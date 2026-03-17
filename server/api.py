"""
server/api.py - consolidated API (M1 + M2 + M3)

Endpoints
---------
GET  /health                  liveness check
GET  /info                    model + config summary

POST /generate                submit via scheduler (M3, batched)
POST /generate/direct         bypass scheduler, call model directly (M1 baseline)

GET  /scheduler/stats         priority queue, rate limiter, batch metrics
POST /scheduler/config        update max_batch_size or max_wait_ms at runtime
DELETE /scheduler/drain       wait for queue to empty

GET  /cache/stats             KV block pool utilisation
DELETE /cache/flush           free all active sequences

GET  /metrics                 single combined snapshot of everything
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.model import load_model, generate as direct_generate
from core.kv_cache import BlockConfig
from core.cache_manager import CacheManager
from scheduler.scheduler import Scheduler
from scheduler.priority_queue import Request

# shared state 

state: dict = {}


# startup / shutdown 

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  inference-cluster starting up")
    print("=" * 50)

    # 1. load model
    print("[1/4] loading model...")
    model, tokenizer = load_model()
    state["model"] = model
    state["tokenizer"] = tokenizer
    device = str(next(model.parameters()).device)
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"      model on {device}, dtype={dtype}")

    # 2. KV cache
    print("[2/4] initialising KV cache...")
    cfg = BlockConfig(
        num_layers=24,
        num_heads=16,
        head_dim=64,
        block_size=16,
        num_blocks=256,
        dtype=dtype,
    )
    cm = CacheManager(cfg, device=device)
    state["cache_manager"] = cm
    state["block_config"] = cfg
    print(f"      {cfg.num_blocks} blocks x {cfg.block_size} tokens/block")

    # 3. scheduler
    print("[3/4] starting scheduler...")
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_wait_ms=20.0,
        rate=50.0,
        burst=20.0,
    )
    scheduler.start()
    state["scheduler"] = scheduler

    # 4. warmup
    print("[4/4] warming up model...")
    t0 = time.perf_counter()
    warmup = Request(prompt="warmup", max_new_tokens=5)
    scheduler.submit(warmup).result(timeout=60)
    print(f"      warmup done in {(time.perf_counter()-t0)*1000:.0f}ms")

    state["start_time"] = time.perf_counter()
    print("=" * 50)
    print("  ready - http://localhost:8000/docs")
    print("=" * 50)

    yield

    print("shutting down...")
    scheduler.stop()
    state.clear()


app = FastAPI(
    title="inference-cluster",
    description="Single-node inference server - M1 / M2 / M3",
    version="0.3.0",
    lifespan=lifespan,
)


# schemas 

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int   = Field(100,  ge=1,    le=1000)
    temperature: float    = Field(0.8,  ge=0.01, le=2.0)
    top_p: float          = Field(0.95, ge=0.0,  le=1.0)
    priority: int         = Field(0,    ge=0,    le=10,
                                  description="0=highest priority, 10=lowest")

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    tokens_per_sec: float
    batch_size: Optional[int] = None

class SchedulerConfigRequest(BaseModel):
    max_batch_size: Optional[int]   = Field(None, ge=1,   le=64)
    max_wait_ms:    Optional[float] = Field(None, ge=1.0, le=5000.0)


# /health 

@app.get("/health", tags=["system"])
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "model_loaded":      "model"     in state,
        "scheduler_running": "scheduler" in state,
        "uptime_s": round(time.perf_counter() - state.get("start_time", 0), 1),
    }


# /info 

@app.get("/info", tags=["system"])
def info():
    """Model and config summary."""
    cfg: BlockConfig = state.get("block_config")
    model = state.get("model")
    device = str(next(model.parameters()).device) if model else "unknown"
    param_count = sum(p.numel() for p in model.parameters()) if model else 0

    return {
        "device": device,
        "param_count_M": round(param_count / 1e6, 1),
        "kv_cache": {
            "num_layers":   cfg.num_layers,
            "num_heads":    cfg.num_heads,
            "head_dim":     cfg.head_dim,
            "block_size":   cfg.block_size,
            "num_blocks":   cfg.num_blocks,
            "pool_size_mb": round(
                cfg.num_blocks * cfg.block_size * cfg.num_layers
                * cfg.num_heads * cfg.head_dim * 2 * 2 / 1e6, 1
            ),
        } if cfg else None,
        "scheduler": {
            "max_batch_size": state["scheduler"].max_batch_size,
            "max_wait_ms":    state["scheduler"].max_wait_ms,
            "rate_per_sec":   state["scheduler"].rate_limiter.rate,
            "burst_capacity": state["scheduler"].rate_limiter.capacity,
        } if "scheduler" in state else None,
    }


# /generate (scheduler path, M3 batched)

@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate(req: GenerateRequest):
    """
    Submit a request through the priority scheduler.
    Requests arriving within max_wait_ms of each other are batched
    into a single forward pass for higher throughput.
    Use priority 0-10 to control queue position (0 = served first).
    """
    if "scheduler" not in state:
        raise HTTPException(503, "scheduler not running")

    request = Request(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        priority=req.priority,
    )

    t0 = time.perf_counter()
    fut = state["scheduler"].submit(request)

    try:
        result = fut.result(timeout=120)
    except RuntimeError as e:
        raise HTTPException(429, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"generation failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        request_id=request.request_id,
        text=result["text"],
        prompt_tokens=result["prompt_tokens"],
        generated_tokens=result["generated_tokens"],
        latency_ms=round(latency_ms, 1),
        tokens_per_sec=round(result["generated_tokens"] / (latency_ms / 1000), 1),
        batch_size=result.get("batch_size"),
    )


# /generate/direct (M1 baseline, no scheduler)

@app.post("/generate/direct", response_model=GenerateResponse, tags=["inference"])
def generate_direct(req: GenerateRequest):
    """
    Call the model directly, bypassing the scheduler entirely.
    No batching, no rate limiting, no priority queue.
    Use this as your M1 baseline to compare against /generate.
    """
    if "model" not in state:
        raise HTTPException(503, "model not loaded")

    t0 = time.perf_counter()
    result = direct_generate(
        state["model"],
        state["tokenizer"],
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        request_id=str(uuid.uuid4())[:8],
        text=result["text"],
        prompt_tokens=result["prompt_tokens"],
        generated_tokens=result["generated_tokens"],
        latency_ms=round(latency_ms, 1),
        tokens_per_sec=round(result["generated_tokens"] / (latency_ms / 1000), 1),
    )


# /scheduler/stats 

@app.get("/scheduler/stats", tags=["scheduler"])
def scheduler_stats():
    """
    Live scheduler metrics: queue depth, rate limiter state,
    avg batch size, total tokens generated, avg batch latency.
    """
    if "scheduler" not in state:
        raise HTTPException(503, "scheduler not running")
    return state["scheduler"].stats()


# /scheduler/config 

@app.post("/scheduler/config", tags=["scheduler"])
def scheduler_config(req: SchedulerConfigRequest):
    """
    Update scheduler parameters at runtime without restarting.
    Only provided fields are changed.

    Tuning guide:
      max_batch_size  higher = more throughput, higher latency per request
      max_wait_ms     higher = batches fill more, higher latency for first request in batch
    """
    if "scheduler" not in state:
        raise HTTPException(503, "scheduler not running")

    s: Scheduler = state["scheduler"]
    changed = {}

    if req.max_batch_size is not None:
        s.max_batch_size = req.max_batch_size
        changed["max_batch_size"] = req.max_batch_size

    if req.max_wait_ms is not None:
        s.max_wait_ms = req.max_wait_ms
        changed["max_wait_ms"] = req.max_wait_ms

    return {
        "updated": changed,
        "current": {
            "max_batch_size": s.max_batch_size,
            "max_wait_ms":    s.max_wait_ms,
        },
    }


# /scheduler/drain 

@app.delete("/scheduler/drain", tags=["scheduler"])
def scheduler_drain(timeout_s: float = 30.0):
    """
    Block until the request queue is empty or timeout_s elapses.
    Useful at the end of a benchmark to ensure all requests are flushed.
    """
    if "scheduler" not in state:
        raise HTTPException(503, "scheduler not running")

    s: Scheduler = state["scheduler"]
    deadline = time.perf_counter() + timeout_s

    while s.queue.size() > 0:
        if time.perf_counter() > deadline:
            return {"drained": False, "remaining": s.queue.size()}
        time.sleep(0.05)

    return {"drained": True, "remaining": 0}


# /cache/stats 

@app.get("/cache/stats", tags=["cache"])
def cache_stats():
    """
    KV cache block pool state: total / used / free blocks,
    utilisation ratio, active sequences, prefix cache entries.
    """
    if "cache_manager" not in state:
        raise HTTPException(503, "cache not initialised")
    return state["cache_manager"].stats()


# /cache/flush 

@app.delete("/cache/flush", tags=["cache"])
def cache_flush():
    """
    Free all active sequence caches.
    Resets utilisation to 0. Useful between benchmark runs.
    """
    cm: CacheManager = state["cache_manager"]
    freed = list(cm._sequences.keys())
    for seq_id in freed:
        cm.free_sequence(seq_id)
    return {"flushed_sequences": freed, "stats": cm.stats()}


# /metrics 

@app.get("/metrics", tags=["system"])
def metrics():
    """
    Single combined snapshot of all subsystems.
    Poll this from a monitoring script or dashboard.
    """
    return {
        "uptime_s":  round(time.perf_counter() - state.get("start_time", 0), 1),
        "scheduler": state["scheduler"].stats()    if "scheduler"    in state else None,
        "cache":     state["cache_manager"].stats() if "cache_manager" in state else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


