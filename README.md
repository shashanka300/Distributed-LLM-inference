# Inference Cluster

A learning-oriented distributed inference cluster built with FastAPI, HuggingFace Transformers, and Python background workers.

This project demonstrates how to combine:

- Multi-node request routing with consistent hashing
- Session affinity for cache locality
- Worker-side priority scheduling and continuous batching
- Basic rate limiting and queue control
- Cluster metrics, anomaly detection, and PID-based autoscaling

## Open Source License

This project is licensed under the MIT License.

See [LICENSE](./LICENSE) for full terms.

## What This System Does

The codebase runs a small cluster with:

- A router (`:8000`) that receives client requests and forwards them
- Two baseline workers (`:8001`, `:8002`) that host the model API
- Optional autoscaled workers (`:8003+`) spawned under load

At runtime:

1. Client sends `POST /generate` to the router with a `session_id`
2. Router selects a healthy worker from a consistent-hash ring
3. Worker enqueues request into a priority queue
4. Scheduler batches nearby requests and calls model generation
5. Router returns generated text plus routing metadata
6. Metrics collector and autoscaler run in background loops

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the cluster

```bash
python start_cluster.py
```

This launches:

- `server.api:app` as `worker-a` on `127.0.0.1:8001`
- `server.api:app` as `worker-b` on `127.0.0.1:8002`
- `router.router:app` on `127.0.0.1:8000`

### 3. Check health

```bash
python start_cluster.py status
```

### 4. Smoke test session affinity

```bash
python start_cluster.py test
```

### 5. Open docs

- Router docs: `http://127.0.0.1:8000/docs`
- Worker docs: `http://127.0.0.1:8001/docs` and `http://127.0.0.1:8002/docs`

## Detailed Walkthrough

## 1) Worker Service (`server/api.py`)

Each worker process does four startup steps:

1. Loads tokenizer + model (`Qwen/Qwen2.5-0.5B`)
2. Initializes KV block allocator structures
3. Starts scheduler background thread
4. Runs warmup generation request

Main worker endpoints:

- `GET /health`: liveness and startup state
- `GET /info`: model + scheduler + KV config summary
- `POST /generate`: scheduler path (priority queue + batching + rate limit)
- `POST /generate/direct`: direct model path (baseline without scheduler)
- `GET /scheduler/stats`: queue, limiter, batching metrics
- `POST /scheduler/config`: runtime tuning for batching parameters
- `DELETE /scheduler/drain`: wait for queue to flush
- `GET /cache/stats`: KV allocator usage snapshot
- `DELETE /cache/flush`: clear active sequence caches
- `GET /metrics`: combined worker metrics

## 2) Scheduling Layer (`scheduler/`)

Scheduler design:

- HTTP threads submit `Request` objects to `PriorityQueue`
- Background worker thread pops requests and forms micro-batches
- Batch loop waits up to `max_wait_ms` to fill up to `max_batch_size`
- Single forward/generate call serves multiple requests
- Each request resolves via a `Future`

Key files:

- `priority_queue.py`: min-heap by `(priority, arrival_time)` for FIFO within tier
- `rate_limiter.py`: token-bucket limiter (burst + refill behavior)
- `scheduler.py`: batching worker loop, batch execution, scheduler metrics

## 3) Router Layer (`router/`)

Router is the cluster ingress and control plane.

Core behavior:

- Maintains consistent-hash ring of worker nodes
- Maintains heartbeat monitor state for health-based routing
- Routes by `session_id` for stable worker affinity
- Falls back to next ring node if primary is unhealthy

Key files:

- `hash_ring.py`: virtual-node consistent hashing + distribution stats
- `heartbeat.py`: periodic `/health` polling with up/down thresholds
- `router.py`: FastAPI ingress, proxy endpoints, metrics/autoscaler wiring

Router endpoints:

- `GET /health`: router liveness + healthy workers
- `GET /nodes`: ring stats + heartbeat + sample distribution
- `GET /nodes/{node_id}/stats`: proxy worker `/metrics`
- `POST /nodes/{node_id}/config`: proxy worker scheduler config
- `POST /generate`: ingress generation endpoint
- `GET /metrics`: cluster aggregated metrics
- `GET /autoscaler/stats`: autoscaler + PID status
- `GET /autoscaler/pid/history`: PID tick history
- `POST /autoscaler/config`: live PID/cooldown tuning

## 4) Metrics and Autoscaling (`metrics/`)

The observability/autoscaling loop runs in router process:

- `collector.py` polls worker `/metrics`
- Tracks rolling latency/queue windows
- Runs simple z-score anomaly detector
- Exposes latest cluster snapshot

PID control:

- `pid.py` computes control output from queue depth error
- Output interpreted as scale pressure (up/down)
- Integral clamp limits windup
- Tick history is retained for tuning/debugging

Autoscaler:

- `autoscaler.py` reads queue depth periodically
- Uses PID output + thresholds + cooldown to decide actions
- `scale_up`: spawn new worker process, wait for `/health`, register in ring+monitor
- `scale_down`: remove owned worker from ring, terminate process

## 5) Core Model/Cache Modules (`core/`)

- `model.py`: model/tokenizer loading and direct generation helper
- `attention.py`: educational NumPy self-attention implementation + causal mask
- `kv_cache.py`: block allocator, prefix block sharing, copy-on-write semantics
- `cache_manager.py`: sequence-level block table management

Note: worker generation currently uses HuggingFace `use_cache=True` path; the custom KV cache layer exists for experimentation and instrumentation.

## Code Structure

```text
inference_cluster/
|-- core/
|   |-- attention.py
|   |-- cache_manager.py
|   |-- kv_cache.py
|   `-- model.py
|-- metrics/
|   |-- autoscaler.py
|   |-- collector.py
|   `-- pid.py
|-- router/
|   |-- hash_ring.py
|   |-- heartbeat.py
|   `-- router.py
|-- scheduler/
|   |-- priority_queue.py
|   |-- rate_limiter.py
|   `-- scheduler.py
|-- server/
|   `-- api.py
|-- tests/
|   |-- bench_m1.py
|   |-- bench_m2.py
|   |-- flood_test.py
|   |-- test_attention.py
|   |-- test_kv_cache.py
|   |-- test_pid.py
|   |-- test_scheduler.py
|   `-- tests_hast_ring.py
|-- main.py
|-- pyproject.toml
|-- start_cluster.py
`-- README.md
```

## Running Tests

Run with `pytest`:

```bash
pytest -q
```

Or run individual test scripts directly:

```bash
python tests/test_scheduler.py
python tests/test_pid.py
python tests/tests_hast_ring.py
```

## Benchmarking and Load Exercises

- `tests/bench_m1.py`: basic sequential benchmark
- `tests/bench_m2.py`: scheduler/batching comparisons and latency studies
- `tests/flood_test.py`: cluster flood scenarios to observe autoscaler behavior

## Configuration Notes

- Ports are defined in `start_cluster.py` and `router/router.py`
- Model name defaults to `Qwen/Qwen2.5-0.5B` in `core/model.py`
- Scheduler defaults are set in `server/api.py`
- Autoscaler/PID defaults are set in `router/router.py`

## Known Scope and Constraints

- Intended as an educational/experimental project, not production-hardened
- Uses local process spawning, not container orchestration
- No persistent queue or distributed state store
- Health/routing/scale behavior is process-local

## Contributing

Issues and pull requests are welcome. If you plan major changes, open an issue first so design direction can be discussed.
