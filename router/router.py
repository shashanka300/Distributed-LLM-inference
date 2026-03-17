"""
router/router.py - M4 + M5 complete router

Endpoints
---------
GET  /health                    liveness + healthy nodes
GET  /nodes                     ring stats + heartbeat + distribution
GET  /nodes/{node_id}/stats     proxy /metrics from a specific worker
POST /nodes/{node_id}/config    proxy /scheduler/config to a specific worker

POST /generate                  route to worker via consistent hash (session affinity)

GET  /metrics                   cluster-wide latency, queue depth, anomalies
GET  /autoscaler/stats          PID state + scaling event history
GET  /autoscaler/pid/history    full PID tick history (plot the control signal)
"""

import time
import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException

from router.hash_ring import ConsistentHashRing, Node
from router.heartbeat import HeartbeatMonitor
from metrics.collector import MetricsCollector
from metrics.pid import PIDController
from metrics.autoscaler import Autoscaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router")

# cluster config 

WORKER_NODES = [
    Node(node_id="worker-a", host="127.0.0.1", port=8001),
    Node(node_id="worker-b", host="127.0.0.1", port=8002),
]

WORKER_URLS = [f"http://127.0.0.1:{n.port}" for n in WORKER_NODES]

# subsystems 

ring = ConsistentHashRing(vnodes_per_node=150)

monitor = HeartbeatMonitor(
    interval_s=2.0,
    failure_threshold=3,
    recovery_threshold=2,
    on_node_down=lambda nid: (
        logger.warning(f"node {nid} DOWN - removing from ring"),
        ring.remove_node(nid),
    ),
    on_node_up=lambda nid: (
        logger.info(f"node {nid} UP - re-adding to ring"),
        ring.add_node(next(n for n in WORKER_NODES if n.node_id == nid)),
    ),
)

collector = MetricsCollector(worker_urls=WORKER_URLS, interval_s=3.0)

pid = PIDController(
    kp=0.5, ki=0.1, kd=0.05,
    target=0.0,
    output_min=-2.0,
    output_max=2.0,
)

autoscaler = Autoscaler(
    collector=collector,
    pid=pid,
    min_workers=1,
    max_workers=4,
    tick_s=5.0,
    cooldown_s=30.0,
    scale_up_threshold=0.6,
    scale_down_threshold=-0.6,
    ring=ring,        # <-- new
    monitor=monitor,  # <-- new
)

# startup / shutdown 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # register all worker nodes
    for node in WORKER_NODES:
        ring.add_node(node)
        monitor.register(node.node_id, node.address)
        autoscaler.register_existing(node.node_id, node.port)
        logger.info(f"registered {node.node_id} at {node.address}")

    # start background threads
    monitor.start()
    collector.start()
    autoscaler.start()

    print("=" * 50)
    print("  router started (M4 + M5)")
    print(f"  workers : {[n.address for n in WORKER_NODES]}")
    print(f"  ring    : {ring.stats()}")
    print(f"  docs    : http://127.0.0.1:8000/docs")
    print("=" * 50)

    yield

    # clean shutdown
    autoscaler.stop()
    collector.stop()
    monitor.stop()
    logger.info("router shut down")


app = FastAPI(
    title="inference-router",
    description="Consistent-hash router with heartbeat, metrics, and PID autoscaler",
    version="0.5.0",
    lifespan=lifespan,
)


# routing helpers 

def route(session_id: str) -> Node:
    """
    Pick a healthy node for this session_id.
    Tries primary first, falls back through ring order if primary is down.
    """
    candidates = ring.get_nodes_for_key(session_id, n=len(WORKER_NODES))
    for node in candidates:
        if monitor.is_healthy(node.node_id):
            return node
    raise HTTPException(503, "no healthy workers available")


def forward(node: Node, method: str, path: str, **kwargs) -> dict:
    """Forward a request to a worker and return its JSON."""
    url = f"{node.address}{path}"
    try:
        r = getattr(requests, method)(url, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, f"worker {node.node_id} unreachable")
    except requests.exceptions.Timeout:
        raise HTTPException(504, f"worker {node.node_id} timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(r.status_code, detail=str(e))


# /health 

@app.get("/health", tags=["system"])
def health():
    """Liveness check that returns healthy node list and ring summary."""
    return {
        "status":        "ok",
        "healthy_nodes": monitor.healthy_nodes(),
        "ring":          ring.stats(),
    }


# /nodes 

@app.get("/nodes", tags=["system"])
def nodes():
    """
    Full cluster view: ring stats, per-node heartbeat status,
    and simulated traffic distribution across the ring.
    """
    return {
        "ring":         ring.stats(),
        "heartbeat":    monitor.stats(),
        "distribution": ring.distribution(n_samples=1000),
    }


@app.get("/nodes/{node_id}/stats", tags=["system"])
def node_stats(node_id: str):
    """Proxy /metrics from a specific worker node."""
    node = ring._nodes.get(node_id)
    if not node:
        raise HTTPException(404, f"node {node_id!r} not found")
    return forward(node, "get", "/metrics")


@app.post("/nodes/{node_id}/config", tags=["system"])
def node_config(node_id: str, body: dict):
    """Proxy /scheduler/config to a specific worker node."""
    node = ring._nodes.get(node_id)
    if not node:
        raise HTTPException(404, f"node {node_id!r} not found")
    return forward(node, "post", "/scheduler/config", json=body)


# /generate 

@app.post("/generate", tags=["inference"])
def generate(request_body: dict, session_id: str = "default"):
    """
    Route a generation request to the correct worker via consistent hashing.

    session_id pins a conversation to a node; the same session always
    hits the same worker, keeping its KV cache warm (session affinity).

    Pass ?session_id=<any-string> in the query string.
    Different session IDs are spread across the ring.
    """
    node = route(session_id)
    t0   = time.perf_counter()
    result = forward(node, "post", "/generate", json=request_body)
    router_ms = round((time.perf_counter() - t0) * 1000, 1)

    result["routed_to"]  = node.node_id
    result["router_ms"]  = router_ms
    result["session_id"] = session_id
    return result


# /metrics 

@app.get("/metrics", tags=["observability"])
def cluster_metrics():
    """
    Cluster-wide aggregated metrics:
    latency window (p50/p95/p99), queue depth history, anomaly count,
    and per-node snapshots.
    """
    return collector.current()


# /autoscaler/stats 

@app.get("/autoscaler/stats", tags=["observability"])
def autoscaler_stats():
    """
    PID controller state and autoscaling event history.

    Includes:
      - current integral, last error, last output
      - Kp / Ki / Kd tuning parameters
      - list of recent scale_up / scale_down events
      - cooldown remaining before next scaling action is allowed
    """
    return autoscaler.stats()


# /autoscaler/pid/history 

@app.get("/autoscaler/pid/history", tags=["observability"])
def pid_history():
    """
    Full tick-by-tick PID history: measured, error, p, i, d, output.
    Use this to plot the control signal over time and tune Kp/Ki/Kd.

    Example (run in a separate terminal while the cluster is under load):
      curl http://127.0.0.1:8000/autoscaler/pid/history | python -m json.tool
    """
    return pid.history()


# /autoscaler/config 

@app.post("/autoscaler/config", tags=["observability"])
def autoscaler_config(body: dict):
    """
    Tune PID parameters at runtime without restarting.

    Body fields (all optional):
      kp, ki, kd: PID gains
      target: desired queue depth (default 0.0)
      cooldown_s: minimum seconds between scale events

    Example:
      curl -X POST http://127.0.0.1:8000/autoscaler/config \\
        -H 'Content-Type: application/json' \\
        -d '{"kp": 0.8, "ki": 0.15}'
    """
    changed = {}

    if "kp" in body:
        pid.kp = float(body["kp"])
        changed["kp"] = pid.kp
    if "ki" in body:
        pid.ki = float(body["ki"])
        changed["ki"] = pid.ki
    if "kd" in body:
        pid.kd = float(body["kd"])
        changed["kd"] = pid.kd
    if "target" in body:
        pid.target = float(body["target"])
        changed["target"] = pid.target
    if "cooldown_s" in body:
        autoscaler.cooldown_s = float(body["cooldown_s"])
        changed["cooldown_s"] = autoscaler.cooldown_s

    pid.reset()   # reset integral/derivative after tuning

    return {
        "updated": changed,
        "current_pid": {"kp": pid.kp, "ki": pid.ki, "kd": pid.kd,
                        "target": pid.target},
    }


