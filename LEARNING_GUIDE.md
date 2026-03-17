# Learning Guide: How To Navigate This Codebase

This guide is for developers who want to understand the project quickly and deeply.

## Goal

By the end, you should be able to answer:

- How a request moves from router to worker and back
- How scheduling and batching work
- How health checks and consistent hashing decide routing
- How metrics feed the PID autoscaler
- Where to change behavior safely

## Recommended Reading Order

1. `start_cluster.py`
2. `router/router.py`
3. `router/hash_ring.py`
4. `router/heartbeat.py`
5. `server/api.py`
6. `scheduler/scheduler.py`
7. `scheduler/priority_queue.py`
8. `scheduler/rate_limiter.py`
9. `metrics/collector.py`
10. `metrics/pid.py`
11. `metrics/autoscaler.py`
12. `core/model.py`
13. `core/kv_cache.py` and `core/cache_manager.py`
14. `tests/*` scripts for behavior validation

## Big Picture Map

## Router-side control plane

- `router/router.py`: ingress API + subsystem wiring
- `router/hash_ring.py`: deterministic placement by session key
- `router/heartbeat.py`: node up/down transitions

## Worker-side data plane

- `server/api.py`: inference API and subsystem lifecycle
- `scheduler/*`: priority queue, rate limiting, micro-batching
- `core/model.py`: model/tokenizer loading and generation path

## Observability and scaling

- `metrics/collector.py`: polling and aggregation
- `metrics/pid.py`: control signal
- `metrics/autoscaler.py`: process-level scale actions

## Follow One Request End-to-End

Use this trace when reading:

1. Client calls `POST /generate` on router (`router/router.py`)
2. Router selects node using `route(session_id)`
3. Router forwards JSON payload to worker `/generate`
4. Worker builds `Request` object (`server/api.py`)
5. Scheduler `submit()` enqueues request (`scheduler/scheduler.py`)
6. Scheduler loop batches and calls model generate
7. Future resolves, worker returns response
8. Router decorates response with `routed_to`, `router_ms`, `session_id`

## Understanding Each Module Quickly

## `router/hash_ring.py`

Focus on:

- `add_node()`
- `remove_node()`
- `get_node()`
- `get_nodes_for_key()`

What to verify:

- Same key maps to same node
- Adding/removing node remaps only a subset of keys

Use tests:

- `tests/tests_hast_ring.py`

## `router/heartbeat.py`

Focus on:

- Failure and recovery thresholds
- Streak logic and state transitions
- Callbacks `on_node_down`, `on_node_up`

Why it matters:

- Routing decisions depend on health status

## `scheduler/scheduler.py`

Focus on:

- `submit()`
- `_worker_loop()`
- `_process_batch()`
- `_run_batch()`

What to watch:

- Batch fill delay (`max_wait_ms`) vs latency tradeoff
- `max_batch_size` throughput effect
- Temperature/top_p behavior in mixed requests

## `metrics/pid.py` and `metrics/autoscaler.py`

Focus on:

- Error sign convention (`measured - target`)
- Integral clamp (`integral_max`)
- Scale thresholds + cooldown
- Ring registration for autoscaled nodes

What to observe live:

- `GET /autoscaler/stats`
- `GET /autoscaler/pid/history`

## Hands-On Learning Path

## Step 1: Bring up the cluster

```bash
python start_cluster.py
python start_cluster.py status
```

## Step 2: Send manual traffic

Use router docs: `http://127.0.0.1:8000/docs`

Try multiple `session_id` values and inspect `routed_to`.

## Step 3: Watch scheduler behavior

Call worker metrics:

- `http://127.0.0.1:8001/metrics`
- `http://127.0.0.1:8002/metrics`

Inspect:

- queue depth
- avg batch size
- avg batch latency

## Step 4: Stress the cluster

```bash
python tests/flood_test.py quick
python tests/flood_test.py burst
```

Watch:

- autoscaler events
- node count changes
- queue depth dynamics

## Step 5: Tune and compare

Adjust at runtime:

- `POST /scheduler/config` on workers
- `POST /autoscaler/config` on router

Compare changes in:

- p95 latency
- throughput
- scaling behavior

## Where To Make Common Changes

## Change model

- Edit `DEFAULT_MODEL` in `core/model.py`

## Change batching policy

- Edit defaults in `server/api.py` scheduler init
- Runtime tune via `/scheduler/config`

## Change routing behavior

- `router/hash_ring.py` for placement logic
- `router/router.py` for candidate/healthy fallback behavior

## Change autoscaling aggressiveness

- PID gains and thresholds in `router/router.py`
- Logic and cooldown in `metrics/autoscaler.py`

## Debug Checklist

If generation fails:

1. Check router health endpoint
2. Check worker health endpoints
3. Check worker logs for model load errors
4. Check scheduler queue depth and rejection rate
5. Check heartbeat node status for unhealthy transitions

If scaling does not trigger:

1. Confirm collector has non-zero queue depth
2. Inspect `autoscaler/stats` output and cooldown remaining
3. Verify PID output crosses thresholds
4. Confirm spawned workers pass `/health` and get ring registration

## Suggested Next Improvements (for learning exercises)

1. Add async HTTP forwarding in router.
2. Add per-node weighted routing from live utilization.
3. Add request tracing IDs and end-to-end timing spans.
4. Integrate custom KV cache into actual decode path.
5. Add persistent config profiles for scheduler and PID tuning.
