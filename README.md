# Inference Cluster

A learning-oriented distributed inference cluster built with FastAPI, HuggingFace Transformers, and Python background workers.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dependency Export](#dependency-export)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Benchmarks and Load Tests](#benchmarks-and-load-tests)
- [Learning Guide](#learning-guide)
- [Conceptual Book Placeholder](#conceptual-book-placeholder)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project simulates a small inference cluster with:

- A router (`127.0.0.1:8000`)
- Two baseline workers (`127.0.0.1:8001`, `127.0.0.1:8002`)
- Optional autoscaled workers (`127.0.0.1:8003+`)

It is designed to teach how routing, scheduling, observability, and autoscaling fit together for model-serving systems.

## Features

- Consistent-hash routing for stable session affinity
- Heartbeat-driven health monitoring and failover
- Worker-side priority queue with continuous batching
- Token-bucket rate limiting
- Cluster metrics aggregation and anomaly detection
- PID-based autoscaling with cooldown controls

## Architecture

High-level request path:

1. Client sends `POST /generate` to router.
2. Router selects healthy worker by `session_id` hash.
3. Worker enqueues request into scheduler queue.
4. Scheduler forms micro-batch and runs model generation.
5. Worker returns generation result.
6. Router returns response with routing metadata.

Primary modules:

- Router and control plane: `router/`
- Worker API and scheduling: `server/`, `scheduler/`
- Metrics and autoscaling: `metrics/`
- Model and cache primitives: `core/`

## Repository Structure

```text
inference_cluster/
|-- core/
|   |-- __init__.py
|   |-- attention.py
|   |-- cache_manager.py
|   |-- kv_cache.py
|   `-- model.py
|-- metrics/
|   |-- autoscaler.py
|   |-- collector.py
|   `-- pid.py
|-- resources/
|   `-- CONCEPTUAL_BOOK_PLACEHOLDER.md
|-- router/
|   |-- hash_ring.py
|   |-- heartbeat.py
|   `-- router.py
|-- scheduler/
|   |-- __init__.py
|   |-- priority_queue.py
|   |-- rate_limiter.py
|   `-- scheduler.py
|-- server/
|   |-- __init__.py
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
|-- LEARNING_GUIDE.md
|-- LICENSE
|-- main.py
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- uv.lock
`-- start_cluster.py
```

## Prerequisites

- Python 3.10+
- `uv` (recommended dependency manager) or `pip`

## Installation

```bash
uv sync
```

Alternative (`pip`) setup:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Dependency Export

`pyproject.toml` contains the project-declared dependencies.  
This repository also includes [requirements.txt](./requirements.txt), exported from the `.venv` environment, to capture additional installed packages that may not be explicitly listed in `pyproject.toml`.

Use:

- `uv sync` for lockfile-driven reproducible setup
- `pip install -r requirements.txt` when you want environment parity with the exported venv package set

## Usage

Start the cluster:

```bash
python start_cluster.py
```

Check status:

```bash
python start_cluster.py status
```

Run session-affinity smoke test:

```bash
python start_cluster.py test
```

API docs:

- Router docs: `http://127.0.0.1:8000/docs`
- Worker docs: `http://127.0.0.1:8001/docs` and `http://127.0.0.1:8002/docs`

## Configuration

Main defaults are defined in:

- Ports and process startup: `start_cluster.py`
- Router nodes and autoscaler settings: `router/router.py`
- Scheduler defaults: `server/api.py`
- Default model name: `core/model.py`

## Testing

Run all tests:

```bash
pytest -q
```

Run individual suites:

```bash
python tests/test_scheduler.py
python tests/test_pid.py
python tests/tests_hast_ring.py
```

## Benchmarks and Load Tests

- `tests/bench_m1.py`: basic sequential latency baseline
- `tests/bench_m2.py`: scheduler and batching comparisons
- `tests/flood_test.py`: cluster flood scenarios for autoscaler behavior

## Learning Guide

For a detailed walkthrough and code navigation path:

- [LEARNING_GUIDE.md](./LEARNING_GUIDE.md)

## Conceptual Book Placeholder

You can upload your future conceptual PDF to:

- `resources/conceptual_understanding_book.pdf`

A tracked placeholder file exists here:

- [resources/CONCEPTUAL_BOOK_PLACEHOLDER.md](./resources/inference_cluster_complete.pdf)

## Known Limitations

- Educational/experimental project, not production hardened
- Local process orchestration only
- No distributed persistent queue/state backend
- Custom KV cache exists as infrastructure; generation path currently uses HF `use_cache=True`

## Contributing

Issues and pull requests are welcome.  
For major changes, open an issue first to align on direction.

## License

MIT License. See [LICENSE](./LICENSE).
