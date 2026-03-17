"""
start_cluster.py — starts worker A, worker B, and the router
Run from your project root:  python start_cluster.py

Commands
--------
python start_cluster.py          start all 3 processes
python start_cluster.py stop     kill all 3 processes
python start_cluster.py status   check if all 3 are responding
python start_cluster.py test     run session-affinity smoke test
"""

import sys
import time
import json
import subprocess
import requests

# ── config ────────────────────────────────────────────────────────────────────

WORKERS = [
    {"id": "worker-a", "port": 8001},
    {"id": "worker-b", "port": 8002},
]
ROUTER_PORT = 8000

# path to your python / uvicorn inside the venv
PYTHON = sys.executable   # uses whichever python is currently active

# ── helpers ───────────────────────────────────────────────────────────────────

def run_uvicorn(app: str, port: int, env_extras: dict | None = None) -> subprocess.Popen:
    """Start a uvicorn process in a new visible window on Windows."""
    import os
    env = os.environ.copy()
    if env_extras:
        env.update(env_extras)

    cmd = [
        PYTHON, "-m", "uvicorn", app,
        "--host", "127.0.0.1",
        "--port", str(port),
    ]

    # on Windows, open each process in its own console window
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    proc = subprocess.Popen(cmd, env=env, **kwargs)
    return proc


def wait_for_health(url: str, label: str, timeout: int = 120) -> bool:
    print(f"  waiting for {label} at {url} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f" ready ({r.json().get('status','ok')})")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print(f" TIMED OUT after {timeout}s")
    return False


def print_json(data: dict):
    print(json.dumps(data, indent=2))


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_start():
    procs = []

    # start worker A
    print(f"[1/3] starting worker-a on port {WORKERS[0]['port']}...")
    pa = run_uvicorn("server.api:app", WORKERS[0]["port"],
                     {"WORKER_ID": "worker-a"})
    procs.append(pa)

    # start worker B
    print(f"[2/3] starting worker-b on port {WORKERS[1]['port']}...")
    pb = run_uvicorn("server.api:app", WORKERS[1]["port"],
                     {"WORKER_ID": "worker-b"})
    procs.append(pb)

    # wait for both workers to be healthy before starting router
    ok_a = wait_for_health(
        f"http://127.0.0.1:{WORKERS[0]['port']}/health", "worker-a"
    )
    ok_b = wait_for_health(
        f"http://127.0.0.1:{WORKERS[1]['port']}/health", "worker-b"
    )

    if not (ok_a and ok_b):
        print("ERROR: one or more workers failed to start. Check their console windows.")
        return

    # start router
    print(f"[3/3] starting router on port {ROUTER_PORT}...")
    pr = run_uvicorn("router.router:app", ROUTER_PORT)
    procs.append(pr)

    ok_r = wait_for_health(f"http://127.0.0.1:{ROUTER_PORT}/health", "router")

    if ok_r:
        print()
        print("=" * 50)
        print("  cluster is up")
        print(f"  router  : http://127.0.0.1:{ROUTER_PORT}")
        print(f"  worker-a: http://127.0.0.1:{WORKERS[0]['port']}")
        print(f"  worker-b: http://127.0.0.1:{WORKERS[1]['port']}")
        print(f"  docs    : http://127.0.0.1:{ROUTER_PORT}/docs")
        print("=" * 50)
        print()
        print("Press Ctrl+C to stop all processes.")
        try:
            # keep main process alive — kill all children on Ctrl+C
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            print("\nstopping cluster...")
            for p in procs:
                p.terminate()
            print("done.")
    else:
        print("Router failed to start.")


def cmd_status():
    targets = [
        ("router",   f"http://127.0.0.1:{ROUTER_PORT}/health"),
        ("worker-a", f"http://127.0.0.1:{WORKERS[0]['port']}/health"),
        ("worker-b", f"http://127.0.0.1:{WORKERS[1]['port']}/health"),
    ]
    all_ok = True
    for label, url in targets:
        try:
            r = requests.get(url, timeout=2)
            status = r.json()
            print(f"  {label:<12} UP    {status}")
        except Exception as e:
            print(f"  {label:<12} DOWN  ({e})")
            all_ok = False

    if all_ok:
        print("\nall nodes healthy")
        # show ring distribution
        try:
            nodes = requests.get(
                f"http://127.0.0.1:{ROUTER_PORT}/nodes", timeout=3
            ).json()
            print("\ndistribution (1000 sample keys):")
            for nid, d in nodes.get("distribution", {}).items():
                bar = "#" * int(d["fraction"] * 40)
                print(f"  {nid}: {bar} {d['fraction']:.3f} "
                      f"({d['deviation_pct']:+.1f}%)")
        except Exception:
            pass


def cmd_test():
    """Session affinity smoke test — same session_id must always hit same node."""
    print("session affinity test")
    print("-" * 40)

    base = f"http://127.0.0.1:{ROUTER_PORT}"
    session_id = "test-session-42"
    n = 6

    nodes_seen = []
    for i in range(n):
        try:
            r = requests.post(
                f"{base}/generate",
                params={"session_id": session_id},
                json={"prompt": "Hello:", "max_new_tokens": 10},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            routed_to = data.get("routed_to", "unknown")
            nodes_seen.append(routed_to)
            print(f"  req {i+1}: routed_to={routed_to}  "
                  f"latency={data.get('latency_ms')}ms  "
                  f"router_ms={data.get('router_ms')}ms")
        except Exception as e:
            print(f"  req {i+1}: ERROR {e}")

    print()
    unique = set(nodes_seen)
    if len(unique) == 1:
        print(f"PASS — all {n} requests routed to {unique.pop()} (affinity working)")
    else:
        print(f"FAIL — requests spread across {unique} (affinity broken)")

    # now test that a different session hits a different node
    print()
    print("testing second session routes differently...")
    hits = {}
    for sid in ["session-aaa", "session-bbb", "session-ccc",
                "session-ddd", "session-eee", "session-fff"]:
        try:
            r = requests.post(
                f"{base}/generate",
                params={"session_id": sid},
                json={"prompt": "Hi:", "max_new_tokens": 5},
                timeout=60,
            )
            node = r.json().get("routed_to", "?")
            hits[node] = hits.get(node, 0) + 1
        except Exception as e:
            print(f"  {sid}: ERROR {e}")

    print(f"distribution across 6 sessions: {hits}")
    if len(hits) > 1:
        print("PASS — sessions spread across multiple nodes")
    else:
        print("NOTE — all sessions hit one node (possible with only 6 samples)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "start"

    if command == "start":
        cmd_start()
    elif command == "status":
        cmd_status()
    elif command == "test":
        cmd_test()
    else:
        print(f"unknown command: {command}")
        print("usage: python start_cluster.py [start|status|test]")
        sys.exit(1)