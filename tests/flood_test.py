"""
tests/flood_test.py  M5 flood test
Hammers the cluster and watches the PID autoscaler react in real time.

Usage
-----
python tests/flood_test.py            run full flood + monitor
python tests/flood_test.py monitor    just watch PID ticks (no flood)
python tests/flood_test.py quick      small flood (10 requests)
"""

import sys
import time
import threading
import statistics
import concurrent.futures
import requests

BASE = "http://127.0.0.1:8000"


# helpers 

def get(path: str) -> dict:
    return requests.get(f"{BASE}{path}", timeout=5).json()

def post(path: str, **kwargs) -> dict:
    return requests.post(f"{BASE}{path}", timeout=120, **kwargs).json()

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_pid_line(label: str = ""):
    try:
        s = get("/autoscaler/stats")
        last = s["pid"].get("last", {})
        workers = s["active_workers"]
        events  = s["scale_events"]
        last_event = events[-1]["action"] if events else "none"
        print(
            f"  {label:<12}"
            f"queue={last.get('measured', 0):>5.1f}  "
            f"error={last.get('error', 0):>6.2f}  "
            f"P={last.get('p', 0):>6.3f}  "
            f"I={last.get('i', 0):>6.3f}  "
            f"D={last.get('d', 0):>6.3f}  "
            f"out={last.get('output', 0):>6.3f}  "
            f"workers={workers}  "
            f"last_event={last_event}"
        )
    except Exception as e:
        print(f"  {label:<12} [could not reach router: {e}]")


# monitor thread 

class PIDMonitor(threading.Thread):
    """
    Runs in the background during the flood.
    Prints a PID snapshot every `interval_s` seconds.
    Records all snapshots so we can summarise after the flood.
    """
    def __init__(self, interval_s: float = 3.0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.snapshots: list[dict] = []
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                s = get("/autoscaler/stats")
                last = s["pid"].get("last", {})
                snap = {
                    "ts":       time.time(),
                    "measured": last.get("measured", 0),
                    "error":    last.get("error", 0),
                    "output":   last.get("output", 0),
                    "p":        last.get("p", 0),
                    "i":        last.get("i", 0),
                    "d":        last.get("d", 0),
                    "workers":  s["active_workers"],
                }
                self.snapshots.append(snap)
                print_pid_line(label=f"t+{len(self.snapshots)*self.interval_s:.0f}s")
            except Exception:
                pass
            self._stop.wait(timeout=self.interval_s)

    def summary(self):
        if not self.snapshots:
            print("  no snapshots recorded")
            return
        queues  = [s["measured"] for s in self.snapshots]
        outputs = [s["output"]   for s in self.snapshots]
        workers = [s["workers"]  for s in self.snapshots]
        print(f"  snapshots      : {len(self.snapshots)}")
        print(f"  queue depth    : mean={statistics.mean(queues):.2f}  "
              f"max={max(queues):.2f}  min={min(queues):.2f}")
        print(f"  PID output     : mean={statistics.mean(outputs):.3f}  "
              f"max={max(outputs):.3f}  min={min(outputs):.3f}")
        print(f"  workers seen   : {sorted(set(workers))}")

        # sparkline of queue depth over time
        if len(queues) > 1:
            max_q = max(queues) or 1
            line = "".join("._-:=+*#"[min(int(q / max_q * 7), 7)] for q in queues)
            print(f"  queue sparkline: {line}")


# flood functions 

def single_request(session_id: str = "flood", max_new_tokens: int = 100) -> dict:
    try:
        r = requests.post(
            f"{BASE}/generate",
            params={"session_id": session_id},
            json={
                "prompt": "Explain the transformer architecture in detail, "
                          "including attention mechanisms and layer normalisation:",
                "max_new_tokens": max_new_tokens,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "ok":         True,
            "latency_ms": data.get("latency_ms", 0),
            "tokens":     data.get("generated_tokens", 0),
            "tok_s":      data.get("tokens_per_sec", 0),
            "node":       data.get("routed_to", "?"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "latency_ms": 0, "tokens": 0}


def run_flood(
    n_requests:  int   = 50,
    concurrency: int   = 20,
    max_new_tokens: int = 100,
    label: str = "flood",
):
    print(f"\n  sending {n_requests} requests "
          f"(concurrency={concurrency}, max_tokens={max_new_tokens})...")

    results = []
    t_wall  = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(single_request, f"session-{i % 10}", max_new_tokens): i
            for i in range(n_requests)
        }
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    wall_ms = (time.perf_counter() - t_wall) * 1000

    ok      = [r for r in results if r["ok"]]
    errors  = [r for r in results if not r["ok"]]
    lats    = sorted(r["latency_ms"] for r in ok)
    toks    = [r["tokens"] for r in ok]
    tok_s   = [r["tok_s"]  for r in ok]
    nodes   = {}
    for r in ok:
        nodes[r["node"]] = nodes.get(r["node"], 0) + 1

    total_tokens = sum(toks)
    agg_tok_s    = total_tokens / (wall_ms / 1000) if wall_ms > 0 else 0

    n = len(lats)
    print(f"\n  {label} results:")
    print(f"    completed  : {len(ok)}/{n_requests}  errors={len(errors)}")
    print(f"    wall time  : {wall_ms:.0f}ms")
    print(f"    agg tok/s  : {agg_tok_s:.1f}")
    if lats:
        print(f"    latency    : "
              f"mean={statistics.mean(lats):.0f}ms  "
              f"median={statistics.median(lats):.0f}ms  "
              f"p95={lats[max(int(n*0.95)-1,0)]:.0f}ms  "
              f"p99={lats[max(int(n*0.99)-1,0)]:.0f}ms  "
              f"max={max(lats):.0f}ms")
        print(f"    tok/s/req  : mean={statistics.mean(tok_s):.1f}")
    print(f"    node dist  : {nodes}")

    if errors:
        print(f"    errors     :")
        for e in errors[:5]:
            print(f"      {e.get('error', '?')}")

    return results


# test scenarios 

def scenario_baseline():
    """Light load  PID should stay near 0, no scaling."""
    print_header("Scenario 1: baseline (light load)")
    print("  Expected: PID output near 0, no scale events")

    monitor = PIDMonitor(interval_s=2.0)
    monitor.start()
    print(f"\n  {'label':<12}{'queue':>8}  {'error':>8}  "
          f"{'P':>8}  {'I':>8}  {'D':>8}  {'output':>8}  workers  last_event")
    print(f"  {'-'*85}")

    run_flood(n_requests=8, concurrency=2, max_new_tokens=30, label="baseline")
    time.sleep(6)   # let PID settle

    monitor.stop()
    print("\n  PID monitor summary:")
    monitor.summary()


def scenario_burst():
    """Heavy burst  queue depth spikes, PID should output positive (scale up)."""
    print_header("Scenario 2: burst (heavy load)")
    print("  Expected: queue depth rises, PID output goes positive, "
          "possible scale-up event after cooldown")

    monitor = PIDMonitor(interval_s=3.0)
    monitor.start()
    print(f"\n  {'label':<12}{'queue':>8}  {'error':>8}  "
          f"{'P':>8}  {'I':>8}  {'D':>8}  {'output':>8}  workers  last_event")
    print(f"  {'-'*85}")

    run_flood(n_requests=50, concurrency=20, max_new_tokens=100, label="burst")
    time.sleep(15)  # watch PID react after flood

    monitor.stop()
    print("\n  PID monitor summary:")
    monitor.summary()


def scenario_sustained():
    """
    Two waves of load  tests integral term accumulation.
    After the first wave the integral should be non-zero,
    making the second wave trigger scaling sooner.
    """
    print_header("Scenario 3: two waves (integral accumulation)")
    print("  Expected: second wave triggers faster than first")

    monitor = PIDMonitor(interval_s=3.0)
    monitor.start()
    print(f"\n  {'label':<12}{'queue':>8}  {'error':>8}  "
          f"{'P':>8}  {'I':>8}  {'D':>8}  {'output':>8}  workers  last_event")
    print(f"  {'-'*85}")

    run_flood(n_requests=20, concurrency=10, max_new_tokens=80, label="wave-1")
    print("\n  waiting 10s between waves...")
    time.sleep(10)
    run_flood(n_requests=20, concurrency=10, max_new_tokens=80, label="wave-2")
    time.sleep(10)

    monitor.stop()
    print("\n  PID monitor summary:")
    monitor.summary()


def scenario_monitor_only(duration_s: int = 60):
    """Just watch the PID ticks  no flood."""
    print_header(f"Monitoring PID for {duration_s}s (no flood)")
    print(f"\n  {'label':<12}{'queue':>8}  {'error':>8}  "
          f"{'P':>8}  {'I':>8}  {'D':>8}  {'output':>8}  workers  last_event")
    print(f"  {'-'*85}")

    monitor = PIDMonitor(interval_s=3.0)
    monitor.start()
    try:
        time.sleep(duration_s)
    except KeyboardInterrupt:
        pass
    monitor.stop()
    print("\n  summary:")
    monitor.summary()


def scenario_quick():
    """Quick smoke test  10 requests, just verify routing works."""
    print_header("Quick smoke test (10 requests)")
    run_flood(n_requests=10, concurrency=4, max_new_tokens=30, label="quick")
    print_pid_line(label="after")


# main 

if __name__ == "__main__":
    # check server is up
    try:
        h = requests.get(f"{BASE}/health", timeout=3).json()
        healthy = h.get("healthy_nodes", [])
        print(f"cluster healthy  nodes: {healthy}")
        if not healthy:
            print("WARNING: no healthy nodes. Start the cluster first:")
            print("  python start_cluster.py")
            sys.exit(1)
    except Exception:
        print(f"ERROR: router not reachable at {BASE}")
        print("Start the cluster first:  python start_cluster.py")
        sys.exit(1)

    # initial PID state
    print(f"\ninitial autoscaler state:")
    s = get("/autoscaler/stats")
    pid_cfg = s["pid"]
    print(f"  Kp={pid_cfg['kp']}  Ki={pid_cfg['ki']}  Kd={pid_cfg['kd']}  "
          f"target={pid_cfg['target']}  workers={s['active_workers']}")

    command = sys.argv[1] if len(sys.argv) > 1 else "full"

    if command == "monitor":
        scenario_monitor_only(duration_s=60)

    elif command == "quick":
        scenario_quick()

    elif command == "burst":
        scenario_burst()

    elif command == "full":
        scenario_baseline()
        print("\nwaiting 15s before burst scenario...")
        time.sleep(15)
        scenario_burst()
        print("\nwaiting 20s before two-wave scenario...")
        time.sleep(20)
        scenario_sustained()

    else:
        print(f"unknown command: {command!r}")
        print("usage: python tests/flood_test.py [full|burst|quick|monitor]")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  flood test complete")
    final = get("/autoscaler/stats")
    print(f"  scale events: {final['scale_events']}")
    print(f"  final workers: {final['active_workers']}")
    print("=" * 60 + "\n")



