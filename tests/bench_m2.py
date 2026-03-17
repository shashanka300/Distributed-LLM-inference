"""
bench_m2.py  M2/M3 benchmark suite
Run with the server already started:  uvicorn server.api:app --host 127.0.0.1 --port 8000
"""

import time
import statistics
import threading
import concurrent.futures
import requests

BASE = "http://127.0.0.1:8000"


# helpers 

def generate(prompt: str, max_new_tokens: int = 50, temperature: float = 0.8,
             priority: int = 0, direct: bool = False) -> dict:
    endpoint = "/generate/direct" if direct else "/generate"
    r = requests.post(f"{BASE}{endpoint}", json={
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "priority": priority,
    })
    r.raise_for_status()
    return r.json()

def cache_stats() -> dict:
    return requests.get(f"{BASE}/cache/stats").json()

def scheduler_stats() -> dict:
    return requests.get(f"{BASE}/scheduler/stats").json()

def flush_cache():
    requests.delete(f"{BASE}/cache/flush")

def drain_scheduler():
    requests.delete(f"{BASE}/scheduler/drain")

def set_scheduler(max_batch_size=None, max_wait_ms=None):
    body = {}
    if max_batch_size is not None:
        body["max_batch_size"] = max_batch_size
    if max_wait_ms is not None:
        body["max_wait_ms"] = max_wait_ms
    requests.post(f"{BASE}/scheduler/config", json=body)

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_stats(label: str, values: list, unit: str = "ms"):
    if not values:
        return
    s = sorted(values)
    n = len(s)
    print(f"  {label}")
    print(f"    n={n}  "
          f"mean={statistics.mean(values):.1f}{unit}  "
          f"median={statistics.median(values):.1f}{unit}  "
          f"p95={s[max(int(n*0.95)-1,0)]:.1f}{unit}  "
          f"p99={s[max(int(n*0.99)-1,0)]:.1f}{unit}  "
          f"min={min(values):.1f}{unit}  "
          f"max={max(values):.1f}{unit}")


# test 1: sequential latency 

def bench_sequential(n: int = 8, max_new_tokens: int = 50):
    print_header(f"1. Sequential latency  (n={n}, max_new_tokens={max_new_tokens})")
    flush_cache()

    prompt = "The key insight of the transformer architecture is"
    latencies, tok_per_sec = [], []

    for i in range(n):
        result = generate(prompt, max_new_tokens)
        latencies.append(result["latency_ms"])
        tok_per_sec.append(result["tokens_per_sec"])
        print(f"  req {i+1:>2}: {result['latency_ms']:>7.1f}ms  "
              f"{result['generated_tokens']:>3} tokens  "
              f"{result['tokens_per_sec']:>6.1f} tok/s  "
              f"batch={result.get('batch_size', 1)}")

    print()
    print_stats("latency",    latencies,   "ms")
    print_stats("throughput", tok_per_sec, " tok/s")


# test 2: concurrent load 

def bench_concurrent(concurrency: int = 4, total: int = 12, max_new_tokens: int = 40):
    print_header(f"2. Concurrent load  (concurrency={concurrency}, total={total})")
    flush_cache()

    prompt = "Explain the concept of attention in neural networks:"
    latencies, tok_per_sec, batch_sizes = [], [], []
    errors = 0

    def single():
        try:
            return generate(prompt, max_new_tokens)
        except Exception as e:
            return {"error": str(e)}

    t_wall = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(single) for _ in range(total)]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if "error" in result:
                errors += 1
                print(f"  ERROR: {result['error']}")
            else:
                latencies.append(result["latency_ms"])
                tok_per_sec.append(result["tokens_per_sec"])
                if result.get("batch_size"):
                    batch_sizes.append(result["batch_size"])

    wall_ms = (time.perf_counter() - t_wall) * 1000
    total_tokens = sum(
        tok_per_sec[i] * latencies[i] / 1000
        for i in range(len(latencies))
    )

    print(f"  wall time: {wall_ms:.0f}ms  |  errors: {errors}")
    print(f"  aggregate throughput: {total_tokens / (wall_ms/1000):.1f} tok/s")
    if batch_sizes:
        print(f"  avg batch size: {statistics.mean(batch_sizes):.2f}  "
              f"max: {max(batch_sizes)}")
    print()
    print_stats("per-request latency", latencies,   "ms")
    print_stats("per-request tok/s",   tok_per_sec, " tok/s")

    s = scheduler_stats()
    print(f"\n  scheduler after burst:")
    print(f"    batches={s['batches_processed']}  "
          f"avg_batch={s['avg_batch_size']}  "
          f"avg_batch_latency={s['avg_batch_latency_ms']}ms")


# test 3: scheduler vs direct comparison 

def bench_scheduler_vs_direct(n: int = 5, max_new_tokens: int = 40):
    print_header(f"3. Scheduler vs direct  (n={n} sequential each)")
    flush_cache()

    prompt = "Explain gradient descent in machine learning:"

    sched_lat, direct_lat = [], []

    for i in range(n):
        r = generate(prompt, max_new_tokens, direct=False)
        sched_lat.append(r["latency_ms"])

    for i in range(n):
        r = generate(prompt, max_new_tokens, direct=True)
        direct_lat.append(r["latency_ms"])

    print(f"  {'path':<12} {'mean':>8} {'median':>8} {'p95':>8} {'tok/s':>8}")
    print(f"  {'-'*48}")

    def row(label, lats):
        s = sorted(lats)
        tps = max_new_tokens / (statistics.mean(lats) / 1000)
        print(f"  {label:<12} "
              f"{statistics.mean(lats):>7.1f}ms "
              f"{statistics.median(lats):>7.1f}ms "
              f"{s[max(int(len(s)*0.95)-1,0)]:>7.1f}ms "
              f"{tps:>7.1f}")

    row("/generate",        sched_lat)
    row("/generate/direct", direct_lat)

    diff = statistics.mean(direct_lat) - statistics.mean(sched_lat)
    print(f"\n  scheduler overhead vs direct: {diff:+.1f}ms mean")


# test 4: prompt length vs latency 

def bench_prompt_lengths():
    print_header("4. Latency vs prompt length")
    flush_cache()

    prompts = {
        "short  ( ~4 tok)": "What is attention?",
        "medium (~27 tok)": (
            "Explain in detail how the transformer architecture uses "
            "self-attention to process sequences of tokens in parallel "
            "rather than sequentially like an RNN."
        ),
        "long   (~73 tok)": (
            "You are an expert in machine learning. "
            "The transformer model was introduced in the paper "
            "'Attention is All You Need' by Vaswani et al. in 2017. "
            "It replaced recurrent neural networks with a purely "
            "attention-based mechanism. Explain the key components "
            "of the transformer encoder and decoder, focusing on "
            "how multi-head self-attention works mathematically."
        ),
    }

    for label, prompt in prompts.items():
        r = generate(prompt, max_new_tokens=40)
        print(f"  {label}  prompt_tokens={r['prompt_tokens']:>3}  "
              f"latency={r['latency_ms']:>7.1f}ms  "
              f"{r['tokens_per_sec']:>6.1f} tok/s")


# test 5: priority queue ordering 

def bench_priority(n_per_tier: int = 3):
    print_header(f"5. Priority ordering  ({n_per_tier} requests per tier)")
    flush_cache()

    # submit low-priority first, then high-priority
    # high-priority should complete sooner despite arriving later
    results = {}
    lock = threading.Lock()

    def submit_and_record(priority: int, label: str):
        t0 = time.perf_counter()
        r = generate("Explain neural networks:", max_new_tokens=30, priority=priority)
        elapsed = (time.perf_counter() - t0) * 1000
        with lock:
            results.setdefault(label, []).append(elapsed)

    threads = []
    # submit low priority first
    for _ in range(n_per_tier):
        t = threading.Thread(target=submit_and_record, args=(10, "low  (p=10)"))
        threads.append(t)
    time.sleep(0.01)
    # then high priority
    for _ in range(n_per_tier):
        t = threading.Thread(target=submit_and_record, args=(0, "high (p=0) "))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for label, lats in sorted(results.items()):
        print(f"  {label}  mean={statistics.mean(lats):.1f}ms  "
              f"min={min(lats):.1f}ms  max={max(lats):.1f}ms")


# test 6: output length vs tok/s 

def bench_output_lengths():
    print_header("6. Token/s vs output length")
    flush_cache()

    prompt = "Explain gradient descent:"
    for max_tok in [20, 50, 100, 200]:
        r = generate(prompt, max_new_tokens=max_tok)
        print(f"  max_new_tokens={max_tok:>3}  "
              f"generated={r['generated_tokens']:>3}  "
              f"latency={r['latency_ms']:>7.1f}ms  "
              f"{r['tokens_per_sec']:>6.1f} tok/s")


# test 7: cache utilisation under load 

def bench_cache_utilisation(n_requests: int = 6):
    print_header(f"7. Cache utilisation  (n={n_requests} concurrent)")
    flush_cache()

    snapshots = []
    stop_flag = threading.Event()

    def poll():
        while not stop_flag.is_set():
            try:
                snapshots.append(cache_stats())
            except Exception:
                pass
            time.sleep(0.1)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    prompt = "Describe the mathematics behind the softmax function:"
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as ex:
        futures = [ex.submit(generate, prompt, 80) for _ in range(n_requests)]
        concurrent.futures.wait(futures)

    stop_flag.set()
    poller.join(timeout=1)

    if snapshots:
        utils = [s["utilization"] for s in snapshots]
        used  = [s["used_blocks"] for s in snapshots]
        print(f"  snapshots: {len(snapshots)}  "
              f"peak={max(used)} blocks ({max(utils):.3f})  "
              f"mean={statistics.mean(utils):.3f}")
        print(f"  utilisation curve: ", end="")
        step = max(1, len(utils) // 40)
        for u in utils[::step]:
            print("._-:=+*#"[min(int(u * 8), 7)], end="")
        print()
    else:
        print("  no snapshots captured")

    print(f"  final cache: {cache_stats()}")


# main 

if __name__ == "__main__":
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        r.raise_for_status()
    except Exception:
        print(f"ERROR: server not running at {BASE}")
        print("Start with:  uvicorn server.api:app --host 127.0.0.1 --port 8000")
        raise SystemExit(1)

    print(f"\nM2/M3 benchmark suite    {BASE}")
    print(f"initial metrics: {requests.get(f'{BASE}/metrics').json()}")

    bench_sequential(n=8, max_new_tokens=50)
    bench_concurrent(concurrency=4, total=12, max_new_tokens=40)
    bench_scheduler_vs_direct(n=5, max_new_tokens=40)
    bench_prompt_lengths()
    bench_priority(n_per_tier=3)
    bench_output_lengths()
    bench_cache_utilisation(n_requests=6)

    print(f"\n{'=' * 60}")
    print("  benchmark complete")
    print(f"  final scheduler stats: {scheduler_stats()}")
    print(f"{'=' * 60}\n")



