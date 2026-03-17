# tests/bench_m1.py
import time, requests, statistics, concurrent.futures

URL = "http://localhost:8000/generate"
PROMPT = "The key insight of the transformer architecture is"
N_REQUESTS = 10

def single_request():
    t0 = time.perf_counter()
    r = requests.post(URL, json={"prompt": PROMPT, "max_new_tokens": 50})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, r.json()["generated_tokens"]

# sequential
print("sequential:")
latencies = []
for i in range(N_REQUESTS):
    ms, toks = single_request()
    latencies.append(ms)
    print(f"  req {i+1}: {ms:.0f}ms, {toks} tokens, {toks/(ms/1000):.1f} tok/s")

print(f"\np50={statistics.median(latencies):.0f}ms  "
      f"p95={sorted(latencies)[int(N_REQUESTS*0.95)-1]:.0f}ms  "
      f"mean={statistics.mean(latencies):.0f}ms")


