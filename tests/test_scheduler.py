# tests/test_scheduler.py
import time
import threading
from scheduler.priority_queue import PriorityQueue, Request
from scheduler.rate_limiter import TokenBucket

passed = failed = 0

def run(name, fn):
    global passed, failed
    t0 = time.perf_counter()
    try:
        fn()
        print(f"  PASS  {name:<50} {(time.perf_counter()-t0)*1000:.2f}ms")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name:<50} --> {e}")
        failed += 1

def test_priority_ordering():
    q = PriorityQueue()
    q.push(Request(prompt="low",    priority=5))
    q.push(Request(prompt="high",   priority=0))
    q.push(Request(prompt="medium", priority=2))

    first  = q.pop(timeout=1)
    second = q.pop(timeout=1)
    third  = q.pop(timeout=1)

    assert first.priority  == 0, f"expected 0 got {first.priority}"
    assert second.priority == 2, f"expected 2 got {second.priority}"
    assert third.priority  == 5, f"expected 5 got {third.priority}"
    print(f"        order: {first.prompt}  {second.prompt}  {third.prompt}")

def test_fifo_within_same_priority():
    q = PriorityQueue()
    for i in range(4):
        time.sleep(0.001)   # ensure distinct arrival_time
        q.push(Request(prompt=f"req{i}", priority=0))

    order = [q.pop(timeout=1).prompt for _ in range(4)]
    assert order == ["req0","req1","req2","req3"], f"FIFO broken: {order}"
    print(f"        FIFO order preserved: {order}")

def test_pop_batch():
    q = PriorityQueue()
    for i in range(10):
        q.push(Request(prompt=f"r{i}", priority=0))

    batch = q.pop_batch(max_size=4)
    assert len(batch) == 4
    assert q.size() == 6
    print(f"        popped batch of 4, {q.size()} remaining")

def test_pop_timeout():
    q = PriorityQueue()
    t0 = time.perf_counter()
    result = q.pop(timeout=0.1)
    elapsed = time.perf_counter() - t0
    assert result is None
    assert elapsed >= 0.09, f"returned too fast: {elapsed:.3f}s"
    print(f"        timeout respected: returned None after {elapsed*1000:.1f}ms")

def test_queue_stats():
    q = PriorityQueue()
    for p in [0, 0, 2, 5]:
        q.push(Request(prompt="x", priority=p))
    s = q.stats()
    assert s["queue_depth"] == 4
    assert s["priority_counts"][0] == 2
    print(f"        queue stats: {s}")

def test_token_bucket_allows():
    tb = TokenBucket(rate=100, capacity=10)
    results = [tb.acquire() for _ in range(10)]
    assert all(results), "should allow first 10 (full bucket)"
    assert not tb.acquire(), "11th should be rejected (bucket empty)"
    s = tb.stats()
    print(f"        token bucket stats: {s}")

def test_token_bucket_refill():
    tb = TokenBucket(rate=100, capacity=10)
    for _ in range(10):
        tb.acquire()                # drain
    time.sleep(0.05)                # wait 50ms  5 tokens refilled at 100/s
    results = [tb.acquire() for _ in range(5)]
    assert all(results), "should allow 5 after refill"
    s = tb.stats()
    print(f"        after 50ms refill: {s}")

def test_token_bucket_rejection_rate():
    tb = TokenBucket(rate=1, capacity=3)   # very tight limit
    allowed = sum(1 for _ in range(20) if tb.acquire())
    s = tb.stats()
    assert s["rejection_rate"] > 0
    print(f"        tight bucket  allowed={allowed}/20, stats={s}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Scheduler test suite")
    print("="*60)
    run("priority ordering (0 < 2 < 5)",         test_priority_ordering)
    run("FIFO within same priority",              test_fifo_within_same_priority)
    run("pop_batch respects max_size",            test_pop_batch)
    run("pop returns None on timeout",            test_pop_timeout)
    run("queue stats",                            test_queue_stats)
    run("token bucket allows up to capacity",     test_token_bucket_allows)
    run("token bucket refills over time",         test_token_bucket_refill)
    run("token bucket rejection rate",            test_token_bucket_rejection_rate)
    print("="*60)
    print(f"  {passed} passed   {failed} failed")
    print("="*60 + "\n")
    if failed:
        raise SystemExit(1)


