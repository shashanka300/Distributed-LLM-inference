import time
import torch
from core.kv_cache import BlockAllocator, BlockConfig


# helpers 

def make_allocator(num_blocks=16, block_size=4):
    cfg = BlockConfig(
        num_layers=2, num_heads=4, head_dim=8,
        block_size=block_size, num_blocks=num_blocks,
        dtype=torch.float32,
    )
    return BlockAllocator(cfg, device="cpu")

passed = 0
failed = 0

def run(name: str, fn):
    global passed, failed
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  PASS  {name:<45} {ms:.2f}ms")
        passed += 1
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  FAIL  {name:<45} {ms:.2f}ms  -->  {e}")
        failed += 1


# tests 

def test_alloc_and_free():
    alloc = make_allocator(num_blocks=4)
    s0 = alloc.stats()
    assert s0["free_blocks"] == 4, f"expected 4 free, got {s0['free_blocks']}"

    b = alloc.allocate()
    s1 = alloc.stats()
    assert s1["free_blocks"] == 3, f"expected 3 free after alloc, got {s1['free_blocks']}"
    assert s1["used_blocks"] == 1
    assert s1["utilization"] == 0.25

    alloc.free(b)
    s2 = alloc.stats()
    assert s2["free_blocks"] == 4, f"expected 4 free after free, got {s2['free_blocks']}"
    assert s2["used_blocks"] == 0
    assert s2["utilization"] == 0.0

    print(f"        stats after alloc:  {s1}")
    print(f"        stats after free:   {s2}")


def test_multi_alloc():
    alloc = make_allocator(num_blocks=8)
    blocks = [alloc.allocate() for _ in range(5)]
    s = alloc.stats()
    assert s["used_blocks"] == 5
    assert s["free_blocks"] == 3
    print(f"        allocated 5 of 8 blocks: {s}")

    for b in blocks:
        alloc.free(b)
    s2 = alloc.stats()
    assert s2["free_blocks"] == 8
    print(f"        after freeing all 5:     {s2}")


def test_exhaustion_raises():
    alloc = make_allocator(num_blocks=2)
    b0 = alloc.allocate()
    b1 = alloc.allocate()
    s = alloc.stats()
    assert s["free_blocks"] == 0
    print(f"        pool fully used: {s}")

    try:
        alloc.allocate()
        assert False, "should have raised MemoryError"
    except MemoryError as e:
        print(f"        MemoryError correctly raised: '{e}'")

    alloc.free(b0)
    alloc.free(b1)
    s2 = alloc.stats()
    assert s2["free_blocks"] == 2
    print(f"        pool restored after free: {s2}")


def test_prefix_cache_hit():
    alloc = make_allocator(num_blocks=8)
    h = hash("system_prompt_v1")

    b1, hit1 = alloc.get_or_create_prefix_block(h)
    s1 = alloc.stats()
    assert not hit1, "first call should be a miss"
    assert s1["prefix_cache_entries"] == 1
    assert b1.ref_count == 1
    print(f"        first call   hit={hit1}, ref_count={b1.ref_count}, {s1}")

    b2, hit2 = alloc.get_or_create_prefix_block(h)
    s2 = alloc.stats()
    assert hit2, "second call should be a hit"
    assert b1.block_id == b2.block_id, "should return same physical block"
    assert b2.ref_count == 2
    print(f"        second call  hit={hit2}, ref_count={b2.ref_count}, {s2}")

    # cleanup
    alloc.free(b1)
    alloc.free(b2)
    s3 = alloc.stats()
    assert s3["prefix_cache_entries"] == 0
    print(f"        after freeing both refs: {s3}")


def test_copy_on_write():
    alloc = make_allocator(num_blocks=8)
    h = hash("shared_prefix")

    b, _ = alloc.get_or_create_prefix_block(h)
    b2, _ = alloc.get_or_create_prefix_block(h)   # second ref
    assert b.ref_count == 2
    print(f"        shared block ref_count before CoW: {b.ref_count}")

    original_id = b.block_id
    new_b = alloc.copy_on_write(b)

    assert new_b.block_id != original_id, "CoW must produce a new block"
    assert b.ref_count == 1, f"old block ref should drop to 1, got {b.ref_count}"
    assert new_b.ref_count == 1
    s = alloc.stats()
    print(f"        after CoW  old block_id={original_id}, new block_id={new_b.block_id}")
    print(f"        old ref_count={b.ref_count}, new ref_count={new_b.ref_count}, {s}")

    alloc.free(b)
    alloc.free(new_b)


def test_utilization_curve():
    """Allocate blocks one by one and log utilization at each step."""
    n = 8
    alloc = make_allocator(num_blocks=n)
    blocks = []
    print(f"        utilization curve ({n} blocks):")
    for i in range(n):
        blocks.append(alloc.allocate())
        s = alloc.stats()
        bar = "#" * s["used_blocks"] + "." * s["free_blocks"]
        print(f"          [{bar}]  used={s['used_blocks']}  util={s['utilization']:.2f}")
    for b in blocks:
        alloc.free(b)


def test_alloc_speed():
    """Measure raw allocation throughput."""
    n = 512
    alloc = make_allocator(num_blocks=n, block_size=16)
    blocks = []

    t0 = time.perf_counter()
    for _ in range(n):
        blocks.append(alloc.allocate())
    alloc_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    for b in blocks:
        alloc.free(b)
    free_ms = (time.perf_counter() - t1) * 1000

    print(f"        {n} allocs in {alloc_ms:.2f}ms  ({n/alloc_ms*1000:.0f} allocs/sec)")
    print(f"        {n} frees  in {free_ms:.2f}ms   ({n/free_ms*1000:.0f} frees/sec)")
    assert alloc.stats()["free_blocks"] == n


# runner 

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  KV cache test suite")
    print("=" * 60)

    run("alloc and free (basic)",           test_alloc_and_free)
    run("multi alloc and bulk free",        test_multi_alloc)
    run("exhaustion raises MemoryError",    test_exhaustion_raises)
    run("prefix cache hit / miss",          test_prefix_cache_hit)
    run("copy-on-write new block",          test_copy_on_write)
    run("utilization curve",               test_utilization_curve)
    run("alloc/free throughput",           test_alloc_speed)

    print("=" * 60)
    print(f"  {passed} passed   {failed} failed")
    print("=" * 60 + "\n")

    if failed:
        raise SystemExit(1)


