# tests/test_hash_ring.py
import time
from router.hash_ring import ConsistentHashRing, Node

passed = failed = 0

def run(name, fn):
    global passed, failed
    t0 = time.perf_counter()
    try:
        fn()
        print(f"  PASS  {name:<55} {(time.perf_counter()-t0)*1000:.2f}ms")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name:<55} --> {e}")
        failed += 1

def make_ring(n_nodes=3):
    ring = ConsistentHashRing(vnodes_per_node=150)
    for i in range(n_nodes):
        ring.add_node(Node(f"node-{i}", "127.0.0.1", 8000 + i))
    return ring

def test_basic_routing():
    ring = make_ring(3)
    node = ring.get_node("session-abc")
    assert node is not None
    assert node.node_id.startswith("node-")
    print(f"        session-abc  {node.node_id}")

def test_same_key_same_node():
    ring = make_ring(3)
    results = {ring.get_node("session-xyz").node_id for _ in range(100)}
    assert len(results) == 1, f"key mapped to multiple nodes: {results}"
    print(f"        100 lookups of same key  always {results.pop()}")

def test_minimal_remapping_on_add():
    ring = make_ring(3)
    keys = [f"session-{i}" for i in range(1000)]
    before = {k: ring.get_node(k).node_id for k in keys}

    ring.add_node(Node("node-3", "127.0.0.1", 8003))
    after = {k: ring.get_node(k).node_id for k in keys}

    remapped = sum(1 for k in keys if before[k] != after[k])
    pct = remapped / len(keys) * 100
    ideal = 1 / 4 * 100   # adding 1 node to 3  expect ~25% remapped
    print(f"        remapped {remapped}/1000 ({pct:.1f}%)  ideal{ideal:.1f}%")
    assert pct < ideal * 1.5, f"too many remapped: {pct:.1f}% > {ideal*1.5:.1f}%"

def test_minimal_remapping_on_remove():
    ring = make_ring(4)
    keys = [f"session-{i}" for i in range(1000)]
    before = {k: ring.get_node(k).node_id for k in keys}

    ring.remove_node("node-2")
    after = {k: ring.get_node(k).node_id for k in keys}

    remapped = sum(1 for k in keys if before[k] != after[k])
    pct = remapped / len(keys) * 100
    ideal = 1 / 4 * 100
    print(f"        remapped {remapped}/1000 ({pct:.1f}%)  ideal{ideal:.1f}%")
    assert pct < ideal * 1.5, f"too many remapped: {pct:.1f}% > {ideal*1.5:.1f}%"

def test_empty_ring_returns_none():
    ring = ConsistentHashRing()
    assert ring.get_node("anything") is None
    print(f"        empty ring correctly returns None")

def test_single_node_handles_all():
    ring = ConsistentHashRing(vnodes_per_node=50)
    ring.add_node(Node("solo", "127.0.0.1", 9000))
    keys = [f"k{i}" for i in range(200)]
    nodes = {ring.get_node(k).node_id for k in keys}
    assert nodes == {"solo"}
    print(f"        all 200 keys  solo")

def test_distribution_evenness():
    ring = make_ring(3)
    dist = ring.distribution(n_samples=10_000)
    print(f"        distribution across 3 nodes (10k samples):")
    for nid, d in dist.items():
        bar = "#" * int(d["fraction"] * 40)
        print(f"          {nid}: {bar} {d['fraction']:.3f} "
              f"(deviation {d['deviation_pct']:+.1f}%)")
    deviations = [abs(d["deviation_pct"]) for d in dist.values()]
    assert max(deviations) < 15, f"distribution too uneven: {deviations}"

def test_replica_nodes():
    ring = make_ring(3)
    replicas = ring.get_nodes_for_key("session-foo", n=2)
    assert len(replicas) == 2
    assert replicas[0].node_id != replicas[1].node_id
    print(f"        primary={replicas[0].node_id}  replica={replicas[1].node_id}")

def test_weight_shifts_distribution():
    ring = ConsistentHashRing(vnodes_per_node=100)
    ring.add_node(Node("small", "127.0.0.1", 8001, weight=1))
    ring.add_node(Node("large", "127.0.0.1", 8002, weight=3))
    dist = ring.distribution(n_samples=10_000)
    large_frac = dist["large"]["fraction"]
    small_frac = dist["small"]["fraction"]
    print(f"        weight 1 vs 3  small={small_frac:.3f}  large={large_frac:.3f}")
    assert large_frac > small_frac * 2, "weighted node should get more traffic"

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Hash ring test suite")
    print("=" * 60)
    run("basic routing returns a node",              test_basic_routing)
    run("same key always maps to same node",         test_same_key_same_node)
    run("minimal remapping on node add",             test_minimal_remapping_on_add)
    run("minimal remapping on node remove",          test_minimal_remapping_on_remove)
    run("empty ring returns None",                   test_empty_ring_returns_none)
    run("single node handles all keys",              test_single_node_handles_all)
    run("distribution evenness across 3 nodes",     test_distribution_evenness)
    run("replica nodes are distinct",                test_replica_nodes)
    run("weighted node gets proportional traffic",   test_weight_shifts_distribution)
    print("=" * 60)
    print(f"  {passed} passed   {failed} failed")
    print("=" * 60 + "\n")
    if failed:
        raise SystemExit(1)


