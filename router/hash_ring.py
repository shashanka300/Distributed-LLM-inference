# router/hash_ring.py
import hashlib
import bisect
import threading
from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: str
    host: str
    port: int
    weight: int = 1        # higher weight = more virtual nodes = more traffic

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"


class ConsistentHashRing:
    """
    Virtual-node consistent hash ring.

    Each physical node is replicated `weight * vnodes_per_node` times
    around a 2^32 ring. A key is placed at hash(key) and walks clockwise
    to the nearest virtual node; that virtual node's physical node
    handles the request.

    Adding/removing a node only remaps keys in its arc: O(k/n) fraction
    of all keys, where k = virtual nodes, n = physical nodes.
    Compare this to naive hash(key) % n_nodes, which remaps ~all keys
    when n changes.

    The ring is backed by a sorted list of (ring_position, node_id)
    pairs. Lookup is O(log k) via bisect.
    """

    def __init__(self, vnodes_per_node: int = 150):
        self.vnodes_per_node = vnodes_per_node
        self._ring: list[tuple[int, str]] = []   # (position, node_id)
        self._nodes: dict[str, Node] = {}
        self._lock = threading.RLock()

    # Ring management

    def add_node(self, node: Node) -> None:
        with self._lock:
            if node.node_id in self._nodes:
                return
            self._nodes[node.node_id] = node
            total_vnodes = self.vnodes_per_node * node.weight
            for i in range(total_vnodes):
                pos = self._hash(f"{node.node_id}:vnode:{i}")
                bisect.insort(self._ring, (pos, node.node_id))

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._nodes:
                return
            node = self._nodes.pop(node_id)
            total_vnodes = self.vnodes_per_node * node.weight
            to_remove = set()
            for i in range(total_vnodes):
                pos = self._hash(f"{node_id}:vnode:{i}")
                to_remove.add((pos, node_id))
            self._ring = [x for x in self._ring if x not in to_remove]

    def get_node(self, key: str) -> Node | None:
        """
        Return the node responsible for `key`.
        Walk clockwise from hash(key) to nearest virtual node.
        O(log k) via bisect on the sorted ring.
        """
        with self._lock:
            if not self._ring:
                return None
            pos = self._hash(key)
            idx = bisect.bisect_left(self._ring, (pos, ""))
            if idx >= len(self._ring):
                idx = 0   # wrap around
            _, node_id = self._ring[idx]
            return self._nodes[node_id]

    def get_nodes_for_key(self, key: str, n: int = 2) -> list[Node]:
        """
        Return n distinct nodes starting from hash(key).
        Useful for replication: primary + replica.
        """
        with self._lock:
            if not self._ring:
                return []
            pos = self._hash(key)
            idx = bisect.bisect_left(self._ring, (pos, ""))
            seen, result = set(), []
            for i in range(len(self._ring)):
                _, node_id = self._ring[(idx + i) % len(self._ring)]
                if node_id not in seen:
                    seen.add(node_id)
                    result.append(self._nodes[node_id])
                if len(result) == n:
                    break
            return result

    # Diagnostics

    def distribution(self, n_samples: int = 10_000) -> dict:
        """
        Simulate n_samples random keys and measure how evenly
        traffic would be distributed across nodes.
        Ideal = 1/num_nodes per node. Deviation shows imbalance.
        """
        import random
        if not self._nodes:
            return {}
        counts: dict[str, int] = {nid: 0 for nid in self._nodes}
        for _ in range(n_samples):
            key = str(random.random())
            node = self.get_node(key)
            if node:
                counts[node.node_id] += 1
        ideal = n_samples / len(self._nodes)
        return {
            nid: {
                "count": c,
                "fraction": round(c / n_samples, 3),
                "deviation_pct": round((c - ideal) / ideal * 100, 1),
            }
            for nid, c in counts.items()
        }

    def stats(self) -> dict:
        with self._lock:
            return {
                "nodes": list(self._nodes.keys()),
                "num_nodes": len(self._nodes),
                "num_virtual_nodes": len(self._ring),
                "vnodes_per_node": self.vnodes_per_node,
            }

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


