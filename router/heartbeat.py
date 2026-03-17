# router/heartbeat.py
import time
import threading
import logging
import requests

logger = logging.getLogger("heartbeat")


class NodeStatus:
    HEALTHY   = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN   = "unknown"


class HeartbeatMonitor:
    """
    Polls each node's /health endpoint every `interval_s` seconds.
    Marks a node unhealthy after `failure_threshold` consecutive failures.
    Marks it healthy again after `recovery_threshold` consecutive successes.

    Calls on_node_down(node_id) / on_node_up(node_id) callbacks so the
    router can add/remove the node from the hash ring automatically.
    """

    def __init__(
        self,
        interval_s: float = 2.0,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        timeout_s: float = 1.0,
        on_node_down=None,
        on_node_up=None,
    ):
        self.interval_s          = interval_s
        self.failure_threshold   = failure_threshold
        self.recovery_threshold  = recovery_threshold
        self.timeout_s           = timeout_s
        self.on_node_down        = on_node_down or (lambda nid: None)
        self.on_node_up          = on_node_up   or (lambda nid: None)

        self._nodes: dict[str, dict] = {}   # node_id -> {address, status, streak}
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def register(self, node_id: str, address: str) -> None:
        with self._lock:
            self._nodes[node_id] = {
                "address":        address,
                "status":         NodeStatus.UNKNOWN,
                "streak":         0,      # consecutive same-result count
                "last_check":     None,
                "total_checks":   0,
                "total_failures": 0,
            }

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="heartbeat", daemon=True
        )
        self._thread.start()
        logger.info("heartbeat monitor started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def is_healthy(self, node_id: str) -> bool:
        with self._lock:
            n = self._nodes.get(node_id)
            return n is not None and n["status"] == NodeStatus.HEALTHY

    def healthy_nodes(self) -> list[str]:
        with self._lock:
            return [
                nid for nid, n in self._nodes.items()
                if n["status"] == NodeStatus.HEALTHY
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                nid: {
                    "status":         n["status"],
                    "streak":         n["streak"],
                    "last_check":     n["last_check"],
                    "total_checks":   n["total_checks"],
                    "total_failures": n["total_failures"],
                    "failure_rate":   round(
                        n["total_failures"] / n["total_checks"], 3
                    ) if n["total_checks"] > 0 else 0.0,
                }
                for nid, n in self._nodes.items()
            }

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                node_ids = list(self._nodes.keys())

            for node_id in node_ids:
                self._check(node_id)

            time.sleep(self.interval_s)

    def _check(self, node_id: str) -> None:
        with self._lock:
            n = self._nodes.get(node_id)
            if n is None:
                return
            address = n["address"]

        try:
            r = requests.get(
                f"{address}/health",
                timeout=self.timeout_s
            )
            success = r.status_code == 200
        except Exception:
            success = False

        with self._lock:
            n = self._nodes[node_id]
            n["total_checks"] += 1
            n["last_check"] = time.time()
            if not success:
                n["total_failures"] += 1

            prev_status = n["status"]

            if success:
                if prev_status == NodeStatus.HEALTHY:
                    n["streak"] += 1
                else:
                    n["streak"] = 1 if n["streak"] <= 0 else n["streak"] + 1
                    if n["streak"] >= self.recovery_threshold:
                        n["status"] = NodeStatus.HEALTHY
                        logger.info(f"node {node_id} is UP")
                        threading.Thread(
                            target=self.on_node_up, args=(node_id,), daemon=True
                        ).start()
            else:
                if prev_status != NodeStatus.HEALTHY:
                    n["streak"] -= 1
                else:
                    n["streak"] = -1 if n["streak"] >= 0 else n["streak"] - 1
                    if abs(n["streak"]) >= self.failure_threshold:
                        n["status"] = NodeStatus.UNHEALTHY
                        logger.warning(f"node {node_id} is DOWN")
                        threading.Thread(
                            target=self.on_node_down, args=(node_id,), daemon=True
                        ).start()


