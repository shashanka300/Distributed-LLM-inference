"""
metrics/autoscaler.py - M5 autoscaler with ring registration

When a new worker is scaled up:
  1. Spawns the uvicorn process
  2. Polls /health until the worker is ready (model loaded)
  3. Registers the new node into the hash ring + heartbeat monitor
     so it actually receives traffic

This closes the loop from M4: new workers are fully integrated,
not just spawned and ignored.
"""

import time
import threading
import subprocess
import sys
import logging
import requests

logger = logging.getLogger("autoscaler")


class Autoscaler:
    """
    Ties together the MetricsCollector and PIDController.

    Every tick_s seconds:
      1. Read cluster queue depth from MetricsCollector
      2. Feed it into the PID controller
      3. If output >= scale_up_threshold:   spin up a new worker
      4. If output <= scale_down_threshold: spin down least-busy owned worker

    New workers are registered into the hash ring + heartbeat monitor
    once their /health endpoint responds, so traffic flows to them
    automatically.

    Pass ring and monitor from router.router so the autoscaler can
    mutate them directly. If not passed (e.g. in unit tests), scale
    events are still recorded but ring registration is skipped.
    """

    def __init__(
        self,
        collector,
        pid,
        min_workers: int   = 1,
        max_workers: int   = 4,
        tick_s: float      = 5.0,
        scale_up_threshold: float   =  0.6,
        scale_down_threshold: float = -0.6,
        cooldown_s: float  = 30.0,
        ring=None,          # router.hash_ring.ConsistentHashRing instance
        monitor=None,       # router.heartbeat.HeartbeatMonitor instance
    ):
        self.collector   = collector
        self.pid         = pid
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.tick_s      = tick_s
        self.scale_up_threshold   = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_s  = cooldown_s
        self.ring        = ring
        self.monitor     = monitor

        # node_id -> Popen | None  (None = externally managed)
        self._workers: dict[str, subprocess.Popen | None] = {}
        self._next_port   = 8003
        self._last_scale  = 0.0
        self._scale_events: list[dict] = []
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # Lifecycle

    def register_existing(self, node_id: str, port: int) -> None:
        """
        Register a worker that was started externally (e.g. start_cluster.py).
        These workers are tracked for headcount but never terminated by the autoscaler.
        """
        with self._lock:
            self._workers[node_id] = None
            self._next_port = max(self._next_port, port + 1)

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="autoscaler", daemon=True
        )
        self._thread.start()
        logger.info("autoscaler started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    # Tick loop

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"autoscaler tick error: {e}")
            time.sleep(self.tick_s)

    def _tick(self) -> None:
        queue_depth = self.collector.queue_depth_now()
        output      = self.pid.update(queue_depth)

        with self._lock:
            n_workers  = len(self._workers)
            in_cooldown = (time.time() - self._last_scale) < self.cooldown_s

        logger.info(
            f"tick: queue={queue_depth:.1f}  pid={output:.3f}  "
            f"workers={n_workers}  cooldown={in_cooldown}"
        )

        if in_cooldown:
            return

        if output >= self.scale_up_threshold and n_workers < self.max_workers:
            self._scale_up()
        elif output <= self.scale_down_threshold and n_workers > self.min_workers:
            self._scale_down()

    # Scale up

    def _scale_up(self) -> None:
        with self._lock:
            port      = self._next_port
            self._next_port += 1
            node_id   = f"worker-auto-{port}"
            self._last_scale = time.time()
            self._scale_events.append({
                "ts":          time.time(),
                "action":      "scale_up",
                "node_id":     node_id,
                "port":        port,
                "queue_depth": self.collector.queue_depth_now(),
            })

        logger.info(f"scaling UP - starting {node_id} on port {port}")

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.api:app",
             "--host", "127.0.0.1", "--port", str(port)],
            **kwargs,
        )

        with self._lock:
            self._workers[node_id] = proc

        logger.info(f"process started for {node_id}, waiting for health check...")

        # register into ring + monitor once worker is ready
        # runs in background so the autoscaler tick doesn't block
        threading.Thread(
            target=self._register_when_ready,
            args=(node_id, f"http://127.0.0.1:{port}", port),
            daemon=True,
            name=f"register-{node_id}",
        ).start()

    def _register_when_ready(
        self, node_id: str, address: str, port: int,
        timeout_s: int = 120, poll_interval_s: float = 3.0,
    ) -> None:
        """
        Poll /health until the worker responds, then register it
        into the hash ring and heartbeat monitor so it receives traffic.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                r = requests.get(f"{address}/health", timeout=2)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    logger.info(f"{node_id} is healthy - registering in ring")
                    self._add_to_ring(node_id, address, port)
                    return
            except Exception:
                pass
            time.sleep(poll_interval_s)

        logger.warning(
            f"{node_id} did not become healthy within {timeout_s}s "
            f" - not added to ring"
        )

    def _add_to_ring(self, node_id: str, address: str, port: int) -> None:
        """Add a new node to the consistent hash ring and heartbeat monitor."""
        if self.ring is None or self.monitor is None:
            logger.debug(
                "ring/monitor not set - skipping ring registration "
                "(normal in unit tests)"
            )
            return

        try:
            from router.hash_ring import Node as RingNode
            self.ring.add_node(RingNode(
                node_id=node_id,
                host="127.0.0.1",
                port=port,
            ))
            self.monitor.register(node_id, address)
            logger.info(
                f"{node_id} added to ring - "
                f"ring now has {self.ring.stats()['num_nodes']} nodes"
            )
        except Exception as e:
            logger.error(f"failed to add {node_id} to ring: {e}")

    # Scale down

    def _scale_down(self) -> None:
        with self._lock:
            # only terminate workers we own (have a Popen handle)
            owned = {k: v for k, v in self._workers.items() if v is not None}
            if not owned:
                logger.info("scale down requested but no owned workers to stop")
                return
            node_id, proc = next(iter(owned.items()))
            del self._workers[node_id]
            self._last_scale = time.time()
            self._scale_events.append({
                "ts":          time.time(),
                "action":      "scale_down",
                "node_id":     node_id,
                "queue_depth": self.collector.queue_depth_now(),
            })

        logger.info(f"scaling DOWN - stopping {node_id}")

        # remove from ring before terminating so no new requests route there
        if self.ring is not None:
            try:
                self.ring.remove_node(node_id)
                logger.info(f"{node_id} removed from ring")
            except Exception as e:
                logger.error(f"failed to remove {node_id} from ring: {e}")

        proc.terminate()
        logger.info(f"{node_id} terminated")

    # Stats

    def stats(self) -> dict:
        with self._lock:
            cooldown_remaining = max(
                0.0, self.cooldown_s - (time.time() - self._last_scale)
            )
            return {
                "active_workers":       len(self._workers),
                "worker_ids":           list(self._workers.keys()),
                "min_workers":          self.min_workers,
                "max_workers":          self.max_workers,
                "scale_up_threshold":   self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "cooldown_s":           self.cooldown_s,
                "cooldown_remaining_s": round(cooldown_remaining, 1),
                "scale_events":         list(self._scale_events[-10:]),
                "pid":                  self.pid.stats(),
            }



