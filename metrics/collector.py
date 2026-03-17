# metrics/collector.py
import time
import threading
import statistics
import collections
import requests
import logging

logger = logging.getLogger("metrics")


class LatencyWindow:
    """
    Rolling window of latency samples.
    Keeps the last `maxlen` values and computes percentiles on demand.
    Uses a deque with O(1) append and O(1) pop from left.
    """

    def __init__(self, maxlen: int = 200):
        self._data: collections.deque[float] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, value_ms: float):
        with self._lock:
            self._data.append(value_ms)

    def stats(self) -> dict:
        with self._lock:
            if not self._data:
                return {"n": 0}
            data = sorted(self._data)
            n = len(data)
            return {
                "n":      n,
                "mean":   round(statistics.mean(data), 1),
                "median": round(statistics.median(data), 1),
                "p95":    round(data[max(int(n * 0.95) - 1, 0)], 1),
                "p99":    round(data[max(int(n * 0.99) - 1, 0)], 1),
                "min":    round(min(data), 1),
                "max":    round(max(data), 1),
                "stdev":  round(statistics.stdev(data), 1) if n > 1 else 0.0,
            }


class AnomalyDetector:
    """
    Z-score anomaly detection on a rolling window.

    A sample is anomalous if:
        |sample - mean| > threshold * stdev

    This is the simplest form of statistical process control.
    The threshold of 3.0 corresponds to the 99.7% rule for a
    normal distribution; values beyond 3 standard deviations
    are flagged as anomalies.

    In probability terms: P(|X - mean| > 3 * stdev) ~= 0.003
    """

    def __init__(self, window: int = 100, threshold: float = 3.0):
        self._window  = collections.deque(maxlen=window)
        self._threshold = threshold
        self._lock    = threading.Lock()
        self.anomaly_count = 0

    def check(self, value: float) -> tuple[bool, float]:
        """
        Returns (is_anomaly, z_score).
        Records value into the window after checking.
        """
        with self._lock:
            if len(self._window) < 10:
                self._window.append(value)
                return False, 0.0

            mean  = statistics.mean(self._window)
            stdev = statistics.stdev(self._window)

            if stdev == 0:
                self._window.append(value)
                return False, 0.0

            z = (value - mean) / stdev
            is_anomaly = abs(z) > self._threshold

            if is_anomaly:
                self.anomaly_count += 1
                logger.warning(
                    f"anomaly detected: value={value:.1f} "
                    f"mean={mean:.1f} stdev={stdev:.1f} z={z:.2f}"
                )

            self._window.append(value)
            return is_anomaly, round(z, 2)


class MetricsCollector:
    """
    Polls all worker nodes every `interval_s` seconds.
    Aggregates latency, queue depth, token throughput across the cluster.
    """

    def __init__(self, worker_urls: list[str], interval_s: float = 3.0):
        self.worker_urls = worker_urls
        self.interval_s  = interval_s

        self.latency    = LatencyWindow(maxlen=500)
        self.queue_depth = LatencyWindow(maxlen=200)   # reusing for queue depth
        self.anomaly     = AnomalyDetector(window=100, threshold=3.0)

        self._snapshots: collections.deque[dict] = collections.deque(maxlen=100)
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # per-node history
        self._node_stats: dict[str, dict] = {}

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="metrics", daemon=True
        )
        self._thread.start()
        logger.info("metrics collector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while self._running:
            self._collect()
            time.sleep(self.interval_s)

    def _collect(self):
        cluster_queue_depth = 0
        cluster_tokens      = 0
        cluster_batches     = 0

        for url in self.worker_urls:
            try:
                r = requests.get(f"{url}/metrics", timeout=2)
                if r.status_code != 200:
                    continue
                data = r.json()

                sched = data.get("scheduler", {})
                queue = sched.get("queue", {})
                depth = queue.get("queue_depth", 0)
                tokens = sched.get("total_tokens_generated", 0)
                batches = sched.get("batches_processed", 0)
                avg_lat = sched.get("avg_batch_latency_ms", 0)

                cluster_queue_depth += depth
                cluster_tokens      += tokens
                cluster_batches     += batches

                if avg_lat > 0:
                    self.latency.record(avg_lat)
                    is_anom, z = self.anomaly.check(avg_lat)

                with self._lock:
                    self._node_stats[url] = {
                        "queue_depth":   depth,
                        "tokens":        tokens,
                        "batches":       batches,
                        "avg_latency_ms": avg_lat,
                        "cache":         data.get("cache", {}),
                        "ts":            time.time(),
                    }

            except Exception as e:
                logger.debug(f"failed to collect from {url}: {e}")

        self.queue_depth.record(cluster_queue_depth)

        snapshot = {
            "ts":                  time.time(),
            "cluster_queue_depth": cluster_queue_depth,
            "cluster_tokens":      cluster_tokens,
            "cluster_batches":     cluster_batches,
            "latency":             self.latency.stats(),
            "anomalies":           self.anomaly.anomaly_count,
        }
        with self._lock:
            self._snapshots.append(snapshot)

    def current(self) -> dict:
        with self._lock:
            latest = self._snapshots[-1] if self._snapshots else {}
            return {
                "latest_snapshot":  latest,
                "latency_window":   self.latency.stats(),
                "queue_window":     self.queue_depth.stats(),
                "anomaly_count":    self.anomaly.anomaly_count,
                "node_stats":       dict(self._node_stats),
            }

    def queue_depth_now(self) -> float:
        with self._lock:
            if not self._snapshots:
                return 0.0
            return self._snapshots[-1].get("cluster_queue_depth", 0.0)


