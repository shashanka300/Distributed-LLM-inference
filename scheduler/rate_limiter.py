# scheduler/rate_limiter.py
import time
import threading


class TokenBucket:
    """
    Classic token bucket for rate limiting.
    Tokens refill at `rate` per second up to `capacity`.
    Each request costs `cost` tokens (default 1).

    This is O(1) per check: just a counter and a timestamp.
    The interesting bit is that it naturally handles bursts:
    if no requests come in for 5 seconds, 5*rate tokens accumulate,
    allowing a short burst before throttling kicks in.
    """

    def __init__(self, rate: float, capacity: float):
        self.rate = rate          # tokens added per second
        self.capacity = capacity  # max tokens (burst ceiling)
        self._tokens = capacity   # start full
        self._last_refill = time.perf_counter()
        self._lock = threading.Lock()
        self._total_allowed = 0
        self._total_rejected = 0

    def _refill(self):
        """Add tokens based on elapsed time. Called inside lock."""
        now = time.perf_counter()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def acquire(self, cost: float = 1.0) -> bool:
        """
        Try to consume `cost` tokens.
        Returns True if allowed, False if rate limit exceeded.
        """
        with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                self._total_allowed += 1
                return True
            self._total_rejected += 1
            return False

    def stats(self) -> dict:
        with self._lock:
            self._refill()
            total = self._total_allowed + self._total_rejected
            return {
                "tokens_available": round(self._tokens, 2),
                "capacity": self.capacity,
                "rate_per_sec": self.rate,
                "total_allowed": self._total_allowed,
                "total_rejected": self._total_rejected,
                "rejection_rate": round(
                    self._total_rejected / total, 3
                ) if total > 0 else 0.0,
            }


