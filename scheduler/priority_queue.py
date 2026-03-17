# scheduler/priority_queue.py
import heapq
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class Request:
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95
    priority: int = 0          # lower = higher priority (min-heap)
    arrival_time: float = field(default_factory=time.perf_counter)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # result filled in by scheduler after generation
    result: Optional[dict] = field(default=None, repr=False)
    error: Optional[str] = field(default=None, repr=False)

    # Heap comparison sorts by (priority, arrival_time) so equal-priority
    # requests are served FIFO
    def __lt__(self, other):
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)

    def __eq__(self, other):
        return self.request_id == other.request_id


class PriorityQueue:
    """
    Thread-safe min-heap priority queue.
    Lower priority value = served first (0 = highest priority).
    Equal priority = FIFO by arrival_time.
    """

    def __init__(self):
        self._heap: list[Request] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push(self, request: Request) -> None:
        with self._not_empty:
            heapq.heappush(self._heap, request)
            self._not_empty.notify()          # wake up any waiting pop()

    def pop(self, timeout: float = 1.0) -> Optional[Request]:
        """Block until a request is available or timeout expires."""
        with self._not_empty:
            deadline = time.perf_counter() + timeout
            while not self._heap:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)
            return heapq.heappop(self._heap)

    def pop_batch(self, max_size: int = 8) -> list[Request]:
        """Pop up to max_size requests at once; used by the batch loop."""
        with self._not_empty:
            batch = []
            while self._heap and len(batch) < max_size:
                batch.append(heapq.heappop(self._heap))
            return batch

    def peek(self) -> Optional[Request]:
        with self._lock:
            return self._heap[0] if self._heap else None

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def stats(self) -> dict:
        with self._lock:
            priorities = [r.priority for r in self._heap]
            wait_times = [time.perf_counter() - r.arrival_time for r in self._heap]
            return {
                "queue_depth": len(self._heap),
                "priority_counts": {
                    p: priorities.count(p) for p in set(priorities)
                } if priorities else {},
                "max_wait_ms": round(max(wait_times) * 1000, 1) if wait_times else 0,
                "mean_wait_ms": round(
                    sum(wait_times) / len(wait_times) * 1000, 1
                ) if wait_times else 0,
            }


