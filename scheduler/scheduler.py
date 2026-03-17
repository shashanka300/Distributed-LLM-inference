# scheduler/scheduler.py
import time
import threading
import concurrent.futures
import logging
from typing import Optional

from scheduler.priority_queue import PriorityQueue, Request
from scheduler.rate_limiter import TokenBucket

logger = logging.getLogger("scheduler")


class Scheduler:
    """
    Continuous batching scheduler.

    Background worker thread:
      1. Pulls up to `max_batch_size` requests from the priority queue
      2. Pads/groups prompts into a single batch
      3. Runs one model forward pass for the whole batch
      4. Resolves each request's Future with its result

    API threads submit requests and block on a Future until the
    scheduler resolves it, providing clean separation of HTTP handling and
    model execution.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,   # how long to wait for batch to fill
        rate: float = 50.0,          # requests/sec allowed
        burst: float = 20.0,         # burst capacity
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.queue = PriorityQueue()
        self.rate_limiter = TokenBucket(rate=rate, capacity=burst)

        # metrics
        self._batches_processed = 0
        self._requests_processed = 0
        self._total_tokens_generated = 0
        self._total_batch_latency_ms = 0.0
        self._lock = threading.Lock()

        # worker thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # Lifecycle

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, name="scheduler-worker", daemon=True
        )
        self._thread.start()
        logger.info("scheduler started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("scheduler stopped")

    # Public API

    def submit(self, request: Request) -> concurrent.futures.Future:
        """
        Submit a request. Returns a Future that resolves when generation completes.
        The calling thread blocks on future.result(); the scheduler resolves it.
        """
        if not self.rate_limiter.acquire():
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_exception(
                RuntimeError("rate limit exceeded - try again shortly")
            )
            return fut

        fut: concurrent.futures.Future = concurrent.futures.Future()
        request._future = fut
        self.queue.push(request)
        return fut

    # Worker loop

    def _worker_loop(self):
        logger.info("worker loop running")
        while self._running:
            # wait for at least one request
            first = self.queue.pop(timeout=0.1)
            if first is None:
                continue

            # collect more requests up to max_batch_size within max_wait_ms
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                more = self.queue.pop(timeout=remaining)
                if more is None:
                    break
                batch.append(more)

            self._process_batch(batch)

    def _process_batch(self, batch: list[Request]):
        t0 = time.perf_counter()
        logger.debug(f"processing batch of {len(batch)}")

        try:
            results = self._run_batch(batch)
            for req, result in zip(batch, results):
                req._future.set_result(result)
        except Exception as e:
            logger.error(f"batch failed: {e}")
            for req in batch:
                req._future.set_exception(e)

        batch_ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._batches_processed += 1
            self._requests_processed += len(batch)
            self._total_tokens_generated += sum(
                r.get("generated_tokens", 0)
                for r in [req._future.result() for req in batch
                          if req._future.done() and not req._future.exception()]
            )
            self._total_batch_latency_ms += batch_ms

    def _run_batch(self, batch: list[Request]) -> list[dict]:
        """
        Run the model on a batch of requests.
        Pads inputs to the same length, runs one forward pass,
        then trims each output back to its own length.
        """
        import torch

        # tokenize all prompts
        encodings = [
            self.tokenizer(req.prompt, return_tensors="pt")
            for req in batch
        ]

        if len(batch) == 1:
            # fast path: single request, no padding needed
            req = batch[0]
            enc = encodings[0].to(self.model.device)
            input_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=True,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    use_cache=True,
                )
            new_ids = out[0][input_len:]
            return [{
                "text": self.tokenizer.decode(new_ids, skip_special_tokens=True),
                "prompt_tokens": input_len,
                "generated_tokens": len(new_ids),
            }]

        # multi-request: pad to longest prompt, run together
        max_len = max(e["input_ids"].shape[1] for e in encodings)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full(
            (len(batch), max_len), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, enc in enumerate(encodings):
            seq_len = enc["input_ids"].shape[1]
            # right-pad: fill from position 0
            input_ids[i, :seq_len] = enc["input_ids"][0]
            attention_mask[i, :seq_len] = 1

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # use the shortest max_new_tokens in the batch so we don't
        # over-generate for short requests
        min_max_tokens = min(req.max_new_tokens for req in batch)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min_max_tokens,
                do_sample=True,
                temperature=batch[0].temperature,  # use first req's params
                top_p=batch[0].top_p,
                use_cache=True,
                pad_token_id=pad_id,
            )

        results = []
        for i, req in enumerate(batch):
            prompt_len = encodings[i]["input_ids"].shape[1]
            new_ids = out[i][max_len:]   # skip the padded prompt
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append({
                "text": text,
                "prompt_tokens": prompt_len,
                "generated_tokens": len(new_ids),
            })
        return results

    # Stats

    def stats(self) -> dict:
        with self._lock:
            avg_batch = (
                self._requests_processed / self._batches_processed
                if self._batches_processed > 0 else 0
            )
            avg_latency = (
                self._total_batch_latency_ms / self._batches_processed
                if self._batches_processed > 0 else 0
            )
        return {
            "queue": self.queue.stats(),
            "rate_limiter": self.rate_limiter.stats(),
            "batches_processed": self._batches_processed,
            "requests_processed": self._requests_processed,
            "total_tokens_generated": self._total_tokens_generated,
            "avg_batch_size": round(avg_batch, 2),
            "avg_batch_latency_ms": round(avg_latency, 1),
        }


