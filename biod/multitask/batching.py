"""Bounded microbatch queue for a single inference device."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic


@dataclass
class Work:
    key: tuple
    item: object
    tokens: int
    submitted: float
    future: Future


class InferenceQueue:
    def __init__(self, predict, *, max_size, wait_ms, max_tokens, capacity=128):
        self.predict = predict
        self.max_size = max_size
        self.wait_seconds = wait_ms / 1000
        self.max_tokens = max_tokens
        self.capacity = capacity
        self.pending = deque()
        self.condition = Condition()
        Thread(target=self._worker, name="biod-microbatch", daemon=True).start()

    def depth(self):
        with self.condition:
            return len(self.pending)

    def submit(self, key, item, tokens):
        work = Work(key, item, tokens, monotonic(), Future())
        with self.condition:
            if len(self.pending) >= self.capacity:
                raise RuntimeError("decision inference queue is full")
            self.pending.append(work)
            self.condition.notify()
        return work.future.result()

    def _take(self):
        with self.condition:
            while not self.pending:
                self.condition.wait()
            first = self.pending[0]
            remaining = first.submitted + self.wait_seconds - monotonic()
            while len(self.pending) < self.max_size and remaining > 0:
                self.condition.wait(remaining)
                remaining = first.submitted + self.wait_seconds - monotonic()
            batch = [self.pending.popleft()]
            longest = first.tokens
            for work in list(self.pending):
                if len(batch) >= self.max_size:
                    break
                length = max(longest, work.tokens)
                if work.key == first.key and length * (len(batch) + 1) <= self.max_tokens:
                    self.pending.remove(work)
                    batch.append(work)
                    longest = length
            return batch

    def _worker(self):
        while True:
            batch = self._take()
            started = monotonic()
            try:
                outputs = self.predict(batch[0].key, [work.item for work in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("decision batch output size mismatch")
            except Exception as error:
                for work in batch:
                    work.future.set_exception(error)
                continue
            elapsed = (monotonic() - started) * 1000
            for work, output in zip(batch, outputs):
                work.future.set_result((output, {
                    "queueMs": round((started - work.submitted) * 1000, 2),
                    "inferMs": round(elapsed, 2),
                    "batchSize": len(batch),
                }))
