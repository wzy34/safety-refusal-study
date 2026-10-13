"""Per-provider concurrency cap (semaphore) and request-rate limit (token bucket)."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager


class TokenBucket:
    def __init__(self, rate_per_s: float, capacity: float):
        self.rate = rate_per_s
        self.capacity = max(1.0, capacity)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                await asyncio.sleep((1 - self._tokens) / self.rate)


class ProviderLimiter:
    def __init__(self, max_concurrency: int, requests_per_minute: float):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.bucket = TokenBucket(requests_per_minute / 60.0, capacity=max_concurrency)

    @asynccontextmanager
    async def slot(self):
        async with self.semaphore:
            await self.bucket.acquire()
            yield
