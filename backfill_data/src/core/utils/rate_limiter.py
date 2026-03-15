"""Shared token bucket rate limiter used by all providers"""

import time
import threading
from typing import Optional
import logging

from src.core.interfaces import RateLimiterInterface

logger = logging.getLogger('backfill_system')


class TokenBucketRateLimiter(RateLimiterInterface):
    """
    Thread-safe token bucket rate limiter.

    All providers share this implementation — only the default parameters
    and logger name differ.  Subclass and override ``provider_name`` if
    you need a distinct log prefix; otherwise instantiate directly.
    """

    provider_name: str = "provider"

    def __init__(
        self,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        safety_margin: float = 0.9,
    ):
        self.requests_per_window = int(requests_per_window * safety_margin)
        self.window_seconds = window_seconds
        self.safety_margin = safety_margin

        self.tokens = float(self.requests_per_window)
        self.max_tokens = float(self.requests_per_window)
        self.refill_rate = self.max_tokens / self.window_seconds

        self.lock = threading.Lock()
        self.last_refill = time.time()

        self.total_requests = 0
        self.total_wait_time = 0.0

        self._log = logging.getLogger(f'backfill_system.{self.provider_name}')
        self._log.info(
            f"{self.provider_name} rate limiter: {self.requests_per_window} "
            f"req/{window_seconds}s (safety margin: {safety_margin})"
        )

    def _refill_tokens(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self) -> None:
        with self.lock:
            self._refill_tokens()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self.total_requests += 1
                return
            tokens_needed = 1.0 - self.tokens
            wait_time = tokens_needed / self.refill_rate
            self._log.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            self.total_wait_time += wait_time

        time.sleep(wait_time)

        with self.lock:
            self._refill_tokens()
            self.tokens -= 1.0
            self.total_requests += 1

    def try_acquire(self) -> bool:
        with self.lock:
            self._refill_tokens()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self.total_requests += 1
                return True
            return False

    def get_current_rate(self) -> float:
        with self.lock:
            return self.refill_rate

    def get_remaining_quota(self) -> Optional[int]:
        with self.lock:
            self._refill_tokens()
            return int(self.tokens)

    def get_statistics(self) -> dict:
        with self.lock:
            return {
                'total_requests': self.total_requests,
                'total_wait_time': self.total_wait_time,
                'avg_wait_time': (
                    self.total_wait_time / self.total_requests
                    if self.total_requests > 0 else 0
                ),
                'available_tokens': int(self.tokens),
                'max_tokens': int(self.max_tokens),
            }

    def reset(self) -> None:
        with self.lock:
            self.tokens = float(self.max_tokens)
            self.last_refill = time.time()
            self.total_requests = 0
            self.total_wait_time = 0.0
            self._log.debug(f"{self.provider_name} rate limiter reset")
