"""Thread-safe rate limiter for Dune Analytics API"""

import time
import threading
from typing import Optional
import logging
from src.core.interfaces import RateLimiterInterface

logger = logging.getLogger('backfill_system.dune')


class DuneRateLimiter(RateLimiterInterface):
    """
    Token bucket rate limiter for Dune Analytics API.

    Dune standard tier: ~40 requests per minute.
    Premium tiers may have higher limits.
    """

    def __init__(
        self,
        requests_per_window: int = 40,
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

        logger.info(
            f"Dune rate limiter: {self.requests_per_window} req/{window_seconds}s "
            f"(safety margin: {safety_margin})"
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
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
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
            logger.debug("Dune rate limiter reset")
