"""
Rate limiting utilities for API requests.
Provides token bucket algorithm implementation for managing request rates.
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter state."""

    requests_per_minute: int
    current_tokens: float
    last_refill: datetime
    total_requests: int
    throttled_requests: int
    average_wait_time: float


class RateLimiter:
    """
    Token bucket rate limiter.

    Implements a token bucket algorithm to control request rates.
    Tokens are refilled at a constant rate, and requests consume tokens.
    """

    def __init__(self, requests_per_minute: int, burst_limit: Optional[int] = None):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_limit: Maximum burst capacity (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit or requests_per_minute

        # Token bucket state
        self.tokens = float(self.burst_limit)
        self.last_refill = datetime.utcnow()

        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0
        self.wait_times: list[float] = []

        # Refill rate (tokens per second)
        self.refill_rate = requests_per_minute / 60.0

        logger.debug(
            f"Initialized rate limiter: {requests_per_minute} req/min, "
            f"burst={self.burst_limit}, refill_rate={self.refill_rate:.2f} tokens/sec"
        )

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()

        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)

        self.last_refill = now

    def _get_wait_time(self) -> float:
        """
        Calculate how long to wait for a token.

        Returns:
            float: Wait time in seconds (0 if token available)
        """
        if self.tokens >= 1.0:
            return 0.0

        # Calculate wait time for token refill
        tokens_needed = 1.0 - self.tokens
        wait_time = tokens_needed / self.refill_rate

        return wait_time

    async def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.
        Consumes one token from the bucket.
        """
        self._refill_tokens()
        self.total_requests += 1

        wait_time = self._get_wait_time()

        if wait_time > 0:
            self.throttled_requests += 1
            self.wait_times.append(wait_time)

            # Keep only last 100 wait times for average calculation
            if len(self.wait_times) > 100:
                self.wait_times.pop(0)

            logger.debug(".2f")
            await asyncio.sleep(wait_time)

        # Consume token
        self.tokens -= 1.0

    def handle_rate_limit(self) -> None:
        """
        Handle external rate limit response.
        Reduces token count to prevent further requests.
        """
        # Reduce tokens significantly when we hit a rate limit
        self.tokens = max(0, self.tokens - (self.burst_limit * 0.5))
        logger.warning("Rate limit hit, reducing token count")

    def get_stats(self) -> RateLimiterStats:
        """Get current rate limiter statistics."""
        avg_wait = (
            sum(self.wait_times) / len(self.wait_times)
            if self.wait_times else 0.0
        )

        return RateLimiterStats(
            requests_per_minute=self.requests_per_minute,
            current_tokens=self.tokens,
            last_refill=self.last_refill,
            total_requests=self.total_requests,
            throttled_requests=self.throttled_requests,
            average_wait_time=avg_wait,
        )

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self.tokens = float(self.burst_limit)
        self.last_refill = datetime.utcnow()
        self.total_requests = 0
        self.throttled_requests = 0
        self.wait_times.clear()
        logger.debug("Rate limiter reset")


class MultiRateLimiter:
    """
    Manages multiple rate limiters for different endpoints or providers.
    """

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.logger = get_logger(__name__)

    def get_limiter(self, key: str, requests_per_minute: int) -> RateLimiter:
        """Get or create a rate limiter for the given key."""
        if key not in self.limiters:
            self.limiters[key] = RateLimiter(requests_per_minute)
            self.logger.debug(f"Created rate limiter for {key}")

        return self.limiters[key]

    def remove_limiter(self, key: str) -> None:
        """Remove a rate limiter."""
        if key in self.limiters:
            del self.limiters[key]
            self.logger.debug(f"Removed rate limiter for {key}")

    def get_all_stats(self) -> Dict[str, RateLimiterStats]:
        """Get statistics for all rate limiters."""
        return {
            key: limiter.get_stats()
            for key, limiter in self.limiters.items()
        }

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
        self.logger.debug("Reset all rate limiters")


# Global multi-rate limiter instance
multi_rate_limiter = MultiRateLimiter()
