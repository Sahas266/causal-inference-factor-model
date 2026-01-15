"""Thread-safe rate limiter for CoinMetrics API"""

import time
import threading
from typing import Optional
import logging
from src.core.interfaces import RateLimiterInterface

logger = logging.getLogger('backfill_system.coinmetrics')


class CoinMetricsRateLimiter(RateLimiterInterface):
    """
    Token bucket rate limiter for CoinMetrics API.
    
    CoinMetrics limits: 6000 requests per 20 seconds
    Implementation uses token bucket with safety margin.
    """
    
    def __init__(
        self,
        requests_per_window: int = 6000,
        window_seconds: int = 20,
        safety_margin: float = 0.9
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Time window in seconds
            safety_margin: Safety factor (0.9 = use 90% of limit)
        """
        self.requests_per_window = int(requests_per_window * safety_margin)
        self.window_seconds = window_seconds
        self.safety_margin = safety_margin
        
        # Token bucket parameters
        self.tokens = float(self.requests_per_window)
        self.max_tokens = float(self.requests_per_window)
        self.refill_rate = self.max_tokens / self.window_seconds  # tokens per second
        
        # Thread safety
        self.lock = threading.Lock()
        self.last_refill = time.time()
        
        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        
        logger.info(
            f"Rate limiter initialized: {self.requests_per_window} req/{window_seconds}s "
            f"(safety margin: {safety_margin})"
        )
    
    def _refill_tokens(self) -> None:
        """
        Refill tokens based on elapsed time.
        Must be called with lock held.
        """
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self) -> None:
        """
        Acquire permission to make one request.
        Blocks until a token is available.
        """
        with self.lock:
            self._refill_tokens()
            
            # If we have tokens, consume one and return
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self.total_requests += 1
                return
            
            # Calculate wait time needed
            tokens_needed = 1.0 - self.tokens
            wait_time = tokens_needed / self.refill_rate
            
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            self.total_wait_time += wait_time
        
        # Wait outside the lock to allow other threads to check
        time.sleep(wait_time)
        
        # Acquire again after waiting
        with self.lock:
            self._refill_tokens()
            self.tokens -= 1.0
            self.total_requests += 1
    
    def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.
        
        Returns:
            True if token was acquired, False otherwise
        """
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self.total_requests += 1
                return True
            
            return False
    
    def get_current_rate(self) -> float:
        """
        Get current request rate (requests per second).
        
        Returns:
            Current requests per second
        """
        with self.lock:
            # Estimate based on token availability
            return self.refill_rate
    
    def get_remaining_quota(self) -> Optional[int]:
        """
        Get remaining requests available immediately.
        
        Returns:
            Number of available tokens (requests)
        """
        with self.lock:
            self._refill_tokens()
            return int(self.tokens)
    
    def get_statistics(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            return {
                'total_requests': self.total_requests,
                'total_wait_time': self.total_wait_time,
                'avg_wait_time': (
                    self.total_wait_time / self.total_requests 
                    if self.total_requests > 0 else 0
                ),
                'available_tokens': int(self.tokens),
                'max_tokens': int(self.max_tokens)
            }
    
    def reset(self) -> None:
        """
        Reset rate limiter state (mainly for testing).
        """
        with self.lock:
            self.tokens = float(self.max_tokens)
            self.last_refill = time.time()
            self.total_requests = 0
            self.total_wait_time = 0.0
            logger.debug("Rate limiter reset")

