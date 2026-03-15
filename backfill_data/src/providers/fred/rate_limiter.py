"""Rate limiter for FRED API — 120 req/min"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class FredRateLimiter(TokenBucketRateLimiter):
    provider_name = "fred"
