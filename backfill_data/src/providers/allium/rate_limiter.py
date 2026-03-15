"""Rate limiter for Allium API — 60 req / 60s default"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class AlliumRateLimiter(TokenBucketRateLimiter):
    provider_name = "allium"
