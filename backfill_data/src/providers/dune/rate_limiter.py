"""Rate limiter for Dune Analytics API — 40 req/min standard"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class DuneRateLimiter(TokenBucketRateLimiter):
    provider_name = "dune"
