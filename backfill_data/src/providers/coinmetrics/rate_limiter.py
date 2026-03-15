"""Rate limiter for CoinMetrics API — 6000 req / 20s"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class CoinMetricsRateLimiter(TokenBucketRateLimiter):
    provider_name = "coinmetrics"
