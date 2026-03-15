"""Rate limiter for CoinGecko API — 30 req/min (free) or 500 req/min (pro)"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class CoinGeckoRateLimiter(TokenBucketRateLimiter):
    provider_name = "coingecko"
