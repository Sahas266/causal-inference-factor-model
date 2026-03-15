"""Rate limiter for DefiLlama API — 200 req / 60s default"""

from src.core.utils.rate_limiter import TokenBucketRateLimiter


class DefiLlamaRateLimiter(TokenBucketRateLimiter):
    provider_name = "defillama"
