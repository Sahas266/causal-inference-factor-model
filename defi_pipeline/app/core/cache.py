"""
Redis cache management for the DeFi data pipeline.
Provides caching functionality for API responses and temporary data storage.
"""

import json
import pickle
from typing import Any, Optional, Union, Dict
from datetime import timedelta
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import cache_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global Redis client
_redis_client: Optional[Redis] = None


def get_redis_client() -> Redis:
    """Get or create the Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            cache_settings.url,
            decode_responses=True,  # Decode strings automatically
            max_connections=cache_settings.max_connections,
            health_check_interval=cache_settings.health_check_interval,
        )
        logger.info("Redis client created")

    return _redis_client


async def init_cache() -> None:
    """Initialize the cache connection."""
    try:
        client = get_redis_client()
        await client.ping()
        logger.info("Cache connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        raise


async def close_cache() -> None:
    """Close the cache connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Cache connection closed")


@asynccontextmanager
async def get_cache():
    """Context manager for cache operations."""
    client = get_redis_client()
    try:
        yield client
    except Exception as e:
        logger.error(f"Cache operation failed: {e}")
        raise


class CacheManager:
    """
    High-level cache manager with serialization and key management.
    """

    def __init__(self, prefix: str = "defi_pipeline"):
        self.prefix = prefix
        self.client = get_redis_client()

    def _make_key(self, *parts: str) -> str:
        """Create a namespaced cache key."""
        return f"{self.prefix}:{':'.join(parts)}"

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            value = await self.client.get(self._make_key(key))
            if value is None:
                return default

            # Try to parse as JSON first, then fallback to pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)

        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_pickle: bool = False
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            use_pickle: Use pickle serialization for complex objects

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            ttl = ttl or cache_settings.ttl

            # Serialize value
            if use_pickle or not isinstance(value, (str, int, float, bool, dict, list)):
                serialized = pickle.dumps(value)
            else:
                serialized = json.dumps(value)

            return await self.client.setex(
                self._make_key(key),
                ttl,
                serialized
            )

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key was deleted, False otherwise
        """
        try:
            return bool(await self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key exists, False otherwise
        """
        try:
            return bool(await self.client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """
        Get the TTL (time to live) for a key.

        Args:
            key: Cache key

        Returns:
            int: TTL in seconds (-2 if key doesn't exist, -1 if no TTL)
        """
        try:
            return await self.client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Cache TTL check failed for key {key}: {e}")
            return -2

    async def set_multiple(
        self,
        key_value_pairs: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple key-value pairs in cache.

        Args:
            key_value_pairs: Dictionary of key-value pairs
            ttl: Time to live for all keys

        Returns:
            bool: True if all sets were successful
        """
        try:
            ttl = ttl or cache_settings.ttl
            pipeline = self.client.pipeline()

            for key, value in key_value_pairs.items():
                # Serialize value
                if not isinstance(value, (str, int, float, bool, dict, list)):
                    serialized = pickle.dumps(value)
                else:
                    serialized = json.dumps(value)

                pipeline.setex(self._make_key(key), ttl, serialized)

            await pipeline.execute()
            return True

        except Exception as e:
            logger.error(f"Cache set_multiple failed: {e}")
            return False

    async def get_multiple(self, keys: list[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            dict: Dictionary of found key-value pairs
        """
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.client.mget(cache_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = pickle.loads(value)

            return result

        except Exception as e:
            logger.error(f"Cache get_multiple failed: {e}")
            return {}

    async def clear_namespace(self) -> int:
        """
        Clear all keys in this cache manager's namespace.

        Returns:
            int: Number of keys deleted
        """
        try:
            pattern = f"{self.prefix}:*"
            keys = await self.client.keys(pattern)

            if keys:
                return await self.client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Cache clear_namespace failed: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and info.

        Returns:
            dict: Cache statistics
        """
        try:
            info = await self.client.info()
            return {
                "connected": True,
                "db_size": await self.client.dbsize(),
                "namespace_keys": len(await self.client.keys(f"{self.prefix}:*")),
                "memory_used": info.get("used_memory_human", "unknown"),
                "connections": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
            }
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            return {"connected": False, "error": str(e)}


# Global cache manager instance
cache_manager = CacheManager()
