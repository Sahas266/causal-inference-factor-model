"""
Abstract base classes for data providers.
Provides common interfaces and utilities for fetching data from external APIs.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import httpx

from app.config import provider_settings
from app.utils.logger import get_logger
from app.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class ProviderType(Enum):
    """Enumeration of supported provider types."""

    BLOCKCHAIN_EXPLORER = "blockchain_explorer"
    DEX_PROTOCOL = "dex_protocol"
    MARKET_DATA = "market_data"
    ANALYTICS_PLATFORM = "analytics_platform"
    CROSS_CHAIN_BRIDGE = "cross_chain_bridge"


@dataclass
class ProviderConfig:
    """Configuration for external data providers."""

    name: str
    provider_type: ProviderType
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = provider_settings.timeout
    headers: Optional[Dict[str, str]] = None
    auth_type: Optional[str] = None  # 'bearer', 'api_key', 'basic', etc.


@dataclass
class ProviderResponse:
    """Response from a provider API call."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


class BaseProvider(ABC):
    """
    Abstract base class for all external data providers.

    Provides common functionality for:
    - HTTP client management
    - Rate limiting
    - Authentication
    - Error handling and retries
    - Health checking
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")

        # HTTP client and rate limiting
        self.client: Optional[httpx.AsyncClient] = None
        self.rate_limiter = RateLimiter(requests_per_minute=config.rate_limit)

        # Provider state
        self.last_request: Optional[datetime] = None
        self.request_count: int = 0
        self.error_count: int = 0
        self.consecutive_failures: int = 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the provider (create HTTP client, etc.)."""
        if self.client is None:
            # Build headers
            headers = self.config.headers or {}
            if self.config.auth_type == "bearer" and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            elif self.config.auth_type == "api_key" and self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
                follow_redirects=True
            )

        self.logger.info(f"Initialized provider {self.config.name}")

    async def cleanup(self) -> None:
        """Clean up provider resources."""
        if self.client:
            await self.client.aclose()
            self.client = None

        self.logger.info(f"Cleaned up provider {self.config.name}")

    @abstractmethod
    async def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> ProviderResponse:
        """
        Fetch data from a specific endpoint.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            ProviderResponse: The API response
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    async def make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ProviderResponse:
        """
        Make an HTTP request with rate limiting and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json: JSON data
            headers: Additional headers

        Returns:
            ProviderResponse: The response
        """
        if not self.client:
            return ProviderResponse(
                success=False,
                error="Provider not initialized",
                status_code=500
            )

        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()

        start_time = time.time()

        try:
            self.logger.debug(f"Making {method} request to {endpoint}")

            # Make the request
            response = await self.client.request(
                method=method,
                url=endpoint,
                params=params,
                data=data,
                json=json,
                headers=headers
            )

            response_time = time.time() - start_time

            # Update provider state
            self._update_request_state(success=True)
            self.last_request = datetime.utcnow()
            self.request_count += 1

            # Check for rate limit headers
            rate_limit_remaining = None
            rate_limit_reset = None

            if "X-RateLimit-Remaining" in response.headers:
                rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])

            if "X-RateLimit-Reset" in response.headers:
                try:
                    rate_limit_reset = datetime.fromtimestamp(
                        int(response.headers["X-RateLimit-Reset"])
                    )
                except (ValueError, TypeError):
                    pass

            # Handle response
            if response.is_success:
                try:
                    response_data = response.json()
                except Exception:
                    response_data = response.text

                return ProviderResponse(
                    success=True,
                    data=response_data,
                    status_code=response.status_code,
                    response_time=response_time,
                    rate_limit_remaining=rate_limit_remaining,
                    rate_limit_reset=rate_limit_reset
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.warning(f"Request failed: {error_msg}")

                # Update rate limiter if we hit a rate limit
                if response.status_code == 429:
                    self.rate_limiter.handle_rate_limit()

                self._update_request_state(success=False)
                return ProviderResponse(
                    success=False,
                    error=error_msg,
                    status_code=response.status_code,
                    response_time=response_time,
                    rate_limit_remaining=rate_limit_remaining,
                    rate_limit_reset=rate_limit_reset
                )

        except httpx.TimeoutException:
            error_msg = "Request timeout"
            self.logger.warning(error_msg)
            self._update_request_state(success=False)
            return ProviderResponse(
                success=False,
                error=error_msg,
                status_code=408,
                response_time=time.time() - start_time
            )

        except httpx.ConnectError:
            error_msg = "Connection error"
            self.logger.warning(error_msg)
            self._update_request_state(success=False)
            return ProviderResponse(
                success=False,
                error=error_msg,
                status_code=503,
                response_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_request_state(success=False)
            return ProviderResponse(
                success=False,
                error=error_msg,
                status_code=500,
                response_time=time.time() - start_time
            )

    def _update_request_state(self, success: bool) -> None:
        """Update internal request state."""
        if success:
            self.consecutive_failures = 0
        else:
            self.error_count += 1
            self.consecutive_failures += 1

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics for monitoring."""
        return {
            "name": self.config.name,
            "type": self.config.provider_type.value,
            "last_request": self.last_request.isoformat() if self.last_request else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0 else 0
            ),
            "rate_limit": self.config.rate_limit,
            "current_rate_limit_state": self.rate_limiter.get_stats(),
        }

    def is_healthy(self) -> bool:
        """Check if the provider is currently healthy."""
        # Consider unhealthy if too many consecutive failures
        return self.consecutive_failures < 3


class RESTProvider(BaseProvider):
    """
    Base class for REST API providers.

    Provides common REST API functionality.
    """

    async def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> ProviderResponse:
        """Fetch data from a REST API endpoint."""
        return await self.make_request("GET", endpoint, params=params)

    async def post_data(self, endpoint: str, data: Dict[str, Any]) -> ProviderResponse:
        """Post data to a REST API endpoint."""
        return await self.make_request("POST", endpoint, json=data)


class GraphQLProvider(BaseProvider):
    """
    Base class for GraphQL API providers.

    Provides common GraphQL query functionality.
    """

    async def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> ProviderResponse:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        return await self.make_request("POST", "/graphql", json=payload)

    async def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> ProviderResponse:
        """Fetch data using GraphQL query."""
        # For GraphQL, endpoint is treated as the query
        return await self.query(endpoint, params)


class ProviderRegistry:
    """
    Registry for managing multiple providers.

    Provides centralized management and monitoring of all providers.
    """

    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.logger = get_logger(__name__)

    async def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        if provider.config.name in self.providers:
            # Clean up old provider
            await self.providers[provider.config.name].cleanup()
            self.logger.warning(f"Provider {provider.config.name} already registered, replacing")

        self.providers[provider.config.name] = provider
        await provider.initialize()
        self.logger.info(f"Registered provider {provider.config.name}")

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return self.providers.get(name)

    def get_providers_by_type(self, provider_type: ProviderType) -> List[BaseProvider]:
        """Get providers filtered by type."""
        return [
            provider for provider in self.providers.values()
            if provider.config.provider_type == provider_type
        ]

    def get_all_providers(self) -> List[BaseProvider]:
        """Get all registered providers."""
        return list(self.providers.values())

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                results[name] = False

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all providers."""
        status = {
            "overall_healthy": True,
            "providers": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        for name, provider in self.providers.items():
            provider_stats = provider.get_provider_stats()
            is_healthy = provider.is_healthy()

            status["providers"][name] = {
                "healthy": is_healthy,
                "stats": provider_stats
            }

            if not is_healthy:
                status["overall_healthy"] = False

        return status

    async def cleanup_all(self) -> None:
        """Clean up all registered providers."""
        for provider in self.providers.values():
            await provider.cleanup()

        self.providers.clear()
        self.logger.info("Cleaned up all providers")


# Global provider registry instance
provider_registry = ProviderRegistry()
