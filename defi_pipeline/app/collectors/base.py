"""
Abstract base classes for data collectors.
Provides common interfaces and utilities for collecting metrics from various sources.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

from app.config import provider_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class MetricType(Enum):
    """Enumeration of all supported metric types."""

    LIQ_FLOW = "liq_flow"
    STABLEFLOW = "stableflow"
    FUNDING_BASIS = "funding_basis"
    CHAIN_CONGESTION = "chain_congestion"
    STAKING_YIELD = "staking_yield"
    MEV_PRESSURE = "mev_pressure"
    CEX_DEX_FLOW = "cex_dex_flow"


class CollectionPriority(Enum):
    """Priority levels for data collection tasks."""

    HIGH = "high"      # 1-5 minute intervals
    MEDIUM = "medium"  # 15-30 minute intervals
    LOW = "low"        # 1 hour intervals


@dataclass
class CollectionResult(Generic[T]):
    """Result of a data collection operation."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CollectionConfig:
    """Configuration for data collection operations."""

    metric_type: MetricType
    priority: CollectionPriority
    interval_minutes: int
    timeout_seconds: int = provider_settings.timeout
    max_retries: int = provider_settings.max_retries
    backoff_factor: float = provider_settings.backoff_factor


class BaseCollector(ABC):
    """
    Abstract base class for all metric collectors.

    Provides common functionality for:
    - Rate limiting and retry logic
    - Error handling and logging
    - Health checking
    - Collection scheduling
    """

    def __init__(self, metric_type: MetricType, config: CollectionConfig):
        self.metric_type = metric_type
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")

        # Collection state
        self.last_collection: Optional[datetime] = None
        self.collection_count: int = 0
        self.error_count: int = 0
        self.consecutive_failures: int = 0

    @abstractmethod
    async def collect(self) -> CollectionResult:
        """
        Collect data for this metric.

        Returns:
            CollectionResult: The collection result with data or error information
        """
        pass

    @abstractmethod
    async def validate_data(self, data: Any) -> bool:
        """
        Validate collected data before processing.

        Args:
            data: The collected data to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the collector and its dependencies are healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    async def collect_with_retry(self) -> CollectionResult:
        """
        Collect data with retry logic and exponential backoff.

        Returns:
            CollectionResult: The final collection result
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(
                    f"Starting collection attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"for {self.metric_type.value}"
                )

                # Attempt collection
                result = await self.collect()

                if result.success and result.data is not None:
                    # Validate the data
                    if await self.validate_data(result.data):
                        self._update_collection_state(success=True)
                        self.logger.info(
                            f"Successfully collected {self.metric_type.value} data "
                            f"from {result.source}"
                        )
                        return result
                    else:
                        error_msg = f"Data validation failed for {self.metric_type.value}"
                        self.logger.warning(error_msg)
                        last_error = error_msg
                else:
                    error_msg = result.error or f"Collection failed for {self.metric_type.value}"
                    self.logger.warning(f"Collection attempt {attempt + 1} failed: {error_msg}")
                    last_error = error_msg

            except Exception as e:
                error_msg = f"Unexpected error in collection attempt {attempt + 1}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                last_error = error_msg

            # If this wasn't the last attempt, wait before retrying
            if attempt < self.config.max_retries:
                wait_time = self.config.backoff_factor ** attempt
                self.logger.debug(f"Waiting {wait_time:.1f}s before retry")
                await asyncio.sleep(wait_time)

        # All attempts failed
        self._update_collection_state(success=False)
        self.logger.error(
            f"All {self.config.max_retries + 1} collection attempts failed for {self.metric_type.value}"
        )

        return CollectionResult(
            success=False,
            error=last_error,
            timestamp=datetime.utcnow()
        )

    def _update_collection_state(self, success: bool) -> None:
        """Update internal collection state."""
        self.collection_count += 1

        if success:
            self.last_collection = datetime.utcnow()
            self.consecutive_failures = 0
        else:
            self.error_count += 1
            self.consecutive_failures += 1

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for monitoring."""
        return {
            "metric_type": self.metric_type.value,
            "last_collection": self.last_collection.isoformat() if self.last_collection else None,
            "collection_count": self.collection_count,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": (
                (self.collection_count - self.error_count) / self.collection_count
                if self.collection_count > 0 else 0
            ),
        }

    def is_data_fresh(self, max_age_minutes: int = 60) -> bool:
        """
        Check if the last collected data is still fresh.

        Args:
            max_age_minutes: Maximum age in minutes for data to be considered fresh

        Returns:
            bool: True if data is fresh, False otherwise
        """
        if self.last_collection is None:
            return False

        age = datetime.utcnow() - self.last_collection
        return age <= timedelta(minutes=max_age_minutes)


class CollectorRegistry:
    """
    Registry for managing multiple collectors.

    Provides centralized management and monitoring of all collectors.
    """

    def __init__(self):
        self.collectors: Dict[MetricType, BaseCollector] = {}
        self.logger = get_logger(__name__)

    def register(self, collector: BaseCollector) -> None:
        """Register a collector instance."""
        if collector.metric_type in self.collectors:
            self.logger.warning(
                f"Collector for {collector.metric_type.value} already registered, replacing"
            )

        self.collectors[collector.metric_type] = collector
        self.logger.info(f"Registered collector for {collector.metric_type.value}")

    def get_collector(self, metric_type: MetricType) -> Optional[BaseCollector]:
        """Get a collector by metric type."""
        return self.collectors.get(metric_type)

    def get_all_collectors(self) -> List[BaseCollector]:
        """Get all registered collectors."""
        return list(self.collectors.values())

    def get_collectors_by_priority(self, priority: CollectionPriority) -> List[BaseCollector]:
        """Get collectors filtered by priority."""
        return [
            collector for collector in self.collectors.values()
            if collector.config.priority == priority
        ]

    async def collect_all(self) -> Dict[MetricType, CollectionResult]:
        """Collect data from all registered collectors."""
        results = {}

        for metric_type, collector in self.collectors.items():
            self.logger.debug(f"Starting collection for {metric_type.value}")
            result = await collector.collect_with_retry()
            results[metric_type] = result

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all collectors."""
        status = {
            "overall_healthy": True,
            "collectors": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        for metric_type, collector in self.collectors.items():
            collector_stats = collector.get_collection_stats()
            is_healthy = collector.consecutive_failures == 0

            status["collectors"][metric_type.value] = {
                "healthy": is_healthy,
                "stats": collector_stats
            }

            if not is_healthy:
                status["overall_healthy"] = False

        return status


# Global collector registry instance
collector_registry = CollectorRegistry()
