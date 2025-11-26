"""
Collector registry and management.
Provides centralized registration and management of metric collectors.
"""

from typing import Dict, List, Optional
from app.collectors.base import BaseCollector, MetricType, CollectionPriority, CollectionConfig, collector_registry


def get_collector_registry() -> BaseCollector:
    """
    Get the global collector registry instance.

    Returns:
        The global collector registry
    """
    return collector_registry


def register_collector(
    collector_class,
    metric_type: MetricType,
    priority: CollectionPriority,
    interval_minutes: int,
    **kwargs
) -> None:
    """
    Register a collector class with the registry.

    Args:
        collector_class: The collector class to instantiate
        metric_type: Type of metric this collector handles
        priority: Collection priority level
        interval_minutes: Collection interval in minutes
        **kwargs: Additional arguments for collector initialization
    """
    config = CollectionConfig(
        metric_type=metric_type,
        priority=priority,
        interval_minutes=interval_minutes
    )

    # Instantiate the collector
    collector = collector_class(config=config, **kwargs)

    # Register with the global registry
    collector_registry.register(collector)


def create_default_collectors() -> None:
    """
    Create and register default collectors for all metric types.
    This is a placeholder - in real implementation, specific collector classes would be imported.
    """
    # Placeholder for default collector registration
    # In a real implementation, you would import specific collector classes
    # and register them here, for example:
    #
    # from app.collectors.chain_congestion import ChainCongestionCollector
    # from app.collectors.funding_basis import FundingBasisCollector
    #
    # register_collector(
    #     ChainCongestionCollector,
    #     MetricType.CHAIN_CONGESTION,
    #     CollectionPriority.HIGH,
    #     1
    # )
    #
    # register_collector(
    #     FundingBasisCollector,
    #     MetricType.FUNDING_BASIS,
    #     CollectionPriority.HIGH,
    #     5
    # )

    # For now, just log that this would be implemented
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Default collectors registration placeholder - implement specific collectors")


# Initialize default collectors on import
create_default_collectors()
