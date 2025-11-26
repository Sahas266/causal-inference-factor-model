"""
Celery tasks for scheduled data collection.
Provides periodic tasks for collecting metrics from various sources.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from celery import Celery
from celery.schedules import crontab

from app.config import task_settings, app_settings
from app.collectors.base import MetricType, CollectionPriority, CollectionConfig
from app.collectors.registry import get_collector_registry
from app.core.cache import cache_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Create Celery app
celery_app = Celery(
    "defi_pipeline",
    broker=task_settings.broker_url,
    backend=task_settings.result_backend,
    timezone=task_settings.timezone,
    enable_utc=task_settings.enable_utc,
)

# Configure Celery
celery_app.conf.update(
    task_serializer=task_settings.task_serializer,
    result_serializer=task_settings.result_serializer,
    accept_content=task_settings.accept_content,
    worker_prefetch_multiplier=task_settings.worker_prefetch_multiplier,
    worker_max_tasks_per_child=task_settings.worker_max_tasks_per_child,
    task_routes={
        "app.tasks.scheduled.collect_chain_congestion": {"queue": "high_priority"},
        "app.tasks.scheduled.collect_funding_basis": {"queue": "high_priority"},
        "app.tasks.scheduled.collect_liq_flow": {"queue": "medium_priority"},
        "app.tasks.scheduled.collect_stableflow": {"queue": "medium_priority"},
        "app.tasks.scheduled.collect_mev_pressure": {"queue": "medium_priority"},
        "app.tasks.scheduled.collect_staking_yield": {"queue": "low_priority"},
        "app.tasks.scheduled.collect_cex_dex_flow": {"queue": "low_priority"},
    },
    beat_schedule={
        # High priority - frequent collection
        "collect-chain-congestion": {
            "task": "app.tasks.scheduled.collect_chain_congestion",
            "schedule": 60.0,  # Every 1 minute
        },
        "collect-funding-basis": {
            "task": "app.tasks.scheduled.collect_funding_basis",
            "schedule": 300.0,  # Every 5 minutes
        },
        # Medium priority - moderate frequency
        "collect-liq-flow": {
            "task": "app.tasks.scheduled.collect_liq_flow",
            "schedule": 900.0,  # Every 15 minutes
        },
        "collect-stableflow": {
            "task": "app.tasks.scheduled.collect_stableflow",
            "schedule": 900.0,  # Every 15 minutes
        },
        "collect-mev-pressure": {
            "task": "app.tasks.scheduled.collect_mev_pressure",
            "schedule": 1800.0,  # Every 30 minutes
        },
        # Low priority - infrequent collection
        "collect-staking-yield": {
            "task": "app.tasks.scheduled.collect_staking_yield",
            "schedule": 3600.0,  # Every 1 hour
        },
        "collect-cex-dex-flow": {
            "task": "app.tasks.scheduled.collect_cex_dex_flow",
            "schedule": 3600.0,  # Every 1 hour
        },
        # Maintenance tasks
        "cleanup-old-cache": {
            "task": "app.tasks.scheduled.cleanup_old_cache",
            "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        "update-data-freshness": {
            "task": "app.tasks.scheduled.update_data_freshness_status",
            "schedule": 300.0,  # Every 5 minutes
        },
    },
)


@celery_app.task(bind=True, name="collect_chain_congestion")
def collect_chain_congestion(self) -> Dict[str, Any]:
    """
    Collect chain congestion metrics.
    High priority task - runs every minute.
    """
    return _collect_metric_task(MetricType.CHAIN_CONGESTION, "collect_chain_congestion")


@celery_app.task(bind=True, name="collect_funding_basis")
def collect_funding_basis(self) -> Dict[str, Any]:
    """
    Collect funding basis metrics.
    High priority task - runs every 5 minutes.
    """
    return _collect_metric_task(MetricType.FUNDING_BASIS, "collect_funding_basis")


@celery_app.task(bind=True, name="collect_liq_flow")
def collect_liq_flow(self) -> Dict[str, Any]:
    """
    Collect liquidity flow metrics.
    Medium priority task - runs every 15 minutes.
    """
    return _collect_metric_task(MetricType.LIQ_FLOW, "collect_liq_flow")


@celery_app.task(bind=True, name="collect_stableflow")
def collect_stableflow(self) -> Dict[str, Any]:
    """
    Collect stablecoin flow metrics.
    Medium priority task - runs every 15 minutes.
    """
    return _collect_metric_task(MetricType.STABLEFLOW, "collect_stableflow")


@celery_app.task(bind=True, name="collect_mev_pressure")
def collect_mev_pressure(self) -> Dict[str, Any]:
    """
    Collect MEV pressure metrics.
    Medium priority task - runs every 30 minutes.
    """
    return _collect_metric_task(MetricType.MEV_PRESSURE, "collect_mev_pressure")


@celery_app.task(bind=True, name="collect_staking_yield")
def collect_staking_yield(self) -> Dict[str, Any]:
    """
    Collect staking yield metrics.
    Low priority task - runs every hour.
    """
    return _collect_metric_task(MetricType.STAKING_YIELD, "collect_staking_yield")


@celery_app.task(bind=True, name="collect_cex_dex_flow")
def collect_cex_dex_flow(self) -> Dict[str, Any]:
    """
    Collect CEX/DEX flow metrics.
    Low priority task - runs every hour.
    """
    return _collect_metric_task(MetricType.CEX_DEX_FLOW, "collect_cex_dex_flow")


@celery_app.task(bind=True, name="collect_all_metrics")
def collect_all_metrics(self) -> Dict[str, Any]:
    """
    Collect all metrics.
    Manual/batch collection task.
    """
    logger.info("Starting batch collection of all metrics")

    registry = get_collector_registry()
    results = {}

    # Collect all metrics
    collection_results = registry.collect_all()

    # Process results
    for metric_type, result in collection_results.items():
        results[metric_type.value] = {
            "success": result.success,
            "error": result.error,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "source": result.source
        }

        if result.success:
            logger.info(f"Successfully collected {metric_type.value}")
        else:
            logger.error(f"Failed to collect {metric_type.value}: {result.error}")

    # Update cache with collection status
    cache_manager.set("last_batch_collection", {
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    })

    logger.info("Batch collection completed")
    return results


@celery_app.task(bind=True, name="cleanup_old_cache")
def cleanup_old_cache(self) -> Dict[str, Any]:
    """
    Clean up old cache entries.
    Maintenance task - runs daily.
    """
    logger.info("Starting cache cleanup")

    try:
        # Clear old cache entries (implementation depends on cache strategy)
        # For now, just log that cleanup would happen
        result = {
            "status": "completed",
            "message": "Cache cleanup placeholder - implement based on retention policy"
        }

        logger.info("Cache cleanup completed")
        return result

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, name="update_data_freshness_status")
def update_data_freshness_status(self) -> Dict[str, Any]:
    """
    Update data freshness status for monitoring.
    Runs every 5 minutes.
    """
    logger.info("Updating data freshness status")

    try:
        from app.services.data_processor import DataProcessor

        processor = DataProcessor()
        freshness_status = processor.get_data_freshness_status()

        # Cache the freshness status
        cache_manager.set("data_freshness", freshness_status, ttl=600)  # 10 minutes

        # Check for stale data
        stale_metrics = []
        for metric_name, status in freshness_status.items():
            if not status["is_fresh"]:
                stale_metrics.append(metric_name)
                logger.warning(f"Stale data detected for {metric_name}")

        result = {
            "status": "completed",
            "freshness_status": freshness_status,
            "stale_metrics": stale_metrics,
            "stale_count": len(stale_metrics)
        }

        logger.info(f"Data freshness update completed - {len(stale_metrics)} stale metrics")
        return result

    except Exception as e:
        logger.error(f"Data freshness update failed: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, name="backfill_historical_data")
def backfill_historical_data(
    self,
    metric_type: str,
    start_date: str,
    end_date: str,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Backfill historical data for a specific metric.
    Manual task for filling data gaps.
    """
    logger.info(f"Starting historical backfill for {metric_type} from {start_date} to {end_date}")

    try:
        # Parse dates
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Validate metric type
        try:
            metric_enum = MetricType(metric_type)
        except ValueError:
            raise ValueError(f"Invalid metric type: {metric_type}")

        # Calculate date ranges for batching
        current_date = start
        total_batches = 0
        successful_batches = 0

        while current_date < end:
            batch_end = min(current_date + timedelta(days=1), end)

            try:
                # Collect data for this batch
                registry = get_collector_registry()
                collector = registry.get_collector(metric_enum)

                if collector:
                    # Note: This is a simplified backfill - real implementation
                    # would need to handle historical data collection properly
                    result = collector.collect_with_retry()

                    if result.success:
                        successful_batches += 1
                    else:
                        logger.warning(f"Failed batch {current_date}: {result.error}")
                else:
                    logger.warning(f"No collector found for {metric_type}")

            except Exception as e:
                logger.error(f"Error in batch {current_date}: {e}")

            current_date = batch_end
            total_batches += 1

        result = {
            "status": "completed",
            "metric_type": metric_type,
            "start_date": start_date,
            "end_date": end_date,
            "total_batches": total_batches,
            "successful_batches": successful_batches,
            "success_rate": successful_batches / total_batches if total_batches > 0 else 0
        }

        logger.info(f"Historical backfill completed for {metric_type}")
        return result

    except Exception as e:
        logger.error(f"Historical backfill failed: {e}")
        return {"status": "failed", "error": str(e)}


# Helper functions

def _collect_metric_task(metric_type: MetricType, task_name: str) -> Dict[str, Any]:
    """
    Generic metric collection task implementation.

    Args:
        metric_type: The metric type to collect
        task_name: Name of the calling task for logging

    Returns:
        Dictionary with collection results
    """
    logger.info(f"Starting {task_name} collection")

    try:
        # Get collector registry
        registry = get_collector_registry()
        collector = registry.get_collector(metric_type)

        if not collector:
            error_msg = f"No collector registered for {metric_type.value}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "metric_type": metric_type.value
            }

        # Collect data
        result = collector.collect_with_retry()

        # Prepare response
        response = {
            "success": result.success,
            "metric_type": metric_type.value,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "source": result.source
        }

        if result.success:
            logger.info(f"Successfully collected {metric_type.value} data")
            response["data_points"] = len(result.data) if isinstance(result.data, list) else 1
        else:
            logger.error(f"Failed to collect {metric_type.value}: {result.error}")
            response["error"] = result.error

        return response

    except Exception as e:
        error_msg = f"Unexpected error in {task_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "metric_type": metric_type.value
        }


# Celery beat configuration for production
if app_settings.is_production:
    celery_app.conf.beat_schedule.update({
        # Add production-specific schedules here
        "health-check-all-collectors": {
            "task": "app.tasks.scheduled.health_check_all_collectors",
            "schedule": 600.0,  # Every 10 minutes
        },
    })


@celery_app.task(bind=True, name="health_check_all_collectors")
def health_check_all_collectors(self) -> Dict[str, Any]:
    """
    Perform health checks on all collectors.
    Production monitoring task.
    """
    logger.info("Performing health checks on all collectors")

    try:
        registry = get_collector_registry()
        health_status = registry.get_health_status()

        # Log unhealthy collectors
        for metric_name, status in health_status["collectors"].items():
            if not status["healthy"]:
                logger.warning(f"Unhealthy collector: {metric_name}")

        # Cache health status
        cache_manager.set("collector_health", health_status, ttl=600)

        logger.info("Collector health checks completed")
        return health_status

    except Exception as e:
        logger.error(f"Collector health check failed: {e}")
        return {"status": "failed", "error": str(e)}
