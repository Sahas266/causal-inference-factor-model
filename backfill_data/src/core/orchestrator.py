"""Core orchestrator for backfill operations"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from src.providers.registry import ProviderRegistry
from src.core.storage.supabase_manager import SupabaseManager
from src.core.storage.database_writer import DatabaseWriter
from src.core.storage.progress_tracker import ProgressTracker
from src.core.utils.config_loader import ConfigLoader
from src.core.interfaces import DataProviderInterface

logger = logging.getLogger('backfill_system.orchestrator')

# Canonical metric-name normalization applied at write time. The warehouse
# standardized the price metric to 'price' (2026-03 audit); some provider
# transformers still emit their native names, so re-runs would otherwise
# re-introduce legacy rows that shadow or fragment the 'price' series.
METRIC_NORMALIZATION = {
    'PriceUSD': 'price',
    'price_usd': 'price',
}

# Canonical provider priorities — single source of truth, matching the
# audited warehouse state (lower wins in asset_metrics_best). Committed
# endpoint configs drifted over time; values stamped here override config
# `priority` so re-runs cannot revert the audit. Unknown providers fall
# back to the config value.
PROVIDER_PRIORITY = {
    'coinmetrics': 1,   # non-price metrics (price is special-cased to 3)
    'fred': 1,
    'artemis': 2,
    'defillama': 2,
    'coingecko': 3,
    'hyperliquid': 3,
    'dune': 4,
    'allium': 4,
    'derived': 99,
}
# Artemis is the authoritative price source; CoinMetrics price is a fallback.
COINMETRICS_PRICE_PRIORITY = 3


def _parse_date_range_bound(
    value: Optional[str | datetime],
    default_hour: int,
    *,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Parse endpoint date range values (YYYY-MM-DD or ISO) into UTC datetimes.

    Args:
        value: Raw date/date-time string from endpoint config.
        default_hour: Hour to use when only a date is provided.

    Returns:
        Parsed datetime in UTC, or None if no value is provided.
    """
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if value.strip().lower() == 'latest':
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    # Date-only format (YYYY-MM-DD)
    if len(value) == 10:
        dt = datetime.fromisoformat(value)
        return datetime(dt.year, dt.month, dt.day, default_hour, 0, 0, tzinfo=timezone.utc)

    # Full ISO timestamp
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _completed_range_covers(progress: Dict, requested_end: Optional[datetime]) -> bool:
    """Return whether a completed checkpoint already covers this request."""
    if progress.get('status') != 'completed' or requested_end is None:
        return False
    previous_end = (progress.get('config') or {}).get('actual_end')
    try:
        parsed_end = _parse_date_range_bound(previous_end, default_hour=23)
    except (TypeError, ValueError):
        return False
    return parsed_end is not None and parsed_end >= requested_end


class BackfillOrchestrator:
    """
    Provider-agnostic orchestrator for backfill operations.
    Coordinates validation, data fetching, and database writes.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            config_dir: Path to configuration directory
        """
        # Initialize components
        self.config_loader = ConfigLoader(config_dir)
        self.supabase_manager = SupabaseManager()
        self.db_writer = DatabaseWriter(self.supabase_manager)
        self.progress_tracker = ProgressTracker(self.supabase_manager)
        
        # Auto-discover and initialize providers
        ProviderRegistry.auto_discover()
        self.providers = self._initialize_providers()
        
        logger.info(
            f"Orchestrator initialized with {len(self.providers)} providers: "
            f"{', '.join(self.providers.keys())}"
        )
    
    def _initialize_providers(self) -> Dict[str, DataProviderInterface]:
        """
        Initialize all enabled providers.
        
        Returns:
            Dictionary of provider_name -> provider_instance
        """
        providers = {}
        
        for provider_name in ProviderRegistry.list_providers():
            try:
                # Load provider config
                provider_config = self.config_loader.load_provider_config(provider_name)
                
                # Skip if disabled
                if not provider_config.get('enabled', False):
                    logger.info(f"Provider {provider_name} is disabled, skipping")
                    continue
                
                # Create and initialize provider
                provider_class = ProviderRegistry.get_provider(provider_name)
                provider = provider_class()
                provider.initialize(provider_config)
                
                providers[provider_name] = provider
                logger.info(f"Initialized provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
        
        return providers
    
    def run_backfill(
        self,
        endpoint_configs: List[Dict],
        max_workers: int = 10,
        validate_only: bool = False
    ) -> Dict:
        """
        Run backfill for all endpoints across all providers.
        
        Args:
            endpoint_configs: List of endpoint configurations
            max_workers: Maximum number of concurrent workers
            validate_only: If True, only validate endpoints without backfilling
            
        Returns:
            Results dictionary with statistics
        """
        results = {
            'total_endpoints': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'provider_stats': {}
        }
        
        # Validate all endpoints first
        logger.info(f"Validating {len(endpoint_configs)} endpoint configurations...")
        validated_configs = []
        
        for config in endpoint_configs:
            validated = self._validate_endpoint_all_providers(config)
            if validated:
                validated_configs.append(validated)
            else:
                results['skipped'] += 1
        
        logger.info(
            f"Validation complete: {len(validated_configs)} valid, "
            f"{results['skipped']} skipped"
        )
        
        if validate_only:
            logger.info("Validation-only mode, stopping here")
            return results
        
        # Execute backfill with threading
        logger.info(f"Starting backfill with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit tasks for each endpoint+provider combination
            for endpoint_config in validated_configs:
                for provider_config in endpoint_config['providers']:
                    if provider_config.get('enabled', True):
                        future = executor.submit(
                            self._backfill_endpoint_provider,
                            endpoint_config,
                            provider_config
                        )
                        futures.append(future)
                        results['total_endpoints'] += 1
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    
                    if result.get('skipped'):
                        results['skipped'] += 1
                    elif result['success']:
                        results['completed'] += 1
                    else:
                        results['failed'] += 1
                    
                    # Update per-provider stats
                    provider = result['provider']
                    if provider not in results['provider_stats']:
                        results['provider_stats'][provider] = {
                            'completed': 0,
                            'failed': 0,
                            'skipped': 0,
                            'records': 0
                        }
                    
                    results['provider_stats'][provider]['records'] += result.get('records', 0)
                    if result.get('skipped'):
                        results['provider_stats'][provider]['skipped'] += 1
                    elif result['success']:
                        results['provider_stats'][provider]['completed'] += 1
                    else:
                        results['provider_stats'][provider]['failed'] += 1
                    
                except Exception as e:
                    logger.error(f"Backfill task failed with exception: {e}", exc_info=True)
                    results['failed'] += 1
        
        logger.info(
            f"Backfill complete: {results['completed']} completed, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )
        
        return results
    
    def _validate_endpoint_all_providers(
        self,
        endpoint_config: Dict
    ) -> Optional[Dict]:
        """
        Validate endpoint across all specified providers.
        
        Args:
            endpoint_config: Endpoint configuration
            
        Returns:
            Validated configuration or None if no valid providers
        """
        validated_config = endpoint_config.copy()
        validated_providers = []
        
        for provider_index, provider_config in enumerate(endpoint_config.get('providers', [])):
            provider_name = provider_config['name']
            
            if provider_name not in self.providers:
                logger.warning(f"Provider {provider_name} not available")
                continue
            
            provider = self.providers[provider_name]
            
            try:
                validation = provider.validate_endpoint(provider_config.get('config', {}))
                
                if validation.valid:
                    # Prefer provider-adjusted bounds; fall back to endpoint date_range.
                    endpoint_range = endpoint_config.get('date_range', {})
                    actual_start = (
                        validation.adjusted_start_time
                        or _parse_date_range_bound(endpoint_range.get('start'), default_hour=0)
                    )
                    actual_end = (
                        validation.adjusted_end_time
                        or _parse_date_range_bound(endpoint_range.get('end'), default_hour=23)
                    )

                    # Add adjusted date range
                    adjusted_provider_config = provider_config.copy()
                    adjusted_provider_config['actual_start'] = actual_start
                    adjusted_provider_config['actual_end'] = actual_end
                    adjusted_provider_config['provider_instance_id'] = provider_config.get(
                        'provider_instance_id',
                        str(provider_index),
                    )
                    validated_providers.append(adjusted_provider_config)
                    
                    logger.info(
                        f"Validated {endpoint_config['endpoint_id']} for {provider_name}: "
                        f"{actual_start} to {actual_end}"
                    )
                else:
                    logger.warning(
                        f"Validation failed for {endpoint_config['endpoint_id']} "
                        f"on {provider_name}: {validation.reason}"
                    )
                    
            except Exception as e:
                logger.error(
                    f"Validation error for {endpoint_config['endpoint_id']} "
                    f"on {provider_name}: {e}"
                )
        
        if not validated_providers:
            logger.error(f"No valid providers for {endpoint_config['endpoint_id']}")
            return None
        
        validated_config['providers'] = validated_providers
        return validated_config
    
    def _backfill_endpoint_provider(
        self,
        endpoint_config: Dict,
        provider_config: Dict
    ) -> Dict:
        """
        Backfill single endpoint from single provider.
        
        Args:
            endpoint_config: Endpoint configuration
            provider_config: Provider-specific configuration
            
        Returns:
            Result dictionary
        """
        provider_name = provider_config['name']
        provider = self.providers[provider_name]
        provider_instance_id = provider_config.get('provider_instance_id', '0')
        endpoint_id = f"{endpoint_config['endpoint_id']}_{provider_name}_{provider_instance_id}"
        
        try:
            logger.info(f"Starting backfill: {endpoint_id}")
            
            # Check if already completed BEFORE initializing — the upsert in
            # initialize_progress resets status to 'pending' and zeroes the
            # counters, which would make this skip-check dead code and wipe
            # the completed marker on every re-run.
            progress = self.progress_tracker.get_progress(endpoint_id)
            requested_end = provider_config.get('actual_end')
            if progress and _completed_range_covers(progress, requested_end):
                logger.info(f"Endpoint {endpoint_id} already completed, skipping")
                return {
                    'success': True,
                    'provider': provider_name,
                    'endpoint_id': endpoint_id,
                    'records': 0,
                    'skipped': True
                }

            if progress and progress.get('status') == 'completed':
                logger.info(
                    "Reopening %s because requested end advanced to %s",
                    endpoint_id,
                    requested_end,
                )
                self.progress_tracker.reopen_progress(endpoint_id, provider_config)
            else:
                # Initialize progress tracking
                self.progress_tracker.initialize_progress(
                    endpoint_id=endpoint_id,
                    provider=provider_name,
                    endpoint_type=provider_config.get('config', {}).get(
                        'endpoint_type', 'unknown'
                    ),
                    table_name=endpoint_config['table'],
                    config=provider_config
                )

            # Get date range
            start_time = provider_config.get('actual_start')
            end_time = provider_config.get('actual_end')
            
            # Resume from checkpoint if exists
            if progress and progress.get('last_successful_time'):
                checkpoint_time = datetime.fromisoformat(
                    progress['last_successful_time'].replace('Z', '+00:00')
                )
                logger.info(f"Resuming from checkpoint: {checkpoint_time}")
                start_time = checkpoint_time
            
            # Stream and write data
            total_records = 0
            
            for data_chunk in provider.fetch_data_stream(
                provider_config.get('config', {}),
                start_time,
                end_time
            ):
                if not data_chunk:
                    continue
                
                # Add provider metadata to each record. Metric names and
                # priorities are normalized here (single chokepoint) so
                # config drift cannot corrupt the warehouse — see
                # METRIC_NORMALIZATION / PROVIDER_PRIORITY above.
                for record in data_chunk:
                    record['provider'] = provider_name
                    metric = record.get('metric')
                    if metric in METRIC_NORMALIZATION:
                        metric = METRIC_NORMALIZATION[metric]
                        record['metric'] = metric
                    priority = PROVIDER_PRIORITY.get(
                        provider_name, provider_config.get('priority', 999)
                    )
                    if provider_name == 'coinmetrics' and metric == 'price':
                        priority = COINMETRICS_PRICE_PRIORITY
                    record['provider_priority'] = priority
                
                # Write to database
                self.db_writer.upsert_batch(
                    table=endpoint_config['table'],
                    data=data_chunk,
                    primary_keys=endpoint_config['primary_keys'],
                    batch_size=1000
                )
                
                total_records += len(data_chunk)
                
                # Update progress checkpoint
                if data_chunk and 'time' in data_chunk[-1]:
                    last_time_str = data_chunk[-1]['time']
                    last_time = datetime.fromisoformat(last_time_str.replace('Z', '+00:00'))
                    self.progress_tracker.update_progress(
                        endpoint_id=endpoint_id,
                        last_time=last_time,
                        records_count=len(data_chunk)
                    )
                
                logger.info(
                    f"Progress {endpoint_id}: {len(data_chunk)} records "
                    f"(total: {total_records})"
                )
            
            # Mark as completed
            self.progress_tracker.mark_completed(endpoint_id)
            logger.info(f"Completed {endpoint_id}: {total_records} records")
            
            return {
                'success': True,
                'provider': provider_name,
                'endpoint_id': endpoint_id,
                'records': total_records
            }
            
        except Exception as e:
            logger.error(f"Backfill failed for {endpoint_id}: {e}", exc_info=True)
            self.progress_tracker.mark_failed(endpoint_id, str(e))
            
            return {
                'success': False,
                'provider': provider_name,
                'endpoint_id': endpoint_id,
                'error': str(e),
                'records': 0
            }
    
    def list_providers(self) -> List[str]:
        """
        List available providers.
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())
    
    def get_provider_stats(self) -> Dict:
        """
        Get statistics for all providers.
        
        Returns:
            Dictionary of provider statistics
        """
        stats = {}
        
        for provider_name in self.providers.keys():
            # Query progress table for stats
            try:
                results = self.db_writer.query_records(
                    table='backfill_progress',
                    columns='status, COUNT(*) as count, SUM(total_records_fetched) as total_records',
                    filters={'provider': provider_name}
                )
                stats[provider_name] = results
            except Exception as e:
                logger.error(f"Failed to get stats for {provider_name}: {e}")
                stats[provider_name] = {'error': str(e)}
        
        return stats

