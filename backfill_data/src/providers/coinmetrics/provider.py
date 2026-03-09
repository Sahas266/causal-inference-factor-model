"""CoinMetrics provider implementation"""

import os
from typing import Generator, Dict, List, Optional
from datetime import datetime
import logging
import requests

from src.core.interfaces import (
    DataProviderInterface,
    ValidationResult,
    FetchResult,
    RateLimiterInterface
)
from .client import CoinMetricsClient
from .rate_limiter import CoinMetricsRateLimiter
from .transformer import CoinMetricsTransformer

logger = logging.getLogger('backfill_system.coinmetrics')


class CoinMetricsProvider(DataProviderInterface):
    """
    CoinMetrics implementation of DataProviderInterface.
    Coordinates client, rate limiter, and transformer.
    """
    
    def __init__(self):
        """Initialize provider (configuration set via initialize method)"""
        self.config: Optional[Dict] = None
        self.client: Optional[CoinMetricsClient] = None
        self.rate_limiter: Optional[CoinMetricsRateLimiter] = None
        self.transformer = CoinMetricsTransformer()
        self._initialized = False
    
    @property
    def provider_name(self) -> str:
        """Return provider identifier"""
        return "coinmetrics"
    
    def initialize(self, provider_config: Dict) -> None:
        """
        Initialize provider with configuration.
        
        Args:
            provider_config: Configuration dictionary
        """
        self.config = provider_config
        
        api_config = provider_config.get('api_config', {})
        base_url = api_config.get('base_url', 'https://api.coinmetrics.io/v4')
        community_base_url = api_config.get(
            'community_base_url',
            'https://community-api.coinmetrics.io/v4'
        )
        is_community_base = base_url.rstrip('/') == community_base_url.rstrip('/')
        api_key_required = api_config.get('api_key_required', not is_community_base)

        # Initialize client
        api_key = api_config.get('api_key')
        if not api_key:
            api_key = os.getenv(api_config.get('api_key_env', ''))
        
        if api_key_required and not api_key:
            raise ValueError(
                "CoinMetrics API key not found in config or environment "
                f"(base_url={base_url})"
            )

        self.client = CoinMetricsClient(api_key, base_url)
        
        # Initialize rate limiter
        rate_limits = provider_config.get('rate_limits', {})
        self.rate_limiter = CoinMetricsRateLimiter(
            requests_per_window=rate_limits.get('requests_per_window', 6000),
            window_seconds=rate_limits.get('window_seconds', 20),
            safety_margin=rate_limits.get('safety_margin', 0.9)
        )
        
        self._initialized = True
        mode = "community" if is_community_base else "pro"
        logger.info("CoinMetrics provider initialized (%s mode)", mode)
    
    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        """
        Validate endpoint using CoinMetrics catalog API.
        
        Args:
            endpoint_config: Endpoint configuration
            
        Returns:
            ValidationResult with availability information
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            endpoint_type = endpoint_config.get('endpoint_type', '')
            params = endpoint_config.get('params', {})
            
            # Determine catalog endpoint
            if 'asset-metrics' in endpoint_type:
                catalog_endpoint = 'catalog-v2/asset-metrics'
                identifier_key = 'assets'
                identifier = params.get('assets', '').split(',')[0]  # First asset
            elif 'market-trades' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-trades'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'exchange-metrics' in endpoint_type:
                catalog_endpoint = 'catalog-v2/exchange-metrics'
                identifier_key = 'exchanges'
                identifier = params.get('exchanges', '').split(',')[0]
            elif 'pair-candles' in endpoint_type:
                catalog_endpoint = 'catalog-v2/pair-candles'
                identifier_key = 'pairs'
                identifier = params.get('pairs', '').split(',')[0]
            elif 'market-openinterest' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-openinterest'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'market-liquidations' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-liquidations'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'market-funding-rates' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-funding-rates'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'market-candles' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-candles'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'market-implied-volatility' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-implied-volatility'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            elif 'market-greeks' in endpoint_type:
                catalog_endpoint = 'catalog-v2/market-greeks'
                identifier_key = 'markets'
                identifier = params.get('markets', '').split(',')[0]
            else:
                return ValidationResult(
                    valid=False,
                    reason=f"Unknown endpoint type: {endpoint_type}"
                )
            
            # Fetch catalog
            catalog_params = {identifier_key: identifier}
            if 'metrics' in params:
                catalog_params['metrics'] = params['metrics']
            
            self.rate_limiter.acquire()
            catalog_data = self.client.fetch_catalog(catalog_endpoint, catalog_params)
            
            # Parse availability times
            min_time, max_time = self.client.parse_catalog_times(catalog_data, identifier)
            
            if min_time and max_time:
                return ValidationResult(
                    valid=True,
                    adjusted_start_time=min_time,
                    adjusted_end_time=max_time,
                    metadata={'catalog_data': catalog_data}
                )
            else:
                # Some community catalog endpoints omit min/max bounds.
                # In that case, allow backfill to proceed using configured date_range.
                if catalog_data.get('data'):
                    return ValidationResult(
                        valid=True,
                        reason=(
                            f"Catalog available for {identifier} but without min/max bounds; "
                            "using endpoint-configured date range"
                        ),
                        metadata={'catalog_data': catalog_data}
                    )
                return ValidationResult(
                    valid=False,
                    reason=f"No data availability found for {identifier}"
                )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return ValidationResult(
                valid=False,
                reason=f"Validation error: {str(e)}"
            )
    
    def fetch_data_batch(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
        cursor: Optional[str] = None
    ) -> FetchResult:
        """
        Fetch single batch from CoinMetrics.
        
        Args:
            endpoint_config: Endpoint configuration
            start_time: Start of time range
            end_time: End of time range
            cursor: Pagination cursor (next_page_url)
            
        Returns:
            FetchResult with data and pagination info
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            if cursor:
                # Use pagination URL
                response = self.client.fetch_by_url(cursor)
            else:
                # Initial request
                endpoint_type = endpoint_config.get('endpoint_type', '')
                params = endpoint_config.get('params', {}).copy()
                
                # Add time range
                params['start_time'] = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                params['end_time'] = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                response = self.client.fetch(endpoint_type, params)
            
            return FetchResult(
                data=response.get('data', []),
                next_cursor=response.get('next_page_url'),
                has_more=bool(response.get('next_page_url')),
                metadata={'response': response}
            )
            
        except Exception as e:
            logger.error(f"Fetch failed: {e}", exc_info=True)
            raise
    
    def fetch_data_stream(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime
    ) -> Generator[List[Dict], None, None]:
        """
        Stream data from CoinMetrics with automatic pagination.
        
        Args:
            endpoint_config: Endpoint configuration
            start_time: Start of time range
            end_time: End of time range
            
        Yields:
            Lists of standardized records
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        cursor = None
        page = 0
        
        while True:
            page += 1
            
            # Respect rate limits
            self.rate_limiter.acquire()
            
            # Fetch batch
            try:
                result = self.fetch_data_batch(
                    endpoint_config,
                    start_time,
                    end_time,
                    cursor
                )
            except Exception as e:
                error_info = self.handle_error(e, {'endpoint_config': endpoint_config})
                if error_info.get('retry', False):
                    import time
                    wait_seconds = error_info.get('wait_seconds', 10)
                    logger.warning(f"Error occurred, waiting {wait_seconds}s before retry")
                    time.sleep(wait_seconds)
                    continue
                else:
                    raise
            
            if result.data:
                # Determine schema type
                endpoint_type = endpoint_config.get('endpoint_type', '')
                if 'asset-metrics' in endpoint_type:
                    schema_type = 'asset_metrics'
                elif 'market-trades' in endpoint_type:
                    schema_type = 'market_trades'
                elif 'exchange-metrics' in endpoint_type:
                    schema_type = 'exchange_metrics'
                elif 'market-orderbooks' in endpoint_type:
                    schema_type = 'market_orderbooks'
                else:
                    schema_type = self.transformer.infer_schema_type(result.data)
                
                # Transform to standard schema
                standard_data = self.transform_to_standard_schema(
                    result.data,
                    endpoint_config,
                    schema_type
                )
                
                logger.info(
                    f"Fetched page {page}: {len(result.data)} raw records -> "
                    f"{len(standard_data)} standard records"
                )
                
                yield standard_data
            
            # Check if more data exists
            if not result.has_more:
                logger.info(f"Completed streaming after {page} pages")
                break
            
            cursor = result.next_cursor
    
    def get_rate_limiter(self) -> RateLimiterInterface:
        """Return rate limiter instance"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        return self.rate_limiter
    
    def transform_to_standard_schema(
        self,
        raw_data: List[Dict],
        endpoint_config: Dict,
        schema_type: str
    ) -> List[Dict]:
        """
        Transform CoinMetrics data to standard schema.
        
        Args:
            raw_data: Raw data from CoinMetrics
            endpoint_config: Endpoint configuration
            schema_type: Schema type
            
        Returns:
            Standardized records
        """
        return self.transformer.transform(raw_data, schema_type)
    
    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """
        Handle CoinMetrics-specific errors.
        
        Args:
            error: Exception that occurred
            context: Error context
            
        Returns:
            Error handling instructions
        """
        if isinstance(error, requests.HTTPError):
            status_code = error.response.status_code
            
            if status_code == 429:
                # Rate limit exceeded
                retry_after = int(error.response.headers.get('Retry-After', 20))
                return {
                    'retry': True,
                    'wait_seconds': retry_after,
                    'error_type': 'rate_limit'
                }
            elif status_code in [500, 502, 503, 504]:
                # Server errors - retry
                return {
                    'retry': True,
                    'wait_seconds': 5,
                    'error_type': 'server_error'
                }
            elif status_code == 401:
                # Authentication error - don't retry
                return {
                    'retry': False,
                    'error_type': 'auth_error',
                    'fatal': True
                }
            elif status_code == 403:
                # Forbidden - metric not available on community tier
                return {
                    'retry': False,
                    'error_type': 'forbidden',
                    'fatal': True
                }
            elif status_code == 404:
                # Not found - don't retry
                return {
                    'retry': False,
                    'error_type': 'not_found',
                    'fatal': True
                }

        # Default: retry with backoff
        return {
            'retry': True,
            'wait_seconds': 10,
            'error_type': 'unknown'
        }

