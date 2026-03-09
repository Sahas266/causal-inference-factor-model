"""Allium Developer API data provider"""

import os
import time
from typing import Generator, Dict, List, Optional
from datetime import datetime
import logging
import requests

from src.core.interfaces import (
    DataProviderInterface,
    ValidationResult,
    FetchResult,
    RateLimiterInterface,
)
from .client import AlliumClient
from .rate_limiter import AlliumRateLimiter
from .transformer import AlliumTransformer

logger = logging.getLogger('backfill_system.allium')


class AlliumProvider(DataProviderInterface):
    """
    Allium Developer REST API implementation of DataProviderInterface.

    Uses Allium's pre-built Developer endpoints (not the Explorer SQL API
    which requires separate compute credits).

    Supported endpoint types
    ------------------------
    ``developer/prices/history``
        POST /developer/prices/history
        OHLCV token prices for any EVM/Solana token.
        Required params: ``addresses`` (list of {chain, token_address}),
                         ``time_granularity`` ("1d" default)
        Schema type: ``token_price_history``

    ``developer/{chain}/dex/trades``
        GET /developer/{chain}/dex/trades
        DEX trade events (paginated).
        Required params: ``chain``
        Schema type: ``dex_trades``

    Endpoint config example:

    .. code-block:: json

        {
          "endpoint_type": "developer/prices/history",
          "params": {
            "schema_type": "token_price_history",
            "asset": "weth",
            "addresses": [
              {"chain": "ethereum",
               "token_address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"}
            ],
            "time_granularity": "1d"
          }
        }
    """

    def __init__(self):
        self.config: Optional[Dict] = None
        self.client: Optional[AlliumClient] = None
        self.rate_limiter: Optional[AlliumRateLimiter] = None
        self.transformer = AlliumTransformer()
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "allium"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config

        api_key = provider_config['api_config'].get('api_key')
        if not api_key:
            api_key = os.getenv(provider_config['api_config'].get('api_key_env', 'ALLIUM_API_KEY'))
        if not api_key:
            raise ValueError("Allium API key not found in config or ALLIUM_API_KEY env var")

        base_url = provider_config['api_config'].get('base_url', 'https://api.allium.so/api/v1')
        self.client = AlliumClient(api_key, base_url)

        rate_limits = provider_config.get('rate_limits', {})
        self.rate_limiter = AlliumRateLimiter(
            requests_per_window=rate_limits.get('requests_per_window', 60),
            window_seconds=rate_limits.get('window_seconds', 60),
            safety_margin=rate_limits.get('safety_margin', 0.9),
        )

        self._initialized = True
        logger.info("Allium provider initialised (Developer API mode)")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})

        try:
            self.rate_limiter.acquire()

            if endpoint_type == 'developer/prices/history':
                addresses = params.get('addresses', [])
                if not addresses:
                    return ValidationResult(valid=False, reason="'addresses' param required")
                # Probe with a 1-day window
                result = self.client.get_price_history(
                    addresses=addresses[:1],
                    start_timestamp='2024-01-01T00:00:00Z',
                    end_timestamp='2024-01-02T00:00:00Z',
                    time_granularity=params.get('time_granularity', '1d'),
                )
                return ValidationResult(
                    valid=True,
                    reason=f"Probe returned {len(result.get('items', []))} token(s)"
                )

            elif endpoint_type.startswith('developer/') and '/dex/trades' in endpoint_type:
                chain = params.get('chain', endpoint_type.split('/')[1])
                result = self.client.get_dex_trades(chain=chain, limit=1)
                return ValidationResult(
                    valid='items' in result,
                    reason="DEX trades endpoint accessible"
                )

            else:
                return ValidationResult(
                    valid=False,
                    reason=f"Unsupported endpoint_type '{endpoint_type}'"
                )

        except Exception as e:
            logger.error(f"Allium validation failed: {e}", exc_info=True)
            return ValidationResult(valid=False, reason=f"Validation error: {e}")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_data_batch(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
        cursor: Optional[str] = None,
    ) -> FetchResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})

        start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        if endpoint_type == 'developer/prices/history':
            raw = self.client.get_price_history(
                addresses=params['addresses'],
                start_timestamp=start_iso,
                end_timestamp=end_iso,
                time_granularity=params.get('time_granularity', '1d'),
                cursor=cursor,
            )
            next_cursor = raw.get('next_cursor') or raw.get('cursor')
            return FetchResult(
                data=[raw],
                next_cursor=next_cursor,
                has_more=bool(next_cursor),
                metadata={},
            )

        elif endpoint_type.startswith('developer/') and '/dex/trades' in endpoint_type:
            chain = params.get('chain', endpoint_type.split('/')[1])
            raw = self.client.get_dex_trades(
                chain=chain,
                limit=params.get('limit', 1000),
                start_time=start_iso,
                end_time=end_iso,
                cursor=cursor,
            )
            next_cursor = raw.get('next_cursor') or raw.get('cursor')
            return FetchResult(
                data=[raw],
                next_cursor=next_cursor,
                has_more=bool(next_cursor),
                metadata={},
            )

        else:
            raise ValueError(f"Unsupported endpoint_type: '{endpoint_type}'")

    def fetch_data_stream(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[List[Dict], None, None]:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        params = endpoint_config.get('params', {})
        schema_type = params.get('schema_type', 'token_price_history')
        asset_hint = params.get('asset', '')

        cursor = None
        page = 0

        while True:
            page += 1
            self.rate_limiter.acquire()

            try:
                result = self.fetch_data_batch(endpoint_config, start_time, end_time, cursor)
            except Exception as e:
                error_info = self.handle_error(e, {'endpoint_config': endpoint_config})
                if error_info.get('retry'):
                    wait = error_info.get('wait_seconds', 10)
                    logger.warning(f"Retrying after {wait}s...")
                    time.sleep(wait)
                    continue
                raise

            raw_payload = result.data[0] if result.data else {}
            if raw_payload:
                standard = self.transformer.transform(
                    raw_payload,
                    schema_type=schema_type,
                    start_time=start_time,
                    end_time=end_time,
                    asset_hint=asset_hint,
                )
                if standard:
                    logger.info(f"Page {page}: {len(standard)} records")
                    yield standard

            if not result.has_more:
                logger.info(f"Allium stream complete after {page} page(s)")
                break

            cursor = result.next_cursor

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_rate_limiter(self) -> RateLimiterInterface:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        return self.rate_limiter

    def transform_to_standard_schema(
        self,
        raw_data: List[Dict],
        endpoint_config: Dict,
        schema_type: str,
    ) -> List[Dict]:
        params = endpoint_config.get('params', {})
        return self.transformer.transform(
            raw_data[0] if raw_data else {},
            schema_type=schema_type,
            asset_hint=params.get('asset', ''),
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def handle_error(self, error: Exception, context: Dict) -> Dict:
        if isinstance(error, requests.HTTPError):
            status = error.response.status_code if error.response else 0
            if status == 429:
                retry_after = int(error.response.headers.get('Retry-After', 60))
                return {'retry': True, 'wait_seconds': retry_after, 'error_type': 'rate_limit'}
            if status in (500, 502, 503, 504):
                return {'retry': True, 'wait_seconds': 10, 'error_type': 'server_error'}
            if status == 401:
                return {'retry': False, 'error_type': 'auth_error', 'fatal': True}
            if status == 402:
                return {'retry': False, 'error_type': 'insufficient_credits', 'fatal': True}
            if status == 404:
                return {'retry': False, 'error_type': 'not_found', 'fatal': True}
        return {'retry': True, 'wait_seconds': 15, 'error_type': 'unknown'}
