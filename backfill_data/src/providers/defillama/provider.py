"""DefiLlama provider implementation"""

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
from .client import DefiLlamaClient
from .rate_limiter import DefiLlamaRateLimiter
from .transformer import DefiLlamaTransformer

logger = logging.getLogger('backfill_system.defillama')

# Endpoint types and the client method that fetches them
_ENDPOINT_DISPATCH = {
    'protocol/tvl': '_fetch_protocol_tvl',
    'chain/tvl': '_fetch_chain_tvl',
    'dex/summary': '_fetch_dex_summary',
    'fees/summary': '_fetch_fees_summary',
    'stablecoin/charts': '_fetch_stablecoin_charts',
    'coin/chart': '_fetch_coin_chart',
    'yields/pool-chart': '_fetch_yield_pool_chart',
}


class DefiLlamaProvider(DataProviderInterface):
    """
    DefiLlama implementation of DataProviderInterface.

    Most DefiLlama endpoints return the full historical series in a single
    request (no server-side pagination). The provider fetches all data once
    and the transformer filters by the requested date range.

    Endpoint config shape expected under ``providers[].config``:

    .. code-block:: json

        {
          "endpoint_type": "chain/tvl",
          "params": {
            "chain": "Ethereum",
            "schema_type": "chain_tvl",
            "asset": "ethereum"
          }
        }

    Supported endpoint types: protocol/tvl, chain/tvl, dex/summary,
    fees/summary, stablecoin/charts, coin/chart, yields/pool-chart.
    """

    def __init__(self):
        self.config: Optional[Dict] = None
        self.client: Optional[DefiLlamaClient] = None
        self.rate_limiter: Optional[DefiLlamaRateLimiter] = None
        self.transformer = DefiLlamaTransformer()
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "defillama"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config

        # API key is optional (only needed for Pro endpoints)
        api_key = provider_config.get('api_config', {}).get('api_key')
        if not api_key:
            api_key = os.getenv(
                provider_config.get('api_config', {}).get('api_key_env', 'DEFILLAMA_API_KEY'),
                None
            )

        free_base = provider_config.get('api_config', {}).get(
            'free_base_url', 'https://api.llama.fi'
        )
        stablecoins_base = provider_config.get('api_config', {}).get(
            'stablecoins_base_url', 'https://stablecoins.llama.fi'
        )
        coins_base = provider_config.get('api_config', {}).get(
            'coins_base_url', 'https://coins.llama.fi'
        )
        pro_base = provider_config.get('api_config', {}).get(
            'pro_base_url', 'https://pro-api.llama.fi'
        )

        self.client = DefiLlamaClient(
            api_key=api_key,
            free_base_url=free_base,
            stablecoins_base_url=stablecoins_base,
            coins_base_url=coins_base,
            pro_base_url=pro_base,
        )

        rate_limits = provider_config.get('rate_limits', {})
        self.rate_limiter = DefiLlamaRateLimiter(
            requests_per_window=rate_limits.get('requests_per_window', 200),
            window_seconds=rate_limits.get('window_seconds', 60),
            safety_margin=rate_limits.get('safety_margin', 0.9),
        )

        self._initialized = True
        logger.info("DefiLlama provider initialised")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        if endpoint_type not in _ENDPOINT_DISPATCH:
            return ValidationResult(
                valid=False,
                reason=f"Unknown endpoint_type '{endpoint_type}'. Supported: {list(_ENDPOINT_DISPATCH)}"
            )

        params = endpoint_config.get('params', {})
        try:
            self.rate_limiter.acquire()
            raw = self._dispatch_fetch(endpoint_type, params)
            if raw is not None:
                return ValidationResult(valid=True, reason="Probe fetch succeeded")
            return ValidationResult(valid=False, reason="Probe returned None")
        except Exception as e:
            logger.warning(f"DefiLlama validation failed: {e}")
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
        """
        DefiLlama returns all historical data in one shot (no pagination).
        This method always fetches the full series; the transformer filters
        by start_time / end_time.  cursor is ignored (always returns has_more=False).
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})

        raw = self._dispatch_fetch(endpoint_type, params)
        return FetchResult(data=[raw], next_cursor=None, has_more=False, metadata={})

    def fetch_data_stream(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[List[Dict], None, None]:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})
        schema_type = params.get('schema_type', 'chain_tvl')
        asset_hint = params.get('asset', '')

        self.rate_limiter.acquire()

        try:
            result = self.fetch_data_batch(endpoint_config, start_time, end_time)
        except Exception as e:
            error_info = self.handle_error(e, {'endpoint_config': endpoint_config})
            if error_info.get('retry'):
                wait = error_info.get('wait_seconds', 10)
                logger.warning(f"Retrying after {wait}s...")
                time.sleep(wait)
                result = self.fetch_data_batch(endpoint_config, start_time, end_time)
            else:
                raise

        raw_payload = result.data[0] if result.data else None
        if raw_payload is None:
            return

        standard = self.transformer.transform(
            raw_payload,
            schema_type=schema_type,
            start_time=start_time,
            end_time=end_time,
            asset_hint=asset_hint,
        )

        if standard:
            logger.info(
                f"DefiLlama '{endpoint_type}': {len(standard)} records "
                f"[{start_time.date()} → {end_time.date()}]"
            )
            yield standard

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
            if status == 404:
                return {'retry': False, 'error_type': 'not_found', 'fatal': True}
        return {'retry': True, 'wait_seconds': 15, 'error_type': 'unknown'}

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch_fetch(self, endpoint_type: str, params: Dict):
        if endpoint_type == 'protocol/tvl':
            protocol = params.get('protocol') or params.get('asset', '')
            return self.client.get_protocol_tvl(protocol)

        elif endpoint_type == 'chain/tvl':
            chain = params.get('chain', 'Ethereum')
            return self.client.get_chain_tvl(chain)

        elif endpoint_type == 'dex/summary':
            protocol = params.get('protocol', '')
            return self.client.get_dex_summary(protocol)

        elif endpoint_type == 'fees/summary':
            protocol = params.get('protocol', '')
            data_type = params.get('data_type', 'dailyFees')
            return self.client.get_fees_summary(protocol, data_type=data_type)

        elif endpoint_type == 'stablecoin/charts':
            chain = params.get('chain', 'all')
            return self.client.get_stablecoin_charts(chain)

        elif endpoint_type == 'coin/chart':
            coins = params.get('coins', '')
            period = params.get('period', '365d')
            span = params.get('span', 0)
            return self.client.get_coin_chart(coins, period=period, span=span)

        elif endpoint_type == 'yields/pool-chart':
            pool_id = params.get('pool_id', '')
            return self.client.get_yield_pool_chart(pool_id)

        else:
            raise ValueError(f"Unknown DefiLlama endpoint_type: '{endpoint_type}'")
