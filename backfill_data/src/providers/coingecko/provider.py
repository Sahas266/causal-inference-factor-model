"""CoinGecko API data provider"""

import os
import time
from typing import Generator, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging
import requests

from src.core.interfaces import (
    DataProviderInterface,
    ValidationResult,
    FetchResult,
    RateLimiterInterface,
)
from .client import CoinGeckoClient
from .rate_limiter import CoinGeckoRateLimiter
from .transformer import CoinGeckoTransformer

logger = logging.getLogger('backfill_system.coingecko')

# Maximum days per market_chart_range request to avoid timeouts
CHUNK_DAYS = 180


class CoinGeckoProvider(DataProviderInterface):
    """
    CoinGecko API implementation of DataProviderInterface.

    Supported endpoint types
    ------------------------
    ``market_chart``
        GET /coins/{id}/market_chart/range
        Daily price, market cap, and volume timeseries.
        Required params: ``coin_id``, ``vs_currency``, ``asset``
        Schema type: ``market_chart``

    ``coin_data``
        GET /coins/{id}
        Current snapshot: fdv, total_supply, max_supply, ath, atl.
        Required params: ``coin_id``, ``asset``
        Schema type: ``coin_data``

    Endpoint config example:

    .. code-block:: json

        {
          "endpoint_type": "market_chart",
          "params": {
            "schema_type": "market_chart",
            "coin_id": "ethereum",
            "vs_currency": "usd",
            "asset": "eth"
          }
        }
    """

    def __init__(self):
        self.config: Optional[Dict] = None
        self.client: Optional[CoinGeckoClient] = None
        self.rate_limiter: Optional[CoinGeckoRateLimiter] = None
        self.transformer = CoinGeckoTransformer()
        self._initialized = False
        self._is_pro = False

    @property
    def provider_name(self) -> str:
        return "coingecko"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config
        api_config = provider_config.get('api_config', {})

        api_key = api_config.get('api_key')
        if not api_key:
            api_key = os.getenv(api_config.get('api_key_env', 'COINGECKO_API_KEY'))

        # Determine tier: Pro keys typically don't start with 'CG-' demo prefix
        # Demo/free keys use api.coingecko.com; Pro keys use pro-api.coingecko.com
        is_pro = bool(api_key) and not (api_key or '').startswith('CG-')
        if is_pro:
            base_url = api_config.get('base_url', 'https://pro-api.coingecko.com/api/v3')
        else:
            base_url = api_config.get('free_base_url', 'https://api.coingecko.com/api/v3')

        self.client = CoinGeckoClient(
            base_url=base_url,
            api_key=api_key,
            is_pro=is_pro,
        )

        rate_limits = provider_config.get('rate_limits', {})
        if not is_pro:
            # Free tier: 30 req/min regardless of config
            req_per_window = min(rate_limits.get('requests_per_window', 30), 30)
        else:
            req_per_window = rate_limits.get('requests_per_window', 500)

        self.rate_limiter = CoinGeckoRateLimiter(
            requests_per_window=req_per_window,
            window_seconds=rate_limits.get('window_seconds', 60),
            safety_margin=rate_limits.get('safety_margin', 0.9),
        )

        self._is_pro = is_pro
        self._initialized = True
        logger.info(f"CoinGecko provider initialised (pro={is_pro})")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})
        coin_id = params.get('coin_id', '')

        if not coin_id:
            return ValidationResult(valid=False, reason="'coin_id' param required")

        try:
            self.rate_limiter.acquire()

            if endpoint_type == 'market_chart':
                vs_currency = params.get('vs_currency', 'usd')
                if not self._is_pro:
                    # Demo tier: probe with /market_chart?days=1
                    result = self.client.get_market_chart_days(
                        coin_id=coin_id,
                        vs_currency=vs_currency,
                        days=1,
                    )
                else:
                    # Pro tier: probe with yesterday's 1-day range
                    now = datetime.now(timezone.utc)
                    probe_from = int((now - timedelta(days=2)).timestamp())
                    probe_to = int((now - timedelta(days=1)).timestamp())
                    result = self.client.get_market_chart_range(
                        coin_id=coin_id,
                        vs_currency=vs_currency,
                        from_ts=probe_from,
                        to_ts=probe_to,
                    )
                n_prices = len(result.get('prices', []))
                return ValidationResult(
                    valid=True,
                    reason=f"Probe returned {n_prices} price point(s)",
                )

            elif endpoint_type == 'coin_data':
                result = self.client.get_coin_data(coin_id=coin_id)
                has_market = 'market_data' in result
                return ValidationResult(
                    valid=has_market,
                    reason="Coin data endpoint accessible" if has_market else "No market_data in response",
                )

            else:
                return ValidationResult(
                    valid=False,
                    reason=f"Unsupported endpoint_type '{endpoint_type}'",
                )

        except Exception as e:
            logger.error(f"CoinGecko validation failed: {e}", exc_info=True)
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
        coin_id = params.get('coin_id', '')

        if endpoint_type == 'market_chart':
            vs_currency = params.get('vs_currency', 'usd')

            if not self._is_pro:
                # Free/demo tier: use /market_chart?days=365 (last ~365 days)
                raw = self.client.get_market_chart_days(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    days=365,
                )
            else:
                from_ts = int(start_time.timestamp())
                to_ts = int(end_time.timestamp())
                raw = self.client.get_market_chart_range(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    from_ts=from_ts,
                    to_ts=to_ts,
                )

            return FetchResult(
                data=[raw],
                next_cursor=None,
                has_more=False,
                metadata={},
            )

        elif endpoint_type == 'coin_data':
            raw = self.client.get_coin_data(coin_id=coin_id)
            return FetchResult(
                data=[raw],
                next_cursor=None,
                has_more=False,
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
        endpoint_type = endpoint_config.get('endpoint_type', '')
        schema_type = params.get('schema_type', 'market_chart')
        asset_hint = params.get('asset', '')

        if endpoint_type == 'coin_data' or not self._is_pro:
            # Single call: coin_data snapshot or demo tier days=365
            label = 'coin_data' if endpoint_type == 'coin_data' else 'demo tier'
            records = self._fetch_chunk(
                endpoint_config, start_time, end_time, schema_type, asset_hint, label,
            )
            if records:
                yield records
            return

        # Pro tier: chunk into CHUNK_DAYS windows to avoid timeouts
        chunk_start = start_time
        chunk_num = 0

        while chunk_start < end_time:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_time)
            chunk_num += 1
            label = f"chunk {chunk_num} ({chunk_start.date()} -> {chunk_end.date()})"

            records = self._fetch_chunk(
                endpoint_config, chunk_start, chunk_end, schema_type, asset_hint, label,
            )
            if records:
                yield records

            chunk_start = chunk_end

        logger.info(f"CoinGecko stream complete after {chunk_num} chunk(s)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_chunk(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
        schema_type: str,
        asset_hint: str,
        label: str,
    ) -> Optional[List[Dict]]:
        """Fetch a single chunk with retry, transform, and return records."""
        self.rate_limiter.acquire()
        try:
            result = self.fetch_data_batch(endpoint_config, start_time, end_time)
        except Exception as e:
            error_info = self.handle_error(e, {'endpoint_config': endpoint_config})
            if error_info.get('retry'):
                wait = error_info.get('wait_seconds', 10)
                logger.warning(f"Retrying {label} after {wait}s...")
                time.sleep(wait)
                result = self.fetch_data_batch(endpoint_config, start_time, end_time)
            else:
                raise

        raw_payload = result.data[0] if result.data else {}
        if not raw_payload:
            return None

        standard = self.transformer.transform(
            raw_payload,
            schema_type=schema_type,
            start_time=start_time,
            end_time=end_time,
            asset_hint=asset_hint,
        )
        if standard:
            logger.info(f"CoinGecko {label}: {len(standard)} records")
            return standard
        return None

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
            if status in (401, 403):
                return {'retry': False, 'error_type': 'auth_error', 'fatal': True}
            if status == 404:
                return {'retry': False, 'error_type': 'not_found', 'fatal': True}
        return {'retry': True, 'wait_seconds': 15, 'error_type': 'unknown'}
