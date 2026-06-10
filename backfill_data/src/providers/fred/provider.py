"""FRED (Federal Reserve Economic Data) provider implementation"""

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
from .client import FredClient
from .rate_limiter import FredRateLimiter
from .transformer import FredTransformer

logger = logging.getLogger('backfill_system.fred')

# Max consecutive retries for a single series before skipping it.
MAX_SERIES_RETRIES = 5


class FredProvider(DataProviderInterface):
    """
    FRED API implementation of DataProviderInterface.

    Supported endpoint types
    ------------------------
    ``series/observations``
        Fetches historical observations for one or more FRED series.
        Params:
            series_ids: comma-separated list of FRED series IDs
            asset: asset name for asset_metrics (default "macro")

    Endpoint config example::

        {
          "endpoint_type": "series/observations",
          "params": {
            "schema_type": "series_observations",
            "series_ids": "DFF,DGS2,DGS10,VIXCLS",
            "asset": "macro"
          }
        }
    """

    def __init__(self):
        self.config: Optional[Dict] = None
        self.client: Optional[FredClient] = None
        self.rate_limiter: Optional[FredRateLimiter] = None
        self.transformer = FredTransformer()
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "fred"

    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config
        api_config = provider_config.get('api_config', {})

        api_key = api_config.get('api_key')
        if not api_key:
            api_key = os.getenv(api_config.get('api_key_env', 'FRED_API_KEY'))
        if not api_key:
            raise ValueError("FRED_API_KEY required")

        base_url = api_config.get('base_url', 'https://api.stlouisfed.org/fred')

        self.client = FredClient(api_key=api_key, base_url=base_url)

        rate_limits = provider_config.get('rate_limits', {})
        self.rate_limiter = FredRateLimiter(
            requests_per_window=rate_limits.get('requests_per_window', 120),
            window_seconds=rate_limits.get('window_seconds', 60),
            safety_margin=rate_limits.get('safety_margin', 0.9),
        )

        self._initialized = True
        logger.info("FRED provider initialised")

    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        if endpoint_type != 'series/observations':
            return ValidationResult(valid=False, reason=f"Unknown endpoint_type '{endpoint_type}'")

        params = endpoint_config.get('params', {})
        series_ids = [s.strip() for s in params.get('series_ids', '').split(',') if s.strip()]
        if not series_ids:
            return ValidationResult(valid=False, reason="'series_ids' param required")

        try:
            self.rate_limiter.acquire()
            result = self.client.get_series_observations(
                series_id=series_ids[0],
                observation_start='2025-01-01',
                observation_end='2025-01-05',
            )
            n_obs = len(result.get('observations', []))
            return ValidationResult(valid=True, reason=f"Probe returned {n_obs} observations for {series_ids[0]}")
        except Exception as e:
            logger.warning(f"FRED validation failed: {e}")
            return ValidationResult(valid=False, reason=f"Validation error: {e}")

    def fetch_data_batch(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
        cursor: Optional[str] = None,
    ) -> FetchResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        params = endpoint_config.get('params', {})
        series_id = params.get('series_ids', '').split(',')[0].strip()

        raw = self.client.get_series_observations(
            series_id=series_id,
            observation_start=start_time.strftime('%Y-%m-%d'),
            observation_end=end_time.strftime('%Y-%m-%d'),
        )
        return FetchResult(data=[raw], next_cursor=None, has_more=False, metadata={})

    def fetch_data_stream(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[List[Dict], None, None]:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        params = endpoint_config.get('params', {})
        asset_hint = params.get('asset', 'macro')
        series_ids = [s.strip() for s in params.get('series_ids', '').split(',') if s.strip()]

        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')

        failed_series: List[str] = []

        for series_id in series_ids:
            raw = None
            retries = 0
            while True:
                self.rate_limiter.acquire()
                try:
                    raw = self.client.get_series_observations(
                        series_id=series_id,
                        observation_start=start_str,
                        observation_end=end_str,
                    )
                    break
                except Exception as e:
                    error_info = self.handle_error(e, {'series_id': series_id})
                    retries += 1
                    if error_info.get('retry') and retries <= MAX_SERIES_RETRIES:
                        # Linear backoff, capped at 4x the base wait.
                        wait = error_info.get('wait_seconds', 10) * min(retries, 4)
                        logger.warning(
                            f"Retry {retries}/{MAX_SERIES_RETRIES} for {series_id} after {wait}s..."
                        )
                        time.sleep(wait)
                        continue
                    logger.error(f"Skipping {series_id}: {e}")
                    failed_series.append(series_id)
                    break

            if raw is None:
                continue

            records = self.transformer.transform(
                raw,
                schema_type='series_observations',
                series_id=series_id,
                start_time=start_time,
                end_time=end_time,
                asset_hint=asset_hint,
            )

            if records:
                logger.info(f"FRED {series_id}: {len(records)} records [{start_str} -> {end_str}]")
                yield records

        # Raise after streaming the successful series so the orchestrator
        # marks this endpoint failed (retryable) instead of completed.
        # Already-yielded records are upserted idempotently, so a re-run
        # is safe.
        if failed_series:
            raise RuntimeError(
                f"FRED fetch failed for {len(failed_series)}/{len(series_ids)} "
                f"series: {', '.join(failed_series)}"
            )

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
            asset_hint=params.get('asset', 'macro'),
        )

    def handle_error(self, error: Exception, context: Dict) -> Dict:
        if isinstance(error, requests.HTTPError):
            status = error.response.status_code if error.response else 0
            if status == 429:
                return {'retry': True, 'wait_seconds': 60, 'error_type': 'rate_limit'}
            if status in (500, 502, 503, 504):
                return {'retry': True, 'wait_seconds': 10, 'error_type': 'server_error'}
            if status in (400, 404):
                return {'retry': False, 'error_type': 'bad_request', 'fatal': True}
        return {'retry': True, 'wait_seconds': 15, 'error_type': 'unknown'}
