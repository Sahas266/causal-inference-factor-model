"""Dune Analytics data provider"""

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
from .client import DuneClient
from .rate_limiter import DuneRateLimiter
from .transformer import DuneTransformer

logger = logging.getLogger('backfill_system.dune')

# Max consecutive retries for a single page before giving up.
MAX_PAGE_RETRIES = 5


class DuneProvider(DataProviderInterface):
    """
    Dune Analytics implementation of DataProviderInterface.

    Fetches results from pre-saved Dune queries. Queries must be created
    in the Dune UI first — the API cannot run ad-hoc SQL.

    Supported endpoint types
    ------------------------
    ``query_results``
        GET /query/{query_id}/results
        Fetches the latest cached results for a saved query.
        Required params: ``query_id`` (int)
        Schema type: ``query_results``

    Endpoint config example:

    .. code-block:: json

        {
          "endpoint_type": "query_results",
          "params": {
            "schema_type": "query_results",
            "query_id": 12345,
            "asset": "eth",
            "limit": 10000
          }
        }
    """

    def __init__(self):
        self.config: Optional[Dict] = None
        self.client: Optional[DuneClient] = None
        self.rate_limiter: Optional[DuneRateLimiter] = None
        self.transformer = DuneTransformer()
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "dune"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, provider_config: Dict) -> None:
        self.config = provider_config

        api_key = provider_config['api_config'].get('api_key')
        if not api_key:
            api_key = os.getenv(provider_config['api_config'].get('api_key_env', 'DUNE_API_KEY'))
        if not api_key:
            raise ValueError("Dune API key not found in config or DUNE_API_KEY env var")

        base_url = provider_config['api_config'].get('base_url', 'https://api.dune.com/api/v1')
        self.client = DuneClient(api_key, base_url)

        rate_limits = provider_config.get('rate_limits', {})
        self.rate_limiter = DuneRateLimiter(
            requests_per_window=rate_limits.get('requests_per_window', 40),
            window_seconds=rate_limits.get('window_seconds', 60),
            safety_margin=rate_limits.get('safety_margin', 0.9),
        )

        self._initialized = True
        logger.info("Dune provider initialised")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_endpoint(self, endpoint_config: Dict) -> ValidationResult:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        endpoint_type = endpoint_config.get('endpoint_type', '')
        params = endpoint_config.get('params', {})

        if endpoint_type != 'query_results':
            return ValidationResult(
                valid=False,
                reason=f"Unsupported endpoint_type '{endpoint_type}'. Only 'query_results' is supported."
            )

        query_id = params.get('query_id')
        if not query_id:
            return ValidationResult(valid=False, reason="'query_id' param is required")

        try:
            self.rate_limiter.acquire()
            result = self.client.get_query_results(int(query_id), limit=1)
            state = result.get('state', '')
            row_count = len(result.get('result', {}).get('rows', []))
            return ValidationResult(
                valid=True,
                reason=f"Query {query_id} accessible, state={state}, probe returned {row_count} row(s)"
            )
        except Exception as e:
            logger.error(f"Dune validation failed for query {query_id}: {e}", exc_info=True)
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

        params = endpoint_config.get('params', {})
        query_id = int(params['query_id'])
        limit = params.get('limit', 10000)
        offset = int(cursor) if cursor else 0

        raw = self.client.get_query_results(query_id, limit=limit, offset=offset)

        rows = raw.get('result', {}).get('rows', [])
        has_more = len(rows) >= limit
        next_cursor = str(offset + len(rows)) if has_more else None

        return FetchResult(
            data=[raw],
            next_cursor=next_cursor,
            has_more=has_more,
            metadata={'query_id': query_id, 'offset': offset, 'rows_fetched': len(rows)},
        )

    def fetch_data_stream(
        self,
        endpoint_config: Dict,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[List[Dict], None, None]:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        params = endpoint_config.get('params', {})
        schema_type = params.get('schema_type', 'query_results')
        asset_hint = params.get('asset', '')

        cursor = None
        page = 0
        retries = 0  # consecutive failures for the current page

        while True:
            page += 1
            self.rate_limiter.acquire()

            try:
                result = self.fetch_data_batch(endpoint_config, start_time, end_time, cursor)
            except Exception as e:
                error_info = self.handle_error(e, {'endpoint_config': endpoint_config})
                retries += 1
                if error_info.get('retry') and retries <= MAX_PAGE_RETRIES:
                    # Linear backoff, capped at 4x the base wait.
                    wait = error_info.get('wait_seconds', 10) * min(retries, 4)
                    logger.warning(
                        f"Retry {retries}/{MAX_PAGE_RETRIES} after {wait}s..."
                    )
                    time.sleep(wait)
                    page -= 1  # keep page numbering accurate on retry
                    continue
                raise

            retries = 0

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
                logger.info(f"Dune stream complete after {page} page(s)")
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
            if status in (401, 403):
                return {'retry': False, 'error_type': 'auth_error', 'fatal': True}
            if status == 404:
                return {'retry': False, 'error_type': 'not_found', 'fatal': True}
        return {'retry': True, 'wait_seconds': 15, 'error_type': 'unknown'}
