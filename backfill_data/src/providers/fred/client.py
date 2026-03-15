"""FRED (Federal Reserve Economic Data) REST API client."""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger('backfill_system.fred')

_BASE_URL = "https://api.stlouisfed.org/fred"


class FredClient:
    """HTTP client for the FRED API."""

    def __init__(self, api_key: str, base_url: str = _BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        logger.info("FRED client initialized: %s", self.base_url)

    def get_series_observations(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
    ) -> Dict:
        """
        GET /series/observations

        Args:
            series_id: FRED series ID (e.g. "DFF", "VIXCLS")
            observation_start: Start date YYYY-MM-DD
            observation_end: End date YYYY-MM-DD

        Returns:
            {"observations": [{"date": "...", "value": "..."}, ...]}
        """
        params: Dict[str, Any] = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
        }
        if observation_start:
            params['observation_start'] = observation_start
        if observation_end:
            params['observation_end'] = observation_end

        return self._get(f"{self.base_url}/series/observations", params)

    def get_series_info(self, series_id: str) -> Dict:
        """GET /series — metadata about a series."""
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
        }
        return self._get(f"{self.base_url}/series", params)

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        try:
            logger.debug("GET %s params=%s", url, {k: v for k, v in (params or {}).items() if k != 'api_key'})
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            body = e.response.text[:300] if e.response and e.response.text else ""
            logger.error("HTTP %s from %s: %s", status, url, body)
            raise
        except requests.RequestException as e:
            logger.error("Request error: %s", e)
            raise

    def close(self):
        self.session.close()
