"""CoinGecko API client"""

import requests
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger('backfill_system.coingecko')


class CoinGeckoClient:
    """
    HTTP client for the CoinGecko API v3.

    Free tier:  https://api.coingecko.com/api/v3
    Pro tier:   https://pro-api.coingecko.com/api/v3

    Auth:
        Free: ``x-cg-demo-api-key`` header (optional)
        Pro:  ``x-cg-pro-api-key`` header

    Key endpoints used:
        /coins/{id}/market_chart/range — OHLCV timeseries (prices, market_caps, total_volumes)
        /coins/{id}                    — Full coin data (ath, atl, supply, fdv)
    """

    def __init__(
        self,
        base_url: str = "https://api.coingecko.com/api/v3",
        api_key: Optional[str] = None,
        is_pro: bool = False,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.is_pro = is_pro
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

        if api_key:
            if is_pro:
                self.session.headers['x-cg-pro-api-key'] = api_key
            else:
                self.session.headers['x-cg-demo-api-key'] = api_key

        logger.info(
            f"CoinGecko client initialised: {self.base_url} "
            f"(pro={is_pro}, key={'set' if api_key else 'none'})"
        )

    # ------------------------------------------------------------------
    # Market chart (timeseries)
    # ------------------------------------------------------------------

    def get_market_chart_range(
        self,
        coin_id: str,
        vs_currency: str,
        from_ts: int,
        to_ts: int,
    ) -> Dict:
        """
        GET /coins/{id}/market_chart/range

        Args:
            coin_id: CoinGecko coin ID, e.g. "ethereum"
            vs_currency: Target currency, e.g. "usd"
            from_ts: Start time as unix timestamp (seconds)
            to_ts: End time as unix timestamp (seconds)

        Returns:
            {
                "prices": [[timestamp_ms, value], ...],
                "market_caps": [[timestamp_ms, value], ...],
                "total_volumes": [[timestamp_ms, value], ...]
            }
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': vs_currency,
            'from': from_ts,
            'to': to_ts,
        }
        return self._get(url, params)

    def get_market_chart_days(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> Dict:
        """
        GET /coins/{id}/market_chart?days=N

        Works on the free/demo tier (up to 365 days from now).
        Returns the same shape as get_market_chart_range.
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily',
        }
        return self._get(url, params)

    # ------------------------------------------------------------------
    # Coin data (snapshot)
    # ------------------------------------------------------------------

    def get_coin_data(self, coin_id: str) -> Dict:
        """
        GET /coins/{id}

        Returns full coin data including market_data with ath, atl,
        total_supply, max_supply, fully_diluted_valuation.

        Args:
            coin_id: CoinGecko coin ID, e.g. "ethereum"

        Returns:
            Full coin object with nested market_data.
        """
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'false',
            'developer_data': 'false',
        }
        return self._get(url, params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        try:
            logger.debug(f"GET {url} params={params}")
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(
                f"HTTP {e.response.status_code} from {url}: "
                f"{e.response.text[:500] if e.response else 'N/A'}"
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Request error from {url}: {e}")
            raise

    def close(self):
        self.session.close()
        logger.debug("CoinGecko client session closed")
