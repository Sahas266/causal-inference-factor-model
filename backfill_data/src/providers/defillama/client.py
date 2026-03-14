"""DefiLlama REST API client."""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger('backfill_system.defillama')

# Public APIs
_FREE_BASE = "https://api.llama.fi"
_STABLECOINS_BASE = "https://stablecoins.llama.fi"
_COINS_BASE = "https://coins.llama.fi"
# Pro API (API key in URL path)
_PRO_BASE = "https://pro-api.llama.fi"


class DefiLlamaClient:
    """HTTP client for DefiLlama free and pro APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        free_base_url: str = _FREE_BASE,
        stablecoins_base_url: str = _STABLECOINS_BASE,
        coins_base_url: str = _COINS_BASE,
        pro_base_url: str = _PRO_BASE,
    ):
        self.api_key = api_key
        self.free_base = free_base_url.rstrip('/')
        self.stablecoins_base = stablecoins_base_url.rstrip('/')
        self.coins_base = coins_base_url.rstrip('/')
        self.pro_base = pro_base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        logger.info("DefiLlama client initialized (pro=%s)", 'yes' if api_key else 'no')

    # ------------------------------------------------------------------
    # TVL
    # ------------------------------------------------------------------

    def get_protocol_tvl(self, protocol: str) -> Dict:
        """GET /protocol/{protocol} - all historical TVL."""
        return self._get(f"{self.free_base}/protocol/{protocol}")

    def get_chain_tvl(self, chain: str) -> List[Dict]:
        """GET /v2/historicalChainTvl/{chain}."""
        return self._get(f"{self.free_base}/v2/historicalChainTvl/{chain}")

    def get_all_chains_tvl(self) -> List[Dict]:
        """GET /v2/historicalChainTvl."""
        return self._get(f"{self.free_base}/v2/historicalChainTvl")

    # ------------------------------------------------------------------
    # DEX volumes
    # ------------------------------------------------------------------

    def get_dex_summary(self, protocol: str) -> Dict:
        """GET /summary/dexs/{protocol} - daily volume history."""
        return self._get(
            f"{self.free_base}/summary/dexs/{protocol}",
            params={"dataType": "dailyVolume"},
        )

    def get_dex_overview(self, chain: Optional[str] = None) -> Dict:
        """GET /overview/dexs or /overview/dexs/{chain}."""
        path = f"/overview/dexs/{chain}" if chain else "/overview/dexs"
        return self._get(
            f"{self.free_base}{path}",
            params={"excludeTotalDataChartBreakdown": "true"},
        )

    # ------------------------------------------------------------------
    # Fees and revenue
    # ------------------------------------------------------------------

    def get_fees_summary(self, protocol: str, data_type: str = "dailyFees") -> Dict:
        """GET /summary/fees/{protocol} - daily fees/revenue history."""
        return self._get(
            f"{self.free_base}/summary/fees/{protocol}",
            params={"dataType": data_type},
        )

    # ------------------------------------------------------------------
    # Stablecoins
    # ------------------------------------------------------------------

    def get_stablecoin_charts(self, chain: str = "all", stablecoin_id: Optional[int] = None) -> List[Dict]:
        """GET /stablecoincharts/{chain} on stablecoins.llama.fi.

        Args:
            chain: Chain name or "all" for aggregate.
            stablecoin_id: Optional DefiLlama stablecoin ID to filter to a single stablecoin.
        """
        params = {}
        if stablecoin_id is not None:
            params["stablecoin"] = stablecoin_id
        return self._get(f"{self.stablecoins_base}/stablecoincharts/{chain}", params=params or None)

    # ------------------------------------------------------------------
    # Coin prices
    # ------------------------------------------------------------------

    def get_coin_chart(self, coins: str, period: str = "365d", span: int = 0) -> Dict:
        """GET /chart/{coins} on coins.llama.fi."""
        params: Dict = {"period": period}
        if span:
            params["span"] = span
        return self._get(f"{self.coins_base}/chart/{coins}", params=params)

    # ------------------------------------------------------------------
    # Pro yields
    # ------------------------------------------------------------------

    def get_yield_pool_chart(self, pool_id: str) -> Dict:
        """GET /yields/chart/{pool} on pro-api.llama.fi (requires API key)."""
        if not self.api_key:
            raise ValueError("DEFILLAMA_API_KEY required for Pro endpoints")
        return self._get(f"{self.pro_base}/{self.api_key}/yields/chart/{pool_id}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        try:
            logger.debug("GET %s params=%s", url, params)
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
