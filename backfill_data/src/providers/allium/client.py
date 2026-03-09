"""Allium Developer REST API client"""

import requests
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger('backfill_system.allium')


class AlliumClient:
    """
    HTTP client for the Allium Developer REST API v1.

    Base URL: ``https://api.allium.so/api/v1``
    Auth: ``X-API-KEY`` header.

    Supported endpoint families
    ---------------------------
    developer/prices/history
        POST /developer/prices/history
        OHLCV price history for any EVM/Solana token by contract address.

    developer/tokens
        GET /developer/tokens
        Token search with live price, volume, market cap.

    developer/{chain}/raw/blocks
        GET /developer/{chain}/raw/blocks
        Raw block data for any supported chain.

    developer/{chain}/raw/transactions
        GET /developer/{chain}/raw/transactions
        Raw transaction data.

    developer/{chain}/dex/trades
        GET /developer/{chain}/dex/trades
        DEX trade events.

    developer/bitcoin/raw/blocks
        GET /developer/bitcoin/raw/blocks
        Bitcoin-specific block data.

    developer/bitcoin/raw/transactions
        GET /developer/bitcoin/raw/transactions
        Bitcoin-specific transaction data.

    Note: The Explorer SQL API (/explorer/queries/{id}/run) requires separate
    compute credits and is NOT used here. The Developer APIs above are
    subscription-based and work with a standard API key.
    """

    # Chains with Developer API price history support
    PRICE_HISTORY_CHAINS = {
        'ethereum', 'arbitrum', 'avalanche', 'bsc', 'base', 'blast',
        'celo', 'hyperevm', 'optimism', 'polygon', 'solana', 'soneium',
        'unichain', 'worldchain', 'zora', 'zksync',
    }

    def __init__(self, api_key: str, base_url: str = "https://api.allium.so/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        })
        logger.info(f"Allium client initialised: {self.base_url}")

    # ------------------------------------------------------------------
    # Token prices (OHLCV)
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        addresses: List[Dict[str, str]],
        start_timestamp: str,
        end_timestamp: str,
        time_granularity: str = "1d",
        cursor: Optional[str] = None,
    ) -> Dict:
        """
        POST /developer/prices/history

        Args:
            addresses: List of ``{"chain": "ethereum", "token_address": "0x..."}`` dicts.
            start_timestamp: ISO-8601 e.g. "2024-01-01T00:00:00Z"
            end_timestamp:   ISO-8601 e.g. "2024-01-31T00:00:00Z"
            time_granularity: "15s" | "1m" | "5m" | "1h" | "1d"
            cursor: pagination cursor from previous response

        Returns:
            ``{"items": [{"mint": "0x...", "chain": "...", "prices": [...]}]}``
            Each price item: ``{"timestamp", "price", "open", "high", "close", "low"}``
        """
        body: Dict[str, Any] = {
            "addresses": addresses,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "time_granularity": time_granularity,
        }
        if cursor:
            body["cursor"] = cursor
        return self._post(f"{self.base_url}/developer/prices/history", body)

    # ------------------------------------------------------------------
    # Token search / metadata
    # ------------------------------------------------------------------

    def search_tokens(
        self,
        query: str,
        chain: Optional[str] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> List[Dict]:
        """
        GET /developer/tokens
        Search tokens by name/symbol with live price data.
        """
        params: Dict = {"q": query, "limit": limit}
        if chain:
            params["chain"] = chain
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/tokens", params)

    def get_token_by_address(self, chain: str, address: str) -> List[Dict]:
        """GET /developer/tokens/chain-address — look up token by contract address."""
        params = {"chain": chain, "address": address}
        return self._get(f"{self.base_url}/developer/tokens/chain-address", params)

    # ------------------------------------------------------------------
    # Raw chain data — EVM chains
    # ------------------------------------------------------------------

    def get_blocks(
        self,
        chain: str,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict:
        """GET /developer/{chain}/raw/blocks — paginated block data."""
        params: Dict = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/{chain}/raw/blocks", params)

    def get_transactions(
        self,
        chain: str,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict:
        """GET /developer/{chain}/raw/transactions — paginated transaction data."""
        params: Dict = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/{chain}/raw/transactions", params)

    def get_dex_trades(
        self,
        chain: str,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict:
        """GET /developer/{chain}/dex/trades — DEX trade events."""
        params: Dict = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/{chain}/dex/trades", params)

    # ------------------------------------------------------------------
    # Raw chain data — Bitcoin
    # ------------------------------------------------------------------

    def get_bitcoin_blocks(
        self,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict:
        """GET /developer/bitcoin/raw/blocks."""
        params: Dict = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/bitcoin/raw/blocks", params)

    def get_bitcoin_transactions(
        self,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict:
        """GET /developer/bitcoin/raw/transactions."""
        params: Dict = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        return self._get(f"{self.base_url}/developer/bitcoin/raw/transactions", params)

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

    def _post(self, url: str, body: Dict) -> Any:
        import json
        try:
            logger.debug(f"POST {url}")
            response = self.session.post(url, data=json.dumps(body), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(
                f"HTTP {e.response.status_code} from {url}: "
                f"{e.response.text[:500] if e.response else 'N/A'}"
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Request error posting to {url}: {e}")
            raise

    def close(self):
        self.session.close()
        logger.debug("Allium client session closed")
