"""CoinMetrics API client implementation"""

import requests
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger('backfill_system.coinmetrics')


class CoinMetricsClient:
    """
    HTTP client for CoinMetrics API v4.
    Handles pagination, catalog queries, and response parsing.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.coinmetrics.io/v4"):
        """
        Initialize CoinMetrics client.
        
        Args:
            api_key: CoinMetrics API key
            base_url: Base URL for CoinMetrics API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Api-Key': self.api_key,
            'Accept': 'application/json'
        })
        
        logger.info(f"CoinMetrics client initialized: {self.base_url}")
    
    def fetch(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch data from a CoinMetrics endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., 'timeseries/asset-metrics')
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"Fetching from {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received {len(data.get('data', []))} records from {endpoint}")
            
            return data
            
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            raise
    
    def fetch_by_url(self, url: str) -> Dict:
        """
        Fetch data using a full URL (for pagination).
        
        Args:
            url: Full URL to fetch
            
        Returns:
            Response data as dictionary
        """
        try:
            logger.debug(f"Fetching from full URL: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received {len(data.get('data', []))} records")
            
            return data
            
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise
    
    def fetch_catalog(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch catalog information for an endpoint.
        Used for validation and availability checks.
        
        Args:
            endpoint: Catalog endpoint (e.g., 'catalog-v2/asset-metrics')
            params: Query parameters (e.g., assets, metrics)
            
        Returns:
            Catalog data
        """
        try:
            logger.debug(f"Fetching catalog: {endpoint}")
            data = self.fetch(endpoint, params)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch catalog {endpoint}: {e}")
            raise
    
    def get_asset_metrics_catalog(self, assets: str = None, metrics: str = None) -> Dict:
        """
        Get catalog for asset metrics.
        
        Args:
            assets: Comma-separated asset list (e.g., 'btc,eth')
            metrics: Comma-separated metric list (e.g., 'PriceUSD')
            
        Returns:
            Catalog data with availability information
        """
        params = {}
        if assets:
            params['assets'] = assets
        if metrics:
            params['metrics'] = metrics
        
        return self.fetch_catalog('catalog-v2/asset-metrics', params)
    
    def get_market_trades_catalog(self, markets: str = None) -> Dict:
        """
        Get catalog for market trades.
        
        Args:
            markets: Comma-separated market list
            
        Returns:
            Catalog data
        """
        params = {}
        if markets:
            params['markets'] = markets
        
        return self.fetch_catalog('catalog-v2/market-trades', params)
    
    def get_exchange_metrics_catalog(self, exchanges: str = None, metrics: str = None) -> Dict:
        """
        Get catalog for exchange metrics.
        
        Args:
            exchanges: Comma-separated exchange list
            metrics: Comma-separated metric list
            
        Returns:
            Catalog data
        """
        params = {}
        if exchanges:
            params['exchanges'] = exchanges
        if metrics:
            params['metrics'] = metrics
        
        return self.fetch_catalog('catalog-v2/exchange-metrics', params)
    
    def parse_catalog_times(self, catalog_data: Dict, asset_or_market: str) -> tuple:
        """
        Parse min and max times from catalog data.
        
        Args:
            catalog_data: Catalog response data
            asset_or_market: Asset or market identifier to look for
            
        Returns:
            Tuple of (min_time, max_time) as datetime objects
        """
        try:
            data = catalog_data.get('data', [])
            
            for item in data:
                if item.get('asset') == asset_or_market or item.get('market') == asset_or_market:
                    min_time_str = item.get('min_time')
                    max_time_str = item.get('max_time')
                    
                    if min_time_str and max_time_str:
                        # Parse ISO format timestamps
                        min_time = datetime.fromisoformat(min_time_str.replace('Z', '+00:00'))
                        max_time = datetime.fromisoformat(max_time_str.replace('Z', '+00:00'))
                        return (min_time, max_time)
            
            # If not found, return None
            return (None, None)
            
        except Exception as e:
            logger.error(f"Failed to parse catalog times: {e}")
            return (None, None)
    
    def fetch_market_orderbooks(
        self,
        markets: str,
        start_time: str,
        end_time: str,
        granularity: str = '1m',
        depth: int = 10,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Fetch market orderbook snapshots.
        
        Args:
            markets: Comma-separated market identifiers (e.g., 'coinbase-btc-usd-spot')
            start_time: Start time in ISO format
            end_time: End time in ISO format
            granularity: Snapshot frequency ('1m', '5m', '1h', etc.)
            depth: Number of bid/ask levels to return
            params: Additional query parameters
            
        Returns:
            Response data with orderbook snapshots
        """
        endpoint = 'timeseries/market-orderbooks'
        
        # Build parameters
        request_params = {
            'markets': markets,
            'start_time': start_time,
            'end_time': end_time,
            'granularity': granularity,
            'depth': depth
        }
        
        # Add any additional params
        if params:
            request_params.update(params)
        
        return self.fetch(endpoint, request_params)
    
    def close(self):
        """Close the HTTP session"""
        self.session.close()
        logger.debug("CoinMetrics client session closed")

