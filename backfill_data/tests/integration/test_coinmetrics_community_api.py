"""
Integration tests for CoinMetrics Community API (no API key required).
Verified endpoints: pair-candles, asset-metrics.
"""

import pytest
import requests
from datetime import datetime, timezone
from src.providers.coinmetrics.transformer import CoinMetricsTransformer

class TestCoinMetricsCommunityAPI:
    """Test suite for CoinMetrics Community API endpoints."""

    def setup_method(self):
        self.transformer = CoinMetricsTransformer()
        self.base_url = "https://community-api.coinmetrics.io/v4/timeseries"

    @pytest.mark.integration
    def test_pair_candles_community(self):
        """Verify pair-candles endpoint works on community tier."""
        url = f"{self.base_url}/pair-candles"
        params = {
            "pairs": "btc-usd",
            "frequency": "1d",
            "limit_per_pair": 2,
            "pretty": "true"
        }
        
        response = requests.get(url, params=params)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json().get("data", [])
        assert len(data) > 0, "No data returned for pair-candles"
        assert data[0]["pair"] == "btc-usd"
        assert "price_close" in data[0]
        
        # Verify transformation
        transformed = self.transformer.transform(data, "pair_candles")
        assert len(transformed) == len(data)
        assert transformed[0]["pair"] == "btc-usd"
        assert "time" in transformed[0]
        assert transformed[0]["frequency"] == "1d"

    @pytest.mark.integration
    def test_asset_metrics_community(self):
        """Verify asset-metrics endpoint works on community tier."""
        url = f"{self.base_url}/asset-metrics"
        params = {
            "assets": "btc",
            "metrics": "PriceUSD",
            "frequency": "1d",
            "limit_per_asset": 2,
            "pretty": "true"
        }
        
        response = requests.get(url, params=params)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json().get("data", [])
        assert len(data) > 0, "No data returned for asset-metrics"
        assert data[0]["asset"] == "btc"
        assert "PriceUSD" in data[0]
        
        # Verify transformation (uses generic metric/value schema)
        transformed = self.transformer.transform(data, "asset_metrics")
        assert len(transformed) >= len(data)
        metrics = [r["metric"] for r in transformed]
        assert "PriceUSD" in metrics

    @pytest.mark.integration
    @pytest.mark.requires_api_key
    def test_pro_endpoints_return_403(self):
        """
        Verify market-level endpoints return 403 on community tier.
        This confirms they require a paid API key as assumed.
        """
        pro_endpoints = [
            "market-candles?markets=coinbase-btc-usd-spot&frequency=1d",
            "market-openinterest?markets=binance-BTCUSDT-future",
            "market-liquidations?markets=binance-BTCUSDT-future",
            "market-funding-rates?markets=binance-BTCUSDT-future",
            "market-implied-volatility?markets=deribit-BTC-28MAR25-200000-C-option",
            "market-greeks?markets=deribit-BTC-28MAR25-200000-C-option"
        ]
        
        for ep_query in pro_endpoints:
            url = f"{self.base_url}/{ep_query}&limit_per_market=1"
            response = requests.get(url)
            # We expect 403 Forbidden for community tier accessing pro endpoints
            assert response.status_code == 403, f"Expected 403 for {ep_query}, got {response.status_code}"
