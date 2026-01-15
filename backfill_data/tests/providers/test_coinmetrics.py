"""Tests for CoinMetrics provider"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.providers.coinmetrics.provider import CoinMetricsProvider
from src.providers.coinmetrics.rate_limiter import CoinMetricsRateLimiter
from src.providers.coinmetrics.transformer import CoinMetricsTransformer


@pytest.fixture
def provider_config():
    """Provider configuration fixture"""
    return {
        'provider_name': 'coinmetrics',
        'enabled': True,
        'api_config': {
            'base_url': 'https://api.coinmetrics.io/v4',
            'api_key': 'test-key'
        },
        'rate_limits': {
            'requests_per_window': 100,
            'window_seconds': 10,
            'safety_margin': 0.9
        }
    }


def test_provider_name():
    """Test provider name property"""
    provider = CoinMetricsProvider()
    assert provider.provider_name == "coinmetrics"


def test_provider_initialization(provider_config):
    """Test provider initialization"""
    provider = CoinMetricsProvider()
    provider.initialize(provider_config)
    
    assert provider._initialized is True
    assert provider.client is not None
    assert provider.rate_limiter is not None


def test_rate_limiter_initialization():
    """Test rate limiter initialization"""
    limiter = CoinMetricsRateLimiter(
        requests_per_window=100,
        window_seconds=10,
        safety_margin=0.9
    )
    
    assert limiter.requests_per_window == 90  # 100 * 0.9
    assert limiter.window_seconds == 10


def test_rate_limiter_acquire():
    """Test rate limiter acquire method"""
    limiter = CoinMetricsRateLimiter(
        requests_per_window=10,
        window_seconds=1,
        safety_margin=1.0
    )
    
    # Should be able to acquire without blocking
    limiter.acquire()
    assert limiter.total_requests == 1


def test_rate_limiter_try_acquire():
    """Test rate limiter try_acquire method"""
    limiter = CoinMetricsRateLimiter(
        requests_per_window=2,
        window_seconds=1,
        safety_margin=1.0
    )
    
    # First two should succeed
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    
    # Third should fail (no tokens left)
    assert limiter.try_acquire() is False


def test_transformer_asset_metrics():
    """Test transformer for asset metrics"""
    transformer = CoinMetricsTransformer()
    
    raw_data = [
        {
            'asset': 'btc',
            'time': '2024-01-01T00:00:00.000000000Z',
            'PriceUSD': '42000.50',
            'CapMrktCurUSD': '800000000000'
        }
    ]
    
    result = transformer.transform_asset_metrics(raw_data)
    
    assert len(result) == 2  # Two metrics
    assert result[0]['asset'] == 'btc'
    assert result[0]['metric'] in ['PriceUSD', 'CapMrktCurUSD']


def test_transformer_infer_schema_type():
    """Test schema type inference"""
    transformer = CoinMetricsTransformer()
    
    # Asset metrics
    data = [{'asset': 'btc', 'time': '2024-01-01T00:00:00Z'}]
    assert transformer.infer_schema_type(data) == 'asset_metrics'
    
    # Market trades
    data = [{'market': 'binance-btc-usdt', 'coin_metrics_id': '123'}]
    assert transformer.infer_schema_type(data) == 'market_trades'
    
    # Exchange metrics
    data = [{'exchange': 'binance', 'time': '2024-01-01T00:00:00Z'}]
    assert transformer.infer_schema_type(data) == 'exchange_metrics'


@patch('src.providers.coinmetrics.client.CoinMetricsClient')
def test_provider_validation(mock_client_class, provider_config):
    """Test endpoint validation"""
    # Setup mock
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    mock_client.fetch_catalog.return_value = {
        'data': [
            {
                'asset': 'btc',
                'min_time': '2015-01-01T00:00:00Z',
                'max_time': '2024-12-31T23:59:59Z'
            }
        ]
    }
    mock_client.parse_catalog_times.return_value = (
        datetime(2015, 1, 1),
        datetime(2024, 12, 31)
    )
    
    # Initialize provider
    provider = CoinMetricsProvider()
    provider.initialize(provider_config)
    provider.client = mock_client
    
    # Validate endpoint
    endpoint_config = {
        'endpoint_type': 'timeseries/asset-metrics',
        'params': {
            'assets': 'btc',
            'metrics': 'PriceUSD'
        }
    }
    
    result = provider.validate_endpoint(endpoint_config)
    
    assert result.valid is True
    assert result.adjusted_start_time is not None


def test_provider_error_handling():
    """Test provider error handling"""
    import requests
    
    provider = CoinMetricsProvider()
    
    # Test 429 rate limit error
    error = requests.HTTPError()
    error.response = Mock()
    error.response.status_code = 429
    error.response.headers = {'Retry-After': '10'}
    
    result = provider.handle_error(error, {})
    assert result['retry'] is True
    assert result['wait_seconds'] == 10
    assert result['error_type'] == 'rate_limit'
    
    # Test 401 auth error
    error.response.status_code = 401
    result = provider.handle_error(error, {})
    assert result['retry'] is False
    assert result['fatal'] is True

