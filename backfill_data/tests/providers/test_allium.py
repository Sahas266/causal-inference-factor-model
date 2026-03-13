"""Unit tests for the Allium Developer REST API provider"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.providers.allium.provider import AlliumProvider
from src.providers.allium.rate_limiter import AlliumRateLimiter
from src.providers.allium.transformer import AlliumTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider_config():
    return {
        'provider_name': 'allium',
        'enabled': True,
        'api_config': {
            'base_url': 'https://api.allium.so/api/v1',
            'api_key': 'test-allium-key',
        },
        'rate_limits': {
            'requests_per_window': 10,
            'window_seconds': 10,
            'safety_margin': 1.0,
        },
    }


@pytest.fixture
def price_history_endpoint():
    return {
        'endpoint_type': 'developer/prices/history',
        'params': {
            'schema_type': 'token_price_history',
            'asset': 'weth',
            'addresses': [
                {
                    'chain': 'ethereum',
                    'token_address': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                }
            ],
            'time_granularity': '1d',
        },
    }


@pytest.fixture
def dex_trades_endpoint():
    return {
        'endpoint_type': 'developer/ethereum/dex/trades',
        'params': {
            'schema_type': 'dex_trades',
            'chain': 'ethereum',
            'asset': 'ethereum_dex',
            'limit': 1000,
        },
    }


@pytest.fixture
def initialized_provider(provider_config):
    provider = AlliumProvider()
    provider.initialize(provider_config)
    return provider


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------

def test_provider_name():
    assert AlliumProvider().provider_name == "allium"


def test_provider_initialization(provider_config):
    provider = AlliumProvider()
    provider.initialize(provider_config)
    assert provider._initialized is True
    assert provider.client is not None
    assert provider.rate_limiter is not None


def test_provider_requires_api_key():
    provider = AlliumProvider()
    with pytest.raises(ValueError, match="API key"):
        provider.initialize({
            'api_config': {'base_url': 'https://api.allium.so/api/v1'},
            'rate_limits': {},
        })


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_initialization():
    rl = AlliumRateLimiter(requests_per_window=100, window_seconds=60, safety_margin=0.9)
    assert rl.requests_per_window == 90   # 100 * 0.9
    assert rl.window_seconds == 60


def test_rate_limiter_acquire():
    rl = AlliumRateLimiter(requests_per_window=10, window_seconds=1, safety_margin=1.0)
    rl.acquire()
    assert rl.total_requests == 1


def test_rate_limiter_try_acquire_exhaustion():
    rl = AlliumRateLimiter(requests_per_window=2, window_seconds=60, safety_margin=1.0)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True
    assert rl.try_acquire() is False  # exhausted


# ---------------------------------------------------------------------------
# Transformer — token_price_history
# ---------------------------------------------------------------------------

@pytest.fixture
def transformer():
    return AlliumTransformer()


def test_transform_token_price_history(transformer):
    raw = {
        'items': [
            {
                'mint': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                'chain': 'ethereum',
                'decimals': 18,
                'prices': [
                    {
                        'timestamp': '2024-01-01T00:00:00Z',
                        'price': 2306.62,
                        'open': 2274.09,
                        'high': 2504.45,
                        'close': 2351.59,
                        'low': 2253.97,
                    },
                    {
                        'timestamp': '2024-01-02T00:00:00Z',
                        'price': 2400.00,
                        'open': 2351.59,
                        'high': 2450.00,
                        'close': 2410.00,
                        'low': 2340.00,
                    },
                ],
            }
        ]
    }
    records = transformer.transform(raw, 'token_price_history', asset_hint='weth')
    # 2 timestamps × 5 metrics = 10 records
    assert len(records) == 10
    metrics = {r['metric'] for r in records}
    assert metrics == {'price_usd', 'open_usd', 'high_usd', 'low_usd', 'close_usd'}
    for r in records:
        assert r['asset'] == 'weth'
        assert r['frequency'] == '1d'
        assert r['value'] is not None


def test_transform_token_price_history_date_filter(transformer):
    raw = {
        'items': [
            {
                'mint': '0xabc',
                'chain': 'ethereum',
                'prices': [
                    {'timestamp': '2024-01-01T00:00:00Z', 'price': 100, 'open': 99, 'high': 110, 'close': 105, 'low': 95},
                    {'timestamp': '2024-06-01T00:00:00Z', 'price': 200, 'open': 190, 'high': 210, 'close': 205, 'low': 185},
                ],
            }
        ]
    }
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 1, tzinfo=timezone.utc)
    records = transformer.transform(raw, 'token_price_history', start_time=start, end_time=end)
    # Only June record passes filter → 5 metrics
    assert len(records) == 5


def test_transform_token_price_history_empty(transformer):
    raw = {'items': []}
    records = transformer.transform(raw, 'token_price_history')
    assert records == []


# ---------------------------------------------------------------------------
# Transformer — dex_trades
# ---------------------------------------------------------------------------

def test_transform_dex_trades(transformer):
    raw = {
        'items': [
            {'block_timestamp': '2024-01-01T10:00:00Z', 'amount_usd': 1000.50},
            {'block_timestamp': '2024-01-01T14:00:00Z', 'amount_usd': 2000.25},
            {'block_timestamp': '2024-01-02T08:00:00Z', 'amount_usd': 500.00},
        ]
    }
    records = transformer.transform(raw, 'dex_trades', asset_hint='ethereum_dex')
    # 2 days × 2 metrics (volume_usd + trade_count) = 4 records
    assert len(records) == 4
    metrics = {r['metric'] for r in records}
    assert metrics == {'volume_usd', 'trade_count'}

    # Check aggregation: day 1 should have 3000.75 volume, 2 trades
    day1_vol = [r for r in records if '2024-01-01' in r['time'] and r['metric'] == 'volume_usd']
    assert len(day1_vol) == 1
    assert float(day1_vol[0]['value']) == pytest.approx(3000.75, rel=1e-4)


def test_transform_dex_trades_empty(transformer):
    raw = {'items': []}
    records = transformer.transform(raw, 'dex_trades')
    assert records == []


# ---------------------------------------------------------------------------
# Transformer — unknown schema
# ---------------------------------------------------------------------------

def test_transform_unknown_schema_type(transformer):
    with pytest.raises(ValueError, match="unknown schema type"):
        transformer.transform({}, 'nonexistent_type')


# ---------------------------------------------------------------------------
# Transformer — infer schema type
# ---------------------------------------------------------------------------

def test_infer_token_price_history(transformer):
    raw = {'items': [{'prices': [{'timestamp': '2024-01-01', 'price': 100}]}]}
    assert transformer.infer_schema_type(raw) == 'token_price_history'


def test_infer_dex_trades(transformer):
    raw = {'items': [{'block_timestamp': '2024-01-01', 'amount_usd': 100}]}
    assert transformer.infer_schema_type(raw) == 'dex_trades'


# ---------------------------------------------------------------------------
# fetch_data_batch — price history (mocked)
# ---------------------------------------------------------------------------

def test_fetch_price_history_batch(initialized_provider, price_history_endpoint):
    mock_response = {
        'items': [
            {
                'mint': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                'chain': 'ethereum',
                'prices': [
                    {'timestamp': '2024-01-01T00:00:00Z', 'price': 2300, 'open': 2290, 'high': 2400, 'close': 2350, 'low': 2250}
                ],
            }
        ],
    }
    initialized_provider.client.get_price_history = Mock(return_value=mock_response)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = initialized_provider.fetch_data_batch(price_history_endpoint, start, end)

    assert len(result.data) == 1
    assert result.has_more is False
    assert result.next_cursor is None


def test_fetch_price_history_pagination(initialized_provider, price_history_endpoint):
    mock_page1 = {
        'items': [{'mint': '0xabc', 'chain': 'ethereum', 'prices': []}],
        'next_cursor': 'cursor-123',
    }
    mock_page2 = {
        'items': [{'mint': '0xabc', 'chain': 'ethereum', 'prices': []}],
    }
    initialized_provider.client.get_price_history = Mock(side_effect=[mock_page1, mock_page2])

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)

    result1 = initialized_provider.fetch_data_batch(price_history_endpoint, start, end)
    assert result1.has_more is True
    assert result1.next_cursor == 'cursor-123'

    result2 = initialized_provider.fetch_data_batch(price_history_endpoint, start, end, cursor='cursor-123')
    assert result2.has_more is False


# ---------------------------------------------------------------------------
# fetch_data_batch — dex trades (mocked)
# ---------------------------------------------------------------------------

def test_fetch_dex_trades_batch(initialized_provider, dex_trades_endpoint):
    mock_response = {
        'items': [
            {'block_timestamp': '2024-01-01T10:00:00Z', 'amount_usd': 1000},
        ],
    }
    initialized_provider.client.get_dex_trades = Mock(return_value=mock_response)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = initialized_provider.fetch_data_batch(dex_trades_endpoint, start, end)

    assert len(result.data) == 1
    assert result.has_more is False


# ---------------------------------------------------------------------------
# fetch_data_stream (mocked)
# ---------------------------------------------------------------------------

def test_fetch_data_stream_prices(initialized_provider, price_history_endpoint):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_price_history = Mock(return_value={
        'items': [
            {
                'mint': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                'chain': 'ethereum',
                'prices': [
                    {'timestamp': '2024-01-01T00:00:00Z', 'price': 2300, 'open': 2290, 'high': 2400, 'close': 2350, 'low': 2250}
                ],
            }
        ],
    })

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    batches = list(initialized_provider.fetch_data_stream(price_history_endpoint, start, end))
    assert len(batches) == 1
    assert len(batches[0]) == 5  # 1 timestamp × 5 metrics
    metrics = {r['metric'] for r in batches[0]}
    assert 'price_usd' in metrics


# ---------------------------------------------------------------------------
# validate_endpoint (mocked)
# ---------------------------------------------------------------------------

def test_validate_price_history_success(initialized_provider, price_history_endpoint):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_price_history = Mock(return_value={
        'items': [{'mint': '0xabc', 'chain': 'ethereum', 'prices': []}]
    })

    result = initialized_provider.validate_endpoint(price_history_endpoint)
    assert result.valid is True


def test_validate_dex_trades_success(initialized_provider, dex_trades_endpoint):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_dex_trades = Mock(return_value={
        'items': [{'block_timestamp': '2024-01-01', 'amount_usd': 100}]
    })

    result = initialized_provider.validate_endpoint(dex_trades_endpoint)
    assert result.valid is True


def test_validate_endpoint_unsupported_type(initialized_provider):
    bad_config = {'endpoint_type': 'explorer/sql', 'params': {}}
    result = initialized_provider.validate_endpoint(bad_config)
    assert result.valid is False
    assert 'Unsupported' in result.reason


def test_validate_price_history_missing_addresses(initialized_provider):
    bad_config = {
        'endpoint_type': 'developer/prices/history',
        'params': {},
    }
    result = initialized_provider.validate_endpoint(bad_config)
    assert result.valid is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

import requests as req_lib

def test_handle_rate_limit_error(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 429
    mock_resp.headers = {'Retry-After': '30'}
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is True
    assert info['wait_seconds'] == 30
    assert info['error_type'] == 'rate_limit'


def test_handle_auth_error(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 401
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is False
    assert info['fatal'] is True


def test_handle_insufficient_credits(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 402
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is False
    assert info['error_type'] == 'insufficient_credits'


def test_handle_server_error(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 503
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is True
    assert info['error_type'] == 'server_error'


def test_handle_not_found_error(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 404
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is False
    assert info['fatal'] is True
