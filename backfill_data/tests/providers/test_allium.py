"""Unit tests for the Allium provider"""

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
def endpoint_config():
    return {
        'endpoint_type': 'explorer/sql',
        'params': {
            'schema_type': 'on_chain_metrics',
            'timeout_seconds': 30,
            'sql': (
                "SELECT DATE_TRUNC('day', block_timestamp) AS time, "
                "'eth' AS asset, COUNT(*) AS tx_count "
                "FROM ethereum.raw.transactions "
                "WHERE block_timestamp >= '{start_time}' AND block_timestamp < '{end_time}' "
                "GROUP BY 1, 2 ORDER BY 1"
            ),
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
# SQL template rendering
# ---------------------------------------------------------------------------

def test_render_sql_substitution():
    template = "SELECT * FROM t WHERE ts >= '{start_time}' AND ts < '{end_time}'"
    rendered = AlliumProvider._render_sql(template, '2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z')
    assert '2024-01-01T00:00:00Z' in rendered
    assert '2024-02-01T00:00:00Z' in rendered
    assert '{start_time}' not in rendered
    assert '{end_time}' not in rendered


# ---------------------------------------------------------------------------
# Transformer — on_chain_metrics
# ---------------------------------------------------------------------------

@pytest.fixture
def transformer():
    return AlliumTransformer()


def test_transform_on_chain_metrics(transformer):
    raw = [
        {'time': '2024-01-01T00:00:00Z', 'asset': 'eth', 'active_addresses': 500000, 'tx_count': 1200000},
        {'time': '2024-01-02T00:00:00Z', 'asset': 'eth', 'active_addresses': 520000, 'tx_count': 1250000},
    ]
    records = transformer.transform(raw, 'on_chain_metrics')
    # 2 rows × 2 metrics = 4 records
    assert len(records) == 4
    metrics = {r['metric'] for r in records}
    assert 'active_addresses' in metrics
    assert 'tx_count' in metrics
    for r in records:
        assert r['asset'] == 'eth'
        assert r['frequency'] == '1d'
        assert r['value'] is not None


def test_transform_on_chain_metrics_null_value(transformer):
    raw = [{'time': '2024-01-01T00:00:00Z', 'asset': 'btc', 'fee': None}]
    records = transformer.transform(raw, 'on_chain_metrics')
    assert len(records) == 1
    assert records[0]['value'] is None


def test_transform_unknown_schema_type(transformer):
    with pytest.raises(ValueError, match="unknown schema type"):
        transformer.transform([], 'nonexistent_type')


# ---------------------------------------------------------------------------
# Transformer — dex_metrics
# ---------------------------------------------------------------------------

def test_transform_dex_metrics(transformer):
    raw = [
        {'time': '2024-01-01', 'protocol': 'uniswap_v3', 'volume_usd': '1000000.50', 'swap_count': 5000},
    ]
    records = transformer.transform(raw, 'dex_metrics')
    assert len(records) == 2  # volume_usd + swap_count
    assets = {r['asset'] for r in records}
    assert 'uniswap_v3' in assets


# ---------------------------------------------------------------------------
# Transformer — infer schema type
# ---------------------------------------------------------------------------

def test_infer_dex_schema(transformer):
    assert transformer.infer_schema_type([{'time': 't', 'protocol': 'uniswap'}]) == 'dex_metrics'


def test_infer_on_chain_schema(transformer):
    assert transformer.infer_schema_type([{'time': 't', 'asset': 'eth', 'tx_count': 100}]) == 'on_chain_metrics'


# ---------------------------------------------------------------------------
# fetch_data_batch (mocked)
# ---------------------------------------------------------------------------

def test_fetch_data_batch(initialized_provider, endpoint_config):
    mock_response = {
        'data': [
            {'time': '2024-01-01T00:00:00Z', 'asset': 'eth', 'tx_count': 1000000}
        ],
        'next_cursor': None,
        'columns': [{'name': 'time'}, {'name': 'asset'}, {'name': 'tx_count'}],
    }
    initialized_provider.client.run_sql = Mock(return_value=mock_response)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = initialized_provider.fetch_data_batch(endpoint_config, start, end)

    assert len(result.data) == 1
    assert result.has_more is False
    assert result.next_cursor is None


def test_fetch_data_batch_pagination(initialized_provider, endpoint_config):
    mock_page1 = {
        'data': [{'time': '2024-01-01', 'asset': 'eth', 'tx_count': 1}],
        'next_cursor': 'cursor-abc',
        'columns': [],
    }
    mock_page2 = {
        'data': [{'time': '2024-01-02', 'asset': 'eth', 'tx_count': 2}],
        'next_cursor': None,
        'columns': [],
    }
    initialized_provider.client.run_sql = Mock(side_effect=[mock_page1, mock_page2])

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    result1 = initialized_provider.fetch_data_batch(endpoint_config, start, end)
    assert result1.has_more is True
    assert result1.next_cursor == 'cursor-abc'

    result2 = initialized_provider.fetch_data_batch(endpoint_config, start, end, cursor='cursor-abc')
    assert result2.has_more is False


# ---------------------------------------------------------------------------
# fetch_data_stream (mocked)
# ---------------------------------------------------------------------------

def test_fetch_data_stream(initialized_provider, endpoint_config):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.run_sql = Mock(return_value={
        'data': [{'time': '2024-01-01T00:00:00Z', 'asset': 'eth', 'tx_count': 999}],
        'next_cursor': None,
        'columns': [],
    })

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    batches = list(initialized_provider.fetch_data_stream(endpoint_config, start, end))
    assert len(batches) == 1
    assert len(batches[0]) == 1   # 1 row × 1 metric
    assert batches[0][0]['metric'] == 'tx_count'


# ---------------------------------------------------------------------------
# validate_endpoint (mocked)
# ---------------------------------------------------------------------------

def test_validate_endpoint_success(initialized_provider, endpoint_config):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.run_sql = Mock(return_value={'data': [{'tx_count': 1}], 'columns': []})

    result = initialized_provider.validate_endpoint(endpoint_config)
    assert result.valid is True


def test_validate_endpoint_unsupported_type(initialized_provider):
    bad_config = {'endpoint_type': 'realtime/token-price-history', 'params': {}}
    result = initialized_provider.validate_endpoint(bad_config)
    assert result.valid is False
    assert 'Unsupported' in result.reason


def test_validate_endpoint_missing_sql(initialized_provider):
    bad_config = {'endpoint_type': 'explorer/sql', 'params': {}}
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
