"""Unit tests for the DefiLlama provider"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.providers.defillama.provider import DefiLlamaProvider
from src.providers.defillama.rate_limiter import DefiLlamaRateLimiter
from src.providers.defillama.transformer import DefiLlamaTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider_config():
    return {
        'provider_name': 'defillama',
        'enabled': True,
        'api_config': {
            'free_base_url': 'https://api.llama.fi',
            'pro_base_url': 'https://pro-api.llama.fi',
        },
        'rate_limits': {
            'requests_per_window': 10,
            'window_seconds': 10,
            'safety_margin': 1.0,
        },
    }


@pytest.fixture
def initialized_provider(provider_config):
    p = DefiLlamaProvider()
    p.initialize(provider_config)
    return p


@pytest.fixture
def transformer():
    return DefiLlamaTransformer()


@pytest.fixture
def time_range():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    return start, end


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------

def test_provider_name():
    assert DefiLlamaProvider().provider_name == "defillama"


def test_provider_initialization(provider_config):
    p = DefiLlamaProvider()
    p.initialize(provider_config)
    assert p._initialized is True
    assert p.client is not None
    assert p.rate_limiter is not None


def test_provider_no_api_key_ok(provider_config):
    """Free endpoints don't require an API key."""
    p = DefiLlamaProvider()
    p.initialize(provider_config)
    assert p.client.api_key is None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_init():
    rl = DefiLlamaRateLimiter(requests_per_window=100, window_seconds=60, safety_margin=0.9)
    assert rl.requests_per_window == 90
    assert rl.window_seconds == 60


def test_rate_limiter_acquire():
    rl = DefiLlamaRateLimiter(requests_per_window=10, window_seconds=1, safety_margin=1.0)
    rl.acquire()
    assert rl.total_requests == 1


# ---------------------------------------------------------------------------
# Transformer — chain_tvl
# ---------------------------------------------------------------------------

def test_transform_chain_tvl_filters_date(transformer, time_range):
    start, end = time_range
    raw = [
        {'date': 1609459200, 'tvl': 100_000_000},   # 2021-01-01 — out of range
        {'date': 1704067200, 'tvl': 150_000_000},   # 2024-01-01 — in range
        {'date': 1735689600, 'tvl': 200_000_000},   # 2025-01-01 — out of range
    ]
    records = transformer.transform(raw, 'chain_tvl', start_time=start, end_time=end, asset_hint='ethereum')
    assert len(records) == 1
    assert records[0]['asset'] == 'ethereum'
    assert records[0]['metric'] == 'tvl_usd'
    assert records[0]['value'] == '150000000'


def test_transform_chain_tvl_no_filter(transformer):
    raw = [
        {'date': 1609459200, 'tvl': 100_000_000},
        {'date': 1640995200, 'tvl': 200_000_000},
    ]
    records = transformer.transform(raw, 'chain_tvl', asset_hint='eth')
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Transformer — protocol_tvl
# ---------------------------------------------------------------------------

def test_transform_protocol_tvl(transformer, time_range):
    start, end = time_range
    raw = {
        'name': 'Aave',
        'symbol': 'AAVE',
        'tvl': [
            {'date': 1704067200, 'totalLiquidityUSD': 5_200_000_000},
        ],
    }
    records = transformer.transform(raw, 'protocol_tvl', start_time=start, end_time=end, asset_hint='aave')
    assert len(records) == 1
    assert records[0]['asset'] == 'aave'
    assert records[0]['metric'] == 'tvl_usd'


# ---------------------------------------------------------------------------
# Transformer — dex_volumes
# ---------------------------------------------------------------------------

def test_transform_dex_volumes(transformer, time_range):
    start, end = time_range
    raw = {
        'name': 'Uniswap V3',
        'slug': 'uniswap-v3',
        'dailyVolume': [
            {'date': 1704067200, 'volume': 1_500_000_000},
        ],
    }
    records = transformer.transform(raw, 'dex_volumes', start_time=start, end_time=end, asset_hint='uniswap_v3')
    assert len(records) == 1
    assert records[0]['metric'] == 'volume_usd'


def test_transform_dex_volumes_tuple_format(transformer, time_range):
    """Some DefiLlama responses use [timestamp, value] tuples."""
    start, end = time_range
    raw = {
        'name': 'Uniswap V3',
        'dailyVolume': [[1704067200, 1_500_000_000]],
    }
    records = transformer.transform(raw, 'dex_volumes', start_time=start, end_time=end, asset_hint='uni')
    assert len(records) == 1
    assert records[0]['value'] == '1500000000'


# ---------------------------------------------------------------------------
# Transformer — fees
# ---------------------------------------------------------------------------

def test_transform_fees(transformer, time_range):
    start, end = time_range
    raw = {
        'name': 'Hyperliquid',
        'totalDataChart': [[1704067200, 1_472_923]],
    }
    records = transformer.transform(raw, 'fees', start_time=start, end_time=end, asset_hint='hyperliquid')
    assert len(records) == 1
    assert records[0]['metric'] == 'fees_usd'
    assert records[0]['asset'] == 'hyperliquid'


# ---------------------------------------------------------------------------
# Transformer — stablecoin_flow
# ---------------------------------------------------------------------------

def test_transform_stablecoin_flow(transformer, time_range):
    start, end = time_range
    raw = [
        {'date': 1704067200, 'totalCirculating': {'peggedUSD': 45_000_000_000}},
    ]
    records = transformer.transform(
        raw, 'stablecoin_flow', start_time=start, end_time=end,
        asset_hint='stablecoins_ethereum'
    )
    assert len(records) == 1
    assert records[0]['metric'] == 'stablecoin_circulating_usd'
    assert records[0]['value'] == '45000000000'


# ---------------------------------------------------------------------------
# Transformer — coin_prices
# ---------------------------------------------------------------------------

def test_transform_coin_prices(transformer, time_range):
    start, end = time_range
    raw = {
        'coins': {
            'coingecko:bitcoin': {
                'symbol': 'BTC',
                'prices': [{'timestamp': 1704067200, 'price': 42000.0}],
                'confidence': 0.99,
            }
        }
    }
    records = transformer.transform(raw, 'coin_prices', start_time=start, end_time=end)
    assert len(records) == 1
    assert records[0]['asset'] == 'btc'
    assert records[0]['metric'] == 'price_usd'
    assert records[0]['metadata']['coin_id'] == 'coingecko:bitcoin'


# ---------------------------------------------------------------------------
# Transformer — unknown schema type
# ---------------------------------------------------------------------------

def test_transform_unknown_type(transformer):
    with pytest.raises(ValueError, match="unknown schema type"):
        transformer.transform({}, 'nonexistent')


# ---------------------------------------------------------------------------
# Provider — validate_endpoint
# ---------------------------------------------------------------------------

def test_validate_endpoint_success(initialized_provider):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_chain_tvl = Mock(return_value=[
        {'date': 1704067200, 'tvl': 100_000_000}
    ])
    result = initialized_provider.validate_endpoint({
        'endpoint_type': 'chain/tvl',
        'params': {'chain': 'Ethereum', 'asset': 'ethereum'},
    })
    assert result.valid is True


def test_validate_endpoint_unknown_type(initialized_provider):
    result = initialized_provider.validate_endpoint({
        'endpoint_type': 'bad/type',
        'params': {},
    })
    assert result.valid is False
    assert 'Unknown' in result.reason


# ---------------------------------------------------------------------------
# Provider — fetch_data_stream
# ---------------------------------------------------------------------------

def test_fetch_data_stream_chain_tvl(initialized_provider):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_chain_tvl = Mock(return_value=[
        {'date': 1704067200, 'tvl': 100_000_000},
    ])

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    endpoint_config = {
        'endpoint_type': 'chain/tvl',
        'params': {'schema_type': 'chain_tvl', 'chain': 'Ethereum', 'asset': 'ethereum'},
    }

    batches = list(initialized_provider.fetch_data_stream(endpoint_config, start, end))
    assert len(batches) == 1
    assert batches[0][0]['metric'] == 'tvl_usd'


def test_fetch_data_stream_protocol_tvl(initialized_provider):
    initialized_provider.rate_limiter.acquire = Mock()
    initialized_provider.client.get_protocol_tvl = Mock(return_value={
        'name': 'Aave',
        'tvl': [{'date': 1704067200, 'totalLiquidityUSD': 5_200_000_000}],
    })

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

    endpoint_config = {
        'endpoint_type': 'protocol/tvl',
        'params': {'schema_type': 'protocol_tvl', 'protocol': 'aave', 'asset': 'aave'},
    }

    batches = list(initialized_provider.fetch_data_stream(endpoint_config, start, end))
    assert len(batches) == 1
    assert batches[0][0]['asset'] == 'aave'


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

import requests as req_lib

def test_handle_rate_limit(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 429
    mock_resp.headers = {'Retry-After': '30'}
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is True
    assert info['wait_seconds'] == 30


def test_handle_auth_error(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 401
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is False
    assert info['fatal'] is True


def test_handle_not_found(initialized_provider):
    mock_resp = Mock()
    mock_resp.status_code = 404
    error = req_lib.HTTPError(response=mock_resp)
    info = initialized_provider.handle_error(error, {})
    assert info['retry'] is False
    assert info['error_type'] == 'not_found'
