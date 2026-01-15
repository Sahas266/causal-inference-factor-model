"""Tests for CoinMetrics orderbook transformation"""

import pytest
from datetime import datetime
from decimal import Decimal
from src.providers.coinmetrics.transformer import CoinMetricsTransformer


@pytest.fixture
def transformer():
    """Transformer fixture"""
    return CoinMetricsTransformer()


@pytest.fixture
def sample_orderbook_data():
    """Sample orderbook data from CoinMetrics API"""
    return [
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2024-01-01T12:00:00.000000000Z",
            "coin_metrics_id": "snapshot_123",
            "asks": [
                {"price": "42100.00", "size": "1.5"},
                {"price": "42101.00", "size": "2.0"},
                {"price": "42102.00", "size": "0.8"}
            ],
            "bids": [
                {"price": "42099.00", "size": "1.2"},
                {"price": "42098.00", "size": "0.9"},
                {"price": "42097.00", "size": "1.1"}
            ]
        }
    ]


def test_transform_market_orderbooks_basic(transformer, sample_orderbook_data):
    """Test basic orderbook transformation"""
    result = transformer.transform_market_orderbooks(sample_orderbook_data)
    
    # Should have 6 records total (3 bids + 3 asks)
    assert len(result) == 6
    
    # Check that we have 3 bids and 3 asks
    bids = [r for r in result if r['side'] == 'bid']
    asks = [r for r in result if r['side'] == 'ask']
    
    assert len(bids) == 3
    assert len(asks) == 3


def test_transform_orderbooks_bid_levels(transformer, sample_orderbook_data):
    """Test that bid levels are assigned correctly"""
    result = transformer.transform_market_orderbooks(sample_orderbook_data)
    
    bids = [r for r in result if r['side'] == 'bid']
    bids_sorted = sorted(bids, key=lambda x: x['level'])
    
    # Level 0 should be best bid (highest price)
    assert bids_sorted[0]['level'] == 0
    assert bids_sorted[0]['price'] == "42099.00"
    assert bids_sorted[0]['size'] == "1.2"
    
    # Level 1
    assert bids_sorted[1]['level'] == 1
    assert bids_sorted[1]['price'] == "42098.00"
    
    # Level 2
    assert bids_sorted[2]['level'] == 2
    assert bids_sorted[2]['price'] == "42097.00"


def test_transform_orderbooks_ask_levels(transformer, sample_orderbook_data):
    """Test that ask levels are assigned correctly"""
    result = transformer.transform_market_orderbooks(sample_orderbook_data)
    
    asks = [r for r in result if r['side'] == 'ask']
    asks_sorted = sorted(asks, key=lambda x: x['level'])
    
    # Level 0 should be best ask (lowest price)
    assert asks_sorted[0]['level'] == 0
    assert asks_sorted[0]['price'] == "42100.00"
    assert asks_sorted[0]['size'] == "1.5"
    
    # Level 1
    assert asks_sorted[1]['level'] == 1
    assert asks_sorted[1]['price'] == "42101.00"
    
    # Level 2
    assert asks_sorted[2]['level'] == 2
    assert asks_sorted[2]['price'] == "42102.00"


def test_transform_orderbooks_metadata(transformer, sample_orderbook_data):
    """Test that metadata is preserved"""
    result = transformer.transform_market_orderbooks(sample_orderbook_data)
    
    # All records should have the same market, time, and snapshot_id
    for record in result:
        assert record['market'] == "coinbase-btc-usd-spot"
        assert record['snapshot_id'] == "snapshot_123"
        assert '2024-01-01T12:00:00' in record['time']


def test_transform_orderbooks_empty_data(transformer):
    """Test transformation with empty data"""
    result = transformer.transform_market_orderbooks([])
    assert len(result) == 0


def test_transform_orderbooks_missing_snapshot_id(transformer):
    """Test transformation when snapshot_id is missing"""
    data = [
        {
            "market": "binance-eth-usdt-spot",
            "time": "2024-01-01T12:00:00.000000000Z",
            "asks": [{"price": "2500.00", "size": "10.0"}],
            "bids": [{"price": "2499.00", "size": "12.0"}]
        }
    ]
    
    result = transformer.transform_market_orderbooks(data)
    
    # Should generate snapshot_id from market and time
    assert len(result) == 2
    assert result[0]['snapshot_id']  # Should not be empty
    assert 'binance-eth-usdt-spot' in result[0]['snapshot_id']


def test_transform_orderbooks_price_decimal_conversion(transformer, sample_orderbook_data):
    """Test that prices are correctly converted to Decimal strings"""
    result = transformer.transform_market_orderbooks(sample_orderbook_data)
    
    for record in result:
        # Should be string representation
        assert isinstance(record['price'], str)
        assert isinstance(record['size'], str)
        
        # Should be valid decimal
        _ = Decimal(record['price'])
        _ = Decimal(record['size'])


def test_infer_schema_type_orderbooks(transformer):
    """Test that orderbooks are correctly identified"""
    data = [
        {
            "market": "test-market",
            "time": "2024-01-01T00:00:00Z",
            "asks": [],
            "bids": []
        }
    ]
    
    schema_type = transformer.infer_schema_type(data)
    assert schema_type == 'market_orderbooks'


def test_transform_orderbooks_multiple_snapshots(transformer):
    """Test transformation with multiple snapshots"""
    data = [
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2024-01-01T12:00:00.000000000Z",
            "coin_metrics_id": "snapshot_1",
            "asks": [{"price": "42100.00", "size": "1.0"}],
            "bids": [{"price": "42099.00", "size": "1.0"}]
        },
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2024-01-01T12:01:00.000000000Z",
            "coin_metrics_id": "snapshot_2",
            "asks": [{"price": "42110.00", "size": "2.0"}],
            "bids": [{"price": "42109.00", "size": "2.0"}]
        }
    ]
    
    result = transformer.transform_market_orderbooks(data)
    
    # Should have 4 records (2 snapshots * 2 levels each)
    assert len(result) == 4
    
    # Check that we have 2 different snapshot IDs
    snapshot_ids = set(r['snapshot_id'] for r in result)
    assert len(snapshot_ids) == 2
    assert 'snapshot_1' in snapshot_ids
    assert 'snapshot_2' in snapshot_ids


def test_transform_orderbooks_invalid_price(transformer):
    """Test handling of invalid price values"""
    data = [
        {
            "market": "test-market",
            "time": "2024-01-01T12:00:00.000000000Z",
            "coin_metrics_id": "snapshot_1",
            "asks": [
                {"price": "invalid", "size": "1.0"},
                {"price": "42100.00", "size": "1.0"}
            ],
            "bids": [{"price": "42099.00", "size": "1.0"}]
        }
    ]
    
    result = transformer.transform_market_orderbooks(data)
    
    # Should skip the invalid record but process valid ones
    # Should have 2 records (1 valid ask + 1 bid)
    assert len(result) >= 2


def test_transform_method_dispatches_orderbooks(transformer, sample_orderbook_data):
    """Test that the main transform method dispatches to orderbooks correctly"""
    result = transformer.transform(sample_orderbook_data, 'market_orderbooks')
    
    # Should return orderbook transformation
    assert len(result) == 6
    assert all(r['side'] in ['bid', 'ask'] for r in result)

