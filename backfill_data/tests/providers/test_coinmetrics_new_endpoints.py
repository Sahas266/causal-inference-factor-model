"""Tests for new CoinMetrics endpoint transformers (FM-13)

Tests for: pair_candles, market_open_interest, market_liquidations,
market_funding_rates, market_candles, market_implied_volatility, market_greeks
"""

import pytest
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.providers.coinmetrics.transformer import CoinMetricsTransformer


@pytest.fixture
def transformer():
    return CoinMetricsTransformer()


# =====================================================
# Pair Candles Tests
# =====================================================

class TestTransformPairCandles:
    def test_basic_transform(self, transformer):
        raw = [{
            'pair': 'btc-usd',
            'time': '2024-01-01T00:00:00.000000000Z',
            'price_open': '42000.00',
            'price_close': '42500.00',
            'price_high': '43000.00',
            'price_low': '41500.00',
            'vwap': '42250.00',
            'volume': '1500.5',
            'candle_usd_volume': '63375000.00'
        }]
        result = transformer.transform_pair_candles(raw)
        assert len(result) == 1
        r = result[0]
        assert r['pair'] == 'btc-usd'
        assert r['price_open'] == '42000.00'
        assert r['price_close'] == '42500.00'
        assert r['price_high'] == '43000.00'
        assert r['price_low'] == '41500.00'
        assert r['vwap'] == '42250.00'
        assert r['volume'] == '1500.5'
        assert r['candle_usd_volume'] == '63375000.00'
        assert r['frequency'] == '1d'

    def test_missing_optional_fields(self, transformer):
        raw = [{
            'pair': 'eth-usd',
            'time': '2024-01-01T00:00:00.000000000Z',
            'price_open': '2200.00',
            'price_close': '2250.00',
            'price_high': '2300.00',
            'price_low': '2150.00'
        }]
        result = transformer.transform_pair_candles(raw)
        assert len(result) == 1
        assert result[0]['volume'] is None
        assert result[0]['vwap'] is None
        assert result[0]['candle_usd_volume'] is None

    def test_empty_data(self, transformer):
        assert transformer.transform_pair_candles([]) == []

    def test_multiple_records(self, transformer):
        raw = [
            {'pair': 'btc-usd', 'time': '2024-01-01T00:00:00Z', 'price_open': '42000', 'price_close': '42500', 'price_high': '43000', 'price_low': '41500'},
            {'pair': 'eth-usd', 'time': '2024-01-01T00:00:00Z', 'price_open': '2200', 'price_close': '2250', 'price_high': '2300', 'price_low': '2150'},
        ]
        result = transformer.transform_pair_candles(raw)
        assert len(result) == 2
        assert result[0]['pair'] == 'btc-usd'
        assert result[1]['pair'] == 'eth-usd'


# =====================================================
# Market Open Interest Tests
# =====================================================

class TestTransformMarketOpenInterest:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'binance-BTCUSDT-future',
            'time': '2024-01-01T00:00:00.000000000Z',
            'contract_count': '50000',
            'value_usd': '2100000000',
            'database_time': '2024-01-01T00:05:00Z',
            'coin_metrics_id': 'abc123'
        }]
        result = transformer.transform_market_open_interest(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'binance-BTCUSDT-future'
        assert r['contract_count'] == '50000'
        assert r['value_usd'] == '2100000000'
        assert r['metadata']['coin_metrics_id'] == 'abc123'

    def test_missing_value_usd(self, transformer):
        raw = [{
            'market': 'binance-ETHUSDT-future',
            'time': '2024-01-01T00:00:00Z',
            'contract_count': '100000'
        }]
        result = transformer.transform_market_open_interest(raw)
        assert result[0]['value_usd'] is None

    def test_empty_data(self, transformer):
        assert transformer.transform_market_open_interest([]) == []


# =====================================================
# Market Liquidations Tests
# =====================================================

class TestTransformMarketLiquidations:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'binance-BTCUSDT-future',
            'time': '2024-01-01T12:00:00.123456789Z',
            'coin_metrics_id': 'liq_001',
            'amount': '0.5',
            'price': '42000.00',
            'type': 'trade',
            'side': 'sell',
            'database_time': '2024-01-01T12:00:01Z'
        }]
        result = transformer.transform_market_liquidations(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'binance-BTCUSDT-future'
        assert r['coin_metrics_id'] == 'liq_001'
        assert r['amount'] == '0.5'
        assert r['price'] == '42000.00'
        assert r['side'] == 'sell'
        assert r['type'] == 'trade'

    def test_missing_optional_fields(self, transformer):
        raw = [{
            'market': 'binance-BTCUSDT-future',
            'time': '2024-01-01T12:00:00Z',
            'coin_metrics_id': 'liq_002'
        }]
        result = transformer.transform_market_liquidations(raw)
        assert result[0]['amount'] is None
        assert result[0]['price'] is None
        assert result[0]['side'] is None

    def test_empty_data(self, transformer):
        assert transformer.transform_market_liquidations([]) == []


# =====================================================
# Market Funding Rates Tests
# =====================================================

class TestTransformMarketFundingRates:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'binance-BTCUSDT-future',
            'time': '2024-01-01T00:00:00.000000000Z',
            'rate': '0.0001',
            'period': '8h',
            'database_time': '2024-01-01T00:05:00Z',
            'coin_metrics_id': 'fr_001'
        }]
        result = transformer.transform_market_funding_rates(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'binance-BTCUSDT-future'
        assert r['rate'] == '0.0001'
        assert r['period'] == '8h'
        assert r['metadata']['coin_metrics_id'] == 'fr_001'

    def test_negative_rate(self, transformer):
        raw = [{
            'market': 'binance-ETHUSDT-future',
            'time': '2024-01-01T08:00:00Z',
            'rate': '-0.0003',
            'period': '8h'
        }]
        result = transformer.transform_market_funding_rates(raw)
        assert result[0]['rate'] == '-0.0003'

    def test_empty_data(self, transformer):
        assert transformer.transform_market_funding_rates([]) == []


# =====================================================
# Market Candles Tests
# =====================================================

class TestTransformMarketCandles:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'coinbase-btc-usd-spot',
            'time': '2024-01-01T00:00:00.000000000Z',
            'price_open': '42000.00',
            'price_close': '42500.00',
            'price_high': '43000.00',
            'price_low': '41500.00',
            'vwap': '42250.00',
            'volume': '800.25',
            'candle_usd_volume': '33810000.00'
        }]
        result = transformer.transform_market_candles(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'coinbase-btc-usd-spot'
        assert r['price_open'] == '42000.00'
        assert r['price_close'] == '42500.00'
        assert r['volume'] == '800.25'
        assert r['frequency'] == '1d'

    def test_missing_optional_fields(self, transformer):
        raw = [{
            'market': 'binance-btc-usdt-spot',
            'time': '2024-01-01T00:00:00Z',
            'price_open': '42000.00',
            'price_close': '42500.00',
            'price_high': '43000.00',
            'price_low': '41500.00'
        }]
        result = transformer.transform_market_candles(raw)
        assert result[0]['volume'] is None
        assert result[0]['vwap'] is None

    def test_empty_data(self, transformer):
        assert transformer.transform_market_candles([]) == []


# =====================================================
# Market Implied Volatility Tests
# =====================================================

class TestTransformMarketImpliedVolatility:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'deribit-BTC-PERPETUAL-future',
            'time': '2024-01-01T00:00:00.000000000Z',
            'implied_volatility': '0.65',
            'database_time': '2024-01-01T00:05:00Z',
            'coin_metrics_id': 'iv_001'
        }]
        result = transformer.transform_market_implied_volatility(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'deribit-BTC-PERPETUAL-future'
        assert r['implied_volatility'] == '0.65'
        assert r['metadata']['coin_metrics_id'] == 'iv_001'

    def test_missing_iv(self, transformer):
        raw = [{
            'market': 'deribit-BTC-PERPETUAL-future',
            'time': '2024-01-01T00:00:00Z'
        }]
        result = transformer.transform_market_implied_volatility(raw)
        assert result[0]['implied_volatility'] is None

    def test_empty_data(self, transformer):
        assert transformer.transform_market_implied_volatility([]) == []


# =====================================================
# Market Greeks Tests
# =====================================================

class TestTransformMarketGreeks:
    def test_basic_transform(self, transformer):
        raw = [{
            'market': 'deribit-ETH-25MAR22-1200-P-option',
            'time': '2024-01-01T00:00:00.000000000Z',
            'delta': '0.45',
            'gamma': '0.002',
            'vega': '150.5',
            'theta': '-25.0',
            'rho': '0.01',
            'database_time': '2024-01-01T00:05:00Z',
            'coin_metrics_id': 'gr_001'
        }]
        result = transformer.transform_market_greeks(raw)
        assert len(result) == 1
        r = result[0]
        assert r['market'] == 'deribit-ETH-25MAR22-1200-P-option'
        assert r['delta'] == '0.45'
        assert r['gamma'] == '0.002'
        assert r['vega'] == '150.5'
        assert r['theta'] == '-25.0'
        assert r['rho'] == '0.01'
        assert r['metadata']['coin_metrics_id'] == 'gr_001'

    def test_partial_greeks(self, transformer):
        raw = [{
            'market': 'deribit-BTC-28DEC22-50000-C-option',
            'time': '2024-01-01T00:00:00Z',
            'delta': '0.7',
            'gamma': '0.001'
        }]
        result = transformer.transform_market_greeks(raw)
        r = result[0]
        assert r['delta'] == '0.7'
        assert r['gamma'] == '0.001'
        assert r['vega'] is None
        assert r['theta'] is None
        assert r['rho'] is None

    def test_negative_greeks(self, transformer):
        raw = [{
            'market': 'deribit-ETH-option',
            'time': '2024-01-01T00:00:00Z',
            'delta': '-0.55',
            'theta': '-30.0',
            'gamma': '0.003',
            'vega': '200.0',
            'rho': '-0.005'
        }]
        result = transformer.transform_market_greeks(raw)
        r = result[0]
        assert r['delta'] == '-0.55'
        assert r['theta'] == '-30.0'

    def test_empty_data(self, transformer):
        assert transformer.transform_market_greeks([]) == []


# =====================================================
# Schema Type Inference Tests
# =====================================================

class TestInferSchemaTypeNew:
    def test_infer_pair_candles(self, transformer):
        data = [{'pair': 'btc-usd', 'time': '2024-01-01', 'price_open': '42000'}]
        assert transformer.infer_schema_type(data) == 'pair_candles'

    def test_infer_market_open_interest(self, transformer):
        data = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01', 'contract_count': '50000'}]
        assert transformer.infer_schema_type(data) == 'market_open_interest'

    def test_infer_market_liquidations(self, transformer):
        data = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01', 'coin_metrics_id': 'abc', 'amount': '0.5'}]
        assert transformer.infer_schema_type(data) == 'market_liquidations'

    def test_infer_market_funding_rates(self, transformer):
        data = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01', 'rate': '0.0001'}]
        assert transformer.infer_schema_type(data) == 'market_funding_rates'

    def test_infer_market_candles(self, transformer):
        data = [{'market': 'coinbase-btc-usd-spot', 'time': '2024-01-01', 'price_open': '42000'}]
        assert transformer.infer_schema_type(data) == 'market_candles'

    def test_infer_market_implied_volatility(self, transformer):
        data = [{'market': 'deribit-BTC-future', 'time': '2024-01-01', 'implied_volatility': '0.65'}]
        assert transformer.infer_schema_type(data) == 'market_implied_volatility'

    def test_infer_market_greeks(self, transformer):
        data = [{'market': 'deribit-option', 'time': '2024-01-01', 'delta': '0.45'}]
        assert transformer.infer_schema_type(data) == 'market_greeks'

    def test_existing_types_still_work(self, transformer):
        """Ensure existing schema type inference is not broken"""
        # Asset metrics
        data = [{'asset': 'btc', 'time': '2024-01-01', 'PriceUSD': '42000'}]
        assert transformer.infer_schema_type(data) == 'asset_metrics'
        
        # Exchange metrics
        data = [{'exchange': 'binance', 'time': '2024-01-01', 'volume_reported_spot_usd_1d': '1000000'}]
        assert transformer.infer_schema_type(data) == 'exchange_metrics'
        
        # Market orderbooks
        data = [{'market': 'coinbase-btc-usd-spot', 'asks': [], 'bids': []}]
        assert transformer.infer_schema_type(data) == 'market_orderbooks'
        
        # Market trades
        data = [{'market': 'coinbase-btc-usd-spot', 'coin_metrics_id': 'abc', 'price': '42000'}]
        assert transformer.infer_schema_type(data) == 'market_trades'


# =====================================================
# Dispatch Tests
# =====================================================

class TestDispatchNew:
    def test_dispatch_pair_candles(self, transformer):
        raw = [{'pair': 'btc-usd', 'time': '2024-01-01T00:00:00Z', 'price_open': '42000', 'price_close': '42500', 'price_high': '43000', 'price_low': '41500'}]
        result = transformer.transform(raw, 'pair_candles')
        assert len(result) == 1
        assert result[0]['pair'] == 'btc-usd'

    def test_dispatch_market_open_interest(self, transformer):
        raw = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01T00:00:00Z', 'contract_count': '50000'}]
        result = transformer.transform(raw, 'market_open_interest')
        assert len(result) == 1

    def test_dispatch_market_liquidations(self, transformer):
        raw = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01T00:00:00Z', 'coin_metrics_id': 'liq_001', 'amount': '0.5'}]
        result = transformer.transform(raw, 'market_liquidations')
        assert len(result) == 1

    def test_dispatch_market_funding_rates(self, transformer):
        raw = [{'market': 'binance-BTCUSDT-future', 'time': '2024-01-01T00:00:00Z', 'rate': '0.0001'}]
        result = transformer.transform(raw, 'market_funding_rates')
        assert len(result) == 1

    def test_dispatch_market_candles(self, transformer):
        raw = [{'market': 'coinbase-btc-usd-spot', 'time': '2024-01-01T00:00:00Z', 'price_open': '42000', 'price_close': '42500', 'price_high': '43000', 'price_low': '41500'}]
        result = transformer.transform(raw, 'market_candles')
        assert len(result) == 1

    def test_dispatch_market_implied_volatility(self, transformer):
        raw = [{'market': 'deribit-BTC-future', 'time': '2024-01-01T00:00:00Z', 'implied_volatility': '0.65'}]
        result = transformer.transform(raw, 'market_implied_volatility')
        assert len(result) == 1

    def test_dispatch_market_greeks(self, transformer):
        raw = [{'market': 'deribit-option', 'time': '2024-01-01T00:00:00Z', 'delta': '0.45'}]
        result = transformer.transform(raw, 'market_greeks')
        assert len(result) == 1

    def test_dispatch_unknown_raises(self, transformer):
        with pytest.raises(ValueError, match="Unknown schema type"):
            transformer.transform([], 'nonexistent')


# =====================================================
# Edge Case Tests
# =====================================================

class TestEdgeCases:
    def test_invalid_decimal_skipped(self, transformer):
        """Records with unparseable values should be skipped gracefully"""
        raw = [{
            'pair': 'btc-usd',
            'time': '2024-01-01T00:00:00Z',
            'price_open': 'not_a_number',
            'price_close': '42500',
            'price_high': '43000',
            'price_low': '41500'
        }]
        result = transformer.transform_pair_candles(raw)
        # Should still process since Decimal('not_a_number') raises and warns
        assert len(result) == 0  # Whole record skipped due to exception

    def test_invalid_time_skipped(self, transformer):
        """Records with invalid timestamps should be skipped"""
        raw = [{
            'market': 'binance-BTCUSDT-future',
            'time': 'invalid-time',
            'contract_count': '50000'
        }]
        result = transformer.transform_market_open_interest(raw)
        assert len(result) == 0
