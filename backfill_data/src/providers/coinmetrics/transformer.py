"""Data transformer for CoinMetrics to standard schema"""

from typing import List, Dict
from datetime import datetime
from decimal import Decimal
import logging

logger = logging.getLogger('backfill_system.coinmetrics')


class CoinMetricsTransformer:
    """
    Transform CoinMetrics API responses to standard schemas.
    Handles asset metrics, market trades, and exchange metrics.
    """
    
    def transform(self, raw_data: List[Dict], schema_type: str) -> List[Dict]:
        """
        Transform raw data to standard schema based on type.
        
        Args:
            raw_data: Raw data from CoinMetrics API
            schema_type: Type of schema ('asset_metrics', 'market_trades', 'exchange_metrics', 'market_orderbooks')
            
        Returns:
            List of records in standard format
            
        Raises:
            ValueError: If schema_type is unknown
        """
        dispatch = {
            'asset_metrics': self.transform_asset_metrics,
            'market_trades': self.transform_market_trades,
            'exchange_metrics': self.transform_exchange_metrics,
            'market_orderbooks': self.transform_market_orderbooks,
            'pair_candles': self.transform_pair_candles,
            'market_open_interest': self.transform_market_open_interest,
            'market_liquidations': self.transform_market_liquidations,
            'market_funding_rates': self.transform_market_funding_rates,
            'market_candles': self.transform_market_candles,
            'market_implied_volatility': self.transform_market_implied_volatility,
            'market_greeks': self.transform_market_greeks,
        }
        
        handler = dispatch.get(schema_type)
        if handler is None:
            raise ValueError(f"Unknown schema type: {schema_type}")
        return handler(raw_data)
    
    def transform_asset_metrics(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics asset metrics to standard format.
        
        CoinMetrics format:
        {
            "asset": "btc",
            "time": "2024-01-01T00:00:00.000000000Z",
            "PriceUSD": "42000.50",
            "CapMrktCurUSD": "800000000000"
        }
        
        Standard format:
        {
            "asset": "btc",
            "metric": "PriceUSD",
            "time": "2024-01-01T00:00:00Z",
            "value": "42000.50",
            "frequency": "1d"
        }
        
        Args:
            raw_data: Raw CoinMetrics data
            
        Returns:
            List of standardized records
        """
        standard_records = []
        
        for record in raw_data:
            asset = record.get('asset', '').lower()
            time_str = record.get('time', '')
            
            # Parse timestamp
            try:
                time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Failed to parse time '{time_str}': {e}")
                continue
            
            # Extract all metric values (skip 'asset' and 'time' fields)
            for key, value in record.items():
                if key in ['asset', 'time']:
                    continue

                # Skip CoinMetrics status metadata fields (e.g. FlowInExNtv-status)
                if '-status' in key:
                    continue
                
                # Convert value to Decimal if not None
                decimal_value = None
                if value is not None:
                    try:
                        decimal_value = Decimal(str(value))
                    except Exception as e:
                        logger.warning(f"Failed to convert value '{value}' for metric '{key}': {e}")
                
                standard_record = {
                    'asset': asset,
                    'metric': key,
                    'time': time.isoformat(),
                    'value': str(decimal_value) if decimal_value is not None else None,
                    'frequency': '1d',  # Default, should be set from config
                    'metadata': {}
                }
                
                standard_records.append(standard_record)
        
        logger.debug(f"Transformed {len(raw_data)} raw records to {len(standard_records)} asset metrics")
        return standard_records
    
    def transform_market_trades(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market trades to standard format.
        
        CoinMetrics format:
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2024-01-01T00:00:00.123456789Z",
            "coin_metrics_id": "123456789",
            "amount": "0.5",
            "price": "42000.50",
            "database_time": "...",
            "side": "buy"
        }
        
        Args:
            raw_data: Raw CoinMetrics data
            
        Returns:
            List of standardized records
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'price': str(Decimal(str(record['price']))),
                    'amount': str(Decimal(str(record['amount']))),
                    'side': record.get('side', 'unknown'),
                    'trade_id': record.get('coin_metrics_id', ''),
                    'metadata': {
                        'database_time': record.get('database_time')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform trade record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} market trades")
        return standard_records
    
    def transform_exchange_metrics(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics exchange metrics to standard format.
        
        CoinMetrics format:
        {
            "exchange": "binance",
            "time": "2024-01-01T00:00:00.000000000Z",
            "flow_in_btc": "1000.5",
            "flow_out_btc": "950.2"
        }
        
        Args:
            raw_data: Raw CoinMetrics data
            
        Returns:
            List of standardized records
        """
        standard_records = []
        
        for record in raw_data:
            exchange = record.get('exchange', '').lower()
            time_str = record.get('time', '')
            
            # Parse timestamp
            try:
                time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Failed to parse time '{time_str}': {e}")
                continue
            
            # Extract all metric values
            for key, value in record.items():
                if key in ['exchange', 'time']:
                    continue
                
                # Convert value to Decimal if not None
                decimal_value = None
                if value is not None:
                    try:
                        decimal_value = Decimal(str(value))
                    except Exception as e:
                        logger.warning(f"Failed to convert value '{value}' for metric '{key}': {e}")
                
                standard_record = {
                    'exchange': exchange,
                    'metric': key,
                    'time': time.isoformat(),
                    'value': str(decimal_value) if decimal_value is not None else None,
                    'frequency': '1d',
                    'metadata': {}
                }
                
                standard_records.append(standard_record)
        
        logger.debug(f"Transformed {len(raw_data)} raw records to {len(standard_records)} exchange metrics")
        return standard_records
    
    def transform_market_orderbooks(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market orderbooks to standard format.
        
        CoinMetrics format:
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2020-06-08T21:00:00.000000000Z",
            "coin_metrics_id": "123456",
            "asks": [
                {"price": "9500.00", "size": "1.5"},
                {"price": "9501.00", "size": "2.0"}
            ],
            "bids": [
                {"price": "9499.00", "size": "1.2"},
                {"price": "9498.00", "size": "0.8"}
            ]
        }
        
        Standard format (flattened):
        One snapshot with 2 bids and 2 asks becomes 4 records:
        [
            {"market": "...", "side": "bid", "level": 0, "price": "9499.00", "size": "1.2"},
            {"market": "...", "side": "bid", "level": 1, "price": "9498.00", "size": "0.8"},
            {"market": "...", "side": "ask", "level": 0, "price": "9500.00", "size": "1.5"},
            {"market": "...", "side": "ask", "level": 1, "price": "9501.00", "size": "2.0"}
        ]
        
        Args:
            raw_data: Raw CoinMetrics orderbook data
            
        Returns:
            List of flattened orderbook level records
        """
        standard_records = []
        
        for snapshot in raw_data:
            try:
                market = snapshot.get('market', '')
                time_str = snapshot.get('time', '')
                snapshot_id = snapshot.get('coin_metrics_id', '')
                
                # Parse timestamp
                try:
                    time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except Exception as e:
                    logger.warning(f"Failed to parse time '{time_str}': {e}")
                    continue
                
                # Process bids (level 0 = best bid = highest price)
                bids = snapshot.get('bids', [])
                for level, bid in enumerate(bids):
                    try:
                        standard_record = {
                            'market': market,
                            'time': time.isoformat(),
                            'snapshot_id': snapshot_id or f"{market}_{time.isoformat()}",
                            'side': 'bid',
                            'price': str(Decimal(str(bid['price']))),
                            'size': str(Decimal(str(bid['size']))),
                            'level': level,
                            'metadata': {}
                        }
                        standard_records.append(standard_record)
                    except Exception as e:
                        logger.warning(f"Failed to process bid level {level}: {e}")
                        continue
                
                # Process asks (level 0 = best ask = lowest price)
                asks = snapshot.get('asks', [])
                for level, ask in enumerate(asks):
                    try:
                        standard_record = {
                            'market': market,
                            'time': time.isoformat(),
                            'snapshot_id': snapshot_id or f"{market}_{time.isoformat()}",
                            'side': 'ask',
                            'price': str(Decimal(str(ask['price']))),
                            'size': str(Decimal(str(ask['size']))),
                            'level': level,
                            'metadata': {}
                        }
                        standard_records.append(standard_record)
                    except Exception as e:
                        logger.warning(f"Failed to process ask level {level}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to transform orderbook snapshot: {e}")
                continue
        
        logger.debug(
            f"Transformed {len(raw_data)} orderbook snapshots to "
            f"{len(standard_records)} level records"
        )
        return standard_records
    
    def transform_pair_candles(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics pair candles to standard format.
        
        CoinMetrics format:
        {
            "pair": "btc-usd",
            "time": "2024-01-01T00:00:00.000000000Z",
            "price_open": "42000.00",
            "price_close": "42500.00",
            "price_high": "43000.00",
            "price_low": "41500.00",
            "vwap": "42250.00",
            "volume": "1500.5",
            "candle_usd_volume": "63375000.00"
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'pair': record.get('pair', ''),
                    'time': time.isoformat(),
                    'price_open': str(Decimal(str(record['price_open']))) if record.get('price_open') else None,
                    'price_high': str(Decimal(str(record['price_high']))) if record.get('price_high') else None,
                    'price_low': str(Decimal(str(record['price_low']))) if record.get('price_low') else None,
                    'price_close': str(Decimal(str(record['price_close']))) if record.get('price_close') else None,
                    'volume': str(Decimal(str(record['volume']))) if record.get('volume') else None,
                    'vwap': str(Decimal(str(record['vwap']))) if record.get('vwap') else None,
                    'candle_usd_volume': str(Decimal(str(record['candle_usd_volume']))) if record.get('candle_usd_volume') else None,
                    'frequency': '1d',
                    'metadata': {
                        'database_time': record.get('database_time')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform pair candle record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} pair candles")
        return standard_records
    
    def transform_market_open_interest(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market open interest to standard format.
        
        CoinMetrics format:
        {
            "market": "binance-BTCUSDT-future",
            "time": "2024-01-01T00:00:00.000000000Z",
            "contract_count": "50000",
            "value_usd": "2100000000",
            "database_time": "...",
            "coin_metrics_id": "..."
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'contract_count': str(Decimal(str(record['contract_count']))) if record.get('contract_count') else None,
                    'value_usd': str(Decimal(str(record['value_usd']))) if record.get('value_usd') else None,
                    'metadata': {
                        'database_time': record.get('database_time'),
                        'coin_metrics_id': record.get('coin_metrics_id')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform open interest record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} open interest records")
        return standard_records
    
    def transform_market_liquidations(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market liquidations to standard format.
        
        CoinMetrics format:
        {
            "market": "binance-BTCUSDT-future",
            "time": "2024-01-01T00:00:00.123456789Z",
            "coin_metrics_id": "123456",
            "amount": "0.5",
            "price": "42000.00",
            "type": "trade",
            "side": "sell",
            "database_time": "..."
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'coin_metrics_id': record.get('coin_metrics_id', ''),
                    'amount': str(Decimal(str(record['amount']))) if record.get('amount') else None,
                    'price': str(Decimal(str(record['price']))) if record.get('price') else None,
                    'side': record.get('side'),
                    'type': record.get('type'),
                    'metadata': {
                        'database_time': record.get('database_time')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform liquidation record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} liquidation records")
        return standard_records
    
    def transform_market_funding_rates(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market funding rates to standard format.
        
        CoinMetrics format:
        {
            "market": "binance-BTCUSDT-future",
            "time": "2024-01-01T00:00:00.000000000Z",
            "rate": "0.0001",
            "period": "8h",
            "database_time": "...",
            "coin_metrics_id": "..."
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'rate': str(Decimal(str(record['rate']))) if record.get('rate') else None,
                    'period': record.get('period'),
                    'metadata': {
                        'database_time': record.get('database_time'),
                        'coin_metrics_id': record.get('coin_metrics_id')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform funding rate record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} funding rate records")
        return standard_records
    
    def transform_market_candles(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market candles (exchange-specific OHLCV) to standard format.
        
        CoinMetrics format:
        {
            "market": "coinbase-btc-usd-spot",
            "time": "2024-01-01T00:00:00.000000000Z",
            "price_open": "42000.00",
            "price_close": "42500.00",
            "price_high": "43000.00",
            "price_low": "41500.00",
            "vwap": "42250.00",
            "volume": "1500.5",
            "candle_usd_volume": "63375000.00"
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'price_open': str(Decimal(str(record['price_open']))) if record.get('price_open') else None,
                    'price_high': str(Decimal(str(record['price_high']))) if record.get('price_high') else None,
                    'price_low': str(Decimal(str(record['price_low']))) if record.get('price_low') else None,
                    'price_close': str(Decimal(str(record['price_close']))) if record.get('price_close') else None,
                    'volume': str(Decimal(str(record['volume']))) if record.get('volume') else None,
                    'vwap': str(Decimal(str(record['vwap']))) if record.get('vwap') else None,
                    'candle_usd_volume': str(Decimal(str(record['candle_usd_volume']))) if record.get('candle_usd_volume') else None,
                    'frequency': '1d',
                    'metadata': {
                        'database_time': record.get('database_time')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform market candle record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} market candles")
        return standard_records
    
    def transform_market_implied_volatility(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market implied volatility to standard format.
        
        CoinMetrics format:
        {
            "market": "deribit-BTC-PERPETUAL-future",
            "time": "2024-01-01T00:00:00.000000000Z",
            "implied_volatility": "0.65",
            "database_time": "...",
            "coin_metrics_id": "..."
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'implied_volatility': str(Decimal(str(record['implied_volatility']))) if record.get('implied_volatility') else None,
                    'metadata': {
                        'database_time': record.get('database_time'),
                        'coin_metrics_id': record.get('coin_metrics_id')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform implied volatility record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} implied volatility records")
        return standard_records
    
    def transform_market_greeks(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform CoinMetrics market greeks to standard format.
        
        CoinMetrics format:
        {
            "market": "deribit-ETH-25MAR22-1200-P-option",
            "time": "2024-01-01T00:00:00.000000000Z",
            "delta": "0.45",
            "gamma": "0.002",
            "vega": "150.5",
            "theta": "-25.0",
            "rho": "0.01",
            "database_time": "...",
            "coin_metrics_id": "..."
        }
        """
        standard_records = []
        
        for record in raw_data:
            try:
                time = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))
                
                standard_record = {
                    'market': record.get('market', ''),
                    'time': time.isoformat(),
                    'delta': str(Decimal(str(record['delta']))) if record.get('delta') else None,
                    'gamma': str(Decimal(str(record['gamma']))) if record.get('gamma') else None,
                    'vega': str(Decimal(str(record['vega']))) if record.get('vega') else None,
                    'theta': str(Decimal(str(record['theta']))) if record.get('theta') else None,
                    'rho': str(Decimal(str(record['rho']))) if record.get('rho') else None,
                    'metadata': {
                        'database_time': record.get('database_time'),
                        'coin_metrics_id': record.get('coin_metrics_id')
                    }
                }
                
                standard_records.append(standard_record)
                
            except Exception as e:
                logger.warning(f"Failed to transform greeks record: {e}")
                continue
        
        logger.debug(f"Transformed {len(standard_records)} greeks records")
        return standard_records
    
    def infer_schema_type(self, raw_data: List[Dict]) -> str:
        """
        Infer schema type from raw data structure.
        
        Args:
            raw_data: Raw data records
            
        Returns:
            Inferred schema type
        """
        if not raw_data:
            raise ValueError("Cannot infer schema type from empty data")
        
        sample = raw_data[0]
        
        # Check for orderbooks (has both asks and bids arrays)
        if 'asks' in sample and 'bids' in sample:
            return 'market_orderbooks'
        # Check for pair candles (has 'pair' field)
        elif 'pair' in sample:
            return 'pair_candles'
        # Check for asset metrics
        elif 'asset' in sample:
            return 'asset_metrics'
        # Check for exchange metrics
        elif 'exchange' in sample:
            return 'exchange_metrics'
        # Check for greeks (has 'delta' field)
        elif 'market' in sample and 'delta' in sample:
            return 'market_greeks'
        # Check for implied volatility
        elif 'market' in sample and 'implied_volatility' in sample:
            return 'market_implied_volatility'
        # Check for funding rates (has 'rate' field)
        elif 'market' in sample and 'rate' in sample:
            return 'market_funding_rates'
        # Check for open interest (has 'contract_count' field)
        elif 'market' in sample and 'contract_count' in sample:
            return 'market_open_interest'
        # Check for liquidations (has 'amount' and 'coin_metrics_id')
        elif 'market' in sample and 'coin_metrics_id' in sample and 'amount' in sample:
            return 'market_liquidations'
        # Check for market candles (has 'price_open')
        elif 'market' in sample and 'price_open' in sample:
            return 'market_candles'
        # Check for market trades (has coin_metrics_id but not asks/bids)
        elif 'market' in sample and 'coin_metrics_id' in sample:
            return 'market_trades'
        else:
            raise ValueError(f"Cannot infer schema type from data: {list(sample.keys())}")

