"""Transformer: DefiLlama API responses → standard backfill schemas"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger('backfill_system.defillama')


def _to_decimal(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, TypeError):
        return None


def _unix_to_iso(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _in_range(ts, start: datetime, end: datetime) -> bool:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return start <= dt <= end


class DefiLlamaTransformer:
    """
    Transform DefiLlama API responses to the standard backfill schema.

    DefiLlama returns full historical data in one shot (no server-side time
    filtering). All transformers accept ``start_time`` / ``end_time`` and
    filter rows on the client side.

    Supported schema types
    ----------------------
    ``protocol_tvl``
        From GET /api/protocol/{protocol}. Outputs one record per day with
        metric = "tvl_usd".

    ``chain_tvl``
        From GET /api/v2/historicalChainTvl/{chain}. One record per day,
        metric = "tvl_usd".

    ``dex_volumes``
        From GET /api/summary/dexs/{protocol}.  Metric = "volume_usd".

    ``fees``
        From GET /api/summary/fees/{protocol}.  Metrics = "fees_usd" / "revenue_usd".

    ``stablecoin_flow``
        From GET /stablecoins/stablecoincharts/{chain}.
        Metric = "stablecoin_circulating_usd".

    ``coin_prices``
        From GET /coins/chart/{coins}.  Metric = "price_usd" per coin.
    """

    def transform(
        self,
        raw_data,
        schema_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        dispatch = {
            'protocol_tvl': self.transform_protocol_tvl,
            'chain_tvl': self.transform_chain_tvl,
            'dex_volumes': self.transform_dex_volumes,
            'fees': self.transform_fees,
            'stablecoin_flow': self.transform_stablecoin_flow,
            'coin_prices': self.transform_coin_prices,
        }
        handler = dispatch.get(schema_type)
        if handler is None:
            raise ValueError(f"DefiLlama transformer: unknown schema type '{schema_type}'")
        return handler(raw_data, start_time=start_time, end_time=end_time, asset_hint=asset_hint)

    # ------------------------------------------------------------------
    # Protocol TVL
    # ------------------------------------------------------------------

    def transform_protocol_tvl(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from /api/protocol/{protocol}):
        {
            "name": "Aave",
            "tvl": [{"date": 1640995200, "totalLiquidityUSD": 5200000000}, ...]
        }
        """
        records = []
        asset = asset_hint or raw_data.get('symbol', raw_data.get('name', 'unknown')).lower()
        tvl_series = raw_data.get('tvl', [])

        for point in tvl_series:
            ts = point.get('date')
            tvl = point.get('totalLiquidityUSD')
            if ts is None:
                continue
            if start_time and end_time and not _in_range(ts, start_time, end_time):
                continue
            records.append({
                'asset': asset,
                'metric': 'tvl_usd',
                'time': _unix_to_iso(ts),
                'value': _to_decimal(tvl),
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"transform_protocol_tvl: {len(records)} records for '{asset}'")
        return records

    # ------------------------------------------------------------------
    # Chain TVL
    # ------------------------------------------------------------------

    def transform_chain_tvl(
        self,
        raw_data: List[Dict],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from /api/v2/historicalChainTvl/{chain}):
        [{"date": 1640995200, "tvl": 150000000000}, ...]
        """
        records = []
        asset = asset_hint or 'unknown'

        for point in raw_data:
            ts = point.get('date')
            tvl = point.get('tvl')
            if ts is None:
                continue
            if start_time and end_time and not _in_range(ts, start_time, end_time):
                continue
            records.append({
                'asset': asset,
                'metric': 'tvl_usd',
                'time': _unix_to_iso(ts),
                'value': _to_decimal(tvl),
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"transform_chain_tvl: {len(records)} records")
        return records

    # ------------------------------------------------------------------
    # DEX volumes
    # ------------------------------------------------------------------

    def transform_dex_volumes(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from /api/summary/dexs/{protocol}):
        {"name": "Uniswap", "dailyVolume": [{"date": 1640995200, "volume": 1500000000}, ...]}
        """
        records = []
        asset = asset_hint or raw_data.get('slug', raw_data.get('name', 'unknown')).lower()

        # DefiLlama uses 'totalDataChart' (list of [ts, val] tuples) or 'dailyVolume' (list of dicts)
        volume_data = raw_data.get('totalDataChart') or raw_data.get('dailyVolume', [])

        for point in volume_data:
            ts = point.get('date') if isinstance(point, dict) else None
            volume = point.get('volume') if isinstance(point, dict) else None

            # totalDataChart uses [timestamp, value] tuples
            if isinstance(point, (list, tuple)) and len(point) == 2:
                ts, volume = point[0], point[1]

            if ts is None:
                continue
            if start_time and end_time and not _in_range(ts, start_time, end_time):
                continue
            records.append({
                'asset': asset,
                'metric': 'volume_usd',
                'time': _unix_to_iso(ts),
                'value': _to_decimal(volume),
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"transform_dex_volumes: {len(records)} records for '{asset}'")
        return records

    # ------------------------------------------------------------------
    # Fees & revenue
    # ------------------------------------------------------------------

    def transform_fees(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from /api/summary/fees/{protocol}):
        {
            "name": "Hyperliquid",
            "totalDataChart": [[1734912000, 1472923], ...]
        }
        """
        records = []
        asset = asset_hint or raw_data.get('slug', raw_data.get('name', 'unknown')).lower()

        for point in raw_data.get('totalDataChart', []):
            if isinstance(point, (list, tuple)) and len(point) == 2:
                ts, value = point[0], point[1]
            elif isinstance(point, dict):
                ts = point.get('date')
                value = point.get('fees') or point.get('revenue') or point.get('value')
            else:
                continue

            if ts is None:
                continue
            if start_time and end_time and not _in_range(ts, start_time, end_time):
                continue
            records.append({
                'asset': asset,
                'metric': 'fees_usd',
                'time': _unix_to_iso(ts),
                'value': _to_decimal(value),
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"transform_fees: {len(records)} records for '{asset}'")
        return records

    # ------------------------------------------------------------------
    # Stablecoin flow
    # ------------------------------------------------------------------

    def transform_stablecoin_flow(
        self,
        raw_data: List[Dict],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "stablecoins",
    ) -> List[Dict]:
        """
        Input (from /stablecoins/stablecoincharts/{chain}):
        [{"date": 1609459200, "totalCirculating": {"peggedUSD": 45000000000}}, ...]
        """
        records = []
        asset = asset_hint or 'stablecoins'

        for point in raw_data:
            ts = point.get('date')
            circulating = point.get('totalCirculating', {})
            value = circulating.get('peggedUSD') if isinstance(circulating, dict) else circulating

            if ts is None:
                continue
            if start_time and end_time and not _in_range(ts, start_time, end_time):
                continue
            records.append({
                'asset': asset,
                'metric': 'stablecoin_circulating_usd',
                'time': _unix_to_iso(ts),
                'value': _to_decimal(value),
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"transform_stablecoin_flow: {len(records)} records")
        return records

    # ------------------------------------------------------------------
    # Coin prices
    # ------------------------------------------------------------------

    def transform_coin_prices(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from /coins/chart/{coins}):
        {
            "coins": {
                "coingecko:bitcoin": {
                    "prices": [{"timestamp": 1640995200, "price": 46000.0}],
                    "symbol": "BTC"
                }
            }
        }
        """
        records = []

        for coin_id, coin_data in raw_data.get('coins', {}).items():
            symbol = coin_data.get('symbol', coin_id.split(':')[-1]).lower()
            asset = asset_hint or symbol

            for point in coin_data.get('prices', []):
                ts = point.get('timestamp')
                price = point.get('price')
                if ts is None:
                    continue
                if start_time and end_time and not _in_range(ts, start_time, end_time):
                    continue
                records.append({
                    'asset': asset,
                    'metric': 'price_usd',
                    'time': _unix_to_iso(ts),
                    'value': _to_decimal(price),
                    'frequency': '1d',
                    'metadata': {'coin_id': coin_id},
                })

        logger.debug(f"transform_coin_prices: {len(records)} records")
        return records
