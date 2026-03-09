"""Transformer: Allium Developer API responses → standard backfill schemas"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger('backfill_system.allium')


def _to_decimal(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, TypeError):
        return None


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 or date string to UTC datetime."""
    if not ts_str:
        raise ValueError("empty timestamp")
    ts_str = ts_str.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        # Try date-only
        d = date.fromisoformat(ts_str.split('T')[0])
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _in_range(dt: datetime, start: Optional[datetime], end: Optional[datetime]) -> bool:
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


class AlliumTransformer:
    """
    Transform Allium Developer API responses to the standard backfill schemas.

    Supported schema types
    ----------------------
    ``token_price_history``
        From POST /developer/prices/history.
        Outputs one record per token per timestamp with metrics:
        ``price_usd``, ``open_usd``, ``high_usd``, ``low_usd``, ``close_usd``
        All written to the ``asset_metrics`` table.

    ``dex_trades``
        From GET /developer/{chain}/dex/trades.
        Outputs volume and trade count aggregated per day.
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
            'token_price_history': self.transform_token_price_history,
            'dex_trades': self.transform_dex_trades,
        }
        handler = dispatch.get(schema_type)
        if handler is None:
            raise ValueError(f"Allium transformer: unknown schema type '{schema_type}'")
        return handler(raw_data, start_time=start_time, end_time=end_time, asset_hint=asset_hint)

    # ------------------------------------------------------------------
    # token_price_history
    # ------------------------------------------------------------------

    def transform_token_price_history(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from POST /developer/prices/history):
        {
          "items": [
            {
              "mint": "0xc02aaa39...",
              "chain": "ethereum",
              "decimals": 18,
              "prices": [
                {"timestamp": "2024-01-01T00:00:00Z",
                 "price": 2306.62, "open": 2274.09,
                 "high": 2504.45, "close": 2351.59, "low": 2253.97}
              ]
            }
          ]
        }
        Each token+price-point → 5 records (price, open, high, low, close).
        """
        records: List[Dict] = []

        for token in raw_data.get('items', []):
            chain = token.get('chain', 'unknown')
            mint = token.get('mint', '')
            # Use hint, or fall back to "chain_0xaddress"
            asset = asset_hint or f"{chain}_{mint[:8].lower()}"

            for pt in token.get('prices', []):
                try:
                    dt = _parse_ts(pt['timestamp'])
                except Exception as e:
                    logger.warning(f"Bad timestamp {pt.get('timestamp')}: {e}")
                    continue

                if not _in_range(dt, start_time, end_time):
                    continue

                ts_iso = dt.isoformat()
                for metric, field in [
                    ('price_usd', 'price'),
                    ('open_usd', 'open'),
                    ('high_usd', 'high'),
                    ('low_usd', 'low'),
                    ('close_usd', 'close'),
                ]:
                    val = pt.get(field)
                    records.append({
                        'asset': asset,
                        'metric': metric,
                        'time': ts_iso,
                        'value': _to_decimal(val),
                        'frequency': '1d',
                        'metadata': {'chain': chain, 'token_address': mint},
                    })

        logger.debug(f"transform_token_price_history: {len(records)} records")
        return records

    # ------------------------------------------------------------------
    # dex_trades
    # ------------------------------------------------------------------

    def transform_dex_trades(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from GET /developer/{chain}/dex/trades):
        {"items": [{"block_timestamp": "...", "amount_usd": 1234, ...}]}

        Aggregates to daily (asset, metric=volume_usd / trade_count).
        """
        from collections import defaultdict
        daily: Dict[str, Dict[str, float]] = defaultdict(lambda: {'volume_usd': 0.0, 'trade_count': 0})
        chain = asset_hint or 'unknown'

        for trade in raw_data.get('items', []):
            ts_str = trade.get('block_timestamp') or trade.get('timestamp', '')
            try:
                dt = _parse_ts(ts_str)
            except Exception:
                continue
            if not _in_range(dt, start_time, end_time):
                continue

            day_key = dt.strftime('%Y-%m-%dT00:00:00+00:00')
            daily[day_key]['volume_usd'] += float(trade.get('amount_usd') or 0)
            daily[day_key]['trade_count'] += 1

        records: List[Dict] = []
        for day_iso, metrics in sorted(daily.items()):
            for metric, value in metrics.items():
                records.append({
                    'asset': chain,
                    'metric': metric,
                    'time': day_iso,
                    'value': _to_decimal(value),
                    'frequency': '1d',
                    'metadata': {},
                })

        logger.debug(f"transform_dex_trades: {len(records)} records")
        return records

    def infer_schema_type(self, raw_data) -> str:
        """Infer schema type from response shape."""
        if isinstance(raw_data, dict):
            if 'items' in raw_data:
                items = raw_data['items']
                if items and isinstance(items[0], dict):
                    if 'prices' in items[0]:
                        return 'token_price_history'
                    if 'amount_usd' in items[0] or 'block_timestamp' in items[0]:
                        return 'dex_trades'
        raise ValueError(f"Cannot infer schema type from response shape")
