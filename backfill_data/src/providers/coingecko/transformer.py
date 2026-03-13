"""Transformer: CoinGecko API responses -> standard backfill schemas"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger('backfill_system.coingecko')


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
        d = date.fromisoformat(ts_str.split('T')[0])
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _ms_to_datetime(ms: int) -> datetime:
    """Convert unix millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _in_range(dt: datetime, start: Optional[datetime], end: Optional[datetime]) -> bool:
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


class CoinGeckoTransformer:
    """
    Transform CoinGecko API responses to the standard backfill schemas.

    Supported schema types
    ----------------------
    ``market_chart``
        From GET /coins/{id}/market_chart/range.
        Outputs records for metrics: price_usd, market_cap_usd, spot_volume_usd_24h.
        Written to the ``asset_metrics`` table.

    ``coin_data``
        From GET /coins/{id}.
        Outputs snapshot records for: fdv_usd, total_supply, max_supply, ath_usd, atl_usd.
        Written to the ``asset_metrics`` table.
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
            'market_chart': self.transform_market_chart,
            'coin_data': self.transform_coin_data,
        }
        handler = dispatch.get(schema_type)
        if handler is None:
            raise ValueError(f"CoinGecko transformer: unknown schema type '{schema_type}'")
        return handler(raw_data, start_time=start_time, end_time=end_time, asset_hint=asset_hint)

    # ------------------------------------------------------------------
    # market_chart
    # ------------------------------------------------------------------

    def transform_market_chart(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from GET /coins/{id}/market_chart/range):
        {
          "prices": [[unix_ms, value], ...],
          "market_caps": [[unix_ms, value], ...],
          "total_volumes": [[unix_ms, value], ...]
        }

        Each series entry -> one record with the corresponding metric name.
        """
        records: List[Dict] = []
        asset = asset_hint or 'unknown'

        series_map = [
            ('prices', 'price_usd'),
            ('market_caps', 'market_cap_usd'),
            ('total_volumes', 'spot_volume_usd_24h'),
        ]

        for series_key, metric_name in series_map:
            for point in raw_data.get(series_key, []):
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue

                ts_ms, value = point[0], point[1]
                try:
                    dt = _ms_to_datetime(int(ts_ms))
                except (ValueError, TypeError, OSError) as e:
                    logger.warning(f"Bad timestamp {ts_ms}: {e}")
                    continue

                if not _in_range(dt, start_time, end_time):
                    continue

                records.append({
                    'asset': asset,
                    'metric': metric_name,
                    'time': dt.isoformat(),
                    'value': _to_decimal(value),
                    'frequency': '1d',
                    'metadata': {},
                })

        logger.debug(f"transform_market_chart: {len(records)} records")
        return records

    # ------------------------------------------------------------------
    # coin_data
    # ------------------------------------------------------------------

    def transform_coin_data(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from GET /coins/{id}):
        {
          "id": "ethereum",
          "market_data": {
            "fully_diluted_valuation": {"usd": 123456},
            "total_supply": 120000000,
            "max_supply": null,
            "ath": {"usd": 4878.26},
            "atl": {"usd": 0.432979}
          },
          "last_updated": "2026-03-10T12:00:00.000Z"
        }

        Extracts fdv_usd, total_supply, max_supply, ath_usd, atl_usd.
        """
        records: List[Dict] = []
        asset = asset_hint or raw_data.get('id', 'unknown')

        market_data = raw_data.get('market_data', {})
        if not market_data:
            logger.warning("No market_data in coin response")
            return records

        # Determine timestamp
        last_updated = raw_data.get('last_updated', '')
        try:
            dt = _parse_ts(last_updated)
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)
        ts_iso = dt.isoformat()

        # Extract metrics
        metric_extractions = [
            ('fdv_usd', self._nested_get(market_data, 'fully_diluted_valuation', 'usd')),
            ('total_supply', market_data.get('total_supply')),
            ('max_supply', market_data.get('max_supply')),
            ('ath_usd', self._nested_get(market_data, 'ath', 'usd')),
            ('atl_usd', self._nested_get(market_data, 'atl', 'usd')),
        ]

        for metric_name, value in metric_extractions:
            if value is None:
                continue
            records.append({
                'asset': asset,
                'metric': metric_name,
                'time': ts_iso,
                'value': _to_decimal(value),
                'frequency': 'snapshot',
                'metadata': {},
            })

        logger.debug(f"transform_coin_data: {len(records)} records")
        return records

    @staticmethod
    def _nested_get(data: Dict, key1: str, key2: str):
        """Safely get a nested dict value like data[key1][key2]."""
        inner = data.get(key1)
        if isinstance(inner, dict):
            return inner.get(key2)
        return None
