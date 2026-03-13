"""Transformer: Dune Analytics query results → standard backfill schemas"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger('backfill_system.dune')


def _to_decimal(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, TypeError):
        return None


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601, date string, or common Dune timestamp formats to UTC datetime."""
    if not ts_str:
        raise ValueError("empty timestamp")
    ts_str = str(ts_str).strip()
    # Handle 'Z' suffix
    ts_str = ts_str.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    # Try date-only (YYYY-MM-DD)
    try:
        d = date.fromisoformat(ts_str.split('T')[0])
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    except ValueError:
        pass
    # Try common Dune formats: "2024-01-01 00:00:00 UTC", "2024-01-01 00:00:00.000 UTC"
    # Strip trailing " UTC" (not rstrip which eats too many chars)
    cleaned = ts_str
    if cleaned.endswith(' UTC'):
        cleaned = cleaned[:-4]
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def _in_range(dt: datetime, start: Optional[datetime], end: Optional[datetime]) -> bool:
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


# Columns to skip when expanding rows into metric records
_SKIP_COLUMNS = {'time', 'date', 'day', 'dt', 'timestamp', 'asset', 'symbol', 'token'}


class DuneTransformer:
    """
    Transform Dune Analytics query result rows into standard backfill schema records.

    Schema type: ``query_results``
        Input: Dune API response with ``result.rows`` — each row has a time column
        and one or more metric columns. For each row, for each metric column,
        one record is created in the asset_metrics format.
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
            'query_results': self.transform_query_results,
        }
        handler = dispatch.get(schema_type)
        if handler is None:
            raise ValueError(f"Dune transformer: unknown schema type '{schema_type}'")
        return handler(raw_data, start_time=start_time, end_time=end_time, asset_hint=asset_hint)

    def transform_query_results(
        self,
        raw_data: Dict,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "",
    ) -> List[Dict]:
        """
        Input (from Dune API):
        {
          "result": {
            "rows": [
              {"time": "2024-01-01", "eth_staked": 28000000, "validator_count": 800000, ...},
              ...
            ],
            "metadata": {"column_names": ["time", "eth_staked", "validator_count"], ...}
          },
          "execution_id": "...",
          "state": "QUERY_STATE_COMPLETED"
        }

        For each row, for each metric column (skipping time/asset columns),
        create one record in the standard asset_metrics format.
        """
        records: List[Dict] = []

        result = raw_data.get('result', {})
        rows = result.get('rows', [])

        if not rows:
            logger.debug("transform_query_results: no rows in response")
            return records

        # Detect the time column name from first row
        time_col = self._detect_time_column(rows[0])
        if not time_col:
            logger.warning("No time column detected in Dune query results")
            return records

        for row in rows:
            ts_raw = row.get(time_col, '')
            try:
                dt = _parse_ts(str(ts_raw))
            except Exception as e:
                logger.warning(f"Bad timestamp {ts_raw}: {e}")
                continue

            if not _in_range(dt, start_time, end_time):
                continue

            ts_iso = dt.isoformat()
            asset = asset_hint or row.get('asset', row.get('symbol', 'eth'))

            for col, value in row.items():
                if col.lower() in _SKIP_COLUMNS:
                    continue
                if value is None:
                    continue

                decimal_val = _to_decimal(value)
                if decimal_val is None:
                    continue

                records.append({
                    'asset': asset,
                    'metric': col,
                    'time': ts_iso,
                    'value': decimal_val,
                    'frequency': '1d',
                    'metadata': {},
                })

        logger.debug(f"transform_query_results: {len(records)} records from {len(rows)} rows")
        return records

    def _detect_time_column(self, row: Dict) -> Optional[str]:
        """Detect which column in the row holds the timestamp."""
        for candidate in ('time', 'date', 'day', 'dt', 'timestamp'):
            if candidate in row:
                return candidate
        # Fall back to first column that looks like a date string
        for key, value in row.items():
            if isinstance(value, str) and len(value) >= 10:
                try:
                    _parse_ts(value)
                    return key
                except (ValueError, TypeError):
                    continue
        return None
