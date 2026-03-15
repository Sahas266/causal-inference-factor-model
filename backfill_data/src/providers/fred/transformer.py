"""Transformer: FRED API observations -> standard backfill schema"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger('backfill_system.fred')


class FredTransformer:
    """
    Transform FRED series observations into standard asset_metrics records.

    Each observation becomes one record:
        asset = asset_hint (default "macro")
        metric = series_id (e.g. "DFF", "VIXCLS")
        time = observation date
        value = observation value
    """

    def transform(
        self,
        raw_data: Dict,
        schema_type: str,
        series_id: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_hint: str = "macro",
    ) -> List[Dict]:
        if schema_type != 'series_observations':
            raise ValueError(f"FRED transformer: unknown schema type '{schema_type}'")

        if not series_id:
            raise ValueError("FRED transformer: series_id is required")

        records: List[Dict] = []
        observations = raw_data.get('observations', [])

        for obs in observations:
            date_str = obs.get('date', '')
            value_str = obs.get('value', '')

            # FRED uses "." for missing values
            if not value_str or value_str == '.':
                continue

            try:
                Decimal(value_str)
            except (InvalidOperation, TypeError):
                continue

            try:
                dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if start_time and dt < start_time:
                continue
            if end_time and dt > end_time:
                continue

            records.append({
                'asset': asset_hint,
                'metric': series_id,
                'time': dt.isoformat(),
                'value': value_str,
                'frequency': '1d',
                'metadata': {},
            })

        logger.debug(f"FRED transform {series_id}: {len(records)} records from {len(observations)} observations")
        return records
