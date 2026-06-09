"""Load CPCM data from Supabase asset_metrics_best view into pandas DataFrames.

Thin backend over BaseCPCMDataLoader: this class only knows how to fetch
long-format rows from the Supabase REST API; pivoting, returns, and caching
live in the shared base.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from causal_portfolio.data.base_loader import BaseCPCMDataLoader

logger = logging.getLogger("cpcm.data")

PAGE_SIZE = 1000


class CPCMDataLoader(BaseCPCMDataLoader):
    """Loads asset metrics from Supabase and pivots into wide-format DataFrames."""

    cache_tag = "supabase"

    def __init__(self, env_path: Optional[str] = None):
        if env_path:
            load_dotenv(env_path)
        else:
            root = Path(__file__).resolve().parents[2]
            load_dotenv(root / ".env")
            load_dotenv(root / "backfill_data" / ".env", override=True)

        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_KEY", "").strip()
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set (non-empty). "
                "Add them to the process environment, or to `.env` at the repo root or "
                "`backfill_data/.env`. For `docker compose`, list those files under "
                "`env_file` for the dashboard service (do not override them with empty "
                "`environment` entries)."
            )
        self._client: Client = create_client(url, key)

    # ── backend hooks ───────────────────────────────────────────────

    def _fetch_panel_long(self, assets, metrics, start, end) -> pd.DataFrame:
        rows = self._fetch_all(
            table="asset_metrics_best",
            select="asset,metric,time,value",
            filters={"gte": {"time": start}, "lte": {"time": end}},
            in_filters={"asset": assets, "metric": metrics},
        )
        return pd.DataFrame(rows)

    def _fetch_macro_long(self, series_ids, start, end) -> pd.DataFrame:
        rows = self._fetch_all(
            table="asset_metrics_best",
            select="metric,time,value",
            filters={
                "eq": {"asset": "macro"},
                "gte": {"time": start},
                "lte": {"time": end},
            },
            in_filters={"metric": series_ids},
        )
        return pd.DataFrame(rows)

    # ── internal ────────────────────────────────────────────────────

    def _fetch_all(
        self,
        table: str,
        select: str,
        filters: Optional[dict] = None,
        in_filters: Optional[dict] = None,
    ) -> list[dict]:
        """Paginated fetch from Supabase (max 1000 rows per request)."""
        all_rows: list[dict] = []
        offset = 0

        while True:
            query = self._client.table(table).select(select)

            if filters:
                for op, criteria in filters.items():
                    if isinstance(criteria, dict):
                        for col, val in criteria.items():
                            query = getattr(query, op)(col, val)
                    else:
                        query = getattr(query, op)(*criteria)

            if in_filters:
                for col, values in in_filters.items():
                    query = query.in_(col, values)

            query = query.order("time").range(offset, offset + PAGE_SIZE - 1)
            result = query.execute()
            rows = result.data or []
            all_rows.extend(rows)

            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

        logger.info(f"Fetched {len(all_rows)} rows from {table}")
        return all_rows
