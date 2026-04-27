"""Load CPCM data from Supabase asset_metrics_best view into pandas DataFrames."""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

logger = logging.getLogger("cpcm.data")

CACHE_DIR = Path(__file__).parent / "cache"
PAGE_SIZE = 1000


class CPCMDataLoader:
    """Loads asset metrics from Supabase and pivots into wide-format DataFrames."""

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
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────────

    def load_panel(
        self,
        assets: list[str],
        metrics: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Query asset_metrics_best and pivot to wide format.

        Returns DataFrame with DatetimeIndex and columns ``{asset}_{metric}``.
        """
        cache_key = self._cache_key("panel", assets, metrics, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        rows = self._fetch_all(
            table="asset_metrics_best",
            select="asset,metric,time,value",
            filters={"gte": {"time": start}, "lte": {"time": end}},
            in_filters={"asset": assets, "metric": metrics},
        )
        df = self._pivot(rows)
        if use_cache and not df.empty:
            self._write_cache(cache_key, df)
        return df

    def load_returns(
        self,
        assets: list[str],
        start: str,
        end: str,
        price_metric: str = "PriceUSD",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load prices and compute log returns."""
        cache_key = self._cache_key("returns", assets, [price_metric], start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        # Try PriceUSD first, fall back to price
        panel = self.load_panel(assets, [price_metric], start, end, use_cache=False)
        if panel.empty and price_metric == "PriceUSD":
            panel = self.load_panel(assets, ["price"], start, end, use_cache=False)

        if panel.empty:
            return pd.DataFrame()

        returns = np.log(panel / panel.shift(1)).dropna(how="all")
        returns.columns = [c.rsplit("_", 1)[0] + "_return" for c in returns.columns]
        if use_cache and not returns.empty:
            self._write_cache(cache_key, returns)
        return returns

    def load_macro(
        self,
        series_ids: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load FRED macro data (asset='macro')."""
        cache_key = self._cache_key("macro", ["macro"], series_ids, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

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
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.pivot_table(index="time", columns="metric", values="value")
        df = df.sort_index().resample("D").last().ffill()
        # Lowercase column names for consistency
        df.columns = [c.lower() for c in df.columns]

        if use_cache and not df.empty:
            self._write_cache(cache_key, df)
        return df

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

    def _pivot(self, rows: list[dict]) -> pd.DataFrame:
        """Convert long-format rows to wide DataFrame."""
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["col"] = df["asset"] + "_" + df["metric"]
        df = df.pivot_table(index="time", columns="col", values="value")
        return df.sort_index().resample("D").last()

    # ── cache ───────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(prefix: str, *parts) -> str:
        raw = f"{prefix}:{parts}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _read_cache(key: str) -> Optional[pd.DataFrame]:
        path = CACHE_DIR / f"{key}.parquet"
        if path.exists():
            logger.debug(f"Cache hit: {path}")
            return pd.read_parquet(path)
        return None

    @staticmethod
    def _write_cache(key: str, df: pd.DataFrame) -> None:
        path = CACHE_DIR / f"{key}.parquet"
        df.to_parquet(path)
        logger.debug(f"Cached: {path}")
