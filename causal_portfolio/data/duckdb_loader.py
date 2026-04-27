"""Local DuckDB-backed CPCM data loader.

Mirrors CPCMDataLoader's public API (load_panel, load_returns, load_macro) but
reads from a local DuckDB file instead of the Supabase REST API. Use for offline
work or to speed up read-heavy analytics.

Schema parity with Supabase:
- table `asset_metrics` matches the Postgres definition
- view `asset_metrics_best` reproduces Postgres `DISTINCT ON (asset, metric, time)`
  picking the row with lowest `provider_priority`, using DuckDB's QUALIFY.

Snapshot a Supabase warehouse into a DuckDB file with `snapshot.py`.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.data.duckdb")

CACHE_DIR = Path(__file__).parent / "cache"

ASSET_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS asset_metrics (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    asset TEXT NOT NULL,
    metric TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    value DOUBLE,
    frequency TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (provider, asset, metric, time)
);
"""

# Mirrors Postgres `SELECT DISTINCT ON (asset, metric, time) ... ORDER BY ..., provider_priority ASC`
ASSET_METRICS_BEST_VIEW = """
CREATE OR REPLACE VIEW asset_metrics_best AS
SELECT provider, asset, metric, time, value, frequency, metadata, created_at
FROM asset_metrics
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY asset, metric, time
    ORDER BY provider_priority ASC
) = 1;
"""


def _to_utc_bound(s: str, which: str) -> str:
    """Normalize a date or datetime string to UTC ISO with offset.

    DuckDB parses bare 'YYYY-MM-DD' against the session timezone, which shifts
    UTC-midnight rows out of the query window on non-UTC machines. Always pass
    explicit UTC bounds: start → 00:00:00+00:00, end → 23:59:59.999+00:00.
    """
    if "T" in s or " " in s:
        # Already has a time component — trust the caller.
        return s
    return f"{s}T00:00:00+00:00" if which == "start" else f"{s}T23:59:59.999+00:00"


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create asset_metrics table + asset_metrics_best view if missing."""
    con.execute(ASSET_METRICS_SCHEMA)
    con.execute(ASSET_METRICS_BEST_VIEW)


class DuckDBCPCMDataLoader:
    """Drop-in DuckDB replacement for CPCMDataLoader.

    Selected via the CPCM_LOCAL_DB env var (path to .duckdb file). The factory
    in `causal_portfolio.data.__init__` routes here automatically when set.
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = True):
        self.db_path = db_path or os.environ.get("CPCM_LOCAL_DB")
        if not self.db_path:
            raise ValueError(
                "DuckDBCPCMDataLoader requires db_path or CPCM_LOCAL_DB env var"
            )
        # read_only=False on a missing file would create an empty one, which is
        # almost never what callers want. Initialize via snapshot.py instead.
        if read_only and not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"DuckDB file not found: {self.db_path}. "
                f"Run `python -m causal_portfolio.data.snapshot` to create it."
            )
        self._con = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            ensure_schema(self._con)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── public API (mirrors CPCMDataLoader) ─────────────────────────

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
        cache_key = self._cache_key("panel_duckdb", assets, metrics, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        df_long = self._con.execute(
            """
            SELECT asset, metric, time, value
            FROM asset_metrics_best
            WHERE asset = ANY(?) AND metric = ANY(?)
              AND time >= ?::TIMESTAMPTZ AND time <= ?::TIMESTAMPTZ
            ORDER BY time
            """,
            [assets, metrics, _to_utc_bound(start, "start"), _to_utc_bound(end, "end")],
        ).fetchdf()

        df = self._pivot(df_long)
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
        """Load prices and compute log returns. Falls back to 'price' metric."""
        cache_key = self._cache_key("returns_duckdb", assets, [price_metric], start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

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
        """Load FRED macro series (asset='macro')."""
        cache_key = self._cache_key("macro_duckdb", ["macro"], series_ids, start, end)
        if use_cache and (cached := self._read_cache(cache_key)) is not None:
            return cached

        df = self._con.execute(
            """
            SELECT metric, time, value
            FROM asset_metrics_best
            WHERE asset = 'macro' AND metric = ANY(?)
              AND time >= ?::TIMESTAMPTZ AND time <= ?::TIMESTAMPTZ
            ORDER BY time
            """,
            [series_ids, _to_utc_bound(start, "start"), _to_utc_bound(end, "end")],
        ).fetchdf()

        if df.empty:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.pivot_table(index="time", columns="metric", values="value")
        df = df.sort_index().resample("D").last().ffill()
        df.columns = [c.lower() for c in df.columns]

        if use_cache and not df.empty:
            self._write_cache(cache_key, df)
        return df

    # ── write helpers (only available when read_only=False) ─────────

    def upsert_rows(self, rows: list[dict]) -> int:
        """Upsert rows into asset_metrics. For tests + offline backfills.

        Each row needs: provider, provider_priority, asset, metric, time,
        value, frequency. Optional: metadata.
        """
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        if "metadata" not in df.columns:
            df["metadata"] = None
        # DuckDB's INSERT OR REPLACE handles the (provider, asset, metric, time) PK
        self._con.register("incoming", df)
        self._con.execute(
            """
            INSERT OR REPLACE INTO asset_metrics
                (provider, provider_priority, asset, metric, time, value, frequency, metadata)
            SELECT provider, provider_priority, asset, metric, time, value, frequency, metadata
            FROM incoming
            """
        )
        self._con.unregister("incoming")
        return len(df)

    # ── internal ────────────────────────────────────────────────────

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        # Normalize all timestamps to UTC then drop tz so daily resample is
        # deterministic regardless of system local timezone.
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["col"] = df["asset"] + "_" + df["metric"]
        df = df.pivot_table(index="time", columns="col", values="value")
        return df.sort_index().resample("D").last()

    @staticmethod
    def _cache_key(prefix: str, *parts) -> str:
        raw = f"{prefix}:{parts}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _read_cache(key: str) -> Optional[pd.DataFrame]:
        path = CACHE_DIR / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    @staticmethod
    def _write_cache(key: str, df: pd.DataFrame) -> None:
        path = CACHE_DIR / f"{key}.parquet"
        df.to_parquet(path)
