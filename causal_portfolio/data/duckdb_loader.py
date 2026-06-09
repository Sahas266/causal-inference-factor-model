"""Local DuckDB-backed CPCM data loader.

Mirrors CPCMDataLoader's public API (load_panel, load_prices, load_returns,
load_macro) but reads from a local DuckDB file instead of the Supabase REST
API. Use for offline work or to speed up read-heavy analytics. The shared
pivot/returns/caching logic lives in BaseCPCMDataLoader; this class only
implements the SQL fetch hooks and write helpers.

Schema parity with Supabase:
- table `asset_metrics` matches the Postgres definition
- view `asset_metrics_best` reproduces Postgres `DISTINCT ON (asset, metric, time)`
  picking the row with lowest `provider_priority`, using DuckDB's QUALIFY.

Snapshot a Supabase warehouse into a DuckDB file with `snapshot.py`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from causal_portfolio.data.base_loader import BaseCPCMDataLoader

logger = logging.getLogger("cpcm.data.duckdb")

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


class DuckDBCPCMDataLoader(BaseCPCMDataLoader):
    """Drop-in DuckDB replacement for CPCMDataLoader.

    Selected via the CPCM_LOCAL_DB env var (path to .duckdb file). The factory
    in `causal_portfolio.data.__init__` routes here automatically when set.
    """

    cache_tag = "duckdb"

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

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── backend hooks ───────────────────────────────────────────────

    def _fetch_panel_long(self, assets, metrics, start, end) -> pd.DataFrame:
        return self._con.execute(
            """
            SELECT asset, metric, time, value
            FROM asset_metrics_best
            WHERE asset = ANY(?) AND metric = ANY(?)
              AND time >= ?::TIMESTAMPTZ AND time <= ?::TIMESTAMPTZ
            ORDER BY time
            """,
            [assets, metrics, _to_utc_bound(start, "start"), _to_utc_bound(end, "end")],
        ).fetchdf()

    def _fetch_macro_long(self, series_ids, start, end) -> pd.DataFrame:
        return self._con.execute(
            """
            SELECT metric, time, value
            FROM asset_metrics_best
            WHERE asset = 'macro' AND metric = ANY(?)
              AND time >= ?::TIMESTAMPTZ AND time <= ?::TIMESTAMPTZ
            ORDER BY time
            """,
            [series_ids, _to_utc_bound(start, "start"), _to_utc_bound(end, "end")],
        ).fetchdf()

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
