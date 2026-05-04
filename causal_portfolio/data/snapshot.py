"""Snapshot the live Supabase warehouse into a local DuckDB file.

Run once while online; the resulting `.duckdb` file lets the rest of the
pipeline run offline through DuckDBCPCMDataLoader.

Usage:
    python -m causal_portfolio.data.snapshot
    python -m causal_portfolio.data.snapshot --out ~/cpcm_local.duckdb
    python -m causal_portfolio.data.snapshot --assets btc,eth,sol --start 2023-01-01

Pagination strategy: TIME-WINDOWED CHUNKS. The warehouse is split into ~30-day
windows; within each window we offset-paginate. This avoids two failure modes:
  - Deep offsets across the whole table → Postgres statement_timeout
  - Same-timestamp boundaries → keyset cursor stuck in a loop

Resume: by default, picks up from MAX(time) in the local DB. Use --truncate to
wipe and start over, or --start to override the cursor.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

logger = logging.getLogger("cpcm.snapshot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAGE_SIZE = 1000           # PostgREST hard cap
WINDOW_DAYS = 30           # chunk width — small enough to stay well under timeout
DEFAULT_END = "2026-12-31" # upper sentinel for full-range pulls


def _parse_iso(s: str) -> datetime:
    """Parse an ISO date or datetime string to a UTC-aware datetime."""
    if "T" not in s and " " not in s:
        s = s + "T00:00:00+00:00"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _local_max_time(loader: DuckDBCPCMDataLoader) -> str | None:
    row = loader._con.execute("SELECT MAX(time) FROM asset_metrics").fetchone()
    if row and row[0]:
        dt = row[0]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return None


def _fetch_window(client, win_start: datetime, win_end: datetime,
                  assets: list[str] | None) -> list[dict]:
    """Pull every row whose time is in [win_start, win_end) via offset pagination."""
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table("asset_metrics").select(
            "provider,provider_priority,asset,metric,time,value,frequency,metadata"
        )
        if assets:
            q = q.in_("asset", assets)
        q = q.gte("time", win_start.isoformat()).lt("time", win_end.isoformat())
        q = q.order("time").order("provider").order("asset").order("metric")
        q = q.range(offset, offset + PAGE_SIZE - 1)

        result = q.execute()
        page = result.data or []
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def snapshot(
    out_path: str,
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    truncate: bool = False,
) -> int:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / "backfill_data" / ".env")
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    client = create_client(url, key)

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing snapshot to %s", out)

    loader = DuckDBCPCMDataLoader(db_path=str(out), read_only=False)
    if truncate:
        loader._con.execute("DELETE FROM asset_metrics")
        logger.info("Truncated existing asset_metrics")

    cursor_start = start or _local_max_time(loader) or "2021-01-01"
    win_start = _parse_iso(cursor_start)
    final_end = _parse_iso(end or DEFAULT_END)
    logger.info("Snapshotting %s → %s in %d-day windows",
                win_start.date(), final_end.date(), WINDOW_DAYS)

    total = 0
    t0 = time.time()
    while win_start < final_end:
        win_end = min(win_start + timedelta(days=WINDOW_DAYS), final_end)
        try:
            rows = _fetch_window(client, win_start, win_end, assets)
        except Exception as e:
            logger.error("Window %s→%s failed: %s", win_start.date(), win_end.date(), e)
            raise

        if rows:
            loader.upsert_rows(rows)
            total += len(rows)
        elapsed = time.time() - t0
        logger.info(
            "  %s → %s: %d rows (running total %d, %.0f rows/sec)",
            win_start.date(), win_end.date(), len(rows), total,
            total / max(elapsed, 1e-3),
        )
        win_start = win_end

    loader.close()
    logger.info("Done: %d rows in %.1fs", total, time.time() - t0)
    return total


def main():
    from causal_portfolio.data import DEFAULT_LOCAL_DB
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_LOCAL_DB))
    p.add_argument("--assets", help="Comma-separated asset filter (default: all)")
    p.add_argument("--start", help="ISO date lower bound (overrides resume cursor)")
    p.add_argument("--end", help="ISO date upper bound")
    p.add_argument("--truncate", action="store_true", help="Wipe local table first")
    args = p.parse_args()

    assets = args.assets.split(",") if args.assets else None
    snapshot(args.out, assets=assets, start=args.start, end=args.end, truncate=args.truncate)


if __name__ == "__main__":
    main()
