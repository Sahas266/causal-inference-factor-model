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


def _last_sync_time(loader: DuckDBCPCMDataLoader) -> str | None:
    loader._con.execute(
        "CREATE TABLE IF NOT EXISTS snapshot_meta(key TEXT PRIMARY KEY, value TEXT)"
    )
    row = loader._con.execute(
        "SELECT value FROM snapshot_meta WHERE key = 'last_sync_utc'"
    ).fetchone()
    return row[0] if row else None


def _record_sync_time(loader: DuckDBCPCMDataLoader, when: datetime) -> None:
    loader._con.execute(
        "CREATE TABLE IF NOT EXISTS snapshot_meta(key TEXT PRIMARY KEY, value TEXT)"
    )
    loader._con.execute(
        "INSERT OR REPLACE INTO snapshot_meta VALUES ('last_sync_utc', ?)",
        [when.isoformat()],
    )


def _fetch_backfilled(client, created_since: str, time_before: datetime,
                      assets: list[str] | None) -> list[dict]:
    """Rows INSERTED after the previous snapshot but STAMPED before the resume
    cursor — the resume-by-MAX(time) cursor alone silently misses these when a
    backfill adds historical rows, and the local DB diverges from the
    warehouse forever."""
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table("asset_metrics").select(
            "provider,provider_priority,asset,metric,time,value,frequency,metadata"
        )
        if assets:
            q = q.in_("asset", assets)
        q = q.gte("created_at", created_since).lt("time", time_before.isoformat())
        q = q.order("created_at").order("time").order("provider")
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

    sync_started = datetime.now(timezone.utc)
    local_max = _local_max_time(loader)
    cursor_start = start or local_max or "2021-01-01"
    win_start = _parse_iso(cursor_start)
    resume_cursor = win_start  # for the late-backfill sweep below
    is_resume = start is None and not truncate and local_max is not None
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

    # Late-backfill sweep: on resume, also pull rows inserted since the last
    # snapshot run whose timestamps fall BEFORE the resume cursor.
    prev_sync = _last_sync_time(loader)
    if is_resume and prev_sync:
        try:
            late = _fetch_backfilled(client, prev_sync, resume_cursor, assets)
            if late:
                loader.upsert_rows(late)
                total += len(late)
            logger.info(
                "Late-backfill sweep (created_at >= %s, time < %s): %d rows",
                prev_sync, resume_cursor.date(), len(late),
            )
        except Exception as e:
            logger.error("Late-backfill sweep failed (continuing): %s", e)
    _record_sync_time(loader, sync_started)

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
