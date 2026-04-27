"""Snapshot the live Supabase warehouse into a local DuckDB file.

Run once while online; the resulting `.duckdb` file lets the rest of the
pipeline run offline through DuckDBCPCMDataLoader.

Usage:
    python -m causal_portfolio.data.snapshot
    python -m causal_portfolio.data.snapshot --out ~/cpcm_local.duckdb
    python -m causal_portfolio.data.snapshot --assets btc,eth,sol --start 2023-01-01

Strategy: stream `asset_metrics` from Supabase REST in 50k-row pages, then
INSERT OR REPLACE into the local DuckDB file. Avoids the postgres extension
(which needs a direct DB connection and credentials we usually don't carry).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

logger = logging.getLogger("cpcm.snapshot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAGE_SIZE = 1000  # Supabase PostgREST hard cap


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

    total = 0
    offset = 0
    t0 = time.time()
    while True:
        q = client.table("asset_metrics").select(
            "provider,provider_priority,asset,metric,time,value,frequency,metadata"
        )
        if assets:
            q = q.in_("asset", assets)
        if start:
            q = q.gte("time", start)
        if end:
            q = q.lte("time", end)
        q = q.order("time").range(offset, offset + PAGE_SIZE - 1)

        result = q.execute()
        rows = result.data or []
        if not rows:
            break

        n = loader.upsert_rows(rows)
        total += n
        if offset % (PAGE_SIZE * 50) == 0:
            elapsed = time.time() - t0
            logger.info("  %d rows snapshotted (%.0f rows/sec)", total, total / max(elapsed, 1e-3))

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    loader.close()
    logger.info("Done: %d rows in %.1fs", total, time.time() - t0)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path.home() / "cpcm_local.duckdb"))
    p.add_argument("--assets", help="Comma-separated asset filter (default: all)")
    p.add_argument("--start", help="ISO date lower bound")
    p.add_argument("--end", help="ISO date upper bound")
    p.add_argument("--truncate", action="store_true", help="Wipe local table first")
    args = p.parse_args()

    assets = args.assets.split(",") if args.assets else None
    snapshot(args.out, assets=assets, start=args.start, end=args.end, truncate=args.truncate)


if __name__ == "__main__":
    main()
