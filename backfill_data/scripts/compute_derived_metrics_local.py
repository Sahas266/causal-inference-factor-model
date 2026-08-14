"""Compute the derived metrics against the local DuckDB instead of Supabase.

`compute_derived_metrics.py` reads and writes Supabase, but the models read
the local DuckDB mirror. Its formulas are the authoritative definitions of
realized volatility, log returns, Sharpe, beta and the flow ratios, so this
driver swaps only the two I/O seams — `fetch_all` and `upsert_batch` — and
calls the same compute functions.

Reusing rather than reimplementing is the point: a second copy of these
formulas would drift from the warehouse the first time either changed, and
a silently different realized-volatility definition is exactly the kind of
divergence nobody notices until a factor stops matching.

Usage:
    python scripts/compute_derived_metrics_local.py
    python scripts/compute_derived_metrics_local.py --asset btc,eth
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

logger = logging.getLogger("derived.local")


def _connect(db_path: str | None, read_only: bool):
    import duckdb

    from causal_portfolio.data import DEFAULT_LOCAL_DB

    con = duckdb.connect(str(db_path or DEFAULT_LOCAL_DB), read_only=read_only)
    con.execute("SET TimeZone='UTC'")
    return con


def _as_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _daily_prices(rows: list[dict]) -> list[dict]:
    """Keep the last deterministic observation per completed UTC day."""
    chosen: dict[date, tuple[datetime, str, dict]] = {}
    today = datetime.now(timezone.utc).date()
    for row in rows:
        observed = _as_utc(row["time"])
        if observed.date() >= today:
            continue
        # Exact timestamps are unique in the warehouse. Value breaks any tie
        # in malformed inputs deterministically without depending on row order.
        candidate = (observed, str(row.get("value")), row)
        if observed.date() not in chosen or candidate[:2] > chosen[observed.date()][:2]:
            chosen[observed.date()] = candidate
    return [
        {**chosen[day][2], "time": f"{day.isoformat()}T00:00:00+00:00"}
        for day in sorted(chosen)
    ]


def install_duckdb_io(module, db_path: str | None) -> list[dict]:
    """Point the Supabase-shaped helpers at DuckDB. Returns the write buffer.

    Writes are buffered rather than applied per call because DuckDB takes an
    exclusive write lock: holding one open across every compute pass would
    block any other local reader for the whole run.
    """
    pending: list[dict] = []
    price_days: dict[str, set[date]] = {}

    def fetch_all(table, filters, select="*", order_col="time"):
        if table != "asset_metrics":
            return []
        where, params = [], []
        for column, value in filters.items():
            where.append(f"{column} = ?")
            params.append(value)
        where.append("time::date < current_date")
        clause = " where " + " and ".join(where)
        con = _connect(db_path, read_only=True)
        try:
            rows = con.execute(
                f"select time, value, asset, metric, provider from asset_metrics"
                f"{clause} order by time, asset, metric, provider, value",
                params,
            ).fetchall()
        finally:
            con.close()
        # The compute functions expect Supabase's dict rows with ISO strings.
        return [
            {
                "time": t.isoformat() if hasattr(t, "isoformat") else str(t),
                "value": v,
                "asset": a,
                "metric": m,
                "provider": p,
            }
            for t, v, a, m, p in rows
        ]

    def upsert_batch(records, batch_size=500):
        pending.extend(records)
        return len(records)

    def fetch_price_data(asset):
        """Pick the FRESHEST adequate price source, not the first one.

        The base helper walks PRICE_SOURCES (artemis, coinmetrics, coingecko)
        and stops at the first provider with >= 31 rows. Upstream that is
        fine, but locally artemis is frozen at 2026-01-01 while coinmetrics
        and coingecko are current — so the original order would recompute
        every derived metric on eight-month-old prices and report success.
        """
        best_rows, best_source, best_key = [], "none", None
        for source_rank, (provider, metric) in enumerate(module.PRICE_SOURCES):
            rows = _daily_prices(fetch_all(
                "asset_metrics",
                {"provider": provider, "asset": asset, "metric": metric},
                select="time,value",
            ))
            if len(rows) < 31:
                continue
            latest = max(r["time"] for r in rows)
            # Freshest source wins. Declared PRICE_SOURCES order breaks ties.
            key = (latest, -source_rank)
            if best_key is None or key > best_key:
                best_rows, best_source, best_key = rows, f"{provider}:{metric}", key
        price_days[asset] = {date.fromisoformat(row["time"][:10]) for row in best_rows}
        return best_rows, best_source

    original_compute_rv = module.compute_realized_volatility

    def compute_realized_volatility(asset):
        records = original_compute_rv(asset) or []
        available = price_days.get(asset, set())
        return [
            record
            for record in records
            if all(
                date.fromisoformat(record["time"][:10]) - timedelta(days=offset)
                in available
                for offset in range(int(record.get("metadata", {}).get("window", 0)) + 1)
            )
        ]

    module.fetch_all = fetch_all
    module.upsert_batch = upsert_batch
    module.fetch_price_data = fetch_price_data
    module.compute_realized_volatility = compute_realized_volatility
    return pending


def flush(pending: list[dict], db_path: str | None) -> int:
    if not pending:
        return 0
    from causal_portfolio.data import DEFAULT_LOCAL_DB
    from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

    # Derived history is immutable. Only append dates newer than the latest
    # committed row for each series; do not rewrite or clean existing rows.
    con = _connect(db_path, read_only=True)
    try:
        maxima = {
            (asset, metric): timestamp
            for asset, metric, timestamp in con.execute(
                "select asset, metric, max(time) from asset_metrics "
                "where provider = 'derived' group by asset, metric"
            ).fetchall()
        }
    finally:
        con.close()

    unique: dict[tuple[str, str, str], dict] = {}
    today = datetime.now(timezone.utc).date()
    for record in pending:
        observed = _as_utc(record["time"])
        if observed.date() >= today or observed.time().replace(tzinfo=None) != datetime.min.time():
            raise ValueError("derived rows must use a completed UTC day at midnight")
        key = (record["asset"], record["metric"], observed.date().isoformat())
        if key in unique and unique[key] != record:
            raise ValueError(f"conflicting derived rows for {key}")
        latest = maxima.get((record["asset"], record["metric"]))
        if latest is None or observed > _as_utc(latest):
            unique[key] = record
    pending = list(unique.values())
    if not pending:
        return 0

    loader = DuckDBCPCMDataLoader(
        db_path=str(db_path or DEFAULT_LOCAL_DB), read_only=False
    )
    try:
        return loader.upsert_rows(pending)
    finally:
        loader.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default=None, help="comma-separated tickers (default: all with price)")
    p.add_argument("--db-path", default=None)
    args = p.parse_args(argv)

    # Imported after load_dotenv: the module exits at import time without
    # Supabase credentials, even though this driver never calls Supabase.
    import compute_derived_metrics as base

    pending = install_duckdb_io(base, args.db_path)

    assets = (
        [a.strip().lower() for a in args.asset.split(",") if a.strip()]
        if args.asset
        else base.get_all_assets_with_price()
    )
    logger.info("computing derived metrics for %d asset(s)", len(assets))

    # The compute functions RETURN their records; only main() upserts them.
    failed = False
    for asset in assets:
        try:
            pending.extend(base.compute_realized_volatility(asset) or [])
        except Exception as exc:
            logger.error("%s: realized volatility failed: %s", asset, exc)
            failed = True

    try:
        pending.extend(base.compute_dex_cex_volume_ratio() or [])
    except Exception as exc:
        logger.error("dex/cex volume ratio failed: %s", exc)
        failed = True

    if failed:
        logger.error("derived computation failed; no rows written")
        return 1

    produced = len(pending)

    try:
        written = flush(pending, args.db_path)
    except Exception as exc:
        logger.error("derived write failed: %s", exc)
        return 1
    logger.info("computed %d records, upserted %d rows", produced, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
