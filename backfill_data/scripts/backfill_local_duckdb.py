"""Backfill providers straight into the local DuckDB, bypassing Supabase.

The orchestrator writes only to Supabase (`db_writer` is constructed inline,
not injected), and the local DuckDB is what the models actually read. This
script reuses the committed provider implementations and the orchestrator's
normalization rules, but sinks the records into DuckDB instead.

It deliberately reuses `METRIC_NORMALIZATION` / `PROVIDER_PRIORITY` from the
orchestrator rather than restating them: those are the single chokepoint that
keeps metric names and best-provider precedence consistent, and a second copy
here would drift from the warehouse the first time either changed.

Not every provider can be run. `artemis` and `hyperliquid` hold the bulk of
the historical rows but their source and configs are absent from the tree
(only __pycache__ remains), so they are skipped with a clear message rather
than failing halfway.

Usage:
    python scripts/backfill_local_duckdb.py --list
    python scripts/backfill_local_duckdb.py --providers defillama,fred
    python scripts/backfill_local_duckdb.py --providers all --start 2026-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from src.core.orchestrator import (
    METRIC_NORMALIZATION,
    PROVIDER_PRIORITY,
    COINMETRICS_PRICE_PRIORITY,
    _parse_date_range_bound,
)
from src.core.utils.config_loader import ConfigLoader, iter_endpoint_config_files
from src.providers.registry import ProviderRegistry

logger = logging.getLogger("backfill.local")

#: Providers whose implementation is not in the committed tree.
UNAVAILABLE = {"artemis", "hyperliquid"}


def _local_loader(db_path: str | None):
    from causal_portfolio.data import DEFAULT_LOCAL_DB
    from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

    return DuckDBCPCMDataLoader(
        db_path=str(db_path or DEFAULT_LOCAL_DB), read_only=False
    )


def _latest_local_per_provider(db_path: str | None) -> dict[str, str]:
    """Latest local timestamp per provider, minus a day of overlap.

    Re-fetching the whole history for 159 endpoints would take hours and
    mostly rewrite rows that already exist. The one-day overlap means a
    provider revising its most recent value still lands.
    """
    import duckdb
    from datetime import timedelta

    from causal_portfolio.data import DEFAULT_LOCAL_DB

    try:
        con = duckdb.connect(str(db_path or DEFAULT_LOCAL_DB), read_only=True)
    except Exception as exc:
        logger.warning("cannot read local watermarks: %s", exc)
        return {}
    try:
        rows = con.execute(
            "select provider, max(time) from asset_metrics group by provider"
        ).fetchall()
    except Exception:
        return {}
    finally:
        con.close()
    out = {}
    for provider, latest in rows:
        if latest is None:
            continue
        out[provider] = (latest.date() - timedelta(days=1)).isoformat()
    return out


def _normalize(records: list[dict], provider_name: str, provider_config: dict) -> list[dict]:
    """Apply the orchestrator's metric/priority rules to a chunk."""
    for record in records:
        record["provider"] = provider_name
        metric = record.get("metric")
        if metric in METRIC_NORMALIZATION:
            metric = METRIC_NORMALIZATION[metric]
            record["metric"] = metric
        priority = PROVIDER_PRIORITY.get(
            provider_name, provider_config.get("priority", 999)
        )
        if provider_name == "coinmetrics" and metric == "price":
            priority = COINMETRICS_PRICE_PRIORITY
        record["provider_priority"] = priority
    return records


def discover(config_dir: Path) -> list[tuple[str, dict, dict]]:
    """Return (provider_name, endpoint_config, provider_config) triples."""
    loader = ConfigLoader(str(config_dir))
    found = []
    for path in iter_endpoint_config_files(config_dir / "endpoints"):
        try:
            endpoint = loader.load_endpoint_config(str(path))
        except Exception as exc:
            logger.warning("skipping unreadable config %s: %s", path.name, exc)
            continue
        # The local DuckDB mirror only carries asset_metrics; the market_*
        # tables have a different schema (and are empty upstream anyway), so
        # routing them through upsert_rows just raises a binder error.
        if endpoint.get("table") != "asset_metrics":
            continue
        for provider_config in endpoint.get("providers", []):
            if not provider_config.get("enabled", True):
                continue
            found.append((provider_config["name"], endpoint, provider_config))
    return found


def run(
    providers: set[str],
    *,
    start: str | None,
    end: str | None,
    db_path: str | None,
    config_dir: Path,
    limit: int | None,
) -> int:
    ProviderRegistry.auto_discover()
    jobs = [job for job in discover(config_dir) if job[0] in providers]
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        logger.warning("no matching endpoint configs")
        return 0

    # Providers are stateful and hold their API session, so build each once —
    # the orchestrator does the same. A provider that cannot initialize (no
    # key, disabled) drops out here rather than failing once per endpoint.
    config_loader = ConfigLoader(str(config_dir))
    live: dict[str, object] = {}
    for provider_name in sorted({job[0] for job in jobs}):
        try:
            provider_settings = config_loader.load_provider_config(provider_name)
            instance = ProviderRegistry.get_provider(provider_name)()
            instance.initialize(provider_settings)
            live[provider_name] = instance
        except Exception as exc:
            logger.error("provider %s unavailable: %s", provider_name, exc)
    if not live:
        logger.error("no provider could be initialized")
        return 0

    local_since = {} if start else _latest_local_per_provider(db_path)
    if local_since:
        logger.info(
            "resuming from local data: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(local_since.items())),
        )

    loader = _local_loader(db_path)
    written = failures = 0
    try:
        for provider_name, endpoint, provider_config in jobs:
            endpoint_id = endpoint.get("endpoint_id", "?")
            provider = live.get(provider_name)
            if provider is None:
                continue

            # End is "now" unless overridden: the point of this script is to
            # bring local data up to today, and most endpoint configs still
            # carry a stale hardcoded end.
            end_time = _parse_date_range_bound(end or "latest", 23)
            # Start incrementally from what is already local, so re-running is
            # cheap and idempotent. Falls back to the config's start on an
            # empty table. Overlap by a day to catch provider revisions.
            start_time = _parse_date_range_bound(
                start
                or local_since.get(provider_name)
                or provider_config.get("actual_start")
                or endpoint.get("date_range", {}).get("start"),
                0,
            )
            if start_time and end_time and start_time > end_time:
                logger.warning(
                    "%s / %s: start %s is after end %s, skipping",
                    provider_name, endpoint_id,
                    start_time.date(), end_time.date(),
                )
                continue
            rows = 0
            try:
                for chunk in provider.fetch_data_stream(
                    provider_config.get("config", {}), start_time, end_time
                ):
                    if not chunk:
                        continue
                    rows += loader.upsert_rows(
                        _normalize(chunk, provider_name, provider_config)
                    )
            except Exception as exc:
                # One endpoint failing must not abort the whole sweep — a
                # rate limit on one series should not cost the other 150.
                logger.error("%s / %s failed: %s", provider_name, endpoint_id, exc)
                failures += 1
                continue
            written += rows
            logger.info("%-12s %-36s %6d rows", provider_name, endpoint_id[:36], rows)
    finally:
        loader.close()

    logger.info("total rows upserted: %d (%d endpoint failures)", written, failures)
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # The provider keys live in backfill_data/.env, not the repo-root .env.
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--providers", default="all",
                   help="comma-separated provider names, or 'all'")
    p.add_argument("--start", default=None, help="override start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="override end (YYYY-MM-DD or 'latest')")
    p.add_argument("--db-path", default=None, help="target DuckDB (default: local)")
    p.add_argument("--limit", type=int, default=None, help="cap endpoints, for a smoke run")
    p.add_argument("--list", action="store_true", help="show endpoints per provider and exit")
    args = p.parse_args(argv)

    config_dir = PROJECT_ROOT / "config"
    jobs = discover(config_dir)
    by_provider: dict[str, int] = {}
    for name, _endpoint, _pc in jobs:
        by_provider[name] = by_provider.get(name, 0) + 1

    if args.list:
        for name in sorted(by_provider):
            note = "  (source not in tree — skipped)" if name in UNAVAILABLE else ""
            print(f"  {name:<14} {by_provider[name]:3d} endpoints{note}")
        return 0

    requested = (
        set(by_provider) if args.providers == "all"
        else {p.strip() for p in args.providers.split(",") if p.strip()}
    )
    unavailable = requested & UNAVAILABLE
    if unavailable:
        logger.warning(
            "skipping %s: provider source is not in the committed tree",
            ", ".join(sorted(unavailable)),
        )
    runnable = requested - UNAVAILABLE
    unknown = runnable - set(by_provider)
    if unknown:
        logger.warning("no endpoints configured for: %s", ", ".join(sorted(unknown)))
    runnable &= set(by_provider)
    if not runnable:
        logger.error("nothing to run")
        return 1

    logger.info("running providers: %s", ", ".join(sorted(runnable)))
    run(
        runnable,
        start=args.start,
        end=args.end,
        db_path=args.db_path,
        config_dir=config_dir,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
