"""Execute the saved Dune queries so their cached results exist.

The Dune provider only reads `GET /query/{id}/results`, which serves the last
cached execution. Every configured query currently answers:

    404  "No execution found for the latest version of the given query"

because the saved queries were edited (or never run) and Dune keeps no result
for the current version. Warming the cache here means the normal provider path
— and the Supabase backfill — both work afterwards, instead of teaching the
shared provider to execute as a side effect of reading.

Execution consumes Dune credits, so this is deliberately a separate, explicit
step rather than something a routine backfill triggers.

Usage:
    python scripts/warm_dune_queries.py --list
    python scripts/warm_dune_queries.py --limit 1        # try one first
    python scripts/warm_dune_queries.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from src.core.utils.config_loader import ConfigLoader, iter_endpoint_config_files
from src.providers.registry import ProviderRegistry

logger = logging.getLogger("dune.warm")

TERMINAL_OK = {"QUERY_STATE_COMPLETED"}
TERMINAL_BAD = {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"}


def dune_query_ids(config_dir: Path) -> list[tuple[int, str]]:
    """(query_id, endpoint_id) for every enabled Dune endpoint."""
    loader = ConfigLoader(str(config_dir))
    out: dict[int, str] = {}
    for path in iter_endpoint_config_files(config_dir / "endpoints"):
        try:
            endpoint = loader.load_endpoint_config(str(path))
        except Exception:
            continue
        for provider_config in endpoint.get("providers", []):
            if provider_config.get("name") != "dune":
                continue
            if not provider_config.get("enabled", True):
                continue
            query_id = provider_config.get("config", {}).get("params", {}).get("query_id")
            if query_id:
                # Several endpoints can share a query id; execute it once.
                out.setdefault(int(query_id), endpoint.get("endpoint_id", "?"))
    return sorted(out.items())


def warm(client, query_id: int, *, poll_seconds: float, timeout_seconds: float) -> str:
    """Execute one query and wait for it to settle. Returns final state."""
    started = client.execute_query(query_id)
    execution_id = started.get("execution_id")
    if not execution_id:
        return f"no execution_id: {started}"
    deadline = time.monotonic() + timeout_seconds
    state = "QUERY_STATE_PENDING"
    while time.monotonic() < deadline:
        status = client.get_execution_status(execution_id)
        state = status.get("state", "UNKNOWN")
        if state in TERMINAL_OK or state in TERMINAL_BAD:
            return state
        time.sleep(poll_seconds)
    return f"TIMEOUT (last state {state})"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="only warm the first N queries")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--timeout-seconds", type=float, default=300.0)
    p.add_argument("--force", action="store_true",
                   help="re-execute even when results are already cached")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    config_dir = PROJECT_ROOT / "config"
    queries = dune_query_ids(config_dir)
    if args.limit:
        queries = queries[: args.limit]
    if args.list:
        for query_id, endpoint_id in queries:
            print(f"  {query_id}  {endpoint_id}")
        print(f"  ({len(queries)} distinct queries)")
        return 0
    if not queries:
        logger.error("no Dune queries configured")
        return 1

    ProviderRegistry.auto_discover()
    loader = ConfigLoader(str(config_dir))
    provider = ProviderRegistry.get_provider("dune")()
    provider.initialize(loader.load_provider_config("dune"))
    client = provider.client

    ok = bad = skipped = 0
    for query_id, endpoint_id in queries:
        # Resumable and credit-safe: a query that already has cached results
        # needs no execution, so an interrupted run can simply be repeated.
        if not args.force:
            try:
                client.get_query_results(query_id, limit=1, offset=0)
                skipped += 1
                logger.info("%-10s %-34s already cached", query_id, endpoint_id[:34])
                continue
            except Exception:
                pass
        try:
            state = warm(
                client, query_id,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        except Exception as exc:
            logger.error("%-10s %-34s execute failed: %s", query_id, endpoint_id[:34], exc)
            bad += 1
            continue
        if state in TERMINAL_OK:
            ok += 1
            logger.info("%-10s %-34s %s", query_id, endpoint_id[:34], state)
        else:
            bad += 1
            logger.warning("%-10s %-34s %s", query_id, endpoint_id[:34], state)

    logger.info("warmed %d, already cached %d, failed %d", ok, skipped, bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
