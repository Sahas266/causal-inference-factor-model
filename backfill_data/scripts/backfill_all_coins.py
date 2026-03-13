#!/usr/bin/env python3
"""
Backfill all coins defined in coin_manifest.json.

Reads the manifest, finds config directories per coin, and runs backfills
sequentially using BackfillOrchestrator.

Usage:
    cd backfill_data
    python scripts/backfill_all_coins.py                        # all coins
    python scripts/backfill_all_coins.py --asset sol,btc        # specific coins
    python scripts/backfill_all_coins.py --dry-run              # list what would run
    python scripts/backfill_all_coins.py --validate-only        # validate configs only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.orchestrator import BackfillOrchestrator
from src.core.utils.logger import setup_logger
from src.core.utils.config_loader import ConfigLoader

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
ENDPOINTS_DIR = CONFIG_DIR / "endpoints"
MANIFEST_PATH = CONFIG_DIR / "coin_manifest.json"


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Backfill all coins from manifest")
    parser.add_argument("--asset", type=str, help="Comma-separated tickers to backfill (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="List what would run without executing")
    parser.add_argument("--validate-only", action="store_true", help="Validate configs without backfilling")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent workers per coin")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logger(level=args.log_level)
    manifest = load_manifest()

    # Filter coins
    if args.asset:
        requested = set(a.strip().lower() for a in args.asset.split(","))
        coins = [c for c in manifest["coins"] if c["ticker"] in requested]
        missing = requested - {c["ticker"] for c in coins}
        if missing:
            logger.warning(f"Coins not found in manifest: {missing}")
    else:
        coins = [c for c in manifest["coins"] if not c.get("skip")]

    if not coins:
        logger.error("No coins to backfill")
        return 1

    # Collect configs per coin
    coin_configs = []
    for coin in coins:
        t = coin["ticker"]
        coin_dir = ENDPOINTS_DIR / t
        if not coin_dir.is_dir():
            logger.warning(f"No config directory for {t} at {coin_dir} — run generate_coin_configs.py first")
            continue
        config_files = sorted(coin_dir.glob("*.json"))
        if not config_files:
            logger.warning(f"No config files in {coin_dir}")
            continue
        coin_configs.append((coin, config_files))

    if args.dry_run:
        print(f"\nWould backfill {len(coin_configs)} coins:\n")
        for coin, files in coin_configs:
            print(f"  {coin['ticker'].upper()} ({coin['display']}):")
            for f in files:
                print(f"    - {f.name}")
        print(f"\nTotal: {sum(len(f) for _, f in coin_configs)} config files")
        return 0

    # Run backfills
    orchestrator = BackfillOrchestrator(config_dir=str(CONFIG_DIR))
    config_loader = ConfigLoader(str(CONFIG_DIR))
    total_records = 0
    failed_coins = []

    for coin, config_files in coin_configs:
        t = coin["ticker"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Backfilling {coin['display']} ({t.upper()}) — {len(config_files)} configs")
        logger.info(f"{'='*60}")

        endpoint_configs = []
        for cf in config_files:
            try:
                endpoint_configs.append(config_loader.load_endpoint_config(str(cf)))
            except Exception as e:
                logger.error(f"Failed to load {cf}: {e}")

        if not endpoint_configs:
            logger.error(f"No valid configs for {t}")
            failed_coins.append(t)
            continue

        start = datetime.now()
        results = orchestrator.run_backfill(
            endpoint_configs=endpoint_configs,
            max_workers=args.max_workers,
            validate_only=args.validate_only,
        )
        duration = datetime.now() - start

        records = sum(s.get("records", 0) for s in results.get("provider_stats", {}).values())
        total_records += records
        logger.info(f"{t.upper()}: {results['completed']}/{results['total_endpoints']} completed, "
                     f"{records} records, {duration}")

        if results["failed"] > 0:
            failed_coins.append(t)

    # Summary
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Coins processed: {len(coin_configs)}")
    print(f"Total records: {total_records:,}")
    if failed_coins:
        print(f"Failed: {', '.join(failed_coins)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
