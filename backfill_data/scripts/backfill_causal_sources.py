#!/usr/bin/env python3
"""Refresh the exact CoinMetrics inputs registered by causal DAG v2."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.orchestrator import BackfillOrchestrator
from src.core.utils.config_loader import ConfigLoader
from src.core.utils.logger import setup_logger

CONFIG_DIR = PROJECT_ROOT / "config"
CAUSAL_SOURCE_CONFIGS = {
    "btc": Path("btc/btc_coinmetrics.json"),
    "doge": Path("doge/doge_coinmetrics.json"),
    "eth": Path("eth/eth_asset_metrics_primary.json"),
}


def load_causal_source_configs(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Load and verify the three immutable CoinMetrics fee sources."""
    loader = ConfigLoader(str(config_dir))
    configs: list[dict] = []
    for asset, relative_path in CAUSAL_SOURCE_CONFIGS.items():
        source_path = config_dir / "endpoints" / relative_path
        config = deepcopy(loader.load_endpoint_config(str(source_path)))
        providers = [
            provider
            for provider in config["providers"]
            if provider.get("name") == "coinmetrics" and provider.get("enabled", True)
        ]
        if len(providers) != 1:
            raise ValueError(
                f"{relative_path.as_posix()} must have one enabled CoinMetrics provider"
            )
        params = providers[0].get("config", {}).get("params", {})
        metrics = {item.strip() for item in params.get("metrics", "").split(",")}
        if params.get("assets") != asset or "FeeTotNtv" not in metrics:
            raise ValueError(
                f"{relative_path.as_posix()} no longer provides {asset}_FeeTotNtv"
            )
        if config.get("date_range", {}).get("end") != "latest":
            raise ValueError(f"{relative_path.as_posix()} must end at 'latest'")
        config["providers"] = providers
        configs.append(config)
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    logger = setup_logger(level=args.log_level)

    try:
        configs = load_causal_source_configs()
        orchestrator = BackfillOrchestrator(config_dir=str(CONFIG_DIR))
        results = orchestrator.run_backfill(
            configs,
            max_workers=args.max_workers,
            validate_only=args.validate_only,
        )
    except Exception:
        logger.exception("Causal source refresh failed")
        return 1

    expected_jobs = len(configs)
    if args.validate_only:
        if results["skipped"]:
            logger.error("Causal source validation skipped %d config(s)", results["skipped"])
            return 1
        return 0
    if results["failed"] or results["total_endpoints"] != expected_jobs:
        logger.error(
            "Causal source refresh incomplete: jobs=%d/%d failed=%d skipped=%d",
            results["total_endpoints"],
            expected_jobs,
            results["failed"],
            results["skipped"],
        )
        return 1
    logger.info(
        "Causal source refresh complete: completed=%d current=%d",
        results["completed"],
        results["skipped"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
