#!/usr/bin/env python3
"""Run backfill jobs for a single provider across matching endpoint configs."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Add project root (backfill_data/) to import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.utils.config_loader import iter_endpoint_config_files

def parse_args(default_provider: str) -> argparse.Namespace:
    """Parse command-line arguments for provider-specific backfill runs."""
    parser = argparse.ArgumentParser(
        description="Run backfill for a single provider across all matching endpoints",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        help=f"Provider name (default: {default_provider})",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Config directory path relative to backfill_data/ (default: config)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate provider endpoint configs without writing data",
    )
    parser.add_argument(
        "--list-endpoints",
        action="store_true",
        help="List endpoint config files that reference the provider and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file path",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON structured logs",
    )
    return parser.parse_args()


def _resolve_config_dir(config_dir_arg: str) -> Path:
    """Resolve config directory path against project root if needed."""
    config_dir = Path(config_dir_arg)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    return config_dir


def _load_all_endpoint_configs(config_dir: Path) -> List[Dict]:
    """Load every endpoint config from config/endpoints."""
    endpoints_dir = config_dir / "endpoints"
    if not endpoints_dir.exists():
        raise FileNotFoundError(f"Endpoints directory not found: {endpoints_dir}")

    configs: List[Dict] = []
    for endpoint_file in iter_endpoint_config_files(endpoints_dir):
        with endpoint_file.open("r", encoding="utf-8") as handle:
            endpoint_config = json.load(handle)
        endpoint_config["_source_file"] = endpoint_file.relative_to(
            endpoints_dir
        ).as_posix()
        configs.append(endpoint_config)
    return configs


def _filter_for_provider(endpoint_config: Dict, provider_name: str) -> Dict | None:
    """Keep only provider entries matching provider_name for one endpoint config."""
    providers = endpoint_config.get("providers", [])
    matching = [entry for entry in providers if entry.get("name") == provider_name]
    if not matching:
        return None

    filtered = deepcopy(endpoint_config)
    filtered["providers"] = matching
    return filtered


def _print_results(results: Dict) -> None:
    """Print run results in a concise, readable summary."""
    print("\n" + "=" * 72)
    print("PROVIDER BACKFILL RESULTS")
    print("=" * 72)
    print(f"Total Endpoint+Provider Jobs: {results['total_endpoints']}")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")

    if results.get("provider_stats"):
        print("\nProvider Stats:")
        for provider, stats in results["provider_stats"].items():
            print(
                f"  - {provider}: "
                f"completed={stats['completed']}, "
                f"failed={stats['failed']}, "
                f"records={stats['records']}"
            )

    print("=" * 72 + "\n")


def main_for_provider(default_provider: str) -> int:
    """Run a provider-targeted backfill session."""
    load_dotenv()
    args = parse_args(default_provider)
    provider_name = args.provider.strip().lower()

    from src.core.utils.logger import setup_logger

    logger = setup_logger(
        level=args.log_level,
        json_format=args.json_logs,
        log_file=args.log_file,
    )

    try:
        config_dir = _resolve_config_dir(args.config_dir)
        endpoint_configs = _load_all_endpoint_configs(config_dir)

        filtered_configs: List[Dict] = []
        source_files: List[str] = []
        for endpoint_config in endpoint_configs:
            filtered = _filter_for_provider(endpoint_config, provider_name)
            if filtered:
                filtered_configs.append(filtered)
                source_files.append(endpoint_config["_source_file"])

        if args.list_endpoints:
            if source_files:
                print(f"\nEndpoints for provider '{provider_name}':")
                for source_file in source_files:
                    print(f"  - {source_file}")
            else:
                print(f"\nNo endpoint configs found for provider '{provider_name}'.")
            print()
            return 0

        if not filtered_configs:
            logger.error(
                "No endpoint configs found for provider '%s' under %s",
                provider_name,
                str(config_dir / "endpoints"),
            )
            return 2

        logger.info(
            "Starting provider run for '%s' with %d endpoint config(s)",
            provider_name,
            len(filtered_configs),
        )
        from src.core.orchestrator import BackfillOrchestrator

        orchestrator = BackfillOrchestrator(config_dir=str(config_dir))
        available_providers = set(orchestrator.list_providers())
        if provider_name not in available_providers:
            logger.error(
                "Provider '%s' is not registered. Available providers: %s",
                provider_name,
                ", ".join(sorted(available_providers)) or "none",
            )
            logger.error(
                "Add adapter code at src/providers/%s/ and config at config/providers/%s.json",
                provider_name,
                provider_name,
            )
            return 3

        start_time = datetime.now()
        results = orchestrator.run_backfill(
            endpoint_configs=filtered_configs,
            max_workers=args.max_workers,
            validate_only=args.validate_only,
        )
        duration = datetime.now() - start_time
        _print_results(results)
        print(f"Duration: {duration}")

        if results["failed"] > 0:
            return 1
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Provider backfill failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main_for_provider(default_provider="coinmetrics"))
