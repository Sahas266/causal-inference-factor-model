#!/usr/bin/env python3
"""Run backfill jobs for CoinGecko endpoints."""

from run_provider_backfill import main_for_provider


if __name__ == "__main__":
    raise SystemExit(main_for_provider("coingecko"))
