#!/usr/bin/env python3
"""
Generate endpoint config JSON files for all coins in coin_manifest.json.

Reads the manifest and creates per-asset directories under config/endpoints/
with provider-specific configs based on existing ETH/BTC templates.

Idempotent — safe to re-run (overwrites existing configs).

Usage:
    cd backfill_data
    python scripts/generate_coin_configs.py
    python scripts/generate_coin_configs.py --dry-run
"""

import argparse
import json
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
ENDPOINTS_DIR = CONFIG_DIR / "endpoints"
MANIFEST_PATH = CONFIG_DIR / "coin_manifest.json"


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def write_config(path, config, dry_run=False):
    if dry_run:
        print(f"  [dry-run] Would write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def make_endpoint_config(endpoint_id, description, provider_name, priority, endpoint_type, params, date_range):
    """Base builder for endpoint configs — handles shared boilerplate.

    Note: `priority` here is advisory. The orchestrator stamps the canonical
    PROVIDER_PRIORITY (src/core/orchestrator.py) onto every record at write
    time, so the values below just need to stay consistent with that map.
    """
    return {
        "endpoint_id": endpoint_id,
        "description": description,
        "table": "asset_metrics",
        "primary_keys": ["provider", "asset", "metric", "time"],
        "date_range": date_range,
        "providers": [
            {
                "name": provider_name,
                "enabled": True,
                "priority": priority,
                "config": {
                    "endpoint_type": endpoint_type,
                    "params": params,
                },
            }
        ],
        "fallback_strategy": "use_all",
        "deduplication": {
            "strategy": "prefer_priority",
            "conflict_resolution": "keep_highest_priority",
        },
    }


def make_coingecko_market_chart(coin):
    t = coin["ticker"]
    cg = coin["providers"]["coingecko"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"coingecko_{t}_market_chart",
        description=f"Daily {coin['display']} price, market cap, and volume from CoinGecko",
        provider_name="coingecko",
        priority=3,
        endpoint_type="market_chart",
        params={"schema_type": "market_chart", "coin_id": cg["coin_id"], "vs_currency": "usd", "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_coingecko_snapshot(coin):
    t = coin["ticker"]
    cg = coin["providers"]["coingecko"]
    today = date.today().isoformat()
    tomorrow = date.fromordinal(date.today().toordinal() + 1).isoformat()
    return make_endpoint_config(
        endpoint_id=f"coingecko_{t}_coin_data",
        description=f"{coin['display']} snapshot data from CoinGecko: fdv, supply, ath, atl",
        provider_name="coingecko",
        priority=3,
        endpoint_type="coin_data",
        params={"schema_type": "coin_data", "coin_id": cg["coin_id"], "asset": t},
        date_range={"start": today, "end": tomorrow},
    )


def make_coinmetrics(coin):
    t = coin["ticker"]
    cm = coin["providers"]["coinmetrics"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_asset_metrics_coinmetrics",
        description=f"{coin['display']} core market and on-chain metrics from CoinMetrics",
        provider_name="coinmetrics",
        priority=1,
        endpoint_type="timeseries/asset-metrics",
        params={"assets": cm["asset"], "metrics": cm["metrics"], "frequency": "1d", "page_size": 10000},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_chain_tvl(coin):
    t = coin["ticker"]
    dl = coin["providers"]["defillama_chain"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_chain_tvl_defillama",
        description=f"{coin['display']} historical chain TVL from DefiLlama",
        provider_name="defillama",
        priority=2,
        endpoint_type="chain/tvl",
        params={"schema_type": "chain_tvl", "chain": dl["chain"], "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_protocol_tvl(coin):
    t = coin["ticker"]
    dl = coin["providers"]["defillama_protocol"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_protocol_tvl_defillama",
        description=f"{coin['display']} protocol TVL from DefiLlama",
        provider_name="defillama",
        priority=2,
        endpoint_type="protocol/tvl",
        params={"schema_type": "protocol_tvl", "protocol": dl["protocol"], "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_fees(coin):
    t = coin["ticker"]
    dl = coin["providers"]["defillama_fees"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_fees_defillama",
        description=f"{coin['display']} daily fees from DefiLlama",
        provider_name="defillama",
        priority=2,
        endpoint_type="fees/summary",
        params={"schema_type": "fees", "protocol": dl["protocol"], "data_type": "dailyFees", "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_dex_volume(coin):
    t = coin["ticker"]
    dl = coin["providers"]["defillama_dex_volume"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_dex_volume_defillama",
        description=f"{coin['display']} daily DEX volume from DefiLlama",
        provider_name="defillama",
        priority=2,
        endpoint_type="dex/summary",
        params={"schema_type": "dex_volumes", "protocol": dl["protocol"], "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def make_stablecoin_supply(coin):
    t = coin["ticker"]
    dl = coin["providers"]["defillama_stablecoin"]
    dr = coin["date_range"]
    return make_endpoint_config(
        endpoint_id=f"{t}_stablecoin_supply_defillama",
        description=f"{coin['display']} stablecoin circulating supply from DefiLlama",
        provider_name="defillama",
        priority=2,
        endpoint_type="stablecoin/charts",
        params={"schema_type": "stablecoin_flow", "chain": "all", "stablecoin_id": dl["stablecoin_id"], "asset": t},
        date_range={"start": dr["start"], "end": dr["end"]},
    )


def main():
    parser = argparse.ArgumentParser(description="Generate endpoint configs from coin manifest")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing")
    args = parser.parse_args()

    manifest = load_manifest()
    total = 0
    skipped = 0

    for coin in manifest["coins"]:
        t = coin["ticker"]
        if coin.get("skip"):
            print(f"Skipping {t} (marked skip in manifest)")
            skipped += 1
            continue

        providers = coin["providers"]
        coin_dir = ENDPOINTS_DIR / t

        print(f"\n{coin['display']} ({t}):")

        # CoinGecko market chart — all coins
        if "coingecko" in providers:
            cfg = make_coingecko_market_chart(coin)
            write_config(coin_dir / f"{t}_coingecko_market_chart.json", cfg, args.dry_run)
            total += 1

            cfg = make_coingecko_snapshot(coin)
            write_config(coin_dir / f"{t}_coingecko_snapshot.json", cfg, args.dry_run)
            total += 1

        # CoinMetrics
        if "coinmetrics" in providers:
            cfg = make_coinmetrics(coin)
            write_config(coin_dir / f"{t}_coinmetrics.json", cfg, args.dry_run)
            total += 1

        # DefiLlama chain TVL
        if "defillama_chain" in providers:
            cfg = make_chain_tvl(coin)
            write_config(coin_dir / f"{t}_chain_tvl_defillama.json", cfg, args.dry_run)
            total += 1

        # DefiLlama protocol TVL
        if "defillama_protocol" in providers:
            cfg = make_protocol_tvl(coin)
            write_config(coin_dir / f"{t}_protocol_tvl_defillama.json", cfg, args.dry_run)
            total += 1

        # DefiLlama fees
        if "defillama_fees" in providers:
            cfg = make_fees(coin)
            write_config(coin_dir / f"{t}_fees_defillama.json", cfg, args.dry_run)
            total += 1

        # DefiLlama DEX volume
        if "defillama_dex_volume" in providers:
            cfg = make_dex_volume(coin)
            write_config(coin_dir / f"{t}_dex_volume_defillama.json", cfg, args.dry_run)
            total += 1

        # DefiLlama stablecoin supply
        if "defillama_stablecoin" in providers:
            cfg = make_stablecoin_supply(coin)
            write_config(coin_dir / f"{t}_stablecoin_supply_defillama.json", cfg, args.dry_run)
            total += 1

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Generated {total} config files ({skipped} coins skipped)")


if __name__ == "__main__":
    main()
