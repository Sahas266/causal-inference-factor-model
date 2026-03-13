"""
Compute derived metrics from existing asset_metrics data in Supabase.

Derived metrics are calculated purely from data already backfilled — no external
API calls. Each record is tagged with provider='derived' so it's clear these
are computed, not raw.

Metrics computed:
    - realized_volatility_7d:  7-day rolling annualized volatility from daily log returns
    - realized_volatility_30d: 30-day rolling annualized volatility from daily log returns
        Source: CoinMetrics PriceUSD, fallback to CoinGecko price_usd

    - dex_cex_volume_ratio: Daily DEX volume / CEX netflow (absolute value)
        Source: DefiLlama volume_usd (sum across protocols) / Dune cex_netflow_usd (absolute)
        Note: Only available for ETH (requires Dune CEX data)

Usage:
    cd backfill_data
    python scripts/compute_derived_metrics.py                    # all assets with data
    python scripts/compute_derived_metrics.py --asset sol        # single asset
    python scripts/compute_derived_metrics.py --asset btc,sol    # multiple assets
"""

import argparse
import os
import sys
import math
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from collections import defaultdict

from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
load_dotenv()

from supabase import create_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all(table, filters, select='*', order_col='time'):
    """Paginated fetch from Supabase (1000 rows per page)."""
    all_rows = []
    offset = 0
    while True:
        q = client.table(table).select(select).order(order_col)
        for col, val in filters.items():
            q = q.eq(col, val)
        result = q.range(offset, offset + 999).execute()
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return all_rows


def upsert_batch(records, batch_size=500):
    """Upsert records to asset_metrics in batches."""
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        client.table('asset_metrics').upsert(batch).execute()
        total += len(batch)
    return total


def get_all_assets_with_price():
    """Get distinct assets that have price data in asset_metrics."""
    # Check CoinMetrics PriceUSD
    result = client.table('asset_metrics').select('asset').eq(
        'metric', 'PriceUSD'
    ).eq('provider', 'coinmetrics').limit(1000).execute()
    cm_assets = set(r['asset'] for r in result.data)

    # Check CoinGecko price_usd
    result = client.table('asset_metrics').select('asset').eq(
        'metric', 'price_usd'
    ).eq('provider', 'coingecko').limit(1000).execute()
    cg_assets = set(r['asset'] for r in result.data)

    return sorted(cm_assets | cg_assets)


def fetch_price_data(asset):
    """Fetch price data for an asset, trying CoinMetrics first then CoinGecko."""
    # Try CoinMetrics PriceUSD first
    rows = fetch_all('asset_metrics', {'provider': 'coinmetrics', 'asset': asset, 'metric': 'PriceUSD'})
    source = 'coinmetrics:PriceUSD'

    if len(rows) < 31:
        # Fall back to CoinGecko price_usd
        rows = fetch_all('asset_metrics', {'provider': 'coingecko', 'asset': asset, 'metric': 'price_usd'})
        source = 'coingecko:price_usd'

    return rows, source


def compute_realized_volatility(asset):
    """
    Compute 7-day and 30-day realized volatility for a given asset.

    Formula: RV_N = std(log_returns over N days) * sqrt(365) * 100  (annualized %)

    Source: CoinMetrics PriceUSD, fallback to CoinGecko price_usd
    Output: provider=derived, asset={asset}, metric=realized_volatility_7d / realized_volatility_30d
    """
    logger.info(f"Fetching price data for {asset} realized volatility...")
    rows, source = fetch_price_data(asset)

    if len(rows) < 31:
        logger.warning(f"Only {len(rows)} price rows for {asset} — need at least 31 for 30d RV. Skipping.")
        return []

    # Sort by time, parse to (date_str, price) pairs
    prices = []
    for r in sorted(rows, key=lambda x: x['time']):
        try:
            price = float(r['value'])
            if price > 0:
                prices.append((r['time'], price))
        except (ValueError, TypeError):
            continue

    logger.info(f"Got {len(prices)} valid price observations for {asset} (source: {source})")

    # Compute log returns
    log_returns = []
    for i in range(1, len(prices)):
        lr = math.log(prices[i][1] / prices[i - 1][1])
        log_returns.append((prices[i][0], lr))

    records = []
    sqrt_365 = math.sqrt(365)

    for window, metric_name in [(7, 'realized_volatility_7d'), (30, 'realized_volatility_30d')]:
        for i in range(window - 1, len(log_returns)):
            window_returns = [log_returns[j][1] for j in range(i - window + 1, i + 1)]
            mean = sum(window_returns) / len(window_returns)
            variance = sum((r - mean) ** 2 for r in window_returns) / (len(window_returns) - 1)
            rv = math.sqrt(variance) * sqrt_365 * 100  # annualized %

            records.append({
                'provider': 'derived',
                'asset': asset,
                'metric': metric_name,
                'time': log_returns[i][0],
                'value': str(round(rv, 4)),
                'frequency': '1d',
                'provider_priority': 99,
                'metadata': {'source': source, 'window': window},
            })

    logger.info(f"Computed {len(records)} realized volatility records for {asset}")
    return records


def compute_dex_cex_volume_ratio():
    """
    Compute daily DEX/CEX volume ratio (ETH only).

    Formula: dex_cex_volume_ratio = sum(DEX volume) / abs(CEX netflow)

    Source DEX: provider=defillama, metric=volume_usd (all assets)
    Source CEX: provider=dune, asset=eth, metric=cex_netflow_usd
    Output: provider=derived, asset=eth, metric=dex_cex_volume_ratio
    """
    logger.info("Fetching DefiLlama DEX volumes...")
    dex_rows = fetch_all('asset_metrics', {'provider': 'defillama', 'metric': 'volume_usd'})

    logger.info("Fetching Dune CEX netflow...")
    cex_rows = fetch_all('asset_metrics', {'provider': 'dune', 'metric': 'cex_netflow_usd'})

    if not dex_rows or not cex_rows:
        logger.warning("Missing DEX or CEX data. Skipping ratio calculation.")
        return []

    # Aggregate DEX volume by date (sum across protocols like uniswap, curve)
    dex_by_date = defaultdict(float)
    for r in dex_rows:
        try:
            date_key = r['time'][:10]  # YYYY-MM-DD
            dex_by_date[date_key] += abs(float(r['value']))
        except (ValueError, TypeError):
            continue

    # CEX netflow by date
    cex_by_date = {}
    for r in cex_rows:
        try:
            date_key = r['time'][:10]
            cex_by_date[date_key] = abs(float(r['value']))
        except (ValueError, TypeError):
            continue

    records = []
    common_dates = sorted(set(dex_by_date.keys()) & set(cex_by_date.keys()))

    for date_key in common_dates:
        dex_vol = dex_by_date[date_key]
        cex_flow = cex_by_date[date_key]
        if cex_flow > 0:
            ratio = dex_vol / cex_flow
            # Use midnight UTC timestamp matching the date
            time_iso = f"{date_key}T00:00:00+00:00"
            records.append({
                'provider': 'derived',
                'asset': 'eth',
                'metric': 'dex_cex_volume_ratio',
                'time': time_iso,
                'value': str(round(ratio, 6)),
                'frequency': '1d',
                'provider_priority': 99,
                'metadata': {
                    'source_dex': 'defillama:volume_usd',
                    'source_cex': 'dune:cex_netflow_usd',
                },
            })

    logger.info(f"Computed {len(records)} DEX/CEX volume ratio records")
    return records


def main():
    parser = argparse.ArgumentParser(description="Compute derived metrics from existing Supabase data")
    parser.add_argument("--asset", type=str,
                        help="Comma-separated asset tickers (default: all assets with price data)")
    args = parser.parse_args()

    # Determine which assets to process
    if args.asset:
        assets = [a.strip().lower() for a in args.asset.split(",")]
    else:
        logger.info("Discovering assets with price data...")
        assets = get_all_assets_with_price()
        logger.info(f"Found {len(assets)} assets: {', '.join(assets)}")

    all_records = []

    # 1. Realized volatility for each asset
    for asset in assets:
        rv_records = compute_realized_volatility(asset)
        all_records.extend(rv_records)

    # 2. DEX/CEX volume ratio (ETH only — requires Dune CEX data)
    if not args.asset or 'eth' in assets:
        ratio_records = compute_dex_cex_volume_ratio()
        all_records.extend(ratio_records)

    if not all_records:
        logger.info("No derived metrics to upsert.")
        return

    logger.info(f"Upserting {len(all_records)} derived metric records to asset_metrics...")
    total = upsert_batch(all_records)
    logger.info(f"Done. Upserted {total} records with provider='derived'.")

    # Summary
    from collections import Counter
    metric_counts = Counter((r['asset'], r['metric']) for r in all_records)
    for (asset, metric), count in sorted(metric_counts.items()):
        logger.info(f"  {asset}/{metric}: {count} records")


if __name__ == '__main__':
    main()
