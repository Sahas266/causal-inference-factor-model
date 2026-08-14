"""Compute the derived metrics the committed script no longer covers.

`compute_derived_metrics.py` only produces realized volatility and the
DEX/CEX ratio, but the warehouse also holds log_return_7d/30d, sharpe_30d,
beta_to_btc_30d, volume_mcap_ratio, tvl_net_flow_usd and
stablecoin_net_flow_usd. Those came from a fuller version that is not in the
tree, so their definitions had to be recovered from the stored series rather
than assumed.

Each formula below was reverse-engineered by reproducing a known historical
value exactly, so the continuation cannot silently diverge from the history
it is appended to (verified against btc, 2025-12-30):

    log_return_7d    log(P_t / P_{t-7})  * 100, CoinMetrics price   1.2088
    log_return_30d   log(P_t / P_{t-30}) * 100, CoinMetrics price  -2.4345
    sharpe_30d       mean/stdev(ddof=1) of 30 daily log returns
                     * sqrt(365), CoinMetrics price                -0.7738
    tvl_net_flow_usd tvl_usd(t) - tvl_usd(t-1)                -51,014,710.00

`volume_mcap_ratio` is deliberately NOT computed: it is 24h_volume / mc, and
both inputs are Artemis-only. Artemis is frozen at 2026-01-01 and its provider
source is not in the tree, so any continuation would have to substitute
different inputs (spot_volume_usd_24h / market_cap_usd give a materially
different number) and would quietly change the series' meaning.

Usage:
    python scripts/compute_derived_extra_local.py --dry-run
    python scripts/compute_derived_extra_local.py
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("derived.extra")

PRICE_PROVIDER_ORDER = ("coinmetrics", "coingecko", "artemis")
ANNUALISATION = 365
DERIVED_PRIORITY = 99


def _connect(db_path, read_only=True):
    import duckdb

    from causal_portfolio.data import DEFAULT_LOCAL_DB

    con = duckdb.connect(str(db_path or DEFAULT_LOCAL_DB), read_only=read_only)
    # Rows are stored at UTC midnight; without this the session renders them
    # in local time and `time::date` lands on the previous day, silently
    # shifting every derived series by one day against its own history.
    con.execute("SET TimeZone='UTC'")
    return con


def _series(con, asset: str, metric: str, provider: str | None = None):
    """Daily (date, value) for one asset/metric, newest last, one row per day."""
    query = [
        "select time::date d, value from asset_metrics",
        "where asset = ? and metric = ?",
    ]
    params: list = [asset, metric]
    if provider:
        query.append("and provider = ?")
        params.append(provider)
    query.append("qualify row_number() over (partition by time::date "
                 "order by provider_priority) = 1")
    query.append("order by d")
    rows = con.execute(" ".join(query), params).fetchall()
    return [(str(d), float(v)) for d, v in rows if v is not None]


def _price_series(con, asset: str):
    """Prefer the provider the historical series was computed from.

    Falls back only when that provider is too short to support a 30-day
    window, so the continuation matches the history wherever it can.
    """
    for provider in PRICE_PROVIDER_ORDER:
        series = _series(con, asset, "price", provider)
        if len(series) >= 31:
            return series, provider
    return [], "none"


def _row(asset, metric, day, value):
    return {
        "provider": "derived",
        "provider_priority": DERIVED_PRIORITY,
        "asset": asset,
        "metric": metric,
        "time": f"{day}T00:00:00+00:00",
        "value": float(value),
        "frequency": "1d",
        "metadata": None,
    }


def log_returns_and_sharpe(con, asset: str) -> list[dict]:
    series, provider = _price_series(con, asset)
    if len(series) < 31:
        return []
    days = [d for d, _ in series]
    prices = [p for _, p in series]
    daily = [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i] > 0 and prices[i - 1] > 0
    ]
    out: list[dict] = []
    for window, metric in ((7, "log_return_7d"), (30, "log_return_30d")):
        for i in range(window, len(prices)):
            if prices[i] > 0 and prices[i - window] > 0:
                out.append(_row(asset, metric, days[i],
                                math.log(prices[i] / prices[i - window]) * 100))
    # daily[i] is the return ending on days[i + 1]
    for i in range(30, len(daily) + 1):
        window = daily[i - 30:i]
        spread = statistics.stdev(window)          # ddof=1, matches history
        if spread > 0:
            out.append(_row(asset, "sharpe_30d", days[i],
                            statistics.mean(window) / spread * math.sqrt(ANNUALISATION)))
    logger.info("  %-6s log-returns/sharpe from %s (%d rows)", asset, provider, len(out))
    return out


def net_flow(con, asset: str, source_metric: str, metric: str) -> list[dict]:
    """Day-over-day change in a stock series, e.g. TVL or stablecoin supply."""
    series = _series(con, asset, source_metric)
    out = []
    for i in range(1, len(series)):
        out.append(_row(asset, metric, series[i][0], series[i][1] - series[i - 1][1]))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default=None, help="comma-separated tickers")
    p.add_argument("--db-path", default=None)
    p.add_argument("--dry-run", action="store_true", help="compute but do not write")
    args = p.parse_args(argv)

    con = _connect(args.db_path, read_only=True)
    try:
        if args.asset:
            assets = [a.strip().lower() for a in args.asset.split(",") if a.strip()]
        else:
            assets = [
                a for (a,) in con.execute(
                    "select distinct asset from asset_metrics where metric='price' order by asset"
                ).fetchall()
            ]
        logger.info("computing extra derived metrics for %d asset(s)", len(assets))

        records: list[dict] = []
        for asset in assets:
            try:
                records.extend(log_returns_and_sharpe(con, asset))
                records.extend(net_flow(con, asset, "tvl_usd", "tvl_net_flow_usd"))
                records.extend(
                    net_flow(con, asset, "stablecoin_supply_usd", "stablecoin_net_flow_usd")
                )
            except Exception as exc:
                logger.error("%s failed: %s", asset, exc)
    finally:
        con.close()

    by_metric: dict[str, int] = {}
    for r in records:
        by_metric[r["metric"]] = by_metric.get(r["metric"], 0) + 1
    for metric in sorted(by_metric):
        logger.info("  %-26s %6d records", metric, by_metric[metric])

    if args.dry_run:
        logger.info("dry run: %d records not written", len(records))
        return 0
    if not records:
        logger.warning("nothing to write")
        return 0

    from causal_portfolio.data import DEFAULT_LOCAL_DB
    from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

    loader = DuckDBCPCMDataLoader(
        db_path=str(args.db_path or DEFAULT_LOCAL_DB), read_only=False
    )
    try:
        written = loader.upsert_rows(records)
    finally:
        loader.close()
    logger.info("upserted %d rows", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
