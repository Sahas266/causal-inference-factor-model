"""Daily local RP-PCA target generation and optional Hyperliquid execution."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from causal_portfolio.execution import execute_target as execute_model_target
from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.run_logging import execution_run_log
from causal_portfolio.execution.targets import load_target_snapshot

logger = logging.getLogger("cpcm.execution.rppca_daily")

DEFAULT_ASSETS = ["btc", "eth", "sol", "bnb", "avax", "uni", "aave", "link", "doge"]
COINGECKO_IDS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "avax": "avalanche-2",
    "uni": "uniswap",
    "aave": "aave",
    "link": "chainlink",
    "doge": "dogecoin",
}
TIME_COLUMNS = ("date", "time", "ts", "timestamp")


@dataclass(frozen=True)
class RPPCAResult:
    weights: dict[str, float]
    last_data_date: pd.Timestamp
    generated_at: datetime
    gamma_used: float
    n_observations: int
    target_path: Path


def refresh_local_prices(
    assets: list[str],
    *,
    db_path: str | Path | None = None,
    now: datetime | None = None,
) -> int:
    """Append one current CoinGecko price snapshot to the local DuckDB."""
    mapped = {asset: COINGECKO_IDS[asset] for asset in assets if asset in COINGECKO_IDS}
    if not mapped:
        raise ValueError("none of the requested assets have a CoinGecko mapping")
    response = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": ",".join(mapped.values()), "vs_currencies": "usd"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    missing = [asset for asset, coin_id in mapped.items() if not payload.get(coin_id, {}).get("usd")]
    if missing:
        raise ValueError(f"CoinGecko response missing prices: {', '.join(missing)}")

    from causal_portfolio.data import DEFAULT_LOCAL_DB
    from causal_portfolio.data.duckdb_loader import DuckDBCPCMDataLoader

    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target_db = Path(db_path or os.environ.get("CPCM_LOCAL_DB") or DEFAULT_LOCAL_DB)
    rows = [
        {
            "provider": "coingecko",
            "provider_priority": 3,
            "asset": asset,
            "metric": "price",
            "time": timestamp,
            "value": float(payload[coin_id]["usd"]),
            "frequency": "1d",
            "metadata": json.dumps({"fetched_utc": timestamp.isoformat()}),
        }
        for asset, coin_id in mapped.items()
    ]
    with DuckDBCPCMDataLoader(db_path=str(target_db), read_only=False) as loader:
        return loader.upsert_rows(rows)


def _load_csv_prices(path: Path, assets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in TIME_COLUMNS), None)
    if time_col is None:
        raise ValueError(f"{path} needs one timestamp column: {', '.join(TIME_COLUMNS)}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    df = df.set_index(time_col).sort_index()
    df.columns = [c.lower().removeprefix("weight_").removesuffix("_price") for c in df.columns]
    missing = [a for a in assets if a not in df.columns]
    if missing:
        raise ValueError(f"price CSV missing assets: {', '.join(missing)}")
    return df[assets].resample("D").last()


def load_prices(assets: list[str], start: str, end: str, prices_csv: Path | None) -> pd.DataFrame:
    if prices_csv:
        return _load_csv_prices(prices_csv, assets).loc[start:end]
    from causal_portfolio.data import get_loader

    return get_loader().load_prices(assets, start, end, use_cache=False)


def forward_fill_prices(
    prices: pd.DataFrame,
    *,
    end: str,
    max_ffill_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if prices.empty:
        raise ValueError("no price data loaded")
    daily = prices.sort_index().resample("D").last()
    last_data_date = daily.dropna(how="all").index.max()
    end_date = pd.Timestamp(end).tz_localize(None).normalize()
    full_index = pd.date_range(daily.index.min().normalize(), end_date, freq="D")
    limit = None if max_ffill_days < 0 else max_ffill_days
    filled = daily.reindex(full_index).ffill(limit=limit)
    missing = filled.columns[filled.iloc[-1].isna()].tolist()
    if missing:
        raise ValueError(
            "latest row still has missing prices after forward fill: "
            f"{', '.join(missing)}; update data or increase --max-ffill-days"
        )
    return filled, last_data_date


def _winsorize(returns: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if lower <= 0 and upper >= 1:
        return returns
    return returns.clip(returns.quantile(lower), returns.quantile(upper), axis=1)


def rppca_weights(
    prices: pd.DataFrame,
    *,
    cov_window: int = 252,
    mean_window: int = 63,
    n_components: int = 5,
    gamma: float | None = None,
    risk_free_rate: float = 0.05,
    target_gross: float = 1.0,
    max_abs_weight: float = 0.30,
    long_only: bool = False,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
) -> tuple[dict[str, float], float, int]:
    returns = _winsorize(prices.pct_change().dropna(how="any"), winsorize_lower, winsorize_upper)
    if len(returns) < max(mean_window, n_components + 2):
        raise ValueError(f"not enough return rows for RP-PCA: {len(returns)}")

    cov_returns = returns.tail(cov_window)
    mean_returns = returns.tail(mean_window)
    x = cov_returns.to_numpy(dtype=float)
    mu = mean_returns.mean().to_numpy(dtype=float)
    sigma = np.cov(x, rowvar=False)
    if sigma.ndim == 0:
        raise ValueError("RP-PCA needs at least two assets")

    k = min(n_components, x.shape[1])
    gamma_used = float(len(cov_returns) if gamma is None else gamma)
    composite = sigma + gamma_used * np.outer(mu, mu)
    eigenvalues, eigenvectors = np.linalg.eigh(composite)
    loadings = eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]

    factors = x @ loadings
    mu_f = factors.mean(axis=0) - risk_free_rate / 252.0
    sigma_f = np.cov(factors, rowvar=False)
    sigma_f = np.atleast_2d(sigma_f) + 1e-8 * np.eye(k)
    try:
        factor_w = np.linalg.solve(sigma_f, mu_f)
    except np.linalg.LinAlgError:
        factor_w = np.linalg.pinv(sigma_f) @ mu_f

    denom = factor_w.sum()
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        factor_w = np.ones(k) / k
    else:
        factor_w = factor_w / denom

    asset_w = np.nan_to_num(loadings @ factor_w)
    if long_only:
        asset_w = np.maximum(asset_w, 0.0)
    gross = float(np.abs(asset_w).sum())
    if gross <= 1e-12:
        asset_w = np.ones(len(asset_w)) / len(asset_w)
        gross = 1.0
    asset_w = asset_w / gross * target_gross
    if max_abs_weight > 0:
        asset_w = np.clip(asset_w, -max_abs_weight, max_abs_weight)

    weights = {
        asset: float(weight)
        for asset, weight in zip(prices.columns, asset_w)
        if math.isfinite(float(weight)) and abs(float(weight)) > 1e-8
    }
    return weights, gamma_used, len(cov_returns)


def write_target(
    weights: dict[str, float],
    *,
    target_path: Path,
    rebalance_date: pd.Timestamp,
    generated_at: datetime,
    metadata: dict,
) -> None:
    payload = dict(weights)
    payload["_meta"] = {
        **metadata,
        "strategy": "RP-PCA daily tangency",
        "rebalance_date": rebalance_date.date().isoformat(),
        "generated_utc": generated_at.isoformat(timespec="seconds"),
        "format_version": 1,
        "gross_exposure": float(sum(abs(v) for v in weights.values())),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_once(args: argparse.Namespace) -> RPPCAResult:
    assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]
    now = datetime.now(timezone.utc)
    end = args.end or now.date().isoformat()
    if args.refresh_prices and not args.prices_csv:
        try:
            count = refresh_local_prices(assets, now=now)
            logger.info("refreshed %d local CoinGecko prices", count)
        except Exception as exc:
            logger.warning("local price refresh failed; using existing data: %s", exc)
    prices = load_prices(assets, args.start, end, Path(args.prices_csv) if args.prices_csv else None)
    prices, last_data_date = forward_fill_prices(
        prices,
        end=end,
        max_ffill_days=args.max_ffill_days,
    )
    weights, gamma_used, n_obs = rppca_weights(
        prices,
        cov_window=args.cov_window,
        mean_window=args.mean_window,
        n_components=args.n_components,
        gamma=args.gamma,
        risk_free_rate=args.risk_free_rate,
        target_gross=args.target_gross,
        max_abs_weight=args.max_abs_weight,
        long_only=args.long_only,
    )
    target_path = Path(args.target_out)
    write_target(
        weights,
        target_path=target_path,
        rebalance_date=pd.Timestamp(end),
        generated_at=now,
        metadata={
            "assets": assets,
            "cov_window": args.cov_window,
            "mean_window": args.mean_window,
            "n_components": args.n_components,
            "gamma_used": gamma_used,
            "last_data_date": last_data_date.date().isoformat(),
            "price_forward_filled_to": end,
            "max_ffill_days": args.max_ffill_days,
            "source": str(args.prices_csv or "causal_portfolio.data.get_loader"),
            "expected_next_rebalance": (
                now + timedelta(hours=args.every_hours)
            ).isoformat(),
        },
    )
    logger.info("wrote RP-PCA target to %s", target_path)
    from causal_portfolio.execution import notify
    target = load_target_snapshot(target_path)
    expected_next = now + timedelta(hours=args.every_hours)
    try:
        from causal_portfolio.execution import trace

        trace.start_cycle(
            target,
            expected_next_rebalance=expected_next,
            model_prices={
                asset: float(price) for asset, price in prices.iloc[-1].items()
            },
            price_source=str(args.prices_csv or "causal_portfolio.data.get_loader"),
        )
    except Exception:
        logger.exception("RP-PCA trace initialization failed (continuing)")
    weight_lines = [
        f"{'🟢' if w >= 0 else '🔴'} {notify._esc(a)}: <b>{w:+.4f}</b>"
        for a, w in sorted(weights.items())
    ]
    notify.send(
        f"📈 <b>RP-PCA Target</b> — {end} (data through "
        f"{last_data_date.date().isoformat()})\n"
        f"{notify.format_next_rebalance(expected_next)}\n\n" + "\n".join(weight_lines),
        parse_mode="HTML",
    )
    if args.execute:
        _submit_target(args, target_path)
    return RPPCAResult(weights, last_data_date, now, gamma_used, n_obs, target_path)


def _submit_target(args: argparse.Namespace, target_path: Path) -> None:
    if args.mainnet and not args.ack_mainnet:
        raise PermissionError("mainnet execution requires --ack-mainnet")

    cfg = ExecutionConfig(
        testnet=not args.mainnet,
        dry_run=False,
        max_signal_age_hours=args.max_signal_age_hours,
        allow_stale_signal=args.allow_stale_signal,
        twap_minutes=args.twap_minutes,
        twap_slices=args.twap_slices,
        max_transaction_cost_bps=args.max_transaction_cost_bps,
        estimated_taker_fee_bps=args.estimated_taker_fee_bps,
    )
    target = load_target_snapshot(target_path)
    result = execute_model_target(
        target,
        cfg,
        acknowledge_mainnet=args.mainnet,
    )
    if result.error:
        raise RuntimeError(result.error)
    if result.post_submit_error:
        logger.warning("submission completed but post-submit checks failed: %s",
                       result.post_submit_error)
    if result.audit_error:
        logger.error("submission completed but audit append failed: %s",
                     result.audit_error)
    if not result.submitted:
        logger.info("RP-PCA rebalance no-op: no orders submitted")
        return
    logger.info("submitted RP-PCA rebalance: %s", result.response)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate daily RP-PCA targets and optionally execute")
    p.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None, help="Defaults to today's UTC date at each run")
    p.add_argument("--prices-csv", default=None, help="Optional wide CSV with date/time and asset price columns")
    p.add_argument("--target-out", default="tmp/rppca_daily_target.json")
    p.add_argument("--cov-window", type=int, default=252)
    p.add_argument("--mean-window", type=int, default=63)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--risk-free-rate", type=float, default=0.05)
    p.add_argument("--target-gross", type=float, default=1.0)
    p.add_argument("--max-abs-weight", type=float, default=0.30)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--max-ffill-days", type=int, default=7, help="-1 allows unlimited forward fill")
    p.add_argument("--refresh-prices", action="store_true", help="Refresh local prices from CoinGecko first")
    p.add_argument("--execute", action="store_true", help="Submit the target to Hyperliquid")
    p.add_argument("--mainnet", action="store_true", help="Use Hyperliquid mainnet; default is testnet")
    p.add_argument("--ack-mainnet", action="store_true", help="Required with --execute --mainnet")
    p.add_argument("--allow-stale-signal", action="store_true")
    p.add_argument("--max-signal-age-hours", type=float, default=72.0)
    p.add_argument("--twap-minutes", type=float, default=0.0)
    p.add_argument("--twap-slices", type=int, default=5)
    p.add_argument("--max-transaction-cost-bps", type=float, default=15.0)
    p.add_argument("--estimated-taker-fee-bps", type=float, default=4.5)
    p.add_argument("--loop", action="store_true", help="Run forever")
    p.add_argument("--every-hours", type=float, default=24.0)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    while True:
        try:
            with execution_run_log("rppca") as run_log:
                logger.info(
                    "starting RP-PCA model: log=%s network=%s execute=%s assets=%s "
                    "target_gross=%.4f target_out=%s",
                    run_log,
                    "mainnet" if args.mainnet else "testnet",
                    args.execute,
                    args.assets,
                    args.target_gross,
                    args.target_out,
                )
                run_once(args)
        except Exception:
            return 1
        if not args.loop:
            return 0
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
