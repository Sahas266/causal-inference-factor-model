"""Delta-neutral funding carry on Hyperliquid perps — build-out + backtest.

The trade: short the perp, hold equal spot — price risk nets out, and the
position COLLECTS funding whenever it is positive (longs pay shorts).
Hyperliquid pays funding hourly at 1/8 of the quoted 8h rate; the warehouse
`funding_rate_8h` series is those hourly payments (median ≈ 11%/yr
annualized, the canonical baseline; means 15–25%/yr by asset, 2023-10 →
2025-12, 20 assets).

Daily income for 1 notional of carry on asset i = sum of that day's hourly
rates. The backtest is in INCOME space: price PnL of the hedged pair is
assumed zero (basis drift and spot/perp tracking error are real but
second-order at daily horizon; flagged in the doc).

Arms (decisions at day t close, applied to day t+1 income):
  always_on      equal weight all assets, always (the naive baseline —
                 collects negative funding too).
  persist_topk   top-K assets by trailing 7d mean funding, only those with
                 positive trailing funding.
  forecast_topk  top-K by a walk-forward per-asset forecast of next-day
                 funding: [own funding today, own 7d mean, cross-asset mean,
                 chain factor innovations] -> linear, train 252d, refit 5d.
                 Hold only if forecast clears the cost hurdle.

Costs: `fee_bps_oneway` charged on BOTH legs' turnover (perp taker ~3.5bp +
spot ~7bp + slippage ≈ 15bp one-way combined, 30bp round trip — deliberately
conservative).

Honesty: the factor-vs-persistence increment gets a circular-shift placebo
on the innovation block; fold breakdown reported.

Run:  python -m causal_portfolio.experiments.funding_carry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import ANNUALIZATION

logger = logging.getLogger("cpcm.experiments.funding_carry")

DUCKDB_PATH = "causal_portfolio/data/cpcm_local.duckdb"


# ── data ────────────────────────────────────────────────────────────

def load_daily_funding(db_path: str = DUCKDB_PATH) -> pd.DataFrame:
    """(days × assets) daily funding income per 1 notional short (sum of the
    day's hourly payments, UTC days)."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    df = con.sql("""
        select cast(time at time zone 'UTC' as date) as day, asset,
               sum(value) as daily_funding, count(*) as n_hours
        from asset_metrics_best
        where metric = 'funding_rate_8h'
        group by 1, 2
    """).df()
    con.close()
    # Drop partial days (< 20 hourly prints) at the edges.
    df = df[df["n_hours"] >= 20]
    wide = df.pivot(index="day", columns="asset", values="daily_funding")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# ── backtest core ───────────────────────────────────────────────────

@dataclass
class CarryResult:
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    avg_gross: float
    ann_turnover_cost: float
    daily: pd.Series
    fold_sharpes: list[float]


def _stats(name: str, pnl: pd.Series, gross: pd.Series,
           cost: pd.Series, n_folds: int = 4) -> CarryResult:
    eq = (1 + pnl).cumprod()
    peak = eq.cummax()
    dd = float(((eq - peak) / peak).min())
    folds = np.array_split(pnl.values, n_folds)
    fold_sh = [float(np.mean(f) / (np.std(f) + 1e-15) * np.sqrt(ANNUALIZATION))
               for f in folds if len(f) > 20]
    return CarryResult(
        name,
        float(pnl.mean() * ANNUALIZATION),
        float(pnl.mean() / (pnl.std() + 1e-15) * np.sqrt(ANNUALIZATION)),
        dd, float(gross.mean()),
        float(cost.mean() * ANNUALIZATION),
        pnl, fold_sh,
    )


def run_carry(
    funding: pd.DataFrame, weights: pd.DataFrame, name: str,
    fee_bps_oneway: float = 15.0,
) -> CarryResult:
    """PnL_t = w_{t-1} · funding_t − |Δw| · fee. Weights already lagged-safe:
    row t is the position DECIDED at t close; income accrues at t+1."""
    w = weights.shift(1).fillna(0.0)
    common = w.index.intersection(funding.index)
    w, f = w.loc[common], funding.loc[common].fillna(0.0)
    income = (w * f).sum(axis=1)
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = dw * (fee_bps_oneway / 1e4)
    pnl = income - cost
    return _stats(name, pnl, w.abs().sum(axis=1), cost)


# ── strategy arms ───────────────────────────────────────────────────

def weights_always_on(funding: pd.DataFrame) -> pd.DataFrame:
    avail = funding.notna()
    w = avail.div(avail.sum(axis=1), axis=0).fillna(0.0)
    return w


def weights_persist_topk(funding: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    sig = funding.rolling(7, min_periods=5).mean()
    w = pd.DataFrame(0.0, index=funding.index, columns=funding.columns)
    for t in range(len(sig)):
        row = sig.iloc[t].dropna()
        row = row[row > 0]
        top = row.nlargest(k)
        if len(top):
            w.iloc[t, [funding.columns.get_loc(c) for c in top.index]] = 1.0 / k
    return w


def forecast_next_funding(
    funding: pd.DataFrame, innov: pd.DataFrame | None, *,
    train_window: int = 252, refit: int = 5,
) -> pd.DataFrame:
    """Walk-forward per-asset linear forecast of next-day funding."""
    mean_f = funding.mean(axis=1).rename("xmean")
    feats_global = [mean_f]
    if innov is not None:
        feats_global.append(innov.reindex(funding.index).ffill(limit=2))
    G = pd.concat(feats_global, axis=1)

    out = pd.DataFrame(np.nan, index=funding.index, columns=funding.columns)
    for asset in funding.columns:
        own = funding[asset]
        X = pd.concat([own.rename("own"),
                       own.rolling(7, min_periods=5).mean().rename("own7"),
                       G], axis=1)
        features = X.dropna()
        labels = own.shift(-1).reindex(features.index)
        if labels.notna().sum() < train_window:
            continue
        beta = None
        last_fit = -refit
        for t, date in enumerate(features.index):
            prior = labels.iloc[:t].dropna().index
            if len(prior) < train_window:
                continue
            if beta is None or t - last_fit >= refit:
                train = prior[-train_window:]
                Xtr = np.column_stack([np.ones(train_window),
                                       features.loc[train].to_numpy()])
                beta, *_ = np.linalg.lstsq(Xtr, labels.loc[train].to_numpy(),
                                           rcond=None)
                last_fit = t
            out.loc[date, asset] = float(
                np.concatenate([[1.0], features.loc[date].to_numpy()]) @ beta
            )
    return out


def weights_forecast_topk(
    forecast: pd.DataFrame, k: int = 5, hurdle: float = 0.0,
) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=forecast.index, columns=forecast.columns)
    for t in range(len(forecast)):
        row = forecast.iloc[t].dropna()
        row = row[row > hurdle]
        top = row.nlargest(k)
        if len(top):
            w.iloc[t, [forecast.columns.get_loc(c) for c in top.index]] = 1.0 / k
    return w


def common_forecast_start(*forecasts: pd.DataFrame) -> pd.Timestamp:
    """First date on which every forecast arm can make an OOS decision."""
    available = pd.concat(
        [forecast.notna().any(axis=1) for forecast in forecasts], axis=1
    ).all(axis=1)
    if not available.any():
        raise ValueError("forecast arms have no common out-of-sample date")
    return pd.Timestamp(available.index[available][0])


# ── report ──────────────────────────────────────────────────────────

def render_markdown(results: list[CarryResult], placebo_p: float | None,
                    args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Delta-neutral funding carry (Hyperliquid)\n"]
    L.append("Short perp + long spot, collecting hourly funding. Income-space "
             "backtest: hedged price PnL assumed zero (basis drift and "
             "execution tracking are real second-order risks, not modeled). "
             f"Costs: {args['fee_bps_oneway']:.0f}bp one-way on combined-leg "
             "turnover. Decisions at t close earn day t+1 funding.\n")
    L.append(f"- Universe: {args['n_assets']} HL perps | window "
             f"{args['lo']} → {args['hi']} | top-K={args['k']}")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Arm | Ann. return | Sharpe | Max DD | avg gross | "
             "ann. cost drag | fold Sharpes |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r in results:
        L.append(f"| {r.name} | {r.ann_return:+.1%} | {r.sharpe:.2f} | "
                 f"{r.max_dd:.1%} | {r.avg_gross:.2f} | "
                 f"{r.ann_turnover_cost:.2%} | "
                 f"{', '.join(f'{s:.1f}' for s in r.fold_sharpes)} |")
    L.append("")

    if placebo_p is not None:
        L.append(f"**Factor-timing placebo** (circular shifts of the "
                 f"innovation block; share of shifts whose forecast arm "
                 f"Sharpe ≥ real): p = {placebo_p:.2f}\n")

    L.append("## Notes / risks not in the income backtest\n")
    L.append("- Spot leg assumed available at perp size (HL spot or external "
             "venue); borrow/withdrawal frictions ignored.")
    L.append("- Basis risk: spot-perp spread moves; severe in squeezes.")
    L.append("- Funding can flip intraday; the daily model reacts next day.")
    L.append("- Capacity: fine at personal size on majors; thin on the tail "
             "assets.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--fee-bps-oneway", type=float, default=15.0)
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--n-placebo", type=int, default=50)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument(
        "--out", default="causal_portfolio/docs/funding_carry_generated.md"
    )
    args = p.parse_args()

    funding = load_daily_funding().loc[args.start:args.end]
    logger.info("funding: %d days x %d assets (%s -> %s)", len(funding),
                funding.shape[1], funding.index[0].date(),
                funding.index[-1].date())

    # Factor innovations (chain factors) aligned to funding days.
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
        innovation_factors,
    )
    assets9 = "btc,eth,sol,bnb,avax,uni,aave,link,doge".split(",")
    loader = get_loader()
    panel = loader.load_panel(list(dict.fromkeys(assets9 + FACTOR_SOURCE_ASSETS)),
                              PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    factors = build_all_factors(panel, macro).dropna(axis=1, how="all")
    chain = [c for c in ("chain_congestion", "mev_pressure", "staking_yield",
                         "liq_flow", "stable_flow") if c in factors.columns]
    innov = innovation_factors(factors[chain])
    innov.index = pd.to_datetime(innov.index)

    fc = forecast_next_funding(funding, innov, train_window=args.train_window)
    fc_nofac = forecast_next_funding(funding, None,
                                     train_window=args.train_window)
    oos_start = common_forecast_start(fc, fc_nofac)
    oos_funding = funding.loc[oos_start:]
    fee = args.fee_bps_oneway
    results = [
        run_carry(
            oos_funding,
            weights_always_on(funding).loc[oos_start:],
            "always_on",
            fee,
        ),
        run_carry(
            oos_funding,
            weights_persist_topk(funding, args.k).loc[oos_start:],
            f"persist_top{args.k}",
            fee,
        ),
        run_carry(
            oos_funding,
            weights_forecast_topk(fc, args.k).loc[oos_start:],
            f"forecast_top{args.k}",
            fee,
        ),
        run_carry(
            oos_funding,
            weights_forecast_topk(fc_nofac, args.k).loc[oos_start:],
            f"forecast_top{args.k}_nofactors",
            fee,
        ),
    ]

    for r in results:
        logger.info("%s: ann=%.1f%% sharpe=%.2f dd=%.1f%%",
                    r.name, r.ann_return * 100, r.sharpe, r.max_dd * 100)

    # Placebo on the factor increment
    real = results[2].sharpe
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(args.n_placebo):
        kshift = int(rng.integers(60, len(innov) - 60))
        shifted = pd.DataFrame(np.roll(innov.values, kshift, axis=0),
                               index=innov.index, columns=innov.columns)
        fcs = forecast_next_funding(funding, shifted,
                                    train_window=args.train_window)
        rs = run_carry(oos_funding, weights_forecast_topk(fcs, args.k).loc[oos_start:],
                       "placebo", fee)
        hits += rs.sharpe >= real
    placebo_p = hits / args.n_placebo
    logger.info("factor-timing placebo p=%.2f", placebo_p)

    md = render_markdown(results, placebo_p, {
        **vars(args), "n_assets": funding.shape[1],
        "lo": oos_funding.index[0].date(), "hi": oos_funding.index[-1].date(),
    })
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
