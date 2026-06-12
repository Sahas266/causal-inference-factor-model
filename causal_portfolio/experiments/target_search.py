"""Search over PREDICTION TARGETS: spreads, vol, funding, flows, tails.

Daily aggregate returns were the hardest possible target and the factors
failed it. This asks where the factors (as AR(1) innovations) DO carry
out-of-sample information, scoring each target as INCREMENTAL skill over its
natural persistence baseline — because "vol predicts vol" and "funding
predicts funding" are free, and a factor only matters if it adds to them.

Targets (all y at t+1, predictors known at t):
  eth_btc_spread  y = r_eth - r_btc.   X = ETH-chain factor innovations
                  (congestion/mev/staking/liq/stable). Baseline: AR(1) of
                  the spread. The chain-specific factors SHOULD show here
                  if anywhere — the spread differences out the market.
  ew_fwd_absret   y = |r_ew| (next-day vol proxy, non-overlapping).
                  X = innovations + vix innovation. Baseline: trailing 5d
                  vol + today's |r|.
  funding_next    y = cross-asset mean funding_rate_8h. X = innovations.
                  Baseline: today's funding (persistence — very strong).
  liq_flow_next   y = liq_flow factor. X = ew_return, btc_return at t.
                  Baseline: AR(1) of liq_flow. This is the DAG v2 stable
                  edge (returns -> flows) tested PREDICTIVELY out of sample.
  tail_next       y = 1(r_ew below its trailing 5th pctile). X = innovations
                  + vix innovation. Baseline: trailing 5d vol. (Linear
                  probability model — crude but comparable.)

Scoring: walk-forward (train 252, refit every 5d, predict 1 step). Skill =
OOS R^2 of full model minus OOS R^2 of baseline (R^2 vs the baseline's own
unconditional mean). Positive skill that survives a circular-shift placebo
of the NON-baseline predictors = the factors add real information.

Run:  python -m causal_portfolio.experiments.target_search
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.experiments.target_search")


# ── walk-forward incremental skill ──────────────────────────────────

def _walk_forward_pred(y: np.ndarray, X: np.ndarray, train_window: int,
                       refit: int = 5) -> np.ndarray:
    """1-step-ahead OOS predictions from a linear model, refit every `refit`."""
    T = len(y)
    pred = np.full(T, np.nan)
    beta = None
    for t in range(train_window, T):
        if beta is None or (t - train_window) % refit == 0:
            Xtr = np.column_stack([np.ones(train_window),
                                   X[t - train_window:t]])
            beta, *_ = np.linalg.lstsq(Xtr, y[t - train_window:t], rcond=None)
        pred[t] = float(np.concatenate([[1.0], X[t]]) @ beta)
    return pred


def _oos_r2(y: np.ndarray, pred: np.ndarray, train_window: int) -> float:
    """R^2 of OOS predictions vs the expanding-mean baseline."""
    sl = slice(train_window, None)
    yy, pp = y[sl], pred[sl]
    ok = ~np.isnan(pp)
    yy, pp = yy[ok], pp[ok]
    denom = np.sum((yy - yy.mean()) ** 2)
    return float(1 - np.sum((yy - pp) ** 2) / (denom + 1e-30))


@dataclass
class TargetResult:
    name: str
    n_oos: int
    r2_base: float
    r2_full: float
    skill: float           # r2_full - r2_base
    placebo_p: float | None = None


def score_target(
    y: pd.Series, X_base: pd.DataFrame, X_extra: pd.DataFrame, *,
    train_window: int = 252, n_placebo: int = 50, seed: int = 0,
) -> TargetResult | None:
    """Incremental OOS skill of X_extra over X_base for target y."""
    data = pd.concat([y.rename("y"), X_base, X_extra], axis=1).dropna()
    if len(data) < train_window + 120:
        return None
    yv = data["y"].values
    Xb = data[X_base.columns].values
    Xf = data[list(X_base.columns) + list(X_extra.columns)].values

    r2_b = _oos_r2(yv, _walk_forward_pred(yv, Xb, train_window), train_window)
    r2_f = _oos_r2(yv, _walk_forward_pred(yv, Xf, train_window), train_window)
    skill = r2_f - r2_b

    placebo_p = None
    if skill > 0:
        rng = np.random.default_rng(seed)
        hits = 0
        E = data[X_extra.columns].values
        for _ in range(n_placebo):
            k = int(rng.integers(60, len(data) - 60))
            Xs = np.column_stack([Xb, np.roll(E, k, axis=0)])
            r2_s = _oos_r2(yv, _walk_forward_pred(yv, Xs, train_window),
                           train_window)
            hits += (r2_s - r2_b) >= skill
        placebo_p = hits / n_placebo
    return TargetResult("", len(data) - train_window, r2_b, r2_f, skill,
                        placebo_p)


# ── report ──────────────────────────────────────────────────────────

def render_markdown(rows: list[TargetResult], args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Prediction-target search: spreads, vol, funding, flows, tails\n"]
    L.append("Walk-forward 1-step OOS predictions (train 252d, refit 5d). "
             "Skill = OOS R² of (baseline + factor innovations) minus OOS R² "
             "of the persistence baseline alone. Placebo: circular shifts of "
             "the factor block only (baseline kept intact), p = share of "
             "shifts with ≥ the real skill.\n")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")
    L.append("| Target | OOS days | R² baseline | R² +factors | skill | "
             "placebo p |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        p = f"{r.placebo_p:.2f}" if r.placebo_p is not None else "—"
        L.append(f"| {r.name} | {r.n_oos} | {r.r2_base:.4f} | "
                 f"{r.r2_full:.4f} | {r.skill:+.4f} | {p} |")
    L.append("")

    winners = [r for r in rows
               if r.skill > 0 and r.placebo_p is not None
               and r.placebo_p < 0.05]
    L.append("## Verdict\n")
    if winners:
        for r in winners:
            L.append(f"- **{r.name}**: the factor innovations add "
                     f"{r.skill:+.4f} OOS R² over the persistence baseline "
                     f"(placebo p={r.placebo_p:.2f}).")
    else:
        L.append("**No target gains placebo-robust skill from the factor "
                 "innovations beyond its persistence baseline.** Where R² "
                 "is high, persistence earns it; the factors do not add.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
        innovation_factors,
    )

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--n-placebo", type=int, default=50)
    p.add_argument("--out", default="causal_portfolio/docs/target_search.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro).dropna(axis=1, how="all")
    innov = innovation_factors(factors)

    ew = returns.mean(axis=1)
    btc = returns["btc_return"]
    eth = returns["eth_return"]
    fund_cols = [c for c in panel.columns if c.endswith("_funding_rate_8h")]
    funding = panel[fund_cols].mean(axis=1) if fund_cols else None

    eth_chain = [c for c in ("chain_congestion", "mev_pressure",
                             "staking_yield", "liq_flow", "stable_flow")
                 if c in innov.columns]
    vix_in = [c for c in ("vixcls",) if c in innov.columns]

    specs = []
    # 1. ETH-BTC spread
    spread = (eth - btc).rename("spread")
    specs.append(("eth_btc_spread", spread.shift(-1),
                  spread.to_frame("spread_lag"), innov[eth_chain]))
    # 2. next-day |EW return| (vol proxy)
    absret = ew.abs()
    base_vol = pd.concat([ew.rolling(5).std().rename("trail5"),
                          absret.rename("abs_today")], axis=1)
    specs.append(("ew_fwd_absret", absret.shift(-1), base_vol,
                  innov[[c for c in innov.columns
                         if c in eth_chain + vix_in + ["funding_basis",
                                                       "cex_dex_flow"]]]))
    # 3. next-day funding
    if funding is not None:
        specs.append(("funding_next", funding.shift(-1),
                      funding.to_frame("funding_today"), innov[eth_chain]))
    # 4. liq_flow from returns (the DAG v2 stable edge, predictively)
    if "liq_flow" in factors.columns:
        lf = factors["liq_flow"]
        rets = pd.concat([ew.rename("ew_ret"), btc.rename("btc_ret")], axis=1)
        specs.append(("liq_flow_next", lf.shift(-1),
                      lf.to_frame("liq_flow_today"), rets))
    # 5. tail probability (linear probability model)
    q = ew.expanding(120).quantile(0.05)
    tail = (ew < q).astype(float)
    specs.append(("tail_next", tail.shift(-1),
                  ew.rolling(5).std().to_frame("trail5"),
                  innov[[c for c in innov.columns
                         if c in eth_chain + vix_in]]))

    rows = []
    for name, y, Xb, Xe in specs:
        res = score_target(y, Xb, Xe, train_window=args.train_window,
                           n_placebo=args.n_placebo)
        if res is None:
            logger.info("%s: too few rows", name)
            continue
        res.name = name
        logger.info("%s: base R2=%.4f full R2=%.4f skill=%+.4f p=%s",
                    name, res.r2_base, res.r2_full, res.skill,
                    f"{res.placebo_p:.2f}" if res.placebo_p is not None else "—")
        rows.append(res)

    md = render_markdown(rows, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
