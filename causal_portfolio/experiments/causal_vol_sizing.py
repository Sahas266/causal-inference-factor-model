"""Causal vol forecast -> position sizing on the PROFITABLE trend strategy.

`vol_effects.md` established two facts that point in opposite directions:
factor levels (VIX, t10y2y, liq_flow) carry HAC-significant information about
forward 5d realized vol beyond trailing vol (real), but vol-targeting
buy-and-hold BTC failed the circular-shift placebo (any leverage overlay
scored in that window). The untested combination: size the strategy that
actually beats BH on Sharpe — dual_confirm trend (close > 100d MA AND 30d
return > 0, Sharpe 0.95 vs BH 0.63, `strategy_trend_rotation.md`) — with the
causal vol forecast, instead of sizing buy-and-hold.

Pre-registered design (single spec, no sweeps):

- Data: BTC daily returns 2021-2026 (local DuckDB loader); features are
  yesterday's VIX level, t10y2y, and |liq_flow| from `build_all_factors`.
- Target: forward 5d realized vol (std of returns over (t, t+5]) — less
  noisy than |r_{t+1}|, chosen up front.
- Forecasts, refit monthly (21 rows) on trailing 252d windows, training rows
  restricted to those whose forward-vol window is fully known at fit time:
    arm1 (HAR baseline):  fwd5vol ~ 1 + |r_t| + rv5_t + rv22_t
    arm2 (HAR + causal):  arm1 regressors + vix_{t-1}, t10y2y_{t-1},
                          |liq_flow|_{t-1}
- Sizing: exposure_t = gate_t * clip(target_vol / pred_vol, 0, 1), where
  gate_t is the exact dual_confirm signal (reused from
  `market_making/strategy_execution_benchmark.py`) and target_vol is the
  trailing 252d median of rv5. No leverage, cap 1 (house rule).
- Arms: (0) unsized dual_confirm, (1) HAR-sized, (2) HAR+causal-sized.
  All evaluated on the common window where both forecasts exist;
  exposure applied to day t+1's return, 5bp one-way on |dexposure|.
- Placebo: circular-shift the SIZING multiplier (not the trend gate),
  n=200 — does the *timing* of the sizing beat random sizing with the same
  distribution? Reported for arms 1 and 2.
- Forecast comparison: Diebold-Mariano-style HAC regression of the squared
  forecast-error differential (arm1 - arm2) on a constant.

Note on factor construction: `build_all_factors` z-scores macro levels over
the full sample. With an intercept in the OLS, an affine transform of a
regressor leaves predictions unchanged, so no look-ahead enters the
forecasts through that path; the one exception is |liq_flow| (abs of a
full-sample-centered series), whose centering constant is a mild, static
full-sample statistic — flagged in the doc caveats.

Run:  python -m causal_portfolio.experiments.causal_vol_sizing
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import (
    ANNUALIZATION,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
)
from causal_portfolio.market_making.strategy_execution_benchmark import (
    dual_confirm_signal,
)

logger = logging.getLogger("cpcm.experiments.causal_vol_sizing")

VOL_H = 5                 # forecast horizon: 5d realized vol
TRAIN_WINDOW = 252
REFIT_EVERY = 21          # monthly
FEE_BPS_ONEWAY = 5.0
PLACEBO_N = 200
HAR_COLS = ["rv1", "rv5", "rv22"]
CAUSAL_COLS = ["vixcls_lag1", "t10y2y_lag1", "abs_liq_flow_lag1"]


# -- dataset ----------------------------------------------------------

def build_dataset(
    returns: pd.Series,
    features: pd.DataFrame,
    *,
    median_window: int = TRAIN_WINDOW,
) -> pd.DataFrame:
    """One row per day t. Everything except fwd_* uses data <= t only;
    causal features enter as yesterday's value (lag 1)."""
    r = returns.dropna().astype(float)
    px = (1.0 + r).cumprod()
    ds = pd.DataFrame({
        "ret": r,
        "fwd_ret": r.shift(-1),                                   # (t, t+1]
        "fwd_vol": r.shift(-1).rolling(VOL_H).std().shift(-(VOL_H - 1)),
        "gate": dual_confirm_signal(px),                          # data <= t
        "rv1": r.abs(),
        "rv5": r.rolling(VOL_H).std(),
        "rv22": r.rolling(22).std(),
    })
    for col in features.columns:
        ds[f"{col}_lag1"] = features[col].reindex(ds.index).ffill(limit=5).shift(1)
    ds["tgt_vol"] = ds["rv5"].rolling(
        median_window, min_periods=median_window // 2
    ).median()
    return ds


# -- walk-forward vol forecast ---------------------------------------

def walk_forward_forecast(
    ds: pd.DataFrame,
    cols: list[str],
    *,
    train_window: int = TRAIN_WINDOW,
    refit_every: int = REFIT_EVERY,
) -> pd.Series:
    """OLS fwd_vol ~ 1 + cols, refit every `refit_every` rows on the trailing
    `train_window` rows whose forward-vol window is fully realized by t."""
    X_all = np.column_stack([np.ones(len(ds)), ds[cols].values])
    y_all = ds["fwd_vol"].values
    n = len(ds)
    pred = np.full(n, np.nan)
    beta = None
    first = train_window + VOL_H
    for t in range(first, n):
        if (t - first) % refit_every == 0:
            lo, hi = t - VOL_H - train_window, t - VOL_H
            Xw, yw = X_all[lo:hi], y_all[lo:hi]
            ok = ~(np.isnan(Xw).any(axis=1) | np.isnan(yw))
            beta = None
            if ok.sum() >= train_window // 2:
                beta, *_ = np.linalg.lstsq(Xw[ok], yw[ok], rcond=None)
        if beta is not None and not np.isnan(X_all[t]).any():
            pred[t] = max(float(X_all[t] @ beta), 1e-12)
    return pd.Series(pred, index=ds.index, name="pred")


def sizing_multiplier(ds: pd.DataFrame, pred: pd.Series) -> pd.Series:
    """clip(target_vol / predicted_vol, 0, 1) — de-risk only, no leverage."""
    return (ds["tgt_vol"] / pred).clip(0.0, 1.0)


# -- backtest ---------------------------------------------------------

@dataclass
class ArmResult:
    name: str
    ann_return: float
    sharpe: float
    max_dd: float
    calmar: float
    avg_exposure: float
    net: pd.Series


def backtest(fwd_ret: pd.Series, exposure: pd.Series, name: str) -> ArmResult:
    e = exposure.astype(float).clip(0.0, 1.0).fillna(0.0)
    net = e * fwd_ret - e.diff().abs().fillna(e.abs()) * FEE_BPS_ONEWAY / 1e4
    net = net.dropna()
    return ArmResult(
        name=name,
        ann_return=float(net.mean() * ANNUALIZATION),
        sharpe=float(sharpe_ratio(net.values)),
        max_dd=float(max_drawdown(net.values)),
        calmar=float(calmar_ratio(net.values)),
        avg_exposure=float(e.mean()),
        net=net,
    )


def placebo_p(
    fwd_ret: pd.Series, gate: pd.Series, mult: pd.Series,
    real_sharpe: float, *, n: int = PLACEBO_N, seed: int = 0,
) -> float:
    """Circular-shift the sizing multiplier ONLY (gate stays put): fraction
    of random sizings with the same distribution that do at least as well."""
    rng = np.random.default_rng(seed)
    m = mult.values
    hits = 0
    for _ in range(n):
        k = int(rng.integers(30, len(m) - 30))
        shifted = pd.Series(np.roll(m, k), index=mult.index)
        hits += backtest(fwd_ret, gate * shifted, "shift").sharpe >= real_sharpe
    return hits / n


def dm_test(fwd_vol: pd.Series, pred1: pd.Series, pred2: pd.Series) -> dict:
    """Diebold-Mariano-style: HAC(10) t on d_t = err1^2 - err2^2.
    Positive mean d => arm2 (causal) forecasts better."""
    import statsmodels.api as sm

    d = ((fwd_vol - pred1) ** 2 - (fwd_vol - pred2) ** 2).dropna()
    fit = sm.OLS(d.values, np.ones(len(d))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    return {
        "n": len(d),
        "mse1": float(((fwd_vol - pred1) ** 2).dropna().mean()),
        "mse2": float(((fwd_vol - pred2) ** 2).dropna().mean()),
        "mean_d": float(fit.params[0]),
        "t": float(fit.tvalues[0]),
        "p": float(fit.pvalues[0]),
    }


# -- experiment -------------------------------------------------------

def run_experiment(returns: pd.Series, features: pd.DataFrame) -> dict:
    ds = build_dataset(returns, features)
    pred1 = walk_forward_forecast(ds, HAR_COLS)
    pred2 = walk_forward_forecast(ds, HAR_COLS + CAUSAL_COLS)

    # common evaluation window: both forecasts + target vol + next-day return
    common = ds.index[
        pred1.notna() & pred2.notna()
        & ds["tgt_vol"].notna() & ds["fwd_ret"].notna()
    ]
    dse = ds.loc[common]
    mult1 = sizing_multiplier(dse, pred1.loc[common])
    mult2 = sizing_multiplier(dse, pred2.loc[common])

    arms = {
        "unsized_dual_confirm": backtest(dse["fwd_ret"], dse["gate"], "unsized"),
        "har_sized": backtest(dse["fwd_ret"], dse["gate"] * mult1, "har"),
        "har_causal_sized": backtest(dse["fwd_ret"], dse["gate"] * mult2, "har_causal"),
    }
    placebos = {
        "har_sized": placebo_p(
            dse["fwd_ret"], dse["gate"], mult1, arms["har_sized"].sharpe),
        "har_causal_sized": placebo_p(
            dse["fwd_ret"], dse["gate"], mult2, arms["har_causal_sized"].sharpe),
    }
    dm = dm_test(dse["fwd_vol"], pred1.loc[common], pred2.loc[common])
    return {
        "start": str(common[0].date()), "end": str(common[-1].date()),
        "n_days": len(common),
        "pct_gate_on": float(dse["gate"].mean()),
        "arms": arms, "placebos": placebos, "dm": dm,
        "delta_sharpe_causal": arms["har_causal_sized"].sharpe
        - arms["har_sized"].sharpe,
    }


def render(out: dict) -> str:
    lines = [
        f"Window {out['start']} -> {out['end']}  ({out['n_days']} days, "
        f"gate on {out['pct_gate_on']:.0%} of days)",
        "",
        "| arm | ann | Sharpe | maxDD | Calmar | avg exp | placebo p |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for key, r in out["arms"].items():
        p = out["placebos"].get(key)
        lines.append(
            f"| {key} | {r.ann_return:+.1%} | {r.sharpe:.2f} | {r.max_dd:.1%} "
            f"| {r.calmar:.2f} | {r.avg_exposure:.2f} "
            f"| {'-' if p is None else f'{p:.3f}'} |"
        )
    dm = out["dm"]
    lines += [
        "",
        f"causal increment (arm2 - arm1) Sharpe: {out['delta_sharpe_causal']:+.3f}",
        f"forecast MSE  HAR={dm['mse1']:.3e}  HAR+causal={dm['mse2']:.3e}  "
        f"(n={dm['n']})",
        f"DM (err1^2 - err2^2, HAC10): mean={dm['mean_d']:+.3e}  "
        f"t={dm['t']:+.2f}  p={dm['p']:.3f}  (positive = causal better)",
    ]
    return "\n".join(lines)


def load_inputs(start: str, end: str) -> tuple[pd.Series, pd.DataFrame]:
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
    )

    assets = ["btc", "eth", "sol", "bnb", "avax", "uni", "aave", "link", "doge"]
    loader = get_loader()
    panel = loader.load_panel(
        list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS)),
        PANEL_METRICS, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    factors = build_all_factors(panel, macro)
    returns = loader.load_returns(["btc"], start, end)["btc_return"]
    features = pd.DataFrame({
        "vixcls": factors["vixcls"],
        "t10y2y": factors["t10y2y"],
        "abs_liq_flow": factors["liq_flow"].abs(),
    })
    return returns, features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    args = p.parse_args()

    returns, features = load_inputs(args.start, args.end)
    logger.info("btc returns: %d days | features: %s",
                returns.notna().sum(), list(features.columns))
    print(render(run_experiment(returns, features)))


if __name__ == "__main__":
    main()
