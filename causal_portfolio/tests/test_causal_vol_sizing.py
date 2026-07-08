"""Causality tests for the causal vol-sizing experiment.

The experiment's claim rests on the alignment contract: everything used to
set exposure at day t (trend gate, vol forecast, target vol, sizing
multiplier) uses data <= t only; the return earned is (t, t+1]. Pinned with
synthetic data via the perturb-future pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.causal_vol_sizing import (
    CAUSAL_COLS,
    HAR_COLS,
    backtest,
    build_dataset,
    sizing_multiplier,
    walk_forward_forecast,
)

N_DAYS = 700
TRAIN = 120   # smaller walk-forward params for test speed
REFIT = 10


def _synthetic(n: int = N_DAYS) -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(5e-4, 0.03, n), index=idx)
    features = pd.DataFrame({
        "vixcls": np.cumsum(rng.normal(0, 0.1, n)),
        "t10y2y": np.cumsum(rng.normal(0, 0.05, n)),
        "abs_liq_flow": np.abs(rng.normal(0, 1, n)),
    }, index=idx)
    return returns, features


def _pipeline(returns, features):
    ds = build_dataset(returns, features, median_window=TRAIN)
    preds = {
        "har": walk_forward_forecast(
            ds, HAR_COLS, train_window=TRAIN, refit_every=REFIT),
        "har_causal": walk_forward_forecast(
            ds, HAR_COLS + CAUSAL_COLS, train_window=TRAIN, refit_every=REFIT),
    }
    return ds, preds


def test_forecasts_and_exposures_do_not_use_future_data():
    returns, features = _synthetic()
    base_ds, base_preds = _pipeline(returns, features)

    # Perturb ONLY the last 60 days of returns and features; every quantity
    # stamped at t < cutoff must be bit-identical.
    r2, f2 = returns.copy(), features.copy()
    r2.iloc[-60:] = r2.iloc[-60:] * 3.0 + 0.01
    f2.iloc[-60:] += 5.0
    ds2, preds2 = _pipeline(r2, f2)

    cutoff = returns.index[-60]
    safe = base_ds.index[base_ds.index < cutoff]
    for col in ("gate", "rv1", "rv5", "rv22", "tgt_vol",
                "vixcls_lag1", "t10y2y_lag1", "abs_liq_flow_lag1"):
        pd.testing.assert_series_equal(
            base_ds.loc[safe, col], ds2.loc[safe, col], check_names=False)
    for arm in ("har", "har_causal"):
        pd.testing.assert_series_equal(
            base_preds[arm].loc[safe], preds2[arm].loc[safe],
            check_names=False)
        mult_base = sizing_multiplier(base_ds, base_preds[arm])
        mult2 = sizing_multiplier(ds2, preds2[arm])
        pd.testing.assert_series_equal(
            (base_ds["gate"] * mult_base).loc[safe],
            (ds2["gate"] * mult2).loc[safe], check_names=False)


def test_multiplier_bounded_and_gate_dominates():
    returns, features = _synthetic()
    ds, preds = _pipeline(returns, features)
    mult = sizing_multiplier(ds, preds["har_causal"]).dropna()
    assert ((mult >= 0) & (mult <= 1)).all()          # no leverage, cap 1
    exposure = (ds["gate"] * sizing_multiplier(ds, preds["har_causal"]))
    assert (exposure[ds["gate"] == 0].fillna(0) == 0).all()   # flat when trend off


def test_backtest_charges_fee_per_switch():
    returns, features = _synthetic(300)
    ds = build_dataset(returns, features, median_window=TRAIN)
    ds = ds.dropna(subset=["fwd_ret"])
    e = pd.Series(0.0, index=ds.index)
    e.iloc[10:20] = 1.0   # one round trip: 2 switches, 2 fees
    r = backtest(ds["fwd_ret"], e, "pulse")
    held = ds["fwd_ret"].iloc[10:20].sum()
    assert r.net.sum() == pytest.approx(held - 2 * 5.0 / 1e4)
    flat = backtest(ds["fwd_ret"], pd.Series(0.0, index=ds.index), "flat")
    assert flat.net.abs().sum() == 0.0
