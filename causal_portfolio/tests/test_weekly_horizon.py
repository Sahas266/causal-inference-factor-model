"""Alignment tests for the weekly-horizon experiment.

The claim rests on the causality contract: the factor value at week t uses
only daily data <= t, and the return earned is (t, t+1w]. Pinned here with
synthetic data, following tests/test_funding_intraday.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.weekly_horizon import backtest, build_grid


def _daily(n_days: int = 700, start: str = "2022-01-03"):
    idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    factors = pd.DataFrame(
        {"liq_flow": rng.normal(size=n_days), "vixcls": rng.normal(size=n_days)},
        index=idx,
    )
    returns = pd.DataFrame(
        {"btc": rng.normal(0, 0.01, n_days), "eth": rng.normal(0, 0.01, n_days)},
        index=idx,
    )
    return factors, returns


def test_fwd_btc_is_the_next_weeks_compounded_return():
    factors, returns = _daily()
    grid = build_grid(factors, returns, "W-FRI")
    t = grid.index[10]
    next_week = returns["btc"].loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=7)]
    assert grid.loc[t, "fwd_btc"] == pytest.approx((1 + next_week).prod() - 1)


def test_factor_at_week_t_is_last_daily_value_leq_t():
    factors, returns = _daily()
    grid = build_grid(factors, returns, "W-FRI")
    t = grid.index[10]
    assert grid.loc[t, "liq_flow"] == factors["liq_flow"].loc[:t].iloc[-1]


def test_signals_and_targets_do_not_use_future_data():
    factors, returns = _daily()
    base = build_grid(factors, returns, "W-FRI")

    # Perturb ONLY the final 30 days; every column at weeks whose forward
    # window ends before the perturbation must be bit-identical.
    factors2, returns2 = factors.copy(), returns.copy()
    factors2.iloc[-30:] += 100.0
    returns2.iloc[-30:] += 0.5
    changed = build_grid(factors2, returns2, "W-FRI")

    cutoff = factors.index[-30]
    safe = base.index[base.index < cutoff - pd.Timedelta(days=7)]
    for col in base.columns:
        pd.testing.assert_series_equal(
            base.loc[safe, col], changed.loc[safe, col], check_names=False
        )


def test_backtest_charges_fee_per_switch():
    factors, returns = _daily()
    grid = build_grid(factors, returns, "W-FRI").dropna(subset=["fwd_btc"])
    fwd = grid["fwd_btc"]

    never = backtest(fwd, pd.Series(0.0, index=grid.index), 52, "flat")
    assert never.net.abs().sum() == 0.0 and never.n_switches == 0

    e = pd.Series(0.0, index=grid.index)
    e.iloc[10:20] = 1.0  # one round trip -> 2 switches, 2 fees
    r = backtest(fwd, e, 52, "pulse")
    assert r.n_switches == 2
    assert r.net.sum() == pytest.approx(fwd.iloc[10:20].sum() - 2 * 5.0 / 1e4)
