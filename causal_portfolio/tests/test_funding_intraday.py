"""Alignment tests for the intraday funding experiment.

The experiment's claim rests entirely on the causality contract: signals at
bar-start t use only data <= t, and the return earned is (t, t+8h]. These
tests pin that contract with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.funding_intraday import backtest, build_panel


def _hourly_prices(n_hours: int, start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_hours))), index=idx)


def _hourly_funding(prices: pd.Series) -> pd.Series:
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(1e-5, 5e-6, len(prices)), index=prices.index)


def test_fwd_ret_is_the_next_bar_return():
    prices = _hourly_prices(24 * 10)
    panel = build_panel(prices, _hourly_funding(prices), "8h")
    t = panel.index[5]
    expected = prices[t + pd.Timedelta(hours=8)] / prices[t] - 1.0
    assert panel.loc[t, "fwd_ret"] == pytest.approx(expected)


def test_signals_do_not_use_future_data():
    prices = _hourly_prices(24 * 200)
    funding = _hourly_funding(prices)
    base = build_panel(prices, funding, "8h")

    # Perturb ONLY the final day of prices and funding; every signal column
    # at bars ending before the perturbation must be bit-identical.
    prices2, funding2 = prices.copy(), funding.copy()
    prices2.iloc[-24:] *= 1.5
    funding2.iloc[-24:] += 1e-3
    changed = build_panel(prices2, funding2, "8h")

    cutoff = prices.index[-25]
    safe = base.index[base.index < cutoff - pd.Timedelta(hours=8)]
    for col in ("fund", "z", "q10", "q90", "trail_ret", "trail_vol"):
        pd.testing.assert_series_equal(
            base.loc[safe, col], changed.loc[safe, col], check_names=False
        )


def test_backtest_charges_fee_per_switch_and_respects_exposure():
    prices = _hourly_prices(24 * 40)
    panel = build_panel(prices, _hourly_funding(prices), "8h")
    panel = panel.dropna(subset=["fwd_ret"])

    always = backtest(panel, pd.Series(1.0, index=panel.index), "8h", "bh")
    never = backtest(panel, pd.Series(0.0, index=panel.index), "8h", "flat")
    assert never.net.abs().sum() == 0.0
    assert never.n_switches == 0

    # single round trip: enter on bar 10, exit on bar 20 -> 2 switches,
    # and net return equals the held bars' returns minus two fees
    e = pd.Series(0.0, index=panel.index)
    e.iloc[10:20] = 1.0
    r = backtest(panel, e, "8h", "pulse")
    assert r.n_switches == 2
    held = panel["fwd_ret"].iloc[10:20].sum()
    assert r.net.sum() == pytest.approx(held - 2 * 5.0 / 1e4)
    assert always.pct_in_market == 1.0
