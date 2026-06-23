from __future__ import annotations

import numpy as np
import pytest

from causal_portfolio.market_making.pin import (
    fit_pin,
    pin_posteriors,
    rolling_pin,
    simulate_pin_counts,
)


def test_pin_recovers_synthetic_probability():
    buys, sells = simulate_pin_counts(
        1_500, alpha=0.3, delta=0.6, mu=40.0, epsilon=20.0, seed=7
    )
    fit = fit_pin(buys, sells, min_days=60, n_starts=8, seed=11)
    expected = 0.3 * 40 / (0.3 * 40 + 2 * 20)
    assert fit.converged
    assert fit.pin == pytest.approx(expected, abs=0.04)
    assert fit.epsilon == pytest.approx(20.0, rel=0.15)
    assert fit.mu == pytest.approx(40.0, rel=0.2)


def test_posterior_identifies_directional_imbalance():
    buys, sells = simulate_pin_counts(
        500, alpha=0.35, delta=0.5, mu=30.0, epsilon=10.0, seed=3
    )
    fit = fit_pin(buys, sells, n_starts=6)
    posterior = pin_posteriors([80, 2], [2, 80], fit)
    assert posterior[0].informed_buy > 0.99
    assert posterior[1].informed_sell > 0.99
    assert posterior[0].no_news + posterior[0].informed == pytest.approx(1.0)


def test_rolling_pin_has_no_lookahead_shape():
    buys = np.full(25, 10)
    sells = np.full(25, 10)
    result = rolling_pin(buys, sells, window=20, n_starts=2)
    assert result[:19] == [None] * 19
    assert all(fit is not None for fit in result[19:])


def test_pin_rejects_fractional_counts_and_short_samples():
    with pytest.raises(ValueError, match="integer"):
        fit_pin([1.5] * 20, [1] * 20)
    with pytest.raises(ValueError, match="at least"):
        fit_pin([1] * 10, [1] * 10, min_days=20)
