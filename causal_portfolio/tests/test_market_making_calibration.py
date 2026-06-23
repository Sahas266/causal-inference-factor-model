from __future__ import annotations

import math

import pytest

from causal_portfolio.market_making.calibration import (
    FillObservation,
    MarkoutObservation,
    adverse_markout_bps,
    ewma_absolute_volatility,
    fit_exponential_intensity,
    fit_markout_curve,
)
from causal_portfolio.market_making.toxicity import (
    ToxicityConfig,
    ToxicitySignal,
    toxicity_adjustment,
)
from causal_portfolio.market_making.types import AggressorSide, QuotePolicy


def test_ewma_volatility_handles_irregular_time():
    sigma = ewma_absolute_volatility(
        [100.0, 101.0, 100.0, 102.0], [0, 1_000, 3_000, 7_000], half_life_seconds=10
    )
    assert math.isfinite(sigma) and sigma > 0


def test_fill_intensity_recovers_exponential_shape():
    observations = []
    for distance in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        fills = round(10_000 * 0.2 * math.exp(-2.5 * distance))
        observations.append(FillObservation(distance, 10_000, fills))
    fit = fit_exponential_intensity(observations)
    assert fit.converged
    assert fit.arrival_rate == pytest.approx(0.2, rel=0.03)
    assert fit.kappa == pytest.approx(2.5, rel=0.03)


def test_markout_curve_drives_spread_and_posterior_policies():
    observations = [
        MarkoutObservation(0.3, AggressorSide.BUY, 8.0),
        MarkoutObservation(0.3, AggressorSide.SELL, 4.0),
    ]
    curve = fit_markout_curve(observations, edges=(0.0, 0.5, 1.0), prior_weight=0)
    signal = ToxicitySignal(0.3, 0.6, 0.2)
    symmetric = toxicity_adjustment(
        signal, ToxicityConfig(policy=QuotePolicy.AS_PIN_SPREAD), markouts=curve
    )
    assert symmetric.bid_extra_bps == symmetric.ask_extra_bps == 6.0
    posterior = toxicity_adjustment(
        signal, ToxicityConfig(policy=QuotePolicy.AS_PIN_POSTERIOR), markouts=curve
    )
    assert posterior.bid_extra_bps == pytest.approx(0.8)
    assert posterior.ask_extra_bps == pytest.approx(4.8)


def test_size_policy_and_markout_sign():
    adjustment = toxicity_adjustment(
        ToxicitySignal(0.8, 0.1, 0.1),
        ToxicityConfig(
            policy=QuotePolicy.AS_PIN_SIZE,
            min_size_scale=0.25,
            size_toxicity_scale=1.0,
        ),
    )
    assert adjustment.bid_size_scale == adjustment.ask_size_scale == 0.25
    assert adverse_markout_bps(is_maker_buy=True, fill_price=100, future_mid=99) == 100
    assert adverse_markout_bps(is_maker_buy=False, fill_price=100, future_mid=101) == 100
