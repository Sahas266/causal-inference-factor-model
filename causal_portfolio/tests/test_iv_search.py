"""Tests for experiments/iv_search.py — IV candidate screening."""

import numpy as np
import pandas as pd
import pytest

from causal_portfolio.experiments.iv_search import (
    IVCandidate, _direct_path_t, _first_stage_f, screen_candidates,
)


@pytest.fixture
def synthetic():
    """800 days; treatment liq_flow driven by a hidden series Z_raw."""
    rng = np.random.default_rng(7)
    T_len = 800
    idx = pd.date_range("2022-01-01", periods=T_len, freq="D")

    z_raw = rng.standard_normal(T_len).cumsum()         # level with persistence
    z_diff = np.diff(z_raw, prepend=z_raw[0])

    # Treatment: strongly driven by YESTERDAY's z change (so the lag-1
    # candidate transform aligns), plus enough noise that corr(Z,T) < 0.9
    # (a perfectly clean driver would trip the tautology screen, correctly).
    liq = 2.0 * np.roll(z_diff, 1) + 1.5 * rng.standard_normal(T_len)
    liq[0] = np.nan

    # Returns: depend on treatment only (exclusion holds for Z).
    r = 0.01 * liq + 0.01 * rng.standard_normal(T_len)

    factors = pd.DataFrame({"liq_flow": liq}, index=idx)
    returns = pd.DataFrame({"btc_return": r, "eth_return": r * 0.9}, index=idx)
    raw = pd.DataFrame({
        "eth_driver_metric": z_raw,                      # the true instrument
        "eth_noise_metric": rng.standard_normal(T_len).cumsum(),
    }, index=idx)
    return raw, factors, returns


class TestStats:
    def test_first_stage_f_strong(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal(252)
        T = 2 * Z + 0.1 * rng.standard_normal(252)
        assert _first_stage_f(T, Z) > 100

    def test_first_stage_f_irrelevant(self):
        rng = np.random.default_rng(0)
        assert _first_stage_f(rng.standard_normal(252),
                              rng.standard_normal(252)) < 10

    def test_direct_path_t_detects(self):
        rng = np.random.default_rng(0)
        T = rng.standard_normal(252)
        Z = rng.standard_normal(252)
        R = 0.5 * T + 0.5 * Z + 0.1 * rng.standard_normal(252)
        assert _direct_path_t(R, T, Z) > 2.0

    def test_direct_path_t_clean(self):
        rng = np.random.default_rng(0)
        T = rng.standard_normal(252)
        Z = rng.standard_normal(252)
        R = 0.5 * T + 0.1 * rng.standard_normal(252)
        assert _direct_path_t(R, T, Z) < 2.0


class TestScreen:
    def test_true_driver_found_clean(self, synthetic):
        raw, factors, returns = synthetic
        res = screen_candidates(raw, factors, returns)
        driver = [c for c in res if c.source_metric == "driver_metric"
                  and c.transform == "diff"]
        assert driver, "true instrument not surfaced"
        best = driver[0]
        assert best.treatment == "liq_flow"
        assert best.train_f >= 10
        assert not best.flags, f"unexpected flags: {best.flags}"

    def test_noise_not_surfaced(self, synthetic):
        raw, factors, returns = synthetic
        res = screen_candidates(raw, factors, returns)
        assert not any(c.source_metric == "noise_metric" for c in res)

    def test_tautology_flagged(self, synthetic):
        raw, factors, returns = synthetic
        # Candidate whose lag-1 diff IS the treatment (stablecoin_mint trap):
        # treatment_t = 2*z_diff_{t-1}, so cumsum(T)/2 has lag-1 diff == T...
        # simplest: a metric whose diff equals the treatment shifted +1.
        taut = factors["liq_flow"].shift(-1).fillna(0.0).cumsum()
        raw = raw.assign(eth_taut_metric=taut.values)
        res = screen_candidates(raw, factors, returns)
        hits = [c for c in res if c.source_metric == "taut_metric"
                and c.transform == "diff"]
        assert hits and any("TAUTOLOGY" in f for f in hits[0].flags)

    def test_price_like_excluded(self, synthetic):
        raw, factors, returns = synthetic
        raw = raw.assign(eth_price=np.linspace(1, 100, len(raw)))
        res = screen_candidates(raw, factors, returns)
        assert not any("price" in c.candidate for c in res)

    def test_direct_path_flagged(self, synthetic):
        raw, factors, returns = synthetic
        rng = np.random.default_rng(3)
        # Z relevant to T but ALSO directly in next-day returns
        z2 = rng.standard_normal(len(raw)).cumsum()
        z2d = np.diff(z2, prepend=z2[0])
        factors = factors.copy()
        factors["liq_flow"] = (factors["liq_flow"]
                               + 1.5 * pd.Series(np.roll(z2d, 1),
                                                 index=factors.index))
        returns = returns.copy()
        # The probe regresses R_{t+1} on Z_t where Z_t = z2d_{t-1} (lag-1
        # diff transform), so a direct path means returns_s contains z2d_{s-2}.
        direct = pd.Series(z2d, index=returns.index).shift(2).fillna(0.0)
        returns["btc_return"] = returns["btc_return"] + 0.05 * direct
        raw = raw.assign(eth_leaky_metric=z2)
        res = screen_candidates(raw, factors, returns)
        hits = [c for c in res if c.source_metric == "leaky_metric"]
        assert hits and any(
            any("DIRECT-PATH" in f for f in c.flags) for c in hits)
