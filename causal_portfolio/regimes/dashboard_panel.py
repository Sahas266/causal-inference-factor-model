"""Regime-analysis logic for the dashboard's Regimes tab — no Streamlit dep.

Kept separate from dashboard.py so it can be unit-tested without the Streamlit
runtime. The tab is thin UI over `analyze_regimes`.

Honesty note baked into the API: the `causal` flag switches between
  - causal=False : full-sample Viterbi decode (non-causal, look-ahead). Shows
    the rosiest in-sample regime structure. Useful to ask "does structure
    exist?" but NOT what a live strategy would see.
  - causal=True  : rolling-window fit + forward filter (strictly causal). What
    a real-time system would actually have known at each point. Our research
    found per-regime driver winners differ substantially between the two —
    the dashboard surfaces both so the difference is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import ANNUALIZATION
from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS,
    MACRO_SERIES,
    PANEL_METRICS,
    build_all_factors,
)
from causal_portfolio.factors.combo_selector import ComboDriverSelector
from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    build_regime_features,
    dwell_stats,
    rolling_fit_decode,
)


@dataclass
class RegimePanelResult:
    n_states: int
    causal: bool
    feature_columns: list[str]
    dates: pd.DatetimeIndex                       # labeled period (post warmup)
    labels: np.ndarray                            # int regime label per date
    transition_matrix: np.ndarray                 # (n_states, n_states)
    state_means: np.ndarray                       # (n_states, n_features)
    dwell: dict                                   # state -> dwell stats
    per_regime_drivers: dict                      # state -> {"winner": tuple, "score": float, "ranking": [...]}
    per_regime_returns: dict                      # state -> {"ann_return","ann_vol","sharpe","days"}
    market_curve: pd.Series                       # equal-weight basket growth-of-1 over labeled period
    notes: list[str] = field(default_factory=list)


def _load(assets, start, end):
    from causal_portfolio.data import get_loader
    loader = get_loader()
    panel_assets = list(dict.fromkeys(list(assets) + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    return panel, returns, macro


def analyze_regimes(
    assets: list[str],
    start: str,
    end: str,
    *,
    n_states: int = 2,
    hmm_window: int = 504,
    hmm_refit_every: int = 63,
    n_restarts: int = 5,
    m_drivers: int = 3,
    causal: bool = True,
    min_obs_per_regime: int = 60,
) -> RegimePanelResult:
    """Fit an HMM regime model and characterize each regime.

    Returns everything the Regimes tab renders: regime labels over time,
    transition matrix, per-state feature means + dwell, per-regime top driver
    subset, per-regime market-return stats, and the market equity curve for
    the timeline ribbon.
    """
    notes: list[str] = []
    panel, returns, macro = _load(assets, start, end)
    if "btc_return" not in returns.columns:
        raise ValueError(
            "Regime features need BTC realized vol — include 'btc' in the asset "
            "universe."
        )

    factors = build_all_factors(panel, macro)
    available = factors.dropna(axis=1, how="all")
    feats = build_regime_features(macro, returns)

    # Align everything to the feature index (feats has a 21d vol warmup already)
    common = feats.index.intersection(returns.index).intersection(available.index)
    feats = feats.loc[common]
    returns = returns.loc[common]
    available = available.loc[common]

    n_states = max(1, min(int(n_states), 3))
    m_actual = min(int(m_drivers), available.shape[1])

    # ── Characterization classifier (full-sample fit) ────────────────
    # Used for the transition matrix + per-state feature means — these are
    # inherently full-sample *descriptions* of the regimes, regardless of the
    # causal toggle (which governs how we LABEL each day, below).
    char = RegimeClassifier(n_states=n_states, n_restarts=n_restarts).fit(feats)
    transition_matrix = char.transition_matrix()
    state_means = char.state_means()

    # ── Labels: causal (rolling forward filter) vs non-causal (Viterbi) ──
    if n_states == 1:
        label_series = pd.Series(np.zeros(len(feats), dtype=int), index=feats.index)
        notes.append("n_states=1: single regime (no switching) — a baseline.")
    elif causal:
        label_series = rolling_fit_decode(
            feats, window_size=hmm_window, refit_every=hmm_refit_every,
            n_states=n_states, n_restarts=n_restarts,
        ).dropna().astype(int)
        if len(label_series) < min_obs_per_regime:
            notes.append("Causal labels very short — try a smaller HMM window "
                         "or a wider date range.")
    else:
        label_series = pd.Series(char.decode(feats), index=feats.index)

    labeled_idx = label_series.index
    labels = label_series.values.astype(int)

    # ── Dwell stats ──────────────────────────────────────────────────
    dwell = dwell_stats(labels)

    # ── Per-regime top driver subset ─────────────────────────────────
    selector = ComboDriverSelector()
    per_regime_drivers: dict = {}
    for state in range(n_states):
        state_dates = labeled_idx[labels == state]
        if len(state_dates) < min_obs_per_regime:
            per_regime_drivers[state] = {
                "winner": None, "score": None, "ranking": [],
                "note": f"only {len(state_dates)} obs (< {min_obs_per_regime})",
            }
            continue
        R_k = returns.loc[state_dates]
        F_k = available.loc[state_dates].dropna(axis=1, how="all")
        if F_k.shape[1] < m_actual:
            per_regime_drivers[state] = {
                "winner": None, "score": None, "ranking": [],
                "note": "too few non-NaN factors in this regime",
            }
            continue
        ranking = selector.rank_all_subsets(R_k, F_k, m=m_actual)
        per_regime_drivers[state] = {
            "winner": list(ranking[0][0]),
            "score": float(ranking[0][1]),
            "ranking": [(list(s), float(sc)) for s, sc in ranking[:5]],
            "note": None,
        }

    # ── Per-regime market-return stats ───────────────────────────────
    # "Market" proxy = equal-weight of the loaded assets' daily returns.
    ret_cols = [c for c in returns.columns if c.endswith("_return")]
    market_r = returns[ret_cols].mean(axis=1)
    per_regime_returns: dict = {}
    for state in range(n_states):
        state_dates = labeled_idx[labels == state]
        r = market_r.loc[state_dates].dropna()
        if len(r) < 2:
            per_regime_returns[state] = {"ann_return": float("nan"),
                                         "ann_vol": float("nan"),
                                         "sharpe": float("nan"), "days": len(r)}
            continue
        mu = float(r.mean()) * ANNUALIZATION
        vol = float(r.std()) * np.sqrt(ANNUALIZATION)
        per_regime_returns[state] = {
            "ann_return": mu, "ann_vol": vol,
            "sharpe": mu / vol if vol > 1e-9 else 0.0,
            "days": int(len(r)),
        }

    # ── Market equity curve over labeled period (for the ribbon) ─────
    market_curve = (1.0 + market_r.loc[labeled_idx].fillna(0.0)).cumprod()

    return RegimePanelResult(
        n_states=n_states, causal=causal,
        feature_columns=list(feats.columns),
        dates=labeled_idx, labels=labels,
        transition_matrix=transition_matrix, state_means=state_means,
        dwell=dwell, per_regime_drivers=per_regime_drivers,
        per_regime_returns=per_regime_returns,
        market_curve=market_curve, notes=notes,
    )


def contiguous_runs(dates: pd.DatetimeIndex, labels: np.ndarray):
    """Yield (start_date, end_date, state) for each contiguous regime run.

    Used by the timeline ribbon to draw one colored band per run.
    """
    if len(labels) == 0:
        return
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            yield dates[run_start], dates[i - 1], int(labels[run_start])
            run_start = i
    yield dates[run_start], dates[len(labels) - 1], int(labels[run_start])
