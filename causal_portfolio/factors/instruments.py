"""Instrumental-variable construction for 2SLS — Python port of instruments.rs.

The DAG declares four instruments, each lagged >= 1 day to break simultaneity:

    gas_spike          instruments  liq_flow
    liquidation_level  instruments  funding_basis
    stablecoin_mint    instruments  stable_flow
    protocol_event     instruments  chain_congestion

Each is built from warehouse columns the same way the Rust reference does:
a lagged spike indicator (ROLLING z-score of the day-over-day change > 2σ,
window 252 / min_periods 60, so the label at t uses only data ≤ t), except
stablecoin_mint which is the lagged level of net minting (diff of supply).

Column names here are mapped to what our warehouse actually has (the Rust code
hard-coded ETH Dune column names that differ slightly). Where a single ETH
series is unavailable we aggregate across assets.

IMPORTANT: these are deliberately weak-by-nature signals (mostly-zero spike
indicators). 2SLS is only valid if the first-stage F-statistic shows the
instrument actually moves its treatment factor — that gate lives in the 2SLS
diagnostics, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.factors.builder import FACTOR_SOURCE_ASSETS

# treatment_factor -> (instrument_name, builder_kind, candidate_source_columns)
# builder_kind: "spike" = lagged z-score spike indicator; "level_diff" = lagged diff
_INSTRUMENT_SPEC = {
    "liq_flow": ("gas_spike", "spike",
                 ["avg_gas_utilization", "stddev_base_fee_gwei", "avg_base_fee_gwei", "FeeTotNtv"]),
    "funding_basis": ("liquidation_level", "spike",
                      ["liquidation_volume_usd", "perp_liquidation", "avg_liquidation_usd"]),
    "stable_flow": ("stablecoin_mint", "level_diff",
                    ["SplyCur", "stablecoin_circulating_usd"]),
    "chain_congestion": ("protocol_event", "spike",
                         ["fees", "total_fees_usd", "base_fees", "FeeTotNtv"]),
}

# Stablecoin assets whose supply we sum for stablecoin_mint
_STABLE_ASSETS = FACTOR_SOURCE_ASSETS


# ── helpers (faithful to instruments.rs) ────────────────────────────


def _diff(x: np.ndarray) -> np.ndarray:
    """First difference; first element NaN, NaN-propagating."""
    d = np.full(len(x), np.nan)
    d[1:] = x[1:] - x[:-1]
    return d


def _z_score_abs(x: np.ndarray, window: int = 252, min_periods: int = 60) -> np.ndarray:
    """ROLLING z-score of absolute values: (|x_t| - mean_t) / std_t over a
    trailing window (population std, ddof=0), ignoring NaN.

    Trailing window ⇒ the label at t uses only data ≤ t (the previous
    full-sample version leaked future data into spike thresholds). NaN until
    `min_periods` non-NaN values are in the window. Matches the Rust
    rolling_z_score_abs in cpcm-factors/instruments.rs.
    """
    s = pd.Series(np.abs(np.asarray(x, dtype=float)))
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std
    z[(std < 1e-15) & s.notna() & mean.notna()] = 0.0
    return z.to_numpy()


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """Shift forward by k; first k elements NaN."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k > 0 else x
    return out


def _lagged_z_score_spike(
    x: np.ndarray, threshold: float = 2.0, lag: int = 1,
    window: int = 252, min_periods: int = 60,
) -> np.ndarray:
    """1.0 if rolling-z(diff(x)) > threshold else 0.0, then lagged by `lag`."""
    z = _z_score_abs(_diff(x), window=window, min_periods=min_periods)
    ind = np.where(np.isnan(z), np.nan, (z > threshold).astype(float))
    return _lag(ind, lag)


def _find_columns(panel: pd.DataFrame, suffixes: list[str]) -> list[str]:
    """All panel columns ending in any of the given metric suffixes."""
    cols = []
    for suf in suffixes:
        cols.extend([c for c in panel.columns if c.endswith(f"_{suf}") or c == suf])
        if cols:
            break  # use the first available source family, like the Rust fallback chain
    return cols


# ── public API ──────────────────────────────────────────────────────


def build_instruments(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build instrument time series from a wide panel.

    Returns:
        (instruments_df, iv_map) where instruments_df has one column per
        successfully-built instrument (indexed like panel), and iv_map maps
        treatment_factor -> instrument_name for the factors we could instrument.
    """
    out: dict[str, np.ndarray] = {}
    iv_map: dict[str, str] = {}

    for factor, (iv_name, kind, sources) in _INSTRUMENT_SPEC.items():
        if factor == "stable_flow":
            # stablecoin_mint = lagged diff of summed stablecoin supply
            supply_cols = [c for c in panel.columns
                           if any(c.startswith(f"{a}_") for a in _STABLE_ASSETS)
                           and (c.endswith("_SplyCur") or c.endswith("_stablecoin_circulating_usd"))]
            if not supply_cols:
                continue
            total = panel[supply_cols].sum(axis=1).values.astype(float)
            out[iv_name] = _lag(_diff(total), 1)
            iv_map[factor] = iv_name
            continue

        cols = _find_columns(panel, sources)
        if not cols:
            continue
        # Aggregate the source across available assets (sum), then build spike
        series = panel[cols].sum(axis=1).values.astype(float)
        out[iv_name] = _lagged_z_score_spike(series, threshold=2.0, lag=1)
        iv_map[factor] = iv_name

    instruments_df = pd.DataFrame(out, index=panel.index)
    return instruments_df, iv_map
