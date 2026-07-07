"""Instrumental variable (IV) construction for CPCM identification.

Mirrors the Rust INSTRUMENTS constant from cpcm_dag.rs. Each IV must:
1. Be relevant (correlated with its treatment factor)
2. Satisfy exclusion (affects outcome only through the treatment)

Reference: Section 3.3 of arXiv:2509.09585v2
"""

import pandas as pd

from .graph import INSTRUMENTS

# Rolling window for spike thresholds: the quantile at t uses only data <= t
# (a full-sample quantile leaks future data into the spike labels).
_ROLL_WINDOW = 252
_ROLL_MIN_PERIODS = 60


def _rolling_q(s: pd.Series, q: float = 0.9) -> pd.Series:
    return s.rolling(_ROLL_WINDOW, min_periods=_ROLL_MIN_PERIODS).quantile(q)


def create_instruments(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Construct instrumental variables from raw panel data.

    Args:
        panel: Wide-format panel from CPCMDataLoader.
        factors: Computed factor DataFrame from builder.

    Returns:
        DataFrame with one column per instrument.

    Raises:
        ValueError: if an instrument's source columns are missing (there is
        deliberately NO fallback — a transform of the treatment itself is not
        a valid instrument).
    """
    instruments = pd.DataFrame(index=factors.index)

    for iv_name, treatment, lag in INSTRUMENTS:
        instruments[iv_name] = _construct_iv(iv_name, treatment, panel, factors, lag)

    return instruments


def _construct_iv(
    iv_name: str,
    treatment: str,
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    lag: int,
) -> pd.Series:
    """Construct a single IV series.

    Strategy: use lagged exogenous shocks that affect the treatment factor
    but have no direct effect on returns.
    """
    if iv_name == "gas_spike":
        # Large gas price jumps → liquidity displacement
        col = _find(panel, "eth_avg_gas_price_gwei")
        if col is not None:
            pct_change = panel[col].pct_change()
            return (pct_change > _rolling_q(pct_change)).astype(float).shift(lag)

    elif iv_name == "stablecoin_mint":
        # Large stablecoin supply changes → stable_flow
        if treatment in factors.columns:
            abs_change = factors[treatment].abs()
            return (abs_change > _rolling_q(abs_change)).astype(float).shift(lag)

    elif iv_name == "liquidation_level":
        # Liquidation events → funding basis shifts
        col = _find(panel, "eth_total_liquidations_usd")
        if col is not None:
            return panel[col].shift(lag)

    elif iv_name == "protocol_event":
        # TVL shocks → chain congestion
        tvl_cols = [c for c in panel.columns if "tvl_usd" in c]
        if tvl_cols:
            total_tvl = panel[tvl_cols].sum(axis=1)
            pct_change = total_tvl.pct_change().abs()
            return (pct_change > _rolling_q(pct_change)).astype(float).shift(lag)

    # No fallback: a transform of the treatment itself is NOT a valid
    # instrument (it cannot satisfy exclusion). Fail loudly instead.
    raise ValueError(
        f"cannot construct instrument {iv_name!r} for treatment {treatment!r}: "
        f"required source columns are missing from the panel"
    )


def _find(df: pd.DataFrame, name: str) -> str | None:
    """Find column by exact name or suffix."""
    if name in df.columns:
        return name
    matches = [c for c in df.columns if c.endswith(f"_{name}")]
    return matches[0] if matches else None
