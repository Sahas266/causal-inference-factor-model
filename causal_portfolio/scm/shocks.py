"""Instrumental variable (IV) construction for CPCM identification.

Mirrors the Rust INSTRUMENTS constant from cpcm_dag.rs. Each IV must:
1. Be relevant (correlated with its treatment factor)
2. Satisfy exclusion (affects outcome only through the treatment)

Reference: Section 3.3 of arXiv:2509.09585v2
"""

import numpy as np
import pandas as pd

from .graph import INSTRUMENTS


def create_instruments(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Construct instrumental variables from raw panel data.

    Args:
        panel: Wide-format panel from CPCMDataLoader.
        factors: Computed factor DataFrame from builder.

    Returns:
        DataFrame with one column per available instrument.
    """
    instruments = pd.DataFrame(index=factors.index)

    for iv_name, treatment, lag in INSTRUMENTS:
        iv_series = _construct_iv(iv_name, treatment, panel, factors, lag)
        if iv_series is not None:
            instruments[iv_name] = iv_series

    return instruments


def _construct_iv(
    iv_name: str,
    treatment: str,
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    lag: int,
) -> pd.Series | None:
    """Construct a single IV series.

    Strategy: use lagged exogenous shocks that affect the treatment factor
    but have no direct effect on returns.
    """
    if iv_name == "gas_spike":
        # Large gas price jumps → liquidity displacement
        col = _find(panel, "eth_avg_gas_price_gwei")
        if col is not None:
            pct_change = panel[col].pct_change()
            return (pct_change > pct_change.quantile(0.9)).astype(float).shift(lag)

    elif iv_name == "stablecoin_mint":
        # Large stablecoin supply changes → stable_flow
        if treatment in factors.columns:
            abs_change = factors[treatment].abs()
            return (abs_change > abs_change.quantile(0.9)).astype(float).shift(lag)

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
            return (pct_change > pct_change.quantile(0.9)).astype(float).shift(lag)

    # Fallback: lagged first-difference of the treatment factor
    if treatment in factors.columns:
        return factors[treatment].diff().shift(lag)

    return None


def _find(df: pd.DataFrame, name: str) -> str | None:
    """Find column by exact name or suffix."""
    if name in df.columns:
        return name
    matches = [c for c in df.columns if c.endswith(f"_{name}")]
    return matches[0] if matches else None
