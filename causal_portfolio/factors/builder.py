"""Compute the 14 CPCM candidate factors from a raw data panel.

Ports the Rust implementation in causal_model/crates/cpcm-factors/src/global_factors.rs
to Python/pandas. Each factor is z-scored for cross-comparability.
"""

import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────

GLOBAL_FACTORS = [
    "liq_flow",
    "stable_flow",
    "funding_basis",
    "chain_congestion",
    "staking_yield",
    "mev_pressure",
    "cex_dex_flow",
]

MACRO_FACTORS = [
    "dff",       # Fed funds rate
    "dgs10",     # 10-year treasury
    "vixcls",    # VIX
    "t10y2y",    # Yield curve slope
    "cpiaucsl",  # CPI
    "m2sl",      # M2 money supply
    "dtwexbgs",  # Trade-weighted dollar index
]

# Canonical warehouse pull for every entry point that builds factors. Anything
# narrower silently starves a factor (e.g. omitting funding_rate_8h made
# funding_basis all-NaN in half the pipelines).
PANEL_METRICS = [
    "PriceUSD", "price", "tvl_usd", "SplyCur",
    "stablecoin_circulating_usd", "FeeTotNtv",
    "FlowInExNtv", "FlowOutExNtv",
    "avg_gas_price_gwei", "avg_base_fee_gwei",
    "stddev_base_fee_gwei", "staking_apr",
    "cex_netflow_usd", "lp_net_flow_usd", "mev_revenue_eth",
    # Perp funding — feeds the funding_basis factor (Hyperliquid, from 2023-10).
    "funding_rate_8h", "funding_premium",
    # Instrument sources (for 2SLS): liquidations + protocol fees.
    "avg_gas_utilization", "liquidation_volume_usd", "perp_liquidation",
    "avg_liquidation_usd", "fees", "total_fees_usd", "base_fees",
]

# FRED series IDs as stored in the warehouse (uppercase asset='macro' metrics).
MACRO_SERIES = [m.upper() for m in MACRO_FACTORS]

# Stablecoins are loaded into the panel as FACTOR INPUTS ONLY (their supply
# feeds stable_flow + the stablecoin_mint instrument). They are never part of
# the trading universe, so returns are not computed for them.
FACTOR_SOURCE_ASSETS = ["usdc", "usdt", "usde"]

TVL_COLUMNS = [
    "aave_tvl_usd", "uni_tvl_usd", "crv_tvl_usd", "pendle_tvl_usd",
    "morpho_tvl_usd", "jup_tvl_usd", "ena_tvl_usd", "aero_tvl_usd",
    "eth_tvl_usd", "sol_tvl_usd", "bnb_tvl_usd", "avax_tvl_usd",
    "pol_tvl_usd", "btc_tvl_usd", "hype_tvl_usd",
]

STABLECOIN_SUPPLY_COLUMNS = [
    "usdc_SplyCur",
    "usdt_SplyCur",
    "usde_stablecoin_circulating_usd",
]


# ── Public API ──────────────────────────────────────────────────────

def build_all_factors(
    panel: pd.DataFrame,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all 14 candidate factors (7 global + 7 macro).

    Args:
        panel: Wide-format DataFrame with ``{asset}_{metric}`` columns and
            DatetimeIndex from ``CPCMDataLoader.load_panel()``.
        macro: Wide-format DataFrame with FRED series columns from
            ``CPCMDataLoader.load_macro()``. If None, macro factors are
            extracted from ``panel`` columns.

    Returns:
        DataFrame with up to 14 columns, one per factor, DatetimeIndex aligned
        to ``panel``.
    """
    factors = pd.DataFrame(index=panel.index)

    # Global factors
    factors["liq_flow"] = _compute_liq_flow(panel)
    factors["stable_flow"] = _compute_stable_flow(panel)
    factors["funding_basis"] = _compute_funding_basis(panel)
    factors["chain_congestion"] = _compute_chain_congestion(panel)
    factors["staking_yield"] = _compute_staking_yield(panel)
    factors["mev_pressure"] = _compute_mev_pressure(panel)
    factors["cex_dex_flow"] = _compute_cex_dex_flow(panel)

    # Macro factors
    macro_src = macro if macro is not None else panel
    for m in MACRO_FACTORS:
        col = _find_column(macro_src, m)
        if col is not None:
            factors[m] = z_score(macro_src[col])
        else:
            factors[m] = np.nan

    # Drop all-NaN factors
    factors = factors.dropna(axis=1, how="all")
    return factors


# ── Global factor computations ──────────────────────────────────────

def _complete_sum(
    df: pd.DataFrame, cols: list[str], ffill_limit: int = 5
) -> pd.Series:
    """Sum components requiring ALL of them present (after short-gap ffill).

    Pandas' default skipna sum lets one missing component drop the total by
    that component's entire magnitude — `.diff()` then fabricates a giant
    fake flow spike that dominates the z-scored factor. Short gaps are
    bridged by ffill; days still missing any component become NaN instead.
    """
    filled = df[cols].ffill(limit=ffill_limit)
    return filled.sum(axis=1, min_count=len(cols))


def _compute_liq_flow(panel: pd.DataFrame) -> pd.Series:
    """diff(sum of protocol TVL) — net liquidity entering/leaving DeFi."""
    available = [c for c in TVL_COLUMNS if c in panel.columns]
    if available:
        total_tvl = _complete_sum(panel, available)
        return z_score(total_tvl.diff())

    # Fallback: Dune LP flow
    col = _find_column(panel, "eth_lp_net_flow_usd")
    if col is not None:
        return z_score(panel[col])

    return pd.Series(np.nan, index=panel.index)


def _compute_stable_flow(panel: pd.DataFrame) -> pd.Series:
    """diff(USDC + USDT + USDe supply) — net stablecoin creation/destruction."""
    available = [c for c in STABLECOIN_SUPPLY_COLUMNS if c in panel.columns]
    if not available:
        return pd.Series(np.nan, index=panel.index)

    total_supply = _complete_sum(panel, available)
    return z_score(total_supply.diff())


def _compute_funding_basis(panel: pd.DataFrame) -> pd.Series:
    """Perp funding basis — mean funding rate across the perp universe, z-scored.

    The CPCM 'Funding Basis' factor. High aggregate funding = leveraged longs
    paying to stay long (overheated/risk-on); negative = shorts paying.
    Source: Hyperliquid funding_rate_8h (8h cadence, daily-resampled in the
    panel). Only available from 2023-10-31 onward, so the series is NaN before
    then and downstream alignment restricts the usable window accordingly.
    """
    funding_cols = [c for c in panel.columns if c.endswith("_funding_rate_8h")]
    if not funding_cols:
        # Fallback: funding premium if rate is absent
        funding_cols = [c for c in panel.columns if c.endswith("_funding_premium")]
    if not funding_cols:
        return pd.Series(np.nan, index=panel.index)
    mean_funding = panel[funding_cols].mean(axis=1)
    return z_score(mean_funding)


def _compute_chain_congestion(panel: pd.DataFrame) -> pd.Series:
    """z-score of gas price or fee data."""
    # Primary: Dune ETH gas data
    col = _find_column(panel, "eth_avg_gas_price_gwei")
    if col is not None:
        return z_score(panel[col])

    # Fallback: CoinMetrics fee data
    fee_cols = [c for c in panel.columns if "FeeTotNtv" in c]
    if fee_cols:
        return z_score(_complete_sum(panel, fee_cols))

    return pd.Series(np.nan, index=panel.index)


def _compute_staking_yield(panel: pd.DataFrame) -> pd.Series:
    """ETH staking APR."""
    col = _find_column(panel, "eth_staking_apr")
    if col is not None:
        return z_score(panel[col])
    return pd.Series(np.nan, index=panel.index)


def _compute_mev_pressure(panel: pd.DataFrame) -> pd.Series:
    """MEV-related volatility proxy: rolling std of base fee or MEV revenue."""
    # Primary: MEV revenue
    col = _find_column(panel, "eth_mev_revenue_eth")
    if col is not None:
        return rolling_std(panel[col], 7)

    # Fallback: base fee std from Dune
    col = _find_column(panel, "eth_stddev_base_fee_gwei")
    if col is not None:
        return z_score(panel[col])

    # Second fallback: rolling std of avg base fee
    col = _find_column(panel, "eth_avg_base_fee_gwei")
    if col is not None:
        return rolling_std(panel[col], 7)

    return pd.Series(np.nan, index=panel.index)


def _compute_cex_dex_flow(panel: pd.DataFrame) -> pd.Series:
    """Net exchange flow (inflow - outflow)."""
    # Primary: Dune CEX netflow
    col = _find_column(panel, "eth_cex_netflow_usd")
    if col is not None:
        return z_score(panel[col])

    # Fallback: CoinMetrics exchange flows
    inflow = _find_column(panel, "eth_FlowInExNtv")
    outflow = _find_column(panel, "eth_FlowOutExNtv")
    if inflow is not None and outflow is not None:
        return z_score(panel[inflow] - panel[outflow])

    return pd.Series(np.nan, index=panel.index)


def innovation_factors(
    factors: pd.DataFrame, min_obs: int = 60, window: int = 252,
) -> pd.DataFrame:
    """AR(1) innovations of each factor, z-scored.

    Most CPCM factors are z-scored LEVELS of persistent series (gas price,
    funding, APR): near-random-walks. Regressions and CI tests on levels are
    dominated by persistence (spurious-regression risk) and the information
    is mostly "where the level already is", not "what just changed". The
    innovation e_t = s_t - (a + rho * s_{t-1}) isolates the day's NEWS.
    Factors that are already diffs (liq_flow, stable_flow) come out nearly
    unchanged (rho ~ 0).

    rho and a are estimated on a TRAILING window and shifted one day, so the
    innovation at t uses only parameters known at t-1 (the previous
    full-sample fit leaked future data into every training window). The final
    z-score remains full-sample — it is affine and absorbed by downstream
    regressions.
    """
    out = {}
    for c in factors.columns:
        s = factors[c]
        lag = s.shift(1)
        roll_cov = lag.rolling(window, min_periods=min_obs).cov(s)
        roll_var = lag.rolling(window, min_periods=min_obs).var()
        rho = (roll_cov / (roll_var + 1e-15)).shift(1)
        a = (
            s.rolling(window, min_periods=min_obs).mean()
            - (roll_cov / (roll_var + 1e-15))
            * lag.rolling(window, min_periods=min_obs).mean()
        ).shift(1)
        innov = s - (a + rho * lag)
        if innov.notna().sum() < min_obs:
            continue
        out[c] = z_score(innov)
    return pd.DataFrame(out, index=factors.index)


# ── Helpers ─────────────────────────────────────────────────────────

def z_score(s: pd.Series) -> pd.Series:
    """Z-score normalization, ignoring NaN."""
    mean = s.mean()
    std = s.std()
    if std is None or std < 1e-15 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def rolling_std(s: pd.Series, window: int = 7) -> pd.Series:
    """Rolling standard deviation, z-scored."""
    rs = s.rolling(window, min_periods=3).std()
    return z_score(rs)


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    """Find a column by exact name or as a suffix (for {asset}_{metric} format).

    Raises on ambiguity — silently picking whichever column pivots first is a
    wrong-column footgun as PANEL_METRICS grows.
    """
    if name in df.columns:
        return name
    matches = [c for c in df.columns if c.endswith(f"_{name}")]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous factor column {name!r}: matches {sorted(matches)}. "
            "Use the fully prefixed '{asset}_{metric}' name."
        )
    return matches[0] if matches else None
