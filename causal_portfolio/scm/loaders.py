"""Data loaders for the CPCM pipeline.

Primary: load from Supabase via CPCMDataLoader.
Fallback: CSV files in data/ (for offline development).
"""

import os
from pathlib import Path

import pandas as pd

from causal_portfolio.data.supabase_loader import CPCMDataLoader
from causal_portfolio.factors.builder import MACRO_FACTORS

DATA_DIR = Path(__file__).parent.parent / "data"

# Default asset universe (liquid assets with good data coverage)
DEFAULT_ASSETS = [
    "btc", "eth", "sol", "bnb", "avax", "xrp", "doge",
    "uni", "aave", "link", "crv", "pendle",
]

# Metrics needed for factor computation
PANEL_METRICS = [
    "PriceUSD", "price",
    "tvl_usd", "SplyCur", "stablecoin_circulating_usd",
    "FeeTotNtv", "FlowInExNtv", "FlowOutExNtv",
    "avg_gas_price_gwei", "avg_base_fee_gwei", "stddev_base_fee_gwei",
    "staking_apr", "cex_netflow_usd", "lp_net_flow_usd",
    "mev_revenue_eth",
    "TxCnt", "AdrActCnt", "CapMrktCurUSD",
]

MACRO_SERIES = [m.upper() for m in MACRO_FACTORS]


def load_from_supabase(
    assets: list[str] | None = None,
    start: str = "2021-01-01",
    end: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load panel, returns, and macro data from Supabase.

    Returns:
        (panel, returns, macro) DataFrames.
    """
    assets = assets or DEFAULT_ASSETS
    loader = CPCMDataLoader()
    panel = loader.load_panel(assets, PANEL_METRICS, start, end)
    returns = loader.load_returns(assets, start, end)
    macro = loader.load_macro(MACRO_SERIES, start, end)
    return panel, returns, macro


def load_factors(path: str = "data/factors.csv") -> pd.DataFrame:
    """Legacy CSV loader (backward compatibility)."""
    csv_path = DATA_DIR / os.path.basename(path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    return pd.DataFrame()


def load_portfolio(path: str = "data/portfolio.csv") -> pd.DataFrame:
    """Legacy CSV loader (backward compatibility)."""
    csv_path = DATA_DIR / os.path.basename(path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    return pd.DataFrame()
