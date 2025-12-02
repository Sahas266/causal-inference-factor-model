import pandas as pd


# TODO remove csv placeholders and replace with DB dataloaders
def load_factors(path: str = "data/factors.csv") -> pd.DataFrame:
    """Load factor time series: macro, crypto, liquidity, vol, etc."""
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")

def load_portfolio(path: str = "data/portfolio.csv") -> pd.DataFrame:
    """Load asset returns or pnl that we want to model."""
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")

def load_shocks(path: str = "data/shocks.csv") -> pd.DataFrame:
    """Load exogenous shock time series (IVs)."""
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")
