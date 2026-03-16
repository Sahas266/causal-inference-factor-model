"""
Cross-validate Rust CPCM OLS results against Python statsmodels.

Reads the same Parquet panel data used by the Rust pipeline, constructs the same
factor features (z-scored), and runs statsmodels OLS for comparison.

Usage:
    cd causal_model
    python scripts/cross_validate_ols.py [--asset eth] [--parquet cache/panel.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ── Factor definitions (must match Rust cpcm-core/src/cpcm_dag.rs) ──────────

GLOBAL_FACTORS = [
    "liq_flow", "stable_flow", "chain_congestion",
    "staking_yield", "mev_pressure", "cex_dex_flow",
]

MACRO_FACTORS = [
    "dff", "dgs10", "vixcls", "t10y2y", "cpiaucsl", "m2sl", "dtwexbgs",
]

ASSET_COVARIATES = ["whale_conc", "protocol_rev", "emissions", "chain_activity"]


def z_score(s: pd.Series) -> pd.Series:
    """Z-score normalize, ignoring NaN."""
    mean = s.mean()
    std = s.std(ddof=0)
    if std < 1e-15:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def compute_liq_flow(df: pd.DataFrame) -> pd.Series:
    tvl_cols = [c for c in df.columns if c.endswith("_tvl_usd")]
    if tvl_cols:
        return z_score(df[tvl_cols].sum(axis=1).diff())
    if "eth_lp_net_flow_usd" in df.columns:
        return z_score(df["eth_lp_net_flow_usd"])
    return pd.Series(np.nan, index=df.index)


def compute_stable_flow(df: pd.DataFrame) -> pd.Series:
    cols = []
    for c in ["usdc_SplyCur", "usdt_SplyCur", "usde_stablecoin_circulating_usd"]:
        if c in df.columns:
            cols.append(c)
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return z_score(df[cols].sum(axis=1).diff())


def compute_chain_congestion(df: pd.DataFrame) -> pd.Series:
    if "eth_avg_gas_price_gwei" in df.columns:
        return z_score(df["eth_avg_gas_price_gwei"])
    fee_cols = [c for c in ["eth_FeeTotNtv", "btc_FeeTotNtv", "bnb_FeeTotNtv"] if c in df.columns]
    if fee_cols:
        return z_score(df[fee_cols].sum(axis=1))
    return pd.Series(np.nan, index=df.index)


def compute_staking_yield(df: pd.DataFrame) -> pd.Series:
    if "eth_staking_apr" in df.columns:
        return z_score(df["eth_staking_apr"])
    return pd.Series(np.nan, index=df.index)


def compute_mev_pressure(df: pd.DataFrame) -> pd.Series:
    if "eth_mev_revenue_eth" in df.columns:
        return z_score(df["eth_mev_revenue_eth"].rolling(7).std())
    if "eth_stddev_base_fee_gwei" in df.columns:
        return z_score(df["eth_stddev_base_fee_gwei"])
    if "eth_avg_base_fee_gwei" in df.columns:
        return z_score(df["eth_avg_base_fee_gwei"].rolling(7).std())
    return pd.Series(np.nan, index=df.index)


def compute_cex_dex_flow(df: pd.DataFrame) -> pd.Series:
    if "eth_cex_netflow_usd" in df.columns:
        return z_score(df["eth_cex_netflow_usd"])
    if "eth_FlowInExNtv" in df.columns and "eth_FlowOutExNtv" in df.columns:
        return z_score(df["eth_FlowInExNtv"] - df["eth_FlowOutExNtv"])
    return pd.Series(np.nan, index=df.index)


def compute_returns(df: pd.DataFrame, asset: str) -> pd.Series:
    for col in [f"{asset}_PriceUSD", f"{asset}_price_usd"]:
        if col in df.columns:
            return np.log(df[col] / df[col].shift(1))
    return pd.Series(np.nan, index=df.index)


def compute_covariate(df: pd.DataFrame, asset: str, cov: str) -> pd.Series:
    mapping = {
        "whale_conc": f"{asset}_top_10_pct_balance",
        "protocol_rev": f"{asset}_fees_usd",
        "emissions": f"{asset}_IssTotNtv",
    }
    if cov == "chain_activity":
        tx = df.get(f"{asset}_TxCnt")
        adr = df.get(f"{asset}_AdrActCnt")
        if tx is not None and adr is not None:
            return (z_score(tx) + z_score(adr)) / 2
        if tx is not None:
            return z_score(tx)
        if adr is not None:
            return z_score(adr)
        return pd.Series(np.nan, index=df.index)

    col = mapping.get(cov, "")
    if col in df.columns:
        return z_score(df[col])
    return pd.Series(np.nan, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Cross-validate CPCM OLS vs statsmodels")
    parser.add_argument("--asset", default="eth", help="Asset to validate (default: eth)")
    parser.add_argument("--parquet", default="cache/panel.parquet", help="Path to panel parquet")
    parser.add_argument("--rust-csv", default="results.csv", help="Path to Rust results CSV")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: Parquet file not found: {parquet_path}")
        print("Run `cargo run -p cpcm-cli -- run` first to generate the panel cache.")
        sys.exit(1)

    print(f"Loading panel from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Panel shape: {df.shape}")

    asset = args.asset

    # ── Compute factors ──────────────────────────────────────────────
    features = {}

    # Global factors
    factor_funcs = {
        "liq_flow": compute_liq_flow,
        "stable_flow": compute_stable_flow,
        "chain_congestion": compute_chain_congestion,
        "staking_yield": compute_staking_yield,
        "mev_pressure": compute_mev_pressure,
        "cex_dex_flow": compute_cex_dex_flow,
    }
    for name, func in factor_funcs.items():
        features[name] = func(df)

    # Macro factors
    for series in MACRO_FACTORS:
        col = f"macro_{series.upper()}"
        if col in df.columns:
            features[series] = z_score(df[col])

    # Asset covariates
    for cov in ASSET_COVARIATES:
        features[f"{asset}_{cov}"] = compute_covariate(df, asset, cov)

    # Returns
    y = compute_returns(df, asset)

    # ── Build regression DataFrame ──────────────────────────────────
    reg_df = pd.DataFrame(features)
    reg_df[f"{asset}_return"] = y

    # Drop features that are >90% NaN
    for col in list(reg_df.columns):
        if col == f"{asset}_return":
            continue
        valid_pct = reg_df[col].notna().mean()
        if valid_pct < 0.10:
            print(f"  Dropping {col}: {valid_pct:.1%} valid")
            reg_df.drop(col, axis=1, inplace=True)

    # Complete cases
    reg_df.dropna(inplace=True)
    print(f"\nComplete cases for {asset}: {len(reg_df)}")

    if len(reg_df) < 30:
        print(f"ERROR: Only {len(reg_df)} complete cases, need >= 30")
        sys.exit(1)

    # ── Run OLS ──────────────────────────────────────────────────────
    feature_cols = [c for c in reg_df.columns if c != f"{asset}_return"]
    X = sm.add_constant(reg_df[feature_cols])
    y_ols = reg_df[f"{asset}_return"]

    model = sm.OLS(y_ols, X).fit()

    print(f"\n{'='*70}")
    print(f"PYTHON statsmodels OLS — {asset}")
    print(f"{'='*70}")
    print(model.summary())

    # ── Compare with Rust results ────────────────────────────────────
    rust_csv = Path(args.rust_csv)
    if rust_csv.exists():
        print(f"\n{'='*70}")
        print(f"COMPARISON: Rust vs Python")
        print(f"{'='*70}")
        rust_df = pd.read_csv(rust_csv)
        rust_asset = rust_df[rust_df["asset"] == asset]
        if not rust_asset.empty:
            row = rust_asset.iloc[0]
            print(f"\n  Rust:   n={row.get('n_obs', '?')}, R²={row.get('r_squared', '?'):.6f}")
            print(f"  Python: n={model.nobs:.0f}, R²={model.rsquared:.6f}")
            print(f"  R² diff: {abs(model.rsquared - float(row.get('r_squared', 0))):.8f}")
        else:
            print(f"  No Rust results found for {asset}")
    else:
        print(f"\n  (No Rust CSV at {rust_csv} — run CPCM pipeline first for comparison)")

    # ── Diagnostics ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}")

    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    dw = durbin_watson(model.resid)
    print(f"  Durbin-Watson: {dw:.4f}")

    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)
    print(f"  Breusch-Pagan: stat={bp_stat:.4f}, p={bp_p:.4f}")

    jb_stat, jb_p = stats.jarque_bera(model.resid)[:2]
    print(f"  Jarque-Bera:   stat={jb_stat:.4f}, p={jb_p:.4f}")

    print(f"\n  VIF:")
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        flag = " *** HIGH" if vif > 10 else ""
        print(f"    {col:25s} VIF={vif:.2f}{flag}")


if __name__ == "__main__":
    main()
