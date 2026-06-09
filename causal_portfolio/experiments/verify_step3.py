"""Step-3 verification: funding_basis + stable_flow populate, instruments build.

Run: python -m causal_portfolio.experiments.verify_step3

Confirms that the factor-side work done in Step 3 actually produces live series
against the warehouse (local DuckDB), not just NaN. Prints per-factor coverage
and per-instrument coverage so we can eyeball that nothing is degenerate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_portfolio.data import get_loader
from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
)
from causal_portfolio.factors.instruments import build_instruments

# Universe: trading assets + perp-funding assets + stablecoins (so stable_flow
# and funding_basis have source data). Stablecoins are inputs only, never traded.
ASSETS = [
    "btc", "eth", "sol", "bnb", "avax", "uni", "aave", "crv", "pendle",
    "ena", "link", "doge", "xrp", "jup", "hype", "aero", "morpho", "tao",
] + FACTOR_SOURCE_ASSETS

METRICS = PANEL_METRICS
MACRO = MACRO_SERIES

START, END = "2021-01-01", "2026-01-01"


def _coverage(s: pd.Series) -> tuple[int, str, str, float]:
    valid = s.dropna()
    if valid.empty:
        return 0, "-", "-", 0.0
    return (len(valid), str(valid.index.min().date()),
            str(valid.index.max().date()), float(valid.std()))


def main() -> None:
    loader = get_loader()
    print(f"Loader: {type(loader).__name__}")
    panel = loader.load_panel(ASSETS, METRICS, START, END)
    macro = loader.load_macro(MACRO, START, END)
    print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} cols")

    factors = build_all_factors(panel, macro)
    print(f"\nFactors built: {list(factors.columns)}")
    print(f"{'factor':<18}{'n_valid':>8}{'  start':>12}{'  end':>12}{'  std':>8}")
    for col in factors.columns:
        n, lo, hi, sd = _coverage(factors[col])
        print(f"{col:<18}{n:>8}{lo:>12}{hi:>12}{sd:>8.3f}")

    # The two factors Step 3 was meant to revive.
    for key in ("funding_basis", "stable_flow"):
        n = factors[key].dropna().shape[0] if key in factors.columns else 0
        status = "OK" if n > 0 else "MISSING"
        print(f"  -> {key}: {n} valid obs [{status}]")

    instruments, iv_map = build_instruments(panel)
    print(f"\nInstruments built: {list(instruments.columns)}")
    print(f"iv_map (treatment -> instrument): {iv_map}")
    print(f"{'instrument':<20}{'n_nonzero':>10}{'n_valid':>10}{'  start':>12}{'  end':>12}")
    for col in instruments.columns:
        s = instruments[col]
        valid = s.dropna()
        nz = int((valid != 0).sum())
        lo = str(valid.index.min().date()) if not valid.empty else "-"
        hi = str(valid.index.max().date()) if not valid.empty else "-"
        print(f"{col:<20}{nz:>10}{len(valid):>10}{lo:>12}{hi:>12}")


if __name__ == "__main__":
    main()
