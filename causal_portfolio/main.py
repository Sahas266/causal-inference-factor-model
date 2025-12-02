from scm.loaders import load_factors, load_portfolio
from scm.shocks import create_shocks
from scm.graph import build_causal_graph
from scm.model import create_scm
from scm.estimation import identify_effect, estimate_effect

import pandas as pd

if __name__ == "__main__":
    # -------------------------
    # Load Data
    # -------------------------
    factors = load_factors()
    portfolio = load_portfolio()

    # Merge into single dataframe
    df = factors.join(portfolio, how="inner")

    # -------------------------
    # Create Exogenous Shocks
    # -------------------------
    shocks = create_shocks(factors)
    df = df.join(shocks)

    # -------------------------
    # Build Causal Graph
    # -------------------------
    graph = build_causal_graph(factors, portfolio)

    # -------------------------
    # Create SCM
    # -------------------------
    model = create_scm(df, graph)

    # -------------------------
    # Identify & Estimate
    # -------------------------
    estimand = identify_effect(model)
    estimate = estimate_effect(model, estimand)

    print("\nFinal causal estimates:")
    print(estimate)
