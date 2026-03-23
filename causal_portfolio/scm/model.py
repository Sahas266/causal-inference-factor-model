"""DoWhy SCM instantiation for CPCM.

Creates a DoWhy CausalModel from the Star-DAG for causal identification.
"""

import pandas as pd
from dowhy import CausalModel

from .graph import build_cpcm_dag, to_gml


def create_scm(
    data: pd.DataFrame,
    assets: list[str],
    selected_drivers: list[str] | None = None,
) -> CausalModel:
    """Create a DoWhy CausalModel from the CPCM Star-DAG.

    Args:
        data: Combined DataFrame with factor, return, and instrument columns.
        assets: List of asset tickers.
        selected_drivers: Combo-selected driver names (if None, uses all).

    Returns:
        DoWhy CausalModel ready for identification/estimation.
    """
    dag = build_cpcm_dag(assets, selected_drivers)
    gml = to_gml(dag)

    # Treatment = selected drivers (or all factors)
    if selected_drivers:
        treatment = [d for d in selected_drivers if d in data.columns]
    else:
        treatment = [c for c in data.columns if c not in
                     [f"{a}_return" for a in assets]]

    # Outcome = asset returns
    outcome = [f"{a}_return" for a in assets if f"{a}_return" in data.columns]

    if not treatment or not outcome:
        raise ValueError(
            f"Missing columns. Treatment: {treatment}, Outcome: {outcome}, "
            f"Available: {list(data.columns)}"
        )

    model = CausalModel(
        data=data,
        graph=gml,
        treatment=treatment,
        outcome=outcome[0] if len(outcome) == 1 else outcome,
    )
    return model
