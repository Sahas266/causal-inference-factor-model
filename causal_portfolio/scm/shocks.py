import pandas as pd
import numpy as np

def create_shocks(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Construct exogenous shock series (IVs) for each factor.
    These *must be uncorrelated with portfolio returns* except through
    the factor they instrument.
    """
    shocks = pd.DataFrame(index=factors.index)

    for col in factors.columns:
        # Option 1: High-frequency surprise (simple demo)
        shocks[f"{col}_shock"] = factors[col].diff().shift(-1)

        # Option 2: Orthogonalized shock to remove correlation with contemporaneous values
        # resid = factors[col] - factors[col].rolling(20).mean()
        # shocks[f"{col}_shock"] = resid

        # Option 3: PCA residual shock template
        # pca_resid = factors[col] - factors.drop(columns=[col]).sum(axis=1)
        # shocks[f"{col}_shock"] = pca_resid

    return shocks
