from econml.iv.dml import DMLIV
from sklearn.ensemble import RandomForestRegressor

def estimate_effect_econml(df, treatment, outcome, instrument):
    """
    EconML 2SLS / DML-IV template.
    """
    model = DMLIV(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        model_z=RandomForestRegressor()
    )

    y = df[outcome].values
    T = df[treatment].values
    Z = df[instrument].values
    X = df.drop(columns=[treatment, outcome, instrument]).values

    model.fit(y, T, Z, X=X)
    return model

def identify_effect(model):
    """Find the estimand based on the DAG."""
    estimand = model.identify_effect()
    print("Identified estimand:\n", estimand)
    return estimand


def estimate_effect(model, estimand, method="iv.instrumental_variable"):
    """
    Estimate causal effects using Instrumental Variables (IV)
    through DoWhy or EconML.
    """
    estimate = model.estimate_effect(
        estimand,
        method_name=method,
        test_significance=True
    )
    print("Causal estimate:\n", estimate)
    return estimate

