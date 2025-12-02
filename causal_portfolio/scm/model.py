import dowhy
from dowhy import CausalModel

def create_scm(data, graph_gml: str) -> CausalModel:
    """
    Create the structural causal model using DoWhy.
    """
    model = CausalModel(
        data=data,
        graph=graph_gml,
        treatment=list(data.columns.drop("portfolio_return")),
        outcome="portfolio_return"
    )
    return model
