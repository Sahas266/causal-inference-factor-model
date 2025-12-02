def build_causal_graph(factors, portfolio):
    """
    Build a causal graph in DoWhy GML format.
    You can expand the DAG later.
    """

    factor_nodes = "\n".join([f'    "{f}" -> "portfolio_return";' 
                              for f in factors.columns])

    gml = f"""
    graph [
        directed 1

        "portfolio_return" [type "outcome"];

{factor_nodes}
    ]
    """

    return gml
