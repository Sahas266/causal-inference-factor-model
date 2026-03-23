"""CPCM Star-DAG construction using networkx.

Ports the Rust implementation in causal_model/crates/cpcm-core/src/cpcm_dag.rs.
Structure: Global/Macro factors → Asset covariates → Asset returns,
with instrumental variables and per-asset unobserved shocks.

Reference: Section 3.1 of arXiv:2509.09585v2 (Star-DAG / SCM structure)
"""

from enum import Enum, auto
from typing import Optional

import networkx as nx

# ── Constants (mirroring Rust cpcm_dag.rs) ──────────────────────────

GLOBAL_FACTORS = [
    "liq_flow", "stable_flow", "funding_basis",
    "chain_congestion", "staking_yield", "mev_pressure", "cex_dex_flow",
]

MACRO_FACTORS = [
    "dff", "dgs10", "vixcls", "t10y2y", "cpiaucsl", "m2sl", "dtwexbgs",
]

ASSET_COVARIATES = ["whale_conc", "protocol_rev", "emissions", "chain_activity"]

# (instrument_name, treatment_factor, lag)
INSTRUMENTS = [
    ("gas_spike", "liq_flow", 1),
    ("liquidation_level", "funding_basis", 1),
    ("stablecoin_mint", "stable_flow", 1),
    ("protocol_event", "chain_congestion", 1),
]

# Factor → covariate interactions
FACTOR_COVARIATE_INTERACTIONS = [
    ("stable_flow", "whale_conc"),
    ("chain_congestion", "protocol_rev"),
    ("staking_yield", "emissions"),
]


class NodeKind(Enum):
    GLOBAL_FACTOR = auto()
    MACRO_FACTOR = auto()
    ASSET_COVARIATE = auto()
    ASSET_RETURN = auto()
    INSTRUMENT = auto()
    UNOBSERVED_SHOCK = auto()


class EdgeKind(Enum):
    CAUSAL = auto()
    INSTRUMENTAL = auto()


def build_cpcm_dag(
    assets: list[str],
    selected_drivers: Optional[list[str]] = None,
) -> nx.DiGraph:
    """Build the full CPCM causal DAG for a set of assets.

    Args:
        assets: List of asset tickers (e.g., ["btc", "eth", "sol"]).
        selected_drivers: If provided, only include these drivers
            (from Combo selection). Otherwise include all 14 factors.

    Returns:
        networkx DiGraph with node/edge metadata.
    """
    G = nx.DiGraph()

    # Determine which factors to include
    if selected_drivers:
        global_factors = [f for f in GLOBAL_FACTORS if f in selected_drivers]
        macro_factors = [f for f in MACRO_FACTORS if f in selected_drivers]
    else:
        global_factors = GLOBAL_FACTORS
        macro_factors = MACRO_FACTORS

    # ── Global factors ──────────────────────────────────────────
    for f in global_factors:
        G.add_node(f, kind=NodeKind.GLOBAL_FACTOR, asset=None)

    # ── Macro factors ───────────────────────────────────────────
    for m in macro_factors:
        G.add_node(m, kind=NodeKind.MACRO_FACTOR, asset=None)

    # ── Instruments ─────────────────────────────────────────────
    for iv_name, treatment, lag in INSTRUMENTS:
        if treatment in global_factors:
            G.add_node(iv_name, kind=NodeKind.INSTRUMENT, asset=None)
            G.add_edge(iv_name, treatment, kind=EdgeKind.INSTRUMENTAL, lag=lag)

    # ── Per-asset nodes ─────────────────────────────────────────
    for asset in assets:
        ret_node = f"{asset}_return"
        G.add_node(ret_node, kind=NodeKind.ASSET_RETURN, asset=asset)

        # Asset covariates → own return
        for cov in ASSET_COVARIATES:
            cov_node = f"{asset}_{cov}"
            G.add_node(cov_node, kind=NodeKind.ASSET_COVARIATE, asset=asset)
            G.add_edge(cov_node, ret_node, kind=EdgeKind.CAUSAL, lag=0)

        # Global factors → asset return
        for f in global_factors:
            G.add_edge(f, ret_node, kind=EdgeKind.CAUSAL, lag=0)

        # Macro factors → asset return (lagged 1 day)
        for m in macro_factors:
            G.add_edge(m, ret_node, kind=EdgeKind.CAUSAL, lag=1)

        # Factor → covariate interactions
        for factor, cov in FACTOR_COVARIATE_INTERACTIONS:
            if factor in global_factors:
                G.add_edge(factor, f"{asset}_{cov}", kind=EdgeKind.CAUSAL, lag=0)

        # Unobserved shock → return
        shock_node = f"{asset}_shock"
        G.add_node(shock_node, kind=NodeKind.UNOBSERVED_SHOCK, asset=asset)
        G.add_edge(shock_node, ret_node, kind=EdgeKind.CAUSAL, lag=0)

    assert nx.is_directed_acyclic_graph(G), "CPCM DAG has a cycle!"
    return G


def summarize_dag(G: nx.DiGraph) -> dict:
    """Summary statistics for the DAG."""
    def count_kind(kind: NodeKind) -> int:
        return sum(1 for _, d in G.nodes(data=True) if d.get("kind") == kind)

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "global_factors": count_kind(NodeKind.GLOBAL_FACTOR),
        "macro_factors": count_kind(NodeKind.MACRO_FACTOR),
        "asset_returns": count_kind(NodeKind.ASSET_RETURN),
        "asset_covariates": count_kind(NodeKind.ASSET_COVARIATE),
        "instruments": count_kind(NodeKind.INSTRUMENT),
        "unobserved_shocks": count_kind(NodeKind.UNOBSERVED_SHOCK),
    }


def to_gml(G: nx.DiGraph) -> str:
    """Convert DAG to GML string for DoWhy compatibility."""
    lines = ["graph [", "  directed 1"]
    for node in G.nodes():
        lines.append(f'  node [ id "{node}" label "{node}" ]')
    for u, v in G.edges():
        lines.append(f'  edge [ source "{u}" target "{v}" ]')
    lines.append("]")
    return "\n".join(lines)
