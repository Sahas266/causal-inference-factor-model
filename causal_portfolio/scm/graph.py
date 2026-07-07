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

    # ── Latent confounders (one per instrumented treatment) ─────
    # u_<treatment> → treatment and u_<treatment> → every asset return.
    # This encodes the endogeneity the IV design exists for: the backdoor
    # criterion now FAILS for instrumented factors (the confounder is
    # UNOBSERVED_SHOCK, so no adjustment set can include or block it) and
    # identification falls through to the IV. Mirrors cpcm_dag.rs.
    for _iv_name, treatment, _lag in INSTRUMENTS:
        if treatment in global_factors:
            u = f"u_{treatment}"
            G.add_node(u, kind=NodeKind.UNOBSERVED_SHOCK, asset=None)
            G.add_edge(u, treatment, kind=EdgeKind.CAUSAL, lag=0)
            for asset in assets:
                G.add_edge(u, f"{asset}_return", kind=EdgeKind.CAUSAL, lag=0)

    assert nx.is_directed_acyclic_graph(G), "CPCM DAG has a cycle!"
    return G


def build_discovered_dag(assets: list[str]) -> nx.DiGraph:
    """DAG v2 — the structure the DATA supports (2026-06 discovery program).

    Differences from the hand-drawn star DAG (build_cpcm_dag):
      - Factors are AR(1) INNOVATIONS (factors/builder.py::innovation_factors),
        not z-scored levels.
      - The dominant causal direction is returns → on-chain factors, not the
        reverse: ``ew_return(t-1) → liq_flow(t)`` replicated across sample
        halves in both level and innovation space (VAR-LiNGAM), and
        ``btc_return(t) → stable_flow(t)`` appears contemporaneously (full
        sample + half 2).
      - Exactly ONE candidate factor→return edge survives orthogonalized
        estimation (DML), the innovation transform, and a circular-shift
        placebo (p=0.01): ``chain_congestion(t-1) → btc_return(t)``,
        ~+39 bps/σ. It is half-2-concentrated (2024–25) and was selected on
        this sample — edge attribute stability="candidate" until it
        validates on data after 2025-12-31.
      - No instruments: with returns upstream, lag does the purging, and the
        IV search showed even strong instruments only hurt (no confounding
        to remove at lag 1).

    Evidence: docs/causal_discovery*.md, docs/dml_effects*.md, docs/dag_v2.md.
    """
    G = nx.DiGraph()
    mkt = "btc_return"
    basket = "ew_return"
    G.add_node(mkt, kind=NodeKind.ASSET_RETURN, asset="btc")
    G.add_node(basket, kind=NodeKind.ASSET_RETURN, asset=None)
    G.add_edge(mkt, basket, kind=EdgeKind.CAUSAL, lag=0,
               stability="stable", evidence="VAR-LiNGAM B0, all runs")

    for f in GLOBAL_FACTORS:
        G.add_node(f, kind=NodeKind.GLOBAL_FACTOR, asset=None,
                   transform="ar1_innovation")

    # Returns drive on-chain state (the replicated direction).
    G.add_edge(basket, "liq_flow", kind=EdgeKind.CAUSAL, lag=1,
               stability="stable",
               evidence="VAR-LiNGAM B1, both halves, levels AND innovations")
    G.add_edge(mkt, "stable_flow", kind=EdgeKind.CAUSAL, lag=0,
               stability="semi-stable",
               evidence="VAR-LiNGAM B0, full sample + half 2")

    # The one surviving candidate factor→return edge.
    G.add_edge("chain_congestion", mkt, kind=EdgeKind.CAUSAL, lag=1,
               stability="candidate",
               evidence="DML +39bps/σ (innovations), timing placebo p=0.01; "
                        "half-2 only — needs post-2025 forward validation")

    # Per-asset returns: market beta + own shock (no factor edges claimed).
    for asset in assets:
        if asset == "btc":
            continue
        r = f"{asset}_return"
        G.add_node(r, kind=NodeKind.ASSET_RETURN, asset=asset)
        G.add_edge(mkt, r, kind=EdgeKind.CAUSAL, lag=0,
                   stability="stable", evidence="market beta")
        shock = f"{asset}_shock"
        G.add_node(shock, kind=NodeKind.UNOBSERVED_SHOCK, asset=asset)
        G.add_edge(shock, r, kind=EdgeKind.CAUSAL, lag=0,
                   stability="stable", evidence="idiosyncratic")
    btc_shock = "btc_shock"
    G.add_node(btc_shock, kind=NodeKind.UNOBSERVED_SHOCK, asset="btc")
    G.add_edge(btc_shock, mkt, kind=EdgeKind.CAUSAL, lag=0,
               stability="stable", evidence="idiosyncratic")

    assert nx.is_directed_acyclic_graph(G), "discovered DAG has a cycle!"
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
