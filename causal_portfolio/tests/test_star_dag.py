"""Tests for scm/graph.py — Star-DAG construction."""

import networkx as nx
import pytest

from causal_portfolio.scm.graph import (
    GLOBAL_FACTORS,
    MACRO_FACTORS,
    ASSET_COVARIATES,
    NodeKind,
    build_cpcm_dag,
    summarize_dag,
    to_gml,
)


class TestBuildCPCMDag:
    def test_single_asset_is_dag(self):
        G = build_cpcm_dag(["eth"])
        assert nx.is_directed_acyclic_graph(G)

    def test_multi_asset_is_dag(self):
        G = build_cpcm_dag(["btc", "eth", "sol"])
        assert nx.is_directed_acyclic_graph(G)

    def test_full_26_coin_is_dag(self):
        assets = [
            "usdc", "usdt", "usde", "btc", "eth", "bnb", "hype", "xrp",
            "pendle", "uni", "jup", "tao", "link", "zec", "ena", "morpho",
            "aero", "sol", "avax", "pol", "wlfi", "crv", "aave", "pepe",
            "shib", "doge",
        ]
        G = build_cpcm_dag(assets)
        assert nx.is_directed_acyclic_graph(G)

    def test_node_counts_single_asset(self):
        G = build_cpcm_dag(["eth"])
        summary = summarize_dag(G)
        assert summary["global_factors"] == 7
        assert summary["macro_factors"] == 7
        assert summary["asset_returns"] == 1
        assert summary["asset_covariates"] == 4
        # 1 per-asset shock + 4 latent confounders (one per instrumented factor)
        assert summary["unobserved_shocks"] == 5

    def test_node_counts_multi_asset(self):
        G = build_cpcm_dag(["btc", "eth", "sol"])
        summary = summarize_dag(G)
        assert summary["asset_returns"] == 3
        assert summary["asset_covariates"] == 12  # 4 per asset
        # 3 per-asset shocks + 4 latent confounders
        assert summary["unobserved_shocks"] == 7

    def test_latent_confounder_per_instrumented_treatment(self):
        G = build_cpcm_dag(["eth"])
        for u, treatment in [("u_liq_flow", "liq_flow"),
                             ("u_funding_basis", "funding_basis"),
                             ("u_stable_flow", "stable_flow"),
                             ("u_chain_congestion", "chain_congestion")]:
            assert G.nodes[u]["kind"] is NodeKind.UNOBSERVED_SHOCK
            assert G.has_edge(u, treatment)
            assert G.has_edge(u, "eth_return")

    def test_factor_causes_return(self):
        G = build_cpcm_dag(["eth"])
        assert G.has_edge("liq_flow", "eth_return")

    def test_macro_causes_return(self):
        G = build_cpcm_dag(["eth"])
        assert G.has_edge("dff", "eth_return")

    def test_instrument_causes_treatment(self):
        G = build_cpcm_dag(["eth"])
        assert G.has_edge("gas_spike", "liq_flow")

    def test_instrument_no_direct_to_return(self):
        G = build_cpcm_dag(["eth"])
        assert not G.has_edge("gas_spike", "eth_return")

    def test_covariate_causes_return(self):
        G = build_cpcm_dag(["eth"])
        assert G.has_edge("eth_whale_conc", "eth_return")

    def test_factor_covariate_interaction(self):
        G = build_cpcm_dag(["eth"])
        assert G.has_edge("stable_flow", "eth_whale_conc")

    def test_d_separation_assets_confounded_despite_factors(self):
        """Updated deliberately (H1 fix): the latent confounders u_<factor>
        point at every return, so conditioning on the observed factors no
        longer d-separates returns — the shared endogeneity is explicit."""
        from networkx.algorithms.d_separation import is_d_separator
        G = build_cpcm_dag(["btc", "eth"])
        cond = set(GLOBAL_FACTORS + MACRO_FACTORS)
        assert not is_d_separator(G, {"btc_return"}, {"eth_return"}, cond)

    def test_assets_correlated_unconditionally(self):
        """Without conditioning, returns share common factor parents."""
        from networkx.algorithms.d_separation import is_d_separator
        G = build_cpcm_dag(["btc", "eth"])
        assert not is_d_separator(G, {"btc_return"}, {"eth_return"}, set())


class TestSelectedDrivers:
    def test_filters_to_selected(self):
        G = build_cpcm_dag(["eth"], selected_drivers=["vixcls", "liq_flow"])
        summary = summarize_dag(G)
        # Only vixcls (macro) + liq_flow (global) = 2 factors
        assert summary["global_factors"] == 1  # liq_flow
        assert summary["macro_factors"] == 1   # vixcls

    def test_selected_still_dag(self):
        G = build_cpcm_dag(
            ["btc", "eth"],
            selected_drivers=["vixcls", "liq_flow", "stable_flow"],
        )
        assert nx.is_directed_acyclic_graph(G)


class TestToGML:
    def test_produces_valid_string(self):
        G = build_cpcm_dag(["eth"])
        gml = to_gml(G)
        assert "graph [" in gml
        assert "directed 1" in gml
        assert "eth_return" in gml
