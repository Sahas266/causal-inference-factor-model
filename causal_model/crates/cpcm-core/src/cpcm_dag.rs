use crate::dag::CausalDag;
use crate::types::{EdgeKind, NodeKind};

/// The 7 global DeFi causal factors from the CPCM paper.
pub const GLOBAL_FACTORS: &[&str] = &[
    "liq_flow",
    "stable_flow",
    "funding_basis",
    "chain_congestion",
    "staking_yield",
    "mev_pressure",
    "cex_dex_flow",
];

/// Macro factors from FRED that affect all asset returns with a 1-day lag.
pub const MACRO_FACTORS: &[&str] = &[
    "dff",      // Fed funds rate
    "dgs10",    // 10-year treasury
    "vixcls",   // VIX
    "t10y2y",   // Yield curve slope
    "cpiaucsl", // CPI
    "m2sl",     // M2 money supply
    "dtwexbgs", // Trade-weighted dollar index
];

/// Per-asset covariate suffixes. Each asset gets `{asset}_{cov}`.
pub const ASSET_COVARIATES: &[&str] = &[
    "whale_conc",
    "protocol_rev",
    "emissions",
    "chain_activity",
];

/// Instrument definitions: (iv_name, instruments_for_factor, lag).
pub const INSTRUMENTS: &[(&str, &str, i32)] = &[
    ("gas_spike", "liq_flow", 1),
    ("liquidation_level", "funding_basis", 1),
    ("stablecoin_mint", "stable_flow", 1),
    ("protocol_event", "chain_congestion", 1),
];

/// Build the full CPCM causal DAG for a set of assets.
///
/// Structure:
/// - 7 global factors → each asset return (lag=0)
/// - 7 macro factors → each asset return (lag=1)
/// - 4 per-asset covariates → own asset return
/// - F→Xi interactions (stable_flow→whale_conc, etc.)
/// - Unobserved shocks → each asset return
/// - Instruments → their respective treatment factors
pub fn build_cpcm_dag(assets: &[&str]) -> CausalDag {
    let mut dag = CausalDag::new();

    // ── Global factors ───────────────────────────────────────────────
    for &f in GLOBAL_FACTORS {
        dag.add_node(f, NodeKind::GlobalFactor, None);
    }

    // ── Macro factors ────────────────────────────────────────────────
    for &m in MACRO_FACTORS {
        dag.add_node(m, NodeKind::MacroFactor, None);
    }

    // ── Instruments ──────────────────────────────────────────────────
    for &(iv_name, treatment, lag) in INSTRUMENTS {
        dag.add_node(iv_name, NodeKind::Instrument, None);
        dag.add_edge(iv_name, treatment, EdgeKind::Instrumental, lag);
    }

    // ── Per-asset nodes ──────────────────────────────────────────────
    for &asset in assets {
        let ret = format!("{asset}_return");
        dag.add_node(&ret, NodeKind::AssetReturn, Some(asset));

        // Asset covariates → own return
        for &cov in ASSET_COVARIATES {
            let cov_name = format!("{asset}_{cov}");
            dag.add_node(&cov_name, NodeKind::AssetCovariate, Some(asset));
            dag.add_edge(&cov_name, &ret, EdgeKind::Causal, 0);
        }

        // Global factors → asset return
        for &f in GLOBAL_FACTORS {
            dag.add_edge(f, &ret, EdgeKind::Causal, 0);
        }

        // Macro factors → asset return (lagged 1 day)
        for &m in MACRO_FACTORS {
            dag.add_edge(m, &ret, EdgeKind::Causal, 1);
        }

        // ── F → Xi interactions ──────────────────────────────────────
        // StableFlow → whale behaviour
        dag.add_edge("stable_flow", &format!("{asset}_whale_conc"), EdgeKind::Causal, 0);
        // Chain congestion → protocol volume/revenue
        dag.add_edge("chain_congestion", &format!("{asset}_protocol_rev"), EdgeKind::Causal, 0);
        // Staking yield → emissions dynamics
        dag.add_edge("staking_yield", &format!("{asset}_emissions"), EdgeKind::Causal, 0);

        // ── Unobserved shock ─────────────────────────────────────────
        let shock = format!("{asset}_shock");
        dag.add_node(&shock, NodeKind::UnobservedShock, Some(asset));
        dag.add_edge(&shock, &ret, EdgeKind::Causal, 0);
    }

    // ── Latent confounders (one per instrumented treatment) ──────────
    // u_<treatment> → treatment and u_<treatment> → every asset return.
    // This encodes the endogeneity the IV design exists for: the backdoor
    // criterion now FAILS for instrumented factors (the confounder is
    // unobservable, so no adjustment set blocks it) and identification
    // falls through to the IV.
    for &(_, treatment, _) in INSTRUMENTS {
        let u = format!("u_{treatment}");
        dag.add_node(&u, NodeKind::UnobservedShock, None);
        dag.add_edge(&u, treatment, EdgeKind::Causal, 0);
        for &asset in assets {
            dag.add_edge(&u, &format!("{asset}_return"), EdgeKind::Causal, 0);
        }
    }

    debug_assert!(dag.is_dag(), "CPCM DAG has a cycle!");
    dag
}

/// Summary of the DAG structure.
#[derive(Debug)]
pub struct DagSummary {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub global_factors: usize,
    pub macro_factors: usize,
    pub asset_returns: usize,
    pub asset_covariates: usize,
    pub instruments: usize,
    pub unobserved_shocks: usize,
}

pub fn summarize_dag(dag: &CausalDag) -> DagSummary {
    DagSummary {
        total_nodes: dag.node_count(),
        total_edges: dag.edge_count(),
        global_factors: dag.nodes_of_kind(NodeKind::GlobalFactor).len(),
        macro_factors: dag.nodes_of_kind(NodeKind::MacroFactor).len(),
        asset_returns: dag.nodes_of_kind(NodeKind::AssetReturn).len(),
        asset_covariates: dag.nodes_of_kind(NodeKind::AssetCovariate).len(),
        instruments: dag.nodes_of_kind(NodeKind::Instrument).len(),
        unobserved_shocks: dag.nodes_of_kind(NodeKind::UnobservedShock).len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsep::d_separated;
    use crate::identify::{check_iv_validity, backdoor_adjustment_set};

    #[test]
    fn test_build_single_asset() {
        let dag = build_cpcm_dag(&["eth"]);
        assert!(dag.is_dag());

        let summary = summarize_dag(&dag);
        assert_eq!(summary.global_factors, 7);
        assert_eq!(summary.macro_factors, 7);
        assert_eq!(summary.asset_returns, 1);
        assert_eq!(summary.asset_covariates, 4);
        assert_eq!(summary.instruments, 4);
        // 1 per-asset shock + 4 latent confounders (one per instrumented factor)
        assert_eq!(summary.unobserved_shocks, 5);
    }

    #[test]
    fn test_build_multi_asset() {
        let assets = ["btc", "eth", "sol"];
        let dag = build_cpcm_dag(&assets);
        assert!(dag.is_dag());

        let summary = summarize_dag(&dag);
        // 7 global + 7 macro + 4 instruments + 3*(1 return + 4 covariates + 1 shock)
        assert_eq!(summary.asset_returns, 3);
        assert_eq!(summary.asset_covariates, 12); // 4 per asset
        // 3 per-asset shocks + 4 latent confounders
        assert_eq!(summary.unobserved_shocks, 7);
    }

    #[test]
    fn test_factor_causes_return() {
        let dag = build_cpcm_dag(&["eth"]);
        // liq_flow should NOT be d-separated from eth_return (direct causal edge)
        assert!(!d_separated(&dag, "liq_flow", "eth_return", &[]));
    }

    #[test]
    fn test_macro_causes_return() {
        let dag = build_cpcm_dag(&["eth"]);
        assert!(!d_separated(&dag, "dff", "eth_return", &[]));
    }

    #[test]
    fn test_asset_returns_correlated_via_common_factors() {
        let dag = build_cpcm_dag(&["btc", "eth"]);
        // btc_return and eth_return share all global + macro factors
        assert!(!d_separated(&dag, "btc_return", "eth_return", &[]));
    }

    #[test]
    fn test_asset_returns_confounded_even_conditioning_on_factors() {
        let dag = build_cpcm_dag(&["btc", "eth"]);
        // Updated deliberately (H1 fix): the latent confounders u_<factor> point
        // at every return, so conditioning on the observed factors no longer
        // d-separates returns — the shared endogeneity is now explicit.
        let mut cond: Vec<&str> = GLOBAL_FACTORS.to_vec();
        cond.extend_from_slice(MACRO_FACTORS);
        assert!(!d_separated(&dag, "btc_return", "eth_return", &cond));
    }

    #[test]
    fn test_iv_gas_spike_valid_for_liq_flow() {
        let dag = build_cpcm_dag(&["eth"]);
        let result = check_iv_validity(&dag, "gas_spike", "liq_flow", "eth_return");
        assert!(result.relevant, "gas_spike should be relevant for liq_flow");
        // Exclusion: gas_spike → liq_flow → eth_return, but gas_spike has no direct
        // edge to eth_return. Conditioning on liq_flow should block the path.
        assert!(
            result.excludable,
            "gas_spike should be excludable given liq_flow"
        );
        assert!(result.valid);
    }

    #[test]
    fn test_instrumented_factor_needs_iv_not_backdoor() {
        // Updated deliberately (H1 fix): liq_flow now has the latent confounder
        // u_liq_flow → {liq_flow, returns}, so NO backdoor set exists and the
        // IV fallback must be used.
        let dag = build_cpcm_dag(&["eth"]);
        let adj = backdoor_adjustment_set(&dag, "liq_flow", "eth_return");
        assert!(adj.is_none(), "backdoor must fail for instrumented factors");
        let iv = check_iv_validity(&dag, "gas_spike", "liq_flow", "eth_return");
        assert!(iv.valid, "gas_spike must remain a valid IV: {}", iv.reason);
    }

    #[test]
    fn test_identify_all_uses_iv_for_instrumented_factors() {
        use crate::identify::{identify_all_effects, IdentificationMethod};
        let dag = build_cpcm_dag(&["eth"]);
        let results = identify_all_effects(&dag);
        for r in &results {
            match r.treatment.as_str() {
                // instrumented factors: backdoor fails, IV succeeds
                "liq_flow" | "funding_basis" | "stable_flow" | "chain_congestion" => {
                    assert!(r.identified, "{} should be IV-identified", r.treatment);
                    assert!(
                        matches!(r.method, IdentificationMethod::Iv { .. }),
                        "{} should use IV, got {:?}",
                        r.treatment,
                        r.method
                    );
                }
                // unconfounded factors: still backdoor-identified
                _ => assert_eq!(r.method, IdentificationMethod::Backdoor, "{}", r.treatment),
            }
        }
    }

    #[test]
    fn test_full_26_coin_dag() {
        let assets: Vec<&str> = vec![
            "usdc", "usdt", "usde", "btc", "eth", "bnb", "hype", "xrp", "pendle",
            "uni", "jup", "tao", "link", "zec", "ena", "morpho", "aero", "sol",
            "avax", "pol", "wlfi", "crv", "aave", "pepe", "shib", "doge",
        ];
        let dag = build_cpcm_dag(&assets);
        assert!(dag.is_dag());

        let summary = summarize_dag(&dag);
        assert_eq!(summary.asset_returns, 26);
        assert_eq!(summary.asset_covariates, 26 * 4);
        // 26 per-asset shocks + 4 latent confounders
        assert_eq!(summary.unobserved_shocks, 30);
        // Total nodes: 7 + 7 + 4 + 26*(1+4+1) + 4 latent confounders = 178
        assert_eq!(summary.total_nodes, 7 + 7 + 4 + 26 * 6 + 4);
    }
}
