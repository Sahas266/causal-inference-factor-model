use chrono::NaiveDate;
use tracing::info;

use cpcm_core::cpcm_dag::{build_cpcm_dag, summarize_dag, GLOBAL_FACTORS, MACRO_FACTORS, ASSET_COVARIATES};
use cpcm_core::identify::identify_all_effects;
use cpcm_data::client::SupabaseClient;
use cpcm_data::frame::{df_to_hashmap, load_parquet, pivot_to_panel, print_summary, save_parquet};
use cpcm_data::types::SupabaseConfig;
use cpcm_estimate::panel::{run_panel_estimation, PanelMode};
use cpcm_estimate::types::EstimationResult;
use cpcm_factors::registry::{compute_all, merge_all, print_coverage};

use crate::config::Config;

/// Full pipeline result.
pub struct PipelineResult {
    pub n_assets: usize,
    pub n_dates: usize,
    pub n_identified: usize,
    pub n_estimated: usize,
    pub results: Vec<EstimationResult>,
}

/// Run the full CPCM pipeline: load → factors → identify → estimate.
pub async fn run_pipeline(config: &Config) -> anyhow::Result<PipelineResult> {
    // ── 1. Load data (cache or Supabase) ────────────────────────────
    let cache_path = &config.data.cache_path;
    let df = match load_parquet(cache_path, config.data.cache_ttl_hours) {
        Ok(Some(cached)) => {
            info!("Using cached panel from {cache_path}");
            cached
        }
        _ => {
            info!("Loading data from Supabase...");
            let sb_config = SupabaseConfig {
                url: config.supabase.url.clone(),
                key: config.supabase.key_env.clone(),
            };
            let client = SupabaseClient::new(&sb_config)?;
            client.health_check().await?;

            let table = if config.data.use_best_view {
                "asset_metrics_best"
            } else {
                "asset_metrics"
            };

            let asset_refs_tmp: Vec<&str> =
                config.assets.include.iter().map(|s| s.as_str()).collect();
            let mut all_assets = asset_refs_tmp;
            // Always fetch stablecoins (for stable_flow factor) and macro data
            for extra in &["usdc", "usdt", "usde", "macro"] {
                if !all_assets.contains(extra) {
                    all_assets.push(extra);
                }
            }

            let rows = client
                .fetch_metrics(
                    table,
                    &all_assets,
                    &[], // all metrics
                    &config.data.start_date,
                    &config.data.end_date,
                )
                .await?;

            info!("Fetched {} rows from Supabase", rows.len());

            let start = NaiveDate::parse_from_str(&config.data.start_date, "%Y-%m-%d")?;
            let end = NaiveDate::parse_from_str(&config.data.end_date, "%Y-%m-%d")?;
            let mut panel_df = pivot_to_panel(rows, start, end)?;

            // Save to cache
            if let Err(e) = save_parquet(&mut panel_df, cache_path) {
                tracing::warn!("Failed to save cache: {e}");
            }

            panel_df
        }
    };

    let asset_refs: Vec<&str> = config.assets.include.iter().map(|s| s.as_str()).collect();
    print_summary(&df);

    let data = df_to_hashmap(&df)?;
    let n_dates = df.height();

    // ── 2. Compute factors, returns, covariates, instruments ─────────
    info!("Computing factors and covariates...");
    let panel = compute_all(&data, &config.assets.include, n_dates);
    print_coverage(&panel, &config.assets.include);

    // ── 3. Build DAG and identify effects ────────────────────────────
    info!("Building causal DAG...");
    let dag = build_cpcm_dag(&asset_refs);
    let summary = summarize_dag(&dag);
    info!(
        "DAG: {} nodes, {} edges ({} factors, {} returns, {} covariates, {} instruments)",
        summary.total_nodes,
        summary.total_edges,
        summary.global_factors + summary.macro_factors,
        summary.asset_returns,
        summary.asset_covariates,
        summary.instruments,
    );

    let id_results = identify_all_effects(&dag);
    let n_identified = id_results.iter().filter(|r| r.identified).count();
    info!(
        "Identification: {}/{} effects identified",
        n_identified,
        id_results.len()
    );

    // ── 4. Estimation ────────────────────────────────────────────────
    info!("Running estimation...");
    let merged = merge_all(&panel);

    let factor_names: Vec<String> = GLOBAL_FACTORS.iter().map(|&s| s.to_string()).collect();
    let macro_names: Vec<String> = MACRO_FACTORS
        .iter()
        .map(|&s| s.to_lowercase())
        .collect();
    let cov_suffixes: Vec<String> = ASSET_COVARIATES.iter().map(|&s| s.to_string()).collect();

    let mode = match config.estimation.mode.as_str() {
        "pooled" => PanelMode::Pooled,
        _ => PanelMode::PerAsset,
    };

    let est_results = run_panel_estimation(
        &merged,
        &config.assets.include,
        &factor_names,
        &macro_names,
        &cov_suffixes,
        &panel.instrument_map,
        mode,
    )?;

    // ── 5. Print results ─────────────────────────────────────────────
    println!("\n=== Estimation Results ===\n");
    for result in &est_results {
        println!("--- {} (n={}) ---", result.asset, result.ols.n_obs);
        println!("  R² = {:.4}, Adj R² = {:.4}", result.ols.r_squared, result.ols.adj_r_squared);
        println!("  DW = {:.3}", result.diagnostics.durbin_watson);

        println!("  OLS Coefficients (standardized):");
        for (name, (&coef, &p)) in result
            .ols
            .feature_names
            .iter()
            .zip(result.ols.coefficients.iter().zip(result.ols.p_values.iter()))
        {
            let sig = if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            };
            println!("    {name:25} {coef:>12.4e}  (p={p:.4}) {sig}");
        }

        if let Some(ref tsls) = result.tsls {
            println!("  2SLS Coefficients:");
            println!("    First-stage F: {:?}", tsls.first_stage_f);
            println!("    Hausman stat: {:.3} (p={:.4})", tsls.hausman_stat, tsls.hausman_p);
            for (name, (&coef, &p)) in tsls
                .feature_names
                .iter()
                .zip(tsls.coefficients.iter().zip(tsls.p_values.iter()))
            {
                let sig = if p < 0.01 {
                    "***"
                } else if p < 0.05 {
                    "**"
                } else if p < 0.10 {
                    "*"
                } else {
                    ""
                };
                println!("    {name:25} {coef:>12.4e}  (p={p:.4}) {sig}");
            }
        }

        // VIF warnings
        for (name, vif) in &result.diagnostics.vif {
            if *vif > 10.0 {
                println!("  WARNING: VIF({name}) = {vif:.1} > 10 (multicollinearity)");
            }
        }
        println!();
    }

    Ok(PipelineResult {
        n_assets: config.assets.include.len(),
        n_dates,
        n_identified,
        n_estimated: est_results.len(),
        results: est_results,
    })
}
