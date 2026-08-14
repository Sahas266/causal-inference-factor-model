use chrono::NaiveDate;
use tracing::info;

use cpcm_core::cpcm_dag::{
    build_cpcm_dag, summarize_dag, ASSET_COVARIATES, GLOBAL_FACTORS, MACRO_FACTORS,
};
use cpcm_core::identify::{identify_all_effects, IdentificationMethod};
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

    // Identification drives estimation: factors identified via IV are
    // instrumented in 2SLS; BACKDOOR factors are estimated by OLS with the
    // adjustment set S as controls (the panel regression already includes all
    // factors + covariates, a superset of every S the star-DAG produces).
    let mut iv_chosen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut method_by_factor: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for r in &id_results {
        match &r.method {
            IdentificationMethod::Iv { instrument } => {
                iv_chosen
                    .entry(r.treatment.clone())
                    .or_insert_with(|| instrument.clone());
                method_by_factor
                    .entry(r.treatment.clone())
                    .or_insert_with(|| format!("2SLS (IV: {instrument})"));
            }
            IdentificationMethod::Backdoor => {
                method_by_factor
                    .entry(r.treatment.clone())
                    .or_insert_with(|| format!("OLS (backdoor, controls: {:?})", r.adjustment_set));
            }
            IdentificationMethod::NotIdentified => {
                method_by_factor
                    .entry(r.treatment.clone())
                    .or_insert_with(|| "NOT IDENTIFIED (reported descriptively)".to_string());
            }
        }
    }
    for (factor, method) in &method_by_factor {
        info!("Identification chose for {factor}: {method}");
    }

    // ── 4. Estimation ────────────────────────────────────────────────
    info!("Running estimation...");
    let mut merged = merge_all(&panel);

    // Apply DAG edge `lag` attributes to treatment→return edges: a macro
    // factor with lag=1 must enter the regression as F(t-1) explaining r(t).
    // (Instrument columns are already lagged at construction; their edges are
    // iv→factor, not →return, so they're untouched here.)
    let mut col_lags: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (from, to, edge) in dag.edges() {
        if to.ends_with("_return") && edge.lag > 0 {
            let entry = col_lags.entry(from.to_string()).or_insert(0);
            *entry = (*entry).max(edge.lag as usize);
        }
    }
    for (col, lag) in &col_lags {
        if let Some(v) = merged.get_mut(col) {
            let mut shifted = vec![f64::NAN; v.len()];
            if *lag <= v.len() {
                shifted[*lag..].copy_from_slice(&v[..v.len() - *lag]);
            }
            *v = shifted;
            info!("Applied DAG lag {lag} to '{col}' before estimation");
        }
    }

    // Only instrument the factors identification actually chose IV for.
    let instrument_map: std::collections::HashMap<String, String> = panel
        .instrument_map
        .iter()
        .filter(|(factor, _)| iv_chosen.contains_key(*factor))
        .map(|(f, z)| (f.clone(), z.clone()))
        .collect();

    let factor_names: Vec<String> = GLOBAL_FACTORS.iter().map(|&s| s.to_string()).collect();
    let macro_names: Vec<String> = MACRO_FACTORS.iter().map(|&s| s.to_lowercase()).collect();
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
        &instrument_map,
        mode,
    )?;

    // ── 5. Print results ─────────────────────────────────────────────
    let stars = |p: f64| {
        if p < 0.01 {
            "***"
        } else if p < 0.05 {
            "**"
        } else if p < 0.10 {
            "*"
        } else {
            ""
        }
    };
    println!("\n=== Estimation Results ===\n");
    println!("Estimator chosen by identification, per factor:");
    {
        let mut factors: Vec<&String> = method_by_factor.keys().collect();
        factors.sort();
        for factor in factors {
            println!("  {factor:20} -> {}", method_by_factor[factor]);
        }
    }
    println!();
    for result in &est_results {
        println!("--- {} (n={}) ---", result.asset, result.ols.n_obs);
        println!(
            "  R² = {:.4}, Adj R² = {:.4}",
            result.ols.r_squared, result.ols.adj_r_squared
        );
        println!("  DW = {:.3}", result.diagnostics.durbin_watson);

        println!("  OLS Coefficients (standardized; stars keyed to HAC p):");
        for (j, name) in result.ols.feature_names.iter().enumerate() {
            let coef = result.ols.coefficients[j];
            let p = result.ols.p_values[j];
            let t_hac = result.ols.hac_t_stats[j];
            let p_hac = result.ols.hac_p_values[j];
            let sig = stars(p_hac);
            println!(
                "    {name:25} {coef:>12.4e}  (p={p:.4}, HAC t={t_hac:+.2} p={p_hac:.4}) {sig}"
            );
        }

        if let Some(ref tsls) = result.tsls {
            println!("  2SLS Coefficients (primary for IV-identified factors):");
            println!("    First-stage partial F: {:?}", tsls.first_stage_f);
            match (tsls.hausman_stat, tsls.hausman_p) {
                (Some(h), Some(hp)) => println!("    Hausman stat: {h:.3} (p={hp:.4})"),
                _ => println!("    Hausman: n/a (variance difference not positive definite)"),
            }
            for (j, name) in tsls.feature_names.iter().enumerate() {
                let coef = tsls.coefficients[j];
                let p = tsls.p_values[j];
                let t_hac = tsls.hac_t_stats[j];
                let p_hac = tsls.hac_p_values[j];
                let sig = stars(p_hac);
                println!(
                    "    {name:25} {coef:>12.4e}  (p={p:.4}, HAC t={t_hac:+.2} p={p_hac:.4}) {sig}"
                );
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
