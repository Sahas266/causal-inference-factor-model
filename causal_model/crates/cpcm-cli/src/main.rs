mod config;
mod output;
mod pipeline;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use cpcm_core::cpcm_dag::{build_cpcm_dag, summarize_dag};
use cpcm_core::identify::identify_all_effects;

use config::Config;
use output::{export_csv_summary, export_json};

#[derive(Parser)]
#[command(name = "cpcm", about = "CPCM Causal Factor Model CLI")]
struct Cli {
    /// Path to config.toml
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the full pipeline: load → factors → identify → estimate
    Run,

    /// Print the causal DAG structure and identification results
    Graph,

    /// Fetch data from Supabase and compute factors (no estimation)
    Factors,

    /// Print data coverage report
    Coverage,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Load .env from parent directory (backfill_data/.env)
    let env_paths = [".env", "../backfill_data/.env", "../../backfill_data/.env"];
    for path in &env_paths {
        if std::path::Path::new(path).exists() {
            dotenvy::from_path(path).ok();
            break;
        }
    }

    let cli = Cli::parse();
    let config = Config::load(&cli.config).unwrap_or_else(|e| {
        tracing::warn!("Could not load config ({}), using defaults", e);
        Config::default()
    });

    match cli.command {
        Commands::Run => {
            let result = pipeline::run_pipeline(&config).await?;
            println!(
                "\nPipeline complete: {} assets, {} dates, {}/{} identified, {} estimated",
                result.n_assets,
                result.n_dates,
                result.n_identified,
                result.n_assets * 14, // 7 factors + 7 macro
                result.n_estimated,
            );

            // Export results
            let json = export_json(&result.results)?;
            std::fs::write("results.json", &json)?;
            let csv = export_csv_summary(&result.results);
            std::fs::write("results.csv", &csv)?;
            println!("Results written to results.json and results.csv");
        }

        Commands::Graph => {
            let asset_refs: Vec<&str> = config.assets.include.iter().map(|s| s.as_str()).collect();
            let dag = build_cpcm_dag(&asset_refs);
            let summary = summarize_dag(&dag);

            println!("=== CPCM Causal DAG ===\n");
            println!("Nodes: {}", summary.total_nodes);
            println!("  Global Factors:    {}", summary.global_factors);
            println!("  Macro Factors:     {}", summary.macro_factors);
            println!("  Asset Returns:     {}", summary.asset_returns);
            println!("  Asset Covariates:  {}", summary.asset_covariates);
            println!("  Instruments:       {}", summary.instruments);
            println!("  Unobserved Shocks: {}", summary.unobserved_shocks);
            println!("Edges: {}", summary.total_edges);
            println!();

            // Identification
            let id_results = identify_all_effects(&dag);
            let identified: Vec<_> = id_results.iter().filter(|r| r.identified).collect();
            let not_identified: Vec<_> = id_results.iter().filter(|r| !r.identified).collect();

            println!("=== Identification Results ===\n");
            println!("Identified: {}/{}", identified.len(), id_results.len());

            if !not_identified.is_empty() {
                println!("\nNot Identified:");
                for r in &not_identified {
                    println!("  {} → {}: {}", r.treatment, r.outcome, r.reason);
                }
            }
        }

        Commands::Factors => {
            let result = pipeline::run_pipeline(&config).await?;
            println!("Factor computation complete ({} dates)", result.n_dates);
        }

        Commands::Coverage => {
            let result = pipeline::run_pipeline(&config).await?;
            println!("Coverage report complete ({} assets)", result.n_assets);
        }
    }

    Ok(())
}
