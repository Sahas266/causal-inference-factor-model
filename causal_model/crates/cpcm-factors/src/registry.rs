use std::collections::HashMap;

use crate::asset_covariates::compute_asset_covariates;
use crate::global_factors::compute_global_factors;
use crate::instruments::compute_instruments;
use crate::returns::compute_all_returns;

/// Full factor computation result.
pub struct FactorPanel {
    /// Global factor time series (7 factors).
    pub global_factors: HashMap<String, Vec<f64>>,
    /// Macro factor time series (FRED series: dff, vixcls, etc.).
    pub macro_factors: HashMap<String, Vec<f64>>,
    /// Per-asset covariate time series.
    pub asset_covariates: HashMap<String, Vec<f64>>,
    /// Log returns per asset.
    pub returns: HashMap<String, Vec<f64>>,
    /// Instrumental variable time series.
    pub instruments: HashMap<String, Vec<f64>>,
    /// Maps factor name → instrument name.
    pub instrument_map: HashMap<String, String>,
    /// Number of dates in the panel.
    pub n_dates: usize,
}

/// FRED macro series names that we map from `macro_X` columns to lowercase `x`.
const MACRO_SERIES: &[&str] = &[
    "DFF", "DGS10", "VIXCLS", "T10Y2Y", "CPIAUCSL", "M2SL", "DTWEXBGS",
    // Additional FRED series available in the database
    "DGS2", "DGS30", "DFEDTARU", "T10Y3M",
    "CPILFESL", "PCEPI", "PCEPILFE", "T5YIE", "T10YIE", "MICH",
    "WALCL", "RRPONTSYD",
    "BAMLH0A0HYM2", "TEDRATE",
    "UNRATE", "PAYEMS", "ICSA", "GDPC1", "INDPRO",
    "DCOILWTICO", "PPIACO",
    "NFCI", "STLFSI2",
];

/// Compute all factors, covariates, returns, and instruments from raw data.
pub fn compute_all(
    data: &HashMap<String, Vec<f64>>,
    assets: &[String],
    n_dates: usize,
) -> FactorPanel {
    let returns = compute_all_returns(data, assets);
    let global_factors = compute_global_factors(data, n_dates);
    let asset_covariates = compute_asset_covariates(data, assets, n_dates);
    let (instruments, instrument_map) = compute_instruments(data, n_dates);

    // Extract macro factors: map `macro_DFF` → `dff`, `macro_VIXCLS` → `vixcls`, etc.
    // Z-score for comparability with other standardized factors.
    let mut macro_factors = HashMap::new();
    for &series in MACRO_SERIES {
        let col_name = format!("macro_{series}");
        if let Some(values) = data.get(&col_name) {
            let valid = values.iter().filter(|v| !v.is_nan()).count();
            if valid > 0 {
                macro_factors.insert(series.to_lowercase(), z_score_vec(values));
            }
        }
    }
    if !macro_factors.is_empty() {
        tracing::info!(
            "Extracted {} macro factors from panel data",
            macro_factors.len()
        );
    }

    FactorPanel {
        global_factors,
        macro_factors,
        asset_covariates,
        returns,
        instruments,
        instrument_map,
        n_dates,
    }
}

/// Merge all factor panel data into a single HashMap for estimation.
pub fn merge_all(panel: &FactorPanel) -> HashMap<String, Vec<f64>> {
    let mut merged = HashMap::new();

    for (k, v) in &panel.global_factors {
        merged.insert(k.clone(), v.clone());
    }
    for (k, v) in &panel.macro_factors {
        merged.insert(k.clone(), v.clone());
    }
    for (k, v) in &panel.asset_covariates {
        merged.insert(k.clone(), v.clone());
    }
    for (k, v) in &panel.returns {
        merged.insert(k.clone(), v.clone());
    }
    for (k, v) in &panel.instruments {
        merged.insert(k.clone(), v.clone());
    }

    merged
}

/// Z-score normalization: (x - mean) / std. NaN-safe.
fn z_score_vec(x: &[f64]) -> Vec<f64> {
    let valid: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return x.to_vec();
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64;
    let std = var.sqrt();
    if std < 1e-15 {
        return vec![0.0; x.len()];
    }
    x.iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { (v - mean) / std })
        .collect()
}

/// Print a coverage report: which factors/covariates have data.
pub fn print_coverage(panel: &FactorPanel, assets: &[String]) {
    println!("\n=== Factor Coverage Report ===\n");

    println!("Global Factors ({} dates):", panel.n_dates);
    for (name, values) in &panel.global_factors {
        let valid = values.iter().filter(|v| !v.is_nan()).count();
        let pct = (valid as f64 / panel.n_dates as f64) * 100.0;
        let status = if pct > 50.0 { "OK" } else if pct > 0.0 { "SPARSE" } else { "MISSING" };
        println!("  {name:20} {valid:>5}/{} ({pct:5.1}%) [{status}]", panel.n_dates);
    }

    println!("\nMacro Factors:");
    for (name, values) in &panel.macro_factors {
        let valid = values.iter().filter(|v| !v.is_nan()).count();
        let pct = (valid as f64 / panel.n_dates as f64) * 100.0;
        let status = if pct > 50.0 { "OK" } else if pct > 0.0 { "SPARSE" } else { "MISSING" };
        println!("  {name:20} {valid:>5}/{} ({pct:5.1}%) [{status}]", panel.n_dates);
    }

    println!("\nAsset Returns:");
    for asset in assets {
        let key = format!("{asset}_return");
        let (valid, total) = match panel.returns.get(&key) {
            Some(v) => (v.iter().filter(|x| !x.is_nan()).count(), v.len()),
            None => (0, 0),
        };
        let status = if valid > 0 { "OK" } else { "MISSING" };
        println!("  {key:20} {valid:>5}/{total} [{status}]");
    }

    println!("\nInstrumental Variables:");
    for (name, values) in &panel.instruments {
        let valid = values.iter().filter(|v| !v.is_nan()).count();
        let spikes = values.iter().filter(|&&v| v == 1.0).count();
        println!("  {name:20} {valid:>5} valid, {spikes} spikes");
    }

    println!("\nInstrument → Factor Mapping:");
    for (factor, iv) in &panel.instrument_map {
        println!("  {iv:20} → {factor}");
    }
}
