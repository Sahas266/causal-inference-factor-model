use nalgebra::{DMatrix, DVector};

use crate::diagnostics::compute_diagnostics;
use crate::ols::{add_intercept, ols};
use crate::tsls::tsls;
use crate::types::EstimationResult;

/// How to structure the panel regression.
#[derive(Debug, Clone, Copy)]
pub enum PanelMode {
    /// Run 26 separate regressions (one per asset).
    PerAsset,
    /// Stack all assets with asset fixed effects.
    Pooled,
}

/// Run panel estimation across multiple assets.
///
/// For each asset, regresses `{asset}_return` on the global factors + macro factors
/// + asset covariates. If instruments are available, also runs 2SLS.
///
/// # Arguments
/// * `data` - A map from column name to its data vector (all same length for per-asset,
///   or stacked for pooled).
/// * `assets` - List of asset tickers.
/// * `factor_names` - Names of global factor columns.
/// * `macro_names` - Names of macro factor columns.
/// * `covariate_suffixes` - Suffixes for per-asset covariates (e.g., "whale_conc").
/// * `instrument_map` - Maps factor name → instrument column name.
/// * `mode` - Per-asset or pooled.
pub fn run_panel_estimation(
    data: &std::collections::HashMap<String, Vec<f64>>,
    assets: &[String],
    factor_names: &[String],
    macro_names: &[String],
    covariate_suffixes: &[String],
    instrument_map: &std::collections::HashMap<String, String>,
    mode: PanelMode,
) -> anyhow::Result<Vec<EstimationResult>> {
    match mode {
        PanelMode::PerAsset => run_per_asset(
            data,
            assets,
            factor_names,
            macro_names,
            covariate_suffixes,
            instrument_map,
        ),
        PanelMode::Pooled => run_pooled(
            data,
            assets,
            factor_names,
            macro_names,
            covariate_suffixes,
            instrument_map,
        ),
    }
}

fn run_per_asset(
    data: &std::collections::HashMap<String, Vec<f64>>,
    assets: &[String],
    factor_names: &[String],
    macro_names: &[String],
    covariate_suffixes: &[String],
    instrument_map: &std::collections::HashMap<String, String>,
) -> anyhow::Result<Vec<EstimationResult>> {
    let mut results = Vec::with_capacity(assets.len());

    for asset in assets {
        let return_col = format!("{asset}_return");
        let y_data = match data.get(&return_col) {
            Some(d) => d,
            None => {
                tracing::warn!("No return data for {asset}, skipping");
                continue;
            }
        };
        let n = y_data.len();
        if n < 30 {
            tracing::warn!("{asset}: only {n} observations, skipping (need >= 30)");
            continue;
        }

        // Build feature matrix: [factors, macro_factors, asset_covariates]
        let mut feature_cols: Vec<Vec<f64>> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();

        // Global factors
        for f in factor_names {
            if let Some(col) = data.get(f) {
                feature_cols.push(col.clone());
                feature_names.push(f.clone());
            }
        }

        // Macro factors
        for m in macro_names {
            if let Some(col) = data.get(m) {
                feature_cols.push(col.clone());
                feature_names.push(m.clone());
            }
        }

        // Asset covariates
        for suffix in covariate_suffixes {
            let col_name = format!("{asset}_{suffix}");
            if let Some(col) = data.get(&col_name) {
                feature_cols.push(col.clone());
                feature_names.push(col_name);
            }
        }

        if feature_cols.is_empty() {
            tracing::warn!("{asset}: no features available, skipping");
            continue;
        }

        // Drop columns that are mostly NaN (< 10% valid data)
        let min_valid = n / 10;
        let mut keep_indices: Vec<usize> = Vec::new();
        for (j, col) in feature_cols.iter().enumerate() {
            let valid_count = col.iter().take(n).filter(|v| !v.is_nan()).count();
            if valid_count > min_valid {
                keep_indices.push(j);
            } else {
                tracing::warn!(
                    "{asset}: dropping feature '{}' ({valid_count}/{n} valid)",
                    feature_names[j]
                );
            }
        }
        feature_cols = keep_indices
            .iter()
            .map(|&j| feature_cols[j].clone())
            .collect();
        feature_names = keep_indices
            .iter()
            .map(|&j| feature_names[j].clone())
            .collect();

        if feature_cols.is_empty() {
            tracing::warn!("{asset}: no features with sufficient data, skipping");
            continue;
        }

        // Find complete cases (no NaN in any remaining column)
        let valid_rows: Vec<usize> = (0..n)
            .filter(|&i| {
                !y_data[i].is_nan()
                    && feature_cols
                        .iter()
                        .all(|col| i < col.len() && !col[i].is_nan())
            })
            .collect();

        if valid_rows.len() < 30 {
            tracing::warn!(
                "{asset}: only {} complete cases, skipping",
                valid_rows.len()
            );
            continue;
        }

        let n_valid = valid_rows.len();
        let y = DVector::from_iterator(n_valid, valid_rows.iter().map(|&i| y_data[i]));

        let k = feature_cols.len();
        let mut x_data = vec![0.0; n_valid * k];
        for (j, col) in feature_cols.iter().enumerate() {
            for (row_idx, &orig_idx) in valid_rows.iter().enumerate() {
                x_data[row_idx + j * n_valid] = col[orig_idx];
            }
        }
        let x_raw = DMatrix::from_column_slice(n_valid, k, &x_data);
        let x = add_intercept(&x_raw);

        let mut all_names = vec!["intercept".to_string()];
        all_names.extend(feature_names.iter().cloned());

        // ── OLS ──────────────────────────────────────────────────────
        let ols_result = ols(&y, &x, &all_names)?;

        // ── Diagnostics ──────────────────────────────────────────────
        let diagnostics = compute_diagnostics(&ols_result.residuals, &x, &all_names)?;

        // ── 2SLS (if instruments available) ──────────────────────────
        let tsls_result = try_tsls(
            &y,
            &x_raw,
            &feature_names,
            data,
            &valid_rows,
            n_valid,
            instrument_map,
        );

        results.push(EstimationResult {
            treatment: "all_factors".to_string(),
            outcome: return_col,
            asset: asset.clone(),
            ols: ols_result,
            tsls: tsls_result,
            diagnostics,
        });
    }

    Ok(results)
}

/// Run pooled panel regression: stack all assets with asset fixed effects.
///
/// The pooled model is: Y_it = α + Σ_j β_j F_jt + Σ_a δ_a D_a + ε_it
/// where D_a are asset dummy variables (fixed effects, with first asset as baseline).
fn run_pooled(
    data: &std::collections::HashMap<String, Vec<f64>>,
    assets: &[String],
    factor_names: &[String],
    macro_names: &[String],
    covariate_suffixes: &[String],
    instrument_map: &std::collections::HashMap<String, String>,
) -> anyhow::Result<Vec<EstimationResult>> {
    // Collect per-asset data, finding complete cases for each
    let mut all_y: Vec<f64> = Vec::new();
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut feature_names_final: Vec<String> = Vec::new();
    let mut asset_dummies: Vec<Vec<f64>> = Vec::new();
    let mut asset_labels: Vec<String> = Vec::new();
    let mut first_asset = true;
    let mut n_features_per_asset = 0;

    for asset in assets {
        let return_col = format!("{asset}_return");
        let y_data = match data.get(&return_col) {
            Some(d) => d,
            None => continue,
        };
        let n = y_data.len();

        // Build features (same logic as per-asset)
        let mut feature_cols: Vec<Vec<f64>> = Vec::new();
        let mut feat_names: Vec<String> = Vec::new();

        for f in factor_names {
            if let Some(col) = data.get(f) {
                feature_cols.push(col.clone());
                feat_names.push(f.clone());
            }
        }
        for m in macro_names {
            if let Some(col) = data.get(m) {
                feature_cols.push(col.clone());
                feat_names.push(m.clone());
            }
        }
        for suffix in covariate_suffixes {
            let col_name = format!("{asset}_{suffix}");
            if let Some(col) = data.get(&col_name) {
                feature_cols.push(col.clone());
                feat_names.push(suffix.to_string()); // use suffix for pooled (not asset-specific)
            }
        }

        if feature_cols.is_empty() {
            continue;
        }

        // Complete cases
        let valid_rows: Vec<usize> = (0..n)
            .filter(|&i| {
                !y_data[i].is_nan()
                    && feature_cols
                        .iter()
                        .all(|col| i < col.len() && !col[i].is_nan())
            })
            .collect();

        if valid_rows.len() < 10 {
            continue;
        }

        if first_asset {
            // Set up feature names from first asset
            feature_names_final = feat_names.clone();
            n_features_per_asset = feat_names.len();
            all_features.resize(n_features_per_asset, Vec::new());
            first_asset = false;
        } else if feat_names.len() != n_features_per_asset {
            // Skip assets with different feature counts (covariate availability)
            tracing::warn!("{asset}: feature count mismatch for pooled panel, skipping");
            continue;
        }

        // Append y
        for &i in &valid_rows {
            all_y.push(y_data[i]);
        }

        // Append features
        for (j, col) in feature_cols.iter().enumerate() {
            for &i in &valid_rows {
                all_features[j].push(col[i]);
            }
        }

        // Asset fixed effect (dummy variable, first asset is baseline)
        if asset_dummies.is_empty() {
            // First asset — baseline, no dummy needed but record label
            asset_labels.push(asset.clone());
            // Pad existing dummies with 0s for this asset's rows
        } else {
            asset_labels.push(asset.clone());
        }

        // Extend all existing dummy columns with 0s for this asset
        for dummy in &mut asset_dummies {
            dummy.extend(std::iter::repeat_n(0.0, valid_rows.len()));
        }

        // Add new dummy column for this asset (1s for its rows, 0s elsewhere)
        if asset_dummies.is_empty() && asset_labels.len() == 1 {
            // First asset is baseline — no dummy
        } else {
            let mut new_dummy = vec![0.0; all_y.len() - valid_rows.len()];
            new_dummy.extend(std::iter::repeat_n(1.0, valid_rows.len()));
            asset_dummies.push(new_dummy);
        }
    }

    let n_total = all_y.len();
    if n_total < 30 {
        anyhow::bail!("Pooled panel: only {n_total} total observations (need >= 30)");
    }

    // Build full X matrix: [factors + macro + covariates + asset dummies]
    let n_dummies = asset_dummies.len();
    let k = n_features_per_asset + n_dummies;

    let mut x_data = vec![0.0; n_total * k];
    for (j, col) in all_features.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            x_data[i + j * n_total] = v;
        }
    }
    for (j, dummy) in asset_dummies.iter().enumerate() {
        for (i, &v) in dummy.iter().enumerate() {
            x_data[i + (n_features_per_asset + j) * n_total] = v;
        }
    }

    let x_raw = DMatrix::from_column_slice(n_total, k, &x_data);
    let x = add_intercept(&x_raw);

    let mut all_names = vec!["intercept".to_string()];
    all_names.extend(feature_names_final.iter().cloned());
    // Dummy names: "FE_{asset}" for each non-baseline asset
    for (i, label) in asset_labels.iter().enumerate().skip(1) {
        if i - 1 < n_dummies {
            all_names.push(format!("FE_{label}"));
        }
    }

    let y = DVector::from_iterator(n_total, all_y.iter().copied());

    let ols_result = ols(&y, &x, &all_names)?;
    let diagnostics = compute_diagnostics(&ols_result.residuals, &x, &all_names)?;

    // 2SLS for pooled (use stacked instrument data)
    let tsls_result = try_tsls(
        &y,
        &x_raw,
        &{
            let mut names = feature_names_final.clone();
            for (i, label) in asset_labels.iter().enumerate().skip(1) {
                if i - 1 < n_dummies {
                    names.push(format!("FE_{label}"));
                }
            }
            names
        },
        data,
        &(0..n_total).collect::<Vec<_>>(),
        n_total,
        instrument_map,
    );

    Ok(vec![EstimationResult {
        treatment: "all_factors".to_string(),
        outcome: "pooled_returns".to_string(),
        asset: "pooled".to_string(),
        ols: ols_result,
        tsls: tsls_result,
        diagnostics,
    }])
}

/// Attempt 2SLS estimation if instruments are available for any factor.
fn try_tsls(
    y: &DVector<f64>,
    x_raw: &DMatrix<f64>,
    feature_names: &[String],
    data: &std::collections::HashMap<String, Vec<f64>>,
    valid_rows: &[usize],
    n_valid: usize,
    instrument_map: &std::collections::HashMap<String, String>,
) -> Option<crate::types::TslsResult> {
    // Find which features have instruments
    let mut endog_indices = Vec::new();
    let mut instrument_cols = Vec::new();
    let mut instrument_names = Vec::new();

    for (j, name) in feature_names.iter().enumerate() {
        if let Some(iv_name) = instrument_map.get(name) {
            if let Some(iv_data) = data.get(iv_name) {
                // Check if IV has valid data for our rows
                let iv_valid: Vec<f64> = valid_rows
                    .iter()
                    .map(|&i| {
                        if i < iv_data.len() {
                            iv_data[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                if iv_valid.iter().all(|v| !v.is_nan()) {
                    endog_indices.push(j);
                    instrument_cols.push(iv_valid);
                    instrument_names.push(iv_name.clone());
                }
            }
        }
    }

    if endog_indices.is_empty() {
        return None;
    }

    // Split X into endogenous and exogenous parts
    let endog_data: Vec<f64> = endog_indices
        .iter()
        .flat_map(|&j| (0..n_valid).map(move |i| x_raw[(i, j)]))
        .collect();
    let x_endog = DMatrix::from_column_slice(n_valid, endog_indices.len(), &endog_data);

    let exog_indices: Vec<usize> = (0..feature_names.len())
        .filter(|j| !endog_indices.contains(j))
        .collect();

    let exog_data: Vec<f64> = exog_indices
        .iter()
        .flat_map(|&j| (0..n_valid).map(move |i| x_raw[(i, j)]))
        .collect();

    // Exogenous matrix with intercept
    let x_exog = if exog_indices.is_empty() {
        DMatrix::from_element(n_valid, 1, 1.0) // just intercept
    } else {
        let raw = DMatrix::from_column_slice(n_valid, exog_indices.len(), &exog_data);
        add_intercept(&raw)
    };

    // Instrument matrix
    let z_data: Vec<f64> = instrument_cols.iter().flatten().copied().collect();
    let z = DMatrix::from_column_slice(n_valid, instrument_cols.len(), &z_data);

    let endog_names: Vec<String> = endog_indices
        .iter()
        .map(|&j| feature_names[j].clone())
        .collect();

    let mut exog_names = vec!["intercept".to_string()];
    exog_names.extend(exog_indices.iter().map(|&j| feature_names[j].clone()));

    match tsls(
        y,
        &x_endog,
        &x_exog,
        &z,
        &endog_names,
        &exog_names,
        &instrument_names,
    ) {
        Ok(result) => Some(result),
        Err(e) => {
            tracing::warn!("2SLS failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build synthetic data: y_asset = 1.0 + 2.0*factor1 + 0.5*factor2 + noise
    fn make_synthetic_data(assets: &[&str], n: usize) -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();

        // Two global factors
        let f1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let f2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).cos()).collect();
        data.insert("factor1".to_string(), f1.clone());
        data.insert("factor2".to_string(), f2.clone());

        // One macro factor
        let m1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.02).sin() * 0.5).collect();
        data.insert("macro1".to_string(), m1.clone());

        // Per-asset returns: y = 1.0 + 2.0*f1 + 0.5*f2 + 0.3*m1 + small noise
        for (a_idx, &asset) in assets.iter().enumerate() {
            let returns: Vec<f64> = (0..n)
                .map(|i| {
                    1.0 + 2.0 * f1[i]
                        + 0.5 * f2[i]
                        + 0.3 * m1[i]
                        + (i as f64 * 0.7 + a_idx as f64).sin() * 0.01
                })
                .collect();
            data.insert(format!("{asset}_return"), returns);
        }

        data
    }

    #[test]
    fn test_per_asset_recovers_coefficients() {
        let assets = ["eth", "btc"];
        let data = make_synthetic_data(&assets, 200);
        let asset_strings: Vec<String> = assets.iter().map(|s| s.to_string()).collect();

        let results = run_panel_estimation(
            &data,
            &asset_strings,
            &["factor1".to_string(), "factor2".to_string()],
            &["macro1".to_string()],
            &[],
            &HashMap::new(),
            PanelMode::PerAsset,
        )
        .unwrap();

        assert_eq!(results.len(), 2);

        for result in &results {
            // intercept ≈ 1.0, factor1 ≈ 2.0, factor2 ≈ 0.5, macro1 ≈ 0.3
            let coefs = &result.ols.coefficients;
            assert!(
                (coefs[0] - 1.0).abs() < 0.1,
                "intercept={}, expected ~1.0",
                coefs[0]
            );
            assert!(
                (coefs[1] - 2.0).abs() < 0.1,
                "factor1={}, expected ~2.0",
                coefs[1]
            );
            assert!(
                (coefs[2] - 0.5).abs() < 0.1,
                "factor2={}, expected ~0.5",
                coefs[2]
            );
            assert!(
                (coefs[3] - 0.3).abs() < 0.1,
                "macro1={}, expected ~0.3",
                coefs[3]
            );
            assert!(result.ols.r_squared > 0.99);
        }
    }

    #[test]
    fn test_per_asset_skips_missing_returns() {
        let data = make_synthetic_data(&["eth"], 100);
        // Remove btc_return so it's missing
        let asset_strings = vec!["eth".to_string(), "btc".to_string()];

        let results = run_panel_estimation(
            &data,
            &asset_strings,
            &["factor1".to_string()],
            &[],
            &[],
            &HashMap::new(),
            PanelMode::PerAsset,
        )
        .unwrap();

        // Only eth should have results
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset, "eth");
    }

    #[test]
    fn test_pooled_panel_runs() {
        let assets = ["eth", "btc", "sol"];
        let data = make_synthetic_data(&assets, 200);
        let asset_strings: Vec<String> = assets.iter().map(|s| s.to_string()).collect();

        let results = run_panel_estimation(
            &data,
            &asset_strings,
            &["factor1".to_string(), "factor2".to_string()],
            &["macro1".to_string()],
            &[],
            &HashMap::new(),
            PanelMode::Pooled,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset, "pooled");

        // Should have: intercept + factor1 + factor2 + macro1 + FE_btc + FE_sol = 6 coefficients
        assert_eq!(results[0].ols.coefficients.len(), 6);
        assert_eq!(results[0].ols.n_obs, 600); // 3 assets × 200 obs

        // Factor coefficients should still be close to true values
        let coefs = &results[0].ols.coefficients;
        assert!(
            (coefs[1] - 2.0).abs() < 0.1,
            "pooled factor1={}, expected ~2.0",
            coefs[1]
        );
        assert!(
            (coefs[2] - 0.5).abs() < 0.1,
            "pooled factor2={}, expected ~0.5",
            coefs[2]
        );
        assert!(results[0].ols.r_squared > 0.99);
    }

    #[test]
    fn test_per_asset_diagnostics() {
        let data = make_synthetic_data(&["eth"], 200);
        let results = run_panel_estimation(
            &data,
            &["eth".to_string()],
            &["factor1".to_string()],
            &[],
            &[],
            &HashMap::new(),
            PanelMode::PerAsset,
        )
        .unwrap();

        let diag = &results[0].diagnostics;
        // DW is computed and is a valid number (synthetic sine data will show autocorrelation)
        assert!(
            diag.durbin_watson >= 0.0 && diag.durbin_watson <= 4.0,
            "DW={}",
            diag.durbin_watson
        );
        // Breusch-Pagan is computed
        assert!(diag.breusch_pagan.0 >= 0.0);
        // VIF should be present
        assert!(!diag.vif.is_empty());
    }
}
