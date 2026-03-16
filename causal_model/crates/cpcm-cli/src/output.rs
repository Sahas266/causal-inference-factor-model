use cpcm_estimate::types::EstimationResult;

/// Export estimation results as JSON.
pub fn export_json(results: &[EstimationResult]) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(results)?)
}

/// Export a summary table as CSV.
pub fn export_csv_summary(results: &[EstimationResult]) -> String {
    let mut lines = vec!["asset,feature,ols_coef,ols_se,ols_p,ols_r2,has_2sls".to_string()];

    for result in results {
        for (i, name) in result.ols.feature_names.iter().enumerate() {
            let has_2sls = if result.tsls.is_some() { "yes" } else { "no" };
            lines.push(format!(
                "{},{},{:.6},{:.6},{:.6},{:.4},{}",
                result.asset,
                name,
                result.ols.coefficients[i],
                result.ols.std_errors[i],
                result.ols.p_values[i],
                result.ols.r_squared,
                has_2sls,
            ));
        }
    }

    lines.join("\n")
}
