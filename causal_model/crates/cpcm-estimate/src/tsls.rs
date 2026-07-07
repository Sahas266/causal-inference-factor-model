use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, ChiSquared, StudentsT};

use crate::ols::{hac_covariance, newey_west_maxlags, ols, project_onto};
use crate::types::TslsResult;

/// Two-Stage Least Squares estimation.
///
/// Estimates the causal effect of endogenous regressors on the outcome using
/// instrumental variables.
///
/// # Arguments
/// * `y` - Outcome vector (n × 1)
/// * `x_endog` - Endogenous regressors (n × k1)
/// * `x_exog` - Exogenous controls (n × k2), should already include intercept
/// * `z` - Instruments (n × m), must have m >= k1
/// * `endog_names` - Names for endogenous regressors
/// * `exog_names` - Names for exogenous controls
/// * `instrument_names` - Names for instruments
pub fn tsls(
    y: &DVector<f64>,
    x_endog: &DMatrix<f64>,
    x_exog: &DMatrix<f64>,
    z: &DMatrix<f64>,
    endog_names: &[String],
    exog_names: &[String],
    instrument_names: &[String],
) -> anyhow::Result<TslsResult> {
    let n = y.len();
    let k1 = x_endog.ncols(); // endogenous count
    let k2 = x_exog.ncols(); // exogenous count (includes intercept)
    let m = z.ncols(); // instrument count

    anyhow::ensure!(n == x_endog.nrows(), "x_endog row mismatch");
    anyhow::ensure!(n == x_exog.nrows(), "x_exog row mismatch");
    anyhow::ensure!(n == z.nrows(), "z row mismatch");
    anyhow::ensure!(m >= k1, "Need at least as many instruments ({m}) as endogenous vars ({k1})");

    // ── Stage 1: Regress each endogenous var on [Z, X_exog] ──────────
    let z_full = hstack(&[z, x_exog]);
    let _ = instrument_names; // names carried for the caller; numerics don't need them

    let x_hat = project_onto(x_endog, &z_full)?;

    // First-stage PARTIAL F-statistics (excluded instruments only)
    let first_stage_f = compute_first_stage_f(x_endog, z, x_exog)?;

    // ── Stage 2: Regress y on [X̂, X_exog] ───────────────────────────
    let x_second = hstack(&[&x_hat, x_exog]);
    let all_names: Vec<String> = endog_names
        .iter()
        .chain(exog_names.iter())
        .cloned()
        .collect();

    // Get beta from Stage 2
    let stage2_result = ols(y, &x_second, &all_names)?;
    let beta = DVector::from_column_slice(&stage2_result.coefficients);

    // ── Correct standard errors using original X (not X̂) ────────────
    // Residuals from the TRUE model: e = y - X_original * β
    let x_original = hstack(&[x_endog, x_exog]);
    let residuals = y - &x_original * &beta;

    let k_total = k1 + k2;
    let df_resid = (n - k_total) as f64;
    let sigma2 = residuals.dot(&residuals) / df_resid;

    // Var(β) = σ² * (X̂'X̂)⁻¹ X̂'X X̂(X̂'X̂)⁻¹
    // Simplified: σ² * (X̂'X̂)⁻¹ (corrected for 2SLS)
    let xtx_hat = x_second.transpose() * &x_second;
    let svd = xtx_hat.clone().svd(true, true);
    let u = svd.u.as_ref().unwrap();
    let vt = svd.v_t.as_ref().unwrap();
    let threshold = 1e-10 * svd.singular_values.max();
    let s_inv_diag = DMatrix::from_diagonal(&DVector::from_iterator(
        svd.singular_values.len(),
        svd.singular_values.iter().map(|&s| {
            if s > threshold { 1.0 / s } else { 0.0 }
        }),
    ));
    let xtx_hat_inv = vt.transpose() * s_inv_diag * u.transpose();

    let var_beta = sigma2 * &xtx_hat_inv;
    let std_errors: Vec<f64> = (0..k_total)
        .map(|j| var_beta[(j, j)].max(0.0).sqrt())
        .collect();

    let t_dist = StudentsT::new(0.0, 1.0, df_resid).unwrap();
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| if se > 1e-15 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
        .collect();

    // R² (using original X, not X̂)
    let y_mean = y.mean();
    let sst = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
    let sse = residuals.dot(&residuals);
    let r_squared = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };

    // ── Sargan test (overidentification) ─────────────────────────────
    let (sargan_stat, sargan_p) = if m > k1 {
        compute_sargan(y, &x_original, z, x_exog, &beta, k1)?
    } else {
        (None, None)
    };

    // ── Hausman test (uses the CORRECTED 2SLS SEs, not raw stage-2 SEs) ──
    let ols_result = ols(y, &x_original, &all_names)?;
    let (hausman_stat, hausman_p) =
        compute_hausman(&ols_result, beta.as_slice(), &std_errors, k1);

    // HAC (Newey-West) standard errors: bread from [X̂, X_exog], residuals
    // from the original X (the 2SLS correction).
    let hac_var = hac_covariance(&x_second, &residuals, &xtx_hat_inv, newey_west_maxlags(n));
    let hac_std_errors: Vec<f64> = (0..k_total)
        .map(|j| hac_var[(j, j)].max(0.0).sqrt())
        .collect();
    let hac_t_stats: Vec<f64> = beta
        .iter()
        .zip(&hac_std_errors)
        .map(|(&b, &se)| if se > 1e-15 { b / se } else { 0.0 })
        .collect();
    let hac_p_values: Vec<f64> = hac_t_stats
        .iter()
        .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
        .collect();

    Ok(TslsResult {
        coefficients: beta.as_slice().to_vec(),
        std_errors,
        t_stats,
        p_values,
        r_squared,
        residuals: residuals.as_slice().to_vec(),
        n_obs: n,
        feature_names: all_names,
        first_stage_f,
        sargan_stat,
        sargan_p,
        hausman_stat,
        hausman_p,
        hac_std_errors,
        hac_t_stats,
        hac_p_values,
    })
}

/// First-stage PARTIAL F of the excluded instruments per endogenous variable.
///
/// Restricted: x_j on X_exog only (R²r). Full: x_j on [Z, X_exog] (R²f).
/// F = ((R²f - R²r)/m) / ((1 - R²f)/(n - m - k2)).
/// The omnibus F used previously let exogenous controls inflate the statistic.
fn compute_first_stage_f(
    x_endog: &DMatrix<f64>,
    z: &DMatrix<f64>,
    x_exog: &DMatrix<f64>,
) -> anyhow::Result<Vec<f64>> {
    let n = x_endog.nrows();
    let k1 = x_endog.ncols();
    let m = z.ncols();
    let k2 = x_exog.ncols();
    let z_full = hstack(&[z, x_exog]);
    let full_names: Vec<String> = (0..m + k2).map(|i| format!("c{i}")).collect();
    let exog_names: Vec<String> = (0..k2).map(|i| format!("e{i}")).collect();
    let df2 = (n as f64) - (m as f64) - (k2 as f64);

    let mut f_stats = Vec::with_capacity(k1);
    for j in 0..k1 {
        let y_j = x_endog.column(j).into_owned();
        let r2_r = if k2 > 0 {
            ols(&y_j, x_exog, &exog_names)?.r_squared
        } else {
            0.0
        };
        let r2_f = ols(&y_j, &z_full, &full_names)?.r_squared;
        let f = if r2_f < 1.0 && m > 0 && df2 > 0.0 {
            ((r2_f - r2_r) / (m as f64)) / ((1.0 - r2_f) / df2)
        } else {
            0.0
        };
        f_stats.push(f);
    }

    Ok(f_stats)
}

/// Sargan overidentification test: n·R² of 2SLS residuals on [Z, X_exog].
/// H0: all instruments are valid (uncorrelated with the error term).
///
/// df = m - k1 (instruments minus endogenous). The aux regression must include
/// X_exog: residuals are orthogonal to exog by construction, and omitting it
/// misattributes exog variation to the instruments.
fn compute_sargan(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    z: &DMatrix<f64>,
    x_exog: &DMatrix<f64>,
    beta: &DVector<f64>,
    k1: usize,
) -> anyhow::Result<(Option<f64>, Option<f64>)> {
    let residuals = y - x * beta;
    let n = residuals.len() as f64;
    let m = z.ncols();

    let zx = hstack(&[z, x_exog]);
    let names: Vec<String> = (0..m)
        .map(|i| format!("z{i}"))
        .chain((0..x_exog.ncols()).map(|i| format!("e{i}")))
        .collect();
    let aux = ols(&residuals, &zx, &names)?;

    let stat = n * aux.r_squared;
    let df = m as f64 - k1 as f64;

    if df > 0.0 {
        let chi2 = ChiSquared::new(df).unwrap();
        let p = 1.0 - chi2.cdf(stat);
        Ok((Some(stat), Some(p)))
    } else {
        Ok((None, None))
    }
}

/// Hausman test comparing OLS and 2SLS estimates (diagonal simplification).
/// H0: OLS is consistent (no endogeneity).
///
/// Uses the CORRECTED 2SLS SEs. Terms with a non-positive variance difference
/// are SKIPPED (standard practice — the difference matrix is not positive
/// definite there); df = number of terms used. (None, None) if none usable.
fn compute_hausman(
    ols_result: &crate::types::OlsResult,
    tsls_beta: &[f64],
    tsls_se: &[f64],
    k_endog: usize,
) -> (Option<f64>, Option<f64>) {
    let mut stat = 0.0;
    let mut df = 0usize;
    for j in 0..k_endog {
        let d = ols_result.coefficients[j] - tsls_beta[j];
        let v = tsls_se[j] * tsls_se[j] - ols_result.std_errors[j] * ols_result.std_errors[j];
        if v > 0.0 {
            stat += d * d / v;
            df += 1;
        }
    }
    if df == 0 {
        return (None, None);
    }
    let chi2 = ChiSquared::new(df as f64).unwrap();
    let p = 1.0 - chi2.cdf(stat.max(0.0));
    (Some(stat), Some(p))
}

/// Horizontally stack matrices.
fn hstack(matrices: &[&DMatrix<f64>]) -> DMatrix<f64> {
    let n = matrices[0].nrows();
    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
    let mut result = DMatrix::zeros(n, total_cols);
    let mut col_offset = 0;
    for &mat in matrices {
        for j in 0..mat.ncols() {
            result.set_column(col_offset + j, &mat.column(j));
        }
        col_offset += mat.ncols();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// Classic 2SLS test: endogenous X correlated with error term.
    /// True model: y = 1.0 + 0.5*x + e, but x = 0.3*z + v where Cov(v, e) != 0.
    /// OLS is biased, 2SLS should recover β ≈ 0.5.
    #[test]
    fn test_tsls_recovers_causal_effect() {
        let n = 1000;
        // Use a deterministic pseudo-random sequence for reproducibility.
        let mut rng_state: u64 = 42;
        let mut next_f64 = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
        };

        let mut z_data = vec![0.0; n];
        let mut x_data = vec![0.0; n];
        let mut y_data = vec![0.0; n];

        for i in 0..n {
            let z_i = next_f64() * 4.0; // instrument
            let v_i = next_f64(); // endogenous part of x
            let e_i = next_f64() + 0.8 * v_i; // error correlated with v (endogeneity!)

            let x_i = 0.3 * z_i + v_i; // x depends on z and v
            let y_i = 1.0 + 0.5 * x_i + e_i; // true causal effect = 0.5

            z_data[i] = z_i;
            x_data[i] = x_i;
            y_data[i] = y_i;
        }

        let y = DVector::from_column_slice(&y_data);
        let x_endog = DMatrix::from_column_slice(n, 1, &x_data);
        let intercept = DMatrix::from_element(n, 1, 1.0);
        let z = DMatrix::from_column_slice(n, 1, &z_data);

        let result = tsls(
            &y,
            &x_endog,
            &intercept,
            &z,
            &["x".into()],
            &["intercept".into()],
            &["z".into()],
        )
        .unwrap();

        // 2SLS should get closer to 0.5 than OLS
        let tsls_beta = result.coefficients[0]; // coefficient on x
        assert!(
            (tsls_beta - 0.5).abs() < 0.3,
            "2SLS β={tsls_beta}, expected ≈ 0.5"
        );

        // First-stage F should be reasonably large
        assert!(
            result.first_stage_f[0] > 5.0,
            "First-stage F={} too low",
            result.first_stage_f[0]
        );
    }

    #[test]
    fn test_hstack() {
        let a = DMatrix::from_column_slice(3, 1, &[1.0, 2.0, 3.0]);
        let b = DMatrix::from_column_slice(3, 2, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let c = hstack(&[&a, &b]);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 3);
        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 7.0).abs() < 1e-10);
    }
}
