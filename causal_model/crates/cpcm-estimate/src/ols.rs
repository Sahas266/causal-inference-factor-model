use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::types::OlsResult;

/// Ordinary Least Squares regression.
///
/// Solves β = (X'X)⁻¹ X'y with full diagnostics (SE, t-stats, p-values, R²).
/// Uses SVD decomposition for numerical stability when X'X is near-singular.
///
/// # Arguments
/// * `y` - Response vector (n × 1)
/// * `x` - Design matrix (n × k), should include intercept column if desired
/// * `feature_names` - Names for each column of X
pub fn ols(y: &DVector<f64>, x: &DMatrix<f64>, feature_names: &[String]) -> anyhow::Result<OlsResult> {
    let n = y.len();
    let k = x.ncols();

    anyhow::ensure!(
        x.nrows() == n,
        "X has {} rows but y has {} elements",
        x.nrows(),
        n
    );
    anyhow::ensure!(n > k, "Need n > k (got n={n}, k={k})");
    anyhow::ensure!(
        feature_names.len() == k,
        "feature_names len ({}) != X cols ({k})",
        feature_names.len()
    );

    // β = (X'X)⁻¹ X'y — using SVD for stability
    let xtx = x.transpose() * x;
    let xty = x.transpose() * y;

    let beta = solve_via_svd(&xtx, &xty)?;

    // Fitted values and residuals
    let y_hat = x * &beta;
    let residuals = y - &y_hat;

    // Sum of squared residuals and total
    let sse = residuals.dot(&residuals);
    let y_mean = y.mean();
    let sst = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();

    let r_squared = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    let df_resid = (n - k) as f64;
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / df_resid;

    // Standard errors: SE(β_j) = sqrt(σ² * (X'X)⁻¹_jj)
    let sigma2 = sse / df_resid;
    let xtx_inv = invert_via_svd(&xtx)?;

    let std_errors: Vec<f64> = (0..k)
        .map(|j| (sigma2 * xtx_inv[(j, j)]).max(0.0).sqrt())
        .collect();

    // t-statistics and p-values
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

    // HAC (Newey-West) standard errors alongside the iid ones.
    let hac_var = hac_covariance(x, &residuals, &xtx_inv, newey_west_maxlags(n));
    let hac_std_errors: Vec<f64> = (0..k)
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

    Ok(OlsResult {
        coefficients: beta.as_slice().to_vec(),
        std_errors,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        residuals: residuals.as_slice().to_vec(),
        n_obs: n,
        n_features: k,
        feature_names: feature_names.to_vec(),
        hac_std_errors,
        hac_t_stats,
        hac_p_values,
    })
}

/// Default Newey-West truncation lag: floor(4 * (n/100)^(2/9)).
pub fn newey_west_maxlags(n: usize) -> usize {
    (4.0 * ((n as f64) / 100.0).powf(2.0 / 9.0)).floor() as usize
}

/// Newey-West HAC covariance (Bartlett kernel), sandwich form.
///
/// Var(β) = (X'X)⁻¹ S (X'X)⁻¹ with
/// S = Σ_t e_t² x_t x_t' + Σ_{l=1}^{L} w_l Σ_t (x_t e_t e_{t-l} x_{t-l}' + sym),
/// w_l = 1 - l/(L+1). `xw` is the bread's regressor matrix (X for OLS, the
/// projected [X̂, X_exog] for 2SLS); `residuals` are the model residuals.
/// No small-sample correction (the Python port matches this exactly).
pub fn hac_covariance(
    xw: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    maxlags: usize,
) -> DMatrix<f64> {
    let n = xw.nrows();
    let k = xw.ncols();
    let mut xu = xw.clone();
    for i in 0..n {
        for j in 0..k {
            xu[(i, j)] *= residuals[i];
        }
    }
    let mut s = xu.transpose() * &xu;
    let l_max = maxlags.min(n.saturating_sub(1));
    for l in 1..=l_max {
        let w = 1.0 - (l as f64) / ((maxlags + 1) as f64);
        let gamma = xu.rows(l, n - l).transpose() * xu.rows(0, n - l);
        let gamma_t = gamma.transpose();
        s += w * (gamma + gamma_t);
    }
    xtx_inv * s * xtx_inv
}

/// Add an intercept (column of 1s) as the first column of X.
pub fn add_intercept(x: &DMatrix<f64>) -> DMatrix<f64> {
    let n = x.nrows();
    let ones = DVector::from_element(n, 1.0);
    DMatrix::from_columns(
        &std::iter::once(ones.column(0))
            .chain((0..x.ncols()).map(|j| x.column(j)))
            .collect::<Vec<_>>(),
    )
}

/// Solve Ax = b using SVD (handles near-singular A).
fn solve_via_svd(a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<DVector<f64>> {
    let svd = a.clone().svd(true, true);
    let u = svd.u.as_ref().ok_or_else(|| anyhow::anyhow!("SVD failed: no U"))?;
    let vt = svd.v_t.as_ref().ok_or_else(|| anyhow::anyhow!("SVD failed: no V'"))?;

    let threshold = 1e-10 * svd.singular_values.max();
    let s_inv = DVector::from_iterator(
        svd.singular_values.len(),
        svd.singular_values.iter().map(|&s| {
            if s > threshold { 1.0 / s } else { 0.0 }
        }),
    );

    // x = V * S⁻¹ * U' * b
    let ut_b = u.transpose() * b;
    let s_inv_ut_b = DVector::from_iterator(
        s_inv.len(),
        s_inv.iter().zip(ut_b.iter()).map(|(&si, &ub)| si * ub),
    );

    Ok(vt.transpose() * s_inv_ut_b)
}

/// Invert a symmetric matrix using SVD (pseudo-inverse for near-singular matrices).
fn invert_via_svd(a: &DMatrix<f64>) -> anyhow::Result<DMatrix<f64>> {
    let svd = a.clone().svd(true, true);
    let u = svd.u.as_ref().ok_or_else(|| anyhow::anyhow!("SVD failed"))?;
    let vt = svd.v_t.as_ref().ok_or_else(|| anyhow::anyhow!("SVD failed"))?;

    let threshold = 1e-10 * svd.singular_values.max();
    let s_inv_diag = DMatrix::from_diagonal(&DVector::from_iterator(
        svd.singular_values.len(),
        svd.singular_values.iter().map(|&s| {
            if s > threshold { 1.0 / s } else { 0.0 }
        }),
    ));

    Ok(vt.transpose() * s_inv_diag * u.transpose())
}

/// Project columns of X onto the column space of Z: X̂ = Z (Z'Z)⁻¹ Z'X.
/// Used in the first stage of 2SLS.
pub fn project_onto(x: &DMatrix<f64>, z: &DMatrix<f64>) -> anyhow::Result<DMatrix<f64>> {
    let ztz = z.transpose() * z;
    let ztz_inv = invert_via_svd(&ztz)?;
    let projection = z * &ztz_inv * z.transpose();
    Ok(&projection * x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// y = 2*x + 3 + noise. Should recover β ≈ [3, 2].
    #[test]
    fn test_ols_simple_linear() {
        let n = 200;
        let x_raw: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let y_raw: Vec<f64> = x_raw.iter().map(|&xi| 3.0 + 2.0 * xi).collect();

        let x_mat = DMatrix::from_column_slice(n, 1, &x_raw);
        let x_with_intercept = add_intercept(&x_mat);
        let y = DVector::from_column_slice(&y_raw);

        let names = vec!["intercept".into(), "x".into()];
        let result = ols(&y, &x_with_intercept, &names).unwrap();

        assert!((result.coefficients[0] - 3.0).abs() < 0.01, "intercept ≈ 3");
        assert!((result.coefficients[1] - 2.0).abs() < 0.01, "slope ≈ 2");
        assert!(result.r_squared > 0.999, "R² ≈ 1.0 for perfect linear data");
    }

    /// Multiple regression: y = 1 + 2*x1 - 0.5*x2.
    #[test]
    fn test_ols_multiple() {
        let n = 300;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = (i as f64) / 50.0;
            let x2 = ((i * 7 + 3) % 100) as f64 / 20.0;
            x_data.push(x1);
            x_data.push(x2);
            y_data.push(1.0 + 2.0 * x1 - 0.5 * x2);
        }

        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        let x_with_intercept = add_intercept(&x_mat);
        let y = DVector::from_column_slice(&y_data);

        let names = vec!["intercept".into(), "x1".into(), "x2".into()];
        let result = ols(&y, &x_with_intercept, &names).unwrap();

        assert!((result.coefficients[0] - 1.0).abs() < 0.01);
        assert!((result.coefficients[1] - 2.0).abs() < 0.01);
        assert!((result.coefficients[2] - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_add_intercept() {
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let xi = add_intercept(&x);
        assert_eq!(xi.ncols(), 3);
        assert_eq!(xi.nrows(), 3);
        // First column should be all 1s
        assert!((xi[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((xi[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((xi[(2, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_onto() {
        // Project x onto z where z = x (should get x back)
        let z = DMatrix::from_column_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let x = z.clone();
        let projected = project_onto(&x, &z).unwrap();
        for i in 0..4 {
            assert!((projected[(i, 0)] - x[(i, 0)]).abs() < 1e-10);
        }
    }
}
