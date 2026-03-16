use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::ols::ols;
use crate::types::Diagnostics;

/// Compute regression diagnostics from residuals and design matrix.
pub fn compute_diagnostics(
    residuals: &[f64],
    x: &DMatrix<f64>,
    feature_names: &[String],
) -> anyhow::Result<Diagnostics> {
    let dw = durbin_watson(residuals);
    let bp = breusch_pagan(residuals, x, feature_names)?;
    let jb = jarque_bera(residuals);
    let vif = variance_inflation_factors(x, feature_names)?;

    Ok(Diagnostics {
        durbin_watson: dw,
        breusch_pagan: bp,
        jarque_bera: jb,
        vif,
    })
}

/// Durbin-Watson statistic for serial correlation in residuals.
/// DW ≈ 2.0 means no serial correlation.
/// DW < 2.0 suggests positive autocorrelation.
/// DW > 2.0 suggests negative autocorrelation.
pub fn durbin_watson(residuals: &[f64]) -> f64 {
    if residuals.len() < 2 {
        return 2.0;
    }

    let num: f64 = residuals
        .windows(2)
        .map(|w| (w[1] - w[0]).powi(2))
        .sum();
    let den: f64 = residuals.iter().map(|e| e.powi(2)).sum();

    if den > 1e-15 { num / den } else { 2.0 }
}

/// Breusch-Pagan test for heteroskedasticity.
/// Regresses squared residuals on X. Under H0 (homoskedasticity), nR² ~ χ²(k-1).
fn breusch_pagan(
    residuals: &[f64],
    x: &DMatrix<f64>,
    feature_names: &[String],
) -> anyhow::Result<(f64, f64)> {
    let n = residuals.len();
    let e_sq: Vec<f64> = residuals.iter().map(|e| e * e).collect();
    let y_sq = DVector::from_column_slice(&e_sq);

    let aux = ols(&y_sq, x, feature_names)?;

    let stat = n as f64 * aux.r_squared;
    let df = (x.ncols() - 1).max(1) as f64; // k-1 (excluding intercept)
    let chi2 = ChiSquared::new(df).unwrap();
    let p = 1.0 - chi2.cdf(stat.max(0.0));

    Ok((stat, p))
}

/// Jarque-Bera test for normality of residuals.
/// Under H0 (normality), JB ~ χ²(2).
fn jarque_bera(residuals: &[f64]) -> (f64, f64) {
    let n = residuals.len() as f64;
    if n < 3.0 {
        return (0.0, 1.0);
    }

    let mean = residuals.iter().sum::<f64>() / n;
    let m2 = residuals.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
    let m3 = residuals.iter().map(|e| (e - mean).powi(3)).sum::<f64>() / n;
    let m4 = residuals.iter().map(|e| (e - mean).powi(4)).sum::<f64>() / n;

    if m2 < 1e-15 {
        return (0.0, 1.0);
    }

    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);
    let excess_kurtosis = kurtosis - 3.0;

    let jb = (n / 6.0) * (skewness.powi(2) + excess_kurtosis.powi(2) / 4.0);

    let chi2 = ChiSquared::new(2.0).unwrap();
    let p = 1.0 - chi2.cdf(jb.max(0.0));

    (jb, p)
}

/// Variance Inflation Factor for each feature.
/// VIF_j = 1 / (1 - R²_j) where R²_j is from regressing X_j on all other X columns.
/// VIF > 10 suggests problematic multicollinearity.
fn variance_inflation_factors(
    x: &DMatrix<f64>,
    feature_names: &[String],
) -> anyhow::Result<Vec<(String, f64)>> {
    let k = x.ncols();
    let n = x.nrows();
    let mut vifs = Vec::with_capacity(k);

    for j in 0..k {
        // Skip intercept (all 1s column)
        if x.column(j).iter().all(|&v| (v - 1.0).abs() < 1e-10) {
            vifs.push((feature_names[j].clone(), 1.0));
            continue;
        }

        let y_j = x.column(j).into_owned();

        // Build X_other: all columns except j
        let other_cols: Vec<usize> = (0..k).filter(|&c| c != j).collect();
        if other_cols.is_empty() {
            vifs.push((feature_names[j].clone(), 1.0));
            continue;
        }

        let x_other = DMatrix::from_columns(
            &other_cols.iter().map(|&c| x.column(c)).collect::<Vec<_>>(),
        );

        let other_names: Vec<String> = other_cols
            .iter()
            .map(|&c| feature_names[c].clone())
            .collect();

        if n <= other_cols.len() + 1 {
            vifs.push((feature_names[j].clone(), f64::INFINITY));
            continue;
        }

        match ols(&y_j, &x_other, &other_names) {
            Ok(result) => {
                let vif = if result.r_squared < 1.0 {
                    1.0 / (1.0 - result.r_squared)
                } else {
                    f64::INFINITY
                };
                vifs.push((feature_names[j].clone(), vif));
            }
            Err(_) => {
                vifs.push((feature_names[j].clone(), f64::INFINITY));
            }
        }
    }

    Ok(vifs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_durbin_watson_no_autocorrelation() {
        // Alternating residuals: no serial correlation
        let residuals = vec![0.1, -0.1, 0.1, -0.1, 0.1, -0.1];
        let dw = durbin_watson(&residuals);
        // For perfectly alternating, DW should be close to (or above) 2
        assert!(dw > 1.5, "DW={dw}");
    }

    #[test]
    fn test_durbin_watson_strong_positive() {
        // Monotone residuals: strong positive autocorrelation
        let residuals = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let dw = durbin_watson(&residuals);
        assert!(dw < 1.0, "DW={dw} should be < 1 for positive autocorrelation");
    }

    #[test]
    fn test_jarque_bera_normal_ish() {
        // Uniformly spaced values — should be roughly normal-ish (low JB)
        let residuals: Vec<f64> = (-50..50).map(|i| i as f64 / 50.0).collect();
        let (jb, _p) = jarque_bera(&residuals);
        // Uniform isn't perfectly normal, but JB shouldn't be extreme
        assert!(jb < 100.0, "JB={jb}");
    }
}
