//! Shared NaN-safe series transforms used by factor, covariate, and
//! instrument construction.

/// Z-score normalization: (x - mean) / std. NaN-safe — NaNs are excluded
/// from the mean/std and preserved in the output.
pub fn z_score(x: &[f64]) -> Vec<f64> {
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

/// Z-score of absolute values: (|x_i| - mean(|x|)) / std(|x|).
pub fn z_score_abs(x: &[f64]) -> Vec<f64> {
    let abs: Vec<f64> = x.iter().map(|v| v.abs()).collect();
    z_score(&abs)
}

/// First-difference: diff[i] = x[i] - x[i-1]. First element is NaN.
pub fn diff(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mut d = vec![f64::NAN];
    for i in 1..x.len() {
        d.push(if x[i].is_nan() || x[i - 1].is_nan() {
            f64::NAN
        } else {
            x[i] - x[i - 1]
        });
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff() {
        let x = vec![10.0, 12.0, 11.0, 15.0];
        let d = diff(&x);
        assert!(d[0].is_nan());
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - (-1.0)).abs() < 1e-10);
        assert!((d[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_score() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = z_score(&x);
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-10, "z-scored mean should be ~0");
    }

    #[test]
    fn test_z_score_abs() {
        let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0];
        let z = z_score_abs(&x);
        // same as z_score of [1,2,3,4,5]
        let expected = z_score(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        for (a, b) in z.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_z_score_preserves_nan() {
        let x = vec![1.0, f64::NAN, 3.0];
        let z = z_score(&x);
        assert!(z[1].is_nan());
        assert!(!z[0].is_nan() && !z[2].is_nan());
    }
}
