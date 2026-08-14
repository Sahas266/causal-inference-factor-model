//! Emits canonical estimator fixtures for cross-validating the Python port.
//!
//! Run with:  cargo test -p cpcm-estimate --test emit_fixtures -- --nocapture
//!
//! Each fixture uses a deterministic input (fixed data or an LCG) so the
//! Python port can reproduce the exact same inputs and assert it matches the
//! Rust outputs to ~1e-9. Lines are `KEY = v0,v1,...` for easy parsing.

use cpcm_estimate::ols::{add_intercept, ols};
use cpcm_estimate::tsls::tsls;
use nalgebra::{DMatrix, DVector};

fn vec_str(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.12e}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn emit_fixtures() {
    // ── Fixture 1: OLS simple linear, y = 3 + 2x (exact) ──────────────
    let n = 200;
    let x_raw: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y_raw: Vec<f64> = x_raw.iter().map(|&xi| 3.0 + 2.0 * xi).collect();
    let x_mat = DMatrix::from_column_slice(n, 1, &x_raw);
    let xi = add_intercept(&x_mat);
    let y = DVector::from_column_slice(&y_raw);
    let r = ols(&y, &xi, &["intercept".into(), "x".into()]).unwrap();
    println!("OLS_SIMPLE_COEF = {}", vec_str(&r.coefficients));
    println!("OLS_SIMPLE_SE = {}", vec_str(&r.std_errors));
    println!("OLS_SIMPLE_R2 = {:.12e}", r.r_squared);
    println!("OLS_SIMPLE_HAC_SE = {}", vec_str(&r.hac_std_errors));

    // ── Fixture 2: OLS multiple, y = 1 + 2x1 - 0.5x2 (exact) ──────────
    let nm = 300;
    let mut xd = Vec::with_capacity(nm * 2);
    let mut yd = Vec::with_capacity(nm);
    for i in 0..nm {
        let x1 = (i as f64) / 50.0;
        let x2 = ((i * 7 + 3) % 100) as f64 / 20.0;
        xd.push(x1);
        xd.push(x2);
        yd.push(1.0 + 2.0 * x1 - 0.5 * x2);
    }
    let xm = DMatrix::from_row_slice(nm, 2, &xd);
    let xmi = add_intercept(&xm);
    let ym = DVector::from_column_slice(&yd);
    let rm = ols(&ym, &xmi, &["intercept".into(), "x1".into(), "x2".into()]).unwrap();
    println!("OLS_MULTI_COEF = {}", vec_str(&rm.coefficients));

    // ── Fixture 3: 2SLS canonical (LCG seed 42) ───────────────────────
    // Mirrors test_tsls_recovers_causal_effect exactly.
    let n2 = 1000usize;
    let mut rng_state: u64 = 42;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    let mut z_data = vec![0.0; n2];
    let mut x_data = vec![0.0; n2];
    let mut y_data = vec![0.0; n2];
    for i in 0..n2 {
        let z_i = next_f64() * 4.0;
        let v_i = next_f64();
        let e_i = next_f64() + 0.8 * v_i;
        let x_i = 0.3 * z_i + v_i;
        let y_i = 1.0 + 0.5 * x_i + e_i;
        z_data[i] = z_i;
        x_data[i] = x_i;
        y_data[i] = y_i;
    }
    let y2 = DVector::from_column_slice(&y_data);
    let x_endog = DMatrix::from_column_slice(n2, 1, &x_data);
    let intercept = DMatrix::from_element(n2, 1, 1.0);
    let z = DMatrix::from_column_slice(n2, 1, &z_data);
    let t = tsls(
        &y2,
        &x_endog,
        &intercept,
        &z,
        &["x".into()],
        &["intercept".into()],
        &["z".into()],
    )
    .unwrap();
    println!("TSLS_COEF = {}", vec_str(&t.coefficients));
    println!("TSLS_SE = {}", vec_str(&t.std_errors));
    println!("TSLS_F = {}", vec_str(&t.first_stage_f));
    println!("TSLS_R2 = {:.12e}", t.r_squared);
    println!("TSLS_HAC_SE = {}", vec_str(&t.hac_std_errors));
    match (t.hausman_stat, t.hausman_p) {
        (Some(h), Some(p)) => println!("TSLS_HAUSMAN = {h:.12e},{p:.12e}"),
        _ => println!("TSLS_HAUSMAN = None"),
    }

    // Also emit the first 5 generated rows so Python can verify identical inputs.
    println!("DATA_Z5 = {}", vec_str(&z_data[..5]));
    println!("DATA_X5 = {}", vec_str(&x_data[..5]));
    println!("DATA_Y5 = {}", vec_str(&y_data[..5]));
}
