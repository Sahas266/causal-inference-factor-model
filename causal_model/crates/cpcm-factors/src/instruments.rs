use std::collections::HashMap;

/// Compute instrumental variables from the data panel.
///
/// Each IV is lagged by >= 1 day to avoid simultaneity.
/// Returns a map of iv_name → Vec<f64> and a map of factor → iv_name.
pub fn compute_instruments(
    data: &HashMap<String, Vec<f64>>,
    n_dates: usize,
) -> (HashMap<String, Vec<f64>>, HashMap<String, String>) {
    let mut iv_data = HashMap::new();
    let mut iv_map = HashMap::new(); // factor -> iv_name

    // gas_spike → instruments for liq_flow
    // Z-score of gas diff > 2σ, lagged by 1 day
    // Try multiple column names: Dune query 6831658 returns avg_base_fee_gwei, not avg_gas_price_gwei
    let gas_col = data.get("eth_avg_gas_price_gwei")
        .or_else(|| data.get("eth_avg_base_fee_gwei"))
        .or_else(|| data.get("eth_FeeTotNtv"));
    if let Some(gas) = gas_col {
        let spike = lagged_z_score_spike(gas, 2.0, 1);
        iv_data.insert("gas_spike".to_string(), spike);
        iv_map.insert("liq_flow".to_string(), "gas_spike".to_string());
    }

    // liquidation_level → instruments for funding_basis
    // Liquidation volume spikes, lagged by 1 day
    if let Some(liq) = data.get("eth_liquidation_volume_usd") {
        let spike = lagged_z_score_spike(liq, 2.0, 1);
        iv_data.insert("liquidation_level".to_string(), spike);
        iv_map.insert("funding_basis".to_string(), "liquidation_level".to_string());
    }

    // stablecoin_mint → instruments for stable_flow
    // Lagged daily change in stablecoin supply
    let supply_cols = ["usdc_SplyCur", "usdt_SplyCur"];
    let available: Vec<&Vec<f64>> = supply_cols
        .iter()
        .filter_map(|&col| data.get(col))
        .collect();

    if !available.is_empty() {
        let total: Vec<f64> = (0..n_dates)
            .map(|i| {
                available
                    .iter()
                    .map(|col| if i < col.len() && !col[i].is_nan() { col[i] } else { 0.0 })
                    .sum()
            })
            .collect();
        let mint = lag(&diff(&total), 1);
        iv_data.insert("stablecoin_mint".to_string(), mint);
        iv_map.insert("stable_flow".to_string(), "stablecoin_mint".to_string());
    }

    // protocol_event → instruments for chain_congestion
    // Fee regime change: abs(diff(fees)) > 2σ, lagged by 1 day
    let fees_col = data.get("eth_fees_usd")
        .or_else(|| data.get("eth_FeeTotNtv"));
    if let Some(fees) = fees_col {
        let spike = lagged_z_score_spike(fees, 2.0, 1);
        iv_data.insert("protocol_event".to_string(), spike);
        iv_map.insert("chain_congestion".to_string(), "protocol_event".to_string());
    }

    (iv_data, iv_map)
}

/// Compute a z-scored spike indicator: 1.0 if |z(diff(x))| > threshold, else 0.0.
/// Then lag by `lag_days`.
fn lagged_z_score_spike(x: &[f64], threshold: f64, lag_days: usize) -> Vec<f64> {
    let d = diff(x);
    let z = z_score_abs(&d);
    let indicator: Vec<f64> = z
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if v > threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    lag(&indicator, lag_days)
}

/// First-difference.
fn diff(x: &[f64]) -> Vec<f64> {
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

/// Z-score of absolute values: |x_i - mean| / std.
fn z_score_abs(x: &[f64]) -> Vec<f64> {
    let valid: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return x.to_vec();
    }
    let abs_vals: Vec<f64> = valid.iter().map(|v| v.abs()).collect();
    let mean = abs_vals.iter().sum::<f64>() / abs_vals.len() as f64;
    let var = abs_vals
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / abs_vals.len() as f64;
    let std = var.sqrt();
    if std < 1e-15 {
        return vec![0.0; x.len()];
    }
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                (v.abs() - mean) / std
            }
        })
        .collect()
}

/// Lag a series by `k` positions (shift forward). First `k` elements become NaN.
fn lag(x: &[f64], k: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; x.len()];
    for i in k..x.len() {
        result[i] = x[i - k];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lagged = lag(&x, 1);
        assert!(lagged[0].is_nan());
        assert!((lagged[1] - 1.0).abs() < 1e-10);
        assert!((lagged[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff() {
        let x = vec![10.0, 12.0, 11.0];
        let d = diff(&x);
        assert!(d[0].is_nan());
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_instruments_returns_map() {
        let mut data = HashMap::new();
        data.insert(
            "eth_avg_gas_price_gwei".to_string(),
            vec![10.0, 12.0, 50.0, 11.0, 13.0, 100.0, 10.0, 12.0, 11.0, 10.0],
        );

        let (iv_data, iv_map) = compute_instruments(&data, 10);
        assert!(iv_data.contains_key("gas_spike"));
        assert_eq!(iv_map.get("liq_flow"), Some(&"gas_spike".to_string()));
    }
}
