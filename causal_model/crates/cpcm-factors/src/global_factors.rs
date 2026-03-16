use std::collections::HashMap;

/// Compute the 7 global CPCM factors from the data panel.
///
/// Each factor maps raw Supabase metrics into a single time series.
/// Returns a map of factor_name → Vec<f64>.
pub fn compute_global_factors(
    data: &HashMap<String, Vec<f64>>,
    n_dates: usize,
) -> HashMap<String, Vec<f64>> {
    let mut factors = HashMap::new();

    // Compute raw factors, then z-score all for comparability.
    // After standardization, each coefficient represents "per 1-std-dev change in the factor".
    factors.insert("liq_flow".to_string(), z_score(&compute_liq_flow(data, n_dates)));
    factors.insert("stable_flow".to_string(), z_score(&compute_stable_flow(data, n_dates)));
    factors.insert("funding_basis".to_string(), vec![f64::NAN; n_dates]); // GAP: no data
    factors.insert("chain_congestion".to_string(), compute_chain_congestion(data, n_dates)); // already z-scored
    factors.insert("staking_yield".to_string(), z_score(&compute_staking_yield(data, n_dates)));
    factors.insert("mev_pressure".to_string(), compute_mev_pressure(data, n_dates)); // already z-scored or rolling_std
    factors.insert("cex_dex_flow".to_string(), z_score(&compute_cex_dex_flow(data, n_dates)));

    factors
}

/// **LiqFlow**: diff(sum of protocol TVL across all tracked protocols).
/// Measures net liquidity entering/leaving DeFi.
fn compute_liq_flow(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    // Sum TVL across all protocols that have tvl_usd data
    let tvl_cols: Vec<&str> = [
        "aave_tvl_usd", "uni_tvl_usd", "crv_tvl_usd", "pendle_tvl_usd",
        "morpho_tvl_usd", "jup_tvl_usd", "ena_tvl_usd", "aero_tvl_usd",
        "eth_tvl_usd", "sol_tvl_usd", "bnb_tvl_usd", "avax_tvl_usd",
        "pol_tvl_usd", "btc_tvl_usd", "hype_tvl_usd",
    ]
    .iter()
    .filter(|&&col| data.contains_key(col))
    .copied()
    .collect();

    if tvl_cols.is_empty() {
        // Fallback: try Dune LP flow data
        if let Some(lp_flow) = data.get("eth_lp_net_flow_usd") {
            return lp_flow.clone();
        }
        return vec![f64::NAN; n];
    }

    let total_tvl = sum_columns(data, &tvl_cols, n);
    diff(&total_tvl)
}

/// **StableFlow**: diff(USDC supply + USDT supply + USDe supply).
/// Measures net stablecoin creation/destruction in the system.
fn compute_stable_flow(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    let supply_cols = [
        "usdc_SplyCur",
        "usdt_SplyCur",
        "usde_stablecoin_circulating_usd",
    ];
    let available: Vec<&str> = supply_cols
        .iter()
        .filter(|&&col| data.contains_key(col))
        .copied()
        .collect();

    if available.is_empty() {
        return vec![f64::NAN; n];
    }

    let total_supply = sum_columns(data, &available, n);
    diff(&total_supply)
}

/// **ChainCongestion**: weighted average of gas/block utilization metrics.
fn compute_chain_congestion(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    // Primary: Dune ETH gas data
    if let Some(gas) = data.get("eth_avg_gas_price_gwei") {
        // Z-score normalize for cross-comparability
        return z_score(gas);
    }

    // Fallback: CoinMetrics fee data
    let fee_cols: Vec<&str> = ["eth_FeeTotNtv", "btc_FeeTotNtv", "bnb_FeeTotNtv"]
        .iter()
        .filter(|&&col| data.contains_key(col))
        .copied()
        .collect();

    if !fee_cols.is_empty() {
        let total_fees = sum_columns(data, &fee_cols, n);
        return z_score(&total_fees);
    }

    vec![f64::NAN; n]
}

/// **StakingYield**: ETH staking APR.
fn compute_staking_yield(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    data.get("eth_staking_apr")
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; n])
}

/// **MEVPressure**: MEV-related volatility proxy.
/// Primary: rolling std of MEV revenue. Fallback: base fee volatility (stddev_base_fee_gwei)
/// as a proxy — high base fee variance correlates with MEV activity and orderflow chaos.
fn compute_mev_pressure(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    // Primary: direct MEV revenue data
    if let Some(mev) = data.get("eth_mev_revenue_eth") {
        return rolling_std(mev, 7);
    }

    // Fallback: base fee volatility from Dune query 6831658
    if let Some(base_fee_std) = data.get("eth_stddev_base_fee_gwei") {
        return z_score(base_fee_std);
    }

    // Second fallback: compute rolling std of avg base fee
    if let Some(base_fee) = data.get("eth_avg_base_fee_gwei") {
        return rolling_std(base_fee, 7);
    }

    vec![f64::NAN; n]
}

/// **CEXDEXFlow**: net exchange flow (inflow - outflow).
fn compute_cex_dex_flow(data: &HashMap<String, Vec<f64>>, n: usize) -> Vec<f64> {
    // Primary: Dune CEX netflow
    if let Some(netflow) = data.get("eth_cex_netflow_usd") {
        return netflow.clone();
    }

    // Fallback: CoinMetrics exchange flows
    let inflow = data.get("eth_FlowInExNtv");
    let outflow = data.get("eth_FlowOutExNtv");

    if let (Some(inf), Some(outf)) = (inflow, outflow) {
        return inf
            .iter()
            .zip(outf.iter())
            .map(|(&i, &o)| {
                if i.is_nan() || o.is_nan() {
                    f64::NAN
                } else {
                    i - o
                }
            })
            .collect();
    }

    vec![f64::NAN; n]
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Sum multiple columns element-wise, treating NaN as 0.
fn sum_columns(data: &HashMap<String, Vec<f64>>, cols: &[&str], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for &col in cols {
        if let Some(values) = data.get(col) {
            for (i, &v) in values.iter().enumerate().take(n) {
                if !v.is_nan() {
                    result[i] += v;
                }
            }
        }
    }
    result
}

/// First-difference: diff[i] = x[i] - x[i-1]. First element is NaN.
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

/// Z-score normalization: (x - mean) / std.
fn z_score(x: &[f64]) -> Vec<f64> {
    let valid: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return x.to_vec();
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64;
    let std = variance.sqrt();
    if std < 1e-15 {
        return vec![0.0; x.len()];
    }
    x.iter()
        .map(|&v| if v.is_nan() { f64::NAN } else { (v - mean) / std })
        .collect()
}

/// Rolling standard deviation with the given window size.
fn rolling_std(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let window_slice: Vec<f64> = x[(i + 1 - window)..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_slice.len() < 3 {
            continue;
        }

        let mean = window_slice.iter().sum::<f64>() / window_slice.len() as f64;
        let var = window_slice
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (window_slice.len() - 1) as f64;
        result[i] = var.sqrt();
    }

    result
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
        // Mean=3, std=sqrt(2)
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-10, "z-scored mean should be ~0");
    }

    #[test]
    fn test_rolling_std() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rs = rolling_std(&x, 3);
        assert!(rs[0].is_nan());
        assert!(rs[1].is_nan());
        assert!(!rs[2].is_nan()); // first valid window: [1,2,3]
        assert!((rs[2] - 1.0).abs() < 1e-10); // std([1,2,3]) = 1.0
    }

    #[test]
    fn test_sum_columns() {
        let mut data = HashMap::new();
        data.insert("a".to_string(), vec![1.0, 2.0, 3.0]);
        data.insert("b".to_string(), vec![10.0, 20.0, 30.0]);
        let result = sum_columns(&data, &["a", "b"], 3);
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }
}
