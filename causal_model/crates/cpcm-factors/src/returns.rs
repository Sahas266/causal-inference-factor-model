use std::collections::HashMap;

/// Compute log returns: ln(P_t / P_{t-1}).
///
/// First element is NaN (no prior price). NaN prices propagate as NaN.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let mut returns = vec![f64::NAN]; // no return for t=0
    for i in 1..prices.len() {
        let ret = if prices[i].is_nan() || prices[i - 1].is_nan() || prices[i - 1] <= 0.0 {
            f64::NAN
        } else {
            (prices[i] / prices[i - 1]).ln()
        };
        returns.push(ret);
    }
    returns
}

/// Compute returns for all assets from a data HashMap.
///
/// Looks for price data in this priority order:
/// 1. `{asset}_PriceUSD` (CoinMetrics)
/// 2. `{asset}_price_usd` (CoinGecko)
///
/// Returns a map of `{asset}_return` → Vec<f64>.
pub fn compute_all_returns(
    data: &HashMap<String, Vec<f64>>,
    assets: &[String],
) -> HashMap<String, Vec<f64>> {
    let mut returns = HashMap::new();

    for asset in assets {
        // Try CoinMetrics first (priority 1), then CoinGecko (priority 3)
        let price_col = format!("{asset}_PriceUSD");
        let fallback_col = format!("{asset}_price_usd");

        let prices = data
            .get(&price_col)
            .or_else(|| data.get(&fallback_col));

        if let Some(prices) = prices {
            let ret = log_returns(prices);
            returns.insert(format!("{asset}_return"), ret);
        } else {
            tracing::warn!("No price data for {asset} — cannot compute returns");
        }
    }

    returns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert!(returns[0].is_nan()); // no prior
        assert!((returns[1] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[2] - (105.0_f64 / 110.0).ln()).abs() < 1e-10);
        assert!((returns[3] - (115.0_f64 / 105.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_handles_nan() {
        let prices = vec![100.0, f64::NAN, 110.0];
        let returns = log_returns(&prices);
        assert!(returns[1].is_nan());
        assert!(returns[2].is_nan()); // NaN prior
    }

    #[test]
    fn test_log_returns_empty() {
        assert!(log_returns(&[]).is_empty());
    }
}
