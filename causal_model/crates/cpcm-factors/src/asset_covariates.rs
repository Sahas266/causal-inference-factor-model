use std::collections::HashMap;

use crate::stats::z_score;

/// Compute per-asset covariates from the data panel.
///
/// Returns a map of `{asset}_{covariate}` → Vec<f64>.
pub fn compute_asset_covariates(
    data: &HashMap<String, Vec<f64>>,
    assets: &[String],
    n_dates: usize,
) -> HashMap<String, Vec<f64>> {
    let mut covariates = HashMap::new();

    for asset in assets {
        // whale_conc: Dune whale concentration (ETH only currently) — z-scored
        let whale_key = format!("{asset}_whale_conc");
        let whale_data = data
            .get(&format!("{asset}_top_10_pct_balance"))
            .map(|v| z_score(v))
            .unwrap_or_else(|| vec![f64::NAN; n_dates]);
        covariates.insert(whale_key, whale_data);

        // protocol_rev: DefiLlama fees — z-scored
        let rev_key = format!("{asset}_protocol_rev");
        let rev_data = data
            .get(&format!("{asset}_fees_usd"))
            .map(|v| z_score(v))
            .unwrap_or_else(|| vec![f64::NAN; n_dates]);
        covariates.insert(rev_key, rev_data);

        // emissions: CoinMetrics net issuance — z-scored
        let emit_key = format!("{asset}_emissions");
        let emit_data = data
            .get(&format!("{asset}_IssTotNtv"))
            .map(|v| z_score(v))
            .unwrap_or_else(|| vec![f64::NAN; n_dates]);
        covariates.insert(emit_key, emit_data);

        // chain_activity: composite of TxCnt + AdrActCnt (z-scored and averaged)
        let activity_key = format!("{asset}_chain_activity");
        let tx_cnt = data.get(&format!("{asset}_TxCnt"));
        let adr_act = data.get(&format!("{asset}_AdrActCnt"));

        let activity = match (tx_cnt, adr_act) {
            (Some(tx), Some(adr)) => {
                let tx_z = z_score(tx);
                let adr_z = z_score(adr);
                tx_z.iter()
                    .zip(adr_z.iter())
                    .map(|(&t, &a)| {
                        if t.is_nan() && a.is_nan() {
                            f64::NAN
                        } else if t.is_nan() {
                            a
                        } else if a.is_nan() {
                            t
                        } else {
                            (t + a) / 2.0
                        }
                    })
                    .collect()
            }
            (Some(tx), None) => z_score(tx),
            (None, Some(adr)) => z_score(adr),
            (None, None) => vec![f64::NAN; n_dates],
        };
        covariates.insert(activity_key, activity);
    }

    covariates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_covariates_with_data() {
        let mut data = HashMap::new();
        data.insert("eth_fees_usd".to_string(), vec![1.0, 2.0, 3.0]);
        data.insert("eth_IssTotNtv".to_string(), vec![10.0, 11.0, 12.0]);
        data.insert("eth_TxCnt".to_string(), vec![100.0, 200.0, 300.0]);
        data.insert("eth_AdrActCnt".to_string(), vec![50.0, 100.0, 150.0]);

        let covs = compute_asset_covariates(&data, &["eth".to_string()], 3);
        assert!(covs.contains_key("eth_protocol_rev"));
        assert!(covs.contains_key("eth_emissions"));
        assert!(covs.contains_key("eth_chain_activity"));
        assert!(covs.contains_key("eth_whale_conc"));

        // Protocol rev should be z-scored fees_usd: z([1,2,3]) ≈ [-1.22, 0, 1.22]
        let rev = &covs["eth_protocol_rev"];
        assert!((rev[1] - 0.0).abs() < 1e-10, "middle z-score should be 0");
        assert!(rev[0] < 0.0, "first z-score should be negative");
        assert!(rev[2] > 0.0, "last z-score should be positive");
    }

    #[test]
    fn test_compute_covariates_missing_data() {
        let data = HashMap::new();
        let covs = compute_asset_covariates(&data, &["btc".to_string()], 5);

        // All should be NaN vectors
        assert!(covs["btc_whale_conc"].iter().all(|v| v.is_nan()));
        assert!(covs["btc_protocol_rev"].iter().all(|v| v.is_nan()));
    }
}
