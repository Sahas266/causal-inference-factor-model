# IV Search — new instrument candidates from the warehouse

Screened **1602** candidate columns × 3 transforms (diff/spike/level, all lag-1) against every global treatment factor. Relevance bar: train-window first-stage F ≥ 10.0. Selection stats use the first 252 aligned rows only.

**Multiple-testing warning:** with thousands of (candidate, treatment) pairs, some F ≥ 10 hits are flukes. Clean candidates below must still clear the per-fold partial-F gate inside the 2SLS A/B before any claim is made.

## Clean candidates (381)

| Treatment | Candidate | train F | stability | |corr(Z,T)| | direct-path t |
|---|---|---:|---:|---:|---:|
| cex_dex_flow | `pol_application_fees[spike]` | 69.5 | 1/3 | 0.47 | 0.68 |
| cex_dex_flow | `usdt_realized_volatility_7d[spike]` | 68.4 | 1/3 | 0.46 | 0.97 |
| cex_dex_flow | `crv_txns[spike]` | 58.6 | 1/3 | 0.44 | 1.49 |
| cex_dex_flow | `pol_avg_stablecoin_fees[spike]` | 54.9 | 1/3 | 0.42 | 1.49 |
| cex_dex_flow | `eth_total_economic_activity[diff]` | 54.8 | 3/3 | 0.42 | 0.09 |
| cex_dex_flow | `usdt_realized_volatility_30d[spike]` | 50.8 | 1/3 | 0.41 | 0.66 |
| cex_dex_flow | `pol_median_stablecoin_fees[spike]` | 49.4 | 1/3 | 0.41 | 1.75 |
| cex_dex_flow | `sol_90d_ann_vol[diff]` | 48.2 | 1/3 | 0.40 | 1.07 |
| cex_dex_flow | `uni_capital_efficiency[spike]` | 45.8 | 1/3 | 0.39 | 1.76 |
| cex_dex_flow | `pol_fees_native[spike]` | 45.6 | 1/3 | 0.39 | 0.54 |
| cex_dex_flow | `eth_stablecoin_fees_native[spike]` | 38.9 | 1/3 | 0.37 | 0.80 |
| cex_dex_flow | `pol_stablecoin_fees_native[spike]` | 38.9 | 1/3 | 0.37 | 0.93 |
| cex_dex_flow | `eth_stablecoin_fees[spike]` | 37.5 | 1/3 | 0.36 | 1.11 |
| cex_dex_flow | `avax_realized_volatility_30d[diff]` | 36.7 | 1/3 | 0.36 | 1.58 |
| cex_dex_flow | `avax_90d_ann_vol[diff]` | 36.2 | 1/3 | 0.36 | 1.18 |
| cex_dex_flow | `eth_whale_transfer_count[diff]` | 33.9 | 3/3 | 0.35 | 0.57 |
| cex_dex_flow | `eth_chain_lending_deposits[spike]` | 33.2 | 1/3 | 0.34 | 0.36 |
| cex_dex_flow | `sol_realized_volatility_30d[diff]` | 32.8 | 2/3 | 0.34 | 0.95 |
| cex_dex_flow | `crv_txns[diff]` | 32.3 | 2/3 | 0.34 | 0.66 |
| cex_dex_flow | `usdc_p2p_stablecoin_supply[spike]` | 32.3 | 1/3 | 0.34 | 0.61 |
| cex_dex_flow | `ena_tvl_usd[diff]` | 30.8 | 1/2 | 0.33 | 0.07 |
| cex_dex_flow | `bnb_artemis_stablecoin_daily_txns[diff]` | 30.5 | 3/3 | 0.33 | 0.30 |
| cex_dex_flow | `ena_fees[diff]` | 29.9 | 2/2 | 0.33 | 0.13 |
| cex_dex_flow | `crv_fees[spike]` | 29.7 | 1/3 | 0.33 | 0.25 |
| cex_dex_flow | `crv_lp_fee_allocation[spike]` | 29.7 | 1/3 | 0.33 | 0.25 |
| cex_dex_flow | `crv_staking_fee_allocation[spike]` | 29.7 | 1/3 | 0.33 | 0.25 |
| cex_dex_flow | `ena_yield_fees[diff]` | 29.6 | 2/2 | 0.33 | 0.15 |
| cex_dex_flow | `eth_stablecoin_fees_native[diff]` | 29.3 | 2/3 | 0.32 | 0.85 |
| cex_dex_flow | `eth_stablecoin_fees[diff]` | 28.5 | 2/3 | 0.32 | 0.52 |
| cex_dex_flow | `crv_mc_revenue_ratio[diff]` | 28.3 | 3/3 | 0.32 | 0.09 |
| cex_dex_flow | `usdt_realized_volatility_7d[diff]` | 28.0 | 1/3 | 0.32 | 1.27 |
| cex_dex_flow | `bnb_realized_volatility_30d[diff]` | 27.8 | 2/3 | 0.32 | 0.11 |
| cex_dex_flow | `bnb_trade_count[diff]` | 27.7 | 2/3 | 0.32 | 0.59 |
| cex_dex_flow | `avax_liquidation_count[spike]` | 27.6 | 1/3 | 0.32 | 0.52 |
| cex_dex_flow | `uni_capital_efficiency[diff]` | 27.4 | 2/3 | 0.31 | 0.68 |
| cex_dex_flow | `eth_whale_transfer_count[spike]` | 27.3 | 1/3 | 0.31 | 1.98 |
| cex_dex_flow | `eth_p2p_stablecoin_daily_txns[diff]` | 26.9 | 2/3 | 0.31 | 0.47 |
| cex_dex_flow | `aero_fees_usd[diff]` | 26.7 | 1/1 | 0.31 | 0.10 |
| cex_dex_flow | `uni_fees_usd[spike]` | 26.6 | 1/3 | 0.31 | 0.09 |
| cex_dex_flow | `crv_active_revenue[spike]` | 26.5 | 1/3 | 0.31 | 1.84 |
| cex_dex_flow | `crv_revenue[spike]` | 26.5 | 1/3 | 0.31 | 1.84 |
| cex_dex_flow | `uni_90d_ann_vol[diff]` | 26.0 | 1/3 | 0.31 | 1.45 |
| cex_dex_flow | `eth_p2p_stablecoin_avg_txn_value[diff]` | 25.9 | 3/3 | 0.31 | 0.04 |
| cex_dex_flow | `avax_30d_ann_vol[diff]` | 25.3 | 1/3 | 0.30 | 1.78 |
| cex_dex_flow | `usdc_SplyCur[spike]` | 25.1 | 1/3 | 0.30 | 0.62 |
| cex_dex_flow | `usdt_p2p_stablecoin_avg_txn_value[diff]` | 24.6 | 3/3 | 0.30 | 0.47 |
| cex_dex_flow | `bnb_fees_native[diff]` | 24.2 | 2/3 | 0.30 | 0.16 |
| cex_dex_flow | `bnb_total_fees_bnb[diff]` | 24.2 | 2/3 | 0.30 | 0.16 |
| cex_dex_flow | `crv_tvl_usd[diff]` | 24.1 | 1/3 | 0.30 | 0.34 |
| cex_dex_flow | `eth_artemis_stablecoin_daily_txns[diff]` | 24.1 | 3/3 | 0.30 | 0.50 |
| cex_dex_flow | `bnb_p2p_stablecoin_avg_txn_value[diff]` | 24.0 | 1/3 | 0.30 | 0.98 |
| cex_dex_flow | `link_realized_volatility_30d[diff]` | 23.8 | 1/3 | 0.30 | 1.85 |
| cex_dex_flow | `bnb_30d_ann_vol[diff]` | 23.4 | 1/3 | 0.29 | 0.31 |
| cex_dex_flow | `eth_average_transaction_value[diff]` | 23.1 | 3/3 | 0.29 | 0.14 |
| cex_dex_flow | `avax_trade_count[spike]` | 23.1 | 2/3 | 0.29 | 0.35 |
| cex_dex_flow | `ena_fees_usd[diff]` | 23.0 | 2/2 | 0.29 | 0.29 |
| cex_dex_flow | `eth_realized_volatility_30d[diff]` | 22.9 | 3/3 | 0.29 | 0.86 |
| cex_dex_flow | `uni_realized_volatility_30d[diff]` | 22.7 | 1/3 | 0.29 | 0.29 |
| cex_dex_flow | `uni_fees_usd[diff]` | 22.2 | 2/3 | 0.29 | 0.54 |
| cex_dex_flow | `bnb_chain_txns[diff]` | 22.0 | 2/3 | 0.28 | 0.64 |
| cex_dex_flow | `bnb_tx_count[diff]` | 22.0 | 2/3 | 0.28 | 0.64 |
| cex_dex_flow | `bnb_txns[diff]` | 22.0 | 2/3 | 0.28 | 0.64 |
| cex_dex_flow | `uni_fees[diff]` | 21.4 | 2/3 | 0.28 | 0.57 |
| cex_dex_flow | `uni_lp_fee_allocation[diff]` | 21.4 | 2/3 | 0.28 | 0.57 |
| cex_dex_flow | `uni_spot_fees[diff]` | 21.4 | 2/3 | 0.28 | 0.57 |
| cex_dex_flow | `usdt_artemis_stablecoin_avg_txn_value[diff]` | 21.3 | 3/3 | 0.28 | 1.05 |
| cex_dex_flow | `hype_perp_fees[diff]` | 21.3 | 1/1 | 0.28 | 0.62 |
| cex_dex_flow | `hype_perp_fees_excluding_hip3[diff]` | 21.3 | 1/1 | 0.28 | 0.62 |
| cex_dex_flow | `usdc_p2p_stablecoin_dau[diff]` | 21.3 | 2/3 | 0.28 | 0.10 |
| cex_dex_flow | `crv_realized_volatility_30d[diff]` | 21.3 | 1/3 | 0.28 | 1.03 |
| cex_dex_flow | `eth_min_base_fee_gwei[diff]` | 21.2 | 2/3 | 0.28 | 0.23 |
| cex_dex_flow | `eth_etf_aum_native[diff]` | 21.2 | 1/1 | 0.28 | 0.15 |
| cex_dex_flow | `usdc_TxTfrCnt[diff]` | 21.2 | 2/3 | 0.28 | 0.36 |
| cex_dex_flow | `sol_30d_ann_vol[diff]` | 21.2 | 2/3 | 0.28 | 1.01 |
| cex_dex_flow | `eth_SplyCur[diff]` | 21.1 | 2/3 | 0.28 | 0.52 |
| cex_dex_flow | `eth_avg_stablecoin_fees[diff]` | 20.8 | 2/3 | 0.28 | 0.89 |
| cex_dex_flow | `eth_total_supply_native[diff]` | 20.8 | 2/3 | 0.28 | 0.51 |
| cex_dex_flow | `eth_p2p_stablecoin_dau[diff]` | 20.7 | 2/3 | 0.28 | 0.47 |
| cex_dex_flow | `ena_tvl[diff]` | 20.6 | 1/2 | 0.28 | 0.18 |
| cex_dex_flow | `usde_stablecoin_supply[diff]` | 20.5 | 1/2 | 0.28 | 0.18 |
| cex_dex_flow | `ena_stablecoin_total_supply[diff]` | 20.5 | 1/2 | 0.28 | 0.18 |
| cex_dex_flow | `usde_stablecoin_total_supply[diff]` | 20.5 | 1/2 | 0.28 | 0.18 |
| cex_dex_flow | `eth_median_stablecoin_fees[diff]` | 20.3 | 2/3 | 0.27 | 1.22 |
| cex_dex_flow | `eth_etf_aum_to_mcap[diff]` | 19.9 | 1/1 | 0.27 | 0.35 |
| cex_dex_flow | `doge_realized_volatility_30d[diff]` | 19.9 | 1/3 | 0.27 | 1.09 |
| cex_dex_flow | `eth_application_fees[diff]` | 19.9 | 2/3 | 0.27 | 0.46 |
| cex_dex_flow | `eth_circulating_supply_native[diff]` | 19.9 | 2/3 | 0.27 | 0.44 |
| cex_dex_flow | `eth_outstanding_supply_native[diff]` | 19.9 | 2/3 | 0.27 | 0.44 |
| cex_dex_flow | `pol_artemis_stablecoin_daily_txns[diff]` | 19.7 | 3/3 | 0.27 | 0.18 |
| cex_dex_flow | `hype_chain_txns[diff]` | 19.6 | 0/0 | 0.27 | 1.13 |
| cex_dex_flow | `eth_artemis_stablecoin_dau[diff]` | 19.5 | 3/3 | 0.27 | 0.68 |
| cex_dex_flow | `link_30d_ann_vol[diff]` | 19.1 | 1/3 | 0.27 | 1.22 |
| cex_dex_flow | `usdc_artemis_stablecoin_avg_txn_value[diff]` | 19.1 | 1/3 | 0.27 | 0.13 |
| cex_dex_flow | `ena_staking_fee_allocation[diff]` | 18.9 | 2/2 | 0.26 | 0.89 |
| cex_dex_flow | `avax_artemis_stablecoin_daily_txns[diff]` | 17.7 | 2/3 | 0.26 | 0.55 |
| cex_dex_flow | `usdc_TxCnt[diff]` | 17.6 | 2/3 | 0.26 | 0.33 |
| cex_dex_flow | `eth_whale_transfer_usd[diff]` | 17.5 | 3/3 | 0.26 | 0.26 |
| cex_dex_flow | `avax_flashloan_count[spike]` | 17.5 | 1/3 | 0.26 | 0.19 |
| cex_dex_flow | `eth_avg_stablecoin_fees[spike]` | 17.4 | 1/3 | 0.26 | 0.64 |
| cex_dex_flow | `usdt_artemis_stablecoin_avg_txn_value[spike]` | 17.4 | 1/3 | 0.25 | 1.25 |
| cex_dex_flow | `doge_90d_ann_vol[diff]` | 17.1 | 1/3 | 0.25 | 0.24 |
| cex_dex_flow | `aave_realized_volatility_30d[diff]` | 17.0 | 2/3 | 0.25 | 0.20 |
| cex_dex_flow | `eth_artemis_stablecoin_avg_txn_value[diff]` | 16.7 | 3/3 | 0.25 | 0.25 |
| cex_dex_flow | `usdt_stablecoin_circulating_usd[diff]` | 16.5 | 2/3 | 0.25 | 0.03 |
| cex_dex_flow | `avax_realized_volatility_7d[spike]` | 16.2 | 1/3 | 0.25 | 0.60 |
| cex_dex_flow | `crv_fees_usd[diff]` | 16.0 | 1/3 | 0.25 | 0.66 |
| cex_dex_flow | `bnb_artemis_stablecoin_dau[diff]` | 16.0 | 3/3 | 0.25 | 0.14 |
| cex_dex_flow | `bnb_realized_volatility_7d[diff]` | 15.8 | 1/3 | 0.24 | 0.60 |
| cex_dex_flow | `avax_trade_count[diff]` | 15.7 | 1/3 | 0.24 | 0.42 |
| cex_dex_flow | `usdt_TxTfrCnt[diff]` | 15.6 | 2/3 | 0.24 | 0.62 |
| cex_dex_flow | `sol_realized_volatility_7d[spike]` | 15.5 | 1/3 | 0.24 | 0.71 |
| cex_dex_flow | `ena_mc_fees_ratio[diff]` | 15.4 | 1/2 | 0.24 | 1.31 |
| cex_dex_flow | `eth_chain_median_txn_fee[diff]` | 15.4 | 2/3 | 0.24 | 1.38 |
| cex_dex_flow | `eth_90d_ann_vol[diff]` | 15.2 | 3/3 | 0.24 | 0.38 |
| cex_dex_flow | `usdt_TxCnt[diff]` | 15.2 | 2/3 | 0.24 | 0.62 |
| cex_dex_flow | `crv_tvl_usd[spike]` | 15.0 | 1/3 | 0.24 | 0.67 |
| cex_dex_flow | `usdt_p2p_stablecoin_daily_txns[diff]` | 14.8 | 2/3 | 0.24 | 0.49 |
| cex_dex_flow | `aero_gross_emissions[diff]` | 14.5 | 2/2 | 0.23 | 1.22 |
| cex_dex_flow | `usdc_p2p_stablecoin_avg_txn_value[diff]` | 14.5 | 3/3 | 0.23 | 0.55 |
| cex_dex_flow | `pendle_percent_ybs_in_pendle[diff]` | 14.5 | 1/2 | 0.23 | 1.08 |
| cex_dex_flow | `doge_realized_volatility_7d[spike]` | 14.5 | 1/3 | 0.23 | 0.24 |
| cex_dex_flow | `xrp_realized_volatility_30d[diff]` | 14.5 | 1/3 | 0.23 | 0.94 |
| cex_dex_flow | `usdc_p2p_stablecoin_daily_txns[diff]` | 14.5 | 2/3 | 0.23 | 0.24 |
| cex_dex_flow | `eth_etf_aum_native[spike]` | 14.4 | 1/1 | 0.23 | 0.03 |
| cex_dex_flow | `btc_90d_ann_vol[diff]` | 14.3 | 3/3 | 0.23 | 1.02 |
| cex_dex_flow | `bnb_non_sybil_users[diff]` | 14.3 | 3/3 | 0.23 | 0.53 |
| cex_dex_flow | `aero_staking_fee_allocation[diff]` | 14.1 | 1/2 | 0.23 | 1.20 |
| cex_dex_flow | `link_oracle_txns[diff]` | 14.1 | 3/3 | 0.23 | 1.30 |
| cex_dex_flow | `aero_active_revenue[diff]` | 14.0 | 2/2 | 0.23 | 1.63 |
| cex_dex_flow | `usdt_beta_to_btc_30d[spike]` | 14.0 | 1/3 | 0.23 | 1.04 |
| cex_dex_flow | `eth_30d_ann_vol[diff]` | 13.9 | 3/3 | 0.23 | 1.16 |
| cex_dex_flow | `bnb_stablecoin_dau[diff]` | 13.7 | 2/3 | 0.23 | 0.53 |
| cex_dex_flow | `pol_p2p_stablecoin_daily_txns[diff]` | 13.5 | 1/3 | 0.23 | 0.41 |
| cex_dex_flow | `jup_fees_usd[diff]` | 13.4 | 2/2 | 0.23 | 0.20 |
| cex_dex_flow | `bnb_avg_gas_utilization[diff]` | 13.3 | 2/3 | 0.22 | 0.28 |
| cex_dex_flow | `aero_third_party_incentives[diff]` | 13.3 | 1/2 | 0.22 | 1.60 |
| cex_dex_flow | `aero_bribes[diff]` | 13.3 | 2/2 | 0.22 | 1.60 |
| cex_dex_flow | `eth_median_stablecoin_fees[spike]` | 13.2 | 1/3 | 0.22 | 1.97 |
| cex_dex_flow | `uni_spot_txns[diff]` | 12.9 | 3/3 | 0.22 | 0.12 |
| cex_dex_flow | `uni_txns[diff]` | 12.9 | 3/3 | 0.22 | 0.12 |
| cex_dex_flow | `avax_realized_volatility_7d[diff]` | 12.7 | 1/3 | 0.22 | 0.87 |
| cex_dex_flow | `avax_total_economic_activity[diff]` | 12.6 | 3/3 | 0.22 | 0.52 |
| cex_dex_flow | `aero_passive_revenue[diff]` | 12.4 | 2/2 | 0.22 | 1.92 |
| cex_dex_flow | `pol_stablecoin_fees[diff]` | 12.1 | 1/3 | 0.21 | 0.47 |
| cex_dex_flow | `xrp_90d_ann_vol[diff]` | 11.9 | 1/3 | 0.21 | 0.46 |
| cex_dex_flow | `bnb_p2p_stablecoin_dau[diff]` | 11.9 | 2/3 | 0.21 | 0.30 |
| cex_dex_flow | `aero_earnings[diff]` | 11.8 | 2/2 | 0.21 | 1.96 |
| cex_dex_flow | `doge_30d_ann_vol[diff]` | 11.8 | 1/3 | 0.21 | 1.38 |
| cex_dex_flow | `crv_mc_fees_ratio[diff]` | 11.8 | 1/3 | 0.21 | 1.35 |
| cex_dex_flow | `jup_tvl[diff]` | 11.6 | 1/3 | 0.21 | 0.62 |
| cex_dex_flow | `pol_p2p_stablecoin_dau[diff]` | 11.6 | 1/3 | 0.21 | 1.02 |
| cex_dex_flow | `crv_total_supply_native[diff]` | 11.5 | 1/3 | 0.21 | 0.15 |
| cex_dex_flow | `sol_mc_revenue_ratio[spike]` | 11.4 | 1/3 | 0.21 | 1.21 |
| cex_dex_flow | `aero_txns[diff]` | 11.4 | 2/2 | 0.21 | 0.65 |
| cex_dex_flow | `crv_30d_ann_vol[diff]` | 11.1 | 1/3 | 0.21 | 0.48 |
| cex_dex_flow | `pol_artemis_stablecoin_dau[diff]` | 11.1 | 2/3 | 0.21 | 1.02 |
| cex_dex_flow | `aave_90d_ann_vol[diff]` | 11.0 | 1/3 | 0.21 | 1.83 |
| cex_dex_flow | `crv_90d_ann_vol[diff]` | 11.0 | 1/3 | 0.20 | 0.56 |
| cex_dex_flow | `eth_stablecoin_dau[diff]` | 10.9 | 2/3 | 0.20 | 1.11 |
| cex_dex_flow | `bnb_chain_dau[diff]` | 10.9 | 1/3 | 0.20 | 0.76 |
| cex_dex_flow | `bnb_dau[diff]` | 10.9 | 1/3 | 0.20 | 0.76 |
| cex_dex_flow | `sol_mc_fees_ratio[spike]` | 10.9 | 1/3 | 0.20 | 0.58 |
| cex_dex_flow | `uni_30d_ann_vol[diff]` | 10.7 | 1/3 | 0.20 | 0.95 |
| cex_dex_flow | `pol_stablecoin_dau[diff]` | 10.4 | 2/3 | 0.20 | 0.36 |
| cex_dex_flow | `btc_realized_volatility_30d[diff]` | 10.3 | 3/3 | 0.20 | 0.36 |
| cex_dex_flow | `eth_chain_tvl[spike]` | 10.1 | 1/3 | 0.20 | 1.21 |
| cex_dex_flow | `eth_tvl_usd[spike]` | 10.1 | 1/3 | 0.20 | 1.21 |
| cex_dex_flow | `aero_capital_efficiency[diff]` | 10.1 | 2/2 | 0.20 | 0.51 |
| cex_dex_flow | `bnb_dau_over_100_balance[diff]` | 10.0 | 3/3 | 0.20 | 0.69 |
| chain_congestion | `ena_tvl_usd[diff]` | 29.4 | 1/2 | 0.32 | 0.20 |
| chain_congestion | `eth_avg_mib_per_second[spike]` | 21.0 | 1/2 | 0.28 | 0.97 |
| chain_congestion | `eth_blob_size_mib[spike]` | 21.0 | 1/2 | 0.28 | 0.97 |
| chain_congestion | `hype_fees_usd[spike]` | 18.2 | 0/0 | 0.26 | 0.02 |
| chain_congestion | `hype_30d_ann_vol[spike]` | 13.1 | 0/0 | 0.22 | 1.15 |
| chain_congestion | `bnb_artemis_stablecoin_avg_txn_value[spike]` | 11.8 | 1/3 | 0.21 | 1.05 |
| funding_basis | `ena_tvl_usd[diff]` | 100.5 | 1/2 | 0.54 | 0.28 |
| funding_basis | `eth_etf_aum_native[diff]` | 34.0 | 1/1 | 0.35 | 0.27 |
| funding_basis | `eth_etf_aum_to_mcap[diff]` | 31.7 | 1/1 | 0.34 | 0.46 |
| funding_basis | `usde_stablecoin_circulating_usd[diff]` | 27.0 | 1/2 | 0.31 | 1.65 |
| funding_basis | `usde_tvl_usd[diff]` | 26.9 | 1/2 | 0.31 | 1.74 |
| funding_basis | `btc_earnings[spike]` | 24.7 | 1/2 | 0.30 | 0.96 |
| funding_basis | `btc_token_incentives[spike]` | 24.7 | 1/2 | 0.30 | 0.96 |
| funding_basis | `eth_etf_aum[diff]` | 21.2 | 1/1 | 0.28 | 0.32 |
| funding_basis | `usdt_stablecoin_total_supply[diff]` | 16.7 | 1/2 | 0.25 | 1.91 |
| funding_basis | `bnb_lst_tvl[diff]` | 16.6 | 2/2 | 0.25 | 1.11 |
| funding_basis | `usdt_stablecoin_supply[diff]` | 16.5 | 1/2 | 0.25 | 1.88 |
| funding_basis | `ena_fed_funds_rate_spread[diff]` | 15.3 | 1/2 | 0.24 | 1.25 |
| funding_basis | `ena_apy_30day[diff]` | 14.8 | 1/2 | 0.24 | 1.30 |
| funding_basis | `usdt_stablecoin_circulating_usd[diff]` | 13.1 | 1/2 | 0.22 | 1.61 |
| funding_basis | `btc_etf_aum[diff]` | 13.0 | 1/2 | 0.22 | 0.60 |
| funding_basis | `eth_chain_lending_deposits[diff]` | 12.9 | 1/2 | 0.22 | 1.41 |
| funding_basis | `uni_treasury[diff]` | 11.9 | 1/2 | 0.21 | 1.13 |
| funding_basis | `uni_own_token_treasury[diff]` | 11.9 | 1/2 | 0.21 | 1.13 |
| funding_basis | `doge_90d_ann_vol[diff]` | 11.2 | 1/2 | 0.21 | 0.70 |
| funding_basis | `eth_total_staked[diff]` | 10.9 | 2/2 | 0.20 | 1.32 |
| funding_basis | `eth_gross_emissions[diff]` | 10.8 | 2/2 | 0.20 | 1.34 |
| funding_basis | `eth_token_incentives[diff]` | 10.8 | 2/2 | 0.20 | 1.34 |
| funding_basis | `avax_total_staked[diff]` | 10.7 | 1/2 | 0.20 | 0.29 |
| liq_flow | `eth_gross_emissions[diff]` | 1030.9 | 2/3 | 0.90 | 0.94 |
| liq_flow | `eth_token_incentives[diff]` | 1030.9 | 2/3 | 0.90 | 0.94 |
| liq_flow | `sol_chain_median_txn_fee[diff]` | 547.4 | 3/3 | 0.83 | 1.35 |
| liq_flow | `sol_median_stablecoin_fees[diff]` | 547.4 | 1/3 | 0.83 | 1.35 |
| liq_flow | `aave_net_treasury[diff]` | 392.1 | 3/3 | 0.78 | 1.06 |
| liq_flow | `aave_treasury[diff]` | 389.4 | 3/3 | 0.78 | 0.98 |
| liq_flow | `aave_own_token_treasury[diff]` | 359.4 | 3/3 | 0.77 | 0.80 |
| liq_flow | `btc_mc_fees_ratio[diff]` | 359.1 | 3/3 | 0.77 | 1.05 |
| liq_flow | `avax_total_staked[diff]` | 321.7 | 3/3 | 0.75 | 0.55 |
| liq_flow | `bnb_mc_fees_ratio[diff]` | 289.2 | 3/3 | 0.73 | 1.86 |
| liq_flow | `bnb_mc_revenue_ratio[diff]` | 289.2 | 3/3 | 0.73 | 1.86 |
| liq_flow | `sol_mc_revenue_ratio[diff]` | 286.0 | 3/3 | 0.73 | 0.95 |
| liq_flow | `sol_mc_fees_ratio[diff]` | 286.0 | 3/3 | 0.73 | 0.95 |
| liq_flow | `sol_chain_avg_txn_fee[diff]` | 227.6 | 3/3 | 0.69 | 0.10 |
| liq_flow | `aave_mc_revenue_ratio[diff]` | 224.2 | 3/3 | 0.69 | 0.57 |
| liq_flow | `crv_mc_fees_ratio[diff]` | 208.4 | 3/3 | 0.67 | 1.24 |
| liq_flow | `btc_etf_aum[diff]` | 195.3 | 2/2 | 0.66 | 1.60 |
| liq_flow | `aave_mc_fees_ratio[diff]` | 181.6 | 3/3 | 0.65 | 0.66 |
| liq_flow | `eth_sharpe_30d[diff]` | 155.8 | 3/3 | 0.62 | 1.39 |
| liq_flow | `link_mc_fees_ratio[diff]` | 138.4 | 3/3 | 0.60 | 1.19 |
| liq_flow | `link_mc_revenue_ratio[diff]` | 138.4 | 3/3 | 0.60 | 1.19 |
| liq_flow | `hype_total_staked[diff]` | 136.0 | 0/0 | 0.59 | 1.06 |
| liq_flow | `btc_sharpe_30d[diff]` | 135.3 | 3/3 | 0.59 | 0.59 |
| liq_flow | `eth_chain_lending_deposits[diff]` | 130.1 | 3/3 | 0.59 | 0.89 |
| liq_flow | `bnb_sharpe_30d[diff]` | 128.3 | 3/3 | 0.58 | 1.09 |
| liq_flow | `uni_tvl[diff]` | 126.3 | 3/3 | 0.58 | 0.66 |
| liq_flow | `crv_tvl[diff]` | 113.2 | 2/3 | 0.56 | 0.72 |
| liq_flow | `jup_lst_tvl[diff]` | 111.1 | 2/2 | 0.55 | 0.98 |
| liq_flow | `aave_sharpe_30d[diff]` | 108.3 | 3/3 | 0.55 | 0.37 |
| liq_flow | `sol_sharpe_30d[diff]` | 104.6 | 3/3 | 0.54 | 1.91 |
| liq_flow | `sol_total_staked[diff]` | 101.1 | 3/3 | 0.54 | 0.31 |
| liq_flow | `uni_sharpe_30d[diff]` | 100.6 | 3/3 | 0.54 | 1.17 |
| liq_flow | `xrp_mc_fees_ratio[diff]` | 99.5 | 3/3 | 0.53 | 0.17 |
| liq_flow | `xrp_mc_revenue_ratio[diff]` | 99.5 | 3/3 | 0.53 | 0.17 |
| liq_flow | `bnb_chain_avg_txn_fee[diff]` | 98.9 | 1/3 | 0.53 | 0.11 |
| liq_flow | `avax_sharpe_30d[diff]` | 96.8 | 3/3 | 0.53 | 1.83 |
| liq_flow | `crv_sharpe_30d[diff]` | 94.1 | 3/3 | 0.52 | 0.92 |
| liq_flow | `xrp_sharpe_30d[diff]` | 93.2 | 3/3 | 0.52 | 1.37 |
| liq_flow | `doge_sharpe_30d[diff]` | 93.1 | 3/3 | 0.52 | 1.07 |
| liq_flow | `sol_rev[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_revenue[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_chain_fees[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_fees[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_passive_revenue[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_validator_fee_allocation[diff]` | 91.9 | 3/3 | 0.52 | 0.95 |
| liq_flow | `sol_voting_fees[diff]` | 89.5 | 3/3 | 0.51 | 1.01 |
| liq_flow | `morpho_mc_fees_ratio[diff]` | 82.5 | 1/1 | 0.50 | 1.12 |
| liq_flow | `shib_sharpe_30d[diff]` | 79.6 | 3/3 | 0.49 | 1.80 |
| liq_flow | `link_sharpe_30d[diff]` | 79.4 | 3/3 | 0.49 | 0.73 |
| liq_flow | `bnb_chain_median_txn_fee[diff]` | 78.9 | 1/3 | 0.49 | 0.03 |
| liq_flow | `jup_mc_fees_ratio[diff]` | 77.6 | 2/2 | 0.49 | 0.37 |
| liq_flow | `morpho_sharpe_30d[diff]` | 73.6 | 0/0 | 0.48 | 0.54 |
| liq_flow | `eth_mc_fees_ratio[diff]` | 70.2 | 3/3 | 0.47 | 0.26 |
| liq_flow | `jup_tvl[diff]` | 70.2 | 3/3 | 0.47 | 0.08 |
| liq_flow | `pol_sharpe_30d[diff]` | 69.7 | 2/2 | 0.47 | 0.68 |
| liq_flow | `sol_base_fees[diff]` | 68.7 | 3/3 | 0.46 | 0.56 |
| liq_flow | `zec_sharpe_30d[diff]` | 67.4 | 3/3 | 0.46 | 1.46 |
| liq_flow | `bnb_lst_tvl[diff]` | 67.3 | 3/3 | 0.46 | 0.08 |
| liq_flow | `uni_treasury[diff]` | 63.7 | 3/3 | 0.45 | 0.34 |
| liq_flow | `uni_own_token_treasury[diff]` | 63.7 | 3/3 | 0.45 | 0.34 |
| liq_flow | `eth_mc_revenue_ratio[diff]` | 61.1 | 3/3 | 0.44 | 0.37 |
| liq_flow | `hype_sharpe_30d[diff]` | 58.7 | 0/0 | 0.44 | 0.17 |
| liq_flow | `pepe_sharpe_30d[diff]` | 55.6 | 3/3 | 0.43 | 0.37 |
| liq_flow | `ena_sharpe_30d[diff]` | 55.4 | 1/1 | 0.43 | 0.70 |
| liq_flow | `jup_sharpe_30d[diff]` | 55.3 | 2/2 | 0.43 | 0.59 |
| liq_flow | `usdc_p2p_stablecoin_supply[diff]` | 54.7 | 1/3 | 0.42 | 0.77 |
| liq_flow | `hype_mc_revenue_ratio[diff]` | 53.6 | 1/1 | 0.42 | 1.31 |
| liq_flow | `pol_funding_premium[diff]` | 52.5 | 1/1 | 0.42 | 1.31 |
| liq_flow | `link_treasury[diff]` | 52.0 | 3/3 | 0.41 | 0.23 |
| liq_flow | `morpho_tvl[diff]` | 49.7 | 2/2 | 0.41 | 0.27 |
| liq_flow | `uni_capital_efficiency[diff]` | 48.2 | 1/3 | 0.40 | 1.05 |
| liq_flow | `pendle_tvl[diff]` | 47.4 | 3/3 | 0.40 | 0.95 |
| liq_flow | `eth_flashloan_count[diff]` | 47.2 | 2/3 | 0.40 | 1.22 |
| liq_flow | `aave_lending_liquidation_bonus[diff]` | 46.5 | 1/3 | 0.40 | 0.23 |
| liq_flow | `eth_application_fees[diff]` | 44.8 | 1/3 | 0.39 | 0.79 |
| liq_flow | `crv_txns[diff]` | 44.5 | 2/3 | 0.39 | 0.33 |
| liq_flow | `avax_unique_liquidators[diff]` | 44.4 | 3/3 | 0.39 | 0.40 |
| liq_flow | `eth_stablecoin_fees_native[diff]` | 44.0 | 1/3 | 0.39 | 1.20 |
| liq_flow | `avax_liquidation_count[diff]` | 43.4 | 2/3 | 0.38 | 1.74 |
| liq_flow | `pendle_own_token_treasury[diff]` | 42.5 | 3/3 | 0.38 | 0.43 |
| liq_flow | `pendle_treasury[diff]` | 42.5 | 3/3 | 0.38 | 0.43 |
| liq_flow | `eth_liquidation_count[diff]` | 41.3 | 2/3 | 0.38 | 1.87 |
| liq_flow | `aero_mc_revenue_ratio[diff]` | 40.7 | 2/2 | 0.37 | 1.63 |
| liq_flow | `aero_mc_fees_ratio[diff]` | 40.2 | 2/2 | 0.37 | 0.40 |
| liq_flow | `sol_total_fees_usd[diff]` | 38.8 | 3/3 | 0.37 | 0.50 |
| liq_flow | `pol_stablecoin_fees[diff]` | 37.9 | 1/3 | 0.36 | 0.77 |
| liq_flow | `pol_application_fees[diff]` | 37.2 | 2/3 | 0.36 | 0.28 |
| liq_flow | `avax_mc_fees_ratio[diff]` | 37.1 | 3/3 | 0.36 | 0.94 |
| liq_flow | `avax_mc_revenue_ratio[diff]` | 37.1 | 3/3 | 0.36 | 0.94 |
| liq_flow | `morpho_funding_premium[diff]` | 37.1 | 0/0 | 0.36 | 0.31 |
| liq_flow | `morpho_treasury[diff]` | 36.9 | 1/1 | 0.36 | 0.74 |
| liq_flow | `morpho_own_token_treasury[diff]` | 36.9 | 1/1 | 0.36 | 0.74 |
| liq_flow | `pol_stablecoin_fees_native[diff]` | 36.2 | 2/3 | 0.36 | 1.48 |
| liq_flow | `eth_whale_transfer_count[diff]` | 35.9 | 1/3 | 0.35 | 0.89 |
| liq_flow | `aave_fees[diff]` | 35.7 | 2/3 | 0.35 | 0.16 |
| liq_flow | `uni_fees[diff]` | 35.5 | 1/3 | 0.35 | 0.88 |
| liq_flow | `uni_lp_fee_allocation[diff]` | 35.5 | 1/3 | 0.35 | 0.88 |
| liq_flow | `uni_spot_fees[diff]` | 35.5 | 1/3 | 0.35 | 0.88 |
| liq_flow | `pol_fees_native[diff]` | 34.0 | 2/3 | 0.35 | 1.42 |
| liq_flow | `crv_fees[diff]` | 33.3 | 1/3 | 0.34 | 1.51 |
| liq_flow | `crv_lp_fee_allocation[diff]` | 33.3 | 1/3 | 0.34 | 1.51 |
| liq_flow | `crv_staking_fee_allocation[diff]` | 33.3 | 1/3 | 0.34 | 1.51 |
| liq_flow | `jup_own_token_treasury[diff]` | 32.5 | 2/2 | 0.34 | 0.11 |
| liq_flow | `jup_treasury[diff]` | 32.5 | 2/2 | 0.34 | 0.11 |
| liq_flow | `crv_mc_revenue_ratio[diff]` | 32.4 | 2/3 | 0.34 | 0.39 |
| liq_flow | `uni_tvl[spike]` | 32.1 | 1/3 | 0.34 | 0.10 |
| liq_flow | `avax_artemis_stablecoin_daily_txns[diff]` | 31.8 | 1/3 | 0.34 | 0.84 |
| liq_flow | `eth_unique_liquidators[diff]` | 30.8 | 3/3 | 0.33 | 0.37 |
| liq_flow | `aave_lending_liquidations[diff]` | 30.7 | 1/3 | 0.33 | 0.02 |
| liq_flow | `eth_whale_transfer_usd[diff]` | 30.3 | 1/3 | 0.33 | 0.53 |
| liq_flow | `avax_unique_flashloan_users[diff]` | 30.2 | 1/3 | 0.33 | 1.27 |
| liq_flow | `pol_avg_stablecoin_fees[diff]` | 29.6 | 1/3 | 0.33 | 0.59 |
| liq_flow | `eth_stablecoin_fees[diff]` | 29.6 | 1/3 | 0.33 | 0.81 |
| liq_flow | `aave_funding_premium[diff]` | 28.5 | 1/2 | 0.32 | 1.48 |
| liq_flow | `uni_fees[spike]` | 28.3 | 1/3 | 0.32 | 0.11 |
| liq_flow | `uni_lp_fee_allocation[spike]` | 28.3 | 1/3 | 0.32 | 0.11 |
| liq_flow | `uni_spot_fees[spike]` | 28.3 | 1/3 | 0.32 | 0.11 |
| liq_flow | `sol_realized_volatility_30d[spike]` | 25.9 | 1/3 | 0.31 | 1.34 |
| liq_flow | `sol_chain_lending_deposits[diff]` | 25.4 | 2/2 | 0.30 | 0.34 |
| liq_flow | `aave_funding_rate_8h[diff]` | 24.9 | 1/2 | 0.30 | 1.50 |
| liq_flow | `sol_p2p_stablecoin_supply[diff]` | 24.9 | 3/3 | 0.30 | 0.66 |
| liq_flow | `morpho_funding_rate_8h[diff]` | 24.2 | 0/0 | 0.30 | 0.11 |
| liq_flow | `pendle_sharpe_30d[diff]` | 24.2 | 3/3 | 0.30 | 0.58 |
| liq_flow | `avax_flashloan_count[diff]` | 23.6 | 2/3 | 0.29 | 1.92 |
| liq_flow | `pol_application_fees[spike]` | 23.3 | 1/3 | 0.29 | 0.30 |
| liq_flow | `avax_trade_count[diff]` | 22.6 | 1/3 | 0.29 | 0.66 |
| liq_flow | `usdt_realized_volatility_30d[spike]` | 22.4 | 1/3 | 0.29 | 0.32 |
| liq_flow | `link_funding_premium[diff]` | 22.2 | 2/2 | 0.29 | 1.25 |
| liq_flow | `aave_txns[diff]` | 21.9 | 1/3 | 0.28 | 0.78 |
| liq_flow | `pol_stablecoin_fees_native[spike]` | 21.1 | 1/3 | 0.28 | 0.62 |
| liq_flow | `sol_chain_lending_loans[diff]` | 20.6 | 2/2 | 0.28 | 0.64 |
| liq_flow | `tao_active_revenue[diff]` | 20.3 | 1/3 | 0.27 | 0.88 |
| liq_flow | `tao_revenue[diff]` | 20.3 | 1/3 | 0.27 | 0.88 |
| liq_flow | `eth_artemis_stablecoin_daily_txns[diff]` | 20.3 | 1/3 | 0.27 | 0.75 |
| liq_flow | `xrp_funding_premium[diff]` | 20.0 | 2/2 | 0.27 | 0.14 |
| liq_flow | `usdc_artemis_stablecoin_avg_txn_value[diff]` | 19.8 | 1/3 | 0.27 | 0.37 |
| liq_flow | `avax_funding_premium[diff]` | 19.5 | 2/2 | 0.27 | 1.14 |
| liq_flow | `bnb_30d_ann_vol[diff]` | 19.2 | 1/3 | 0.27 | 0.55 |
| liq_flow | `link_oracle_txns[diff]` | 18.9 | 2/3 | 0.27 | 1.53 |
| liq_flow | `sol_90d_ann_vol[diff]` | 18.0 | 1/3 | 0.26 | 0.73 |
| liq_flow | `eth_chain_lending_deposits[spike]` | 17.7 | 1/3 | 0.26 | 0.09 |
| liq_flow | `uni_spot_txns[diff]` | 17.0 | 1/3 | 0.25 | 0.33 |
| liq_flow | `uni_txns[diff]` | 17.0 | 1/3 | 0.25 | 0.33 |
| liq_flow | `ena_funding_premium[diff]` | 16.5 | 1/2 | 0.25 | 1.29 |
| liq_flow | `pol_funding_rate_8h[diff]` | 16.2 | 1/1 | 0.25 | 1.24 |
| liq_flow | `bnb_realized_volatility_30d[diff]` | 16.2 | 1/3 | 0.25 | 0.35 |
| liq_flow | `link_funding_rate_8h[diff]` | 16.0 | 1/2 | 0.25 | 1.48 |
| liq_flow | `tao_sharpe_30d[diff]` | 15.9 | 3/3 | 0.24 | 0.58 |
| liq_flow | `bnb_funding_premium[diff]` | 15.9 | 2/2 | 0.24 | 0.14 |
| liq_flow | `pol_artemis_stablecoin_daily_txns[diff]` | 15.4 | 1/3 | 0.24 | 0.04 |
| liq_flow | `eth_unique_flashloan_users[diff]` | 14.6 | 2/3 | 0.24 | 0.42 |
| liq_flow | `link_realized_volatility_30d[diff]` | 14.4 | 1/3 | 0.23 | 1.58 |
| liq_flow | `sol_total_fees_sol[diff]` | 13.6 | 1/3 | 0.23 | 0.41 |
| liq_flow | `avax_fees_native[diff]` | 13.0 | 2/3 | 0.22 | 0.04 |
| liq_flow | `avax_total_fees_avax[diff]` | 13.0 | 2/3 | 0.22 | 0.04 |
| liq_flow | `ena_mc_fees_ratio[diff]` | 12.7 | 2/2 | 0.22 | 0.85 |
| liq_flow | `uni_own_token_treasury[spike]` | 12.6 | 1/3 | 0.22 | 0.06 |
| liq_flow | `uni_treasury[spike]` | 12.6 | 1/3 | 0.22 | 0.06 |
| liq_flow | `xrp_funding_rate_8h[diff]` | 12.4 | 2/2 | 0.22 | 0.77 |
| liq_flow | `aave_own_token_treasury[spike]` | 11.8 | 1/3 | 0.21 | 1.22 |
| liq_flow | `usdc_beta_to_btc_30d[diff]` | 11.7 | 1/3 | 0.21 | 1.07 |
| liq_flow | `uni_funding_premium[diff]` | 11.3 | 2/2 | 0.21 | 0.11 |
| liq_flow | `eth_realized_volatility_30d[diff]` | 10.8 | 1/3 | 0.20 | 1.06 |
| liq_flow | `usdt_realized_volatility_7d[diff]` | 10.8 | 1/3 | 0.20 | 1.46 |
| liq_flow | `sol_funding_rate_8h[diff]` | 10.7 | 1/2 | 0.20 | 1.83 |
| liq_flow | `hype_chain_avg_txn_fee[diff]` | 10.7 | 0/0 | 0.20 | 0.22 |
| liq_flow | `ena_funding_rate_8h[diff]` | 10.6 | 1/2 | 0.20 | 1.11 |
| liq_flow | `avax_funding_rate_8h[diff]` | 10.4 | 2/2 | 0.20 | 0.82 |
| liq_flow | `jup_fees_usd[diff]` | 10.1 | 1/2 | 0.20 | 0.25 |
| liq_flow | `aave_treasury[spike]` | 10.1 | 1/3 | 0.20 | 1.79 |
| mev_pressure | `jup_own_token_treasury[spike]` | 27.4 | 1/2 | 0.31 | 0.61 |
| mev_pressure | `jup_treasury[spike]` | 27.4 | 1/2 | 0.31 | 0.61 |
| mev_pressure | `bnb_fees_native[spike]` | 14.6 | 1/3 | 0.23 | 0.01 |
| mev_pressure | `bnb_total_fees_bnb[spike]` | 14.6 | 1/3 | 0.23 | 0.01 |
| stable_flow | `eth_etf_aum_to_mcap[diff]` | 24.0 | 1/1 | 0.30 | 0.46 |
| stable_flow | `eth_etf_aum_native[diff]` | 23.0 | 1/1 | 0.29 | 0.25 |
| stable_flow | `eth_l2_settlement_value_usd[spike]` | 22.5 | 1/3 | 0.29 | 0.99 |
| staking_yield | `eth_queue_active_amount[diff]` | 87.5 | 1/3 | 0.51 | 0.22 |
| staking_yield | `usde_tvl_usd[diff]` | 22.5 | 2/2 | 0.29 | 1.71 |
| staking_yield | `usde_stablecoin_circulating_usd[diff]` | 21.9 | 2/2 | 0.28 | 1.61 |
| staking_yield | `morpho_90d_ann_vol[diff]` | 15.6 | 0/0 | 0.24 | 1.03 |
| staking_yield | `hype_fees_usd[spike]` | 13.0 | 0/0 | 0.22 | 0.30 |
| staking_yield | `hype_30d_ann_vol[spike]` | 10.1 | 0/0 | 0.20 | 0.83 |

## Flagged (relevant but suspect) (562)

| Treatment | Candidate | train F | stability | Flags |
|---|---|---:|---:|---|
| cex_dex_flow | `eth_liquidation_count[spike]` | 71.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_p2p_stablecoin_supply[diff]` | 65.0 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_tvl_usd[spike]` | 59.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `pol_stablecoin_fees[spike]` | 55.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_p2p_stablecoin_supply[spike]` | 50.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aero_mc_fees_ratio[spike]` | 50.3 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_chain_lending_loans[diff]` | 50.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `crv_tvl[diff]` | 48.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `crv_tvl[spike]` | 47.8 | 1/3 | DIRECT-PATH |t|=2.2 |
| cex_dex_flow | `eth_application_fees[spike]` | 42.4 | 0/3 | DIRECT-PATH |t|=2.3; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_90d_ann_vol[diff]` | 42.3 | 3/3 | DIRECT-PATH |t|=2.2 |
| cex_dex_flow | `eth_average_transaction_value[spike]` | 39.2 | 0/3 | DIRECT-PATH |t|=2.7; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_lending_deposits[spike]` | 39.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_fees[spike]` | 38.3 | 0/3 | DIRECT-PATH |t|=2.6; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `link_90d_ann_vol[diff]` | 38.1 | 0/3 | DIRECT-PATH |t|=3.7; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_tvl[spike]` | 37.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_realized_volatility_30d[diff]` | 37.5 | 2/3 | DIRECT-PATH |t|=2.4 |
| cex_dex_flow | `pol_p2p_stablecoin_supply[diff]` | 34.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_SplyCur[diff]` | 33.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `sol_realized_volatility_30d[spike]` | 33.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_avg_base_fee_gwei[spike]` | 32.9 | 0/3 | DIRECT-PATH |t|=2.6; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_eth_burned[spike]` | 32.9 | 0/3 | DIRECT-PATH |t|=2.6; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_eth_burned_calc[spike]` | 32.9 | 0/3 | DIRECT-PATH |t|=2.6; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_chain_lending_deposits[diff]` | 32.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `sol_chain_mau[diff]` | 32.9 | 0/3 | DIRECT-PATH |t|=2.5; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_avg_base_fee_gwei[spike]` | 31.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_fees_native[spike]` | 31.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_total_fees_avax[spike]` | 31.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_FeeTotNtv[spike]` | 30.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_fees_native[spike]` | 30.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `crv_fees_usd[spike]` | 29.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_p2p_stablecoin_supply[diff]` | 28.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_stablecoin_total_supply[spike]` | 28.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_tvl_usd[diff]` | 28.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_chain_tvl[diff]` | 28.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_chain_tvl[spike]` | 28.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aero_mc_revenue_ratio[spike]` | 28.1 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_tvl[spike]` | 27.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_chain_lending_loans[spike]` | 25.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_lending_liquidation_bonus[spike]` | 25.3 | 0/3 | DIRECT-PATH |t|=2.3; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_chain_lending_deposits[spike]` | 25.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_total_supply_native[spike]` | 24.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `tao_funding_rate_8h[spike]` | 24.2 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `pol_p2p_stablecoin_supply[spike]` | 23.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `hype_auction_fees[spike]` | 23.7 | 0/1 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_stablecoin_supply[diff]` | 23.6 | 2/3 | DIRECT-PATH |t|=3.0 |
| cex_dex_flow | `usdt_stablecoin_total_supply[diff]` | 23.6 | 2/3 | DIRECT-PATH |t|=3.0 |
| cex_dex_flow | `eth_total_economic_activity[spike]` | 23.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_FeeTotNtv[diff]` | 23.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_p2p_stablecoin_supply[spike]` | 23.1 | 0/3 | DIRECT-PATH |t|=2.8; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_SplyCur[spike]` | 23.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_l2_settlement_value_usd[diff]` | 23.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `link_beta_to_btc_30d[spike]` | 22.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_tvl[diff]` | 22.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_fees[spike]` | 22.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_lp_fee_allocation[spike]` | 22.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_spot_fees[spike]` | 22.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_fees_native[diff]` | 22.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_validator_fee_allocation_native[diff]` | 22.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `ena_stablecoin_total_supply_adjusted[diff]` | 21.8 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `hype_fees_usd[diff]` | 21.7 | 0/0 | DIRECT-PATH |t|=2.1 |
| cex_dex_flow | `link_oracle_txns[spike]` | 21.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_lending_loans[diff]` | 20.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `shib_realized_volatility_30d[spike]` | 20.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `sol_trade_count[diff]` | 20.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `ena_other_fee_allocation[diff]` | 19.5 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `tao_funding_premium[spike]` | 18.9 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_tvl_usd[spike]` | 18.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_lending_deposits[diff]` | 18.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_stablecoin_supply[diff]` | 18.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_stablecoin_total_supply[diff]` | 18.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_stablecoin_supply[spike]` | 18.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_difficulty[diff]` | 17.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_fees[diff]` | 17.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `pol_funding_rate_8h[spike]` | 17.5 | 0/1 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `pol_funding_premium[spike]` | 17.4 | 0/1 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_stablecoin_supply[diff]` | 17.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_stablecoin_total_supply[diff]` | 17.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aero_mc_revenue_ratio[diff]` | 17.1 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_total_fees_usd[spike]` | 17.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_stablecoin_supply[spike]` | 16.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `sol_90d_ann_vol[spike]` | 16.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_chain_tvl[spike]` | 16.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aave_txns[spike]` | 16.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_funding_rate_8h[spike]` | 16.5 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_avg_mib_per_second[diff]` | 16.4 | 1/2 | DIRECT-PATH |t|=2.5 |
| cex_dex_flow | `eth_blob_size_mib[diff]` | 16.4 | 1/2 | DIRECT-PATH |t|=2.5 |
| cex_dex_flow | `hype_spot_fees[diff]` | 15.8 | 0/1 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_fees_native[spike]` | 15.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_total_fees_bnb[spike]` | 15.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `btc_chain_avg_txn_fee[diff]` | 15.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_p2p_stablecoin_supply[diff]` | 15.4 | 1/3 | DIRECT-PATH |t|=2.4 |
| cex_dex_flow | `shib_sharpe_30d[spike]` | 15.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `xrp_30d_ann_vol[diff]` | 15.2 | 1/3 | DIRECT-PATH |t|=2.6 |
| cex_dex_flow | `avax_total_economic_activity[spike]` | 15.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_chain_tvl[diff]` | 14.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_realized_volatility_30d[spike]` | 14.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `doge_30d_ann_vol[spike]` | 14.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_unique_liquidators[spike]` | 14.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_p2p_stablecoin_daily_txns[diff]` | 14.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `doge_funding_rate_8h[spike]` | 14.3 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_stablecoin_total_supply[spike]` | 14.2 | 0/3 | DIRECT-PATH |t|=3.2; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `xrp_30d_ann_vol[spike]` | 13.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_tvl_usd[diff]` | 13.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_stablecoin_circulating_usd[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_chain_lending_loans[diff]` | 13.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_chain_lending_deposits[diff]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_beta_to_btc_30d[diff]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `link_active_revenue[spike]` | 13.0 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_90d_ann_vol[spike]` | 12.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `xrp_realized_volatility_30d[spike]` | 12.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_p2p_stablecoin_supply[diff]` | 12.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_p2p_stablecoin_avg_txn_value[spike]` | 12.6 | 1/3 | DIRECT-PATH |t|=2.4 |
| cex_dex_flow | `uni_net_treasury[diff]` | 12.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_30d_ann_vol[spike]` | 12.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `zec_realized_volatility_7d[spike]` | 12.4 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aero_dau[diff]` | 12.2 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `doge_realized_volatility_30d[spike]` | 12.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `uni_90d_ann_vol[spike]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_tvl_usd[diff]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_chain_tvl[diff]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_validator_fee_allocation_native[spike]` | 11.9 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_total_fees_avax[diff]` | 11.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_fees_native[diff]` | 11.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_90d_ann_vol[spike]` | 11.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `crv_outstanding_supply_native[diff]` | 11.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_90d_ann_vol[spike]` | 11.5 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `jup_30d_ann_vol[diff]` | 11.4 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `aero_realized_volatility_7d[spike]` | 11.3 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_total_staked[diff]` | 11.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_stablecoin_growth[diff]` | 11.2 | 0/3 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_artemis_stablecoin_avg_txn_value[spike]` | 10.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_gross_emissions[diff]` | 10.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_token_incentives[diff]` | 10.8 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `xrp_realized_volatility_7d[diff]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `pol_stablecoin_fees_native[diff]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `sol_avg_stablecoin_fees[diff]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `eth_artemis_stablecoin_daily_txns[spike]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_tvl_usd[spike]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `jup_lst_tvl[spike]` | 10.4 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_stablecoin_supply[spike]` | 10.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdt_stablecoin_total_supply[spike]` | 10.3 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `jup_funding_premium[spike]` | 10.2 | 0/2 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `bnb_artemis_stablecoin_avg_txn_value[spike]` | 10.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `usdc_artemis_stablecoin_daily_txns[spike]` | 10.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_avg_base_fee_gwei[diff]` | 10.2 | 0/3 | UNSTABLE (relevance dies after train) |
| cex_dex_flow | `avax_total_staked[spike]` | 10.1 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `xrp_realized_volatility_30d[spike]` | 76.3 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_average_revenue_per_user[spike]` | 62.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_funding_rate_8h[spike]` | 62.0 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `tao_funding_rate_8h[spike]` | 51.8 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pendle_funding_rate_8h[spike]` | 39.8 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_total_supply_native[diff]` | 39.2 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `tao_funding_premium[spike]` | 31.1 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `aero_tvl_usd[spike]` | 30.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `usde_p2p_stablecoin_avg_txn_value[spike]` | 25.7 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_circulating_supply_native[diff]` | 25.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pol_funding_rate_8h[spike]` | 23.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_take_rate[diff]` | 23.3 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `morpho_own_token_treasury[spike]` | 22.3 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `morpho_treasury[spike]` | 22.3 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_buybacks_native[diff]` | 21.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pol_funding_premium[spike]` | 20.9 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `xrp_realized_volatility_30d[diff]` | 20.6 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_funding_premium[spike]` | 20.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_outstanding_supply_native[diff]` | 20.3 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_average_revenue_per_user[diff]` | 20.2 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `aero_beta_to_btc_30d[spike]` | 19.7 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `usde_artemis_stablecoin_avg_txn_value[spike]` | 19.5 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_buybacks[spike]` | 19.5 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_buybacks_native[spike]` | 19.5 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_passive_revenue[spike]` | 19.5 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_revenue[spike]` | 19.5 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_take_rate[spike]` | 19.5 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_buybacks[diff]` | 19.2 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_passive_revenue[diff]` | 19.2 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_revenue[diff]` | 19.2 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `avax_funding_rate_8h[spike]` | 15.8 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `eth_etf_aum_native[diff]` | 15.3 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `sol_funding_rate_8h[spike]` | 14.8 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `eth_etf_aum_to_mcap[diff]` | 14.4 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_tvl_usd[diff]` | 14.4 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `btc_90d_ann_vol[spike]` | 13.2 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `avax_funding_premium[spike]` | 13.0 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `xrp_artemis_stablecoin_avg_txn_value[spike]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `xrp_p2p_stablecoin_avg_txn_value[spike]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `eth_da_dau[spike]` | 12.5 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `link_active_revenue[spike]` | 12.2 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `usde_realized_volatility_30d[spike]` | 11.9 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `jup_30d_ann_vol[diff]` | 11.5 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `aero_90d_ann_vol[diff]` | 11.4 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pendle_sharpe_30d[spike]` | 11.2 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `hype_realized_volatility_7d[spike]` | 11.1 | 0/1 | UNSTABLE (relevance dies after train) |
| chain_congestion | `shib_realized_volatility_30d[spike]` | 10.9 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `btc_30d_ann_vol[spike]` | 10.9 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `xrp_beta_to_btc_30d[diff]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pol_chain_fees[spike]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pol_fees[spike]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `pol_revenue[spike]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| chain_congestion | `doge_funding_rate_8h[spike]` | 10.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_circulating_supply_native[diff]` | 173.0 | 0/2 | DIRECT-PATH |t|=2.9; UNSTABLE (relevance dies after train) |
| funding_basis | `eth_outstanding_supply_native[diff]` | 173.0 | 0/2 | DIRECT-PATH |t|=2.9; UNSTABLE (relevance dies after train) |
| funding_basis | `eth_total_supply_native[diff]` | 170.1 | 1/2 | DIRECT-PATH |t|=2.9 |
| funding_basis | `eth_SplyCur[diff]` | 154.9 | 1/2 | DIRECT-PATH |t|=2.2 |
| funding_basis | `hype_average_revenue_per_user[spike]` | 89.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `btc_SplyCur[diff]` | 60.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `btc_circulating_supply_native[diff]` | 60.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `btc_total_supply_native[diff]` | 60.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `ena_fees_usd[spike]` | 50.0 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_chain_mau[diff]` | 47.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_mau[diff]` | 47.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `tao_circulating_supply_native[diff]` | 42.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `tao_outstanding_supply_native[diff]` | 42.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `tao_total_issuance_native[diff]` | 42.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `tao_total_supply_native[diff]` | 42.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `pol_l1_fee_allocation[spike]` | 33.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `xrp_artemis_stablecoin_avg_txn_value[spike]` | 31.3 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `xrp_p2p_stablecoin_avg_txn_value[spike]` | 31.3 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_da_dau[spike]` | 26.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_mc_revenue_ratio[spike]` | 25.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `jup_own_token_treasury[diff]` | 20.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `jup_treasury[diff]` | 20.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_perp_fees[spike]` | 20.4 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_perp_fees_excluding_hip3[spike]` | 20.4 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `bnb_chain_mau[diff]` | 19.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `usdt_p2p_stablecoin_supply[diff]` | 19.8 | 1/2 | DIRECT-PATH |t|=2.0 |
| funding_basis | `eth_etf_aum_native[spike]` | 19.7 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `ena_tvl_usd[spike]` | 18.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_fees[spike]` | 18.1 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `pendle_token_holder_count[diff]` | 17.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `pendle_own_token_treasury[spike]` | 17.7 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `pendle_treasury[spike]` | 17.7 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_active_revenue[spike]` | 17.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `sol_chain_lending_deposits[diff]` | 16.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_take_rate[diff]` | 16.5 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_buybacks_native[diff]` | 16.2 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `pepe_realized_volatility_30d[diff]` | 15.4 | 0/2 | DIRECT-PATH |t|=2.7; UNSTABLE (relevance dies after train) |
| funding_basis | `avax_trade_count[spike]` | 15.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `usde_realized_volatility_30d[spike]` | 15.0 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_buybacks[spike]` | 14.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_buybacks_native[spike]` | 14.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_passive_revenue[spike]` | 14.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_revenue[spike]` | 14.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_take_rate[spike]` | 14.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `shib_realized_volatility_30d[diff]` | 14.8 | 0/2 | DIRECT-PATH |t|=2.2; UNSTABLE (relevance dies after train) |
| funding_basis | `avax_chain_tvl[diff]` | 14.5 | 0/2 | DIRECT-PATH |t|=2.0; UNSTABLE (relevance dies after train) |
| funding_basis | `uni_tvl[diff]` | 14.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_passive_revenue[diff]` | 14.4 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_revenue[diff]` | 14.4 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_buybacks[diff]` | 14.3 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `pol_chain_tvl[diff]` | 14.0 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_average_revenue_per_user[diff]` | 13.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_min_base_fee_gwei[spike]` | 13.7 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_mc_fees_ratio[diff]` | 13.7 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `pendle_spot_fees[spike]` | 13.7 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_90d_ann_vol[spike]` | 13.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_median_stablecoin_fees[spike]` | 13.6 | 0/2 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| funding_basis | `avax_dau_over_100_balance[spike]` | 13.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `hype_mc_fees_ratio[spike]` | 13.5 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_mc_fees_ratio[spike]` | 13.4 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `shib_realized_volatility_7d[spike]` | 13.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `sol_chain_lending_loans[diff]` | 13.0 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_bribes[spike]` | 12.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `bnb_tvl_usd[diff]` | 12.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `bnb_chain_tvl[diff]` | 12.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `jup_own_token_treasury_native[spike]` | 12.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_token_incentives[spike]` | 12.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `jup_own_token_treasury_native[diff]` | 12.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_avg_mib_per_second[diff]` | 12.1 | 0/2 | DIRECT-PATH |t|=2.2; UNSTABLE (relevance dies after train) |
| funding_basis | `eth_blob_size_mib[diff]` | 12.1 | 0/2 | DIRECT-PATH |t|=2.2; UNSTABLE (relevance dies after train) |
| funding_basis | `doge_realized_volatility_30d[diff]` | 12.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_cumulative_staked_eth[diff]` | 11.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_cumulative_eth_staked[diff]` | 11.9 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_chain_lending_deposits[diff]` | 11.8 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `morpho_own_token_treasury[diff]` | 11.8 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `morpho_treasury[diff]` | 11.8 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_total_supply_native[diff]` | 11.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_chain_txns[spike]` | 11.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_tx_count[spike]` | 11.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_txns[spike]` | 11.5 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `morpho_mc_fees_ratio[spike]` | 11.3 | 1/1 | DIRECT-PATH |t|=2.2 |
| funding_basis | `btc_weekly_commits_sub_ecosystem[spike]` | 11.3 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `jup_lst_tvl[spike]` | 11.2 | 0/2 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| funding_basis | `aero_beta_to_btc_30d[spike]` | 11.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_chain_wau[diff]` | 11.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `avax_wau[diff]` | 11.1 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `eth_etf_aum_to_mcap[spike]` | 10.9 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `morpho_net_treasury[spike]` | 10.8 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `btc_tvl_usd[diff]` | 10.7 | 0/2 | DIRECT-PATH |t|=2.2; UNSTABLE (relevance dies after train) |
| funding_basis | `morpho_net_treasury[diff]` | 10.6 | 0/1 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_tvl[diff]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `aero_realized_volatility_7d[spike]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| funding_basis | `pol_token_incentives[diff]` | 10.1 | 1/2 | DIRECT-PATH |t|=2.8 |
| liq_flow | `eth_etf_aum[diff]` | 2081.8 | 1/1 | TAUTOLOGY |corr(Z,T)|=0.94 |
| liq_flow | `eth_total_staked[diff]` | 1185.5 | 3/3 | TAUTOLOGY |corr(Z,T)|=0.91 |
| liq_flow | `sol_avg_stablecoin_fees[diff]` | 366.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `uni_mc_fees_ratio[diff]` | 96.0 | 3/3 | DIRECT-PATH |t|=2.0 |
| liq_flow | `bnb_chain_tvl[diff]` | 62.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_p2p_stablecoin_supply[diff]` | 50.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `lido_fees_usd[diff]` | 47.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_fees_usd[diff]` | 47.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_median_stablecoin_fees[spike]` | 46.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_stablecoin_fees[spike]` | 43.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_p2p_stablecoin_supply[diff]` | 40.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_avg_stablecoin_fees[spike]` | 40.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_p2p_stablecoin_supply[diff]` | 39.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `uni_capital_efficiency[spike]` | 36.4 | 2/3 | DIRECT-PATH |t|=2.1 |
| liq_flow | `uni_fees_usd[spike]` | 34.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `uni_fees_usd[diff]` | 32.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_p2p_stablecoin_supply[spike]` | 31.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_chain_tvl[diff]` | 30.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_stablecoin_supply[spike]` | 28.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_fees_native[spike]` | 28.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_fees_usd[spike]` | 27.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_funding_premium[diff]` | 27.2 | 2/2 | DIRECT-PATH |t|=2.4 |
| liq_flow | `eth_etf_aum[spike]` | 26.2 | 0/1 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_chain_lending_loans[diff]` | 26.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_p2p_stablecoin_supply[spike]` | 25.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `link_beta_to_btc_30d[spike]` | 25.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_liquidation_count[spike]` | 24.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_90d_ann_vol[spike]` | 23.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pol_p2p_stablecoin_supply[spike]` | 23.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_90d_ann_vol[diff]` | 21.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `btc_difficulty[diff]` | 21.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_p2p_stablecoin_supply[diff]` | 21.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_stablecoin_fees_native[spike]` | 21.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aave_dau[diff]` | 21.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_fees[spike]` | 20.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_lp_fee_allocation[spike]` | 20.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_staking_fee_allocation[spike]` | 20.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_txns[spike]` | 20.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_p2p_stablecoin_supply[spike]` | 20.3 | 0/3 | DIRECT-PATH |t|=2.6; UNSTABLE (relevance dies after train) |
| liq_flow | `eth_average_transaction_value[spike]` | 19.8 | 0/3 | DIRECT-PATH |t|=2.9; UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_beta_to_btc_30d[diff]` | 19.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `btc_earnings[diff]` | 18.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `btc_token_incentives[diff]` | 18.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `link_90d_ann_vol[diff]` | 18.8 | 0/3 | DIRECT-PATH |t|=3.3; UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_p2p_stablecoin_daily_txns[diff]` | 18.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `link_30d_ann_vol[diff]` | 18.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aero_txns[diff]` | 17.6 | 0/2 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_90d_ann_vol[diff]` | 17.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_average_transaction_value[diff]` | 17.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `link_realized_volatility_30d[spike]` | 17.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `jup_30d_ann_vol[spike]` | 17.2 | 0/2 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdt_artemis_stablecoin_avg_txn_value[diff]` | 17.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_liquidation_count[spike]` | 17.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_l2_settlement_value_usd[diff]` | 16.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_SplyCur[spike]` | 16.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_stablecoin_fees[spike]` | 16.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_funding_rate_8h[spike]` | 16.5 | 0/2 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| liq_flow | `ena_30d_ann_vol[spike]` | 16.4 | 0/1 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_realized_volatility_30d[spike]` | 16.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_30d_ann_vol[diff]` | 16.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_trade_count[diff]` | 16.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aave_lending_liquidation_bonus[spike]` | 16.1 | 0/3 | DIRECT-PATH |t|=2.3; UNSTABLE (relevance dies after train) |
| liq_flow | `uni_dau[diff]` | 16.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `uni_spot_dau[diff]` | 16.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_avg_stablecoin_fees[diff]` | 16.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_application_fees[spike]` | 15.9 | 0/3 | DIRECT-PATH |t|=2.5; UNSTABLE (relevance dies after train) |
| liq_flow | `usdt_p2p_stablecoin_avg_txn_value[diff]` | 15.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_chain_tvl[diff]` | 15.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_90d_ann_vol[spike]` | 15.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_90d_ann_vol[spike]` | 15.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_realized_volatility_7d[spike]` | 14.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aave_fees[spike]` | 14.8 | 0/3 | DIRECT-PATH |t|=2.8; UNSTABLE (relevance dies after train) |
| liq_flow | `uni_net_treasury[diff]` | 14.8 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_total_staked[spike]` | 14.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_chain_tvl[spike]` | 14.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdt_realized_volatility_7d[spike]` | 14.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `crv_tvl[spike]` | 14.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_30d_ann_vol[diff]` | 14.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aave_token_holder_value_accrual[spike]` | 14.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_realized_volatility_30d[spike]` | 13.8 | 0/3 | DIRECT-PATH |t|=2.5; UNSTABLE (relevance dies after train) |
| liq_flow | `sol_chain_lending_deposits[spike]` | 13.7 | 0/2 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_total_economic_activity[spike]` | 13.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_active_revenue[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_chain_fees[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_earnings[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_fees[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_passive_revenue[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_revenue[diff]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_artemis_stablecoin_daily_txns[diff]` | 13.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_chain_txns[diff]` | 13.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_txns[diff]` | 13.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_artemis_stablecoin_avg_txn_value[spike]` | 13.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_tx_count[diff]` | 13.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `xrp_realized_volatility_30d[diff]` | 13.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_chain_dau[diff]` | 13.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_dau[diff]` | 13.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_p2p_stablecoin_avg_txn_value[diff]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `pendle_beta_to_btc_30d[diff]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_whale_transfer_count[spike]` | 13.1 | 0/3 | DIRECT-PATH |t|=2.2; UNSTABLE (relevance dies after train) |
| liq_flow | `ena_30d_ann_vol[diff]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_artemis_stablecoin_daily_txns[spike]` | 13.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `aave_net_treasury[spike]` | 13.0 | 1/3 | DIRECT-PATH |t|=2.4 |
| liq_flow | `bnb_application_fees[diff]` | 13.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_funding_rate_8h[diff]` | 12.9 | 2/2 | DIRECT-PATH |t|=2.5 |
| liq_flow | `pol_p2p_stablecoin_avg_txn_value[spike]` | 12.9 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `btc_difficulty[spike]` | 12.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_new_users[diff]` | 12.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_p2p_stablecoin_dau[diff]` | 12.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_active_revenue[diff]` | 12.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_whale_transfer_usd[spike]` | 12.3 | 0/3 | DIRECT-PATH |t|=2.0; UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_30d_ann_vol[spike]` | 12.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_chain_txns[diff]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_tx_count[diff]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_txns[diff]` | 12.1 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `ena_realized_volatility_30d[spike]` | 11.9 | 0/1 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_stablecoin_supply[diff]` | 11.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_stablecoin_total_supply[diff]` | 11.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `zec_realized_volatility_7d[spike]` | 11.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_p2p_stablecoin_daily_txns[diff]` | 11.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_realized_volatility_30d[diff]` | 11.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_flashloan_count[spike]` | 11.0 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_stablecoin_total_supply[spike]` | 10.8 | 0/3 | DIRECT-PATH |t|=3.0; UNSTABLE (relevance dies after train) |
| liq_flow | `sol_validator_fee_allocation_native[diff]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_burns_native[diff]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_fees_native[diff]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_stablecoin_supply[spike]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdc_stablecoin_total_supply[spike]` | 10.6 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `doge_90d_ann_vol[diff]` | 10.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdt_beta_to_btc_30d[spike]` | 10.5 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `eth_artemis_stablecoin_avg_txn_value[diff]` | 10.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `uni_90d_ann_vol[diff]` | 10.4 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `avax_trade_count[spike]` | 10.3 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `sol_stablecoin_fees[diff]` | 10.2 | 0/3 | UNSTABLE (relevance dies after train) |
| liq_flow | `usdt_realized_volatility_30d[diff]` | 10.2 | 1/3 | DIRECT-PATH |t|=2.5 |
| liq_flow | `bnb_stablecoin_supply[spike]` | 10.2 | 0/3 | DIRECT-PATH |t|=3.9; UNSTABLE (relevance dies after train) |
| liq_flow | `xrp_30d_ann_vol[diff]` | 10.1 | 0/3 | DIRECT-PATH |t|=2.8; UNSTABLE (relevance dies after train) |
| liq_flow | `bnb_realized_volatility_7d[diff]` | 10.0 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_new_users[spike]` | 76.0 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_sybil_users[diff]` | 42.3 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_new_users[diff]` | 21.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_funding_rate_8h[spike]` | 20.9 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `morpho_own_token_treasury[spike]` | 20.5 | 0/1 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| mev_pressure | `morpho_treasury[spike]` | 20.5 | 0/1 | DIRECT-PATH |t|=2.1; UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_chain_wau[diff]` | 19.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_take_rate[diff]` | 16.2 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_chain_mau[diff]` | 15.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `aave_30d_ann_vol[spike]` | 14.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_buybacks[spike]` | 14.6 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_buybacks_native[spike]` | 14.6 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_passive_revenue[spike]` | 14.6 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_revenue[spike]` | 14.6 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_take_rate[spike]` | 14.6 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_buybacks_native[diff]` | 14.3 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_average_revenue_per_user[diff]` | 14.0 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_passive_revenue[diff]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_revenue[diff]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `hype_buybacks[diff]` | 13.0 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `bnb_avg_gas_utilization[spike]` | 13.0 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `uni_funding_rate_8h[spike]` | 12.1 | 0/2 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_chain_dau[diff]` | 11.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_dau[diff]` | 11.9 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `ena_artemis_stablecoin_dau[spike]` | 11.3 | 0/2 | UNSTABLE (relevance dies after train) |
| mev_pressure | `usde_artemis_stablecoin_dau[spike]` | 11.3 | 0/2 | UNSTABLE (relevance dies after train) |
| mev_pressure | `eth_etf_aum[diff]` | 10.9 | 0/1 | UNSTABLE (relevance dies after train) |
| mev_pressure | `aave_earnings[spike]` | 10.8 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `aave_token_incentives[spike]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pendle_percent_ybs_in_pendle[diff]` | 10.7 | 0/2 | UNSTABLE (relevance dies after train) |
| mev_pressure | `pol_chain_avg_txn_fee[spike]` | 10.3 | 0/2 | UNSTABLE (relevance dies after train) |
| mev_pressure | `sol_funding_rate_8h[spike]` | 10.0 | 0/2 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_stablecoin_fees_native[spike]` | 62.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_unique_flashloan_users[spike]` | 48.8 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_fees_native[spike]` | 44.6 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_liquidation_count[spike]` | 44.0 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `ena_fees_usd[spike]` | 38.7 | 0/2 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_liquidation_count[spike]` | 30.2 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_avg_base_fee_gwei[spike]` | 28.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_fees_native[spike]` | 28.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_total_fees_avax[spike]` | 28.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_HashRate[diff]` | 24.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_weekly_commits_sub[diff]` | 23.8 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_90d_ann_vol[spike]` | 23.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_application_fees[spike]` | 23.2 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `doge_realized_volatility_30d[spike]` | 22.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_30d_ann_vol[spike]` | 22.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_HashRate[spike]` | 21.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_unique_liquidators[spike]` | 21.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_30d_ann_vol[spike]` | 20.6 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_stablecoin_fees_native[spike]` | 20.4 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_application_fees[spike]` | 19.8 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `aave_token_holder_value_accrual[spike]` | 19.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `aave_fees[diff]` | 19.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_average_transaction_value[spike]` | 18.9 | 0/3 | DIRECT-PATH |t|=2.4; UNSTABLE (relevance dies after train) |
| stable_flow | `eth_etf_aum_to_mcap[spike]` | 18.8 | 0/1 | UNSTABLE (relevance dies after train) |
| stable_flow | `doge_30d_ann_vol[spike]` | 18.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `link_weekly_commits_core[spike]` | 16.4 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_30d_ann_vol[diff]` | 16.4 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_90d_ann_vol[diff]` | 16.0 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `aave_weekly_devs_core[diff]` | 15.8 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_realized_volatility_30d[diff]` | 15.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_avg_stablecoin_fees[spike]` | 15.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_realized_volatility_7d[diff]` | 14.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `aave_lending_liquidation_bonus[diff]` | 14.2 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_realized_volatility_7d[spike]` | 13.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_realized_volatility_30d[spike]` | 13.6 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_unique_liquidators[diff]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `bnb_tvl_usd[spike]` | 13.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `crv_weekly_devs_core[spike]` | 12.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `uni_capital_efficiency[spike]` | 12.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `sol_weekly_devs_sub[spike]` | 12.6 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `uni_90d_ann_vol[diff]` | 12.4 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `aave_fees[spike]` | 12.2 | 0/3 | DIRECT-PATH |t|=2.3; UNSTABLE (relevance dies after train) |
| stable_flow | `sol_weekly_devs_sub[diff]` | 11.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_weekly_devs_sub[diff]` | 11.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_fees_native[diff]` | 11.8 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `btc_weekly_commits_core[diff]` | 11.6 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_liquidation_count[diff]` | 11.4 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_30d_ann_vol[diff]` | 11.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_sharpe_30d[spike]` | 11.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `uni_realized_volatility_30d[diff]` | 11.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_TxCnt[spike]` | 11.3 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `bnb_30d_ann_vol[diff]` | 11.2 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `bnb_realized_volatility_30d[diff]` | 11.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `usdt_artemis_stablecoin_daily_txns[spike]` | 11.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_etf_aum_native[spike]` | 11.1 | 0/1 | UNSTABLE (relevance dies after train) |
| stable_flow | `tao_sharpe_30d[spike]` | 11.0 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_liquidation_count[diff]` | 11.0 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `sol_weekly_commits_core[diff]` | 10.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `ena_tvl_usd[diff]` | 10.9 | 0/2 | UNSTABLE (relevance dies after train) |
| stable_flow | `avax_weekly_commits_core[diff]` | 10.9 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `doge_sharpe_30d[spike]` | 10.7 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `pendle_percent_ybs_in_pendle[spike]` | 10.6 | 0/2 | UNSTABLE (relevance dies after train) |
| stable_flow | `tao_realized_volatility_30d[spike]` | 10.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `zec_weekly_devs_sub[spike]` | 10.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `uni_30d_ann_vol[diff]` | 10.5 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `ena_fees_usd[diff]` | 10.4 | 0/2 | UNSTABLE (relevance dies after train) |
| stable_flow | `pol_funding_rate_8h[spike]` | 10.2 | 0/1 | UNSTABLE (relevance dies after train) |
| stable_flow | `doge_30d_ann_vol[diff]` | 10.2 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `sol_stablecoin_supply[spike]` | 10.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `sol_stablecoin_total_supply[spike]` | 10.1 | 0/3 | UNSTABLE (relevance dies after train) |
| stable_flow | `eth_total_economic_activity[spike]` | 10.1 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `eth_queue_active_amount[spike]` | 365.1 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `eth_queue_exit_amount[spike]` | 365.1 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_total_supply_native[diff]` | 130.6 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_circulating_supply_native[diff]` | 88.8 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `eth_queue_exit_amount[diff]` | 80.1 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_outstanding_supply_native[diff]` | 70.3 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_funding_rate_8h[spike]` | 59.3 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `pepe_realized_volatility_30d[diff]` | 53.7 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_auction_fees[spike]` | 39.7 | 0/1 | DIRECT-PATH |t|=2.4; UNSTABLE (relevance dies after train) |
| staking_yield | `morpho_own_token_treasury[spike]` | 37.9 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `morpho_treasury[spike]` | 37.9 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `jup_funding_rate_8h[diff]` | 32.5 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `eth_da_dau[spike]` | 31.6 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `jup_funding_premium[diff]` | 29.1 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `pepe_realized_volatility_30d[spike]` | 25.0 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_realized_volatility_7d[spike]` | 17.4 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_average_revenue_per_user[spike]` | 16.3 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `doge_funding_premium[spike]` | 15.9 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `doge_funding_rate_8h[spike]` | 14.8 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `aero_fees_usd[spike]` | 14.3 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `sol_funding_rate_8h[spike]` | 13.4 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `eth_funding_rate_8h[spike]` | 13.3 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `jup_mc_fees_ratio[spike]` | 12.0 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `pepe_realized_volatility_7d[spike]` | 11.7 | 0/3 | UNSTABLE (relevance dies after train) |
| staking_yield | `tao_funding_rate_8h[spike]` | 11.6 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `jup_beta_to_btc_30d[spike]` | 11.1 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `ena_fees_usd[spike]` | 11.0 | 0/2 | UNSTABLE (relevance dies after train) |
| staking_yield | `hype_funding_premium[spike]` | 10.3 | 0/1 | UNSTABLE (relevance dies after train) |
| staking_yield | `jup_funding_premium[spike]` | 10.1 | 0/2 | UNSTABLE (relevance dies after train) |

## Gated 2SLS A/B with the shortlist instruments

One run per instrument (others exogenous), so short-history candidates keep their full sample. Gate: per-window first-stage partial F ≥ 10.0. Win-rates are walk-forward folds vs BH BTC.

| Treatment | Instrument | Gate (passed/seen) | mean F | OLS wr | 2SLS wr | OLS medSh | 2SLS medSh | OOS days |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| liq_flow | `aave_net_treasury[diff]` | 108/108 | 258.8 | 50% | 50% | 0.831 | 0.485 | 540 |
| cex_dex_flow | `eth_total_economic_activity[diff]` | 108/108 | 69.8 | 50% | 50% | 0.831 | 0.668 | 540 |
| funding_basis | `ena_tvl_usd[diff]` | 46/78 | 16.2 | 100% | 67% | 1.095 | 0.458 | 386 |
| staking_yield | `eth_queue_active_amount[diff]` | 0/107 | 0.8 | 50% | 50% | 0.030 | 0.030 | 534 |
| chain_congestion | `eth_blob_size_mib[spike]` | 1/81 | 1.3 | 67% | 67% | 0.760 | 0.760 | 405 |
| stable_flow | `eth_etf_aum_native[diff]` | 0/55 | 3.9 | 50% | 50% | 1.236 | 1.236 | 273 |
| mev_pressure | `bnb_fees_native[spike]` | 0/108 | 2.7 | 50% | 50% | 0.831 | 0.831 | 540 |

- **liq_flow** ← `aave_net_treasury[diff]`: protocol treasury marks move with DeFi liquidity (F=392, 3/3)
- **cex_dex_flow** ← `eth_total_economic_activity[diff]`: on-chain activity precedes exchange flows (F=55, 3/3, direct-path t=0.09)
- **funding_basis** ← `ena_tvl_usd[diff]`: Ethena TVL ~ basis-trade size: USDe is minted by shorting perps for funding (F=100)
- **staking_yield** ← `eth_queue_active_amount[diff]`: validator entry queue mechanically dilutes APR (F=88)
- **chain_congestion** ← `eth_blob_size_mib[spike]`: blob-space demand shocks congest blockspace (F=21, post-Dencun only)
- **stable_flow** ← `eth_etf_aum_native[diff]`: ETF demand shock -> stablecoin on-ramp (F=23, 2024+ only)
- **mev_pressure** ← `bnb_fees_native[spike]`: cross-chain activity bursts (weak: F=15, 1/3)

### Verdict

3 instrument(s) engaged the gate and made 2SLS genuinely diverge from OLS. Of those, **3 made OOS performance worse** and 0 made it better. With strong, non-tautological instruments the causal arm still does not beat the correlational arm — consistent with the structural finding that the DAG encodes no unobserved confounding for 2SLS to purge: when there is no endogeneity bias to remove, instrumenting only adds estimation variance. Neither arm robustly beats BH BTC.
