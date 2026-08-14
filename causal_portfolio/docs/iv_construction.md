# IV construction — richer transforms + over-identification (Sargan)

Extends the warehouse IV search beyond diff/spike with five new causal, lag-1 instrument constructions and adds multi-instrument over-identification tests. Screened **1602** candidate columns × 7 transforms (`diff, spike, ar1_innov, event, sign, qbucket, cumchg7`) against every global treatment. Relevance bar: train-window first-stage F ≥ 10.0; selection uses the first 252 aligned rows only.

**Multiple-testing warning:** transforms × metrics × treatments is a large grid; the count of clean strong hits is upward-biased by selection. Exclusion is untestable in general (the screen is purely statistical) and this is ONE market cycle. The ultimate arbiter remains OOS 2SLS-vs-OLS, below.

## 1. Per-transform candidate yield

Which construction surfaces strong (F≥10), non-tautological, low-direct-path candidates — and how many survive to *strong & stable* (clean, F≥20, ≥1 later window holds relevance).

| Transform | Relevant (F≥10) | Clean | Strong & stable | Best clean candidate (treatment, F) |
|---|---:|---:|---:|---|
| `diff` | 570 | 317 | 193 | `eth_gross_emissions[diff]` — liq_flow, F=1031 |
| `spike` | 373 | 64 | 38 | `pol_application_fees[spike]` — cex_dex_flow, F=70 |
| `ar1_innov` *(new)* | 721 | 399 | 247 | `eth_gross_emissions[ar1_innov]` — liq_flow, F=1030 |
| `event` *(new)* | 373 | 64 | 38 | `pol_application_fees[event]` — cex_dex_flow, F=70 |
| `sign` *(new)* | 298 | 199 | 118 | `eth_etf_aum[sign]` — liq_flow, F=225 |
| `qbucket` *(new)* | 1448 | 958 | 576 | `jup_net_treasury[qbucket]` — chain_congestion, F=596 |
| `cumchg7` *(new)* | 808 | 365 | 220 | `eth_circulating_supply_native[cumchg7]` — funding_basis, F=215 |

## 2. Strongest clean candidate per treatment, by transform

Best clean (no tautology/direct-path/unstable flag) candidate for each treatment under each NEW transform, vs the diff/spike baseline. Blank = no clean candidate cleared F≥10 for that cell.

| Treatment | `diff` | `spike` | `ar1_innov` | `event` | `sign` | `qbucket` | `cumchg7` |
|---|---|---|---|---|---|---|---|
| cex_dex_flow | eth_total_economic_activity F=55 (3/3) | pol_application_fees F=70 (1/3) | eth_total_economic_activity F=154 (3/3) | pol_application_fees F=70 (1/3) | aero_active_revenue F=39 (2/2) | eth_total_economic_activity F=146 (3/3) | jup_tvl F=121 (2/3) |
| chain_congestion | ena_tvl_usd F=29 (1/2) | eth_avg_mib_per_second F=21 (1/2) | tao_staked_alpha F=213 (0/0) | eth_avg_mib_per_second F=21 (1/2) | usde_tvl_usd F=33 (2/2) | jup_net_treasury F=596 (2/2) | aero_total_supply_native F=81 (2/2) |
| funding_basis | ena_tvl_usd F=100 (1/2) | btc_earnings F=25 (1/2) | ena_tvl_usd F=103 (1/2) | btc_earnings F=25 (1/2) | eth_SplyCur F=80 (1/2) | jup_net_treasury F=495 (2/2) | eth_circulating_supply_native F=215 (1/2) |
| liq_flow | eth_gross_emissions F=1031 (2/3) | uni_tvl F=32 (1/3) | eth_gross_emissions F=1030 (2/3) | uni_tvl F=32 (1/3) | eth_etf_aum F=225 (1/1) | bnb_mc_fees_ratio F=51 (1/3) | eth_total_staked F=61 (3/3) |
| mev_pressure | — | jup_own_token_treasury F=27 (1/2) | — | jup_own_token_treasury F=27 (1/2) | — | aero_sharpe_30d F=59 (1/2) | usde_tvl_usd F=19 (1/2) |
| stable_flow | eth_etf_aum_to_mcap F=24 (1/1) | eth_l2_settlement_value_usd F=23 (1/3) | eth_etf_aum_to_mcap F=24 (1/1) | eth_l2_settlement_value_usd F=23 (1/3) | — | jup_net_treasury F=19 (2/2) | eth_etf_aum_to_mcap F=12 (1/1) |
| staking_yield | eth_queue_active_amount F=87 (1/3) | hype_fees_usd F=13 (0/0) | tao_staked_alpha F=235 (0/0) | hype_fees_usd F=13 (0/0) | usde_stablecoin_circulating_usd F=51 (1/2) | jup_net_treasury F=324 (2/2) | eth_queue_active_amount F=144 (2/3) |

## 3. Multi-instrument over-identification (Sargan)

For treatments with ≥2 clean strong instruments of LOW mutual correlation (|corr| ≤ 0.6, so they carry independent relevance), over-identified 2SLS on the train window with the mean cross-asset next-day return as the structural outcome. A VALID instrument set should **not reject** (Sargan p > 0.05). We report the estimators.py Rust-parity Sargan AND an independently recomputed standard Sargan (n·R² of 2SLS residuals on all instruments+exog, χ²(m−k)) as a cross-check.

| Treatment | Instruments | n | min 1st-F | max |corr| | Sargan (std) | p (std) | Sargan (rust) | p (rust) | df | Valid set? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cex_dex_flow | `eth_total_economic_activity[ar1_innov]`, `avax_total_economic_activity[qbucket]`, `avax_artemis_stablecoin_daily_txns[qbucket]` | 223 | 92 | 0.48 | 0.21 | 0.900 | 0.14 | 0.704 | 2 | not rejected |
| funding_basis | `jup_net_treasury[qbucket]`, `jup_own_token_treasury[qbucket]`, `ena_tvl_usd[cumchg7]` | 76 | 1 | 0.45 | 1.29 | 0.526 | 1.25 | 0.263 | 2 | not rejected |
| liq_flow | `eth_gross_emissions[diff]`, `link_mc_fees_ratio[ar1_innov]`, `eth_chain_lending_deposits[diff]` | 251 | 129 | 0.60 | 0.92 | 0.630 | 0.26 | 0.611 | 2 | not rejected |

**3/3** over-identified set(s) were NOT rejected by the standard Sargan at 5%.
A non-rejection is necessary but NOT sufficient for validity: Sargan only tests whether the instruments *agree* on the same structural β — if they share the SAME exclusion violation it cannot detect it. With one structural equation and one cycle it is low-powered. Treat a pass as 'no internal contradiction', not 'exclusion confirmed'.

## 4. Gated 2SLS-vs-OLS OOS A/B with the best NEW constructions

Best CLEAN candidate per treatment whose transform is new to this module (tie-break: stability, then F), each run as the sole instrument in the walk-forward A/B (others exogenous). Gate: per-window first-stage partial F ≥ 10.0.

| Treatment | Instrument | Gate | mean F | OLS medSh | 2SLS medSh | OLS wr | 2SLS wr | OOS days |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cex_dex_flow | `eth_total_economic_activity[ar1_innov]` | 108/108 | 80.6 | 0.831 | 0.398 | 50% | 50% | 540 |
| chain_congestion | `jup_net_treasury[qbucket]` | 16/85 | 18.9 | -0.156 | -0.156 | 33% | 33% | 424 |
| funding_basis | `jup_net_treasury[qbucket]` | 11/85 | 4.5 | -0.156 | -0.156 | 33% | 33% | 424 |
| liq_flow | `sol_chain_median_txn_fee[ar1_innov]` | 90/108 | 19.5 | 0.831 | 0.031 | 50% | 50% | 540 |
| mev_pressure | `sol_active_revenue[qbucket]` | 3/108 | 2.7 | 0.831 | 0.897 | 50% | 50% | 540 |
| stable_flow | `eth_etf_aum_to_mcap[ar1_innov]` | 0/55 | 4.3 | 1.236 | 1.236 | 50% | 50% | 273 |
| staking_yield | `jup_net_treasury[qbucket]` | 18/85 | 6.0 | -0.156 | -1.159 | 33% | 33% | 424 |

### Verdict

4 instrument(s) engaged the gate; **1 made 2SLS ≥ OLS OOS** (median Sharpe). This would be the first construction to flip the result — discount heavily by the gate column and the one-cycle / multiple-testing caveats before believing it.
