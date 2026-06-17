# On-chain demand/flow DAGs (state -> risk-on/off -> return)

Each rule rotates one asset between full exposure and 0% (cash) on a single **on-chain demand/flow** state variable. Decision at day t (data through t, plus a 1-day publication lag on the raw series), applied to day t+1's return, 5bp one-way switching cost. Every signal is causal (trailing 30d MA cross or trailing z-score; no full-sample thresholds). Benchmark is buy-and-hold of the SAME asset.

- Window `2021-01-01` -> `2025-12-31`
- Placebo: 200 circular time-shifts of the exposure path; p = share matching/beating the real rule's Sharpe. High p => the timing carries no information.
- Run UTC: `2026-06-17T16:01:00+00:00`

## BTC

| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | placebo p | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| exch_flow[btc|FlowInExNtv-FlowOutExNtv] | +40% | 0.74 | -69% | 0.58 | 91% | 243 | 0.09 | 2020-12-31->2025-12-31 |
| always_in <- BH | +37% | 0.63 | -77% | 0.48 | 100% | 1 | — | — |
| stbl_supply[btc|USDT+USDC+USDe] | +33% | 0.64 | -69% | 0.47 | 73% | 50 | 0.39 | 2020-12-31->2025-12-31 |
| tvl_trend[btc|tvl_usd] | +20% | 0.49 | -67% | 0.29 | 55% | 131 | — | 2021-03-20->2025-12-31 |
| addr_growth[btc|AdrActCnt] | +19% | 0.47 | -73% | 0.26 | 51% | 673 | — | 2020-12-31->2025-12-31 |
| exch_flow_strict[btc|FlowInExNtv-FlowOutExNtv] | +16% | 0.44 | -63% | 0.25 | 40% | 745 | — | 2020-12-31->2025-12-31 |
| fee_demand[btc|fees] | +2% | 0.05 | -65% | 0.03 | 36% | 323 | — | 2020-12-31->2025-12-31 |

Buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48.

## ETH

| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | placebo p | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stbl_supply[eth|USDT+USDC+USDe] | +53% | 0.78 | -65% | 0.82 | 73% | 50 | 0.26 | 2020-12-31->2025-12-31 |
| always_in <- BH | +57% | 0.73 | -79% | 0.72 | 100% | 1 | — | — |
| fee_demand[eth|fees] | +28% | 0.65 | -45% | 0.64 | 39% | 352 | — | 2020-12-31->2025-12-31 |
| exch_flow_strict[eth|cex_netflow_usd] | +27% | 0.49 | -62% | 0.43 | 51% | 629 | — | 2020-12-31->2025-12-30 |
| addr_growth[eth|AdrActCnt] | +29% | 0.57 | -67% | 0.43 | 46% | 431 | — | 2020-12-31->2025-12-31 |
| exch_flow[eth|cex_netflow_usd] | +33% | 0.47 | -82% | 0.40 | 79% | 423 | — | 2020-12-31->2025-12-30 |
| tvl_trend[eth|tvl_usd] | +16% | 0.28 | -77% | 0.20 | 50% | 144 | — | 2020-12-31->2025-12-31 |

Buy-and-hold ETH: ann +57%, Sharpe 0.73, maxDD -79%, Calmar 0.72.

## Verdict

Rules that beat BH on Sharpe AND whose timing survives the placebo (p < 0.10):
- **exch_flow[btc|FlowInExNtv-FlowOutExNtv]** (BTC) — Sharpe 0.74 vs 0.63, maxDD -69% vs -77%, 243 switches (placebo p=0.09)

Read this conservatively: a placebo p near the 0.10 cutoff is a weak pass, not a strong edge. Survivors with heavy turnover (hundreds of switches) are also the most cost- and lag-sensitive, and none clears BH out-of-sample on an independent cycle (we only have one).

### Caveats

- **Uneven, late-starting coverage.** On-chain series differ in history (see the coverage column). BTC `tvl_usd` starts 2021-03; USDe joins the stablecoin aggregate only in late 2023; `cex_netflow_usd` is ETH-only (BTC uses a native FlowIn-FlowOut proxy in different units).
- **Publication lag.** A 1-day lag is applied to every raw series before the harness's own 1-day decision->return lag; real feeds can be later or revised. Gaps are forward-filled at most 5 days, so a stale read never carries indefinitely.
- **One cycle.** 2021-2025 is a single bull/bear/bull regime. These rules are not validated out-of-sample on an independent cycle; a placebo pass is necessary, not sufficient.
- **Cash, not stable yield.** Risk-off earns 0%. A real stable yield would only improve every gated rule, never hurt it.
- **Consistent with the repo's standing finding:** buy-and-hold is the benchmark nothing here beats out-of-sample.