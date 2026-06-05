# DAG Variant Search

Each variant is a different return-equation structure (which factors are direct causes of returns, at what lag, with/without Combo selection), run through the identical downstream pipeline (V1 solver + EKF + manifold optimizer + walk-forward backtest). Differences are attributable to DAG structure alone.

**Lag grid:** every factor-pool config is run both contemporaneous (`_lag0`, edges use same-day factor values) and fully lagged (`_lag1`, edges use prior-day values *everywhere* — predictive, no contemporaneous look-ahead). This tests lagging across all configs.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Full window: `2022-01-01` → `2025-12-31`
- OOS window: `2024-01-01` → `2025-12-31`
- Run UTC: `2026-06-05T03:45:28+00:00`

## FULL window

**Buy & Hold BTC:** total +9.7%, Sharpe 0.248, MaxDD -72.9%

| Variant | Drivers | Total | Sharpe | Sortino | MaxDD | Beats BH? |
|---|---|---:|---:|---:|---:|:--:|
| onchain_core_lag0 | chain_congestion, mev_pressure, liq_flow | -45.3% | -0.455 | -0.446 | -55.0% | — |
| macro_lag0 | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | -30.5% | -0.346 | -0.340 | -47.9% | — |
| combo3_lag0 | cex_dex_flow, vixcls, t10y2y | -24.0% | -0.109 | -0.107 | -40.3% | — |
| global_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | +5.4% | 0.149 | 0.152 | -41.8% | — |
| macro_lag1 | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +9.4% | 0.201 | 0.202 | -32.7% | — |
| combo3_lag1 | chain_congestion, dff, vixcls | +14.6% | 0.226 | 0.217 | -46.6% | ✓ |
| onchain_core_lag1 | chain_congestion, mev_pressure, liq_flow | +36.9% | 0.393 | 0.412 | -36.9% | ✓ |
| all_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +49.5% | 0.563 | 0.595 | -22.7% | ✓ |
| all_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +69.0% | 0.723 | 0.743 | -16.1% | ✓ |
| global_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | +101.8% | 0.884 | 0.920 | -29.7% | ✓ |

## OOS window

**Buy & Hold BTC:** total +60.1%, Sharpe 0.647, MaxDD -33.1%

| Variant | Drivers | Total | Sharpe | Sortino | MaxDD | Beats BH? |
|---|---|---:|---:|---:|---:|:--:|
| onchain_core_lag1 | chain_congestion, mev_pressure, liq_flow | -33.3% | -0.920 | -0.862 | -39.5% | — |
| global_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | -21.8% | -0.579 | -0.526 | -47.1% | — |
| combo3_lag1 | liq_flow, chain_congestion, t10y2y | -18.0% | -0.261 | -0.238 | -35.8% | — |
| all_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | -7.3% | -0.059 | -0.054 | -19.9% | — |
| global_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | -0.8% | 0.107 | 0.105 | -23.2% | — |
| all_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +4.0% | 0.238 | 0.246 | -17.7% | — |
| combo3_lag0 | chain_congestion, mev_pressure, vixcls | +7.5% | 0.292 | 0.280 | -25.5% | — |
| onchain_core_lag0 | chain_congestion, mev_pressure, liq_flow | +8.6% | 0.333 | 0.336 | -31.2% | — |
| macro_lag0 | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +13.9% | 0.452 | 0.454 | -12.0% | — |
| macro_lag1 | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +31.4% | 1.045 | 1.083 | -10.9% | — |

## Verdict

**No variant beats Buy & Hold BTC in both windows.** Restructuring the DAG's return equation does not, on this data, produce a configuration that robustly beats simply holding BTC. Consistent with every other strategy result in this project.

## Lag analysis (lag-0 vs lag-1-everywhere, per config)

Lagging every edge by one day, applied to all configs:

| Config | Full lag0 | Full lag1 | OOS lag0 | OOS lag1 | Lag effect |
|---|---:|---:|---:|---:|---|
| combo3 | −24.0% | +14.6% | +7.5% | −18.0% | helps in-sample, **hurts OOS** |
| all | +69.0% | +49.5% | −7.3% | +4.0% | hurts in-sample, mildly helps OOS (both lose) |
| global | +5.4% | **+101.8%** | −21.8% | −0.8% | huge in-sample jump, **collapses OOS** |
| macro | −30.5% | +9.4% | +13.9% | **+31.4%** | helps **both** windows |
| onchain_core | −45.3% | +36.9% | +8.6% | −33.3% | helps in-sample, **destroys OOS** |

Three conclusions:

1. **Lagging swings in-sample results enormously — and most of it is
   overfit.** `global_lag1` jumped +96pp in-sample (+5.4% → +101.8%) yet
   went from −21.8% to −0.8% OOS — still a loss. `onchain_core_lag1` gained
   +82pp in-sample then posted the **single worst OOS result** in the whole
   grid (−33.3%). A one-day lag does not plausibly add ~90pp of real edge;
   it's aligning the factor with the next-period return in a way that fit
   the 2022-23 path and doesn't repeat.

2. **The only config where lagging helps in BOTH windows is `macro`.**
   `macro_lag1` is the most robust arrangement in the entire search —
   OOS Sharpe **1.045**, max DD just −10.9%, and lagging improved it over
   `macro_lag0` in both windows. Slow-moving macro series genuinely carry
   predictive (lagged) information that on-chain factors don't. Its total
   return (+31.4% OOS) still trails buy-and-hold BTC (+60.1%), so it is not
   a winner — but it is the one structurally-sound signal the search found.

3. **Lag-everywhere does not rescue any config past buy-and-hold BTC.**
   Across all 10 configs and both lag settings, nothing clears the BH bar
   out of sample. Lagging changes *which* configs look good in-sample but
   not the bottom line.

### Net takeaway on lag structure

- Contemporaneous (`lag0`) results are contaminated by same-day look-ahead;
  the lag-1 forms are the honest predictive versions and should be preferred
  on principle.
- But the dramatic in-sample improvements from lagging the on-chain pools
  are overfit and vanish OOS — a textbook case for why the two-window
  discipline is non-negotiable here.
- `macro_lag1` is the single arrangement worth remembering: genuinely
  predictive, most OOS-robust, lowest drawdown — yet still short of BH BTC.
