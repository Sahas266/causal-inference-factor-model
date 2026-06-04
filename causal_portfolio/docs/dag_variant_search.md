# DAG Variant Search

Each variant is a different return-equation structure (which factors are direct causes of returns, at what lag, with/without Combo selection), run through the identical downstream pipeline (V1 solver + EKF + manifold optimizer + walk-forward backtest). Differences are attributable to DAG structure alone.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Full window: `2022-01-01` → `2025-12-31`
- OOS window: `2024-01-01` → `2025-12-31`
- Run UTC: `2026-06-04T15:38:50+00:00`

## FULL window

**Buy & Hold BTC:** total +9.7%, Sharpe 0.248, MaxDD -72.9%

| Variant | Drivers | Total | Sharpe | Sortino | MaxDD | Beats BH? |
|---|---|---:|---:|---:|---:|:--:|
| onchain_core | chain_congestion, mev_pressure, liq_flow | -45.3% | -0.455 | -0.446 | -55.0% | — |
| baseline_combo | cex_dex_flow, vixcls, t10y2y | -24.0% | -0.109 | -0.107 | -40.3% | — |
| global_only | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | +5.4% | 0.149 | 0.152 | -41.8% | — |
| baseline_macro_lag1 | cex_dex_flow, dff, vixcls | +0.9% | 0.168 | 0.168 | -51.3% | — |
| macro_only | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +9.4% | 0.201 | 0.202 | -32.7% | — |
| combo_all_lag1 | chain_congestion, dff, vixcls | +14.6% | 0.226 | 0.217 | -46.6% | ✓ |
| all_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +49.5% | 0.563 | 0.595 | -22.7% | ✓ |
| all_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +69.0% | 0.723 | 0.743 | -16.1% | ✓ |

## OOS window

**Buy & Hold BTC:** total +60.1%, Sharpe 0.647, MaxDD -33.1%

| Variant | Drivers | Total | Sharpe | Sortino | MaxDD | Beats BH? |
|---|---|---:|---:|---:|---:|:--:|
| global_only | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow | -21.8% | -0.579 | -0.526 | -47.1% | — |
| combo_all_lag1 | liq_flow, chain_congestion, t10y2y | -18.0% | -0.261 | -0.238 | -35.8% | — |
| baseline_macro_lag1 | chain_congestion, mev_pressure, cex_dex_flow | -12.9% | -0.234 | -0.220 | -31.1% | — |
| all_lag0 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | -7.3% | -0.059 | -0.054 | -19.9% | — |
| all_lag1 | liq_flow, chain_congestion, staking_yield, mev_pressure, cex_dex_flow, dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +4.0% | 0.238 | 0.246 | -17.7% | — |
| baseline_combo | chain_congestion, mev_pressure, vixcls | +7.5% | 0.292 | 0.280 | -25.5% | — |
| onchain_core | chain_congestion, mev_pressure, liq_flow | +8.6% | 0.333 | 0.336 | -31.2% | — |
| macro_only | dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs | +31.4% | 1.045 | 1.083 | -10.9% | — |

## Verdict

**No variant beats Buy & Hold BTC in both windows.** Restructuring the DAG's return equation does not, on this data, produce a configuration that robustly beats simply holding BTC. Consistent with every other strategy result in this project.

## Analysis — which DAGs produced the best results

Even though none beat buy-and-hold BTC out of sample, the search produced
clear, actionable findings about DAG *structure*:

**1. Denser DAGs dominate the sparse Combo-selected baseline (in-sample).**
The current pipeline uses a narrow DAG — Combo-selects just 3 driver edges
into the return node. That arrangement (`baseline_combo`) is among the
*worst*: −24.0% full window. The densest arrangement (`all_lag0`, all 12
available factors → returns) returns **+69.0%** full window with the best
Sharpe (0.723) and a −16.1% max drawdown vs BH's −72.9%. Adding every
factor edge beats the curated 3-driver DAG by ~93 percentage points
in-sample. **The Combo selection the pipeline relies on is discarding
useful structure.**

**2. But the in-sample winners are drawdown-avoidance, not generalizable
alpha.** `all_lag0` and `all_lag1` top the full window almost entirely by
dodging the 2022 crash (max DD −16%/−23% vs BH −73%). In the OOS window
(2024-25, no comparable crash) that edge evaporates: `all_lag0` → −7.3%,
both lose to BH's +60.1%. This is the same overfitting signature that has
recurred throughout the project — a structure that fits the one big
historical drawdown looks brilliant in-sample and mediocre forward.

**3. Most OOS-robust arrangement: `macro_only`.** Macro factors alone
(lagged 1 day) give the best OOS *risk-adjusted* return — Sharpe **1.045**,
max DD only −10.9% — more stable than any on-chain or dense arrangement.
Its total return (+31.4%) still trails buy-and-hold BTC (+60.1%), so it's
not a winner, but it's the one structure whose edge didn't fully collapse
OOS. A defensive (drawdown-minimizing) mandate might find it interesting;
a return-maximizing one would not.

**4. Lag structure matters and is currently mis-specified.** The stated DAG
puts macro on a 1-day lag, but the live backtest never applied any lag.
Contemporaneous (`all_lag0`) beats lagged (`all_lag1`) in-sample (+69% vs
+49.5%) but the gap reverses OOS (−7.3% vs +4.0%) — lagged is slightly more
robust forward, as you'd expect from removing a contemporaneous look-ahead
edge.

### Practical takeaways for the pipeline

- The current 3-driver Combo DAG is a **poor default** — if keeping the CPCM
  approach, wire **all factor edges** into the return node, not a curated 3.
- **Lag the macro edges** as the DAG actually specifies (the backtest
  silently ignores this today); it's modestly more robust OOS.
- None of this clears the bar of beating buy-and-hold BTC out of sample,
  which remains the honest benchmark for this project.

### Caveat

Eight structures were searched and ranked; the full-window "winners" should
be read as in-sample fits, not forward expectations. The two-window
(full + OOS) discipline is what exposed `all_lag0`'s collapse — without it
we'd have wrongly crowned a +69% "DAG improvement."
