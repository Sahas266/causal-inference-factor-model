# Regime × DAG (exploratory)

Crosses the causal factor loadings with HMM regime labels (risk-on/off from VIX + BTC realized vol, forward-filter = causal). Pooled OLS loadings vs regime-conditional OLS loadings (fit only on past days sharing the current regime), same optimizer / cadence / factor pool, scored OOS vs BH BTC.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | 2 regimes | train 252d, rebalance 5d, regime lookback 504d
- Run UTC: `2026-06-08T17:54:03+00:00`

## Regime occupancy (causal labels)

| Regime | Days | % | Mean run (d) |
|---|---:|---:|---:|
| 0 | 757 | 64% | 40 |
| 1 | 431 | 36% | 23 |

Regime-specific fits used at 188 rebalances; fell back to pooled at 0 (insufficient same-regime history).

## Full OOS window

| Arm | Total | Sharpe |
|---|---:|---:|
| Pooled | +59.6% | 0.904 |
| Regime | -5.9% | 0.040 |
| BH BTC | +238.5% | 1.245 |

## Walk-forward folds vs BH BTC

2 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 2 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Arm | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| Pooled | 57% | 1.222 | 7 |
| Regime | 57% | 0.137 | 7 |

## Verdict

Regime conditioning did NOT help: 57% vs pooled 57% fold win-rate vs BH. Splitting the sample by regime cut estimation data without a generalizing payoff — and neither arm beats buy-and-hold BTC. Consistent with every prior result in this project.