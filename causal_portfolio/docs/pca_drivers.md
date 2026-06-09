# PCA-orthogonalized drivers vs raw factors

Same optimizer / cadence / train window; only the OLS regressors differ — the 14 raw CPCM factors vs their causal principal components (fit on the trailing window, 90% variance kept). Scored OOS vs BH BTC.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | train 252d, rebalance 5d
- Mean components kept: 7.3
- Run UTC: `2026-06-09T17:57:51+00:00`

## Full OOS window
| Arm | Total | Sharpe |
|---|---:|---:|
| Raw factors | +55.7% | 0.710 |
| PCA drivers | +28.2% | 0.419 |
| BH BTC | +291.8% | 1.127 |

## Walk-forward folds vs BH BTC

2 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 2 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Arm | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| Raw | 44% | 0.609 | 9 |
| PCA | 44% | 0.579 | 9 |

## Verdict

PCA and raw drivers tied OOS (44% fold win-rate vs BH). Orthogonalizing didn't change the generalization story.