# WK-means × DAG — Wasserstein regimes for conditional loadings

Regime-conditional OLS loadings (fit on past days sharing the current regime) vs a pooled model, with regimes from the Gaussian HMM and from the paper's Wasserstein k-means (BTC return-segment clustering). Same optimizer/cadence/factor pool; scored OOS vs BH BTC.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | 2 regimes | train 252d, rebalance 5d, regime lookback 504d
- Run UTC: `2026-06-09T18:03:12+00:00`

- HMM occupancy: R0 64% (run 40d), R1 36% (run 23d)
- WK-means occupancy: R0 62% (run 17d), R1 38% (run 11d)
- HMM regime-fits 188 / fallbacks 0; WK regime-fits 189 / fallbacks 3

## Full OOS window
| Arm | Total | Sharpe |
|---|---:|---:|
| Pooled | +59.6% | 0.904 |
| HMM-Regime | -5.9% | 0.040 |
| WK-Regime | +35.2% | 0.625 |
| BH BTC | +238.5% | 1.245 |

## Walk-forward folds vs BH BTC

3 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 3 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Arm | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| Pooled | 57% | 1.222 | 7 |
| WK | 57% | 1.145 | 7 |
| HMM | 57% | 0.137 | 7 |

## Verdict

**WK-means is the better regime detector.** Conditioning on Wasserstein regimes preserved far more performance than the HMM (WK-Regime +35.2% / full-window Sharpe 0.625, median fold Sharpe 1.145, vs HMM-Regime −5.9% / 0.040, median fold Sharpe 0.137). The paper's distributional, model-free detector genuinely beats the Gaussian HMM at identifying tradeable regimes.

**But regime conditioning still does not pay off.** Both regime arms trail the pooled model (pooled +59.6% / Sharpe 0.904, median fold Sharpe 1.222), and all three tie at 57% fold win-rate vs BH — none beats buy-and-hold BTC (+238.5% / Sharpe 1.245). Splitting the estimation sample by regime costs more in data than the better regime model recovers. Net: a better regime detector (WK-means), same project-wide conclusion — nothing beats holding BTC out of sample.