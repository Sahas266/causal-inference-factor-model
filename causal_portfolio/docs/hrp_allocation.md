# Hierarchical Risk Parity (HRP) vs Equal-Weight vs BH BTC

Correlation-distance hierarchical clustering -> quasi-diagonal -> recursive bisection (López de Prado HRP), long-only, refit on a trailing window. Pure correlation-based diversification, no alpha signal. Scored OOS vs an equal-weight book and BH BTC.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge, xrp, crv, ltc`
- Window: `2022-01-01` → `2025-12-31` | train 252d, rebalance 21d
- Run UTC: `2026-06-09T18:00:53+00:00`

## Full OOS window
| Strategy | Total | Sharpe | MaxDD |
|---|---:|---:|---:|
| HRP | +281.4% | 0.976 | -46.3% |
| Equal-Weight | +150.9% | 0.747 | -54.8% |
| BH BTC | +302.6% | 1.127 | -32.1% |

## Walk-forward folds vs BH BTC

2 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 2 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Strategy | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| HRP | 56% | 0.441 | 9 |
| EqualWeight | 33% | 0.535 | 9 |

## Verdict

HRP did not beat BH BTC (56% of folds; equal-weight 33%). Smarter correlation-based weighting of the same long crypto universe still doesn't escape BTC's dominance — the problem isn't weighting, it's that the universe is a high-correlation, BTC-led basket. Diversification reduces drawdown but caps upside below BH BTC.