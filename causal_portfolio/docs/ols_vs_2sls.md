# OLS vs 2SLS — does causal identification generalize OOS?

Same walk-forward, optimizer, covariance, cadence and train window for both arms; the only difference is OLS vs 2SLS estimation of the per-asset factor loadings each window. 2SLS instruments an endogenous factor only when its first-stage partial F clears the weak-instrument gate (F ≥ 10.0).

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | train 252d, rebalance 5d
- Run UTC: `2026-06-08T18:11:04+00:00`

## Instrument strength (first-stage partial F gate)

| Instrument | Windows passed / seen | Mean F |
|---|---:|---:|
| `gas_spike` | 0/108 | 0.44 |
| `liquidation_level` | 0/108 | 0.03 |
| `protocol_event` | 0/108 | 0.28 |
| `stablecoin_mint` | 0/108 | 1.35 |

## Full OOS window

| Arm | Total | Sharpe |
|---|---:|---:|
| OLS | +11.2% | 0.420 |
| 2SLS | +11.2% | 0.420 |
| BH BTC | +51.7% | 0.839 |

## Walk-forward folds vs BH BTC

2 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 2 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Arm | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| OLS | 50% | 0.831 | 4 |
| 2SLS | 50% | 0.831 | 4 |

## Verdict

2SLS produced results **identical** to OLS: every instrument failed the F ≥ 10.0 gate, so 2SLS reduced to OLS by design. Fold win-rate vs BH BTC 50% for both — the instruments add nothing exploitable, and the correlational fit itself does not reliably beat holding BTC.