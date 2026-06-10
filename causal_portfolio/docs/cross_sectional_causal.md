# Cross-sectional market-neutral test (Direction 1)

Beta-hedged residual returns (trailing β known at t-1), predicted from lag-1 global factors via per-asset OLS, traded dollar-neutral. The market factor — the thing BH BTC owns — is hedged out, so this isolates whether differential factor sensitivities carry relative-return information.

- Assets: `b, t, c, ,, e, t, h, ,, s, o, l, ,, b, n, b, ,, a, v, a, x, ,, u, n, i, ,, a, a, v, e, ,, l, i, n, k, ,, d, o, g, e` (BTC = hedge leg)
- Window: `2022-01-01` → `2025-12-31` | train 252d | rebalance 5d | β window 90d
- Run UTC: `2026-06-10T23:17:09+00:00`

## All factors jointly

| Metric | Value |
|---|---:|
| OOS L/S Sharpe | 0.966 |
| Newey-West t (mean) | 1.13 |
| Mean rank IC (5d fwd) | 0.0567 |
| IC t-stat | 1.44 |
| Fold Sharpes | 0.62, 0.37, 3.02, 0.78 |
| Placebo p-value (50 circular shifts) | 0.14 |
| Placebo Sharpe range | [-1.20, 1.94] (median 0.08) |

## Single-factor L/S

| Factor | Sharpe | NW t | mean IC | IC t |
|---|---:|---:|---:|---:|
| liq_flow | -0.141 | -0.24 | 0.0044 | 0.17 |
| cex_dex_flow | -0.141 | -0.25 | 0.0016 | 0.07 |
| mev_pressure | -0.174 | -0.29 | -0.0023 | -0.10 |
| m2sl | -0.268 | -0.46 | -0.0101 | -0.35 |
| funding_basis | -0.354 | -0.42 | 0.0042 | 0.11 |
| staking_yield | -0.500 | -0.83 | -0.0033 | -0.13 |
| dff | -0.562 | -0.91 | -0.0038 | -0.15 |
| t10y2y | -0.582 | -0.99 | -0.0055 | -0.21 |
| chain_congestion | -0.754 | -1.32 | -0.0086 | -0.32 |
| stable_flow | -0.785 | -1.24 | -0.0069 | -0.27 |
| dtwexbgs | -0.792 | -1.39 | -0.0506 | -1.87 |
| dgs10 | -0.844 | -1.42 | -0.0335 | -1.26 |
| cpiaucsl | -0.886 | -1.54 | -0.0250 | -0.87 |
| vixcls | -1.088 | -1.84 | -0.0370 | -1.36 |

## Verdict

**No reliable cross-sectional signal.** Joint L/S Sharpe 0.966 (NW t=1.13) is indistinguishable from the circular-shift placebo (p=0.14). Hedging out the market factor does not reveal relative-return predictability from the CPCM factor loadings at the daily horizon.