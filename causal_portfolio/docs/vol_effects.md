# Factor effects on volatility + vol targeting (Direction 5)

Part 1: forward 5d realized vol (EW basket) on factors at t, controlling for trailing 5d vol, HAC(10) errors. Part 2: walk-forward vol-targeting backtest — if factors carry vol information beyond persistence, the factor-augmented arm should beat plain trailing-vol targeting.

- Window: `2022-01-01` → `2025-12-31` | n=787 | train 252d

- Run UTC: `2026-06-10T23:27:04+00:00`

## Part 1 — incremental vol information (HAC t-stats)

Trailing 5d vol t-stat: **0.4** (persistence dominates, as expected).

| Factor | coef (vol/σ) | t | p |
|---|---:|---:|---:|
| vixcls | +0.00458 | 2.71 ** | 0.007 |
| t10y2y | -0.01425 | -2.61 ** | 0.009 |
| liq_flow | -0.00106 | -2.40 ** | 0.016 |
| chain_congestion | -0.00159 | -1.61 | 0.107 |
| funding_basis | +0.00137 | 1.56 | 0.118 |
| dtwexbgs | +0.00278 | 1.35 | 0.175 |
| dff | -0.01106 | -0.84 | 0.401 |
| cex_dex_flow | -0.00073 | -0.82 | 0.414 |
| staking_yield | +0.00776 | 0.53 | 0.598 |
| stable_flow | -0.00029 | -0.48 | 0.632 |
| dgs10 | -0.00151 | -0.43 | 0.667 |
| cpiaucsl | +0.00946 | 0.38 | 0.703 |
| mev_pressure | -0.00145 | -0.31 | 0.757 |
| m2sl | +0.00252 | 0.21 | 0.836 |

3/14 factors significant at 5% after the trailing-vol control (~0.7 expected by chance).

## Part 2 — vol-targeting backtest (BH BTC)

| Arm | OOS Sharpe | Total | Realized vol | max leverage |
|---|---:|---:|---:|---:|
| constant | 0.839 | +50.9% | 45.3% | 1.00 |
| trail_vol_target | 0.657 | +34.0% | 46.7% | 2.00 |
| factor_vol_target | 1.158 | +91.6% | 47.5% | 2.00 |

## Part 2 — vol-targeting backtest (EW basket)

| Arm | OOS Sharpe | Total | Realized vol | max leverage |
|---|---:|---:|---:|---:|
| constant | 0.526 | +19.4% | 71.3% | 1.00 |
| trail_vol_target | 0.679 | +40.8% | 75.5% | 2.00 |
| factor_vol_target | 0.970 | +85.5% | 63.9% | 2.00 |

## Verdict

Factor-augmented vol targeting vs plain trailing-vol targeting: ΔSharpe +0.501 (BTC), +0.292 (EW). 

**Placebo check (circular factor shifts):** p=0.43 (BTC), p=0.47 (EW) — the fraction of random factor alignments whose vol-target arm does at least as well.

**The apparent improvement does not survive the placebo**: a large share of random factor alignments produce equal or better Sharpe, i.e. ANY time-varying leverage overlay tends to score in this OOS window — the gain is leverage-timing luck, not factor information. The HAC-significant coefficients in Part 1 (mostly VIX, itself a vol index) are real statistically but already subsumed by trailing vol + noise at the portfolio level.