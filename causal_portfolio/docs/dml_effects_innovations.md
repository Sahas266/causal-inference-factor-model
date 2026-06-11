# Double ML effect estimates (Direction 4)

Cross-fitted LinearDML: effect of each factor (at t) on next-day return, residualizing both on all other factors with time-ordered folds. Effects are bps of next-day return per +1σ of the factor; 95% CIs from statsmodels inference.

**28 tests** — Bonferroni bar for family-wise 5% is CI exclusion at α=0.0018; the table flags plain 95% exclusions, so discount accordingly.

- Run UTC: `2026-06-10T23:55:09+00:00`

| Treatment | Outcome | Effect (bps/σ) | 95% CI | sig? |
|---|---|---:|---|:-:|
| staking_yield | ew_next | +539.9 | [-191.4, +1271.2] |  |
| staking_yield | btc_next | +184.3 | [-249.4, +618.0] |  |
| mev_pressure | ew_next | +71.4 | [-129.8, +272.6] |  |
| chain_congestion | btc_next | +38.8 | [+8.4, +69.1] | **yes** |
| chain_congestion | ew_next | +37.1 | [+1.5, +72.7] | **yes** |
| funding_basis | ew_next | +25.3 | [+0.2, +50.4] | **yes** |
| t10y2y | ew_next | +24.3 | [-11.2, +59.8] |  |
| stable_flow | ew_next | +21.2 | [-1.3, +43.8] |  |
| cpiaucsl | btc_next | -18.8 | [-61.9, +24.3] |  |
| t10y2y | btc_next | +18.8 | [-6.5, +44.0] |  |
| liq_flow | ew_next | -18.7 | [-43.1, +5.7] |  |
| m2sl | btc_next | +17.7 | [-27.8, +63.1] |  |
| funding_basis | btc_next | +16.7 | [-0.4, +33.8] |  |
| vixcls | btc_next | +16.7 | [+1.1, +32.3] | **yes** |
| vixcls | ew_next | +14.3 | [-11.3, +39.9] |  |
| cpiaucsl | ew_next | +10.9 | [-59.0, +80.7] |  |
| mev_pressure | btc_next | -10.8 | [-140.3, +118.8] |  |
| dgs10 | ew_next | -9.0 | [-42.5, +24.4] |  |
| cex_dex_flow | btc_next | +8.3 | [-6.4, +22.9] |  |
| dff | ew_next | -7.1 | [-36.2, +22.0] |  |
| liq_flow | btc_next | -6.5 | [-20.7, +7.7] |  |
| stable_flow | btc_next | +6.3 | [-9.0, +21.6] |  |
| dgs10 | btc_next | +5.4 | [-18.7, +29.6] |  |
| dff | btc_next | +5.3 | [-9.5, +20.2] |  |
| dtwexbgs | btc_next | -4.6 | [-26.0, +16.9] |  |
| cex_dex_flow | ew_next | +4.5 | [-18.2, +27.2] |  |
| m2sl | ew_next | +2.0 | [-68.2, +72.2] |  |
| dtwexbgs | ew_next | +0.7 | [-28.8, +30.3] |  |

## Heterogeneity (CausalForestDML): `staking_yield` on next-day EW return

| Moderator | effect @ 20th pct (CI) | effect @ 80th pct (CI) |
|---|---|---|
| vixcls | +4901.3 bps [-3656.2, +13458.9] | +1347.1 bps [-3493.3, +6187.4] |
| funding_basis | +1920.4 bps [-3656.0, +7496.9] | +706.3 bps [-935.8, +2348.4] |

## Verdict

4/28 effects clear plain 95% CIs: funding_basis→ew_next (+25.3 bps), chain_congestion→btc_next (+38.8 bps), chain_congestion→ew_next (+37.1 bps), vixcls→btc_next (+16.7 bps). At 28 tests, ~1.4 false positives are expected; none survives Bonferroni unless its CI is far from zero. Check the heterogeneity table before reading anything into these.