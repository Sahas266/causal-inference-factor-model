# Double ML effect estimates (Direction 4)

Cross-fitted LinearDML: effect of each factor (at t) on next-day return, residualizing both on all other factors with time-ordered folds. Effects are bps of next-day return per +1σ of the factor; 95% CIs from statsmodels inference.

**28 tests** — Bonferroni bar for family-wise 5% is CI exclusion at α=0.0018; the table flags plain 95% exclusions, so discount accordingly.

- Run UTC: `2026-06-10T23:21:51+00:00`

| Treatment | Outcome | Effect (bps/σ) | 95% CI | sig? |
|---|---|---:|---|:-:|
| staking_yield | ew_next | +744.7 | [-204.8, +1694.2] |  |
| cpiaucsl | ew_next | -627.5 | [-1083.9, -171.1] | **yes** |
| cpiaucsl | btc_next | -317.3 | [-641.1, +6.5] |  |
| mev_pressure | ew_next | +125.8 | [-79.2, +330.8] |  |
| t10y2y | ew_next | +65.1 | [-6.5, +136.8] |  |
| t10y2y | btc_next | +61.6 | [+14.2, +109.0] | **yes** |
| staking_yield | btc_next | +52.4 | [-534.4, +639.1] |  |
| m2sl | btc_next | -51.3 | [-193.1, +90.5] |  |
| mev_pressure | btc_next | +45.6 | [-88.7, +180.0] |  |
| chain_congestion | ew_next | +45.3 | [+17.5, +73.0] | **yes** |
| chain_congestion | btc_next | +40.9 | [+17.7, +64.1] | **yes** |
| m2sl | ew_next | +26.4 | [-176.9, +229.6] |  |
| liq_flow | ew_next | -22.4 | [-46.9, +2.0] |  |
| stable_flow | ew_next | +20.1 | [-4.1, +44.4] |  |
| vixcls | btc_next | +15.1 | [-11.9, +42.0] |  |
| dtwexbgs | ew_next | -13.9 | [-37.9, +10.0] |  |
| funding_basis | ew_next | +13.5 | [-9.5, +36.4] |  |
| cex_dex_flow | btc_next | +12.3 | [-6.2, +30.8] |  |
| dff | btc_next | -11.8 | [-116.9, +93.4] |  |
| dff | ew_next | +9.8 | [-137.7, +157.4] |  |
| funding_basis | btc_next | +9.8 | [-9.6, +29.3] |  |
| liq_flow | btc_next | -8.6 | [-23.2, +5.9] |  |
| dgs10 | btc_next | +8.4 | [-41.9, +58.7] |  |
| stable_flow | btc_next | +8.3 | [-7.6, +24.3] |  |
| vixcls | ew_next | +5.8 | [-33.3, +44.9] |  |
| dgs10 | ew_next | +5.3 | [-65.5, +76.1] |  |
| dtwexbgs | btc_next | +5.0 | [-11.0, +21.0] |  |
| cex_dex_flow | ew_next | -2.5 | [-31.0, +26.0] |  |

## Heterogeneity (CausalForestDML): `staking_yield` on next-day EW return

| Moderator | effect @ 20th pct (CI) | effect @ 80th pct (CI) |
|---|---|---|
| vixcls | -1454.1 bps [-7570.7, +4662.6] | +6563.4 bps [-7775.3, +20902.2] |
| funding_basis | +8745.2 bps [+1371.1, +16119.3] | +644.8 bps [-7176.6, +8466.2] |

## Verdict

4/28 effects clear plain 95% CIs: chain_congestion→btc_next (+40.9 bps), chain_congestion→ew_next (+45.3 bps), t10y2y→btc_next (+61.6 bps), cpiaucsl→ew_next (-627.5 bps). At 28 tests, ~1.4 false positives are expected; none survives Bonferroni unless its CI is far from zero. Check the heterogeneity table before reading anything into these.