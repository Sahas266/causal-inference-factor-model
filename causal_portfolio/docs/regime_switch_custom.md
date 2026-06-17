# Custom-regime DAG switching (regime → per-regime rule → return)

Each strategy assigns a **different** exposure rule per regime (menu: hold / trend_50 / vol_target / flat / mean_revert / momentum). Two assignment methods: a **pre-registered** economic mapping fixed before results, and an **adaptive** causal selection (at each rebalance, pick the menu rule with best trailing-window Sharpe in the current regime, past data only). Exposure applied to BTC and the equal-weight majors basket; decision at t close → t+1 return, 5bp one-way cost.

- Universe: `btc, eth, sol, bnb, avax, xrp, doge, link, uni, aave, crv`
- Window `2021-01-01` → `2025-12-31` (1825 days)
- Drawdown regime threshold: 10%
- Run UTC: `2026-06-17T18:02:06+00:00`

## Benchmarks

| Benchmark | Ann.ret | Sharpe | Sortino | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|
| BH BTC | +37% | 0.63 | 0.66 | -77% | 0.48 |
| trend_50 on BTC | +34% | 0.88 | 0.72 | -57% | 0.60 |
| BH basket | +85% | 1.05 | 1.03 | -80% | 1.07 |
| trend_50 on basket | +67% | 1.22 | 0.93 | -47% | 1.43 |

## Regime dwell / switches (full-sample label distribution)

| Regime def | States (share) | Switches |
|---|---|---:|
| drawdown | at_highs 26%, in_drawdown 74% | 69 |
| trending | ranging 52%, trending 48% | 283 |
| corr | low_corr 66%, high_corr 34% | 92 |
| breadth | low_breadth 57%, high_breadth 43% | 116 |

## Results — switching strategies

Sorted within asset by Sharpe. `placebo p` = regime-shuffle (circular-shift the regime labels, re-run under the same mapping); high p ⇒ regime *timing* carries no edge. `vs trend_50` compares Sharpe to the unconditional trend_50 benchmark on the same asset.

| Strategy | Method | Asset | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in | switches | placebo p | vs trend_50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| adaptive:drawdown | adaptive | BTC | +27% | 0.70 | 0.54 | -39% | 0.68 | 43% | 140 | 0.23 | below (-0.18) |
| prereg:trending | pre-registered | BTC | +16% | 0.62 | 0.41 | -49% | 0.34 | 26% | 152 | — | below (-0.26) |
| prereg:corr_vt | pre-registered | BTC | +35% | 0.61 | 0.64 | -75% | 0.46 | 100% | 222 | — | below (-0.27) |
| prereg:corr | pre-registered | BTC | +29% | 0.61 | 0.55 | -63% | 0.46 | 66% | 93 | — | below (-0.27) |
| adaptive:breadth | adaptive | BTC | +21% | 0.55 | 0.43 | -64% | 0.32 | 42% | 206 | — | below (-0.33) |
| prereg:trending_alt | pre-registered | BTC | +14% | 0.55 | 0.35 | -41% | 0.35 | 26% | 134 | — | below (-0.33) |
| adaptive:corr | adaptive | BTC | +19% | 0.44 | 0.33 | -62% | 0.30 | 46% | 133 | — | below (-0.45) |
| prereg:breadth_mr | pre-registered | BTC | +15% | 0.39 | 0.29 | -55% | 0.27 | 47% | 154 | — | below (-0.49) |
| adaptive:trending | adaptive | BTC | +16% | 0.34 | 0.30 | -74% | 0.21 | 60% | 194 | — | below (-0.54) |
| prereg:breadth | pre-registered | BTC | +12% | 0.33 | 0.24 | -52% | 0.23 | 43% | 116 | — | below (-0.55) |
| prereg:drawdown_mr | pre-registered | BTC | +9% | 0.29 | 0.19 | -33% | 0.27 | 29% | 112 | — | below (-0.59) |
| prereg:drawdown | pre-registered | BTC | +6% | 0.24 | 0.15 | -33% | 0.20 | 26% | 70 | — | below (-0.64) |
| prereg:corr | pre-registered | basket | +76% | 1.18 | 1.01 | -64% | 1.18 | 66% | 93 | 0.18 | below (-0.04) |
| prereg:corr_vt | pre-registered | basket | +82% | 1.07 | 1.05 | -78% | 1.05 | 100% | 248 | 0.05 | below (-0.15) |
| adaptive:breadth | adaptive | basket | +70% | 1.02 | 0.82 | -77% | 0.90 | 61% | 259 | — | below (-0.20) |
| adaptive:trending | adaptive | basket | +63% | 0.99 | 0.79 | -75% | 0.84 | 53% | 202 | — | below (-0.22) |
| prereg:breadth | pre-registered | basket | +47% | 0.94 | 0.63 | -48% | 0.99 | 43% | 116 | — | below (-0.28) |
| prereg:trending | pre-registered | basket | +37% | 0.93 | 0.52 | -44% | 0.85 | 26% | 190 | — | below (-0.29) |
| prereg:trending_alt | pre-registered | basket | +34% | 0.85 | 0.47 | -45% | 0.77 | 26% | 168 | — | below (-0.36) |
| prereg:drawdown | pre-registered | basket | +34% | 0.85 | 0.47 | -54% | 0.63 | 26% | 70 | — | below (-0.37) |
| adaptive:corr | adaptive | basket | +54% | 0.81 | 0.64 | -72% | 0.75 | 55% | 147 | — | below (-0.40) |
| prereg:breadth_mr | pre-registered | basket | +43% | 0.80 | 0.55 | -53% | 0.80 | 45% | 154 | — | below (-0.42) |
| adaptive:drawdown | adaptive | basket | +47% | 0.70 | 0.52 | -81% | 0.59 | 55% | 108 | — | below (-0.52) |
| prereg:drawdown_mr | pre-registered | basket | +29% | 0.67 | 0.38 | -58% | 0.51 | 28% | 104 | — | below (-0.55) |

## Verdict

- **BH BTC**: Sharpe 0.63, maxDD -77%. **trend_50 (BTC)**: Sharpe 0.88. **BH basket**: Sharpe 1.05. **trend_50 (basket)**: Sharpe 1.22.

**No custom-regime switching strategy both beats trend_50 on Sharpe (same asset) and survives the regime-shuffle placebo.** 0 strategy/asset combo(s) edge out trend_50 on Sharpe, but either fail the shuffle placebo (their edge is reproducible with scrambled regime labels ⇒ it comes from the menu rules / average exposure, not regime *timing*) or do not clear it cleanly. The regime label adds churn, not skill.

## Caveats

- **One cycle.** 2021–2025 is a single bull→bear→recovery; any regime gate that helps does so largely by truncating the *one* 2022 bear. The placebo validates timing *within* this sample, not across regime change.
- **Mechanically endogenous regimes.** Drawdown, efficiency ratio, correlation and breadth are all derived from the *same* prices being traded, so part of any edge is tautological (a price below its peak both defines 'drawdown' and is a recent loss).
- **Multiple testing.** 8 pre-registered mappings × 2 assets + 4 adaptive regimes × 2 assets were tried; the best in-sample numbers are upward-biased. The shuffle placebo guards against regime-timing luck on a *fixed* mapping, not against having picked the lucky mapping/regime.
- **Funding regime omitted from the headline.** Funding data only starts 2023-11 (~2 yr), too short for a 252-day rolling-quantile regime over the full window; run `--with-funding` to include it on the 2023-11+ subsample, but treat it as exploratory.
- **Stables = 0%.** Risk-off legs earn 0% (no stable yield modeled); a real cash yield would only lift the flat-rule regimes slightly and does not change rankings vs the benchmarks.
- **No leverage.** Exposure capped at 1, so every overlay can only de-risk — structurally disadvantaged on raw return in a bull market.