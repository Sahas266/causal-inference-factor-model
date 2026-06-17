# Market-breadth & correlation-regime DAGs (state → exposure → return)

Each rule turns a single cross-sectional market state (breadth, avg pairwise correlation, cross-sectional dispersion, BTC relative strength) into a 0/1 exposure, applied to **both** BTC and the equal-weight majors **basket**. Decision at day t close, applied to t+1 return, 5bp one-way switching cost. Sorted within each asset by Calmar.

- Universe: `btc, eth, sol, bnb, avax, xrp, doge, link, uni, aave, crv`
- Window `2021-01-01` → `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T16:01:53+00:00`

| Rule | Asset | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | placebo p (Sharpe) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| btcdom_alts_on_lead | BTC | +31% | 0.78 | -52% | 0.60 | 51% | 153 | 0.13 |
| breadth_mom_rising | BTC | +25% | 0.79 | -46% | 0.54 | 40% | 286 | 0.06 |
| disp_off_low | BTC | +19% | 0.75 | -35% | 0.54 | 22% | 94 | 0.12 |
| corr_off_high | BTC | +32% | 0.68 | -64% | 0.49 | 66% | 93 | 0.26 |
| always_in ⟵ BH | BTC | +37% | 0.63 | -77% | 0.48 | 100% | 1 | — |
| breadth_ma50_gt50 | BTC | +14% | 0.41 | -44% | 0.33 | 43% | 116 | — |
| disp_off_high | BTC | +16% | 0.31 | -76% | 0.22 | 78% | 95 | — |
| breadth_ma50_gtmed | BTC | +10% | 0.33 | -56% | 0.19 | 40% | 140 | — |
| breadth_ma100_gt50 | BTC | +6% | 0.18 | -71% | 0.09 | 43% | 72 | — |
| corr_off_low | BTC | +3% | 0.10 | -59% | 0.06 | 34% | 92 | — |
| btcdom_alts_on_lag | BTC | +3% | 0.06 | -77% | 0.03 | 49% | 154 | — |
| corr_off_high | basket | +92% | 1.47 | -56% | 1.64 | 66% | 93 | 0.04 |
| breadth_ma50_gt50 | basket | +53% | 1.08 | -39% | 1.35 | 43% | 116 | 0.14 |
| always_in ⟵ BH | basket | +85% | 1.05 | -80% | 1.07 | 100% | 1 | — |
| disp_off_low | basket | +37% | 1.01 | -41% | 0.88 | 22% | 94 | — |
| breadth_mom_rising | basket | +40% | 1.02 | -46% | 0.87 | 40% | 286 | — |
| breadth_ma50_gtmed | basket | +37% | 0.85 | -57% | 0.65 | 40% | 140 | — |
| btcdom_alts_on_lag | basket | +49% | 0.81 | -78% | 0.63 | 49% | 154 | — |
| disp_off_high | basket | +47% | 0.64 | -86% | 0.55 | 78% | 95 | — |
| btcdom_alts_on_lead | basket | +33% | 0.62 | -64% | 0.52 | 51% | 153 | — |
| breadth_ma100_gt50 | basket | +21% | 0.43 | -71% | 0.30 | 43% | 72 | — |
| corr_off_low | basket | -9% | -0.17 | -89% | -0.10 | 34% | 92 | — |

## Verdict

**Buy-and-hold BTC**: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48.
**Buy-and-hold basket**: ann +85%, Sharpe 1.05, maxDD -80%, Calmar 1.07.

Rules that beat their BH benchmark on Sharpe AND whose timing survives the placebo (p<0.10):
- **corr_off_high** on basket — Sharpe 1.47 vs 1.05, maxDD -56% vs -80% (placebo p=0.04)
- **breadth_mom_rising** on BTC — Sharpe 0.79 vs 0.63, maxDD -46% vs -77% (placebo p=0.06)

## Caveats

- **One cycle.** This is a single 2021–2025 crypto regime (2021 mania → 2022 bear → 2023–25 recovery). Breadth and correlation gates are exactly the kind of rule that overfits to one drawdown.
- **Mechanically endogenous.** Breadth, correlation and dispersion are all derived from the *same* prices being traded, so part of any apparent edge is mechanical (e.g. a price below its MA both lowers breadth and is, tautologically, a recent loss).
- **Two-sided sign tests inflate significance.** We report both signs of the correlation, dispersion and dominance gates. Testing a hypothesis and its negation roughly doubles the chance one looks good by luck — treat any single survivor with suspicion and discount the placebo threshold accordingly.
- **Stables = 0%.** Risk-off earns 0% (no stable yield modeled); a real cash/stable yield would only improve the risk-off legs, so these gate results are conservative on that axis.
- **Read the survivors skeptically.** Any rule that clears the placebo here is one survivor out of ~20 rule/asset combos under two-sided sign testing on a single cycle; the placebo only rules out trivial average-exposure effects, not overfitting, so the honest read is "suggestive, not validated — would need a second regime / market to trust."