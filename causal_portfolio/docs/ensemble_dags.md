# Ensemble / combination allocation DAGs (state -> allocation -> return)

Each rule combines signals that each looked useful on their own and asks whether the COMBINATION beats the best SINGLE signal. Allocations are causal (trailing windows only); the decision at t close is applied to t+1 return with a 5bp one-way switching cost; stables earn 0%. Sorted by Sharpe.

- Universe: btc, eth, sol, bnb, avax, xrp, doge, link, uni, aave, crv
- Window `2021-01-01` -> `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T15:59:45+00:00`

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | turnover | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| breadth_trend_basket | +58% | 1.27 | 1.22 | -47% | 1.24 | 80% | 0.062 t/o | 0.09 |
| trend_on_basket | +67% | 1.22 | 0.93 | -47% | 1.43 | 50% | 0.059 t/o | 0.09 |
| btctrend_on_basket | +61% | 1.18 | 0.89 | -54% | 1.12 | 51% | 0.057 t/o | 0.10 |
| ew_basket ⟵ 1/N | +85% | 1.05 | 1.03 | -80% | 1.07 | 100% | 0.001 t/o | — |
| sig_trend50 | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 sw | 0.03 |
| trend_AVG_funding_full | +25% | 0.88 | 0.81 | -33% | 0.76 | 65% | 181 sw | 0.08 |
| trend_AND_funding_full | +19% | 0.81 | 0.48 | -26% | 0.71 | 25% | 60 sw | 0.14 |
| dual_confirm_basket | +33% | 0.69 | 0.44 | -56% | 0.58 | 40% | 0.062 t/o | 0.41 |
| bh_btc ⟵ BH BTC | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 0.001 t/o | — |
| vote_frac | +23% | 0.60 | 0.60 | -61% | 0.37 | 87% | 266 sw | — |
| sig_vix_calm | +13% | 0.28 | 0.25 | -71% | 0.18 | 70% | 103 sw | — |

### Strategy 5 — fair funding-window comparison (2023-11 onward, 792 days)

Funding only exists from 2023-11, so the full-sample `trend_*_funding` rows above pad a long NaN-funding stretch. Here every rule is judged over exactly the funding window, against trend-only and BH BTC on the SAME days.

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fw_trend50_only | +41% | 1.19 | 1.06 | -26% | 1.57 | 57% | 50 sw | 0.18 |
| fw_trend_AND_funding | +40% | 1.16 | 1.02 | -26% | 1.52 | 55% | 60 sw | 0.20 |
| fw_bh_btc ⟵ BH BTC (sub) | +53% | 1.11 | 1.20 | -32% | 1.64 | 100% | 1 sw | — |
| fw_trend_AVG_funding | +40% | 1.09 | 1.19 | -30% | 1.36 | 91% | 127 sw | — |
| fw_funding_only | +39% | 0.90 | 0.94 | -35% | 1.10 | 89% | 85 sw | — |

## Verdict

Benchmarks — buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48. 1/N basket: ann +85%, Sharpe 1.05, maxDD -80%, Calmar 1.07.

Combinations that beat the best single benchmark on Sharpe AND whose timing survives the placebo (p<0.10):
- **breadth_trend_basket** — Sharpe 1.27 vs best-single 1.05, maxDD -47% (placebo p=0.09)
- **trend_on_basket** — Sharpe 1.22 vs best-single 1.05, maxDD -47% (placebo p=0.09)

## Caveats

- **One market cycle.** 2021–2025 is a single bull→bear→recovery path. Trend and breadth filters look heroic when they truncate exactly one bear; that is not evidence of a repeatable edge.
- **Combinations multiply researcher DoF.** Each combo bakes in choices (which signals, which windows, AND vs average vs vote). The placebo guards against *timing* luck on a fixed rule, not against having picked the lucky combination among many.
- **Funding is short** (2023-11+) and the full-sample trend+funding rows pad a NaN stretch; trust the funding-window sub-table for that comparison.
- Stables modeled at 0% (no yield, no slippage beyond the switching cost); the 5bp/side fee is optimistic for the smaller majors.