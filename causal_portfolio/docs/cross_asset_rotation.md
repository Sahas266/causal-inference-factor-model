# Simple cross-asset rotation strategies (state → weights → return)

Each strategy maps a *cross-section* of trailing momentum / vol across the majors to a long-only weight vector over {majors, stables}. Stables earn 0%. Decision at t close, weights applied to t+1 return, 5bp one-way switching cost on every weight change. Sorted by Sharpe.

- Universe: btc, eth, sol, bnb, avax, xrp, doge, link, uni, aave, crv
- Window `2021-01-01` → `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T15:49:45+00:00`

| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | avg turnover | %in mkt | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relmom_top3_30 | +96% | 1.08 | 1.15 | -82% | 1.17 | 0.254 | 98% | 0.16 |
| ew_basket ⟵ 1/N | +85% | 1.05 | 1.03 | -80% | 1.07 | 0.001 | 100% | 1.00 |
| inv_vol_20 | +72% | 0.93 | 0.92 | -79% | 0.91 | 0.033 | 99% | 0.99 |
| relmom_top3_90 | +71% | 0.85 | 0.86 | -87% | 0.81 | 0.132 | 95% | 0.61 |
| best_of_two_30 | +52% | 0.78 | 0.81 | -70% | 0.73 | 0.156 | 98% | 0.24 |
| bh_btc ⟵ BH BTC | +37% | 0.63 | 0.66 | -77% | 0.48 | 0.001 | 100% | — |
| relmom_top1_90 | +64% | 0.56 | 0.66 | -95% | 0.68 | 0.216 | 95% | 0.69 |
| relmom_top1_30 | +66% | 0.54 | 0.62 | -95% | 0.70 | 0.375 | 98% | 0.73 |
| ts_mom_basket_90 | +16% | 0.34 | 0.30 | -73% | 0.22 | 0.046 | 80% | — |
| dual_mom_90 | +30% | 0.28 | 0.31 | -96% | 0.31 | 0.192 | 80% | — |

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, Sortino 0.66, maxDD -77%, Calmar 0.48.
1/N equal-weight basket: ann +85%, Sharpe 1.05, maxDD -80%, Calmar 1.07.

**No strategy both beats BH BTC on Sharpe AND survives the placebo (p<0.10).**

5 strategies do beat BH BTC on Sharpe — `relmom_top3_30` (Sharpe 1.08, p=0.16), `ew_basket` (Sharpe 1.05, p=1.00), `inv_vol_20` (Sharpe 0.93, p=0.99), `relmom_top3_90` (Sharpe 0.85, p=0.61), `best_of_two_30` (Sharpe 0.78, p=0.24) — but every one has a high placebo p, meaning random-timed copies of the same weights reproduce the Sharpe just as often. The lift over BH BTC comes from holding a *diversified basket of majors* (the 1/N basket alone clears BH BTC: Sharpe 1.05 vs 0.63), not from skillful rotation timing. The best rotation, `relmom_top3_30`, edges the basket on Sharpe (1.08 vs 1.05) but its placebo p (~0.16) is not low enough to call the timing real, and its turnover is ~250x the basket's.

## Caveats

- **Costs / turnover** dominate the high-churn rotations (top-1 momentum flips often); the 5bp/side fee is optimistic for the smaller majors, which have wider spreads and thinner books.
- **Single market cycle.** 2021–2025 is one bull→bear→recovery regime. Cross-sectional momentum looks very different across cycles; one path is not evidence of a durable edge.
- **Multiple testing.** Nine rules over a handful of lookbacks were tried; the best in-sample Sharpe is upward-biased. The placebo guards against *timing* luck but not against having picked the lucky lookback.
- Stables are modeled at 0% return (no yield, no slippage to/from stables beyond the switching cost).