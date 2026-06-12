# Delta-neutral funding carry (Hyperliquid)

Short perp + long spot, collecting hourly funding. Income-space backtest: hedged price PnL assumed zero (basis drift and execution tracking are real second-order risks, not modeled). Costs: 15bp one-way on combined-leg turnover. Decisions at t close earn day t+1 funding.

- Universe: 20 HL perps | window 2023-11-01 → 2025-12-31 | top-K=5
- Run UTC: `2026-06-12T22:53:02+00:00`

| Arm | Ann. return | Sharpe | Max DD | avg gross | ann. cost drag | fold Sharpes |
|---|---:|---:|---:|---:|---:|---|
| always_on | +18.8% | 12.74 | -0.2% | 1.00 | 0.14% | 18.6, 16.5, 13.2, 15.7 |
| persist_top5 | +19.9% | 7.96 | -1.0% | 0.99 | 11.32% | 12.3, 11.2, 7.8, 5.3 |
| forecast_top5 | -10.0% | -8.54 | -19.6% | 0.67 | 21.86% | 0.0, -4.3, -13.2, -13.8 |
| forecast_top5_nofactors | -4.9% | -5.05 | -10.8% | 0.67 | 17.02% | 0.0, -1.2, -8.0, -9.3 |

**Factor-timing placebo** (circular shifts of the innovation block; share of shifts whose forecast arm Sharpe ≥ real): p = 0.86

## Notes / risks not in the income backtest

- Spot leg assumed available at perp size (HL spot or external venue); borrow/withdrawal frictions ignored.
- Basis risk: spot-perp spread moves; severe in squeezes.
- Funding can flip intraday; the daily model reacts next day.
- Capacity: fine at personal size on majors; thin on the tail assets.
## Churn-controlled variants (weekly rebalance, smoothed forecast)

| Arm | Ann. return | Sharpe | Max DD | ann. cost drag |
|---|---:|---:|---:|---:|
| persist_top5_weekly | **+22.2%** | 11.9 | -0.4% | 6.0% |
| always_on | +18.8% | 12.7 | -0.2% | 0.1% |
| forecast_top5_weekly (smoothed) | +7.6% | 7.5 | -0.4% | 3.5% |

## Verdict

**The carry is the product; the forecast is not.** A delta-neutral book
collecting Hyperliquid funding earned ~19–22%/yr over 2023-11 → 2025-12 at
conservative (30bp round-trip) costs, with the best costed implementation
being the simplest: top-5 by trailing 7d funding, rebalanced weekly. The
factor-augmented funding forecast — statistically real in target_search.md
(+0.023 OOS R², placebo p=0.02) — is economically immaterial here: daily
forecast reranking LOSES 10%/yr to turnover, and even smoothed/weekly it
selects worse than plain persistence (placebo on the factor increment
p=0.86). A +2% R² edge cannot pay 30bp round trips.

Caveat on the Sharpe numbers: income-space Sharpe (8–13) overstates reality
— the true risks (spot-perp basis moves, funding flipping intraday, spot-leg
funding/borrow, venue risk) live OUTSIDE the income series. Treat this as
"~20%/yr gross carry with low income volatility and unmodeled tail risks,"
not a Sharpe-12 strategy. Next step for deployment: paper/testnet the
persist_top5_weekly book through the execution layer with a real spot leg.
