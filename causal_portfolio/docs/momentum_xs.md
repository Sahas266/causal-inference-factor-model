# Cross-sectional momentum / reversal (per-asset drivers)

Daily rank L/S on beta-hedged residual returns over a 20-asset universe (NaN-tolerant; late listings enter when listed). No transaction costs — Sharpes are upper bounds for a daily-churn strategy.

- Universe: `btc,eth,sol,bnb,avax,uni,aave,link,doge,crv,pendle,xrp,shib,zec,tao,pepe,aero,jup,ena,hype`
- Window: `2022-01-01` → `2025-12-31` | β window 90d
- Run UTC: `2026-06-11T00:06:41+00:00`

| Signal | Sharpe | NW t | total | placebo p | avg breadth | fold Sharpes |
|---|---:|---:|---:|---:|---:|---|
| rev1 | 0.474 | 0.86 | +40.4% | 0.46 | 16.0 | -1.25, -0.54, 2.30, 1.09 |
| mom7 | 1.960 | 3.45 | +546.1% | 0.00 | 15.9 | 0.44, 1.58, 3.28, 2.23 |
| mom30 | 1.587 | 2.77 | +331.1% | 0.00 | 15.9 | 0.80, 0.49, 3.59, 1.03 |
| mom30_7 | 0.938 | 1.82 | +110.6% | 0.06 | 15.9 | 0.94, -0.12, 2.38, 0.53 |
| combo | -0.084 | -0.15 | -18.5% | 0.89 | 15.9 | 1.15, -0.07, -0.60, -0.80 |

## Verdict

Signals clearing both NW |t|>2 and placebo p<0.05: **mom7** (Sharpe 1.96, p=0.00), **mom30** (Sharpe 1.59, p=0.00). Before believing: subtract realistic costs (daily rank churn ≈ 50–150% turnover/day) and re-test net.
## Transaction-cost reality check (post-hoc, same loop with turnover tracking)

| Signal | gross Sharpe | avg daily turnover | net @10bps | net @20bps | net @30bps |
|---|---:|---:|---:|---:|---:|
| mom7 | 1.96 | 0.51x gross | 1.29 (+224%) | 0.61 (+63%) | -0.06 (-18%) |
| mom30 | 1.59 | 0.25x gross | 1.25 (+207%) | 0.91 (+119%) | 0.58 (+56%) |

Note the beta-hedge legs roughly double traded notional (each residual
position implies a BTC hedge), so effective per-unit-turnover cost is
~1.5–2× the headline fee. On Hyperliquid majors (taker ~3–5bps + slippage),
**mom30 plausibly survives real costs; mom7 is marginal.** Both are positive
in all four walk-forward folds and pass the circular placebo at 0/100 —
the only signals in this project to do so.
