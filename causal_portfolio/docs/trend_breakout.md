# Simple trend-following & breakout rules on BTC (→ stables)

Each rule holds BTC (exposure 1) or sits in stables (0, =0% return) on a single causal trend/breakout signal; decision at t close, applied to t+1 return, 5bp one-way switching cost. Compared to buy-and-hold BTC (`always_in`) and the incumbent `trend_50` (price > 50d MA) from the regime-rotation study. Sorted by Sharpe.

- Asset: BTC | window `2021-01-01` → `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T15:51:50+00:00`

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dual_confirm | +28% | 0.95 | 0.70 | -26% | 1.10 | 37% | 98 | 0.04 |
| tsmom_30 | +33% | 0.89 | 0.76 | -46% | 0.71 | 51% | 158 | 0.05 |
| trend_50 ⟵ incumbent | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | 0.03 |
| always_in ⟵ BH | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 1 | — |
| tsmom_180 | +23% | 0.63 | 0.52 | -54% | 0.43 | 58% | 44 | — |
| donchian_55 | +23% | 0.62 | 0.47 | -49% | 0.46 | 50% | 22 | — |
| donchian_20 | +24% | 0.61 | 0.49 | -55% | 0.44 | 51% | 60 | — |
| macd_sign | +20% | 0.52 | 0.43 | -52% | 0.39 | 49% | 139 | — |
| ma_cross_20_100 | +17% | 0.46 | 0.35 | -43% | 0.40 | 51% | 22 | — |
| tsmom_90 | +17% | 0.46 | 0.37 | -57% | 0.30 | 53% | 74 | — |
| ma_cross_50_200 | +16% | 0.45 | 0.35 | -45% | 0.36 | 53% | 10 | — |

## Time-series momentum (ROC sign) robustness (lookback sweep, placebo each)

Is the best family a real trend BAND or a single-lookback spike? A finer grid, placebo each:

| N (days) | Sharpe | Calmar | MaxDD | %in mkt | switches | placebo p |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.60 | 0.40 | -57% | 51% | 200 | 0.17 |
| **30** | 0.89 | 0.71 | -46% | 51% | 158 | **0.05** |
| 45 | 0.56 | 0.43 | -49% | 52% | 112 | 0.25 |
| 60 | 0.69 | 0.44 | -60% | 52% | 86 | 0.14 |
| 90 | 0.46 | 0.30 | -57% | 53% | 74 | 0.41 |
| 120 | 0.47 | 0.35 | -50% | 53% | 52 | 0.32 |
| 180 | 0.63 | 0.43 | -54% | 58% | 44 | 0.28 |

(BH BTC: Sharpe 0.63, Calmar 0.48, maxDD -77%. Incumbent trend_50: Sharpe 0.88, Calmar 0.60, maxDD -57%.)

Only N=30 clears the placebo — an isolated spike, which reads as overfit rather than a robust trend band.

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48. Incumbent trend_50: ann +34%, Sharpe 0.88, maxDD -57%, Calmar 0.60.

Rules that beat BOTH BH and the incumbent trend_50 on Sharpe AND survive the placebo (p<0.10):
- **dual_confirm** — Sharpe 0.95 (vs BH 0.63, trend_50 0.88), maxDD -26%, Calmar 1.10 (placebo p=0.04)
- **tsmom_30** — Sharpe 0.89 (vs BH 0.63, trend_50 0.88), maxDD -46%, Calmar 0.71 (placebo p=0.05)

**Read the sweep before trusting these.** The time-series momentum (roc sign) family is NOT a contiguous band — only a single lookback clears the placebo while its neighbours fail, the classic overfit signature. The single-rule placebo passing is therefore weak evidence; `dual_confirm` (price>100d MA AND 30d ROC>0, a 2-of-2 AND filter, not a swept lookback) is the more credible survivor here, and even it only edges the incumbent `trend_50` on Sharpe while winning clearly on drawdown and Calmar.

## Caveats

- **Trend-following on BTC is the single most data-mined strategy in crypto.** Every variant here has been published and backtested thousands of times; surviving an in-sample placebo is a low bar, not proof of a forward edge.
- **2021–2025 is ONE cycle** — one real bear (2022) and a couple of sharp selloffs. The placebo validates that timing carries information *within this sample*; it cannot validate against regime change. A band that looks robust here can still be a single-cycle artifact.
- **Stables earn 0%** in this backtest. A real stable yield would only help the out-of-market periods, so these are conservative; but slippage on the switch days is modelled only as a flat 5bp fee, which understates cost in the fast (high-switch) variants.
- Treat any survivor as a **risk overlay to paper-trade forward**, not a guaranteed edge — same discipline as every other candidate in this repo, where buy-and-hold BTC is the benchmark nothing has reliably beaten out-of-sample.