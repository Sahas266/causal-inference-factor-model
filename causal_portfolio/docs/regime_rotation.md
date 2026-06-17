# Simple regime-rotation rules (state → risk-on/off → return)

Each rule rotates BTC↔stables (0% when out) on a single causal state variable; decision at t close, applied to t+1 return, 5bp one-way switching cost. Sorted by Calmar (return / max drawdown) — the metric a risk overlay should win on.

- Asset: BTC | window `2021-01-01` → `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T15:40:39+00:00`

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trend_100 | +25% | 0.66 | 0.53 | -37% | 0.67 | 53% | 72 | 0.15 |
| trend_50 | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | 0.03 |
| always_in ⟵ BH | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 1 | — |
| funding_off | +17% | 0.59 | 0.41 | -35% | 0.48 | 39% | 85 | — |
| vix_off_70 | +17% | 0.48 | 0.37 | -41% | 0.42 | 42% | 129 | — |
| trend_200 | +16% | 0.42 | 0.34 | -50% | 0.32 | 55% | 52 | — |
| vol_target | +20% | 0.39 | 0.39 | -75% | 0.27 | 96% | 411 | — |
| dd_off_30 | +14% | 0.35 | 0.27 | -55% | 0.26 | 52% | 37 | — |
| dd_off_20 | +11% | 0.31 | 0.22 | -44% | 0.26 | 41% | 46 | — |
| vol_off_70 | +13% | 0.35 | 0.26 | -61% | 0.22 | 42% | 95 | — |
| vol_off_80 | +8% | 0.24 | 0.16 | -60% | 0.14 | 32% | 85 | — |
| trend200_and_vix | +4% | 0.17 | 0.10 | -37% | 0.11 | 26% | 96 | — |
| trend200_and_vol | -12% | -0.54 | -0.28 | -66% | -0.18 | 21% | 76 | — |

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48.

Rules that beat BH on Sharpe AND whose timing survives the placebo (p<0.10):
- **trend_50** — Sharpe 0.88 vs 0.63, maxDD -57% vs -77% (placebo p=0.03)

*Note: trend filters typically improve Calmar by truncating bear markets; check the placebo column to see whether that is timing skill or just reduced exposure.*
## Trend-filter robustness (MA-length sweep, placebo each)

The headline rule (trend_50) picked its lookback from {50,100,200}, so a
single winner could be luck. A finer sweep shows it is **not** a one-lookback
spike — a contiguous band clears the placebo:

| MA days | Sharpe | Calmar | maxDD | %in mkt | placebo p |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.28 | 0.16 | -65% | 50% | 0.61 |
| 30 | 0.59 | 0.43 | -52% | 50% | 0.19 |
| **40** | **1.09** | 0.83 | -49% | 50% | **0.01** |
| **50** | 0.88 | 0.60 | -57% | 51% | **0.03** |
| **60** | 0.82 | 0.52 | -60% | 52% | **0.04** |
| 75 | 0.67 | 0.46 | -57% | 53% | 0.14 |
| 100 | 0.66 | 0.67 | -37% | 53% | 0.15 |
| **125** | 0.93 | **1.04** | -33% | 54% | **0.06** |
| 150 | 0.58 | 0.49 | -44% | 54% | 0.30 |
| 200 | 0.42 | 0.32 | -50% | 55% | 0.46 |

(BH BTC: Sharpe 0.63, Calmar 0.48, maxDD -77%.)

Two placebo-robust regions: a **fast band (40–60d)** that maximizes Sharpe
(~0.8–1.1 vs BH 0.63) by sidestepping sharp selloffs, and a **slow point
(~125d)** that maximizes Calmar (1.04) by truncating whole bear markets to
−33% drawdown. The endpoints (20d whipsaw, 200d lag) correctly fail. A real
trend effect should look like a band, and it does.

## What this means

- **The "rotate to stables when volatile" idea underperforms.** vol_off /
  vol_target / dd_off all cut return more than risk (Calmar below BH); their
  drawdown relief is just lower average exposure, not skillful timing.
- **The trend filter is the one simple rule that works on BTC** here: price
  above its ~40–60d (Sharpe) or ~125d (drawdown) moving average → hold,
  else → stables. It roughly matches BH return while cutting max drawdown
  from −77% to the −33% to −57% range and lifting Sharpe.
- **funding_off** is a respectable honorable mention (Sharpe 0.59, maxDD
  −35%, only 39% time in market) and is economically distinct (positioning,
  not price), worth combining with trend rather than the vol rules (the
  trend+vol AND-combo was the worst rule of all — over-de-risking).
- **Caveat:** trend-following BTC is the single most data-mined strategy in
  crypto, and 2021–2025 is ONE cycle (one real bear, 2022). The placebo
  validates timing WITHIN this sample; it cannot validate against regime
  change. Treat as a risk overlay to paper-trade forward, not a guaranteed
  edge — same discipline as the chain_congestion candidate.
