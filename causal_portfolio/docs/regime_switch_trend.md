# Trend/MA regime switching — a DAG per regime

A moving-average **trend regime** (bull = price > N-day MA, bear below; or a 3-state bull/neutral/bear band) selects a different base allocation rule per regime. Each map is one causal DAG: `regime_t -> rule -> exposure_t -> return_{t+1}`, decided at close t, applied to t+1 with a 5bp one-way switching cost.

- Asset: BTC | window `2021-01-01` -> `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T20:32:10+00:00`
- **The bar is `trend_50`, not BH BTC** — the trend filter is *itself* a bull/bear regime switch, so a regime-conditional DAG must beat it AND survive the regime-shuffle placebo.

Benchmarks: BH BTC — ann +37%, Sharpe 0.63, Sortino 0.66, maxDD -77%, Calmar 0.48.  trend_50 — ann +34%, Sharpe 0.88, Sortino 0.72, maxDD -57%, Calmar 0.60.

| Strategy | Regime def | Ann | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | shuffle p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ma50_3s::bull_hold__neu_flat__bear_flat` | 3-state MA50 (+/-5% band) | +36% | 1.08 | 0.81 | -34% | 1.05 | 36% | 84 | 0.01 |
| `ma50_2s::bull_hold__bear_meanrev` | 2-state MA50 (bull=px>MA) | +42% | 1.01 | 0.86 | -55% | 0.76 | 55% | 146 | 0.03 |
| `ma50_2s::bull_hold__bear_flat` | 2-state MA50 (bull=px>MA) | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | — |
| `ma50_2s::bull_trend50__bear_flat` | 2-state MA50 (bull=px>MA) | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | — |
| `ma50_3s::bull_hold__neu_trend50__bear_flat` | 3-state MA50 (+/-5% band) | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | — |
| `ma100_2s::bull_hold__bear_meanrev` | 2-state MA100 (bull=px>MA) | +33% | 0.81 | 0.67 | -38% | 0.87 | 57% | 104 | 0.12 |
| `ma100_3s::bull_hold__neu_voltgt__bear_meanrev` | 3-state MA100 (+/-5% band) | +34% | 0.79 | 0.70 | -51% | 0.67 | 67% | 165 | 0.11 |
| `ma100_3s::bull_hold__neu_trend50__bear_flat` | 3-state MA100 (+/-5% band) | +29% | 0.78 | 0.62 | -45% | 0.65 | 53% | 76 | 0.14 |
| `ma100_2s::bull_trend50__bear_flat` | 2-state MA100 (bull=px>MA) | +26% | 0.77 | 0.57 | -38% | 0.70 | 44% | 80 | 0.28 |
| `ma200_3s::bull_hold__neu_voltgt__bear_meanrev` | 3-state MA200 (+/-5% band) | +31% | 0.75 | 0.66 | -47% | 0.66 | 66% | 118 | 0.18 |
| `ma50_2s::ADAPTIVE` | 2s MA50 adaptive (trailing-Sharpe pick) | +27% | 0.71 | 0.58 | -57% | 0.48 | 53% | 191 | — |
| `ma100_3s::bull_hold__neu_flat__bear_flat` | 3-state MA100 (+/-5% band) | +24% | 0.70 | 0.51 | -38% | 0.64 | 43% | 68 | 0.10 |
| `ma100_2s::bull_hold__bear_flat` | 2-state MA100 (bull=px>MA) | +25% | 0.66 | 0.53 | -37% | 0.67 | 53% | 72 | 0.15 |
| `ma100_2s::ADAPTIVE` | 2s MA100 adaptive (trailing-Sharpe pick) | +23% | 0.65 | 0.52 | -40% | 0.58 | 52% | 160 | — |
| `ma200_2s::bull_trend50__bear_flat` | 2-state MA200 (bull=px>MA) | +20% | 0.65 | 0.46 | -49% | 0.41 | 39% | 80 | — |
| `ma100_3s::ADAPTIVE` | 3s MA100 adaptive (trailing-Sharpe pick) | +23% | 0.65 | 0.52 | -49% | 0.47 | 55% | 145 | — |
| `ma200_3s::ADAPTIVE` | 3s MA200 adaptive (trailing-Sharpe pick) | +22% | 0.64 | 0.50 | -51% | 0.43 | 51% | 123 | — |
| `ma50_3s::bull_hold__neu_voltgt__bear_meanrev` | 3-state MA50 (+/-5% band) | +28% | 0.63 | 0.61 | -59% | 0.48 | 73% | 267 | — |
| `ma200_2s::ADAPTIVE` | 2s MA200 adaptive (trailing-Sharpe pick) | +21% | 0.63 | 0.49 | -43% | 0.49 | 50% | 128 | — |
| `ma50_2s::bull_voltgt__bear_flat` | 2-state MA50 (bull=px>MA) | +20% | 0.61 | 0.47 | -54% | 0.37 | 48% | 257 | — |
| `ma100_2s::bull_voltgt__bear_flat` | 2-state MA100 (bull=px>MA) | +20% | 0.58 | 0.46 | -37% | 0.53 | 52% | 245 | — |
| `ma200_3s::bull_hold__neu_trend50__bear_flat` | 3-state MA200 (+/-5% band) | +21% | 0.57 | 0.46 | -44% | 0.48 | 55% | 58 | — |
| `ma200_2s::bull_hold__bear_meanrev` | 2-state MA200 (bull=px>MA) | +21% | 0.51 | 0.43 | -58% | 0.35 | 58% | 80 | — |
| `ma50_3s::ADAPTIVE` | 3s MA50 adaptive (trailing-Sharpe pick) | +20% | 0.50 | 0.44 | -61% | 0.33 | 64% | 213 | — |
| `ma200_2s::bull_hold__bear_flat` | 2-state MA200 (bull=px>MA) | +16% | 0.42 | 0.34 | -50% | 0.32 | 55% | 52 | — |
| `ma200_2s::bull_voltgt__bear_flat` | 2-state MA200 (bull=px>MA) | +14% | 0.40 | 0.32 | -50% | 0.28 | 55% | 277 | — |
| `ma200_3s::bull_hold__neu_flat__bear_flat` | 3-state MA200 (+/-5% band) | +14% | 0.40 | 0.30 | -48% | 0.29 | 50% | 50 | — |

## Verdict

**Read the shuffle gate carefully.** `trend_50` *itself* passes a regime-shuffle placebo (own-shuffle p≈0.03): the test only confirms that *trend timing carries information*, which the benchmark already exploits. So shuffle p<0.10 is **necessary but not sufficient** — a mapping can inherit trend_50's timing edge and pass the shuffle without adding anything *over* trend_50. The honest question is whether the *marginal* lift over trend_50 survives, and with one cycle + many maps tried, that marginal lift is suggestive at best.

Strategies that beat `trend_50` on Sharpe AND survive the regime-shuffle placebo (p < 0.10):
- **`ma50_3s::bull_hold__neu_flat__bear_flat`** — Sharpe 1.08 vs 0.88, maxDD -34%, shuffle p=0.01
- **`ma50_2s::bull_hold__bear_meanrev`** — Sharpe 1.01 vs 0.88, maxDD -55%, shuffle p=0.03

But both reduce to trend, not to genuine per-regime switching:
- `ma50_3s::bull_hold__neu_flat__bear_flat` maps *both* neutral and bear to flat, so it collapses to a **2-state on/off trend rule with a +5% confirmation band** — a stricter trend_50, not a different DAG per regime. Its higher Sharpe (1.08) and far lower drawdown (-34%) come from being invested only 36% of the time in the strongest uptrends; that is exposure-timing on the *same* trend signal.
- `ma50_2s::bull_hold__bear_meanrev` is the only true regime-conditional DAG that clears the bar: it dip-buys (RSI<30) *inside* downtrends instead of going flat. Its edge over bull_hold/bear_flat is a handful of oversold-bounce days in the 2022 bear (~64 active bear-days); that is a thin, cycle-specific basis and should be paper-traded forward, not trusted.

**Bottom line:** MA-regime DAG switching does *not* convincingly beat `trend_50`. The two strategies that clear the Sharpe+shuffle gate are either a relabelled stricter trend filter or lean on a few 2022-bear mean-reversion days — neither is a robust, regime-specific edge over the trend benchmark.

## Caveats

- **One market cycle.** 2021-2025 is a single bull->bear->recovery; every MA-regime winner looks good mostly by truncating the one 2022 bear. The placebo validates regime *timing within* this sample, not against regime change.
- **The bar is self-similar.** The benchmark (`trend_50`) is itself a trend regime switch, so a trend-regime DAG that maps bull->long, bear->cash is nearly the same strategy — beating it requires the *per-regime rule choice* to add value beyond the on/off trend call.
- **Pre-registered vs adaptive.** Pre-registered maps were fixed before seeing results (no map-selection leak). The adaptive variant picks rules by trailing per-regime Sharpe on past-only data; it has no look-ahead but does pay extra turnover and estimation noise.
- **The shuffle is a weak gate here.** Because the benchmark `trend_50` already passes a regime-shuffle (own p≈0.03), passing the shuffle proves only that the regime label is informative about trend — not that the mapping adds value *over* trend_50. Treat shuffle p<0.10 as a floor (rules out pure exposure-level luck), not as evidence of a regime-specific edge.
- **Multiple testing.** ~27 regime/map/state variants were run; the best in-sample numbers are upward-biased. The shuffle guards timing luck on a *fixed* mapping, not the luck of having tried many mappings.
- **Costs / stables.** 5bp one-way is optimistic; high-switch maps degrade at 10-20bp. Stables earn 0% here — a real yield lifts every risk-off leg slightly but does not change the ranking vs trend_50.