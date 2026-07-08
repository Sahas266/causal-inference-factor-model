# Weekly/monthly horizon: factor → next-week returns

**Date:** 2026-07-07. **Module:** `experiments/weekly_horizon.py`
(tests: `tests/test_weekly_horizon.py`).

## Why this experiment

The daily-horizon program closed with a structural null — on-chain factors
are downstream of daily returns (`causal_directions_summary.md`) — and the
intraday escape hatch is closed (`funding_intraday.md`). The last open avenue
named in the summary is **longer horizons**: perhaps factor state says
nothing about tomorrow but something about next week or next month. This
experiment tests that, under the house honesty bar
(`dag_strategies_findings.md` §Method).

## Design (pre-registered, deliberately small)

- **Grid:** daily factors from `build_all_factors` (9-asset default
  universe + stablecoin factor inputs, 2021-01-01 → 2025-12-31, local
  DuckDB), resampled to weekly Friday close via `.last()`. The factor value
  at week *t* uses only daily data ≤ *t*; the target is the compounded
  return over (*t*, *t*+1w] — non-overlapping. Two targets: **BTC** and the
  **equal-weight 9-asset basket** (weekly-rebalanced 1/N).
- **Regressions:** for each of the 7 core CPCM factors plus `vixcls` and
  `t10y2y` (none all-NaN): OLS of forward weekly return on the
  (full-sample-affine z-scored) factor, Newey-West HAC (maxlags 4). 9
  factors × 2 targets = **18 regressions, all reported**; Bonferroni
  threshold p < 0.0028.
- **Monthly variant:** identical machinery at month-end (~60 obs —
  reported, flagged underpowered).
- **Overlays** only for weekly hits at p < 0.05 pre-Bonferroni: long/flat
  gate whose direction **and** gate center are fit on the **first half** of
  the weekly grid only, evaluated on the **second half** only
  (2023-07 → 2025-12). 5bp one-way costs, weekly rebalance, vs buy-and-hold
  on the same grid; circular-shift placebo (n=200, annualization 52) when
  the overlay beats BH on Sharpe. One spec per factor, no sign/lookback
  sweeps.
- **Causality contract** (unit-tested): perturbing the last 30 days of daily
  data leaves every grid column at earlier weeks bit-identical.

Grid: 262 weeks (2021-01-01 → 2026-01-02), 60 months. `stable_flow` has
weekly n=100 (supply components' common window) and `funding_basis` n=113
(Hyperliquid funding starts 2023-10).

## Results

### Weekly (18 regressions; Bonferroni p < 0.0028)

| factor | target | n | β (bps/σ) | HAC t | p | p<0.05 | Bonf |
|---|---|--:|--:|--:|--:|:-:|:-:|
| liq_flow | btc | 249 | +19.9 | +0.49 | 0.624 | | |
| liq_flow | basket | 249 | +73.5 | +0.97 | 0.332 | | |
| stable_flow | btc | 100 | +5.7 | +0.09 | 0.925 | | |
| stable_flow | basket | 100 | +74.6 | +0.76 | 0.445 | | |
| funding_basis | btc | 113 | +63.4 | +1.84 | 0.066 | | |
| funding_basis | basket | 113 | +106.3 | +1.78 | 0.075 | | |
| chain_congestion | btc | 260 | +13.0 | +0.14 | 0.886 | | |
| chain_congestion | basket | 260 | +220.5 | +1.59 | 0.112 | | |
| staking_yield | btc | 260 | −62.8 | −6.43 | 0.000 | * | ** |
| staking_yield | basket | 260 | −87.1 | −5.76 | 0.000 | * | ** |
| mev_pressure | btc | 230 | −32.2 | −0.68 | 0.494 | | |
| mev_pressure | basket | 230 | −23.9 | −0.36 | 0.718 | | |
| cex_dex_flow | btc | 260 | +80.2 | +1.99 | 0.047 | * | |
| cex_dex_flow | basket | 260 | +97.9 | +1.60 | 0.111 | | |
| vixcls | btc | 260 | −13.2 | −0.28 | 0.782 | | |
| vixcls | basket | 260 | −14.4 | −0.17 | 0.867 | | |
| t10y2y | btc | 260 | −35.8 | −0.61 | 0.539 | | |
| t10y2y | basket | 260 | +143.8 | +1.29 | 0.196 | | |

3 hits at plain 0.05 (≈0.9 expected by chance from 18 tests, and two of the
three are the same factor on correlated targets). Two survive Bonferroni —
both `staking_yield`.

### Monthly (18 regressions, ~60 obs — underpowered)

| factor | target | n | β (bps/σ) | HAC t | p |
|---|---|--:|--:|--:|--:|
| liq_flow | btc | 57 | +62.4 | +0.26 | 0.792 |
| liq_flow | basket | 57 | +283.1 | +0.84 | 0.403 |
| stable_flow | btc | 22 | −200.6 | −0.87 | 0.384 |
| stable_flow | basket | 22 | −318.7 | −1.17 | 0.241 |
| funding_basis | btc | 25 | −297.6 | −1.94 | 0.053 |
| funding_basis | basket | 25 | −294.3 | −1.00 | 0.319 |
| chain_congestion | btc | 59 | +318.9 | +1.99 | 0.047 |
| chain_congestion | basket | 59 | +1007.9 | +1.98 | 0.048 |
| staking_yield | btc | 59 | +153.2 | +0.99 | 0.323 |
| staking_yield | basket | 59 | −138.2 | −0.54 | 0.588 |
| mev_pressure | btc | 52 | +0.6 | +0.00 | 0.997 |
| mev_pressure | basket | 52 | +10.5 | +0.07 | 0.942 |
| cex_dex_flow | btc | 59 | +467.1 | +2.41 | 0.016 |
| cex_dex_flow | basket | 59 | +583.0 | +1.87 | 0.061 |
| vixcls | btc | 59 | +195.2 | +0.67 | 0.504 |
| vixcls | basket | 59 | +482.7 | +1.02 | 0.309 |
| t10y2y | btc | 59 | −83.5 | −0.44 | 0.662 |
| t10y2y | basket | 59 | +485.1 | +1.25 | 0.210 |

Nothing survives Bonferroni. Note the sign flips versus weekly:
`staking_yield` flips to positive on BTC, `funding_basis` flips to negative
— sign instability across horizons is what noise looks like.

### The Bonferroni survivor is a first-half artifact

Split-stability diagnostic (fit on each half of the weekly grid):

| hit | 1st-half β (t) | 2nd-half β (t) |
|---|--:|--:|
| staking_yield → btc | −88.9 (−6.89) | **+14.5 (+0.25)** |
| staking_yield → basket | −128.2 (−5.56) | **+65.7 (+1.04)** |
| cex_dex_flow → btc | +59.6 (+0.71) | +118.6 (+2.69) |

The full-sample `staking_yield` t = −6.43 lives entirely in 2021-H1-2023
(the bear leg); in the second half the coefficient is zero with the **wrong
sign**. The pre-registered split-sample overlay confirms — it loses to BH
outright on both targets:

| overlay (eval 2023-07 → 2025-12) | ann | Sharpe | maxDD | in-mkt | placebo p |
|---|--:|--:|--:|--:|--:|
| **BH btc** | **+51.8%** | **1.18** | −30.4% | 100% | — |
| staking_yield gate (btc) | +31.0% | 0.78 | −30.4% | 81% | — |
| **BH basket** | **+59.3%** | **0.97** | −52.3% | 100% | — |
| staking_yield gate (basket) | +27.2% | 0.48 | −52.3% | 81% | — |
| cex_dex_flow gate (btc) | +59.0% | 1.83 | −14.8% | 42% | 0.03 |

### cex_dex_flow: the one lead, and why it isn't a confirmation

The `cex_dex_flow → btc` gate (long when ETH CEX netflow is above its
first-half mean) beats BH on the second half — Sharpe 1.83 vs 1.18, smaller
drawdown, and the circular-shift placebo passes (p = 0.03). Before getting
excited, the honesty bar:

- The regression that licensed the overlay is **1 of 18 at p = 0.047** —
  exactly the multiple-testing expectation — and does **not** survive
  Bonferroni. It also fails to show up on the correlated basket target
  (p = 0.111).
- The split diagnostic shows the coefficient lives in the **second half**
  (t = +2.69 vs +0.71 first half). The overlay's evaluation window is that
  same second half, so the "out-of-sample" protocol here protected only the
  *sign choice*, not the *decision to test this factor* — which was made by
  a full-sample p-value dominated by the eval window. This is
  hypothesis-generation and evaluation on overlapping data.
- The sign is *positive* (inflows to CEXes → higher next-week BTC return),
  the opposite of the usual "exchange inflows = sell pressure" story, which
  raises the prior that this is window-specific.

Filed as **suggestive, unconfirmed**. The clean test is a true holdout:
re-run this exact spec (frozen sign, frozen center) on post-2025 data as it
accrues, before believing anything.

## Verdict

**The weekly/monthly horizon does not rescue the factor→return premise.**
Of 18 pre-registered weekly regressions, the only Bonferroni survivor
(`staking_yield`) flips sign across sample halves and its split-sample
overlay loses to buy-and-hold by 20+ points of annualized return — a
first-half regime artifact, the weekly cousin of the trend artifacts the DML
study flagged. The monthly grid (~60 obs) produces nothing under
multiplicity and flips signs versus weekly. The single residual lead,
`cex_dex_flow → btc` (p = 0.047, overlay Sharpe 1.83 vs BH 1.18, placebo
0.03), sits at the multiple-testing expectation and its overlay was
evaluated on the same window that generated the hypothesis — suggestive at
best, and explicitly **not** a beat of buy-and-hold by this program's
standards. All three open avenues from `causal_directions_summary.md` are
now tested: intraday (null), longer horizons (null with one unconfirmed
lead); only exogenous event calendars remain.

## Caveats

- **~5 years, one macro regime cycle:** 262 weekly obs spanning exactly one
  bear (2021-11 → 2022-12) and one bull leg — the staking_yield artifact
  shows how easily a full-sample t = −6 can be pure regime coincidence.
- **Full-sample z-scores:** factor scaling is full-sample affine (documented
  acceptable — absorbed by the regression intercept/slope); all *trading*
  thresholds are first-half-only.
- **Coverage asymmetry:** `stable_flow` (n=100) and `funding_basis` (n=113)
  are tested on shorter, mostly-bull windows than the n=260 factors.
- **cex_dex_flow source is ETH-only** (Dune CEX netflow for Ethereum) used
  to predict BTC — plausible as a market-wide flow proxy, but another
  degree of removal.
- **No short leg, long/flat only** — by pre-registration; a long/short
  variant of the cex_dex_flow gate is post-hoc and intentionally untested.
