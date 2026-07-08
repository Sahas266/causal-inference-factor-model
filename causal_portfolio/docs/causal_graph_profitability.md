# Can the causal graph be made profitable? — three routes, tested

**Date:** 2026-07-07. Run AFTER the engine fixes (train-window driver
selection, real costs, causal transforms — see `CAUSAL_ENGINE_REVIEW.md`),
so nothing below inherits the old look-ahead.

## Framing

The honest engine now reports the direct route is dead: CPCM V1/V4 lose
~60% where buy-and-hold BTC made +37%/yr, and the structural diagnosis
(`causal_directions_summary.md`) is that prices sit upstream of on-chain
state. The remaining candidates for a *profitable* causal graph are the
directions the evidence actually supports:

| Route | Idea | Module / doc |
|---|---|---|
| A | Graph points INTO funding (`congestion→funding` survived DML) — forecast funding, time the delta-neutral carry | `experiments/funding_carry_timing.py` |
| B | Daily was null; maybe factor→return exists at weekly/monthly horizons | `experiments/weekly_horizon.py` / `docs/weekly_horizon.md` |
| C | Factor→VOL information is real (`vol_effects.md` Part 1) — size the profitable trend strategy with a causal vol forecast | `experiments/causal_vol_sizing.py` / `docs/causal_vol_sizing.md` |

All three under the house honesty bar: trailing fits only, decisions lagged,
real switch costs, circular-shift placebos, multiplicity reported.

## Route A — timing the funding carry with causal parents: **null**

Arms decided at end of day t, earning day t+1's accrual (unit notional,
short-perp+long-spot, 20bp per switch). Forecasts = trailing-252d OLS.

**BTC (2024-07-10 → 2025-12-30, 539 days)** — forecast MSE: AR(1) 8.81e-10,
AR+parents(gas utilization, BTC return) 8.59e-10 (**−2.45%**, the graph does
improve the funding *forecast* slightly).

| arm | ann | maxDD | in-mkt | switches |
|---|--:|--:|--:|--:|
| always_on | **+1.53%** | −0.02% | 100% | 0 |
| sign_gate | −2.82% | −4.50% | 96% | 32 |
| ar_gate | +0.45% | −1.11% | 99% | 8 |
| causal_gate | −1.18% | −3.08% | 97% | 20 |

ETH: same shape, worse (causal forecast delta −0.08%; every gate deeply
negative). Placebo p's are *low* (0.01–0.05) — the gates do time something
real (they avoid negative-funding days better than random placement) — but
20bp switch costs plus missed accrual exceed what the timing saves. **The
carry is best left unmanaged**, and the causal parents' forecast edge is too
small to monetize.

Caveats: accrual-only PnL (basis/mark-to-market noise of the hedge excluded,
so per-arm "Sharpe" is not comparable to price-risk Sharpe — the honest
number is annualized return); HL-only funding; the evaluable window
(2024-07+) is a low-funding era, which is why always-on earns +1.5%/yr here
versus the ~19–22%/yr documented for the 2023–24 high-funding period.

## Route B — weekly/monthly horizon: **null, one unconfirmed lead**

18 pre-registered weekly regressions (9 factors × BTC/basket, HAC, 262
weeks). Three hit p<0.05 (~0.9 expected by chance). The only Bonferroni
survivor, `staking_yield` (t=−6.4), **flips sign between sample halves**
(−88.9 bps/σ → +14.5) and its overlay loses to buy-and-hold — a regime
artifact, not an edge. Monthly (~60 obs): nothing, signs unstable.

The residual lead: `cex_dex_flow → btc` weekly — first-half-fit long/flat
gate beats BH on the second half (Sharpe 1.83 vs 1.18, maxDD −14.8% vs
−30.4%, placebo p=0.03) — but it is licensed by a 1-of-18 regression at
p=0.047 (exactly the multiplicity expectation) and the coefficient lives in
the same half the overlay is scored on. **Filed as suggestive/unconfirmed**;
requires a frozen-spec rerun on post-2025 holdout before being believed.

## Route C — causal vol-sizing of the trend strategy: **double null**

2021-09-16 → 2025-12-30, 1,567 days, dual_confirm gate on 40% of days.

| arm | ann | Sharpe | maxDD | Calmar | placebo p |
|---|--:|--:|--:|--:|--:|
| unsized dual_confirm | **+35.4%** | **1.17** | −26.0% | **1.36** | — |
| HAR-sized | +29.0% | 1.09 | −24.8% | 1.17 | 0.93 |
| HAR+causal-sized | +32.5% | 1.17 | −25.8% | 1.26 | 0.37 |

Both sized arms trail unsized; the sizing timing is indistinguishable from
random (p=0.93 / 0.37). Worse: the causal factors **degrade the vol forecast
itself** out of sample (MSE 1.95e-4 vs 1.78e-4, DM t=−1.65) — the
full-sample HAC significance in `vol_effects.md` does not survive
trailing-window estimation. Mechanism: the trend gate already exits in
exactly the high-vol episodes a vol forecast would de-risk; post-gate there
is nothing left for sizing to do. This closes the vol-sizing branch that
`vol_effects.md` left open.

## Verdict

**No tested route makes the causal graph profitable.** The graph's one
validated direction (into funding) improves the funding forecast by ~2% MSE
— too little to beat switch costs; the horizon escape hatch is null; the vol
channel is real in-sample but evaporates under walk-forward estimation and
is redundant with the trend gate. After this round, the profitable things in
this repo remain exactly the non-causal ones: **medium-term trend on a
diversified basket** (Sharpe ~1.2–1.3), **diversification itself** (1/N
1.05), and **unmanaged funding carry** as structural income (~1.5%/yr in the
current low-funding era; ~19–22%/yr when funding was hot).

## What is still genuinely open

1. **Exogenous event calendars** (FOMC/CPI release days — the release-lag
   map added in `base_loader.py` provides the dates — plus token unlocks):
   the last untested avenue from `causal_directions_summary.md`.
2. **`cex_dex_flow → btc` weekly**, frozen spec, post-2025 holdout only.
3. **`corr_off_high`** forward test (from `dag_strategies_findings.md` —
   non-causal but the strongest unvalidated signal in the repo).
4. Re-test carry timing if/when funding returns to a high-rate regime,
   where the accrual dwarfs switch costs and timing has room to pay.
