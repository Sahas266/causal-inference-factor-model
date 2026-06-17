# Regime-switching DAG on a causal HMM

A causal Hidden Markov Model (rolling-window fit + forward filter on VIX + BTC realized vol — the live-system label, *not* full-sample Viterbi) identifies the regime; a per-state exposure rule from a fixed menu is applied to BTC (stables = 0%). Two mappings: a **pre-registered** one (risk-on state -> `trend_50`, risk-off -> `flat`, mapping fixed before seeing OOS) and an **adaptive causal** one (per state, pick the menu rule with best *trailing* Sharpe in that state, past data only). Compared to BH BTC, the best unconditional rule (`trend_50`), a **regime-shuffle placebo**, and a **look-ahead Viterbi** version that is marked NOT tradable.

- Asset: BTC | window `2021-01-01` -> `2025-12-31`
- HMM: rolling fit window `252`d, refit every `21`d, multi-start | fee `5`bp one-way | placebo shuffles `200`
- Menu: `hold` (=BH), `trend_50`, `vol_target`, `flat` (stables), `mean_revert` (RSI14<30 long else flat)
- Run UTC: `2026-06-17T20:35:45+00:00`

> **Prior finding (`experiments/regime_dag.py`):** crossing the HMM regime with per-regime OLS factor loadings HURT — the per-regime driver winners were a look-ahead artifact and the causal A/B lost to BH BTC. This experiment removes the driver-attribution step and tests only allocation-rule switching.


## 2-state HMM

**HMM states** (full-sample z-scored feature means, sorted by VIX; state 0 = lowest-stress):

| State | VIX z | BTC vol z | Read |
|---:|---:|---:|---|
| 0 | -0.48 | -0.52 | low-stress (risk-on) |
| 1 | +0.88 | +0.95 | high-stress (risk-off) |

**Causal-label occupancy / dwell** (forward-filter, tradable):

| State | Days | % | Runs | Mean run (d) | Max run (d) |
|---:|---:|---:|---:|---:|---:|
| 0 | 890 | 57% | 33 | 27 | 172 |
| 1 | 663 | 43% | 33 | 20 | 116 |

- Pre-registered rule usage (causal): `flat` 54%, `trend_50` 46%
- Adaptive rule usage (causal): `mean_revert` 38%, `trend_50` 30%, `flat` 14%, `vol_target` 12%, `hold` 7%

**Performance** (placebo p = share of HMM-label circular shifts with Sharpe >= real; low = timing matters):

| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BH BTC | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 1 | — |
| trend_50 (uncond) | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | — |
| prereg (causal) | +0% | 0.03 | 0.01 | -33% | 0.01 | 16% | 102 | 0.94 |
| adaptive (causal) | +7% | 0.25 | 0.17 | -44% | 0.17 | 36% | 169 | 0.79 |
| *— look-ahead (NOT tradable) —* | | | | | | | | |
| prereg (Viterbi look-ahead) | +31% | 0.93 | 0.65 | -52% | 0.61 | 35% | 92 | — |
| adaptive (Viterbi look-ahead) | +3% | 0.07 | 0.05 | -72% | 0.04 | 47% | 142 | — |

**Verdict (2-state):** best causal arm `adaptive (causal)` Sharpe 0.25 vs trend_50 0.88 vs BH 0.63. Does NOT beat trend_50. Placebo p=0.79 (timing is NOT informative). Look-ahead Viterbi best Sharpe 0.93 (+0.68 over causal — the look-ahead inflation).


## 3-state HMM

**HMM states** (full-sample z-scored feature means, sorted by VIX; state 0 = lowest-stress):

| State | VIX z | BTC vol z | Read |
|---:|---:|---:|---|
| 0 | -0.51 | -0.64 | low-stress (risk-on) |
| 1 | -0.18 | +1.11 | intermediate |
| 2 | +1.49 | +0.48 | high-stress (risk-off) |

**Causal-label occupancy / dwell** (forward-filter, tradable):

| State | Days | % | Runs | Mean run (d) | Max run (d) |
|---:|---:|---:|---:|---:|---:|
| 0 | 743 | 48% | 26 | 29 | 149 |
| 1 | 424 | 27% | 35 | 12 | 48 |
| 2 | 386 | 25% | 29 | 13 | 65 |

- Pre-registered rule usage (causal): `trend_50` 41%, `vol_target` 40%, `flat` 18%
- Adaptive rule usage (causal): `mean_revert` 38%, `trend_50` 33%, `flat` 20%, `vol_target` 6%, `hold` 3%

**Performance** (placebo p = share of HMM-label circular shifts with Sharpe >= real; low = timing matters):

| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BH BTC | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 1 | — |
| trend_50 (uncond) | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | — |
| prereg (causal) | +3% | 0.10 | 0.08 | -66% | 0.05 | 52% | 344 | 0.93 |
| adaptive (causal) | +3% | 0.14 | 0.08 | -39% | 0.09 | 25% | 124 | 0.78 |
| *— look-ahead (NOT tradable) —* | | | | | | | | |
| prereg (Viterbi look-ahead) | +20% | 0.45 | 0.38 | -73% | 0.27 | 60% | 277 | — |
| adaptive (Viterbi look-ahead) | +9% | 0.24 | 0.17 | -51% | 0.17 | 43% | 160 | — |

**Verdict (3-state):** best causal arm `adaptive (causal)` Sharpe 0.14 vs trend_50 0.88 vs BH 0.63. Does NOT beat trend_50. Placebo p=0.78 (timing is NOT informative). Look-ahead Viterbi best Sharpe 0.45 (+0.31 over causal — the look-ahead inflation).


## Overall verdict

**No causal-HMM regime-switch arm both beats `trend_50` on Sharpe and survives the regime-shuffle placebo.** The HMM-state conditioning does not add information over simply applying the single best unconditional rule. Where a regime arm's headline Sharpe looks competitive, the placebo shows the result is reproducible by random-timed labels of the same dwell — i.e. it is the *average exposure* the mapping happens to set, not the HMM's *timing*. The look-ahead Viterbi version looks materially better, which is exactly the look-ahead inflation that doomed the prior `regime_dag.py` finding. Consistent with that prior negative result and with the project-wide finding that nothing beats BH BTC out-of-sample.

## Caveats

- **One market cycle.** 2021-2025 is a single bull->bear->recovery; any regime overlay that 'works' largely does so by truncating the 2022 bear once. The placebo validates timing *within* this sample, not across regime change.
- **HMM refit instability.** Rolling-window forward-filter labels flap near transitions and the 3-state fit is fragile (multi-start mitigates but does not remove this); causal labels agree with the full-sample Viterbi only ~77% at the best config (see `driver_selection_regimes.md`).
- **Look-ahead is large.** The Viterbi rows quantify how much a labeler that sees the whole sequence overstates the result; the causal (tradable) rows are the honest ones.
- **Multiple testing / menu choice.** Two mappings × two state counts × a 5-rule menu were tried; the placebo guards timing luck on a *fixed* mapping, not the menu/config search.
- **Prior negative finding.** `experiments/regime_dag.py` already found HMM regime conditioning (over factor loadings) hurt; this cleaner setup tests the allocation-rule variant honestly.