# Causal vol forecast → sizing the trend strategy

**Date:** 2026-07-07. **Module:** `experiments/causal_vol_sizing.py`
(tests: `tests/test_causal_vol_sizing.py`).

## Why this experiment

`vol_effects.md` left a genuinely open combination. Part 1 there is real:
VIX, t10y2y, and liq_flow carry HAC-significant information about forward 5d
realized vol *beyond* trailing vol. But Part 2 (vol-targeting BH BTC) failed
the circular-shift placebo (p≈0.43–0.47) — any time-varying leverage overlay
scored in that window. The untested idea: apply the causal vol forecast not
to buy-and-hold but to the strategy that actually beats BH on Sharpe —
`dual_confirm` trend (close > 100d MA AND 30d return > 0, Sharpe 0.95 vs BH
0.63, `strategy_trend_rotation.md`). Does causal vol forecasting add anything
over (a) the unsized trend and (b) the trend sized by a plain HAR forecast?

## Design (pre-registered, single spec, no sweeps)

- **Data:** BTC daily returns 2021-01-01 → 2025-12-31 (local DuckDB);
  features are *yesterday's* VIX level, t10y2y, and |liq_flow| from
  `build_all_factors`.
- **Target:** forward 5d realized vol (std of returns over (t, t+5]) —
  chosen up front as less noisy than |r_{t+1}|.
- **Forecasts,** refit monthly on trailing 252d windows, training rows
  restricted to those whose forward-vol window is fully realized at fit time:
  - **arm1 (HAR baseline):** fwd5vol ~ 1 + |r_t| + rv5_t + rv22_t
  - **arm2 (HAR + causal):** arm1 + vix_{t−1} + t10y2y_{t−1} + |liq_flow|_{t−1}
- **Sizing:** exposure_t = gate_t × clip(target_vol / pred_vol, 0, 1);
  gate = exact `dual_confirm_signal` (reused from
  `market_making/strategy_execution_benchmark.py`), target_vol = trailing
  252d median of rv5. **No leverage, cap 1** (house rule) — sizing can only
  de-risk.
- **Arms:** (0) unsized dual_confirm, (1) HAR-sized, (2) HAR+causal-sized.
  Common evaluation window (both forecasts defined); exposure applied to day
  t+1's return; 5bp one-way on |Δexposure|.
- **Placebo:** circular-shift the *sizing multiplier* only (trend gate stays
  put), n=200 — does the timing of the sizing beat random sizing with the
  same distribution?
- **Forecast comparison:** Diebold-Mariano-style HAC(10) test on the squared
  forecast-error differential.
- **Causality contract** (unit-tested, perturb-future): gate, vol features,
  target vol, forecasts, and multiplier at day t use data ≤ t only; causal
  factors enter at t−1; the return earned is (t, t+1].

## Results

Window **2021-09-16 → 2025-12-30** (1,567 days; gate on 40% of days).

| arm | ann | Sharpe | maxDD | Calmar | avg exp | placebo p |
|---|--:|--:|--:|--:|--:|--:|
| unsized_dual_confirm | **+35.4%** | **1.17** | **−26.0%** | **1.36** | 0.40 | — |
| har_sized | +29.0% | 1.09 | −24.8% | 1.17 | 0.36 | 0.930 |
| har_causal_sized | +32.5% | 1.17 | −25.8% | 1.26 | 0.37 | 0.370 |

- **Causal increment (arm2 − arm1):** ΔSharpe +0.074 — but both sized arms
  sit at or below the unsized strategy on every metric except a 1pt maxDD
  trim.
- **Forecast MSE:** HAR 1.780e−04 vs HAR+causal **1.950e−04** — adding the
  causal factors makes the vol forecast *worse* out of sample. DM on the
  squared-error differential: mean −1.69e−05, t = −1.65, p = 0.099
  (negative = causal worse; borderline-significantly so).

## Verdict

**Null, and doubly so.** (1) Vol-sizing the profitable trend strategy does
not help: both sized arms give up return relative to the unsized gate while
trimming drawdown by only ~1pt, and the sizing *timing* carries no
information — 93% of random circular shifts of the HAR multiplier do at
least as well (p=0.93), and 37% for the causal version (p=0.37). Whatever
the sized arms achieve, a random multiplier with the same distribution
achieves too. (2) The causal factors don't even improve the vol *forecast*
at this refit cadence: OOS MSE is higher with them (DM t=−1.65) — the
full-sample HAC significance in `vol_effects.md` Part 1 does not survive
being estimated on trailing 252d windows, where the extra regressors mostly
add estimation noise. The apparent arm2 > arm1 portfolio ordering is a
worse-forecast-by-luck artifact, not causal information: arm2's forecasts
are worse, its placebo is unremarkable, and its metrics still trail the
unsized strategy. The trend gate already de-risks in exactly the episodes
where a vol forecast would — after the gate, there is nothing left for vol
sizing to do. This closes the vol-sizing branch left open by
`vol_effects.md`: the buy-and-hold-vs-dual-confirm verdict stands, unsized.

## Caveats

- **Different window than the headline Sharpe 0.95.** The 252d train + 22d
  vol warmup pushes evaluation to 2021-09 → 2025-12, where unsized
  dual_confirm scores 1.17; comparisons are within-window and unaffected.
- **De-risk-only sizing.** With exposure capped at 1 and target_vol at the
  trailing median, the multiplier binds mainly in high-vol spells — which
  the trend gate has often already exited. A leverage-permitting variant
  would be a different (and placebo-flunked, per `vol_effects.md`) trade.
- **Factor construction look-ahead is cosmetic.** `build_all_factors`
  z-scores levels over the full sample; with an intercept, affine transforms
  of regressors leave OLS predictions unchanged. The one exception is
  |liq_flow| (abs of a full-sample-centered series), whose centering
  constant is a static full-sample statistic — a mild leak that, if
  anything, *flatters* arm2, which still loses.
- **One asset, one cycle, ~4.3 years.** Same single-cycle caveat as every
  trend result in this repo.
- **Monthly refit, 252d windows — pre-registered, not tuned.** Faster refit
  or longer windows might rescue the causal forecast's MSE; testing that
  would be the parameter sweep this design forbids.
