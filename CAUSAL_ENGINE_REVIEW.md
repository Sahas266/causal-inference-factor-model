# Causal Engine + Dashboard Review

## Fix status (2026-07-07) — all findings addressed

Everything below was fixed on 2026-07-07 unless marked otherwise. Verification:
**482 Python tests + 66 Rust tests pass**; Rust↔Python parity fixtures
regenerated and green (coef/SE/F/R² to 1e-6, incl. new HAC SEs).

| Finding | Status |
|---|---|
| C1 driver-selection look-ahead | ✅ selection now runs on the initial train window only (`run_backtest.py`) |
| C2 NaN-poisoned component sums | ✅ `_complete_sum` (ffill limit 5 + `min_count=all`) for TVL/stablecoin/fee sums |
| C3 zero transaction costs | ✅ `CPCMBacktester` + `RegimeConditionalBacktester` charge 5+5 bps on L1 turnover (matches costs.py baselines) |
| C4 wrong covariance in regime fallback | ✅ precomputed asset-return covariance; first-rebalance failure now raises instead of silently trading 1/n |
| C5 Hausman broken | ✅ corrected 2SLS SEs; var_diff≤0 terms skipped, df = terms used (both languages; fixture Hausman 90.92→197.74) |
| H1 DAG encodes no endogeneity | ✅ latent `u_<treatment>` confounders added; instrumented factors now resolve via IV, not vacuous BACKDOOR/∅ (both languages) |
| H2 exclusion test conditions on collider | ✅ Pearl's criterion: remove all edges out of T, test d_sep(Z,Y) — textbook IV with latent confounder now accepted (both languages) |
| H3 decorative identify→estimate chain | ✅ pipeline.rs applies edge lags, gates 2SLS on identification outcome, logs chosen estimator |
| Macro publication lag | ✅ CPI/M2 shifted 45d to release date (`MACRO_RELEASE_LAG_DAYS`); CACHE_VERSION→3 |
| AR(1) innovation full-sample ρ | ✅ trailing-252d ρ/a, shifted one day (causal) |
| Instrument/shock binarization leakage | ✅ rolling z/quantile (252/60) in instruments.py + shocks.py + Rust mirror; treatment-as-instrument fallback deleted (raises) |
| Full-sample factor z-score | ⚪ kept deliberately — affine, absorbed by per-window refits; documented. Same for hmm feature scaling (EM-init only) |
| Coherence score mislabeled | ✅ labeled "in-sample diag" in summary |
| main.py weights ≠ backtested strategy | ✅ now ManifoldOptimizer + EKF + total-return covariance + max-weight cap (same machinery as engine) |
| v4 warm-start + no seed | ✅ fresh seeded init per fit; train history reset per fit |
| Omnibus first-stage F | ✅ partial F (matches ols_vs_2sls to 1e-8), both languages |
| Sargan df / missing X_exog | ✅ df=m−k1, aux includes X_exog, both languages |
| No HAC SEs | ✅ Newey-West added both languages; pipeline prints HAC t/p |
| dropna(how="any") silent shrinkage | ✅ loud warning with worst-offender assets (run_backtest + main) |
| Snapshot resume misses late backfills | ✅ created_at-based sweep + snapshot_meta last-sync record |
| Cache never invalidates | ✅ `CPCM_REFRESH_CACHE=1` escape hatch + version bump |
| `_find_column` ambiguity | ✅ raises on multiple suffix matches |
| `load_prices` name mangling | ✅ removesuffix |
| threshold counts no-op rebalances | ✅ no-op is not a trade; test updated |
| First-rebalance failure silent 1/n | ✅ raises; later failures counted in `n_failed_rebalances` |
| dead `scm/estimation.py` | ✅ deleted (zero importers) |
| All dashboard findings (params staleness, fit_kwargs, diag clobber, maxDD color, preview caps, regime captions, audit network, Sharpe ddof, load_market dupe) | ✅ fixed; 17 dashboard tests incl. 3 new |
| Rust/Python DAG drift (driver filtering, DAG v2 not in Rust) | ⏭️ skipped — porting research-side DAG v2 to Rust is new scope, not a bug fix |
| strategies._walk no inter-rebalance drift | ⏭️ acknowledged by design (documented in code) |

**Numbers caveat:** C1–C3 change backtest results by design (less look-ahead,
real costs). Every previously documented CPCM Sharpe is now stale-optimistic.

**Post-fix re-run (2026-07-07, defaults: 9 assets, m=3, 2022-01-01→2025-12-31,
train-window-only driver selection = `['liq_flow', 'cex_dex_flow', 'vixcls']`,
5+5 bps costs, 242 rebalances):**

| Solver | Total return | Sharpe | MaxDD | Calmar |
|---|--:|--:|--:|--:|
| Combo-EKF-V1 | **−62.0%** | **−0.861** | −64.7% | −0.387 |
| Combo-EKF-V4 (PINN) | **−59.6%** | **−0.759** | −68.9% | −0.332 |
| Buy-and-hold BTC (same era, reference) | +37%/yr | 0.63 | −77% | 0.48 |

Under honest evaluation the CPCM strategies are not merely "not better than
buy-and-hold" — they are deeply negative. The previously reported positive
CPCM figures (e.g. the paper-referenced V4 Sharpe 1.266) were the product of
full-sample driver selection, full-sample factor normalization artifacts, and
zero transaction costs. This closes the loop on the program's structural
conclusion: the factor→return premise fails, and the engine now says so
honestly.

---

**Date:** 2026-07-06. **Scope:** end-to-end — data loaders → factor builder → driver
selection → solvers (v1/v4) → backtest engines → SCM/identification/2SLS (Python +
Rust) → both Streamlit dashboards. Four parallel reviewers (data+factors,
solvers+backtest, SCM, dashboards); every HIGH finding re-verified by direct read
before inclusion.

## Verdict

The engine's **honest-evaluation experiments** (`ols_vs_2sls.py`, the placebo-gated
experiment suite) are clean — proper partial-F gates, train-window-only selection,
correct walk-forward. The problems concentrate in the **production pipeline path**
(`main.py` / `run_backtest.py` / `dashboard.py`): several layers of look-ahead
flatter the headline CPCM backtest, the identify→estimate chain is decorative, and
the dashboards can silently display results that don't match the parameters shown.
Nothing here changes the program's *negative* conclusions (the nulls were
established by the clean experiment harnesses); the bugs mostly make the CPCM
pipeline look **better** than it is, which the buy-and-hold verdict already
discounts.

Execution-dashboard **safety is clean**: no code path submits orders; all configs
are `dry_run=True`; the signing client is constructed lazily so previews never
need the private key.

---

## Critical findings (fix before trusting any CPCM backtest number)

| # | Where | What |
|---|---|---|
| C1 | `run_backtest.py:68-70`, `main.py` | **Driver selection runs on the full sample before the walk-forward.** The m drivers are chosen using the entire test period, then the backtest "discovers" they work. Every reported OOS Sharpe is upward-biased. (`regime_engine.py` does per-window selection correctly — the global runner doesn't.) |
| C2 | `factors/builder.py:121,138,171` | **NaN-poisoned sums fabricate flow spikes.** `panel[available].sum(axis=1)` with default `skipna=True, min_count=0`: one missing day in one TVL/supply component drops the total by that component's whole magnitude; `.diff()` then invents a multi-billion fake outflow + next-day inflow. These spikes dominate the z-scored factor. Fix: `min_count=len(available)` or ffill components first. |
| C3 | `backtest/engine.py` | **CPCM backtester charges zero transaction costs** (no cost/fee anywhere in the file) while `strategies.py` baselines pay 10bp round-trip. Every CPCM-vs-baseline comparison is cost-asymmetric in CPCM's favor at 5-day rebalancing. |
| C4 | `backtest/regime_engine.py:340-343` | **Global fallback passes the factor-panel covariance where the optimizer needs asset-return covariance.** n_factors≠n_assets → exception → swallowed at `run():215` → fallback silently never runs (stale/equal weights). If dims coincide → optimizes on nonsense covariance. |
| C5 | `scm/estimators.py:290-300` + `tsls.rs:200-234` | **Hausman test broken twice** (bit-identical in Rust): uses raw stage-2 OLS SEs instead of the corrected 2SLS SEs computed 30 lines earlier, and clamps `var_diff` to 1e-15 so `stat = d²·10¹⁵` → p≈0. Pipeline near-always "detects endogeneity". |

## High — structural (already known in spirit, still current in code)

| # | Where | What |
|---|---|---|
| H1 | `scm/graph.py:91-123` = `cpcm_dag.rs:65-104` | DAG encodes no endogeneity: factors' only parents are instruments, shocks feed only returns. `identify_all_effects` always returns BACKDOOR/∅; IV fallback unreachable; the graph asserts OLS is unbiased by construction. |
| H2 | `scm/identification.py:240-241` = `identify.rs:177-181` | `check_iv_validity` tests exclusion as `d_sep(Z,Y|T)` — conditions on collider T, so the moment H1 is fixed, every valid instrument gets rejected. Currently masked only because H1 keeps it unreachable. |
| H3 | `cpcm-cli/pipeline.rs:108-140` | **Identify→estimate chain is decorative**: identification results computed, logged, then ignored — estimation regresses everything regardless; edge `lag` attributes consumed by no estimator (macro regressed contemporaneously despite lag=1 in the graph). Python `main.py` never calls identification at all. |

## Look-ahead family (beyond C1)

Ranked by real impact — note the reviewers *disagreed* on the z-score item; resolution below.

- **Macro publication lag** (`base_loader.py:150-163`, MED): FRED series ffilled from observation date, not release date — CPI/M2 visible ~4-6 weeks early to factor construction and driver selection.
- **AR(1) innovation transform** (`builder.py:230-242`, MED): ρ estimated full-sample. Non-affine — genuinely leaks. This transform produced the innovations behind the one "surviving" DML edge (`chain_congestion→btc_next`), adding a leakage caveat to that already-caveated result.
- **Instrument/shock binarization** (`factors/instruments.py:58-90`, `scm/shocks.py:56-76`, MED): full-sample z/quantile thresholds; binarization is non-affine, so spike labels in train windows depend on future data. Feeds `ols_vs_2sls.py`'s instrument matrix.
- **Full-sample `z_score` on factors** (`builder.py:247-253`) — flagged HIGH by one reviewer, "benign" by another. **Resolution: mostly benign for the linear solvers** (affine transform absorbed by per-window regression coefficients) **but not fully benign** for V4-PINN (scale-sensitive init/Jacobian penalty λ) or anything thresholding factor levels. Real but second-order next to C1/C2. Same class: `hmm.py:78-82` full-sample feature scaling (EM init sensitivity only).
- **"Coherence Score"** (`engine.py:176-185`, LOW): EKF fit on the whole test period at once, printed alongside OOS metrics. Diagnostic-only; label it.

## Medium — engine

- `main.py:94-100`: **pipeline "current weights" ≠ backtested strategy** — different covariance (`residual_cov_` vs total), no manifold projection, no max-weight cap, gross-normalization div-by-zero when raw weights ≈ 0. What `main.py` prints was never backtested.
- `v4_pinn.py` + `engine.py:209`: solver instance **warm-starts across rebalances** (contradicts "re-fitted at each rebalance" doc; folds non-independent; `_train_history` unbounded) and has **no torch seed** — v4 backtests non-reproducible.
- First-stage F in `tsls` (`estimators.py:255-270` = `tsls.rs:139-167`) is the **omnibus F, not partial F** — exog controls inflate it past the F>10 bar. (The experiment harness implements partial F correctly; the library/Rust diagnostic doesn't.)
- Sargan df bug (`estimators.py:273-287` = `tsls.rs`): df = m−(k1+k2) instead of m−k1 → silently never runs when any exog control exists; aux regression also omits X_exog.
- No HAC/robust SEs anywhere in scm/ or Rust estimators (daily crypto returns); DW/BP computed then ignored — printed p-values overstated.
- `returns.dropna(how="any")` (`run_backtest.py:73`, `main.py:85`): one spotty asset shrinks the aligned sample for all assets, silently.
- Snapshot resume cursor `MAX(time)` (`snapshot.py:110`): late-backfilled historical rows never sync; local DuckDB silently diverges from warehouse.
- Parquet cache never invalidates within a `CACHE_VERSION` (`base_loader.py:85-92`): partial results (end beyond warehouse coverage) cached forever.

## Medium — dashboards

- `dashboard.py:459,804,915,423-429`: **charts indexed with the live `train_window` slider, not the run's value** — drag slider after a run and every x-axis/config row silently lies. Store run params in `pipeline_data`.
- `dashboard.py:270` vs `engine.py:209`: **V4 sidebar hyperparams (epochs/lr/λ) never reach walk-forward refits** — sweeps return byte-identical backtests.
- `dashboard.py:281-289` + Solver tab: full-sample diagnostic solver clobbered by the backtest's last-window refit — Solver tab shows last-window fit labeled as full-sample.
- `execution_dashboard.py:96-97`: **maxDD delta `delta_color="inverse"` is backwards** — worse drawdown renders green.
- `execution/dashboard_data.py:94-95`: **plan preview disables safety caps** (`max_position_pct=1.0, max_single_trade_pct=1.0` vs CLI defaults 0.30/0.10) — dashboard "what would fire" can materially disagree with what the CLI would actually do.
- Regimes tab (`dashboard_panel.py:106-127`): causal/Viterbi toggle wired correctly, but in causal mode the transition matrix/state means come from a separate **full-sample** fit shown under the "causal" header; rolling-fit state semantics can diverge from full-sample state ordering (esp. 3-state).
- Audit history table ignores the `network` field — testnet and mainnet executions indistinguishable in one table.
- Stale `regime_result` in session_state captioned with *current* sidebar params; Viterbi mode silently ignores the window/cadence sliders.
- Per-regime Sharpe hand-rolled with ddof=1 vs canonical `metrics.sharpe_ratio` ddof=0.

## Low / hygiene

- `scm/estimation.py`: dead module, would crash on first use (DoWhy API called on an EconML DMLIV); delete.
- `scm/shocks.py:78-80`: fallback "instrument" is a transform of the treatment itself — dead today, loaded trap.
- `_find_column` suffix matching returns first match (`builder.py:262-267`) — wrong-column footgun as metrics grow; `load_prices` name-mangling if `price_metric` contains `_`.
- `threshold_l1<=0` counts a "rebalance" every day (`threshold.py:44`) — costless but metadata (n_rebalances in dumped weights JSON) wrong.
- First-rebalance failure silently trades equal-weight 1/n (`engine.py:144,160`) with only a log warning.
- Rust/Python DAG drift: Python has driver filtering + DAG v2; Rust has neither. Estimator numerics are deliberately bit-parity — which means C5/partial-F/Sargan bugs exist identically in both.
- `execution/dashboard_data.py:16-24` duplicates `data/__init__.py` loader alignment verbatim.

## Clean areas (checked, no findings)

Weight-application lag in both backtest engines (internally consistent, no peek);
`RollingHMM` refit/segment indexing; regime one-step-ahead posterior;
`metrics.py` annualization consistency; `costs.py` L1 math; constraint
cap-and-renormalize; Bayes-Ball d-separation (both languages — faithful, correct
collider handling); 2SLS point estimates + corrected-SE formula; `ols_vs_2sls.py`
walk-forward A/B (partial-F gate, explicit reported fallback); duckdb/supabase
loader dedup (thin hooks over shared base); execution dashboard order-safety;
Supabase pagination ordering.

## Suggested fix order

1. **C1–C4** — they corrupt the numbers every other decision leans on (per-window
   driver selection; `min_count` on component sums; charge costs in
   `CPCMBacktester`; pass asset-return covariance in the regime fallback + stop
   swallowing that exception).
2. **Dashboard truthfulness** — store run params in session state; thread
   `fit_kwargs` into refits; fix `delta_color`; use default caps in
   `build_plan_preview`.
3. **C5 + partial-F + Sargan df** (Python and Rust together — they're bit-parity)
   if 2SLS diagnostics will ever be quoted again.
4. **Macro release-lag + AR(1)/binarization leakage** before any new experiment
   builds on those transforms.
5. Delete `scm/estimation.py`, fix `shocks.py` fallback, hygiene items
   opportunistically.
