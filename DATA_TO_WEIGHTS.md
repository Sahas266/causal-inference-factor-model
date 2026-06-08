# Data → Weights: The Full CPCM Pipeline

This document traces the end-to-end flow from raw warehouse data to output
portfolio weights, stage by stage, with the **current state** and **where it
needs to improve** for each. It is the map for anyone (re)entering this project.

> **One-line version**
> `DuckDB/Supabase → build_all_factors → Combo-select drivers → [Ridge OLS *or*
> identified 2SLS] β → EKF-filtered driver state → ManifoldOptimizer (Σ⁻¹μ
> projected onto the Jacobian tangent space, constrained) → weights → (backtest
> loop | Hyperliquid execution)`

## The headline finding (read this first)

Across every experiment in this repo — the CPCM pipeline, regime gating,
REGIME_EW, baskets, all DAG variants × lag grid, OLS vs gated/ungated 2SLS, and
regime × DAG — **nothing beats buy-and-hold BTC out of sample.** The causal
machinery (2SLS, identification, diagnostics) is now real and validated against
the Rust reference, but the binding constraint is **instrument relevance**: the
declared instruments have near-zero first-stage strength, so causal estimation
cannot even separate from plain OLS, and OLS doesn't beat BH BTC either.

There are **two estimation paths** that share the same data→factors front-end
(Stages 0–2) and the same optimizer back-end (Stages 4–5):

- **Production / backtest path** — Ridge OLS via `V1LinearSolver`. This is what
  the dashboards, `main.py`, and `run_backtest.py` actually run. The DAG is
  *decorative* here (plain correlational fit; edges/instruments unused).
- **Causal path** — `scm/` + `experiments/` (Steps 4–6 of the causal goal):
  graph identification + gated 2SLS. Built to test whether causal estimation
  generalizes better. It does not, on this data, because the instruments are
  weak.

---

## Stage 0 — Data access

**Code:** `causal_portfolio/data/__init__.py::get_loader()`,
`data/duckdb_loader.py`, `data/supabase_loader.py`

`get_loader()` resolution order:
1. `CPCM_FORCE_REMOTE=1` → Supabase (escape hatch).
2. `CPCM_LOCAL_DB` env var → DuckDB at that path.
3. Default local snapshot `causal_portfolio/data/cpcm_local.duckdb` if present.
4. Fall back to Supabase REST.

Three reads, all off the `asset_metrics_best` view (best-provider-priority row
per `(asset, metric, time)`):
- `load_panel(assets, metrics, start, end)` → wide DataFrame, columns
  `{asset}_{metric}`, daily-resampled (`.resample("D").last()`).
- `load_returns(assets, …)` → **log** returns `log(price / price.shift(1))`,
  columns `{asset}_return` (falls back `PriceUSD` → `price`).
- `load_macro(series, …)` → FRED series (`asset='macro'`), daily ffill.

Both loaders share a parquet cache (`data/cache/`) keyed by query hash.

**Current state:** Solid. Local-first DuckDB (~216 MB snapshot, 2.85M rows,
2021–2026) means reads don't consume the Supabase quota. DuckDB `asset_metrics_best`
reproduces the Postgres `DISTINCT ON ... provider_priority` via `QUALIFY`.
Timezone bounds normalized to UTC (`_to_utc_bound`).

**Where to improve:**
- Snapshot is a point-in-time copy — add a scheduled `snapshot.py` refresh so
  local doesn't silently drift from the live warehouse.
- `load_returns` uses log returns, but the backtester/optimizer compound with
  `cumprod(1+r)` (simple-return semantics). The causal experiments fix this by
  `expm1`-ing to simple returns; the **V1 backtest path still mixes the two**.
  Unify on one convention end-to-end.

## Stage 1 — Factor construction

**Code:** `factors/builder.py::build_all_factors(panel, macro)`,
`factors/instruments.py::build_instruments(panel)`

**14 candidate factors**, each z-scored for cross-comparability:
- 7 global (on-chain): `liq_flow, stable_flow, funding_basis, chain_congestion,
  staking_yield, mev_pressure, cex_dex_flow`
- 7 macro: `dff, dgs10, vixcls, t10y2y, cpiaucsl, m2sl, dtwexbgs`

Each factor has a primary source + fallback chain (e.g. `chain_congestion` =
Dune gas price → CoinMetrics `FeeTotNtv`). All-NaN factors are dropped.

`build_instruments` produces 4 IV series + `iv_map` (treatment → instrument):
`gas_spike→liq_flow`, `liquidation_level→funding_basis`,
`stablecoin_mint→stable_flow`, `protocol_event→chain_congestion`. Spike
instruments are lagged |z(Δ)|>2σ indicators; `stablecoin_mint` is a lagged level
diff.

**Current state:**
- `funding_basis` was hard-coded NaN; **now live** from Hyperliquid
  `funding_rate_8h` (792 obs from 2023-11). `stable_flow` live from stablecoin
  supply diffs (usdc/usdt/usde). Both wired into the analysis path
  (`experiments/dag_variants.load_inputs`), with stablecoins loaded as factor
  inputs only (never traded).
- Instruments build cleanly but are **weak by construction** (see Stage 3).

**Where to improve — THIS IS THE #1 BOTTLENECK:**
- **Find stronger instruments.** The current IVs are sparse spike indicators or
  near-trivial diffs with first-stage partial F ≈ 0.03–1.35 (need ≥ 10). They
  have essentially no relevance, so 2SLS can't identify anything. Candidate
  genuinely-exogenous instruments to source:
  - **Scheduled protocol events** — hard-forks, EIP activations, halvings,
    token-unlock cliffs (calendar-known, plausibly exogenous to same-day return
    shocks, move on-chain factors).
  - **ETF / fund flow announcements** — creation/redemption prints, 13F filing
    dates.
  - **Regulatory / macro calendar shocks** — FOMC surprise component, CPI
    surprise (actual − consensus), not the level.
  - **Oracle / liquidation cascades** triggered by exogenous price gaps.
  - **Cross-venue funding dislocations** as instruments for funding_basis.
  - Validate each candidate with both the **graph** exclusion check
    (`scm.identification.check_iv_validity`) AND the **data** first-stage F
    BEFORE trusting it. (See the `stablecoin_mint` cautionary tale in
    `docs/causal_dag_variant_search.md`: it "passed" the strength gate under
    lag1 only because it was the lagged treatment up to scaling — a relevance
    pass masking an exclusion violation.)
- `staking_yield`, `mev_pressure`, `cex_dex_flow` are ETH-only single-series
  proxies — broaden to cross-asset.
- `funding_basis` only spans 2023-11+, which truncates any window that includes
  it. Backfill earlier funding or accept the shorter window explicitly.

## Stage 2 — Driver selection

**Code:** `factors/combo_selector.py::ComboDriverSelector`

`rank_all_subsets(R, F, m)` ranks every `m`-factor subset by a **commonality
score** = largest eigenvalue (λ_max) of the residual-return correlation matrix —
i.e. the subset whose joint variation best explains common return movement.
Selected subset becomes the regressor pool ("which factors are direct causes of
returns").

**Current state:** Works; used by the `combo_m` DAG variants and the regime
backtester. This is the closest thing to a learned DAG structure (which edges
into the return node).

**Where to improve:**
- Combinatorial `rank_all_subsets` is exponential in the pool size — fine for
  ~14 factors, won't scale if the factor set grows. Add a greedy / forward-
  selection mode.
- Selection happens per training window inside the walk-forward, which is
  correct, but the **selection itself is a multiple-testing surface** — record
  selection stability across windows (how often the same subset wins).

## Stage 3 — Estimation (driver → return loadings β) — *the fork*

### Production / backtest path
**Code:** `solvers/v1_linear.py::V1LinearSolver`

`returns = drivers @ β + intercept` via **Ridge (α=1)**, fit jointly across
assets, + Ledoit-Wolf residual covariance. Exposes `predict(F)` (→ μ) and
`jacobian(F)` (→ βᵀ). The DAG is not consulted — purely correlational.

### Causal path
**Code:** `scm/graph.py`, `scm/identification.py`, `scm/estimators.py`,
`experiments/ols_vs_2sls.py`

1. `build_cpcm_dag(assets, selected_drivers)` → networkx star-DAG
   (factors → covariates → returns, + instruments + unobserved shocks).
2. **Identification** (`identification.py`, port of Rust `dsep.rs`+`identify.rs`):
   - `d_separated` — Bayes-Ball.
   - `backdoor_adjustment_set` — Pearl backdoor (empty → singles → pairs → full).
   - `check_iv_validity` — graph relevance (Z ⊥̸ X) + exclusion (Z ⊥ Y | X).
   - `identify_all_effects` — backdoor-first, IV-fallback.
3. **Estimation** (`estimators.py`, port of Rust `ols.rs`/`tsls.rs`/
   `diagnostics.rs`): `ols`, `tsls` (first-stage F, Sargan, Hausman), and
   diagnostics (Durbin-Watson, Breusch-Pagan, Jarque-Bera, VIF). SVD pseudo-
   inverse with the same cutoff as Rust; scipy only for the t/χ² CDFs.
4. **Gated 2SLS** (`ols_vs_2sls.py`): instrument an endogenous factor only when
   its first-stage **partial F ≥ 10**; otherwise fold it into the exogenous set
   (2SLS degrades to OLS for weak instruments).

Both paths emit the same `predict(F)` / `jacobian(F)` interface, so Stage 5 is
identical.

**Current state:**
- Causal core is **complete and validated to ~1e-6 against Rust fixtures**
  (`tests/fixtures/rust_estimator_fixtures.json`, emitted by
  `cpcm-estimate/tests/emit_fixtures.rs`); every Rust dsep/identify unit test is
  mirrored 1:1. 27 estimator/identification tests pass.
- In the current star-DAG, **all factor→return effects are backdoor-identified
  with the empty set** (only confounding is via unobserved shocks, which can't
  open a backdoor) — so the DAG as drawn says **OLS is already causally valid**.
  2SLS only diverges under genuine factor endogeneity, which Hausman detects.
- **Empirical result:** gated 2SLS = OLS (all instruments fail the F gate);
  ungated 2SLS is *worse* than OLS (median Sharpe +0.83 → −0.60), the textbook
  weak-instrument failure. See `docs/ols_vs_2sls.md` and
  `docs/ols_vs_2sls_ungated.md`.

**Where to improve:**
- Wire the **causal path into the shipping backtester** so the dashboard can run
  identified 2SLS, not just Ridge OLS. Today the causal layer lives in
  `experiments/`; `engine.py` only knows `CPCMSolver` (Ridge). A `TslsSolver`
  that carries aligned instruments would unify them — but it's only worth doing
  once the instruments are strong enough to matter (Stage 1 bottleneck).
- The Hausman/Sargan use the Rust's diagonal/total-column simplifications (kept
  for fixture parity). For real inference, upgrade to the full
  variance-difference Hausman and the canonical Sargan df = m − k_endog.
- Diagnostics (`compute_diagnostics`) are ported but not yet surfaced in any
  report — feed first-stage F / Hausman / VIF into the dashboards.

## Stage 4 — State filtering

**Code:** `filters/ekf.py::CPCMKalmanFilter`

`fit_dynamics(driver_history)` learns a linear state-transition for the driver
vector; `filter(...)` returns the filtered series; the **last filtered state** is
`F_current`. Denoises drivers before they drive weights. Toggleable; off → raw
last driver row.

**Current state:** Works in the V1 backtest path. The causal experiments
(`ols_vs_2sls`, `regime_dag`) **skip EKF** (use the last training row) to keep
the OLS/2SLS A/B clean — both arms identical, so the comparison is fair.

**Where to improve:**
- EKF dynamics are assumed linear/Gaussian; no validation that the learned
  transition actually improves OOS driver prediction. Ablate EKF-on vs EKF-off
  on the production path.
- Reconcile the two paths: either bring EKF into the causal experiments or
  document why it's excluded as a permanent choice.

## Stage 5 — Weights

**Code:** `optimizer/manifold.py::ManifoldOptimizer.optimize(solver, F_current, cov)`

1. μ = `solver.predict(F_current)` — expected returns (n_assets,).
2. J = `solver.jacobian(F_current)`; tangent-space projector **P = UUᵀ** from the
   SVD of J (keep singular vectors above 1e-10·σ_max).
3. Mean-variance: **w_mv = (1/λ) Σ⁻¹ μ** (Σ = Ledoit-Wolf covariance, +1e-6 I).
4. Manifold projection: **w = P · w_mv** — weights must lie in the span of how
   drivers move returns, not idiosyncratic noise.
5. Constraints: optional long-only clip → clip to ±`max_weight` → normalize so
   Σ|w| = 1 (fully invested).

→ **output: a weight vector over assets.**

**Current state:** Works; shared by both estimation paths and the regime
backtester. The manifold projection is the distinctive CPCM step.

**Where to improve:**
- No transaction-cost or turnover penalty in the objective — turnover is only
  *measured* post-hoc (`average_turnover`). For live use, add a cost term or
  no-trade band.
- `Σ⁻¹` via dense `np.linalg.inv` with a fixed 1e-6 ridge; use the same
  Ledoit-Wolf-shrunk Σ consistently and a stable solve.
- `risk_aversion` (λ) and `max_weight` are fixed hyperparameters, never tuned
  walk-forward. Either justify the defaults or tune them OOS-honestly.

## Stage 6 — Walk-forward orchestration

**Code:** `backtest/engine.py::CPCMBacktester.run(returns, drivers, dates)`,
`validation/walk_forward.py`

Every `rebalance_freq` days: refit Stage 3 on a trailing `train_window`, run EKF
(Stage 4), optimize weights (Stage 5); hold between rebalances; accumulate the
portfolio return series. That OOS series is scored by the walk-forward harness
(`validation/walk_forward.py`): partition into non-overlapping folds (with
purge/embargo), report per-fold win-rate vs BH BTC + a multiple-testing note.

**Current state:** The harness is the honesty backbone — it converts "one
arbitrary OOS number" into a distribution and flags selection bias.
`compare_variants` records trial count.

**Where to improve:**
- The harness scores an *already-walk-forward* series into folds; it does not
  itself re-fit. Fine, but make sure callers always feed it genuinely OOS
  series (the experiments do).
- Add deflated-Sharpe / PBO (probability of backtest overfitting) to quantify
  the multiple-testing across the many variants tried.

## Stage 7 — Execution (live)

**Code:** `execution/` (`cli.py`, planner, order-book-aware executor)

Turns a weight vector → Hyperliquid orders: read account state, diff to target
weights, order-book-aware IOC slicing against the live L2 book (handles thin
books, partial fills, oracle-band rejects). **Testnet by default**; `--mainnet`
requires explicit confirmation. Credentials via `HYPERLIQUID_*` env vars
(gitignored `.env`).

**Current state:** Order-book-aware execution built and testnet-verified.
Dust-filter removed (closes residual positions). Audit logs gitignored.

**Where to improve:**
- No live reconciliation loop / fill-quality reporting fed back to the strategy.
- Mainnet path exercised only minimally — slippage/impact modeling vs realized
  fills would close the loop between backtest assumptions and live results.
- The private key in `.env` should be rotated (it was exposed during debugging).

---

## Reference reports

| Report | What it shows |
|---|---|
| `causal_portfolio/docs/dag_variant_search.md` | 10 DAG structures × lag grid, correlational (Ridge) backtest |
| `causal_portfolio/docs/causal_dag_variant_search.md` | Same variants under OLS vs gated 2SLS |
| `causal_portfolio/docs/ols_vs_2sls.md` | Core OLS-vs-2SLS OOS test (gated) |
| `causal_portfolio/docs/ols_vs_2sls_ungated.md` | Ungated 2SLS — weak-instrument failure demo |
| `causal_portfolio/docs/regime_dag_search.md` | Regime × DAG (exploratory) |
| `causal_portfolio/docs/regime_ew_production_readiness.md` | Why REGIME_EW's headline was an artifact |
| `causal_model/README.md` | Rust engine = reference-only port oracle |

## The single most important next step

**Source and validate stronger instruments (Stage 1).** Everything causal in
this pipeline is gated on first-stage instrument relevance, and the current IVs
have none. Until there is at least one instrument that clears F ≥ 10 *and* passes
the exclusion check on real data, 2SLS cannot do anything OLS doesn't, and the
whole causal apparatus — however correct — cannot earn its keep. The
correlational ceiling is already known: it loses to buy-and-hold BTC.
