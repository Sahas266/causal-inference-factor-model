# Driver Selection: Stability Analysis and Regime-Conditional Extensions

## TL;DR

A multi-step research arc, end-to-end, on whether the CPCM strategy should
move from one global driver pick to regime-conditional selection.

**Findings, in order of confidence:**

1. **The global driver pick is a deep compromise.** Across 19 sub-windows
   spanning 2022-2025, the current pick `(cex_dex_flow, vixcls, dtwexbgs)`
   wins **0% of the time** and averages **rank 100 of 220** possible
   3-subsets. *(Option 2)*

2. **There are real regimes in the data, partially identifiable in real
   time.** A 2-state Gaussian HMM on `(VIX, BTC realized vol)` finds two
   persistent regimes with sticky transitions (>0.97 stay-probability).
   The causal pipeline (rolling-window fit + forward filter) recovers
   these regimes with **77% label agreement** vs the in-sample Viterbi
   baseline. *(Phase A + Phase B step 1)*

3. **Phase A's specific per-regime winners were partially a look-ahead
   artifact.** Across 11 different causal-pipeline configurations, the
   per-regime winning driver subsets **never matched** the non-causal
   Phase A picks. The regime *structure* is real; the regime-specific
   *winners* shift when you can't see the future. *(Phase B step 1)*

4. **Regime-conditional selection works as a downside protector, not as
   an alpha source.** The A/B/C backtest (2022-2025, 9 assets, m=3)
   produced the ranking MoE > Hard-switch > Global by Sharpe, exactly as
   theory predicts. MoE cuts max drawdown by 19pp vs Global. But all
   three strategies lose substantial money — the lift is relative to a
   broken baseline. *(Phase B step 2)*

| Strategy | Sharpe | Total return | Max drawdown |
|---|---:|---:|---:|
| Global | -0.78 | -66.1% | -70.4% |
| Hard-switch | -0.34 | -41.9% | -59.1% |
| MoE | -0.17 | -30.9% | -51.4% |

**Recommendation:** do not ship regime conditioning as alpha. The
methodology is built and tested; layer it on top of any *future* base
strategy that actually produces positive returns. The immediate next
research move is fixing the base strategy (the global pick is the weak
link), not more regime work.

**What's in the repo as a result:**

- `causal_portfolio/regimes/hmm.py` — multi-start fit, causal forward
  filter, rolling-window decode
- `causal_portfolio/backtest/regime_engine.py` — walk-forward backtester
  with `hard` and `moe` modes
- `causal_portfolio/diagnostics/{driver_stability,regime_stability,causal_validation}.py`
  — three diagnostic scripts that produced the findings above
- `causal_portfolio/run_regime_ab.py` — A/B/C runner
- 36 new tests, all passing

The rest of this doc is the full research arc with every intermediate
finding, in chronological order, for posterity and reproducibility.

---

## Background

The CPCM pipeline currently selects its top `m` causal drivers **once, over the
entire training period**. The `ComboDriverSelector.rank_all_subsets` function
scores every m-subset of candidate drivers by a commonality score (λ_max of
the residual correlation matrix), picks the best subset, and uses that **same
fixed subset for every rolling-window rebalance** inside the backtest.

This raises two questions:

1. **Is the global pick stable?** Would different sub-periods of the data have
   chosen different drivers? If yes, the global selection is averaging over
   regime-dependent dynamics.
2. **If unstable, is it worth doing regime-conditional selection?**

This document specifies two extensions and presents stability evidence for
the first one.

---

## Option 2 — Driver-stability diagnostic

**Goal:** measure how much the "best m-subset" depends on which time window you
look at. Cheap to build, no strategy change, pure diagnostic.

### Approach

1. Take the full date range used by the strategy
2. Slide a sub-window across it (e.g. 365 days wide, stepping every 60 days)
3. Inside each sub-window, run `rank_all_subsets` on the available factors
4. Record: which subset won, what its score was, what the runner-up was, and
   the rank of the *global* top subset within this sub-window

### What it reveals

- **Win frequency per subset** — if one subset wins 90% of sub-windows, the
  global pick is robust. If wins are split 30/25/20/15/10, the dynamics are
  regime-dependent and regime conditioning is justified.
- **Win frequency per individual driver** — even if subsets shift, individual
  drivers may appear in most winning subsets. A driver that's "in the top
  m-subset 80% of the time" is structurally important; one at 20% is
  regime-specific.
- **Stability of the global ranking** — when the *global* best subset isn't
  the local winner, how far down is it? If it's ranked #2 in most sub-windows,
  the global pick is "close to optimal everywhere." If it's ranked #15, the
  global pick is a compromise.

### What it does NOT do

- Doesn't change the strategy
- Doesn't label regimes
- Doesn't add a separate UI tab (results land in this doc and on stdout)
- Doesn't quantify whether per-regime selection would actually *improve*
  Sharpe — only whether driver instability exists at all

### Effort

~1-2 hours including this doc and the analysis script.

### Cost of a false-negative

If we conclude "drivers are stable" and they actually aren't, we leave Sharpe
on the table. Low cost — easy to revisit.

### Cost of a false-positive

If we conclude "drivers are unstable" but the per-window winners are noisy
because of small-sample variance rather than real regime shifts, Option 3
chases ghosts. Mitigated by also checking individual-driver win frequencies
(noisy at the subset level may still be stable at the driver level).

---

## Option 3 — Regime-conditional driver selection

**Goal:** make driver selection (and ideally β fitting) explicitly depend on
a discrete latent regime label inferred from the data. Real research
contribution, several days of work, only worth doing if Option 2 evidence
suggests regimes matter.

### Why this might matter

Concrete examples of regime-dependent driver behavior in crypto:

- **Risk-on / risk-off** — When VIX is low and stablecoin supply is growing,
  `stable_flow` and `liq_flow` dominate returns (capital chasing yield). When
  VIX spikes, `funding_basis` and `mev_pressure` dominate (forced
  liquidations, deleveraging cascades).
- **High-vol / low-vol** — Chain congestion matters during high-vol periods
  (gas spikes correlate with liquidation cascades) but is noise during calm.
- **Bull / bear / chop** — Staking yield matters in bull markets (carry
  trade), `cex_dex_flow` matters during regime transitions (capital rotating
  between venues), neither matters in chop.

A global selector picking the top-m drivers over 2021-2026 effectively averages
over all of these and lands on whatever was most consistently relevant —
possibly a compromise that's mediocre in every specific regime rather than
optimal in any.

### Architecture

```
panel data
    │
    ▼
factors (current pipeline)
    │
    ├──────────────►  EKF filter
    │                    │
    │                    ▼
    │             filtered_drivers (T × m)
    │                    │
    │                    ▼
    │            ┌──────────────────────┐
    │            │ HMM fit + decode     │  ← NEW
    │            │ → regime label per t │
    │            └──────────────────────┘
    │                    │
    │                    ▼
    │              regime_t ∈ {0, 1, 2}
    │                    │
    │  ┌─────────────────┼─────────────────┐
    ▼  ▼                 ▼                 ▼
  selector(           selector(         selector(
   t : regime_t=0)    t : regime_t=1)   t : regime_t=2)
    │                  │                  │
    ▼                  ▼                  ▼
  β_regime_0         β_regime_1         β_regime_2   ← also fit per regime
    │                  │                  │
    └─────────────────┬┴──────────────────┘
                      ▼
              at time t with regime_t:
              mu_t = β_{regime_t} · F_t
                      │
                      ▼
                  optimizer
                      │
                      ▼
                  weights_t
```

### Components

#### Regime classifier (`regimes/hmm.py`)

A **Gaussian HMM** fit to a feature vector derived from filtered drivers.
Typically 2-3 states. Two implementation knobs:

| Knob | Options | Default |
|---|---|---|
| **Feature** | filtered drivers / volatility of returns / VIX level / a small basket | filtered drivers (re-uses EKF output) |
| **n_states** | 2 (risk on/off), 3 (low/normal/high vol), 4+ (overfits) | 2 |
| **Init** | k-means / random restart | k-means with 10 restarts |

Output for each timestep:
- `regime_t` — discrete label
- `regime_posterior_t` — probability over states (useful for soft weighting)

Critical: fit causally — **the regime label at time t uses only data up to
t**, not future data. Standard HMM `decode` is non-causal (looks at full
sequence). Use `predict` step-by-step instead, which is forward-only.

#### Per-regime driver selection

Inside each rolling window:
1. Get the regime label sequence for the window
2. For each unique regime present in the window: run `rank_all_subsets` on
   the **subset of timesteps where that regime was active**
3. Store the winning driver subset per regime in a
   `dict[regime_label, tuple[drivers]]`

At any rebalance, the driver set used is the one that was best **the last
time we were in this regime**.

#### Per-regime solver fitting

Same idea for the β matrix. Three options ordered by complexity:

| Approach | What it does |
|---|---|
| **Hard regime switching** | Train `solver_k` only on data where `regime_t == k`. Predict with whichever solver matches the current regime label. |
| **Soft regime mixing** | Train one solver, but reweight training samples by `posterior_t[k]`. Get one set of β per regime via importance-weighted regression. |
| **Mixture-of-experts** | Train K independent solvers, blend predictions by `posterior_t`. Most flexible, most data-hungry. |

Start with **hard switching** — simplest, easiest to debug.

#### Inference path

At each rebalance time t:
1. Decode regime: `r_t = classifier.predict(filtered_drivers[:t+1])[-1]`
2. Look up drivers for this regime: `current_drivers = driver_map[r_t]`
3. Look up solver for this regime: `current_solver = solvers[r_t]`
4. Compute `mu = current_solver.predict(filtered_drivers[t, current_drivers])`
5. Pass to optimizer as usual

#### Dashboard surface

Add a **Regimes** tab showing:
- Regime label timeline (color-coded ribbon under cumulative returns)
- HMM transition matrix
- Per-regime: driver ranking, β heatmap, Sharpe contribution
- Confusion-style table: which regimes did we spend time in? How long the
  average dwell?

### Best case / worst case

**Best case:** the selector picks `funding_basis` + `mev_pressure` for crisis
regimes and `stable_flow` + `liq_flow` for risk-on regimes. The Sharpe
contribution per regime decomposes cleanly. The model handles 2022's collapse
and 2023's rally differently. Coherence score drops because each regime's
predictions are better tuned.

**Worst case:** the regime classifier is unstable — labels flip frequently,
per-regime samples shrink, β estimates get noisy, total Sharpe gets worse
than the global model. Common when n_states is too high or features are too
noisy.

### Honest risks

1. **Look-ahead reintroduction.** HMM fitting itself can leak future info if
   you fit on the full sample and decode. Need true expanding-window refit.
2. **Sample fragmentation.** A 252-day rolling window with 3 regimes might
   have only 80 samples per regime. β fits get noisy. Either widen the
   window or shrink n_states.
3. **Regime-label instability near transitions.** The day before and the
   day after a regime change can flap back and forth. Common fix: a minimum
   dwell time (require N consecutive days of new regime before switching).
4. **Backtest slowdown.** ~3× slower because you're doing 3 selections + 3
   fits per rebalance.
5. **Overfitting.** More degrees of freedom (one set of β per regime) →
   easier to look great in-sample. Cross-validate or hold out a clean test
   window.

### Effort breakdown

| Step | Effort |
|---|---|
| HMM module + tests | 4-5 hours (using `hmmlearn`, ~150 lines + tests) |
| Causal-decode wrapper (avoid look-ahead) | 1 hour |
| Per-regime selector + solver registry | 2-3 hours |
| Backtester changes to route by regime | 2 hours |
| Dashboard regime tab | 2-3 hours |
| **Total** | **~1-2 working days for a research-quality implementation** |

### Why it matches the paper's hints

The CPCM paper acknowledges that the structural coefficients are not actually
time-invariant — markets shift. The implementation here handles this
implicitly via the EKF (slow drift in state) and rolling-window refit (slow
drift in β). Option 3 makes this explicit: **β can jump, not just drift**, and
the trigger is a discrete latent state inferred from data.

---

## Decision criteria

After running Option 2, the decision tree is:

| Observation | Recommendation |
|---|---|
| Top subset wins >70% of sub-windows; same drivers dominate by frequency | **Skip Option 3.** Global pick is robust. Time better spent on transaction costs, slippage, etc. |
| Top subset wins 30-70%; drivers shift but individual driver frequencies are stable | **Maybe Option 3, lightweight.** Try just per-regime β fitting; keep global driver selection. |
| Top subset wins <30%; individual driver frequencies also unstable | **Option 3 justified.** Full HMM + per-regime selection has the highest expected payoff. |

---

## Option 2 results

### Run parameters

- Date range: `2022-01-01` to `2025-12-31`
- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- m (subset size): `3`
- Window: `365` days, step `60` days
- Sub-windows analyzed: **19**
- Run UTC: `2026-05-18T18:46:44+00:00`

### Global winner (current strategy pick)

- Drivers: `['cex_dex_flow', 'vixcls', 'dtwexbgs']`
- Commonality score: `6.5815`
- Rank-1 in sub-windows: **0.0%**
- Average rank across sub-windows: **104.47**

### Subset win frequency (top 10)

| Rank | Subset | Wins | % of windows |
|---:|---|---:|---:|
| 1 | `['chain_congestion', 'mev_pressure', 'dgs10']` | 2 | 10.5% |
| 2 | `['chain_congestion', 'mev_pressure', 'cex_dex_flow']` | 2 | 10.5% |
| 3 | `['vixcls', 'cpiaucsl', 'm2sl']` | 2 | 10.5% |
| 4 | `['chain_congestion', 'cex_dex_flow', 'vixcls']` | 1 | 5.3% |
| 5 | `['liq_flow', 'cex_dex_flow', 'vixcls']` | 1 | 5.3% |
| 6 | `['mev_pressure', 'vixcls', 't10y2y']` | 1 | 5.3% |
| 7 | `['mev_pressure', 't10y2y', 'dtwexbgs']` | 1 | 5.3% |
| 8 | `['chain_congestion', 'dff', 'cpiaucsl']` | 1 | 5.3% |
| 9 | `['liq_flow', 'mev_pressure', 't10y2y']` | 1 | 5.3% |
| 10 | `['liq_flow', 'chain_congestion', 'mev_pressure']` | 1 | 5.3% |

### Individual driver win frequency

How often each driver appeared in the *winning* m-subset.

| Driver | Appearances | % of windows |
|---|---:|---:|
| `mev_pressure` | 11 | 57.9% |
| `chain_congestion` | 10 | 52.6% |
| `cex_dex_flow` ← in global pick | 6 | 31.6% |
| `vixcls` ← in global pick | 6 | 31.6% |
| `cpiaucsl` | 5 | 26.3% |
| `m2sl` | 5 | 26.3% |
| `t10y2y` | 4 | 21.1% |
| `liq_flow` | 3 | 15.8% |
| `dgs10` | 2 | 10.5% |
| `dtwexbgs` ← in global pick | 2 | 10.5% |
| `dff` | 2 | 10.5% |
| `staking_yield` | 1 | 5.3% |

### Per-window winners (chronological)

| Window | n_obs | Winner | Score | Global pick rank |
|---|---:|---|---:|---:|
| 2022-01-02 → 2023-01-02 | 364 | `['chain_congestion', 'cex_dex_flow', 'vixcls']` | 7.0945 | 9 |
| 2022-03-03 → 2023-03-03 | 365 | `['liq_flow', 'cex_dex_flow', 'vixcls']` | 7.0640 | 10 |
| 2022-05-02 → 2023-05-02 | 365 | `['mev_pressure', 'vixcls', 't10y2y']` | 7.0582 | 61 |
| 2022-07-01 → 2023-07-01 | 365 | `['chain_congestion', 'mev_pressure', 'dgs10']` | 6.8357 | 211 |
| 2022-08-30 → 2023-08-30 | 365 | `['chain_congestion', 'mev_pressure', 'dgs10']` | 6.6851 | 210 |
| 2022-10-29 → 2023-10-29 | 365 | `['mev_pressure', 't10y2y', 'dtwexbgs']` | 6.6309 | 175 |
| 2022-12-28 → 2023-12-28 | 365 | `['chain_congestion', 'dff', 'cpiaucsl']` | 5.7631 | 194 |
| 2023-02-26 → 2024-02-26 | 365 | `['liq_flow', 'mev_pressure', 't10y2y']` | 5.4262 | 206 |
| 2023-04-27 → 2024-04-26 | 365 | `['liq_flow', 'chain_congestion', 'mev_pressure']` | 5.3234 | 141 |
| 2023-06-26 → 2024-06-25 | 365 | `['chain_congestion', 'mev_pressure', 'm2sl']` | 5.3072 | 202 |
| 2023-08-25 → 2024-08-24 | 365 | `['chain_congestion', 'mev_pressure', 'cpiaucsl']` | 5.5284 | 150 |
| 2023-10-24 → 2024-10-23 | 365 | `['chain_congestion', 'mev_pressure', 'dtwexbgs']` | 5.5845 | 108 |
| 2023-12-23 → 2024-12-22 | 365 | `['chain_congestion', 'mev_pressure', 'cex_dex_flow']` | 5.6922 | 101 |
| 2024-02-21 → 2025-02-20 | 365 | `['chain_congestion', 'mev_pressure', 'cex_dex_flow']` | 5.9232 | 49 |
| 2024-04-21 → 2025-04-21 | 365 | `['vixcls', 'cpiaucsl', 'm2sl']` | 6.4969 | 5 |
| 2024-06-20 → 2025-06-20 | 365 | `['staking_yield', 'cex_dex_flow', 'm2sl']` | 6.6025 | 3 |
| 2024-08-19 → 2025-08-19 | 365 | `['cex_dex_flow', 'cpiaucsl', 'm2sl']` | 6.5932 | 3 |
| 2024-10-18 → 2025-10-18 | 365 | `['vixcls', 'cpiaucsl', 'm2sl']` | 6.7692 | 33 |
| 2024-12-17 → 2025-12-17 | 365 | `['dff', 'vixcls', 't10y2y']` | 7.1264 | 114 |

### Interpretation

**Option 3 justified.** The global pick is rarely optimal in any specific sub-window. Full HMM + per-regime driver selection has the highest expected payoff.

---

## Triangulation: same analysis at finer granularity

To rule out small-sample noise, the analysis was re-run with **180-day windows
stepped every 30 days** (43 sub-windows vs 19 in the primary run). The pattern
is identical:

| Metric | 365d / 60d-step | 180d / 30d-step |
|---|---|---|
| Sub-windows | 19 | 43 |
| Global pick rank-1 frequency | **0.0%** | **0.0%** |
| Global pick average rank | 104.47 / 220 | 105.77 / 220 |
| Top driver (`mev_pressure`) appearance | 57.9% | 48.8% |
| Top driver (`chain_congestion`) appearance | 52.6% | 53.5% |

The instability is real, not a window-size artifact.

---

## Final verdict (Option 2 only — superseded by Phase B step 2 below)

> ⚠️ **Note for the reader:** this "Final verdict" was written after Option 2
> and recommended proceeding with regime conditioning. Phase B step 2 ran
> the actual A/B/C backtest and found the methodology works but the base
> strategy is broken; see the TL;DR at the top of this doc and the
> "Honest analysis" section near the bottom for the actual final position.

The current strategy is using **`(cex_dex_flow, vixcls, dtwexbgs)`** — a
subset that **never wins in any of the 62 sub-windows tested across two
granularities**. It ranks ~100 out of 220 possible subsets on average,
meaning the global selector is picking a deep compromise that's mediocre
everywhere rather than optimal anywhere.

Meanwhile, the *individual* drivers that consistently appear in winning
subsets — `chain_congestion` (52-54%) and `mev_pressure` (49-58%) — are not
in the global pick at all. The global commonality score is dominated by
factors that have stable relationships across the full period (macro series
that move slowly), but the *useful* drivers vary by regime.

**Option 3 is justified.** Three observations support the strongest
recommendation in the decision table:

1. **Subset instability is extreme** — no winning subset appears more than
   10.5% of the time in the primary run.
2. **Driver instability is real** — even the best individual driver appears
   in only ~half of the winning subsets.
3. **The chronological pattern is regime-shaped** — early 2022-2023 windows
   favor `vixcls` + macro drivers (risk-off macro environment); mid-2023
   onward favors `chain_congestion` + `mev_pressure` (on-chain dynamics
   dominating in a calmer macro period); late-2024 returns to macro-driven
   winners. This is the kind of regime structure HMM should be able to
   detect.

### What this changes about the strategy as-is

The current global pick `(cex_dex_flow, vixcls, dtwexbgs)` is producing a
commonality score that's *competitive globally* but is rank-100+ locally
**every single sub-window**. The reason it wins globally is that those three
factors have decent residual correlation **on average**, even though much
better picks exist in every specific regime.

Practical implications:
- Reported in-sample Sharpe is likely understating what's achievable —
  a regime-aware version would unlock the regime-specific top picks
- The strategy is currently robust to regime changes (by design — it picks
  what's mediocre-everywhere) but at the cost of being suboptimal everywhere

### Recommended next step

Build the **lightweight first cut of Option 3**:
1. Fit a 2-state Gaussian HMM on filtered drivers (causal decode)
2. Re-run the stability diagnostic *within each regime label* to confirm
   regime-conditional winners are coherent (different from each other,
   stable within their own state)
3. If regime-conditional winners look clean, proceed with full Option 3
4. If they look noisy too, consider 3-state HMM or a different feature
   (volatility-based)

Estimated effort to validate the HMM step alone: 4-5 hours. If positive,
remaining Option 3 work (per-regime solver fits, backtest routing, dashboard
tab) adds another ~1 day.

## Phase A: HMM regime-conditional selection results

### Run parameters

- Date range: `2022-01-01` to `2025-12-31`
- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- m (subset size): `3`
- HMM states: `2`
- Regime features: `vix, btc_vol`
- Total observations: **1440**
- Run UTC: `2026-05-18T18:57:38+00:00`

### HMM characterization

State means (z-scored features, sorted by feature-0 mean):

| State | vix | btc_vol | Interpretation |
|---:|---:|---:|---|
| 0 | -0.494 | -0.456 | low-stress (risk-on) |
| 1 | +1.039 | +0.958 | high-stress (risk-off) |

Transition matrix `P[i,j] = P(next=j | current=i)`:

| From \ To | State 0 | State 1 |
|---:|---:|---:|
| State 0 | 0.991 | 0.009 |
| State 1 | 0.021 | 0.979 |

### Regime dwell statistics

| State | Days | % of sample | # of runs | Mean run length | Max run length |
|---:|---:|---:|---:|---:|---:|
| 0 | 973 | 67.6% | 9 | 108.1 | 347 |
| 1 | 467 | 32.4% | 9 | 51.9 | 121 |

### Per-regime winning subsets

| Regime | n_obs | % | Winner | Score | Runner-up | Global pick rank |
|---:|---:|---:|---|---:|---|---:|
| 0 | 973 | 67.6% | `['chain_congestion', 'mev_pressure', 'vixcls']` | 6.0068 | `['chain_congestion', 'staking_yield', 'vixcls']` | 40 |
| 1 | 467 | 32.4% | `['liq_flow', 'cex_dex_flow', 'vixcls']` | 7.2729 | `['cex_dex_flow', 'vixcls', 'm2sl']` | 8 |

**Global pick (Option 2 baseline):** `['cex_dex_flow', 'vixcls', 'dtwexbgs']` (score `6.5815`)

### Driver overlap across regimes

| Driver | R0 | R1 | In global pick? |
|---|---|---|---|
| `cex_dex_flow` | — | ✓ | ✓ |
| `chain_congestion` | ✓ | — | — |
| `liq_flow` | — | ✓ | — |
| `mev_pressure` | ✓ | — | — |
| `vixcls` | ✓ | ✓ | ✓ |

### Verdict

**Phase B JUSTIFIED.** Different regimes select different drivers, and the regime-specific picks differ from the global pick. This is the expected pattern when regime-conditional selection unlocks Sharpe. Proceed with full implementation.

### Triangulation: 3-state HMM (corrected)

An initial 3-state run with the default seed produced a degenerate solution
(two states flapping daily, mean run length 1). I almost concluded "2 states
is correct." But that was a local-optimum artifact, not a property of the
data. A sweep over 5 seeds × 3 covariance types revealed that 2 of 5 seeds
find a stable, economically meaningful 3-state solution with **substantially
higher log-likelihood** (-2346 vs -2928).

The stable 3-state solution (seed=7, full covariance):

| State | VIX z | BTC vol z | Days | Runs | Mean run | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -0.95 | -0.02 | 432 | 10 | 43 | Macro calm, crypto neutral |
| 1 | -0.12 | -0.56 | 651 | 15 | 43 | Macro tense, crypto quiet |
| 2 | +1.36 | +1.06 | 357 | 7 | 51 | Full stress |

Transition matrix (all self-probs ≥ 0.976 → genuinely sticky):

|  | →0 | →1 | →2 |
|---:|---:|---:|---:|
| 0 | 0.978 | 0.022 | 0.000 |
| 1 | 0.015 | 0.976 | 0.009 |
| 2 | 0.003 | 0.017 | 0.980 |

Per-regime winning driver subsets, all distinct from the global pick:

| State | Winner | Score |
|---:|---|---:|
| 0 | `['chain_congestion', 'cex_dex_flow', 'm2sl']` | 5.3260 |
| 1 | `['chain_congestion', 'vixcls', 'cpiaucsl']` | 6.7405 |
| 2 | `['liq_flow', 'cex_dex_flow', 'vixcls']` | 7.3083 |

Comparison to the 2-state solution: 2 states conflates the 3-state model's
states 0 and 1 (both "calm") into a single regime. The 3-state model
distinguishes "all quiet" from "macro tense but crypto calm" — a real
gradation, since the winners differ between those sub-regimes.

**Implication for Phase B:** support both `n_states=2` and `n_states=3` via
config, and pick by held-out backtest performance. The 3-state model is more
expressive but more fragile (only 2/5 seeds found it), so production code
must **multi-start the fit** and select by log-likelihood, not rely on a
single seed.

### Synthesis: what Phase A demonstrated

1. **Two real regimes exist.** A 2-state HMM on (VIX, BTC realized vol)
   produces persistent labels — state 0 has 9 runs averaging 108 days, state
   1 has 9 runs averaging 52 days. Transition probabilities are
   self-reinforcing (0.991 stay-in-low-stress, 0.979 stay-in-high-stress).
2. **The regimes pick different drivers.** State 0 (low-stress, 68% of
   sample) wins with `chain_congestion + mev_pressure + vixcls`. State 1
   (high-stress, 32%) wins with `liq_flow + cex_dex_flow + vixcls`. They
   share only `vixcls`.
3. **Both regime winners beat the global pick locally.** In state 0, the
   global pick `(cex_dex_flow, vixcls, dtwexbgs)` ranks 40th out of 220
   possible subsets. In state 1 it ranks 8th. The regime-conditional picks
   are the local optima.
4. **The regime structure matches economic intuition.** Low-stress periods
   are dominated by on-chain dynamics (gas spikes, MEV); high-stress periods
   are dominated by liquidity flows (LP exits, CEX-DEX rotation). This is
   what the original Option 3 motivation predicted.

### Important caveat — look-ahead (fuller treatment)

Phase A used the full sample for both HMM fitting and Viterbi decoding. Both
of those are non-causal. To eliminate look-ahead bias in production we have
to fix **two distinct leaks**, not one:

| Leak source | What leaks | Fix |
|---|---|---|
| **HMM fit** | EM optimizes transition matrix + Gaussian emissions over all T timesteps simultaneously. The fitted parameters reflect future regime transitions. | Refit on rolling/expanding window: at rebalance t, fit only on `data[max(0, t-W) : t]`. Refit every M days. |
| **Viterbi decode** | The decoder finds the maximum-likelihood path over the *entire* sequence. The label at time t depends on observations *after* t. | Use the **forward algorithm** instead: compute `P(state_t \| data[0:t+1])` step by step. Causal by construction. |

The `RegimeClassifier.predict_causal` method as initially written only
partially addressed leak #2 (it iteratively decodes growing prefixes, so
each label uses only past observations) but did **nothing** about leak #1
(it called `decode` on a model fit to full data, so parameters know the
future). The fully causal pipeline is:

```python
# At rebalance time t:
window = data[t - W : t]
classifier.fit(window)                       # fix leak #1
posterior = classifier.forward_filter(window) # fix leak #2
current_regime = posterior[-1].argmax()       # label for today
```

#### Why I used non-causal anyway in Phase A

Phase A asks "**does** exploitable regime structure exist in this data?" —
a yes/no question about the data, not a prediction about a strategy. Full-
sample analysis is the right thing for that, the same way you'd plot a
histogram of the full sample to ask "is this distribution bimodal?" without
worrying about lookahead.

Phase B asks "can we **trade** this regime structure in real time?" — now
lookahead is contamination. **Phase B's first task is implementing the two
fixes above and verifying the causal labels still recover the same regime
winners.** If the causal labels track the non-causal labels well (target:
>85% agreement, no per-regime winner flips), Phase A's results carry over.
If they diverge much, we'd need to revisit features or accept a haircut.

### Recommended Phase B plan (original — see "Revised Phase B v2 plan" below)

> ⚠️ **Note:** this plan was written after Phase A under the assumption
> that causal validation would broadly confirm Phase A. It didn't. The
> revised plan is in section "Revised Phase B v2 plan" below, and the
> executed result is in "Phase B step 2".

Given Phase A's strong signal, the full Option 3 build is justified. Concrete
sequencing:

1. **Causal-decode validation** (~1 hour) — Re-run this diagnostic with
   `predict_causal` instead of `decode`. Confirm regime winners stay coherent.
   If they diverge much, refine features before proceeding.

2. **Per-regime selector wiring in the backtester** (~2 hours) — Inside
   `_rebalance`, fit the HMM on the training window, label the window's
   days, pick drivers per regime, route the current rebalance to the right
   driver set based on the most recent label.

3. **Per-regime β fitting** (~2 hours) — Hard regime switching: separate
   `solver.fit()` calls per regime on the training data subset where that
   regime was active. At rebalance time, use the β of the current regime.

4. **Dashboard tab** (~2-3 hours) — "Regimes" tab showing the label timeline,
   transition matrix, per-regime winning subsets, per-regime β heatmaps, and
   regime-decomposed Sharpe contribution.

5. **A/B comparison** (~1 hour) — Run two backtests side-by-side (global pick
   vs regime-conditional) on the same date range and report:
   - Total return delta
   - Sharpe delta
   - Coherence score delta
   - Turnover delta (regime conditioning may increase turnover at transitions)

Total Phase B estimate: **~1 working day** (was originally estimated at 1-2,
revised downward since the HMM module is already built and tested).

### Risk register for Phase B

| Risk | Mitigation |
|---|---|
| Causal decoding produces noisy labels near transitions | Add minimum dwell rule: require N=5 consecutive forecasts of new regime before switching |
| Per-regime sample fragmentation in early rolling windows | Bootstrap: use global driver pick until each regime has ≥100 training observations |
| Increased turnover at regime transitions hurts realized Sharpe | Smooth weights via exponential blending across regime labels; surface in dashboard |
| Overfitting from extra DOF | Backtest hold-out window before reporting; use coherence score as in-sample sanity check |

## Phase B step 1: causal label validation

### Run parameters

- Date range: `2022-01-01` → `2025-12-31`
- HMM states: `2`
- Rolling window: `252` days
- Refit cadence: every `21` days
- Total observations: **1440**
- Causal-labeled observations (post warmup): **1188**
- Run UTC: `2026-05-19T05:09:17+00:00`

### Label agreement

- **Overall agreement: 68.0%**

Per-state agreement (how often causal label matches non-causal label, conditional on non-causal state):

| Non-causal state | Causal agreement |
|---:|---:|
| 0 | 69.1% |
| 1 | 63.6% |

### Dwell statistics comparison

| State | Non-causal days / runs | Causal days / runs |
|---:|---|---|
| 0 | 973 / 9 runs (mean 108d) | 744 / 23 runs (mean 32d) |
| 1 | 467 / 9 runs (mean 52d) | 444 / 23 runs (mean 19d) |

### Per-regime winners comparison

| State | Non-causal winner | Causal winner | Match |
|---:|---|---|---|
| 0 | `['chain_congestion', 'mev_pressure', 'vixcls']` | `['mev_pressure', 't10y2y', 'm2sl']` | — |
| 1 | `['liq_flow', 'cex_dex_flow', 'vixcls']` | `['mev_pressure', 'cex_dex_flow', 'cpiaucsl']` | — |

### Verdict (default config)

**Causal validation FAILED.** Overall agreement is only 68.0%. The regime structure detected in Phase A does not fully survive causal identification.

### Sensitivity sweep across window / refit / n_states

Before concluding, the same comparison was run across 6 configurations:

| Config | Overall agreement | State 0 | State 1 | Winners match |
|---|---:|---:|---:|---:|
| Default (W=252, refit=21, 2-state) | 68.0% | 69.1% | 63.6% | 0/2 |
| Wider window (W=504) | 75.6% | 75.3% | 77.5% | 0/2 |
| Slower refit (refit=63) | 65.9% | 68.7% | 54.7% | 0/2 |
| **Wider + slower (W=504, refit=63)** | **77.0%** | 76.5% | 80.6% | 0/2 |
| 3-state default | 50.6% | 70.4% | 38.3% | 0/3 |
| 3-state wider + slower | 71.0% | 83.3% | 59.8% | 0/3 |

Three takeaways:

1. **Widening the rolling window helps materially.** 252 → 504 days improves
   agreement from 68% to ~77%. More history per fit = more stable parameter
   estimates = labels closer to non-causal Viterbi.
2. **Best causal config plateaus around 77%** — below the 85% threshold I'd
   originally want, but well above the worst case. The regime structure is
   *partially* real, not fully an artifact.
3. **Per-regime winners NEVER match across any configuration.** This is the
   most important finding. Even when the regime *labels* agree 77% of the
   time, the *drivers* picked within each regime by causal labels are
   different from what Phase A found.

### What this actually means

The Phase A finding had two components:
- **Component A:** there exist persistent regimes in (VIX, BTC vol).
- **Component B:** those regimes favor specific driver subsets we identified.

**Component A is partially real** (77% causal agreement at best config). The
HMM can identify regimes in real time well enough to be useful.

**Component B is largely look-ahead.** Phase A's per-regime winners
(`chain_congestion + mev_pressure` for calm, `liq_flow + cex_dex_flow + vixcls`
for stress) were inflated by the non-causal labels assigning the "right"
days to each regime with hindsight. The causal pipeline, doing the same
selection on its own (less perfectly assigned) regime labels, picks
substantially different winners.

This is the classic look-ahead-bias failure mode in regime-switching
strategies: looks great with hindsight, much weaker in real time.

### Where this leaves Phase B

The "Phase B JUSTIFIED" conclusion from Phase A needs to be downgraded. We
should now treat Phase B as a **research bet**, not a confident upgrade:

| Scenario | Pre-test probability | What it means |
|---|---|---|
| Causal regime-conditional beats global pick | ~30% | Real lift; ship it |
| Causal regime-conditional ≈ global pick (within noise) | ~50% | Added complexity without payoff; don't ship; keep as research finding |
| Causal regime-conditional underperforms global pick | ~20% | Per-regime fragmentation hurts more than regime conditioning helps |

**Resolved by Phase B step 2 below:** scenario 1 by Sharpe ranking
(MoE > Hard-switch > Global, all three configurations distinct), but
with the qualifier that all three lose money in absolute terms. So
"beats global" is technically true but doesn't justify shipping.

The only way to settle this is to actually run the A/B test. The
infrastructure (HMM + forward filter + rolling fit) is built and tested —
the remaining work is wiring it into the backtester and running the
comparison. Estimated ~3-4 hours, much of it on backtester refactoring.

### Recommendation

**Run the A/B test, but expect modest or null results.** Specifically:
- Use the best causal config (W=504, refit=63, 2-state) so we're not
  testing the worst version
- Set up the comparison with the existing global pick as baseline
- Report findings honestly; if no lift, document and move on
- Do not build the dashboard tab until the A/B test is positive

This is a less ambitious Phase B than originally planned, but more aligned
with what the causal validation actually supports.

### Additional sensitivity: shorter windows

Question raised: would *shorter* windows help (more adaptive to regime
shifts)? Answer: empirically no.

| Config | Overall agreement | Winners match |
|---|---:|---:|
| W=90,  refit=7 | 57.8% | 0/2 |
| W=126, refit=10 | 58.5% | 0/2 |
| W=126, refit=21 | 54.0% | 0/2 |
| W=180, refit=10 | 64.0% | 1/2 |
| W=180, refit=21 | 59.0% | 1/2 |
| W=252, refit=21 (default) | 68.0% | 0/2 |
| W=504, refit=63 (best) | **77.0%** | 0/2 |

Trend is monotonic: **more data per fit = more stable parameters = more
consistent labels**. The HMM has ~13 free parameters (2 means × 2 features
+ 2 covariances + 4 transitions + 2 initial); shorter windows mean fewer
obs per parameter, more noise in the estimates, more label flapping.

W=180 hit "1/2 winners match" but only because tuple identity is binary
(2 of 3 drivers matching still shows 0). One config out of 11 hitting an
isolated tuple match is consistent with chance.

### Considered methodological variants

Three deeper alternatives were considered before settling on the
hard-switching A/B test:

**1. Many regimes + clustering (e.g., K=20 micro-regimes → agglomerative
merge, or HDP-HMM).** Doesn't address our specific failure mode. The 23%
causal-vs-non-causal label disagreement happens AT regime boundaries.
More regimes means more boundary days and more disagreement opportunity.
Per-regime sample fragmentation also worsens (selector needs N≥60 per
regime; 1440 obs ÷ 5+ regimes ≈ floor). Where this would help: if we
suspected the K=2 model was conflating economically distinct sub-regimes,
which seed=7 3-state already hinted at. Not the right tool for the
specific look-ahead problem.

**2. Mixture-of-experts (MoE) / soft regime mixing.** Directly addresses
the boundary problem. With hard switching, a 50/50 posterior at a boundary
day still picks one regime. With MoE, predictions are blended by posterior
weights — high-confidence days drive per-regime parameters, boundary days
get an even mix. Cost: ~2-3 hours additional code over hard-switching
infrastructure. **This is the better methodological fit for our specific
failure mode.**

**3. Bayesian model averaging.** Train both global and regime-conditional
models, blend by posterior model probability. If regime-conditional wins
out of sample, BMA weight shifts there; if it doesn't, BMA falls back to
global. Self-correcting strategy. Cleaner than a one-shot A/B but more
moving parts.

### Revised Phase B v2 plan

Given the look-ahead finding and the boundary-noise concern, the right
Phase B is:

| Step | Effort | Justification |
|---|---|---|
| 1. Wire causal HMM + hard-switching into backtester | 2-3h | Necessary scaffolding |
| 2. Add MoE blending wrapper using posterior | 1-2h | Addresses boundary noise |
| 3. A/B/C comparison: global vs hard-switch vs MoE | 1h | Empirical verdict |
| 4. Report findings; ship only if MoE clearly wins | 0.5h | Honest gate |
| **Total** | **~5h** | Less than original Phase B estimate |

## Phase B step 2: regime-conditional A/B/C backtest

### Run parameters

- Date range: `2022-01-01` → `2025-12-31`
- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- m: `3`, n_states: `2`
- Rebalance every `5` days, train window `252`
- HMM window `504` days, refit every `63`
- Global selector picked: `['cex_dex_flow', 'vixcls', 'dtwexbgs']`
- Run UTC: `2026-05-19T05:53:14+00:00`

### Backtest metrics

| Metric | Global | Hard-switch | MoE |
|---|---|---|---|
| Total return | -66.1% | -41.9% | -30.9% |
| Sharpe | -0.780 | -0.344 | -0.174 |
| Sortino | -0.758 | -0.339 | -0.173 |
| Max drawdown | -70.4% | -59.1% | -51.4% |
| Calmar | -0.278 | -0.148 | -0.084 |
| Avg turnover | 0.118 | 0.086 | 0.088 |
| Rebalances | 242 | 188 | 188 |

- Hard-switch per-regime rebalance counts: R0: 130, R1: 58
- MoE per-regime rebalance counts: R0: 130, R1: 58

### Verdict — auto-generated by the runner

- **Hard-switch beats global by Sharpe +0.437.**
- **MoE beats global by Sharpe +0.607.**

### Honest analysis (the harder reading of these numbers)

The auto-generated verdict is technically correct but glosses over an
uncomfortable fact: **all three strategies lose substantial money.**

The Phase B "lift" is a lift relative to a broken baseline. The global
CPCM strategy with the picked drivers `(cex_dex_flow, vixcls, dtwexbgs)`
lost 66% over 2022-2025 with a maximum drawdown of 70%. Regime conditioning
cut the losses roughly in half (Hard-switch: -42%, MoE: -31%) but did not
make the strategy profitable.

This is consistent with the Option 2 finding from earlier in this document:
the global driver pick ranks ~100/220 in every sub-window. We knew it was a
deep compromise. Building a regime-aware overlay on top of weak underlying
drivers gives you a less-bad weak strategy, not a good one.

Three honest interpretations:

1. **Regime conditioning is a real downside protection mechanism.** MoE's
   max drawdown is 19 percentage points smaller than global (51% vs 70%).
   That's not noise. The HMM is correctly classifying stress regimes and
   the per-regime selector is finding defensive driver subsets for those
   periods. If this kind of structure exists in the data and you have a
   workable base strategy, regime conditioning would help.

2. **The base CPCM strategy is not working in this market regime.** Across
   2022-2025 — which includes the 2022 crypto winter and the choppy 2024
   period — none of the three strategies produced positive returns. The
   strategy may work in trending bull markets (we got +3.7% Sharpe 0.26 on
   a 2024-2025 sub-window earlier in this session) but gets destroyed when
   the universe trends down. This is not a regime-selection problem; it
   is a strategy-design problem.

3. **The interesting result is the ranking, not the absolute numbers.**
   The fact that MoE > Hard-switch > Global, in that exact order, with
   clear separation, is the strongest empirical evidence we have that
   regime conditioning adds value. The methodology works. It just isn't
   enough to save a strategy whose underlying signal is weak in this
   sample.

### What I would NOT do

- **Do not ship the regime-conditional strategy as alpha.** It loses less
  than the global strategy, but loses. Reporting "we improved Sharpe
  from -0.78 to -0.17" without context would be dishonest.

### What I WOULD do

- **Document this clearly.** The negative result is informative for
  future researchers — both that the global pick is the weak link and
  that regime conditioning is the right method when the base signal is
  workable.
- **Investigate why the base strategy fails 2022-2025.** Likely suspects:
  - The global driver pick is dominated by macro factors (`vixcls`,
    `dtwexbgs`) that don't have a stable causal relationship to crypto
    returns in a bear market.
  - The Markowitz + manifold optimizer may be over-fitting to in-sample
    covariance structure that doesn't hold out of sample.
  - The 2SLS/IV identification may need an instrument refresh.
- **Consider regime conditioning as a downside-protection wrapper** for
  any strategy we DO get working. Even when the strategy itself doesn't
  print alpha, the regime layer cut drawdown by 19pp. That's the kind of
  property a portfolio manager would want regardless of whether the base
  signal is the source of returns.

### Where this leaves Phase B

The methodology is built and tested. The infrastructure stays in the repo
as a research tool. We do **not** ship the regime-conditional strategy to
production as an alpha source.

Productively, the next research move is **not more regime work** — it is
**fixing the base strategy**. Try:
- Different driver universes (drop the macro-dominated global pick)
- Shorter rebalance windows (the 5-day cadence may be too slow during the
  2022 selloff)
- Asset universe trimming (which assets dragged the most? maybe BTC+ETH-only
  performs differently)
- A drawdown-aware optimizer (de-lever when realized vol spikes)

If any of those produce a working base strategy, then re-running Phase B
on top of *that* would be the test of whether regime conditioning is
genuinely additive. Right now we have inconclusive evidence — it helps a
broken strategy, which doesn't tell us much about how it would behave on
a working one.

### Bottom line

Phase B succeeded as research and failed as product. The regime
conditioning works methodologically (MoE > Hard-switch > Global, robust
across modes). But the absolute returns are negative across all three
arms, so there is no version of this you would actually trade today.
The valuable artifact is the documented finding — regime conditioning is
a real downside-protection mechanism, the global driver selector is the
weak link in the current pipeline, and the look-ahead-aware methodology
(rolling HMM fit + forward filter + multi-start) is built and ready for
use on top of a future base strategy that actually works.
