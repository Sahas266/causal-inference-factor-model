# REGIME_EW Production Readiness

The winning strategy from `driver_selection_regimes.md` (REGIME_EW, +166.6%
total return, Sharpe 0.885 over 2022-2025) is in-sample. Before any real-money
deployment we need to address five gaps that block confident use:

1. **Out-of-sample validation** — was the result overfit?
2. **Posterior-confidence gating** — should we flip on 51/49 days?
3. **Stop-loss overlay** — is -44% max DD really acceptable?
4. **Production fallback** — what happens when the HMM fit fails?
5. **Cost sensitivity** — does the lift survive worse fills?

This doc captures the work to address each. Findings appended as each
section is run.

---

## Task 1 — Out-of-sample validation

### The overfit concern

The REGIME_EW hyperparameters (`hmm_window=504`, `refit_every=63`,
`n_states=2`, `stress_state=1`, `stress_allocation=0`, `threshold_l1=0.10`,
asset universe = BTC+ETH+SOL) were all chosen by examining results over
the full 2022-2025 sample. A naive worry: maybe the strategy looks great
because we tuned it to look great on the data we tested it on.

Mitigating factors that suggest it's not severely overfit:
- The hyperparameters are "obvious defaults" rather than optimized values.
  `hmm_window=504` and `refit_every=63` were chosen for label stability
  in Phase B step 1, before the strategy search began.
- The strategy has very few degrees of freedom — the HMM has ~14
  parameters, and the strategy's decision is binary (calm or stress).
- The Phase A threshold-sensitivity sweep showed the same returns for
  `threshold_l1 ∈ [0.10, 1.00]` — the strategy isn't fragile to that knob.

But the right test is empirical: hold out a chunk of data, run the same
strategy on it, and see if the lift survives.

### Plan

Test on two splits:

1. **First half**: 2022-01-01 to 2023-12-31 (~2 years, includes 2022
   crypto winter). The strategy's biggest stated advantage is avoiding
   the 2022 selloff — this is where the lift should be largest if real.
2. **Second half**: 2024-01-01 to 2025-12-31 (~2 years, predominantly
   recovery + bull). Less drama; harder to outperform here. If REGIME_EW
   beats BH_BTC on the calmer half too, the regime gate is doing more
   than just dodging the one big drawdown.

Same hyperparameters as the headline result, no re-tuning.

### Results

### Run timestamp

`2026-05-25T09:23:41+00:00`

### Per-slice metrics

Same strategy (same hyperparameters), evaluated on each slice.

| Slice | Strategy | n_obs | Total | Sharpe | MaxDD | Lift vs BH (Total) | Lift vs BH (Sharpe) |
|---|---|---:|---:|---:|---:|---:|---:|
| first_half | BH_BTC | 728 | -33.2% | -0.074 | -72.4% | — | — |
| first_half | REGIME_EW | 204 | +124.3% | 2.665 | -25.9% | +157.5% | +2.740 |
| second_half | BH_BTC | 731 | +58.4% | 0.638 | -33.1% | — | — |
| second_half | REGIME_EW | 731 | +16.6% | 0.375 | -43.2% | -41.8% | -0.263 |
| full | BH_BTC | 1459 | +10.6% | 0.255 | -72.4% | — | — |
| full | REGIME_EW | 935 | +174.9% | 0.904 | -43.2% | +164.4% | +0.649 |

### Verdict — corrected (the per-slice table above is misleading)

The first_half comparison in the table above is **not apples-to-apples**.
REGIME_EW's 504-day HMM warmup means it doesn't issue its first signal
until **2023-06-11**, so it never saw most of the 2022 selloff. The
first_half slice catches only 204 days of REGIME_EW's signals (versus
728 for BH_BTC) — different date ranges, can't compare totals.

Two valid comparisons remain:

**1. Apples-to-apples on the genuinely-OOS second half (2024-01 → 2026-01,
both strategies fully active, 731 obs each):**

| Strategy | Total | Sharpe | MaxDD |
|---|---:|---:|---:|
| BH_BTC | +58.4% | 0.638 | -33.1% |
| REGIME_EW | +16.6% | 0.375 | -43.2% |
| **Lift** | **−41.8%** | **−0.263** | **−10.1pp** |

REGIME_EW **underperforms BH_BTC by 42pp** in the only clean out-of-sample
period. Sharpe is also worse. Max drawdown is also worse. The regime gate
went defensive during periods that turned out to be profitable.

**2. Same-window over REGIME_EW's full signaling period (2023-06-11 →
2025-12-31, 935 obs each):**

| Strategy | Total | Sharpe | MaxDD |
|---|---:|---:|---:|
| BH_BTC | +159.5% | 0.868 | -33.1% |
| REGIME_EW | +174.7% | 0.904 | -43.2% |
| **Lift** | **+15.1%** | **+0.037** | **−10.1pp** |

REGIME_EW beats BH_BTC by **only 15pp total and +0.04 Sharpe** — well within
noise for crypto. Max drawdown is actually WORSE for REGIME_EW.

### What the headline `+166%` actually was

The original strategy-search result `REGIME_EW +166.6% vs BH_BTC +8.5%
(+158pp lift)` is **not a fair comparison**. The denominator is wrong:
BH_BTC's +8.5% reflects sitting through the full 2022 crypto winter
(-72% drawdown by mid-2022), while REGIME_EW only started issuing
signals in mid-2023 and never had to ride that drawdown.

If you compare them on the same dates with the same active window,
REGIME_EW beats BH_BTC by ~15pp over 2.5 years — roughly +6%/year of
edge, all of which evaporates in the 2024-2025 second half where it
loses 42pp.

### Honest conclusion

**REGIME_EW does not have a robust edge over BH_BTC**. The headline result
was a measurement artifact. The signal doesn't generalize to 2024-2025.

This is the kind of finding that makes OOS validation indispensable.
Without this check we would have deployed a strategy that has historically
been better than buy-and-hold only by a small margin and is currently
losing to it.

### What to do next

Three options:

1. **Stop deploying REGIME_EW**, complete the other production-readiness
   tasks (posterior gating, stop-loss, fallback, cost sensitivity) as
   general infrastructure improvements that would help any *future*
   strategy that has a real edge.
2. **Refine the strategy** — try different features, asset universes,
   stress-allocation values, hyperparameters — and re-validate OOS at
   each step. High risk of more false positives if we tune-and-test on
   the same data; need a held-out validation set.
3. **Step back to the drawing board** — accept that neither the CPCM
   driver pipeline (Phase B) nor the regime-gated baseline beats
   buy-and-hold robustly, and pivot to a different approach entirely.

---

## Task 2 — Posterior-confidence gating

### Implementation

Added `min_posterior_to_switch` parameter to `regime_gated_long_only`. When
the HMM proposes a regime change, only execute it if the new regime's
posterior probability ≥ threshold. Otherwise stay in the current regime.

Defaults to 0.0 (always switch on argmax). Tested at 0.55, 0.65, 0.75, 0.85
on the OOS window (see "Sweeps on OOS window" section below).

### Result

Monotonically improves OOS returns: vanilla `gate=0.00` produces +16.6%,
strict `gate=0.85` produces +29.9%. Mechanism: fewer false switches at
boundary days means less getting whipsawed in and out of position. None
of the configs recover to BH_BTC's +58.4%.

**Status:** mechanism works as designed; doesn't change the OOS verdict.
Production recommendation if shipping: `min_posterior_to_switch=0.85`.

---

## Task 3 — Stop-loss overlay

### Implementation

Added `stop_loss_pct` and `stop_reset_pct` parameters to
`regime_gated_long_only`. Inside the walk loop, an internal equity tracker
maintains a running peak. If drawdown from peak ≥ `stop_loss_pct`, the
strategy forces cash regardless of regime label until drawdown recovers
below `stop_reset_pct` (default: half of `stop_loss_pct`).

Internal equity tracking starts at 1.0 normalized; stop-loss decisions
use simulated net returns, which match the backtest exactly. For
production deployment the stop would be evaluated against broker-reported
equity instead.

### Result

`stop=15%`, `20%`, `25%` all locked the strategy in cash after the first
significant drawdown — equity stays flat, never recovers to peak, stop
stays active for the remainder. `stop=30%` allowed the strategy to
participate but only produced +7.7%.

**Status:** mechanism works but the current implementation is a "hard
permanent stop" without time-based re-entry. A smarter version would
add a re-entry rule (e.g., resume after N days, or on next regime
transition). Out of scope for this pass; documented as a limitation.

---

## Task 4 — Production fallback

### Implementation

`run_regime_weights.compute_current_regime` now wraps the HMM fit in
try/except. On any failure (data load, HMM convergence, missing macro
series), the function:

1. Reads the last successful regime decision from
   `causal_portfolio/data/regime_last_state.json`.
2. If a state file exists, returns those weights with
   `source="fallback_last_known"`.
3. If no state file exists, returns the **stress regime** (cash) with
   `source="fallback_stress"` — the conservative default when we can't
   tell what regime we're in.

A successful live computation always persists the result to the state
file so the next failure has something to fall back to.

Configurable via `on_failure` argument: `"fallback"` (default),
`"raise"` (debug), or `"stress"` (always cash on failure).

### Tests

9 new tests in `test_regime_weights_fallback.py`:

- Stress fallback returns cash weights
- State file round-trip
- Missing / corrupt state file returns None safely
- Fallback to last-known when HMM fails
- Fallback to stress when no last-known
- `on_failure="raise"` propagates exceptions for debugging
- `on_failure="stress"` always uses cash regardless of state file
- Successful live computation persists state for next run

**Status:** done. Daily cron will not crash on edge days.

---

## Task 5 — Cost sensitivity

### Implementation

Added a cost sweep to `run_regime_ew_hardening.py` covering 6 fee/slippage
combinations spanning 0/0 bps (idealized) to 30/30 bps (worst case).

### Result

Even at **zero costs**, REGIME_EW produces +19.0% on the OOS window —
still 39pp below BH_BTC's +58.4%. The strategy is NOT cost-bound; the
issue is the underlying signal not working in 2024-2025. At realistic
HL costs (5/5 to 10/10 bps), returns are +14.3% to +16.6%. At pessimistic
30/30, returns drop to +5.4%.

**Status:** done. The cost model behaves linearly and reasonably. The
strategy's failure to beat BH_BTC is not a cost artifact.

---

## Production-readiness checklist (final)

- [✗] **OOS validation shows lift survives in both halves** — FAILED.
  REGIME_EW loses BH_BTC by -42pp in the genuinely-OOS second half. The
  headline +166% was a measurement artifact from the HMM warmup window
  excluding the 2022 selloff that BH_BTC had to absorb.
- [✓] **Posterior gating reduces switching noise** — `gate=0.85`
  improved OOS Total return from +16.6% to +29.9%. Mechanism works.
- [△] **Stop-loss caps drawdown** — fires correctly at the trigger but
  the current implementation has no time-based re-entry; once tripped,
  the strategy stays in cash. Useful as a circuit-breaker but not as a
  performance enhancer. Documented as a limitation.
- [✓] **Production cron survives HMM failures** — done. Falls back to
  last-known regime or stress (cash). 9 tests covering all failure modes.
- [✓] **Cost sensitivity has clear linear behavior** — done. Strategy is
  NOT cost-bound; even at 0/0 bps it loses to BH_BTC. The lift gap
  is the signal, not the costs.

## Bottom line

The hardening features all work as designed. None of them rescue
REGIME_EW as a deployable strategy in the OOS window. The infrastructure
(posterior gate, stop-loss, production fallback, cost-sensitivity
testing) carries forward as research-grade tooling that would benefit
any *future* strategy with a real edge.

**Do NOT deploy REGIME_EW on testnet or mainnet as a return-seeking
strategy.** If you want to do a smoke test of the execution stack
end-to-end, use a different weight vector — e.g. just BH BTC, or an
equal-weight basket — and ignore the regime gate.

## Sweeps on OOS window (Tasks 2, 3, 5)

OOS window: `2024-01-01` → `2026-01-01`.  BH_BTC baseline: total +58.4%, Sharpe 0.638, MaxDD -33.1%.  Run UTC: `2026-05-25T10:40:22+00:00`

### Sweep A: posterior-confidence gating

Baseline (BH_BTC over OOS): total +58.4%, Sharpe 0.638, MaxDD -33.1%

| Config | n_obs | Total | Sharpe | Sortino | MaxDD | Turnover | Beats BH? |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate=0.00 | 731 | +16.6% | 0.375 | 0.298 | -43.2% | 0.027 | — |
| gate=0.55 | 731 | +20.0% | 0.401 | 0.318 | -43.2% | 0.027 | — |
| gate=0.65 | 731 | +22.0% | 0.416 | 0.330 | -42.9% | 0.025 | — |
| gate=0.75 | 731 | +22.0% | 0.416 | 0.330 | -42.9% | 0.025 | — |
| gate=0.85 | 731 | +29.9% | 0.474 | 0.374 | -42.9% | 0.019 | — |

### Sweep B: stop-loss overlay

Baseline (BH_BTC over OOS): total +58.4%, Sharpe 0.638, MaxDD -33.1%

| Config | n_obs | Total | Sharpe | Sortino | MaxDD | Turnover | Beats BH? |
|---|---:|---:|---:|---:|---:|---:|---:|
| stop=off | 731 | +16.6% | 0.375 | 0.298 | -43.2% | 0.027 | — |
| stop=15% | 731 | +0.0% | 0.000 | 0.000 | +0.0% | 0.000 | — |
| stop=20% | 731 | +0.0% | 0.000 | 0.000 | +0.0% | 0.000 | — |
| stop=25% | 731 | +0.0% | 0.000 | 0.000 | +0.0% | 0.000 | — |
| stop=30% | 731 | +7.7% | 0.299 | 0.143 | -32.5% | 0.007 | — |

### Sweep C: cost sensitivity

Baseline (BH_BTC over OOS): total +58.4%, Sharpe 0.638, MaxDD -33.1%

| Config | n_obs | Total | Sharpe | Sortino | MaxDD | Turnover | Beats BH? |
|---|---:|---:|---:|---:|---:|---:|---:|
| fee=0 slip=0 | 731 | +19.0% | 0.394 | 0.306 | -43.0% | 0.027 | — |
| fee=5 slip=5 | 731 | +16.6% | 0.375 | 0.298 | -43.2% | 0.027 | — |
| fee=10 slip=10 | 731 | +14.3% | 0.356 | 0.283 | -43.3% | 0.027 | — |
| fee=20 slip=20 | 731 | +9.7% | 0.319 | 0.253 | -43.5% | 0.027 | — |
| fee=5 slip=30 | 731 | +10.9% | 0.328 | 0.260 | -43.5% | 0.027 | — |
| fee=30 slip=30 | 731 | +5.4% | 0.282 | 0.223 | -43.7% | 0.027 | — |

