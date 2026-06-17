# Regime-switching DAGs — consolidated findings

**Question:** is it viable to identify the current regime, then apply a
regime-specific allocation rule ("a DAG per regime") — e.g. a rule tuned for
volatile regimes used only when volatile, bull/bear rules, etc.?

**Answer: not as tested.** Across four regime-identification families × two
mapping methods, *no* regime-conditional scheme reliably beats the single best
**unconditional** rule (`trend_50`) once a regime-shuffle placebo is applied.
The one form of "regime switching" that *does* work is the trivial one we
already had — and the sophisticated methods do not improve on it.

## How it was tested (the honesty bar)

For each regime family: identify the regime **causally** (label at day *t*
from data through *t−1* / forward-filtered — never full-sample), then assign a
rule from a fixed menu (`hold`, `trend_50`, `vol_target`, `flat`=stables,
`mean_revert`) to each regime two ways:

- **Pre-registered** — an economic mapping fixed *before* seeing OOS results
  (e.g. bull→hold, bear→flat; volatile→de-risk, calm→momentum).
- **Adaptive (causal model selection)** — at each rebalance, for the current
  regime, pick the menu rule with the best *trailing-window* Sharpe in that
  regime using only past data. (The illegitimate alternative — ranking rules
  by full-sample per-regime performance — is the leak that produced false
  positives in this project's history; it was *not* used.)

Each result is judged against **three** bars: BH BTC (Sharpe 0.63, maxDD
−77%), the best unconditional rule **`trend_50`** (Sharpe 0.88, maxDD −57%),
and a **regime-shuffle placebo** (circular-shift the regime-label series,
re-run the same mapping; `p` = share of shuffles with Sharpe ≥ the real one).
A regime scheme only "works" if it beats `trend_50` **and** clears the shuffle.

## Results by regime family

| Family | Doc | Best causal Sharpe | vs trend_50 (0.88) | Shuffle placebo | Verdict |
|---|---|---:|---|---|---|
| Trend / MA (bull/bear) | `regime_switch_trend.md` | **1.01** (ma50 bull→hold/bear→mean_revert) | beats by 0.13 | weak gate† | borderline / cycle-specific |
| Volatility (calm/volatile, +VIX) | `regime_switch_vol.md` | 0.67 | loses | not reached | **null** |
| Causal HMM (2 & 3 state) | `regime_switch_hmm.md` | 0.25 | loses badly | p=0.78–0.94 | **null** |
| Custom (drawdown / efficiency / correlation / breadth) | `regime_switch_custom.md` | < 0.88 | loses | fails | **null** |

## What each family showed

- **Trend / MA regimes** — the *only* family that produced anything above
  `trend_50`: `bull→hold, bear→mean_revert` (Sharpe 1.01, ann +42%). But two
  caveats gut it. (1) `trend_50` is *itself* a bull/bear MA regime switch
  (bull→hold, bear→flat), and it **passes its own shuffle placebo** (p≈0.025) —
  so "shuffle p<0.10" only proves the MA label is informative about trend,
  which the benchmark already exploits; it is *not* evidence of an edge over
  trend_50. (2) The marginal gain is ~64 oversold-bounce days in the single
  2022 bear — thin and cycle-specific. The adaptive variants all *under*-
  performed their best pre-registered cousin (turnover + estimation noise).

- **Volatility regimes** — clean **null**. Nothing beat `trend_50`; most
  configs landed below BH BTC. The decisive evidence is the **adversarial
  control**: the *inverted* mapping (de-risk in calm, momentum in volatile —
  the opposite of the hypothesis) was *no worse* than the "correct" one. If the
  vol label separated up-days from down-days, inverting it would hurt. It
  doesn't — the volatility regime carries no directional information. (In
  crypto, high vol clusters around both crashes *and* parabolic rallies.)

- **Causal HMM** — clear **null** (best causal Sharpe 0.14–0.25 vs 0.88),
  placebo p=0.78–0.94 (timing uninformative). This family also delivers the
  **most important diagnostic in the whole study**: the *look-ahead* Viterbi
  labeling (which sees the full sequence) scores Sharpe **0.93** where the
  honest causal forward-filter scores **0.03** (2-state, pre-registered) — a
  +0.90 inflation from look-ahead alone. That gap is exactly the artifact that
  makes regime strategies look brilliant in careless backtests, and it is why
  the prior `regime_dag.py` finding (HMM × per-regime factor loadings) was a
  mirage.

- **Custom regimes** (drawdown-state, trending-vs-ranging efficiency ratio,
  correlation regime, breadth) — **null**. None beat `trend_50` and survived
  the shuffle; and these regimes are partly *mechanically* derived from the
  same prices being traded, so any apparent edge is partly tautological.

## The synthesis (answer to the question)

1. **Regime switching "works" only in its trivial form, which we already had.**
   A bull/bear MA filter (`trend_50`: hold above the 50-day MA, stables below)
   *is* a regime-conditional DAG, and it is the best risk overlay in the whole
   project (Sharpe 0.88 vs BH 0.63, drawdown roughly halved). So the *idea* is
   sound — but it bottoms out at the simplest possible implementation.

2. **More sophisticated regime ID does not pay.** HMM, volatility buckets, and
   custom composite regimes all *lose* to the simple MA switch once evaluated
   causally. The added machinery buys churn and estimation noise, not skill.

3. **Per-regime strategy *optimization* does not pay either.** Adaptive
   trailing-window selection of the best rule per regime underperformed fixed
   pre-registered mappings everywhere — selecting on short per-regime windows
   overfits.

4. **The apparent wins of regime strategies are two illusions**, both caught
   here: (a) the **shuffle illusion** — a mapping's Sharpe is often reproducible
   with randomly-timed labels of the same dwell, i.e. it is the *average
   exposure* the mapping sets, not the regime *timing*; and (b) the
   **look-ahead illusion** — non-causal (full-sample Viterbi / cluster) labels
   inflate Sharpe by ~0.3–0.9, and that inflation evaporates under honest
   forward-filtering.

**Bottom line:** *yes*, regime → regime-specific allocation is viable, but only
as the one-line MA trend filter we already use; layering HMM/volatility/custom
regime detection and per-regime strategy optimization on top does **not** beat
it out-of-sample. The honest ceiling for "regime ID → per-regime DAG" on this
data is `trend_50` (and, on the diversified basket, `trend_on_basket`). This is
consistent with every other strand of the project: simple beats clever, and
buy-and-hold BTC remains unbeaten on raw direction.

## Caveats (apply to all four families)

- **One market cycle.** 2021–2025 has a single 2022 bear; every regime overlay
  that "helps" does so largely by truncating that one event. The placebo
  validates timing *within* this sample, not across a genuine regime change.
- **Multiple testing.** ~20–27 configs per family (regime defs × mappings ×
  state counts); best in-sample numbers are upward-biased, and the shuffle
  placebo guards timing luck on a *fixed* mapping, not the config search.
- **Look-ahead is the dominant trap** (quantified in the HMM family). Any
  regime backtest that does not forward-filter is not tradable.
- **Stables = 0%, no leverage**: overlays can only de-risk, structurally
  disadvantaged on raw return in a bull market; a real cash yield lifts
  risk-off legs slightly without changing rankings.
