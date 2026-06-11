# DAG v2 — the structure the data supports

Date: 2026-06-10. Code: `scm/graph.py::build_discovered_dag`,
`factors/builder.py::innovation_factors`. Evidence:
`causal_discovery.md`, `causal_discovery_innovations.md`,
`dml_effects.md`, `dml_effects_innovations.md`.

## Why redesign

The original star DAG (`build_cpcm_dag`) was a hand-drawn *sample*: all 14
factors → all returns contemporaneously, four instruments, no confounding,
no simultaneity. Discovery showed the data disagrees on every count, and the
factor *construction* itself (z-scored levels of persistent series) poisoned
both regression and CI testing with persistence.

Two changes:

1. **Factors become AR(1) innovations** (`innovation_factors`): the day's
   news, not the level. Validation that this transform does real work: the
   spurious macro "effects" in level-space DML (cpiaucsl −628 bps!,
   t10y2y) **vanish** in innovation space, exactly as persistence artifacts
   should.
2. **Edges are only what replicates.** Each edge carries
   `stability` ∈ {stable, semi-stable, candidate} and an `evidence` string.

## The discovered structure

```
btc_shock ─────────────→ btc_return ──(lag 0)──→ {alt returns}
                            │   ▲
              (lag 0)       │   │ (lag 1, CANDIDATE)
btc_return ──→ stable_flow │ chain_congestion
                            │
ew_return ──(lag 1)──→ liq_flow        [stable, both halves, both spaces]
```

- **`ew_return(t-1) → liq_flow(t)`** — the most replicated edge in the
  panel: both sample halves, both level and innovation space. Yesterday's
  market move drives today's DeFi liquidity flows. (Exploitable for LP/flow
  positioning, not return prediction.)
- **`btc_return(t) → stable_flow(t)`** — contemporaneous, full sample +
  half 2: price moves drive same-day stablecoin issuance/redemption.
- **`chain_congestion(t-1) → btc_return(t)`** — the ONE candidate
  factor→return edge produced by the entire program:
  - DML (orthogonalized, cross-fit): +41 bps/σ in level space (survives
    Bonferroni on btc_next), **+39 bps/σ in innovation space**
    (CI [+8, +69]; survives plain 95%, not Bonferroni).
  - Timing backtest: BTC exposure tilted by yesterday's congestion
    innovation (lev = clip(1 + 0.5·z, 0, 2)) gives **Sharpe 0.729 vs 0.551
    BH** (+167% vs +83% total), circular-shift **placebo p = 0.010** —
    the first signal in the project to pass the placebo.
  - **Caveats that keep it "candidate":** the hypothesis was *selected on
    this sample* (the placebo doesn't price in selection); the effect is
    half-2 concentrated (2024–25 tilt Sharpe 1.33 vs BH 0.96; half 1 flat
    0.16 vs 0.20 — plausibly a post-Dencun/ETF-era regime); zero costs and
    leverage ≤ 2 assumed.
  - Economic story: a congestion *surprise* is blockspace-demand news —
    surges in on-chain demand continue into next-day buying. Post-2024,
    gas dynamics (blobs) and ETF flows may have strengthened the
    demand→price channel.
- **No instruments.** With returns causally upstream and lag-1 timing,
  predetermined regressors need no purging; the IV program (iv_search.md)
  showed even strong instruments only add variance.
- **No claimed edges from the other six factors to returns** — momentum/
  reversal per-asset drivers also failed (momentum_xs.md), so the DAG does
  not pretend otherwise.

## Pre-registered forward validation

The candidate edge is frozen here, before any new data:

> **Test:** daily BTC exposure = clip(1 + 0.5 · z(chain_congestion AR(1)
> innovation, known at t−1), 0, 2), evaluated on data strictly after
> 2025-12-31, against BH BTC on the same window. **Pass:** tilt Sharpe ≥ BH
> Sharpe + 0.10 over ≥ 180 trading days. Anything else demotes the edge to
> "unstable" and DAG v2 loses its only factor→return arrow.

Until that test runs, the honest summary of DAG v2 is: **prices drive
on-chain state; on-chain state does not drive prices — with one
sample-mined, regime-suspect, placebo-surviving exception worth watching.**
