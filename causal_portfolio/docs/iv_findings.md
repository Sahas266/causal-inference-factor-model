# Instrumental-variable findings — consolidated

What we have tried in the search for valid, strong instruments to identify
factor→return effects causally (2SLS), and what we learned. This curates the
results scattered across `iv_search.md`, `ols_vs_2sls.md`,
`ols_vs_2sls_ungated.md`, `causal_dag_variant_search.md`, and `dag_v2.md`.

## Why instruments at all

The CPCM thesis is that on-chain factors *cause* returns. If a factor is
endogenous (correlated with the return shock — e.g. reverse causality or a
common driver), OLS loadings are biased and a portfolio built on them is
mis-weighted. An instrument **Z** for treatment **T** that (1) moves T
(relevance) and (2) affects returns *only through* T (exclusion) lets 2SLS
recover the causal loading. The whole question: do such instruments exist in
this warehouse, and does using them generalize better out of sample than OLS?

## What was tried

### 1. The four declared instruments (the hand-drawn DAG)
`scm/graph.py` ships four instruments, each lag-1:

| Instrument | Treatment | Result |
|---|---|---|
| `gas_spike` | liq_flow | first-stage F ≈ 0 — irrelevant |
| `liquidation_level` | funding_basis | first-stage F ≈ 0 — irrelevant |
| `stablecoin_mint` | stable_flow | "strong" only because it **is** the treatment |
| `protocol_event` | chain_congestion | first-stage F ≈ 0 — irrelevant |

Mean partial F across windows 0.03–1.35, all far below the weak-instrument
bar of 10. So **gated 2SLS degrades to OLS in every window** — the causal arm
and the correlational arm are bit-identical, and the "does causal identification
generalize better?" question cannot even be posed on this data.

The one apparent exception, `stablecoin_mint` under lag-1, is a **fake**: the
treatment `stable_flow` is `z_score(diff(supply))` lagged one day and the
instrument is `diff(supply)` lagged one day — the same series up to scaling.
The first-stage F is near-infinite by construction (the instrument is the
treatment), which violates exclusion. Tellingly, where this fake-strong
instrument was used, **2SLS did WORSE OOS than OLS** (`global_lag1`: 2SLS 25%
fold win-rate vs OLS 50%).

### 2. Full-warehouse IV search (`iv_search.py` → `iv_search.md`)
Screened **all 308 warehouse metrics** (1,602 per-asset columns) × lag-1
transforms as candidate instruments for the 7 global treatment factors, with
real discipline:
- **Selection on the train window only** (first 252 aligned rows) — no OOS leak.
- **Tautology screen** — reject `|corr(Z,T)| > 0.9` or name-listed factor-source
  metrics (automates the `stablecoin_mint` trap).
- **Direct-path probe** — flag Z if it predicts returns *beyond* T (exclusion
  red flag).
- **Stability** — relevance rechecked on later non-overlapping windows.
- The `level` transform was dropped after it produced 866 spurious-regression
  hits (regressing one persistent series on another inflates F regardless of
  any link).

Result: **943 relevant candidates, 381 clean**. For the first time this found
genuinely **strong, non-tautological instruments**:

| Treatment | Instrument | Gate (windows passed) | mean partial F |
|---|---|---:|---:|
| liq_flow | `aave_net_treasury[diff]` | 108/108 | 259 |
| cex_dex_flow | `eth_total_economic_activity[diff]` | 108/108 | 70 |
| funding_basis | `ena_tvl_usd[diff]` | 46/78 | 16 |

### 3. The decisive A/B (gated 2SLS vs OLS, walk-forward OOS)
Wiring those strong instruments into the OLS-vs-gated-2SLS A/B
(`ols_vs_2sls.py`): **everywhere 2SLS genuinely diverged from OLS, it did
worse out of sample.**

| Treatment ← instrument | OLS median Sharpe | 2SLS median Sharpe |
|---|---:|---:|
| liq_flow ← aave_net_treasury | 0.831 | **0.485** |
| cex_dex_flow ← eth_total_economic_activity | 0.831 | **0.668** |
| funding_basis ← ena_tvl_usd | 1.095 | **0.458** |

The other four shortlist candidates failed the per-window gate (train-window
relevance did not persist).

## The structural reason it fails

Two independent results explain *why* even strong instruments hurt:

1. **The DAG encodes no endogeneity** (`scm/graph.py` audit). Factors have no
   unobserved confounders — their only parents are instruments; per-asset
   shocks feed only returns. So `identify_all_effects` always returns BACKDOOR
   with an empty adjustment set, the IV fallback never fires, and there is
   nothing for 2SLS to purge. (Side effect: the latent flaw in
   `check_iv_validity` — testing exclusion as `d_sep(Z,Y|T)`, which conditions
   on the collider T — is unreachable.)

2. **Causal discovery says the arrow points the other way** (`dag_v2.md`).
   PC + VAR-LiNGAM find the stable edge is `returns → on-chain factors`
   (e.g. `ew_return → liq_flow`), not factor → returns. If factors are
   *downstream* of returns, instrumenting them to predict returns is
   ill-posed.

3. **Prediction does not want causal purging anyway.** 2SLS estimates the
   *structural* β (what returns would do under an exogenous intervention on
   the factor); OLS estimates the *best linear predictor*, which keeps the
   endogenous component because correlated-with-the-error variation still
   *predicts*. For a portfolio you want the forecast, not the counterfactual —
   so even a valid, strong instrument throws away first-stage variance and
   makes the OOS forecast worse. "2SLS < OLS with strong instruments" is the
   textbook outcome when the target is prediction, not inference.

## Bottom line

- **No valid, strong instrument helps** on this data. Weak instruments collapse
  2SLS to OLS; strong ones (now that we have them) make 2SLS worse OOS.
- The causal-estimation question is **closed under the current framing**:
  factor→return identification via IV/2SLS does not beat plain OLS, which does
  not beat buy-and-hold BTC.
- **Where instruments could still matter** (open, not yet exhausted): instruments
  with a genuine *exclusion story* (the warehouse search was purely statistical),
  instruments for the relationships that DID survive (chain_congestion→BTC,
  factor→funding), richer instrument *construction* (multi-lag, event dummies,
  overid/Sargan with several instruments), and *cross-asset* instruments. These
  are the targets of the follow-up search below.

---

## Follow-up IV search (fan-out, 2026-06-17)

Four parallel agents attacked the angles the original purely-statistical
warehouse search left open. Each has its own module + doc. Summary:

| Angle | Experiment / doc | Result |
|---|---|---|
| Economically-motivated (exclusion story) | `iv_economic` | **Null.** The exclusion-clean families (mining/hashrate, issuance schedule, stablecoin mechanics, macro surprise) are all **irrelevant** (first-stage F≈0); the only relevant ones (ETF native flows F=23–34, validator queue F=80–88) carry exclusion risk and **fail the per-window gate** (mean F 0.8–3.9), so 2SLS collapses to OLS. 2/20 candidates clean, 0 help OOS. |
| IVs for the surviving edges | `iv_surviving_edges` | **Unidentified (inconclusive, not refuted).** For `chain_congestion→btc_return` the best instruments (eth_new_users F=7.8, sol_tx_count F=5.8, blob_size F=4.6) **never clear F≥10**, in any regime incl. post-Dencun. For `congestion→funding` the most-relevant instrument (blob_size F=6.0) **trips the direct-path exclusion flag** (t=2.85). No valid strong instrument exists to corroborate either edge; the DML/placebo evidence stands but IV can't confirm causality. |
| Richer construction + over-id (Sargan) | `iv_construction` | Richer transforms (ar1_innov, event, sign, qbucket, cumchg7) surface **many more** strong clean candidates (4,591 relevant / 2,366 clean across 7 transforms), and **3/3 over-identified sets are NOT rejected by Sargan** (cex_dex_flow, funding_basis, liq_flow). But the OOS A/B is the decisive test, and it is the **strongest refutation yet**: where the gate truly engaged with a Sargan-valid strong instrument, **2SLS was worse OOS** — `cex_dex_flow ← eth_total_economic_activity[ar1_innov]` (gate 108/108, meanF 81) OLS medSharpe 0.831 → **2SLS 0.398**; `liq_flow ← sol_chain_median_txn_fee[ar1_innov]` (gate 90/108) 0.831 → **0.031**. Sargan-validity + strength + richer construction still does not make 2SLS beat OLS. |
| Cross-asset instruments | `iv_cross_asset` | **Null.** Only 2/12 cross-asset candidates were strong+clean (`sol_dex_volume→cex_dex_flow` for BTC F=18, clean direct-path; `bnb_dex_volume→cex_dex_flow` for ETH F=16). Both engaged the gate but on **full-window** OOS Sharpe 2SLS = OLS (0.148 vs 0.140) or **worse** (0.052 vs 0.140) — the median-Sharpe "win" was a 4-fold artifact. Cross-asset exclusion buys nothing. |

**Net:** the follow-up reproduces and *strengthens* the original conclusion under
four better framings. An economic exclusion story doesn't help (the clean
instruments are irrelevant); the relationships that DID survive (congestion→BTC,
congestion→funding) **cannot be instrumented** (no strong valid Z exists), so
they remain predictive-not-proven-causal; richer construction finds more strong
candidates and even passes over-identification, yet still doesn't make 2SLS beat
OLS OOS; and cross-asset exclusion produces clean strong instruments that still
don't help. Every angle lands on the same structural truth from the section
above: **there is no factor→return endogeneity in this data for an instrument to
exploit.**
