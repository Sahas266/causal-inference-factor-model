# Instrumenting the two surviving edges (congestion -> BTC, congestion -> funding)

The whole-warehouse IV search closed factor->RETURNS 2SLS (`iv_findings.md`). This tests the two relationships that DID survive their placebo tests, with instruments chosen for an a-priori **exclusion story**, not just statistical relevance.

- Assets: `btc,eth,sol,bnb,avax,uni,aave,link,doge`
- Window: `2022-01-01` -> `2025-12-31` | train 252d | weak-instrument bar F ≥ 10.0
- Run UTC: `2026-06-17T17:45:31+00:00`

**Selection discipline:** first-stage F, |corr(Z,T)| and direct-path t use the TRAIN window only. Exclusion is **untestable** — these instruments rest on an economic argument. Read every number as one-cycle / suggestive.

---

## Task 1 — congestion(t-1) → btc_return(t)

Treatment = chain_congestion AR(1) innovation known at t-1. Outcome = BTC simple return at t. A valid Z shifts yesterday's congestion but reaches today's BTC return only through it. Direct-path t flags Z that predicts BTC return *beyond* congestion (exclusion red flag).

### Candidate screen (train window)

| Candidate instrument | train F | \|corr(Z,T)\| | direct-path t | n | flags | exclusion story |
|---|---:|---:|---:|---:|---|---|
| `eth_new_users[diff]` | 7.8 | 0.17 | 0.74 | 1459 | clean | new-address onboarding = organic on-chain demand -> congestion |
| `sol_tx_count[diff]` | 5.8 | 0.15 | 0.75 | 1459 | clean | Solana transaction count: cross-chain demand proxy |
| `sol_total_fees_sol[diff]` | 5.0 | 0.14 | 0.61 | 1459 | clean | Solana chain fees: cross-chain congestion proxy |
| `eth_blob_size_mib[diff]` | 4.6 | 0.13 | 0.41 | 657 | clean | L2 data-availability demand drives ETH gas; rollup posting schedule, arguably orthogonal to BTC return (post-Dencun 2024-03 only) |
| `eth_TxCnt[diff]` | 2.1 | 0.09 | 0.25 | 1459 | clean | raw ETH transaction demand drives fees; not a direct claim on BTC |
| `eth_blob_fees[diff]` | 1.5 | 0.08 | 0.50 | 657 | clean | blob-fee market = DA demand pressure on blockspace (post-Dencun only) |
| `eth_chain_nft_trading_volume[diff]` | 0.8 | 0.06 | 0.75 | 1459 | clean | NFT mints/trades are a non-financial blockspace consumer that congests the chain without being a BTC price bet |
| `bnb_tx_count[diff]` | 0.5 | 0.04 | 0.13 | 1459 | clean | BNB transaction count: cross-chain demand proxy |
| `bnb_total_fees_bnb[diff]` | 0.4 | 0.04 | 0.05 | 1459 | clean | BNB chain fees: cross-chain congestion co-moves with the demand cycle, does not touch ETH-fee congestion's residual or BTC price directly |
| `eth_bridge_deposit_count[diff]` | 0.2 | 0.03 | 0.17 | 1459 | clean | bridge deposit count: cross-chain activity consumes ETH blockspace |
| `btc_TxCnt[diff]` | 0.2 | 0.03 | 0.34 | 1459 | clean | BTC transaction count: chain-demand cycle proxy (different chain's congestion, excludable from ETH-fee residual) |
| `sol_priority_fees[diff]` | 0.1 | 0.01 | 0.47 | 1459 | clean | Solana priority-fee spend: cross-chain blockspace contention |
| `avax_total_fees_avax[diff]` | 0.0 | 0.01 | 0.52 | 1459 | clean | Avalanche chain fees: cross-chain congestion proxy |

### First-stage relevance across regimes (diagnostic)

Selection uses the 2022 train window (top), but the edge lives post-Dencun (2024+). This shows whether instruments are any more relevant where the edge actually appears. Uses only Z and T (never the outcome), so it cannot leak selection.

| Instrument | train-window F | full-sample F | post-2024 F | n post-2024 |
|---|---:|---:|---:|---:|
| `eth_new_users[diff]` | 7.8 | 11.7 | 0.0 | 731 |
| `eth_blob_size_mib[diff]` | 4.6 | 4.8 | 4.8 | 657 |
| `eth_chain_nft_trading_volume[diff]` | 0.8 | 3.4 | 0.0 | 731 |
| `sol_tx_count[diff]` | 5.8 | 2.6 | 0.5 | 731 |
| `eth_blob_fees[diff]` | 1.5 | 2.1 | 2.1 | 657 |
| `eth_TxCnt[diff]` | 2.1 | 1.9 | 0.4 | 731 |
| `bnb_total_fees_bnb[diff]` | 0.4 | 0.6 | 0.1 | 731 |
| `avax_total_fees_avax[diff]` | 0.0 | 0.3 | 0.0 | 731 |
| `sol_total_fees_sol[diff]` | 5.0 | 0.2 | 0.6 | 731 |
| `sol_priority_fees[diff]` | 0.1 | 0.1 | 0.5 | 731 |
| `btc_TxCnt[diff]` | 0.2 | 0.1 | 0.0 | 731 |
| `eth_bridge_deposit_count[diff]` | 0.2 | 0.0 | 0.1 | 731 |
| `bnb_tx_count[diff]` | 0.5 | 0.0 | 0.0 | 731 |

### 2SLS vs OLS (full sample + walk-forward OOS)

*No congestion instrument cleared the F ≥ 10.0 relevance bar without a tautology flag — no instrument passes the gate, so no causal estimate is licensed.*

#### Illustrative 2SLS on the most-relevant (sub-bar) instruments

**Below the F ≥ 10 weak-instrument bar — NOT a causal claim.** Shown only to see what instrumenting *would* do to the effect. Weak-instrument 2SLS is biased toward OLS and has inflated variance; large |2SLS β| with tiny F is the signature of an unidentified estimate, not a finding.

| Instrument | OLS β (t) | 2SLS β (t) | full F | OLS OOS R² | 2SLS OOS R² | OLS β̄ | 2SLS β̄ | sign agree | mean win F |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `eth_new_users[diff]` | +0.0003 (0.4) | +0.0086 (1.0) | 11.7 | -0.0103 | -17.7825 | +0.0006 | +0.0876 | 48% | 0.9 |
| `sol_tx_count[diff]` | +0.0003 (0.4) | +0.0003 (0.0) | 2.6 | -0.0103 | -326.3655 | +0.0006 | -0.2421 | 41% | 1.7 |
| `sol_total_fees_sol[diff]` | +0.0003 (0.4) | -0.0272 (0.3) | 0.2 | -0.0103 | -5.9889 | +0.0006 | -0.0225 | 59% | 3.7 |

### Verdict — Task 1

**Unidentified.** No instrument with an exclusion story is relevant enough (F ≥ bar) to congestion on the train window, so the congestion→BTC edge cannot be tested causally here. It remains *predictive only* — DML/placebo evidence stands, but no IV corroborates a structural interpretation.

---

## Task 2 — congestion innovation(t) → funding(t+1)

Treatment = chain_congestion innovation at t. Outcome = cross-asset mean funding_rate_8h at t+1. Funding starts 2023-11, so this sample is ~2 years and the walk-forward is short.

### Candidate screen (train window)

| Candidate instrument | train F | \|corr(Z,T)\| | direct-path t | n | flags | exclusion story |
|---|---:|---:|---:|---:|---|---|
| `eth_blob_size_mib[diff]` | 6.0 | 0.18 | 2.85 | 657 | DIRECT-PATH |t|=2.9 | L2 data-availability demand drives ETH gas; rollup posting schedule, arguably orthogonal to BTC return (post-Dencun 2024-03 only) |
| `eth_new_users[diff]` | 2.8 | 0.13 | 0.13 | 792 | clean | new-address onboarding = organic on-chain demand -> congestion |
| `eth_blob_fees[diff]` | 1.8 | 0.10 | 0.74 | 657 | clean | blob-fee market = DA demand pressure on blockspace (post-Dencun only) |
| `sol_total_fees_sol[diff]` | 1.8 | 0.10 | 0.27 | 792 | clean | Solana chain fees: cross-chain congestion proxy |
| `sol_priority_fees[diff]` | 1.3 | 0.08 | 0.54 | 792 | clean | Solana priority-fee spend: cross-chain blockspace contention |
| `sol_tx_count[diff]` | 0.5 | 0.05 | 0.49 | 792 | clean | Solana transaction count: cross-chain demand proxy |
| `eth_TxCnt[diff]` | 0.4 | 0.05 | 0.22 | 792 | clean | raw ETH transaction demand drives fees; not a direct claim on BTC |
| `avax_total_fees_avax[diff]` | 0.2 | 0.04 | 0.90 | 792 | clean | Avalanche chain fees: cross-chain congestion proxy |
| `eth_chain_nft_trading_volume[diff]` | 0.2 | 0.03 | 0.68 | 792 | clean | NFT mints/trades are a non-financial blockspace consumer that congests the chain without being a BTC price bet |
| `bnb_tx_count[diff]` | 0.1 | 0.03 | 1.03 | 792 | clean | BNB transaction count: cross-chain demand proxy |
| `btc_TxCnt[diff]` | 0.1 | 0.02 | 0.40 | 792 | clean | BTC transaction count: chain-demand cycle proxy (different chain's congestion, excludable from ETH-fee residual) |
| `eth_bridge_deposit_count[diff]` | 0.0 | 0.02 | 0.50 | 792 | clean | bridge deposit count: cross-chain activity consumes ETH blockspace |
| `bnb_total_fees_bnb[diff]` | 0.0 | 0.01 | 1.17 | 792 | clean | BNB chain fees: cross-chain congestion co-moves with the demand cycle, does not touch ETH-fee congestion's residual or BTC price directly |

### First-stage relevance across regimes (diagnostic)

| Instrument | train-window F | full-sample F | post-2024 F | n post-2024 |
|---|---:|---:|---:|---:|
| `eth_blob_size_mib[diff]` | 6.0 | 4.8 | 4.8 | 657 |
| `eth_blob_fees[diff]` | 1.8 | 2.1 | 2.1 | 657 |
| `sol_tx_count[diff]` | 0.5 | 1.0 | 0.5 | 730 |
| `avax_total_fees_avax[diff]` | 0.2 | 0.8 | 0.0 | 730 |
| `sol_total_fees_sol[diff]` | 1.8 | 0.6 | 0.6 | 730 |
| `eth_new_users[diff]` | 2.8 | 0.6 | 0.0 | 730 |
| `eth_TxCnt[diff]` | 0.4 | 0.5 | 0.4 | 730 |
| `eth_bridge_deposit_count[diff]` | 0.0 | 0.4 | 0.0 | 730 |
| `sol_priority_fees[diff]` | 1.3 | 0.4 | 0.5 | 730 |
| `bnb_tx_count[diff]` | 0.1 | 0.2 | 0.0 | 730 |
| `eth_chain_nft_trading_volume[diff]` | 0.2 | 0.1 | 0.0 | 730 |
| `btc_TxCnt[diff]` | 0.1 | 0.0 | 0.0 | 730 |
| `bnb_total_fees_bnb[diff]` | 0.0 | 0.0 | 0.2 | 730 |

### 2SLS vs OLS (full sample + walk-forward OOS)

*No congestion instrument cleared the relevance bar for the funding outcome — no instrument passes the gate, so no causal estimate is licensed.*

#### Illustrative 2SLS on the most-relevant (sub-bar) instruments

**Below the F ≥ 10 bar — NOT a causal claim** (see Task 1 note). Shown only to display the instrumented effect.

| Instrument | OLS β (t) | 2SLS β (t) | full F | OLS OOS R² | 2SLS OOS R² | OLS β̄ | 2SLS β̄ | sign agree | mean win F |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `eth_new_users[diff]` | +0.0000 (14.4) | +0.0000 (0.0) | 0.6 | -0.0287 | -2066.6674 | +0.0001 | -0.0004 | 74% | 0.6 |
| `eth_blob_fees[diff]` | +0.0001 (11.5) | +0.0001 (1.0) | 2.1 | +0.0989 | -108.1033 | +0.0001 | +0.0003 | 43% | 2.2 |
| `sol_total_fees_sol[diff]` | +0.0000 (14.4) | +0.0000 (0.4) | 0.6 | -0.0287 | -803.3852 | +0.0001 | -0.0002 | 74% | 3.6 |

### Verdict — Task 2

**Unidentified.** No exclusion-story instrument is relevant enough to congestion on the funding sample's train window, so the congestion→funding edge cannot be tested causally. It remains *predictive only* (target_search +0.023 OOS R²).

---

## Caveats (apply to both)

- **Exclusion is untestable.** A clean direct-path t only means Z does not *visibly* predict the outcome beyond T on the train window; it cannot prove Z affects the outcome solely through T.
- **The congestion factor here is fee-based.** `avg_gas_price_gwei` is absent from this snapshot, so chain_congestion falls back to z(Σ FeeTotNtv) over {btc, doge, eth}. That SUM includes `btc_FeeTotNtv`, which co-moves with BTC activity — a built-in exclusion hazard for the congestion→BTC test specifically.
- **One cycle, sample-mined hypotheses.** Both edges were selected on this very sample; the placebo tests that anointed them do not price in that selection. Task 2's funding window is short (~2y).
- **2SLS for prediction throws away first-stage variance.** As `iv_findings.md` notes, even a valid strong instrument typically *lowers* OOS fit because it discards the endogenous component that still predicts. A 2SLS OOS R² below OLS is therefore NOT by itself evidence against causality; the diagnostic that matters is whether the *effect estimate* (sign/size) survives instrumenting.
