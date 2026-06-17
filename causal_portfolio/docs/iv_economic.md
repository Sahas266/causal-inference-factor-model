# Economically-motivated IV search — exclusion-first

Unlike the purely statistical warehouse sweep (`iv_search.md`), every candidate here was chosen FIRST for a plausible **exclusion story** (a supply-side / scheduled / mechanical driver of the treatment with no obvious second channel to returns), then tested for relevance and the same falsifiers. See the module docstring for the per-instrument economic rationale.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | train 252d | relevance bar F ≥ 10.0
- Run UTC: `2026-06-17T17:40:45+00:00`

**Exclusion is untestable in general.** The direct-path probe (does Z predict returns beyond T?) is a *necessary* falsifier, not a proof of exclusion. A passed probe + a clean economic story is the best we can do; treat every 'clean' below as *not yet falsified*, not *validated*.

## Candidate screen

Selection stats on the train window only. `corr(Z,T)` is the tautology check (>0.9 ⇒ Z *is* T); `direct-path t` is the exclusion red flag (>2 ⇒ Z predicts returns past T); `stability` = later non-overlapping windows with F ≥ bar.

| # | Treatment | Candidate | train F | stab | |corr(Z,T)| | direct-path t | flags |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | chain_congestion | `btc_HashRate[diff]` | 0.4 | 0/3 | 0.04 | 0.93 | WEAK (train F < bar) |
| 2 | chain_congestion | `btc_difficulty[diff]` | 0.0 | 0/3 | 0.00 | 0.02 | WEAK (train F < bar) |
| 3 | chain_congestion | `btc_HashRate[spike]` | 0.0 | 0/3 | 0.00 | 0.00 | WEAK (train F < bar) |
| 4 | liq_flow | `btc_HashRate[diff]` | 0.0 | 0/3 | 0.00 | 0.91 | WEAK (train F < bar) |
| 5 | staking_yield | `eth_gross_emissions[diff]` | 0.0 | 0/3 | 0.00 | 0.85 | WEAK (train F < bar) |
| 6 | staking_yield | `eth_IssTotNtv[diff]` | 0.0 | 0/3 | 0.00 | 0.36 | WEAK (train F < bar) |
| 7 | staking_yield | `eth_consensus_rewards_eth[diff]` | 0.0 | 2/3 | 0.00 | 0.00 | WEAK (train F < bar) |
| 8 | staking_yield | `eth_queue_exit_amount[diff]` | 80.1 | 0/3 | 0.49 | 0.27 | UNSTABLE (relevance dies after train) |
| 9 | staking_yield | `eth_queue_entry_amount[diff]` | 2.2 | 0/3 | 0.09 | 1.09 | WEAK (train F < bar) |
| 10 | stable_flow | `eth_etf_aum_native[diff]` | 23.0 | 1/1 | 0.29 | 0.25 | clean |
| 11 | funding_basis | `eth_etf_aum_native[diff]` | 34.0 | 1/1 | 0.35 | 0.27 | clean |
| 12 | funding_basis | `btc_etf_flows_native[diff]` | 0.2 | 0/2 | 0.02 | 0.57 | WEAK (train F < bar) |
| 13 | funding_basis | `eth_etf_aum_native[spike]` | 19.7 | 0/1 | 0.27 | 0.04 | UNSTABLE (relevance dies after train) |
| 14 | stable_flow | `eth_stablecoin_minted_usd[diff]` | 0.0 | 0/3 | 0.00 | 0.00 | WEAK (train F < bar) |
| 15 | stable_flow | `eth_stablecoin_burned_usd[diff]` | 0.0 | 0/3 | 0.00 | 0.00 | WEAK (train F < bar) |
| 16 | stable_flow | `bnb_stablecoin_supply[diff]` | 0.5 | 0/3 | 0.04 | 0.57 | WEAK (train F < bar) |
| 17 | funding_basis | `macro_vixcls[diff]` | 0.2 | 0/2 | 0.03 | 1.45 | WEAK (train F < bar) |
| 18 | funding_basis | `macro_vixcls[spike]` | 0.0 | 0/2 | 0.00 | 0.00 | WEAK (train F < bar) |
| 19 | funding_basis | `macro_bamlh0a0hym2[diff]` | 1.1 | 0/2 | 0.07 | 0.54 | WEAK (train F < bar) |
| 20 | funding_basis | `macro_dgs2[diff]` | 0.9 | 0/2 | 0.06 | 0.61 | WEAK (train F < bar) |

**4/20 candidates clear the relevance bar** (F ≥ 10.0, non-tautological); **2 are fully clean** (also no direct-path / stability flag).

### Exclusion-story read per category

- **Mining/security (hashrate, difficulty → congestion/liq):** clean economic story, but first-stage relevance is ~0 — daily hashrate/difficulty innovations do not move the gas/TVL-based treatments at the train window. Irrelevant ⇒ unusable.
- **Emission/issuance (gross_emissions, issuance, consensus rewards → staking_yield):** cleanest exclusion (pure schedule), but F ≈ 0 — the smooth staking-APR level barely responds to daily issuance innovations. The validator **queue** series move APR strongly (F up to ~88) but the queue also signals demand (weaker exclusion) and relevance is **unstable** out of window.
- **ETF native flows → stable_flow / funding_basis:** the only instruments with a genuine exclusion story that ALSO clear relevance (F≈23–34, direct-path t≈0.25, non-tautological). Short history (2024+, ~525 rows).
- **Stablecoin mint/burn/other-chain → stable_flow:** tautology successfully avoided (corr≈0), but the issuer-side and single-chain series are **irrelevant** to aggregate stable flow at the train window (F ≈ 0).
- **Macro surprise (VIX, HY spread, 2y → funding_basis):** clean-ish story but exclusion-fragile (risk-off hits all crypto directly), and empirically **irrelevant** here (F < 2).

### Instruments carried to the A/B (best per treatment)

- **staking_yield** ← `eth_queue_exit_amount[diff]`: validator EXIT queue drains at a fixed per-epoch churn limit (mechanical); more exits mechanically raise APR for those remaining (F=80, stab 0/3, direct-t=0.27) [FLAGS: UNSTABLE (relevance dies after train)]
- **stable_flow** ← `eth_etf_aum_native[diff]`: ETF AUM in coins = TradFi positioning; AP creation sources coins via stablecoin rails, so it should hit price via on-ramp stable flow (F=23, stab 1/1, direct-t=0.25)
- **funding_basis** ← `eth_etf_aum_native[diff]`: ETF positioning shock; APs hedge in perps, transmitting only via funding (F=34, stab 1/1, direct-t=0.27)

## Gated 2SLS A/B (economically-motivated shortlist)

One run per instrument (all other factors stay exogenous), so short-history candidates keep their full sample. The per-window first-stage partial-F gate (≥ 10.0) decides, window by window, whether the factor is actually instrumented; where it fails, 2SLS = OLS. Win-rates are walk-forward folds vs BH BTC.

| Treatment | Instrument | Gate (passed/seen) | mean F | OLS wr | 2SLS wr | OLS medSh | 2SLS medSh | OOS days |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| staking_yield | `eth_queue_exit_amount[diff]` | 0/108 | 0.8 | 50% | 50% | 0.831 | 0.831 | 540 |
| stable_flow | `eth_etf_aum_native[diff]` | 0/55 | 3.9 | 50% | 50% | 1.236 | 1.236 | 273 |
| funding_basis | `eth_etf_aum_native[diff]` | 0/55 | 3.4 | 50% | 50% | 1.236 | 1.236 | 273 |

## Verdict

Every shortlist instrument **failed the per-window gate** — 2SLS collapsed to OLS in every window (identical median Sharpe and fold win-rate). Note the gate's mean partial F (~0.8–3.9) is far below the train-window univariate F (23–88): once the other exogenous factors are partialled out and the window rolls forward, the instruments' relevance evaporates. Same outcome as the original four declared instruments: nothing exploitable.

### Caveats

- **Exclusion is ultimately untestable.** The direct-path probe rejects only the most blatant second channels; a clean probe is necessary, not sufficient.
- **One cycle.** 2022–2025 is a single crypto macro regime; out-of-window stability (the `stab` column) is the closest we get to a second sample, and the relevant instruments fail it.
- **Multiple testing.** ~19 (treatment, instrument) pairs were screened; any single 'clean + helps' result needs a Bonferroni-style discount before it is believed.
- **Short history.** The ETF instruments only exist from 2024, so their A/B runs on a truncated, single-regime OOS window.
