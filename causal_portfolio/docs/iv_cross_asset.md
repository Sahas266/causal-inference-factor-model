# Cross-asset instrumental variables

**Question.** The within-asset warehouse IV search (`iv_findings.md`) found strong instruments make 2SLS WORSE out of sample, but it never used an economic exclusion argument. Here we test the *cross-asset* exclusion story: a shock on asset **X**'s chain plausibly hits asset **Y**'s return ONLY through a shared GLOBAL factor (aggregate congestion / liquidity / risk), never through Y's own chain-local return shock. That is the exclusion the within-asset search lacked.

- Assets: `btc, eth, sol, bnb, avax, xrp, doge, link, uni, aave, crv`
- Window: `2022-01-01` → `2025-12-31` | train 252d, rebalance 5d, F-gate 10.0
- Run UTC: `2026-06-17T17:45:03+00:00`

**Exclusion test = the cross-asset direct path.** For each triple (Z on chain X, treatment T, target asset Y) we regress Y's next-day return on `[T, Z]` over the train window. If Z's coefficient is significant (|t| ≥ 2), Z reaches Y by a path that bypasses T — the cross-asset exclusion is violated. |t| < 2 is *consistent with* (never proof of) exclusion.

## Cross-asset candidates (train-window screen)

| Treatment ← Z (chain X) | Target Y | train F | |corr(Z,T)| | direct-path t on Y | vs mean-ret t | stability | flags |
|---|---|---:|---:|---:|---:|---:|---|
| cex_dex_flow ← `sol_dex_volume_usd[diff]` | btc | 17.7 | 0.26 | 0.11 | 0.27 | 1/3 | clean |
| cex_dex_flow ← `bnb_dex_volume_usd[diff]` | eth | 15.5 | 0.24 | 0.56 | 0.13 | 1/3 | clean |
| cex_dex_flow ← `sol_total_fees_usd[spike]` | eth | 0.0 | 0.00 | 0.00 | 0.00 | 0/3 | WEAK F=0.0; UNSTABLE (relevance dies after train) |
| chain_congestion ← `eth_AdrActCnt[diff]` | btc | 0.4 | 0.04 | 0.41 | 0.51 | 0/3 | WEAK F=0.4; UNSTABLE (relevance dies after train) |
| chain_congestion ← `eth_txns[diff]` | btc | 0.2 | 0.03 | 1.16 | 1.07 | 0/3 | WEAK F=0.2; UNSTABLE (relevance dies after train) |
| chain_congestion ← `eth_avg_gas_utilization[diff]` | btc | 0.0 | 0.00 | 2.01 | 1.63 | 0/3 | WEAK F=0.0; CROSS-DIRECT-PATH |t|=2.0 on btc; UNSTABLE (relevance dies after train) |
| liq_flow ← `crv_tvl_usd[diff]` | sol | 6.9 | 0.16 | 0.26 | 0.17 | 0/3 | WEAK F=6.9; NAME-TAUTOLOGY (source is a factor input); UNSTABLE (relevance dies after train) |
| liq_flow ← `uni_tvl_usd[diff]` | btc | 2.3 | 0.10 | 0.90 | 1.75 | 0/3 | WEAK F=2.3; NAME-TAUTOLOGY (source is a factor input); UNSTABLE (relevance dies after train) |
| liq_flow ← `aave_tvl_usd[diff]` | eth | 1.1 | 0.07 | 1.15 | 1.24 | 0/3 | WEAK F=1.1; NAME-TAUTOLOGY (source is a factor input); UNSTABLE (relevance dies after train) |
| stable_flow ← `sol_stablecoin_supply[diff]` | eth | 8.7 | 0.18 | 0.77 | 0.93 | 0/3 | WEAK F=8.7; UNSTABLE (relevance dies after train) |
| stable_flow ← `bnb_stablecoin_supply[diff]` | btc | 0.5 | 0.04 | 0.83 | 0.48 | 0/3 | WEAK F=0.5; UNSTABLE (relevance dies after train) |
| stable_flow ← `avax_stablecoin_supply[diff]` | btc | 0.1 | 0.02 | 0.22 | 0.12 | 0/3 | WEAK F=0.1; UNSTABLE (relevance dies after train) |

**2** of 12 cross-asset candidates clear the relevance bar (train F ≥ 10.0); **2** are fully clean (relevant, non-tautological, no cross-asset direct path, stable).

### Economic rationale per candidate

- **cex_dex_flow ← `sol_dex_volume_usd[diff]` → btc:** Solana DEX-volume surge as an on-chain trading-activity shock; reaches BTC only via aggregate CEX/DEX flow, not BTC's own flow.
- **cex_dex_flow ← `bnb_dex_volume_usd[diff]` → eth:** BNB-chain DEX volume shock; cross-chain to ETH return through the global exchange-flow factor.
- **cex_dex_flow ← `sol_total_fees_usd[spike]` → eth:** Solana fee spike = activity burst; cross-asset exclusion to ETH return via aggregate flow.
- **chain_congestion ← `eth_AdrActCnt[diff]` → btc:** ETH active-address surge congests the dominant smart-contract chain; hits BTC only via aggregate 'space is busy / risk-on' congestion, not BTC's own block space.
- **chain_congestion ← `eth_txns[diff]` → btc:** ETH transaction-count innovation as a blockspace-demand shock; no chain-local path into BTC price.
- **chain_congestion ← `eth_avg_gas_utilization[diff]` → btc:** ETH gas-utilization innovation (how full blocks are) — a clean congestion shock with no BTC-local channel.
- **liq_flow ← `crv_tvl_usd[diff]` → sol:** Curve TVL shock on Ethereum DeFi; cross-chain to SOL return, so any path should be the aggregate liquidity factor.
- **liq_flow ← `uni_tvl_usd[diff]` → btc:** Uniswap TVL shock -> aggregate on-chain liquidity -> BTC only through the global liquidity channel.
- **liq_flow ← `aave_tvl_usd[diff]` → eth:** AAVE TVL inflow as a DeFi-liquidity shock; should reach ETH return only through aggregate DeFi liquidity, not via ETH's own price shock.
- **stable_flow ← `sol_stablecoin_supply[diff]` → eth:** Solana-chain stablecoin supply shock; cross-chain on-ramp to ETH return via the global stable_flow channel.
- **stable_flow ← `bnb_stablecoin_supply[diff]` → btc:** BNB-chain stablecoin minting as a chain-local liquidity on-ramp; hits BTC only through aggregate stablecoin liquidity (stable_flow is built from issuer-level USDC/USDT/USDe supply, not chain-level).
- **stable_flow ← `avax_stablecoin_supply[diff]` → btc:** Avalanche-chain stablecoin supply innovation; chain-specific, so not the aggregate-treatment tautology.

## Gated 2SLS vs OLS — out-of-sample A/B

One run per RELEVANT cross-asset instrument (others exogenous), reusing `ols_vs_2sls.run_ab` unchanged. Per-window first-stage partial F ≥ 10.0 gate; folds walk-forward vs BH BTC. Median Sharpe is the headline.

Median Sharpe is a 4-fold median; **full-window Sharpe** is the whole OOS series and is the honest tie-breaker — a fold median can drift up on a near-zero baseline while the full series is flat or worse.

| Treatment ← Z | Target | Gate | mean F | OLS medSh | 2SLS medSh | OLS fullSh | 2SLS fullSh | wr (both) | OOS days |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cex_dex_flow ← `sol_dex_volume_usd[diff]` | btc | 46/108 | 9.2 | -0.012 | 0.150 | 0.140 | 0.148 | 50% | 540 |
| cex_dex_flow ← `bnb_dex_volume_usd[diff]` | eth | 39/108 | 8.0 | -0.012 | 0.202 | 0.140 | 0.052 | 50% | 540 |

## Verdict

2 cross-asset instrument(s) engaged the gate and made 2SLS genuinely diverge from OLS (others stay exogenous). On the noisy 4-fold MEDIAN Sharpe, 2 of them look better than OLS. **But the full-window OOS Sharpe — the honest arbiter (margin ±0.05) — tells a different story: 0 materially improve the full series, 1 make it materially worse, and 1 are a wash.**

2 candidate(s) are a MEDIAN MIRAGE: the 4-fold median says 2SLS wins while the full-window Sharpe says it ties or loses. With only 4 folds over a baseline OLS Sharpe near zero, where the median falls is mostly luck, and 2SLS's per-fold Sharpes are far more dispersed (one fold can swing to +1.7 and another to -2.0).

**Verdict: cross-asset exclusion buys nothing the within-asset search didn't.** No cross-asset instrument robustly improves the full OOS series, and the fold win-rate vs BH BTC is unchanged at ~50% — 2SLS beats buy-and-hold BTC no more often than OLS does, and neither comes near BH BTC's full-window Sharpe. This reproduces the within-asset finding under a genuine economic exclusion argument, consistent with the structural diagnosis: the DAG carries no unobserved confounding for 2SLS to purge, and a portfolio wants the best linear predictor, not the structural β.

### Caveats (honesty bar)

- **Cross-asset exclusion is an assumption, not a proof.** A common risk shock (a macro/liquidity event that simultaneously spikes one chain's activity AND moves the target asset directly) violates it: Z would then have a path to Y outside the aggregate treatment. The direct-path t is only a weak, linear, train-window check of that — it cannot rule out a nonlinear or regime-specific common shock.
- **One cycle.** The sample is a single 2022–2025 crypto cycle; relevance and exclusion could both look different in another regime.
- **Multiple testing.** A dozen hand-picked triples across four families is small, but combined with the prior warehouse search the family-wise error budget is already spent — treat any single apparent win with suspicion.
- **Prediction ≠ identification.** Even a perfectly valid instrument throws away the endogenous first-stage variance that *predicts* returns; for a portfolio that is a cost, not a benefit. This is structural, not a data artifact.
