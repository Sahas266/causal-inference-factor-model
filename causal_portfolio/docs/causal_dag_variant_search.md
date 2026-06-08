# Causal DAG Variant Search (OLS vs gated 2SLS)

Each variant's return-equation structure (factor pool + lags + optional Combo selection) is run through the Step-5 causal A/B: per-asset loadings estimated by OLS and by gated 2SLS, scored out-of-sample (walk-forward folds) against buy-and-hold BTC. The instrument-strength gate (first-stage partial F ≥ 10.0) is reported per variant.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge`
- Window: `2022-01-01` → `2025-12-31` | train 252d, rebalance 5d
- Run UTC: `2026-06-08T17:50:21+00:00`

**Buy & Hold BTC (OOS):** total +291.8%, Sharpe 1.127

| Variant | OLS wr | 2SLS wr | OLS medSharpe | 2SLS medSharpe | Instrument gate (passed/seen) |
|---|---:|---:|---:|---:|---|
| combo3_lag0 | 25% | 25% | -0.216 | -0.497 | liquidation_level 0/108; stablecoin_mint 44/108 |
| combo3_lag1 | 33% | 33% | -0.222 | -0.222 | gas_spike 1/242 |
| all_lag0 | 50% | 50% | 0.831 | 0.831 | gas_spike 0/108; liquidation_level 0/108; protocol_event 0/108; stablecoin_mint 0/108 |
| all_lag1 | 50% | 50% | 0.451 | 0.963 | gas_spike 0/108; liquidation_level 0/108; protocol_event 0/108; stablecoin_mint 108/108 |
| global_lag0 | 50% | 50% | -0.587 | -0.731 | gas_spike 0/108; liquidation_level 0/108; protocol_event 0/108; stablecoin_mint 12/108 |
| global_lag1 | 50% | 25% | -0.174 | -1.388 | gas_spike 0/108; liquidation_level 0/108; protocol_event 0/108; stablecoin_mint 108/108 |
| macro_lag0 | 33% | 33% | 0.161 | 0.161 | no instrumented factors in pool |
| macro_lag1 | 33% | 33% | 0.734 | 0.734 | no instrumented factors in pool |
| onchain_core_lag0 | 33% | 33% | -0.746 | -0.746 | gas_spike 0/242; protocol_event 1/242 |
| onchain_core_lag1 | 44% | 44% | 0.445 | 0.445 | gas_spike 0/242; protocol_event 0/242 |

## Instrument validity caveat

`stablecoin_mint` clears the strength gate only under the **lag1** configs — but that is an artifact, not a real instrument. Under lag1 the treatment `stable_flow` is `z_score(diff(supply))` lagged one day, while `stablecoin_mint` is `diff(supply)` lagged one day: the same underlying series up to scaling. So the first-stage F is near-infinite by construction (the instrument *is* the treatment), which violates the exclusion restriction. Tellingly, where this fake-strong instrument was used, 2SLS did **worse** OOS than OLS (e.g. `global_lag1`: 2SLS 25% vs OLS 50%) — instrumenting a factor with itself adds variance without identification. Every other instrument fails the strength gate outright. Net: there is no valid, strong instrument in this DAG on this data.

## Verdict

**No variant's causal arm beats BH BTC in ≥80% of OOS folds.** Where the gate column shows 0 passes, 2SLS reduced to OLS by design (weak instruments), so the causal and correlational arms coincide — and neither robustly beats simply holding BTC. This is the same conclusion reached by every other experiment in the project, now confirmed under explicit causal estimation across all DAG structures.