# causal_model/ — Rust CPCM engine (REFERENCE ONLY, not used at runtime)

> **Status: reference implementation. Nothing in the live pipeline calls this.**
>
> All results, dashboards, backtests, regime analysis, the DAG-variant search,
> and the Hyperliquid execution layer run on the **Python** package in
> `causal_portfolio/`. The Python code never invokes this Rust engine — verified:
> there are no `subprocess`/`cargo`/`cpcm` binary calls from any `.py` file
> (only docstrings noting that some Python modules "port" Rust logic).

## Why it's kept

This crate set is the **authoritative reference** for the causal-estimation
machinery the Python pipeline is porting:

- `crates/cpcm-estimate/` — `ols.rs`, `tsls.rs` (2SLS), `diagnostics.rs`
  (first-stage F, Hausman, Sargan). Tested.
- `crates/cpcm-core/` — `dag.rs`, `cpcm_dag.rs`, `dsep.rs` (Bayes-Ball
  d-separation), `identify.rs` (backdoor identification). Tested.
- `crates/cpcm-factors/` — `global_factors.rs`, `instruments.rs`
  (instrument time-series construction). Tested.

When porting these to Python (`causal_portfolio/`), validate the Python
outputs against the Rust unit-test fixtures so the port is provably correct.

## What it does NOT do

The Rust pipeline (`cpcm-cli`) **only estimates causal effects** — it
identifies effects and runs panel OLS/2SLS, then exports `EstimationResult`.
It has **no backtest, no portfolio optimizer, no walk-forward evaluation, no
Sharpe** — none of the machinery that turns causal estimates into a tested
return-prediction strategy. That all lives in Python.

## Running it (only if you want the reference outputs)

```bash
cd causal_model
cargo test -q     # run the reference unit tests (the port targets)
cargo run --bin cpcm -- graph   # DAG summary; `run` needs Supabase creds
```

## Do not delete

Keep this until the Python 2SLS/diagnostics/identification port is complete
and validated against these fixtures. After that, deletion can be
reconsidered — but it remains the only tested reference for the causal math.
