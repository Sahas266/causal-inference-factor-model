# Project: Causal Inference Factor Model (CPCM)

This repository implements **Causal PDE-Control Models (CPCMs)** for DeFi portfolio optimization. It bridges causal inference, nonlinear filtering, and stochastic control to create alpha-generating signals from on-chain and market data, solving the problem of correlation breakdowns in crypto markets by modeling true causal mechanisms.

## System Architecture & Trust Model

The monorepo is divided into four main areas. **Important**: Always verify the readiness of local environments before assuming they run cleanly.

### 1. `backfill_data/` (Python)
The operational data ingestion system. **Highly mature and reliable.**
- **Purpose:** Fetches, normalizes, and stores data from providers like CoinMetrics, DeFi Llama, Allium, and Dune.
- **Storage:** Writes to a Supabase warehouse (PostgreSQL/TimescaleDB).
- **Architecture:** Plugin-based provider system, concurrent thread-pool backfilling, and checkpoint resumption.
- **Key Files:** `backfill.py` (orchestrator), `src/providers/` (adapters), `config/endpoints/` (configurations).

### 2. `causal_model/` (Rust)
The core causal graph and estimation engine. **Mature and tests cleanly.**
- **Purpose:** Implements the Structural Causal Model (SCM). Runs DAG generation, d-separation, Instrumental Variables (IV) validation, OLS, and 2SLS regressions.
- **Tech Stack:** Rust, Polars, Petgraph, Nalgebra.
- **Key Files:** `crates/cpcm-core/` (DAG logic), `crates/cpcm-data/` (Supabase/Parquet client), `crates/cpcm-cli/src/main.rs`.

### 3. `causal_portfolio/` (Python)
The research, backtesting, and visualization layer. **Substantive code, but local env may be incomplete.**
- **Purpose:** Implements DoWhy SCM logic, portfolio optimization solvers, and the Streamlit dashboard.
- **Tech Stack:** DoWhy/EconML, CVXPY, Streamlit.
- **Key Files:** `run_backtest.py`, `dashboard.py`, `scm/` (causal graph definitions).

### 4. `defi_pipeline/` (Python)
A FastAPI-based scaffold for serving real-time DeFi metrics. **Partial implementation/scaffold only.**
- **Purpose:** Serve real-time causal indicators (Liq Flow, Chain Congestion, etc.) via REST API.

---

## Causal Theory & Key Factors

We identify 7 global macroeconomic forces that drive crypto returns:
1. **LiqFlow**: Net LP inflows/outflows across AMMs.
2. **StableFlow**: Net minting/burning of major stablecoins.
3. **FundingBasis**: Perp funding premium vs. spot returns.
4. **ChainCongestion**: Gas fees and block utilization.
5. **StakingYield**: Real staking APR.
6. **MEVPressure**: MEV revenue volatility / base fee volatility.
7. **CEXDEXFlow**: Net exchange-to-chain transfer flows.

Our model uses **Instrumental Variables (IVs)** (like gas fee spikes, lagged supply changes) combined with **Two-Stage Least Squares (2SLS)** to isolate causal effects from noisy correlations and reverse causality.

---

## Environment & Data Truth

- **Source of Truth:** The live Supabase warehouse is the source of truth for data coverage (Project ID: `jnulpcqpftnwvknwuqpa`). Trust the `public.asset_metrics_best` table over local configurations.
- **Python Setup:** Use the checked-in virtualenv on Windows (`venv\Scripts\python.exe`). Note: Some packages may be missing for the portfolio stack.
- **Missing Data:** The committed repository currently lacks adapters for Artemis and Hyperliquid, which exist in the live database. Funding rate data relies on Hyperliquid.
- **Required Env Vars:** Use `.env.example` as a template. You will need `SUPABASE_KEY` and specific provider API keys to run ingestion. Keep keys out of version control.

---

## Key Commands

### Data Ingestion (`backfill_data/`)
```powershell
cd backfill_data
# List providers
python backfill.py --list-providers
# Validate endpoints
python backfill.py --config config/endpoints/ --validate-only
# Run tests
python -m pytest
```

### Causal Estimation (`causal_model/`)
```powershell
cd causal_model
# Run Rust tests
cargo test -q
# Run the full estimation pipeline
cargo run -p cpcm-cli -- run
```

### Research & Backtesting (`causal_portfolio/`)
```powershell
# Run portfolio backtest
python -m causal_portfolio.run_backtest --solver v1 --m 3 --assets btc,eth,sol
# Launch dashboard
streamlit run causal_portfolio/dashboard.py
```

### API Pipeline (`defi_pipeline/`)
```powershell
cd defi_pipeline
uvicorn app.main:app --reload
```

---

## Development & Contribution Guidelines

1. **Testing First:** Python uses `pytest` (use markers like `@pytest.mark.integration`). Rust uses `cargo test`. Ensure new functionality is backed by tests.
2. **Causal Integrity:** Changes to causal graphs (DAGs) and IV strategies must be justified theoretically in documentation before implementation.
3. **Commit Messages:** Keep titles concise, imperative, and scoped (e.g., `feat: add l2 activity factor`). Include test commands run and screenshot evidence for UI changes.
4. **Cross-Platform Caveats:** `uvloop` should remain conditional in `requirements.txt` to preserve Windows compatibility.
5. **Historical Context:** Previous uncommitted implementation work (e.g., Artemis, Hyperliquid) may be stored in local `.claude` session memories. Consult these for historical reconstruction if needed.