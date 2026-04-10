# Repository Guidelines

## Current Repo Reality

This monorepo has four main areas:

- `backfill_data/`: the operational data ingestion system and the most mature committed Python module.
- `causal_model/`: the Rust CPCM causal graph and estimation engine; this is real code and tests cleanly.
- `causal_portfolio/`: the Python CPCM research, backtest, and Streamlit layer; the code is substantive, but the checked-in Windows `venv` is incomplete for the full quant stack.
- `defi_pipeline/`: a FastAPI scaffold that is still partial and should not be treated like the other three modules.

If you need the safest working mental model:

- trust committed orchestration and provider code in `backfill_data/`
- trust `causal_model/` implementation and tests
- trust `causal_portfolio/` structure, but verify local env readiness before assuming it runs
- treat `defi_pipeline/` as planned / partial

## Live Warehouse Notes

Supabase is the live source of truth for data coverage.

- Project ID: `jnulpcqpftnwvknwuqpa`
- Verified via Supabase MCP on `2026-04-10`
- `public.asset_metrics`: `2,848,307` rows
- `public.asset_metrics_best`: `2,817,204` rows
- Providers in fact tables: `9`
- Assets: `29`
- Metrics: `338`
- Global date range: `2021-01-01` -> `2026-01-01`
- Database size after index cleanup: about `1,088 MB`

Important mismatch:

- the live warehouse includes large Artemis and Hyperliquid datasets
- the current committed worktree does not include Artemis or Hyperliquid provider source/config files
- `backfill_progress`, `provider_health`, and `recent_backfill_activity` only reflect the committed-provider operational metadata, not the full live provider set

Trust order for data questions:

1. Supabase fact tables for live coverage
2. committed repo code for what can be re-run locally
3. Claude session history and memory for historical reconstruction

Current committed providers clearly present:

- `allium`
- `coingecko`
- `coinmetrics`
- `defillama`
- `dune`
- `fred`

Current warehouse-only historical providers:

- `artemis`
- `hyperliquid`

Additional live assets beyond the 26-coin target universe:

- `macro`
- `matic`
- legacy `lido`

## Project Structure & Module Organization

- `backfill_data/`: core code in `src/`, endpoint configs in `config/endpoints/`, schema in `config/database_schema.sql`, helper scripts in `scripts/`, tests in `tests/`.
- `causal_model/`: Rust workspace with crates under `crates/cpcm-*`; CLI entrypoint is `crates/cpcm-cli/src/main.rs`.
- `causal_portfolio/`: Python CPCM layer; main entrypoints are `main.py`, `run_backtest.py`, and `dashboard.py`; tests live in `tests/`.
- `defi_pipeline/`: FastAPI scaffold with app code in `app/`, scripts in `scripts/`, and tests in `tests/`.

Runtime assumptions visible in code:

- `causal_portfolio/data/supabase_loader.py` reads from `asset_metrics_best`
- `causal_model` can read from `asset_metrics_best` or `asset_metrics`
- the operational env file on this machine is `backfill_data/.env`
- `.env.example` exists at repo root, but root `.env` may be absent

## Build, Test, and Development Commands

Use the repo virtualenv on Windows: `venv\Scripts\python.exe`.

- `python backfill_data/backfill.py --list-providers`: inspect committed backfill providers.
- `python backfill_data/backfill.py --config backfill_data/config/endpoints/btc/ --validate-only`: validate endpoint configs before running them.
- `cd backfill_data; python -m pytest`: run backfill tests.
- `cd causal_model; cargo test -q`: run Rust workspace tests.
- `python -m causal_portfolio.run_backtest --solver v1 --m 3 --assets btc,eth,sol`: run the portfolio backtest path.
- `streamlit run causal_portfolio/dashboard.py`: launch the dashboard locally.
- `cd defi_pipeline; uvicorn app.main:app --reload`: start the API scaffold.
- `make start|logs|stop`: manage the Dockerized pipeline/dashboard stack.

## Testing Guidelines

Pytest is the Python test runner; Rust uses `cargo test`.

- name Python tests `test_*.py`
- mirror the package or feature under test
- in `backfill_data`, use the existing markers: `unit`, `integration`, `slow`, `requires_api_key`, `requires_database`

Currently verified locally:

- `backfill_data/tests/providers/test_allium.py`
- `backfill_data/tests/providers/test_coinmetrics.py`
- `backfill_data/tests/providers/test_defillama.py`
- `cargo test -q` in `causal_model/`

Current local env caveat:

- `causal_portfolio` test collection is blocked in the checked-in Windows `venv` because packages such as `pandas`, `sklearn`, and `torch` are missing

## Known Mismatches & Cleanup Targets

These are still worth keeping in mind while working:

- `requirements.txt` still pins `uvloop==0.22.1` unconditionally even though docs correctly say it must be conditional on Windows
- `docker-compose.yml` still health-checks `http://localhost:8501/healthz`, but the Streamlit dashboard does not expose `/healthz`
- `funding_basis` is still treated as a gap in the current Python and Rust factor builders even though Hyperliquid funding data now exists in the warehouse
- `defi_pipeline/` docs and direction can overstate readiness relative to the committed code

Storage note:

- on `2026-04-10`, `public.idx_asset_metrics_metric_time` was dropped because it showed `0` scans and cost about `60 MB`
- `backfill_data/config/database_schema.sql` was updated to keep the repo in sync with that live schema change

## Historical Reconstruction Notes

Claude session history matters for this repo because some capabilities existed in Claude-local state without staying committed in git.

Most important Claude-era workstreams:

- Allium / FM-13 implementation
- 26-coin backfill architecture buildout
- POL / MATIC lineage work
- earlier Artemis / Hyperliquid provider work that no longer exists in the current tree

If Artemis or Hyperliquid need to be restored, the likely source order is:

1. local Claude file-history backups
2. local Claude project transcripts
3. current docs and notes

Useful local locations on this machine:

- `C:\Users\sahas\.claude\projects\c--Users-sahas-Github-Repos-causal-inference-factor-model\`
- `C:\Users\sahas\.claude\projects\c--Users-sahas-Github-Repos-causal-inference-factor-model\memory\MEMORY.md`
- `C:\Users\sahas\.claude\file-history\`

Interpret Claude memory carefully:

- it records operational truth and local session state
- it is not proof that every described file is still committed in git

## Commit & Pull Request Guidelines

Recent history favors short subjects such as `more backfill`, `causal model work`, and `ready for PR`.

- keep commit titles concise, imperative, and scoped to one change
- include affected modules
- summarize behavior or schema changes
- list exact test commands you ran
- include screenshots for Streamlit or API UI changes
- call out any new env vars, provider keys, or Supabase assumptions

## Security & Configuration Tips

Never commit live secrets.

- start from `.env.example`
- keep real keys in local `.env` files
- document any new required variables in the PR description
- prefer Supabase MCP for live warehouse inspection when available
- if Supabase MCP is unavailable, the repo fallback path is `backfill_data/.env` plus `backfill_data/src/core/storage/supabase_manager.py`
