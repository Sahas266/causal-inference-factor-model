# Repository Guidelines

## Project Structure & Module Organization

This monorepo contains four distinct components:

- `backfill_data/`: production-oriented Python ingestion code in `src/`, endpoint definitions in `config/endpoints/`, scripts in `scripts/`, and tests in `tests/`.
- `causal_model/`: Rust CPCM engine organized as workspace crates under `crates/cpcm-*`; the CLI is in `crates/cpcm-cli/src/main.rs`.
- `causal_portfolio/`: Python research, backtesting, and Streamlit code. Primary entry points are `main.py`, `run_backtest.py`, and `dashboard.py`; tests are in `tests/`.
- `defi_pipeline/`: partial FastAPI scaffold with application code in `app/`, scripts in `scripts/`, and tests in `tests/`.

Keep changes within the relevant component. Treat `defi_pipeline/` as incomplete, and verify dependencies before running `causal_portfolio/`.

## Build, Test, and Development Commands

Use `venv\Scripts\python.exe` on Windows when the checked-in environment has the required packages.

- `python backfill_data/backfill.py --list-providers`: list committed ingestion providers.
- `cd backfill_data; python -m pytest`: run ingestion tests.
- `cd causal_model; cargo test -q`: test the Rust workspace.
- `python -m causal_portfolio.run_backtest --solver v1 --m 3 --assets btc,eth,sol`: run a focused portfolio backtest.
- `streamlit run causal_portfolio/dashboard.py`: launch the dashboard.
- `cd defi_pipeline; uvicorn app.main:app --reload`: start the API scaffold.
- `make start`, `make logs`, and `make stop`: manage the Docker stack.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for Python functions/modules, and `PascalCase` for classes. Add type hints to public Python interfaces. Format `defi_pipeline` Python with Black (100-character lines) and keep imports grouped. For Rust, follow standard `rustfmt` output and idiomatic `snake_case`; run `cargo fmt --check` before review. Name Python tests `test_*.py` and place them beside the component they cover.

## Testing Guidelines

Python uses pytest; Rust uses Cargo tests. Add focused regression tests for behavior changes. In `backfill_data`, apply existing markers such as `unit`, `integration`, `slow`, `requires_api_key`, and `requires_database`. Coverage is available with `pytest --cov=src --cov-report=html`; no repository-wide threshold is enforced. Tests requiring live APIs or Supabase must be explicitly identified.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects describing one change. Include affected modules and exact test commands in the PR description. Explain schema, environment-variable, provider, or Supabase assumptions; link relevant issues and include screenshots for dashboard or API UI changes.

## Security & Configuration

Never commit credentials. Copy `.env.example` for documented settings and keep live values in local `.env` files. The operational ingestion environment is `backfill_data/.env`.
