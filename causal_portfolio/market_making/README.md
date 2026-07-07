# Market-Making Engine

This package implements the Avellaneda-Stoikov (AS) and Easley PIN research plan without modifying the portfolio rebalancer. It is safe by default: the live runner starts on testnet in dry-run mode, submits only post-only `Alo` orders, and cancels only strategy-owned CLOIDs. Benchmark harnesses in this package may deliberately use IOC taker orders for comparison arms.

## Components

- `avellaneda_stoikov.py`: finite-horizon approximation, stationary reservation prices, and a bounded-inventory numerical HJB benchmark.
- `pin.py`: stable symmetric-uninformed-intensity three-state Poisson-mixture likelihood, multi-start MLE, posteriors, and rolling estimates.
- `calibration.py` / `toxicity.py`: EWMA volatility, exponential fill intensity, realized-markout curves, and four quote policies.
- `simulator.py`: deterministic event replay with latency, conservative queue position, partial fills, maker fees, funding, and 1/5/30-second markouts.
- `engine.py`: inventory, stale-book, margin, loss, volatility, spread, and toxicity gates.
- `hyperliquid.py`: WebSockets, `Alo` placement, targeted cancel-by-CLOID, persisted ownership, and an opt-in dead-man switch.
- `recorder.py`: deduplicated raw `trades`, `l2Book`, and `bbo` capture in DuckDB.
- `strategy_execution_benchmark.py` / `testnet_execution_benchmark.py`: identical-signal IOC versus target-relative AS+PIN execution bounds and minimal-size testnet trials.

## Data Capture

Run from the repository root with the checked-in environment:

```powershell
venv\Scripts\python.exe -m causal_portfolio.market_making.record `
  --coins BTC,ETH --output data/market_making/hyperliquid.duckdb
```

The recorder is read-only. Use `MicrostructureStore.coverage()` to inspect channel coverage and `replay_events(coin="BTC")` for deterministic simulation input. Treat PIN as research-only until at least 60 complete UTC days pass validation.

Create a versioned calibration artifact after the coverage gate passes:

```powershell
venv\Scripts\python.exe -m causal_portfolio.market_making.calibrate `
  --database data/market_making/hyperliquid.duckdb --coin BTC --window 60 `
  --markouts-csv data/market_making/btc_markouts.csv `
  --fill-observations-csv data/market_making/btc_quote_exposures.csv `
  --output data/market_making/btc_calibration.json
```

Markout CSV columns are `pin,aggressor,adverse_bps,horizon_ms`; exposure columns are `distance,exposure_seconds,fills,aggressor`. Calibration rejects partial days or L2 gaps above 60 seconds.

## Shadow and Testnet

All parameters use explicit units: `sigma` is absolute price volatility per square-root second; inventory is base units; AS internally converts inventory to order-size units.

```powershell
venv\Scripts\python.exe -m causal_portfolio.market_making.run `
  --coin BTC --order-size 0.001 --tick-size 1 --size-decimals 5 `
  --gamma 0.0001 --kappa 0.02 --horizon-seconds 300 `
  --sigma 10 --max-inventory 0.005 --duration-seconds 300
```

This command only logs intended testnet quotes. Add `--live` for testnet submission after calibrating parameters. Values above demonstrate CLI shape; they are not trading recommendations.

Select `as_pin_size`, `as_pin_spread`, or `as_pin_posterior` with `--policy` and `--calibration`. Spread policies refuse to start without realized-markout calibration; `--use-calibrated-intensity` replaces symmetric `A` and `kappa` with the average of separately fitted buy/sell intensities.

Compare every feasible policy on the same post-calibration event window:

```powershell
venv\Scripts\python.exe -m causal_portfolio.market_making.backtest `
  --database data/market_making/hyperliquid.duckdb `
  --calibration data/market_making/btc_calibration.json --coin BTC `
  --order-size 0.001 --tick-size 1 --size-decimals 5 `
  --gamma 0.0001 --kappa 0.02 --sigma 10 --max-inventory 0.005
```

Replay defaults to the first event at or after calibration `as_of` and rejects look-ahead. Results include spread capture, inventory PnL, fees, funding, fill rate, quote uptime, cancel rate, drawdown, tail inventory, and markouts.

Live mainnet additionally requires `--mainnet --live --allow-mainnet --acknowledge-mainnet --dedicated-account`. The dedicated-account gate is mandatory because Hyperliquid's scheduled-cancel dead-man switch affects every order on that account.

## Verification

```powershell
$tests = Get-ChildItem causal_portfolio\tests\test_market_making_*.py
venv\Scripts\python.exe -m pytest $tests -q
```

Mainnet remains an operational promotion, not a code toggle: review walk-forward replay, shadow, testnet reconnect, CLOID reconciliation, and loss-limit reports first.
