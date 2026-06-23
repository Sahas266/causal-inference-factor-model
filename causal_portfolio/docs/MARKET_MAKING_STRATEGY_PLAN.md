# Market-Making and Informed-Flow Implementation Plan

## Implementation Status

The code path is implemented under `causal_portfolio/market_making/`: AS approximation and numerical HJB, PIN MLE/posteriors, calibration and toxicity policies, event replay, safety-gated quote state, Hyperliquid WebSockets/post-only execution, strategy-owned CLOID persistence, dead-man gating, and DuckDB capture. Focused tests live in `causal_portfolio/tests/test_market_making_*.py`.

Operational evidence is deliberately not marked complete. Sixty-day exchange-specific capture, walk-forward policy comparison, shadow reports, testnet fault injection, and controlled mainnet promotion require elapsed data and human review.

## Source Material

- Avellaneda and Stoikov (2006 working paper), *High-frequency trading in a limit order book*: original utility-maximization model, HJB, finite- and infinite-horizon reservation prices, fill-intensity assumptions, and finite-horizon approximation. The article was later published in 2008.
- Hummingbot (2021), *Guide to the Avellaneda & Stoikov Strategy*: operational interpretation using target inventory, configurable sessions, and spread-based parameter settings for crypto.
- Hummingbot/Medium (2021), *A comprehensive guide to Avellaneda & Stoikov's market-making strategy*: a reposted implementation summary.
- Easley, Kiefer, O'Hara, and Paperman (1996), *Liquidity, Information, and Infrequently Traded Stocks*: structural estimation of informed and uninformed trade arrivals and the probability of informed trading (PIN).

Local PDFs are organized under the ignored `references/` directory and are not required at runtime.

Treat the two practitioner guides as implementation context, not independent validation of the academic model.

## Strategy Interpretation

Avellaneda-Stoikov (AS) answers where to quote. Under symmetric exponential arrival intensity `lambda(delta) = A * exp(-kappa * delta)`, the paper obtains the following finite-horizon approximation for midprice `s`, inventory `q`, risk aversion `gamma`, volatility `sigma`, and remaining horizon `tau`:

```text
reservation_price = s - q * gamma * sigma^2 * tau
total_spread = gamma * sigma^2 * tau + (2 / gamma) * log(1 + gamma / kappa)
bid = reservation_price - total_spread / 2
ask = reservation_price + total_spread / 2
```

These controls follow an inventory expansion and a first-order linearization of the arrival term. They are not the exact solution of the HJB. `A` affects simulated fill frequency and PnL even though it drops out of the approximate quote formula.

The model also assumes an arithmetic Brownian midprice with no drift, exponential utility, one-unit fills, independent Poisson arrivals, costless continuous quote updates, terminal marking at the midprice, and no queue position, latency, fees, funding, or informed flow. Every production deviation should be explicit and tested.

The paper's infinite-horizon formulation is a distinct discounted-utility problem with parameter `omega` and nonlinear reservation bid/ask prices. Repeatedly resetting `tau` for a 24/7 market is a practical finite-horizon policy, not that infinite-horizon solution.

Easley answers when passive liquidity is exposed to informed flow. Daily aggressive-buy and aggressive-sell counts are modeled as a three-component Poisson mixture: no information event, good news, or bad news. With event probability `alpha`, informed arrival rate `mu`, and symmetric uninformed arrival rate `epsilon`:

```text
PIN = alpha * mu / (alpha * mu + 2 * epsilon)
```

PIN is not directional alpha by itself. It should widen quotes, reduce size, or suspend quoting when adverse-selection risk is high. The paper's good-news/bad-news posterior can later make the widening side-specific.

## Current Repository Fit

Reusable components already exist:

- `execution/orderbook.py`: typed L2 books, price bands, and fill parsing.
- `execution/hyperliquid.py`: account state, metadata, L2 snapshots, order submission, reconciliation, and audit logging.
- `backfill_data`: `market_trades` and `market_orderbooks` schemas plus CoinMetrics ingestion scaffolding.
- `backtest/costs.py`: fee and slippage primitives.

The existing execution path is a taker/rebalancing engine. It uses IOC orders and cancels all open orders before submission. A market maker requires persistent post-only quotes, targeted cancel/replace, WebSocket events, and queue-aware simulation. Keep it as a separate engine so portfolio rebalancing safety remains unchanged.

## Proposed Architecture

Create `causal_portfolio/market_making/`:

```text
types.py                 TradePrint, QuoteState, QuotePair, MMInventory
avellaneda_stoikov.py    Pure reservation-price and spread calculations
pin.py                   Stable EKOP likelihood, rolling MLE, diagnostics
toxicity.py              PIN and online good/bad-news posterior overlays
calibration.py           EWMA sigma, kappa/fill-intensity, markout estimates
simulator.py             Event-driven queue/fill/inventory simulator
engine.py                Quote state machine and safety gates
```

Add a dedicated `HLMarketMakerAdapter` rather than changing `execute_plan()`:

- subscribe to `trades`, `l2Book`, `orderUpdates`, and `userFills`;
- submit `Alo` post-only quotes with deterministic strategy CLOIDs;
- cancel only strategy-owned quotes using CLOID, never `cancel_all_open()`;
- refresh quotes on material price, inventory, volatility, or toxicity changes;
- use Hyperliquid's scheduled-cancel dead-man switch;
- preserve the current account/network/freshness/audit safety gates.

## Delivery Phases

### Phase 0 - Capture and Validate Microstructure Data

1. Build a read-only Hyperliquid WebSocket recorder for BTC and ETH first.
2. Persist trade prints, top-ten L2 snapshots, BBO, and timestamps to local Parquet or DuckDB. Reuse the warehouse schemas after the event contract stabilizes.
3. Verify whether Hyperliquid trade side `B/A` represents the aggressor side using simultaneous book changes and sample fills.
4. Measure gaps, duplicate hashes, clock skew, and reconnect behavior.
5. Collect at least 60 complete UTC days before treating rolling PIN as production quality. CoinMetrics history can prototype the estimator, but exchange-specific calibration must use Hyperliquid data.

Deliverable: a coverage report and deterministic replay files. Do not place orders in this phase.

### Phase 1 - Pure Models

Implement AS as a dependency-free pure function with these invariants:

- finite inputs and positive `gamma`, `sigma`, and `kappa`;
- explicit units for inventory, volatility, horizon, and risk aversion;
- `bid < reservation_price < ask`;
- inventory sign moves the reservation price in the liquidation direction;
- tick-size rounding never crosses the spread;
- configurable rolling finite horizon for 24/7 markets, clearly labeled as an implementation policy;
- separate estimates of `A` and `kappa` from quote-distance/fill observations, including side-specific diagnostics.

Add a discrete-inventory numerical HJB solver as a research benchmark. Compare the closed-form approximation with the numerical controls across volatility, horizon, inventory, `gamma`, and `kappa`; establish a domain in which the approximation is accepted. Keep the paper's stationary reservation-price model behind an experimental interface until its `omega` interpretation and quote controls are resolved.

Implement the four-parameter Easley model:

- aggregate aggressive buys and sells by UTC day;
- maximize the log-sum-exp form of the three-state Poisson likelihood;
- use logit transforms for `alpha` and `delta`, log transforms for `mu` and `epsilon`;
- use multi-start L-BFGS-B and retain convergence, Hessian, sample-size, and boundary diagnostics;
- estimate on trailing 60-day windows and expose PIN only when diagnostics pass;
- validate recovery on synthetic data generated from known parameters.

### Phase 2 - Combined Quote Policy

Start with four separately measurable policies:

1. `AS_BASELINE`: inventory and volatility only.
2. `AS_PIN_SPREAD`: add a symmetric adverse-selection premium calibrated from post-fill markouts by PIN bucket.
3. `AS_PIN_SIZE`: reduce displayed size as PIN rises.
4. `AS_PIN_POSTERIOR`: widen the ask when informed-buy probability rises and widen the bid when informed-sell probability rises.

Do not choose arbitrary PIN multipliers for the production policy. Estimate the premium from realized 1-second, 5-second, and 30-second markouts after hypothetical maker fills. A simple initial form is:

```text
bid_half_spread = as_half_spread + expected_sell_side_adverse_markout
ask_half_spread = as_half_spread + expected_buy_side_adverse_markout
quote_size = base_size * clip(1 - toxicity_scale * PIN, min_size_scale, 1)
```

Hard gates should stop quoting on stale books, disconnects, crossed books, estimator failure, excessive inventory, insufficient free margin, abnormal volatility, or a configured loss limit.

### Phase 3 - Event-Driven Backtest

The daily portfolio backtester is not suitable. Replay trades and L2 events in timestamp order and model:

- post-only acceptance and tick/size constraints;
- conservative queue position and partial fills;
- cancel/replace latency and stale-quote exposure;
- maker/taker fees, funding, and inventory mark-to-market;
- quote refresh throttling and exchange rate limits;
- liquidation and margin constraints.

Use walk-forward calibration: fit `sigma`, `kappa`, PIN, and markout curves only on prior data. Compare all four policies on identical event windows. Report spread capture, adverse-selection markout, fill rate, inventory PnL, total PnL, drawdown, quote uptime, cancel rate, and tail inventory.

### Phase 4 - Testnet and Controlled Mainnet

1. Run a read-only quote shadow that logs intended quotes without submission.
2. Run testnet with minimal sizes and forced disconnect/restart tests.
3. Require reconciliation of every fill and verify the dead-man switch.
4. Mainnet canary: BTC only, one level per side, minimal size, strict inventory and daily-loss caps.
5. Add ETH and additional assets only after each asset has independent PIN and fill-intensity calibration.

## Acceptance Criteria

- Synthetic PIN estimates recover known parameters without look-ahead.
- Quote calculations satisfy inventory, spread, tick, and finite-value invariants.
- Replays are deterministic and contain no future data in calibration.
- The combined policy improves adverse-selection-adjusted spread capture over `AS_BASELINE` after fees and funding.
- Approximate AS quotes remain within documented tolerances of the numerical HJB benchmark over the enabled parameter domain.
- No strategy code calls the portfolio rebalancer's global cancel path.
- Testnet survives reconnects, duplicate events, ambiguous fills, and stale data without opening unintended exposure.
- Mainnet remains disabled until the shadow and testnet reports are reviewed.

## Questions for Sasha Stoikov

These answers would materially reduce implementation ambiguity:

1. For a continuously traded crypto perpetual, would you favor rolling finite-horizon controls, the paper's discounted infinite-horizon formulation, or a later stationary AS extension? How should `omega` be selected operationally?
2. What normalization of `q`, `sigma`, and `gamma` do you recommend so risk aversion is stable across assets with different prices, contract sizes, and volatility regimes?
3. With queue priority and cancel latency, should `A` and `kappa` be estimated from market-order depth, our conditional fill probability, or a state-dependent intensity model? Which definition best preserves the model's economics?
4. When adding informed-flow toxicity, is it more principled to alter side-specific arrival intensities, shift the reservation price, or add an adverse-selection premium to each half-spread?
5. In practice, which parameter regimes make the paper's inventory/arrival linearization unreliable enough that a numerical HJB or later extension should replace the closed-form approximation?

## Next Operational Milestone

Run the recorder continuously for BTC and ETH, review its gap/clock-skew report, and accumulate 60 complete UTC days. Then create pre-window calibration artifacts, compare all four policies on identical post-calibration replays, and begin a dry-run quote shadow. Testnet and mainnet promotion remain gated on those reports.
