# Trend-Rotation Execution Benchmark

## Scope

This benchmark holds the strategy constant and changes only execution. The strategy is the preferred single-asset rule from `strategy_trend_rotation.md`: hold BTC when the close is above its 100-day moving average and the trailing 30-day return is positive; otherwise hold stables. The signal is formed at day `t` and executed for day `t+1`.

The arms are:

- **Naive:** immediate IOC execution. Historical cost uses the account's 4.5 bp testnet taker rate and therefore excludes unknown historical spread/slippage.
- **AS+PIN:** a target-relative Avellaneda-Stoikov passive quote, widened on the threatened side by a PIN/posterior adverse-markout premium, followed by IOC fallback after the time budget.

## Historical Result

Window: **2021-04-12 through 2025-10-11**, covering all **98** overlapping `dual_confirm` switches.

| Arm | Mean cost/switch | Total switch cost | Passive fill rate | Ann. return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Naive IOC lower bound | **4.50 bp** | **441.00 bp** | n/a | **31.75%** | **1.002** | **-25.93%** |
| AS+PIN, daily high/low touch | 11.12 bp | 1,089.71 bp | 85.7% | 30.31% | 0.945 | -26.00% |
| AS+PIN, daily close cross | 94.46 bp | 9,256.76 bp | 42.9% | 12.17% | 0.381 | -39.94% |

**Historical verdict: naive wins.** AS+PIN did not improve this slow, directional strategy under either passive-fill bound. Its unfilled orders incurred opportunity cost before IOC fallback, while the candle-classified toxicity premium occasionally became far too wide.

Important evidence limits:

- The local warehouse provides daily BTC prices but no historical aggressor-side trades or L2 queue state.
- Public Hyperliquid daily candles supply open/high/low/close, volume, and trade count. Buy/sell counts are therefore bulk-volume-style candle classifications, not observed trade sides.
- The resulting 60-day PIN proxy ranged from 0.0000 to 0.7009; only 75.5% of switch-date fits passed full estimator diagnostics.
- Estimated side premiums ranged from 0 to 329.1 bp. That instability is itself evidence against deploying this proxy.
- High/low touch assumes a fill without queue information and is optimistic; close cross is deliberately conservative. Neither is a substitute for event-level replay.

## Testnet Result

The live `dual_confirm` signal was **risk-off** on 2026-06-23: BTC closed at $62,302 versus a $72,124 MA100, with a -19.1% trailing 30-day return. Because the testnet account was already flat, following the strategy literally was a no-op. To obtain execution observations without changing its final target, the benchmark ran an isolated entry/exit drill for each arm.

One BTC round trip per arm was run with **0.0002 BTC** legs after a **90-second** flow capture. The account began and ended flat, with no residual orders. The bootstrap used 35 five-second buckets; its PIN proxy was 0.6800, with 5-second adverse markouts of 2.44 bp after aggressive buys and 2.37 bp after aggressive sells.

| Leg | Maker share | Fallback share | Shortfall | Elapsed |
|---|---:|---:|---:|---:|
| Naive buy | 0% | 100% | +10.20 bp | 5.70 s |
| Naive sell | 0% | 100% | +8.73 bp | 2.67 s |
| AS+PIN buy | 0% | 100% | -7.39 bp | 17.27 s |
| AS+PIN sell | 100% | 0% | -4.21 bp | 6.97 s |

- Naive round-trip PnL: **-$0.021451**.
- AS+PIN round-trip PnL: **-$0.008158**.
- The observed loss was 62.0% smaller for AS+PIN in this single trial.

This is **not evidence of superiority**. The AS+PIN buy never received a maker fill; it benefited from the market moving lower during its timeout before IOC fallback. Only the sell leg tested passive execution successfully. The arms were sequential rather than simultaneous, so market path and latency confound the comparison. The bucket PIN is a short testnet bootstrap, not the daily structural PIN defined by Easley et al.

## Decision

Do **not** replace naive execution for `dual_confirm` based on this benchmark. The current evidence supports a narrower next experiment:

1. Keep IOC as the deterministic fallback for daily strategy switches.
2. Test AS+PIN as a short-lived passive pre-stage with a strict opportunity-cost budget.
3. Use true 60-day aggressor-side PIN data; do not use candle- or five-second-bucket PIN in production.
4. Run at least 30 randomized, paired testnet entry/exit trials and report maker fill rate, time-to-fill, fallback rate, shortfall, and post-fill markouts.
5. Repeat historical evaluation only after event-level trades and L2 snapshots are available.

Reproducible harnesses are in `causal_portfolio/market_making/strategy_execution_benchmark.py` and `testnet_execution_benchmark.py`. The artifacts supporting the reported tables are committed under `causal_portfolio/docs/execution_artifacts/`; ad hoc reruns may also write raw outputs under ignored `tmp/` paths.
