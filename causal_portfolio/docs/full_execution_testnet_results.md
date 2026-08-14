# Full Execution Testnet Run Results

Run directory: `tmp/full_execution_testnet_20260629T094847Z`<br>
Network: Hyperliquid testnet<br>
Instrument: BTC for AS/PIN benchmark; BTC/ETH/SOL for the placeholder strategy rebalance<br>
Benchmark size: `0.00034 BTC` per leg

## Executive Summary

The run passed final acceptance. The execution arm completed the placeholder strategy rebalance, flattened material exposure, then ran the AS+PIN benchmark campaign. The final state had no BTC position and no open orders.

| Metric | Result |
|---|---:|
| Total AS/PIN attempts | 5 |
| Successful restored attempts | 3 |
| Usable PIN attempts | 3 |
| AS/PIN maker-buy attempts | 3 |
| AS/PIN maker-sell attempts | 2 |
| Fatal errors | 0 |

Successful attempts were attempts 1, 4, and 5. Attempts 2 and 3 failed non-fatally and were safely cleaned up.

## Result Table

| Attempt | PIN | Naive PnL | AS+PIN PnL | AS+PIN Edge | AS+PIN maker fill |
|---:|---:|---:|---:|---:|---|
| 1 | 0.5767 | -$0.0340 | +$0.0266 | +$0.0606 | buy + sell |
| 4 | 0.5118 | -$0.0395 | +$0.0147 | +$0.0542 | buy + sell |
| 5 | 0.6335 | -$0.0217 | -$0.0405 | -$0.0188 | buy only |

Average over successful attempts:

| Arm | Avg roundtrip PnL | Avg fees |
|---|---:|---:|
| Naive IOC | -$0.0317 | $0.0180 |
| AS+PIN | +$0.0003 | $0.0080 |
| Difference | +$0.0320 | -$0.0100 |

## How the Calculations Work

For each leg, the benchmark computes:

```text
average_price = sum(fill_size * fill_price) / sum(fill_size)
buy_effective_price = (buy_notional + buy_fee) / buy_size
sell_effective_price = (sell_notional - sell_fee) / sell_size
```

Implementation shortfall is measured against the reference mid:

```text
buy_shortfall_bps = (buy_effective_price - reference_mid) / reference_mid * 10,000
sell_shortfall_bps = (reference_mid - sell_effective_price) / reference_mid * 10,000
```

Roundtrip PnL uses the common filled size:

```text
roundtrip_pnl = size * (sell_average_price - buy_average_price) - buy_fee - sell_fee
```

The naive arm buys BTC with a marketable IOC order, then sells with a reduce-only IOC order. This crosses the spread on both sides and pays taker-style costs.

The AS+PIN arm computes Avellaneda-Stoikov quotes from the live mid, volatility, inventory, liquidity slope, and horizon. PIN then adjusts those quotes for toxicity: when informed flow risk is higher, the quote is made more conservative. The benchmark submits post-only AS+PIN quotes first, cancels after the passive window if needed, and uses IOC only for the unfilled remainder.

## Why AS+PIN Was Better Here

AS+PIN was better on average because it captured maker economics instead of always crossing the spread. Attempts 1 and 4 filled both AS+PIN buy and sell legs passively, producing positive roundtrip PnL while the naive IOC arm lost money. Fees were also materially lower: about `$0.0080` per AS+PIN roundtrip versus `$0.0180` for naive IOC.

Attempt 5 shows the limitation: AS+PIN bought passively but had to sell through fallback IOC, and that adverse sell overwhelmed the maker benefit. So the conclusion is not that AS+PIN wins every trade; it is that, when the post-only quotes fill, AS+PIN can reduce spread/fee drag and improve execution quality. In this run, that was enough to turn the average successful roundtrip from negative to roughly flat/slightly positive.

## Safety and Cleanup

The campaign ended with:

- `0 BTC` final position.
- `0` open orders.
- No fatal errors.
- Only tiny ETH/SOL dust from the placeholder rebalance, below practical cleanup size.

Detailed artifacts are in `tmp/full_execution_testnet_20260629T094847Z/`, especially `full_execution_testnet_summary.json` and `terminal_transcript.txt`.
