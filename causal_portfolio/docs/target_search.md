# Prediction-target search: spreads, vol, funding, flows, tails

Walk-forward 1-step OOS predictions (train 252d, refit 5d). Skill = OOS R² of (baseline + factor innovations) minus OOS R² of the persistence baseline alone. Placebo: circular shifts of the factor block only (baseline kept intact), p = share of shifts with ≥ the real skill.

- Run UTC: `2026-06-12T20:07:27+00:00`

| Target | OOS days | R² baseline | R² +factors | skill | placebo p |
|---|---:|---:|---:|---:|---:|
| eth_btc_spread | 1206 | -0.0148 | -0.0459 | -0.0311 | — |
| ew_fwd_absret | 538 | -0.0071 | -0.0693 | -0.0621 | — |
| funding_next | 539 | 0.2468 | 0.2700 | +0.0233 | 0.02 |
| liq_flow_next | 1207 | -0.0108 | 0.7747 | +0.7855 | 0.00 |
| tail_next | 1203 | -0.0072 | -0.0762 | -0.0689 | — |

## Verdict

- **funding_next**: the factor innovations add +0.0233 OOS R² over the persistence baseline (placebo p=0.02).
- **liq_flow_next**: the factor innovations add +0.7855 OOS R² over the persistence baseline (placebo p=0.00).
## Interpretation

- **liq_flow_next (skill +0.79, p=0.00)** — the DAG v2 stable edge
  (`returns → liq_flow`) confirmed predictively, but the MAGNITUDE (OOS R²
  0.77 from yesterday's returns, while liq_flow's own AR(1) has none) points
  to a substantially MECHANICAL channel: TVL is marked in USD, so a ~1-day
  provider recording lag makes "yesterday's return predicts today's TVL
  change" partly price marking rather than behavioral flow. Usable for
  anticipating TVL/flow prints (LP context), not evidence of economic flow
  response per se. Distinguishing marks from flows needs token-quantity TVL
  (DefiLlama provides USD; Dune LP events could separate it).
- **funding_next (skill +0.023, p=0.02)** — real, placebo-robust incremental
  information: factor innovations improve next-day funding forecasts over
  funding's own (strong) persistence. Directly relevant to a funding-carry
  strategy: better funding forecasts = better entry/exit timing for the
  carry book.
- **eth_btc_spread / ew_fwd_absret / tail_next** — negative skill: the
  factors don't predict even the chain-specific spread, and add nothing to
  vol persistence (consistent with Direction 5) or tail risk.
