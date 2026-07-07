# 8h funding → next-8h returns (intraday alignment)

**Date:** 2026-07-02. **Module:** `experiments/funding_intraday.py`
(tests: `tests/test_funding_intraday.py`).

## Why this experiment

The daily-horizon program closed with a structural null — on-chain factors are
downstream of daily returns (`causal_directions_summary.md`), and the daily
funding gates were exact ties with buy-and-hold (`dag_strategies_findings.md`
§D). Its post-mortem left one alignment explicitly open: **the funding epoch
itself**. The rate paid at epoch *t* is fixed by the premium *before* *t* and
the next-8h return is measured strictly after, so the simultaneity /
reverse-causality objection is weakest exactly here. This experiment tests
that remaining escape hatch.

## Design (pre-registered, deliberately small)

- **Regression:** next-8h return ~ funding z-score (rolling 90d, data ≤ t),
  Newey-West HAC (3 lags), with and without trailing-return/vol controls.
- **Overlays (long/flat, 5bp one-way, vs BH on the same grid):**
  `extreme_off` (flat above rolling q90 — crowded longs), `carry_on` (long
  only when funding ≤ 0), `band` (long inside rolling [q10, q90]).
  Circular-shift placebo (n=200) for anything that beats BH on Sharpe.
- **Control:** identical machinery on the 24h grid (the known daily tie).
- **Data:** hourly WBTC/WETH spot from Dune `prices.hour`
  ([query 7868219](https://dune.com/queries/7868219), cached at
  `data/cache/btc_eth_hourly_dune.csv`); hourly `funding_rate_8h`
  (provider=hyperliquid) from the warehouse. Window common to both:
  **2023-12-01 → 2025-12-31** (2,286 8h bars).
- **Causality contract** (unit-tested): signals at bar-start *t* use only
  funding in (t−8h, t] and prices ≤ t; return earned is (t, t+8h].

## Results

### BTC — the headline test is a clean null

| | 8h | 24h |
|---|---|---|
| β (bps per σ of funding) | −1.13 | −0.03 |
| HAC t | −0.37 | −0.00 |
| p | 0.71 | 1.00 |
| with controls | β −1.75, t −0.58 | β +0.23, t +0.02 |

| rule (8h) | ann | Sharpe | maxDD | in-mkt | placebo p |
|---|--:|--:|--:|--:|--:|
| **bh** | **+50.4%** | **1.13** | −33.7% | 100% | — |
| extreme_off | +42.5% | 1.01 | −31.7% | 90% | — |
| carry_on | +10.0% | 0.93 | −8.7% | 6% | — |
| band | +11.1% | 0.29 | −37.0% | 78% | — |

No overlay beats buy-and-hold at either horizon. The one apparent exception —
`carry_on` @ 24h (Sharpe 1.18 vs BH 1.11, shuffle p=0.03) — is **not an
edge**: it is in the market **4%** of the time (~30 bars), returns +10%/yr
versus BH's +51%, and its Sharpe is the cash-like-series artifact the house
docs warn about. BTC funding was negative only during brief panic dips in
this bull-heavy window; the rule bottom-ticks a handful of days. Its 8h twin
(6% in-market) loses to BH outright.

### ETH — a weak coefficient with the *wrong* sign, fading under controls

| | 8h | 24h |
|---|---|---|
| β | **+9.68** | +24.05 |
| HAC t | +2.09 | +1.68 |
| p | 0.037 | 0.092 |
| with controls | β +8.04, t +1.76, p=0.078 | β +24.22, t +1.73, p=0.083 |

The sign is **positive** — high funding predicts *higher* next-8h returns
(crowding-continuation / momentum), the opposite of the contrarian premise
behind all three pre-registered rules. Accordingly every ETH overlay loses to
BH, badly. The p=0.037 headline does not survive the honesty bar: it fades to
p≈0.08 under trailing-return/vol controls (so it is partly repackaged
momentum), and it is 1 of 8 regressions run (2 assets × 2 horizons × 2 specs)
— right at the multiple-testing expectation. No positive-sign rule was
pre-registered, and building one post-hoc from this coefficient would be
exactly the sample-mining this program's placebo discipline exists to reject.

## Verdict

**The intraday escape hatch is closed.** The daily funding null was not an
artifact of daily aggregation: at funding's native 8h epoch — the alignment
where reverse causality is structurally weakest — funding carries no
exploitable directional information for BTC (t = −0.37), and for ETH only a
weak positive coefficient that shrinks under controls and is consistent with
chance under multiplicity. No pre-registered overlay beats buy-and-hold on
either asset at either horizon.

This strengthens, rather than merely repeats, the program's structural
conclusion: even at the alignment most favorable to the factor→return
premise, the premise fails. Of the three open avenues in
`causal_directions_summary.md`, intraday alignment is now resolved (null);
**exogenous event calendars** and **longer horizons** remain untested.

## Caveats

- **Spot proxy:** WBTC/WETH spot stands in for the perp; the basis is a few
  bps versus ~1% 8h vol. A perp-candle rerun (Dune query is pinned; HL
  `candleSnapshot` when network allows) would be cosmetic, not directional.
- **One venue, ~2.1 years:** Hyperliquid funding only, Dec-2023 → Dec-2025,
  a predominantly bull window — negative-funding regimes are underrepresented.
- **Long/flat only:** the positive ETH coefficient would, if real, want a
  *long-when-crowded* rule; that is post-hoc here and intentionally untested.
