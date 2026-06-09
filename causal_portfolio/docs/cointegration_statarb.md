# Cointegration Stat-Arb (market-neutral) vs BH BTC

Walk-forward Engle-Granger pair selection (ADF-gated) + banded spread mean-reversion, dollar-neutral, equal-weighted across active pairs. Scored OOS against buy-and-hold BTC.

- Assets: `btc, eth, sol, bnb, avax, uni, aave, link, doge, xrp, crv, ltc`
- Window: `2022-01-01` → `2025-12-31` | train 252d, refit 21d | entry z=1.5, exit z=0.5, ADF p<0.05

- Run UTC: `2026-06-09T18:01:50+00:00`

Cointegrated pairs found per refit: mean 8.3, max 10.

Most-selected pairs (refits active): `avax~crv` (18), `doge~avax` (15), `aave~crv` (14), `bnb~btc` (14), `bnb~uni` (12), `bnb~crv` (12), `link~aave` (11), `bnb~link` (11)

## Full OOS window

| Strategy | Total | Sharpe | MaxDD |
|---|---:|---:|---:|
| Stat-Arb | +16.1% | 0.393 | -21.2% |
| BH BTC | +303.2% | 1.137 | -32.1% |

## Walk-forward folds vs BH BTC

1 variants tested. With ~0.5 per-fold coin-flip odds under no edge, the best win-rate is upward-biased by selection across 1 trials — treat a single top performer with suspicion unless its margin over the field is large.

| Strategy | Win-rate vs BH | Median Sharpe | Folds |
|---|---:|---:|---:|
| StatArb | 44% | 1.391 | 9 |

## Verdict

*A market-neutral book should NOT be judged on beating BH BTC's total return — that is a directional benchmark and not its job. The honest questions are: is its standalone risk-adjusted return positive and stable, and does it diversify BTC?*

- BTC correlation of daily returns: **+0.09** (near zero confirms market-neutral by construction).
- Full-window Sharpe (the honest aggregate, no costs): **0.39**.
- Per-fold median Sharpe 1.39 vs full-window 0.39: a large gap means good folds are offset by bad stretches / autocorrelated drawdowns — inconsistent, not a stable edge.

Standalone gross Sharpe 0.39 — **no edge.** Cointegration relationships in this universe were too unstable OOS to trade profitably at these thresholds, before any costs. Consistent with the rest of the project.