# Strategy: BTC ↔ stables trend rotation

The best **risk-adjusted** strategy found in this project. Hold the risk asset
while its price is in an uptrend; rotate to stables when it is not. It does not
beat buy-and-hold BTC on raw return — it roughly **matches the return while
cutting drawdown by a third to a half** and lifting Sharpe. It is a price-trend
(momentum) overlay, **not** a causal factor→return model.

> Honesty note: trend-following BTC is the most data-mined strategy in crypto,
> and 2021–2025 is a single market cycle. The placebo tests below validate the
> *timing* within this sample; they cannot validate it against a future regime
> change. Treat this as a risk overlay to paper-trade forward, not a guaranteed
> edge. See "Risks & caveats".

## The rule (exact, reproducible)

State → exposure → return, fully causal:

1. **Signal (at the close of day *t*, from data through *t* only):**
   - `trend_N`:  `exposure = 1 if price_t > MA_N(price)_t else 0`
   - `dual_confirm` (preferred on BTC): `exposure = 1 only if (price_t > MA_100) AND (trailing 30-day return > 0), else 0`

   where `price = cumprod(1 + daily_return)` and `MA_N` is the simple
   N-day moving average.
2. **Execution:** the exposure is applied to day **t+1**'s return (one-day
   lag — no look-ahead). Out-of-market capital sits in stables (modeled at 0%).
3. **Cost:** 5 bp one-way on every change in exposure (`|Δexposure| × 5bp`).
   Turnover is low because the signal is a slow moving average (≈ 100 switches
   over 4 years on BTC; ≈ 0.06 of gross/day on the basket).

Reference implementation: `causal_portfolio/experiments/regime_rotation.py`
(`backtest_rule`, `build_rules`) and `trend_breakout.py` (`_dual_confirm`).

## Variants & recommended parameters

| Variant | Asset | Signal | When to prefer |
|---|---|---|---|
| **trend_on_basket** | EW basket of 11 majors | price > 50d MA | **best overall** — stacks trend (drawdown control) with diversification (free Sharpe) |
| **dual_confirm** | BTC | px > 100d MA AND 30d ret > 0 | best **single-asset** risk control (lowest drawdown); an AND-filter, not a swept lookback |
| **trend_50 / trend_40–60** | BTC | price > 40–60d MA | simplest; the canonical incumbent |
| **trend_125** | BTC | price > 125d MA | maximizes Calmar (deepest drawdown cut) at slightly lower Sharpe |

## Performance (2021-01-01 → 2025-12-31, 1,825 days, 5bp costs)

### On BTC
| Strategy | Ann. ret | Sharpe | Sortino | Max DD | Calmar | % in market | switches | placebo p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold BTC | +37% | 0.63 | 0.66 | −77% | 0.48 | 100% | 1 | — |
| dual_confirm | +28% | **0.95** | 0.70 | **−26%** | **1.10** | 37% | 98 | **0.04** |
| trend_50 | +34% | 0.88 | 0.72 | −57% | 0.60 | 51% | 104 | **0.03** |

### On the equal-weight basket (11 majors)
| Strategy | Ann. ret | Sharpe | Sortino | Max DD | Calmar | % in mkt | placebo p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold basket (1/N) | +85% | 1.05 | 1.03 | −80% | 1.07 | 100% | — |
| breadth_trend_basket | +58% | **1.27** | 1.22 | −47% | 1.24 | 80% | 0.09 |
| trend_on_basket | +67% | **1.22** | 0.93 | −47% | **1.43** | 50% | 0.09 |

The basket version is the strongest: the 1/N basket already beats BTC on Sharpe
(1.05 vs 0.63) for free, and the trend filter on top cuts its drawdown from
−80% to −47% while keeping Sharpe ≥ 1.2.

## Robustness — the lookback is a band, not a lucky point

Sweeping the moving-average length (placebo-tested at each), the edge appears as
a **contiguous band**, which is the signature of a real effect rather than an
overfit parameter:

| MA days | Sharpe | Calmar | max DD | placebo p |
|---:|---:|---:|---:|---:|
| 20 | 0.28 | 0.16 | −65% | 0.61 |
| 30 | 0.59 | 0.43 | −52% | 0.19 |
| **40** | **1.09** | 0.83 | −49% | **0.01** |
| **50** | 0.88 | 0.60 | −57% | **0.03** |
| **60** | 0.82 | 0.52 | −60% | **0.04** |
| 75 | 0.67 | 0.46 | −57% | 0.14 |
| 100 | 0.66 | 0.67 | −37% | 0.15 |
| **125** | 0.93 | **1.04** | −33% | **0.06** |
| 150 | 0.58 | 0.49 | −44% | 0.30 |
| 200 | 0.42 | 0.32 | −50% | 0.46 |

Two placebo-robust regions: a **fast band (40–60d)** maximizing Sharpe (sidesteps
sharp selloffs) and a **slow point (~125d)** maximizing Calmar (truncates whole
bear markets to −33%). The endpoints fail as expected — 20d whipsaws, 200d lags.

## Why it works (mechanism)

Crypto prices exhibit medium-term **momentum / trend persistence**: drawdowns
cluster and compound (the 2022 bear ran for ~12 months), so a slow trend filter
exits early in a sustained downtrend and re-enters after it turns. The edge is
**drawdown truncation**, not return enhancement — you give up some upside (you
re-enter late) in exchange for sitting out the worst of the bear. This is a
statistical regularity in price, **not** a causal mechanism; there is no
on-chain factor driving the decision.

## What does NOT work (tested, rejected)

- **Volatility-triggered rotation** ("rotate to stables when volatile") —
  `vol_off` (Sharpe 0.35), `vol_target` (0.39), `dd_off` (0.31) all **underperform
  BH BTC**: they cut return more than risk. In crypto, high volatility clusters
  around both crashes *and* parabolic rallies, so de-risking on vol discards the
  upside with the downside. The vol regime carries no directional information
  (inverting the rule was no worse).
- **Fancier regime detection** (HMM, custom drawdown/correlation/breadth
  regimes) does **not** beat this simple MA switch — see
  `regime_switching_findings.md`. More machinery buys churn, not skill.
- **Stacking de-risking filters** (`trend AND vol`) was the *worst* rule of all —
  it sits out the recoveries.

Honorable mention: **`funding_off`** (sit out when aggregate perp funding is
negative) — Sharpe 0.59, maxDD −35%, in market only 39% of the time. It is
economically *distinct* from trend (positioning, not price), so it is the one
signal worth **combining** with the trend filter rather than the vol rules.

## Risks & caveats

- **One market cycle.** The drawdown benefit comes largely from truncating the
  single 2022 bear. The placebo validates timing within-sample only.
- **Whipsaw in choppy/ranging markets.** A flat, mean-reverting market makes the
  filter cross repeatedly, paying costs and missing snap-backs (visible in the
  −0.46 placebo gap at 200d and the 20–30d whipsaw).
- **Late re-entry.** Trend exits below the MA and re-enters above it, so it gives
  back the first leg of every recovery — the cost of the drawdown protection.
- **Stables = 0%.** A real cash/stablecoin yield (or T-bills) would add to every
  risk-off day and only improve the numbers; not modeled here.
- **No leverage / exposure capped at 1.** The overlay can only de-risk, so it is
  structurally behind BH on *raw return* in a bull market — it wins on Sharpe and
  drawdown, not total return.
- **Costs.** 5bp one-way is fine for BTC/major perps; verify against the live
  Hyperliquid taker fee + slippage before sizing.

## Deployment path

Execution A/B results are documented in `trend_rotation_execution_benchmark.md`. The historical proxy favored immediate IOC; the single testnet AS+PIN round trip was cheaper but confounded by sequential market movement and only one maker-filled leg. Keep IOC as the production fallback until event-level replay and repeated paired testnet trials pass.

1. Pick the variant: **`trend_on_basket`** (50d MA on the 1/N basket of majors)
   for the best risk-adjusted profile, or **`dual_confirm`** on BTC for the
   tightest single-asset drawdown.
2. Wire the exposure signal into the execution layer
   (`causal_portfolio/execution/`) — it already turns weight vectors into
   Hyperliquid orders; this strategy just emits a {0, 1} (or per-asset {0, 1})
   weight each day.
3. **Paper-trade / testnet forward** for ≥ ~6 months before real size — the
   same forward-validation discipline applied to the `chain_congestion`
   candidate edge (`dag_v2.md`). The within-sample placebo is necessary but not
   sufficient.
4. Consider combining with `funding_off` (the one orthogonal signal) and a real
   stablecoin yield on the risk-off leg.

## Reproduce

```bash
# Single-asset battery incl. trend_50, vol rules, dd stops, + MA sweep
python -m causal_portfolio.experiments.regime_rotation

# Trend/breakout family incl. dual_confirm, Donchian, MACD, tsmom
python -m causal_portfolio.experiments.trend_breakout

# Basket overlays incl. trend_on_basket, breadth_trend_basket
python -m causal_portfolio.experiments.ensemble_dags
```

Outputs: `docs/regime_rotation.md`, `docs/trend_breakout.md`,
`docs/ensemble_dags.md`. Consolidated comparison across all ~60 DAG strategies:
`docs/dag_strategies_findings.md`. Regime-ID comparison:
`docs/regime_switching_findings.md`.
