# Macro- and on-chain-conditioned risk-on/off gates for BTC

Each rule gates BTC↔stables (0% when out) on a single causal macro or crypto-native state variable; decision at t close, applied to t+1 return, 5bp one-way switching cost. Sorted by Sharpe. Baselines: `always_in` (= buy-and-hold BTC) and `trend_50` (BTC price > 50d MA).

- Asset: BTC | window `2021-01-01` → `2025-12-31` (1825 days)
- Run UTC: `2026-06-17T15:50:42+00:00`

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| curve_flatten_off | +38% | 1.05 | 0.84 | -38% | 0.99 | 48% | 133 | 0.04 |
| curve_flatten_off_and_trend50 | +27% | 1.00 | 0.64 | -24% | 1.13 | 27% | 116 | 0.04 |
| trend_50 ⟵ trend baseline | +34% | 0.88 | 0.72 | -57% | 0.60 | 51% | 104 | 0.03 |
| always_in ⟵ BH | +37% | 0.63 | 0.66 | -77% | 0.48 | 100% | 1 | — |
| funding_q_off | +35% | 0.63 | 0.64 | -77% | 0.45 | 88% | 121 | — |
| funding_neg_off | +17% | 0.59 | 0.41 | -35% | 0.48 | 39% | 85 | — |
| stablecoin_flow_on | +25% | 0.49 | 0.44 | -75% | 0.34 | 73% | 50 | — |
| rates_off | +19% | 0.48 | 0.36 | -58% | 0.32 | 46% | 108 | — |
| m2_liquidity_on | +11% | 0.23 | 0.19 | -61% | 0.17 | 63% | 3 | — |
| dollar_trend_off | +2% | 0.06 | 0.05 | -73% | 0.03 | 47% | 109 | — |
| curve_steepen_off | -4% | -0.08 | -0.06 | -81% | -0.04 | 52% | 134 | — |

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48.

Price-trend baseline `trend_50`: ann +34%, Sharpe 0.88, maxDD -57%, Calmar 0.60.

Gates that beat BH on Sharpe AND whose timing survives the placebo (p < 0.10):
- **curve_flatten_off** — Sharpe 1.05 vs BH 0.63, maxDD -38% vs -77% (placebo p=0.04)
- **curve_flatten_off_and_trend50** — Sharpe 1.00 vs BH 0.63, maxDD -24% vs -77% (placebo p=0.04)
- **trend_50** — Sharpe 0.88 vs BH 0.63, maxDD -57% vs -77% (placebo p=0.03)

**Read this skeptically.** The 2s10s curve has only a few independent regime turns over 2021–2025, so a low placebo p can be a small-sample artifact (the circular shift cannot manufacture many distinct exposure paths from such a slow signal). Note also that the *flattening/inverting* sign is the one that helps; the opposite `curve_steepen_off` sign is among the worst rules — so the sign was chosen by which helped, exactly the kind of two-way bet that inflates in-sample significance. Treat the curve result as a hypothesis to test out-of-sample, not a deployable edge.

## Does a macro/on-chain gate ADD to price trend?

Best standalone gate: **curve_flatten_off** (Sharpe 1.05). AND-combined with `trend_50` → **curve_flatten_off_and_trend50**: Sharpe 1.00 vs `trend_50` alone 0.88, Calmar 1.13 vs 0.60. The gate **ADDS** to the price trend on this sample (higher Sharpe than trend alone), though it also cuts %-in-market to 27% from 51% — check the placebo before trusting it.

## Caveats

- **Publication lag (macro).** FRED series are forward-filled to the BTC daily index. Releases are backward-dated, so ffill of *realized* values introduces no look-ahead, but the actual *publication* of a month's M2 / CPI lags the reference date by days-to-weeks; the live tradability of M2- and CPI-based gates is mildly optimistic here.
- **Few independent regime changes.** Over 2021–2025 the macro series (dollar trend, 10y level, curve, M2 trend) have only a handful of independent regime turns. The placebo and any implied degrees of freedom are therefore weak — a 'survivor' on this sample is not strong evidence, and a non-survivor is the safer read.
- **On-chain funding starts 2023-11.** The aggregate perp funding series only begins 2023-11, so `funding_*` rules are evaluated on a much shorter, mostly-bull sub-window and are not comparable to the full-sample macro rules.
- **Stables earn 0%.** Risk-off legs sit in cash at 0% return (conservative); a real stablecoin yield would only improve the gated rules relative to BH.