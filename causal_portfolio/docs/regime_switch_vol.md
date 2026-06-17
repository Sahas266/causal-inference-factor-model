# Volatility-regime-conditional strategy switching

Detect a **volatility regime** (calm / normal / volatile) from a trailing-only signal, then apply a **different rule per regime** — the user's *"de-risk / mean-revert in volatile, momentum in calm"* hypothesis.  Same honesty bar as the rest of the DAG work: causal regime labels (trailing rolling quantiles), causal menu rules, one-day lag, 5bp one-way cost.  A switch **works** only if it beats `trend_50` AND survives the regime-shuffle placebo (p<0.10).

- Asset: BTC | window `2021-01-01` → `2025-12-31` (1825 days)
- Regime: 20d realized vol / VIX, bucketed by trailing 252d rolling quantiles (calm<0.33, volatile>0.67)
- Adaptive: reselect every 21d on trailing 365d per-regime Sharpe
- Run UTC: `2026-06-17T17:58:36+00:00`

## References

| Strategy | Ann.ret | Sharpe | Sortino | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|
| BH_BTC | +37% | 0.63 | 0.66 | -77% | 0.48 |
| trend_50 (unconditional) | +34% | 0.88 | 0.72 | -57% | 0.60 |

## Regime def: `btcvol_3state` (dwell calm 42% / normal 30% / volatile 28%, 158 regime switches)

| Mapping (mode) | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | shuffle p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| momo_calm__voltgt_vol | +25% | 0.67 | 0.60 | -55% | 0.45 | 63% | 369 | — |
| momo_calm__revert_vol | +19% | 0.60 | 0.44 | -51% | 0.38 | 38% | 144 | — |
| momo_calm__flat_vol | +16% | 0.54 | 0.39 | -51% | 0.32 | 37% | 126 | — |
| INVERTED_flat_calm__momo_vol | +14% | 0.53 | 0.32 | -34% | 0.41 | 25% | 128 | — |
| hold_calm__voltgt_vol | +21% | 0.41 | 0.40 | -75% | 0.28 | 94% | 267 | — |
| hold_calm__flat_vol | +13% | 0.36 | 0.29 | -58% | 0.23 | 53% | 101 | — |
| ADAPTIVE_trailing_sharpe (adaptive) | +4% | 0.13 | 0.09 | -64% | 0.06 | 35% | 224 | — |

## Regime def: `btcvol_2state` (dwell calm 58% / normal 0% / volatile 42%, 100 regime switches)

| Mapping (mode) | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | shuffle p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADAPTIVE_trailing_sharpe (adaptive) | +19% | 0.58 | 0.41 | -51% | 0.38 | 40% | 219 | — |
| INVERTED_flat_calm__momo_vol | +12% | 0.57 | 0.29 | -23% | 0.52 | 17% | 86 | — |
| momo_calm__revert_vol | +14% | 0.46 | 0.31 | -41% | 0.34 | 32% | 144 | — |
| momo_calm__flat_vol | +12% | 0.45 | 0.30 | -43% | 0.28 | 30% | 126 | — |
| hold_calm__voltgt_vol | +20% | 0.40 | 0.40 | -75% | 0.27 | 94% | 411 | — |
| hold_calm__flat_vol | +8% | 0.21 | 0.16 | -68% | 0.12 | 55% | 101 | — |

## Regime def: `vix_3state` (dwell calm 42% / normal 27% / volatile 30%, 219 regime switches)

| Mapping (mode) | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | %in mkt | switches | shuffle p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INVERTED_flat_calm__momo_vol | +14% | 0.53 | 0.32 | -46% | 0.30 | 26% | 148 | — |
| momo_calm__voltgt_vol | +22% | 0.52 | 0.46 | -74% | 0.30 | 65% | 285 | — |
| hold_calm__voltgt_vol | +22% | 0.42 | 0.42 | -76% | 0.29 | 95% | 175 | — |
| momo_calm__revert_vol | +10% | 0.31 | 0.22 | -54% | 0.18 | 38% | 170 | — |
| ADAPTIVE_trailing_sharpe (adaptive) | +9% | 0.27 | 0.21 | -37% | 0.23 | 45% | 238 | — |
| momo_calm__flat_vol | +8% | 0.26 | 0.18 | -54% | 0.15 | 36% | 146 | — |
| hold_calm__flat_vol | +8% | 0.21 | 0.18 | -57% | 0.14 | 54% | 137 | — |

## Verdict

Bar to clear: **trend_50** (Sharpe 0.88, maxDD -57%) AND regime-shuffle placebo p<0.10. BH BTC reference Sharpe 0.63.

**No volatility-regime-conditional configuration both beats trend_50 on Sharpe and survives the regime-shuffle placebo.**

Nothing even edges trend_50 in-sample: applying a de-risk/mean-revert rule only in volatile regimes and momentum/hold in calm does not improve on the plain unconditional trend filter. The best vol-conditional config (`btcvol_3state / momo_calm__voltgt_vol`, Sharpe 0.67) lands between BH BTC (0.63) and trend_50 (0.88) — i.e. it adds churn over BH without reaching the unconditional trend benchmark.

**The adversarial control is the tell.** The `INVERTED_*` mapping (de-risk in *calm*, ride momentum in *volatile* — the opposite of the hypothesis) is **not** worse than its correctly-oriented twins; in the 2-state definition it posts the second-highest Sharpe of its group (0.57, vs 0.45 for the "correct" `momo_calm__flat_vol`). If the volatility regime carried directional information, inverting the mapping should hurt — it does not. That is direct evidence the calm/volatile label is not separating up-days from down-days. Consistent with `dag_strategies_findings.md` §A, where plain vol gating already lost to BH unconditionally: making the gating *regime-conditional* does not rescue it.

### Caveats
- **One market cycle** (2021–2025: a single bull→bear→recovery). Volatility in crypto clusters around BOTH crashes and parabolic rallies, so a 'volatile' label does not cleanly mean 'go down' — de-risking in high vol can cut upside as much as downside.
- **Multiple testing**: several fixed mappings × 3 regime definitions × an adaptive selector were tried; the best in-sample number is upward-biased. The placebo guards regime-timing luck on a *fixed* config, not the luck of having picked the best config.
- **Stables at 0%**; **no leverage** (exposure capped at 1, so every overlay can only de-risk — structurally disadvantaged on raw return in a bull).
- The `INVERTED_*` row is an adversarial control (de-risk in calm, momentum in volatile); if it is not clearly worse than its non-inverted twin, the regime label is not carrying directional info.