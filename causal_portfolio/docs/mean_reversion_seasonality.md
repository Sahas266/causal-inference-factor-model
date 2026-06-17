# Mean-reversion & calendar/seasonality rules on BTC

Textbook 'buy-the-dip' mean-reversion plus calendar/seasonality rules, each as a daily BTC exposure in [0,1] (stables = 0% when out). Signals are causal (trailing windows only); decision at t close applied to t+1, switching cost charged on every exposure change. Seasonality is estimated on a TRAILING 1-year window, never the full sample. Sorted by Calmar.

- Asset: BTC | window `2021-01-01` -> `2025-12-31` (1825 days)
- Headline switching cost: `5` bp one-way; net-Sharpe columns show 5 / 10 / 20 bp sensitivity
- Run UTC: `2026-06-17T15:50:48+00:00`

| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | placebo p | net Sharpe 5bp | 10bp | 20bp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bh <- BH | +37% | 0.63 | -77% | 0.48 | 100% | 1 | — | 0.63 | 0.63 | 0.63 |
| rsi_dipbuyer | +8% | 0.52 | -18% | 0.44 | 4% | 42 | — | 0.52 | 0.49 | 0.43 |
| bollinger | +18% | 0.54 | -54% | 0.34 | 24% | 68 | — | 0.54 | 0.52 | 0.48 |
| turn_of_month | +8% | 0.39 | -25% | 0.32 | 13% | 120 | — | 0.39 | 0.33 | 0.21 |
| rsi_reversal | +11% | 0.27 | -60% | 0.18 | 40% | 15 | — | 0.27 | 0.26 | 0.25 |
| st_reversal | +3% | 0.07 | -63% | 0.04 | 45% | 72 | — | 0.07 | 0.05 | 0.01 |
| dow_seasonality | +1% | 0.03 | -72% | 0.02 | 62% | 776 | — | 0.03 | -0.15 | -0.49 |

## Day-of-week: which weekdays the rule favors

Fraction of each weekday the trailing-1y rule held BTC (higher = that weekday's trailing mean was usually positive):

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---:|---:|---:|---:|---:|---:|---:|
| 67% | 48% | 82% | 39% | 67% | 64% | 70% |

Most-favored weekdays (held >=50% of the time): **Mon, Wed, Fri, Sat, Sun**.

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, maxDD -77%, Calmar 0.48.

**No rule clears even the first bar.** Not one beats buy-and-hold BTC on gross Sharpe (0.63) — so none warranted placebo testing. Every rule de-risks (lower maxDD via lower average exposure) but at the cost of giving up so much upside that risk-adjusted return falls below BH. Mean-reversion does not time BTC's regime here.

## Robustness: best mean-reversion rule on the EW basket

Applied **rsi_dipbuyer** (best BTC mean-reversion rule by Calmar) to the equal-weight basket of 10 coins, to check it isn't BTC-specific.

| Rule | Ann.ret | Sharpe | MaxDD | Calmar | %in mkt | switches | placebo p | net 5/10/20bp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basket BH | +87% | 1.09 | -79% | 1.09 | 100% | 1 | — | 1.09/1.09/1.09 |
| basket rsi_dipbuyer | +11% | 0.59 | -24% | 0.48 | 2% | 32 | 0.21 | 0.59/0.58/0.55 |

Result: the rule does NOT beat basket BH either — not a BTC-specific fluke, just a weak edge everywhere.

## Caveats

- **These churn.** Mean-reversion rules flip in/out far more than the regime overlays; every switch pays the spread, so the 10/20bp columns matter more here than anywhere else in this repo.
- **Mean-reversion is fragile and regime-dependent.** 'Buy the dip' works in choppy/ranging markets and gets run over in trends (crashes keep crashing, melt-ups keep melting up). A single sample can flatter or bury it depending on which regime dominated.
- **One market cycle.** 2021-2025 is essentially one bull-bear-bull crypto cycle; weekday/turn-of-month effects estimated on it are thin and unstable, and any apparent seasonality may not persist.
- **Stables = 0%.** Out-of-market days earn nothing here; a real stable yield would lift every rule's return modestly but not change the risk-adjusted ranking vs BH.
- **Placebo = circular time-shift.** It asks whether the *timing* beats a random-timed overlay with the same average exposure; a high p means the rule's only 'edge' is being in the market a certain fraction of the time, not skill.