# Continuous risk-targeting DAGs (state → continuous exposure → return)

Each rule maps a single causal state variable to a *continuous* BTC exposure in [0, 1] (not a 0/1 switch).  Decision at t close applies to t+1 return; 5bp one-way cost on turnover.  Exposure is capped at 1 (no leverage), so vol-targeting can only de-risk; the vol *budget* is a trailing-median of realized vol so average exposure sits near 1 (causal, no full-sample tuning).

- Asset: BTC | window `2021-01-01` → `2025-12-31` (1825 days)
- Binary baselines for reference: **trend_50** Sharpe ~0.88, plain **vol_target** Sharpe ~0.39 (from `regime_rotation`).
- Run UTC: `2026-06-17T16:00:29+00:00`

| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | Avg exp | Turnover | Placebo p (Sharpe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bh ⟵ BH | +37% | 0.63 | 0.66 | -77% | 0.48 | 1.00 | 1.0 | — |
| trend50_x_vol | +20% | 0.61 | 0.47 | -54% | 0.37 | 0.45 | 104.5 | — |
| trend_zscore | +17% | 0.60 | 0.43 | -42% | 0.40 | 0.31 | 43.7 | — |
| vol_target_ewma | +21% | 0.41 | 0.41 | -74% | 0.28 | 0.89 | 32.9 | — |
| vol_target_realized | +20% | 0.39 | 0.39 | -75% | 0.27 | 0.88 | 23.5 | — |
| tsmom | +14% | 0.39 | 0.31 | -57% | 0.25 | 0.50 | 80.8 | — |
| downside_target | +17% | 0.35 | 0.35 | -75% | 0.22 | 0.84 | 29.3 | — |

## Per-rule notes

- **bh** — Buy-and-hold BTC. The benchmark nothing in this repo beats OOS. 
- **trend50_x_vol** — Binary trend_50 filter times the vol scale — combines the binary trend baseline with continuous de-risking. Sharpe below BH (0.61 vs 0.63). Calmar below BH (0.37 vs 0.48).
- **trend_zscore** — Continuous trend strength: clip(z-score of (price-200dMA)/MA, 0, 1) — scales in with how far price sits above its 200d MA. Sharpe below BH (0.60 vs 0.63). Calmar below BH (0.40 vs 0.48).
- **vol_target_ewma** — Exposure = clip(median(ewVol)/ewVol, 0, 1), span-20 EWMA vol; de-risks smoothly as vol rises above its trailing median. Sharpe below BH (0.41 vs 0.63). Calmar below BH (0.28 vs 0.48).
- **vol_target_realized** — Same form with rolling-20d realized std instead of EWMA; less responsive but lower turnover than the EWMA version. Sharpe below BH (0.39 vs 0.63). Calmar below BH (0.27 vs 0.48).
- **tsmom** — Classic single-asset TSMOM (Moskowitz-Ooi-Pedersen, N=90): trend gate (sign of 90d return) times the vol scale. Sharpe below BH (0.39 vs 0.63). Calmar below BH (0.25 vs 0.48).
- **downside_target** — Scales by trailing downside semi-deviation (std of negative days only), so upside vol is not penalized. Sharpe below BH (0.35 vs 0.63). Calmar below BH (0.22 vs 0.48).

## Verdict

Buy-and-hold BTC: ann +37%, Sharpe 0.63, Sortino 0.66, maxDD -77%, Calmar 0.48.

**No continuous risk-targeting rule beats BH on Sharpe OR Calmar.** All variants de-risk (avg exposure < 1) and therefore leave return on the table in this bull-dominated cycle; cutting drawdown does not raise the Calmar enough to compensate, and the best Sharpe (trend50_x_vol, 0.61) still sits just under BH's 0.63. No rule cleared the bar to even warrant a placebo test of its timing.

### Continuous vs binary

Best continuous rule by Sharpe is **trend50_x_vol** (0.61).  The binary **trend_50** baseline (Sharpe ~0.88) is **not** beaten by any continuous variant here — continuous scaling does not buy a robust risk-adjusted edge over the simple binary trend gate on this single BTC cycle.

## Caveats

- **No leverage (exposure cap 1):** vol-targeting can only de-risk, never add risk in calm periods, so it mechanically *reduces* return in a sustained bull market (2021-2025 BTC).
- **One cycle:** a single 2021-2025 BTC sample. Trend/vol overlays look good largely because they sat out the 2022 bear; that is one event, not a distribution.
- **Placebo is the honest test:** a high placebo p means the rule's edge is its average exposure level (reproducible by random timing), not the timing of its calls.
- Causal throughout: trailing/EW stats and rolling medians only; no thresholds fit on the full sample; one-day decision lag.