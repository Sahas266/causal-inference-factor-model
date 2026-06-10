# Five directions to improve the causal model — results summary

Date: 2026-06-10. Context: after the 14-method re-audit (Batch 6) and the
full-warehouse IV search established that (a) the hand-drawn DAG encodes no
endogeneity, and (b) even strong instruments make 2SLS worse OOS, five new
directions were explored that change the *question* rather than the
estimator. Full detail in the per-direction docs linked below.

| # | Direction | Doc | Verdict |
|---|---|---|---|
| 1 | Cross-sectional market-neutral target | `cross_sectional_causal.md` | **Null.** Joint L/S Sharpe 0.97 but NW t=1.13, placebo p=0.14; every single factor alone is negative — the joint number is a lucky covariance window. |
| 2 | Event studies on on-chain shocks | `event_studies.md` | **Null.** 4/64 return cells at p<0.05 (~3 expected by chance); no vol effects vs block-bootstrap null. The only suggestive cells are next-day bounces after stress events — conditioning, not causation. |
| 3 | Causal discovery (PC + VAR-LiNGAM) | `causal_discovery.md` | **Structural answer.** Zero stable lagged factor→return edges by either method. The only stable cross-edge is `ew_return → liq_flow` (lag 1): returns drive next-day DeFi flows — the *reverse* of the CPCM premise. |
| 4 | Double ML effect estimates | `dml_effects.md` | **Mostly null, one survivor with caveats.** 4/28 significant at plain 95% (~1.4 expected). `chain_congestion → btc_next` (+41 bps/σ, CI [+18,+64]) survives Bonferroni — but it is a persistent regressor (HAC-naive CIs are optimistic) and its own trading backtests are negative. The macro "hits" (cpiaucsl, t10y2y) are nonstationary levels — trend artifacts. |
| 5 | Vol effects + vol targeting | `vol_effects.md` | **Statistically real, practically subsumed.** VIX/t10y2y/liq_flow add HAC-significant vol information beyond trailing vol, and factor-augmented vol targeting shows ΔSharpe +0.50 (BTC) — but the circular-shift placebo kills it (p≈0.43–0.47): any time-varying leverage overlay scores in this window. |

## The synthesis

The five directions converge on one structural picture, articulated most
sharply by Direction 3:

**In this panel, prices are causally upstream of on-chain activity, not
downstream.** Returns drive next-day liquidity flows; no factor→return edge
replicates across sample halves; the orthogonalized daily effect of every
on-chain factor on next-day returns is statistically zero (Direction 4's one
survivor doesn't survive its own caveats or translate into a backtest). The
factors are *consequences* dressed as *causes* — which is why fourteen
estimation methods, strong instruments, market-neutralization, event
framing, and DML all return the same null.

What this implies for the project:

1. **The CPCM premise needs inversion, not better estimation.** A model where
   returns cause on-chain state can still be useful — for *risk* (Direction
   5's Part 1 is real: factor levels carry vol information) and for *state
   estimation* (knowing where flows will go after a price move) — but not
   for daily return prediction from these factors.
2. **The honest-evaluation toolkit is now built and reusable**: circular-shift
   placebos (`cross_sectional.py`, `vol_effects.py`), block-bootstrap event
   nulls (`event_studies.py`), train-window-only selection (`iv_search.py`),
   per-fold gates. Any future "win" must clear these before it's believed —
   in this session alone they correctly executed two seductive false
   positives (the L/S Sharpe 0.97 and the vol-targeting ΔSharpe +0.50).
3. **Remaining genuinely-open avenues** (not explored here): intraday
   alignment (8h funding → next-8h returns, where simultaneity is weaker);
   exogenous calendars (token unlocks, FOMC/CPI surprise days) as cleaner
   natural experiments than warehouse-derived shocks; and longer horizons
   (weekly/monthly factor effects), which need more history than 2022–2025
   provides for honest folds.

**The buy-and-hold BTC verdict stands across all five directions.**
