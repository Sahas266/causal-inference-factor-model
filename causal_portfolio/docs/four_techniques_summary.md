# Four Techniques: PCA, Cointegration, Correlation (HRP), Wasserstein Regimes

Summary of an attempt to improve on the standing result (nothing beats
buy-and-hold BTC out of sample) using four additional techniques, including the
regime-ID method from Horvath, Issa & Muguruza (2021), *Clustering Market
Regimes using the Wasserstein Distance* (SSRN 3947905).

All four are evaluated the same honest way: walk-forward, scored against
buy-and-hold BTC via `validation/walk_forward.py` (per-fold win-rate +
multiple-testing note), on `btc..doge`-class universes over 2022–2025.

| Technique | Module / experiment | Result doc | OOS vs BH BTC |
|---|---|---|---|
| **PCA drivers** | `analysis/pca.py`, `experiments/pca_drivers.py` | `pca_drivers.md` | Tied raw factors, both lose to BH |
| **Cointegration stat-arb** | `analysis/cointegration.py`, `experiments/statarb.py` | `cointegration_statarb.md` | Market-neutral, gross Sharpe 0.39, no edge |
| **Correlation / HRP** | `analysis/correlation.py`, `experiments/hrp_alloc.py` | `hrp_allocation.md` | Best long-only diversifier, still loses to BH |
| **Wasserstein regimes** | `regimes/wkmeans.py`, `experiments/wkmeans_regime.py` | `wkmeans_regime.md` | Better detector than HMM; conditioning still doesn't beat BH |

## 1. PCA-orthogonalized drivers

**Idea:** the 14 CPCM factors are collinear (high VIF), which destabilizes the
per-asset loadings. Rotate them onto principal components (causal, fit on the
trailing window) and regress on those instead.

**Result:** Raw factors +55.7% / Sharpe 0.71; PCA drivers +28.2% / 0.42; BH BTC
+291.8% / 1.13. Fold win-rate vs BH 44% for both. **PCA did not help** — it lost
signal compressing variance, and collinearity was not the binding constraint.
Neither beats BH.

## 2. Cointegration stat-arb (the most promising — market-neutral)

**Idea:** the one strategy class that is *not* a bet on BTC going up. Engle-
Granger pair selection (ADF-gated) each refit, banded spread mean-reversion,
dollar-neutral, equal-weighted across active pairs.

**Result:** ~8 cointegrated pairs/refit (e.g. `avax~crv`, `doge~avax`,
`bnb~btc`). Daily-return correlation to BTC **+0.09** (genuinely market-neutral).
But full-window gross Sharpe **0.39** (no costs charged), +16.1% total. The big
gap between median fold Sharpe (1.39) and full-window Sharpe (0.39) shows
good folds offset by autocorrelated drawdowns — inconsistent, not a stable edge.
After realistic spread-trading costs it would be ~0. **No tradeable edge as-is**,
though the near-zero BTC correlation is the only genuinely interesting property
and the diversification angle merits a costed, vol-targeted refinement.

## 3. Correlation / Hierarchical Risk Parity

**Idea:** isolate the portfolio-construction lever — weight the same long crypto
universe by its correlation structure (HRP: correlation-distance clustering →
quasi-diagonalization → recursive bisection), no return forecast.

**Result:** HRP +281.4% / Sharpe 0.976 / MaxDD −46.3%; Equal-Weight +150.9% /
0.747 / −54.8%; BH BTC +302.6% / 1.127 / −32.1%. HRP is the **best long-only
diversifier** (beats equal-weight clearly, 56% vs 33% fold win-rate) and nearly
matches BH's total return — but loses on Sharpe AND drawdown. The universe is a
high-correlation, BTC-led basket; smarter weighting reduces neither the
correlation to BTC nor the gap to just holding BTC.

## 4. Wasserstein k-means regimes (the paper's method)

**Idea:** replace the Gaussian HMM regime detector with WK-means — cluster the
*distributions* of short BTC return segments under the p-Wasserstein distance
(model-free, no Markov/Gaussian assumptions). Use the regimes to condition the
factor loadings.

Port (`regimes/wkmeans.py`) is faithful to the paper: stream-lift segments,
W_p via sorted order statistics (eq 21), Wasserstein barycenter = per-order-
statistic median (Prop 2.6), k-means loop, MMD self-similarity validation.

**Result — the most interesting of the four:** WK-means is **materially the
better regime detector**. Regime-conditioning with Wasserstein labels preserved
+35.2% / Sharpe 0.625 (median fold Sharpe 1.145), versus the HMM's −5.9% /
0.040 (median fold 0.137). So the paper's claim holds here: the distributional
detector beats the HMM. **But regime conditioning itself still does not pay off**
— both regime arms trail the pooled (no-regime) model (+59.6% / 0.904, median
fold 1.222), all three tie at 57% fold win-rate vs BH, and none beats BH BTC
(+238.5% / 1.245). Splitting the estimation sample by regime costs more data
than the better detector recovers.

## Overall conclusion

A better regime detector (WK-means), a genuinely market-neutral sleeve
(cointegration), cleaner drivers (PCA), and smarter diversification (HRP) — and
**still nothing beats buy-and-hold BTC out of sample.** Each technique improved
the *thing it targets* (WK-means > HMM at regime ID; HRP > equal-weight at
diversification), but none changed the headline. The two with standalone merit
worth further (costed) work are **cointegration stat-arb** (uncorrelated with
BTC — diversification value even at low standalone Sharpe) and **WK-means**
(a strictly better regime model to swap in wherever regimes are used).

### Where each could still be pushed
- **Cointegration:** add a transaction-cost model + vol-targeting; filter on
  *pair stability* (rolling half-life, persistence of the ADF rejection) rather
  than a single in-window test; size by spread half-life. Test as a *sleeve*
  blended with a BTC core, judged on the blend's Sharpe/drawdown, not on beating
  BTC outright.
- **WK-means:** swap it in for the HMM in the dashboard Regimes tab and anywhere
  regimes drive decisions; try the sliced/multivariate Wasserstein on a
  (BTC-return, VIX) joint segment instead of BTC returns alone.
- **PCA:** try supervised dimension reduction (PLS) — components chosen for
  return-predictiveness, not just factor variance.
- **HRP:** apply HRP *within* a market-neutral or regime sleeve, not to the raw
  long basket where BTC dominance is unavoidable.
