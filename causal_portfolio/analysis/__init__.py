"""Statistical analysis tools layered on the CPCM data.

Modules:
  - cointegration: Engle-Granger / Johansen cointegration + spread signals
    (market-neutral stat-arb).
  - pca: PCA-orthogonalized drivers.
  - correlation: correlation-distance clustering + Hierarchical Risk Parity.

These are exploratory levers tested against the project's standing benchmark
(buy-and-hold BTC) via the walk-forward harness.
"""
