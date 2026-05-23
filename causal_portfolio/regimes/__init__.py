"""Market regime classification for CPCM.

Currently houses a Gaussian HMM classifier used by the Phase A diagnostic
in `diagnostics/regime_stability.py`. Production regime-conditional selection
(Phase B) would build on this.
"""

from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    build_regime_features,
    dwell_stats,
    rolling_fit_decode,
)

__all__ = [
    "RegimeClassifier",
    "build_regime_features",
    "dwell_stats",
    "rolling_fit_decode",
]
