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
from causal_portfolio.regimes.wkmeans import (
    WassersteinKMeans,
    mmd_self_similarity,
    rolling_fit_label,
    segment_stream,
    wasserstein_distance_sorted,
)

__all__ = [
    "RegimeClassifier",
    "build_regime_features",
    "dwell_stats",
    "rolling_fit_decode",
    "WassersteinKMeans",
    "mmd_self_similarity",
    "rolling_fit_label",
    "segment_stream",
    "wasserstein_distance_sorted",
]
