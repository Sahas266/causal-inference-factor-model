"""CPCM structural causal model layer.

Pure-Python ports of the Rust cpcm-core / cpcm-estimate engine:
  - graph: the CPCM star-DAG (networkx)
  - identification: d-separation, backdoor sets, IV validity (port of
    cpcm-core dsep.rs + identify.rs)
  - estimators: OLS / 2SLS / diagnostics (port of cpcm-estimate)

These give the Python backtest genuine causal identification + 2SLS, validated
against the Rust reference fixtures (see tests/test_estimators.py,
tests/test_identification.py).
"""

from causal_portfolio.scm.estimators import (
    Diagnostics,
    OlsResult,
    TslsResult,
    add_intercept,
    compute_diagnostics,
    ols,
    project_onto,
    tsls,
)
from causal_portfolio.scm.graph import (
    EdgeKind,
    NodeKind,
    build_cpcm_dag,
    summarize_dag,
)
from causal_portfolio.scm.identification import (
    IdentificationMethod,
    IdentificationResult,
    IvValidityResult,
    backdoor_adjustment_set,
    check_iv_validity,
    d_connected_set,
    d_separated,
    identify_all_effects,
)

__all__ = [
    "Diagnostics", "OlsResult", "TslsResult",
    "add_intercept", "compute_diagnostics", "ols", "project_onto", "tsls",
    "EdgeKind", "NodeKind", "build_cpcm_dag", "summarize_dag",
    "IdentificationMethod", "IdentificationResult", "IvValidityResult",
    "backdoor_adjustment_set", "check_iv_validity", "d_connected_set",
    "d_separated", "identify_all_effects",
]
