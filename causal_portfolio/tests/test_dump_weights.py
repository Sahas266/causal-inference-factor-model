"""Tests for the pipeline → weights JSON dumper.

Unit-tests the `_dump_weights` helper without running a full backtest by
constructing a synthetic BacktestResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest


@dataclass
class FakeResult:
    """Minimal stand-in for BacktestResult with just what _dump_weights reads."""
    weights_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rebalance_dates: list = field(default_factory=list)


def test_dump_weights_writes_valid_executor_json(tmp_path):
    from causal_portfolio.run_backtest import _dump_weights

    result = FakeResult(
        weights_history=np.array([
            [0.10, 0.20, 0.05],
            [0.30, 0.25, -0.10],   # most recent
        ]),
        rebalance_dates=["2025-01-01", "2025-01-06"],
    )
    out = tmp_path / "w.json"
    _dump_weights(
        result=result, assets=["btc", "eth", "sol"],
        returns_cols=["btc_return", "eth_return", "sol_return"],
        out_path=str(out),
        metadata={"solver": "v1", "m": 3, "use_ekf": True,
                  "start": "2022-01-01", "end": "2025-12-31",
                  "selected_drivers": ["stable_flow", "funding_basis"],
                  "rebalance_freq": 5},
    )

    raw = json.loads(out.read_text())
    # Asset weights from the LAST row of weights_history
    assert raw["btc"] == pytest.approx(0.30)
    assert raw["eth"] == pytest.approx(0.25)
    assert raw["sol"] == pytest.approx(-0.10)
    # Provenance
    assert raw["_meta"]["solver"] == "v1"
    assert raw["_meta"]["rebalance_date"] == "2025-01-06"
    assert raw["_meta"]["gross_exposure"] == pytest.approx(0.65)
    assert raw["_meta"]["n_nonzero"] == 3


def test_dump_weights_drops_zero_weights(tmp_path):
    from causal_portfolio.run_backtest import _dump_weights

    result = FakeResult(
        weights_history=np.array([[0.5, 0.0, 0.5]]),
        rebalance_dates=["2025-01-01"],
    )
    out = tmp_path / "w.json"
    _dump_weights(result=result, assets=["btc", "eth", "sol"],
                  returns_cols=["btc_return", "eth_return", "sol_return"],
                  out_path=str(out), metadata={})

    raw = json.loads(out.read_text())
    assert "btc" in raw and "sol" in raw
    assert "eth" not in raw  # zero weight dropped
    assert raw["_meta"]["n_nonzero"] == 2


def test_dump_weights_handles_empty_result(tmp_path, caplog):
    from causal_portfolio.run_backtest import _dump_weights

    result = FakeResult()
    out = tmp_path / "w.json"
    _dump_weights(result=result, assets=[], returns_cols=[],
                  out_path=str(out), metadata={})
    assert not out.exists()  # nothing written


def test_dump_weights_handles_length_mismatch(tmp_path, caplog):
    from causal_portfolio.run_backtest import _dump_weights

    result = FakeResult(
        weights_history=np.array([[0.5, 0.5]]),  # 2 weights
        rebalance_dates=["2025-01-01"],
    )
    out = tmp_path / "w.json"
    _dump_weights(result=result, assets=["btc", "eth", "sol"],  # 3 assets
                  returns_cols=["btc_return", "eth_return", "sol_return"],
                  out_path=str(out), metadata={})
    assert not out.exists()  # skipped due to mismatch


# ── CLI _load_weights now strips _meta ──────────────────────────────


def test_cli_load_weights_skips_meta_field(tmp_path):
    """The CLI loader must accept dumper output (which includes _meta)."""
    from causal_portfolio.execution.cli import _load_weights

    payload = {
        "btc": 0.3, "eth": 0.2,
        "_meta": {"solver": "v1", "n_nonzero": 2, "selected_drivers": ["x"]},
    }
    f = tmp_path / "w.json"
    f.write_text(json.dumps(payload))
    weights = _load_weights(str(f))
    assert weights == {"btc": 0.3, "eth": 0.2}
    assert "_meta" not in weights


def test_cli_load_weights_still_rejects_non_numeric_real_keys(tmp_path):
    """Only underscore-prefixed keys are exempt from numeric validation."""
    from causal_portfolio.execution.cli import _load_weights

    payload = {"btc": "not a number"}
    f = tmp_path / "w.json"
    f.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="numeric"):
        _load_weights(str(f))


def test_cli_load_weights_dumper_roundtrip(tmp_path):
    """The CLI loader can consume what _dump_weights writes."""
    from causal_portfolio.execution.cli import _load_weights
    from causal_portfolio.run_backtest import _dump_weights

    result = FakeResult(
        weights_history=np.array([[0.3, 0.2, -0.1]]),
        rebalance_dates=["2025-01-01"],
    )
    out = tmp_path / "w.json"
    _dump_weights(result=result, assets=["btc", "eth", "sol"],
                  returns_cols=["btc_return", "eth_return", "sol_return"],
                  out_path=str(out), metadata={"solver": "v1"})

    weights = _load_weights(str(out))
    assert weights == {"btc": 0.3, "eth": 0.2, "sol": -0.1}
