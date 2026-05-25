"""Tests for production-fallback in run_regime_weights.compute_current_regime."""

from __future__ import annotations

import json

import pytest

from causal_portfolio.run_regime_weights import (
    _stress_fallback,
    _read_last_state,
    _write_last_state,
    compute_current_regime,
    regime_to_weights,
)


def test_stress_fallback_returns_cash_target():
    info = _stress_fallback(n_states=2, reason="test")
    assert info["regime"] == 1
    assert info["source"] == "fallback_stress"
    weights = regime_to_weights(info["regime"], ["btc", "eth", "sol"])
    # Stress regime → all zero
    assert all(v == 0.0 for v in weights.values())


def test_state_file_round_trip(tmp_path):
    state_file = tmp_path / "state.json"
    info = {"regime": 0, "posterior": [0.9, 0.1], "as_of": "2026-01-01"}
    _write_last_state(state_file, info)
    loaded = _read_last_state(state_file)
    assert loaded == info


def test_read_missing_state_returns_none(tmp_path):
    assert _read_last_state(tmp_path / "nope.json") is None


def test_read_corrupt_state_returns_none(tmp_path):
    state_file = tmp_path / "bad.json"
    state_file.write_text("not json")
    assert _read_last_state(state_file) is None


def test_fallback_uses_last_known_on_failure(tmp_path, monkeypatch):
    """If HMM fit raises, we should fall back to last-known state."""
    state_file = tmp_path / "state.json"
    last_known = {
        "regime": 0, "posterior": [0.8, 0.2], "as_of": "2026-01-15",
        "feature_columns": ["vix", "btc_vol"], "hmm_window": 504, "n_states": 2,
        "source": "live",
    }
    _write_last_state(state_file, last_known)

    # Mock the live path to always raise
    import causal_portfolio.run_regime_weights as mod
    monkeypatch.setattr(mod, "_compute_regime_live",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("HMM died")))

    result = compute_current_regime(
        assets=["btc"], fallback_state_file=state_file,
    )
    assert result["source"] == "fallback_last_known"
    assert result["regime"] == 0
    assert result["fallback_reason"] == "HMM died"


def test_fallback_to_stress_when_no_state_file(tmp_path, monkeypatch):
    """If HMM fails AND no state file exists, default to stress regime."""
    import causal_portfolio.run_regime_weights as mod
    monkeypatch.setattr(mod, "_compute_regime_live",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("data load died")))

    result = compute_current_regime(
        assets=["btc"], fallback_state_file=tmp_path / "nonexistent.json",
    )
    assert result["source"] == "fallback_stress"
    assert result["regime"] == 1  # stress
    assert "data load died" in result["fallback_reason"]


def test_on_failure_raise_propagates(tmp_path, monkeypatch):
    """on_failure='raise' should re-raise the exception."""
    import causal_portfolio.run_regime_weights as mod
    monkeypatch.setattr(mod, "_compute_regime_live",
                        lambda **kwargs: (_ for _ in ()).throw(ValueError("debug me")))

    with pytest.raises(ValueError, match="debug me"):
        compute_current_regime(
            assets=["btc"], fallback_state_file=tmp_path / "x.json",
            on_failure="raise",
        )


def test_on_failure_stress_always_stress(tmp_path, monkeypatch):
    """on_failure='stress' should skip the last-known lookup."""
    state_file = tmp_path / "state.json"
    _write_last_state(state_file, {
        "regime": 0, "posterior": [0.9, 0.1], "as_of": "2026-01-15",
    })

    import causal_portfolio.run_regime_weights as mod
    monkeypatch.setattr(mod, "_compute_regime_live",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("x")))

    result = compute_current_regime(
        assets=["btc"], fallback_state_file=state_file, on_failure="stress",
    )
    # Even though state file has regime=0, we went to stress
    assert result["source"] == "fallback_stress"
    assert result["regime"] == 1


def test_live_path_persists_state(tmp_path, monkeypatch):
    """On a successful live computation, state should be written to disk."""
    state_file = tmp_path / "state.json"
    fake_info = {
        "regime": 0, "posterior": [0.7, 0.3], "as_of": "2026-05-25",
        "feature_columns": ["vix", "btc_vol"], "hmm_window": 504, "n_states": 2,
    }
    import causal_portfolio.run_regime_weights as mod
    monkeypatch.setattr(mod, "_compute_regime_live", lambda **kwargs: dict(fake_info))

    result = compute_current_regime(
        assets=["btc"], fallback_state_file=state_file,
    )
    assert result["source"] == "live"
    assert state_file.exists()
    saved = json.loads(state_file.read_text())
    assert saved["regime"] == 0
