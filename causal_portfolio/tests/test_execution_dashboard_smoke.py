"""Smoke tests for the execution dashboard data functions.

These exercise causal_portfolio.execution.dashboard_data against the local
DuckDB and mocked adapters. No Streamlit runtime needed — the data functions
live in a plain module the UI imports.
"""

from __future__ import annotations

import pytest

from causal_portfolio.execution import dashboard_data as dd


def test_run_strategy_bh_btc():
    res = dd.run_strategy("Buy & Hold BTC", ("btc", "eth", "sol"),
                          "2023-06-11", "2025-12-31", 5.0, 5.0, 21)
    assert "strat_curve" in res
    assert len(res["strat_curve"]) > 100
    assert res["target_weights"] == {"btc": 1.0}
    # For BH BTC the strategy curve IS the benchmark curve
    assert res["strat_metrics"]["total"] == pytest.approx(res["bh_metrics"]["total"], abs=1e-6)


def test_run_strategy_fixed_weight_targets():
    res = dd.run_strategy("Fixed 60/30/10 BTC/ETH/SOL", ("btc", "eth", "sol"),
                          "2023-06-11", "2025-12-31", 5.0, 5.0, 21)
    tw = res["target_weights"]
    assert tw["btc"] == pytest.approx(0.6)
    assert tw["eth"] == pytest.approx(0.3)
    assert tw["sol"] == pytest.approx(0.1)
    # Curve and metrics populated
    assert len(res["dates"]) == len(res["strat_curve"])
    assert res["strat_metrics"]["rebalances"] > 0


def test_run_strategy_equal_weight():
    res = dd.run_strategy("Equal-weight basket", ("btc", "eth", "sol"),
                          "2023-06-11", "2025-12-31", 5.0, 5.0, 21)
    tw = res["target_weights"]
    assert tw["btc"] == pytest.approx(1 / 3)
    assert len(tw) == 3


def test_fetch_live_account_handles_failure(monkeypatch):
    """fetch_live_account returns an error string instead of raising."""
    import causal_portfolio.execution.hyperliquid as hl

    def boom(cfg):
        raise RuntimeError("no creds / no network")
    monkeypatch.setattr(hl, "HLAdapter", boom)

    state, mids, err = dd.fetch_live_account(testnet=True)
    assert state is None
    assert mids is None
    assert "no creds" in err


def test_load_audit_history_returns_list():
    records = dd.load_audit_history(n_days=3)
    assert isinstance(records, list)
    # Records (if any) should be dicts with expected keys
    for rec in records:
        assert "ts_utc" in rec or "submitted" in rec
