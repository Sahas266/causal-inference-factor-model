"""Data functions for the execution dashboard — no Streamlit dependency.

Kept separate from execution_dashboard.py so they can be unit-tested without
the Streamlit runtime. The dashboard imports these and wraps the expensive
ones in st.cache_data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_market(assets: tuple[str, ...], start: str, end: str):
    """Return (returns, macro) aligned on common dates (macro ffilled).

    Thin wrapper over the shared loader helper. Note: the old local copy
    also dropped all-NaN return rows before aligning; the shared helper
    keeps such rows if macro covers them — shared behavior wins.
    """
    from causal_portfolio.data import load_returns_and_macro
    return load_returns_and_macro(assets, start, end)


def run_strategy(strategy: str, assets: tuple[str, ...], start: str, end: str,
                 fee_bps: float, slippage_bps: float, rebalance_freq: int) -> dict:
    """Run the selected strategy + BH_BTC benchmark; return curves + metrics."""
    from causal_portfolio.backtest.strategies import (
        buy_and_hold, equal_weight_basket, fixed_weight_portfolio,
    )
    returns, _ = load_market(assets, start, end)

    bh = buy_and_hold(returns, "btc_return", fee_bps=fee_bps, slippage_bps=slippage_bps)
    if strategy == "Buy & Hold BTC":
        strat = bh
        target_weights = {"btc": 1.0}
    elif strategy == "Fixed 60/30/10 BTC/ETH/SOL":
        strat = fixed_weight_portfolio(
            returns, fee_bps=fee_bps, slippage_bps=slippage_bps,
            rebalance_freq=rebalance_freq)
        target_weights = {k: v for k, v in {"btc": 0.6, "eth": 0.3, "sol": 0.1}.items()
                          if f"{k}_return" in returns.columns}
    else:  # Equal-weight basket
        strat = equal_weight_basket(
            returns, fee_bps=fee_bps, slippage_bps=slippage_bps,
            rebalance_freq=rebalance_freq)
        n = returns.shape[1]
        target_weights = {c.replace("_return", ""): 1.0 / n for c in returns.columns}

    n_eval = len(strat.returns_series)
    idx = returns.index[-n_eval:]
    strat_curve = np.cumprod(1 + strat.returns_series)
    bh_curve = np.cumprod(1 + bh.returns_series[-n_eval:])

    return {
        "dates": idx,
        "strat_curve": strat_curve,
        "bh_curve": bh_curve,
        "strat_metrics": {
            "total": strat.total_return, "sharpe": strat.sharpe,
            "sortino": strat.sortino, "max_dd": strat.max_dd,
            "calmar": strat.calmar, "turnover": strat.avg_turnover,
            "rebalances": len(strat.rebalance_dates),
        },
        "bh_metrics": {
            # cumprod starts at 1+r[0]; dividing by bh_curve[0] would drop
            # the first day's return (same convention as engine.from_returns).
            "total": float(bh_curve[-1] - 1.0),
            "sharpe": bh.sharpe, "max_dd": bh.max_dd,
        },
        "target_weights": target_weights,
    }


def fetch_live_account(testnet: bool = True):
    """Return (state, mids, error). error is None on success."""
    try:
        from causal_portfolio.execution.config import ExecutionConfig
        from causal_portfolio.execution.hyperliquid import HLAdapter
        cfg = ExecutionConfig(testnet=testnet, dry_run=True)
        adapter = HLAdapter(cfg)
        return adapter.fetch_state(), adapter.fetch_mids(), None
    except Exception as e:
        return None, None, str(e)


def build_plan_preview(target_weights: dict, testnet: bool = True):
    """Dry-run plan: what orders would fire to reach target_weights."""
    from causal_portfolio.execution.config import ExecutionConfig
    from causal_portfolio.execution.hyperliquid import HLAdapter
    from causal_portfolio.execution.rebalancer import plan_rebalance
    # Default caps on purpose: the preview must show what the CLI would
    # actually do, safety caps included.
    cfg = ExecutionConfig(testnet=testnet, dry_run=True)
    adapter = HLAdapter(cfg)
    state = adapter.fetch_state()
    mids = adapter.fetch_mids()
    meta = adapter.fetch_meta()
    return plan_rebalance(target_weights, state, mids, meta, cfg)


def load_audit_history(n_days: int = 7, network: str | None = None) -> list[dict]:
    """Recent audit records, newest days first.

    If `network` is given ("testnet"/"mainnet"), keep only records for that
    network — plus records with no network field (old logs), which are kept
    visible under both.
    """
    from causal_portfolio.execution import audit
    rows: list[dict] = []
    for i in range(n_days):
        date = (datetime.now(timezone.utc) - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows.extend(audit.read_log(date))
    if network is not None:
        rows = [r for r in rows if r.get("network") in (network, None)]
    return rows
