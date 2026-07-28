"""CPCM Execution Dashboard — strategy → weights → live Hyperliquid account.

A focused Streamlit app that visualizes the deployable-strategy pipeline:

  1. Pick a strategy and date range. See its backtest equity curve vs the
     buy-and-hold-BTC benchmark, with cost-adjusted metrics.
  2. See the strategy's CURRENT target weights (what it would hold today).
  3. See the LIVE Hyperliquid testnet account (equity, open positions).
  4. Preview the rebalance plan (dry-run): exactly what orders would fire to
     move the live account to the target weights. Read-only — nothing submits.
  5. Browse the audit log of past executions.

Honest framing baked in: buy-and-hold BTC is the default because it is the
one thing in our research that robustly beat everything else. The backtest
panel shows the numbers so the strategy choice is transparent.

The data functions live in causal_portfolio.execution_dashboard_data (no
Streamlit dependency, unit-tested). This file is UI only.

Run:
    streamlit run causal_portfolio/execution_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from causal_portfolio import execution_dashboard_data as dd

st.set_page_config(page_title="CPCM Execution", page_icon="⚡", layout="wide")

DEFAULT_ASSETS = ("btc", "eth", "sol")

# Cache the expensive backtest + data load
run_strategy = st.cache_data(show_spinner="Running backtest…", ttl=3600)(dd.run_strategy)
# Cache live account reads briefly so every widget interaction doesn't
# re-hit the Hyperliquid API (3 network calls per rerun otherwise).
# Errors are raised inside the cached function so st.cache_data never
# stores a failure (a cached error tuple would poison reads for the TTL).


@st.cache_data(show_spinner=False, ttl=30)
def _fetch_live_account_ok(testnet: bool):
    state, mids, err = dd.fetch_live_account(testnet=testnet)
    if err:
        raise RuntimeError(err)
    return state, mids


def fetch_live_account(testnet: bool):
    try:
        state, mids = _fetch_live_account_ok(testnet=testnet)
        return state, mids, None
    except Exception as e:
        return None, None, str(e)


st.title("⚡ CPCM Execution Dashboard")
st.caption("Strategy → target weights → live Hyperliquid account. Testnet by default.")

with st.sidebar:
    st.header("Configuration")
    strategy = st.selectbox(
        "Strategy",
        ["Buy & Hold BTC", "Fixed 60/30/10 BTC/ETH/SOL", "Equal-weight basket"],
        help="Buy & Hold BTC is the research benchmark nothing else beat.",
    )
    start = st.text_input("Start date", "2023-06-11")
    end = st.text_input("End date", "2025-12-31")
    rebalance_freq = st.slider("Rebalance frequency (days)", 1, 63, 21)
    fee_bps = st.number_input("Fee (bps)", 0.0, 50.0, 5.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 50.0, 5.0)
    network = st.radio("Network", ["testnet", "mainnet"], index=0)

# ── Backtest panel ──────────────────────────────────────────────────

st.subheader("📈 Backtest vs Buy & Hold BTC")
try:
    res = run_strategy(strategy, DEFAULT_ASSETS, start, end,
                       fee_bps, slippage_bps, rebalance_freq)
except Exception as e:
    st.error(f"Backtest failed: {e}")
    st.stop()

sm, bm = res["strat_metrics"], res["bh_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total return", f"{sm['total']:+.1%}", f"{sm['total']-bm['total']:+.1%} vs BH")
c2.metric("Sharpe", f"{sm['sharpe']:.3f}", f"{sm['sharpe']-bm['sharpe']:+.3f} vs BH")
# max_dd is negative; a higher (closer-to-zero) delta is better, so the
# default delta coloring is already correct.
c3.metric("Max drawdown", f"{sm['max_dd']:+.1%}", f"{sm['max_dd']-bm['max_dd']:+.1%} vs BH")
c4.metric("Rebalances", f"{sm['rebalances']}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=res["dates"], y=res["strat_curve"], name=strategy,
                         line=dict(width=2)))
fig.add_trace(go.Scatter(x=res["dates"], y=res["bh_curve"], name="Buy & Hold BTC",
                         line=dict(width=1.5, dash="dot")))
fig.update_layout(height=380, yaxis_title="Growth of $1", template="plotly_dark",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

if sm["total"] < bm["total"]:
    st.info(f"ℹ️ **{strategy}** underperforms Buy & Hold BTC over this window "
            f"({sm['total']:+.1%} vs {bm['total']:+.1%}). BH BTC remains the "
            f"benchmark to beat. This strategy is shown for diversification / "
            f"multi-leg execution demonstration.")

# ── Target weights ──────────────────────────────────────────────────

st.subheader("🎯 Current Target Weights")
tw = res["target_weights"]
wc1, wc2 = st.columns([1, 2])
with wc1:
    st.dataframe(pd.DataFrame(
        [{"Asset": k.upper(), "Weight": f"{v:.1%}"} for k, v in tw.items()],
    ), hide_index=True, use_container_width=True)
with wc2:
    bar = go.Figure(go.Bar(x=[k.upper() for k in tw], y=list(tw.values()),
                           marker_color="#00cc96"))
    bar.update_layout(height=240, yaxis_title="Weight", template="plotly_dark",
                      yaxis_tickformat=".0%")
    st.plotly_chart(bar, use_container_width=True)

# ── Live account ────────────────────────────────────────────────────

st.subheader(f"💰 Live {network.title()} Account")
state, mids, err = fetch_live_account(testnet=(network == "testnet"))
if err:
    st.warning(f"Could not reach Hyperliquid ({network}): {err}")
elif state is not None:
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Equity", f"${state.account_value_usd:,.2f}")
    ac2.metric("Margin used", f"${state.margin_used_usd:,.2f}")
    ac3.metric("Open positions", f"{len(state.positions)}")
    if state.positions:
        pos_rows = []
        for coin, p in state.positions.items():
            mark = (mids or {}).get(coin, p.entry_px)
            pos_rows.append({
                "Coin": coin, "Side": "LONG" if p.is_long else "SHORT",
                "Size": f"{p.size:+.6f}", "Entry": f"{p.entry_px:,.2f}",
                "Mark": f"{mark:,.2f}", "Notional": f"${p.notional_usd:+,.2f}",
            })
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("Account is flat (no open positions).")

    # ── Plan preview ────────────────────────────────────────────────
    st.subheader("📋 Rebalance Plan Preview (dry-run)")
    st.caption("What orders would fire to move the live account to the target "
               "weights. Read-only — nothing is submitted.")
    if st.button("Compute plan"):
        try:
            plan = dd.build_plan_preview(tw, testnet=(network == "testnet"))
            st.text(plan.summary())
            if plan.orders:
                order_rows = [{
                    "Coin": o.coin, "Side": "BUY" if o.is_buy else "SELL",
                    "Size": f"{o.size:.6f}", "Limit Px": f"{o.limit_px:,.2f}",
                    "Notional": f"${o.size * o.limit_px:,.2f}",
                } for o in plan.orders]
                st.dataframe(pd.DataFrame(order_rows), hide_index=True,
                             use_container_width=True)
            else:
                st.success("Already on target — no orders needed.")
            if plan.skipped:
                st.caption("Skipped: " + "; ".join(
                    f"{c} ({r.value})" for c, r, _ in plan.skipped))
        except Exception as e:
            st.error(f"Plan preview failed: {e}")

# ── Audit history ───────────────────────────────────────────────────

st.subheader("📜 Recent Execution History (audit log)")
st.caption(f"Filtered to **{network}** (records with unknown network shown as '?').")
try:
    records = dd.load_audit_history(n_days=7, network=network)
    if records:
        hist_rows = []
        for rec in records:
            plan = rec.get("plan") or {}
            n_orders = len(plan.get("orders") or [])
            gross = sum(abs(v) for v in (plan.get("deltas_usd") or {}).values())
            hist_rows.append({
                "Time (UTC)": rec.get("ts_utc", "?")[:19],
                "Network": rec.get("network") or "?",
                "Submitted": "✓" if rec.get("submitted") else "—",
                "Orders": n_orders,
                "Gross $": f"${gross:,.0f}",
                "Error": (rec.get("error") or "")[:40],
            })
        st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
    else:
        st.caption(f"No {network} execution history in the last 7 days.")
except Exception as e:
    st.caption(f"Could not load audit history: {e}")
