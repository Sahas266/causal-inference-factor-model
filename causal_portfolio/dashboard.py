"""CPCM Quant Dashboard — Streamlit app.

Full-featured dashboard for running and analyzing the Causal PDE-Control Model
portfolio optimization pipeline. Quants can configure every parameter, run
the full Combo-EKF-V1/V4 pipeline, and inspect results interactively.

Usage:
    streamlit run causal_portfolio/dashboard.py
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CPCM Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #0e1117;
    border: 1px solid #262730;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-label { color: #888; font-size: 12px; text-transform: uppercase; }
.metric-value { color: #fff; font-size: 28px; font-weight: 700; margin: 4px 0; }
.metric-delta-pos { color: #00cc88; font-size: 12px; }
.metric-delta-neg { color: #ff4b4b; font-size: 12px; }
.section-header {
    font-size: 14px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 16px 0 8px 0; border-bottom: 1px solid #262730; padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────
ALL_ASSETS = [
    "btc", "eth", "sol", "bnb", "avax", "xrp", "doge",
    "uni", "aave", "link", "crv", "pendle", "morpho", "ena", "aero",
    "usdc", "usdt", "usde",
    "hype", "tao", "wlfi", "jup", "pepe", "shib", "zec", "pol",
]
DEFAULT_ASSETS = ["btc", "eth", "sol", "bnb", "avax", "uni", "aave", "link", "doge"]


# ── Sidebar configuration ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ CPCM Configuration")
    st.markdown("---")

    # ── Asset universe
    st.markdown('<div class="section-header">Asset Universe</div>', unsafe_allow_html=True)
    selected_assets = st.multiselect(
        "Select assets", ALL_ASSETS, default=DEFAULT_ASSETS,
        help="Assets to include in the portfolio universe"
    )
    if not selected_assets:
        selected_assets = DEFAULT_ASSETS

    # ── Date range
    st.markdown('<div class="section-header">Date Range</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2022-01-01"),
                                   min_value=pd.Timestamp("2021-01-01"),
                                   max_value=pd.Timestamp("2025-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2025-12-31"),
                                 min_value=pd.Timestamp("2021-06-01"),
                                 max_value=pd.Timestamp("2026-01-01"))

    # ── Driver selection
    st.markdown('<div class="section-header">Causal Drivers</div>', unsafe_allow_html=True)
    m_drivers = st.slider("Number of drivers (m)", 1, 7, 3,
                          help="Combo selects the best m drivers from 14 candidates")

    # ── Solver
    st.markdown('<div class="section-header">Solver</div>', unsafe_allow_html=True)
    solver_choice = st.radio("Solver variant", ["V1 — Ridge OLS", "V4 — PINN"],
                             help="V1 is fast/linear; V4 is neural with Jacobian smoothness")
    solver_name = "v1" if "V1" in solver_choice else "v4"

    if solver_name == "v1":
        ridge_alpha = st.slider("Ridge α", 0.001, 10.0, 1.0, 0.1,
                                help="Regularization strength")
    else:
        pinn_epochs = st.slider("PINN epochs", 50, 1000, 300, 50)
        pinn_lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
        pinn_lambda_j = st.slider("λ_Jacobian", 0.0, 0.1, 0.01, 0.005,
                                  format="%.3f",
                                  help="Jacobian smoothness penalty weight")
        pinn_hidden = st.select_slider("Hidden dim", [32, 64, 128, 256], value=64)

    # ── EKF
    st.markdown('<div class="section-header">Kalman Filter</div>', unsafe_allow_html=True)
    use_ekf = st.toggle("Enable EKF filtering", value=True,
                        help="VAR(1) Extended Kalman Filter to denoise driver observations")
    obs_noise = st.slider("Observation noise scale", 0.01, 0.5, 0.1, 0.01,
                          disabled=not use_ekf,
                          help="R = scale × diag(var(drivers))")

    # ── Backtest
    st.markdown('<div class="section-header">Backtest Settings</div>', unsafe_allow_html=True)
    train_window = st.slider("Training window (days)", 60, 500, 252, 10)
    rebalance_freq = st.slider("Rebalance frequency (days)", 1, 30, 5,
                               help="How often to recompute weights")

    # ── Optimizer
    st.markdown('<div class="section-header">Optimizer</div>', unsafe_allow_html=True)
    risk_aversion = st.slider("Risk aversion λ", 0.1, 10.0, 1.0, 0.1,
                              help="Mean-variance trade-off: higher = more conservative")
    max_weight = st.slider("Max weight per asset", 0.05, 0.5, 0.25, 0.05,
                           help="Upper bound on |w_i|")
    long_only = st.toggle("Long-only portfolio", value=False)

    # ── Use cached data
    use_cache = st.toggle("Use data cache", value=True,
                          help="Cache Supabase queries to parquet for faster reloads")

    st.markdown("---")
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)


# ── Main layout ───────────────────────────────────────────────────────
st.title("📊 CPCM Quant Dashboard")
st.caption("Causal PDE-Control Model · Combo-EKF-V1/V4 · arXiv:2509.09585v2")

# Tabs
tabs = st.tabs([
    "📋 Overview",
    "📈 Factors & Drivers",
    "🕸️ Causal DAG",
    "🔬 Solver",
    "📉 Backtest",
    "🔍 EKF Filter",
    "🔀 Regimes",
])
t_overview, t_factors, t_dag, t_solver, t_backtest, t_ekf, t_regimes = tabs


# ── Session state ─────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None


# ── Helper: plotly dark theme ─────────────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
)


def _metric(label: str, value: str, delta: str | None = None, good_positive: bool = True):
    """Render a styled metric card."""
    if delta:
        delta_class = "metric-delta-pos" if (
            ("+" in str(delta) and good_positive) or
            ("-" in str(delta) and not good_positive)
        ) else "metric-delta-neg"
        delta_html = f'<div class="{delta_class}">{delta}</div>'
    else:
        delta_html = ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _load_data(assets, start, end, use_cache_flag):
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS,
    )
    loader = get_loader()
    panel_assets = list(dict.fromkeys(list(assets) + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, start, end,
                              use_cache=use_cache_flag)
    returns = loader.load_returns(list(assets), start, end, use_cache=use_cache_flag)
    macro = loader.load_macro(MACRO_SERIES, start, end, use_cache=use_cache_flag)
    return panel, returns, macro


def run_pipeline():
    """Run the full CPCM pipeline and store results in session state."""
    start_str = str(start_date)
    end_str = str(end_date)

    with st.spinner("Loading data from Supabase..."):
        try:
            panel, returns, macro = _load_data(
                tuple(selected_assets), start_str, end_str, use_cache
            )
        except Exception as e:
            st.error(f"Data loading failed: {e}\n\n{traceback.format_exc()}")
            return

    if panel.empty or returns.empty:
        st.error("No data returned. Check Supabase connection and asset selection.")
        return

    with st.spinner("Computing factors..."):
        from causal_portfolio.factors.builder import build_all_factors
        factors = build_all_factors(panel, macro)
        available = factors.dropna(axis=1, how="all")

    with st.spinner(f"Running Combo driver selection (m={m_drivers})..."):
        from causal_portfolio.factors.combo_selector import ComboDriverSelector
        selector = ComboDriverSelector()
        m_actual = min(m_drivers, len(available.columns))
        ranking = selector.rank_all_subsets(returns, available, m=m_actual)
        selected_drivers = list(ranking[0][0])

    with st.spinner("Building Star-DAG..."):
        from causal_portfolio.scm.graph import build_cpcm_dag, summarize_dag
        dag = build_cpcm_dag(selected_assets, selected_drivers)
        dag_summary = summarize_dag(dag)

    # Align data
    common = (
        factors[selected_drivers].dropna().index
        .intersection(returns.dropna().index)
    )
    D = factors.loc[common, selected_drivers].values
    R = returns.loc[common].values
    dates = common.values

    if len(D) <= train_window:
        st.error(f"Not enough data after alignment: {len(D)} rows ≤ train_window={train_window}")
        return

    with st.spinner(f"Running {solver_choice} backtest..."):
        from causal_portfolio.backtest.engine import CPCMBacktester
        from causal_portfolio.optimizer.manifold import ManifoldOptimizer
        from causal_portfolio.solvers.v1_linear import V1LinearSolver

        if solver_name == "v4":
            from causal_portfolio.solvers.v4_pinn import V4PINNSolver
            solver = V4PINNSolver(
                m_drivers=m_actual, n_assets=R.shape[1],
                hidden_dim=pinn_hidden, n_layers=3,
            )
            fit_kwargs = dict(n_epochs=pinn_epochs, lr=pinn_lr, lambda_J=pinn_lambda_j)
        else:
            solver = V1LinearSolver(alpha=ridge_alpha)
            fit_kwargs = {}

        optimizer = ManifoldOptimizer(
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            long_only=long_only,
        )

        # Fit solver on full data for diagnostics
        solver.fit(D, R, **fit_kwargs) if fit_kwargs else solver.fit(D, R)

        backtester = CPCMBacktester(
            solver=solver, optimizer=optimizer,
            use_ekf=use_ekf, rebalance_freq=rebalance_freq,
            train_window=train_window,
        )
        result = backtester.run(R, D, dates)

    # EKF diagnostics
    ekf_data = None
    if use_ekf:
        with st.spinner("Computing EKF diagnostics..."):
            from causal_portfolio.filters.ekf import CPCMKalmanFilter
            ekf = CPCMKalmanFilter(m=m_actual)
            ekf.fit_dynamics(D, obs_noise_scale=obs_noise)
            filtered, covs = ekf.filter(D)
            innov_diag = ekf.innovation_diagnostics(D)
            ekf_data = {
                "filtered": filtered,
                "covs": covs,  # (T, m, m) state covariance — needed for trace(P_t) plot
                "raw": D,
                "driver_names": selected_drivers,
                "diagnostics": innov_diag,
            }

    # Combo ranking
    return_cols = list(returns.columns)
    asset_names = [c.replace("_return", "") for c in return_cols]

    st.session_state.pipeline_data = {
        "factors": factors,
        "available_factors": available,
        "returns": returns.loc[common],
        "drivers": pd.DataFrame(D, index=common, columns=selected_drivers),
        "panel": panel,
        "macro": macro,
        "selected_drivers": selected_drivers,
        "ranking": ranking,
        "dag": dag,
        "dag_summary": dag_summary,
        "solver": solver,
        "solver_name": solver_name,
        "result": result,
        "ekf_data": ekf_data,
        "asset_names": asset_names,
        "dates": common,
        "D": D,
        "R": R,
        "m_actual": m_actual,
    }
    st.session_state.result = result
    st.success("Pipeline complete!")
    st.rerun()


if run_btn:
    run_pipeline()


# ── Overview tab ──────────────────────────────────────────────────────
with t_overview:
    if st.session_state.result is None:
        st.info("Configure the pipeline in the sidebar and click **Run Pipeline** to start.")
        # Show pipeline diagram
        st.markdown("### CPCM Pipeline Architecture")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
**Causal PDE-Control Model** (arXiv:2509.09585v2) — best config: Combo-EKF-V4, m=3 → Sharpe 1.266

```
Supabase Data (430K+ rows)
       │
       ▼
 Factor Builder          7 global + 7 macro candidate factors
       │
       ▼
 Combo Selection         Minimizes residual commonality λ_max(Σ_ε)
       │                 C(14,3)=364 subsets evaluated
       ▼
  Star-DAG (SCM)         Z_t → F_t → A_t → p_t
       │                 IV nodes: gas_spike, stablecoin_mint, etc.
       ▼
  EKF Filter             VAR(1) Kalman Filter: x̂_t|t = Ax̂_t-1 + K(y_t - Ax̂_t-1)
  (optional)             Denoises driver observations
       │
       ▼
  Solver                 V1: Ridge OLS (constant Jacobian)
       │                 V4: PINN + Jacobian smoothness (nonlinear)
       ▼
 Manifold Optimizer      w = ProjTangent(Σ⁻¹μ), P = UUᵀ via SVD(∂A/∂F)
       │
       ▼
Walk-Forward Backtest    Rolling 252-day train, rebalance every N days
       │
       ▼
    Metrics              Sharpe, Sortino, MaxDD, Calmar, Turnover
```
""")
        with col2:
            st.markdown("### Paper Results (Benchmark)")
            bench = pd.DataFrame({
                "Config": ["Combo-EKF-V4 m=3", "Combo-EKF-V1 m=3", "Mean-Variance", "Equal Weight"],
                "Sharpe": [1.266, 1.143, 0.847, 0.634],
                "Sortino": [1.969, 1.752, 1.201, 0.891],
                "MaxDD": [-0.194, -0.211, -0.287, -0.341],
                "Turnover": [0.153, 0.168, 0.204, 0.012],
            })
            st.dataframe(bench, hide_index=True, use_container_width=True)

    else:
        data = st.session_state.pipeline_data
        result = st.session_state.result

        # ── Key metrics ──
        st.markdown("### Performance Metrics")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            _metric("Sharpe Ratio", f"{result.sharpe:.3f}",
                    f"+{result.sharpe - 0.847:.3f} vs MV" if result.sharpe > 0.847 else f"{result.sharpe - 0.847:.3f} vs MV")
        with c2:
            _metric("Sortino Ratio", f"{result.sortino:.3f}")
        with c3:
            _metric("Max Drawdown", f"{result.max_dd:.1%}", None, good_positive=False)
        with c4:
            _metric("Calmar Ratio", f"{result.calmar:.3f}")
        with c5:
            _metric("Avg Turnover", f"{result.avg_turnover:.3f}")
        with c6:
            color = "✅" if result.coherence_score < 0.01 else "⚠️"
            _metric("Coherence", f"{color} {result.coherence_score:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Config summary ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Run Configuration")
            cfg = {
                "Assets": ", ".join(data["asset_names"]),
                "Date range": f"{start_date} → {end_date}",
                "Selected drivers": ", ".join(data["selected_drivers"]),
                "Solver": data["solver_name"].upper(),
                "EKF enabled": str(use_ekf),
                "Train window": f"{train_window} days",
                "Rebalance freq": f"every {rebalance_freq} days",
                "Rebalances": str(len(result.rebalance_dates)),
            }
            st.table(pd.DataFrame(cfg.items(), columns=["Parameter", "Value"]))

        with col_b:
            st.markdown("### DAG Structure")
            summary = data["dag_summary"]
            dag_df = pd.DataFrame([
                ("Global factors", summary["global_factors"]),
                ("Macro factors", summary["macro_factors"]),
                ("Asset returns", summary["asset_returns"]),
                ("Asset covariates", summary["asset_covariates"]),
                ("Instruments", summary["instruments"]),
                ("Unobserved shocks", summary["unobserved_shocks"]),
                ("Total nodes", summary["total_nodes"]),
                ("Total edges", summary["total_edges"]),
            ], columns=["Element", "Count"])
            st.dataframe(dag_df, hide_index=True, use_container_width=True)

        # ── Cumulative returns ──
        st.markdown("### Cumulative Portfolio Returns")
        port_cum = np.cumprod(1 + result.returns_series)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=port_cum, mode="lines", name="CPCM Portfolio",
            line=dict(color="#00cc88", width=2),
        ))
        # Buy & Hold BTC benchmark over the same test window
        ov_returns = data.get("returns")
        if ov_returns is not None and "btc_return" in ov_returns.columns:
            test_dates = data["dates"][train_window:train_window + len(port_cum)]
            btc_r = ov_returns["btc_return"].reindex(test_dates).fillna(0.0)
            fig.add_trace(go.Scatter(
                y=np.cumprod(1 + btc_r.values), mode="lines", name="Buy & Hold BTC",
                line=dict(color="#f7931a", width=1.6, dash="dash"),
            ))
        fig.update_layout(**DARK, title="", height=300,
                          xaxis_title="Days", yaxis_title="Cumulative Value",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── Factors & Drivers tab ─────────────────────────────────────────────
with t_factors:
    if st.session_state.pipeline_data is None:
        st.info("Run the pipeline first.")
    else:
        data = st.session_state.pipeline_data

        # ── Combo ranking ──
        st.markdown("### Combo Driver Selection Ranking")
        st.caption(f"Best m={data['m_actual']} drivers minimizing residual commonality λ_max(Σ_ε). Lower = better.")

        ranking = data["ranking"]
        top_n = st.slider("Show top N subsets", 5, 50, 20, key="ranking_top_n")
        rank_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Subset": " + ".join(s),
                "Commonality Score": f"{score:.4f}",
                "Selected": "⭐" if i == 0 else "",
            }
            for i, (s, score) in enumerate(ranking[:top_n])
        ])
        st.dataframe(rank_df, hide_index=True, use_container_width=True)

        # ── Factor time series ──
        st.markdown("### Factor Time Series")
        factors = data["factors"]
        drivers_df = data["drivers"]

        factor_to_plot = st.multiselect(
            "Select factors to display",
            list(factors.dropna(axis=1, how="all").columns),
            default=data["selected_drivers"],
            key="factor_plot_sel",
        )

        if factor_to_plot:
            fig = make_subplots(
                rows=len(factor_to_plot), cols=1,
                shared_xaxes=True,
                subplot_titles=factor_to_plot,
                vertical_spacing=0.03,
            )
            colors = px.colors.qualitative.Plotly
            for i, col in enumerate(factor_to_plot):
                if col in factors.columns:
                    s = factors[col].dropna()
                    fig.add_trace(
                        go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=col, line=dict(color=colors[i % len(colors)], width=1),
                            showlegend=True,
                        ),
                        row=i + 1, col=1,
                    )
                    # Mark selected drivers
                    if col in data["selected_drivers"]:
                        fig.add_trace(
                            go.Scatter(
                                x=s.index[[0, -1]], y=[0, 0], mode="lines",
                                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                                showlegend=False,
                            ),
                            row=i + 1, col=1,
                        )
            fig.update_layout(
                **DARK, height=250 * len(factor_to_plot),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation matrix ──
        st.markdown("### Factor Correlation Matrix")
        avail = data["available_factors"]
        corr = avail.dropna().corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="",
        )
        fig_corr.update_layout(**DARK, height=450, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)


# ── Causal DAG tab ────────────────────────────────────────────────────
with t_dag:
    if st.session_state.pipeline_data is None:
        st.info("Run the pipeline first.")
    else:
        data = st.session_state.pipeline_data
        dag = data["dag"]

        st.markdown("### CPCM Star-DAG")
        col_ctrl, col_info = st.columns([3, 1])

        with col_info:
            st.markdown("**Node types**")
            st.markdown("""
- 🔵 Global factor
- 🟣 Macro factor
- 🔴 Asset return
- 🟡 Instrument (IV)
- 🟠 Asset covariate
- ⚫ Unobserved shock
""")
            summary = data["dag_summary"]
            st.markdown(f"**{summary['total_nodes']}** nodes, **{summary['total_edges']}** edges")

        from causal_portfolio.scm.graph import NodeKind
        node_colors = {
            NodeKind.GLOBAL_FACTOR: "#1f77b4",
            NodeKind.MACRO_FACTOR: "#9467bd",
            NodeKind.ASSET_RETURN: "#d62728",
            NodeKind.INSTRUMENT: "#bcbd22",
            NodeKind.ASSET_COVARIATE: "#ff7f0e",
            NodeKind.UNOBSERVED_SHOCK: "#7f7f7f",
        }

        # Build layout using hierarchical positioning
        pos = {}
        kinds = nx.get_node_attributes(dag, "kind")

        def _layout_dag(G, kinds):
            """Hierarchical layout: instruments → factors → covariates → returns."""
            layer_map = {
                NodeKind.INSTRUMENT: 0,
                NodeKind.GLOBAL_FACTOR: 1,
                NodeKind.MACRO_FACTOR: 1,
                NodeKind.UNOBSERVED_SHOCK: 2,
                NodeKind.ASSET_COVARIATE: 2,
                NodeKind.ASSET_RETURN: 3,
            }
            layers: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
            for n, k in kinds.items():
                layers[layer_map.get(k, 2)].append(n)

            pos = {}
            for layer, nodes in layers.items():
                x_pos = layer * 3
                for j, n in enumerate(sorted(nodes)):
                    y_pos = j - len(nodes) / 2
                    pos[n] = (x_pos, y_pos)
            return pos

        pos = _layout_dag(dag, kinds)

        # Extract edges and nodes for plotly
        edge_x, edge_y = [], []
        for u, v in dag.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in dag.nodes() if n in pos]
        node_y = [pos[n][1] for n in dag.nodes() if n in pos]
        node_color = [node_colors.get(kinds.get(n, NodeKind.GLOBAL_FACTOR), "#aaa")
                      for n in dag.nodes() if n in pos]
        node_text = list(dag.nodes())

        fig_dag = go.Figure()
        fig_dag.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.7, color="#444"),
            hoverinfo="none", showlegend=False,
        ))
        fig_dag.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=10, color=node_color, line=dict(width=1, color="#333")),
            text=node_text,
            textposition="top center",
            textfont=dict(size=8, color="white"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))
        fig_dag.update_layout(
            **DARK, height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_dag, use_container_width=True)

        # d-separation explorer
        st.markdown("### d-Separation Explorer")
        st.caption("Check if two nodes are d-separated given a conditioning set.")
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        node_list = sorted(dag.nodes())
        with col_ds1:
            node_a = st.selectbox("Node A", node_list, index=0)
        with col_ds2:
            node_b = st.selectbox("Node B", node_list, index=min(1, len(node_list)-1))
        with col_ds3:
            cond_nodes = st.multiselect("Conditioning set", node_list)

        if st.button("Check d-separation"):
            from networkx.algorithms.d_separation import is_d_separator
            result_dsep = is_d_separator(dag, {node_a}, {node_b}, set(cond_nodes))
            if result_dsep:
                st.success(f"✅ {node_a} and {node_b} ARE d-separated given {{{', '.join(cond_nodes) or '∅'}}}.")
            else:
                st.error(f"❌ {node_a} and {node_b} are NOT d-separated given {{{', '.join(cond_nodes) or '∅'}}}.")


# ── Solver tab ────────────────────────────────────────────────────────
with t_solver:
    if st.session_state.pipeline_data is None:
        st.info("Run the pipeline first.")
    else:
        data = st.session_state.pipeline_data
        solver = data["solver"]

        st.markdown(f"### {data['solver_name'].upper()} Solver Diagnostics")

        if data["solver_name"] == "v1":
            diag = solver.diagnostics
            st.markdown(f"**Mean R²**: `{diag['mean_r2']:.4f}` | "
                        f"**N obs**: `{diag['n_observations']}` | "
                        f"**Residual σ**: `{diag['residual_std']:.4f}`")

            # R² per asset bar chart
            r2_vals = diag["r2_per_asset"]
            asset_names = data["asset_names"]
            fig_r2 = go.Figure(go.Bar(
                x=asset_names[:len(r2_vals)],
                y=r2_vals,
                marker_color=["#00cc88" if v > 0.1 else "#ff4b4b" for v in r2_vals],
                text=[f"{v:.3f}" for v in r2_vals],
                textposition="outside",
            ))
            fig_r2.add_hline(y=0, line_color="white", line_width=0.5)
            fig_r2.update_layout(
                **DARK, title="R² per Asset (V1 Ridge OLS)",
                height=350, yaxis_title="R²",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            # Beta heatmap
            st.markdown("### Coefficient Matrix β (n_assets × m_drivers)")
            beta = solver.beta_  # (m, n_assets)
            beta_df = pd.DataFrame(
                beta.T,
                index=asset_names[:beta.shape[1]] if beta.shape[1] <= len(asset_names) else range(beta.shape[1]),
                columns=data["selected_drivers"],
            )
            fig_beta = px.imshow(
                beta_df,
                text_auto=".3f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="",
            )
            fig_beta.update_layout(**DARK, height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_beta, use_container_width=True)

            # Jacobian = beta.T (n_assets × m_drivers)
            st.markdown("### Jacobian dA/dF (constant for V1)")
            J = solver.jacobian(np.zeros(data["m_actual"]))
            st.caption("For V1 the Jacobian is constant (= βᵀ). For V4 it varies point-by-point.")
            j_df = pd.DataFrame(J, index=asset_names[:J.shape[0]], columns=data["selected_drivers"])
            st.dataframe(j_df.style.format("{:.4f}").background_gradient(cmap="RdBu_r", axis=None),
                         use_container_width=True)

            # Residual covariance
            st.markdown("### Residual Covariance (Ledoit-Wolf shrinkage)")
            cov_df = pd.DataFrame(
                solver.residual_cov_,
                index=asset_names[:solver.residual_cov_.shape[0]],
                columns=asset_names[:solver.residual_cov_.shape[1]],
            )
            fig_cov = px.imshow(cov_df, text_auto=".4f",
                                color_continuous_scale="Blues", aspect="auto")
            fig_cov.update_layout(**DARK, height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cov, use_container_width=True)

        else:  # V4 PINN
            history = solver.train_history
            if history:
                h_df = pd.DataFrame(history)
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=h_df["epoch"], y=h_df["train_loss"],
                    mode="lines", name="Train Loss", line=dict(color="#1f77b4", width=1.5),
                ))
                fig_loss.add_trace(go.Scatter(
                    x=h_df["epoch"], y=h_df["val_loss"],
                    mode="lines", name="Val Loss", line=dict(color="#ff7f0e", width=1.5),
                ))
                fig_loss.update_layout(
                    **DARK, title="V4 PINN Training Curve",
                    height=350, xaxis_title="Epoch", yaxis_title="MSE Loss",
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            # Jacobian at a single point
            st.markdown("### Jacobian dA/dF at Current Driver State")
            D = data["D"]
            J = solver.jacobian(D[-1])
            asset_names = data["asset_names"]
            j_df = pd.DataFrame(J, index=asset_names[:J.shape[0]], columns=data["selected_drivers"])
            st.dataframe(j_df.style.format("{:.4f}").background_gradient(cmap="RdBu_r", axis=None),
                         use_container_width=True)
            st.caption("For V4 PINN, the Jacobian is computed via autodiff and varies with driver state.")


# ── Backtest tab ──────────────────────────────────────────────────────
with t_backtest:
    if st.session_state.pipeline_data is None:
        st.info("Run the pipeline first.")
    else:
        result = st.session_state.result
        data = st.session_state.pipeline_data

        # ── Performance metrics ──
        st.markdown("### Risk/Return Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_row = [
            ("Sharpe", f"{result.sharpe:.3f}"),
            ("Sortino", f"{result.sortino:.3f}"),
            ("MaxDD", f"{result.max_dd:.1%}"),
            ("Calmar", f"{result.calmar:.3f}"),
            ("Turnover", f"{result.avg_turnover:.3f}"),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics_row):
            col.metric(label, val)

        # ── Cumulative returns ──
        st.markdown("### Cumulative Returns")
        port_cum = np.cumprod(1 + result.returns_series)
        port_cum_series = pd.Series(
            port_cum, index=data["dates"][train_window:train_window + len(port_cum)]
        )

        # ── Buy & Hold BTC benchmark (same window) ──
        # Align BTC daily returns to the portfolio's test window and compound.
        # Skip gracefully if BTC isn't in the selected universe.
        bh_cum_series = None
        returns_df = data.get("returns")
        if returns_df is not None and "btc_return" in returns_df.columns:
            btc_r = returns_df["btc_return"].reindex(port_cum_series.index).fillna(0.0)
            bh_cum_series = pd.Series(
                np.cumprod(1 + btc_r.values), index=port_cum_series.index,
            )

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=port_cum_series.index, y=port_cum_series.values,
            mode="lines", name="CPCM Portfolio",
            line=dict(color="#00cc88", width=2),
            fill="tozeroy", fillcolor="rgba(0,204,136,0.1)",
        ))
        if bh_cum_series is not None:
            fig_cum.add_trace(go.Scatter(
                x=bh_cum_series.index, y=bh_cum_series.values,
                mode="lines", name="Buy & Hold BTC",
                line=dict(color="#f7931a", width=1.8, dash="dash"),
            ))
        # Drawdown shading
        peak = np.maximum.accumulate(port_cum)
        drawdown = (port_cum - peak) / peak
        fig_cum.add_trace(go.Scatter(
            x=port_cum_series.index, y=peak,
            mode="lines", name="Peak",
            line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
        ))
        fig_cum.update_layout(
            **DARK, height=350, xaxis_title="Date", yaxis_title="Cumulative Return",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Strategy vs Buy & Hold BTC comparison ──
        if bh_cum_series is not None:
            port_total = float(port_cum_series.iloc[-1] - 1)
            bh_total = float(bh_cum_series.iloc[-1] - 1)
            lift = port_total - bh_total
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("CPCM total return", f"{port_total:+.1%}")
            cc2.metric("Buy & Hold BTC", f"{bh_total:+.1%}")
            cc3.metric("CPCM − BH BTC", f"{lift:+.1%}",
                       delta=f"{lift:+.1%}",
                       delta_color="normal" if lift >= 0 else "inverse")
            if lift < 0:
                st.caption("⚠️ The CPCM strategy underperforms buy-and-hold BTC over "
                           "this window. BH BTC remains the benchmark to beat.")

        # ── Drawdown ──
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=port_cum_series.index, y=drawdown * 100,
            mode="lines", name="Drawdown",
            line=dict(color="#ff4b4b", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,75,75,0.15)",
        ))
        fig_dd.update_layout(
            **DARK, height=200, xaxis_title="", yaxis_title="Drawdown (%)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Daily returns distribution ──
        col_hist, col_weights = st.columns(2)
        with col_hist:
            st.markdown("### Daily Return Distribution")
            fig_hist = px.histogram(
                result.returns_series, nbins=60,
                color_discrete_sequence=["#1f77b4"],
                labels={"value": "Daily Return"},
            )
            fig_hist.update_layout(
                **DARK, height=300, showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_weights:
            st.markdown("### Final Portfolio Weights")
            if len(result.weights_history) > 0:
                final_w = result.weights_history[-1]
                asset_names = data["asset_names"]
                w_df = pd.DataFrame({
                    "Asset": asset_names[:len(final_w)],
                    "Weight": final_w,
                }).sort_values("Weight", ascending=True)
                colors = ["#ff4b4b" if w < 0 else "#00cc88" for w in w_df["Weight"]]
                fig_w = go.Figure(go.Bar(
                    y=w_df["Asset"], x=w_df["Weight"], orientation="h",
                    marker_color=colors,
                    text=[f"{w:.3f}" for w in w_df["Weight"]],
                    textposition="outside",
                ))
                fig_w.update_layout(
                    **DARK, height=300,
                    xaxis_title="Weight", margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_w, use_container_width=True)

        # ── Weight evolution ──
        st.markdown("### Portfolio Weight Evolution")
        wh = result.weights_history
        asset_names = data["asset_names"]
        wh_idx = data["dates"][train_window:train_window + len(wh)]
        wh_df = pd.DataFrame(wh, index=wh_idx,
                             columns=asset_names[:wh.shape[1]])
        fig_wev = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, col in enumerate(wh_df.columns):
            fig_wev.add_trace(go.Scatter(
                x=wh_df.index, y=wh_df[col],
                mode="lines", name=col,
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig_wev.update_layout(
            **DARK, height=350, xaxis_title="Date", yaxis_title="Weight",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_wev, use_container_width=True)

        # ── Rolling Sharpe ──
        st.markdown("### Rolling Sharpe Ratio (63-day)")
        roll_window = 63
        if len(result.returns_series) > roll_window:
            from causal_portfolio.backtest.metrics import ANNUALIZATION
            # Trailing window ending at i-1, population std (ddof=0)
            roll = pd.Series(result.returns_series).rolling(roll_window)
            roll_sharpe = (
                roll.mean() / roll.std(ddof=0).clip(lower=1e-10)
                * np.sqrt(ANNUALIZATION)
            ).to_numpy()[roll_window - 1 : -1]
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=port_cum_series.index[roll_window:], y=roll_sharpe,
                mode="lines", name="Rolling Sharpe",
                line=dict(color="#9467bd", width=1.5),
            ))
            fig_rs.add_hline(y=0, line_color="white", line_width=0.5, line_dash="dot")
            fig_rs.add_hline(y=1.266, line_color="#bcbd22", line_width=1,
                            line_dash="dash", annotation_text="Paper target (1.266)")
            fig_rs.update_layout(
                **DARK, height=250, xaxis_title="Date", yaxis_title="Sharpe",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_rs, use_container_width=True)

        # ── Raw data download ──
        st.markdown("### Export Results")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            returns_csv = pd.Series(result.returns_series).to_csv().encode()
            st.download_button("📥 Download Returns (.csv)", returns_csv,
                               "cpcm_returns.csv", "text/csv")
        with col_dl2:
            weights_csv = wh_df.to_csv().encode()
            st.download_button("📥 Download Weights (.csv)", weights_csv,
                               "cpcm_weights.csv", "text/csv")


# ── EKF tab ───────────────────────────────────────────────────────────
with t_ekf:
    if st.session_state.pipeline_data is None:
        st.info("Run the pipeline first.")
    elif st.session_state.pipeline_data["ekf_data"] is None:
        st.info("EKF was not enabled for this run. Toggle **Enable EKF filtering** in the sidebar and re-run.")
    else:
        ekf_data = st.session_state.pipeline_data["ekf_data"]
        data = st.session_state.pipeline_data

        st.markdown("### EKF Filter: Raw vs Filtered Drivers")
        filtered = ekf_data["filtered"]
        raw = ekf_data["raw"]
        driver_names = ekf_data["driver_names"]
        dates = data["dates"]

        driver_sel = st.selectbox("Select driver", driver_names, key="ekf_driver_sel")
        d_idx = driver_names.index(driver_sel)

        fig_ekf = go.Figure()
        fig_ekf.add_trace(go.Scatter(
            x=dates, y=raw[:, d_idx],
            mode="lines", name="Raw (noisy)",
            line=dict(color="rgba(100,150,255,0.5)", width=1),
        ))
        fig_ekf.add_trace(go.Scatter(
            x=dates, y=filtered[:, d_idx],
            mode="lines", name="EKF Filtered",
            line=dict(color="#00cc88", width=2),
        ))
        fig_ekf.update_layout(
            **DARK, title=f"Driver: {driver_sel}",
            height=300, xaxis_title="Date", yaxis_title="z-score",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_ekf, use_container_width=True)

        # All drivers side by side
        st.markdown("### All Drivers: Raw vs Filtered")
        fig_all = make_subplots(
            rows=len(driver_names), cols=1, shared_xaxes=True,
            subplot_titles=driver_names, vertical_spacing=0.04,
        )
        for i, dname in enumerate(driver_names):
            fig_all.add_trace(go.Scatter(
                x=dates, y=raw[:, i], mode="lines", name=f"{dname} (raw)",
                line=dict(color="rgba(100,150,255,0.4)", width=1), showlegend=(i == 0),
            ), row=i + 1, col=1)
            fig_all.add_trace(go.Scatter(
                x=dates, y=filtered[:, i], mode="lines", name=f"{dname} (filtered)",
                line=dict(color="#00cc88", width=1.5), showlegend=(i == 0),
            ), row=i + 1, col=1)
        fig_all.update_layout(
            **DARK, height=220 * len(driver_names),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # Innovation diagnostics
        st.markdown("### Innovation Diagnostics")
        st.caption("Normalized innovations should be ~N(0,1) if the model is well-specified.")
        diag = ekf_data["diagnostics"]
        innov_df = pd.DataFrame({
            "Driver": driver_names,
            "Mean (should ≈ 0)": [f"{m:.4f}" for m in diag["mean"]],
            "Std (should ≈ 1)": [f"{s:.4f}" for s in diag["std"]],
            "Autocorr lag-1 (should ≈ 0)": [f"{a:.4f}" for a in diag["autocorr_lag1"]],
        })

        def _style_innov(df):
            return df

        st.dataframe(innov_df, hide_index=True, use_container_width=True)

        # State prediction uncertainty
        st.markdown("### State Prediction Uncertainty (Trace of P_t)")
        covs = ekf_data.get("covs")
        if covs is None:
            st.info("Re-run with EKF enabled to see time-varying state covariance (P_t) evolution.")
        else:
            # trace(P_t) = sum of diagonal — total state uncertainty at each time step
            trace_pt = np.einsum("tii->t", covs)
            dates = data.get("dates")
            pt_df = pd.DataFrame({"trace(P_t)": trace_pt}, index=dates if dates is not None else None)
            fig_pt = px.line(
                pt_df, y="trace(P_t)",
                title="Total state uncertainty over time (lower = filter more confident)",
            )
            fig_pt.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pt, use_container_width=True)
            st.caption(
                "Trace of the posterior covariance matrix P_{t|t}. "
                "Should drop quickly from the initial value and stabilize. "
                "Spikes indicate periods where observations were noisy or missing."
            )


# ── Regimes tab ───────────────────────────────────────────────────────
with t_regimes:
    st.markdown("## 🔀 Market Regime Analysis")
    st.caption(
        "Fits a Gaussian HMM on (VIX, BTC realized vol) and characterizes each "
        "regime. Independent of the CPCM **Run Pipeline** flow — set the "
        "parameters below and click **Analyze Regimes**. Uses the sidebar's "
        f"asset universe ({', '.join(selected_assets)}) and date range."
    )

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        regime_n_states = st.slider("Number of regimes", 1, 3, 2)
        regime_m = st.slider("Drivers per regime (m)", 1, 3, 3)
    with rc2:
        regime_window = st.slider("HMM window (days)", 90, 756, 504, 30)
        regime_refit = st.slider("Causal refit cadence (days)", 7, 126, 63, 7)
    with rc3:
        regime_label_mode = st.radio(
            "Labeling method",
            ["Causal (forward filter)", "Non-causal (Viterbi · look-ahead)"],
        )
        st.caption(
            "**Causal** = what a live system would have known at each point "
            "(rolling fit + forward filter). **Non-causal** = full-sample "
            "hindsight; overstates the structure (see research doc)."
        )

    if st.button("Analyze Regimes", type="primary"):
        from causal_portfolio.regimes.dashboard_panel import analyze_regimes
        # Cache on params so re-clicking (e.g. toggling causal/non-causal back
        # and forth) doesn't refit the whole rolling HMM from scratch.
        analyze_regimes_cached = st.cache_data(show_spinner=False, ttl=3600)(analyze_regimes)
        with st.spinner("Fitting HMM + per-regime driver selection…"):
            try:
                st.session_state.regime_result = analyze_regimes_cached(
                    assets=selected_assets,
                    start=str(start_date), end=str(end_date),
                    n_states=regime_n_states, hmm_window=regime_window,
                    hmm_refit_every=regime_refit, m_drivers=regime_m,
                    causal=regime_label_mode.startswith("Causal"),
                )
            except Exception as e:
                st.session_state.regime_result = None
                st.error(f"Regime analysis failed: {e}")

    rr = st.session_state.get("regime_result")
    if rr is None:
        st.info("Set parameters and click **Analyze Regimes** to run.")
    else:
        _REGIME_COLORS = ["#00cc88", "#f7931a", "#e45756"]  # calm → transitional → stress

        def _regime_name(state: int, n: int) -> str:
            if n == 1:
                return "All"
            if n == 2:
                return ["Calm", "Stress"][state]
            return ["Calm", "Transitional", "Stress"][state]

        for note in rr.notes:
            st.warning(note)

        mode_tag = "causal (forward filter)" if rr.causal else "non-causal (Viterbi, look-ahead)"
        st.markdown(f"**Labeling:** {mode_tag} · **features:** {', '.join(rr.feature_columns)} "
                    f"· **labeled days:** {len(rr.dates)}")

        # ── Characterization table ──
        st.markdown("### Regime Characterization")
        char_rows = []
        for s in range(rr.n_states):
            d = rr.dwell.get(s, {})
            means = rr.state_means[s] if s < len(rr.state_means) else []
            row = {"Regime": f"{s} · {_regime_name(s, rr.n_states)}"}
            for col, mval in zip(rr.feature_columns, means):
                row[f"{col} (z)"] = f"{mval:+.2f}"
            row["Days"] = d.get("days", 0)
            row["% sample"] = f"{d.get('pct', 0):.1%}"
            row["Mean run (d)"] = f"{d.get('mean_run_length', 0):.0f}"
            char_rows.append(row)
        st.dataframe(pd.DataFrame(char_rows), hide_index=True, use_container_width=True)

        # ── Transition matrix ──
        if rr.n_states > 1:
            st.markdown("### Transition Matrix  ·  P(next | current)")
            labels_tm = [f"{s}·{_regime_name(s, rr.n_states)}" for s in range(rr.n_states)]
            fig_tm = go.Figure(go.Heatmap(
                z=rr.transition_matrix, x=labels_tm, y=labels_tm,
                text=[[f"{v:.3f}" for v in row] for row in rr.transition_matrix],
                texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
                showscale=False,
            ))
            fig_tm.update_layout(**DARK, height=260, xaxis_title="To",
                                 yaxis_title="From", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_tm, use_container_width=True)

        # ── Timeline ribbon ──
        st.markdown("### Regime Timeline  ·  market equity colored by regime")
        from causal_portfolio.regimes.dashboard_panel import contiguous_runs
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=rr.market_curve.index, y=rr.market_curve.values,
            mode="lines", name="Equal-weight basket",
            line=dict(color="#e8e8e8", width=1.6),
        ))
        for s0, s1, state in contiguous_runs(rr.dates, rr.labels):
            fig_tl.add_vrect(
                x0=s0, x1=s1, fillcolor=_REGIME_COLORS[state % len(_REGIME_COLORS)],
                opacity=0.18, line_width=0, layer="below",
            )
        # Legend proxies for the regime bands
        for s in range(rr.n_states):
            fig_tl.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=_REGIME_COLORS[s % len(_REGIME_COLORS)]),
                name=f"{s}·{_regime_name(s, rr.n_states)}",
            ))
        fig_tl.update_layout(**DARK, height=340, yaxis_title="Growth of $1",
                             margin=dict(l=0, r=0, t=10, b=0),
                             legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_tl, use_container_width=True)

        # ── Top drivers per regime + return stats ──
        cda, cdb = st.columns(2)
        with cda:
            st.markdown("### Top Drivers per Regime")
            drv_rows = []
            for s in range(rr.n_states):
                info = rr.per_regime_drivers.get(s, {})
                winner = info.get("winner")
                drv_rows.append({
                    "Regime": f"{s}·{_regime_name(s, rr.n_states)}",
                    "Top drivers": ", ".join(winner) if winner else (info.get("note") or "—"),
                    "Score": f"{info['score']:.3f}" if info.get("score") is not None else "—",
                })
            st.dataframe(pd.DataFrame(drv_rows), hide_index=True, use_container_width=True)
            st.caption("Lowest commonality score = best subset (λ_max of residual "
                       "correlation). Different drivers per regime is the whole point.")
        with cdb:
            st.markdown("### Market Returns by Regime")
            ret_rows = []
            for s in range(rr.n_states):
                rstat = rr.per_regime_returns.get(s, {})
                ret_rows.append({
                    "Regime": f"{s}·{_regime_name(s, rr.n_states)}",
                    "Ann. return": f"{rstat.get('ann_return', float('nan')):+.1%}",
                    "Ann. vol": f"{rstat.get('ann_vol', float('nan')):.1%}",
                    "Sharpe": f"{rstat.get('sharpe', float('nan')):.2f}",
                    "Days": rstat.get("days", 0),
                })
            st.dataframe(pd.DataFrame(ret_rows), hide_index=True, use_container_width=True)
            st.caption("Equal-weight basket performance while in each regime. "
                       "A useful regime split shows clearly different return/risk.")
