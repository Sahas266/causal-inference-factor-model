"""Instrument the TWO relationships that survived placebo testing.

The whole-warehouse IV program (`iv_search.py` -> `iv_findings.md`) closed the
question "does instrumenting factor->RETURNS beat OLS?" with a clean NO: weak
instruments collapse 2SLS to OLS, strong ones make 2SLS worse OOS, and causal
discovery says returns are *upstream* of the factors anyway.

But two relationships did survive their placebo tests and were left as the open
follow-up (iv_findings.md "Where instruments could still matter"):

  TASK 1  chain_congestion(t-1) -> btc_return(t)
          The ONE candidate factor->return edge (dag_v2.md): DML +39 bps/sigma
          in innovation space, timing placebo p = 0.01. Predictive — is it
          CAUSAL? We need Z that shifts yesterday's congestion but reaches
          today's BTC return ONLY through congestion.

  TASK 2  chain_congestion innovation(t) -> aggregate funding(t+1)
          target_search.md: factor innovations add +0.023 OOS R2 to next-day
          funding over funding's own (strong) persistence, placebo p = 0.02,
          driven mostly by chain_congestion. Predictive — is it CAUSAL? We
          need Z that shifts congestion but is excludable from funding.

Both treatments are the SAME factor (congestion); the exclusion *story*
differs by outcome. Candidate instruments are chosen for an a-priori
exclusion argument the pure-statistical warehouse screen never had:

  - blob-space demand (eth_blob_size_mib / eth_blob_fees): L2 data-availability
    demand drives ETH gas/fees (relevance) but is a function of rollup posting
    schedules, arguably orthogonal to BTC's own return shock and to perp
    funding. Post-Dencun (2024-03) only.
  - NFT trading volume (eth_chain_nft_trading_volume): a non-financial
    blockspace consumer — mints/trades congest the chain without being a BTC
    price bet.
  - non-financial transaction / user counts (eth_TxCnt, eth_new_users,
    btc_TxCnt, *_tx_count): raw on-chain demand, drives fees, not a direct
    claim on BTC return.
  - gas/fees from a DIFFERENT chain (bnb_total_fees_bnb, sol_total_fees_sol,
    avax_total_fees_avax, sol_priority_fees): cross-chain congestion that
    co-moves with the demand cycle but does not mechanically touch ETH-fee
    congestion's own residual or BTC's price.

HONESTY BAR (identical to iv_search.py):
  - Relevance, |corr(Z,T)|, direct-path t are computed on the TRAIN window
    only (first `train_window` aligned rows). Instrument choice cannot leak OOS.
  - Tautology screen: |corr(Z,T)| > 0.9 => Z *is* T up to scaling, reject.
  - Direct-path probe: a significant Z coefficient in
    outcome ~ [1, T, Z] flags a path that bypasses T (exclusion red flag).
  - Exclusion itself is UNTESTABLE. These instruments rest on an economic
    argument, not a proof.
  - One-cycle results. Task 2's funding sample starts 2023-11 (~2 years), so
    the walk-forward there is short. Read every number as suggestive.

The 2SLS-vs-OLS comparison is the verdict: if instrumenting congestion leaves
the effect's sign and size ~unchanged and the OOS fit holds, the relationship
behaves causally; if 2SLS blows up / flips / dies OOS, it was predictive
co-movement, not a structural congestion->outcome channel.

Run:  python -m causal_portfolio.experiments.iv_surviving_edges --help
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS,
    build_all_factors, innovation_factors,
)
# Reused screening primitives (NOT modified).
from causal_portfolio.experiments.iv_search import (
    _candidate_series, _first_stage_f, _direct_path_t,
)
from causal_portfolio.experiments.ols_vs_2sls import _partial_first_stage_f
from causal_portfolio.scm.estimators import add_intercept, ols, tsls

logger = logging.getLogger("cpcm.experiments.iv_surviving_edges")


# ── candidate instruments (asset, metric, transform, exclusion story) ──
# Each tuple: source asset, warehouse metric, transform key from
# _candidate_series ("diff" or "spike"), and the a-priori exclusion rationale.
CONGESTION_INSTRUMENTS: list[tuple[str, str, str, str]] = [
    ("eth", "blob_size_mib", "diff",
     "L2 data-availability demand drives ETH gas; rollup posting schedule, "
     "arguably orthogonal to BTC return (post-Dencun 2024-03 only)"),
    ("eth", "blob_fees", "diff",
     "blob-fee market = DA demand pressure on blockspace (post-Dencun only)"),
    ("eth", "chain_nft_trading_volume", "diff",
     "NFT mints/trades are a non-financial blockspace consumer that congests "
     "the chain without being a BTC price bet"),
    ("eth", "TxCnt", "diff",
     "raw ETH transaction demand drives fees; not a direct claim on BTC"),
    ("eth", "new_users", "diff",
     "new-address onboarding = organic on-chain demand -> congestion"),
    ("eth", "bridge_deposit_count", "diff",
     "bridge deposit count: cross-chain activity consumes ETH blockspace"),
    ("btc", "TxCnt", "diff",
     "BTC transaction count: chain-demand cycle proxy (different chain's "
     "congestion, excludable from ETH-fee residual)"),
    ("bnb", "total_fees_bnb", "diff",
     "BNB chain fees: cross-chain congestion co-moves with the demand cycle, "
     "does not touch ETH-fee congestion's residual or BTC price directly"),
    ("sol", "total_fees_sol", "diff",
     "Solana chain fees: cross-chain congestion proxy"),
    ("sol", "priority_fees", "diff",
     "Solana priority-fee spend: cross-chain blockspace contention"),
    ("avax", "total_fees_avax", "diff",
     "Avalanche chain fees: cross-chain congestion proxy"),
    ("bnb", "tx_count", "diff",
     "BNB transaction count: cross-chain demand proxy"),
    ("sol", "tx_count", "diff",
     "Solana transaction count: cross-chain demand proxy"),
]


# ── candidate-instrument loader ──────────────────────────────────────

def load_candidate_columns(
    pairs: list[tuple[str, str]], start: str, end: str,
) -> pd.DataFrame:
    """Load (asset, metric) series wide as `{asset}_{metric}`, daily-resampled.

    Uses the active loader's backend. For the DuckDB loader we hit the
    connection directly; for Supabase we fall back to a per-pair panel pull.
    The same UTC/naive/daily conventions as base_loader so columns align with
    the factor frame.
    """
    from causal_portfolio.data import get_loader
    loader = get_loader()
    frames: dict[str, pd.Series] = {}

    con = getattr(loader, "_con", None)
    if con is not None:  # DuckDB fast path
        for a, m in pairs:
            df = con.execute(
                """
                SELECT time, value FROM asset_metrics_best
                WHERE asset = ? AND metric = ?
                  AND time >= ?::TIMESTAMPTZ AND time <= ?::TIMESTAMPTZ
                ORDER BY time
                """,
                [a, m, f"{start}T00:00:00+00:00", f"{end}T23:59:59.999+00:00"],
            ).fetchdf()
            if df.empty:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            frames[f"{a}_{m}"] = (
                df.set_index("time")["value"].sort_index().resample("D").last())
    else:  # generic loader path (e.g. Supabase)
        by_metric: dict[str, list[str]] = {}
        for a, m in pairs:
            by_metric.setdefault(m, []).append(a)
        for m, assets in by_metric.items():
            panel = loader.load_panel(assets, [m], start, end)
            for c in panel.columns:
                frames[c] = panel[c]

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


# ── screening (train-window only) ────────────────────────────────────

@dataclass
class Screen:
    name: str            # e.g. "eth_blob_size_mib[diff]"
    story: str
    train_f: float       # first-stage F of T ~ [1, Z] on train
    corr_zt: float       # |corr(Z, T)| on train
    direct_t: float      # |t| of Z in outcome ~ [1, T, Z] on train (exclusion)
    n_obs: int
    flags: list[str] = field(default_factory=list)


def screen_instrument(
    T: pd.Series, Z: pd.Series, outcome: pd.Series, *,
    train_window: int, taut_corr: float = 0.9,
) -> Screen | None:
    """Relevance / tautology / direct-path screen on the TRAIN window only.

    T, Z, outcome are aligned-by-index series; outcome is the dependent
    variable already shifted so that row t pairs T_t/Z_t with the outcome they
    are meant to explain (caller handles the lag structure).
    """
    data = pd.concat([T.rename("T"), Z.rename("Z"), outcome.rename("Y")],
                     axis=1).dropna()
    if len(data) < train_window + 30:
        return None
    train = data.iloc[:train_window]
    Tt, Zt, Yt = train["T"].values, train["Z"].values, train["Y"].values

    f = _first_stage_f(Tt, Zt)
    corr = float(abs(np.corrcoef(Zt, Tt)[0, 1])) if np.std(Zt) > 1e-12 else 0.0
    direct = _direct_path_t(Yt, Tt, Zt)

    flags = []
    if corr > taut_corr:
        flags.append(f"TAUTOLOGY |corr(Z,T)|={corr:.2f}")
    if direct > 2.0:
        flags.append(f"DIRECT-PATH |t|={direct:.1f}")
    return Screen(name="", story="", train_f=f, corr_zt=corr, direct_t=direct,
                  n_obs=len(data), flags=flags)


# ── bespoke single-treatment 2SLS vs OLS (walk-forward OOS) ──────────

@dataclass
class TwoSLSResult:
    label: str
    instrument: str
    # Full-sample point estimates (for the headline effect comparison).
    ols_beta: float
    ols_t: float
    tsls_beta: float
    tsls_t: float
    first_stage_f_full: float
    # Walk-forward OOS.
    n_oos: int
    ols_oos_r2: float
    tsls_oos_r2: float
    ols_beta_mean: float
    tsls_beta_mean: float
    ols_beta_signflip: float   # share of windows where OLS beta flips vs full
    tsls_beta_signflip: float
    sign_agree: float          # share of OOS windows where 2SLS beta sign == OLS
    mean_window_f: float
    notes: str = ""


def _ols_single(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """OLS of y on [1, t]; return (slope, |t-stat of slope|)."""
    X = add_intercept(t.reshape(-1, 1))
    res = ols(y, X, ["intercept", "T"])
    return float(res.coefficients[1]), float(abs(res.t_stats[1]))


def _tsls_single(
    y: np.ndarray, t: np.ndarray, z: np.ndarray,
) -> tuple[float, float, float]:
    """2SLS of y on endogenous t, instrument z (intercept exogenous).

    Returns (slope, |t-stat of slope|, first-stage F).
    """
    n = len(y)
    x_exog = np.ones((n, 1))
    res = tsls(y, t.reshape(n, 1), x_exog, z.reshape(n, 1),
               ["T"], ["intercept"], ["Z"])
    return (float(res.coefficients[0]), float(abs(res.t_stats[0])),
            float(res.first_stage_f[0]))


def run_2sls_vs_ols(
    treatment: pd.Series, instrument: pd.Series, outcome: pd.Series, *,
    label: str, instrument_name: str, train_window: int, refit: int = 5,
    f_threshold: float = 10.0,
) -> TwoSLSResult | None:
    """Single-treatment 2SLS vs OLS, full-sample + walk-forward OOS.

    All three series are pre-aligned so row t holds (T_t, Z_t, outcome_t) — the
    caller has already applied the t-1 / t+1 lag structure of the edge under
    test. OOS predictions are 1-step (predict outcome_t from T_t using betas
    fit on the prior window); R^2 is vs the train-mean baseline. The economic
    question is whether instrumenting T changes the estimated effect and whether
    the instrumented forecast holds up OOS relative to OLS.
    """
    data = pd.concat([treatment.rename("T"), instrument.rename("Z"),
                      outcome.rename("Y")], axis=1).dropna()
    if len(data) < train_window + 60:
        return None
    Y = data["Y"].values
    Tt = data["T"].values
    Zt = data["Z"].values
    n = len(data)

    # Full-sample estimates (headline effect).
    ols_b, ols_tstat = _ols_single(Y, Tt)
    try:
        tsls_b, tsls_tstat, fs_f_full = _tsls_single(Y, Tt, Zt)
    except Exception as e:  # pragma: no cover - degenerate
        logger.warning("full-sample 2SLS failed (%s): %s", label, e)
        return None

    # Walk-forward: refit betas each `refit`, predict 1 step OOS.
    ols_pred = np.full(n, np.nan)
    tsls_pred = np.full(n, np.nan)
    ols_betas: list[float] = []
    tsls_betas: list[float] = []
    window_fs: list[float] = []
    ob = (0.0, 0.0)      # (intercept, slope) for OLS
    tb = (0.0, 0.0)      # (intercept, slope) for 2SLS
    for t in range(train_window, n):
        if (t - train_window) % refit == 0:
            ytr, ttr, ztr = Y[t - train_window:t], Tt[t - train_window:t], Zt[t - train_window:t]
            # OLS window fit
            Xo = add_intercept(ttr.reshape(-1, 1))
            res_o = ols(ytr, Xo, ["intercept", "T"])
            ob = (float(res_o.coefficients[0]), float(res_o.coefficients[1]))
            ols_betas.append(ob[1])
            # 2SLS window fit (gate on first-stage F; fall back to OLS if weak)
            try:
                f_win = _partial_first_stage_f(
                    ttr, ztr, np.empty((train_window, 0)))
                window_fs.append(f_win)
                if f_win >= f_threshold:
                    xe = np.ones((train_window, 1))
                    res_t = tsls(ytr, ttr.reshape(-1, 1), xe, ztr.reshape(-1, 1),
                                 ["T"], ["intercept"], ["Z"])
                    tb = (float(res_t.coefficients[1]), float(res_t.coefficients[0]))
                else:
                    tb = ob  # weak instrument this window -> degrade to OLS
            except Exception:
                tb = ob
            tsls_betas.append(tb[1])
        ols_pred[t] = ob[0] + ob[1] * Tt[t]
        tsls_pred[t] = tb[0] + tb[1] * Tt[t]

    sl = slice(train_window, None)
    yy = Y[sl]
    base = yy.mean()
    denom = float(np.sum((yy - base) ** 2)) + 1e-30

    def _r2(pred):
        pp = pred[sl]
        return float(1.0 - np.sum((yy - pp) ** 2) / denom)

    ols_betas_a = np.array(ols_betas)
    tsls_betas_a = np.array(tsls_betas)
    ols_flip = float(np.mean(np.sign(ols_betas_a) != np.sign(ols_b))) if len(ols_betas_a) else float("nan")
    tsls_flip = float(np.mean(np.sign(tsls_betas_a) != np.sign(tsls_b))) if len(tsls_betas_a) else float("nan")
    k = min(len(ols_betas_a), len(tsls_betas_a))
    sign_agree = (float(np.mean(np.sign(ols_betas_a[:k]) == np.sign(tsls_betas_a[:k])))
                  if k else float("nan"))

    return TwoSLSResult(
        label=label, instrument=instrument_name,
        ols_beta=ols_b, ols_t=ols_tstat,
        tsls_beta=tsls_b, tsls_t=tsls_tstat,
        first_stage_f_full=fs_f_full,
        n_oos=int(np.sum(~np.isnan(ols_pred[sl]))),
        ols_oos_r2=_r2(ols_pred), tsls_oos_r2=_r2(tsls_pred),
        ols_beta_mean=float(np.mean(ols_betas_a)) if len(ols_betas_a) else float("nan"),
        tsls_beta_mean=float(np.mean(tsls_betas_a)) if len(tsls_betas_a) else float("nan"),
        ols_beta_signflip=ols_flip, tsls_beta_signflip=tsls_flip,
        sign_agree=sign_agree,
        mean_window_f=float(np.mean(window_fs)) if window_fs else float("nan"),
    )


# ── task drivers ─────────────────────────────────────────────────────

def _aligned_funding(panel: pd.DataFrame) -> pd.Series:
    """Aggregate next-day funding target: mean across assets of funding_rate_8h.

    The daily panel already holds the day's last 8h funding print per asset
    (funding_rate_8h, Hyperliquid, daily-resampled .last()). We take the
    cross-asset mean — the same aggregate target_search.md forecasts. (Summing
    the 8h prints per day is unavailable post-resample; the cross-asset mean of
    the daily print is the canonical aggregate used elsewhere in the repo.)
    """
    fund_cols = [c for c in panel.columns if c.endswith("_funding_rate_8h")]
    if not fund_cols:
        return pd.Series(dtype=float)
    return panel[fund_cols].mean(axis=1)


@dataclass
class RegimeF:
    """First-stage F of an instrument across regimes (diagnostic, not selection)."""
    name: str
    train_f: float
    full_f: float
    post2024_f: float
    n_post2024: int


@dataclass
class TaskOutput:
    screens: list[Screen]
    results: list[TwoSLSResult]
    ols_baseline_note: str
    illustrative: list[TwoSLSResult] = field(default_factory=list)
    regimes: list[RegimeF] = field(default_factory=list)


def _regime_relevance(
    T: pd.Series, outcome: pd.Series,
    instruments: list[tuple[str, str, str, str]],
    raw: pd.DataFrame, *, train_window: int, lag_z: bool, post_cut: str,
) -> list[RegimeF]:
    """First-stage F of each instrument on train / full / post-2024 windows.

    Pure diagnostic — uses only Z and T (never the outcome's VALUES), so it
    cannot leak strategy selection. The outcome is passed only to align rows to
    the SAME sample the screen uses (so the "train F" column matches the
    screen's train window, which for funding starts 2023-11 not 2022). It
    answers: are these instruments relevant in the regime where the edge
    actually lives (post-Dencun 2024+), even though the train window says they
    are weak?
    """
    out: list[RegimeF] = []
    for a, m, tf, _ in instruments:
        col = f"{a}_{m}"
        if col not in raw.columns:
            continue
        Z = _candidate_series(raw[col].astype(float))[tf]
        if lag_z:               # contemporaneous-with-treatment variant
            Z = Z.shift(-1)
        d = pd.concat([T.rename("T"), Z.rename("Z"),
                       outcome.rename("Y")], axis=1).dropna()
        if len(d) < train_window + 30:
            continue
        f_tr = _first_stage_f(d["T"].values[:train_window], d["Z"].values[:train_window])
        f_full = _first_stage_f(d["T"].values, d["Z"].values)
        dp = d[d.index >= post_cut]
        f_post = _first_stage_f(dp["T"].values, dp["Z"].values) if len(dp) > 40 else float("nan")
        out.append(RegimeF(f"{col}[{tf}]", f_tr, f_full, f_post, len(dp)))
    out.sort(key=lambda r: -r.full_f)
    return out


def run_task1_congestion_to_btc(
    factors: pd.DataFrame, innov: pd.DataFrame, returns: pd.DataFrame,
    raw: pd.DataFrame, *, train_window: int, f_threshold: float,
) -> TaskOutput:
    """congestion(t-1) -> btc_return(t), instrumented.

    Treatment = chain_congestion AR(1) innovation known at t-1 (shift +1 so the
    row's congestion is yesterday's). Outcome = btc_return(t). Instrument =
    candidate metric innovation, also at t-1 (predetermined). This matches the
    dag_v2 edge exactly: predetermined congestion -> next-day BTC.
    """
    cc = innov["chain_congestion"]
    T = cc.shift(1).rename("congestion_lag1")          # known at t-1
    btc = returns["btc_return"]

    screens: list[Screen] = []
    results: list[TwoSLSResult] = []
    zser: dict[str, pd.Series] = {}
    for asset, metric, tf, story in CONGESTION_INSTRUMENTS:
        col = f"{asset}_{metric}"
        if col not in raw.columns:
            continue
        Z = _candidate_series(raw[col].astype(float))[tf]   # already lag-1
        name = f"{col}[{tf}]"
        zser[name] = Z
        sc = screen_instrument(T, Z, btc, train_window=train_window)
        if sc is None:
            continue
        sc.name, sc.story = name, story
        screens.append(sc)

        if sc.train_f >= f_threshold and not any(
                fl.startswith("TAUTOLOGY") for fl in sc.flags):
            res = run_2sls_vs_ols(
                T, Z, btc, label="congestion(t-1)->btc_return(t)",
                instrument_name=name, train_window=train_window,
                f_threshold=f_threshold)
            if res is not None:
                results.append(res)

    screens.sort(key=lambda s: -s.train_f)

    # Illustrative 2SLS on the 3 most-relevant CLEAN instruments even below the
    # bar (gate disabled), so the doc shows what instrumenting *would* do to the
    # effect. Explicitly NOT a causal claim — these are weak instruments.
    illustrative: list[TwoSLSResult] = []
    clean = [s for s in screens if not s.flags and s.name not in
             {r.instrument for r in results}]
    for sc in clean[:3]:
        res = run_2sls_vs_ols(
            T, zser[sc.name], btc, label="congestion(t-1)->btc_return(t)",
            instrument_name=sc.name, train_window=train_window,
            f_threshold=0.0)  # gate off -> always instruments
        if res is not None:
            illustrative.append(res)

    regimes = _regime_relevance(T, btc, CONGESTION_INSTRUMENTS, raw,
                                train_window=train_window, lag_z=False,
                                post_cut="2024-01-01")
    return TaskOutput(screens, results,
                      "OLS effect = correlational loading of btc_return on "
                      "yesterday's congestion innovation.",
                      illustrative=illustrative, regimes=regimes)


def run_task2_congestion_to_funding(
    factors: pd.DataFrame, innov: pd.DataFrame, panel: pd.DataFrame,
    raw: pd.DataFrame, *, train_window: int, f_threshold: float,
) -> TaskOutput:
    """congestion innovation(t) -> aggregate funding(t+1), instrumented.

    Treatment = chain_congestion innovation at t. Outcome = aggregate funding at
    t+1 (shift -1 so the row pairs today's congestion with tomorrow's funding).
    Instrument = candidate metric innovation at t (contemporaneous with the
    treatment, predetermined w.r.t. tomorrow's funding).
    """
    cc = innov["chain_congestion"].rename("congestion")
    funding = _aligned_funding(panel)
    if funding.empty:
        return TaskOutput([], [], "no funding columns in panel")
    funding_next = funding.shift(-1).rename("funding_next")

    screens: list[Screen] = []
    results: list[TwoSLSResult] = []
    zser: dict[str, pd.Series] = {}
    for asset, metric, tf, story in CONGESTION_INSTRUMENTS:
        col = f"{asset}_{metric}"
        if col not in raw.columns:
            continue
        # Instrument contemporaneous with the treatment: undo the +1 lag that
        # _candidate_series applies so Z_t aligns with congestion_t.
        Zc = _candidate_series(raw[col].astype(float))[tf].shift(-1)
        name = f"{col}[{tf}]"
        zser[name] = Zc
        sc = screen_instrument(cc, Zc, funding_next, train_window=train_window)
        if sc is None:
            continue
        sc.name, sc.story = name, story
        screens.append(sc)

        if sc.train_f >= f_threshold and not any(
                fl.startswith("TAUTOLOGY") for fl in sc.flags):
            res = run_2sls_vs_ols(
                cc, Zc, funding_next, label="congestion(t)->funding(t+1)",
                instrument_name=name, train_window=train_window,
                f_threshold=f_threshold)
            if res is not None:
                results.append(res)

    screens.sort(key=lambda s: -s.train_f)

    illustrative: list[TwoSLSResult] = []
    clean = [s for s in screens if not s.flags and s.name not in
             {r.instrument for r in results}]
    for sc in clean[:3]:
        res = run_2sls_vs_ols(
            cc, zser[sc.name], funding_next, label="congestion(t)->funding(t+1)",
            instrument_name=sc.name, train_window=train_window, f_threshold=0.0)
        if res is not None:
            illustrative.append(res)

    regimes = _regime_relevance(cc, funding_next, CONGESTION_INSTRUMENTS, raw,
                                train_window=train_window, lag_z=True,
                                post_cut="2024-01-01")
    return TaskOutput(screens, results,
                      "OLS effect = correlational loading of next-day funding "
                      "on today's congestion innovation.",
                      illustrative=illustrative, regimes=regimes)


# ── markdown ─────────────────────────────────────────────────────────

def _screen_table(screens: list[Screen], f_threshold: float) -> list[str]:
    L = ["| Candidate instrument | train F | \\|corr(Z,T)\\| | direct-path t | "
         "n | flags | exclusion story |",
         "|---|---:|---:|---:|---:|---|---|"]
    for s in screens:
        flag = "; ".join(s.flags) if s.flags else "clean"
        L.append(f"| `{s.name}` | {s.train_f:.1f} | {s.corr_zt:.2f} | "
                 f"{s.direct_t:.2f} | {s.n_obs} | {flag} | {s.story} |")
    return L


def _ab_table(results: list[TwoSLSResult]) -> list[str]:
    L = ["| Instrument | OLS β (t) | 2SLS β (t) | full F | OLS OOS R² | "
         "2SLS OOS R² | OLS β̄ | 2SLS β̄ | sign agree | mean win F |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        L.append(
            f"| `{r.instrument}` | {r.ols_beta:+.4f} ({r.ols_t:.1f}) | "
            f"{r.tsls_beta:+.4f} ({r.tsls_t:.1f}) | {r.first_stage_f_full:.1f} | "
            f"{r.ols_oos_r2:+.4f} | {r.tsls_oos_r2:+.4f} | {r.ols_beta_mean:+.4f} | "
            f"{r.tsls_beta_mean:+.4f} | {r.sign_agree:.0%} | {r.mean_window_f:.1f} |")
    return L


def _regime_table(regimes: list[RegimeF]) -> list[str]:
    L = ["| Instrument | train-window F | full-sample F | post-2024 F | n post-2024 |",
         "|---|---:|---:|---:|---:|"]
    for r in regimes:
        pf = f"{r.post2024_f:.1f}" if r.post2024_f == r.post2024_f else "—"
        L.append(f"| `{r.name}` | {r.train_f:.1f} | {r.full_f:.1f} | {pf} | "
                 f"{r.n_post2024} |")
    return L


def render_markdown(
    t1: TaskOutput, t2: TaskOutput, args: dict,
) -> str:
    L = ["# Instrumenting the two surviving edges (congestion -> BTC, "
         "congestion -> funding)\n"]
    L.append("The whole-warehouse IV search closed factor->RETURNS 2SLS "
             "(`iv_findings.md`). This tests the two relationships that DID "
             "survive their placebo tests, with instruments chosen for an "
             "a-priori **exclusion story**, not just statistical relevance.\n")
    asset_str = (args["assets"] if isinstance(args["assets"], str)
                 else ", ".join(args["assets"]))
    L.append(f"- Assets: `{asset_str}`")
    L.append(f"- Window: `{args['start']}` -> `{args['end']}` | train "
             f"{args['train_window']}d | weak-instrument bar F ≥ "
             f"{args['f_threshold']}")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")
    L.append("**Selection discipline:** first-stage F, |corr(Z,T)| and "
             "direct-path t use the TRAIN window only. Exclusion is "
             "**untestable** — these instruments rest on an economic argument. "
             "Read every number as one-cycle / suggestive.\n")

    # ── Task 1 ──
    L.append("---\n\n## Task 1 — congestion(t-1) → btc_return(t)\n")
    L.append("Treatment = chain_congestion AR(1) innovation known at t-1. "
             "Outcome = BTC simple return at t. A valid Z shifts yesterday's "
             "congestion but reaches today's BTC return only through it. "
             "Direct-path t flags Z that predicts BTC return *beyond* "
             "congestion (exclusion red flag).\n")
    L.append("### Candidate screen (train window)\n")
    L += _screen_table(t1.screens, args["f_threshold"])
    L.append("")
    if t1.regimes:
        L.append("### First-stage relevance across regimes (diagnostic)\n")
        L.append("Selection uses the 2022 train window (top), but the edge "
                 "lives post-Dencun (2024+). This shows whether instruments are "
                 "any more relevant where the edge actually appears. Uses only "
                 "Z and T (never the outcome), so it cannot leak selection.\n")
        L += _regime_table(t1.regimes)
        L.append("")
    L.append("### 2SLS vs OLS (full sample + walk-forward OOS)\n")
    if t1.results:
        L += _ab_table(t1.results)
        L.append("")
        L.append("β is the per-unit effect of yesterday's congestion "
                 "innovation on today's BTC return (return units; ×1e4 ≈ bps). "
                 "OOS R² is 1-step walk-forward vs the train-mean baseline "
                 "(negative is normal for daily returns). 2SLS gates on "
                 "per-window first-stage F.\n")
    else:
        L.append("*No congestion instrument cleared the F ≥ "
                 f"{args['f_threshold']} relevance bar without a tautology "
                 "flag — no instrument passes the gate, so no causal estimate "
                 "is licensed.*\n")
    if t1.illustrative:
        L.append("#### Illustrative 2SLS on the most-relevant (sub-bar) "
                 "instruments\n")
        L.append("**Below the F ≥ 10 weak-instrument bar — NOT a causal "
                 "claim.** Shown only to see what instrumenting *would* do to "
                 "the effect. Weak-instrument 2SLS is biased toward OLS and has "
                 "inflated variance; large |2SLS β| with tiny F is the "
                 "signature of an unidentified estimate, not a finding.\n")
        L += _ab_table(t1.illustrative)
        L.append("")
    L.append(_verdict_task1(t1, args))

    # ── Task 2 ──
    L.append("\n---\n\n## Task 2 — congestion innovation(t) → funding(t+1)\n")
    L.append("Treatment = chain_congestion innovation at t. Outcome = "
             "cross-asset mean funding_rate_8h at t+1. Funding starts "
             "2023-11, so this sample is ~2 years and the walk-forward is "
             "short.\n")
    L.append("### Candidate screen (train window)\n")
    if t2.screens:
        L += _screen_table(t2.screens, args["f_threshold"])
    else:
        L.append(f"*{t2.ols_baseline_note}*")
    L.append("")
    if t2.regimes:
        L.append("### First-stage relevance across regimes (diagnostic)\n")
        L += _regime_table(t2.regimes)
        L.append("")
    L.append("### 2SLS vs OLS (full sample + walk-forward OOS)\n")
    if t2.results:
        L += _ab_table(t2.results)
        L.append("")
        L.append("β is the per-unit effect of today's congestion innovation on "
                 "tomorrow's aggregate funding (funding-rate units).\n")
    else:
        L.append("*No congestion instrument cleared the relevance bar for the "
                 "funding outcome — no instrument passes the gate, so no causal "
                 "estimate is licensed.*\n")
    if t2.illustrative:
        L.append("#### Illustrative 2SLS on the most-relevant (sub-bar) "
                 "instruments\n")
        L.append("**Below the F ≥ 10 bar — NOT a causal claim** (see Task 1 "
                 "note). Shown only to display the instrumented effect.\n")
        L += _ab_table(t2.illustrative)
        L.append("")
    L.append(_verdict_task2(t2, args))

    # ── overall caveats ──
    L.append("\n---\n\n## Caveats (apply to both)\n")
    L.append("- **Exclusion is untestable.** A clean direct-path t only means "
             "Z does not *visibly* predict the outcome beyond T on the train "
             "window; it cannot prove Z affects the outcome solely through T.")
    L.append("- **The congestion factor here is fee-based.** "
             "`avg_gas_price_gwei` is absent from this snapshot, so "
             "chain_congestion falls back to z(Σ FeeTotNtv) over "
             "{btc, doge, eth}. That SUM includes `btc_FeeTotNtv`, which "
             "co-moves with BTC activity — a built-in exclusion hazard for the "
             "congestion→BTC test specifically.")
    L.append("- **One cycle, sample-mined hypotheses.** Both edges were "
             "selected on this very sample; the placebo tests that anointed "
             "them do not price in that selection. Task 2's funding window is "
             "short (~2y).")
    L.append("- **2SLS for prediction throws away first-stage variance.** As "
             "`iv_findings.md` notes, even a valid strong instrument typically "
             "*lowers* OOS fit because it discards the endogenous component "
             "that still predicts. A 2SLS OOS R² below OLS is therefore NOT by "
             "itself evidence against causality; the diagnostic that matters "
             "is whether the *effect estimate* (sign/size) survives "
             "instrumenting.")
    L.append("")
    return "\n".join(L)


def _verdict_task1(t1: TaskOutput, args: dict) -> str:
    L = ["### Verdict — Task 1\n"]
    if not t1.results:
        L.append("**Unidentified.** No instrument with an exclusion story is "
                 "relevant enough (F ≥ bar) to congestion on the train window, "
                 "so the congestion→BTC edge cannot be tested causally here. "
                 "It remains *predictive only* — DML/placebo evidence stands, "
                 "but no IV corroborates a structural interpretation.")
        return "\n".join(L)
    strong = [r for r in t1.results if r.first_stage_f_full >= args["f_threshold"]]
    same_sign = [r for r in strong if np.sign(r.tsls_beta) == np.sign(r.ols_beta)]
    held = [r for r in same_sign if abs(r.tsls_beta) > 0.25 * abs(r.ols_beta)]
    if held:
        L.append(f"{len(held)}/{len(strong)} strong instruments leave the "
                 "congestion→BTC effect the **same sign** and a comparable "
                 "magnitude after instrumenting — consistent with a CAUSAL "
                 "reading: purging the endogenous part of congestion does not "
                 "erase the effect. But the OOS fit is the usual daily-return "
                 "noise and 2SLS standard errors widen, so this is corroboration"
                 ", not proof.")
    elif same_sign:
        L.append(f"Instruments preserve the effect's SIGN ({len(same_sign)}/"
                 f"{len(strong)}) but shrink/inflate its magnitude under "
                 "instrumenting — weak-to-mixed support for causality; the "
                 "point estimate is unstable.")
    else:
        L.append("Instrumenting **flips or destroys** the effect sign in every "
                 "strong instrument — the congestion→BTC relationship looks "
                 "PREDICTIVE (co-movement), not causal: the part of congestion "
                 "an instrument can move does not drive BTC the way the raw "
                 "factor's correlation implied.")
    return "\n".join(L)


def _verdict_task2(t2: TaskOutput, args: dict) -> str:
    L = ["### Verdict — Task 2\n"]
    if not t2.results:
        L.append("**Unidentified.** No exclusion-story instrument is relevant "
                 "enough to congestion on the funding sample's train window, so "
                 "the congestion→funding edge cannot be tested causally. It "
                 "remains *predictive only* (target_search +0.023 OOS R²).")
        return "\n".join(L)
    strong = [r for r in t2.results if r.first_stage_f_full >= args["f_threshold"]]
    same_sign = [r for r in strong if np.sign(r.tsls_beta) == np.sign(r.ols_beta)]
    held = [r for r in same_sign if abs(r.tsls_beta) > 0.25 * abs(r.ols_beta)]
    if held:
        L.append(f"{len(held)}/{len(strong)} strong instruments leave the "
                 "congestion→funding effect the **same sign** and comparable "
                 "size after instrumenting — consistent with a CAUSAL channel "
                 "(congestion surprise -> next-day funding), though on a short "
                 "~2y funding sample with one cycle.")
    elif same_sign:
        L.append(f"Sign preserved ({len(same_sign)}/{len(strong)}) but "
                 "magnitude unstable under instrumenting — weak support.")
    else:
        L.append("Instrumenting flips/destroys the effect in every strong "
                 "instrument — congestion→funding looks PREDICTIVE, not causal.")
    return "\n".join(L)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--f-threshold", type=float, default=10.0)
    p.add_argument("--out", default="causal_portfolio/docs/iv_surviving_edges.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    from causal_portfolio.data import get_loader
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro)
    innov = innovation_factors(factors)

    pairs = [(a, m) for a, m, _, _ in CONGESTION_INSTRUMENTS]
    raw = load_candidate_columns(pairs, args.start, args.end)
    raw.index = pd.to_datetime(raw.index)

    # Align candidate raw to factor index.
    common = raw.index.intersection(factors.index)
    raw = raw.loc[common]

    t1 = run_task1_congestion_to_btc(
        factors, innov, returns, raw,
        train_window=args.train_window, f_threshold=args.f_threshold)
    # Task 2 needs the post-2023-11 funding window; use a shorter train window
    # automatically if 252 leaves too little OOS.
    t2 = run_task2_congestion_to_funding(
        factors, innov, panel, raw,
        train_window=min(args.train_window, 180),
        f_threshold=args.f_threshold)

    md = render_markdown(t1, t2, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")

    print("\n=== TASK 1: congestion(t-1) -> btc_return(t) ===")
    for s in t1.screens:
        print(f"  {s.name:36s} F={s.train_f:7.1f} corr={s.corr_zt:.2f} "
              f"direct_t={s.direct_t:.2f} {'|'.join(s.flags) or 'clean'}")
    for r in t1.results + t1.illustrative:
        tag = "" if r in t1.results else "  [illustrative, sub-bar]"
        print(f"  -> {r.instrument}: OLS b={r.ols_beta:+.4f}(t{r.ols_t:.1f}) "
              f"2SLS b={r.tsls_beta:+.4f}(t{r.tsls_t:.1f}) fullF={r.first_stage_f_full:.1f} "
              f"OLS_oosR2={r.ols_oos_r2:+.4f} 2SLS_oosR2={r.tsls_oos_r2:+.4f}{tag}")
    print("\n=== TASK 2: congestion(t) -> funding(t+1) ===")
    for s in t2.screens:
        print(f"  {s.name:36s} F={s.train_f:7.1f} corr={s.corr_zt:.2f} "
              f"direct_t={s.direct_t:.2f} {'|'.join(s.flags) or 'clean'}")
    for r in t2.results + t2.illustrative:
        tag = "" if r in t2.results else "  [illustrative, sub-bar]"
        print(f"  -> {r.instrument}: OLS b={r.ols_beta:+.4f}(t{r.ols_t:.1f}) "
              f"2SLS b={r.tsls_beta:+.4f}(t{r.tsls_t:.1f}) fullF={r.first_stage_f_full:.1f} "
              f"OLS_oosR2={r.ols_oos_r2:+.4f} 2SLS_oosR2={r.tsls_oos_r2:+.4f}{tag}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
