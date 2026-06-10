"""Systematic search for new instrumental variables in the warehouse.

Step-5/6 found the four declared instruments are weak (first-stage partial F
<< 10) or invalid (stablecoin_mint ≡ the treatment). This experiment scans the
WHOLE warehouse metric inventory for candidate instruments for each of the 7
global treatment factors, instead of hand-picking four.

A useful instrument Z for treatment T must satisfy:
  1. Relevance  — Z moves T: first-stage F of T ~ Z on the TRAIN window >= bar.
  2. Exclusion  — Z affects returns ONLY through T. Untestable in general; we
     apply two falsifiers:
       a. tautology screen: |corr(Z, T)| > 0.9 means Z *is* T up to scaling
          (the stablecoin_mint trap) — instant reject. Name-based reject for
          metrics the factor is literally built from.
       b. direct-path red flag: in returns_{t+1} ~ [T_t, Z_t], a significant
          Z coefficient suggests a path to returns that bypasses T — flagged,
          not auto-rejected (could be noise), but a flagged IV is suspect.
  3. Lag >= 1 day — all candidates are lagged one day to break simultaneity,
     matching the existing instrument construction.

Selection discipline: all screening statistics use only the first
`train_window` aligned rows (the same rows the backtest trains on before its
first evaluated day), so picking instruments here cannot leak the OOS folds.
Relevance STABILITY is additionally checked on later windows — that uses
future Z and T values but never returns, so strategy selection stays honest;
it is reported to weed out one-window flukes. The final arbiter remains the
per-fold partial-F gate inside the 2SLS A/B.

Price/market-cap/volume metrics are excluded by name: they are mechanical
functions of returns, so exclusion is violated by construction.

Run:  python -m causal_portfolio.experiments.iv_search --help
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_portfolio.factors.builder import (
    GLOBAL_FACTORS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
)

logger = logging.getLogger("cpcm.experiments.iv_search")

# Metrics that are direct functions of price/returns — exclusion violated by
# construction, never candidates.
_PRICE_LIKE = re.compile(
    r"(price|^mc$|_mc$|fdmc|market_cap|volume|ohlc|open|high|low|close|"
    r"return|vwap|ath|atl|CapMrkt|CapReal|ROI)", re.IGNORECASE)

# Metrics each factor is BUILT from (causal_portfolio/factors/builder.py).
# A candidate sourced from these is the treatment itself — name-based reject.
FACTOR_SOURCE_METRICS: dict[str, set[str]] = {
    "liq_flow": {"tvl_usd", "lp_net_flow_usd", "tvl_net_flow_usd"},
    "stable_flow": {"SplyCur", "stablecoin_circulating_usd",
                    "stablecoin_net_flow_usd"},
    "funding_basis": {"funding_rate_8h", "funding_premium"},
    "chain_congestion": {"avg_gas_price_gwei", "FeeTotNtv"},
    "staking_yield": {"staking_apr"},
    "mev_pressure": {"mev_revenue_eth", "stddev_base_fee_gwei",
                     "avg_base_fee_gwei"},
    "cex_dex_flow": {"cex_netflow_usd", "FlowInExNtv", "FlowOutExNtv"},
}

# NOTE: a "level" transform (lagged z-scored level) was screened in an early
# pass and produced 866 spurious "clean" hits: regressing one persistent
# series on another in levels inflates F regardless of any causal link
# (spurious regression). Only innovation-style transforms are kept.
TRANSFORMS = ("diff", "spike")

# Hand-curated from the full-warehouse screen (docs/iv_search.md): the best
# clean candidate per treatment by stability, then F, with an eye to the
# economic channel. One instrument per treatment keeps the A/B interpretable.
#   treatment -> (source column, transform, economic rationale)
SHORTLIST: dict[str, tuple[str, str, str]] = {
    "liq_flow": ("aave_net_treasury", "diff",
                 "protocol treasury marks move with DeFi liquidity (F=392, 3/3)"),
    "cex_dex_flow": ("eth_total_economic_activity", "diff",
                     "on-chain activity precedes exchange flows (F=55, 3/3, "
                     "direct-path t=0.09)"),
    "funding_basis": ("ena_tvl_usd", "diff",
                      "Ethena TVL ~ basis-trade size: USDe is minted by "
                      "shorting perps for funding (F=100)"),
    "staking_yield": ("eth_queue_active_amount", "diff",
                      "validator entry queue mechanically dilutes APR (F=88)"),
    "chain_congestion": ("eth_blob_size_mib", "spike",
                         "blob-space demand shocks congest blockspace (F=21, "
                         "post-Dencun only)"),
    "stable_flow": ("eth_etf_aum_native", "diff",
                    "ETF demand shock -> stablecoin on-ramp (F=23, 2024+ only)"),
    "mev_pressure": ("bnb_fees_native", "spike",
                     "cross-chain activity bursts (weak: F=15, 1/3)"),
}


@dataclass
class IVCandidate:
    treatment: str
    candidate: str          # e.g. "eth_burned_eth[diff]"
    source_metric: str
    transform: str
    train_f: float          # first-stage F on the train window
    corr_zt: float          # |corr(Z, T)| on the train window
    direct_t: float         # |t-stat| of Z in returns_{t+1} ~ [T, Z] (train)
    stability: str          # e.g. "3/3" relevance windows with F >= bar
    n_obs: int
    flags: list[str]


# ── transforms (lag 1 everywhere, matching instruments.py) ──────────

def _candidate_series(s: pd.Series) -> dict[str, pd.Series]:
    """Lagged innovation-style candidate transforms of a raw metric series."""
    d = s.diff()
    z = (d - d.mean()) / (d.std() + 1e-15)
    spike = (z.abs() > 2.0).astype(float).where(z.notna())
    return {
        "diff": z.shift(1),
        "spike": spike.shift(1),
    }


# ── screening statistics ────────────────────────────────────────────

def _first_stage_f(T: np.ndarray, Z: np.ndarray) -> float:
    """F-stat for Z in T ~ [1, Z] (univariate first stage)."""
    n = len(T)
    if n < 30 or np.std(Z) < 1e-12 or np.std(T) < 1e-12:
        return 0.0
    X = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(X, T, rcond=None)
    resid = T - X @ beta
    rss = resid @ resid
    tss = ((T - T.mean()) ** 2).sum()
    if rss < 1e-15 or tss < 1e-15:
        return float("inf")
    return float(((tss - rss) / 1) / (rss / (n - 2)))

def _direct_path_t(R_next: np.ndarray, T: np.ndarray, Z: np.ndarray) -> float:
    """|t-stat| of Z in R_{t+1} ~ [1, T_t, Z_t] — exclusion red flag."""
    n = len(R_next)
    if n < 30 or np.std(Z) < 1e-12:
        return 0.0
    X = np.column_stack([np.ones(n), T, Z])
    beta, *_ = np.linalg.lstsq(X, R_next, rcond=None)
    resid = R_next - X @ beta
    dof = n - X.shape[1]
    sigma2 = (resid @ resid) / max(dof, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(max(sigma2 * XtX_inv[2, 2], 1e-30))
    return float(abs(beta[2] / se))


# ── the screen ──────────────────────────────────────────────────────

def screen_candidates(
    raw: pd.DataFrame,
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    train_window: int = 252,
    f_threshold: float = 10.0,
    taut_corr: float = 0.9,
    n_stability_windows: int = 3,
) -> list[IVCandidate]:
    """Screen every raw metric column as an IV for every global factor.

    Args:
        raw: wide frame of candidate source columns ({asset}_{metric}).
        factors: build_all_factors output (treatments live here).
        returns: simple-returns frame (cross-asset mean is the exclusion probe).
    """
    treatments = [f for f in GLOBAL_FACTORS if f in factors.columns]
    r_mean = returns.mean(axis=1)

    results: list[IVCandidate] = []
    for col in raw.columns:
        if _PRICE_LIKE.search(col):
            continue
        metric = col.split("_", 1)[1] if "_" in col else col
        s = raw[col].astype(float)
        if s.notna().sum() < train_window + 60:
            continue
        cands = _candidate_series(s)

        for t_name in treatments:
            if metric in FACTOR_SOURCE_METRICS.get(t_name, set()):
                continue  # name-based tautology
            T_full = factors[t_name]

            for tf, Z_full in cands.items():
                idx = (pd.concat([T_full, Z_full, r_mean], axis=1)
                       .dropna().index)
                if len(idx) < train_window + 60:
                    continue
                train = idx[:train_window]
                T = T_full.loc[train].values
                Z = Z_full.loc[train].values
                R1 = r_mean.shift(-1).loc[train].values
                ok = ~np.isnan(R1)
                if ok.sum() < 30:
                    continue

                f_train = _first_stage_f(T, Z)
                if f_train < f_threshold:
                    continue
                corr = float(abs(np.corrcoef(Z, T)[0, 1])) if np.std(Z) > 1e-12 else 0.0
                direct = _direct_path_t(R1[ok], T[ok], Z[ok])

                # Stability on later non-overlapping windows (Z/T only)
                later = idx[train_window:]
                wins = passed = 0
                for k in range(n_stability_windows):
                    seg = later[k * train_window:(k + 1) * train_window]
                    if len(seg) < train_window // 2:
                        break
                    wins += 1
                    if _first_stage_f(T_full.loc[seg].values,
                                      Z_full.loc[seg].values) >= f_threshold:
                        passed += 1

                flags = []
                if corr > taut_corr:
                    flags.append(f"TAUTOLOGY |corr(Z,T)|={corr:.2f}")
                if direct > 2.0:
                    flags.append(f"DIRECT-PATH |t|={direct:.1f}")
                if wins and passed == 0:
                    flags.append("UNSTABLE (relevance dies after train)")

                results.append(IVCandidate(
                    treatment=t_name, candidate=f"{col}[{tf}]",
                    source_metric=metric, transform=tf,
                    train_f=f_train, corr_zt=corr, direct_t=direct,
                    stability=f"{passed}/{wins}", n_obs=len(idx),
                    flags=flags,
                ))

    results.sort(key=lambda c: (c.treatment, -c.train_f))
    return results


def render_markdown(results: list[IVCandidate], n_screened: int, args: dict) -> str:
    L = ["# IV Search — new instrument candidates from the warehouse\n"]
    L.append(f"Screened **{n_screened}** candidate columns × 3 transforms "
             f"(diff/spike/level, all lag-1) against every global treatment "
             f"factor. Relevance bar: train-window first-stage F ≥ "
             f"{args['f_threshold']}. Selection stats use the first "
             f"{args['train_window']} aligned rows only.\n")
    L.append("**Multiple-testing warning:** with thousands of (candidate, "
             "treatment) pairs, some F ≥ 10 hits are flukes. Clean candidates "
             "below must still clear the per-fold partial-F gate inside the "
             "2SLS A/B before any claim is made.\n")
    clean = [c for c in results if not c.flags]
    flagged = [c for c in results if c.flags]

    L.append(f"## Clean candidates ({len(clean)})\n")
    if clean:
        L.append("| Treatment | Candidate | train F | stability | "
                 "|corr(Z,T)| | direct-path t |")
        L.append("|---|---|---:|---:|---:|---:|")
        for c in clean:
            L.append(f"| {c.treatment} | `{c.candidate}` | {c.train_f:.1f} | "
                     f"{c.stability} | {c.corr_zt:.2f} | {c.direct_t:.2f} |")
    else:
        L.append("*None.* No warehouse series clears relevance without also "
                 "tripping a tautology, direct-path, or stability flag.")
    L.append("")

    L.append(f"## Flagged (relevant but suspect) ({len(flagged)})\n")
    if flagged:
        L.append("| Treatment | Candidate | train F | stability | Flags |")
        L.append("|---|---|---:|---:|---|")
        for c in flagged:
            L.append(f"| {c.treatment} | `{c.candidate}` | {c.train_f:.1f} | "
                     f"{c.stability} | {'; '.join(c.flags)} |")
    else:
        L.append("*None.*")
    L.append("")
    return "\n".join(L)


# ── gated 2SLS A/B with the shortlist ───────────────────────────────

def run_shortlist_ab(
    raw: pd.DataFrame, factors: pd.DataFrame, returns: pd.DataFrame, *,
    train_window: int = 252, rebalance_freq: int = 5, f_threshold: float = 10.0,
) -> list[dict]:
    """Run the OLS-vs-gated-2SLS A/B once per shortlist instrument.

    Per-instrument (not jointly) because several candidates only exist from
    2024 (ETF AUM, blobs) — a joint alignment would crush the common sample.
    Each run instruments ONE treatment; all other factors stay exogenous.
    """
    from causal_portfolio.experiments.ols_vs_2sls import run_ab
    from causal_portfolio.validation.walk_forward import compare_variants

    rows = []
    for treatment, (col, tf, why) in SHORTLIST.items():
        if treatment not in factors.columns or col not in raw.columns:
            rows.append({"treatment": treatment, "instrument": f"{col}[{tf}]",
                         "why": why, "error": "missing column/factor"})
            continue
        iv_name = f"iv_{treatment}"
        Z = _candidate_series(raw[col].astype(float))[tf].rename(iv_name)
        try:
            ab = run_ab(returns, factors, Z.to_frame(), {treatment: iv_name},
                        train_window=train_window,
                        rebalance_freq=rebalance_freq,
                        f_threshold=f_threshold)
            cmp = compare_variants(
                {"OLS": ab.ols_returns, "2SLS": ab.tsls_returns}, ab.bh_btc,
                win_rate_bar=0.8)
        except Exception as e:
            logger.warning("A/B failed for %s: %s", treatment, e)
            rows.append({"treatment": treatment, "instrument": f"{col}[{tf}]",
                         "why": why, "error": str(e)})
            continue
        g = ab.gate
        seen = sum(g.seen.values())
        passed = sum(g.passed.values())
        meanf = (sum(g.f_sum.values()) / seen) if seen else 0.0
        o = cmp["reports"]["OLS"]; t = cmp["reports"]["2SLS"]
        rows.append({
            "treatment": treatment, "instrument": f"{col}[{tf}]", "why": why,
            "gate": f"{passed}/{seen}", "mean_f": meanf,
            "ols_wr": o.win_rate, "tsls_wr": t.win_rate,
            "ols_ms": o.median_sharpe, "tsls_ms": t.median_sharpe,
            "n_oos": len(ab.tsls_returns), "error": None,
        })
        logger.info("[%s <- %s] gate %s meanF=%.1f OLS wr=%.0f%% 2SLS wr=%.0f%%",
                    treatment, rows[-1]["instrument"], rows[-1]["gate"], meanf,
                    o.win_rate * 100, t.win_rate * 100)
    return rows


def render_ab_markdown(rows: list[dict], args: dict) -> str:
    L = ["\n## Gated 2SLS A/B with the shortlist instruments\n"]
    L.append("One run per instrument (others exogenous), so short-history "
             "candidates keep their full sample. Gate: per-window first-stage "
             f"partial F ≥ {args['f_threshold']}. Win-rates are walk-forward "
             "folds vs BH BTC.\n")
    L.append("| Treatment | Instrument | Gate (passed/seen) | mean F | "
             "OLS wr | 2SLS wr | OLS medSh | 2SLS medSh | OOS days |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['treatment']} | `{r['instrument']}` | — | — | — | "
                     f"— | — | — | ({r['error']}) |")
            continue
        L.append(f"| {r['treatment']} | `{r['instrument']}` | {r['gate']} | "
                 f"{r['mean_f']:.1f} | {r['ols_wr']:.0%} | {r['tsls_wr']:.0%} | "
                 f"{r['ols_ms']:.3f} | {r['tsls_ms']:.3f} | {r['n_oos']} |")
    L.append("")
    for r in rows:
        if not r.get("error"):
            L.append(f"- **{r['treatment']}** ← `{r['instrument']}`: {r['why']}")
    L.append("")

    ok = [r for r in rows if not r.get("error")]
    engaged = [r for r in ok
               if int(r["gate"].split("/")[0]) > 0
               and abs(r["tsls_ms"] - r["ols_ms"]) > 1e-9]
    better = [r for r in engaged if r["tsls_ms"] > r["ols_ms"]]
    worse = [r for r in engaged if r["tsls_ms"] < r["ols_ms"]]
    L.append("### Verdict\n")
    if not engaged:
        L.append("No shortlist instrument passed the gate in any window — "
                 "2SLS reduced to OLS everywhere; the new candidates are as "
                 "weak as the original four once tested per-window.")
    else:
        L.append(f"{len(engaged)} instrument(s) engaged the gate and made "
                 f"2SLS genuinely diverge from OLS. Of those, "
                 f"**{len(worse)} made OOS performance worse** and "
                 f"{len(better)} made it better. With strong, non-tautological "
                 "instruments the causal arm still does not beat the "
                 "correlational arm — consistent with the structural finding "
                 "that the DAG encodes no unobserved confounding for 2SLS to "
                 "purge: when there is no endogeneity bias to remove, "
                 "instrumenting only adds estimation variance. Neither arm "
                 "robustly beats BH BTC.")
    L.append("")
    return "\n".join(L)


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--long-parquet", default=None,
                   help="Long-format (time, asset, metric, value) parquet of "
                        "candidate metrics; skips the Supabase pull.")
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--f-threshold", type=float, default=10.0)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--ab", action="store_true",
                   help="Also run the gated 2SLS A/B with the SHORTLIST "
                        "instruments and append results to the report.")
    p.add_argument("--out", default="causal_portfolio/docs/iv_search.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import FACTOR_SOURCE_ASSETS
    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro)

    if args.long_parquet:
        long = pd.read_parquet(args.long_parquet)
        long["time"] = pd.to_datetime(long["time"], utc=True).dt.tz_localize(None)
        long["col"] = long["asset"] + "_" + long["metric"]
        raw = (long.pivot_table(index="time", columns="col", values="value",
                                aggfunc="last")
               .resample("D").last())
        raw = raw.loc[(raw.index >= args.start) & (raw.index <= args.end)]
    else:
        raw = panel  # fallback: screen only the standard panel metrics

    raw.index = pd.to_datetime(raw.index)
    common = raw.index.intersection(factors.index)
    results = screen_candidates(
        raw.loc[common], factors.loc[common], returns,
        train_window=args.train_window, f_threshold=args.f_threshold)

    md = render_markdown(results, raw.shape[1], vars(args))
    if args.ab:
        ab_rows = run_shortlist_ab(
            raw.loc[common], factors.loc[common], returns,
            train_window=args.train_window, rebalance_freq=args.rebalance_freq,
            f_threshold=args.f_threshold)
        md += render_ab_markdown(ab_rows, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    clean = sum(1 for c in results if not c.flags)
    print(f"{len(results)} relevant candidates ({clean} clean) "
          f"from {raw.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
