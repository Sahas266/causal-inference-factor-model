"""Direction 4 — Double ML causal effect estimates (+ heterogeneity).

Orthogonalized, cross-fitted estimates of each global factor's effect on
next-day returns, using econml's LinearDML: both the outcome (next-day
return) and the treatment (factor) are residualized on the OTHER factors
with cross-fitted nuisance models before the effect regression. This kills
the overfitting channel that plagued the naive OLS/2SLS loops, and the
statsmodels inference gives honest CIs.

Timing is strictly predictive: Y = r_{t+1}, T = factor_t, W = other
factors_t. Cross-fitting uses time-ordered splits (no shuffling).

Heterogeneity: CausalForestDML with X = [vixcls, funding_basis] asks whether
the effect of the strongest treatment differs by market state (the regime
hypothesis, tested properly this time).

Run:  python -m causal_portfolio.experiments.dml_effects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.experiments.dml_effects")


@dataclass
class DMLRow:
    treatment: str
    outcome: str
    effect_bps: float       # effect of +1σ factor on next-day return, bps
    ci_lo_bps: float
    ci_hi_bps: float
    significant: bool       # 95% CI excludes 0 (pre multiple-testing)


def _ts_splits(n: int, k: int = 5):
    """Time-ordered K-fold indices (contiguous test blocks, no shuffle)."""
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=k, shuffle=False).split(np.arange(n)))


def run_dml(
    factors: pd.DataFrame, returns: pd.DataFrame, *, alpha: float = 0.05,
) -> list[DMLRow]:
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV

    outcomes = {
        "btc_next": returns["btc_return"].shift(-1),
        "ew_next": returns.mean(axis=1).shift(-1),
    }
    rows: list[DMLRow] = []
    for t_name in factors.columns:
        others = [c for c in factors.columns if c != t_name]
        for o_name, y_s in outcomes.items():
            data = pd.concat(
                [y_s.rename("y"), factors[t_name].rename("t"),
                 factors[others]], axis=1).dropna()
            if len(data) < 300:
                continue
            Y = data["y"].values
            T = data["t"].values
            W = data[others].values
            est = LinearDML(
                model_y=LassoCV(max_iter=5000),
                model_t=LassoCV(max_iter=5000),
                cv=_ts_splits(len(Y)),
                random_state=0,
            )
            est.fit(Y, T, X=None, W=W, inference="statsmodels")
            eff = float(est.const_marginal_effect()) * 1e4
            lo, hi = est.const_marginal_effect_interval(alpha=alpha)
            lo, hi = float(lo) * 1e4, float(hi) * 1e4
            rows.append(DMLRow(t_name, o_name, eff, lo, hi,
                               significant=(lo > 0 or hi < 0)))
            logger.info("%s -> %s: %.1f bps [%.1f, %.1f]%s",
                        t_name, o_name, eff, lo, hi,
                        "  *" if rows[-1].significant else "")
    return rows


def run_cate(
    factors: pd.DataFrame, returns: pd.DataFrame, treatment: str,
    het_cols: list[str],
) -> dict | None:
    """CausalForestDML heterogeneity of `treatment`'s effect by market state."""
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LassoCV

    others = [c for c in factors.columns if c != treatment]
    y = returns.mean(axis=1).shift(-1).rename("y")
    data = pd.concat([y, factors[[treatment] + others]], axis=1).dropna()
    het = [h for h in het_cols if h in others]
    if len(data) < 300 or not het:
        return None
    Y = data["y"].values
    T = data[treatment].values
    X = data[het].values
    W = data[[c for c in others if c not in het]].values
    est = CausalForestDML(
        model_y=LassoCV(max_iter=5000), model_t=LassoCV(max_iter=5000),
        cv=_ts_splits(len(Y)), n_estimators=500, random_state=0,
    )
    est.fit(Y, T, X=X, W=W)

    out = {"treatment": treatment, "het_cols": het, "rows": []}
    for j, h in enumerate(het):
        lo_q, hi_q = np.quantile(X[:, j], [0.2, 0.8])
        x_lo, x_hi = np.median(X, axis=0).copy(), np.median(X, axis=0).copy()
        x_lo[j], x_hi[j] = lo_q, hi_q
        e_lo = float(est.effect(x_lo.reshape(1, -1))) * 1e4
        e_hi = float(est.effect(x_hi.reshape(1, -1))) * 1e4
        lb_lo, ub_lo = (float(v) * 1e4 for v in
                        est.effect_interval(x_lo.reshape(1, -1)))
        lb_hi, ub_hi = (float(v) * 1e4 for v in
                        est.effect_interval(x_hi.reshape(1, -1)))
        out["rows"].append((h, e_lo, (lb_lo, ub_lo), e_hi, (lb_hi, ub_hi)))
    return out


def render_markdown(rows: list[DMLRow], cate: dict | None, args: dict) -> str:
    from datetime import datetime, timezone
    L = ["# Double ML effect estimates (Direction 4)\n"]
    L.append("Cross-fitted LinearDML: effect of each factor (at t) on "
             "next-day return, residualizing both on all other factors with "
             "time-ordered folds. Effects are bps of next-day return per "
             "+1σ of the factor; 95% CIs from statsmodels inference.\n")
    n_tests = len(rows)
    L.append(f"**{n_tests} tests** — Bonferroni bar for family-wise 5% is "
             f"CI exclusion at α={0.05 / max(n_tests, 1):.4f}; the table "
             "flags plain 95% exclusions, so discount accordingly.\n")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")

    L.append("| Treatment | Outcome | Effect (bps/σ) | 95% CI | sig? |")
    L.append("|---|---|---:|---|:-:|")
    for r in sorted(rows, key=lambda r: -abs(r.effect_bps)):
        L.append(f"| {r.treatment} | {r.outcome} | {r.effect_bps:+.1f} | "
                 f"[{r.ci_lo_bps:+.1f}, {r.ci_hi_bps:+.1f}] | "
                 f"{'**yes**' if r.significant else ''} |")
    L.append("")

    if cate:
        L.append(f"## Heterogeneity (CausalForestDML): `{cate['treatment']}` "
                 f"on next-day EW return\n")
        L.append("| Moderator | effect @ 20th pct (CI) | effect @ 80th pct (CI) |")
        L.append("|---|---|---|")
        for h, e_lo, (a1, b1), e_hi, (a2, b2) in cate["rows"]:
            L.append(f"| {h} | {e_lo:+.1f} bps [{a1:+.1f}, {b1:+.1f}] | "
                     f"{e_hi:+.1f} bps [{a2:+.1f}, {b2:+.1f}] |")
        L.append("")

    sig = [r for r in rows if r.significant]
    L.append("## Verdict\n")
    if not sig:
        L.append("**No factor has a 95%-significant orthogonalized effect on "
                 "next-day returns** — fully consistent with every backtest "
                 "in the project. The proper, cross-fitted causal estimate of "
                 "the daily factor→return effect is indistinguishable from "
                 "zero across the board.")
    else:
        names = ", ".join(f"{r.treatment}→{r.outcome} ({r.effect_bps:+.1f} "
                          f"bps)" for r in sig)
        L.append(f"{len(sig)}/{n_tests} effects clear plain 95% CIs: {names}. "
                 f"At {n_tests} tests, ~{0.05 * n_tests:.1f} false positives "
                 "are expected; none survives Bonferroni unless its CI is "
                 "far from zero. Check the heterogeneity table before "
                 "reading anything into these.")
    return "\n".join(L)


def main() -> None:
    import argparse
    from pathlib import Path
    from causal_portfolio.data import get_loader
    from causal_portfolio.factors.builder import (
        FACTOR_SOURCE_ASSETS, GLOBAL_FACTORS, MACRO_SERIES, PANEL_METRICS,
        build_all_factors,
    )

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--innovations", action="store_true",
                   help="Use AR(1) innovations of the factors — if a "
                        "level effect was a persistence artifact, its "
                        "innovation version will be null.")
    p.add_argument("--out", default="causal_portfolio/docs/dml_effects.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS))
    panel = loader.load_panel(panel_assets, PANEL_METRICS, args.start, args.end)
    macro = loader.load_macro(MACRO_SERIES, args.start, args.end)
    returns = loader.load_returns(assets, args.start, args.end)
    factors = build_all_factors(panel, macro).dropna(axis=1, how="all")
    if args.innovations:
        from causal_portfolio.factors.builder import innovation_factors
        factors = innovation_factors(factors).dropna(axis=1, how="all")

    rows = run_dml(factors, returns)

    # Heterogeneity for the largest |effect| global treatment
    glob = [r for r in rows if r.treatment in GLOBAL_FACTORS
            and r.outcome == "ew_next"]
    cate = None
    if glob:
        top = max(glob, key=lambda r: abs(r.effect_bps)).treatment
        logger.info("CATE for top treatment: %s", top)
        cate = run_cate(factors, returns, top, ["vixcls", "funding_basis"])

    md = render_markdown(rows, cate, vars(args))
    Path(args.out).write_text(md, encoding="utf-8")
    sig = sum(r.significant for r in rows)
    print(f"{len(rows)} DML estimates, {sig} significant -> {args.out}")


if __name__ == "__main__":
    main()
