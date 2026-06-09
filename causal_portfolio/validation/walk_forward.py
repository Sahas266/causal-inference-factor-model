"""Walk-forward validation harness.

Design (see the project discussion): the CPCMBacktester already refits per
rebalance on a trailing window, so its output return series is *already*
walk-forward — every daily return reflects a decision made from past data
only. This harness therefore does NOT re-backtest. It takes that already-OOS
return series, partitions it into contiguous non-overlapping test folds, and
scores each fold against a benchmark — giving a *distribution* of performance
across sub-periods instead of one number, plus a robustness "win-rate" bar.

Why folds instead of one OOS number:
  - One fixed split is one arbitrary cut; a strategy can win/lose by luck of
    where the line falls. K folds show consistency across regimes.
  - Selecting a variant by its single OOS number silently turns that window
    into part of the selection. Judging by a win-rate across folds is harder
    to overfit, and `compare_variants` records the trial count so the best of
    many can be discounted.

Purge/embargo: because features use rolling windows (e.g. 21-day realized
vol), we drop `purge_days` at the start of each fold as a defensive trim
against any residual window bleed from the prior period. (The per-rebalance
refit already prevents decision-level look-ahead; this is belt-and-suspenders
at the scoring seam.) `embargo_days` leaves a gap between consecutive folds.

Annualization is 365 (crypto trades 24/7), matching build_regime_features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_portfolio.backtest.metrics import (
    ANNUALIZATION, max_drawdown, sharpe_ratio,
)


@dataclass(frozen=True)
class FoldResult:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    strat_total: float
    strat_sharpe: float
    strat_max_dd: float
    bench_total: float
    bench_sharpe: float
    beats_bench: bool


@dataclass
class WalkForwardReport:
    n_folds: int
    folds: list[FoldResult]
    median_sharpe: float
    mean_total: float
    win_rate: float                 # fraction of folds beating benchmark
    bench_median_sharpe: float
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.n_folds} folds | win-rate vs bench {self.win_rate:.0%} | "
                f"median Sharpe {self.median_sharpe:.3f} (bench "
                f"{self.bench_median_sharpe:.3f}) | mean total {self.mean_total:+.1%}")


def _total_return(r: np.ndarray) -> float:
    if len(r) == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


# ── shared report helpers (used by the experiment writers) ───────────


def metric_row(name: str, returns, *, with_dd: bool = False) -> str:
    """Markdown table row: total return, annualized Sharpe (365), optional MaxDD.

    One implementation so every experiment's OOS summary is computed identically
    (was copy-pasted as a local ``_line`` closure in each report writer).
    """
    r = np.asarray(returns)
    row = f"| {name} | {_total_return(r):+.1%} | {sharpe_ratio(r, annualization=ANNUALIZATION):.3f} |"
    return row + f" {max_drawdown(r):+.1%} |" if with_dd else row


def bh_btc_returns(returns: pd.DataFrame, index) -> pd.Series:
    """Buy-and-hold BTC benchmark aligned to ``index``.

    ``returns`` holds SIMPLE returns (loader default), so the benchmark
    compounds consistently with the strategy return series as-is.
    """
    if "btc_return" in returns:
        return returns["btc_return"].reindex(index).fillna(0.0)
    return pd.Series(0.0, index=index)


def make_test_folds(
    n: int, test_days: int, purge_days: int, embargo_days: int,
    min_fold_obs: int,
) -> list[tuple[int, int]]:
    """Partition an index range [0, n) into contiguous test folds.

    Returns a list of (start, end) index pairs (end exclusive). Each fold
    spans `test_days`; `purge_days` are trimmed from the start of each fold
    and `embargo_days` separate consecutive folds. A trailing partial fold
    shorter than `min_fold_obs` is dropped.
    """
    folds: list[tuple[int, int]] = []
    cursor = 0
    while cursor < n:
        raw_start = cursor
        raw_end = min(cursor + test_days, n)
        start = raw_start + purge_days
        end = raw_end
        if end - start >= min_fold_obs:
            folds.append((start, end))
        cursor = raw_end + embargo_days
    return folds


def evaluate(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    test_days: int = 126,
    purge_days: int = 21,
    embargo_days: int = 5,
    min_fold_obs: int = 40,
) -> WalkForwardReport:
    """Score an already-walk-forward strategy return series across folds.

    Args:
        strategy_returns: daily strategy returns, datetime-indexed (the OOS
            output of a walk-forward backtest).
        benchmark_returns: daily benchmark returns (e.g. BH BTC), same index
            space; aligned internally.
        test_days / purge_days / embargo_days / min_fold_obs: fold geometry.

    Returns:
        WalkForwardReport with per-fold metrics + aggregates.
    """
    common = strategy_returns.dropna().index.intersection(
        benchmark_returns.dropna().index)
    s = strategy_returns.loc[common]
    b = benchmark_returns.loc[common]
    n = len(common)

    fold_spans = make_test_folds(n, test_days, purge_days, embargo_days, min_fold_obs)
    folds: list[FoldResult] = []
    for i, (a, e) in enumerate(fold_spans):
        sr = s.iloc[a:e].values
        br = b.iloc[a:e].values
        s_total = _total_return(sr)
        b_total = _total_return(br)
        folds.append(FoldResult(
            index=i, start=common[a], end=common[e - 1], n_obs=e - a,
            strat_total=s_total,
            strat_sharpe=sharpe_ratio(sr, annualization=ANNUALIZATION),
            strat_max_dd=max_drawdown(sr),
            bench_total=b_total,
            bench_sharpe=sharpe_ratio(br, annualization=ANNUALIZATION),
            beats_bench=s_total > b_total,
        ))

    if folds:
        median_sharpe = float(np.median([f.strat_sharpe for f in folds]))
        mean_total = float(np.mean([f.strat_total for f in folds]))
        win_rate = float(np.mean([1.0 if f.beats_bench else 0.0 for f in folds]))
        bench_median = float(np.median([f.bench_sharpe for f in folds]))
    else:
        median_sharpe = mean_total = win_rate = bench_median = float("nan")

    return WalkForwardReport(
        n_folds=len(folds), folds=folds,
        median_sharpe=median_sharpe, mean_total=mean_total,
        win_rate=win_rate, bench_median_sharpe=bench_median,
        params=dict(test_days=test_days, purge_days=purge_days,
                    embargo_days=embargo_days, min_fold_obs=min_fold_obs),
    )


def compare_variants(
    variants: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    *,
    win_rate_bar: float = 0.8,
    **fold_kwargs,
) -> dict:
    """Evaluate many strategies on the SAME folds and rank by win-rate.

    Records the trial count so the best of many can be discounted (the more
    variants tried, the more likely the top one is luck). A variant "passes"
    only if win_rate >= win_rate_bar.

    Returns:
        {
          "reports": {name: WalkForwardReport},
          "ranking": [(name, win_rate, median_sharpe)],  # best first
          "n_variants": int,
          "passers": [names with win_rate >= bar],
          "multiple_testing_note": str,
        }
    """
    reports = {name: evaluate(r, benchmark_returns, **fold_kwargs)
               for name, r in variants.items()}
    ranking = sorted(
        ((n, rep.win_rate, rep.median_sharpe) for n, rep in reports.items()),
        key=lambda t: (t[1], t[2]), reverse=True,
    )
    passers = [n for n, wr, _ in ranking if wr >= win_rate_bar]
    n_var = len(variants)
    # Under pure noise, P(a variant beats bench in a fold) ~ 0.5, so the
    # expected max win-rate across n variants drifts up with n — flag it.
    note = (f"{n_var} variants tested. With ~0.5 per-fold coin-flip odds under "
            f"no edge, the best win-rate is upward-biased by selection across "
            f"{n_var} trials — treat a single top performer with suspicion "
            f"unless its margin over the field is large.")
    return {
        "reports": reports, "ranking": ranking, "n_variants": n_var,
        "passers": passers, "multiple_testing_note": note,
    }
