"""Daily target assessment for the repository's data-supported causal DAG.

RP-PCA was only an execution placeholder.  The actual causal model is DAG v2:
the frozen candidate edge is the one-day-lagged AR(1) innovation in aggregate
chain fees to next-day BTC return. The runner reproduces the corrected rule,
checks the prospective holdout registered on 2026-08-12, and refuses to emit
or execute a target until the causal edge earns deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from causal_portfolio.backtest.metrics import max_drawdown, sharpe_ratio
from causal_portfolio.execution import execute_target as execute_model_target
from causal_portfolio.execution.config import DEFAULT_ASSET_MAP, ExecutionConfig
from causal_portfolio.execution.run_logging import execution_run_log
from causal_portfolio.execution.targets import load_target_snapshot
from causal_portfolio.execution.types import TargetSnapshot
from causal_portfolio.scm.graph import GLOBAL_FACTORS, build_discovered_dag, summarize_dag

logger = logging.getLogger("cpcm.models.causal_daily")

MODEL_ASSET = "btc"
FACTOR_ASSETS = ("btc", "doge", "eth")
FACTOR_METRIC = "FeeTotNtv"
FACTOR_COLUMNS = tuple(f"{asset}_{FACTOR_METRIC}" for asset in FACTOR_ASSETS)
FACTOR_NAME = "chain_congestion"
MODEL_HISTORY_START = "2022-01-01"
MODEL_REGISTERED_AT = pd.Timestamp("2026-08-12")
VALIDATION_DECISION_START = MODEL_REGISTERED_AT + pd.Timedelta(days=1)
VALIDATION_START = VALIDATION_DECISION_START + pd.Timedelta(days=1)
VALIDATION_OBSERVATIONS = 180
VALIDATION_SHARPE_LIFT = 0.10
VALIDATION_COST_BPS = 5.0
AR_MIN_OBSERVATIONS = 60
AR_WINDOW_DAYS = 252
SNAPSHOT_OVERLAP_DAYS = 7
STRATEGY_NAME = "CPCM DAG v2 causal forward validation"
RP_PCA_STRATEGY_NAME = "RP-PCA daily tangency"
DEFAULT_RP_PCA_REFERENCE_TARGET = Path("tmp/rppca_daily_target.json")
DEFAULT_SIGNAL_LEDGER = Path(
    "causal_portfolio/execution/state/cpcm_causal_signal_ledger.jsonl"
)

MODEL_SPEC = {
    "version": "dag_v2_causal_daily_2026_08_12",
    "registered_at": MODEL_REGISTERED_AT.date().isoformat(),
    "history_start": MODEL_HISTORY_START,
    "factor": FACTOR_NAME,
    "source_columns": FACTOR_COLUMNS,
    "source_fill": "none",
    "innovation": "trailing_ar1_then_prefix_zscore",
    "ar_min_observations": AR_MIN_OBSERVATIONS,
    "ar_window_days": AR_WINDOW_DAYS,
    "signal_lag_calendar_days": 1,
    "exposure_rule": "clip(1 + 0.5 * z, 0, 2)",
    "outcome": "next_scheduled_decision_btc_mark_to_mark_return",
    "observation_store": "append_only_sha256_chain",
    "decision_start": VALIDATION_DECISION_START.date().isoformat(),
    "validation_start": VALIDATION_START.date().isoformat(),
    "validation_observations": VALIDATION_OBSERVATIONS,
    "validation_required_sharpe_lift": VALIDATION_SHARPE_LIFT,
}
MODEL_SPEC_SHA256 = hashlib.sha256(
    json.dumps(MODEL_SPEC, separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()


class CausalModelNotDeployable(RuntimeError):
    """The frozen causal model has not passed its deployment gate."""


@dataclass(frozen=True)
class ForwardValidation:
    status: str
    passed: bool
    n_available: int
    n_evaluated: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    strategy_total_return: float | None
    benchmark_total_return: float | None
    strategy_sharpe: float | None
    benchmark_sharpe: float | None
    sharpe_lift: float | None
    net_strategy_sharpe_5bps: float | None
    net_sharpe_lift_5bps: float | None
    strategy_max_drawdown: float | None
    benchmark_max_drawdown: float | None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "passed": self.passed,
            "n_available": self.n_available,
            "n_evaluated": self.n_evaluated,
            "start": self.start.date().isoformat() if self.start is not None else None,
            "end": self.end.date().isoformat() if self.end is not None else None,
            "required_observations": VALIDATION_OBSERVATIONS,
            "required_sharpe_lift": VALIDATION_SHARPE_LIFT,
            "strategy_total_return": self.strategy_total_return,
            "benchmark_total_return": self.benchmark_total_return,
            "strategy_sharpe": self.strategy_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "sharpe_lift": self.sharpe_lift,
            "net_strategy_sharpe_5bps": self.net_strategy_sharpe_5bps,
            "net_sharpe_lift_5bps": self.net_sharpe_lift_5bps,
            "diagnostic_cost_bps": VALIDATION_COST_BPS,
            "strategy_max_drawdown": self.strategy_max_drawdown,
            "benchmark_max_drawdown": self.benchmark_max_drawdown,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CausalDailyResult:
    weights: dict[str, float]
    last_data_date: pd.Timestamp
    generated_at: datetime
    target_path: Path
    signal_z: float
    exposure_multiplier: float
    base_weight: float
    validation: ForwardValidation
    model_prices: dict[str, float]


def _daily_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"no {name} data loaded")
    out = frame.copy().sort_index()
    index = pd.to_datetime(out.index, utc=True).tz_convert(None).normalize()
    if index.has_duplicates:
        duplicate = index[index.duplicated()][0]
        raise ValueError(f"{name} has duplicate daily row {duplicate.date()}")
    out.index = index
    return out


def validate_model_signature() -> dict[str, object]:
    """Assert that the frozen production rule still matches discovered DAG v2."""
    dag = build_discovered_dag([MODEL_ASSET])
    edge = dag.get_edge_data(FACTOR_NAME, f"{MODEL_ASSET}_return")
    if edge is None:
        raise ValueError("DAG v2 is missing its frozen congestion-to-BTC edge")
    if edge.get("lag") != 1 or edge.get("stability") != "candidate":
        raise ValueError(f"DAG v2 edge signature changed: {edge}")
    if dag.nodes[FACTOR_NAME].get("transform") != "ar1_innovation":
        raise ValueError("DAG v2 chain_congestion transform is not ar1_innovation")
    direct_factors = {
        source
        for source, target in dag.in_edges(f"{MODEL_ASSET}_return")
        if source in GLOBAL_FACTORS
    }
    if direct_factors != {FACTOR_NAME}:
        raise ValueError(
            "DAG v2 executable factor set changed: "
            f"expected {[FACTOR_NAME]}, got {sorted(direct_factors)}"
        )
    return {
        "version": MODEL_SPEC["version"],
        "spec_sha256": MODEL_SPEC_SHA256,
        "registered_at": MODEL_SPEC["registered_at"],
        "validation_start": MODEL_SPEC["validation_start"],
        "treatment": FACTOR_NAME,
        "outcome": f"{MODEL_ASSET}_return",
        "lag_days": 1,
        "transform": "ar1_innovation",
        "stability": edge["stability"],
        "summary": summarize_dag(dag),
    }


def build_fixed_factor(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen fee aggregate without imputing any source."""
    panel = _daily_frame(panel, "factor source")
    missing = [column for column in FACTOR_COLUMNS if column not in panel]
    if missing:
        raise ValueError(f"missing frozen DAG v2 source columns: {', '.join(missing)}")
    source = panel[list(FACTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan)
    empty = [column for column in source if source[column].notna().sum() == 0]
    if empty:
        raise ValueError(f"empty frozen DAG v2 source columns: {', '.join(empty)}")
    return source.sum(axis=1, min_count=len(FACTOR_COLUMNS)).rename(
        FACTOR_NAME
    ).to_frame()


def _z_score(series: pd.Series) -> pd.Series:
    std = series.std()
    if std is None or not math.isfinite(float(std)) or float(std) < 1e-15:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - series.mean()) / std


def _frozen_innovation(factor: pd.Series) -> pd.Series:
    """Frozen leak-free AR(1) innovation transform for one history prefix."""
    level = _z_score(factor.astype(float))
    lag = level.shift(1)
    rolling_cov = lag.rolling(
        AR_WINDOW_DAYS, min_periods=AR_MIN_OBSERVATIONS
    ).cov(level)
    rolling_var = lag.rolling(
        AR_WINDOW_DAYS, min_periods=AR_MIN_OBSERVATIONS
    ).var()
    rho_unshifted = rolling_cov / (rolling_var + 1e-15)
    rho = rho_unshifted.shift(1)
    intercept = (
        level.rolling(AR_WINDOW_DAYS, min_periods=AR_MIN_OBSERVATIONS).mean()
        - rho_unshifted
        * lag.rolling(AR_WINDOW_DAYS, min_periods=AR_MIN_OBSERVATIONS).mean()
    ).shift(1)
    return _z_score(level - (intercept + rho * lag))


def online_innovation_signal(
    factors: pd.DataFrame,
    *,
    decision_start: pd.Timestamp = VALIDATION_DECISION_START - pd.Timedelta(days=1),
) -> pd.Series:
    """Reproduce the innovation transform as it would have looked each day.

    The transform is implemented here rather than imported from the research
    helper so later research changes cannot silently alter the registered model.
    Prefix recomputation also makes every normalization causal: no later
    observation can revise an earlier decision.
    """
    factors = _daily_frame(factors, "factor")
    if list(factors.columns) != [FACTOR_NAME]:
        raise ValueError(f"expected only the frozen factor {FACTOR_NAME!r}")
    values: dict[pd.Timestamp, float] = {}
    for date in factors.index[factors.index >= decision_start]:
        transformed = _frozen_innovation(factors.loc[:date, FACTOR_NAME])
        if transformed.empty:
            continue
        value = float(transformed.iloc[-1])
        if math.isfinite(value):
            values[date] = value
    if not values:
        raise ValueError("no finite DAG v2 innovation signals are available")
    return pd.Series(values, name="congestion_innovation_z", dtype=float)


def exposure_from_signal(signal: pd.Series | float) -> pd.Series | float:
    """Frozen DAG v2 rule: clip(1 + 0.5 z, 0, 2)."""
    if isinstance(signal, pd.Series):
        return (1.0 + 0.5 * signal.astype(float)).clip(0.0, 2.0)
    return float(np.clip(1.0 + 0.5 * float(signal), 0.0, 2.0))


def _signal_record_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_signal_ledger(path: str | Path) -> list[dict[str, object]]:
    """Load and authenticate the append-only prospective signal ledger."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    records: list[dict[str, object]] = []
    previous_hash: str | None = None
    previous_date: pd.Timestamp | None = None
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank causal ledger line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid causal ledger line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"causal ledger line {line_number} is not an object")
        stored_hash = record.get("record_sha256")
        unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
        if stored_hash != _signal_record_hash(unhashed):
            raise ValueError(f"causal ledger hash mismatch on line {line_number}")
        if record.get("previous_record_sha256") != previous_hash:
            raise ValueError(f"causal ledger chain mismatch on line {line_number}")
        if record.get("model_spec_sha256") != MODEL_SPEC_SHA256:
            raise ValueError(f"causal ledger model mismatch on line {line_number}")
        decision_date = pd.Timestamp(str(record.get("decision_date"))).normalize()
        source_date = pd.Timestamp(str(record.get("source_date"))).normalize()
        if source_date != decision_date - pd.Timedelta(days=1):
            raise ValueError(f"causal ledger source lag mismatch on line {line_number}")
        if previous_date is not None and decision_date <= previous_date:
            raise ValueError(f"causal ledger dates are not increasing on line {line_number}")
        for field in ("signal_z", "exposure_multiplier", "btc_reference_price"):
            value = float(record[field])
            if not math.isfinite(value):
                raise ValueError(
                    f"causal ledger {field} is non-finite on line {line_number}"
                )
        if float(record["btc_reference_price"]) <= 0:
            raise ValueError(
                f"causal ledger BTC price is not positive on line {line_number}"
            )
        records.append(record)
        previous_hash = str(stored_hash)
        previous_date = decision_date
    return records


def record_signal_observation(
    path: str | Path,
    *,
    now: datetime,
    source_date: pd.Timestamp,
    signal_z: float,
    btc_reference_price: float,
) -> dict[str, object]:
    """Append one immutable daily decision, or return the existing exact one."""
    decision_time = now.astimezone(timezone.utc)
    decision_date = pd.Timestamp(decision_time.date())
    if decision_date < VALIDATION_DECISION_START:
        raise CausalModelNotDeployable(
            "DAG v2 prospective registration is not active until "
            f"{VALIDATION_DECISION_START.date().isoformat()}"
        )
    source_date = pd.Timestamp(source_date).normalize()
    if source_date != decision_date - pd.Timedelta(days=1):
        raise ValueError(
            "DAG v2 decision requires the previous complete UTC day: "
            f"decision={decision_date.date()}, source={source_date.date()}"
        )
    if not math.isfinite(signal_z):
        raise ValueError(f"invalid causal signal {signal_z}")
    if not math.isfinite(btc_reference_price) or btc_reference_price <= 0:
        raise ValueError(f"invalid BTC reference price {btc_reference_price}")

    ledger_path = Path(path)
    records = load_signal_ledger(ledger_path)
    if records:
        existing_date = pd.Timestamp(str(records[-1]["decision_date"]))
        if existing_date == decision_date:
            existing = records[-1]
            if (
                str(existing["source_date"]) != source_date.date().isoformat()
                or not math.isclose(
                    float(existing["signal_z"]), signal_z, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    "immutable causal decision conflicts with revised source data"
                )
            return existing
        if existing_date > decision_date:
            raise ValueError("causal ledger already contains a later decision")

    payload: dict[str, object] = {
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "decision_date": decision_date.date().isoformat(),
        "decision_utc": decision_time.isoformat(timespec="seconds"),
        "source_date": source_date.date().isoformat(),
        "signal_z": float(signal_z),
        "exposure_multiplier": float(exposure_from_signal(signal_z)),
        "btc_reference_price": float(btc_reference_price),
        "previous_record_sha256": (
            str(records[-1]["record_sha256"]) if records else None
        ),
    }
    record = {**payload, "record_sha256": _signal_record_hash(payload)}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        ledger_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return record


def evaluate_signal_ledger(
    records: list[dict[str, object]],
) -> ForwardValidation:
    """Score the first 180 outcomes from immutable scheduled decision marks."""
    decision_dates = [
        pd.Timestamp(str(record["decision_date"])).normalize() for record in records
    ]
    if decision_dates and decision_dates[0] != VALIDATION_DECISION_START:
        return ForwardValidation(
            status="invalid_data",
            passed=False,
            n_available=max(len(records) - 1, 0),
            n_evaluated=0,
            start=None,
            end=None,
            strategy_total_return=None,
            benchmark_total_return=None,
            strategy_sharpe=None,
            benchmark_sharpe=None,
            sharpe_lift=None,
            net_strategy_sharpe_5bps=None,
            net_sharpe_lift_5bps=None,
            strategy_max_drawdown=None,
            benchmark_max_drawdown=None,
            reason=(
                "first decision must be "
                f"{VALIDATION_DECISION_START.date().isoformat()}"
            ),
        )
    for left, right in zip(decision_dates, decision_dates[1:]):
        if right != left + pd.Timedelta(days=1):
            return ForwardValidation(
                status="invalid_data",
                passed=False,
                n_available=max(len(records) - 1, 0),
                n_evaluated=0,
                start=decision_dates[1] if len(decision_dates) > 1 else None,
                end=decision_dates[-1] if len(decision_dates) > 1 else None,
                strategy_total_return=None,
                benchmark_total_return=None,
                strategy_sharpe=None,
                benchmark_sharpe=None,
                sharpe_lift=None,
                net_strategy_sharpe_5bps=None,
                net_sharpe_lift_5bps=None,
                strategy_max_drawdown=None,
                benchmark_max_drawdown=None,
                reason=f"missing decision date after {left.date().isoformat()}",
            )

    available = max(len(records) - 1, 0)
    outcome_dates = decision_dates[1 : VALIDATION_OBSERVATIONS + 1]
    if available < VALIDATION_OBSERVATIONS:
        return ForwardValidation(
            status="pending",
            passed=False,
            n_available=available,
            n_evaluated=available,
            start=outcome_dates[0] if outcome_dates else None,
            end=outcome_dates[-1] if outcome_dates else None,
            strategy_total_return=None,
            benchmark_total_return=None,
            strategy_sharpe=None,
            benchmark_sharpe=None,
            sharpe_lift=None,
            net_strategy_sharpe_5bps=None,
            net_sharpe_lift_5bps=None,
            strategy_max_drawdown=None,
            benchmark_max_drawdown=None,
        )

    frozen = records[: VALIDATION_OBSERVATIONS + 1]
    prices = np.asarray(
        [float(record["btc_reference_price"]) for record in frozen], dtype=float
    )
    benchmark = prices[1:] / prices[:-1] - 1.0
    held = np.asarray(
        [float(record["exposure_multiplier"]) for record in frozen[:-1]],
        dtype=float,
    )
    strategy = held * benchmark
    turnover = np.abs(np.diff(np.concatenate(([1.0], held))))
    net_strategy = strategy - turnover * VALIDATION_COST_BPS / 10_000.0
    strategy_sharpe = float(sharpe_ratio(strategy))
    benchmark_sharpe = float(sharpe_ratio(benchmark))
    net_strategy_sharpe = float(sharpe_ratio(net_strategy))
    lift = strategy_sharpe - benchmark_sharpe
    return ForwardValidation(
        status="passed" if lift >= VALIDATION_SHARPE_LIFT else "failed",
        passed=lift >= VALIDATION_SHARPE_LIFT,
        n_available=available,
        n_evaluated=VALIDATION_OBSERVATIONS,
        start=decision_dates[1],
        end=decision_dates[VALIDATION_OBSERVATIONS],
        strategy_total_return=float(np.prod(1.0 + strategy) - 1.0),
        benchmark_total_return=float(np.prod(1.0 + benchmark) - 1.0),
        strategy_sharpe=strategy_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        sharpe_lift=lift,
        net_strategy_sharpe_5bps=net_strategy_sharpe,
        net_sharpe_lift_5bps=net_strategy_sharpe - benchmark_sharpe,
        strategy_max_drawdown=float(max_drawdown(strategy)),
        benchmark_max_drawdown=float(max_drawdown(benchmark)),
    )


def evaluate_forward_validation(
    signal: pd.Series,
    returns: pd.DataFrame,
    *,
    validation_start: pd.Timestamp = VALIDATION_START,
) -> ForwardValidation:
    """Evaluate the first 180 exact calendar outcomes after registration."""
    returns = _daily_frame(returns, "return")
    return_col = f"{MODEL_ASSET}_return"
    if return_col not in returns:
        raise ValueError(f"missing return column {return_col!r}")
    start = pd.Timestamp(validation_start).normalize()
    clean_signal = signal.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    clean_return = (
        returns[return_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    )
    if clean_signal.empty or clean_return.empty:
        available_end = start - pd.Timedelta(days=1)
    else:
        available_end = min(
            clean_return.index.max(),
            clean_signal.index.max() + pd.Timedelta(days=1),
        )
    calendar = (
        pd.date_range(start, available_end, freq="D")
        if available_end >= start
        else pd.DatetimeIndex([])
    )
    frozen_calendar = calendar[:VALIDATION_OBSERVATIONS]
    decision_calendar = frozen_calendar - pd.Timedelta(days=1)
    missing_returns = frozen_calendar.difference(clean_return.index)
    missing_signals = decision_calendar.difference(clean_signal.index)
    if len(missing_returns) or len(missing_signals):
        parts = []
        if len(missing_returns):
            parts.append(
                "missing return date(s): "
                + ", ".join(day.date().isoformat() for day in missing_returns[:5])
            )
        if len(missing_signals):
            parts.append(
                "missing decision date(s): "
                + ", ".join(day.date().isoformat() for day in missing_signals[:5])
            )
        reason = "; ".join(parts)
        return ForwardValidation(
            status="invalid_data",
            passed=False,
            n_available=len(frozen_calendar),
            n_evaluated=0,
            start=frozen_calendar.min() if len(frozen_calendar) else None,
            end=frozen_calendar.max() if len(frozen_calendar) else None,
            strategy_total_return=None,
            benchmark_total_return=None,
            strategy_sharpe=None,
            benchmark_sharpe=None,
            sharpe_lift=None,
            net_strategy_sharpe_5bps=None,
            net_sharpe_lift_5bps=None,
            strategy_max_drawdown=None,
            benchmark_max_drawdown=None,
            reason=reason,
        )

    available = len(calendar)
    if available < VALIDATION_OBSERVATIONS:
        return ForwardValidation(
            status="pending",
            passed=False,
            n_available=available,
            n_evaluated=available,
            start=frozen_calendar.min() if len(frozen_calendar) else None,
            end=frozen_calendar.max() if len(frozen_calendar) else None,
            strategy_total_return=None,
            benchmark_total_return=None,
            strategy_sharpe=None,
            benchmark_sharpe=None,
            sharpe_lift=None,
            net_strategy_sharpe_5bps=None,
            net_sharpe_lift_5bps=None,
            strategy_max_drawdown=None,
            benchmark_max_drawdown=None,
        )

    frozen = calendar[:VALIDATION_OBSERVATIONS]
    decisions = frozen - pd.Timedelta(days=1)
    held = pd.Series(
        exposure_from_signal(clean_signal.reindex(decisions)).to_numpy(dtype=float),
        index=frozen,
        name="exposure",
    )
    previous_decision = decisions[0] - pd.Timedelta(days=1)
    previous_signal = clean_signal.get(previous_decision)
    initial_exposure = (
        float(exposure_from_signal(previous_signal))
        if previous_signal is not None and math.isfinite(float(previous_signal))
        else 1.0
    )
    turnover = held.diff()
    turnover.iloc[0] = held.iloc[0] - initial_exposure
    turnover = turnover.abs()
    benchmark = np.expm1(clean_return.reindex(frozen).to_numpy(dtype=float))
    strategy = held.to_numpy(dtype=float) * benchmark
    net_strategy = strategy - (
        turnover.to_numpy(dtype=float) * VALIDATION_COST_BPS / 10_000.0
    )
    strategy_sharpe = float(sharpe_ratio(strategy))
    benchmark_sharpe = float(sharpe_ratio(benchmark))
    net_strategy_sharpe = float(sharpe_ratio(net_strategy))
    lift = strategy_sharpe - benchmark_sharpe
    return ForwardValidation(
        status="passed" if lift >= VALIDATION_SHARPE_LIFT else "failed",
        passed=lift >= VALIDATION_SHARPE_LIFT,
        n_available=available,
        n_evaluated=len(frozen),
        start=frozen[0],
        end=frozen[-1],
        strategy_total_return=float(np.prod(1.0 + strategy) - 1.0),
        benchmark_total_return=float(np.prod(1.0 + benchmark) - 1.0),
        strategy_sharpe=strategy_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        sharpe_lift=lift,
        net_strategy_sharpe_5bps=net_strategy_sharpe,
        net_sharpe_lift_5bps=net_strategy_sharpe - benchmark_sharpe,
        strategy_max_drawdown=float(max_drawdown(strategy)),
        benchmark_max_drawdown=float(max_drawdown(benchmark)),
    )


def _validate_latest_window(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    now: datetime,
    min_history_days: int,
    max_data_age_hours: float,
    allow_stale: bool,
) -> pd.Timestamp:
    if min_history_days < 60:
        raise ValueError(f"min_history_days must be >= 60, got {min_history_days}")
    panel = _daily_frame(panel, "factor source")
    prices = _daily_frame(prices, "price")
    if MODEL_ASSET not in prices:
        raise ValueError(f"missing price column {MODEL_ASSET!r}")
    source = panel[list(FACTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan)
    first_required = pd.Timestamp(MODEL_HISTORY_START)
    if source.index.min() > first_required:
        raise ValueError(
            "DAG v2 source history starts too late: "
            f"{source.index.min().date().isoformat()} > {MODEL_HISTORY_START}"
        )
    latest_by_source = {
        column: source[column].last_valid_index() for column in FACTOR_COLUMNS
    }
    if any(date is None for date in latest_by_source.values()):
        empty = [column for column, date in latest_by_source.items() if date is None]
        raise ValueError(f"empty frozen DAG v2 source columns: {', '.join(empty)}")
    # Trim to the last day every source covers — the newest date the
    # aggregate chain-fee factor can be computed honestly.
    #
    # Requiring the sources to agree stalled the model for hours a day: Coin
    # Metrics publishes per asset at different times, so btc can sit a day
    # behind eth/doge through no fault of the data. No calendar cap is applied
    # on top: every writer into this table records completed days only (Coin
    # Metrics publishes closed days; the CoinGecko filler drops the partial
    # current day), so the floor is already a settled date. Contiguity within
    # the window is still enforced below, so this trims the tail without
    # tolerating a hole.
    last_date = min(latest_by_source.values())
    lagging = sorted(
        column for column, date in latest_by_source.items() if date > last_date
    )
    if lagging:
        logger.info(
            "DAG v2 trimmed to %s; ahead of the window: %s",
            last_date.date().isoformat(), ", ".join(lagging),
        )
    source = source.loc[source.index <= last_date]
    expected = pd.date_range(first_required, last_date, freq="D")
    source_window = source.reindex(expected)
    missing_source = source_window.isna()
    if missing_source.any().any():
        missing = [
            f"{column}@{date.date().isoformat()}"
            for date, row in missing_source.iterrows()
            for column, is_missing in row.items()
            if is_missing
        ]
        raise ValueError(
            "DAG v2 raw sources are not calendar-contiguous; missing "
            + ", ".join(missing[:8])
        )
    factor_window = factors[FACTOR_NAME].reindex(expected)
    if factor_window.isna().any():
        missing = factor_window.index[factor_window.isna()]
        preview = ", ".join(day.date().isoformat() for day in missing[:8])
        raise ValueError(f"DAG v2 factor has missing calendar day(s): {preview}")
    price_window = (
        pd.to_numeric(prices[MODEL_ASSET], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .reindex(expected)
    )
    if price_window.isna().any():
        missing = price_window.index[price_window.isna()]
        preview = ", ".join(day.date().isoformat() for day in missing[:8])
        raise ValueError(f"DAG v2 BTC prices have missing calendar day(s): {preview}")
    if len(expected) < min_history_days:
        raise ValueError(
            f"only {len(expected)} complete DAG v2 rows; need {min_history_days}"
        )
    current = now.astimezone(timezone.utc)
    signal_time = last_date.to_pydatetime().replace(tzinfo=timezone.utc)
    age_hours = (current - signal_time).total_seconds() / 3600.0
    if age_hours < -(5.0 / 60.0):
        raise ValueError(f"DAG v2 data date is {-age_hours:.1f}h in the future")
    if not allow_stale and age_hours > max_data_age_hours:
        raise ValueError(
            f"DAG v2 raw inputs are stale: latest complete row is "
            f"{last_date.date().isoformat()} ({age_hours:.1f}h old; "
            f"limit {max_data_age_hours:.1f}h)"
        )
    return last_date


def build_causal_target(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    now: datetime,
    base_weight: float = 0.05,
    min_history_days: int = 252,
    max_data_age_hours: float = 72.0,
    allow_stale: bool = False,
    validation_override: ForwardValidation | None = None,
    model_price_override: float | None = None,
    allow_unvalidated_edge: bool = False,
) -> CausalDailyResult:
    """Assess DAG v2 and return a target, gated on its frozen validation.

    `allow_unvalidated_edge` downgrades the deployment gate from a refusal to
    a warning, so the model emits a target while the edge is still unproven.
    It exists for research and dry-run inspection. The validation result is
    still computed and still recorded on the result, so nothing is hidden —
    only the refusal is suspended.
    """
    if not (0.0 < base_weight <= 0.5):
        raise ValueError(f"base_weight must be in (0, 0.5], got {base_weight}")
    validate_model_signature()
    panel = _daily_frame(panel, "factor source").loc[MODEL_HISTORY_START:]
    returns = _daily_frame(returns, "return").loc[MODEL_HISTORY_START:]
    prices = _daily_frame(prices, "price").loc[MODEL_HISTORY_START:]
    factors = build_fixed_factor(panel)
    last_date = _validate_latest_window(
        panel,
        factors,
        prices,
        now=now,
        min_history_days=min_history_days,
        max_data_age_hours=max_data_age_hours,
        allow_stale=allow_stale,
    )
    signal = online_innovation_signal(factors)
    validation = validation_override or evaluate_forward_validation(signal, returns)
    if not validation.passed:
        lift = (
            "unavailable"
            if validation.sharpe_lift is None
            else f"{validation.sharpe_lift:+.4f}"
        )
        detail = (
            "frozen forward validation "
            f"status={validation.status}, observations={validation.n_available}, "
            f"Sharpe lift={lift}, required={VALIDATION_SHARPE_LIFT:+.2f}"
        )
        if not allow_unvalidated_edge:
            raise CausalModelNotDeployable(f"DAG v2 is not deployable: {detail}")
        logger.warning(
            "DAG v2 deployment gate OVERRIDDEN — emitting a target on an "
            "unproven edge: %s",
            detail,
        )
    if last_date not in signal.index:
        raise ValueError(f"no causal innovation signal for {last_date.date()}")
    signal_z = float(signal.loc[last_date])
    multiplier = float(exposure_from_signal(signal_z))
    weight = base_weight * multiplier
    if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
        raise ValueError(f"DAG v2 produced invalid BTC weight {weight}")
    prices = _daily_frame(prices, "price")
    price = (
        float(model_price_override)
        if model_price_override is not None
        else float(prices.loc[last_date, MODEL_ASSET])
    )
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"invalid BTC model price {price}")
    return CausalDailyResult(
        weights={MODEL_ASSET: weight},
        last_data_date=last_date,
        generated_at=now.astimezone(timezone.utc),
        target_path=Path(),
        signal_z=signal_z,
        exposure_multiplier=multiplier,
        base_weight=base_weight,
        validation=validation,
        model_prices={MODEL_ASSET: price},
    )


def write_target(
    result: CausalDailyResult,
    *,
    target_path: Path,
    expected_next_rebalance: datetime,
    metadata: dict[str, object],
) -> CausalDailyResult:
    payload: dict[str, object] = dict(result.weights)
    payload["_meta"] = {
        **metadata,
        "strategy": STRATEGY_NAME,
        "rebalance_date": result.last_data_date.date().isoformat(),
        "generated_utc": result.generated_at.isoformat(timespec="seconds"),
        "expected_next_rebalance": expected_next_rebalance.isoformat(),
        "format_version": 1,
        "gross_exposure": float(sum(abs(v) for v in result.weights.values())),
        "signal_z": result.signal_z,
        "exposure_multiplier": result.exposure_multiplier,
        "base_weight": result.base_weight,
        "forward_validation": result.validation.as_dict(),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return CausalDailyResult(
        weights=result.weights,
        last_data_date=result.last_data_date,
        generated_at=result.generated_at,
        target_path=target_path,
        signal_z=result.signal_z,
        exposure_multiplier=result.exposure_multiplier,
        base_weight=result.base_weight,
        validation=result.validation,
        model_prices=result.model_prices,
    )


def _required_snapshot_start(db_path: Path) -> str:
    """Return an overlap cursor based on the oldest required live series."""
    if not db_path.exists():
        return MODEL_HISTORY_START
    import duckdb

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'asset_metrics'"
        ).fetchone()[0]
        if not exists:
            return MODEL_HISTORY_START
        cursors = []
        for asset in FACTOR_ASSETS:
            row = connection.execute(
                "SELECT MAX(time) FROM asset_metrics "
                "WHERE provider = 'coinmetrics' AND asset = ? AND metric = ?",
                [asset, FACTOR_METRIC],
            ).fetchone()
            cursors.append(row[0] if row else None)
        price_row = connection.execute(
            "SELECT MAX(time) FROM asset_metrics "
            "WHERE asset = ? AND metric IN ('price', 'PriceUSD', 'price_usd')",
            [MODEL_ASSET],
        ).fetchone()
        cursors.append(price_row[0] if price_row else None)
        if any(cursor is None for cursor in cursors):
            return MODEL_HISTORY_START
        oldest = pd.Timestamp(min(cursors))
        if oldest.tzinfo is not None:
            oldest = oldest.tz_convert("UTC").tz_localize(None)
        oldest = oldest.normalize() - pd.Timedelta(days=SNAPSHOT_OVERLAP_DAYS)
        return max(oldest, pd.Timestamp(MODEL_HISTORY_START)).date().isoformat()
    finally:
        connection.close()


def refresh_local_data(
    *,
    now: datetime,
    db_path: str | Path | None = None,
) -> int:
    """Refresh the exact BTC/ETH/DOGE source set used by frozen DAG v2."""
    from causal_portfolio.data import DEFAULT_LOCAL_DB
    from causal_portfolio.data.coinmetrics_refresh import refresh_causal_sources

    target_db = Path(
        db_path or os.environ.get("CPCM_LOCAL_DB") or DEFAULT_LOCAL_DB
    ).expanduser().resolve()
    start = _required_snapshot_start(target_db)
    end = (now.astimezone(timezone.utc).date() - timedelta(days=1)).isoformat()
    return refresh_causal_sources(
        target_db,
        start=start,
        end=end,
    )


def _load_frames(end: str):
    from causal_portfolio.data import get_loader

    loader = get_loader()
    try:
        panel = loader.load_panel(
            list(FACTOR_ASSETS),
            [FACTOR_METRIC],
            MODEL_HISTORY_START,
            end,
            use_cache=False,
        )
        returns = loader.load_returns(
            [MODEL_ASSET], MODEL_HISTORY_START, end, use_cache=False
        )
        prices = loader.load_prices(
            [MODEL_ASSET], MODEL_HISTORY_START, end, use_cache=False
        )
    finally:
        close = getattr(loader, "close", None)
        if close is not None:
            close()
    return panel, returns, prices


def _configured_execution_address() -> str:
    """Load and return the wallet address used by the execution adapter."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    for name in ("HL_ADDRESS", "HYPERLIQUID_WALLET_ADDRESS"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ValueError(
        "RP-PCA retirement requires HL_ADDRESS or HYPERLIQUID_WALLET_ADDRESS"
    )


def _load_rppca_reference(path: Path) -> TargetSnapshot:
    """Load the last placeholder target as the retirement ownership record."""
    if not path.exists():
        raise FileNotFoundError(f"RP-PCA reference target not found: {path}")
    reference = load_target_snapshot(path)
    if reference.strategy != RP_PCA_STRATEGY_NAME:
        raise ValueError(
            "RP-PCA reference target has unexpected strategy: "
            f"{reference.strategy!r}"
        )
    if not reference.weights:
        raise ValueError("RP-PCA reference target has no owned asset universe")
    unknown = sorted(set(reference.weights) - set(DEFAULT_ASSET_MAP))
    if unknown:
        raise ValueError(f"RP-PCA reference target contains unmapped assets: {unknown}")
    if any(weight == 0 for weight in reference.weights.values()):
        raise ValueError("RP-PCA reference target contains a zero-side asset")
    return reference


def retire_rppca_positions(args: argparse.Namespace, *, now: datetime) -> Path:
    """Close placeholder positions once, through the audited execution path."""
    network = "mainnet" if args.mainnet else "testnet"
    expected_address = _configured_execution_address()
    reference_path = Path(args.rppca_reference_target)
    reference = _load_rppca_reference(reference_path)
    account_tag = hashlib.sha256(
        expected_address.casefold().encode("utf-8")
    ).hexdigest()[:12]
    marker = Path(
        args.retirement_marker
        or (
            "causal_portfolio/execution/state/"
            f"rppca_retired_{network}_{account_tag}.json"
        )
    )
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "retired"
            or payload.get("network") != network
            or str(payload.get("address", "")).casefold()
            != expected_address.casefold()
            or payload.get("reference_target_id") != reference.target_id
        ):
            raise RuntimeError(f"invalid RP-PCA retirement marker: {marker}")
        return marker
    if args.mainnet and not args.ack_mainnet:
        raise PermissionError("mainnet RP-PCA retirement requires --ack-mainnet")

    target = TargetSnapshot(
        weights={ticker: 0.0 for ticker in reference.weights},
        as_of=now,
        generated_at=now,
        strategy="RP-PCA placeholder retirement",
        source="causal_portfolio.models.causal_daily",
        metadata={
            "purpose": "one-time model cutover",
            "network": network,
            "reference_target": str(reference_path),
            "reference_target_id": reference.target_id,
        },
    )
    expected_sides = tuple(
        sorted(
            (
                DEFAULT_ASSET_MAP[ticker],
                1 if weight > 0 else -1,
            )
            for ticker, weight in reference.weights.items()
        )
    )
    config = ExecutionConfig(
        testnet=not args.mainnet,
        dry_run=False,
        long_only=True,
        max_position_pct=1.0,
        max_single_trade_pct=1.0,
        max_transaction_cost_bps=args.max_transaction_cost_bps,
        estimated_taker_fee_bps=args.estimated_taker_fee_bps,
        min_position_change_pct=0.0,
        min_rebalance_completeness=1.0,
        account_decommission=True,
        account_decommission_expected_address=expected_address,
        account_decommission_expected_sides=expected_sides,
    )
    result = execute_model_target(
        target,
        config,
        acknowledge_mainnet=args.mainnet,
    )
    if result.error:
        raise RuntimeError(f"RP-PCA retirement failed: {result.error}")
    if result.post_submit_error:
        raise RuntimeError(
            "RP-PCA retirement could not verify final state: "
            f"{result.post_submit_error}"
        )
    if result.audit_error:
        raise RuntimeError(f"RP-PCA retirement audit failed: {result.audit_error}")
    if result.submitted and result.post_state is None:
        raise RuntimeError("RP-PCA retirement submitted without a verified post-state")
    final_state = result.post_state or result.plan.current_state
    remaining = {
        coin: position.notional_usd
        for coin, position in final_state.positions.items()
        if abs(position.notional_usd) > 1e-6
    }
    if remaining:
        raise RuntimeError(f"RP-PCA retirement left open positions: {remaining}")

    marker_payload = {
        "status": "retired",
        "network": network,
        "retired_utc": now.isoformat(),
        "address": final_state.address,
        "submitted": result.submitted,
        "target_id": target.target_id,
        "reference_target": str(reference_path),
        "reference_target_id": reference.target_id,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(marker)
    logger.info("RP-PCA placeholder retirement verified: %s", marker)
    return marker


def _fetch_btc_reference_price(*, testnet: bool) -> float:
    """Read the executable BTC mid used for prospective mark-to-mark scoring."""
    from causal_portfolio.execution.hyperliquid import HLAdapter

    adapter = HLAdapter(
        ExecutionConfig(
            testnet=testnet,
            dry_run=True,
            network_timeout_seconds=15.0,
        )
    )
    price = float(adapter.fetch_mids().get("BTC", float("nan")))
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"invalid Hyperliquid BTC reference price {price}")
    return price


def run_once(args: argparse.Namespace) -> CausalDailyResult | None:
    now = datetime.now(timezone.utc)
    end = args.end or now.date().isoformat()
    if args.retire_rppca:
        retire_rppca_positions(args, now=now)
    if args.refresh_data:
        count = refresh_local_data(now=now)
        logger.info("refreshed %d DAG v2 warehouse rows", count)
    if pd.Timestamp(now.date()) < VALIDATION_DECISION_START:
        logger.info(
            "DAG v2 prospective registration is pending until %s; no target",
            VALIDATION_DECISION_START.date().isoformat(),
        )
        return None
    panel, returns, prices = _load_frames(end)
    normalized_panel = _daily_frame(panel, "factor source").loc[MODEL_HISTORY_START:]
    normalized_prices = _daily_frame(prices, "price").loc[MODEL_HISTORY_START:]
    factors = build_fixed_factor(normalized_panel)
    last_date = _validate_latest_window(
        normalized_panel,
        factors,
        normalized_prices,
        now=now,
        min_history_days=args.min_history_days,
        max_data_age_hours=args.max_signal_age_hours,
        allow_stale=args.allow_stale_signal,
    )
    signal = online_innovation_signal(factors)
    if last_date not in signal.index:
        raise ValueError(f"no causal innovation signal for {last_date.date()}")
    signal_z = float(signal.loc[last_date])
    reference_price = _fetch_btc_reference_price(testnet=not args.mainnet)
    record_signal_observation(
        args.signal_ledger,
        now=now,
        source_date=last_date,
        signal_z=signal_z,
        btc_reference_price=reference_price,
    )
    validation = evaluate_signal_ledger(load_signal_ledger(args.signal_ledger))
    if validation.status == "invalid_data":
        raise CausalModelNotDeployable(
            f"DAG v2 prospective ledger is invalid: {validation.reason}"
        )
    if not validation.passed:
        logger.info(
            "DAG v2 prospective validation %s: %d/%d outcomes; no target",
            validation.status,
            validation.n_available,
            VALIDATION_OBSERVATIONS,
        )
        from causal_portfolio.execution import notify

        notify.send(
            f"<b>CPCM DAG v2 Assessment</b> - {validation.status}\n"
            f"Prospective outcomes: <b>{validation.n_available}/"
            f"{VALIDATION_OBSERVATIONS}</b>\nNo target emitted.",
            parse_mode="HTML",
        )
        return None
    result = build_causal_target(
        panel,
        returns,
        prices,
        now=now,
        base_weight=args.base_weight,
        min_history_days=args.min_history_days,
        max_data_age_hours=args.max_signal_age_hours,
        allow_stale=args.allow_stale_signal,
        validation_override=validation,
        model_price_override=reference_price,
        allow_unvalidated_edge=args.allow_unvalidated_edge,
    )
    target_path = Path(args.target_out)
    expected_next = now + timedelta(hours=args.every_hours)
    result = write_target(
        result,
        target_path=target_path,
        expected_next_rebalance=expected_next,
        metadata={
            "model_signature": validate_model_signature(),
            "factor_source_columns": list(FACTOR_COLUMNS),
            "rule": "btc_weight = base_weight * clip(1 + 0.5 * signal_z, 0, 2)",
            "source": str(
                os.environ.get("CPCM_LOCAL_DB", "causal_portfolio.data.get_loader")
            ),
        },
    )
    logger.info(
        "wrote DAG v2 target to %s (signal=%+.4f multiplier=%.4f data=%s)",
        target_path,
        result.signal_z,
        result.exposure_multiplier,
        result.last_data_date.date().isoformat(),
    )

    target = load_target_snapshot(target_path)
    try:
        from causal_portfolio.execution import trace

        trace.start_cycle(
            target,
            expected_next_rebalance=expected_next,
            model_prices=result.model_prices,
            price_source="causal_portfolio.data.get_loader",
        )
    except Exception:
        logger.exception("DAG v2 trace initialization failed (continuing)")

    from causal_portfolio.execution import notify

    notify.send(
        f"<b>CPCM DAG v2 Target</b> - data through "
        f"{result.last_data_date.date().isoformat()}\n"
        f"Signal z: <b>{result.signal_z:+.4f}</b> | "
        f"BTC weight: <b>{result.weights[MODEL_ASSET]:.4f}</b>\n"
        f"Frozen holdout: <b>passed</b> "
        f"(Sharpe lift {result.validation.sharpe_lift:+.3f})\n"
        f"{notify.format_next_rebalance(expected_next)}",
        parse_mode="HTML",
    )
    if args.execute:
        _submit_target(args, target_path)
    return result


def _submit_target(args: argparse.Namespace, target_path: Path) -> None:
    if args.mainnet and not args.ack_mainnet:
        raise PermissionError("mainnet execution requires --ack-mainnet")
    config = ExecutionConfig(
        testnet=not args.mainnet,
        dry_run=False,
        long_only=True,
        max_signal_age_hours=args.max_signal_age_hours,
        allow_stale_signal=args.allow_stale_signal,
        twap_minutes=args.twap_minutes,
        twap_slices=args.twap_slices,
        max_transaction_cost_bps=args.max_transaction_cost_bps,
        estimated_taker_fee_bps=args.estimated_taker_fee_bps,
        min_position_change_pct=args.min_position_change_pct,
        min_rebalance_completeness=args.min_rebalance_completeness,
    )
    result = execute_model_target(
        load_target_snapshot(target_path),
        config,
        acknowledge_mainnet=args.mainnet,
    )
    if result.error:
        raise RuntimeError(result.error)
    if result.post_submit_error:
        logger.warning(
            "submission completed but post-submit checks failed: %s",
            result.post_submit_error,
        )
    if result.audit_error:
        logger.error("submission completed but audit append failed: %s", result.audit_error)
    if not result.submitted:
        logger.info("DAG v2 rebalance no-op: no orders submitted")
        return
    logger.info("submitted DAG v2 rebalance: %s", result.response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess frozen CPCM DAG v2 and optionally execute its BTC target"
    )
    parser.add_argument("--end", default=None, help="Defaults to today's UTC date")
    parser.add_argument("--target-out", default="tmp/cpcm_causal_daily_target.json")
    parser.add_argument("--signal-ledger", default=str(DEFAULT_SIGNAL_LEDGER))
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--retire-rppca",
        action="store_true",
        help="Close and audit placeholder positions once before causal assessment",
    )
    parser.add_argument(
        "--retire-rppca-only",
        action="store_true",
        help="Perform only the one-time audited placeholder retirement",
    )
    parser.add_argument(
        "--retirement-marker",
        default=None,
        help="Override the one-time RP-PCA retirement marker path",
    )
    parser.add_argument(
        "--rppca-reference-target",
        default=str(DEFAULT_RP_PCA_REFERENCE_TARGET),
        help="Last audited RP-PCA target used to scope placeholder retirement",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mainnet", action="store_true")
    parser.add_argument("--ack-mainnet", action="store_true")
    parser.add_argument("--allow-stale-signal", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-edge",
        action="store_true",
        help=(
            "Emit a target even when the frozen forward validation has not "
            "passed. The gate is reported as a warning instead of refusing. "
            "Research/dry-run use: the edge is unproven by definition."
        ),
    )
    parser.add_argument("--max-signal-age-hours", type=float, default=72.0)
    parser.add_argument("--twap-minutes", type=float, default=0.0)
    parser.add_argument("--twap-slices", type=int, default=5)
    parser.add_argument("--max-transaction-cost-bps", type=float, default=15.0)
    parser.add_argument("--estimated-taker-fee-bps", type=float, default=4.5)
    parser.add_argument("--min-position-change-pct", type=float, default=0.0)
    parser.add_argument("--min-rebalance-completeness", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--every-hours", type=float, default=24.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.retire_rppca_only:
        if args.loop or args.retire_rppca:
            logger.error("--retire-rppca-only cannot be combined with loop/combined retirement")
            return 2
        try:
            with execution_run_log("rppca-retirement") as run_log:
                logger.info("starting one-time RP-PCA retirement: log=%s", run_log)
                retire_rppca_positions(args, now=datetime.now(timezone.utc))
        except Exception:
            logger.exception("RP-PCA retirement failed")
            return 1
        return 0
    while True:
        try:
            with execution_run_log("cpcm-causal") as run_log:
                logger.info(
                    "starting CPCM DAG v2 assessment: log=%s network=%s "
                    "execute=%s base_weight=%.4f target_out=%s",
                    run_log,
                    "mainnet" if args.mainnet else "testnet",
                    args.execute,
                    args.base_weight,
                    args.target_out,
                )
                run_once(args)
        except Exception:
            logger.exception("CPCM DAG v2 daily run failed")
            return 1
        if not args.loop:
            return 0
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
