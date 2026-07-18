"""Hyperliquid execution layer for CPCM-generated portfolio weights.

Translates a target weight vector w into a list of orders, with safety rails
(dust thresholds, position caps, slippage limits, dry-run by default).

Key separation:
  - rebalancer.py: PURE function (no network) — fully unit-testable
  - hyperliquid.py: thin SDK adapter — touches network, lazy-imports the SDK

Public entry points:
  - plan_rebalance: weights + state → RebalancePlan (no orders submitted)
  - execute_target(target, config): TargetSnapshot → SubmitResult
"""

# Library version (semver). Single source of truth — pyproject.toml reads it.
__version__ = "0.1.0"

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.hyperliquid import execute_target
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.reconcile import reconcile
from causal_portfolio.execution.run_logging import execution_run_log
from causal_portfolio.execution.targets import load_target_snapshot
from causal_portfolio.execution.types import (
    AccountState,
    AssetMeta,
    Order,
    Position,
    RebalancePlan,
    ReconcileDrift,
    SkipReason,
    SubmitResult,
    TargetSnapshot,
)

__all__ = [
    "AccountState",
    "AssetMeta",
    "ExecutionConfig",
    "Order",
    "Position",
    "RebalancePlan",
    "ReconcileDrift",
    "SkipReason",
    "SubmitResult",
    "TargetSnapshot",
    "execute_target",
    "load_target_snapshot",
    "execution_run_log",
    "plan_rebalance",
    "reconcile",
]
