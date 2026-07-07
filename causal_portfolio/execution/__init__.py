"""Hyperliquid execution layer for CPCM-generated portfolio weights.

Translates a target weight vector w into a list of orders, with safety rails
(dust thresholds, position caps, slippage limits, dry-run by default).

Key separation:
  - rebalancer.py: PURE function (no network) — fully unit-testable
  - hyperliquid.py: thin SDK adapter — touches network, lazy-imports the SDK

Public entry points:
  - plan_rebalance: weights + state → RebalancePlan (no orders submitted)
  - execute_rebalance: plan + adapter → SubmitResult (orders submitted)
"""

from causal_portfolio.execution.config import ExecutionConfig
from causal_portfolio.execution.rebalancer import plan_rebalance
from causal_portfolio.execution.reconcile import reconcile
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
    "load_target_snapshot",
    "plan_rebalance",
    "reconcile",
]
