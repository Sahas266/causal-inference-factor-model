"""Post-execution reconciliation.

Pure function: given a RebalancePlan and the actual AccountState observed
after order submission, compute the drift between target and reality for
each coin we tried to trade. Drift can come from:
  - Partial fills (IOC orders don't always fill in full)
  - Slippage past the limit price
  - Cancellations between cancel and submit (race window)
  - Mid prices moving during submission

The output list is empty when every coin landed inside tolerance.
"""

from __future__ import annotations

from causal_portfolio.execution.types import (
    AccountState,
    RebalancePlan,
    ReconcileDrift,
)


def reconcile(
    plan: RebalancePlan,
    post_state: AccountState,
    tolerance_pct: float = 0.05,
    tolerance_usd: float = 50.0,
) -> list[ReconcileDrift]:
    """Compare plan.target_usd to actual post-execution notionals.

    A coin is reported as drifted when |actual - target| exceeds BOTH:
      - tolerance_usd (absolute floor — ignore $1 drifts on a $100k account)
      - tolerance_pct of |target| (relative floor — 5% drift on a $10k target
        is bigger than on a $100 target)

    Coins not in plan.target_usd are ignored (we only judge what we planned).
    Coins with target=0 that have residual positions are always reported
    (we wanted them closed; anything left is drift).

    Returns:
        List of ReconcileDrift, empty if everything reconciled.
    """
    drifts: list[ReconcileDrift] = []
    actual_by_coin = {c: p.notional_usd for c, p in post_state.positions.items()}

    for coin, target_usd in plan.target_usd.items():
        actual_usd = actual_by_coin.get(coin, 0.0)
        drift_usd = actual_usd - target_usd

        # Skip coins where both target and actual are tiny (sub-tolerance).
        if abs(target_usd) < tolerance_usd and abs(actual_usd) < tolerance_usd:
            continue

        if abs(target_usd) > 0:
            drift_pct = drift_usd / abs(target_usd)
        else:
            # Plan wanted this closed (target=0) but something's still there.
            drift_pct = float("inf") if abs(actual_usd) > tolerance_usd else 0.0

        # Material drift: BOTH thresholds breached.
        breaches_abs = abs(drift_usd) > tolerance_usd
        breaches_pct = abs(drift_pct) > tolerance_pct
        if breaches_abs and breaches_pct:
            drifts.append(ReconcileDrift(
                coin=coin, target_usd=target_usd, actual_usd=actual_usd,
                drift_usd=drift_usd, drift_pct=drift_pct,
            ))

    return drifts


def format_drift_summary(drifts: list[ReconcileDrift]) -> str:
    """Human-readable one-liner per drift, joined by newlines."""
    if not drifts:
        return "Reconciliation OK: all targets within tolerance."
    lines = [f"Reconciliation: {len(drifts)} coin(s) drifted from target:"]
    for d in drifts:
        pct = "inf" if d.drift_pct == float("inf") else f"{d.drift_pct:+.1%}"
        lines.append(
            f"  {d.coin:<8} target=${d.target_usd:+,.2f} "
            f"actual=${d.actual_usd:+,.2f} drift=${d.drift_usd:+,.2f} ({pct})"
        )
    return "\n".join(lines)
