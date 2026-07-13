"""Append-only audit log for every rebalance.

Records inputs, plan, response, post-state in JSON-Lines format. One file per
day, keyed by UTC date. Never compressed or rotated automatically — debugging
a fill from three weeks ago requires the raw record.

File path: <repo>/causal_portfolio/execution/logs/rebalance-YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from causal_portfolio.execution.types import RebalancePlan, SubmitResult

logger = logging.getLogger("cpcm.execution.audit")

LOG_DIR = Path(__file__).parent / "logs"


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-friendly conversion for our dataclasses + enums."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def append(result: SubmitResult, log_dir: Path | None = None) -> Path:
    """Append a SubmitResult to today's audit log. Returns the log path."""
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = log_dir / f"rebalance-{today}.jsonl"

    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "network": result.plan.network,
        "target_id": (
            result.plan.target_snapshot.target_id
            if result.plan.target_snapshot is not None
            else None
        ),
        "submitted": result.submitted,
        "error": result.error,
        "post_submit_error": result.post_submit_error,
        "plan": _serialize(result.plan),
        "response": result.response,
        "post_state": _serialize(result.post_state) if result.post_state else None,
        "drifts": _serialize(result.drifts),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.info("audit: appended to %s", path)
    return path


def read_log(date: str, log_dir: Path | None = None) -> list[dict]:
    """Read all records for a given UTC date (YYYY-MM-DD).

    Malformed lines (e.g. a truncated write from a crash) are skipped with a
    warning rather than poisoning the whole day's log.
    """
    log_dir = log_dir or LOG_DIR
    path = log_dir / f"rebalance-{date}.jsonl"
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("audit: skipping malformed line %d in %s: %s", lineno, path, e)
    return records
