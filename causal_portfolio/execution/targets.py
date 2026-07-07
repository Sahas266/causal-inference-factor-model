"""Load model target weights with provenance intact.

The executor accepts both the repository's JSON handoff and the dated
asset-weight CSV emitted by the reference RP-PCA project. CSV inputs may
contain a full rebalance history; only the latest dated row is executable.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal_portfolio.execution.config import DEFAULT_ASSET_MAP
from causal_portfolio.execution.types import TargetSnapshot


_TIMESTAMP_COLUMNS = ("rebalance_date", "as_of")
_GENERATED_COLUMNS = ("computed_utc", "generated_utc", "generated_at")
_NON_ASSET_COLUMNS = set(_TIMESTAMP_COLUMNS + _GENERATED_COLUMNS + ("strategy",))
_KNOWN_TARGET_ASSETS = set(DEFAULT_ASSET_MAP)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid target timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_timestamp(mapping: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return _parse_timestamp(mapping[key])
    return None


def _csv_asset_name(column: str) -> str | None:
    """Return the target ticker represented by a CSV column, or None.

    Dated CSV execution targets are intentionally strict: every unprefixed
    column must be a known executable asset, a known metadata column, or an
    underscore-prefixed metadata field. This prevents accidental numeric
    analytics columns such as `turnover` from becoming phantom target assets.
    """
    normalized = column.strip().lower()
    if normalized in _NON_ASSET_COLUMNS or normalized.startswith("_"):
        return None
    if normalized.startswith("weight_"):
        normalized = normalized.removeprefix("weight_")
    if normalized in _KNOWN_TARGET_ASSETS:
        return normalized
    raise ValueError(
        f"Unknown target CSV column {column!r}. Prefix metadata columns with '_' "
        "or add the asset to DEFAULT_ASSET_MAP before execution."
    )


def _load_json(path: Path) -> TargetSnapshot:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Weights file must be a JSON object, got {type(raw).__name__}")

    if "weights" in raw:
        raw_weights = raw["weights"]
        if not isinstance(raw_weights, dict):
            raise ValueError("The JSON 'weights' field must be an object")
        metadata = {k: v for k, v in raw.items() if k != "weights"}
        nested_meta = metadata.pop("_meta", {})
        if nested_meta and not isinstance(nested_meta, dict):
            raise ValueError("The JSON '_meta' field must be an object")
        metadata = {**nested_meta, **metadata}
    else:
        nested_meta = raw.get("_meta", {})
        if nested_meta and not isinstance(nested_meta, dict):
            raise ValueError("The JSON '_meta' field must be an object")
        metadata = dict(nested_meta)
        raw_weights = {k: v for k, v in raw.items() if not k.startswith("_")}

    # TargetSnapshot.__post_init__ normalizes tickers and validates weights.
    return TargetSnapshot(
        weights=raw_weights,
        as_of=_first_timestamp(metadata, _TIMESTAMP_COLUMNS),
        generated_at=_first_timestamp(metadata, _GENERATED_COLUMNS),
        strategy=str(metadata["strategy"]) if metadata.get("strategy") else None,
        source=str(path),
        metadata=metadata,
    )


def _load_csv(path: Path) -> TargetSnapshot:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError("Target CSV has no data rows")

    timestamp_column = next((c for c in _TIMESTAMP_COLUMNS if c in fieldnames), None)
    if timestamp_column is None:
        raise ValueError("Target CSV must contain 'rebalance_date' or 'as_of'")

    dated_rows: list[tuple[datetime, dict[str, str]]] = []
    for row in rows:
        timestamp = _parse_timestamp(row.get(timestamp_column))
        if timestamp is None:
            raise ValueError(f"Target CSV row is missing {timestamp_column!r}")
        dated_rows.append((timestamp, row))
    as_of, latest = max(dated_rows, key=lambda item: item[0])

    raw_weights: dict[str, float] = {}
    for column in fieldnames:
        asset = _csv_asset_name(column)
        if asset is None:
            continue
        value = latest.get(column)
        if value is None or value.strip() == "":
            raise ValueError(f"Latest target row has a blank weight for {column!r}")
        try:
            raw_weights[asset] = float(value)
        except ValueError as exc:
            raise ValueError(f"Weight for {column!r} must be numeric, got {value!r}") from exc

    metadata: dict[str, Any] = {
        "format": "rebalance_history_csv",
        "rows": len(rows),
        timestamp_column: latest[timestamp_column],
    }
    for column in fieldnames:
        if column.startswith("_") or column in _GENERATED_COLUMNS or column == "strategy":
            value = latest.get(column)
            if value not in (None, ""):
                metadata[column] = value

    return TargetSnapshot(
        weights=raw_weights,
        as_of=as_of,
        generated_at=_first_timestamp(latest, _GENERATED_COLUMNS),
        strategy=latest.get("strategy") or latest.get("_strategy") or None,
        source=str(path),
        metadata=metadata,
    )


def load_target_snapshot(path: str | Path) -> TargetSnapshot:
    """Load a JSON snapshot or the latest row of a rebalance-history CSV."""
    target_path = Path(path)
    suffix = target_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(target_path)
    if suffix == ".json":
        return _load_json(target_path)
    raise ValueError(f"Unsupported target file type {suffix!r}; use .json or .csv")
