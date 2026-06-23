"""Versioned, freshness-checked calibration artifacts for shadow/live use."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal_portfolio.market_making.calibration import IntensityEstimate, MarkoutCurve
from causal_portfolio.market_making.pin import PINFit, PINPosterior


@dataclass(frozen=True)
class CalibrationArtifact:
    coin: str
    as_of: datetime
    pin_fit: PINFit | None = None
    posterior: PINPosterior | None = None
    markouts: MarkoutCurve | None = None
    buy_intensity: IntensityEstimate | None = None
    sell_intensity: IntensityEstimate | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        if not self.coin:
            raise ValueError("coin cannot be empty")
        value = self.as_of
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "as_of", value.astimezone(timezone.utc))
        if self.format_version != 1:
            raise ValueError(f"unsupported calibration format: {self.format_version}")

    def freshness_error(
        self, max_age_hours: float, *, now: datetime | None = None
    ) -> str | None:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        age = (current.astimezone(timezone.utc) - self.as_of).total_seconds() / 3600
        if age < -5 / 60:
            return f"calibration is {abs(age):.1f}h in the future"
        if age > max_age_hours:
            return f"calibration is {age:.1f}h old (limit {max_age_hours:.1f}h)"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "coin": self.coin,
            "as_of": self.as_of.isoformat(),
            "pin_fit": asdict(self.pin_fit) if self.pin_fit else None,
            "posterior": asdict(self.posterior) if self.posterior else None,
            "markouts": asdict(self.markouts) if self.markouts else None,
            "buy_intensity": asdict(self.buy_intensity) if self.buy_intensity else None,
            "sell_intensity": asdict(self.sell_intensity) if self.sell_intensity else None,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)


def _optional(payload: dict[str, Any], name: str, cls):
    value = payload.get(name)
    if value is None:
        return None
    if cls is MarkoutCurve:
        value = {
            **value,
            "edges": tuple(value["edges"]),
            "buy_bps": tuple(value["buy_bps"]),
            "sell_bps": tuple(value["sell_bps"]),
            "counts": tuple(value["counts"]),
        }
    if cls is PINFit and value.get("standard_errors") is not None:
        value = {**value, "standard_errors": tuple(value["standard_errors"])}
    return cls(**value)


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationArtifact(
        coin=str(payload["coin"]),
        as_of=datetime.fromisoformat(str(payload["as_of"]).replace("Z", "+00:00")),
        pin_fit=_optional(payload, "pin_fit", PINFit),
        posterior=_optional(payload, "posterior", PINPosterior),
        markouts=_optional(payload, "markouts", MarkoutCurve),
        buy_intensity=_optional(payload, "buy_intensity", IntensityEstimate),
        sell_intensity=_optional(payload, "sell_intensity", IntensityEstimate),
        format_version=int(payload.get("format_version", 1)),
    )
