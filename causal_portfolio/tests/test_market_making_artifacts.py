from __future__ import annotations

from datetime import datetime, timedelta, timezone

from causal_portfolio.market_making.artifacts import (
    CalibrationArtifact,
    load_calibration_artifact,
)
from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.pin import PINFit, PINPosterior


def _fit():
    return PINFit(
        alpha=0.3,
        delta=0.5,
        mu=40,
        epsilon=20,
        pin=0.230769,
        log_likelihood=-100,
        sample_days=60,
        converged=True,
        boundary_solution=False,
        hessian_condition=100,
        standard_errors=(0.01, 0.02, 1.0, 0.5),
        starts=8,
        message="ok",
    )


def test_artifact_round_trip_and_freshness(tmp_path):
    now = datetime.now(timezone.utc)
    artifact = CalibrationArtifact(
        coin="BTC",
        as_of=now,
        pin_fit=_fit(),
        posterior=PINPosterior(0.5, 0.3, 0.2),
        markouts=MarkoutCurve((0, 1), (5,), (4,), (10,)),
    )
    path = tmp_path / "calibration.json"
    artifact.save(path)
    loaded = load_calibration_artifact(path)
    assert loaded == artifact
    assert loaded.freshness_error(1, now=now + timedelta(minutes=30)) is None
    assert "old" in loaded.freshness_error(1, now=now + timedelta(hours=2))


def test_pin_fit_usable_requires_diagnostics():
    fit = _fit()
    assert fit.usable
    missing_hessian = PINFit(**{**fit.__dict__, "hessian_condition": None})
    assert not missing_hessian.usable
