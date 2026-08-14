"""Windows console compatibility for research command help."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "causal_portfolio.experiments.cross_asset_rotation",
        "causal_portfolio.experiments.trend_breakout",
    ],
)
def test_help_is_cp1252_safe(module):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
