from __future__ import annotations

from datetime import datetime, timezone

import pytest

from causal_portfolio.execution.rppca_daily import (
    build_parser,
    forward_fill_prices,
    run_once,
)
from causal_portfolio.execution.targets import load_target_snapshot


def test_rppca_daily_generates_loadable_forward_filled_target(tmp_path):
    prices = tmp_path / "prices.csv"
    target = tmp_path / "target.json"
    rows = ["date,btc_price,eth_price,sol_price"]
    for i in range(90):
        rows.append(f"2026-01-{(i % 30) + 1:02d},{100+i},{80+i*0.5},{30+i*0.2}")
    prices.write_text("\n".join(rows), encoding="utf-8")

    args = build_parser().parse_args([
        "--assets", "btc,eth,sol",
        "--prices-csv", str(prices),
        "--start", "2026-01-01",
        "--end", "2026-02-02",
        "--max-ffill-days", "31",
        "--cov-window", "30",
        "--mean-window", "14",
        "--n-components", "2",
        "--target-out", str(target),
    ])

    result = run_once(args)
    loaded = load_target_snapshot(result.target_path)

    assert target.exists()
    assert loaded.strategy == "RP-PCA daily tangency"
    assert loaded.as_of == datetime(2026, 1, 30, tzinfo=timezone.utc)
    assert set(loaded.weights) <= {"btc", "eth", "sol"}


def test_forward_fill_prices_enforces_limit(tmp_path):
    import pandas as pd

    prices = pd.DataFrame(
        {"btc": [100.0], "eth": [50.0]},
        index=pd.DatetimeIndex(["2026-01-01"]),
    )

    with pytest.raises(ValueError, match="increase --max-ffill-days"):
        forward_fill_prices(prices, end="2026-01-10", max_ffill_days=2)


def test_default_end_resolves_at_run_time():
    args = build_parser().parse_args([])

    assert args.end is None
