"""Execution A/B harness for the BTC/stables dual-confirm trend strategy.

Historical mode uses the repository's BTC close series for the strategy signal
and public Hyperliquid daily candles for execution-day open/high/low/close.
Because historical trade sides and L2 queue state are unavailable, PIN is a
candle-classified proxy and passive fills are reported as optimistic
(high/low touch) and conservative (close cross) bounds.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_portfolio.backtest.metrics import max_drawdown, sharpe_ratio
from causal_portfolio.market_making.avellaneda_stoikov import (
    ASModelParams,
    finite_horizon_quotes,
)
from causal_portfolio.market_making.calibration import MarkoutCurve
from causal_portfolio.market_making.pin import PINFit, fit_pin, pin_posteriors
from causal_portfolio.market_making.toxicity import (
    ToxicityConfig,
    ToxicitySignal,
    toxicity_adjustment,
)
from causal_portfolio.market_making.types import QuotePolicy


@dataclass(frozen=True)
class DailyCandle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int

    @property
    def day(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp_ms, unit="ms", tz="UTC").normalize()


@dataclass(frozen=True)
class SwitchExecution:
    signal_day: str
    execution_day: str
    is_buy: bool
    reference_open: float
    naive_cost_bps: float
    as_pin_quote: float
    as_pin_touch_cost_bps: float
    as_pin_close_cost_bps: float
    touch_filled: bool
    close_filled: bool
    pin_proxy: float
    pin_fit_usable: bool
    informed_buy_probability: float
    informed_sell_probability: float
    toxicity_premium_bps: float


@dataclass(frozen=True)
class ArmMetrics:
    mean_cost_bps: float
    median_cost_bps: float
    total_cost_bps: float
    passive_fill_rate: float | None
    annualized_return: float
    sharpe: float
    max_drawdown: float


@dataclass(frozen=True)
class HistoricalExecutionBenchmark:
    strategy: str
    start: str
    end: str
    switches: tuple[SwitchExecution, ...]
    naive: ArmMetrics
    as_pin_touch: ArmMetrics
    as_pin_close: ArmMetrics
    pin_usable_fraction: float
    methodology_quality: str = "proxy_pin_and_daily_fill_bounds"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dual_confirm_signal(prices: pd.Series) -> pd.Series:
    """Exact preferred BTC rule: close > MA100 and trailing 30d return > 0."""
    clean = prices.astype(float).sort_index()
    moving_average = clean.rolling(100, min_periods=100).mean()
    trailing_return = clean.pct_change(30)
    return ((clean > moving_average) & (trailing_return > 0)).astype(float)


def load_local_btc_prices(
    database: str | Path = "causal_portfolio/data/cpcm_local.duckdb",
    *,
    start: str = "2021-01-01",
    end: str = "2025-12-31",
) -> pd.Series:
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT time, value FROM asset_metrics_best
            WHERE lower(asset) = 'btc' AND metric = 'price'
              AND time >= ? AND time < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY
            ORDER BY time
            """,
            [start, end],
        ).df()
    finally:
        connection.close()
    if rows.empty:
        raise ValueError("local warehouse has no BTC price data in the requested range")
    index = pd.to_datetime(rows["time"], utc=True).dt.normalize()
    return pd.Series(rows["value"].astype(float).values, index=index, name="btc_price")


def fetch_hyperliquid_daily_candles(start: str, end: str) -> list[DailyCandle]:
    """Read public mainnet candles; no wallet or private key is used."""
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    raw = info.candles_snapshot(
        "BTC", "1d", int(start_dt.timestamp() * 1_000), int(end_dt.timestamp() * 1_000)
    )
    return [
        DailyCandle(
            timestamp_ms=int(item["t"]),
            open=float(item["o"]),
            high=float(item["h"]),
            low=float(item["l"]),
            close=float(item["c"]),
            volume=float(item["v"]),
            trades=int(item["n"]),
        )
        for item in raw
    ]


def candle_classified_counts(candles: list[DailyCandle]) -> tuple[np.ndarray, np.ndarray]:
    """Bulk-volume-style count classification used only as a PIN proxy."""
    closes = np.array([c.close for c in candles], dtype=float)
    opens = np.array([c.open for c in candles], dtype=float)
    changes = closes - opens
    scale = pd.Series(np.diff(closes, prepend=closes[0])).rolling(
        30, min_periods=10
    ).std()
    fallback = float(np.nanmedian(scale.values))
    if not math.isfinite(fallback) or fallback <= 0:
        fallback = max(float(np.std(changes)), 1.0)
    sigma = np.where(np.isfinite(scale.values) & (scale.values > 0), scale.values, fallback)
    buy_fraction = norm.cdf(changes / sigma)
    trades = np.array([max(c.trades, 1) for c in candles], dtype=float)
    buys = np.rint(trades * buy_fraction).astype(int)
    sells = np.maximum(trades.astype(int) - buys, 0)
    return buys, sells


def _proxy_pin(
    history: list[DailyCandle], *, window: int, starts: int
) -> tuple[PINFit, float, float, MarkoutCurve]:
    sample = history[-window:]
    buys, sells = candle_classified_counts(sample)
    fit = fit_pin(buys, sells, min_days=window, n_starts=starts, seed=17)
    posterior = pin_posteriors([int(buys[-1])], [int(sells[-1])], fit)[0]
    returns = np.array([(c.close - c.open) / c.open for c in sample])
    buy_weights = buys / np.maximum(buys + sells, 1)
    sell_weights = 1.0 - buy_weights
    buy_adverse = float(
        np.sum(buy_weights * np.maximum(returns, 0) * 10_000)
        / max(np.sum(buy_weights), 1e-12)
    )
    sell_adverse = float(
        np.sum(sell_weights * np.maximum(-returns, 0) * 10_000)
        / max(np.sum(sell_weights), 1e-12)
    )
    curve = MarkoutCurve((0.0, 1.0), (buy_adverse,), (sell_adverse,), (window,))
    return fit, posterior.informed_buy, posterior.informed_sell, curve


def _all_in_cost_bps(
    *, is_buy: bool, reference: float, fill_price: float, fee_bps: float
) -> float:
    effective = fill_price * (1 + fee_bps / 10_000) if is_buy else fill_price * (
        1 - fee_bps / 10_000
    )
    direction = 1.0 if is_buy else -1.0
    return direction * (effective - reference) / reference * 10_000


def _strategy_metrics(
    prices: pd.Series,
    exposure: pd.Series,
    switch_costs: dict[pd.Timestamp, float],
    passive_fill_rate: float | None,
) -> ArmMetrics:
    returns = prices.pct_change().fillna(0.0)
    held = exposure.shift(1).fillna(0.0)
    daily = held * returns
    for day, cost_bps in switch_costs.items():
        if day in daily.index:
            daily.loc[day] -= cost_bps / 10_000
    costs = list(switch_costs.values())
    return ArmMetrics(
        mean_cost_bps=float(np.mean(costs)),
        median_cost_bps=float(np.median(costs)),
        total_cost_bps=float(np.sum(costs)),
        passive_fill_rate=passive_fill_rate,
        annualized_return=float(daily.mean() * 365),
        sharpe=float(sharpe_ratio(daily.values)),
        max_drawdown=float(max_drawdown(daily.values)),
    )


def run_historical_execution_benchmark(
    prices: pd.Series,
    candles: list[DailyCandle],
    *,
    pin_window: int = 60,
    pin_starts: int = 6,
    taker_fee_bps: float = 4.5,
    maker_fee_bps: float = 1.5,
    gamma: float = 1e-6,
    baseline_spread_bps: float = 2.0,
) -> HistoricalExecutionBenchmark:
    candle_by_day = {candle.day: candle for candle in candles}
    ordered_days = sorted(candle_by_day)
    exposure = dual_confirm_signal(prices)
    changes = exposure[exposure.diff().abs() > 0]
    executions: list[SwitchExecution] = []
    naive_costs: dict[pd.Timestamp, float] = {}
    touch_costs: dict[pd.Timestamp, float] = {}
    close_costs: dict[pd.Timestamp, float] = {}
    for signal_day, target in changes.items():
        execution_day = signal_day + pd.Timedelta(days=1)
        candle = candle_by_day.get(execution_day)
        if candle is None:
            continue
        prior = [candle_by_day[day] for day in ordered_days if day < execution_day]
        if len(prior) < pin_window:
            continue
        fit, informed_buy, informed_sell, curve = _proxy_pin(
            prior, window=pin_window, starts=pin_starts
        )
        signal = ToxicitySignal(fit.pin, informed_buy, informed_sell, True)
        adjustment = toxicity_adjustment(
            signal,
            ToxicityConfig(policy=QuotePolicy.AS_PIN_POSTERIOR),
            markouts=curve,
        )
        close_history = np.array([item.close for item in prior[-30:]], dtype=float)
        sigma = max(float(np.std(np.diff(close_history), ddof=1)), candle.open * 1e-5)
        target_spread = candle.open * baseline_spread_bps / 10_000
        kappa = 2.0 / max(target_spread, 1e-9)
        params = ASModelParams(gamma, sigma, kappa, 1.0)
        is_buy = bool(target > 0.5)
        inventory_deviation = -1.0 if is_buy else 1.0
        levels = finite_horizon_quotes(candle.open, inventory_deviation, params)
        if is_buy:
            premium = adjustment.bid_extra_bps
            quote = math.floor(levels.bid - candle.open * premium / 10_000)
            touch_filled = candle.low <= quote
            close_filled = candle.close <= quote
        else:
            premium = adjustment.ask_extra_bps
            quote = math.ceil(levels.ask + candle.open * premium / 10_000)
            touch_filled = candle.high >= quote
            close_filled = candle.close >= quote
        naive_cost = _all_in_cost_bps(
            is_buy=is_buy,
            reference=candle.open,
            fill_price=candle.open,
            fee_bps=taker_fee_bps,
        )
        touch_cost = _all_in_cost_bps(
            is_buy=is_buy,
            reference=candle.open,
            fill_price=quote if touch_filled else candle.close,
            fee_bps=maker_fee_bps if touch_filled else taker_fee_bps,
        )
        close_cost = _all_in_cost_bps(
            is_buy=is_buy,
            reference=candle.open,
            fill_price=quote if close_filled else candle.close,
            fee_bps=maker_fee_bps if close_filled else taker_fee_bps,
        )
        executions.append(
            SwitchExecution(
                signal_day=signal_day.date().isoformat(),
                execution_day=execution_day.date().isoformat(),
                is_buy=is_buy,
                reference_open=candle.open,
                naive_cost_bps=naive_cost,
                as_pin_quote=quote,
                as_pin_touch_cost_bps=touch_cost,
                as_pin_close_cost_bps=close_cost,
                touch_filled=touch_filled,
                close_filled=close_filled,
                pin_proxy=fit.pin,
                pin_fit_usable=fit.usable,
                informed_buy_probability=informed_buy,
                informed_sell_probability=informed_sell,
                toxicity_premium_bps=premium,
            )
        )
        naive_costs[execution_day] = naive_cost
        touch_costs[execution_day] = touch_cost
        close_costs[execution_day] = close_cost
    if not executions:
        raise ValueError("no strategy switches overlap usable execution candles")
    evaluation_start = pd.Timestamp(executions[0].execution_day, tz="UTC")
    evaluation_end = pd.Timestamp(executions[-1].execution_day, tz="UTC")
    eval_prices = prices.loc[evaluation_start:evaluation_end]
    eval_exposure = exposure.reindex(eval_prices.index).fillna(0.0)
    touch_fill_rate = float(np.mean([item.touch_filled for item in executions]))
    close_fill_rate = float(np.mean([item.close_filled for item in executions]))
    usable_fraction = float(np.mean([item.pin_fit_usable for item in executions]))
    return HistoricalExecutionBenchmark(
        strategy="dual_confirm_btc_stables",
        start=evaluation_start.date().isoformat(),
        end=evaluation_end.date().isoformat(),
        switches=tuple(executions),
        naive=_strategy_metrics(eval_prices, eval_exposure, naive_costs, None),
        as_pin_touch=_strategy_metrics(
            eval_prices, eval_exposure, touch_costs, touch_fill_rate
        ),
        as_pin_close=_strategy_metrics(
            eval_prices, eval_exposure, close_costs, close_fill_rate
        ),
        pin_usable_fraction=usable_fraction,
    )
