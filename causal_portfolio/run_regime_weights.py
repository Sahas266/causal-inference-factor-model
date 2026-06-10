"""Emit current REGIME_EW weights as JSON for the executor.

This is the bridge between the winning strategy from `run_strategy_search.py`
and the Hyperliquid execution layer. Each invocation:

  1. Loads the recent (VIX, BTC realized vol) history from the local DuckDB.
  2. Fits the HMM on the trailing window.
  3. Runs the forward filter for today's regime label.
  4. Writes a weights JSON file:
       - Calm regime  → {"btc": 0.333, "eth": 0.333, "sol": 0.333}
       - Stress regime → all zeros (cash)

Then you can pipe it to the executor:
    python -m causal_portfolio.run_regime_weights --out weights.json
    python -m causal_portfolio.execution.cli execute \\
        --weights weights.json --live --testnet

Safe to run daily: when the regime label hasn't changed, the resulting
deltas are near zero and the rebalancer skips sizes that round to nothing
(there is no explicit no-trade band — tiny drift trades may still go out).

Usage:
    python -m causal_portfolio.run_regime_weights
    python -m causal_portfolio.run_regime_weights --out weights.json
    python -m causal_portfolio.run_regime_weights --assets btc,eth,sol --end 2026-05-25
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from causal_portfolio.data import get_loader
from causal_portfolio.regimes.hmm import (
    RegimeClassifier,
    build_regime_features,
)

logger = logging.getLogger("cpcm.regime_weights")


STATE_FILE_DEFAULT = Path(__file__).parent / "data" / "regime_last_state.json"


def _read_last_state(path: Path) -> dict | None:
    """Read the last successful regime decision from disk, if any."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("could not read last regime state from %s: %s", path, e)
        return None


def _write_last_state(path: Path, info: dict) -> None:
    """Persist the latest regime decision so we have a fallback next time."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2, default=str))
    except Exception as e:
        logger.warning("could not write regime state to %s: %s", path, e)


def compute_current_regime(
    assets: list[str],
    end_date: str | None = None,
    hmm_window: int = 504,
    n_states: int = 2,
    n_restarts: int = 5,
    lookback_buffer_days: int = 60,
    fallback_state_file: Path | None = None,
    on_failure: str = "fallback",
) -> dict:
    """Compute today's regime label using the trailing window.

    Production-safety: if the HMM fit or data load fails, fall back to the
    last successfully-decoded regime (read from `fallback_state_file`). If
    no fallback exists, default to the stress state (cash) — the
    conservative choice when we can't tell the regime.

    Args:
        assets: asset universe for return loading.
        end_date: ISO date for the regime decision (default: today).
        hmm_window: rolling HMM window in days.
        n_states: HMM components.
        n_restarts: HMM multi-start seed count.
        lookback_buffer_days: padding above hmm_window when fetching data.
        fallback_state_file: where last-known regime is persisted. None
            uses the default `causal_portfolio/data/regime_last_state.json`.
        on_failure: "fallback" (default — use last-known, else stress),
            "raise" (propagate the exception, useful for debugging),
            or "stress" (always fall back to stress regime on failure).

    Returns:
        {
            "regime": int,
            "posterior": list[float],
            "as_of": str (ISO date),
            "feature_columns": list[str],
            "hmm_window": int,
            "source": "live" | "fallback_last_known" | "fallback_stress",
        }
    """
    state_file = fallback_state_file or STATE_FILE_DEFAULT
    try:
        info = _compute_regime_live(
            assets=assets, end_date=end_date, hmm_window=hmm_window,
            n_states=n_states, n_restarts=n_restarts,
            lookback_buffer_days=lookback_buffer_days,
        )
        info["source"] = "live"
        _write_last_state(state_file, info)
        return info
    except Exception as e:
        logger.error("regime computation failed: %s", e)
        if on_failure == "raise":
            raise
        if on_failure == "stress":
            return _stress_fallback(n_states, str(e))
        # default "fallback": try last-known, else stress
        last = _read_last_state(state_file)
        if last is not None:
            logger.warning(
                "using last-known regime from %s (as_of %s)",
                state_file, last.get("as_of", "?"),
            )
            return {**last, "source": "fallback_last_known",
                    "fallback_reason": str(e)}
        logger.warning("no fallback state available — defaulting to stress")
        return _stress_fallback(n_states, str(e))


def _stress_fallback(n_states: int, reason: str) -> dict:
    """Conservative default when we can't tell the regime: go to stress (cash).

    States are sorted by stress (feature-0 mean) at fit time, so the
    highest-index state is always the most stressed one.
    """
    stress = n_states - 1
    return {
        "regime": stress,
        "posterior": [0.0] * stress + [1.0],
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "feature_columns": [],
        "hmm_window": 0,
        "n_states": n_states,
        "source": "fallback_stress",
        "fallback_reason": reason,
    }


def _compute_regime_live(
    assets: list[str],
    end_date: str | None,
    hmm_window: int,
    n_states: int,
    n_restarts: int,
    lookback_buffer_days: int,
) -> dict:
    """The original (HMM-fit) path, factored out for try/except wrapping."""
    loader = get_loader()

    # Pull enough history for the HMM window plus the feature warmup
    end = end_date or datetime.now(timezone.utc).date().isoformat()
    # Approximate: 600 days back covers 504 HMM + 21 vol warmup + buffer
    from datetime import date, timedelta
    end_dt = datetime.fromisoformat(end).date()
    start_dt = end_dt - timedelta(days=hmm_window + lookback_buffer_days + 30)

    returns = loader.load_returns(assets, start_dt.isoformat(), end_dt.isoformat())
    macro = loader.load_macro(["VIXCLS"], start_dt.isoformat(), end_dt.isoformat())

    # Align + ffill macro to returns
    common = returns.index.intersection(macro.index)
    returns = returns.loc[common]
    macro = macro.loc[common].ffill()

    if "btc_return" not in returns.columns:
        raise ValueError("BTC returns required for regime features (btc_vol)")

    feats = build_regime_features(macro, returns)
    if len(feats) < hmm_window:
        raise ValueError(
            f"Not enough feature history: got {len(feats)}, need {hmm_window}. "
            f"Try a smaller --hmm-window or refresh the local DuckDB snapshot."
        )

    window = feats.iloc[-hmm_window:]
    classifier = RegimeClassifier(
        n_states=n_states, n_restarts=n_restarts,
    ).fit(window)
    posterior = classifier.forward_filter(window)[-1]
    regime = int(posterior.argmax())

    return {
        "regime": regime,
        "posterior": [float(x) for x in posterior],
        "as_of": str(feats.index[-1].date()),
        "feature_columns": list(feats.columns),
        "hmm_window": hmm_window,
        "n_states": n_states,
    }


def regime_to_weights(
    regime: int, assets: list[str],
    stress_state: int = 1, stress_allocation: float = 0.0,
) -> dict[str, float]:
    """Convert a regime label to the executor's weights JSON shape.

    `stress_state` must be the highest-index state (n_states - 1) — states
    are sorted by stress at fit time. The default of 1 matches n_states=2.
    """
    n = len(assets)
    if regime == stress_state:
        per_asset = stress_allocation / n if n else 0.0
    else:
        per_asset = 1.0 / n if n else 0.0
    return {a.lower(): per_asset for a in assets}


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=str, default="btc,eth,sol")
    p.add_argument("--end", type=str, default=None,
                   help="ISO date for the regime decision (default: today)")
    p.add_argument("--out", type=str, default=None,
                   help="Write weights JSON here (default: stdout only)")
    p.add_argument("--hmm-window", type=int, default=504)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--stress-allocation", type=float, default=0.0,
                   help="Target weight in basket during stress (default 0 = cash)")
    args = p.parse_args()

    assets = args.assets.split(",")
    info = compute_current_regime(
        assets=assets, end_date=args.end,
        hmm_window=args.hmm_window, n_states=args.n_states,
    )
    stress_state = info.get("n_states", args.n_states) - 1
    weights = regime_to_weights(
        info["regime"], assets,
        stress_state=stress_state,
        stress_allocation=args.stress_allocation,
    )

    payload = {
        **weights,
        "_meta": {
            **info,
            "strategy": "REGIME_EW",
            "stress_allocation": args.stress_allocation,
            "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    # Console summary
    regime_name = (
        "STRESS (cash)" if info["regime"] == stress_state else "CALM (long basket)"
    )
    print(f"As of {info['as_of']}:")
    print(f"  Regime: {info['regime']} — {regime_name}")
    print(f"  Posterior: {[round(x, 3) for x in info['posterior']]}")
    print(f"  Weights: {weights}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
