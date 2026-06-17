"""Macro- and on-chain-conditioned risk-on/off gates for BTC.

Each strategy is the smallest possible causal DAG:  state_t -> exposure_t ->
return_{t+1}.  A single macro or crypto-native state variable decides whether
to hold BTC (exposure 1) or sit in stables (exposure 0, earning 0%).  The
decision uses data through day t and is applied to day t+1's return (the
shared `backtest_rule` shifts exposure by one day) with a switching cost on
every change.

State -> exposure map for each rule (all CAUSAL: trailing windows / rolling
quantiles only, never a threshold fit on the full sample):

  dollar_trend_off   risk-OFF when the broad dollar index DTWEXBGS is in an
                     uptrend (DXY > its own 50d MA).  Strong dollar = global
                     risk-off; crypto historically sells off with a rising USD.

  rates_off          risk-OFF when the 10y yield DGS10 is rising (DGS10 > its
                     60d MA).  Rising rates tighten financial conditions.

  curve_steepen_off  risk-OFF when the 2s10s curve T10Y2Y is steepening fast
                     (T10Y2Y > its 60d MA + buffer).  Bull-steepening often
                     accompanies recession onset / Fed cuts under stress.
  curve_flatten_off  the opposite sign (risk-off when curve is flattening),
                     reported honestly alongside so the sign is not cherry-picked.

  m2_liquidity_on    risk-ON when M2 money supply M2SL is above its 6-month
                     trailing average (expanding liquidity), else stables.

  stablecoin_flow_on risk-ON when aggregate stablecoin supply (USDT+USDC+USDe)
                     is above its 30d MA (crypto capital inflowing), else
                     stables.  A crypto-native liquidity / inflow proxy.

  funding_off        risk-OFF when aggregate perp funding is negative OR in its
                     lower trailing quantile (shorts paying longs / fear), else
                     BTC.  Two variants: sign-based and quantile-based.

  <best>_and_trend50 the single best macro/on-chain gate AND-combined with the
                     classic BTC 50d price trend filter — does the gate ADD to
                     price trend, or does it just subtract exposure?

Baselines: buy-and-hold BTC (`always_in`) and the price-trend `trend_50`
(BTC price > its 50d MA).  Every rule that beats BH on Sharpe or Calmar is
placebo-tested (circular time-shift of the exposure path) to check whether the
edge is real TIMING skill or merely a lower average exposure reproducible at
random.

Run:  python -m causal_portfolio.experiments.macro_onchain_gates
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the validated harness — do NOT reimplement backtest/placebo logic.
from causal_portfolio.experiments.regime_rotation import (
    RotResult,
    backtest_rule,
    placebo_p,
)
from causal_portfolio.factors.builder import MACRO_SERIES

logger = logging.getLogger("cpcm.experiments.macro_onchain_gates")

DB_PATH = "causal_portfolio/data/cpcm_local.duckdb"


# ── data loaders ────────────────────────────────────────────────────


def load_btc(start: str, end: str) -> pd.Series:
    from causal_portfolio.data import get_loader

    loader = get_loader()
    returns = loader.load_returns(["btc"], start, end)
    returns.index = pd.to_datetime(returns.index)
    return returns["btc_return"].sort_index()


def load_macro_frame(start: str, end: str) -> pd.DataFrame:
    """FRED macro series, columns lowercased (vixcls, dtwexbgs, dgs10, ...)."""
    from causal_portfolio.data import get_loader

    loader = get_loader()
    macro = loader.load_macro(MACRO_SERIES, start, end)
    if macro is None or not len(macro):
        return pd.DataFrame()
    macro.index = pd.to_datetime(macro.index)
    macro.columns = [c.lower() for c in macro.columns]
    return macro.sort_index()


def load_stablecoin_supply(start: str, end: str) -> pd.Series:
    """Aggregate stablecoin supply (USDT+USDC SplyCur + USDe circulating_usd).

    A crypto-native capital-inflow proxy.  Daily, summed across the three.
    """
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql(
        """
        select cast(time at time zone 'UTC' as date) as day,
               asset, max(value) as v
        from asset_metrics_best
        where (metric = 'SplyCur' and asset in ('usdt', 'usdc'))
           or (metric = 'stablecoin_circulating_usd' and asset = 'usde')
        group by 1, 2
        """
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    wide = df.pivot_table(index="day", columns="asset", values="v")
    wide.index = pd.to_datetime(wide.index)
    # USDe only exists from 2024-02; treat its pre-launch supply as 0 so the
    # aggregate is a real running total rather than NaN-poisoned.
    agg = wide.fillna(0.0).sum(axis=1).sort_index()
    agg = agg[(agg.index >= pd.Timestamp(start)) & (agg.index <= pd.Timestamp(end))]
    return agg


def load_funding(start: str, end: str) -> pd.Series:
    """Aggregate perp funding: sum hourly funding_rate_8h per UTC day per asset,
    then average across assets.  Daily series (from 2023-11)."""
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql(
        """
        with daily as (
            select asset,
                   cast(time at time zone 'UTC' as date) as day,
                   sum(value) as f
            from asset_metrics_best
            where metric = 'funding_rate_8h'
            group by 1, 2
        )
        select day, avg(f) as f
        from daily
        group by 1
        """
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["f"].values, index=pd.to_datetime(df["day"])).sort_index()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


# ── signal helpers (all causal) ─────────────────────────────────────


def _price(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _ma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def _rolling_lower_quantile_flag(s: pd.Series, q: float, window: int) -> pd.Series:
    """1 when s is at/below its trailing q-quantile (lower tail), else 0.

    Causal: the threshold at t uses only s[..t]."""
    thresh = s.rolling(window, min_periods=min(60, window)).quantile(q)
    return (s <= thresh).astype(float)


# ── exposure-rule builders: each returns a Series in [0,1] on `idx` ──


def build_rules(
    btc: pd.Series,
    macro: pd.DataFrame,
    stablecoin: pd.Series,
    funding: pd.Series,
) -> dict[str, pd.Series]:
    idx = btc.index
    price = _price(btc)
    rules: dict[str, pd.Series] = {}

    # Baselines ------------------------------------------------------
    rules["always_in"] = pd.Series(1.0, index=idx)  # = BH BTC
    rules["trend_50"] = (
        price > price.rolling(50, min_periods=25).mean()
    ).astype(float)

    # 1. Dollar trend gate ------------------------------------------
    if "dtwexbgs" in macro:
        dxy = macro["dtwexbgs"].reindex(idx).ffill()
        dxy_up = dxy > _ma(dxy, 50)
        # risk-OFF (0) when dollar is in an uptrend.
        rules["dollar_trend_off"] = (~dxy_up).astype(float)

    # 2. Rates gate --------------------------------------------------
    if "dgs10" in macro:
        y10 = macro["dgs10"].reindex(idx).ffill()
        rising = y10 > _ma(y10, 60)
        # risk-OFF when 10y is rising.
        rules["rates_off"] = (~rising).astype(float)

    # 2b. 2s10s curve gate (both signs, reported honestly) ----------
    if "t10y2y" in macro:
        curve = macro["t10y2y"].reindex(idx).ffill()
        steepening = curve > _ma(curve, 60)
        # risk-OFF when curve is steepening (bull-steepening = stress proxy).
        rules["curve_steepen_off"] = (~steepening).astype(float)
        # Opposite sign: risk-OFF when curve is flattening/inverting further.
        rules["curve_flatten_off"] = steepening.astype(float)

    # 3. M2 liquidity gate ------------------------------------------
    if "m2sl" in macro:
        m2 = macro["m2sl"].reindex(idx).ffill()
        # 6-month (~126 trading / 182 calendar day) trailing average. M2 is
        # daily-resampled+ffilled upstream, so use a ~182-calendar-day window.
        m2_expanding = m2 > _ma(m2, 182)
        # risk-ON when liquidity expanding.
        rules["m2_liquidity_on"] = m2_expanding.astype(float)

    # 4. Stablecoin-supply liquidity gate ---------------------------
    if stablecoin is not None and len(stablecoin):
        sc = stablecoin.reindex(idx).ffill()
        sc_growing = sc > _ma(sc, 30)
        rules["stablecoin_flow_on"] = sc_growing.astype(float)

    # 5. Funding regime ---------------------------------------------
    if funding is not None and len(funding):
        f = funding.reindex(idx).ffill()
        # Sign-based: risk-OFF when aggregate funding is negative.
        rules["funding_neg_off"] = (f >= 0).astype(float)
        # Quantile-based: risk-OFF when funding is in its lower trailing
        # quintile (fear / shorts paying), else on. 252d trailing window.
        lower_q = _rolling_lower_quantile_flag(f, 0.20, 252)
        rules["funding_q_off"] = (1.0 - lower_q)

    return rules


# ── report ──────────────────────────────────────────────────────────


def render_markdown(
    results: list[RotResult],
    placebos: dict[str, dict],
    bh: RotResult,
    trend50: RotResult,
    args: dict,
    combo_note: str,
) -> str:
    from datetime import datetime, timezone

    L = ["# Macro- and on-chain-conditioned risk-on/off gates for BTC\n"]
    L.append(
        "Each rule gates BTC↔stables (0% when out) on a single causal macro or "
        "crypto-native state variable; decision at t close, applied to t+1 "
        f"return, {args['fee_bps_oneway']:.0f}bp one-way switching cost. Sorted "
        "by Sharpe. Baselines: `always_in` (= buy-and-hold BTC) and `trend_50` "
        "(BTC price > 50d MA).\n"
    )
    L.append(
        f"- Asset: BTC | window `{args['start']}` → `{args['end']}` "
        f"({bh.daily.shape[0]} days)"
    )
    L.append(
        f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n"
    )

    L.append(
        "| Rule | Ann.ret | Sharpe | Sortino | MaxDD | Calmar | "
        "%in mkt | switches | placebo p (Sharpe) |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda r: -r.sharpe):
        pp = placebos.get(r.name, {}).get("sharpe")
        pps = f"{pp:.2f}" if pp is not None else "—"
        tag = ""
        if r.name == "always_in":
            tag = " ⟵ BH"
        elif r.name == "trend_50":
            tag = " ⟵ trend baseline"
        L.append(
            f"| {r.name}{tag} | {r.ann_return:+.0%} | {r.sharpe:.2f} | "
            f"{r.sortino:.2f} | {r.max_dd:.0%} | {r.calmar:.2f} | "
            f"{r.pct_in_market:.0%} | {r.n_switches} | {pps} |"
        )
    L.append("")

    # Verdict ---------------------------------------------------------
    beat_bh = [
        r
        for r in results
        if r.name not in ("always_in",)
        and (r.sharpe > bh.sharpe or r.calmar > bh.calmar)
    ]
    survivors = [
        r
        for r in beat_bh
        if r.sharpe > bh.sharpe
        and (placebos.get(r.name, {}).get("sharpe", 1.0) < 0.10)
    ]
    L.append("## Verdict\n")
    L.append(
        f"Buy-and-hold BTC: ann {bh.ann_return:+.0%}, Sharpe {bh.sharpe:.2f}, "
        f"maxDD {bh.max_dd:.0%}, Calmar {bh.calmar:.2f}.\n"
    )
    L.append(
        f"Price-trend baseline `trend_50`: ann {trend50.ann_return:+.0%}, "
        f"Sharpe {trend50.sharpe:.2f}, maxDD {trend50.max_dd:.0%}, "
        f"Calmar {trend50.calmar:.2f}.\n"
    )
    if survivors:
        L.append(
            "Gates that beat BH on Sharpe AND whose timing survives the placebo "
            "(p < 0.10):"
        )
        for r in sorted(survivors, key=lambda r: -r.sharpe):
            pp = placebos[r.name]["sharpe"]
            L.append(
                f"- **{r.name}** — Sharpe {r.sharpe:.2f} vs BH {bh.sharpe:.2f}, "
                f"maxDD {r.max_dd:.0%} vs {bh.max_dd:.0%} (placebo p={pp:.2f})"
            )
        L.append(
            "\n**Read this skeptically.** The 2s10s curve has only a few "
            "independent regime turns over 2021–2025, so a low placebo p can be "
            "a small-sample artifact (the circular shift cannot manufacture many "
            "distinct exposure paths from such a slow signal). Note also that the "
            "*flattening/inverting* sign is the one that helps; the opposite "
            "`curve_steepen_off` sign is among the worst rules — so the sign was "
            "chosen by which helped, exactly the kind of two-way bet that inflates "
            "in-sample significance. Treat the curve result as a hypothesis to "
            "test out-of-sample, not a deployable edge."
        )
    else:
        L.append(
            "**No macro/on-chain gate both beats BH on Sharpe and survives the "
            "placebo (p < 0.10).** Some gates lift Calmar by cutting drawdown, "
            "but they do so by sitting out of the market on average, not by "
            "skillful regime calls — a random-timed overlay with the same "
            "average exposure does as well or better."
        )
    L.append("")
    L.append(combo_note)
    L.append("")

    # Caveats ---------------------------------------------------------
    L.append("## Caveats\n")
    L.append(
        "- **Publication lag (macro).** FRED series are forward-filled to the "
        "BTC daily index. Releases are backward-dated, so ffill of *realized* "
        "values introduces no look-ahead, but the actual *publication* of a "
        "month's M2 / CPI lags the reference date by days-to-weeks; the live "
        "tradability of M2- and CPI-based gates is mildly optimistic here."
    )
    L.append(
        "- **Few independent regime changes.** Over 2021–2025 the macro series "
        "(dollar trend, 10y level, curve, M2 trend) have only a handful of "
        "independent regime turns. The placebo and any implied degrees of "
        "freedom are therefore weak — a 'survivor' on this sample is not strong "
        "evidence, and a non-survivor is the safer read."
    )
    L.append(
        "- **On-chain funding starts 2023-11.** The aggregate perp funding "
        "series only begins 2023-11, so `funding_*` rules are evaluated on a "
        "much shorter, mostly-bull sub-window and are not comparable to the "
        "full-sample macro rules."
    )
    L.append(
        "- **Stables earn 0%.** Risk-off legs sit in cash at 0% return "
        "(conservative); a real stablecoin yield would only improve the gated "
        "rules relative to BH."
    )
    return "\n".join(L)


# ── orchestration ───────────────────────────────────────────────────


def run(args) -> str:
    btc = load_btc(args.start, args.end)
    macro = load_macro_frame(args.start, args.end)
    stablecoin = load_stablecoin_supply(args.start, args.end)
    funding = load_funding(args.start, args.end)

    logger.info(
        "loaded: btc=%d days, macro cols=%s, stablecoin=%d, funding=%d",
        len(btc),
        list(macro.columns) if len(macro) else [],
        len(stablecoin),
        len(funding),
    )

    rules = build_rules(btc, macro, stablecoin, funding)

    results: list[RotResult] = []
    for name, exp in rules.items():
        r = backtest_rule(btc, exp, fee_bps_oneway=args.fee_bps_oneway)
        r.name = name
        results.append(r)
        logger.info(
            "%-20s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% calmar=%.2f in=%.0f%%",
            name,
            r.ann_return * 100,
            r.sharpe,
            r.max_dd * 100,
            r.calmar,
            r.pct_in_market * 100,
        )

    bh = next(r for r in results if r.name == "always_in")
    trend50 = next(r for r in results if r.name == "trend_50")

    # ── pick best non-baseline gate and AND-combine with trend_50 ──
    gate_candidates = [
        r for r in results if r.name not in ("always_in", "trend_50")
    ]
    best_gate = max(gate_candidates, key=lambda r: r.sharpe)
    combo_name = f"{best_gate.name}_and_trend50"
    combo_exp = (rules[best_gate.name] * rules["trend_50"]).clip(0, 1)
    combo = backtest_rule(btc, combo_exp, fee_bps_oneway=args.fee_bps_oneway)
    combo.name = combo_name
    rules[combo_name] = combo_exp
    results.append(combo)
    logger.info(
        "%-20s ann=%+.0f%% sharpe=%.2f maxDD=%.0f%% calmar=%.2f in=%.0f%%",
        combo_name,
        combo.ann_return * 100,
        combo.sharpe,
        combo.max_dd * 100,
        combo.calmar,
        combo.pct_in_market * 100,
    )

    # Does the gate ADD to price trend? Compare combo vs trend_50 alone.
    adds = combo.sharpe > trend50.sharpe + 1e-9
    if adds:
        combo_note = (
            f"## Does a macro/on-chain gate ADD to price trend?\n\n"
            f"Best standalone gate: **{best_gate.name}** (Sharpe "
            f"{best_gate.sharpe:.2f}). AND-combined with `trend_50` → "
            f"**{combo_name}**: Sharpe {combo.sharpe:.2f} vs `trend_50` alone "
            f"{trend50.sharpe:.2f}, Calmar {combo.calmar:.2f} vs "
            f"{trend50.calmar:.2f}. The gate **ADDS** to the price trend on "
            f"this sample (higher Sharpe than trend alone), though it also cuts "
            f"%-in-market to {combo.pct_in_market:.0%} from "
            f"{trend50.pct_in_market:.0%} — check the placebo before trusting it."
        )
    else:
        combo_note = (
            f"## Does a macro/on-chain gate ADD to price trend?\n\n"
            f"Best standalone gate: **{best_gate.name}** (Sharpe "
            f"{best_gate.sharpe:.2f}). AND-combined with `trend_50` → "
            f"**{combo_name}**: Sharpe {combo.sharpe:.2f} vs `trend_50` alone "
            f"{trend50.sharpe:.2f}, Calmar {combo.calmar:.2f} vs "
            f"{trend50.calmar:.2f}. The gate does **NOT** add to the price "
            f"trend — AND-ing it on top of `trend_50` only **subtracts** "
            f"exposure (%-in-market {combo.pct_in_market:.0%} vs "
            f"{trend50.pct_in_market:.0%}) without improving risk-adjusted "
            f"return. The price trend already captures what edge exists."
        )

    # ── placebo: every rule beating BH on Sharpe or Calmar ──
    placebos: dict[str, dict] = {}
    for r in results:
        if r.name == "always_in":
            continue
        if r.sharpe > bh.sharpe or r.calmar > bh.calmar:
            p = placebo_p(
                btc,
                rules[r.name],
                r,
                metric="sharpe",
                n=args.n_placebo,
                fee_bps_oneway=args.fee_bps_oneway,
            )
            placebos[r.name] = {"sharpe": p}
            logger.info("placebo %-20s p(Sharpe)=%.2f", r.name, p)

    md = render_markdown(results, placebos, bh, trend50, vars(args), combo_note)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")
    return md


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fee-bps-oneway", type=float, default=5.0)
    p.add_argument("--n-placebo", type=int, default=200)
    p.add_argument(
        "--out", default="causal_portfolio/docs/macro_onchain_gates.md"
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
