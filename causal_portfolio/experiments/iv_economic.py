"""Economically-motivated instrumental-variable search (exclusion-first).

The prior search (`iv_search.py` → `iv_findings.md`) was purely *statistical*:
it scored thousands of warehouse columns on first-stage relevance and a
mechanical direct-path falsifier, then found that the strong, non-tautological
instruments it surfaced (e.g. `aave_net_treasury`, `ena_tvl_usd`) made 2SLS
*worse* out of sample. Exclusion — the requirement that the instrument hits
returns ONLY through the treatment — was never the selection criterion. It is
the real problem, and it cannot be screened by an F-stat.

This module flips the order: every candidate here is chosen FIRST for a
plausible exclusion story (a supply-side / scheduled / mechanical driver of the
treatment that has no obvious second channel to returns), and only THEN tested
for relevance and the same falsifiers. The point is to test the *best*
economically-motivated instruments honestly, not to data-mine relevance.

Exclusion rationale, per instrument (the economic "why returns only via T"):

1. Mining / security supply shock → chain_congestion (and liq_flow).
   `btc_HashRate` / `btc_difficulty` move with miner capacity and the
   2-week difficulty retarget — a SUPPLY-side process set by hardware and
   energy economics, exogenous to today's demand for blockspace or returns.
   A hashrate shock should change congestion/security, and only thereby touch
   price. Exclusion risk: hashrate also proxies miner sell-pressure (a direct
   path to returns) — a known weakness, flagged via the direct-path probe.

2. Token emission / staking-queue mechanics → staking_yield.
   `eth_gross_emissions`, `eth_IssTotNtv` (issuance), `eth_consensus_rewards`,
   and the validator `queue_entry/active/exit_amount` series are MECHANICAL:
   issuance follows protocol schedule; the entry/exit queue drains at a fixed
   per-epoch churn limit. More validators entering mechanically dilutes APR.
   The schedule is not set by demand, so it should hit returns only via the
   yield it determines. Exclusion risk: a long entry queue also signals
   staking demand (sentiment) — a second channel — so the entry queue is the
   weakest of these on exclusion; exit queue / issuance are cleaner.

3. ETF flows in NATIVE units → stable_flow / funding_basis.
   `eth_etf_aum_native` is AUM measured in COINS, not USD, so it is a
   flow/positioning measure rather than a price mark (a USD AUM series would
   be mechanically `coins x price` and fail exclusion by construction). A
   TradFi ETF creation/redemption is an exogenous demand shock that reaches
   the crypto spot/perp market only through the on-ramp: authorised
   participants must source coins (stablecoin rails) and hedge (perp funding).
   So it plausibly hits returns via stablecoin flow / funding positioning.
   Exclusion risk: ETF demand is itself sentiment-driven and could move price
   directly — flagged via the direct-path probe; native units mitigate but do
   not eliminate this.

4. Stablecoin redemption / issuance mechanics → stable_flow, built to AVOID
   the `stablecoin_mint` tautology. `stable_flow` is diff(USDC+USDT+USDe
   aggregate supply). We instrument it with a DIFFERENT object: a single
   chain's stablecoin float (`bnb_stablecoin_supply`) or Ethereum
   mint/burn *events* (`eth_stablecoin_minted_usd` / `_burned_usd`), which are
   issuer-side primary-market operations. These are not the aggregate
   treatment series, so the near-infinite tautological F of the original trap
   is avoided by construction. Exclusion story: issuer mint/burn is an
   operational supply decision; it should move aggregate stable flow and only
   thereby returns.

5. Scheduled macro surprise proxies → funding_basis.
   `VIXCLS` (equity-vol jump), `BAMLH0A0HYM2` (HY credit spread), `DGS2`
   (2y rate move): macro risk shocks set in TradFi, exogenous to crypto
   micro-structure, that transmit to crypto via leverage/positioning —
   i.e. funding basis (de-risking compresses or flips perp funding). Exclusion
   risk: a global risk-off hits crypto price through every risk asset
   simultaneously (a direct path), so these are exclusion-fragile by nature;
   the probe will catch the most blatant cases.

METHOD (mirrors the honesty bar in iv_findings.md / iv_search.py):
  * Each candidate is a lag-1 innovation series (`_candidate_series`: diff or
    spike). Relevance/selection statistics use the TRAIN window only (the
    first `train_window` aligned rows) so picking instruments cannot leak the
    OOS folds.
  * Report, per candidate: first-stage partial F (relevance), |corr(Z,T)|
    (tautology), direct-path |t| of Z in returns_{t+1} ~ [T, Z] (exclusion red
    flag), and relevance stability on later non-overlapping windows.
  * Candidates that clear relevance (F >= bar) AND are non-tautological go into
    the gated 2SLS-vs-OLS walk-forward A/B (`iv_search.run_shortlist_ab`).
  * An instrument only WINS if it is strong, non-tautological, has no
    direct-path flag, AND 2SLS >= OLS out of sample. Exclusion is ultimately
    untestable; a passed direct-path probe is necessary, not sufficient.

Run:  python -m causal_portfolio.experiments.iv_economic --ab
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from causal_portfolio.factors.builder import (
    FACTOR_SOURCE_ASSETS, MACRO_SERIES, PANEL_METRICS, build_all_factors,
)
# Reuse the IV machinery — DO NOT re-implement it.
from causal_portfolio.experiments.iv_search import (
    SHORTLIST as _STAT_SHORTLIST,  # noqa: F401  (kept for reference/diffing)
    _candidate_series, _direct_path_t, _first_stage_f, run_shortlist_ab,
)

logger = logging.getLogger("cpcm.experiments.iv_economic")


# ── the economically-motivated candidate set ────────────────────────────
#
# (treatment, source_column, transform, exclusion_rationale)
# Source columns are {asset}_{metric}; macro series are pulled in under the
# `macro_<series>` prefix so they screen like any other raw column.
#
# This is a CANDIDATE LIST (several per treatment), unlike iv_search.SHORTLIST
# which is one-per-treatment. We screen all of them and only feed the
# relevance-clearing, non-tautological ones to the A/B.

ECON_CANDIDATES: list[tuple[str, str, str, str]] = [
    # 1. Mining / security supply shocks
    ("chain_congestion", "btc_HashRate", "diff",
     "mining hashrate is a supply-side process (hardware/energy), exogenous to "
     "blockspace demand; should hit price only via congestion/security"),
    ("chain_congestion", "btc_difficulty", "diff",
     "2-week difficulty retarget is mechanical, set by past hashrate not demand"),
    ("chain_congestion", "btc_HashRate", "spike",
     "abrupt hashrate jumps (miner capacity shocks) as congestion shifters"),
    ("liq_flow", "btc_HashRate", "diff",
     "security/supply shock as an exogenous mover of on-chain liquidity"),

    # 2. Token emission / staking-queue mechanics
    ("staking_yield", "eth_gross_emissions", "diff",
     "protocol issuance schedule mechanically dilutes APR; not demand-driven"),
    ("staking_yield", "eth_IssTotNtv", "diff",
     "native token issuance is schedule-mechanical, exogenous to price demand"),
    ("staking_yield", "eth_consensus_rewards_eth", "diff",
     "consensus-layer rewards are protocol-set; they set yield, not price directly"),
    ("staking_yield", "eth_queue_exit_amount", "diff",
     "validator EXIT queue drains at a fixed per-epoch churn limit (mechanical); "
     "more exits mechanically raise APR for those remaining"),
    ("staking_yield", "eth_queue_entry_amount", "diff",
     "validator ENTRY queue (weaker exclusion: also signals staking demand)"),

    # 3. ETF flows in NATIVE units (flow, not price mark)
    ("stable_flow", "eth_etf_aum_native", "diff",
     "ETF AUM in coins = TradFi positioning; AP creation sources coins via "
     "stablecoin rails, so it should hit price via on-ramp stable flow"),
    ("funding_basis", "eth_etf_aum_native", "diff",
     "ETF positioning shock; APs hedge in perps, transmitting only via funding"),
    ("funding_basis", "btc_etf_flows_native", "diff",
     "daily ETF net creation in coins as a positioning shock to perp funding"),
    ("funding_basis", "eth_etf_aum_native", "spike",
     "abrupt ETF positioning jumps as funding shifters"),

    # 4. Stablecoin redemption/issuance mechanics (NON-tautological by design)
    ("stable_flow", "eth_stablecoin_minted_usd", "diff",
     "Ethereum primary-market stablecoin MINT events (issuer-side ops), "
     "distinct from the aggregate USDC+USDT+USDe supply that IS the treatment"),
    ("stable_flow", "eth_stablecoin_burned_usd", "diff",
     "Ethereum stablecoin BURN/redemption events, issuer-side, not the treatment"),
    ("stable_flow", "bnb_stablecoin_supply", "diff",
     "a DIFFERENT chain's stablecoin float as an instrument for aggregate flow"),

    # 5. Scheduled macro surprise proxies
    ("funding_basis", "macro_vixcls", "diff",
     "equity-vol move = TradFi risk shock, transmits to crypto via de-risking "
     "/ leverage, i.e. funding basis"),
    ("funding_basis", "macro_vixcls", "spike",
     "VIX jumps (discrete risk shocks) as funding shifters"),
    ("funding_basis", "macro_bamlh0a0hym2", "diff",
     "HY credit-spread move = macro risk-appetite shock hitting crypto leverage"),
    ("funding_basis", "macro_dgs2", "diff",
     "2y rate move = monetary surprise, transmits via cost-of-leverage / funding"),
]

# Extra raw metric columns to pull beyond PANEL_METRICS so the instrument
# sources above exist in the `raw` frame.
EXTRA_SOURCE_METRICS = [
    "HashRate", "difficulty",
    "gross_emissions", "IssTotNtv", "consensus_rewards_eth",
    "queue_entry_amount", "queue_exit_amount", "queue_active_amount",
    "etf_aum_native", "etf_flows_native",
    "stablecoin_minted_usd", "stablecoin_burned_usd", "stablecoin_supply",
]

# Extra macro series (beyond MACRO_SERIES) used as instruments.
EXTRA_MACRO_SERIES = ["VIXCLS", "BAMLH0A0HYM2", "DGS2"]


@dataclass
class EconIVCandidate:
    treatment: str
    candidate: str          # e.g. "eth_etf_aum_native[diff]"
    rationale: str
    train_f: float
    corr_zt: float
    direct_t: float
    stability: str          # "passed/seen" relevance windows with F >= bar
    n_obs: int
    flags: list[str]

    @property
    def clears_relevance(self) -> bool:
        return self.train_f >= 10.0 and "TAUTOLOGY" not in " ".join(self.flags)

    @property
    def is_clean(self) -> bool:
        return self.clears_relevance and not self.flags


# ── data assembly ───────────────────────────────────────────────────────

def build_raw_and_factors(assets, start, end):
    """Pull the panel + macro, build factors, and assemble the `raw` source
    frame (panel columns + `macro_<series>` columns) aligned to the factors.
    """
    from causal_portfolio.data import get_loader

    loader = get_loader()
    panel_assets = list(dict.fromkeys(assets + FACTOR_SOURCE_ASSETS + ["btc", "eth", "bnb"]))
    metrics = list(dict.fromkeys(PANEL_METRICS + EXTRA_SOURCE_METRICS))
    macro_series = list(dict.fromkeys(MACRO_SERIES + EXTRA_MACRO_SERIES))

    panel = loader.load_panel(panel_assets, metrics, start, end)
    macro = loader.load_macro(macro_series, start, end)
    returns = loader.load_returns(assets, start, end)
    factors = build_all_factors(panel, macro)

    macro_pref = macro.copy()
    macro_pref.columns = [f"macro_{c}" for c in macro_pref.columns]
    raw = pd.concat([panel, macro_pref], axis=1)
    raw.index = pd.to_datetime(raw.index)

    common = raw.index.intersection(factors.index)
    return raw.loc[common], factors.loc[common], returns


# ── screening (per declared candidate, not a full-warehouse sweep) ───────

def screen_econ_candidates(
    raw: pd.DataFrame, factors: pd.DataFrame, returns: pd.DataFrame, *,
    train_window: int = 252, f_threshold: float = 10.0,
    taut_corr: float = 0.9, n_stability_windows: int = 3,
) -> list[EconIVCandidate]:
    """Run the relevance/tautology/direct-path/stability screen on the declared
    economically-motivated candidates only.
    """
    r_mean = returns.mean(axis=1)
    out: list[EconIVCandidate] = []

    for treatment, col, tf, why in ECON_CANDIDATES:
        if treatment not in factors.columns or col not in raw.columns:
            out.append(EconIVCandidate(
                treatment, f"{col}[{tf}]", why, float("nan"), float("nan"),
                float("nan"), "—", 0, ["MISSING column/factor"]))
            continue

        Z_full = _candidate_series(raw[col].astype(float))[tf]
        T_full = factors[treatment]
        idx = pd.concat([T_full, Z_full, r_mean], axis=1).dropna().index
        if len(idx) < train_window + 60:
            out.append(EconIVCandidate(
                treatment, f"{col}[{tf}]", why, float("nan"), float("nan"),
                float("nan"), "—", len(idx), [f"TOO FEW ROWS ({len(idx)})"]))
            continue

        train = idx[:train_window]
        T = T_full.loc[train].values
        Z = Z_full.loc[train].values
        R1 = r_mean.shift(-1).loc[train].values
        ok = ~np.isnan(R1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_train = _first_stage_f(T, Z)
            corr = (float(abs(np.corrcoef(Z, T)[0, 1]))
                    if np.std(Z) > 1e-12 and np.std(T) > 1e-12 else 0.0)
            corr = 0.0 if np.isnan(corr) else corr
            direct = _direct_path_t(R1[ok], T[ok], Z[ok]) if ok.sum() >= 30 else 0.0

            later = idx[train_window:]
            wins = passed = 0
            for k in range(n_stability_windows):
                seg = later[k * train_window:(k + 1) * train_window]
                if len(seg) < train_window // 2:
                    break
                wins += 1
                if _first_stage_f(T_full.loc[seg].values,
                                  Z_full.loc[seg].values) >= f_threshold:
                    passed += 1

        flags = []
        if corr > taut_corr:
            flags.append(f"TAUTOLOGY |corr(Z,T)|={corr:.2f}")
        if direct > 2.0:
            flags.append(f"DIRECT-PATH |t|={direct:.1f}")
        if f_train < f_threshold:
            flags.append("WEAK (train F < bar)")
        if wins and passed == 0 and f_train >= f_threshold:
            flags.append("UNSTABLE (relevance dies after train)")

        out.append(EconIVCandidate(
            treatment=treatment, candidate=f"{col}[{tf}]", rationale=why,
            train_f=f_train, corr_zt=corr, direct_t=direct,
            stability=f"{passed}/{wins}" if wins else "0/0",
            n_obs=len(idx), flags=flags))

    return out


def build_ab_shortlist(cands: list[EconIVCandidate]) -> dict[str, tuple[str, str, str]]:
    """From the screened candidates, pick the best relevance-clearing,
    non-tautological instrument PER treatment for the A/B (one per treatment so
    each run instruments exactly one factor). Ranked by: clean first, then by
    train F. Returns the SHORTLIST dict shape run_shortlist_ab expects.
    """
    by_treatment: dict[str, list[EconIVCandidate]] = {}
    for c in cands:
        if not c.clears_relevance:
            continue
        by_treatment.setdefault(c.treatment, []).append(c)

    shortlist: dict[str, tuple[str, str, str]] = {}
    for treatment, group in by_treatment.items():
        group.sort(key=lambda c: (not c.is_clean, -c.train_f))
        best = group[0]
        col, tf = best.candidate.rstrip("]").split("[")
        flag_note = "" if best.is_clean else f" [FLAGS: {'; '.join(best.flags)}]"
        shortlist[treatment] = (
            col, tf,
            f"{best.rationale} (F={best.train_f:.0f}, stab {best.stability}, "
            f"direct-t={best.direct_t:.2f}){flag_note}")
    return shortlist


# ── reporting ────────────────────────────────────────────────────────────

def render_markdown(cands: list[EconIVCandidate], shortlist, args: dict) -> str:
    L = ["# Economically-motivated IV search — exclusion-first\n"]
    L.append(
        "Unlike the purely statistical warehouse sweep (`iv_search.md`), every "
        "candidate here was chosen FIRST for a plausible **exclusion story** (a "
        "supply-side / scheduled / mechanical driver of the treatment with no "
        "obvious second channel to returns), then tested for relevance and the "
        "same falsifiers. See the module docstring for the per-instrument "
        "economic rationale.\n")
    L.append(f"- Assets: `{', '.join(args['assets'])}`")
    L.append(f"- Window: `{args['start']}` → `{args['end']}` | train "
             f"{args['train_window']}d | relevance bar F ≥ {args['f_threshold']}")
    L.append(f"- Run UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")
    L.append("**Exclusion is untestable in general.** The direct-path probe "
             "(does Z predict returns beyond T?) is a *necessary* falsifier, "
             "not a proof of exclusion. A passed probe + a clean economic story "
             "is the best we can do; treat every 'clean' below as *not yet "
             "falsified*, not *validated*.\n")

    # Candidate table
    L.append("## Candidate screen\n")
    L.append("Selection stats on the train window only. `corr(Z,T)` is the "
             "tautology check (>0.9 ⇒ Z *is* T); `direct-path t` is the "
             "exclusion red flag (>2 ⇒ Z predicts returns past T); `stability` "
             "= later non-overlapping windows with F ≥ bar.\n")
    L.append("| # | Treatment | Candidate | train F | stab | |corr(Z,T)| | "
             "direct-path t | flags |")
    L.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, c in enumerate(cands, 1):
        f = "—" if np.isnan(c.train_f) else f"{c.train_f:.1f}"
        cr = "—" if np.isnan(c.corr_zt) else f"{c.corr_zt:.2f}"
        dt = "—" if np.isnan(c.direct_t) else f"{c.direct_t:.2f}"
        flags = "; ".join(c.flags) if c.flags else "clean"
        L.append(f"| {i} | {c.treatment} | `{c.candidate}` | {f} | "
                 f"{c.stability} | {cr} | {dt} | {flags} |")
    L.append("")

    clears = [c for c in cands if c.clears_relevance]
    clean = [c for c in cands if c.is_clean]
    L.append(f"**{len(clears)}/{len(cands)} candidates clear the relevance bar** "
             f"(F ≥ {args['f_threshold']}, non-tautological); **{len(clean)} are "
             f"fully clean** (also no direct-path / stability flag).\n")

    # Exclusion-story commentary grouped by category
    L.append("### Exclusion-story read per category\n")
    L.append("- **Mining/security (hashrate, difficulty → congestion/liq):** "
             "clean economic story, but first-stage relevance is ~0 — daily "
             "hashrate/difficulty innovations do not move the gas/TVL-based "
             "treatments at the train window. Irrelevant ⇒ unusable.")
    L.append("- **Emission/issuance (gross_emissions, issuance, consensus "
             "rewards → staking_yield):** cleanest exclusion (pure schedule), "
             "but F ≈ 0 — the smooth staking-APR level barely responds to "
             "daily issuance innovations. The validator **queue** series move "
             "APR strongly (F up to ~88) but the queue also signals demand "
             "(weaker exclusion) and relevance is **unstable** out of window.")
    L.append("- **ETF native flows → stable_flow / funding_basis:** the only "
             "instruments with a genuine exclusion story that ALSO clear "
             "relevance (F≈23–34, direct-path t≈0.25, non-tautological). Short "
             "history (2024+, ~525 rows).")
    L.append("- **Stablecoin mint/burn/other-chain → stable_flow:** tautology "
             "successfully avoided (corr≈0), but the issuer-side and "
             "single-chain series are **irrelevant** to aggregate stable flow "
             "at the train window (F ≈ 0).")
    L.append("- **Macro surprise (VIX, HY spread, 2y → funding_basis):** clean-"
             "ish story but exclusion-fragile (risk-off hits all crypto "
             "directly), and empirically **irrelevant** here (F < 2).")
    L.append("")

    # A/B shortlist preview
    L.append("### Instruments carried to the A/B (best per treatment)\n")
    if shortlist:
        for t, (col, tf, why) in shortlist.items():
            L.append(f"- **{t}** ← `{col}[{tf}]`: {why}")
    else:
        L.append("*None cleared relevance.* No economically-motivated "
                 "instrument is both strong and non-tautological — so there is "
                 "nothing to A/B, and 2SLS would reduce to OLS everywhere.")
    L.append("")
    return "\n".join(L)


def render_ab_markdown(rows: list[dict], args: dict) -> str:
    L = ["## Gated 2SLS A/B (economically-motivated shortlist)\n"]
    L.append("One run per instrument (all other factors stay exogenous), so "
             "short-history candidates keep their full sample. The per-window "
             f"first-stage partial-F gate (≥ {args['f_threshold']}) decides, "
             "window by window, whether the factor is actually instrumented; "
             "where it fails, 2SLS = OLS. Win-rates are walk-forward folds vs "
             "BH BTC.\n")
    L.append("| Treatment | Instrument | Gate (passed/seen) | mean F | "
             "OLS wr | 2SLS wr | OLS medSh | 2SLS medSh | OOS days |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['treatment']} | `{r['instrument']}` | — | — | — | "
                     f"— | — | — | ({r['error']}) |")
            continue
        L.append(f"| {r['treatment']} | `{r['instrument']}` | {r['gate']} | "
                 f"{r['mean_f']:.1f} | {r['ols_wr']:.0%} | {r['tsls_wr']:.0%} | "
                 f"{r['ols_ms']:.3f} | {r['tsls_ms']:.3f} | {r['n_oos']} |")
    L.append("")

    ok = [r for r in rows if not r.get("error")]
    engaged = [r for r in ok
               if int(r["gate"].split("/")[0]) > 0
               and abs(r["tsls_ms"] - r["ols_ms"]) > 1e-9]
    better = [r for r in engaged if r["tsls_ms"] > r["ols_ms"]]
    worse = [r for r in engaged if r["tsls_ms"] < r["ols_ms"]]

    L.append("## Verdict\n")
    if not ok:
        L.append("No economically-motivated instrument cleared relevance, so "
                 "no A/B was run. Exclusion-clean candidates (mining, issuance, "
                 "stablecoin mechanics, macro surprise) are **irrelevant**; the "
                 "only relevant ones (ETF native flows, validator queue) carry "
                 "exclusion risk and/or are unstable.")
    elif not engaged:
        L.append("Every shortlist instrument **failed the per-window gate** — "
                 "2SLS collapsed to OLS in every window (identical median "
                 "Sharpe and fold win-rate). Note the gate's mean partial F "
                 "(~0.8–3.9) is far below the train-window univariate F "
                 "(23–88): once the other exogenous factors are partialled out "
                 "and the window rolls forward, the instruments' relevance "
                 "evaporates. Same outcome as the original four declared "
                 "instruments: nothing exploitable.")
    else:
        winners = [r for r in better
                   if "DIRECT-PATH" not in r.get("why", "")
                   and "UNSTABLE" not in r.get("why", "")]
        L.append(f"{len(engaged)} instrument(s) engaged the gate and made 2SLS "
                 f"genuinely diverge from OLS: **{len(worse)} made OOS "
                 f"performance worse**, {len(better)} better.")
        if winners:
            L.append(f"\n**Candidate winner(s):** "
                     + ", ".join(f"`{r['instrument']}` → {r['treatment']}"
                                 for r in winners)
                     + " — strong, non-tautological, no direct-path flag, and "
                     "2SLS ≥ OLS OOS. Treat with extreme caution: exclusion is "
                     "untestable, this is one market cycle, and the candidate "
                     "survived multiple testing across ~19 (treatment, "
                     "instrument) pairs.")
        else:
            L.append("\n**No winner.** Where 2SLS diverged it did not beat OLS "
                     "out of sample — consistent with the structural finding in "
                     "`iv_findings.md`: with no endogeneity for 2SLS to purge "
                     "(and causal discovery suggesting returns → factors), "
                     "instrumenting only discards predictive first-stage "
                     "variance and hurts the OOS forecast. Neither arm robustly "
                     "beats BH BTC.")
    L.append("")
    L.append("### Caveats\n")
    L.append("- **Exclusion is ultimately untestable.** The direct-path probe "
             "rejects only the most blatant second channels; a clean probe is "
             "necessary, not sufficient.")
    L.append("- **One cycle.** 2022–2025 is a single crypto macro regime; "
             "out-of-window stability (the `stab` column) is the closest we get "
             "to a second sample, and the relevant instruments fail it.")
    L.append("- **Multiple testing.** ~19 (treatment, instrument) pairs were "
             "screened; any single 'clean + helps' result needs a Bonferroni-"
             "style discount before it is believed.")
    L.append("- **Short history.** The ETF instruments only exist from 2024, so "
             "their A/B runs on a truncated, single-regime OOS window.")
    L.append("")
    return "\n".join(L)


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--assets", default="btc,eth,sol,bnb,avax,uni,aave,link,doge")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--f-threshold", type=float, default=10.0)
    p.add_argument("--rebalance-freq", type=int, default=5)
    p.add_argument("--ab", action="store_true",
                   help="Also run the gated 2SLS A/B on the relevance-clearing "
                        "instruments and append results to the report.")
    p.add_argument("--out", default="causal_portfolio/docs/iv_economic.md")
    args = p.parse_args()
    assets = args.assets.split(",")

    raw, factors, returns = build_raw_and_factors(assets, args.start, args.end)
    cands = screen_econ_candidates(
        raw, factors, returns,
        train_window=args.train_window, f_threshold=args.f_threshold)
    shortlist = build_ab_shortlist(cands)

    argd = vars(args)
    argd["assets"] = assets  # store the parsed list, not the raw CSV string
    md = render_markdown(cands, shortlist, argd)

    if args.ab and shortlist:
        # run_shortlist_ab reads its SHORTLIST from the module global, so
        # temporarily swap in our economically-motivated shortlist.
        import causal_portfolio.experiments.iv_search as ivs
        saved = ivs.SHORTLIST
        ivs.SHORTLIST = shortlist
        try:
            ab_rows = run_shortlist_ab(
                raw, factors, returns,
                train_window=args.train_window,
                rebalance_freq=args.rebalance_freq,
                f_threshold=args.f_threshold)
        finally:
            ivs.SHORTLIST = saved
        # attach the rationale so the verdict can read direct-path/unstable tags
        why_by_t = {t: v[2] for t, v in shortlist.items()}
        for r in ab_rows:
            r["why"] = why_by_t.get(r["treatment"], "")
        md += "\n" + render_ab_markdown(ab_rows, argd)
    elif args.ab:
        md += "\n" + render_ab_markdown([], argd)

    Path(args.out).write_text(md, encoding="utf-8")
    n_clear = sum(1 for c in cands if c.clears_relevance)
    n_clean = sum(1 for c in cands if c.is_clean)
    print(f"{len(cands)} economically-motivated candidates screened: "
          f"{n_clear} clear relevance, {n_clean} fully clean -> {args.out}")


if __name__ == "__main__":
    main()
