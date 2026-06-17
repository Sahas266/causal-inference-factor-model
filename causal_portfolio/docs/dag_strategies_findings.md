# Simple DAG strategies — consolidated findings

A "simple DAG" here is the smallest causal trading model: **one state variable
→ an allocation decision → next-day return.** Risk-on holds the asset (BTC, an
alt, or the equal-weight basket); risk-off rotates into stables (0% return,
conservative). Every rule below is evaluated the same honest way, so results
are comparable across families.

## Method (the honesty bar)

- **Causal signals only** — trailing windows / rolling quantiles; never a
  threshold or ranking fit using the full sample.
- **One-day lag** — the decision uses data through day *t* and is applied to
  day *t+1*'s return. Switching cost (5bp one-way) on every position change.
- **Benchmarks** — buy-and-hold BTC (ann **+37%**, Sharpe **0.63**, maxDD
  **−77%**, Calmar **0.48**) and, for multi-asset rules, the daily 1/N
  equal-weight basket of 11 majors (ann **+85%**, Sharpe **1.05**, maxDD
  **−80%**, Calmar **1.07**).
- **Placebo gate** — any rule that beats its benchmark on Sharpe or Calmar is
  circular-shift placebo-tested: `p` = share of random-timed copies of the
  same signal that match/beat it. **Low p (<0.10) = the timing carries
  information; high p = the "edge" is just average exposure level.**
- Window: BTC/2021–2025 (1,825 days) unless a series starts later (funding
  2023-11; some on-chain later).

Source experiments (one module + doc each, all reuse
`experiments/regime_rotation.py`): `regime_rotation`, `cross_asset_rotation`,
`trend_breakout`, `macro_onchain_gates`, `mean_reversion_seasonality`,
`ensemble_dags`, `breadth_correlation_dags`, `onchain_flow_dags`,
`risk_targeting_dags`.

## Master results table

Sorted by Sharpe within family. `bench` = benchmark the row is measured
against (BTC or basket). `p` = placebo (— = not tested, didn't beat bench).

### A. Volatility / drawdown gates (BTC) — *the "rotate out when volatile" idea*
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| vix_off_70 | +17% | 0.48 | −41% | 0.42 | — |
| vol_target | +20% | 0.39 | −75% | 0.27 | — |
| dd_off_30 | +14% | 0.35 | −55% | 0.26 | — |
| vol_off_70 | +13% | 0.35 | −61% | 0.22 | — |
| dd_off_20 | +11% | 0.31 | −44% | 0.26 | — |
| vol_off_80 | +8% | 0.24 | −60% | 0.14 | — |

→ **All lose to BH BTC.** Vol/drawdown gates de-risk only by cutting average
exposure; none beats the benchmark, so none was even placebo-tested.

### B. Trend / breakout (BTC)
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| trend_40 (MA sweep) | ~+? | **1.09** | −49% | 0.83 | **0.01** |
| dual_confirm (px>100dMA AND 30d ret>0) | +28% | **0.95** | **−26%** | **1.10** | **0.04** |
| trend_125 (MA sweep) | +? | 0.93 | −33% | 1.04 | **0.06** |
| tsmom_30 (ROC sign) | +33% | 0.89 | −46% | 0.71 | 0.05* |
| trend_50 (px>50d MA) | +34% | 0.88 | −57% | 0.60 | **0.03** |
| trend_60 (MA sweep) | +? | 0.82 | −60% | 0.52 | **0.04** |
| trend_100 | +25% | 0.66 | −37% | 0.67 | 0.15 |
| donchian_55 / 20 | +23% | 0.62 | −49% | 0.46 | — |
| macd_sign | +20% | 0.52 | −52% | 0.39 | — |
| ma_cross_20_100 / 50_200 | +16% | 0.45 | −45% | 0.40 | — |
| trend_200 | +16% | 0.42 | −50% | 0.32 | — |

→ **The strongest family.** A contiguous MA band (40–60d, p=0.01–0.04) and the
2-of-2 `dual_confirm` filter beat BH on Sharpe *and* survive the placebo.
*tsmom_30's lookback sweep is a single-point spike (only N=30 passes) = overfit;
MA crossover, Donchian, MACD all fail.

### C. Cross-asset rotation (basket bench)
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| relmom_top3_30 | +96% | 1.08 | −82% | 1.17 | 0.16 |
| ew_basket (1/N) | +85% | 1.05 | −80% | 1.07 | — |
| inv_vol_20 | +72% | 0.93 | −79% | 0.91 | 0.99 |
| relmom_top3_90 | +71% | 0.85 | −87% | 0.81 | 0.61 |
| best_of_two_30 | +52% | 0.78 | −70% | 0.73 | 0.24 |
| relmom_top1_30/90 | +66% | 0.55 | −95% | 0.70 | — |
| dual_mom_90 | +30% | 0.28 | −96% | 0.31 | — |

→ **No rotation timing survives the placebo.** The entire lift over BTC is
*diversification* — the plain 1/N basket (Sharpe 1.05) does it at ~zero
turnover; rotation adds churn, not skill.

### D. Macro / on-chain gates (BTC)
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| curve_flatten_off (2s10s) | +38% | 1.05 | −38% | 0.99 | 0.04† |
| curve_flatten_off AND trend50 | +27% | 1.00 | −24% | 1.13 | 0.04† |
| funding_q_off | +35% | 0.63 | −77% | 0.45 | — |
| funding_neg_off | +17% | 0.59 | −35% | 0.48 | — |
| stablecoin_flow_on | +25% | 0.49 | −75% | 0.34 | — |
| rates_off | +19% | 0.48 | −58% | 0.32 | — |
| m2_liquidity_on | +11% | 0.23 | −61% | 0.17 | — |
| dollar_trend_off | +2% | 0.06 | −73% | 0.03 | — |
| curve_steepen_off | −4% | −0.08 | −81% | −0.04 | — |

→ The 2s10s curve gate beats BH+placebo but is **†flagged as a likely
small-sample artifact** (few independent macro regime turns; the helping sign
was chosen post-hoc). No macro *level* gate (dollar/rates/M2) or stablecoin
gate beats BH.

### E. On-chain demand/flow (BTC unless noted)
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| exch_flow (in−out z) | +40% | 0.74 | −69% | 0.58 | 0.09 |
| stbl_supply (ETH) | +53% | 0.78 | −65% | 0.82 | 0.26 |
| stbl_supply (BTC) | +33% | 0.64 | −69% | 0.47 | 0.39 |
| tvl_trend | +20% | 0.49 | −67% | 0.29 | — |
| addr_growth | +19% | 0.47 | −73% | 0.26 | — |
| fee_demand | +2% | 0.05 | −65% | 0.03 | — |

→ Only the BTC exchange-flow proxy marginally passes (p=0.09) at heavy
turnover; the headline-looking stablecoin-supply gate fails the placebo
(it's just being ~73% invested). No clean on-chain edge.

### F. Breadth / correlation regime (BTC + basket)
| Rule | Bench | Ann | Sharpe | MaxDD | Calmar | p |
|---|--|--:|--:|--:|--:|--:|
| corr_off_high | basket | +92% | **1.47** | −56% | **1.64** | **0.04**‡ |
| breadth_ma50_gt50 | basket | +53% | 1.08 | −39% | 1.35 | 0.14 |
| breadth_mom_rising | BTC | +25% | 0.79 | −46% | 0.54 | 0.08 |
| btcdom_alts_on_lead | BTC | +31% | 0.78 | −52% | 0.60 | 0.13 |
| disp_off_low | BTC | +19% | 0.75 | −35% | 0.54 | 0.12 |
| corr_off_low | basket | −9% | −0.17 | −89% | −0.10 | — |

→ **Risk-off when pairwise correlation is high** (crashes are correlated) beats
BH basket and survives the placebo — the most promising *new* single signal.
‡Caveat: breadth/correlation are derived from the same prices (partly
mechanical), and both signs were tested (inflates significance). Suggestive,
not validated.

### G. Ensemble / combination (BTC + basket)
| Rule | Bench | Ann | Sharpe | MaxDD | Calmar | p |
|---|--|--:|--:|--:|--:|--:|
| breadth_trend_basket | basket | +58% | **1.27** | −47% | 1.24 | 0.06 |
| trend_on_basket | basket | +67% | **1.22** | −47% | **1.43** | 0.08 |
| btctrend_on_basket | basket | +61% | 1.18 | −54% | 1.12 | 0.08 |
| trend_AVG_funding | BTC | +25% | 0.88 | −33% | 0.76 | 0.09 |
| trend_AND_funding | BTC | +19% | 0.81 | −26% | 0.71 | 0.13 |
| dual_confirm_basket | basket | +33% | 0.69 | −56% | 0.58 | 0.41 |
| vote_frac (3-signal) | BTC | +23% | 0.60 | −61% | 0.37 | — |

→ **Diversification × trend stacks; signal-voting does not.** Applying the
trend filter to the basket (trend_on_basket Sharpe 1.22, breadth_trend_basket
1.27) beats both BH basket and trend-on-BTC. But naive ensembles (voting,
dual-confirm-on-basket) *underperform their best component*, and funding adds
nothing on top of trend.

### H. Continuous risk-targeting (BTC)
| Rule | Ann | Sharpe | MaxDD | Calmar | p |
|---|--:|--:|--:|--:|--:|
| trend50_x_vol | +20% | 0.61 | −54% | 0.37 | — |
| trend_zscore | +17% | 0.60 | −42% | 0.40 | — |
| vol_target_ewma | +21% | 0.41 | −74% | 0.28 | — |
| downside_target | +17% | 0.35 | −75% | 0.22 | — |

→ **All lose to BH BTC.** With no leverage the exposure cap means vol-targeting
can only de-risk, which gives up return in a bull; continuous scaling ≈ binary
trend minus extra turnover.

### I. Mean-reversion / seasonality (BTC)
| Rule | Ann | Sharpe | MaxDD | Calmar | net Sharpe @20bp |
|---|--:|--:|--:|--:|--:|
| bollinger | +18% | 0.54 | −54% | 0.34 | 0.48 |
| rsi_dipbuyer | +8% | 0.52 | −18% | 0.44 | 0.43 |
| turn_of_month | +8% | 0.39 | −25% | 0.32 | 0.21 |
| rsi_reversal | +11% | 0.27 | −60% | 0.18 | 0.25 |
| st_reversal | +3% | 0.07 | −63% | 0.04 | 0.01 |
| dow_seasonality | +1% | 0.03 | −72% | 0.02 | −0.49 |

→ **A clean null.** Nothing beats BH even before costs; "buy the −2σ dip" gets
run over by trends; seasonality goes net-negative once realistic costs hit its
churn.

## Robust survivors (beat benchmark on Sharpe AND placebo p < 0.10)

| Rule | Bench | Sharpe (vs bench) | MaxDD (vs bench) | p | Trust |
|---|--|--:|--:|--:|--|
| breadth_trend_basket | basket 1.05 | **1.27** | −47% vs −80% | 0.06 | trend×diversification — credible |
| trend_on_basket | basket 1.05 | 1.22 | −47% vs −80% | 0.08 | same idea |
| corr_off_high | basket 1.05 | **1.47** | −56% vs −80% | 0.04 | suggestive (mechanical/2-sided) |
| dual_confirm | BTC 0.63 | 0.95 | **−26%** vs −77% | 0.04 | AND-filter, not swept — credible |
| trend_40–60 band | BTC 0.63 | 0.82–1.09 | ~−50% vs −77% | 0.01–0.04 | contiguous band — credible |
| curve_flatten_off | BTC 0.63 | 1.05 | −38% vs −77% | 0.04 | likely small-sample artifact |

The credible, non-mechanical survivors all reduce to **one idea: medium-term
price trend** (on BTC, or — better — on the diversified basket). It does not
raise return; it roughly halves max drawdown while keeping Sharpe at/above the
benchmark.

## Strategies that performed *comparably* to buy-and-hold BTC

"Comparable" = kept pace with BH BTC's **+37%/yr return** without a clearly
better risk profile — i.e. you could have just held BTC and done about as well.
These neither convincingly win nor lose; they are the **statistical ties**:

| Rule | Ann ret | Sharpe | MaxDD | Read vs BH BTC |
|---|--:|--:|--:|--|
| **trend_50** | +34% | 0.88 | −57% | same return, better Sharpe/DD — a *risk* tie-plus |
| **tsmom_30** | +33% | 0.89 | −46% | same return, but overfit lookback → treat as a tie |
| **funding_q_off** | +35% | 0.63 | −77% | **identical** Sharpe & DD — a pure tie (no edge) |
| **curve_flatten_off** | +38% | 1.05 | −38% | same return; Sharpe edge is a likely artifact → tie |
| **sig_trend50 (ensemble)** | +34% | 0.88 | −57% | = trend_50 (same rule) |
| **exch_flow (on-chain)** | +40% | 0.74 | −69% | ~same return, marginal Sharpe, heavy churn → tie |

And on the **basket** benchmark, the rules that merely *match* BH basket
(Sharpe ~1.05) rather than beat it: **inv_vol_20** (0.93), **relmom_top3_90**
(0.85) — comparable-to-slightly-worse, no timing edge.

**Takeaway on "comparable":** apart from the trend family's drawdown benefit,
most rules that look like they "kept up" with BTC did so by *being BTC most of
the time* (funding_q_off is 88% invested; its numbers are BH BTC to two
decimals). When a rule's return matches BH but its placebo `p` is high or
untested, the honest reading is **tie, not edge** — you are being paid the same
as holding BTC for taking on extra complexity and turnover.

## Synthesis

1. **One real idea survives, three ways:** medium-term **price trend** — as a
   binary BTC filter (40–60d MA, dual_confirm), and best of all applied to the
   **diversified basket** (trend_on_basket / breadth_trend_basket, Sharpe
   ~1.2–1.3). It is a *risk overlay* (halves drawdown), not a return enhancer.
2. **Diversification is the only free Sharpe:** the 1/N basket beats BH BTC
   (1.05 vs 0.63) with no timing at all.
3. **Everything else is exposure-reduction, overfit, or mechanical:** vol/
   drawdown gates, rotation timing, macro-level gates, on-chain flow gates,
   continuous vol-targeting, mean-reversion and seasonality all fail the
   placebo or the benchmark.
4. **`corr_off_high` is the one genuinely new lead** worth a forward test, with
   the caveat that it is partly mechanical and two-sided-selected.

## Caveats (apply to every row)

- **One market cycle.** 2021–2025 is a single bull→bear→recovery; the trend/
  correlation winners look good largely by truncating the *one* 2022 bear.
  The placebo validates timing *within* this sample — it cannot validate
  against regime change.
- **Multiple testing.** ~60 rule/lookback variants were tried; the best
  in-sample numbers are upward-biased. The placebo guards against timing luck
  on a *fixed* rule, not against having picked the lucky rule/lookback.
- **Costs.** 5bp one-way is optimistic for smaller majors; high-churn rules
  (seasonality, on-chain flow, voting) degrade materially at 10–20bp.
- **Stables at 0%.** A real stable yield lifts every risk-off rule slightly but
  does not change the rankings vs BH.
- **No leverage.** Exposure capped at 1, so every overlay can only de-risk —
  structurally disadvantaged in a bull market on raw return.

**Bottom line:** of all the simple DAGs, a **medium-term trend filter on a
diversified basket** is the only thing that beats the benchmarks on a
risk-adjusted basis and survives the placebo with a non-mechanical signal.
Buy-and-hold BTC remains unbeaten on raw direction; the wins are drawdown
control, and they are one-cycle evidence to paper-trade forward, not deploy on
faith.
