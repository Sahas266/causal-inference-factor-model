# What This Project Is Doing — A Detailed Explanation

## The One-Sentence Version

We're building a system that figures out **what actually causes crypto prices to move** (not just what correlates with them), and then uses those causal relationships to make better trading decisions across 26 tokens.

---

## The Problem: Correlation is Not Causation

Most quant models in crypto work like this: "When gas fees go up, ETH price tends to go down — let's trade on that." The problem is that this is a **correlation**. Maybe gas fees go up *because* prices are crashing (everyone is panic-selling, clogging the network). If you trade on "high gas = short ETH," you're not exploiting a real signal — you're chasing a side effect.

This matters because:
- Correlations break down. They work until they don't, and when they break it's usually during the exact market conditions where you need your model most (crashes, squeezes, depegs).
- You can't simulate interventions. If you can't tell cause from effect, you can't answer "what happens to ETH if stablecoin supply suddenly drops 10%?" — which is the kind of question a portfolio manager actually needs answered.
- You overfit. Without causal structure, your model memorizes historical accidents instead of learning durable mechanisms.

**Causal inference** solves this. Instead of asking "what moved together in the past?", it asks "if I surgically changed X, what would happen to Y?" That's what a Structural Causal Model (SCM) does.

---

## The Approach: Causal PDE-Control Models (CPCM)

The [CPCM paper](https://arxiv.org/pdf/2509.09585v2) proposes a three-layer framework:

1. **SCM Layer** (Phase 2.1 — what we're building now): Define the causal graph. Which things cause which other things? How do we prove it statistically?
2. **Filtering Layer** (Phase 2.2 — future): Use Kalman/particle filters to estimate hidden state variables (things we can't directly observe, like "true market sentiment") from noisy data.
3. **Control Layer** (Phase 2.3 — future): Given the causal model and the estimated state, compute optimal portfolio weights that maximize risk-adjusted returns.

The paper applied this to US equities. We're applying it to DeFi/crypto, which is harder (more noise, less history, wilder dynamics) but also has richer on-chain data that traditional markets don't have.

---

## Phase 2.1: What We're Actually Building

### The Causal Graph (DAG)

A **Directed Acyclic Graph (DAG)** is a diagram where arrows mean "causes." Our DAG has three types of nodes:

#### 1. Global Factors (F) — 7 forces that move the entire crypto market

These are macro-level on-chain signals. Every token's return is affected by all of them:

| Factor | What it is | Why it causes price moves |
|--------|-----------|--------------------------|
| **LiqFlow** | Net LP inflows/outflows across AMMs (Uniswap, Curve, etc.) | When liquidity providers pull money out, pools get thinner, slippage increases, and prices become more volatile. When they add money, the opposite. |
| **StableFlow** | Net minting/burning of USDC + USDT + USDe | Stablecoins are the "cash" of crypto. When $500M of new USDC gets minted, that's $500M of fresh buying power entering the system. When stables get burned, liquidity is leaving. This is arguably the most important systemic indicator. |
| **FundingBasis** | The gap between perp funding rates and spot returns | When perps trade at a premium (positive funding), longs are paying shorts. Extreme funding = crowded positioning = likely mean reversion. This is one of the strongest short-term signals in crypto. *(We don't have this data on free tier yet.)* |
| **ChainCongestion** | Gas fees, block utilization | High congestion = high demand for blockspace = expensive to execute trades. This reduces arbitrage efficiency, which increases price dislocations and volatility. |
| **StakingYield** | Real staking APR (e.g., ETH ~3.5%) | The "risk-free rate" of crypto. When staking yield rises, it raises the opportunity cost of holding non-yielding tokens, repricing everything. |
| **MEVPressure** | MEV (Maximal Extractable Value) revenue volatility | When MEV bots are extracting a lot, it means orderflow is chaotic — sandwiches, frontrunning, backrunning. This widens spreads and increases short-term variance. |
| **CEXDEXFlow** | Net flow from centralized exchanges (Binance, Coinbase) to on-chain | Large outflows from CEXes to wallets = accumulation (bullish). Large inflows to CEXes = distribution/selling (bearish). |

#### 2. Asset-Specific Covariates (Xi) — per-token fundamentals

Each token also has its own unique drivers:

| Covariate | What it is | Example |
|-----------|-----------|---------|
| **Whale concentration** | Are the top 10 wallets accumulating or distributing? | If Vitalik sells, that's different from a random wallet selling. |
| **Protocol revenue** | Fees the protocol earns (swap fees for Uniswap, borrow interest for Aave) | Higher revenue = more valuable protocol = should support token price. |
| **Emissions/inflation** | How many new tokens are being created (vesting, mining, staking rewards) | More emissions = more sell pressure. Bitcoin's halving is the most famous example. |
| **Chain activity** | Transaction count, active addresses | More users = more demand for the token (gas, staking, governance). |

#### 3. Unobserved Shocks (εi) — things we can't predict

Protocol exploits, chain halts, oracle failures, liquidation cascades. We can't model these directly, but we can detect them after the fact (e.g., via liquidation volume spikes) and account for their impact.

#### The Arrows (Causal Edges)

The DAG says:
- Every global factor F → every asset's return (all 7 factors affect all 26 tokens)
- Every macro factor (interest rates, inflation, VIX, dollar index) → every asset's return (with a 1-day lag)
- Each asset's own covariates Xi → that asset's return only
- Some factors affect some covariates: e.g., StableFlow affects whale behavior, ChainCongestion affects protocol volume
- Unobserved shocks → returns

### The Identification Problem

Drawing arrows on a graph is easy. The hard part is **proving** that the arrow is real — that the relationship is truly causal and not just a spurious correlation driven by some hidden third variable.

Example: We claim "CEX outflows cause price increases." But maybe both are caused by a third thing — say, a rumor on Twitter. People hear the rumor, withdraw from CEXes, and also bid up the price. The CEX outflow didn't cause the price increase; the rumor caused both.

To handle this, we use two techniques:

#### Backdoor Criterion
If we can identify and statistically control for all the confounders (common causes), we can isolate the true causal effect. The DAG tells us exactly which variables to control for — that's the "minimal adjustment set."

**d-separation** is the graph algorithm that figures this out. Given two nodes X and Y and a set of variables Z we're controlling for, d-separation tells us whether X and Y are still connected through any unblocked causal path. If not, they're independent given Z. The Bayes-Ball algorithm implements this efficiently.

#### Instrumental Variables (IVs)
Sometimes you can't control for everything (especially unobserved confounders). IVs are a workaround. An instrument Z is a variable that:
1. **Affects the treatment** (Z → X): it pushes X around
2. **Only affects the outcome through the treatment** (Z → X → Y, but not Z → Y directly): it has no direct effect on Y except through X

Example: We want to know if LP outflows cause price drops (or if it's the reverse — prices drop, then LPs flee). We use **gas fee spikes** as an instrument. Gas spikes make it expensive to rebalance, forcing some LPs out — but gas spikes don't directly cause price drops (they're a cost friction, not a valuation signal). So gas spikes push LP outflows without directly pushing prices, letting us isolate the causal effect of LP outflows on price.

Our IVs:
| What we want to prove | The instrument | Why it works |
|----------------------|---------------|-------------|
| LP outflow → price drop | Gas fee spikes | Gas makes LPs exit but doesn't directly move price |
| Funding rate → returns | Liquidation volume spikes | Liquidations shock funding but aren't directly priced |
| Stablecoin issuance → market beta | Lagged supply changes | Yesterday's mint doesn't directly cause today's price (temporal separation) |
| Protocol revenue → token return | Fee regime changes (>2σ events) | Protocol fee changes are governance decisions, not market-driven |

### The Estimation

Once we know which effects are causal (via the DAG + identification), we estimate their magnitude using regression:

#### OLS (Ordinary Least Squares) — the baseline
Standard linear regression: `return = β₁·LiqFlow + β₂·StableFlow + ... + β₇·CEXDEXFlow + controls + ε`

This gives us coefficients (β) that say "a 1-unit increase in LiqFlow is associated with a β₁ change in returns." But OLS is biased when there's reverse causality or omitted confounders (which there always is in finance).

#### 2SLS (Two-Stage Least Squares) — the causal estimator
This uses the instrumental variables to remove the bias:

- **Stage 1**: Regress the treatment (e.g., LiqFlow) on the instrument (e.g., gas spikes) plus controls. This gives us the "exogenous part" of LiqFlow — the variation in LiqFlow that's driven purely by gas costs, not by prices or anything else.
- **Stage 2**: Regress returns on this cleaned-up version of LiqFlow. The resulting coefficient is the **causal effect**.

If the 2SLS coefficient is very different from the OLS coefficient, that's evidence that the OLS estimate was biased (usually by reverse causality). The Hausman test formally checks this.

#### Panel Structure
We run this for each of the 26 tokens individually (26 separate regressions). Later, we can pool all tokens together with fixed effects for more statistical power.

---

## The Software Architecture (What Each Rust Crate Does)

### `cpcm-core` — The Graph Engine
Pure algorithms, no data dependencies. Builds the DAG, runs d-separation, computes backdoor adjustment sets, validates instruments. This is the "brain" — it encodes the causal theory and checks whether our statistical strategy is valid.

Think of it as a theorem prover for causal claims. You assert "LiqFlow causes ETH returns" and it tells you "to prove that, you need to control for these 5 variables" or "you need an instrument that satisfies these conditions."

### `cpcm-data` — The Data Pipeline
Connects to Supabase (where Phase 1 stored all 430K rows), fetches the metrics we need, and pivots them into a wide table where each row is a date and each column is a metric (like `eth_PriceUSD`, `btc_TxCnt`, `usdc_SplyCur`). Caches locally to Parquet so we don't re-fetch every run.

### `cpcm-factors` — The Feature Engineering
Transforms raw database metrics into the 7 CPCM factors, asset covariates, and instrumental variables. For example:
- Raw data: `usdc_SplyCur = 32.5B`, `usdt_SplyCur = 112.3B`, `usde_stablecoin_circulating_usd = 3.2B`
- Factor output: `StableFlow = diff(32.5 + 112.3 + 3.2) = +0.4B` (net $400M stablecoins minted today)

Each factor has a `FactorBuilder` trait implementation that declares what raw metrics it needs and how to compute the factor from them.

### `cpcm-estimate` — The Statistics Engine
Implements OLS and 2SLS from scratch using `nalgebra` (a linear algebra library). Also computes diagnostics:
- **First-stage F-statistic** > 10: are our instruments strong enough? (Staiger-Stock rule)
- **Sargan test**: if we have more instruments than treatments, are they all valid? (overidentification)
- **Hausman test**: is OLS biased? (compares OLS vs 2SLS coefficients)
- **Durbin-Watson**: is there serial correlation in residuals? (common in time series)
- **VIF**: are our factors too correlated with each other? (multicollinearity)

### `cpcm-cli` — The Runner
Wires everything together into a command-line tool:
```
cpcm-cli run        # full pipeline: fetch data → build factors → identify → estimate → report
cpcm-cli coverage   # "which coins have which data?" report
cpcm-cli graph      # print the DAG and d-separation results
cpcm-cli factors    # compute and inspect factor time series
cpcm-cli estimate   # run OLS + 2SLS and print coefficients
```

---

## What We Get at the End

After running the full pipeline, we'll have:

1. **A causal DAG** encoding our theory of what drives crypto returns, validated for identifiability.
2. **Causal effect estimates** for each factor on each token. E.g., "A $1B increase in StableFlow causes a +0.8% return for ETH (p < 0.01), a +1.2% return for SOL (p < 0.05), and has no significant effect on DOGE (p = 0.34)."
3. **Diagnostic reports** telling us which estimates are trustworthy (strong instruments, passing specification tests) and which are suspect.
4. **A comparison of OLS vs 2SLS** showing where naive correlation-based estimates are biased and by how much.

This is the foundation for Phases 2.2 (state filtering) and 2.3 (portfolio optimization). Once we know the causal coefficients, we can feed them into a Kalman filter to estimate the current hidden state, and then into an optimizer to compute portfolio weights. But that's future work — Phase 2.1 is about getting the causal graph and estimation right.

---

## Why Rust?

- **Performance**: Matrix operations on 430K+ rows with 26 assets, 7 factors, rolling windows, and bootstrap confidence intervals. Rust's zero-cost abstractions and lack of GC pauses make this fast.
- **Correctness**: Rust's type system prevents entire classes of bugs (null references, data races, type mismatches). When you're computing causal effects that drive trading decisions, correctness matters more than in most software.
- **Deployment**: The eventual live trading system needs to run continuously with low latency. A compiled Rust binary is easier to deploy and monitor than a Python script with a fragile dependency chain.
- **Ecosystem**: `nalgebra` (linear algebra), `polars` (DataFrames), `petgraph` (graphs), `reqwest` (HTTP) — all mature, well-maintained crates that cover our needs.

---

## Current Status (2026-03-16)

The system is **live and producing results** across 23 crypto assets:

- **~430K rows** from Supabase → **1,827 × 250+ panel** → **23 assets estimated**
- **All 7 factors z-scored** — coefficients represent effect per 1-standard-deviation change
- **6/7 global factors active** (only funding_basis missing — no free perp funding rate source)
- **7 macro factors active** in estimation: VIX (vixcls), fed funds (dff), 10Y yield (dgs10), yield curve (t10y2y), CPI (cpiaucsl), M2 (m2sl), dollar index (dtwexbgs)
- **MEV pressure fixed** — uses base fee volatility as proxy (significant for AAVE)
- **vixcls** (VIX) is the most broadly significant factor (10/23 assets)
- **liq_flow** dominant for shorter-history assets (14/23 significant)
- 2SLS ran for all assets — Hausman tests confirm OLS consistency
- 61 unit tests pass, results exported to JSON + CSV (338 rows), panel cached to Parquet

Run it yourself: `cargo run -p cpcm-cli -- run` (needs `SUPABASE_KEY` in env)

---

## What's Missing (Known Gaps)

| Gap | Impact | Workaround |
|-----|--------|-----------|
| **Funding rates** (perp funding - spot) | Can't compute FundingBasis factor (1/7 missing) | Dropped from regression automatically. Add Hyperliquid API later. |
| **CoinGecko full history** (need Pro key) | 14 assets only have ~365 obs (high R² may be spurious) | CoinMetrics covers 9 assets back to 2021. |
| **Whale concentration** (ETH only) | Covariate NaN for non-ETH assets | Dropped from regression. Create Dune queries for other tokens. |
| **VIF multicollinearity** | cpiaucsl & m2sl VIF 50-75 — inflated SEs | Both kept; consider dropping one or PCA. |
| **POL price data** | No price on free tier → NaN returns, R²=0 | Included but effectively empty. |
