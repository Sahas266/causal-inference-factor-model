# CPCM Phase 2.1 — Causal SCM Layer in Rust

## Context
Phase 1 (data collection) is complete: ~430,300 rows in Supabase across 26 coins, 7 providers, 30 macro series. The existing Python `causal_portfolio/` is a skeleton using DoWhy/EconML with empty CSV data files. We rebuild the causal inference layer in Rust for performance and correctness, starting with Phase 2.1: **SCM Layer — Causal Graph for DeFi Asset Returns**.

---

## Cargo Workspace Layout

```
causal_model/                         # at repo root
  Cargo.toml                          # workspace manifest
  config.toml                         # runtime config (Supabase creds, asset list)
  crates/
    cpcm-core/                        # DAG, d-separation, identification (pure, no I/O)
      src/ { lib.rs, dag.rs, dsep.rs, identify.rs, types.rs }
    cpcm-data/                        # Supabase PostgREST client, DataFrame pivot
      src/ { lib.rs, client.rs, queries.rs, frame.rs, types.rs }
    cpcm-factors/                     # raw metrics → 7 CPCM factors + covariates + IVs
      src/ { lib.rs, global_factors.rs, asset_covariates.rs, instruments.rs, returns.rs, registry.rs }
    cpcm-estimate/                    # OLS, 2SLS, diagnostics (nalgebra)
      src/ { lib.rs, ols.rs, tsls.rs, diagnostics.rs, panel.rs, types.rs }
    cpcm-cli/                         # binary: CLI runner
      src/ { main.rs, config.rs, pipeline.rs, output.rs }
```

**Rationale**: Each crate has a single responsibility. `cpcm-core` is pure graph algorithms (zero I/O). `cpcm-data` owns all Supabase access. `cpcm-factors` maps raw metrics to CPCM factors. `cpcm-estimate` is pure numerical linear algebra. `cpcm-cli` wires them together.

---

## Step 1: `cpcm-core` — Causal DAG Engine

### Node & Edge Types
```rust
enum NodeKind { GlobalFactor, AssetCovariate, AssetReturn, Instrument, UnobservedShock, MacroFactor }
enum EdgeKind { Causal, Instrumental, Confounded }
struct Node { id: NodeId, name: String, kind: NodeKind, asset: Option<String> }
struct Edge { from: NodeId, to: NodeId, kind: EdgeKind, lag: i32 }
```

### CausalDag (wraps `petgraph::DiGraph<Node, Edge>`)
- `add_node()`, `add_edge()`, `parents()`, `children()`, `ancestors()`, `descendants()`
- `is_dag()` — topological sort check
- `d_separated(x, y, conditioning_set)` — Bayes-Ball algorithm (Shachter 1998)
- `backdoor_criterion(treatment, outcome) -> Option<Vec<NodeId>>` — minimal adjustment set
- `valid_instrument(z, treatment, outcome)` — relevance + exclusion via d-sep

### D-Separation Algorithm (Bayes-Ball)
1. Mark all nodes in conditioning set Z as "observed"
2. From source X, run reachability search respecting three junction rules:
   - Chain (A→B→C): blocked if B in Z
   - Fork (A←B→C): blocked if B in Z
   - Collider (A→B←C): blocked if B NOT in Z and no descendant of B in Z
3. If Y is unreachable, X and Y are d-separated given Z

### CPCM DAG Builder (`build_cpcm_dag(assets)`)
Hardcodes the DAG from the paper:
- **7 Global Factors** → each `{asset}_return` (contemporaneous edges)
- **7 Macro Factors** (DFF, DGS10, VIXCLS, T10Y2Y, CPIAUCSL, M2SL, DTWEXBGS) → each `{asset}_return` (lag=1)
- **4 Per-asset Covariates** (whale_conc, protocol_rev, emissions, chain_activity) → own `{asset}_return`
- **F→Xi interactions**: stable_flow→whale_conc, chain_congestion→protocol_rev, staking_yield→emissions
- **Unobserved shocks** εi → `{asset}_return`

```rust
pub fn build_cpcm_dag(assets: &[&str]) -> CausalDag {
    let mut dag = CausalDag::new();

    // 7 Global Factors
    let factors = ["liq_flow", "stable_flow", "funding_basis",
                   "chain_congestion", "staking_yield", "mev_pressure", "cex_dex_flow"];
    for f in &factors {
        dag.add_node(f, GlobalFactor, None);
    }

    // Macro factors (FRED)
    let macro_factors = ["dff", "dgs10", "vixcls", "t10y2y", "cpiaucsl", "m2sl", "dtwexbgs"];
    for m in &macro_factors {
        dag.add_node(m, MacroFactor, None);
    }

    // Per-asset nodes
    for asset in assets {
        dag.add_node(&format!("{}_return", asset), AssetReturn, Some(asset));

        for cov in &["whale_conc", "protocol_rev", "emissions", "chain_activity"] {
            let name = format!("{}_{}", asset, cov);
            dag.add_node(&name, AssetCovariate, Some(asset));
            dag.add_edge(&name, &format!("{}_return", asset), Causal, 0);
        }

        // All global factors -> asset return
        for f in &factors {
            dag.add_edge(f, &format!("{}_return", asset), Causal, 0);
        }

        // All macro factors -> asset return (lagged)
        for m in &macro_factors {
            dag.add_edge(m, &format!("{}_return", asset), Causal, 1);
        }

        // F -> Xi interactions
        dag.add_edge("stable_flow", &format!("{}_whale_conc", asset), Causal, 0);
        dag.add_edge("chain_congestion", &format!("{}_protocol_rev", asset), Causal, 0);
        dag.add_edge("staking_yield", &format!("{}_emissions", asset), Causal, 0);

        // Unobserved shock
        let shock = format!("{}_shock", asset);
        dag.add_node(&shock, UnobservedShock, Some(asset));
        dag.add_edge(&shock, &format!("{}_return", asset), Causal, 0);
    }

    dag
}
```

---

## Step 2: `cpcm-data` — Supabase Data Layer

### SupabaseClient
- Auth: `apikey` + `Authorization: Bearer {key}` headers against `{url}/rest/v1/`
- Pagination: `Range` header (0-999, 1000-1999, ...), loop until response count < 1000
- Queries the `asset_metrics_best` view (deduplicated, best-provider-per-metric)
- Filters: `asset=eq.eth`, `metric=in.(PriceUSD,TxCnt)`, `time=gte.2021-01-01`

```rust
pub struct SupabaseClient {
    http: reqwest::Client,
    base_url: String,      // e.g. "https://jnulpcqpftnwvknwuqpa.supabase.co"
    api_key: String,       // service_role key
}

impl SupabaseClient {
    // Paginated fetch of asset_metrics rows
    async fn fetch_metrics(
        &self,
        assets: &[&str],
        metrics: &[&str],
        start: NaiveDate,
        end: NaiveDate,
        use_best_view: bool,
    ) -> Result<Vec<AssetMetricRow>>

    // Single convenience: fetch one metric for one asset
    async fn fetch_single(
        &self, asset: &str, metric: &str,
        start: NaiveDate, end: NaiveDate,
    ) -> Result<Vec<(NaiveDate, f64)>>
}
```

### Row Type (mirrors `asset_metrics` table)
```rust
#[derive(Debug, Deserialize)]
pub struct AssetMetricRow {
    pub provider: String,
    pub provider_priority: i32,
    pub asset: String,
    pub metric: String,
    pub time: String,           // ISO-8601, parse to NaiveDate
    pub value: Option<String>,  // stored as text in Supabase
    pub frequency: String,
    pub metadata: Option<serde_json::Value>,
}
```

### DataFrame Pivot
`pivot_to_panel(rows) -> DataFrame` — pivots raw rows into wide format:
- Index = `date` (daily, NaiveDate)
- Columns = `{asset}_{metric}` (e.g., `eth_PriceUSD`, `btc_TxCnt`)
- Values = f64 (parsed from text)
- Forward-fill for irregular metrics (TVL), NaN for missing dates
- Alignment to daily frequency (all downstream estimation assumes daily panels)

### Parquet Caching
~430K rows = ~430 paginated requests. Cache the DataFrame locally to skip re-fetching:
```toml
[data]
cache_path = "cache/panel.parquet"
cache_ttl_hours = 24
```

---

## Step 3: `cpcm-factors` — Factor Construction

### 7 Global Factors (mapping raw metrics → CPCM factors)

| Factor | Computation | Supabase Sources |
|--------|-------------|------------------|
| **LiqFlow** | `diff(sum(tvl_usd))` across protocols | `defillama/*/tvl_usd` + `dune/eth/lp_net_flow_usd` |
| **StableFlow** | `diff(usdc + usdt + usde supply)` | `coinmetrics/usdc,usdt/SplyCur` + `defillama/usde/stablecoin_circulating_usd` |
| **FundingBasis** | NaN placeholder (no free source) | GAP — mark as missing column |
| **ChainCongestion** | Weighted avg gas/block utilization | `dune/eth/avg_gas_price_gwei` + `coinmetrics/*/FeeTotNtv` |
| **StakingYield** | ETH staking APR | `dune/eth/staking_apr` |
| **MEVPressure** | MEV revenue level/variance | `dune/eth/mev_revenue_eth` |
| **CEXDEXFlow** | Net exchange flow | `dune/eth/cex_inflow-outflow` + `coinmetrics/*/FlowInExNtv-FlowOutExNtv` |

### FactorBuilder Trait
```rust
pub trait FactorBuilder {
    /// Name of the factor (e.g., "liq_flow")
    fn name(&self) -> &str;

    /// List of (asset, metric) pairs needed from Supabase
    fn required_metrics(&self) -> Vec<(&str, &str)>;

    /// Compute the factor series from a panel DataFrame
    fn compute(&self, panel: &DataFrame) -> Result<Series>;
}
```

Each factor implements this trait. The `FactorRegistry` collects all builders and runs them, producing a DataFrame with columns `[date, liq_flow, stable_flow, ..., cex_dex_flow]`.

### Per-asset Covariates

| Covariate | Computation | Source |
|-----------|-------------|--------|
| `whale_conc` | Raw value from Dune | `dune/eth/top_10_pct_balance` (ETH only) |
| `protocol_rev` | DefiLlama fees | `defillama/*/fees_usd` |
| `emissions` | CoinMetrics issuance | `coinmetrics/*/IssTotNtv` |
| `chain_activity` | Composite: TxCnt + AdrActCnt | `coinmetrics/*/TxCnt`, `coinmetrics/*/AdrActCnt` |

### Instrumental Variables

| IV | Instruments For | Source | Computation |
|----|-----------------|--------|-------------|
| `gas_spike` | LiqFlow→return | `dune/eth/avg_gas_price_gwei` | z-score of gas diff > 2σ |
| `liquidation_level` | FundingBasis→return | `dune/eth/liquidation_volume_usd` | Volume spikes |
| `stablecoin_mint` | StableFlow→return | `coinmetrics/usdc,usdt/SplyCur` | Lagged supply diff |
| `protocol_event` | ProtocolRev→return | `defillama/*/fees_usd` | Fee regime change (diff > 2σ) |

Key constraint: IVs must be lagged by ≥1 day relative to outcome to avoid simultaneity.

### InstrumentBuilder Trait
```rust
pub struct InstrumentDef {
    pub name: String,
    pub instruments_for: String,  // treatment factor name
    pub target: String,           // outcome ("{asset}_return")
    pub lag: i32,                 // must be >= 1
}

pub trait InstrumentBuilder {
    fn name(&self) -> &str;
    fn def(&self) -> &InstrumentDef;
    fn required_metrics(&self) -> Vec<(&str, &str)>;
    fn compute(&self, panel: &DataFrame) -> Result<Series>;
}
```

### Return Computation
```rust
pub fn compute_log_returns(prices: &Series) -> Series {
    // ln(P_t / P_{t-1})
    let shifted = prices.shift(1);
    (prices / &shifted).ln()
}
```
Source: `coinmetrics/{asset}/PriceUSD` (priority 1) → `coingecko/{asset}/price_usd` fallback.

### Missing Data Handling
- Factors that aggregate across assets (LiqFlow, StableFlow) sum whatever is available
- Per-asset covariates use NaN for missing coins
- Estimation layer uses complete-case rows or handles NaN explicitly
- `DataCoverage` report struct lists which factors/covariates have data for which assets and date ranges

---

## Step 4: `cpcm-estimate` — Causal Estimation

### OLS (nalgebra)
```rust
pub fn ols(y: &DVector<f64>, x: &DMatrix<f64>) -> OlsResult {
    // β = (X'X)^{-1} X'y
    let xtx = x.transpose() * x;
    let xty = x.transpose() * y;
    let beta = xtx.try_inverse()?.unwrap() * &xty;

    // Residuals, R², standard errors
    let y_hat = x * &beta;
    let resid = y - &y_hat;
    let sse = resid.dot(&resid);
    let n = y.len() as f64;
    let k = x.ncols() as f64;
    let sigma2 = sse / (n - k);
    let var_beta = sigma2 * xtx.try_inverse().unwrap();
    let se = DVector::from_fn(beta.len(), |i, _| var_beta[(i, i)].sqrt());
    // ...
}

pub struct OlsResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residuals: Vec<f64>,
    pub n_obs: usize,
    pub feature_names: Vec<String>,
}
```

### Two-Stage Least Squares (2SLS)
```rust
pub fn tsls(
    y: &DVector<f64>,        // outcome (n × 1)
    x_endog: &DMatrix<f64>,  // endogenous treatment (n × k1)
    x_exog: &DMatrix<f64>,   // exogenous controls (n × k2), includes intercept
    z: &DMatrix<f64>,        // instruments (n × m), m >= k1
) -> TslsResult {
    // Stage 1: Regress each endogenous var on [Z, X_exog]
    let z_full = hstack(&[z, x_exog]);
    let x_hat = project_onto(x_endog, &z_full);

    // Stage 2: Regress y on [X̂, X_exog]
    let x_second = hstack(&[&x_hat, x_exog]);
    let beta = ols_core(y, &x_second);

    // Correct standard errors using original X (not X̂)
    let x_original = hstack(&[x_endog, x_exog]);
    let resid = y - &(x_original * &beta);
    // ... 2SLS-corrected variance
}

pub struct TslsResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,     // 2SLS-corrected
    pub first_stage_f: Vec<f64>,  // one per endogenous variable
    pub sargan_stat: Option<f64>, // overidentification test (if m > k1)
    pub sargan_p: Option<f64>,
    pub hausman_stat: f64,        // OLS vs 2SLS comparison
    pub hausman_p: f64,
    pub n_obs: usize,
    pub feature_names: Vec<String>,
}
```

### IV Quality Pre-checks
```rust
pub struct IvDiagnostics {
    pub first_stage_f_stat: f64,      // Must be > 10 (Staiger-Stock rule)
    pub partial_r_squared: f64,       // First-stage partial R²
    pub correlation_with_outcome: f64, // Should be ~0 conditional on treatment
}
```

### Panel Modes
```rust
pub enum PanelMode {
    PerAsset,              // 26 separate regressions (start here)
    Pooled,                // stack all assets, add asset fixed effects
    PooledWithInteraction, // pooled + factor × asset interaction terms
}

pub fn run_panel_estimation(
    panel: &DataFrame,
    dag: &CausalDag,
    factors: &[String],
    assets: &[String],
    instruments: &HashMap<String, String>,
    mode: PanelMode,
) -> Vec<EstimationResult>
```

### Diagnostics
```rust
pub struct Diagnostics {
    pub durbin_watson: f64,           // serial correlation
    pub breusch_pagan: (f64, f64),    // heteroskedasticity (stat, p-value)
    pub jarque_bera: (f64, f64),      // normality of residuals
    pub vif: Vec<(String, f64)>,      // variance inflation factors
}
```

---

## Step 5: `cpcm-cli` — Binary

### Configuration (TOML)
```toml
[supabase]
url = "https://jnulpcqpftnwvknwuqpa.supabase.co"
key_env = "SUPABASE_KEY"

[data]
start_date = "2021-01-01"
end_date = "2026-01-01"
cache_path = "cache/panel.parquet"
cache_ttl_hours = 24
use_best_view = true

[assets]
include = [
    "btc", "eth", "bnb", "sol", "avax", "xrp", "pol",
    "hype", "tao", "wlfi",
    "uni", "aave", "crv", "pendle", "morpho", "aero", "link", "ena", "jup",
    "zec", "pepe", "shib", "doge",
]

[estimation]
mode = "per_asset"
run_ols = true
run_2sls = true
confidence_level = 0.95
```

### CLI Commands
```
cpcm-cli run              # full pipeline: load → factor → identify → estimate
cpcm-cli load              # just fetch data and cache
cpcm-cli factors           # compute factors from cached data, print summary
cpcm-cli graph             # print DAG structure, d-separation checks
cpcm-cli estimate          # run estimation from cached factors
cpcm-cli coverage          # print data coverage report
cpcm-cli export --format json  # export results
```

### Pipeline Orchestration
```rust
async fn run_pipeline(config: &Config) -> Result<PipelineResult> {
    // 1. Load data
    let client = SupabaseClient::new(&config.supabase)?;
    let panel = load_or_cache(&client, &config.data).await?;

    // 2. Compute returns
    let returns = compute_all_returns(&panel, &config.assets.include)?;

    // 3. Compute factors
    let registry = FactorRegistry::default();
    let factors = registry.compute_all(&panel)?;
    let covariates = compute_all_covariates(&panel, &config.assets.include)?;

    // 4. Compute instruments
    let instruments = InstrumentRegistry::default().compute_all(&panel)?;

    // 5. Build DAG
    let dag = build_cpcm_dag(&config.assets.include);
    let id_report = identify_all_effects(&dag, &factors, &returns)?;

    // 6. Estimate
    let results = run_panel_estimation(
        &merge_all(&[&returns, &factors, &covariates, &instruments]),
        &dag, &factor_names, &config.assets.include,
        &instrument_map, config.estimation.mode.into(),
    )?;

    // 7. Diagnostics
    let diagnostics = compute_diagnostics(&results)?;

    Ok(PipelineResult { id_report, results, diagnostics })
}
```

---

## Dependencies

| Crate | Key Dependencies |
|-------|-----------------|
| `cpcm-core` | petgraph 0.6, serde 1 |
| `cpcm-data` | reqwest 0.12 (rustls-tls), tokio 1, polars 0.46 (lazy, parquet, temporal), chrono 0.4, dotenvy 0.15, serde_json 1 |
| `cpcm-factors` | polars 0.46, cpcm-data |
| `cpcm-estimate` | nalgebra 0.33, statrs 0.17 |
| `cpcm-cli` | clap 4 (derive), toml 0.8, tracing 0.1, tracing-subscriber 0.3, all cpcm-* crates |

**Dev dependencies (all crates):** proptest 1.4, approx 0.5, tokio-test 0.4

---

## Implementation Sequence

### Week 1: Foundation — DONE
1. ~~Scaffold Cargo workspace, all 5 crates with stub `lib.rs` / `main.rs`~~
2. ~~Implement `cpcm-core`: DAG, d-separation (Bayes-Ball), backdoor criterion, IV validity checks~~
3. ~~Write comprehensive unit tests for graph algorithms~~ (32 tests)

### Week 2: Data Pipeline — DONE
4. ~~Implement `cpcm-data`: Supabase client, paginated fetch, DataFrame pivot~~
5. ~~Add Parquet caching~~ (save/load with TTL)
6. ~~Integration test against live Supabase~~ (340,934 rows fetched)

### Week 3: Factors and IVs — DONE (85%)
7. ~~Implement all 7 factor builders~~ (5/7 producing data; funding_basis + mev_pressure missing)
8. ~~Implement asset covariate builders~~ (4 covariates)
9. ~~Implement IV builders~~ (2/4 producing data)
10. ~~Data coverage report~~

### Week 4: Estimation — DONE
11. ~~Implement OLS in `cpcm-estimate` with full diagnostics~~
12. ~~Implement 2SLS~~
13. ~~Panel estimation modes~~ (per-asset + pooled with fixed effects)
14. ~~Wire CLI pipeline end-to-end~~

### Week 5: Validation — DONE
15. ~~Run full pipeline on real data~~ (23 assets estimated, results exported)
16. ~~Cross-validation script created~~ (`scripts/cross_validate_ols.py` — statsmodels baseline)
17. ~~Add `cpcm-cli export` for results serialization~~ (JSON + CSV)
18. ~~Fix remaining factor data gaps~~:
    - MEV pressure: 3-level fallback (mev_revenue_eth → stddev_base_fee_gwei → rolling_std(avg_base_fee_gwei))
    - Macro factors: 30 FRED series extracted, z-scored, mapped (macro_DFF → dff)
    - Factor standardization: all factors uniformly z-scored
    - 23-coin expansion: all coins with price data now estimated
    - IV column name fallbacks: gas_spike tries avg_gas_price_gwei → avg_base_fee_gwei → FeeTotNtv

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| **FundingBasis data gap** | NaN column, estimate without. Add Hyperliquid API later. |
| **Sparse IV coverage** (ETH-only for most IVs) | Full 2SLS for ETH, OLS fallback for other assets |
| **Near-singular X'X** (multicollinear factors) | VIF pre-check, SVD pseudo-inverse fallback |
| **Supabase hibernation** | Health check at startup, warn user to `restore_project` |
| **Polars API churn** | Pin exact version (0.46.x), use lazy mode |

---

## Verification

| Check | Status | Result |
|-------|--------|--------|
| `cargo build --workspace` compiles | **PASS** | Zero errors, zero warnings |
| `cargo test --workspace` — all 61 tests | **PASS** | 32 core + 4 data + 13 estimate + 12 factors |
| `cargo run -p cpcm-cli -- graph` — offline DAG test | **PASS** | 72 nodes, 202 edges, 126/126 identified |
| `cargo run -p cpcm-cli -- run` — live pipeline | **PASS** | ~430K rows → 23 assets estimated → JSON/CSV (338 result rows) |
| Parquet cache — subsequent runs skip Supabase | **PASS** | Loads from `cache/panel.parquet` within TTL |
| MEV pressure producing data | **PASS** | Base fee volatility proxy, significant for AAVE |
| Macro factors in estimation | **PASS** | 7 primary FRED series active (vixcls significant for 10/23 assets) |
| All factors z-scored | **PASS** | Standardized coefficients (effect per 1-std-dev change) |
| Cross-validation script ready | **DONE** | `scripts/cross_validate_ols.py` (statsmodels baseline) |

---

## Data Available in Supabase (~430,300 rows)

| Provider | Rows | Coverage |
|----------|------|----------|
| CoinMetrics | 140,679 | btc, eth, bnb, xrp, doge, zec, aave, uni, link, usdc, usdt — 13 metrics |
| Dune | 138,137 | ETH (42 metrics, 12 queries) + SOL/BNB/AVAX activity, DEX, staking, congestion, liquidations |
| Derived | 49,330 | All 26 coins — realized_volatility_7d/30d, dex_cex_volume_ratio (ETH) |
| DefiLlama | 40,667 | 19 assets — tvl_usd, fees_usd, volume_usd, stablecoin metrics |
| CoinGecko | 22,772 | All 26 coins — snapshots + market_chart (last 365 days) |
| FRED | 20,452 | 30 macro series (interest rates, inflation, money supply, labor, commodities, dollar) |
| Allium | 18,270 | BTC + ETH OHLCV fallback |

**26 coins:** USDC, USDT, USDe, BTC, ETH, BNB, HYPE, XRP, PENDLE, UNI, JUP, TAO, LINK, ZEC, ENA, MORPHO, AERO, SOL, AVAX, POL, WLFI, CRV, AAVE, PEPE, SHIB, DOGE

---

## Critical Reference Files
- `causal_portfolio/scm/graph.py` — Python DAG definition to expand in Rust
- `causal_portfolio/scm/estimation.py` — Python 2SLS template to reimplement
- `backfill_data/src/schemas/asset_metrics.py` — Row schema the Rust data layer must mirror
- `backfill_data/scripts/compute_derived_metrics.py` — Factor computation patterns to port
- `docs/Causal PDE-Control Models for Portfolio Optimization.md` — DAG structure, factor defs, IV specs
