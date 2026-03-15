## **Causal PDE-Control Models for Portfolio Optimization**

[Causal PDE-Control Models (CPCMs)](https://arxiv.org/pdf/2509.09585v2) is a structural framework that bridges causal inference, nonlinear filtering, and stochastic control to create portfolio allocations under partial information regimes 

CPCMs achieve higher Sharpe and lower turnover. They provide a deployable middle ground between rigid classical models and black-box ML

While prior research focuses on US equities, we will build our factor model for defi, back test results, generate signals, and use it generate alpha on chain 

## **Implementation Roadmap for Live Trading**

### Phase 1 Build Driver Library for DeFi:

* Build driver library: 300+ time series, standardized, cleaned  
  * On-chain metrics: TVL, active addresses, gas fees, MEV volume, liquidation flows  
  * Protocol-specific: Utilization rates, borrow APYs, trading volumes per DEX  
  * Cross-chain: Bridge flows, L2 activity, validator/staking dynamics  
  * Token-specific: Holder concentration, smart contract interactions, vesting schedules  
  * Use socials to get sentiment data telegram, discord  
    * Use embeddings, semantic search   
    *   
* Implement EKF/PF with stratified resampling  
* Set up real-time observation pipeline

### Phase 2: SCM Layer

#### 2.1 Causal Graph for DeFi Asset Returns

Here are my thoughts on our initial DAG:

*Stable inflow → market liquidity → returns*

*Funding rate → perp flow → returns*

*LP incentives → LP inflow → AMM depth → returns*

*Gas fees → tx volume → volatility*

These factors will be used to represent causal drivers of crypto returns, not just correlations.

Constructing SCM where:

*F \= causal DeFi factors*

| Factor | Description | Causal Mechanism | Data source |
| ----- | ----- | ----- | ----- |
| Liq Flow | Net LP inflows/outflows | Predicts volatility, direction |  |
| Stableflow | Net stablecoin mint/burn | Measures systemic liquidity |  |
| Funding Basis | Perp funding premium | Predicts mean reversion/momentum |  |
| Chain Congestion | Gas fees, block fullness | Predicts execution cost, volatility |  |
| Staking Yield | Real staking APR | Predicts carry/valuation |  |
| MEV Pressure | MEV extraction volatility | Predicts spreads, variance |  |
| CEX/DEX Flow | Bridged inflow/outflow | Predicts buy/sell pressure |  |

In order to define the nodes of our SCM we must compute:

1. LiqFlow: Net LP inflows/outflows across AMMs.  
   * Mechanism: changes liquidity depth, volatility, and price impact.  
2. StableFlow: Net mint/burn of top stablecoins USDC/USDT/DAI across chains.  
   * Mechanism: system-level liquidity changes → beta compression/expansion.  
3. Funding Basis: Perp funding \- spot return.  
   * Mechanism: imbalance means directional reversion or continuation depending on the OI regime.  
4. Chain Congestion: Gas fees, block fullness, sequencer delays.  
   * Mechanism: execution friction lowers arbitrage efficiency which increases volatility.  
5. StakeYield: Real staking APR  
   * Mechanism: affects opportunity cost and reprices yield assets.  
6. MEV Pressure: Variance in MEV revenue per block.  
   * Mechanism: unpredictable orderflow affects spreads and short term returns.  
7. CEX to DEX Flow: Exchange-to-chain net transfer flow.  
   * Mechanism: buy/sell pressure shock to AMMs.

Next we need to get asset specific Covariates (Xᵢ) these are protocol level and asset specific fundamentals: 

*Xi \= asset-specific causal covariates*

1. Whale concentration delta: Changes in concentration among top addresses.  
2. Protocol revenues: Fees, swaps, borrow interest.  
3. Emissions schedule / inflation: Modulates selling pressure.  
4. Governance events: Proposal timing leads to temporary uncertainty/sentiment.  
5. Chain-specific activity: TPS, gas burnt, L1/L2 adoption

Next lets collect datapoints on unobserved shocks these will be: *ϵi(t)ϵi​(t)*

* Protocol exploits  
* Chain halts  
* Oracle failures  
* Liquidation cascades

Here is my initial thesis we will see from once the full SCM DAG is complete: 

Global DeFi factors point to returns. All F factors point into each asset return rᵢ

Asset specific covariates point to asset returns. Each Xᵢ points into its own asset’s return.

*F → Xᵢ interactions*

Some asset specific features depend on factors:

* StableFlow to whale concentration  
* Chain congestion to protocol volume  
* Funding basis to protocol OI

Unobserved shocks point to returns

Meaning: εᵢ → rᵢ

Once we have our data we can build the SCM following this [project structure](https://github.com/Sahas266/causal-inference-factor-model/blob/v1/causal_portfolio/project_goals.md)

#### 2.2 IVs: instrumental variables

IVs disentangle protocol activity from price, liquidity shocks from volatility, flows from returns, etc.

| Causal Path | Problem | IV |
| ----- | ----- | ----- |
| LP outflow → price drop | Reverse causality | Use gas spikes, incentive changes |
| Perp funding → future returns | Confounding by carry traders | Use OI shocks, liquidation levels |
| Stablecoin issuance → market beta | Macro cycle confounding | Use treasury inflow, chain-level mint events |
| Protocol revenue → token return | Price affects user activity | Use external protocol events |

// TODO

