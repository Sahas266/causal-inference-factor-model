# Phase 2: Structural Causal Model (SCM) Implementation

Phase 2 focuses on the **SCM Layer**, which bridges the gap between raw data (Phase 1) and portfolio optimization (Phase 3). Its primary goal is to identify true causal mechanisms in DeFi markets—distinguishing between things that merely "move together" (correlation) and things where one actually "causes" the other (causation).

This layer is implemented across two parallel stacks: a production-grade **Rust engine** (`causal_model/`) and a **Python research stack** (`causal_portfolio/`).

---

## Step 1: Causal Graph (DAG) Definition
**What it does:** Defines the "map" of the market. We represent our causal theory using a **Directed Acyclic Graph (DAG)**, where nodes are variables (Factors, Covariates, Returns) and arrows represent the direction of causality.
- **Nodes:** Global Factors (Macro forces like StableFlow), Asset Covariates (Token-specific metrics), and Returns.
- **Edges:** We assert, for example, that *Stablecoin Inflow → Market Liquidity → Asset Returns*.
- **Files:**
    - `causal_model/crates/cpcm-core/src/dag.rs`: Generic graph data structures.
    - `causal_model/crates/cpcm-core/src/cpcm_dag.rs`: The specific 7-factor DAG used in the CPCM model.
    - `causal_portfolio/scm/graph.py`: NetworkX/DoWhy definitions for research.

## Step 2: Identification & d-Separation
**What it does:** Mathematically verifies if a causal effect *can* be estimated. We use the **Backdoor Criterion** and **d-separation** (using the Bayes-Ball algorithm) to find a "minimal adjustment set"—the specific variables we must control for to block "backdoor paths" (spurious correlations) and isolate the true effect of $X$ on $Y$.
- **Files:**
    - `causal_model/crates/cpcm-core/src/dsep.rs`: Implementation of the Bayes-Ball path-blocking algorithm.
    - `causal_model/crates/cpcm-core/src/identify.rs`: Logic for identifying identifiable causal effects.
    - `causal_portfolio/scm/model.py`: DoWhy integration for automated identification.

## Step 3: Factor Engineering & Instrument Generation
**What it does:** Prepares the data for regression. Raw metrics (e.g., total supply in USD) are transformed into normalized, stationary causal factors (e.g., net supply change, z-scored). Crucially, we also generate **Instrumental Variables (IVs)**—"shocks" that affect the cause but not the effect directly (e.g., gas spikes that force LP exits but don't directly change the token's value).
- **Files:**
    - `causal_model/crates/cpcm-factors/src/global_factors.rs`: Computation of the 7 CPCM factors.
    - `causal_model/crates/cpcm-factors/src/instruments.rs`: Logic for generating exogenous shocks/IVs.
    - `causal_portfolio/factors/builder.py`: Python-based factor construction.
    - `causal_portfolio/scm/shocks.py`: Instrument construction for research.

## Step 4: Causal Estimation (OLS vs. 2SLS)
**What it does:** The core statistical engine. We run two types of regressions:
1. **OLS (Ordinary Least Squares):** Provides the baseline "correlation" estimate.
2. **2SLS (Two-Stage Least Squares):** The "causal" estimator. In Stage 1, we use our IVs to "clean" the factor of its endogenous noise. In Stage 2, we regress returns on this "clean" factor. If the 2SLS coefficient differs significantly from OLS, we've found a causal relationship that correlation alone would have missed.
- **Files:**
    - `causal_model/crates/cpcm-estimate/src/ols.rs`: High-performance OLS implementation.
    - `causal_model/crates/cpcm-estimate/src/tsls.rs`: Two-Stage Least Squares implementation.
    - `causal_portfolio/scm/estimation.py`: EconML/DoWhy wrappers for advanced estimation.

## Step 5: Validation & Diagnostics
**What it does:** Stress-tests our estimates to ensure they aren't artifacts of bad data or weak theory.
- **Hausman Test:** Checks if OLS is "consistent" (i.e., if correlation is a good enough proxy for causation in this specific case).
- **First-Stage F-Statistic:** Ensures our instruments are "strong" enough to actually push the factor around.
- **Sargan Test:** Checks for "over-identification" if we have multiple instruments.
- **Files:**
    - `causal_model/crates/cpcm-estimate/src/diagnostics.rs`: Implementation of all Rust-based statistical tests.
    - `causal_portfolio/optimizer/diagnostics.py`: Optimization-specific validation.

## Step 6: Pipeline Orchestration
**What it does:** Wires all the steps together into a single automated workflow: Fetch Data → Build Factors → Identify Effects → Estimate Coefficients → Export Results.
- **Files:**
    - `causal_model/crates/cpcm-cli/src/pipeline.rs`: The "Main Loop" for the Rust CPCM-CLI.
    - `causal_portfolio/main.py`: The research-oriented execution script.
