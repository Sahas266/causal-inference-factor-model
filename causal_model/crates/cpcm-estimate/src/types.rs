use serde::{Deserialize, Serialize};

/// Result of an OLS regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residuals: Vec<f64>,
    pub n_obs: usize,
    pub n_features: usize,
    pub feature_names: Vec<String>,
    /// Newey-West HAC (Bartlett kernel) standard errors and derived stats.
    pub hac_std_errors: Vec<f64>,
    pub hac_t_stats: Vec<f64>,
    pub hac_p_values: Vec<f64>,
}

/// Result of a 2SLS regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TslsResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub residuals: Vec<f64>,
    pub n_obs: usize,
    pub feature_names: Vec<String>,

    /// First-stage PARTIAL F-statistic (excluded instruments) per endogenous variable.
    pub first_stage_f: Vec<f64>,
    /// Sargan overidentification test (only if instruments > endogenous vars).
    pub sargan_stat: Option<f64>,
    pub sargan_p: Option<f64>,
    /// Hausman test: OLS vs 2SLS. None when no variance-difference term is
    /// positive (difference matrix not positive definite).
    pub hausman_stat: Option<f64>,
    pub hausman_p: Option<f64>,
    /// Newey-West HAC (Bartlett kernel) standard errors and derived stats.
    pub hac_std_errors: Vec<f64>,
    pub hac_t_stats: Vec<f64>,
    pub hac_p_values: Vec<f64>,
}

/// Regression diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Durbin-Watson statistic (serial correlation; ~2.0 = none).
    pub durbin_watson: f64,
    /// Breusch-Pagan test for heteroskedasticity (stat, p-value).
    pub breusch_pagan: (f64, f64),
    /// Jarque-Bera test for normality of residuals (stat, p-value).
    pub jarque_bera: (f64, f64),
    /// Variance Inflation Factor for each feature.
    pub vif: Vec<(String, f64)>,
}

/// Combined estimation result for one treatment-outcome pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationResult {
    pub treatment: String,
    pub outcome: String,
    pub asset: String,
    pub ols: OlsResult,
    pub tsls: Option<TslsResult>,
    pub diagnostics: Diagnostics,
}
