use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub supabase: SupabaseSection,
    pub data: DataSection,
    pub assets: AssetsSection,
    pub estimation: EstimationSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupabaseSection {
    pub url: String,
    /// Either a raw key (starts with "eyJ") or an env var name.
    pub key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSection {
    pub start_date: String,
    pub end_date: String,
    pub cache_path: String,
    pub cache_ttl_hours: u64,
    pub use_best_view: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetsSection {
    pub include: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationSection {
    pub mode: String,
    pub run_ols: bool,
    pub run_2sls: bool,
    pub confidence_level: f64,
}

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read config file '{path}': {e}"))?;
        let mut config: Config = toml::from_str(&content)?;
        if let Ok(url) = std::env::var("SUPABASE_URL") {
            if !url.trim().is_empty() {
                config.supabase.url = url.trim().to_string();
            }
        }
        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            supabase: SupabaseSection {
                url: "https://your-project-id.supabase.co".to_string(),
                key_env: "SUPABASE_KEY".to_string(),
            },
            data: DataSection {
                start_date: "2021-01-01".to_string(),
                end_date: "2026-01-01".to_string(),
                cache_path: "cache/panel.parquet".to_string(),
                cache_ttl_hours: 24,
                use_best_view: true,
            },
            assets: AssetsSection {
                include: vec![
                    "btc", "eth", "bnb", "sol", "avax", "xrp", "pol", "hype", "tao", "wlfi", "uni",
                    "aave", "crv", "pendle", "morpho", "aero", "link", "ena", "jup", "zec", "pepe",
                    "shib", "doge",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            },
            estimation: EstimationSection {
                mode: "per_asset".to_string(),
                run_ols: true,
                run_2sls: true,
                confidence_level: 0.95,
            },
        }
    }
}
