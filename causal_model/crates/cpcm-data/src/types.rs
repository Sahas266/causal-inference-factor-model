use serde::{Deserialize, Deserializer, Serialize};

/// Raw row from the `asset_metrics` (or `asset_metrics_best`) table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMetricRow {
    pub provider: String,
    #[serde(default)]
    pub provider_priority: Option<i32>,
    pub asset: String,
    pub metric: String,
    pub time: String,
    #[serde(default, deserialize_with = "deserialize_value")]
    pub value: Option<String>,
    pub frequency: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Deserialize `value` from either a JSON string ("123.45") or a JSON number (123.45).
fn deserialize_value<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match v {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Number(n)) => Ok(Some(n.to_string())),
        Some(serde_json::Value::Null) => Ok(None),
        Some(other) => Ok(Some(other.to_string())),
    }
}

/// Configuration for the Supabase connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupabaseConfig {
    pub url: String,
    pub key: String,
}

/// Configuration for the data pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub start_date: String,
    pub end_date: String,
    pub cache_path: Option<String>,
    pub cache_ttl_hours: Option<u64>,
    pub use_best_view: bool,
}
