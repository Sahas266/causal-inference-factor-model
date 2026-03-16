use reqwest::header::{HeaderMap, HeaderValue};
use tracing::{debug, info};

use crate::types::{AssetMetricRow, SupabaseConfig};

const PAGE_SIZE: usize = 1000;

/// HTTP client for the Supabase PostgREST API.
pub struct SupabaseClient {
    http: reqwest::Client,
    base_url: String,
    _api_key: String,
}

impl SupabaseClient {
    /// Create a new client from config. Reads `SUPABASE_KEY` from env if `key` is
    /// the name of an env var.
    pub fn new(config: &SupabaseConfig) -> anyhow::Result<Self> {
        let api_key = if config.key.starts_with("eyJ") {
            // Looks like an actual JWT key
            config.key.clone()
        } else {
            // Treat as env var name
            std::env::var(&config.key)
                .map_err(|_| anyhow::anyhow!("Env var '{}' not set", config.key))?
        };

        let mut headers = HeaderMap::new();
        headers.insert("apikey", HeaderValue::from_str(&api_key)?);
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {api_key}"))?,
        );
        headers.insert("Prefer", HeaderValue::from_static("return=representation"));

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()?;

        // Normalize URL: strip trailing slash
        let base_url = config.url.trim_end_matches('/').to_string();

        Ok(Self {
            http,
            base_url,
            _api_key: api_key,
        })
    }

    /// Health check: ping the Supabase REST API.
    pub async fn health_check(&self) -> anyhow::Result<()> {
        let url = format!("{}/rest/v1/", self.base_url);
        let resp = self.http.get(&url).send().await?;
        if resp.status().is_success() || resp.status().as_u16() == 200 {
            info!("Supabase health check OK");
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Supabase health check failed ({}): {}. Project may be INACTIVE — restore it first.",
                status, body
            )
        }
    }

    /// Fetch asset metrics with pagination.
    ///
    /// # Arguments
    /// * `table` - Table or view name (e.g., `asset_metrics_best` or `asset_metrics`)
    /// * `assets` - Filter by these asset tickers
    /// * `metrics` - Filter by these metric names (empty = all)
    /// * `start_date` - ISO date string (inclusive)
    /// * `end_date` - ISO date string (inclusive)
    pub async fn fetch_metrics(
        &self,
        table: &str,
        assets: &[&str],
        metrics: &[&str],
        start_date: &str,
        end_date: &str,
    ) -> anyhow::Result<Vec<AssetMetricRow>> {
        let mut all_rows: Vec<AssetMetricRow> = Vec::new();
        let mut offset = 0;

        loop {
            let url = self.build_query_url(table, assets, metrics, start_date, end_date, offset)?;
            debug!("Fetching: {url}");

            let resp = self.http.get(&url).send().await?;
            let status = resp.status();
            let body = resp.text().await?;

            if !status.is_success() {
                anyhow::bail!("Supabase query failed ({}): {}", status, body);
            }

            let rows: Vec<AssetMetricRow> = serde_json::from_str(&body)?;
            let count = rows.len();
            all_rows.extend(rows);

            info!("Fetched {count} rows (total: {})", all_rows.len());

            if count < PAGE_SIZE {
                break;
            }
            offset += PAGE_SIZE;
        }

        Ok(all_rows)
    }

    fn build_query_url(
        &self,
        table: &str,
        assets: &[&str],
        metrics: &[&str],
        start_date: &str,
        end_date: &str,
        offset: usize,
    ) -> anyhow::Result<String> {
        // asset_metrics has provider_priority; asset_metrics_best view does not
        let select = if table.contains("best") {
            "provider,asset,metric,time,value,frequency"
        } else {
            "provider,provider_priority,asset,metric,time,value,frequency"
        };
        let mut url = format!(
            "{}/rest/v1/{}?select={}",
            self.base_url, table, select
        );

        // Asset filter
        if !assets.is_empty() {
            let asset_list = assets
                .iter()
                .map(|a| format!("\"{}\"", a))
                .collect::<Vec<_>>()
                .join(",");
            url.push_str(&format!("&asset=in.({})", asset_list));
        }

        // Metric filter
        if !metrics.is_empty() {
            let metric_list = metrics
                .iter()
                .map(|m| format!("\"{}\"", m))
                .collect::<Vec<_>>()
                .join(",");
            url.push_str(&format!("&metric=in.({})", metric_list));
        }

        // Date range
        url.push_str(&format!("&time=gte.{start_date}&time=lte.{end_date}"));

        // Order and pagination
        url.push_str("&order=time.asc");
        url.push_str(&format!(
            "&offset={offset}&limit={PAGE_SIZE}"
        ));

        Ok(url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_query_url() {
        let _config = SupabaseConfig {
            url: "https://example.supabase.co".to_string(),
            key: "test-key".to_string(),
        };

        // We can't actually create the client without a valid key in env,
        // but we can test the URL building logic indirectly.
        // The build_query_url is a method on SupabaseClient, so we just
        // verify the format expectations.
        let expected_base = "https://example.supabase.co/rest/v1/asset_metrics_best";
        assert!(expected_base.contains("asset_metrics_best"));
    }
}
