use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use polars::prelude::*;
use tracing::{info, warn};

use crate::types::AssetMetricRow;

/// Pivot raw `AssetMetricRow`s into a wide Polars DataFrame.
///
/// Output format:
/// - Index column: `date` (NaiveDate, daily)
/// - One column per unique `{asset}_{metric}` combination
/// - Values are f64 (parsed from the string `value` field)
/// - Missing dates within the range are filled with null
/// - Forward-fill is applied for slowly-updating metrics (TVL, supply)
pub fn pivot_to_panel(
    rows: Vec<AssetMetricRow>,
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> anyhow::Result<DataFrame> {
    if rows.is_empty() {
        anyhow::bail!("No rows to pivot");
    }

    // Group by (asset_metric, date) → value
    let mut column_data: HashMap<String, BTreeMap<NaiveDate, f64>> = HashMap::new();

    for row in &rows {
        let col_name = format!("{}_{}", row.asset, row.metric);
        let date = parse_date(&row.time)?;
        let value = match &row.value {
            Some(v) => match v.parse::<f64>() {
                Ok(f) => f,
                Err(_) => {
                    warn!("Cannot parse value '{}' for {}", v, col_name);
                    continue;
                }
            },
            None => continue,
        };

        column_data.entry(col_name).or_default().insert(date, value);
    }

    // Generate the full date range
    let dates = generate_date_range(start_date, end_date);
    let n_dates = dates.len();

    info!(
        "Pivoting {} rows into {} columns × {} dates",
        rows.len(),
        column_data.len(),
        n_dates
    );

    // Build columns
    let mut df_columns: Vec<Column> = Vec::new();

    // Date column
    let date_series = Series::new(
        "date".into(),
        dates.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
    );
    df_columns.push(date_series.into());

    // Sort column names for deterministic output
    let mut col_names: Vec<String> = column_data.keys().cloned().collect();
    col_names.sort();

    for col_name in &col_names {
        let data_map = &column_data[col_name];
        let values: Vec<Option<f64>> = dates
            .iter()
            .map(|date| data_map.get(date).copied())
            .collect();

        let series = Series::new(col_name.as_str().into(), values);
        df_columns.push(series.into());
    }

    let mut df = DataFrame::new(df_columns)?;

    // Forward-fill null values (for TVL, supply, and other slowly-updating metrics)
    df = forward_fill_columns(df, &col_names)?;

    Ok(df)
}

/// Parse an ISO-8601 datetime or date string to NaiveDate.
fn parse_date(s: &str) -> anyhow::Result<NaiveDate> {
    // Try datetime first (e.g., "2024-01-15T00:00:00+00:00")
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.date_naive());
    }
    // Try datetime without timezone
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.date());
    }
    // Try plain date
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d);
    }
    // Try with fractional seconds
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Ok(dt.date());
    }
    anyhow::bail!("Cannot parse date: '{s}'")
}

/// Generate a vector of all dates from start to end (inclusive).
fn generate_date_range(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    let mut current = start;
    while current <= end {
        dates.push(current);
        current += chrono::Duration::days(1);
    }
    dates
}

/// Forward-fill null values in each column.
fn forward_fill_columns(mut df: DataFrame, col_names: &[String]) -> anyhow::Result<DataFrame> {
    for col_name in col_names {
        let col = df.column(col_name.as_str())?.clone();
        let filled = col
            .as_materialized_series()
            .fill_null(FillNullStrategy::Forward(None))?;
        let _ = df.replace(col_name.as_str(), filled);
    }
    Ok(df)
}

/// Extract a single column as Vec<f64> from the DataFrame, replacing nulls with NaN.
pub fn column_to_vec(df: &DataFrame, col_name: &str) -> anyhow::Result<Vec<f64>> {
    let col = df
        .column(col_name)
        .map_err(|_| anyhow::anyhow!("Column '{col_name}' not found"))?;

    let series = col.as_materialized_series();
    let ca = series
        .f64()
        .map_err(|_| anyhow::anyhow!("Column '{col_name}' is not f64"))?;

    Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Get all numeric column names (everything except "date").
pub fn numeric_columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .filter(|&name| name != "date")
        .map(|s| s.to_string())
        .collect()
}

/// Build a HashMap<String, Vec<f64>> from a DataFrame for use with the estimation layer.
pub fn df_to_hashmap(df: &DataFrame) -> anyhow::Result<HashMap<String, Vec<f64>>> {
    let mut map = HashMap::new();
    for col_name in numeric_columns(df) {
        map.insert(col_name.clone(), column_to_vec(df, &col_name)?);
    }
    Ok(map)
}

/// Save a DataFrame to Parquet.
pub fn save_parquet(df: &mut DataFrame, path: &str) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file).finish(df)?;
    info!("Saved panel to {path}");
    Ok(())
}

/// Load a DataFrame from Parquet if it exists and is within TTL.
pub fn load_parquet(path: &str, ttl_hours: u64) -> anyhow::Result<Option<DataFrame>> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Ok(None);
    }
    let metadata = std::fs::metadata(p)?;
    let modified = metadata.modified()?;
    let age = std::time::SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default();
    if age.as_secs() > ttl_hours * 3600 {
        info!(
            "Cache expired ({:.1}h old, TTL={}h)",
            age.as_secs() as f64 / 3600.0,
            ttl_hours
        );
        return Ok(None);
    }
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    info!(
        "Loaded cached panel from {path} ({} rows × {} cols)",
        df.height(),
        df.width()
    );
    Ok(Some(df))
}

/// Print a summary of the DataFrame: shape, columns, null counts.
pub fn print_summary(df: &DataFrame) {
    println!("Panel shape: {} rows × {} cols", df.height(), df.width());
    println!("Columns:");
    for name in df.get_column_names() {
        let col = df.column(name).unwrap();
        let null_count = col.null_count();
        let non_null = col.len() - null_count;
        println!("  {name}: {non_null} values, {null_count} nulls");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(asset: &str, metric: &str, time: &str, value: &str) -> AssetMetricRow {
        AssetMetricRow {
            provider: "test".to_string(),
            provider_priority: Some(1),
            asset: asset.to_string(),
            metric: metric.to_string(),
            time: time.to_string(),
            value: Some(value.to_string()),
            frequency: Some("1d".to_string()),
            metadata: None,
        }
    }

    #[test]
    fn test_pivot_basic() {
        let rows = vec![
            make_row("eth", "PriceUSD", "2024-01-01", "2300.5"),
            make_row("eth", "PriceUSD", "2024-01-02", "2350.0"),
            make_row("eth", "TxCnt", "2024-01-01", "1200000"),
            make_row("btc", "PriceUSD", "2024-01-01", "42000"),
        ];

        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
        let df = pivot_to_panel(rows, start, end).unwrap();

        assert_eq!(df.height(), 3); // 3 days
        assert!(df.column("eth_PriceUSD").is_ok());
        assert!(df.column("eth_TxCnt").is_ok());
        assert!(df.column("btc_PriceUSD").is_ok());
    }

    #[test]
    fn test_parse_date_variants() {
        assert_eq!(
            parse_date("2024-01-15").unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()
        );
        assert_eq!(
            parse_date("2024-01-15T00:00:00+00:00").unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()
        );
        assert_eq!(
            parse_date("2024-01-15T12:30:00").unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()
        );
    }

    #[test]
    fn test_generate_date_range() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
        let dates = generate_date_range(start, end);
        assert_eq!(dates.len(), 5);
    }
}
