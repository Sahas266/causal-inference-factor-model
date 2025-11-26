-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Set up TimescaleDB configuration for better performance
-- These settings can be adjusted based on your hardware and workload

-- Enable compression by default for new hypertables
ALTER DATABASE defi_pipeline SET timescaledb.compress = true;

-- Set compression policy to compress chunks after 7 days
-- ALTER DATABASE defi_pipeline SET timescaledb.compress_segmentby = true;
-- ALTER DATABASE defi_pipeline SET timescaledb.compress_orderby = 'timestamp';

-- Set chunk interval policy (can be adjusted per table)
-- Default chunk interval is 1 day for most metrics, but can be customized

-- Create a maintenance job to compress old data
-- This will run daily to compress chunks older than 7 days
SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_liq_flow',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_stableflow',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_funding_basis',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_chain_congestion',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_staking_yield',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_mev_pressure',
        'compress_before', '7 days'::interval
    )
) AS job_id;

SELECT add_job(
    'compress_chunks',
    '1 day',
    config => json_build_object(
        'hypertable', 'metrics_cex_dex_flow',
        'compress_before', '7 days'::interval
    )
) AS job_id;

-- Create retention policies (optional - adjust based on your needs)
-- These will drop data older than 1 year to manage database size

-- SELECT add_retention_policy('metrics_liq_flow', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_stableflow', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_funding_basis', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_chain_congestion', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_staking_yield', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_mev_pressure', INTERVAL '1 year') AS job_id;
-- SELECT add_retention_policy('metrics_cex_dex_flow', INTERVAL '1 year') AS job_id;
