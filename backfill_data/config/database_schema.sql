-- Multi-Provider Data Backfill System - Database Schema
-- PostgreSQL / Supabase compatible

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- PROGRESS TRACKING TABLE
-- ============================================================================
-- Tracks backfill progress for each endpoint+provider combination
-- Supports checkpoint resumption and error tracking

CREATE TABLE IF NOT EXISTS backfill_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id TEXT NOT NULL,              -- Unique identifier: endpoint_provider
    provider TEXT NOT NULL,                 -- Provider name (coinmetrics, glassnode, etc.)
    endpoint_type TEXT NOT NULL,            -- Type of endpoint (asset-metrics, market-trades, etc.)
    table_name TEXT NOT NULL,               -- Target table for data
    status TEXT NOT NULL DEFAULT 'pending'  -- Status: pending, validating, running, completed, failed
        CHECK (status IN ('pending', 'validating', 'running', 'completed', 'failed')),
    
    -- Progress tracking
    last_successful_time TIMESTAMPTZ,       -- Last successfully processed timestamp (checkpoint)
    total_records_fetched BIGINT DEFAULT 0, -- Total records fetched
    total_api_calls INTEGER DEFAULT 0,      -- Total API calls made
    
    -- Timing
    started_at TIMESTAMPTZ,                 -- When backfill started
    completed_at TIMESTAMPTZ,               -- When backfill completed
    
    -- Error handling
    error_message TEXT,                     -- Last error message
    retry_count INTEGER DEFAULT 0,          -- Number of retries attempted
    
    -- Configuration snapshot
    config JSONB,                           -- Configuration used for this backfill
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(endpoint_id, provider)
);

-- Indexes for progress tracking
CREATE INDEX IF NOT EXISTS idx_progress_provider ON backfill_progress(provider);
CREATE INDEX IF NOT EXISTS idx_progress_status ON backfill_progress(status);
CREATE INDEX IF NOT EXISTS idx_progress_updated ON backfill_progress(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_progress_endpoint_id ON backfill_progress(endpoint_id);

-- ============================================================================
-- ASSET METRICS TABLE
-- ============================================================================
-- Stores asset-level metrics (price, market cap, volume, on-chain metrics, etc.)
-- Multi-provider with priority-based deduplication support

CREATE TABLE IF NOT EXISTS asset_metrics (
    -- Provider metadata
    provider TEXT NOT NULL,                 -- Data provider
    provider_priority INTEGER NOT NULL,     -- Priority for deduplication (lower = higher priority)
    
    -- Asset identification
    asset TEXT NOT NULL,                    -- Asset symbol (btc, eth, etc.)
    metric TEXT NOT NULL,                   -- Metric name (PriceUSD, CapMrktCurUSD, etc.)
    time TIMESTAMPTZ NOT NULL,              -- Metric timestamp
    
    -- Data
    value NUMERIC,                          -- Metric value (can be NULL for missing data)
    frequency TEXT NOT NULL,                -- Data frequency (1d, 1h, 5m, etc.)
    metadata JSONB,                         -- Additional provider-specific metadata
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key: provider + asset + metric + time
    PRIMARY KEY (provider, asset, metric, time)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_asset_metrics_composite 
    ON asset_metrics(asset, metric, time DESC);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_priority 
    ON asset_metrics(asset, metric, time DESC, provider_priority ASC);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_provider 
    ON asset_metrics(provider);
CREATE INDEX IF NOT EXISTS idx_asset_metrics_time 
    ON asset_metrics(time DESC);

-- View: Get best data per metric (highest priority provider)
CREATE OR REPLACE VIEW asset_metrics_best AS
SELECT DISTINCT ON (asset, metric, time)
    provider,
    asset,
    metric,
    time,
    value,
    frequency,
    metadata,
    created_at
FROM asset_metrics
ORDER BY asset, metric, time, provider_priority ASC;

-- ============================================================================
-- MARKET TRADES TABLE
-- ============================================================================
-- Stores individual trades from exchanges
-- Multi-provider support

CREATE TABLE IF NOT EXISTS market_trades (
    -- Provider metadata
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    
    -- Trade identification
    market TEXT NOT NULL,                   -- Market identifier (coinbase-btc-usd-spot)
    time TIMESTAMPTZ NOT NULL,              -- Trade timestamp
    trade_id TEXT NOT NULL,                 -- Unique trade ID from provider
    
    -- Trade data
    price NUMERIC NOT NULL,                 -- Trade price
    amount NUMERIC NOT NULL,                -- Trade amount/volume
    side TEXT CHECK (side IN ('buy', 'sell', 'unknown')), -- Trade side
    
    -- Metadata
    metadata JSONB,                         -- Additional trade metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key: provider + market + time + trade_id
    PRIMARY KEY (provider, market, time, trade_id)
);

-- Indexes for market trades
CREATE INDEX IF NOT EXISTS idx_market_trades_composite 
    ON market_trades(market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_trades_provider 
    ON market_trades(provider);
CREATE INDEX IF NOT EXISTS idx_market_trades_time 
    ON market_trades(time DESC);

-- ============================================================================
-- EXCHANGE METRICS TABLE
-- ============================================================================
-- Stores exchange-level metrics (flows, balances, etc.)
-- Multi-provider support

CREATE TABLE IF NOT EXISTS exchange_metrics (
    -- Provider metadata
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    
    -- Exchange identification
    exchange TEXT NOT NULL,                 -- Exchange name (binance, coinbase, etc.)
    metric TEXT NOT NULL,                   -- Metric name (flow_in_btc, balance_btc, etc.)
    time TIMESTAMPTZ NOT NULL,              -- Metric timestamp
    
    -- Data
    value NUMERIC,                          -- Metric value
    frequency TEXT NOT NULL,                -- Data frequency
    metadata JSONB,                         -- Additional metadata
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key: provider + exchange + metric + time
    PRIMARY KEY (provider, exchange, metric, time)
);

-- Indexes for exchange metrics
CREATE INDEX IF NOT EXISTS idx_exchange_metrics_composite 
    ON exchange_metrics(exchange, metric, time DESC);
CREATE INDEX IF NOT EXISTS idx_exchange_metrics_priority 
    ON exchange_metrics(exchange, metric, time DESC, provider_priority ASC);
CREATE INDEX IF NOT EXISTS idx_exchange_metrics_provider 
    ON exchange_metrics(provider);

-- View: Get best exchange metrics (highest priority provider)
CREATE OR REPLACE VIEW exchange_metrics_best AS
SELECT DISTINCT ON (exchange, metric, time)
    provider,
    exchange,
    metric,
    time,
    value,
    frequency,
    metadata,
    created_at
FROM exchange_metrics
ORDER BY exchange, metric, time, provider_priority ASC;

-- ============================================================================
-- MARKET ORDERBOOKS TABLE
-- ============================================================================
-- Stores orderbook snapshots (bids and asks) from markets
-- Each row represents one level in the orderbook
-- Multi-provider support with flattened structure for efficient querying

CREATE TABLE IF NOT EXISTS market_orderbooks (
    -- Provider metadata
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    
    -- Market and snapshot identification
    market TEXT NOT NULL,                   -- Market identifier (coinbase-btc-usd-spot)
    time TIMESTAMPTZ NOT NULL,              -- Snapshot timestamp
    snapshot_id TEXT NOT NULL,              -- Unique snapshot ID (coin_metrics_id)
    
    -- Orderbook level data
    side TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
    price NUMERIC NOT NULL,                 -- Price at this level
    size NUMERIC NOT NULL,                  -- Size/volume at this level
    level INTEGER NOT NULL,                 -- Orderbook level (0 = best bid/ask, 1 = second best, etc.)
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key: provider + market + time + snapshot_id + side + level
    PRIMARY KEY (provider, market, time, snapshot_id, side, level)
);

-- Indexes for orderbook queries
CREATE INDEX IF NOT EXISTS idx_orderbooks_market_time 
    ON market_orderbooks(market, time DESC);
CREATE INDEX IF NOT EXISTS idx_orderbooks_side 
    ON market_orderbooks(market, side, time DESC);
CREATE INDEX IF NOT EXISTS idx_orderbooks_level 
    ON market_orderbooks(market, time DESC, level ASC);
CREATE INDEX IF NOT EXISTS idx_orderbooks_provider 
    ON market_orderbooks(provider);

-- View: Get best bid/ask (level 0) across all snapshots
CREATE OR REPLACE VIEW market_orderbooks_best_quotes AS
SELECT 
    provider,
    market,
    time,
    side,
    price,
    size,
    snapshot_id
FROM market_orderbooks
WHERE level = 0
ORDER BY market, time DESC, side;

-- ============================================================================
-- HELPER VIEWS FOR MONITORING
-- ============================================================================

-- Provider health check view
CREATE OR REPLACE VIEW provider_health AS
SELECT 
    provider,
    COUNT(*) as total_endpoints,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
    SUM(total_records_fetched) as total_records,
    AVG(total_api_calls) as avg_api_calls,
    MAX(updated_at) as last_activity
FROM backfill_progress
GROUP BY provider
ORDER BY provider;

-- Data coverage comparison by provider
CREATE OR REPLACE VIEW data_coverage_by_provider AS
SELECT 
    asset,
    metric,
    provider,
    MIN(time) as earliest_data,
    MAX(time) as latest_data,
    COUNT(*) as data_points,
    COUNT(CASE WHEN value IS NOT NULL THEN 1 END) as non_null_points
FROM asset_metrics
GROUP BY asset, metric, provider
ORDER BY asset, metric, provider;

-- Recent backfill activity
CREATE OR REPLACE VIEW recent_backfill_activity AS
SELECT 
    endpoint_id,
    provider,
    status,
    total_records_fetched,
    last_successful_time,
    error_message,
    updated_at
FROM backfill_progress
WHERE updated_at > NOW() - INTERVAL '24 hours'
ORDER BY updated_at DESC;

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for backfill_progress
DROP TRIGGER IF EXISTS update_backfill_progress_updated_at ON backfill_progress;
CREATE TRIGGER update_backfill_progress_updated_at
    BEFORE UPDATE ON backfill_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for asset_metrics
DROP TRIGGER IF EXISTS update_asset_metrics_updated_at ON asset_metrics;
CREATE TRIGGER update_asset_metrics_updated_at
    BEFORE UPDATE ON asset_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for exchange_metrics
DROP TRIGGER IF EXISTS update_exchange_metrics_updated_at ON exchange_metrics;
CREATE TRIGGER update_exchange_metrics_updated_at
    BEFORE UPDATE ON exchange_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PAIR CANDLES (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS pair_candles (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    pair TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    price_open NUMERIC,
    price_high NUMERIC,
    price_low NUMERIC,
    price_close NUMERIC,
    volume NUMERIC,
    vwap NUMERIC,
    candle_usd_volume NUMERIC,
    frequency TEXT NOT NULL DEFAULT '1d',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, pair, time)
);

CREATE INDEX IF NOT EXISTS idx_pair_candles_pair_time ON pair_candles (pair, time DESC);
CREATE INDEX IF NOT EXISTS idx_pair_candles_provider ON pair_candles (provider);

-- ============================================================================
-- MARKET OPEN INTEREST (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_open_interest (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    contract_count NUMERIC,
    value_usd NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time)
);

CREATE INDEX IF NOT EXISTS idx_market_oi_market_time ON market_open_interest (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_oi_provider ON market_open_interest (provider);

-- ============================================================================
-- MARKET LIQUIDATIONS (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_liquidations (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    coin_metrics_id TEXT NOT NULL,
    amount NUMERIC,
    price NUMERIC,
    side TEXT CHECK (side IN ('buy', 'sell', 'unknown')),
    type TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time, coin_metrics_id)
);

CREATE INDEX IF NOT EXISTS idx_market_liq_market_time ON market_liquidations (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_liq_provider ON market_liquidations (provider);

-- ============================================================================
-- MARKET FUNDING RATES (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_funding_rates (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    rate NUMERIC,
    period TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time)
);

CREATE INDEX IF NOT EXISTS idx_market_fr_market_time ON market_funding_rates (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_fr_provider ON market_funding_rates (provider);

-- ============================================================================
-- MARKET CANDLES (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_candles (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    price_open NUMERIC,
    price_high NUMERIC,
    price_low NUMERIC,
    price_close NUMERIC,
    volume NUMERIC,
    vwap NUMERIC,
    candle_usd_volume NUMERIC,
    frequency TEXT NOT NULL DEFAULT '1d',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time)
);

CREATE INDEX IF NOT EXISTS idx_market_candles_market_time ON market_candles (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_candles_provider ON market_candles (provider);

-- ============================================================================
-- MARKET IMPLIED VOLATILITY (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_implied_volatility (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    implied_volatility NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time)
);

CREATE INDEX IF NOT EXISTS idx_market_iv_market_time ON market_implied_volatility (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_iv_provider ON market_implied_volatility (provider);

-- ============================================================================
-- MARKET GREEKS (FM-13)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_greeks (
    provider TEXT NOT NULL,
    provider_priority INTEGER NOT NULL,
    market TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    delta NUMERIC,
    gamma NUMERIC,
    vega NUMERIC,
    theta NUMERIC,
    rho NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (provider, market, time)
);

CREATE INDEX IF NOT EXISTS idx_market_greeks_market_time ON market_greeks (market, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_greeks_provider ON market_greeks (provider);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE backfill_progress IS 'Tracks progress of backfill operations for checkpoint resumption';
COMMENT ON TABLE asset_metrics IS 'Asset-level metrics from multiple providers with priority-based deduplication';
COMMENT ON TABLE market_trades IS 'Individual trades from exchanges across multiple providers';
COMMENT ON TABLE exchange_metrics IS 'Exchange-level metrics (flows, balances) from multiple providers';
COMMENT ON TABLE pair_candles IS 'OHLCV candle data for asset pairs from multiple providers';
COMMENT ON TABLE market_open_interest IS 'Open interest data for futures markets from multiple providers';
COMMENT ON TABLE market_liquidations IS 'Liquidation events for futures markets from multiple providers';
COMMENT ON TABLE market_funding_rates IS 'Funding rate data for perpetual futures from multiple providers';
COMMENT ON TABLE market_candles IS 'Exchange-specific OHLCV candle data from multiple providers';
COMMENT ON TABLE market_implied_volatility IS 'Implied volatility data for options markets from multiple providers';
COMMENT ON TABLE market_greeks IS 'Option greeks data from multiple providers';

COMMENT ON VIEW asset_metrics_best IS 'Returns best available data per metric based on provider priority';
COMMENT ON VIEW exchange_metrics_best IS 'Returns best available exchange metrics based on provider priority';
COMMENT ON VIEW provider_health IS 'Summary of provider health and activity';
COMMENT ON VIEW data_coverage_by_provider IS 'Data coverage comparison across providers';

