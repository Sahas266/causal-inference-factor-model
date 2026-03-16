/// Pre-defined metric groups for common queries.

/// CoinMetrics asset metrics available on community tier.
pub const COINMETRICS_METRICS: &[&str] = &[
    "PriceUSD",
    "CapMrktCurUSD",
    "TxCnt",
    "AdrActCnt",
    "SplyCur",
    "BlkCnt",
    "HashRate",
    "ROI30d",
    "FeeTotNtv",
    "IssTotNtv",
    "FlowInExNtv",
    "FlowOutExNtv",
    "TxTfrCnt",
];

/// CoinGecko metrics.
pub const COINGECKO_METRICS: &[&str] = &[
    "price_usd",
    "market_cap_usd",
    "total_volume_usd",
    "fdv_usd",
    "ath_usd",
    "atl_usd",
    "total_supply",
    "max_supply",
];

/// DefiLlama metrics.
pub const DEFILLAMA_METRICS: &[&str] = &[
    "tvl_usd",
    "fees_usd",
    "volume_usd",
    "stablecoin_circulating_usd",
];

/// Dune-derived metrics (varies by asset).
pub const DUNE_ETH_METRICS: &[&str] = &[
    "staking_apr",
    "mev_revenue_eth",
    "avg_gas_price_gwei",
    "avg_block_utilization",
    "cex_inflow_usd",
    "cex_outflow_usd",
    "cex_netflow_usd",
    "lp_net_flow_usd",
    "flashloan_volume_usd",
    "liquidation_volume_usd",
    "top_10_pct_balance",
    "eth_staked",
    "net_burn_eth",
    "bridge_inflow_usd",
    "bridge_outflow_usd",
    "l2_blob_fees_eth",
];

/// FRED macro series (stored with asset="macro").
pub const FRED_METRICS: &[&str] = &[
    "DFF", "DGS2", "DGS10", "DGS30", "DFEDTARU", "T10Y2Y", "T10Y3M",
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "T5YIE", "T10YIE", "MICH",
    "M2SL", "WALCL", "RRPONTSYD",
    "VIXCLS", "BAMLH0A0HYM2", "TEDRATE",
    "UNRATE", "PAYEMS", "ICSA", "GDPC1", "INDPRO",
    "DCOILWTICO", "PPIACO",
    "NFCI", "STLFSI2",
    "DTWEXBGS",
];

/// Derived metrics (computed, not fetched).
pub const DERIVED_METRICS: &[&str] = &[
    "realized_volatility_7d",
    "realized_volatility_30d",
    "dex_cex_volume_ratio",
];

/// All 26 target assets.
pub const ALL_ASSETS: &[&str] = &[
    "usdc", "usdt", "usde", "btc", "eth", "bnb", "hype", "xrp", "pendle",
    "uni", "jup", "tao", "link", "zec", "ena", "morpho", "aero", "sol",
    "avax", "pol", "wlfi", "crv", "aave", "pepe", "shib", "doge",
];

/// Assets with CoinMetrics coverage.
pub const COINMETRICS_ASSETS: &[&str] = &[
    "btc", "eth", "bnb", "xrp", "doge", "zec", "aave", "uni", "link", "usdc", "usdt",
];
