"""
SQLAlchemy models for DeFi data pipeline with TimescaleDB support.
All models are designed for time-series data with hypertables.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from sqlalchemy import Column, String, Integer, Numeric, DateTime, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class LiqFlow(Base):
    """Liquidity flow metrics table."""

    __tablename__ = "metrics_liq_flow"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    chain = Column(String(50), primary_key=True, nullable=False)
    protocol = Column(String(100), nullable=False)
    net_inflow = Column(Numeric(36, 18), nullable=False)
    tvl_change = Column(Numeric(36, 18), nullable=False)
    token_flows = Column(JSONB, nullable=True)
    add_liquidity_events = Column(Integer, nullable=True)
    remove_liquidity_events = Column(Integer, nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_liq_flow_timestamp_chain", timestamp, chain),
        Index("idx_liq_flow_protocol", protocol),
    )


class Stableflow(Base):
    """Stablecoin flow metrics table."""

    __tablename__ = "metrics_stableflow"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    stablecoin = Column(String(10), primary_key=True, nullable=False)
    chain = Column(String(50), nullable=False)
    minted = Column(Numeric(36, 18), nullable=False)
    burned = Column(Numeric(36, 18), nullable=False)
    net_flow = Column(Numeric(36, 18), nullable=False)
    mint_events = Column(Integer, nullable=True)
    burn_events = Column(Integer, nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_stableflow_timestamp_stablecoin", timestamp, stablecoin),
        Index("idx_stableflow_chain", chain),
    )


class FundingBasis(Base):
    """Funding basis metrics table."""

    __tablename__ = "metrics_funding_basis"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    exchange = Column(String(50), primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    funding_rate = Column(Numeric(10, 6), nullable=False)
    premium_index = Column(Numeric(10, 6), nullable=False)
    mark_price = Column(Numeric(36, 18), nullable=True)
    index_price = Column(Numeric(36, 18), nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_funding_basis_timestamp_exchange", timestamp, exchange),
        Index("idx_funding_basis_symbol", symbol),
    )


class ChainCongestion(Base):
    """Chain congestion metrics table."""

    __tablename__ = "metrics_chain_congestion"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    chain = Column(String(50), primary_key=True, nullable=False)
    avg_gas_price = Column(Numeric(20, 10), nullable=False)
    block_fullness = Column(Numeric(5, 4), nullable=False)
    pending_txs = Column(Integer, nullable=False)
    base_fee = Column(Numeric(20, 10), nullable=True)
    priority_fee = Column(Numeric(20, 10), nullable=True)
    block_time = Column(Numeric(10, 3), nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_chain_congestion_timestamp_chain", timestamp, chain),
    )


class StakingYield(Base):
    """Staking yield metrics table."""

    __tablename__ = "metrics_staking_yield"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    chain = Column(String(50), primary_key=True, nullable=False)
    staked_amount = Column(Numeric(36, 18), nullable=False)
    apr = Column(Numeric(10, 6), nullable=False)
    validator_count = Column(Integer, nullable=False)
    total_validators = Column(Integer, nullable=True)
    slashing_events = Column(Integer, nullable=True)
    rewards_distributed = Column(Numeric(36, 18), nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_staking_yield_timestamp_chain", timestamp, chain),
    )


class MEVPressure(Base):
    """MEV pressure metrics table."""

    __tablename__ = "metrics_mev_pressure"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    chain = Column(String(50), primary_key=True, nullable=False)
    mev_extracted = Column(Numeric(36, 18), nullable=False)
    mev_blocks_pct = Column(Numeric(5, 4), nullable=False)
    top_searcher_share = Column(Numeric(5, 4), nullable=False)
    sandwich_attacks = Column(Integer, nullable=True)
    frontrunning_events = Column(Integer, nullable=True)
    arbitrage_opportunities = Column(Integer, nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_mev_pressure_timestamp_chain", timestamp, chain),
    )


class CEXDEXFlow(Base):
    """CEX/DEX flow metrics table."""

    __tablename__ = "metrics_cex_dex_flow"

    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    bridge = Column(String(100), primary_key=True, nullable=False)
    direction = Column(String(10), primary_key=True, nullable=False)
    volume = Column(Numeric(36, 18), nullable=False)
    unique_users = Column(Integer, nullable=False)
    transaction_count = Column(Integer, nullable=True)
    avg_transaction_size = Column(Numeric(36, 18), nullable=True)

    __table_args__ = (
        # Indexes for performance
        Index("idx_cex_dex_flow_timestamp_bridge", timestamp, bridge),
        Index("idx_cex_dex_flow_direction", direction),
    )


# Utility functions for TimescaleDB operations
def create_hypertables(engine) -> None:
    """Create TimescaleDB hypertables for all metric tables."""

    from sqlalchemy import text

    tables = [
        ("metrics_liq_flow", "timestamp"),
        ("metrics_stableflow", "timestamp"),
        ("metrics_funding_basis", "timestamp"),
        ("metrics_chain_congestion", "timestamp"),
        ("metrics_staking_yield", "timestamp"),
        ("metrics_mev_pressure", "timestamp"),
        ("metrics_cex_dex_flow", "timestamp"),
    ]

    with engine.connect() as conn:
        for table_name, time_column in tables:
            # Check if already a hypertable
            result = conn.execute(
                text(f"SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = '{table_name}'")
            )
            if not result.fetchone():
                # Create hypertable
                conn.execute(
                    text(f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE)")
                )
                conn.commit()


def create_continuous_aggregates(engine) -> None:
    """Create continuous aggregates for real-time analytics."""

    with engine.connect() as conn:
        # Example: Hourly aggregates for chain congestion
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_chain_congestion_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', timestamp) AS bucket,
                chain,
                AVG(avg_gas_price) AS avg_gas_price,
                AVG(block_fullness) AS block_fullness,
                MAX(pending_txs) AS max_pending_txs
            FROM metrics_chain_congestion
            GROUP BY bucket, chain
            WITH NO DATA;
        """))

        # Example: Daily aggregates for funding basis
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_funding_basis_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', timestamp) AS bucket,
                exchange,
                symbol,
                AVG(funding_rate) AS avg_funding_rate,
                MIN(funding_rate) AS min_funding_rate,
                MAX(funding_rate) AS max_funding_rate
            FROM metrics_funding_basis
            GROUP BY bucket, exchange, symbol
            WITH NO DATA;
        """))

        conn.commit()


# Model registry for easy access
ALL_MODELS = [
    LiqFlow,
    Stableflow,
    FundingBasis,
    ChainCongestion,
    StakingYield,
    MEVPressure,
    CEXDEXFlow,
]

MODEL_MAP = {
    "liq_flow": LiqFlow,
    "stableflow": Stableflow,
    "funding_basis": FundingBasis,
    "chain_congestion": ChainCongestion,
    "staking_yield": StakingYield,
    "mev_pressure": MEVPressure,
    "cex_dex_flow": CEXDEXFlow,
}
