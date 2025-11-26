"""
Data processing and aggregation service.
Handles data transformation, aggregation, and caching for all metrics.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal

from app.models.schemas import (
    LiqFlowMetric, StableflowMetric, FundingBasisMetric,
    ChainCongestionMetric, StakingYieldMetric, MEVPressureMetric,
    CEXDEXFlowMetric, DashboardSummary
)
from app.models.database import (
    LiqFlow, Stableflow, FundingBasis, ChainCongestion,
    StakingYield, MEVPressure, CEXDEXFlow
)
from app.core.database import get_db_session
# from app.core.cache import cache_manager  # Commented out for testing
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Service for processing and aggregating DeFi metrics data.
    Provides high-level data operations with caching and optimization.
    """

    def __init__(self):
        self.cache_ttl = 300  # 5 minutes default cache TTL

    def get_latest_metric(
        self,
        metric_class,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Get the latest data for a specific metric.

        Args:
            metric_class: SQLAlchemy model class
            filters: Additional filters to apply
            use_cache: Whether to check cache first

        Returns:
            Latest metric data or None
        """
        # Simplified version without cache for testing
        try:
            with get_db_session() as db:
                query = db.query(metric_class).order_by(metric_class.timestamp.desc())

                if filters:
                    for key, value in filters.items():
                        if hasattr(metric_class, key):
                            query = query.filter(getattr(metric_class, key) == value)

                result = query.first()

                if result:
                    # Convert to dict
                    data_dict = {column.name: getattr(result, column.name)
                               for column in result.__table__.columns}

                    return data_dict

        except Exception as e:
            logger.error(f"Error fetching latest {metric_class.__tablename__}: {e}")

        return None

    def get_metric_history(
        self,
        metric_class,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific metric.

        Args:
            metric_class: SQLAlchemy model class
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            filters: Additional filters
            limit: Maximum number of records
            use_cache: Whether to check cache first

        Returns:
            List of metric data dictionaries
        """
        # Simplified version without cache for testing
        try:
            with get_db_session() as db:
                query = db.query(metric_class).order_by(metric_class.timestamp.desc())

                if from_timestamp:
                    query = query.filter(metric_class.timestamp >= from_timestamp)
                if to_timestamp:
                    query = query.filter(metric_class.timestamp <= to_timestamp)

                if filters:
                    for key, value in filters.items():
                        if hasattr(metric_class, key):
                            query = query.filter(getattr(metric_class, key) == value)

                query = query.limit(limit)
                results = query.all()

                # Convert to list of dicts
                data_list = []
                for result in results:
                    data_dict = {column.name: getattr(result, column.name)
                               for column in result.__table__.columns}
                    data_list.append(data_dict)

                return data_list

        except Exception as e:
            logger.error(f"Error fetching history for {metric_class.__tablename__}: {e}")

        return []

    def get_dashboard_summary(self) -> DashboardSummary:
        """
        Get a summary of the latest data for all metrics.

        Returns:
            DashboardSummary with latest data for all metrics
        """
        # Simplified version without cache for testing
        summary_data = {}

        # Liquidity Flow
        liq_flow_data = self.get_latest_metric(LiqFlow)
        if liq_flow_data:
            summary_data["liq_flow"] = LiqFlowMetric(**liq_flow_data)

        # Stableflow
        stableflow_data = self.get_latest_metric(Stableflow)
        if stableflow_data:
            summary_data["stableflow"] = StableflowMetric(**stableflow_data)

        # Funding Basis
        funding_data = self.get_latest_metric(FundingBasis)
        if funding_data:
            summary_data["funding_basis"] = FundingBasisMetric(**funding_data)

        # Chain Congestion
        congestion_data = self.get_latest_metric(ChainCongestion)
        if congestion_data:
            summary_data["chain_congestion"] = ChainCongestionMetric(**congestion_data)

        # Staking Yield
        staking_data = self.get_latest_metric(StakingYield)
        if staking_data:
            summary_data["staking_yield"] = StakingYieldMetric(**staking_data)

        # MEV Pressure
        mev_data = self.get_latest_metric(MEVPressure)
        if mev_data:
            summary_data["mev_pressure"] = MEVPressureMetric(**mev_data)

        # CEX/DEX Flow
        cex_dex_data = self.get_latest_metric(CEXDEXFlow)
        if cex_dex_data:
            summary_data["cex_dex_flow"] = CEXDEXFlowMetric(**cex_dex_data)

        # Create dashboard summary
        from app.models.schemas import MetricMetadata
        metadata = MetricMetadata(
            timestamp=datetime.utcnow(),
            source="data_processor",
            cache_hit=False
        )

        dashboard = DashboardSummary(**summary_data, metadata=metadata)

        return dashboard

    def get_historical_data(
        self,
        metrics: List[str],
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical data for multiple metrics.

        Args:
            metrics: List of metric names
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            limit: Maximum records per metric

        Returns:
            Dictionary with metric names as keys and data lists as values
        """
        result = {}

        # Map metric names to model classes
        metric_map = {
            "liq_flow": LiqFlow,
            "stableflow": Stableflow,
            "funding_basis": FundingBasis,
            "chain_congestion": ChainCongestion,
            "staking_yield": StakingYield,
            "mev_pressure": MEVPressure,
            "cex_dex_flow": CEXDEXFlow,
        }

        for metric_name in metrics:
            if metric_name in metric_map:
                metric_class = metric_map[metric_name]
                data = self.get_metric_history(
                    metric_class=metric_class,
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                    limit=limit
                )
                result[metric_name] = data
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return result

    async def aggregate_metric_data(
        self,
        metric_class,
        group_by: List[str],
        time_bucket: str = "1 hour",
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metric data by time buckets and groupings.

        Args:
            metric_class: SQLAlchemy model class
            group_by: Columns to group by (in addition to time)
            time_bucket: TimescaleDB time bucket (e.g., '1 hour', '1 day')
            from_timestamp: Start timestamp
            to_timestamp: End timestamp

        Returns:
            List of aggregated data points
        """
        try:
            with get_db_session() as db:
                # Build the query with TimescaleDB time_bucket function
                time_column = metric_class.timestamp

                # Create select columns
                select_columns = [
                    f"time_bucket('{time_bucket}', {time_column}) as bucket_time"
                ]

                # Add group by columns
                for col in group_by:
                    if hasattr(metric_class, col):
                        select_columns.append(f"{col}")

                # Add aggregate functions for numeric columns
                numeric_columns = []
                for column in metric_class.__table__.columns:
                    if column.type.python_type in (int, float, Decimal):
                        numeric_columns.append(column.name)

                for col_name in numeric_columns:
                    select_columns.extend([
                        f"AVG({col_name}) as avg_{col_name}",
                        f"MIN({col_name}) as min_{col_name}",
                        f"MAX({col_name}) as max_{col_name}"
                    ])

                # Build the raw SQL query
                select_clause = ", ".join(select_columns)
                group_clause = ", ".join(["bucket_time"] + group_by)

                query_str = f"""
                    SELECT {select_clause}
                    FROM {metric_class.__tablename__}
                    WHERE timestamp >= %s AND timestamp <= %s
                    GROUP BY {group_clause}
                    ORDER BY bucket_time DESC
                """

                # Execute the query
                params = [from_timestamp or datetime.min, to_timestamp or datetime.max]
                raw_results = db.execute(query_str, params).fetchall()

                # Convert to dictionaries
                results = []
                for row in raw_results:
                    result_dict = dict(row)
                    results.append(result_dict)

                return results

        except Exception as e:
            logger.error(f"Error aggregating {metric_class.__tablename__}: {e}")
            return []

    async def get_data_freshness_status(self) -> Dict[str, Any]:
        """
        Get data freshness status for all metrics.

        Returns:
            Dictionary with freshness information for each metric
        """
        status = {}
        current_time = datetime.utcnow()

        metric_configs = [
            ("liq_flow", LiqFlow, 900),  # 15 minutes
            ("stableflow", Stableflow, 900),  # 15 minutes
            ("funding_basis", FundingBasis, 300),  # 5 minutes
            ("chain_congestion", ChainCongestion, 60),  # 1 minute
            ("staking_yield", StakingYield, 3600),  # 1 hour
            ("mev_pressure", MEVPressure, 1800),  # 30 minutes
            ("cex_dex_flow", CEXDEXFlow, 3600),  # 1 hour
        ]

        for metric_name, metric_class, max_age_seconds in metric_configs:
            latest_data = await self.get_latest_metric(metric_class, use_cache=False)

            if latest_data:
                last_updated = latest_data.get('timestamp')
                if last_updated:
                    age_seconds = (current_time - last_updated).total_seconds()
                    is_fresh = age_seconds <= max_age_seconds
                else:
                    age_seconds = None
                    is_fresh = False
            else:
                last_updated = None
                age_seconds = None
                is_fresh = False

            status[metric_name] = {
                "last_updated": last_updated.isoformat() if last_updated else None,
                "age_seconds": age_seconds,
                "is_fresh": is_fresh,
                "max_age_seconds": max_age_seconds
            }

        return status
