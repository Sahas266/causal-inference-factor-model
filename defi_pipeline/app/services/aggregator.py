"""
Cross-metric aggregation service.
Provides advanced analytics and correlations across different DeFi metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict

from app.models.database import (
    LiqFlow, Stableflow, FundingBasis, ChainCongestion,
    StakingYield, MEVPressure, CEXDEXFlow
)
from app.core.database import get_db_session
from app.core.cache import cache_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricAggregator:
    """
    Service for aggregating and correlating data across multiple DeFi metrics.
    Provides insights into market relationships and trends.
    """

    def __init__(self):
        self.cache_ttl = 600  # 10 minutes for aggregated data

    async def get_market_correlations(
        self,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        chains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate correlations between different metrics.

        Args:
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            chains: Specific chains to analyze

        Returns:
            Dictionary with correlation matrices and insights
        """
        cache_key = f"correlations:{from_timestamp}:{to_timestamp}:{str(chains)}"

        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # Get time-series data for correlation analysis
            time_series_data = await self._get_time_series_for_correlation(
                from_timestamp, to_timestamp, chains
            )

            # Calculate correlations
            correlations = self._calculate_correlations(time_series_data)

            # Generate insights
            insights = self._generate_correlation_insights(correlations)

            result = {
                "correlations": correlations,
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat(),
                "period": {
                    "from": from_timestamp.isoformat() if from_timestamp else None,
                    "to": to_timestamp.isoformat() if to_timestamp else None,
                },
                "chains": chains
            }

            # Cache result
            await cache_manager.set(cache_key, result, ttl=self.cache_ttl)

            return result

        except Exception as e:
            logger.error(f"Error calculating market correlations: {e}")
            return {"error": str(e)}

    async def get_chain_health_score(
        self,
        chain: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate a comprehensive health score for a blockchain.

        Args:
            chain: Chain identifier
            hours: Hours of historical data to analyze

        Returns:
            Health score and component metrics
        """
        cache_key = f"health_score:{chain}:{hours}"

        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Get relevant metrics for the chain
            metrics_data = await self._get_chain_metrics(chain, start_time, end_time)

            # Calculate health score components
            congestion_score = self._calculate_congestion_score(metrics_data.get("congestion", []))
            liquidity_score = self._calculate_liquidity_score(metrics_data.get("liquidity", []))
            stability_score = self._calculate_stability_score(metrics_data.get("stableflow", []))
            mev_score = self._calculate_mev_score(metrics_data.get("mev", []))

            # Overall health score (weighted average)
            weights = {
                "congestion": 0.3,
                "liquidity": 0.25,
                "stability": 0.25,
                "mev": 0.2
            }

            overall_score = (
                congestion_score * weights["congestion"] +
                liquidity_score * weights["liquidity"] +
                stability_score * weights["stability"] +
                mev_score * weights["mev"]
            )

            result = {
                "chain": chain,
                "overall_score": round(overall_score, 2),
                "components": {
                    "congestion": round(congestion_score, 2),
                    "liquidity": round(liquidity_score, 2),
                    "stability": round(stability_score, 2),
                    "mev": round(mev_score, 2)
                },
                "weights": weights,
                "period_hours": hours,
                "timestamp": datetime.utcnow().isoformat()
            }

            await cache_manager.set(cache_key, result, ttl=self.cache_ttl)
            return result

        except Exception as e:
            logger.error(f"Error calculating health score for {chain}: {e}")
            return {"error": str(e)}

    async def get_market_sentiment_indicators(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate market sentiment indicators from aggregated metrics.

        Args:
            hours: Hours of data to analyze

        Returns:
            Sentiment indicators and trends
        """
        cache_key = f"sentiment:{hours}"

        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Get aggregated market data
            market_data = await self._get_aggregated_market_data(start_time, end_time)

            # Calculate sentiment indicators
            sentiment = self._calculate_sentiment_indicators(market_data)

            result = {
                "sentiment_score": sentiment["overall"],
                "indicators": sentiment["indicators"],
                "trends": sentiment["trends"],
                "period_hours": hours,
                "timestamp": datetime.utcnow().isoformat()
            }

            await cache_manager.set(cache_key, result, ttl=self.cache_ttl)
            return result

        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {"error": str(e)}

    async def get_anomaly_detection(
        self,
        metric_type: str,
        hours: int = 168,  # 1 week
        threshold: float = 2.0  # Standard deviations
    ) -> Dict[str, Any]:
        """
        Detect anomalies in metric data using statistical methods.

        Args:
            metric_type: Type of metric to analyze
            hours: Hours of historical data
            threshold: Anomaly detection threshold in standard deviations

        Returns:
            Anomaly detection results
        """
        cache_key = f"anomalies:{metric_type}:{hours}:{threshold}"

        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # Get historical data
            data = await self._get_metric_data_for_anomaly_detection(
                metric_type, hours
            )

            # Detect anomalies
            anomalies = self._detect_statistical_anomalies(data, threshold)

            result = {
                "metric_type": metric_type,
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies[:50],  # Limit results
                "total_points": len(data),
                "anomaly_rate": len(anomalies) / len(data) if data else 0,
                "threshold_std": threshold,
                "period_hours": hours,
                "timestamp": datetime.utcnow().isoformat()
            }

            await cache_manager.set(cache_key, result, ttl=self.cache_ttl)
            return result

        except Exception as e:
            logger.error(f"Error detecting anomalies for {metric_type}: {e}")
            return {"error": str(e)}

    # Private helper methods

    async def _get_time_series_for_correlation(
        self,
        from_timestamp: Optional[datetime],
        to_timestamp: Optional[datetime],
        chains: Optional[List[str]]
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get aligned time series data for correlation analysis."""
        # This would align data points by timestamp and create time series
        # Implementation would depend on specific correlation needs
        return {}

    def _calculate_correlations(self, time_series_data: Dict[str, List]) -> Dict[str, float]:
        """Calculate Pearson correlation coefficients between metrics."""
        # Placeholder for correlation calculation
        return {
            "congestion_liquidity": 0.0,
            "funding_stableflow": 0.0,
            "mev_congestion": 0.0
        }

    def _generate_correlation_insights(self, correlations: Dict[str, float]) -> List[str]:
        """Generate human-readable insights from correlation data."""
        insights = []

        for pair, correlation in correlations.items():
            if abs(correlation) > 0.7:
                direction = "positive" if correlation > 0 else "negative"
                strength = "strong" if abs(correlation) > 0.8 else "moderate"
                insights.append(
                    f"{strength} {direction} correlation ({correlation:.2f}) "
                    f"between {pair.replace('_', ' and ')}"
                )

        return insights

    async def _get_chain_metrics(
        self, chain: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics data for a specific chain."""
        from app.services.data_processor import DataProcessor

        processor = DataProcessor()

        return {
            "congestion": await processor.get_metric_history(
                ChainCongestion, start_time, end_time, {"chain": chain}, limit=1000
            ),
            "liquidity": await processor.get_metric_history(
                LiqFlow, start_time, end_time, {"chain": chain}, limit=1000
            ),
            "stableflow": await processor.get_metric_history(
                Stableflow, start_time, end_time, {"chain": chain}, limit=1000
            ),
            "mev": await processor.get_metric_history(
                MEVPressure, start_time, end_time, {"chain": chain}, limit=1000
            )
        }

    def _calculate_congestion_score(self, congestion_data: List[Dict[str, Any]]) -> float:
        """Calculate congestion health score (lower congestion = higher score)."""
        if not congestion_data:
            return 50.0  # Neutral score

        # Normalize gas prices and block fullness
        scores = []
        for data in congestion_data[-24:]:  # Last 24 data points
            gas_price = float(data.get("avg_gas_price", 0))
            block_fullness = float(data.get("block_fullness", 0))

            # Normalize gas price (assuming 100 gwei is high, 10 gwei is low)
            gas_score = max(0, min(100, 100 - (gas_price - 10) / (100 - 10) * 100))

            # Block fullness score (lower fullness = higher score)
            fullness_score = 100 - (block_fullness * 100)

            scores.append((gas_score + fullness_score) / 2)

        return sum(scores) / len(scores) if scores else 50.0

    def _calculate_liquidity_score(self, liquidity_data: List[Dict[str, Any]]) -> float:
        """Calculate liquidity health score."""
        if not liquidity_data:
            return 50.0

        # Look at net inflow trends
        inflows = [float(d.get("net_inflow", 0)) for d in liquidity_data[-24:]]
        if inflows:
            positive_trend = sum(1 for i in range(1, len(inflows)) if inflows[i] > inflows[i-1])
            return (positive_trend / len(inflows)) * 100

        return 50.0

    def _calculate_stability_score(self, stableflow_data: List[Dict[str, Any]]) -> float:
        """Calculate stability score from stablecoin flows."""
        if not stableflow_data:
            return 50.0

        # Look at net flows - stable flows are better
        net_flows = [abs(float(d.get("net_flow", 0))) for d in stableflow_data[-24:]]
        if net_flows:
            avg_flow = sum(net_flows) / len(net_flows)
            # Lower average flows = more stable = higher score
            return max(0, min(100, 100 - avg_flow / 1000000))  # Arbitrary scaling

        return 50.0

    def _calculate_mev_score(self, mev_data: List[Dict[str, Any]]) -> float:
        """Calculate MEV health score (lower MEV pressure = higher score)."""
        if not mev_data:
            return 50.0

        # Lower MEV extraction is better
        mev_values = [float(d.get("mev_extracted", 0)) for d in mev_data[-24:]]
        if mev_values:
            avg_mev = sum(mev_values) / len(mev_values)
            # Lower MEV = higher score
            return max(0, min(100, 100 - avg_mev / 1000000))  # Arbitrary scaling

        return 50.0

    async def _get_aggregated_market_data(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get aggregated market data for sentiment analysis."""
        # Implementation would aggregate data across all chains
        return {}

    def _calculate_sentiment_indicators(self, market_data: Dict[str, List]) -> Dict[str, Any]:
        """Calculate market sentiment from aggregated data."""
        # Placeholder implementation
        return {
            "overall": 50.0,
            "indicators": {
                "liquidity_trend": "neutral",
                "volatility": "moderate",
                "congestion_level": "normal"
            },
            "trends": ["Market showing normal activity"]
        }

    async def _get_metric_data_for_anomaly_detection(
        self, metric_type: str, hours: int
    ) -> List[float]:
        """Get metric data for anomaly detection."""
        # Placeholder - would fetch actual metric data
        return []

    def _detect_statistical_anomalies(
        self, data: List[float], threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(data) < 10:
            return []

        # Simple z-score based anomaly detection
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

        anomalies = []
        for i, value in enumerate(data):
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > threshold:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                        "timestamp": datetime.utcnow() - timedelta(hours=len(data) - i)
                    })

        return anomalies
