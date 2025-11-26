"""
Implementation Example: Chain Congestion Collector

This file demonstrates how to implement a complete collector for the Chain Congestion metric.
Use this as a template for implementing the other 6 metrics.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
import asyncio

from app.collectors.base import BaseCollector, MetricType, CollectionConfig, CollectionResult
from app.models.schemas import ChainCongestionMetric
from app.providers.base import RESTProvider, ProviderConfig, ProviderType
from app.core.database import get_db_session
from app.models.database import ChainCongestion
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Step 1: Create a Provider for the data source
class EtherscanProvider(RESTProvider):
    """
    Provider for Etherscan API.
    Handles rate limiting and authentication for Ethereum blockchain data.
    """

    def __init__(self, api_key: str):
        config = ProviderConfig(
            name="etherscan",
            provider_type=ProviderType.BLOCKCHAIN_EXPLORER,
            base_url="https://api.etherscan.io/api",
            api_key=api_key,
            rate_limit=5,  # Etherscan free tier: 5 calls/second
            timeout=30,
            headers={"User-Agent": "DeFi-Pipeline/1.0"}
        )
        super().__init__(config)

    async def get_gas_oracle(self) -> Dict[str, Any]:
        """Get current gas prices from Etherscan."""
        params = {
            "module": "gastracker",
            "action": "gasoracle",
            "apikey": self.config.api_key
        }

        response = await self.fetch_data("", params=params)

        if response.success:
            return response.data
        else:
            logger.error(f"Etherscan gas oracle failed: {response.error}")
            return {}

    async def get_pending_transactions(self) -> Dict[str, Any]:
        """Get pending transaction count (txpool status)."""
        params = {
            "module": "proxy",
            "action": "txpool_status",
            "apikey": self.config.api_key
        }

        response = await self.fetch_data("", params=params)

        if response.success:
            return response.data
        else:
            logger.error(f"Etherscan txpool status failed: {response.error}")
            return {}

    async def get_block_by_number(self, block_number: str) -> Dict[str, Any]:
        """Get block information by block number."""
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": block_number,
            "boolean": "true",
            "apikey": self.config.api_key
        }

        response = await self.fetch_data("", params=params)

        if response.success:
            return response.data
        else:
            logger.error(f"Etherscan block info failed: {response.error}")
            return {}


# Step 2: Create the Collector
class ChainCongestionCollector(BaseCollector):
    """
    Collector for Ethereum chain congestion metrics.

    Collects:
    - Average gas prices (SafeLow, Standard, Fast, Rapid)
    - Block utilization percentage
    - Pending transaction counts
    - Base fee and priority fee data (EIP-1559)
    """

    def __init__(self, config: CollectionConfig, provider: EtherscanProvider):
        super().__init__(MetricType.CHAIN_CONGESTION, config)
        self.provider = provider
        self.chain = "ethereum"  # Could be parameterized for multi-chain

    async def collect(self) -> CollectionResult:
        """
        Collect chain congestion data from multiple sources.

        Strategy:
        1. Get gas prices from Etherscan gas oracle
        2. Get pending transaction counts
        3. Get latest block data for utilization
        4. Combine and validate data
        """
        try:
            logger.info(f"Starting chain congestion collection for {self.chain}")

            # Collect data from multiple sources concurrently
            gas_data, pending_data, block_data = await asyncio.gather(
                self._collect_gas_data(),
                self._collect_pending_data(),
                self._collect_block_data()
            )

            # Combine data into metric
            metric = self._create_congestion_metric(gas_data, pending_data, block_data)

            if metric:
                logger.info(f"Successfully collected congestion data: gas_price={metric.avg_gas_price}")
                return CollectionResult(
                    success=True,
                    data=[metric],
                    timestamp=datetime.utcnow(),
                    source=self.provider.config.name
                )
            else:
                return CollectionResult(
                    success=False,
                    error="Failed to create valid congestion metric",
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Chain congestion collection failed: {e}", exc_info=True)
            return CollectionResult(
                success=False,
                error=f"Collection error: {str(e)}",
                timestamp=datetime.utcnow()
            )

    async def _collect_gas_data(self) -> Dict[str, Any]:
        """Collect gas price data."""
        gas_data = await self.provider.get_gas_oracle()

        if gas_data.get("status") == "1" and "result" in gas_data:
            result = gas_data["result"]
            return {
                "safe_gas_price": int(result.get("SafeGasPrice", 0)),
                "propose_gas_price": int(result.get("ProposeGasPrice", 0)),
                "fast_gas_price": int(result.get("FastGasPrice", 0)),
                "base_fee": float(result.get("suggestBaseFee", 0)),
                "gas_used_ratio": result.get("gasUsedRatio", "").split(",")
            }

        return {}

    async def _collect_pending_data(self) -> Dict[str, Any]:
        """Collect pending transaction data."""
        pending_data = await self.provider.get_pending_transactions()

        if "result" in pending_data:
            result = pending_data["result"]
            return {
                "pending": int(result.get("pending", "0x0"), 16),
                "queued": int(result.get("queued", "0x0"), 16)
            }

        return {"pending": 0, "queued": 0}

    async def _collect_block_data(self) -> Dict[str, Any]:
        """Collect latest block data for utilization calculation."""
        # Get latest block number (simplified - would need to get actual latest)
        # In production, you'd get this from the provider
        block_data = await self.provider.get_block_by_number("latest")

        if "result" in block_data:
            result = block_data["result"]
            gas_used = int(result.get("gasUsed", "0x0"), 16)
            gas_limit = int(result.get("gasLimit", "0x0"), 16)

            return {
                "gas_used": gas_used,
                "gas_limit": gas_limit,
                "transaction_count": len(result.get("transactions", [])),
                "block_fullness": gas_used / gas_limit if gas_limit > 0 else 0
            }

        return {"gas_used": 0, "gas_limit": 0, "transaction_count": 0, "block_fullness": 0}

    def _create_congestion_metric(
        self,
        gas_data: Dict[str, Any],
        pending_data: Dict[str, Any],
        block_data: Dict[str, Any]
    ) -> Optional[ChainCongestionMetric]:
        """Create a ChainCongestionMetric from collected data."""

        try:
            # Calculate average gas price (weighted average of different speeds)
            gas_prices = [
                gas_data.get("safe_gas_price", 0),
                gas_data.get("propose_gas_price", 0),
                gas_data.get("fast_gas_price", 0)
            ]
            valid_prices = [p for p in gas_prices if p > 0]

            if not valid_prices:
                logger.warning("No valid gas prices collected")
                return None

            avg_gas_price = sum(valid_prices) / len(valid_prices)

            # Calculate block fullness
            block_fullness = block_data.get("block_fullness", 0)
            if not (0 <= block_fullness <= 1):
                block_fullness = 0

            # Get pending transaction count
            pending_txs = pending_data.get("pending", 0) + pending_data.get("queued", 0)

            # Create metric
            metric = ChainCongestionMetric(
                timestamp=datetime.utcnow(),
                chain=self.chain,
                avg_gas_price=Decimal(str(avg_gas_price)),
                block_fullness=Decimal(str(block_fullness)),
                pending_txs=pending_txs,
                base_fee=Decimal(str(gas_data.get("base_fee", 0))) if gas_data.get("base_fee") else None,
                priority_fee=None,  # Would need additional data source
                block_time=None     # Would need historical block data
            )

            return metric

        except Exception as e:
            logger.error(f"Failed to create congestion metric: {e}")
            return None

    async def validate_data(self, data: Any) -> bool:
        """Validate collected congestion data."""
        if not isinstance(data, list) or len(data) != 1:
            return False

        metric = data[0]
        if not isinstance(metric, ChainCongestionMetric):
            return False

        # Validate required fields
        if metric.avg_gas_price <= 0:
            return False

        if not (0 <= metric.block_fullness <= 1):
            return False

        if metric.pending_txs < 0:
            return False

        return True


# Step 3: Database Storage Function
def store_chain_congestion_metrics(metrics: List[ChainCongestionMetric]) -> bool:
    """
    Store chain congestion metrics in database.

    Args:
        metrics: List of ChainCongestionMetric objects

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_session() as db:
            for metric in metrics:
                # Convert Pydantic model to SQLAlchemy model
                db_metric = ChainCongestion(
                    timestamp=metric.timestamp,
                    chain=metric.chain,
                    avg_gas_price=float(metric.avg_gas_price),
                    block_fullness=float(metric.block_fullness),
                    pending_txs=metric.pending_txs,
                    base_fee=float(metric.base_fee) if metric.base_fee else None,
                    priority_fee=float(metric.priority_fee) if metric.priority_fee else None,
                    block_time=float(metric.block_time) if metric.block_time else None
                )

                db.add(db_metric)

            db.commit()
            logger.info(f"Stored {len(metrics)} chain congestion metrics")
            return True

    except Exception as e:
        logger.error(f"Failed to store chain congestion metrics: {e}")
        return False


# Step 4: API Integration Helper
def get_chain_congestion_data(
    chain: str = "ethereum",
    limit: int = 100,
    from_timestamp: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve chain congestion data for API responses.

    Args:
        chain: Blockchain identifier
        limit: Maximum number of records
        from_timestamp: Start timestamp for historical data

    Returns:
        List of congestion data dictionaries
    """
    try:
        with get_db_session() as db:
            query = db.query(ChainCongestion)\
                     .filter(ChainCongestion.chain == chain)\
                     .order_by(ChainCongestion.timestamp.desc())

            if from_timestamp:
                query = query.filter(ChainCongestion.timestamp >= from_timestamp)

            results = query.limit(limit).all()

            # Convert to dictionaries for API response
            data = []
            for result in results:
                data.append({
                    "timestamp": result.timestamp.isoformat(),
                    "chain": result.chain,
                    "avg_gas_price": float(result.avg_gas_price),
                    "block_fullness": float(result.block_fullness),
                    "pending_txs": result.pending_txs,
                    "base_fee": float(result.base_fee) if result.base_fee else None,
                    "priority_fee": float(result.priority_fee) if result.priority_fee else None,
                    "block_time": float(result.block_time) if result.block_time else None
                })

            return data

    except Exception as e:
        logger.error(f"Failed to retrieve chain congestion data: {e}")
        return []


# Step 5: Example Usage
async def example_usage():
    """Example of how to use the Chain Congestion Collector."""

    # Initialize provider
    provider = EtherscanProvider(api_key="YOUR_ETHERSCAN_API_KEY")

    # Initialize collector
    config = CollectionConfig(
        metric_type=MetricType.CHAIN_CONGESTION,
        priority=CollectionPriority.HIGH,
        interval_minutes=1,
        timeout_seconds=30,
        max_retries=3
    )

    collector = ChainCongestionCollector(config, provider)

    # Collect data
    result = await collector.collect()

    if result.success:
        print(f"✅ Collected {len(result.data)} congestion metrics")

        # Store in database
        if store_chain_congestion_metrics(result.data):
            print("✅ Metrics stored successfully")

        # Retrieve for API
        api_data = get_chain_congestion_data()
        print(f"📊 Retrieved {len(api_data)} records for API")

    else:
        print(f"❌ Collection failed: {result.error}")


if __name__ == "__main__":
    # Run example (requires API key)
    # asyncio.run(example_usage())
    print("Chain Congestion Collector implementation example")
    print("See docs/developer_guide.md for detailed implementation instructions")
