"""
API v1 router that aggregates all metric endpoints.
Provides the main routing structure for version 1 of the API.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime

from app.models.schemas import APIResponse, MetricMetadata, DashboardSummary
from app.services.data_processor import DataProcessor
# from app.core.cache import cache_manager  # Commented out for testing
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Create the main API router
api_router = APIRouter()

# Import endpoint modules (they will be created)
# from app.api.v1.endpoints import (
#     liq_flow,
#     stableflow,
#     funding_basis,
#     chain_congestion,
#     staking_yield,
#     mev_pressure,
#     cex_dex_flow
# )

# Data processor instance
data_processor = DataProcessor()


def create_api_response(
    data: Any,
    source: str = "cache",
    cache_hit: bool = False,
    freshness_seconds: int = None
) -> APIResponse:
    """Create a standardized API response."""
    metadata = MetricMetadata(
        timestamp=datetime.utcnow(),
        source=source,
        cache_hit=cache_hit,
        freshness_seconds=freshness_seconds
    )

    return APIResponse(
        success=True,
        data=data,
        metadata=metadata
    )


# Placeholder endpoints - these will be replaced with actual implementations
@api_router.get("/liq-flow")
async def get_liq_flow(chain: str = None, from_timestamp: str = None, to_timestamp: str = None):
    """Get liquidity flow metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "Liquidity flow endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/stableflow")
async def get_stableflow(coin: str = None, from_timestamp: str = None, to_timestamp: str = None):
    """Get stablecoin flow metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "Stableflow endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/funding-basis")
async def get_funding_basis(exchange: str = None, symbol: str = None):
    """Get funding basis metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "Funding basis endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/chain-congestion")
async def get_chain_congestion(chain: str = None):
    """Get chain congestion metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "Chain congestion endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/staking-yield")
async def get_staking_yield(chain: str = None):
    """Get staking yield metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "Staking yield endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/mev-pressure")
async def get_mev_pressure(chain: str = None):
    """Get MEV pressure metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "MEV pressure endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/cex-dex-flow")
async def get_cex_dex_flow(bridge: str = None):
    """Get CEX/DEX flow metrics."""
    # Placeholder implementation
    return create_api_response(
        data={"message": "CEX/DEX flow endpoint - implementation pending"},
        source="placeholder"
    )


@api_router.get("/dashboard")
def get_dashboard():
    """Get dashboard summary with latest data for all metrics."""
    try:
        # Get latest data from data processor
        dashboard_data = data_processor.get_dashboard_summary()

        # Convert to dict for API response
        response_data = dashboard_data.model_dump() if hasattr(dashboard_data, 'model_dump') else dashboard_data

        return create_api_response(
            data=response_data,
            source="data_processor",
            cache_hit=False
        )

    except Exception as e:
        logger.error(f"Dashboard endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")


@api_router.get("/historical")
def get_historical(
    metrics: str = None,
    from_timestamp: str = None,
    to_timestamp: str = None,
    limit: int = 100
):
    """Get historical data for specified metrics."""
    try:
        # Parse metrics parameter
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",")]
        else:
            # Default to all metrics
            metric_list = [
                "liq_flow", "stableflow", "funding_basis",
                "chain_congestion", "staking_yield", "mev_pressure", "cex_dex_flow"
            ]

        # Parse timestamps
        from_ts = None
        to_ts = None

        if from_timestamp:
            try:
                from_ts = datetime.fromisoformat(from_timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid from_timestamp format")

        if to_timestamp:
            try:
                to_ts = datetime.fromisoformat(to_timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid to_timestamp format")

        # Get historical data
        historical_data = data_processor.get_historical_data(
            metrics=metric_list,
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            limit=limit
        )

        return create_api_response(
            data=historical_data,
            source="data_processor",
            cache_hit=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")


# Future endpoint registrations will go here:
# api_router.include_router(liq_flow.router, prefix="/liq-flow", tags=["Liquidity Flow"])
# api_router.include_router(stableflow.router, prefix="/stableflow", tags=["Stableflow"])
# api_router.include_router(funding_basis.router, prefix="/funding-basis", tags=["Funding Basis"])
# api_router.include_router(chain_congestion.router, prefix="/chain-congestion", tags=["Chain Congestion"])
# api_router.include_router(staking_yield.router, prefix="/staking-yield", tags=["Staking Yield"])
# api_router.include_router(mev_pressure.router, prefix="/mev-pressure", tags=["MEV Pressure"])
# api_router.include_router(cex_dex_flow.router, prefix="/cex-dex-flow", tags=["CEX/DEX Flow"])
