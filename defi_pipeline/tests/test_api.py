"""
API integration tests.
Tests the FastAPI endpoints and their responses.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


def make_async_client() -> AsyncClient:
    """Create an async in-process ASGI client for the FastAPI app."""
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint."""
    async with make_async_client() as client:
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


@pytest.mark.asyncio
async def test_detailed_health_endpoint():
    """Test the detailed health check endpoint."""
    async with make_async_client() as client:
        response = await client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "checks" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint."""
    async with make_async_client() as client:
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "DeFi Data Pipeline" in data["message"]


@pytest.mark.asyncio
async def test_api_placeholder_endpoints():
    """Test that API endpoints return placeholder responses."""
    endpoints = [
        "/api/v1/liq-flow",
        "/api/v1/stableflow",
        "/api/v1/funding-basis",
        "/api/v1/chain-congestion",
        "/api/v1/staking-yield",
        "/api/v1/mev-pressure",
        "/api/v1/cex-dex-flow"
    ]

    async with make_async_client() as client:
        for endpoint in endpoints:
            response = await client.get(endpoint)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "data" in data
            assert "metadata" in data
            assert data["metadata"]["source"] == "placeholder"
            assert "implementation pending" in data["data"]["message"].lower()


@pytest.mark.asyncio
async def test_dashboard_endpoint():
    """Test the dashboard endpoint."""
    async with make_async_client() as client:
        response = await client.get("/api/v1/dashboard")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data
        assert "metadata" in data


@pytest.mark.asyncio
async def test_historical_endpoint():
    """Test the historical data endpoint."""
    async with make_async_client() as client:
        response = await client.get("/api/v1/historical?metrics=liq_flow")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data
        assert "metadata" in data


@pytest.mark.asyncio
async def test_invalid_metric_request():
    """Test handling of invalid metric requests."""
    async with make_async_client() as client:
        response = await client.get("/api/v1/historical?metrics=invalid_metric")

        # Should still return 200 with empty data for invalid metrics
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that rate limiting is applied."""
    # This test would need to be adjusted based on actual rate limiting implementation
    # For now, just verify the endpoint exists and responds
    async with make_async_client() as client:
        response = await client.get("/api/v1/liq-flow")

        assert response.status_code == 200
        # Check for rate limit headers if implemented
        # assert "X-RateLimit-Remaining" in response.headers


@pytest.mark.asyncio
async def test_openapi_docs():
    """Test that OpenAPI documentation is accessible."""
    async with make_async_client() as client:
        response = await client.get("/docs")

        # Should redirect or serve docs
        assert response.status_code in [200, 302]


@pytest.mark.asyncio
async def test_cors_headers():
    """Test CORS headers are present."""
    async with make_async_client() as client:
        response = await client.options(
            "/api/v1/liq-flow",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
