"""
FastAPI application entry point for the DeFi Data Pipeline.
Provides REST API endpoints for real-time DeFi metrics.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import time

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # Commented out for testing
import structlog

from app.config import app_settings, monitoring_settings
from app.core.database import init_database, check_database_health
# from app.core.cache import init_cache, close_cache  # Commented out for testing
from app.core.exceptions import create_error_response, DeFiPipelineError
from app.utils.logger import setup_logging
from app.utils.rate_limiter import multi_rate_limiter
from app.api.v1.router import api_router
from app.utils.logger import get_logger

# Set up logging (commented out for testing)
# setup_logging()
logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting DeFi Data Pipeline...")

    try:
        # Initialize database
        init_database()
        logger.info("Database initialized")

        # Initialize cache
        # await init_cache()  # Commented out for testing
        # logger.info("Cache initialized")

        # Initialize collectors and providers would go here
        # (They will be initialized when needed)

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down DeFi Data Pipeline...")

    try:
        # Close cache connection
        # await close_cache()  # Commented out for testing
        # logger.info("Cache connection closed")

        # Close any other connections here
        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Application shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="DeFi Data Pipeline API",
    description="Scalable API for collecting and serving real-time DeFi market intelligence metrics",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",  # Always enable docs for development
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Commented out for testing - enable in production
# if app_settings.is_production:
#     app.add_middleware(
#         TrustedHostMiddleware,
#         allowed_hosts=["*"],  # Configure this properly in production
#     )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()

    # Get client info
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get('user-agent', 'unknown')

    logger.info(
        f"Request started: {request.method} {request.url} from {client_host}"
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"Request completed: {request.method} {request.url} -> {response.status_code} in {process_time:.3f}s"
        )

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url} - {str(e)} ({process_time:.3f}s)"
        )
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    error_response = create_error_response(exc)
    logger.error(
        f"Unhandled exception: {error_response.error} - {error_response.message} for {request.url}"
    )
    return JSONResponse(
        status_code=error_response.status_code,
        content=error_response.to_dict()
    )


# Rate limiting dependency
async def check_rate_limit(request: Request) -> None:
    """Check API rate limits."""
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'

    # Use client IP for rate limiting
    limiter = multi_rate_limiter.get_limiter(
        f"api:{client_host}",
        app_settings.rate_limit_per_minute
    )

    await limiter.wait_if_needed()


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": app.version
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check including database and cache status."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": app.version,
        "checks": {}
    }

    # Database health
    try:
        db_healthy = check_database_health()
        health_status["checks"]["database"] = "healthy" if db_healthy else "unhealthy"
        if not db_healthy:
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Cache health
    try:
        from app.core.cache import cache_manager
        cache_stats = await cache_manager.get_stats()
        health_status["checks"]["cache"] = "healthy" if cache_stats.get("connected") else "unhealthy"
        if not cache_stats.get("connected"):
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["cache"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status


@app.get("/status", tags=["Status"])
async def system_status():
    """Get system status including data freshness."""
    # This will be expanded with actual metric freshness checks
    return {
        "status": "operational",
        "timestamp": time.time(),
        "version": app.version,
        "message": "System is operational"
    }


# Prometheus metrics endpoint (commented out for testing)
# @app.get("/metrics", tags=["Monitoring"])
# async def metrics():
#     """Prometheus metrics endpoint."""
#     if not monitoring_settings.enable_metrics:
#         raise HTTPException(status_code=404, detail="Metrics not enabled")
#
#     return JSONResponse(
#         content=generate_latest().decode(),
#         media_type=CONTENT_TYPE_LATEST
#     )


# Include API routers
app.include_router(
    api_router,
    prefix="/api/v1",
    tags=["API v1"],
    dependencies=[Depends(check_rate_limit)]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeFi Data Pipeline API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "status": "/status"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=app_settings.host,
        port=app_settings.port,
        reload=app_settings.reload,
        log_level=app_settings.log_level.lower(),
        access_log=True
    )
