"""FastAPI application for ARC Prize 2025 evaluation system with rate limiting protection.

This module creates the main FastAPI application instance with comprehensive
rate limiting middleware to prevent DoS attacks on evaluation endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.api.middleware.rate_limiter import RateLimitMiddleware, RateLimitConfig
from src.adapters.api.middleware.auth import setup_authentication_middleware
from src.adapters.api.routes.evaluation import router as evaluation_router
from src.adapters.api.routes.auth import router as auth_router
from src.infrastructure.config import get_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Get configuration
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management with startup and shutdown tasks."""
    # Startup tasks
    logger.info("application_starting", platform=config.get_platform_info().platform.value)
    
    # Initialize services here if needed
    # await initialize_services()
    
    yield
    
    # Shutdown tasks
    logger.info("application_shutting_down")
    
    # Cleanup services here if needed
    # await cleanup_services()


# Create FastAPI application
app = FastAPI(
    title="ARC Prize 2025 Evaluation API",
    description="High-performance evaluation system for ARC (Abstraction and Reasoning Corpus) tasks with real-time monitoring",
    version="1.0.0",
    docs_url="/docs" if config.is_development() else None,
    redoc_url="/redoc" if config.is_development() else None,
    lifespan=lifespan,
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.arc-prize.example.com", "description": "Production server"},
    ],
)


# Configure rate limiting
def get_rate_limit_config() -> RateLimitConfig:
    """Get rate limiting configuration from YAML config and environment variables."""
    # Environment variables take precedence over config file
    return RateLimitConfig(
        # Evaluation endpoints - more restrictive
        evaluation_requests_per_minute=int(
            os.getenv("RATE_LIMIT_EVALUATION_RPM") or 
            config.get("security.rate_limit.evaluation.requests_per_minute", 100)
        ),
        evaluation_requests_per_hour=int(
            os.getenv("RATE_LIMIT_EVALUATION_RPH") or 
            config.get("security.rate_limit.evaluation.requests_per_hour", 1000)
        ),
        evaluation_burst_size=int(
            os.getenv("RATE_LIMIT_EVALUATION_BURST") or 
            config.get("security.rate_limit.evaluation.burst_size", 10)
        ),
        
        # Dashboard endpoints - more permissive
        dashboard_requests_per_minute=int(
            os.getenv("RATE_LIMIT_DASHBOARD_RPM") or 
            config.get("security.rate_limit.dashboard.requests_per_minute", 1000)
        ),
        dashboard_requests_per_hour=int(
            os.getenv("RATE_LIMIT_DASHBOARD_RPH") or 
            config.get("security.rate_limit.dashboard.requests_per_hour", 10000)
        ),
        dashboard_burst_size=int(
            os.getenv("RATE_LIMIT_DASHBOARD_BURST") or 
            config.get("security.rate_limit.dashboard.burst_size", 50)
        ),
        
        # WebSocket connections
        websocket_connections_per_ip=int(
            os.getenv("RATE_LIMIT_WS_CONNECTIONS") or 
            config.get("security.rate_limit.websocket.connections_per_ip", 5)
        ),
        
        # Authentication endpoints
        auth_requests_per_minute=int(
            os.getenv("RATE_LIMIT_AUTH_RPM") or 
            config.get("security.rate_limit.auth.requests_per_minute", 20)
        ),
        auth_requests_per_hour=int(
            os.getenv("RATE_LIMIT_AUTH_RPH") or 
            config.get("security.rate_limit.auth.requests_per_hour", 200)
        ),
        
        # Global settings
        enable_rate_limiting=(
            os.getenv("RATE_LIMIT_ENABLED", "").lower() == "true" if os.getenv("RATE_LIMIT_ENABLED") 
            else config.get("security.rate_limit.enabled", True)
        ),
        redis_url=os.getenv("REDIS_URL") or config.get("security.rate_limit.redis_url", "redis://localhost:6379/0"),
        use_redis_backend=(
            os.getenv("RATE_LIMIT_USE_REDIS", "").lower() == "true" if os.getenv("RATE_LIMIT_USE_REDIS") 
            else config.get("security.rate_limit.use_redis", False)
        ),
        
        # Advanced settings from config
        enable_adaptive_limiting=config.get("security.rate_limit.adaptive_limiting", True),
        suspicious_threshold_multiplier=config.get("security.rate_limit.suspicious_threshold_multiplier", 2.0),
        whitelist_ips=config.get("security.rate_limit.whitelist_ips", []),
        blacklist_ips=config.get("security.rate_limit.blacklist_ips", []),
    )


# Add authentication middleware
setup_authentication_middleware(app)

# Add rate limiting middleware
rate_limit_config = get_rate_limit_config()
if rate_limit_config.enable_rate_limiting:
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    logger.info(
        "rate_limiting_enabled",
        evaluation_rpm=rate_limit_config.evaluation_requests_per_minute,
        dashboard_rpm=rate_limit_config.dashboard_requests_per_minute,
        use_redis=rate_limit_config.use_redis_backend,
    )
else:
    logger.warning("rate_limiting_disabled")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api.cors.allowed_origins", ["http://localhost:3000"]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler with structured logging."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    
    # Don't expose internal errors in production
    if config.is_development():
        detail = f"Internal server error: {str(exc)}"
    else:
        detail = "Internal server error"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail, "error_id": "internal_server_error"},
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "platform": config.get_platform_info().platform.value,
        "version": "1.0.0",
    }


# Include routers
app.include_router(auth_router)
app.include_router(evaluation_router)


# Root endpoint
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "ARC Prize 2025 Evaluation API",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs" if config.is_development() else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "src.adapters.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=config.is_development(),
        log_config=None,  # Use structlog configuration
        access_log=False,  # Handled by middleware
    )