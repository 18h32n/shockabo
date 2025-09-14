"""Main FastAPI application entry point for ARC Prize 2025 system.

This module creates and configures the FastAPI application with all
necessary components including health checks, evaluation endpoints,
and monitoring capabilities.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# Add source root to path for absolute imports
_src_root = Path(__file__).parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from src.adapters.api.routes import create_api_router
from src.domain.health_models import HealthCheckConfig
from src.domain.services.health_service import HealthService
from src.infrastructure.config import get_config, initialize_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Application startup time for health checks
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("arc_system_starting_up")

    try:
        # Initialize configuration
        config = initialize_config()
        logger.info("configuration_initialized", platform=config.platform.value)

        # Initialize health service with configuration
        health_config = HealthCheckConfig()

        # Load health check configuration if available
        health_config_data = config.get("health_check", {})
        if health_config_data:
            health_config.enabled = health_config_data.get("enabled", True)
            health_config.check_interval_seconds = health_config_data.get("check_interval_seconds", 30)
            health_config.timeout_seconds = health_config_data.get("timeout_seconds", 5)
            health_config.max_failures = health_config_data.get("max_failures", 3)
            health_config.degraded_threshold_ms = health_config_data.get("degraded_threshold_ms", 1000.0)
            health_config.unhealthy_threshold_ms = health_config_data.get("unhealthy_threshold_ms", 5000.0)
            health_config.components_to_check = health_config_data.get("components_to_check",
                ["database", "cache", "wandb", "storage", "memory", "cpu"])

        # Initialize health service
        health_service = HealthService(health_config)

        # Store references in app state
        app.state.config = config
        app.state.health_service = health_service
        app.state.startup_time = _startup_time

        logger.info("arc_system_startup_completed", startup_duration_ms=(time.time() - _startup_time) * 1000)

    except Exception as e:
        logger.error("arc_system_startup_failed", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("arc_system_shutting_down")

    try:
        # Cleanup resources
        if hasattr(app.state, 'health_service'):
            # Perform any health service cleanup if needed
            pass

        logger.info("arc_system_shutdown_completed")

    except Exception as e:
        logger.error("arc_system_shutdown_failed", error=str(e), exc_info=True)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    # Create FastAPI application
    app = FastAPI(
        title="ARC Prize 2025 System",
        description="Comprehensive system for ARC Prize 2025 competition with evaluation framework and monitoring",
        version="1.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on environment
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add trusted host middleware
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0").split(",")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log HTTP requests with timing and status information."""
        start_time = time.time()

        # Skip logging for health check endpoints to reduce noise
        if request.url.path.startswith("/health") and request.url.path in ["/health/live", "/health/ready"]:
            response = await call_next(request)
            return response

        try:
            response = await call_next(request)

            process_time = (time.time() - start_time) * 1000

            logger.info(
                "http_request",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time_ms=process_time,
                user_agent=request.headers.get("user-agent", ""),
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = (time.time() - start_time) * 1000

            logger.error(
                "http_request_error",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time_ms=process_time,
                exc_info=True
            )

            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "message": str(e)}
            )

    # Include API routers
    api_router = create_api_router()
    app.include_router(api_router)

    # Add root health check endpoint for load balancers
    @app.get("/", status_code=200, include_in_schema=False)
    async def root_health_check():
        """Root endpoint for basic health checking by load balancers."""
        return {"status": "healthy", "service": "arc-prize-2025", "version": "1.3.0"}

    # Add system info endpoint
    @app.get("/info", tags=["system"])
    async def system_info():
        """Get system information and configuration."""
        config = get_config()

        return {
            "service": "ARC Prize 2025 System",
            "version": "1.3.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "platform": config.platform.value,
            "uptime_seconds": time.time() - _startup_time,
            "resources": config.get_platform_info().to_dict() if hasattr(config.get_platform_info(), 'to_dict') else str(config.get_platform_info()),
            "features": {
                "health_checks": True,
                "evaluation_api": True,
                "websocket_support": True,
                "wandb_integration": True,
                "performance_monitoring": True
            }
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(
            "unhandled_exception",
            method=request.method,
            url=str(request.url),
            error=str(exc),
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": id(request)
            }
        )

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    # Configuration for different environments
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "development").lower() == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(
        "starting_uvicorn_server",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        server_header=False,  # Security: don't expose server info
        date_header=False     # Security: don't expose date header
    )
