"""Health check API routes for system monitoring and status reporting.

This module provides comprehensive health check endpoints for monitoring
system status, component health, and performance metrics.
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from src.domain.health_models import HealthStatus
from src.domain.services.health_service import get_health_service

logger = structlog.get_logger(__name__)

# Create health check router
router = APIRouter(prefix="/health", tags=["health"])

# Get health service instance
health_service = get_health_service()


@router.get("/", status_code=status.HTTP_200_OK)
async def basic_health_check():
    """Basic health check endpoint.
    
    This endpoint provides a quick health status check of critical components:
    - Database connectivity
    - Cache system
    - Memory usage
    
    Returns:
        JSON response with basic health status and HTTP status code:
        - 200: System is healthy
        - 503: System is unhealthy or degraded
    """
    try:
        # Perform basic health check
        health_summary = await health_service.perform_basic_health_check()
        
        # Determine HTTP status code based on overall health
        http_status = _get_http_status_from_health(health_summary.overall_status)
        
        response_data = {
            "status": health_summary.overall_status.value,
            "timestamp": health_summary.timestamp.isoformat(),
            "version": health_summary.version,
            "environment": health_summary.environment,
            "uptime_seconds": health_summary.uptime_seconds,
            "components": {
                comp.component_name: {
                    "status": comp.status.value,
                    "response_time_ms": comp.response_time_ms
                }
                for comp in health_summary.components
            }
        }
        
        logger.info(
            "basic_health_check_completed",
            status=health_summary.overall_status.value,
            components_count=len(health_summary.components),
            http_status=http_status
        )
        
        return JSONResponse(
            status_code=http_status,
            content=response_data
        )
        
    except Exception as e:
        logger.error("basic_health_check_failed", error=str(e), exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Health check failed",
                "message": str(e)
            }
        )


@router.get("/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check():
    """Detailed health check endpoint.
    
    This endpoint provides comprehensive health information including:
    - All system components status
    - Performance metrics (CPU, memory, disk usage)
    - Component-specific details (database connections, cache stats, etc.)
    - Resource utilization
    - Error information
    
    Returns:
        JSON response with detailed health status and metrics
    """
    try:
        # Perform detailed health check
        health_summary = await health_service.perform_detailed_health_check()
        
        # Determine HTTP status code based on overall health
        http_status = _get_http_status_from_health(health_summary.overall_status)
        
        response_data = health_summary.to_dict()
        
        logger.info(
            "detailed_health_check_completed",
            status=health_summary.overall_status.value,
            components_count=len(health_summary.components),
            http_status=http_status
        )
        
        return JSONResponse(
            status_code=http_status,
            content=response_data
        )
        
    except Exception as e:
        logger.error("detailed_health_check_failed", error=str(e), exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Detailed health check failed",
                "message": str(e),
                "components": [],
                "performance_metrics": {},
                "detailed_health": {}
            }
        )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe():
    """Kubernetes readiness probe endpoint.
    
    This endpoint indicates whether the service is ready to accept traffic.
    It checks critical dependencies required for normal operation:
    - Database connectivity
    - Cache availability
    - Required external services
    
    Returns:
        HTTP 200: Service is ready
        HTTP 503: Service is not ready
    """
    try:
        is_ready = await health_service.perform_readiness_check()
        
        if is_ready:
            logger.debug("readiness_check_passed")
            return PlainTextResponse(
                status_code=status.HTTP_200_OK,
                content="OK"
            )
        else:
            logger.warning("readiness_check_failed")
            return PlainTextResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="Not Ready"
            )
            
    except Exception as e:
        logger.error("readiness_probe_error", error=str(e), exc_info=True)
        return PlainTextResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content="Error"
        )


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """Kubernetes liveness probe endpoint.
    
    This endpoint indicates whether the service is alive and running.
    It performs minimal checks to verify the service hasn't crashed:
    - Basic memory availability
    - Service responsiveness
    - Core functionality
    
    Returns:
        HTTP 200: Service is alive
        HTTP 503: Service is dead (should be restarted)
    """
    try:
        is_alive = await health_service.perform_liveness_check()
        
        if is_alive:
            logger.debug("liveness_check_passed")
            return PlainTextResponse(
                status_code=status.HTTP_200_OK,
                content="OK"
            )
        else:
            logger.warning("liveness_check_failed")
            return PlainTextResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="Not Alive"
            )
            
    except Exception as e:
        logger.error("liveness_probe_error", error=str(e), exc_info=True)
        return PlainTextResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content="Error"
        )


@router.get("/startup", status_code=status.HTTP_200_OK)
async def startup_probe():
    """Kubernetes startup probe endpoint.
    
    This endpoint indicates whether the service has finished starting up.
    It's used for slow-starting containers to avoid being killed during startup.
    
    Returns:
        HTTP 200: Service has started successfully
        HTTP 503: Service is still starting up
    """
    try:
        # For this implementation, startup is considered complete if basic health check passes
        health_summary = await health_service.perform_basic_health_check()
        
        # Consider startup complete if system is at least not unhealthy
        is_started = health_summary.overall_status != HealthStatus.UNHEALTHY
        
        if is_started:
            logger.debug("startup_check_passed", status=health_summary.overall_status.value)
            return PlainTextResponse(
                status_code=status.HTTP_200_OK,
                content="OK"
            )
        else:
            logger.warning("startup_check_failed", status=health_summary.overall_status.value)
            return PlainTextResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="Starting"
            )
            
    except Exception as e:
        logger.error("startup_probe_error", error=str(e), exc_info=True)
        return PlainTextResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content="Error"
        )


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def performance_metrics():
    """Performance metrics endpoint.
    
    This endpoint provides current performance metrics for monitoring:
    - CPU and memory usage
    - Disk utilization
    - Active connections
    - Request rates
    - Error rates
    - Response times
    
    Returns:
        JSON response with performance metrics
    """
    try:
        # Get detailed health check for metrics
        health_summary = await health_service.perform_detailed_health_check()
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": health_summary.performance_metrics.to_dict(),
            "component_metrics": {
                comp.component_name: {
                    "status": comp.status.value,
                    "response_time_ms": comp.response_time_ms,
                    "last_checked": comp.last_checked.isoformat(),
                    "error_message": comp.error_message
                }
                for comp in health_summary.components
            }
        }
        
        logger.info("performance_metrics_retrieved")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=metrics_data
        )
        
    except Exception as e:
        logger.error("performance_metrics_failed", error=str(e), exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to retrieve performance metrics",
                "message": str(e)
            }
        )


@router.get("/component/{component_name}", status_code=status.HTTP_200_OK)
async def component_health_check(component_name: str):
    """Individual component health check endpoint.
    
    Args:
        component_name: Name of the component to check (database, cache, wandb, etc.)
    
    Returns:
        JSON response with component-specific health information
    """
    try:
        # Get detailed health check
        health_summary = await health_service.perform_detailed_health_check()
        
        # Find the requested component
        component_result = health_summary.get_component_by_name(component_name)
        
        if not component_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component '{component_name}' not found or not monitored"
            )
        
        # Determine HTTP status code
        http_status = _get_http_status_from_health(component_result.status)
        
        response_data = component_result.to_dict()
        
        # Add component-specific detailed information
        if component_name in health_summary.detailed_health:
            response_data["detailed_info"] = health_summary.detailed_health[component_name]
        
        logger.info(
            "component_health_check_completed",
            component_name=component_name,
            status=component_result.status.value,
            response_time_ms=component_result.response_time_ms
        )
        
        return JSONResponse(
            status_code=http_status,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "component_health_check_failed",
            component_name=component_name,
            error=str(e),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "component_name": component_name,
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Component health check failed",
                "message": str(e)
            }
        )


@router.get("/config", status_code=status.HTTP_200_OK)
async def health_check_config():
    """Health check configuration endpoint.
    
    Returns current health check configuration for debugging and monitoring.
    
    Returns:
        JSON response with health check configuration
    """
    try:
        config_data = health_service.config.to_dict()
        
        logger.debug("health_check_config_retrieved")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "timestamp": datetime.now().isoformat(),
                "health_check_config": config_data,
                "registered_checks": list(health_service._last_check_results.keys())
            }
        )
        
    except Exception as e:
        logger.error("health_check_config_failed", error=str(e), exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to retrieve health check configuration",
                "message": str(e)
            }
        )


def _get_http_status_from_health(health_status: HealthStatus) -> int:
    """Convert health status to appropriate HTTP status code.
    
    Args:
        health_status: Health status enum value
        
    Returns:
        HTTP status code
    """
    if health_status == HealthStatus.HEALTHY:
        return status.HTTP_200_OK
    elif health_status == HealthStatus.DEGRADED:
        return status.HTTP_200_OK  # Still operational, just degraded performance
    elif health_status == HealthStatus.UNHEALTHY:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    else:  # UNKNOWN
        return status.HTTP_503_SERVICE_UNAVAILABLE


# Health check status summary for easy integration
@router.get("/status", status_code=status.HTTP_200_OK)
async def health_status_summary():
    """Simple health status summary endpoint.
    
    Provides a minimal response for simple health checks and load balancers.
    
    Returns:
        JSON response with overall status
    """
    try:
        health_summary = await health_service.perform_basic_health_check()
        
        return JSONResponse(
            status_code=_get_http_status_from_health(health_summary.overall_status),
            content={
                "status": health_summary.overall_status.value,
                "timestamp": health_summary.timestamp.isoformat(),
                "uptime": health_summary.uptime_seconds
            }
        )
        
    except Exception as e:
        logger.error("health_status_summary_failed", error=str(e))
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )