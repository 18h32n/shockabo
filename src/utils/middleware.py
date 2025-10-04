"""FastAPI middleware for centralized error handling and request processing.

This module provides middleware components for:
- Global exception handling
- Request/response logging
- Performance monitoring
- Rate limiting support
- Error recovery and circuit breaker patterns
"""

import time
from typing import Any
from uuid import uuid4

import structlog
from fastapi import Request, Response, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from src.utils.error_handling import (
    ARCBaseException,
    ErrorCode,
    ErrorContext,
    ErrorLogger,
    ErrorSeverity,
    create_error_response,
    get_http_status_for_error_code,
)

logger = structlog.get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware for FastAPI applications."""

    def __init__(self, app, error_logger: ErrorLogger | None = None):
        super().__init__(app)
        self.error_logger = error_logger or ErrorLogger()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and handle any exceptions."""
        request_id = str(uuid4())
        request.state.request_id = request_id

        # Add request ID to structured logging context
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.time()

        try:
            # Log incoming request
            logger.info(
                "request_started",
                method=request.method,
                url=str(request.url),
                client_host=getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
                user_agent=request.headers.get('user-agent', 'unknown')
            )

            response = await call_next(request)

            # Log successful response
            processing_time = time.time() - start_time
            logger.info(
                "request_completed",
                status_code=response.status_code,
                processing_time_ms=processing_time * 1000
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except ARCBaseException as exc:
            # Handle our custom exceptions
            processing_time = time.time() - start_time

            # Add context information
            if not exc.context:
                exc.context = ErrorContext()
            exc.context.request_id = request_id
            exc.context.additional_data.update({
                "processing_time_ms": processing_time * 1000,
                "endpoint": str(request.url.path),
                "method": request.method,
            })

            # Determine HTTP status code
            http_status = get_http_status_for_error_code(exc.error_code)

            # Create error response
            response = create_error_response(exc, http_status, request)
            response.headers["X-Request-ID"] = request_id

            logger.warning(
                "request_failed_arc_exception",
                error_code=exc.error_code.value,
                error_id=exc.error_id,
                status_code=http_status,
                processing_time_ms=processing_time * 1000
            )

            return response

        except Exception as exc:
            # Handle unexpected exceptions
            processing_time = time.time() - start_time

            # Create context for the error
            context = ErrorContext(
                request_id=request_id,
                additional_data={
                    "processing_time_ms": processing_time * 1000,
                    "endpoint": str(request.url.path),
                    "method": request.method,
                }
            )

            # Create ARC exception from generic exception
            arc_exc = ARCBaseException(
                message="An unexpected error occurred during request processing",
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                details=str(exc),
                severity=ErrorSeverity.HIGH,
                context=context,
                suggestions=[
                    "Please try the request again",
                    "Contact support if the problem persists",
                ]
            )

            response = create_error_response(arc_exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)
            response.headers["X-Request-ID"] = request_id

            logger.error(
                "request_failed_unexpected_exception",
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=500,
                processing_time_ms=processing_time * 1000,
                exc_info=True
            )

            return response

        finally:
            # Clear structured logging context
            structlog.contextvars.clear_contextvars()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging and performance monitoring."""

    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024 * 10,  # 10KB default
        excluded_paths: set | None = None,
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.excluded_paths = excluded_paths or {"/health", "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log request and response details."""
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()

        # Prepare request logging data
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_host": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
        }

        # Log request body if enabled
        if self.log_request_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    request_data["body"] = body.decode('utf-8', errors='ignore')
                else:
                    request_data["body"] = f"<Body too large: {len(body)} bytes>"
            except Exception as e:
                request_data["body_error"] = str(e)

        logger.info("detailed_request", **request_data)

        # Process request
        response = await call_next(request)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response logging data
        response_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "processing_time_ms": processing_time * 1000,
        }

        # Add performance category
        if processing_time < 0.1:
            response_data["performance"] = "fast"
        elif processing_time < 0.5:
            response_data["performance"] = "normal"
        elif processing_time < 2.0:
            response_data["performance"] = "slow"
        else:
            response_data["performance"] = "very_slow"

        # Log response body if enabled and not too large
        if self.log_response_body and hasattr(response, 'body'):
            try:
                if hasattr(response.body, '__len__') and len(response.body) <= self.max_body_size:
                    response_data["body"] = response.body.decode('utf-8', errors='ignore')
                else:
                    response_data["body"] = "<Body not logged>"
            except Exception as e:
                response_data["body_error"] = str(e)

        # Log based on status code
        if response.status_code >= 500:
            logger.error("detailed_response", **response_data)
        elif response.status_code >= 400:
            logger.warning("detailed_response", **response_data)
        else:
            logger.info("detailed_response", **response_data)

        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and alerting."""

    def __init__(
        self,
        app,
        slow_request_threshold: float = 2.0,  # seconds
        very_slow_request_threshold: float = 5.0,  # seconds
        enable_memory_monitoring: bool = False,
    ):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.very_slow_request_threshold = very_slow_request_threshold
        self.enable_memory_monitoring = enable_memory_monitoring

        # Performance metrics tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.slow_requests = 0
        self.very_slow_requests = 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Monitor request performance and collect metrics."""
        start_time = time.time()
        memory_before = None

        if self.enable_memory_monitoring:
            try:
                import os

                import psutil
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                logger.warning("psutil not available, memory monitoring disabled")
                self.enable_memory_monitoring = False

        try:
            response = await call_next(request)
        except Exception as e:
            # Still track performance for failed requests
            processing_time = time.time() - start_time
            self._update_metrics(request, processing_time, error=str(e))
            raise

        # Calculate performance metrics
        processing_time = time.time() - start_time
        memory_after = None
        memory_delta = None

        if self.enable_memory_monitoring and memory_before:
            try:
                import os

                import psutil
                process = psutil.Process(os.getpid())
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = memory_after - memory_before
            except Exception:
                pass

        # Update metrics
        self._update_metrics(request, processing_time, response.status_code, memory_delta)

        # Add performance headers
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        if memory_delta is not None:
            response.headers["X-Memory-Delta"] = f"{memory_delta:.2f}MB"

        # Log performance alerts
        request_id = getattr(request.state, 'request_id', 'unknown')

        if processing_time >= self.very_slow_request_threshold:
            logger.warning(
                "very_slow_request_alert",
                request_id=request_id,
                endpoint=request.url.path,
                processing_time_ms=processing_time * 1000,
                threshold_ms=self.very_slow_request_threshold * 1000,
                memory_delta_mb=memory_delta
            )
        elif processing_time >= self.slow_request_threshold:
            logger.info(
                "slow_request_alert",
                request_id=request_id,
                endpoint=request.url.path,
                processing_time_ms=processing_time * 1000,
                threshold_ms=self.slow_request_threshold * 1000,
                memory_delta_mb=memory_delta
            )

        return response

    def _update_metrics(
        self,
        request: Request,
        processing_time: float,
        status_code: int | None = None,
        memory_delta: float | None = None,
        error: str | None = None,
    ):
        """Update internal performance metrics."""
        self.request_count += 1
        self.total_processing_time += processing_time

        if processing_time >= self.very_slow_request_threshold:
            self.very_slow_requests += 1
        elif processing_time >= self.slow_request_threshold:
            self.slow_requests += 1

        # Log metrics periodically
        if self.request_count % 100 == 0:
            avg_time = self.total_processing_time / self.request_count
            slow_rate = (self.slow_requests + self.very_slow_requests) / self.request_count

            logger.info(
                "performance_metrics_summary",
                total_requests=self.request_count,
                average_processing_time_ms=avg_time * 1000,
                slow_request_rate=slow_rate,
                slow_requests=self.slow_requests,
                very_slow_requests=self.very_slow_requests
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        if self.request_count == 0:
            return {
                "total_requests": 0,
                "average_processing_time_ms": 0.0,
                "slow_request_rate": 0.0,
                "slow_requests": 0,
                "very_slow_requests": 0,
            }

        avg_time = self.total_processing_time / self.request_count
        slow_rate = (self.slow_requests + self.very_slow_requests) / self.request_count

        return {
            "total_requests": self.request_count,
            "average_processing_time_ms": avg_time * 1000,
            "slow_request_rate": slow_rate,
            "slow_requests": self.slow_requests,
            "very_slow_requests": self.very_slow_requests,
            "slow_threshold_ms": self.slow_request_threshold * 1000,
            "very_slow_threshold_ms": self.very_slow_request_threshold * 1000,
        }


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for handling downstream service failures."""

    def __init__(
        self,
        app,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Implement circuit breaker pattern."""
        # Check circuit breaker state
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("circuit_breaker_half_open")
            else:
                # Circuit is open, return service unavailable
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": "Service temporarily unavailable",
                        "circuit_breaker_state": "open",
                        "retry_after": self.recovery_timeout,
                    }
                )

        try:
            response = await call_next(request)

            # Success - reset failure count if we were in half_open state
            if self.state == "half_open":
                self._reset()
                logger.info("circuit_breaker_reset_success")

            return response

        except self.expected_exception:
            self._record_failure()

            if self.state == "half_open":
                # Failed in half_open, go back to open
                self.state = "open"
                logger.warning("circuit_breaker_failed_half_open")

            # Re-raise the exception
            raise

    def _record_failure(self):
        """Record a failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold and self.state == "closed":
            self.state = "open"
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _reset(self):
        """Reset the circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }


# Global middleware instances for easy access
error_handling_middleware = None
performance_monitoring_middleware = None


def setup_middleware(app, enable_detailed_logging: bool = False):
    """Setup all middleware components for the FastAPI application.

    Args:
        app: FastAPI application instance
        enable_detailed_logging: Whether to enable detailed request/response logging
    """
    global error_handling_middleware, performance_monitoring_middleware

    # Performance monitoring middleware (innermost)
    performance_monitoring_middleware = PerformanceMonitoringMiddleware(
        app,
        slow_request_threshold=2.0,
        very_slow_request_threshold=5.0,
        enable_memory_monitoring=True
    )
    app.add_middleware(PerformanceMonitoringMiddleware)

    # Request logging middleware
    if enable_detailed_logging:
        app.add_middleware(
            RequestLoggingMiddleware,
            log_request_body=False,  # Disabled by default for security
            log_response_body=False,  # Disabled by default for performance
            max_body_size=1024 * 5,  # 5KB limit
        )

    # Circuit breaker middleware (for external service calls)
    app.add_middleware(
        CircuitBreakerMiddleware,
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Exception
    )

    # Error handling middleware (outermost)
    error_handling_middleware = ErrorHandlingMiddleware(app)
    app.add_middleware(ErrorHandlingMiddleware)

    logger.info(
        "middleware_configured",
        error_handling=True,
        performance_monitoring=True,
        detailed_logging=enable_detailed_logging,
        circuit_breaker=True
    )


def get_performance_metrics() -> dict[str, Any] | None:
    """Get current performance metrics from middleware."""
    if performance_monitoring_middleware:
        return performance_monitoring_middleware.get_metrics()
    return None
