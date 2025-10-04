"""Centralized error handling system for the evaluation framework.

This module provides custom exception classes, error response formats,
and middleware for consistent error handling across all modules.
"""

import traceback
from enum import Enum
from typing import Any

import structlog
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Import from comprehensive error handling for compatibility
try:
    from .comprehensive_error_handling import ErrorContext, ErrorSeverity
except ImportError:
    # Fallback definitions if comprehensive_error_handling is not available
    from dataclasses import dataclass
    from enum import Enum as FallbackEnum

    class ErrorSeverity(FallbackEnum):
        """Error severity levels (fallback)."""
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    @dataclass
    class ErrorContext:
        """Error context information (fallback)."""
        operation: str
        additional_data: dict[str, Any] | None = None

logger = structlog.get_logger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for the evaluation framework."""

    # Authentication & Authorization
    AUTH_MISSING_TOKEN = "AUTH_001"
    AUTH_INVALID_TOKEN = "AUTH_002"
    AUTH_EXPIRED_TOKEN = "AUTH_003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_004"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"
    RATE_LIMIT_QUOTA_EXCEEDED = "RATE_002"

    # Validation
    VALIDATION_REQUIRED_FIELD = "VAL_001"
    VALIDATION_INVALID_FORMAT = "VAL_002"
    VALIDATION_OUT_OF_RANGE = "VAL_003"
    VALIDATION_DUPLICATE_VALUE = "VAL_004"

    # Task Processing
    TASK_NOT_FOUND = "TASK_001"
    TASK_INVALID_INPUT = "TASK_002"
    TASK_PROCESSING_FAILED = "TASK_003"
    TASK_EVALUATION_ERROR = "TASK_004"

    # Experiment Management
    EXPERIMENT_NOT_FOUND = "EXP_001"
    EXPERIMENT_ALREADY_RUNNING = "EXP_002"
    EXPERIMENT_FAILED = "EXP_003"
    EXPERIMENT_CANCELLED = "EXP_004"

    # Database
    DATABASE_CONNECTION_ERROR = "DB_001"
    DATABASE_QUERY_ERROR = "DB_002"
    DATABASE_CONSTRAINT_VIOLATION = "DB_003"
    DATABASE_TRANSACTION_FAILED = "DB_004"

    # External Services
    WANDB_CONNECTION_ERROR = "EXT_001"
    WANDB_AUTH_ERROR = "EXT_002"
    WANDB_QUOTA_EXCEEDED = "EXT_003"
    WANDB_API_ERROR = "EXT_004"

    # System
    SYSTEM_RESOURCE_EXHAUSTED = "SYS_001"
    SYSTEM_TIMEOUT = "SYS_002"
    SYSTEM_INTERNAL_ERROR = "SYS_003"
    SYSTEM_CONFIGURATION_ERROR = "SYS_004"

    # WebSocket
    WEBSOCKET_CONNECTION_FAILED = "WS_001"
    WEBSOCKET_AUTH_REQUIRED = "WS_002"
    WEBSOCKET_PROTOCOL_ERROR = "WS_003"
    WEBSOCKET_MAX_CONNECTIONS = "WS_004"


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = None
    message: str
    code: str | None = None
    context: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error: bool = True
    code: ErrorCode
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
    timestamp: str
    recovery_suggestions: list[str] | None = None


class BaseEvaluationError(Exception):
    """Base exception for all evaluation framework errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: list[ErrorDetail] | None = None,
        recovery_suggestions: list[str] | None = None,
        context: dict[str, Any] | None = None
    ):
        """Initialize evaluation error.

        Args:
            message: Human-readable error message
            code: Standardized error code
            details: Optional detailed error information
            recovery_suggestions: Optional suggestions for error recovery
            context: Optional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or []
        self.recovery_suggestions = recovery_suggestions or []
        self.context = context or {}


class ARCBaseException(BaseEvaluationError):
    """ARC-specific base exception class.

    This is the main exception class used throughout the ARC framework.
    It extends BaseEvaluationError with ARC-specific functionality.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SYSTEM_INTERNAL_ERROR,
        severity: str | None = None,
        context: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
        retry_after: int | None = None,
        cause: Exception | None = None,
        **kwargs
    ):
        """Initialize ARC base exception.

        Args:
            message: Human-readable error message
            error_code: Standardized error code
            severity: Error severity level (for compatibility)
            context: Error context information
            suggestions: Recovery suggestions
            retry_after: Suggested retry delay in seconds
            cause: Underlying cause exception
            **kwargs: Additional keyword arguments for compatibility
        """
        # Convert context to appropriate format if needed
        formatted_context = context or {}
        if hasattr(context, 'additional_data'):
            formatted_context = context.additional_data

        super().__init__(
            message=message,
            code=error_code,
            recovery_suggestions=suggestions,
            context=formatted_context
        )

        # Additional ARC-specific attributes
        self.severity = severity
        self.retry_after = retry_after
        self.cause = cause
        self.error_code = error_code  # Alias for compatibility


class AuthenticationError(BaseEvaluationError):
    """Authentication and authorization errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.AUTH_INVALID_TOKEN, **kwargs):
        super().__init__(message, code, **kwargs)


AuthenticationException = AuthenticationError
EvaluationException = BaseEvaluationError


class ValidationError(BaseEvaluationError):
    """Data validation errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.VALIDATION_INVALID_FORMAT, **kwargs):
        super().__init__(message, code, **kwargs)


class TaskProcessingError(BaseEvaluationError):
    """Task processing and evaluation errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.TASK_PROCESSING_FAILED, **kwargs):
        super().__init__(message, code, **kwargs)




class ExperimentError(BaseEvaluationError):
    """Experiment management errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.EXPERIMENT_FAILED, **kwargs):
        super().__init__(message, code, **kwargs)


class DatabaseError(BaseEvaluationError):
    """Database operation errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.DATABASE_QUERY_ERROR, **kwargs):
        super().__init__(message, code, **kwargs)


class ExternalServiceError(BaseEvaluationError):
    """External service integration errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.WANDB_CONNECTION_ERROR, **kwargs):
        super().__init__(message, code, **kwargs)


class SystemError(BaseEvaluationError):
    """System-level errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.SYSTEM_INTERNAL_ERROR, **kwargs):
        super().__init__(message, code, **kwargs)


class WebSocketError(BaseEvaluationError):
    """WebSocket connection errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.WEBSOCKET_CONNECTION_FAILED, **kwargs):
        super().__init__(message, code, **kwargs)


class DataNotFoundException(ARCBaseException):
    """Exception raised when requested data cannot be found."""

    def __init__(
        self,
        data_type: str,
        identifier: str,
        context: dict[str, Any] | None = None,
        **kwargs
    ):
        """Initialize data not found exception.

        Args:
            data_type: Type of data that was not found (e.g., "task", "model", "file")
            identifier: Identifier of the missing data
            context: Additional context information
            **kwargs: Additional arguments passed to ARCBaseException
        """
        message = f"{data_type.capitalize()} '{identifier}' not found"

        super().__init__(
            message=message,
            error_code=ErrorCode.TASK_NOT_FOUND,
            context=context,
            suggestions=[
                f"Verify that {data_type} '{identifier}' exists",
                f"Check {data_type} identifier spelling and format",
                f"Ensure you have access to the {data_type}",
                "Refresh data sources if needed"
            ],
            **kwargs
        )

        self.data_type = data_type
        self.identifier = identifier


# Aliases for backward compatibility
TaskNotFoundException = DataNotFoundException
DataNotFoundError = DataNotFoundException


class ErrorLogger:
    """Error logging utility for ARC framework.

    Provides structured error logging with context and severity tracking.
    This is a compatibility layer that works with the ARC error framework.
    """

    def __init__(self, logger_name: str = __name__):
        """Initialize error logger.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = structlog.get_logger(logger_name)
        self.errors = []

    def log_error(
        self,
        error: ARCBaseException | Exception,
        context: dict[str, Any] | None = None,
        severity: str = "medium"
    ):
        """Log an error with context and severity.

        Args:
            error: The error to log
            context: Additional context information
            severity: Error severity level
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "timestamp": structlog.get_logger().info.__globals__.get('time', __import__('time')).time()
        }

        # Add context information
        if context:
            error_data["context"] = context

        # Handle ARC-specific error attributes
        if isinstance(error, ARCBaseException):
            error_data.update({
                "error_code": error.error_code.value if hasattr(error.error_code, 'value') else str(error.error_code),
                "severity": error.severity or severity,
                "recovery_suggestions": error.recovery_suggestions,
                "arc_context": error.context
            })

        # Store for analysis
        self.errors.append(error_data)

        # Log based on severity
        if severity.lower() in ["critical", "high"]:
            self.logger.error("arc_error", **error_data)
        else:
            self.logger.warning("arc_error", **error_data)

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of logged errors.

        Returns:
            Dictionary containing error statistics and recent errors
        """
        if not self.errors:
            return {"total_errors": 0, "recent_errors": []}

        severity_counts = {}
        for error in self.errors:
            severity = error.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_errors": len(self.errors),
            "severity_distribution": severity_counts,
            "recent_errors": self.errors[-5:],  # Last 5 errors
            "error_types": list({error["error_type"] for error in self.errors})
        }


class ErrorHandler:
    """Centralized error handling utilities."""

    @staticmethod
    def create_error_response(
        error: BaseEvaluationError | Exception,
        request_id: str | None = None
    ) -> ErrorResponse:
        """Create standardized error response.

        Args:
            error: Error to format
            request_id: Optional request ID for tracking

        Returns:
            Formatted error response
        """
        from datetime import datetime

        if isinstance(error, BaseEvaluationError):
            return ErrorResponse(
                code=error.code,
                message=error.message,
                details=error.details,
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                recovery_suggestions=error.recovery_suggestions
            )

        # Handle standard exceptions
        if isinstance(error, HTTPException):
            return ErrorResponse(
                code=ErrorCode.SYSTEM_INTERNAL_ERROR,
                message=error.detail if isinstance(error.detail, str) else str(error.detail),
                request_id=request_id,
                timestamp=datetime.now().isoformat()
            )

        # Generic exception
        return ErrorResponse(
            code=ErrorCode.SYSTEM_INTERNAL_ERROR,
            message=str(error),
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            recovery_suggestions=["Check system logs", "Retry the operation", "Contact support if issue persists"]
        )

    @staticmethod
    def get_http_status_code(error_code: ErrorCode) -> int:
        """Get appropriate HTTP status code for error code.

        Args:
            error_code: Error code

        Returns:
            HTTP status code
        """
        status_map = {
            # 400 Bad Request
            ErrorCode.VALIDATION_REQUIRED_FIELD: 400,
            ErrorCode.VALIDATION_INVALID_FORMAT: 400,
            ErrorCode.VALIDATION_OUT_OF_RANGE: 400,
            ErrorCode.VALIDATION_DUPLICATE_VALUE: 400,
            ErrorCode.TASK_INVALID_INPUT: 400,
            ErrorCode.WEBSOCKET_PROTOCOL_ERROR: 400,

            # 401 Unauthorized
            ErrorCode.AUTH_MISSING_TOKEN: 401,
            ErrorCode.AUTH_INVALID_TOKEN: 401,
            ErrorCode.AUTH_EXPIRED_TOKEN: 401,
            ErrorCode.WANDB_AUTH_ERROR: 401,
            ErrorCode.WEBSOCKET_AUTH_REQUIRED: 401,

            # 403 Forbidden
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: 403,

            # 404 Not Found
            ErrorCode.TASK_NOT_FOUND: 404,
            ErrorCode.EXPERIMENT_NOT_FOUND: 404,

            # 409 Conflict
            ErrorCode.EXPERIMENT_ALREADY_RUNNING: 409,
            ErrorCode.DATABASE_CONSTRAINT_VIOLATION: 409,

            # 429 Too Many Requests
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.RATE_LIMIT_QUOTA_EXCEEDED: 429,
            ErrorCode.WANDB_QUOTA_EXCEEDED: 429,

            # 500 Internal Server Error
            ErrorCode.TASK_PROCESSING_FAILED: 500,
            ErrorCode.TASK_EVALUATION_ERROR: 500,
            ErrorCode.EXPERIMENT_FAILED: 500,
            ErrorCode.DATABASE_CONNECTION_ERROR: 500,
            ErrorCode.DATABASE_QUERY_ERROR: 500,
            ErrorCode.DATABASE_TRANSACTION_FAILED: 500,
            ErrorCode.WANDB_CONNECTION_ERROR: 500,
            ErrorCode.WANDB_API_ERROR: 500,
            ErrorCode.SYSTEM_INTERNAL_ERROR: 500,
            ErrorCode.SYSTEM_CONFIGURATION_ERROR: 500,
            ErrorCode.WEBSOCKET_CONNECTION_FAILED: 500,

            # 503 Service Unavailable
            ErrorCode.SYSTEM_RESOURCE_EXHAUSTED: 503,
            ErrorCode.WEBSOCKET_MAX_CONNECTIONS: 503,

            # 504 Gateway Timeout
            ErrorCode.SYSTEM_TIMEOUT: 504,
        }

        return status_map.get(error_code, 500)

    @staticmethod
    def log_error(
        error: BaseEvaluationError | Exception,
        context: dict[str, Any] | None = None,
        request_id: str | None = None
    ):
        """Log error with structured data.

        Args:
            error: Error to log
            context: Optional context information
            request_id: Optional request ID
        """
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_id": request_id,
            **(context or {})
        }

        if isinstance(error, BaseEvaluationError):
            log_data.update({
                "error_code": error.code.value,
                "error_details": [detail.dict() for detail in error.details],
                "error_context": error.context
            })

        # Add stack trace for debugging
        log_data["stack_trace"] = traceback.format_exc()

        logger.error("evaluation_error", **log_data)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for centralized error handling."""

    async def dispatch(self, request: Request, call_next):
        """Handle request with error catching.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response with error handling
        """
        request_id = request.headers.get("x-request-id") or f"req_{id(request)}"

        try:
            # Add request ID to request state
            request.state.request_id = request_id

            response = await call_next(request)
            return response

        except BaseEvaluationError as error:
            # Handle custom evaluation errors
            ErrorHandler.log_error(error, request_id=request_id)
            error_response = ErrorHandler.create_error_response(error, request_id)
            status_code = ErrorHandler.get_http_status_code(error.code)

            return JSONResponse(
                status_code=status_code,
                content=error_response.dict(),
                headers={"x-request-id": request_id}
            )

        except HTTPException as error:
            # Handle FastAPI HTTP exceptions
            ErrorHandler.log_error(error, request_id=request_id)
            error_response = ErrorHandler.create_error_response(error, request_id)

            return JSONResponse(
                status_code=error.status_code,
                content=error_response.dict(),
                headers={"x-request-id": request_id}
            )

        except Exception as error:
            # Handle unexpected errors
            ErrorHandler.log_error(
                error,
                context={
                    "path": request.url.path,
                    "method": request.method,
                    "client": str(request.client) if request.client else None
                },
                request_id=request_id
            )

            error_response = ErrorHandler.create_error_response(error, request_id)

            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={"x-request-id": request_id}
            )


# Recovery strategies for different error types
RECOVERY_STRATEGIES = {
    ErrorCode.AUTH_EXPIRED_TOKEN: [
        "Refresh your authentication token",
        "Re-authenticate with valid credentials",
        "Check token expiration settings"
    ],
    ErrorCode.RATE_LIMIT_EXCEEDED: [
        "Wait for rate limit window to reset",
        "Reduce request frequency",
        "Consider upgrading to higher rate limits"
    ],
    ErrorCode.TASK_NOT_FOUND: [
        "Verify the task ID is correct",
        "Check if the task has been created",
        "Ensure you have access to the task"
    ],
    ErrorCode.DATABASE_CONNECTION_ERROR: [
        "Check database connectivity",
        "Verify database configuration",
        "Check system resources",
        "Contact system administrator"
    ],
    ErrorCode.WANDB_QUOTA_EXCEEDED: [
        "Check W&B storage usage",
        "Clean up old experiments",
        "Consider upgrading W&B plan",
        "Contact W&B support"
    ]
}


class ErrorRecovery:
    """Error recovery utilities and decorators."""

    # Recovery action constants
    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"

    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        recovery_action: str = None,
        retryable_exceptions: tuple = (Exception,)
    ):
        """Decorator for automatic retry functionality.

        Args:
            max_attempts: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
            recovery_action: Recovery action to take (defaults to RETRY)
            retryable_exceptions: Tuple of exceptions that should trigger retries
        """
        if recovery_action is None:
            recovery_action = ErrorRecovery.RETRY

        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e

                        if attempt == max_attempts - 1:
                            # Last attempt failed
                            break

                        # Calculate delay with exponential backoff
                        wait_time = delay * (backoff_multiplier ** attempt)
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt + 1}/{max_attempts}, "
                            f"retrying in {wait_time}s: {e}"
                        )

                        import time
                        time.sleep(wait_time)

                # All retries failed
                if recovery_action == ErrorRecovery.FAIL_FAST:
                    raise last_exception
                elif recovery_action == ErrorRecovery.IGNORE:
                    logger.error(f"Function {func.__name__} failed but ignoring due to recovery_action=IGNORE")
                    return None
                else:
                    raise ARCBaseException(
                        message=f"Function {func.__name__} failed after {max_attempts} attempts",
                        error_code=ErrorCode.SYSTEM_INTERNAL_ERROR,
                        cause=last_exception,
                        suggestions=[
                            "Check system resources and connectivity",
                            "Verify function parameters",
                            "Try increasing retry attempts or delay"
                        ]
                    ) from last_exception

            return wrapper
        return decorator

    @staticmethod
    def with_fallback(fallback_func):
        """Decorator to provide fallback functionality.

        Args:
            fallback_func: Function to call if primary function fails
        """
        def decorator(primary_func):
            def wrapper(*args, **kwargs):
                try:
                    return primary_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Primary function {primary_func.__name__} failed, "
                        f"trying fallback {fallback_func.__name__}: {e}"
                    )
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        raise ARCBaseException(
                            message=f"Both primary function {primary_func.__name__} and fallback {fallback_func.__name__} failed",
                            error_code=ErrorCode.SYSTEM_INTERNAL_ERROR,
                            cause=fallback_error,
                            suggestions=[
                                "Check system state and dependencies",
                                "Verify input parameters",
                                "Consider implementing additional fallback strategies"
                            ]
                        ) from e

            return wrapper
        return decorator
