"""Centralized error handling system for the evaluation framework.

This module provides custom exception classes, error response formats,
and middleware for consistent error handling across all modules.
"""

import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog

# Handle optional dependencies
try:
    from fastapi import HTTPException, Request, Response, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    # Create minimal replacements for when FastAPI is not available
    class BaseModel:
        pass
    class BaseHTTPMiddleware:
        pass
    HTTPException = Exception
    JSONResponse = dict

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
    
    # Data processing
    DATA_NOT_FOUND = "DATA_001"
    INITIALIZATION_ERROR = "INIT_001"
    
    # WebSocket
    WEBSOCKET_CONNECTION_FAILED = "WS_001"
    WEBSOCKET_AUTH_REQUIRED = "WS_002"
    WEBSOCKET_PROTOCOL_ERROR = "WS_003"
    WEBSOCKET_MAX_CONNECTIONS = "WS_004"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = None
    message: str
    code: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    
    error: bool = True
    code: ErrorCode
    message: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None
    timestamp: str
    recovery_suggestions: Optional[List[str]] = None


class BaseEvaluationError(Exception):
    """Base exception for all evaluation framework errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[List[ErrorDetail]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
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


class AuthenticationError(BaseEvaluationError):
    """Authentication and authorization errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.AUTH_INVALID_TOKEN, **kwargs):
        super().__init__(message, code, **kwargs)


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


# Additional error classes for ARC data processing compatibility
class DataNotFoundException(BaseEvaluationError):
    """Data not found errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.TASK_NOT_FOUND, **kwargs):
        super().__init__(message, code, **kwargs)


class DataCorruptionException(BaseEvaluationError):
    """Data corruption errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.TASK_INVALID_INPUT, **kwargs):
        super().__init__(message, code, **kwargs)


class DataFormatException(BaseEvaluationError):
    """Data format errors."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.VALIDATION_INVALID_FORMAT, **kwargs):
        super().__init__(message, code, **kwargs)


class ARCBaseException(BaseEvaluationError):
    """Base ARC processing exception."""
    pass


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorContext:
    """Error context information."""
    
    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.data = kwargs


class ErrorRecovery(Enum):
    """Error recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL_FAST = "fail_fast"


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


class ErrorHandler:
    """Centralized error handling utilities."""
    
    @staticmethod
    def create_error_response(
        error: Union[BaseEvaluationError, Exception],
        request_id: Optional[str] = None
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
        error: Union[BaseEvaluationError, Exception],
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
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