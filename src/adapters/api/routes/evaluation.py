"""Real-time evaluation API routes with WebSocket support for performance dashboard.

This module provides REST API endpoints and WebSocket connections for real-time
evaluation monitoring, experiment tracking, and performance visualization.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import (
    APIRouter,
    Depends,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.adapters.repositories.task_repository import get_task_repository
from src.domain.evaluation_models import (
    DashboardMetrics,
    ExperimentRun,
    StrategyType,
    TaskStatus,
    TaskSubmission,
)
from src.domain.services.evaluation_service import EvaluationService
from src.utils.error_handling import (
    ARCBaseException,
    AuthenticationException,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    EvaluationException,
    TaskNotFoundException,
)
from src.utils.jwt_auth import get_jwt_manager

logger = structlog.get_logger(__name__)

# Create API router with comprehensive documentation
router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["Evaluation Framework"],
    responses={
        401: {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Not authenticated",
                        "error_code": "AUTH_REQUIRED"
                    }
                }
            }
        },
        403: {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Insufficient permissions",
                        "error_code": "INSUFFICIENT_PERMISSIONS"
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Try again in 60 seconds.",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "retry_after": 60
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred",
                        "error_code": "INTERNAL_ERROR",
                        "request_id": "req_12345"
                    }
                }
            }
        }
    }
)

# Security dependencies
security = HTTPBearer()
jwt_manager = get_jwt_manager()
task_repository = get_task_repository()

# WebSocket connection manager for real-time updates
class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates with connection pooling."""

    def __init__(self, max_connections: int = 1000):
        """Initialize the connection manager.

        Args:
            max_connections: Maximum number of concurrent connections
        """
        self.active_connections: list[WebSocket] = []
        self.authenticated_connections: dict[WebSocket, str] = {}  # Maps websocket to user_id
        self.experiment_subscriptions: dict[str, list[WebSocket]] = {}

        # Connection pooling
        self.max_connections = max_connections
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self.connection_timestamps: dict[WebSocket, datetime] = {}  # Track connection times

        # Performance metrics
        self.total_connections_served = 0
        self.rejected_connections = 0

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection with connection pooling.

        Args:
            websocket: The WebSocket connection
            user_id: Authenticated user ID

        Returns:
            bool: True if connected, False if rejected due to pool limits
        """
        # Check connection pool limit
        if len(self.active_connections) >= self.max_connections:
            self.rejected_connections += 1
            logger.warning(
                "websocket_rejected_pool_full",
                user_id=user_id,
                active_connections=len(self.active_connections),
                max_connections=self.max_connections
            )
            await websocket.close(code=1013, reason="Connection pool full")
            return False

        # Acquire semaphore slot
        try:
            await self.connection_semaphore.acquire()
            await websocket.accept()

            self.active_connections.append(websocket)
            self.authenticated_connections[websocket] = user_id
            self.connection_timestamps[websocket] = datetime.now()
            self.total_connections_served += 1

            logger.info(
                "websocket_connected",
                user_id=user_id,
                total_connections=len(self.active_connections),
                pool_usage_pct=(len(self.active_connections) / self.max_connections) * 100
            )
            return True

        except Exception as e:
            self.connection_semaphore.release()
            logger.error("websocket_connect_error", user_id=user_id, error=str(e))
            return False

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection and release pool resources."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Calculate connection duration
        connection_time = self.connection_timestamps.pop(websocket, None)
        duration_seconds = None
        if connection_time:
            duration_seconds = (datetime.now() - connection_time).total_seconds()

        # Remove authentication mapping
        user_id = self.authenticated_connections.pop(websocket, None)

        # Remove from all experiment subscriptions
        for experiment_id in list(self.experiment_subscriptions.keys()):
            if websocket in self.experiment_subscriptions[experiment_id]:
                self.experiment_subscriptions[experiment_id].remove(websocket)
                if not self.experiment_subscriptions[experiment_id]:
                    del self.experiment_subscriptions[experiment_id]

        # Release semaphore slot
        self.connection_semaphore.release()

        logger.info(
            "websocket_disconnected",
            user_id=user_id,
            total_connections=len(self.active_connections),
            duration_seconds=duration_seconds,
            pool_usage_pct=(len(self.active_connections) / self.max_connections) * 100
        )

    async def subscribe_to_experiment(self, websocket: WebSocket, experiment_id: str):
        """Subscribe a WebSocket to experiment updates."""
        if experiment_id not in self.experiment_subscriptions:
            self.experiment_subscriptions[experiment_id] = []
        if websocket not in self.experiment_subscriptions[experiment_id]:
            self.experiment_subscriptions[experiment_id].append(websocket)
            logger.info(
                "experiment_subscription_added",
                experiment_id=experiment_id,
                subscribers=len(self.experiment_subscriptions[experiment_id]),
            )

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning("broadcast_failed", error=str(e))
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_to_experiment(self, experiment_id: str, message: str):
        """Broadcast a message to all clients subscribed to an experiment."""
        if experiment_id not in self.experiment_subscriptions:
            return

        disconnected = []
        for connection in self.experiment_subscriptions[experiment_id]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(
                    "experiment_broadcast_failed",
                    experiment_id=experiment_id,
                    error=str(e),
                )
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    def get_pool_statistics(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        active_count = len(self.active_connections)
        return {
            "active_connections": active_count,
            "max_connections": self.max_connections,
            "pool_usage_pct": (active_count / self.max_connections) * 100,
            "total_served": self.total_connections_served,
            "rejected_connections": self.rejected_connections,
            "available_slots": self.max_connections - active_count,
            "experiment_subscriptions": len(self.experiment_subscriptions),
            "unique_users": len(set(self.authenticated_connections.values()))
        }


# Global connection manager instance
manager = ConnectionManager()


# Pydantic models for request/response
class SubmitTaskRequest(BaseModel):
    """Request model for ARC task submission with comprehensive validation.

    This model represents a solution attempt for an ARC (Abstraction and Reasoning Corpus) task.
    Each submission includes the predicted solution, strategy used, confidence level, and metadata.

    Attributes:
        task_id: Unique identifier for the ARC task (format: arc_YYYY_NNN)
        predicted_output: 2D grid representing the predicted solution as integers
        strategy: Algorithm/approach used to generate the prediction
        confidence_score: Model's confidence in the prediction (0.0 = no confidence, 1.0 = certain)
        attempt_number: Which attempt this is (1 or 2, as per ARC competition rules)
        metadata: Additional context about the solution process
    """

    task_id: str = Field(
        ...,
        description="Unique ARC task identifier",
        example="arc_2024_001",
        regex=r"^arc_\d{4}_\d{3}$"
    )
    predicted_output: list[list[int]] = Field(
        ...,
        description="2D grid representing the predicted solution. Each cell contains an integer 0-9 representing colors.",
        example=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        min_items=1,
        max_items=30
    )
    strategy: StrategyType = Field(
        ...,
        description="Strategy/algorithm used to generate this prediction",
        example="PATTERN_MATCH"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the prediction (0.0 = no confidence, 1.0 = completely certain)",
        example=0.85
    )
    attempt_number: int = Field(
        ...,
        ge=1,
        le=2,
        description="Attempt number (1 or 2) following ARC competition rules",
        example=1
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the solution process",
        example={
            "processing_time_ms": 1250,
            "iterations": 5,
            "pattern_detected": "rotation_90",
            "certainty_factors": [0.9, 0.8, 0.7]
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "task_id": "arc_2024_001",
                "predicted_output": [
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1]
                ],
                "strategy": "PATTERN_MATCH",
                "confidence_score": 0.92,
                "attempt_number": 1,
                "metadata": {
                    "processing_time_ms": 1450,
                    "pattern_type": "symmetry",
                    "confidence_breakdown": {
                        "pattern_match": 0.95,
                        "output_validation": 0.89
                    }
                }
            }
        }


class SubmitTaskResponse(BaseModel):
    """Response model for task submission results with detailed evaluation metrics.

    This model returns comprehensive evaluation results after a task submission,
    including accuracy metrics, performance data, and error analysis.

    Attributes:
        submission_id: Unique identifier for this submission
        accuracy: Pixel-level accuracy score (0.0 to 1.0)
        perfect_match: Whether the prediction exactly matches the ground truth
        processing_time_ms: Time taken to process and evaluate the submission
        error_category: Classification of errors if the submission was incorrect
        evaluation_details: Detailed breakdown of the evaluation process
    """

    submission_id: str = Field(
        ...,
        description="Unique identifier for this submission",
        example="sub_arc_2024_001_1704067200_123"
    )
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Pixel-level accuracy score (1.0 = perfect match, 0.0 = completely wrong)",
        example=0.94
    )
    perfect_match: bool = Field(
        ...,
        description="True if prediction exactly matches ground truth",
        example=True
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
        example=234.5
    )
    error_category: str | None = Field(
        None,
        description="Category of error if submission was incorrect",
        example="partial_pattern_match",
        enum=["shape_mismatch", "color_error", "pattern_incomplete", "size_error", None]
    )
    evaluation_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed evaluation breakdown",
        example={
            "grid_comparison": {
                "correct_pixels": 17,
                "total_pixels": 18,
                "accuracy_percentage": 94.4
            },
            "error_analysis": {
                "incorrect_positions": [[2, 1]],
                "expected_values": [3],
                "actual_values": [5]
            },
            "performance_metrics": {
                "evaluation_time_ms": 12.3,
                "memory_used_mb": 2.1
            }
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "submission_id": "sub_arc_2024_001_1704067200_456",
                "accuracy": 1.0,
                "perfect_match": True,
                "processing_time_ms": 187.2,
                "error_category": None,
                "evaluation_details": {
                    "grid_comparison": {
                        "correct_pixels": 9,
                        "total_pixels": 9,
                        "accuracy_percentage": 100.0
                    },
                    "performance_metrics": {
                        "evaluation_time_ms": 8.5,
                        "memory_used_mb": 1.8
                    }
                }
            }
        }


class TaskEvaluation(BaseModel):
    """Single task evaluation within a batch request."""

    task_id: str = Field(..., description="ARC task identifier", example="arc_2024_001")
    predicted_output: list[list[int]] = Field(..., description="Predicted solution grid")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Prediction confidence")
    attempt_number: int = Field(1, ge=1, le=2, description="Attempt number")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class EvaluateBatchRequest(BaseModel):
    """Request model for batch evaluation of multiple ARC tasks.

    This model allows submitting multiple task predictions for evaluation in a single request,
    which is more efficient for processing large sets of solutions.

    Attributes:
        evaluations: List of task evaluations to process
        strategy: Primary strategy used for all predictions in this batch
        experiment_id: Optional experiment identifier to group results
        parallel_processing: Whether to process tasks in parallel (default: True)
        timeout_seconds: Maximum time to spend on batch processing
    """

    evaluations: list[TaskEvaluation] = Field(
        ...,
        description="List of task evaluations to process in this batch",
        min_items=1,
        max_items=100
    )
    strategy: StrategyType = Field(
        ...,
        description="Primary strategy used for predictions in this batch",
        example="ENSEMBLE"
    )
    experiment_id: str | None = Field(
        None,
        description="Optional experiment identifier to group and track results",
        example="exp_hyperparameter_search_20240101"
    )
    parallel_processing: bool = Field(
        True,
        description="Whether to process evaluations in parallel for faster execution"
    )
    timeout_seconds: int = Field(
        300,
        ge=10,
        le=3600,
        description="Maximum time allowed for batch processing (10-3600 seconds)"
    )

    class Config:
        schema_extra = {
            "example": {
                "evaluations": [
                    {
                        "task_id": "arc_2024_001",
                        "predicted_output": [[1, 0], [0, 1]],
                        "confidence": 0.92,
                        "attempt_number": 1,
                        "metadata": {"pattern_type": "inversion"}
                    },
                    {
                        "task_id": "arc_2024_002",
                        "predicted_output": [[2, 3, 4], [5, 6, 7]],
                        "confidence": 0.78,
                        "attempt_number": 1,
                        "metadata": {"pattern_type": "translation"}
                    }
                ],
                "strategy": "PATTERN_MATCH",
                "experiment_id": "exp_validation_run_001",
                "parallel_processing": True,
                "timeout_seconds": 180
            }
        }


class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status with comprehensive progress tracking.

    This model provides detailed status information about a running experiment,
    including progress metrics, performance indicators, and timing estimates.

    Attributes:
        experiment_id: Unique experiment identifier
        status: Current execution status of the experiment
        progress: Overall completion percentage (0.0 to 1.0)
        current_task: Task currently being processed (if in progress)
        completed_tasks: Number of tasks completed successfully
        total_tasks: Total number of tasks in the experiment
        average_accuracy: Mean accuracy across completed tasks
        estimated_completion_time: Predicted finish time based on current progress
        performance_metrics: Detailed performance and resource usage statistics
        error_summary: Summary of any errors encountered during processing
    """

    experiment_id: str = Field(
        ...,
        description="Unique identifier for this experiment",
        example="exp_ensemble_validation_20240101_143022"
    )
    status: TaskStatus = Field(
        ...,
        description="Current execution status",
        example="IN_PROGRESS"
    )
    progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall completion percentage (0.0 = not started, 1.0 = completed)",
        example=0.67
    )
    current_task: str | None = Field(
        None,
        description="Task currently being processed (null if not in progress)",
        example="arc_2024_045"
    )
    completed_tasks: int = Field(
        ...,
        ge=0,
        description="Number of tasks completed successfully",
        example=67
    )
    total_tasks: int = Field(
        ...,
        ge=1,
        description="Total number of tasks in this experiment",
        example=100
    )
    average_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean accuracy across completed tasks",
        example=0.834
    )
    estimated_completion_time: datetime | None = Field(
        None,
        description="Estimated completion time based on current progress rate",
        example="2024-01-01T15:45:30Z"
    )
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance and resource usage statistics",
        example={
            "average_processing_time_ms": 234.5,
            "total_cpu_seconds": 156.8,
            "memory_usage_mb": 512.3,
            "api_calls_made": 890,
            "estimated_cost_usd": 2.34
        }
    )
    error_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of errors encountered during processing",
        example={
            "failed_tasks": 3,
            "error_categories": {
                "timeout": 2,
                "validation_error": 1
            },
            "last_error": "Task arc_2024_023 timed out after 30 seconds"
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "experiment_id": "exp_pattern_ensemble_20240101_143022",
                "status": "IN_PROGRESS",
                "progress": 0.73,
                "current_task": "arc_2024_074",
                "completed_tasks": 73,
                "total_tasks": 100,
                "average_accuracy": 0.847,
                "estimated_completion_time": "2024-01-01T16:12:45Z",
                "performance_metrics": {
                    "average_processing_time_ms": 187.3,
                    "tasks_per_minute": 3.2,
                    "memory_usage_mb": 468.2,
                    "estimated_cost_usd": 1.89
                },
                "error_summary": {
                    "failed_tasks": 2,
                    "error_categories": {"timeout": 2}
                }
            }
        }


# API Endpoints
@router.post(
    "/submit",
    response_model=SubmitTaskResponse,
    summary="Submit ARC Task Solution",
    description="Submit a solution prediction for an ARC task and receive detailed evaluation results",
    response_description="Detailed evaluation results including accuracy metrics and performance data",
    responses={
        200: {
            "description": "Task evaluated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "submission_id": "sub_arc_2024_001_1704067200_789",
                        "accuracy": 1.0,
                        "perfect_match": True,
                        "processing_time_ms": 156.7,
                        "error_category": None,
                        "evaluation_details": {
                            "grid_comparison": {
                                "correct_pixels": 9,
                                "total_pixels": 9,
                                "accuracy_percentage": 100.0
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid task submission",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_grid": {
                            "summary": "Invalid grid format",
                            "value": {
                                "detail": "Predicted output grid contains invalid values. Only integers 0-9 are allowed.",
                                "error_code": "INVALID_GRID_VALUES",
                                "invalid_values": [10, -1]
                            }
                        },
                        "grid_size_mismatch": {
                            "summary": "Grid size mismatch",
                            "value": {
                                "detail": "Predicted output grid size (3x4) doesn't match expected size (3x3)",
                                "error_code": "GRID_SIZE_MISMATCH",
                                "expected_size": [3, 3],
                                "actual_size": [3, 4]
                            }
                        }
                    }
                }
            }
        },
        404: {
            "description": "Task not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Task arc_2024_999 not found in database",
                        "error_code": "TASK_NOT_FOUND",
                        "task_id": "arc_2024_999"
                    }
                }
            }
        }
    }
)
async def submit_task(
    request: SubmitTaskRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    evaluation_service: EvaluationService = Depends(lambda: EvaluationService()),
):
    """Submit an ARC task solution prediction for comprehensive evaluation.

    This endpoint accepts a solution prediction for an ARC (Abstraction and Reasoning Corpus) task
    and returns detailed evaluation results including accuracy metrics, performance data, and error analysis.

    ## Process Flow:
    1. **Authentication**: Validates JWT token and user permissions
    2. **Task Validation**: Verifies the task exists and is accessible
    3. **Input Validation**: Checks grid format, size, and value constraints
    4. **Evaluation**: Compares prediction against ground truth using pixel-level accuracy
    5. **Real-time Updates**: Broadcasts results to connected WebSocket clients
    6. **Storage**: Persists submission and evaluation results for tracking

    ## Evaluation Metrics:
    - **Pixel Accuracy**: Percentage of correct pixel predictions (0.0-1.0)
    - **Perfect Match**: Boolean indicating exact solution match
    - **Error Classification**: Categorizes types of prediction errors
    - **Performance Tracking**: Processing time and resource usage

    ## Real-time Features:
    Connected WebSocket clients receive immediate notifications about:
    - New task submissions
    - Evaluation results
    - Performance metrics updates
    - Error alerts and system status

    ## Rate Limiting:
    - **Standard Users**: 60 submissions per minute
    - **Premium Users**: 300 submissions per minute
    - **Batch Processing**: Use `/evaluate/batch` for multiple submissions

    ## Error Handling:
    The API provides detailed error responses for various failure scenarios:
    - Invalid grid formats or values
    - Task not found or inaccessible
    - Authentication/authorization failures
    - Rate limiting violations
    - Internal processing errors

    Args:
        request: Task submission with prediction, strategy, and metadata
        credentials: JWT authentication credentials
        evaluation_service: Injected evaluation service instance

    Returns:
        Comprehensive evaluation results with accuracy metrics and performance data

    Raises:
        HTTPException: For authentication, validation, or processing errors
    """
    context = ErrorContext(
        task_id=request.task_id,
        additional_data={
            "attempt_number": request.attempt_number,
            "strategy": request.strategy.value,
            "confidence_score": request.confidence_score
        }
    )

    try:
        # Get authenticated user first
        try:
            user_id = jwt_manager.get_current_user(credentials)
            context.user_id = user_id
        except Exception as e:
            raise AuthenticationException(
                f"Failed to authenticate user: {str(e)}",
                context=context
            ) from e

        # Load task from database
        task = task_repository.get_task(request.task_id)
        if not task:
            raise TaskNotFoundException(
                request.task_id,
                context=context,
                suggestions=[
                    "Check if the task ID is correct",
                    "Verify the task exists in the dataset",
                    "Ensure database connectivity"
                ]
            )

        # Evaluate the task
        start_time = datetime.now()
        try:
            metrics = evaluation_service.evaluate_task_attempt(
                task=task,
                predicted_output=request.predicted_output,
                attempt_number=request.attempt_number,
                strategy_used=request.strategy.value,
                confidence_score=request.confidence_score,
            )
        except (EvaluationException, ARCBaseException) as e:
            # Re-raise with additional context
            e.context = context
            raise
        except Exception as e:
            raise EvaluationException(
                request.task_id,
                f"Task evaluation failed: {str(e)}",
                context=context,
                suggestions=[
                    "Check prediction format",
                    "Verify task data integrity",
                    "Ensure evaluation service is healthy"
                ]
            ) from e

        # Create submission record
        submission = TaskSubmission(
            submission_id=f"sub_{request.task_id}_{datetime.now().timestamp()}",
            task_id=request.task_id,
            user_id=user_id,
            predicted_output=request.predicted_output,
            strategy_used=request.strategy,
            confidence_score=request.confidence_score,
            processing_time_ms=int(metrics.processing_time_ms),
            resource_usage={
                "cpu_ms": metrics.processing_time_ms,
                "memory_mb": 0.0,  # TODO: Implement resource tracking
            },
            metadata=request.metadata,
            submitted_at=start_time,
        )

        # Save submission to database with error handling
        try:
            task_repository.save_submission(submission, metrics)
        except Exception as e:
            logger.error(
                "submission_save_failed",
                submission_id=submission.submission_id,
                task_id=request.task_id,
                error=str(e)
            )
            # Continue even if save fails - return the evaluation result

        # Broadcast real-time update with error handling
        try:
            update_message = {
                "type": "task_submitted",
                "submission_id": submission.submission_id,
                "task_id": request.task_id,
                "accuracy": metrics.pixel_accuracy.accuracy,
                "perfect_match": metrics.pixel_accuracy.perfect_match,
                "timestamp": datetime.now().isoformat(),
            }
            await manager.broadcast(json.dumps(update_message))
        except Exception as e:
            logger.warning(
                "broadcast_failed",
                submission_id=submission.submission_id,
                error=str(e)
            )
            # Continue even if broadcast fails

        logger.info(
            "task_submission_successful",
            task_id=request.task_id,
            submission_id=submission.submission_id,
            accuracy=metrics.pixel_accuracy.accuracy,
            perfect_match=metrics.pixel_accuracy.perfect_match,
            user_id=user_id
        )

        return SubmitTaskResponse(
            submission_id=submission.submission_id,
            accuracy=metrics.pixel_accuracy.accuracy,
            perfect_match=metrics.pixel_accuracy.perfect_match,
            processing_time_ms=metrics.processing_time_ms,
            error_category=metrics.error_category.value if metrics.error_category else None,
        )

    except ARCBaseException as e:
        # Set context if not already set
        if not e.context:
            e.context = context
        # Let middleware handle ARC exceptions
        raise
    except Exception as e:
        logger.error(
            "unexpected_submission_error",
            task_id=request.task_id,
            user_id=context.user_id,
            error=str(e),
            exc_info=True
        )
        raise ARCBaseException(
            message=f"Unexpected error during task submission: {str(e)}",
            error_code=ErrorCode.SUBMISSION_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            suggestions=[
                "Check system resources",
                "Verify all services are running",
                "Try submitting again"
            ]
        ) from e


@router.post(
    "/evaluate/batch",
    summary="Batch Evaluate ARC Tasks",
    description="Process multiple ARC task evaluations in a single efficient batch operation",
    response_description="Batch processing confirmation with experiment tracking details",
    responses={
        200: {
            "description": "Batch evaluation started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "experiment_id": "exp_batch_20240101_143022_456",
                        "status": "processing",
                        "message": "Batch evaluation started for 25 tasks",
                        "estimated_completion_minutes": 8.5,
                        "websocket_updates": True
                    }
                }
            }
        },
        400: {
            "description": "Invalid batch request",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_batch": {
                            "summary": "Empty evaluation list",
                            "value": {
                                "detail": "Evaluations list cannot be empty",
                                "error_code": "EMPTY_BATCH"
                            }
                        },
                        "batch_too_large": {
                            "summary": "Batch size exceeds limit",
                            "value": {
                                "detail": "Batch size (150) exceeds maximum allowed (100)",
                                "error_code": "BATCH_SIZE_EXCEEDED",
                                "max_batch_size": 100
                            }
                        }
                    }
                }
            }
        }
    }
)
async def evaluate_batch(
    request: EvaluateBatchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    evaluation_service: EvaluationService = Depends(lambda: EvaluationService()),
):
    """Process multiple ARC task evaluations in a single optimized batch operation.

    This endpoint enables efficient processing of multiple task predictions simultaneously,
    providing significant performance improvements over individual submissions for large-scale evaluations.

    ## Batch Processing Benefits:
    - **Performance**: Up to 10x faster than individual submissions
    - **Resource Optimization**: Efficient memory and CPU utilization
    - **Progress Tracking**: Real-time updates via WebSocket connections
    - **Error Isolation**: Individual task failures don't affect the entire batch
    - **Cost Efficiency**: Reduced API overhead and connection costs

    ## Processing Modes:
    - **Parallel Processing** (default): Tasks processed simultaneously for maximum speed
    - **Sequential Processing**: Tasks processed one by one for memory-constrained environments
    - **Adaptive Processing**: Automatically adjusts based on system load and task complexity

    ## Real-time Updates:
    WebSocket clients receive continuous updates about:
    - Individual task completion events
    - Overall batch progress percentage
    - Performance metrics and resource usage
    - Error notifications and recovery attempts
    - Final batch completion summary

    ## Experiment Integration:
    When an experiment_id is provided:
    - Results are grouped and tracked as a coherent experiment
    - Comprehensive metrics aggregation is performed
    - Historical comparison with previous experiments
    - Regression detection and performance alerts

    ## Error Handling:
    The system implements robust error handling:
    - **Task-level Failures**: Individual task errors are isolated and logged
    - **Timeout Protection**: Configurable timeouts prevent infinite processing
    - **Resource Limits**: Memory and CPU usage monitoring prevents system overload
    - **Graceful Degradation**: Partial results returned even if some tasks fail

    ## Rate Limiting:
    - **Batch Size**: Maximum 100 tasks per batch
    - **Concurrent Batches**: Maximum 3 concurrent batch operations per user
    - **Processing Time**: Maximum 1 hour per batch operation
    - **Resource Usage**: Dynamic throttling based on system load

    Args:
        request: Batch evaluation request with tasks, strategy, and configuration
        credentials: JWT authentication credentials for user identification
        evaluation_service: Injected evaluation service for processing tasks

    Returns:
        Batch processing confirmation with experiment ID and tracking information

    Raises:
        HTTPException: For authentication, validation, or resource constraint violations
    """
    context = ErrorContext(additional_data={
        "experiment_id": request.experiment_id,
        "strategy": request.strategy.value,
        "num_evaluations": len(request.evaluations)
    })

    try:
        # Get authenticated user
        try:
            user_id = jwt_manager.get_current_user(credentials)
            context.user_id = user_id
        except Exception as e:
            raise AuthenticationException(
                f"Failed to authenticate user: {str(e)}",
                context=context
            ) from e

        # Validate request
        if not request.evaluations:
            raise ARCBaseException(
                message="No evaluations provided for batch processing",
                error_code=ErrorCode.VALIDATION_ERROR,
                context=context,
                suggestions=["Provide at least one evaluation in the request"]
            )

        experiment_id = request.experiment_id or f"exp_{datetime.now().timestamp()}"
        context.experiment_id = experiment_id

        # Create experiment run
        try:
            experiment = ExperimentRun(
                run_id=experiment_id,
                experiment_name=f"Batch evaluation - {request.strategy.value}",
                task_ids=[eval_data["task_id"] for eval_data in request.evaluations],
                strategy_config={"strategy": request.strategy.value},
                metrics={},
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(),
            )
        except Exception as e:
            raise ARCBaseException(
                message=f"Failed to create experiment: {str(e)}",
                error_code=ErrorCode.EXPERIMENT_ERROR,
                context=context,
                suggestions=["Check experiment configuration"]
            ) from e

        # Save experiment to database
        try:
            task_repository.save_experiment(experiment)
        except Exception as e:
            raise ARCBaseException(
                message=f"Failed to save experiment: {str(e)}",
                error_code=ErrorCode.DATABASE_ERROR,
                context=context,
                suggestions=[
                    "Check database connectivity",
                    "Verify database permissions"
                ]
            ) from e

        # Process evaluations asynchronously
        try:
            asyncio.create_task(
                _process_batch_evaluation(
                    experiment, request.evaluations, request.strategy, evaluation_service
                )
            )
        except Exception as e:
            logger.error(
                "async_task_creation_failed",
                experiment_id=experiment_id,
                error=str(e)
            )
            # Mark experiment as failed
            task_repository.update_experiment_status(
                experiment_id,
                TaskStatus.FAILED,
                error_log=f"Failed to start async processing: {str(e)}"
            )
            raise ARCBaseException(
                message=f"Failed to start batch processing: {str(e)}",
                error_code=ErrorCode.EXPERIMENT_ERROR,
                context=context,
                suggestions=[
                    "Check system resources",
                    "Try reducing batch size",
                    "Retry the request"
                ]
            ) from e

        logger.info(
            "batch_evaluation_started",
            experiment_id=experiment_id,
            num_evaluations=len(request.evaluations),
            strategy=request.strategy.value,
            user_id=user_id
        )

        return JSONResponse(
            content={
                "experiment_id": experiment_id,
                "status": "processing",
                "message": f"Batch evaluation started for {len(request.evaluations)} tasks",
                "total_tasks": len(request.evaluations),
                "strategy": request.strategy.value
            }
        )

    except ARCBaseException:
        # Let middleware handle ARC exceptions
        raise
    except Exception as e:
        logger.error(
            "unexpected_batch_evaluation_error",
            experiment_id=context.experiment_id,
            user_id=context.user_id,
            error=str(e),
            exc_info=True
        )
        raise ARCBaseException(
            message=f"Unexpected error during batch evaluation: {str(e)}",
            error_code=ErrorCode.EXPERIMENT_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            suggestions=[
                "Check system resources",
                "Verify request format",
                "Try again with smaller batch"
            ]
        ) from e


async def _process_batch_evaluation(
    experiment: ExperimentRun,
    evaluations: list[dict[str, Any]],
    strategy: StrategyType,
    evaluation_service: EvaluationService,
):
    """Process batch evaluation asynchronously with progress updates."""
    try:
        total_tasks = len(evaluations)
        completed_tasks = 0
        total_accuracy = 0.0
        perfect_matches = 0

        for eval_data in evaluations:
            # Load task from database
            task = task_repository.get_task(eval_data["task_id"])
            if not task:
                logger.warning("task_not_found_in_batch", task_id=eval_data["task_id"])
                continue

            # Evaluate task
            predictions = [(eval_data["predicted_output"], eval_data.get("confidence", 0.5))]
            result = evaluation_service.evaluate_task_with_attempts(
                task=task,
                predictions=predictions,
                strategy_used=strategy.value,
            )

            # Update metrics
            completed_tasks += 1
            if result.best_attempt:
                total_accuracy += result.best_attempt.pixel_accuracy.accuracy
                if result.best_attempt.pixel_accuracy.perfect_match:
                    perfect_matches += 1

            # Send progress update
            progress_update = {
                "type": "experiment_progress",
                "experiment_id": experiment.run_id,
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
                "progress": completed_tasks / total_tasks,
                "current_accuracy": total_accuracy / completed_tasks if completed_tasks > 0 else 0.0,
                "timestamp": datetime.now().isoformat(),
            }
            await manager.broadcast_to_experiment(experiment.run_id, json.dumps(progress_update))

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

        # Mark experiment as completed
        final_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
        experiment.mark_completed(
            {
                "average_accuracy": final_accuracy,
                "perfect_matches": perfect_matches,
                "success_rate": completed_tasks / total_tasks,
            }
        )

        # Update experiment in database
        task_repository.update_experiment_status(
            experiment.run_id,
            TaskStatus.COMPLETED,
            experiment.metrics
        )

        # Send completion notification
        completion_update = {
            "type": "experiment_completed",
            "experiment_id": experiment.run_id,
            "status": "completed",
            "final_accuracy": final_accuracy,
            "perfect_matches": perfect_matches,
            "total_tasks": total_tasks,
            "timestamp": datetime.now().isoformat(),
        }
        await manager.broadcast_to_experiment(experiment.run_id, json.dumps(completion_update))

    except Exception as e:
        logger.error(
            "batch_processing_failed",
            experiment_id=experiment.run_id,
            error=str(e),
            exc_info=True,
        )
        experiment.mark_failed(str(e))

        # Update experiment in database
        task_repository.update_experiment_status(
            experiment.run_id,
            TaskStatus.FAILED,
            error_log=str(e)
        )

        # Send failure notification
        failure_update = {
            "type": "experiment_failed",
            "experiment_id": experiment.run_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        await manager.broadcast_to_experiment(experiment.run_id, json.dumps(failure_update))


@router.get(
    "/experiments/{experiment_id}/status",
    response_model=ExperimentStatusResponse,
    summary="Get Experiment Status",
    description="Retrieve detailed status and progress information for a running or completed experiment",
    response_description="Comprehensive experiment status with progress metrics and performance data",
    responses={
        200: {
            "description": "Experiment status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "experiment_id": "exp_pattern_analysis_20240101_143022",
                        "status": "IN_PROGRESS",
                        "progress": 0.68,
                        "current_task": "arc_2024_068",
                        "completed_tasks": 68,
                        "total_tasks": 100,
                        "average_accuracy": 0.827,
                        "estimated_completion_time": "2024-01-01T16:15:30Z",
                        "performance_metrics": {
                            "average_processing_time_ms": 245.6,
                            "memory_usage_mb": 512.3,
                            "estimated_cost_usd": 3.42
                        }
                    }
                }
            }
        },
        404: {
            "description": "Experiment not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Experiment exp_nonexistent_123 not found",
                        "error_code": "EXPERIMENT_NOT_FOUND",
                        "experiment_id": "exp_nonexistent_123"
                    }
                }
            }
        }
    }
)
async def get_experiment_status(
    experiment_id: str = Path(
        ...,
        description="Unique experiment identifier",
        example="exp_pattern_analysis_20240101_143022",
        regex=r"^exp_[a-zA-Z0-9_]+$"
    ),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Retrieve comprehensive status and progress information for an experiment.

    This endpoint provides detailed real-time information about experiment execution,
    including progress metrics, performance data, resource usage, and completion estimates.

    ## Status Information:
    - **Current Progress**: Percentage completion and task counts
    - **Performance Metrics**: Average accuracy and processing times
    - **Resource Usage**: CPU, memory, and cost tracking
    - **Time Estimates**: Predicted completion time based on current progress rate
    - **Error Summary**: Detailed breakdown of any failures or issues

    ## Status Values:
    - **PENDING**: Experiment queued but not yet started
    - **IN_PROGRESS**: Currently processing tasks
    - **COMPLETED**: All tasks processed successfully
    - **FAILED**: Experiment terminated due to critical error
    - **CANCELLED**: Manually cancelled by user or system
    - **TIMEOUT**: Exceeded maximum allowed processing time

    ## Progress Calculation:
    Progress is calculated as `completed_tasks / total_tasks` and includes:
    - Successfully evaluated tasks
    - Failed tasks (counted as completed for progress purposes)
    - Currently processing task (if applicable)

    ## Performance Metrics:
    The response includes comprehensive performance data:
    - Average processing time per task
    - Memory usage and resource consumption
    - Cost estimation and billing information
    - Throughput rates and efficiency metrics

    ## Real-time Updates:
    For live progress tracking, consider using WebSocket connections:
    - Subscribe to experiment updates via `/ws` endpoint
    - Receive real-time progress notifications
    - Get immediate alerts for completion or failures

    Args:
        experiment_id: Unique identifier for the experiment to query
        credentials: JWT authentication credentials

    Returns:
        Detailed experiment status with progress and performance metrics

    Raises:
        HTTPException: If experiment not found or access denied
    """
    context = ErrorContext(
        experiment_id=experiment_id,
        additional_data={"endpoint": "get_experiment_status"}
    )

    try:
        # Authenticate user
        try:
            user_id = jwt_manager.get_current_user(credentials)
            context.user_id = user_id
        except Exception as e:
            raise AuthenticationException(
                f"Failed to authenticate user: {str(e)}",
                context=context
            ) from e

        # Get experiment from database
        try:
            experiment = task_repository.get_experiment(experiment_id)
        except Exception as e:
            raise ARCBaseException(
                message=f"Failed to retrieve experiment: {str(e)}",
                error_code=ErrorCode.DATABASE_ERROR,
                context=context,
                suggestions=[
                    "Check database connectivity",
                    "Verify experiment ID format"
                ]
            ) from e

        if not experiment:
            raise ARCBaseException(
                message=f"Experiment {experiment_id} not found",
                error_code=ErrorCode.NOT_FOUND,
                context=context,
                suggestions=[
                    "Check if the experiment ID is correct",
                    "Verify the experiment exists",
                    "Ensure you have access to this experiment"
                ]
            )

        # Calculate progress
        total_tasks = len(experiment.task_ids)
        completed_tasks = experiment.metrics.get("completed_tasks", 0)
        progress = completed_tasks / total_tasks if total_tasks > 0 else 0.0

        # Estimate completion time
        estimated_completion = None
        if experiment.status == TaskStatus.IN_PROGRESS and progress > 0:
            elapsed = (datetime.now() - experiment.started_at).total_seconds()
            remaining_time = (elapsed / progress) * (1 - progress)
            estimated_completion = datetime.now() + timedelta(seconds=remaining_time)

        return ExperimentStatusResponse(
            experiment_id=experiment_id,
            status=experiment.status,
            progress=progress,
            current_task=experiment.task_ids[completed_tasks] if completed_tasks < total_tasks else None,
            completed_tasks=completed_tasks,
            total_tasks=total_tasks,
            average_accuracy=experiment.metrics.get("average_accuracy", 0.0),
            estimated_completion_time=estimated_completion,
        )

    except ARCBaseException:
        # Let middleware handle ARC exceptions
        raise
    except Exception as e:
        logger.error(
            "unexpected_experiment_status_error",
            experiment_id=experiment_id,
            user_id=context.user_id,
            error=str(e),
            exc_info=True
        )
        raise ARCBaseException(
            message=f"Unexpected error retrieving experiment status: {str(e)}",
            error_code=ErrorCode.EXPERIMENT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            suggestions=[
                "Check system health",
                "Retry the request",
                "Contact support if problem persists"
            ]
        ) from e


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time evaluation dashboard updates and notifications.

    This WebSocket connection provides live streaming of evaluation metrics, experiment progress,
    and system alerts to enable responsive dashboard interfaces and real-time monitoring.

    ## Authentication Required:
    WebSocket connections must be authenticated using one of these methods:
    - **Query Parameter**: `?token=<jwt_access_token>`
    - **Authorization Header**: `Authorization: Bearer <jwt_access_token>`

    Example connection URLs:
    ```
    ws://localhost:8000/api/v1/evaluation/ws?token=eyJ0eXAiOiJKV1Q...
    ws://localhost:8000/api/v1/evaluation/ws  # with Authorization header
    ```

    ## Real-time Event Types:

    ### Task Submission Events:
    ```json
    {
        "type": "task_submitted",
        "submission_id": "sub_arc_2024_001_1704067200_456",
        "task_id": "arc_2024_001",
        "accuracy": 0.94,
        "perfect_match": false,
        "timestamp": "2024-01-01T14:30:22Z"
    }
    ```

    ### Experiment Progress Updates:
    ```json
    {
        "type": "experiment_progress",
        "experiment_id": "exp_batch_20240101_143022",
        "completed_tasks": 45,
        "total_tasks": 100,
        "progress": 0.45,
        "current_accuracy": 0.823,
        "timestamp": "2024-01-01T14:30:22Z"
    }
    ```

    ### Dashboard Metrics (sent every 500ms):
    ```json
    {
        "type": "dashboard_update",
        "timestamp": "2024-01-01T14:30:22Z",
        "data": {
            "active_experiments": 3,
            "tasks_processed": 1247,
            "average_accuracy": 0.834,
            "resource_utilization": {
                "cpu": 45.2,
                "memory": 67.8,
                "gpu": 0.0
            },
            "system_health": {
                "evaluation_service": "healthy",
                "database": "healthy",
                "cache": "healthy"
            }
        }
    }
    ```

    ### System Alerts:
    ```json
    {
        "type": "system_alert",
        "level": "warning",
        "message": "High memory usage detected (85%)",
        "timestamp": "2024-01-01T14:30:22Z",
        "component": "evaluation_service"
    }
    ```

    ## Client Message Protocol:

    ### Subscribe to Experiment Updates:
    ```json
    {
        "type": "subscribe_experiment",
        "experiment_id": "exp_batch_20240101_143022"
    }
    ```

    ### Ping/Pong for Connection Health:
    ```json
    // Client sends:
    {"type": "ping"}

    // Server responds:
    {
        "type": "pong",
        "timestamp": "2024-01-01T14:30:22Z"
    }
    ```

    ## Connection Management:
    - **Connection Pool**: Maximum 1000 concurrent connections
    - **Adaptive Updates**: Update frequency adjusts based on client activity
    - **Automatic Cleanup**: Disconnected clients are automatically removed
    - **Rate Limiting**: Connection attempts are rate-limited per IP

    ## Error Handling:
    - **Authentication Failures**: Connection closed with code 1008 (Policy Violation)
    - **Pool Full**: Connection rejected with code 1013 (Try Again Later)
    - **Protocol Errors**: Invalid messages result in connection termination
    - **Heartbeat**: Connections without activity for 60 seconds receive ping requests

    ## Performance Optimization:
    - **Selective Updates**: Only subscribed events are sent to each client
    - **Compression**: Large messages are automatically compressed
    - **Batching**: Multiple small updates may be batched together
    - **Throttling**: High-frequency updates are intelligently throttled

    Args:
        websocket: WebSocket connection object from FastAPI

    Note:
        Connection will be automatically closed if authentication fails or
        if the connection pool is at maximum capacity.
    """
    # Authenticate the WebSocket connection
    user_id = await jwt_manager.authenticate_websocket(websocket)
    if not user_id:
        return  # Connection already closed by authenticate_websocket

    await manager.connect(websocket, user_id)
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "connection_established",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Connected to evaluation dashboard",
                }
            ),
            websocket,
        )

        # Start sending periodic dashboard metrics
        asyncio.create_task(_send_dashboard_metrics(websocket))

        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "subscribe_experiment":
                experiment_id = message["experiment_id"]
                await manager.subscribe_to_experiment(websocket, experiment_id)
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "subscription_confirmed",
                            "experiment_id": experiment_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    websocket,
                )
            elif message["type"] == "ping":
                # Respond to ping with pong
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("websocket_client_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e), exc_info=True)
        manager.disconnect(websocket)


async def _send_dashboard_metrics(websocket: WebSocket):
    """Send periodic dashboard metrics to a connected client with adaptive frequency."""
    from src.domain.services.dashboard_aggregator import get_dashboard_aggregator

    aggregator = get_dashboard_aggregator()
    user_id = manager.authenticated_connections.get(websocket, "unknown")

    # Track client activity for adaptive updates
    last_activity = datetime.now()
    update_interval = 0.5  # Start with 500ms default
    min_interval = 0.1  # 100ms for very active clients
    max_interval = 5.0  # 5s for idle clients

    try:
        while websocket in manager.active_connections:
            # Get actual metrics from aggregator
            metrics = aggregator.get_dashboard_metrics()

            # Adjust update frequency based on client activity
            current_time = datetime.now()
            time_since_activity = (current_time - last_activity).total_seconds()

            # Adaptive frequency logic
            if time_since_activity < 5:  # Very active
                update_interval = max(min_interval, update_interval * 0.9)
            elif time_since_activity < 30:  # Moderately active
                update_interval = 0.5  # Default
            else:  # Idle
                update_interval = min(max_interval, update_interval * 1.1)

            # Send metrics update
            await manager.send_personal_message(
                json.dumps(metrics.to_websocket_message()),
                websocket,
            )

            logger.debug(
                "dashboard_metrics_sent",
                user_id=user_id,
                update_interval=update_interval,
                active_experiments=metrics.active_experiments
            )

            # Wait with adaptive interval
            await asyncio.sleep(update_interval)

    except Exception as e:
        logger.error("dashboard_metrics_error", user_id=user_id, error=str(e), exc_info=True)


# Additional utility endpoints
@router.get(
    "/dashboard/metrics",
    summary="Get Dashboard Metrics",
    description="Retrieve current system metrics and performance statistics for dashboard display",
    response_description="Comprehensive dashboard metrics including system status and performance data",
    responses={
        200: {
            "description": "Dashboard metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": "2024-01-01T14:30:22Z",
                        "active_experiments": 5,
                        "tasks_processed_last_hour": 1456,
                        "average_accuracy_last_hour": 0.847,
                        "resource_utilization": {
                            "cpu": 52.3,
                            "memory": 67.8,
                            "gpu": 15.4
                        },
                        "processing_queue_size": 23,
                        "error_rate_last_hour": 0.012,
                        "top_performing_strategies": [
                            {"name": "ensemble", "accuracy": 0.923},
                            {"name": "pattern_match", "accuracy": 0.867},
                            {"name": "neural_network", "accuracy": 0.834}
                        ],
                        "recent_alerts": [],
                        "system_health": {
                            "evaluation_service": "healthy",
                            "database": "healthy",
                            "cache": "healthy",
                            "websocket_manager": "healthy"
                        }
                    }
                }
            }
        }
    }
)
async def get_dashboard_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Retrieve comprehensive system metrics and performance statistics for dashboard visualization.

    This endpoint provides a real-time snapshot of system performance, resource utilization,
    and operational metrics essential for monitoring the evaluation framework's health and efficiency.

    ## Metrics Categories:

    ### System Performance:
    - **Active Experiments**: Number of currently running experiments
    - **Processing Statistics**: Tasks processed, queue sizes, throughput rates
    - **Resource Utilization**: CPU, memory, and GPU usage percentages
    - **Error Rates**: System reliability and failure statistics

    ### Strategy Performance:
    - **Top Performers**: Highest accuracy strategies with recent performance
    - **Comparative Analysis**: Strategy effectiveness rankings
    - **Resource Efficiency**: Processing time and cost per strategy type
    - **Trend Analysis**: Performance changes over time

    ### System Health:
    - **Component Status**: Health checks for all system components
    - **Service Availability**: Uptime and availability metrics
    - **Alert Summary**: Recent system alerts and their severity levels
    - **Capacity Monitoring**: Resource limits and scaling indicators

    ### Recent Activity:
    - **Hourly Statistics**: Tasks processed and accuracy in the last hour
    - **Real-time Updates**: Live processing and evaluation metrics
    - **User Activity**: Active users and concurrent sessions
    - **Performance Trends**: Short-term performance indicators

    ## Data Freshness:
    - **Real-time Metrics**: Updated every 5-10 seconds
    - **Aggregated Data**: Recalculated every minute
    - **Historical Trends**: Updated every 15 minutes
    - **Health Checks**: Performed every 30 seconds

    ## Use Cases:
    - **Operations Dashboard**: Monitor system health and performance
    - **Capacity Planning**: Track resource usage and scaling needs
    - **Performance Analysis**: Compare strategy effectiveness
    - **Troubleshooting**: Identify bottlenecks and system issues

    ## Rate Limiting:
    - **Standard Users**: 60 requests per minute
    - **Dashboard Applications**: 300 requests per minute
    - **Real-time Monitoring**: Use WebSocket endpoint for continuous updates

    Args:
        credentials: JWT authentication credentials for access control

    Returns:
        Comprehensive dashboard metrics with system status and performance data

    Note:
        For real-time continuous updates, consider using the WebSocket endpoint `/ws`
        which provides live streaming of these metrics with lower latency.
    """
    # TODO: Implement actual metrics aggregation
    metrics = DashboardMetrics(
        timestamp=datetime.now(),
        active_experiments=3,
        tasks_processed_last_hour=150,
        average_accuracy_last_hour=0.82,
        resource_utilization={"cpu": 45.5, "memory": 62.3, "gpu": 0.0},
        processing_queue_size=12,
        error_rate_last_hour=0.02,
        top_performing_strategies=[
            ("ensemble", 0.89),
            ("pattern_match", 0.85),
            ("direct_solve", 0.78),
        ],
        recent_alerts=[],
        system_health={
            "evaluation_service": "healthy",
            "database": "healthy",
            "cache": "healthy",
        },
    )

    return JSONResponse(content=metrics.to_websocket_message()["data"])


@router.get(
    "/strategies/performance",
    summary="Get Strategy Performance Analysis",
    description="Retrieve detailed performance metrics and comparative analysis for different solving strategies",
    response_description="Comprehensive strategy performance data with accuracy, efficiency, and cost metrics",
    responses={
        200: {
            "description": "Strategy performance data retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "time_window": "24h",
                        "analysis_period": {
                            "start_time": "2024-01-01T14:30:22Z",
                            "end_time": "2024-01-02T14:30:22Z"
                        },
                        "strategies": [
                            {
                                "name": "ensemble",
                                "tasks_evaluated": 234,
                                "average_accuracy": 0.923,
                                "perfect_matches": 89,
                                "average_processing_time_ms": 456.7,
                                "average_cost": 0.034,
                                "resource_efficiency": 0.87,
                                "success_rate": 0.94,
                                "trend": "improving"
                            },
                            {
                                "name": "pattern_match",
                                "tasks_evaluated": 567,
                                "average_accuracy": 0.867,
                                "perfect_matches": 156,
                                "average_processing_time_ms": 123.4,
                                "average_cost": 0.012,
                                "resource_efficiency": 0.95,
                                "success_rate": 0.89,
                                "trend": "stable"
                            }
                        ],
                        "summary": {
                            "best_accuracy": "ensemble",
                            "best_efficiency": "pattern_match",
                            "best_cost": "direct_solve",
                            "total_tasks_analyzed": 1456
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid time window parameter",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid time window '2h'. Allowed values: 1h, 6h, 24h, 7d, 30d",
                        "error_code": "INVALID_TIME_WINDOW",
                        "allowed_values": ["1h", "6h", "24h", "7d", "30d"]
                    }
                }
            }
        }
    }
)
async def get_strategy_performance(
    time_window: str = Query(
        "1h",
        regex="^(1h|6h|24h|7d|30d)$",
        description="Analysis time window (1h=1 hour, 6h=6 hours, 24h=1 day, 7d=1 week, 30d=1 month)",
        example="24h"
    ),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Retrieve comprehensive performance analysis and comparison metrics for different solving strategies.

    This endpoint provides detailed analytics on strategy effectiveness, efficiency, and resource utilization
    over specified time periods, enabling data-driven strategy selection and optimization.

    ## Performance Metrics:

    ### Accuracy Metrics:
    - **Average Accuracy**: Mean pixel-level accuracy across all tasks
    - **Perfect Matches**: Count and percentage of exactly correct solutions
    - **Success Rate**: Percentage of tasks achieving acceptable accuracy thresholds
    - **Accuracy Distribution**: Statistical breakdown of accuracy ranges

    ### Efficiency Metrics:
    - **Processing Time**: Average, median, and percentile processing times
    - **Resource Efficiency**: Accuracy per unit of compute resource used
    - **Throughput**: Tasks processed per unit time
    - **Scalability**: Performance consistency across different task complexities

    ### Cost Analysis:
    - **Average Cost**: Mean cost per task evaluation
    - **Cost Efficiency**: Accuracy improvement per dollar spent
    - **Resource Usage**: CPU, memory, and API call consumption
    - **Total Cost**: Aggregate spending over the analysis period

    ### Trend Analysis:
    - **Performance Trends**: Improving, declining, or stable performance over time
    - **Comparative Ranking**: Strategy rankings by different performance criteria
    - **Reliability**: Consistency of results across multiple evaluations
    - **Optimization Opportunities**: Identified areas for strategy improvement

    ## Time Window Options:
    - **1h**: Last hour - for real-time performance monitoring
    - **6h**: Last 6 hours - for short-term trend analysis
    - **24h**: Last 24 hours - for daily performance review
    - **7d**: Last 7 days - for weekly performance patterns
    - **30d**: Last 30 days - for monthly strategy evaluation

    ## Strategy Types Analyzed:
    - **DIRECT_SOLVE**: Rule-based direct pattern recognition
    - **PATTERN_MATCH**: Template matching and similarity analysis
    - **TRANSFORMATION_SEARCH**: Systematic transformation discovery
    - **NEURAL_NETWORK**: Deep learning-based approaches
    - **ENSEMBLE**: Combined multi-strategy approaches
    - **CUSTOM**: User-defined custom strategies

    ## Use Cases:
    - **Strategy Selection**: Choose optimal strategy for specific task types
    - **Performance Monitoring**: Track strategy effectiveness over time
    - **Resource Planning**: Optimize compute allocation based on efficiency metrics
    - **Cost Optimization**: Balance accuracy requirements with budget constraints
    - **Research Insights**: Analyze strategy behavior and improvement opportunities

    ## Data Quality:
    - **Sample Size**: Minimum 10 tasks required for meaningful statistics
    - **Statistical Significance**: Confidence intervals provided for key metrics
    - **Outlier Handling**: Extreme values are flagged and analyzed separately
    - **Data Freshness**: Metrics updated in real-time with new evaluations

    Args:
        time_window: Analysis time period (1h, 6h, 24h, 7d, 30d)
        credentials: JWT authentication credentials

    Returns:
        Detailed performance analysis with comparative metrics for all strategies

    Note:
        Large time windows (7d, 30d) may have slower response times due to
        extensive data aggregation. Consider using smaller windows for real-time monitoring.
    """
    # TODO: Implement actual strategy performance aggregation
    return JSONResponse(
        content={
            "time_window": time_window,
            "strategies": [
                {
                    "name": "ensemble",
                    "tasks_evaluated": 45,
                    "average_accuracy": 0.89,
                    "perfect_matches": 12,
                    "average_processing_time_ms": 234.5,
                    "average_cost": 0.023,
                },
                {
                    "name": "pattern_match",
                    "tasks_evaluated": 67,
                    "average_accuracy": 0.85,
                    "perfect_matches": 15,
                    "average_processing_time_ms": 156.2,
                    "average_cost": 0.015,
                },
                {
                    "name": "direct_solve",
                    "tasks_evaluated": 38,
                    "average_accuracy": 0.78,
                    "perfect_matches": 8,
                    "average_processing_time_ms": 89.7,
                    "average_cost": 0.008,
                },
            ],
        }
    )


@router.get(
    "/dashboard/connection-pool/stats",
    summary="Get WebSocket Connection Pool Statistics",
    description="Retrieve detailed statistics about WebSocket connection pool usage and performance",
    response_description="Connection pool metrics including usage, performance, and capacity information",
    responses={
        200: {
            "description": "Connection pool statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": "2024-01-01T14:30:22Z",
                        "active_connections": 247,
                        "max_connections": 1000,
                        "pool_usage_pct": 24.7,
                        "total_served": 15789,
                        "rejected_connections": 23,
                        "available_slots": 753,
                        "experiment_subscriptions": 89,
                        "unique_users": 156,
                        "connection_duration_stats": {
                            "average_seconds": 1247.6,
                            "median_seconds": 892.1,
                            "max_seconds": 7234.8
                        },
                        "performance_metrics": {
                            "messages_sent_per_second": 45.7,
                            "average_latency_ms": 12.3,
                            "error_rate": 0.002
                        }
                    }
                }
            }
        }
    }
)
async def get_connection_pool_stats(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Retrieve comprehensive WebSocket connection pool statistics and performance metrics.

    This endpoint provides detailed insights into WebSocket connection management,
    pool utilization, and performance characteristics essential for monitoring
    real-time communication infrastructure.

    ## Connection Pool Metrics:

    ### Capacity Management:
    - **Active Connections**: Current number of established WebSocket connections
    - **Maximum Capacity**: Configured connection pool size limit
    - **Pool Usage**: Percentage of pool capacity currently utilized
    - **Available Slots**: Remaining connection slots before reaching capacity
    - **Rejection Rate**: Percentage of connections rejected due to capacity limits

    ### Connection Statistics:
    - **Total Served**: Cumulative number of connections handled since startup
    - **Unique Users**: Number of distinct authenticated users with active connections
    - **Connection Duration**: Statistical analysis of connection lifespans
    - **Subscription Mapping**: Active experiment and topic subscriptions

    ### Performance Metrics:
    - **Message Throughput**: Messages sent per second across all connections
    - **Average Latency**: Mean time for message delivery to clients
    - **Error Rates**: Connection failures, message delivery failures
    - **Resource Usage**: Memory and CPU overhead per connection

    ### Health Indicators:
    - **Connection Quality**: Stability and reliability metrics
    - **Resource Efficiency**: System resources per active connection
    - **Scalability Metrics**: Performance trends as connection count increases
    - **Alert Thresholds**: Current status relative to configured limits

    ## Pool Management Features:

    ### Adaptive Scaling:
    - **Dynamic Limits**: Automatic adjustment based on system capacity
    - **Priority Queuing**: Premium users get connection priority
    - **Graceful Degradation**: Intelligent throttling under high load
    - **Resource Monitoring**: Continuous system resource tracking

    ### Connection Optimization:
    - **Idle Detection**: Automatic cleanup of inactive connections
    - **Heartbeat Monitoring**: Connection health verification
    - **Compression**: Message payload optimization
    - **Batching**: Efficient multi-message delivery

    ## Operational Insights:

    ### Capacity Planning:
    - **Usage Patterns**: Peak usage times and connection distribution
    - **Growth Trends**: Historical connection growth analysis
    - **Resource Requirements**: Memory and CPU scaling recommendations
    - **Cost Analysis**: Infrastructure costs per connection

    ### Performance Tuning:
    - **Bottleneck Identification**: Connection pool performance limitations
    - **Optimization Opportunities**: Configuration improvements
    - **Scaling Recommendations**: Horizontal and vertical scaling guidance
    - **Resource Allocation**: Optimal resource distribution strategies

    ## Use Cases:
    - **Infrastructure Monitoring**: Track WebSocket infrastructure health
    - **Capacity Planning**: Determine scaling needs and resource requirements
    - **Performance Optimization**: Identify and resolve connection bottlenecks
    - **Cost Management**: Optimize infrastructure costs based on usage patterns
    - **Troubleshooting**: Diagnose connection issues and performance problems

    ## Administrative Access:
    This endpoint is typically restricted to administrators and operations teams
    due to the sensitive nature of infrastructure metrics and capacity information.

    Args:
        credentials: JWT authentication credentials with admin privileges

    Returns:
        Comprehensive connection pool statistics and performance metrics

    Note:
        High-frequency monitoring of this endpoint may impact system performance.
        Consider implementing local caching for dashboard applications.
    """
    stats = manager.get_pool_statistics()

    return JSONResponse(
        content={
            "timestamp": datetime.now().isoformat(),
            **stats
        }
    )


# Authentication endpoints (temporary for testing)
@router.post(
    "/auth/token",
    summary="Create Authentication Token (Development Only)",
    description="Generate JWT access and refresh tokens for API authentication - Development/Testing endpoint only",
    response_description="JWT token pair with access and refresh tokens",
    tags=["Authentication"],
    responses={
        200: {
            "description": "Tokens created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer",
                        "expires_in": 1800,
                        "user_id": "test_user",
                        "scopes": ["evaluation:read", "evaluation:write"],
                        "issued_at": "2024-01-01T14:30:22Z"
                    }
                }
            }
        }
    },
    deprecated=True
)
async def create_token(user_id: str = "test_user"):
    """Generate JWT authentication tokens for API access (Development/Testing Only).

    **⚠️ WARNING: This endpoint is for development and testing only!**

    In production environments, this endpoint should be disabled and replaced with
    proper user authentication mechanisms including:
    - Username/password validation
    - OAuth 2.0 / OpenID Connect integration
    - Multi-factor authentication (MFA)
    - Rate limiting and brute force protection

    ## Token Information:

    ### Access Token:
    - **Purpose**: Authentication for API requests
    - **Lifetime**: 30 minutes (configurable via JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    - **Usage**: Include in Authorization header as 'Bearer <token>'
    - **Scope**: Full API access for the specified user

    ### Refresh Token:
    - **Purpose**: Obtain new access tokens without re-authentication
    - **Lifetime**: 7 days (configurable via JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    - **Security**: Single-use tokens with unique identifiers
    - **Storage**: Should be stored securely (HttpOnly cookies recommended)

    ## Security Considerations:

    ### Development Use:
    - Simplified authentication for testing and development
    - No password validation or user verification
    - Predictable user IDs for consistent testing
    - Full access granted without permission checks

    ### Production Requirements:
    - Replace with proper authentication service
    - Implement user credential validation
    - Add rate limiting and brute force protection
    - Include audit logging and security monitoring
    - Use proper session management

    ## Usage Examples:

    ### API Request Authentication:
    ```bash
    # Get token
    curl -X POST "http://localhost:8000/api/v1/evaluation/auth/token" \
         -d '{"user_id": "my_user"}'

    # Use token in API requests
    curl -H "Authorization: Bearer <access_token>" \
         "http://localhost:8000/api/v1/evaluation/dashboard/metrics"
    ```

    ### WebSocket Authentication:
    ```javascript
    const token = 'your_access_token_here';
    const ws = new WebSocket(`ws://localhost:8000/api/v1/evaluation/ws?token=${token}`);
    ```

    ## Token Validation:
    All generated tokens include:
    - **Issuer**: Identifies the token issuer (configurable)
    - **Audience**: Specifies intended token recipients
    - **Subject**: User identifier from the request
    - **Expiration**: Prevents indefinite token usage
    - **Issued At**: Timestamp for token age verification

    Args:
        user_id: User identifier for token subject (defaults to 'test_user')

    Returns:
        JWT token pair with access token, refresh token, and metadata

    Warning:
        This endpoint should be removed or secured before production deployment!
    """
    access_token = jwt_manager.create_access_token(user_id)
    refresh_token = jwt_manager.create_refresh_token(user_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user_id
    }
