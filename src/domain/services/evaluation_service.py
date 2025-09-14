"""Evaluation service for pixel-perfect accuracy calculation and performance metrics.

This service provides comprehensive evaluation capabilities for the ARC Prize 2025 competition,
including pixel-perfect accuracy calculation, per-task metrics tracking, 2-attempt evaluation
support, and detailed error analysis.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.domain.models import ARCTask

logger = structlog.get_logger(__name__)


class AttemptNumber(Enum):
    """Enumeration for evaluation attempt numbers."""

    FIRST = 1
    SECOND = 2


class ErrorCategory(Enum):
    """Categories for error analysis."""

    SHAPE_MISMATCH = "shape_mismatch"
    COLOR_ERROR = "color_error"
    PATTERN_ERROR = "pattern_error"
    TRANSFORMATION_ERROR = "transformation_error"
    PARTIAL_CORRECT = "partial_correct"
    COMPLETE_FAILURE = "complete_failure"


@dataclass
class PixelAccuracy:
    """Detailed pixel-level accuracy metrics."""

    total_pixels: int
    correct_pixels: int
    accuracy: float
    perfect_match: bool
    pixel_diff_map: list[list[int]] | None = None


@dataclass
class TaskMetrics:
    """Per-task performance metrics."""

    task_id: str
    attempt_number: AttemptNumber
    pixel_accuracy: PixelAccuracy
    processing_time_ms: float
    confidence_score: float = 0.0
    error_category: ErrorCategory | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a task submission."""

    task_id: str
    strategy_used: str
    attempts: list[TaskMetrics]
    best_attempt: TaskMetrics | None = None
    final_accuracy: float = 0.0
    total_processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        if self.attempts:
            self.best_attempt = max(self.attempts, key=lambda x: x.pixel_accuracy.accuracy)
            self.final_accuracy = self.best_attempt.pixel_accuracy.accuracy
            self.total_processing_time_ms = sum(a.processing_time_ms for a in self.attempts)


class EvaluationService:
    """Service for evaluating model predictions against ground truth with comprehensive error handling."""

    def __init__(self):
        """Initialize the evaluation service."""
        self.logger = structlog.get_logger(__name__).bind(service="evaluation")

        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "shape_mismatch_errors": 0,
            "format_errors": 0,
            "processing_errors": 0,
            "total_processing_time_ms": 0.0,
        }

    def calculate_pixel_accuracy(
        self, predicted: list[list[int]], ground_truth: list[list[int]], include_diff_map: bool = False
    ) -> PixelAccuracy:
        """Calculate pixel-perfect accuracy between predicted and ground truth grids with error handling.

        Args:
            predicted: Predicted output grid
            ground_truth: Ground truth output grid
            include_diff_map: Whether to include a difference map in the result

        Returns:
            PixelAccuracy object with detailed metrics

        Raises:
            PredictionFormatException: If input formats are invalid
            EvaluationException: If calculation fails
        """
        # Convert to numpy arrays for efficient computation
        try:
            pred_array = np.array(predicted)
            truth_array = np.array(ground_truth)
        except (ValueError, TypeError):
            # Handle irregular shaped grids
            self.logger.warning(
                "irregular_grid_shape",
                error="Cannot convert to numpy array - possibly irregular grid",
            )
            # Return 0 accuracy for irregular grids
            return PixelAccuracy(
                total_pixels=sum(len(row) for row in ground_truth) if ground_truth else 0,
                correct_pixels=0,
                accuracy=0.0,
                perfect_match=False,
                pixel_diff_map=None,
            )

        # Check shape compatibility
        if pred_array.shape != truth_array.shape:
            self.logger.warning(
                "shape_mismatch",
                pred_shape=pred_array.shape,
                truth_shape=truth_array.shape,
            )
            # Return 0 accuracy for shape mismatch
            return PixelAccuracy(
                total_pixels=truth_array.size,
                correct_pixels=0,
                accuracy=0.0,
                perfect_match=False,
                pixel_diff_map=None,
            )

        # Calculate pixel-wise comparison
        correct_mask = pred_array == truth_array
        correct_pixels = np.sum(correct_mask)
        total_pixels = truth_array.size
        accuracy = float(correct_pixels) / total_pixels if total_pixels > 0 else 0.0
        perfect_match = bool(np.all(correct_mask))

        # Create difference map if requested
        pixel_diff_map = None
        if include_diff_map and not perfect_match:
            # 0 = correct, 1 = incorrect
            diff_map = (~correct_mask).astype(int)
            pixel_diff_map = diff_map.tolist()

        return PixelAccuracy(
            total_pixels=total_pixels,
            correct_pixels=int(correct_pixels),
            accuracy=accuracy,
            perfect_match=perfect_match,
            pixel_diff_map=pixel_diff_map,
        )

    def categorize_error(
        self, predicted: list[list[int]], ground_truth: list[list[int]], pixel_accuracy: PixelAccuracy
    ) -> tuple[ErrorCategory, dict[str, Any]]:
        """Categorize the type of error in the prediction.

        Args:
            predicted: Predicted output grid
            ground_truth: Ground truth output grid
            pixel_accuracy: Calculated pixel accuracy

        Returns:
            Tuple of (ErrorCategory, error_details dict)
        """
        details = {}
        pred_array = np.array(predicted)
        truth_array = np.array(ground_truth)

        # Shape mismatch
        if pred_array.shape != truth_array.shape:
            details["pred_shape"] = pred_array.shape
            details["truth_shape"] = truth_array.shape
            details["shape_diff"] = (
                pred_array.shape[0] - truth_array.shape[0],
                pred_array.shape[1] - truth_array.shape[1] if pred_array.ndim > 1 else 0,
            )
            return ErrorCategory.SHAPE_MISMATCH, details

        # Perfect match
        if pixel_accuracy.perfect_match:
            return None, {}

        # Analyze error patterns
        if pixel_accuracy.accuracy == 0.0:
            return ErrorCategory.COMPLETE_FAILURE, {"accuracy": 0.0}

        if pixel_accuracy.accuracy >= 0.7:
            # High accuracy but not perfect - likely small errors
            category = ErrorCategory.PARTIAL_CORRECT
            details["accuracy"] = pixel_accuracy.accuracy
            details["incorrect_pixels"] = pixel_accuracy.total_pixels - pixel_accuracy.correct_pixels
        else:
            # Analyze color distribution differences
            pred_colors = np.unique(pred_array)
            truth_colors = np.unique(truth_array)
            color_diff = set(pred_colors) ^ set(truth_colors)

            if len(color_diff) > 0:
                category = ErrorCategory.COLOR_ERROR
                details["predicted_colors"] = pred_colors.tolist()
                details["truth_colors"] = truth_colors.tolist()
                details["missing_colors"] = list(set(truth_colors) - set(pred_colors))
                details["extra_colors"] = list(set(pred_colors) - set(truth_colors))
            else:
                # Same colors but wrong pattern
                category = ErrorCategory.PATTERN_ERROR
                details["accuracy"] = pixel_accuracy.accuracy

        return category, details

    def evaluate_task_attempt(
        self,
        task: ARCTask,
        predicted_output: list[list[int]],
        attempt_number: AttemptNumber,
        strategy_used: str,
        confidence_score: float = 0.0,
        include_diff_map: bool = False,
    ) -> TaskMetrics:
        """Evaluate a single attempt for a task.

        Args:
            task: The ARC task being evaluated
            predicted_output: The predicted output grid
            attempt_number: Which attempt this is (first or second)
            strategy_used: Name/identifier of the strategy used
            confidence_score: Confidence score from the model (0-1)
            include_diff_map: Whether to include pixel difference map

        Returns:
            TaskMetrics object with detailed evaluation results
        """
        start_time = time.perf_counter()

        # Get ground truth
        if task.test_output is None:
            raise ValueError(f"Task {task.task_id} has no ground truth output for evaluation")

        # Calculate pixel accuracy
        pixel_accuracy = self.calculate_pixel_accuracy(
            predicted_output, task.test_output, include_diff_map=include_diff_map
        )

        # Categorize errors if not perfect
        error_category = None
        error_details = {}
        if not pixel_accuracy.perfect_match:
            error_category, error_details = self.categorize_error(predicted_output, task.test_output, pixel_accuracy)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Create metrics
        metrics = TaskMetrics(
            task_id=task.task_id,
            attempt_number=attempt_number,
            pixel_accuracy=pixel_accuracy,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            error_category=error_category,
            error_details=error_details,
        )

        self.logger.info(
            "task_attempt_evaluated",
            task_id=task.task_id,
            attempt=attempt_number.value,
            accuracy=pixel_accuracy.accuracy,
            perfect_match=pixel_accuracy.perfect_match,
            strategy=strategy_used,
            processing_time_ms=processing_time_ms,
        )

        return metrics

    def evaluate_task_with_attempts(
        self,
        task: ARCTask,
        predictions: list[tuple[list[list[int]], float]],
        strategy_used: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a task with up to 2 attempts.

        Args:
            task: The ARC task to evaluate
            predictions: List of (predicted_output, confidence_score) tuples (max 2)
            strategy_used: Strategy identifier used for predictions
            metadata: Additional metadata to include in result

        Returns:
            EvaluationResult with all attempts and best result
        """
        if not predictions:
            raise ValueError("At least one prediction is required")

        if len(predictions) > 2:
            self.logger.warning(
                "too_many_predictions",
                task_id=task.task_id,
                num_predictions=len(predictions),
                max_allowed=2,
            )
            predictions = predictions[:2]  # Take only first 2

        attempts = []
        for idx, (predicted_output, confidence) in enumerate(predictions):
            attempt_num = AttemptNumber.FIRST if idx == 0 else AttemptNumber.SECOND
            metrics = self.evaluate_task_attempt(
                task=task,
                predicted_output=predicted_output,
                attempt_number=attempt_num,
                strategy_used=strategy_used,
                confidence_score=confidence,
                include_diff_map=True,  # Include diff maps for error analysis
            )
            attempts.append(metrics)

        # Create evaluation result
        result = EvaluationResult(
            task_id=task.task_id,
            strategy_used=strategy_used,
            attempts=attempts,
            metadata=metadata or {},
        )

        self.logger.info(
            "task_evaluation_complete",
            task_id=task.task_id,
            num_attempts=len(attempts),
            best_accuracy=result.final_accuracy,
            total_time_ms=result.total_processing_time_ms,
        )

        return result

    def batch_evaluate(
        self, evaluations: list[tuple[ARCTask, list[tuple[list[list[int]], float]], str]]
    ) -> list[EvaluationResult]:
        """Evaluate multiple tasks in batch.

        Args:
            evaluations: List of (task, predictions, strategy) tuples

        Returns:
            List of EvaluationResult objects
        """
        results = []
        start_time = time.perf_counter()

        for task, predictions, strategy in evaluations:
            try:
                result = self.evaluate_task_with_attempts(
                    task=task,
                    predictions=predictions,
                    strategy_used=strategy,
                )
                results.append(result)
            except Exception as e:
                self.logger.error(
                    "batch_evaluation_error",
                    task_id=task.task_id,
                    error=str(e),
                )
                # Create failed result
                failed_result = EvaluationResult(
                    task_id=task.task_id,
                    strategy_used=strategy,
                    attempts=[],
                    metadata={"evaluation_error": str(e)},
                )
                results.append(failed_result)

        total_time = (time.perf_counter() - start_time) * 1000
        avg_accuracy = sum(r.final_accuracy for r in results) / len(results) if results else 0.0

        self.logger.info(
            "batch_evaluation_complete",
            num_tasks=len(evaluations),
            num_success=sum(1 for r in results if r.attempts),
            avg_accuracy=avg_accuracy,
            total_time_ms=total_time,
        )

        return results

    def get_evaluation_summary(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Generate summary statistics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total_tasks": 0,
                "perfect_matches": 0,
                "average_accuracy": 0.0,
                "accuracy_by_attempt": {},
                "error_distribution": {},
                "processing_time_stats": {},
            }

        # Calculate statistics
        total_tasks = len(results)
        perfect_matches = sum(1 for r in results if r.best_attempt and r.best_attempt.pixel_accuracy.perfect_match)

        # Accuracy statistics
        accuracies = [r.final_accuracy for r in results if r.attempts]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        # Accuracy by attempt
        first_attempts = [a.pixel_accuracy.accuracy for r in results for a in r.attempts if a.attempt_number == AttemptNumber.FIRST]
        second_attempts = [a.pixel_accuracy.accuracy for r in results for a in r.attempts if a.attempt_number == AttemptNumber.SECOND]

        accuracy_by_attempt = {
            "first": {
                "count": len(first_attempts),
                "average": sum(first_attempts) / len(first_attempts) if first_attempts else 0.0,
                "perfect": sum(1 for a in first_attempts if a == 1.0),
            },
            "second": {
                "count": len(second_attempts),
                "average": sum(second_attempts) / len(second_attempts) if second_attempts else 0.0,
                "perfect": sum(1 for a in second_attempts if a == 1.0),
            },
        }

        # Error distribution
        error_counts = {}
        for result in results:
            for attempt in result.attempts:
                if attempt.error_category:
                    error_counts[attempt.error_category.value] = error_counts.get(attempt.error_category.value, 0) + 1

        # Processing time statistics
        processing_times = [r.total_processing_time_ms for r in results if r.attempts]
        time_stats = {
            "mean": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "min": min(processing_times) if processing_times else 0.0,
            "max": max(processing_times) if processing_times else 0.0,
            "total": sum(processing_times),
        }

        return {
            "total_tasks": total_tasks,
            "tasks_evaluated": len(accuracies),
            "perfect_matches": perfect_matches,
            "average_accuracy": avg_accuracy,
            "accuracy_distribution": {
                "0.0": sum(1 for a in accuracies if a == 0.0),
                "0.0-0.5": sum(1 for a in accuracies if 0.0 < a < 0.5),
                "0.5-0.8": sum(1 for a in accuracies if 0.5 <= a < 0.8),
                "0.8-1.0": sum(1 for a in accuracies if 0.8 <= a < 1.0),
                "1.0": sum(1 for a in accuracies if a == 1.0),
            },
            "accuracy_by_attempt": accuracy_by_attempt,
            "error_distribution": error_counts,
            "processing_time_stats": time_stats,
            "strategies_used": list({r.strategy_used for r in results}),
        }

    def get_evaluation_statistics(self) -> dict[str, Any]:
        """Get evaluation service performance statistics.

        Returns:
            Dictionary containing evaluation statistics
        """
        stats = self.evaluation_stats.copy()

        # Calculate derived metrics
        total_evals = stats["total_evaluations"]
        if total_evals > 0:
            stats["success_rate"] = stats["successful_evaluations"] / total_evals
            stats["failure_rate"] = stats["failed_evaluations"] / total_evals
            stats["avg_processing_time_ms"] = stats["total_processing_time_ms"] / stats["successful_evaluations"] if stats["successful_evaluations"] > 0 else 0.0
            stats["shape_mismatch_rate"] = stats["shape_mismatch_errors"] / total_evals
            stats["format_error_rate"] = stats["format_errors"] / total_evals
        else:
            stats.update({
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "avg_processing_time_ms": 0.0,
                "shape_mismatch_rate": 0.0,
                "format_error_rate": 0.0,
            })

        # Service health indicator
        stats["service_healthy"] = (
            stats["success_rate"] > 0.95 and  # > 95% success rate
            stats["avg_processing_time_ms"] < 1000  # < 1 second average processing
        )

        return stats

    def reset_statistics(self) -> None:
        """Reset evaluation statistics."""
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "shape_mismatch_errors": 0,
            "format_errors": 0,
            "processing_errors": 0,
            "total_processing_time_ms": 0.0,
        }

        logger.info("evaluation_statistics_reset")
