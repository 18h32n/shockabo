"""Unit tests for evaluation service with pixel-perfect accuracy calculation."""

from datetime import datetime

import numpy as np
import pytest

from src.domain.models import ARCTask
from src.domain.services.evaluation_service import (
    AttemptNumber,
    ErrorCategory,
    EvaluationResult,
    EvaluationService,
    PixelAccuracy,
    TaskMetrics,
)


class TestPixelAccuracy:
    """Test pixel accuracy calculation."""

    def test_perfect_match(self):
        """Test perfect match calculation."""
        service = EvaluationService()
        predicted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 9
        assert result.correct_pixels == 9
        assert result.accuracy == 1.0
        assert result.perfect_match is True
        assert result.pixel_diff_map is None

    def test_partial_match(self):
        """Test partial match calculation."""
        service = EvaluationService()
        predicted = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # Last pixel different
        ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 9
        assert result.correct_pixels == 8
        assert result.accuracy == pytest.approx(8 / 9)
        assert result.perfect_match is False

    def test_complete_mismatch(self):
        """Test complete mismatch calculation."""
        service = EvaluationService()
        predicted = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 9
        assert result.correct_pixels == 0
        assert result.accuracy == 0.0
        assert result.perfect_match is False

    def test_shape_mismatch(self):
        """Test shape mismatch handling."""
        service = EvaluationService()
        predicted = [[1, 2], [3, 4]]  # 2x2
        ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3x3

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 9  # Ground truth size
        assert result.correct_pixels == 0
        assert result.accuracy == 0.0
        assert result.perfect_match is False

    def test_diff_map_generation(self):
        """Test difference map generation."""
        service = EvaluationService()
        predicted = [[1, 0], [0, 1]]  # Diagonal different
        ground_truth = [[1, 1], [1, 1]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth, include_diff_map=True)

        assert result.pixel_diff_map == [[0, 1], [1, 0]]  # 1 indicates error
        assert result.accuracy == 0.5

    def test_empty_grids(self):
        """Test handling of empty grids."""
        service = EvaluationService()
        predicted = []
        ground_truth = []

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 0
        assert result.correct_pixels == 0
        assert result.accuracy == 0.0
        assert result.perfect_match is True  # Empty grids are considered perfect match


class TestErrorCategorization:
    """Test error categorization functionality."""

    def test_shape_mismatch_error(self):
        """Test shape mismatch error categorization."""
        service = EvaluationService()
        predicted = [[1, 2], [3, 4]]
        ground_truth = [[1, 2, 3], [4, 5, 6]]
        pixel_acc = PixelAccuracy(total_pixels=6, correct_pixels=0, accuracy=0.0, perfect_match=False)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category == ErrorCategory.SHAPE_MISMATCH
        assert "pred_shape" in details
        assert "truth_shape" in details
        assert details["pred_shape"] == (2, 2)
        assert details["truth_shape"] == (2, 3)

    def test_color_error(self):
        """Test color error categorization."""
        service = EvaluationService()
        predicted = [[1, 1], [1, 1]]  # Only uses color 1
        ground_truth = [[1, 2], [3, 4]]  # Uses colors 1, 2, 3, 4
        pixel_acc = PixelAccuracy(total_pixels=4, correct_pixels=1, accuracy=0.25, perfect_match=False)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category == ErrorCategory.COLOR_ERROR
        assert set(details["missing_colors"]) == {2, 3, 4}
        assert details["extra_colors"] == []

    def test_pattern_error(self):
        """Test pattern error categorization."""
        service = EvaluationService()
        predicted = [[1, 2], [2, 1]]  # Same colors, wrong pattern
        ground_truth = [[1, 2], [1, 2]]
        pixel_acc = PixelAccuracy(total_pixels=4, correct_pixels=2, accuracy=0.5, perfect_match=False)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category == ErrorCategory.PATTERN_ERROR
        assert details["accuracy"] == 0.5

    def test_partial_correct_error(self):
        """Test partial correct categorization."""
        service = EvaluationService()
        predicted = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # One pixel wrong
        ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pixel_acc = PixelAccuracy(total_pixels=9, correct_pixels=8, accuracy=8 / 9, perfect_match=False)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category == ErrorCategory.PARTIAL_CORRECT
        assert details["incorrect_pixels"] == 1
        assert details["accuracy"] == pytest.approx(8 / 9)

    def test_complete_failure_error(self):
        """Test complete failure categorization."""
        service = EvaluationService()
        predicted = [[0, 0], [0, 0]]
        ground_truth = [[1, 2], [3, 4]]
        pixel_acc = PixelAccuracy(total_pixels=4, correct_pixels=0, accuracy=0.0, perfect_match=False)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category == ErrorCategory.COMPLETE_FAILURE
        assert details["accuracy"] == 0.0

    def test_perfect_match_no_error(self):
        """Test no error for perfect match."""
        service = EvaluationService()
        predicted = [[1, 2], [3, 4]]
        ground_truth = [[1, 2], [3, 4]]
        pixel_acc = PixelAccuracy(total_pixels=4, correct_pixels=4, accuracy=1.0, perfect_match=True)

        category, details = service.categorize_error(predicted, ground_truth, pixel_acc)

        assert category is None
        assert details == {}


class TestTaskEvaluation:
    """Test task-level evaluation functionality."""

    def test_single_attempt_evaluation(self):
        """Test evaluation of a single attempt."""
        service = EvaluationService()
        task = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=[[5, 6], [7, 8]],
        )
        predicted = [[5, 6], [7, 0]]  # One error

        metrics = service.evaluate_task_attempt(
            task=task,
            predicted_output=predicted,
            attempt_number=AttemptNumber.FIRST,
            strategy_used="test_strategy",
            confidence_score=0.85,
        )

        assert metrics.task_id == "test_001"
        assert metrics.attempt_number == AttemptNumber.FIRST
        assert metrics.pixel_accuracy.accuracy == 0.75
        assert metrics.confidence_score == 0.85
        assert metrics.error_category == ErrorCategory.PARTIAL_CORRECT
        assert metrics.processing_time_ms > 0

    def test_missing_ground_truth(self):
        """Test handling of missing ground truth."""
        service = EvaluationService()
        task = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=None,  # No ground truth
        )

        with pytest.raises(ValueError, match="has no ground truth"):
            service.evaluate_task_attempt(
                task=task,
                predicted_output=[[5, 6], [7, 8]],
                attempt_number=AttemptNumber.FIRST,
                strategy_used="test_strategy",
            )

    def test_two_attempt_evaluation(self):
        """Test evaluation with two attempts."""
        service = EvaluationService()
        task = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=[[5, 6], [7, 8]],
        )

        predictions = [
            ([[5, 0], [0, 8]], 0.6),  # First attempt - 50% accuracy
            ([[5, 6], [7, 8]], 0.9),  # Second attempt - perfect
        ]

        result = service.evaluate_task_with_attempts(
            task=task,
            predictions=predictions,
            strategy_used="test_strategy",
        )

        assert len(result.attempts) == 2
        assert result.attempts[0].attempt_number == AttemptNumber.FIRST
        assert result.attempts[0].pixel_accuracy.accuracy == 0.5
        assert result.attempts[1].attempt_number == AttemptNumber.SECOND
        assert result.attempts[1].pixel_accuracy.accuracy == 1.0
        assert result.best_attempt == result.attempts[1]
        assert result.final_accuracy == 1.0

    def test_max_two_attempts(self):
        """Test that only 2 attempts are evaluated even if more provided."""
        service = EvaluationService()
        task = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=[[5, 6], [7, 8]],
        )

        # Provide 3 predictions
        predictions = [
            ([[0, 0], [0, 0]], 0.3),
            ([[5, 6], [0, 0]], 0.6),
            ([[5, 6], [7, 8]], 0.9),  # This should be ignored
        ]

        result = service.evaluate_task_with_attempts(
            task=task,
            predictions=predictions,
            strategy_used="test_strategy",
        )

        assert len(result.attempts) == 2  # Only first 2 evaluated

    def test_evaluation_with_metadata(self):
        """Test evaluation with custom metadata."""
        service = EvaluationService()
        task = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=[[5, 6], [7, 8]],
        )

        metadata = {"model_version": "1.0", "temperature": 0.7}
        result = service.evaluate_task_with_attempts(
            task=task,
            predictions=[([[5, 6], [7, 8]], 0.9)],
            strategy_used="test_strategy",
            metadata=metadata,
        )

        assert result.metadata == metadata


class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    def test_batch_evaluate_success(self):
        """Test successful batch evaluation."""
        service = EvaluationService()

        # Create test tasks
        tasks = []
        evaluations = []
        for i in range(3):
            task = ARCTask(
                task_id=f"test_{i:03d}",
                task_source="test",
                train_examples=[],
                test_input=[[1, 2], [3, 4]],
                test_output=[[5, 6], [7, 8]],
            )
            tasks.append(task)
            predictions = [([[5, 6], [7, 8]], 0.9)]  # Perfect prediction
            evaluations.append((task, predictions, f"strategy_{i}"))

        results = service.batch_evaluate(evaluations)

        assert len(results) == 3
        assert all(r.final_accuracy == 1.0 for r in results)
        assert all(r.task_id == f"test_{i:03d}" for i, r in enumerate(results))

    def test_batch_evaluate_with_failures(self):
        """Test batch evaluation with some failures."""
        service = EvaluationService()

        # Create mix of valid and invalid tasks
        task1 = ARCTask(
            task_id="test_001",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=[[5, 6], [7, 8]],
        )

        task2 = ARCTask(
            task_id="test_002",
            task_source="test",
            train_examples=[],
            test_input=[[1, 2], [3, 4]],
            test_output=None,  # No ground truth - will fail
        )

        evaluations = [
            (task1, [([[5, 6], [7, 8]], 0.9)], "strategy_1"),
            (task2, [([[5, 6], [7, 8]], 0.9)], "strategy_2"),
        ]

        results = service.batch_evaluate(evaluations)

        assert len(results) == 2
        assert results[0].final_accuracy == 1.0
        assert results[1].final_accuracy == 0.0  # Failed task
        assert "evaluation_error" in results[1].metadata


class TestEvaluationSummary:
    """Test evaluation summary generation."""

    def test_empty_summary(self):
        """Test summary for empty results."""
        service = EvaluationService()
        summary = service.get_evaluation_summary([])

        assert summary["total_tasks"] == 0
        assert summary["perfect_matches"] == 0
        assert summary["average_accuracy"] == 0.0

    def test_comprehensive_summary(self):
        """Test comprehensive summary generation."""
        service = EvaluationService()

        # Create diverse results
        results = []

        # Perfect match task
        metrics1 = TaskMetrics(
            task_id="test_001",
            attempt_number=AttemptNumber.FIRST,
            pixel_accuracy=PixelAccuracy(9, 9, 1.0, True),
            processing_time_ms=10.0,
        )
        result1 = EvaluationResult(
            task_id="test_001",
            strategy_used="strategy_a",
            attempts=[metrics1],
        )
        results.append(result1)

        # Partial match with two attempts
        metrics2a = TaskMetrics(
            task_id="test_002",
            attempt_number=AttemptNumber.FIRST,
            pixel_accuracy=PixelAccuracy(9, 5, 5 / 9, False),
            processing_time_ms=15.0,
            error_category=ErrorCategory.PATTERN_ERROR,
        )
        metrics2b = TaskMetrics(
            task_id="test_002",
            attempt_number=AttemptNumber.SECOND,
            pixel_accuracy=PixelAccuracy(9, 7, 7 / 9, False),
            processing_time_ms=12.0,
            error_category=ErrorCategory.PARTIAL_CORRECT,
        )
        result2 = EvaluationResult(
            task_id="test_002",
            strategy_used="strategy_b",
            attempts=[metrics2a, metrics2b],
        )
        results.append(result2)

        # Complete failure
        metrics3 = TaskMetrics(
            task_id="test_003",
            attempt_number=AttemptNumber.FIRST,
            pixel_accuracy=PixelAccuracy(9, 0, 0.0, False),
            processing_time_ms=5.0,
            error_category=ErrorCategory.COMPLETE_FAILURE,
        )
        result3 = EvaluationResult(
            task_id="test_003",
            strategy_used="strategy_a",
            attempts=[metrics3],
        )
        results.append(result3)

        summary = service.get_evaluation_summary(results)

        assert summary["total_tasks"] == 3
        assert summary["tasks_evaluated"] == 3
        assert summary["perfect_matches"] == 1
        assert summary["average_accuracy"] == pytest.approx((1.0 + 7 / 9 + 0.0) / 3)

        # Check accuracy distribution
        assert summary["accuracy_distribution"]["1.0"] == 1
        assert summary["accuracy_distribution"]["0.0"] == 1
        assert summary["accuracy_distribution"]["0.5-0.8"] == 1

        # Check attempt statistics
        assert summary["accuracy_by_attempt"]["first"]["count"] == 3
        assert summary["accuracy_by_attempt"]["second"]["count"] == 1

        # Check error distribution
        assert summary["error_distribution"]["pattern_error"] == 1
        assert summary["error_distribution"]["partial_correct"] == 1
        assert summary["error_distribution"]["complete_failure"] == 1

        # Check processing time stats
        assert summary["processing_time_stats"]["total"] == 42.0
        assert summary["processing_time_stats"]["mean"] == 14.0

        # Check strategies used
        assert set(summary["strategies_used"]) == {"strategy_a", "strategy_b"}


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_pixel_grid(self):
        """Test evaluation of single pixel grids."""
        service = EvaluationService()
        predicted = [[5]]
        ground_truth = [[5]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.total_pixels == 1
        assert result.correct_pixels == 1
        assert result.accuracy == 1.0
        assert result.perfect_match is True

    def test_large_grid_performance(self):
        """Test performance with large grids."""
        service = EvaluationService()
        size = 100
        predicted = np.random.randint(0, 10, (size, size)).tolist()
        ground_truth = np.random.randint(0, 10, (size, size)).tolist()

        import time

        start = time.time()
        result = service.calculate_pixel_accuracy(predicted, ground_truth)
        duration = time.time() - start

        assert result.total_pixels == size * size
        assert duration < 0.1  # Should be fast even for large grids

    def test_irregular_shaped_grids(self):
        """Test grids with different row lengths (should handle as shape mismatch)."""
        service = EvaluationService()
        predicted = [[1, 2, 3], [4, 5]]  # Irregular
        ground_truth = [[1, 2], [3, 4]]

        # This should handle the irregular shape gracefully
        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.accuracy == 0.0  # Shape mismatch

    def test_unicode_and_special_values(self):
        """Test handling of special values in grids."""
        service = EvaluationService()
        predicted = [[0, -1], [999, 10]]
        ground_truth = [[0, -1], [999, 10]]

        result = service.calculate_pixel_accuracy(predicted, ground_truth)

        assert result.accuracy == 1.0
        assert result.perfect_match is True