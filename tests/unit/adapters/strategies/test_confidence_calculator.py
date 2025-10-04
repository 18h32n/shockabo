"""Unit tests for ConfidenceCalculator."""

import time

import numpy as np
import pytest

from src.adapters.strategies.confidence_calculator import (
    ConfidenceCalculator,
    ProgramMetrics,
)
from src.domain.models import ARCTask


class TestConfidenceCalculatorInitialization:
    """Test ConfidenceCalculator initialization."""

    def test_default_initialization(self):
        calculator = ConfidenceCalculator()
        assert calculator.fitness_weight == 0.5
        assert calculator.diversity_weight == 0.3
        assert calculator.convergence_weight == 0.2
        assert calculator.calibration_table is not None

    def test_custom_weights(self):
        calculator = ConfidenceCalculator(
            fitness_weight=0.6, diversity_weight=0.25, convergence_weight=0.15
        )
        assert calculator.fitness_weight == 0.6
        assert calculator.diversity_weight == 0.25
        assert calculator.convergence_weight == 0.15

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ConfidenceCalculator(
                fitness_weight=0.5, diversity_weight=0.3, convergence_weight=0.3
            )


class TestFitnessBasedConfidence:
    """Test fitness-based confidence calculation."""

    def test_perfect_fitness_perfect_training(self):
        calculator = ConfidenceCalculator()
        metrics = ProgramMetrics(
            fitness=1.0,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=3,
            total_training_examples=3,
        )
        confidence = calculator.calculate_fitness_confidence(metrics)
        assert confidence > 0.9

    def test_high_fitness_partial_training(self):
        calculator = ConfidenceCalculator()
        metrics = ProgramMetrics(
            fitness=0.95,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=2,
            total_training_examples=3,
        )
        confidence = calculator.calculate_fitness_confidence(metrics)
        assert 0.5 < confidence < 0.9

    def test_zero_training_examples(self):
        calculator = ConfidenceCalculator()
        metrics = ProgramMetrics(
            fitness=1.0,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=0,
            total_training_examples=0,
        )
        confidence = calculator.calculate_fitness_confidence(metrics)
        assert confidence == 0.0

    def test_low_fitness_penalty(self):
        calculator = ConfidenceCalculator()
        metrics = ProgramMetrics(
            fitness=0.5,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=3,
            total_training_examples=3,
        )
        confidence = calculator.calculate_fitness_confidence(metrics)
        assert confidence < 0.6


class TestDiversityAdjustment:
    """Test diversity-based confidence adjustment."""

    def test_high_diversity(self):
        calculator = ConfidenceCalculator()
        metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.9, 80, 45.0, 3, 3),
            ProgramMetrics(0.8, 120, 55.0, 3, 3),
            ProgramMetrics(0.7, 90, 48.0, 3, 3),
        ]
        diversity = calculator.calculate_diversity_adjustment(metrics)
        assert diversity > 0.2

    def test_low_diversity(self):
        calculator = ConfidenceCalculator()
        metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.99, 101, 50.5, 3, 3),
            ProgramMetrics(0.98, 102, 51.0, 3, 3),
        ]
        diversity = calculator.calculate_diversity_adjustment(metrics)
        assert diversity < 0.5

    def test_single_program(self):
        calculator = ConfidenceCalculator()
        metrics = [ProgramMetrics(1.0, 100, 50.0, 3, 3)]
        diversity = calculator.calculate_diversity_adjustment(metrics)
        assert diversity == 0.5


class TestConvergenceConfidence:
    """Test convergence-based confidence calculation."""

    def test_strong_convergence(self):
        calculator = ConfidenceCalculator()
        metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.98, 105, 52.0, 3, 3),
            ProgramMetrics(0.97, 98, 49.0, 3, 3),
            ProgramMetrics(0.96, 102, 51.0, 3, 3),
            ProgramMetrics(0.95, 103, 50.5, 3, 3),
        ]
        convergence = calculator.calculate_convergence_confidence(metrics)
        assert convergence > 0.7

    def test_weak_convergence(self):
        calculator = ConfidenceCalculator()
        metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.5, 80, 45.0, 2, 3),
            ProgramMetrics(0.3, 120, 55.0, 1, 3),
            ProgramMetrics(0.2, 90, 48.0, 1, 3),
        ]
        convergence = calculator.calculate_convergence_confidence(metrics)
        assert convergence < 0.5

    def test_few_programs(self):
        calculator = ConfidenceCalculator()
        metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.9, 105, 52.0, 3, 3),
        ]
        convergence = calculator.calculate_convergence_confidence(metrics)
        assert convergence == 0.5


class TestCalibration:
    """Test confidence score calibration."""

    def test_calibrate_exact_match(self):
        calculator = ConfidenceCalculator()
        calibrated = calculator._calibrate_score(0.5)
        assert calibrated == 0.45

    def test_calibrate_interpolation(self):
        calculator = ConfidenceCalculator()
        calibrated = calculator._calibrate_score(0.75)
        assert 0.70 < calibrated < 0.82

    def test_calibrate_bounds(self):
        calculator = ConfidenceCalculator()
        assert calculator._calibrate_score(0.0) == 0.05
        assert calculator._calibrate_score(1.0) == 0.98
        assert calculator._calibrate_score(-0.1) == 0.05
        assert calculator._calibrate_score(1.5) == 0.98


class TestOverallConfidence:
    """Test overall confidence calculation."""

    def test_high_confidence_scenario(self):
        calculator = ConfidenceCalculator()
        task = ARCTask(
            task_id="test_001",
            task_source="training",
            train_examples=[],
            test_input=[],
        )

        best_metrics = ProgramMetrics(
            fitness=1.0,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=3,
            total_training_examples=3,
        )

        all_metrics = [
            ProgramMetrics(1.0, 100, 50.0, 3, 3),
            ProgramMetrics(0.98, 105, 52.0, 3, 3),
            ProgramMetrics(0.97, 98, 49.0, 3, 3),
            ProgramMetrics(0.96, 102, 51.0, 3, 3),
        ]

        confidence = calculator.calculate_confidence(task, best_metrics, all_metrics)
        assert 0.6 < confidence <= 1.0

    def test_low_confidence_scenario(self):
        calculator = ConfidenceCalculator()
        task = ARCTask(
            task_id="test_002",
            task_source="training",
            train_examples=[],
            test_input=[],
        )

        best_metrics = ProgramMetrics(
            fitness=0.5,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=1,
            total_training_examples=3,
        )

        all_metrics = [
            ProgramMetrics(0.5, 100, 50.0, 1, 3),
            ProgramMetrics(0.4, 80, 45.0, 1, 3),
            ProgramMetrics(0.3, 120, 55.0, 1, 3),
        ]

        confidence = calculator.calculate_confidence(task, best_metrics, all_metrics)
        assert 0.0 <= confidence < 0.5

    def test_confidence_bounds(self):
        calculator = ConfidenceCalculator()
        task = ARCTask(
            task_id="test_003",
            task_source="training",
            train_examples=[],
            test_input=[],
        )

        best_metrics = ProgramMetrics(
            fitness=0.8,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=2,
            total_training_examples=3,
        )

        all_metrics = [best_metrics]

        confidence = calculator.calculate_confidence(task, best_metrics, all_metrics)
        assert 0.0 <= confidence <= 1.0


class TestMultiProgramAggregation:
    """Test aggregation of multiple program confidences."""

    def test_equal_weights_aggregation(self):
        calculator = ConfidenceCalculator()
        confidences = [0.8, 0.9, 0.7]
        aggregated = calculator.aggregate_multi_program_confidence(confidences)
        assert np.isclose(aggregated, 0.8)

    def test_weighted_aggregation(self):
        calculator = ConfidenceCalculator()
        confidences = [0.9, 0.5]
        weights = [0.8, 0.2]
        aggregated = calculator.aggregate_multi_program_confidence(confidences, weights)
        assert 0.7 < aggregated < 0.9

    def test_empty_confidences(self):
        calculator = ConfidenceCalculator()
        aggregated = calculator.aggregate_multi_program_confidence([])
        assert aggregated == 0.0

    def test_mismatched_lengths(self):
        calculator = ConfidenceCalculator()
        with pytest.raises(ValueError, match="same length"):
            calculator.aggregate_multi_program_confidence([0.8, 0.9], [1.0])


class TestPerformance:
    """Test performance requirements."""

    def test_overall_confidence_performance(self):
        calculator = ConfidenceCalculator()
        task = ARCTask(
            task_id="perf_test",
            task_source="training",
            train_examples=[],
            test_input=[],
        )

        best_metrics = ProgramMetrics(
            fitness=0.95,
            program_size=100,
            execution_time_ms=50.0,
            passed_training_examples=3,
            total_training_examples=3,
        )

        all_metrics = [
            ProgramMetrics(0.95 - i * 0.05, 100 + i * 10, 50.0 + i * 5, 3, 3)
            for i in range(20)
        ]

        start = time.perf_counter()
        confidence = calculator.calculate_confidence(task, best_metrics, all_metrics)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Calculation took {elapsed_ms:.2f}ms (target: <50ms)"
        assert 0.0 <= confidence <= 1.0

    def test_aggregation_performance(self):
        calculator = ConfidenceCalculator()
        confidences = [0.8 + i * 0.01 for i in range(100)]
        weights = [1.0] * 100

        start = time.perf_counter()
        aggregated = calculator.aggregate_multi_program_confidence(confidences, weights)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Aggregation took {elapsed_ms:.2f}ms (target: <10ms)"
        assert 0.0 <= aggregated <= 1.0
