"""Confidence scoring for program synthesis strategy.

This module calculates confidence scores for evolved programs to enable
ensemble integration and strategy selection in Epic 3.
"""

from dataclasses import dataclass

import numpy as np

from src.domain.models import ARCTask


@dataclass
class ProgramMetrics:
    """Metrics for a single evolved program."""

    fitness: float
    program_size: int
    execution_time_ms: float
    passed_training_examples: int
    total_training_examples: int


class ConfidenceCalculator:
    """Calculate confidence scores for program synthesis outputs.

    Combines multiple signals to produce calibrated confidence scores:
    - Fitness: How well the program performs on training examples
    - Diversity: How different successful programs are from each other
    - Convergence: How stable the evolution process was

    The confidence scores are calibrated against historical success rates
    to enable accurate ensemble weighting in Epic 3.

    Performance:
        <50ms per calculation for real-time ensemble decisions

    Example:
        calculator = ConfidenceCalculator()
        programs = evolution_engine.evolve(task)
        confidence = calculator.calculate_confidence(
            task=task,
            best_program_metrics=programs[0].metrics,
            all_program_metrics=[p.metrics for p in programs[:10]]
        )
    """

    def __init__(
        self,
        fitness_weight: float = 0.5,
        diversity_weight: float = 0.3,
        convergence_weight: float = 0.2,
        calibration_table: dict[float, float] | None = None,
    ):
        """Initialize confidence calculator with configurable weights.

        Args:
            fitness_weight: Weight for fitness-based confidence (0.0-1.0)
            diversity_weight: Weight for diversity-based adjustment (0.0-1.0)
            convergence_weight: Weight for convergence-based confidence (0.0-1.0)
            calibration_table: Optional mapping of raw scores to calibrated probabilities

        Note:
            Weights should sum to 1.0 for normalized scoring.
        """
        if not np.isclose(fitness_weight + diversity_weight + convergence_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")

        self.fitness_weight = fitness_weight
        self.diversity_weight = diversity_weight
        self.convergence_weight = convergence_weight

        self.calibration_table = calibration_table or self._default_calibration_table()

    def _default_calibration_table(self) -> dict[float, float]:
        """Default calibration mapping raw scores to success probabilities.

        Based on historical performance of program synthesis strategy.
        """
        return {
            0.0: 0.05,
            0.1: 0.10,
            0.2: 0.15,
            0.3: 0.25,
            0.4: 0.35,
            0.5: 0.45,
            0.6: 0.55,
            0.7: 0.70,
            0.8: 0.82,
            0.9: 0.92,
            1.0: 0.98,
        }

    def calculate_fitness_confidence(self, metrics: ProgramMetrics) -> float:
        """Calculate confidence based on program fitness.

        Args:
            metrics: Metrics for the best program

        Returns:
            Fitness-based confidence score (0.0-1.0)

        Performance:
            <5ms per calculation
        """
        if metrics.total_training_examples == 0:
            return 0.0

        training_accuracy = metrics.passed_training_examples / metrics.total_training_examples

        penalty = 0.0
        if metrics.fitness < 1.0:
            penalty = 0.1 * (1.0 - metrics.fitness)

        confidence = training_accuracy * metrics.fitness - penalty

        return max(0.0, min(1.0, confidence))

    def calculate_diversity_adjustment(
        self, all_program_metrics: list[ProgramMetrics]
    ) -> float:
        """Calculate diversity-based confidence adjustment.

        Higher diversity among top programs indicates robustness.
        Lower diversity may indicate overfitting or search convergence to local optimum.

        Args:
            all_program_metrics: Metrics for top N programs (typically N=10-20)

        Returns:
            Diversity adjustment factor (0.0-1.0)

        Performance:
            <10ms per calculation
        """
        if len(all_program_metrics) < 2:
            return 0.5

        fitness_values = [m.fitness for m in all_program_metrics]
        size_values = [m.program_size for m in all_program_metrics]

        fitness_std = np.std(fitness_values)
        size_std = np.std(size_values)

        fitness_diversity = min(1.0, float(fitness_std / 0.3))
        size_diversity = min(1.0, float(size_std / 50.0))

        diversity_score = 0.7 * fitness_diversity + 0.3 * size_diversity

        return float(diversity_score)

    def calculate_convergence_confidence(
        self, all_program_metrics: list[ProgramMetrics]
    ) -> float:
        """Calculate confidence based on evolutionary convergence.

        Stable convergence with multiple high-fitness programs indicates
        a reliable solution. Erratic fitness or single outliers suggest uncertainty.

        Args:
            all_program_metrics: Metrics for top N programs across generations

        Returns:
            Convergence-based confidence (0.0-1.0)

        Performance:
            <10ms per calculation
        """
        if len(all_program_metrics) < 3:
            return 0.5

        fitness_values = [m.fitness for m in all_program_metrics]
        top_fitness = fitness_values[0]

        high_fitness_count = sum(1 for f in fitness_values if f >= top_fitness * 0.9)

        convergence_ratio = high_fitness_count / len(all_program_metrics)

        fitness_stability = (
            1.0 - float(np.std(fitness_values[:5])) if len(fitness_values) >= 5 else 0.5
        )

        convergence_score = 0.6 * convergence_ratio + 0.4 * fitness_stability

        return max(0.0, min(1.0, convergence_score))

    def calculate_confidence(
        self,
        task: ARCTask,
        best_program_metrics: ProgramMetrics,
        all_program_metrics: list[ProgramMetrics],
    ) -> float:
        """Calculate overall calibrated confidence score.

        Combines fitness, diversity, and convergence signals with configurable
        weights, then applies calibration to map to historical success rates.

        Args:
            task: The ARC task being solved (for context-aware scoring)
            best_program_metrics: Metrics for the best evolved program
            all_program_metrics: Metrics for top N programs (N=10-20 recommended)

        Returns:
            Calibrated confidence score (0.0-1.0) representing estimated
            probability of correctness on the test output

        Performance:
            <50ms total for ensemble integration

        Example:
            >>> calculator = ConfidenceCalculator()
            >>> confidence = calculator.calculate_confidence(
            ...     task=task,
            ...     best_program_metrics=ProgramMetrics(
            ...         fitness=0.95,
            ...         program_size=120,
            ...         execution_time_ms=50.0,
            ...         passed_training_examples=3,
            ...         total_training_examples=3
            ...     ),
            ...     all_program_metrics=[...]
            ... )
            >>> assert 0.0 <= confidence <= 1.0
        """
        fitness_conf = self.calculate_fitness_confidence(best_program_metrics)
        diversity_adj = self.calculate_diversity_adjustment(all_program_metrics)
        convergence_conf = self.calculate_convergence_confidence(all_program_metrics)

        raw_confidence = (
            self.fitness_weight * fitness_conf
            + self.diversity_weight * diversity_adj
            + self.convergence_weight * convergence_conf
        )

        calibrated_confidence = self._calibrate_score(raw_confidence)

        return max(0.0, min(1.0, calibrated_confidence))

    def _calibrate_score(self, raw_score: float) -> float:
        """Apply calibration mapping to raw confidence score.

        Uses linear interpolation between calibration table entries.

        Args:
            raw_score: Raw confidence before calibration (0.0-1.0)

        Returns:
            Calibrated confidence representing true success probability
        """
        if raw_score <= 0.0:
            return self.calibration_table[0.0]
        if raw_score >= 1.0:
            return self.calibration_table[1.0]

        lower_key = int(raw_score * 10) / 10
        upper_key = min(1.0, lower_key + 0.1)

        if lower_key not in self.calibration_table or upper_key not in self.calibration_table:
            return raw_score

        lower_val = self.calibration_table[lower_key]
        upper_val = self.calibration_table[upper_key]

        fraction = (raw_score - lower_key) / 0.1
        interpolated = lower_val + fraction * (upper_val - lower_val)

        return interpolated

    def aggregate_multi_program_confidence(
        self, program_confidences: list[float], program_weights: list[float] | None = None
    ) -> float:
        """Aggregate confidence scores from multiple programs.

        Used when combining predictions from multiple top programs for
        ensemble voting or uncertainty estimation.

        Args:
            program_confidences: Confidence scores for each program (0.0-1.0)
            program_weights: Optional weights for each program (e.g., based on fitness)

        Returns:
            Aggregated confidence score (0.0-1.0)

        Performance:
            <10ms per calculation
        """
        if not program_confidences:
            return 0.0

        if program_weights is None:
            program_weights = [1.0] * len(program_confidences)

        if len(program_confidences) != len(program_weights):
            raise ValueError("Confidence and weight lists must have same length")

        total_weight = sum(program_weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            c * w for c, w in zip(program_confidences, program_weights, strict=False)
        )
        aggregated = weighted_sum / total_weight

        return max(0.0, min(1.0, aggregated))
