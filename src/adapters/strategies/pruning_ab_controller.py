"""A/B testing controller for intelligent program pruning strategies.

This module implements the A/B testing framework for comparing different
pruning strategies to optimize the balance between performance and accuracy.
"""

import logging
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

from src.domain.models import (
    PruningDecision,
    PruningMetrics,
    PruningResult,
    PruningStrategy,
)

logger = logging.getLogger(__name__)


class ABTestController:
    """Control A/B testing of pruning strategies with statistical analysis."""

    def __init__(
        self,
        strategies: list[PruningStrategy],
        exploration_rate: float = 0.1,
        min_samples: int = 1000,
        confidence_level: float = 0.95,
    ):
        """Initialize A/B test controller.

        Args:
            strategies: List of pruning strategies to test
            exploration_rate: Rate of random strategy selection for exploration
            min_samples: Minimum samples needed per strategy for significance testing
            confidence_level: Confidence level for statistical tests (0.95 = 95%)
        """
        self.strategies = {s.strategy_id: s for s in strategies}
        self.exploration_rate = exploration_rate
        self.min_samples = min_samples
        self.confidence_level = confidence_level

        # Track strategy performance metrics
        self.strategy_stats = defaultdict(lambda: {
            "total_programs": 0,
            "programs_pruned": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "time_saved_ms": [],
            "pruning_times_ms": [],
            "confidence_scores": [],
            "samples": 0,
        })

        # Track allocation percentages
        self.allocations = {
            s.strategy_id: 1.0 / len(strategies) for s in strategies
        }

        # Store recent decisions for analysis
        self.decision_history = []

    def select_strategy(
        self,
        task_features: dict[str, float] | None = None
    ) -> PruningStrategy:
        """Select pruning strategy based on allocation and past performance.

        Args:
            task_features: Optional features about the task for contextual selection

        Returns:
            Selected pruning strategy
        """
        if random.random() < self.exploration_rate:
            # Exploration: random selection
            return self._random_strategy()
        else:
            # Exploitation: weighted selection based on allocation
            return self._weighted_strategy_selection()

    def _random_strategy(self) -> PruningStrategy:
        """Select a random strategy for exploration."""
        strategy_id = random.choice(list(self.strategies.keys()))
        return self.strategies[strategy_id]

    def _weighted_strategy_selection(self) -> PruningStrategy:
        """Select strategy based on current allocation weights."""
        # Convert allocations to probabilities
        total_weight = sum(self.allocations.values())
        probabilities = [
            self.allocations[s_id] / total_weight
            for s_id in self.strategies.keys()
        ]

        # Select based on weights
        strategy_id = np.random.choice(
            list(self.strategies.keys()),
            p=probabilities
        )

        return self.strategies[strategy_id]

    def update_metrics(
        self,
        strategy_id: str,
        metrics: PruningMetrics,
        pruning_results: list[PruningResult],
        false_negatives: list[str] | None = None,
    ):
        """Update strategy performance metrics.

        Args:
            strategy_id: ID of the strategy that was used
            metrics: Overall pruning metrics
            pruning_results: Individual program pruning results
            false_negatives: List of program IDs that were incorrectly pruned
        """
        stats = self.strategy_stats[strategy_id]

        # Update counts
        stats["total_programs"] += metrics.total_programs
        stats["programs_pruned"] += metrics.programs_pruned
        stats["samples"] += 1

        # Update time metrics
        stats["time_saved_ms"].append(metrics.time_saved_ms)
        stats["pruning_times_ms"].extend([
            r.pruning_time_ms for r in pruning_results
        ])

        # Update confidence scores
        stats["confidence_scores"].extend([
            r.confidence_score for r in pruning_results
            if r.decision != PruningDecision.ACCEPT
        ])

        # Update accuracy metrics
        if false_negatives:
            stats["false_negatives"] += len(false_negatives)

        # Record decision
        self.decision_history.append({
            "timestamp": datetime.now(),
            "strategy_id": strategy_id,
            "metrics": metrics,
            "false_negative_rate": metrics.false_negative_rate,
        })

        # Check for statistical significance and adjust allocations
        if self._has_significance():
            self._adjust_allocations()

    def _has_significance(self) -> bool:
        """Check if we have enough data for statistical significance."""
        # All strategies need minimum samples
        for strategy_id in self.strategies.keys():
            strategy_stats = self.strategy_stats[strategy_id]
            if strategy_stats["samples"] < self.min_samples:
                return False
        return True

    def _adjust_allocations(self):
        """Adjust strategy allocations based on performance metrics."""
        # Calculate performance scores for each strategy
        performance_scores = {}

        for strategy_id, strategy_stats in self.strategy_stats.items():
            # Calculate composite performance score
            # Higher is better: time saved, pruning rate
            # Lower is better: false negative rate

            avg_time_saved = np.mean(strategy_stats["time_saved_ms"])
            pruning_rate = strategy_stats["programs_pruned"] / max(strategy_stats["total_programs"], 1)
            false_negative_rate = strategy_stats["false_negatives"] / max(strategy_stats["programs_pruned"], 1)

            # Composite score (weighted combination)
            # Prioritize low false negatives while maximizing time savings
            score = (
                0.4 * (avg_time_saved / 1000)  # Normalized time savings
                + 0.3 * pruning_rate  # Pruning effectiveness
                - 0.3 * (false_negative_rate * 10)  # Heavily penalize false negatives
            )

            performance_scores[strategy_id] = max(score, 0.01)  # Ensure positive

        # Run statistical tests to ensure differences are significant
        if self._test_significance(performance_scores):
            # Update allocations based on performance
            total_score = sum(performance_scores.values())

            for strategy_id, score in performance_scores.items():
                # New allocation proportional to performance
                new_allocation = score / total_score

                # Apply smoothing to prevent extreme changes
                old_allocation = self.allocations[strategy_id]
                self.allocations[strategy_id] = (
                    0.7 * new_allocation + 0.3 * old_allocation
                )

            logger.info(f"Updated allocations: {self.allocations}")

    def _test_significance(self, scores: dict[str, float]) -> bool:
        """Test if performance differences are statistically significant.

        Args:
            scores: Performance scores for each strategy

        Returns:
            True if differences are significant
        """
        # If we only have one strategy, no test needed
        if len(scores) <= 1:
            return False

        # Use ANOVA to test if there are significant differences
        strategy_times = []

        for strategy_id in scores:
            times = self.strategy_stats[strategy_id]["time_saved_ms"]
            if len(times) >= 30:  # Need sufficient samples for normality
                strategy_times.append(times[-self.min_samples:])

        if len(strategy_times) < 2:
            return False

        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*strategy_times)

        # Check if p-value indicates significant difference
        alpha = 1 - self.confidence_level
        return p_value < alpha

    def get_best_strategy(self) -> tuple[str, dict[str, float]]:
        """Get the best performing strategy based on current metrics.

        Returns:
            Tuple of (strategy_id, performance_metrics)
        """
        best_score = -float('inf')
        best_strategy = None

        for strategy_id, strategy_stats in self.strategy_stats.items():
            if strategy_stats["samples"] < 10:  # Need minimum samples
                continue

            # Calculate performance score
            avg_time_saved = np.mean(strategy_stats["time_saved_ms"])
            pruning_rate = strategy_stats["programs_pruned"] / max(strategy_stats["total_programs"], 1)
            false_negative_rate = strategy_stats["false_negatives"] / max(strategy_stats["programs_pruned"], 1)

            score = (
                0.4 * (avg_time_saved / 1000)
                + 0.3 * pruning_rate
                - 0.3 * (false_negative_rate * 10)
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy_id

        if best_strategy:
            return best_strategy, {
                "avg_time_saved_ms": np.mean(
                    self.strategy_stats[best_strategy]["time_saved_ms"]
                ),
                "pruning_rate": (
                    self.strategy_stats[best_strategy]["programs_pruned"] /
                    max(self.strategy_stats[best_strategy]["total_programs"], 1)
                ),
                "false_negative_rate": (
                    self.strategy_stats[best_strategy]["false_negatives"] /
                    max(self.strategy_stats[best_strategy]["programs_pruned"], 1)
                ),
                "samples": self.strategy_stats[best_strategy]["samples"],
            }

        return None, {}

    def get_current_allocations(self) -> dict[str, float]:
        """Get current strategy allocation percentages."""
        return self.allocations.copy()

    def get_performance_summary(self) -> dict[str, dict[str, float]]:
        """Get performance summary for all strategies.

        Returns:
            Dictionary mapping strategy_id to performance metrics
        """
        summary = {}

        for strategy_id, strategy_stats in self.strategy_stats.items():
            if strategy_stats["samples"] == 0:
                continue

            summary[strategy_id] = {
                "samples": strategy_stats["samples"],
                "avg_time_saved_ms": np.mean(strategy_stats["time_saved_ms"]) if strategy_stats["time_saved_ms"] else 0,
                "pruning_rate": strategy_stats["programs_pruned"] / max(strategy_stats["total_programs"], 1),
                "false_negative_rate": strategy_stats["false_negatives"] / max(strategy_stats["programs_pruned"], 1),
                "avg_pruning_time_ms": np.mean(strategy_stats["pruning_times_ms"]) if strategy_stats["pruning_times_ms"] else 0,
                "avg_confidence": np.mean(strategy_stats["confidence_scores"]) if strategy_stats["confidence_scores"] else 0,
                "allocation": self.allocations[strategy_id],
            }

        return summary

    def export_results(self) -> dict:
        """Export A/B test results for analysis.

        Returns:
            Dictionary containing all test results and metrics
        """
        return {
            "test_config": {
                "strategies": list(self.strategies.keys()),
                "exploration_rate": self.exploration_rate,
                "min_samples": self.min_samples,
                "confidence_level": self.confidence_level,
            },
            "performance_summary": self.get_performance_summary(),
            "best_strategy": self.get_best_strategy(),
            "current_allocations": self.get_current_allocations(),
            "decision_history": self.decision_history[-1000:],  # Last 1000 decisions
            "timestamp": datetime.now().isoformat(),
        }
