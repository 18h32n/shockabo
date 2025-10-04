"""Metrics collection system for pruning strategy comparison.

This module collects and aggregates metrics for comparing different pruning
strategies in the A/B testing framework.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.domain.models import (
    DSLProgram,
    EvaluationResult,
    PruningDecision,
    PruningMetrics,
    PruningResult,
)

logger = logging.getLogger(__name__)


class PruningMetricsCollector:
    """Collect and aggregate metrics for pruning strategy comparison."""

    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.

        Args:
            window_size: Size of sliding window for recent metrics
        """
        self.window_size = window_size

        # Store metrics by strategy
        self.strategy_metrics = defaultdict(lambda: {
            "decisions": defaultdict(int),  # Count by decision type
            "timing": {
                "pruning_times": [],
                "evaluation_times": [],
                "total_times": [],
            },
            "accuracy": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            },
            "confidence": {
                "scores": [],
                "thresholds": [],
                "correct_predictions": 0,
                "total_predictions": 0,
            },
            "resource_usage": {
                "memory_mb": [],
                "cpu_percent": [],
            },
            "program_complexity": {
                "operation_counts": [],
                "program_lengths": [],
                "rejected_at_stage": defaultdict(int),
            },
        })

        # Track overall system metrics
        self.system_metrics = {
            "total_programs_processed": 0,
            "total_time_saved_ms": 0,
            "queue_lengths": [],
            "processing_rates": [],  # Programs per second
        }

    def record_pruning_decision(
        self,
        strategy_id: str,
        program: DSLProgram,
        result: PruningResult,
        actual_evaluation: EvaluationResult | None = None,
    ):
        """Record a pruning decision and its outcome.

        Args:
            strategy_id: ID of the strategy that made the decision
            program: The program that was evaluated
            result: The pruning result
            actual_evaluation: Optional actual evaluation result for validation
        """
        metrics = self.strategy_metrics[strategy_id]

        # Record decision type
        metrics["decisions"][result.decision.value] += 1

        # Record timing
        metrics["timing"]["pruning_times"].append(result.pruning_time_ms)
        self._maintain_window(metrics["timing"]["pruning_times"])

        # Record confidence
        metrics["confidence"]["scores"].append(result.confidence_score)
        self._maintain_window(metrics["confidence"]["scores"])

        # Record program complexity
        if hasattr(program, 'operations'):
            metrics["program_complexity"]["operation_counts"].append(
                len(program.operations)
            )
            metrics["program_complexity"]["program_lengths"].append(
                sum(len(str(op)) for op in program.operations)
            )

        # Record rejection stage
        if result.decision != PruningDecision.ACCEPT:
            stage = result.decision.value.replace("reject_", "")
            metrics["program_complexity"]["rejected_at_stage"][stage] += 1

        # If we have actual evaluation, update accuracy metrics
        if actual_evaluation is not None:
            self._update_accuracy_metrics(
                strategy_id, result, actual_evaluation
            )

    def _update_accuracy_metrics(
        self,
        strategy_id: str,
        pruning_result: PruningResult,
        actual_result: EvaluationResult,
    ):
        """Update accuracy metrics based on actual evaluation results.

        Args:
            strategy_id: ID of the strategy
            pruning_result: The pruning decision
            actual_result: The actual evaluation result
        """
        metrics = self.strategy_metrics[strategy_id]["accuracy"]
        confidence_metrics = self.strategy_metrics[strategy_id]["confidence"]

        # Determine if pruning decision was correct
        program_passed = actual_result.passed and actual_result.accuracy > 0.5
        pruned = pruning_result.decision != PruningDecision.ACCEPT

        if pruned and not program_passed:
            # Correctly rejected bad program
            metrics["true_negatives"] += 1
            confidence_metrics["correct_predictions"] += 1
        elif pruned and program_passed:
            # Incorrectly rejected good program (false negative)
            metrics["false_negatives"] += 1
        elif not pruned and program_passed:
            # Correctly accepted good program
            metrics["true_positives"] += 1
            confidence_metrics["correct_predictions"] += 1
        else:
            # Incorrectly accepted bad program (false positive)
            metrics["false_positives"] += 1

        confidence_metrics["total_predictions"] += 1

    def record_batch_metrics(
        self,
        strategy_id: str,
        pruning_metrics: PruningMetrics,
        evaluation_time_ms: float,
        resource_usage: dict[str, float] | None = None,
    ):
        """Record metrics for a batch of programs.

        Args:
            strategy_id: ID of the strategy used
            pruning_metrics: Aggregated pruning metrics
            evaluation_time_ms: Time spent on actual evaluation
            resource_usage: Optional resource usage metrics
        """
        # Update timing metrics
        total_time = pruning_metrics.avg_pruning_time_ms * pruning_metrics.total_programs + evaluation_time_ms
        self.strategy_metrics[strategy_id]["timing"]["total_times"].append(total_time)
        self.strategy_metrics[strategy_id]["timing"]["evaluation_times"].append(evaluation_time_ms)

        # Update resource usage if provided
        if resource_usage:
            if "memory_mb" in resource_usage:
                self.strategy_metrics[strategy_id]["resource_usage"]["memory_mb"].append(
                    resource_usage["memory_mb"]
                )
            if "cpu_percent" in resource_usage:
                self.strategy_metrics[strategy_id]["resource_usage"]["cpu_percent"].append(
                    resource_usage["cpu_percent"]
                )

        # Update system metrics
        self.system_metrics["total_programs_processed"] += pruning_metrics.total_programs
        self.system_metrics["total_time_saved_ms"] += pruning_metrics.time_saved_ms

    def record_queue_metrics(
        self,
        queue_length: int,
        processing_rate: float,
    ):
        """Record system-wide queue metrics.

        Args:
            queue_length: Current evaluation queue length
            processing_rate: Programs processed per second
        """
        self.system_metrics["queue_lengths"].append(queue_length)
        self.system_metrics["processing_rates"].append(processing_rate)

        # Maintain window
        self._maintain_window(self.system_metrics["queue_lengths"])
        self._maintain_window(self.system_metrics["processing_rates"])

    def get_strategy_comparison(self) -> dict[str, dict[str, float]]:
        """Get comparative metrics for all strategies.

        Returns:
            Dictionary mapping strategy_id to comparison metrics
        """
        comparison = {}

        for strategy_id, metrics in self.strategy_metrics.items():
            # Calculate aggregate metrics
            total_decisions = sum(metrics["decisions"].values())
            if total_decisions == 0:
                continue

            accuracy = metrics["accuracy"]
            total_actual = sum(accuracy.values())

            # Calculate rates
            pruning_rate = 1.0 - metrics["decisions"].get("accept", 0) / max(total_decisions, 1)

            if total_actual > 0:
                precision = accuracy["true_negatives"] / max(
                    accuracy["true_negatives"] + accuracy["false_negatives"], 1
                )
                recall = accuracy["true_negatives"] / max(
                    accuracy["true_negatives"] + accuracy["false_positives"], 1
                )
                false_negative_rate = accuracy["false_negatives"] / max(total_actual, 1)
                f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
            else:
                precision = recall = false_negative_rate = f1_score = 0

            # Calculate timing metrics
            avg_pruning_time = np.mean(metrics["timing"]["pruning_times"]) if metrics["timing"]["pruning_times"] else 0
            avg_total_time = np.mean(metrics["timing"]["total_times"]) if metrics["timing"]["total_times"] else 0

            # Calculate confidence metrics
            avg_confidence = np.mean(metrics["confidence"]["scores"]) if metrics["confidence"]["scores"] else 0
            confidence_accuracy = (
                metrics["confidence"]["correct_predictions"] /
                max(metrics["confidence"]["total_predictions"], 1)
            )

            comparison[strategy_id] = {
                "pruning_rate": pruning_rate,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "false_negative_rate": false_negative_rate,
                "avg_pruning_time_ms": avg_pruning_time,
                "avg_total_time_ms": avg_total_time,
                "avg_confidence_score": avg_confidence,
                "confidence_accuracy": confidence_accuracy,
                "total_decisions": total_decisions,
                "decision_distribution": dict(metrics["decisions"]),
                "rejection_stages": dict(metrics["program_complexity"]["rejected_at_stage"]),
            }

        return comparison

    def get_efficiency_metrics(self) -> dict[str, float]:
        """Get overall efficiency metrics across all strategies.

        Returns:
            Dictionary of efficiency metrics
        """
        total_programs = self.system_metrics["total_programs_processed"]

        if total_programs == 0:
            return {}

        avg_queue_length = np.mean(self.system_metrics["queue_lengths"]) if self.system_metrics["queue_lengths"] else 0
        avg_processing_rate = np.mean(self.system_metrics["processing_rates"]) if self.system_metrics["processing_rates"] else 0

        return {
            "total_programs_processed": total_programs,
            "total_time_saved_ms": self.system_metrics["total_time_saved_ms"],
            "avg_time_saved_per_program_ms": self.system_metrics["total_time_saved_ms"] / total_programs,
            "avg_queue_length": avg_queue_length,
            "avg_processing_rate": avg_processing_rate,
            "efficiency_gain_percent": (
                self.system_metrics["total_time_saved_ms"] /
                (self.system_metrics["total_time_saved_ms"] + total_programs * 50) * 100  # Assume 50ms baseline
            ),
        }

    def get_resource_usage_summary(self, strategy_id: str) -> dict[str, float]:
        """Get resource usage summary for a specific strategy.

        Args:
            strategy_id: ID of the strategy

        Returns:
            Dictionary of resource usage metrics
        """
        resources = self.strategy_metrics[strategy_id]["resource_usage"]

        return {
            "avg_memory_mb": np.mean(resources["memory_mb"]) if resources["memory_mb"] else 0,
            "peak_memory_mb": max(resources["memory_mb"]) if resources["memory_mb"] else 0,
            "avg_cpu_percent": np.mean(resources["cpu_percent"]) if resources["cpu_percent"] else 0,
            "peak_cpu_percent": max(resources["cpu_percent"]) if resources["cpu_percent"] else 0,
        }

    def export_metrics(self, filepath: str):
        """Export all collected metrics to a file.

        Args:
            filepath: Path to export metrics to
        """
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_comparison": self.get_strategy_comparison(),
            "efficiency_metrics": self.get_efficiency_metrics(),
            "resource_usage": {
                strategy_id: self.get_resource_usage_summary(strategy_id)
                for strategy_id in self.strategy_metrics.keys()
            },
            "raw_metrics": {
                strategy_id: {
                    "decisions": dict(metrics["decisions"]),
                    "timing_summary": {
                        "pruning": {
                            "mean": np.mean(metrics["timing"]["pruning_times"]) if metrics["timing"]["pruning_times"] else 0,
                            "std": np.std(metrics["timing"]["pruning_times"]) if metrics["timing"]["pruning_times"] else 0,
                            "min": min(metrics["timing"]["pruning_times"]) if metrics["timing"]["pruning_times"] else 0,
                            "max": max(metrics["timing"]["pruning_times"]) if metrics["timing"]["pruning_times"] else 0,
                        },
                    },
                    "accuracy": dict(metrics["accuracy"]),
                    "confidence": {
                        "mean_score": np.mean(metrics["confidence"]["scores"]) if metrics["confidence"]["scores"] else 0,
                        "accuracy": (
                            metrics["confidence"]["correct_predictions"] /
                            max(metrics["confidence"]["total_predictions"], 1)
                        ),
                    },
                }
                for strategy_id, metrics in self.strategy_metrics.items()
            },
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported metrics to {filepath}")

    def _maintain_window(self, data_list: list):
        """Maintain sliding window size for a list.

        Args:
            data_list: List to maintain window size for
        """
        if len(data_list) > self.window_size:
            # Keep only the most recent entries
            del data_list[:-self.window_size]
