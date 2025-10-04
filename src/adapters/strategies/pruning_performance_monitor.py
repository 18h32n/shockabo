"""Performance monitoring for program pruning to track 40% time savings target.

This module tracks pruning performance metrics and validates that we achieve
the target 40% reduction in evaluation time.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import structlog

from src.domain.models import EvaluationResult, PruningMetrics

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceWindow:
    """Rolling window of performance measurements."""
    window_size: int
    measurements: deque[float] = field(default_factory=deque)
    timestamps: deque[float] = field(default_factory=deque)

    def add(self, value: float, timestamp: float | None = None) -> None:
        """Add a measurement to the window."""
        if timestamp is None:
            timestamp = time.time()

        self.measurements.append(value)
        self.timestamps.append(timestamp)

        # Maintain window size
        while len(self.measurements) > self.window_size:
            self.measurements.popleft()
            self.timestamps.popleft()

    def get_average(self) -> float:
        """Get average value in window."""
        if not self.measurements:
            return 0.0
        return sum(self.measurements) / len(self.measurements)

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value in window."""
        if not self.measurements:
            return 0.0
        return float(np.percentile(list(self.measurements), percentile))


@dataclass
class PruningPerformanceStats:
    """Comprehensive pruning performance statistics."""
    total_programs_evaluated: int = 0
    programs_pruned: int = 0
    time_saved_ms: float = 0.0
    baseline_time_ms: float = 0.0
    pruning_overhead_ms: float = 0.0
    false_negatives_detected: int = 0
    true_negatives: int = 0

    @property
    def pruning_rate(self) -> float:
        """Calculate pruning rate."""
        if self.total_programs_evaluated == 0:
            return 0.0
        return self.programs_pruned / self.total_programs_evaluated

    @property
    def false_negative_rate(self) -> float:
        """Calculate false negative rate."""
        total_rejected = self.false_negatives_detected + self.true_negatives
        if total_rejected == 0:
            return 0.0
        return self.false_negatives_detected / total_rejected

    @property
    def time_savings_percent(self) -> float:
        """Calculate percentage time saved."""
        if self.baseline_time_ms == 0:
            return 0.0
        return (self.time_saved_ms / self.baseline_time_ms) * 100

    @property
    def effective_time_with_pruning(self) -> float:
        """Calculate total time including pruning overhead."""
        return self.baseline_time_ms - self.time_saved_ms + self.pruning_overhead_ms


class PruningPerformanceMonitor:
    """Monitor and track pruning performance against targets."""

    def __init__(
        self,
        target_time_savings_percent: float = 40.0,
        target_false_negative_rate: float = 0.05,
        window_size: int = 1000
    ):
        """Initialize the performance monitor.

        Args:
            target_time_savings_percent: Target percentage time savings (default 40%)
            target_false_negative_rate: Maximum acceptable false negative rate (default 5%)
            window_size: Size of rolling window for metrics
        """
        self.target_time_savings_percent = target_time_savings_percent
        self.target_false_negative_rate = target_false_negative_rate
        self.window_size = window_size

        self.logger = structlog.get_logger(__name__).bind(
            service="pruning_performance_monitor",
            target_savings=target_time_savings_percent
        )

        # Rolling windows for key metrics
        self.time_savings_window = PerformanceWindow(window_size)
        self.false_negative_window = PerformanceWindow(window_size)
        self.pruning_time_window = PerformanceWindow(window_size)

        # Cumulative statistics
        self.cumulative_stats = PruningPerformanceStats()

        # Baseline timing estimates
        self.baseline_eval_time_ms = 50.0  # Default estimate per full evaluation

        # Alert thresholds
        self.alerts_sent = {
            "low_time_savings": False,
            "high_false_negatives": False,
            "slow_pruning": False
        }

    def record_evaluation_batch(
        self,
        num_programs: int,
        num_pruned: int,
        pruning_time_ms: float,
        full_eval_time_ms: float,
        false_negatives: int = 0
    ) -> PruningMetrics:
        """Record performance metrics for a batch of evaluations.

        Args:
            num_programs: Total number of programs in batch
            num_pruned: Number of programs pruned (not evaluated fully)
            pruning_time_ms: Total time spent on pruning decisions
            full_eval_time_ms: Time spent on full evaluation of non-pruned programs
            false_negatives: Number of false negatives detected (if known)

        Returns:
            PruningMetrics with current performance data
        """
        # Calculate time saved
        baseline_time = num_programs * self.baseline_eval_time_ms
        actual_time = pruning_time_ms + full_eval_time_ms
        time_saved = max(0, baseline_time - actual_time)

        # Update cumulative stats
        self.cumulative_stats.total_programs_evaluated += num_programs
        self.cumulative_stats.programs_pruned += num_pruned
        self.cumulative_stats.time_saved_ms += time_saved
        self.cumulative_stats.baseline_time_ms += baseline_time
        self.cumulative_stats.pruning_overhead_ms += pruning_time_ms
        self.cumulative_stats.false_negatives_detected += false_negatives
        self.cumulative_stats.true_negatives += max(0, num_pruned - false_negatives)

        # Update rolling windows
        time_savings_percent = (time_saved / baseline_time) * 100 if baseline_time > 0 else 0
        self.time_savings_window.add(time_savings_percent)

        false_negative_rate = false_negatives / num_pruned if num_pruned > 0 else 0
        self.false_negative_window.add(false_negative_rate)

        avg_pruning_time = pruning_time_ms / num_programs if num_programs > 0 else 0
        self.pruning_time_window.add(avg_pruning_time)

        # Create metrics object
        metrics = PruningMetrics(
            strategy_id="current",
            total_programs=num_programs,
            programs_pruned=num_pruned,
            pruning_rate=num_pruned / num_programs if num_programs > 0 else 0,
            false_negatives=false_negatives,
            false_negative_rate=false_negative_rate,
            avg_pruning_time_ms=avg_pruning_time,
            time_saved_ms=time_saved,
            timestamp=datetime.now()
        )

        # Check performance and send alerts if needed
        self._check_performance_targets()

        return metrics

    def update_baseline_timing(self, eval_results: list[EvaluationResult]) -> None:
        """Update baseline evaluation time estimate from actual results.

        Args:
            eval_results: Recent evaluation results with timing data
        """
        if not eval_results:
            return

        # Extract timing data from successful evaluations
        eval_times = [
            result.total_processing_time_ms
            for result in eval_results
            if result.attempts and result.total_processing_time_ms > 0
        ]

        if eval_times:
            # Use 75th percentile as baseline to be conservative
            self.baseline_eval_time_ms = float(np.percentile(eval_times, 75))

            self.logger.info(
                "baseline_timing_updated",
                new_baseline_ms=self.baseline_eval_time_ms,
                sample_size=len(eval_times)
            )

    def _check_performance_targets(self) -> None:
        """Check if performance targets are being met and alert if not."""
        # Check time savings
        current_time_savings = self.time_savings_window.get_average()
        if current_time_savings < self.target_time_savings_percent:
            if not self.alerts_sent["low_time_savings"]:
                self.logger.warning(
                    "low_time_savings_alert",
                    current_savings_percent=current_time_savings,
                    target_percent=self.target_time_savings_percent,
                    window_size=len(self.time_savings_window.measurements)
                )
                self.alerts_sent["low_time_savings"] = True
        else:
            self.alerts_sent["low_time_savings"] = False

        # Check false negative rate
        current_fn_rate = self.false_negative_window.get_average()
        if current_fn_rate > self.target_false_negative_rate:
            if not self.alerts_sent["high_false_negatives"]:
                self.logger.warning(
                    "high_false_negative_alert",
                    current_rate=current_fn_rate,
                    target_rate=self.target_false_negative_rate,
                    window_size=len(self.false_negative_window.measurements)
                )
                self.alerts_sent["high_false_negatives"] = True
        else:
            self.alerts_sent["high_false_negatives"] = False

        # Check pruning speed
        avg_pruning_time = self.pruning_time_window.get_average()
        if avg_pruning_time > 5.0:  # More than 5ms per program
            if not self.alerts_sent["slow_pruning"]:
                self.logger.warning(
                    "slow_pruning_alert",
                    avg_time_ms=avg_pruning_time,
                    target_ms=5.0
                )
                self.alerts_sent["slow_pruning"] = True
        else:
            self.alerts_sent["slow_pruning"] = False

    def get_current_performance(self) -> dict:
        """Get current performance metrics.

        Returns:
            Dictionary with current performance data
        """
        return {
            "current_metrics": {
                "time_savings_percent": self.time_savings_window.get_average(),
                "false_negative_rate": self.false_negative_window.get_average(),
                "avg_pruning_time_ms": self.pruning_time_window.get_average(),
                "pruning_rate": self.cumulative_stats.pruning_rate
            },
            "cumulative_stats": {
                "total_programs": self.cumulative_stats.total_programs_evaluated,
                "programs_pruned": self.cumulative_stats.programs_pruned,
                "time_savings_percent": self.cumulative_stats.time_savings_percent,
                "false_negative_rate": self.cumulative_stats.false_negative_rate,
                "total_time_saved_ms": self.cumulative_stats.time_saved_ms
            },
            "targets": {
                "time_savings_percent": self.target_time_savings_percent,
                "false_negative_rate": self.target_false_negative_rate
            },
            "targets_met": {
                "time_savings": self.time_savings_window.get_average() >= self.target_time_savings_percent,
                "false_negatives": self.false_negative_window.get_average() <= self.target_false_negative_rate
            }
        }

    def get_detailed_report(self) -> str:
        """Generate detailed performance report.

        Returns:
            Formatted performance report string
        """
        perf = self.get_current_performance()

        report = f"""
=== Pruning Performance Report ===

Current Performance (Rolling Window):
  - Time Savings: {perf['current_metrics']['time_savings_percent']:.1f}% (target: {perf['targets']['time_savings_percent']}%)
  - False Negative Rate: {perf['current_metrics']['false_negative_rate']:.3f} (target: <{perf['targets']['false_negative_rate']})
  - Pruning Rate: {perf['current_metrics']['pruning_rate']:.1%}
  - Avg Pruning Time: {perf['current_metrics']['avg_pruning_time_ms']:.2f}ms per program

Cumulative Statistics:
  - Total Programs: {perf['cumulative_stats']['total_programs']:,}
  - Programs Pruned: {perf['cumulative_stats']['programs_pruned']:,}
  - Total Time Saved: {perf['cumulative_stats']['total_time_saved_ms']/1000:.1f}s
  - Overall Time Savings: {perf['cumulative_stats']['time_savings_percent']:.1f}%

Target Achievement:
  - Time Savings Target: {'✓ MET' if perf['targets_met']['time_savings'] else '✗ NOT MET'}
  - False Negative Target: {'✓ MET' if perf['targets_met']['false_negatives'] else '✗ NOT MET'}

Baseline Evaluation Time: {self.baseline_eval_time_ms:.1f}ms per program
"""
        return report

    def export_metrics_for_dashboard(self) -> dict:
        """Export metrics in format suitable for monitoring dashboard.

        Returns:
            Dictionary with dashboard-ready metrics
        """
        return {
            "pruning_time_savings_gauge": self.time_savings_window.get_average(),
            "pruning_false_negative_rate_gauge": self.false_negative_window.get_average(),
            "pruning_rate_gauge": self.cumulative_stats.pruning_rate,
            "pruning_speed_histogram": list(self.pruning_time_window.measurements),
            "total_programs_counter": self.cumulative_stats.total_programs_evaluated,
            "programs_pruned_counter": self.cumulative_stats.programs_pruned,
            "time_saved_counter": self.cumulative_stats.time_saved_ms,
            "targets_met": all(self.get_current_performance()["targets_met"].values())
        }
