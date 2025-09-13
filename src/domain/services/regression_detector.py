"""Regression detection service for automated performance monitoring.

This service provides baseline comparison algorithms and automated alerting
for detecting performance regressions between experiment runs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy import stats

from src.domain.evaluation_models import (
    ExperimentMetrics,
    RegressionAlert,
)
from src.domain.services.dashboard_aggregator import get_dashboard_aggregator
from src.domain.services.evaluation_service import EvaluationResult

logger = structlog.get_logger(__name__)


class RegressionMetric(Enum):
    """Metrics tracked for regression detection."""

    ACCURACY = "accuracy"
    PROCESSING_TIME = "processing_time"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    PERFECT_MATCH_RATE = "perfect_match_rate"
    COST_PER_TASK = "cost_per_task"


class RegressionSeverity(Enum):
    """Severity levels for regression alerts."""

    LOW = "low"  # < 5% regression
    MEDIUM = "medium"  # 5-10% regression
    HIGH = "high"  # 10-20% regression
    CRITICAL = "critical"  # > 20% regression


@dataclass
class BaselineMetrics:
    """Baseline metrics for comparison."""

    experiment_id: str
    metrics: dict[str, float]
    task_results: list[EvaluationResult]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric: RegressionMetric) -> float | None:
        """Get a specific metric value.

        Args:
            metric: Metric to retrieve

        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(metric.value)


@dataclass
class RegressionAnalysis:
    """Detailed regression analysis results."""

    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percentage_change: float
    is_regression: bool
    severity: RegressionSeverity
    statistical_significance: float  # p-value
    confidence_interval: tuple[float, float]
    affected_tasks: list[str]
    detailed_comparison: dict[str, Any]


class RegressionDetector:
    """Service for detecting performance regressions."""

    def __init__(self, baseline_storage_path: str = "./baselines"):
        """Initialize regression detector.

        Args:
            baseline_storage_path: Path to store baseline data
        """
        self.logger = structlog.get_logger(__name__).bind(service="regression_detector")
        self.baseline_path = Path(baseline_storage_path)
        self.baseline_path.mkdir(parents=True, exist_ok=True)

        # Regression thresholds
        self.thresholds = {
            RegressionMetric.ACCURACY: {"threshold": 0.02, "higher_is_better": True},
            RegressionMetric.PROCESSING_TIME: {"threshold": 0.10, "higher_is_better": False},
            RegressionMetric.MEMORY_USAGE: {"threshold": 0.15, "higher_is_better": False},
            RegressionMetric.ERROR_RATE: {"threshold": 0.05, "higher_is_better": False},
            RegressionMetric.PERFECT_MATCH_RATE: {"threshold": 0.05, "higher_is_better": True},
            RegressionMetric.COST_PER_TASK: {"threshold": 0.20, "higher_is_better": False},
        }

    def save_baseline(
        self,
        experiment_id: str,
        metrics: ExperimentMetrics,
        results: list[EvaluationResult],
        name: str | None = None,
    ) -> BaselineMetrics:
        """Save experiment results as a baseline.

        Args:
            experiment_id: ID of the experiment
            metrics: Experiment metrics
            results: List of evaluation results
            name: Optional name for the baseline

        Returns:
            BaselineMetrics object
        """
        # Extract metrics
        baseline_metrics = {
            RegressionMetric.ACCURACY.value: metrics.average_accuracy,
            RegressionMetric.PERFECT_MATCH_RATE.value: metrics.perfect_match_rate,
            RegressionMetric.PROCESSING_TIME.value: metrics.average_processing_time_ms,
            RegressionMetric.ERROR_RATE.value: (
                metrics.failed_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0.0
            ),
            RegressionMetric.COST_PER_TASK.value: (
                metrics.total_resource_cost / metrics.total_tasks if metrics.total_tasks > 0 else 0.0
            ),
        }

        baseline = BaselineMetrics(
            experiment_id=experiment_id,
            metrics=baseline_metrics,
            task_results=results,
            timestamp=datetime.now(),
            metadata={"name": name or experiment_id},
        )

        # Save to file
        baseline_file = self.baseline_path / f"{experiment_id}_baseline.json"
        self._save_baseline_to_file(baseline, baseline_file)

        self.logger.info(
            "baseline_saved",
            experiment_id=experiment_id,
            metrics=list(baseline_metrics.keys()),
        )

        return baseline

    def load_baseline(self, baseline_id: str) -> BaselineMetrics | None:
        """Load a baseline from storage.

        Args:
            baseline_id: ID of the baseline to load

        Returns:
            BaselineMetrics object or None if not found
        """
        baseline_file = self.baseline_path / f"{baseline_id}_baseline.json"
        if not baseline_file.exists():
            self.logger.warning("baseline_not_found", baseline_id=baseline_id)
            return None

        return self._load_baseline_from_file(baseline_file)

    def compare_to_baseline(
        self,
        current_metrics: ExperimentMetrics,
        current_results: list[EvaluationResult],
        baseline: BaselineMetrics,
    ) -> list[RegressionAnalysis]:
        """Compare current results to a baseline.

        Args:
            current_metrics: Current experiment metrics
            current_results: Current evaluation results
            baseline: Baseline to compare against

        Returns:
            List of regression analyses for each metric
        """
        analyses = []

        # Compare each metric
        for metric in RegressionMetric:
            analysis = self._analyze_metric_regression(
                metric, current_metrics, current_results, baseline
            )
            if analysis:
                analyses.append(analysis)

        return analyses

    def _analyze_metric_regression(
        self,
        metric: RegressionMetric,
        current_metrics: ExperimentMetrics,
        current_results: list[EvaluationResult],
        baseline: BaselineMetrics,
    ) -> RegressionAnalysis | None:
        """Analyze regression for a specific metric.

        Args:
            metric: Metric to analyze
            current_metrics: Current experiment metrics
            current_results: Current evaluation results
            baseline: Baseline metrics

        Returns:
            RegressionAnalysis or None if metric not available
        """
        # Get baseline value
        baseline_value = baseline.get_metric(metric)
        if baseline_value is None:
            return None

        # Get current value
        current_value = self._extract_metric_value(metric, current_metrics, current_results)
        if current_value is None:
            return None

        # Calculate changes
        absolute_change = current_value - baseline_value
        percentage_change = (
            (absolute_change / baseline_value * 100) if baseline_value != 0 else 0.0
        )

        # Determine if this is a regression
        config = self.thresholds[metric]
        is_regression = False

        if config["higher_is_better"]:
            # For metrics where higher is better (e.g., accuracy)
            is_regression = percentage_change < -config["threshold"] * 100
        else:
            # For metrics where lower is better (e.g., time, cost)
            is_regression = percentage_change > config["threshold"] * 100

        # Calculate severity
        severity = self._calculate_severity(abs(percentage_change))

        # Statistical significance (simplified)
        p_value = self._calculate_statistical_significance(
            metric, current_results, baseline.task_results
        )

        # Confidence interval
        ci = self._calculate_confidence_interval(
            metric, current_results, baseline.task_results
        )

        # Find affected tasks
        affected_tasks = self._find_affected_tasks(
            metric, current_results, baseline.task_results, config["threshold"]
        )

        # Detailed comparison
        detailed_comparison = {
            "baseline_stats": self._calculate_metric_stats(metric, baseline.task_results),
            "current_stats": self._calculate_metric_stats(metric, current_results),
            "distribution_shift": self._analyze_distribution_shift(
                metric, current_results, baseline.task_results
            ),
        }

        return RegressionAnalysis(
            metric_name=metric.value,
            baseline_value=baseline_value,
            current_value=current_value,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            is_regression=is_regression,
            severity=severity,
            statistical_significance=p_value,
            confidence_interval=ci,
            affected_tasks=affected_tasks,
            detailed_comparison=detailed_comparison,
        )

    def _extract_metric_value(
        self,
        metric: RegressionMetric,
        experiment_metrics: ExperimentMetrics,
        results: list[EvaluationResult],
    ) -> float | None:
        """Extract metric value from experiment data.

        Args:
            metric: Metric to extract
            experiment_metrics: Experiment metrics
            results: Evaluation results

        Returns:
            Metric value or None
        """
        if metric == RegressionMetric.ACCURACY:
            return experiment_metrics.average_accuracy
        elif metric == RegressionMetric.PERFECT_MATCH_RATE:
            return experiment_metrics.perfect_match_rate
        elif metric == RegressionMetric.PROCESSING_TIME:
            return experiment_metrics.average_processing_time_ms
        elif metric == RegressionMetric.ERROR_RATE:
            total = experiment_metrics.total_tasks
            return experiment_metrics.failed_tasks / total if total > 0 else 0.0
        elif metric == RegressionMetric.COST_PER_TASK:
            total = experiment_metrics.total_tasks
            return experiment_metrics.total_resource_cost / total if total > 0 else 0.0
        elif metric == RegressionMetric.MEMORY_USAGE:
            # Extract from results if available
            memory_values = []
            for result in results:
                if "memory_mb" in result.metadata:
                    memory_values.append(result.metadata["memory_mb"])
            return np.mean(memory_values) if memory_values else None

        return None

    def _calculate_severity(self, percentage_change: float) -> RegressionSeverity:
        """Calculate regression severity based on percentage change.

        Args:
            percentage_change: Absolute percentage change

        Returns:
            RegressionSeverity
        """
        if percentage_change < 5:
            return RegressionSeverity.LOW
        elif percentage_change < 10:
            return RegressionSeverity.MEDIUM
        elif percentage_change < 20:
            return RegressionSeverity.HIGH
        else:
            return RegressionSeverity.CRITICAL

    def _calculate_statistical_significance(
        self,
        metric: RegressionMetric,
        current_results: list[EvaluationResult],
        baseline_results: list[EvaluationResult],
    ) -> float:
        """Calculate statistical significance of the difference.

        Args:
            metric: Metric to analyze
            current_results: Current results
            baseline_results: Baseline results

        Returns:
            p-value from statistical test
        """
        # Extract metric values for each task
        current_values = []
        baseline_values = []

        for result in current_results:
            value = self._get_task_metric_value(metric, result)
            if value is not None:
                current_values.append(value)

        for result in baseline_results:
            value = self._get_task_metric_value(metric, result)
            if value is not None:
                baseline_values.append(value)

        if not current_values or not baseline_values:
            return 1.0  # No significance if no data

        # Use appropriate statistical test
        if len(current_values) >= 30 and len(baseline_values) >= 30:
            # Use t-test for large samples
            _, p_value = stats.ttest_ind(current_values, baseline_values)
        else:
            # Use Mann-Whitney U test for small samples
            _, p_value = stats.mannwhitneyu(
                current_values, baseline_values, alternative="two-sided"
            )

        return p_value

    def _calculate_confidence_interval(
        self,
        metric: RegressionMetric,
        current_results: list[EvaluationResult],
        baseline_results: list[EvaluationResult],
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for the difference.

        Args:
            metric: Metric to analyze
            current_results: Current results
            baseline_results: Baseline results
            confidence_level: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Extract differences for each matched task
        differences = []

        current_by_task = {r.task_id: r for r in current_results}
        baseline_by_task = {r.task_id: r for r in baseline_results}

        for task_id in set(current_by_task.keys()) & set(baseline_by_task.keys()):
            current_value = self._get_task_metric_value(metric, current_by_task[task_id])
            baseline_value = self._get_task_metric_value(metric, baseline_by_task[task_id])

            if current_value is not None and baseline_value is not None:
                differences.append(current_value - baseline_value)

        if not differences:
            return (0.0, 0.0)

        # Calculate confidence interval using bootstrap or t-distribution
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)

        # Use t-distribution for confidence interval
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin = t_value * std_diff / np.sqrt(n)

        return (mean_diff - margin, mean_diff + margin)

    def _find_affected_tasks(
        self,
        metric: RegressionMetric,
        current_results: list[EvaluationResult],
        baseline_results: list[EvaluationResult],
        threshold: float,
    ) -> list[str]:
        """Find tasks most affected by regression.

        Args:
            metric: Metric to analyze
            current_results: Current results
            baseline_results: Baseline results
            threshold: Regression threshold

        Returns:
            List of affected task IDs
        """
        affected_tasks = []

        current_by_task = {r.task_id: r for r in current_results}
        baseline_by_task = {r.task_id: r for r in baseline_results}

        for task_id in set(current_by_task.keys()) & set(baseline_by_task.keys()):
            current_value = self._get_task_metric_value(metric, current_by_task[task_id])
            baseline_value = self._get_task_metric_value(metric, baseline_by_task[task_id])

            if current_value is not None and baseline_value is not None:
                # Calculate relative change
                if baseline_value != 0:
                    change = abs((current_value - baseline_value) / baseline_value)
                    if change > threshold:
                        affected_tasks.append(task_id)

        return affected_tasks

    def _get_task_metric_value(
        self, metric: RegressionMetric, result: EvaluationResult
    ) -> float | None:
        """Get metric value for a specific task.

        Args:
            metric: Metric to extract
            result: Task evaluation result

        Returns:
            Metric value or None
        """
        if not result.best_attempt:
            return None

        if metric == RegressionMetric.ACCURACY:
            return result.final_accuracy
        elif metric == RegressionMetric.PERFECT_MATCH_RATE:
            return 1.0 if result.best_attempt.pixel_accuracy.perfect_match else 0.0
        elif metric == RegressionMetric.PROCESSING_TIME:
            return result.total_processing_time_ms
        elif metric == RegressionMetric.ERROR_RATE:
            return 0.0 if result.final_accuracy > 0 else 1.0
        elif metric == RegressionMetric.COST_PER_TASK:
            return result.metadata.get("cost", 0.0)
        elif metric == RegressionMetric.MEMORY_USAGE:
            return result.metadata.get("memory_mb", 0.0)

        return None

    def _calculate_metric_stats(
        self, metric: RegressionMetric, results: list[EvaluationResult]
    ) -> dict[str, float]:
        """Calculate statistical summary for a metric.

        Args:
            metric: Metric to analyze
            results: Evaluation results

        Returns:
            Dictionary with statistical measures
        """
        values = []
        for result in results:
            value = self._get_task_metric_value(metric, result)
            if value is not None:
                values.append(value)

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "percentile_25": np.percentile(values, 25),
            "percentile_75": np.percentile(values, 75),
        }

    def _analyze_distribution_shift(
        self,
        metric: RegressionMetric,
        current_results: list[EvaluationResult],
        baseline_results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Analyze distribution shift between baseline and current.

        Args:
            metric: Metric to analyze
            current_results: Current results
            baseline_results: Baseline results

        Returns:
            Dictionary with distribution analysis
        """
        current_values = []
        baseline_values = []

        for result in current_results:
            value = self._get_task_metric_value(metric, result)
            if value is not None:
                current_values.append(value)

        for result in baseline_results:
            value = self._get_task_metric_value(metric, result)
            if value is not None:
                baseline_values.append(value)

        if not current_values or not baseline_values:
            return {"ks_statistic": None, "ks_pvalue": None}

        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_pvalue = stats.ks_2samp(current_values, baseline_values)

        return {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "distribution_changed": ks_pvalue < 0.05,
        }

    def create_alerts(
        self,
        analyses: list[RegressionAnalysis],
        experiment_id: str,
        baseline_id: str,
    ) -> list[RegressionAlert]:
        """Create alerts for detected regressions.

        Args:
            analyses: List of regression analyses
            experiment_id: Current experiment ID
            baseline_id: Baseline experiment ID

        Returns:
            List of regression alerts
        """
        alerts = []

        for analysis in analyses:
            if not analysis.is_regression:
                continue

            alert = RegressionAlert(
                alert_id=f"alert_{experiment_id}_{analysis.metric_name}_{datetime.now().timestamp()}",
                experiment_id=experiment_id,
                baseline_experiment_id=baseline_id,
                metric_name=analysis.metric_name,
                baseline_value=analysis.baseline_value,
                current_value=analysis.current_value,
                regression_percentage=abs(analysis.percentage_change),
                severity=analysis.severity.value,
                affected_tasks=analysis.affected_tasks[:10],  # Limit to 10 tasks
                details={
                    "statistical_significance": analysis.statistical_significance,
                    "confidence_interval": analysis.confidence_interval,
                    "detailed_stats": analysis.detailed_comparison,
                },
            )

            alerts.append(alert)

            # Send to dashboard aggregator
            aggregator = get_dashboard_aggregator()
            aggregator.add_regression_alert(alert)

            self.logger.warning(
                "regression_detected",
                metric=analysis.metric_name,
                severity=analysis.severity.value,
                regression_pct=abs(analysis.percentage_change),
                affected_tasks_count=len(analysis.affected_tasks),
            )

        return alerts

    def generate_regression_report(
        self,
        analyses: list[RegressionAnalysis],
        experiment_id: str,
        baseline_id: str,
    ) -> dict[str, Any]:
        """Generate a comprehensive regression report.

        Args:
            analyses: List of regression analyses
            experiment_id: Current experiment ID
            baseline_id: Baseline experiment ID

        Returns:
            Report dictionary
        """
        regressions = [a for a in analyses if a.is_regression]
        improvements = [a for a in analyses if not a.is_regression and a.percentage_change != 0]

        report = {
            "experiment_id": experiment_id,
            "baseline_id": baseline_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_metrics": len(analyses),
                "regressions_detected": len(regressions),
                "improvements_detected": len(improvements),
                "critical_regressions": sum(
                    1 for r in regressions if r.severity == RegressionSeverity.CRITICAL
                ),
            },
            "regressions": [
                {
                    "metric": r.metric_name,
                    "severity": r.severity.value,
                    "baseline_value": r.baseline_value,
                    "current_value": r.current_value,
                    "change_percentage": r.percentage_change,
                    "p_value": r.statistical_significance,
                    "affected_tasks_count": len(r.affected_tasks),
                }
                for r in regressions
            ],
            "improvements": [
                {
                    "metric": i.metric_name,
                    "baseline_value": i.baseline_value,
                    "current_value": i.current_value,
                    "improvement_percentage": abs(i.percentage_change),
                }
                for i in improvements
            ],
            "detailed_analyses": [self._format_detailed_analysis(a) for a in analyses],
        }

        return report

    def _format_detailed_analysis(self, analysis: RegressionAnalysis) -> dict[str, Any]:
        """Format detailed analysis for report.

        Args:
            analysis: Regression analysis

        Returns:
            Formatted analysis dictionary
        """
        return {
            "metric": analysis.metric_name,
            "baseline_value": analysis.baseline_value,
            "current_value": analysis.current_value,
            "absolute_change": analysis.absolute_change,
            "percentage_change": analysis.percentage_change,
            "is_regression": analysis.is_regression,
            "severity": analysis.severity.value if analysis.is_regression else None,
            "statistical_significance": analysis.statistical_significance,
            "confidence_interval": analysis.confidence_interval,
            "significant": analysis.statistical_significance < 0.05,
            "affected_tasks": analysis.affected_tasks[:5],  # Top 5
            "baseline_stats": analysis.detailed_comparison.get("baseline_stats", {}),
            "current_stats": analysis.detailed_comparison.get("current_stats", {}),
            "distribution_shift": analysis.detailed_comparison.get("distribution_shift", {}),
        }

    def _save_baseline_to_file(self, baseline: BaselineMetrics, file_path: Path):
        """Save baseline to JSON file.

        Args:
            baseline: Baseline to save
            file_path: Path to save file
        """
        # Convert to serializable format
        data = {
            "experiment_id": baseline.experiment_id,
            "metrics": baseline.metrics,
            "timestamp": baseline.timestamp.isoformat(),
            "metadata": baseline.metadata,
            "task_count": len(baseline.task_results),
            "task_accuracies": {
                r.task_id: r.final_accuracy for r in baseline.task_results
            },
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_baseline_from_file(self, file_path: Path) -> BaselineMetrics:
        """Load baseline from JSON file.

        Args:
            file_path: Path to load from

        Returns:
            BaselineMetrics object
        """
        with open(file_path) as f:
            data = json.load(f)

        return BaselineMetrics(
            experiment_id=data["experiment_id"],
            metrics=data["metrics"],
            task_results=[],  # Not fully restored for efficiency
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"],
        )


# Global detector instance
_regression_detector: RegressionDetector | None = None


def get_regression_detector() -> RegressionDetector:
    """Get the global regression detector instance.

    Returns:
        RegressionDetector instance
    """
    global _regression_detector
    if _regression_detector is None:
        _regression_detector = RegressionDetector()
    return _regression_detector
