"""Dashboard data aggregation service for real-time metrics and visualization.

This service aggregates evaluation metrics, experiment progress, and system performance
data for real-time dashboard display.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.domain.evaluation_models import (
    DashboardMetrics,
    ExperimentMetrics,
    ExperimentRun,
    RegressionAlert,
    ResourceUsage,
)
from src.domain.services.evaluation_service import EvaluationResult

logger = structlog.get_logger(__name__)


class MetricsWindow:
    """Time-windowed metrics collection for rolling statistics."""

    def __init__(self, window_duration: timedelta, max_size: int = 10000):
        """Initialize metrics window.

        Args:
            window_duration: Time window for metrics retention
            max_size: Maximum number of entries to keep
        """
        self.window_duration = window_duration
        self.max_size = max_size
        self.metrics: deque[tuple[datetime, dict[str, Any]]] = deque(maxlen=max_size)

    def add(self, metric: dict[str, Any]) -> None:
        """Add a metric to the window."""
        self.metrics.append((datetime.now(), metric))
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove metrics outside the time window."""
        cutoff_time = datetime.now() - self.window_duration
        while self.metrics and self.metrics[0][0] < cutoff_time:
            self.metrics.popleft()

    def get_metrics(self) -> list[dict[str, Any]]:
        """Get all metrics within the window."""
        self._cleanup()
        return [metric for _, metric in self.metrics]

    def get_count(self) -> int:
        """Get count of metrics in window."""
        self._cleanup()
        return len(self.metrics)

    def get_average(self, key: str) -> float:
        """Calculate average of a numeric metric."""
        metrics = self.get_metrics()
        if not metrics:
            return 0.0
        values = [m.get(key, 0) for m in metrics if key in m]
        return sum(values) / len(values) if values else 0.0


class DashboardAggregator:
    """Service for aggregating and computing dashboard metrics."""

    def __init__(self):
        """Initialize the dashboard aggregator."""
        self.logger = structlog.get_logger(__name__).bind(service="dashboard_aggregator")

        # Time windows for different metrics
        self.task_metrics_1h = MetricsWindow(timedelta(hours=1))
        self.task_metrics_24h = MetricsWindow(timedelta(hours=24))
        self.error_metrics_1h = MetricsWindow(timedelta(hours=1))
        self.resource_metrics_5m = MetricsWindow(timedelta(minutes=5))

        # Active experiments tracking
        self.active_experiments: dict[str, ExperimentRun] = {}

        # Strategy performance tracking
        self.strategy_metrics: dict[str, MetricsWindow] = defaultdict(
            lambda: MetricsWindow(timedelta(hours=1))
        )

        # Recent alerts
        self.recent_alerts: deque[RegressionAlert] = deque(maxlen=100)

        # System health status
        self.system_health: dict[str, str] = {
            "evaluation_service": "healthy",
            "dashboard_service": "healthy",
            "database": "healthy",
            "cache": "healthy",
        }

        # Processing queue (simulated)
        self._processing_queue_size = 0

    def record_task_evaluation(self, result: EvaluationResult) -> None:
        """Record a task evaluation result.

        Args:
            result: Evaluation result to record
        """
        if not result.best_attempt:
            return

        metric = {
            "task_id": result.task_id,
            "strategy": result.strategy_used,
            "accuracy": result.final_accuracy,
            "perfect_match": result.best_attempt.pixel_accuracy.perfect_match,
            "processing_time_ms": result.total_processing_time_ms,
            "timestamp": result.created_at,
        }

        # Add to time windows
        self.task_metrics_1h.add(metric)
        self.task_metrics_24h.add(metric)

        # Add to strategy-specific metrics
        self.strategy_metrics[result.strategy_used].add(metric)

        # Record errors if any
        if result.final_accuracy < 1.0:
            error_metric = {
                "task_id": result.task_id,
                "error_category": (
                    result.best_attempt.error_category.value
                    if result.best_attempt.error_category
                    else "unknown"
                ),
                "accuracy": result.final_accuracy,
            }
            self.error_metrics_1h.add(error_metric)

        self.logger.info(
            "task_evaluation_recorded",
            task_id=result.task_id,
            accuracy=result.final_accuracy,
            strategy=result.strategy_used,
        )

    def record_resource_usage(self, usage: ResourceUsage) -> None:
        """Record resource usage metrics.

        Args:
            usage: Resource usage data
        """
        metric = {
            "task_id": usage.task_id,
            "cpu_seconds": usage.cpu_seconds,
            "memory_mb": usage.memory_mb,
            "gpu_memory_mb": usage.gpu_memory_mb,
            "total_tokens": usage.total_tokens,
            "estimated_cost": usage.estimated_cost,
            "timestamp": usage.timestamp,
        }
        self.resource_metrics_5m.add(metric)

    def start_experiment(self, experiment: ExperimentRun) -> None:
        """Register a new experiment.

        Args:
            experiment: Experiment run to track
        """
        self.active_experiments[experiment.run_id] = experiment
        self.logger.info(
            "experiment_started",
            experiment_id=experiment.run_id,
            num_tasks=len(experiment.task_ids),
        )

    def update_experiment(self, experiment_id: str, metrics: dict[str, float]) -> None:
        """Update experiment metrics.

        Args:
            experiment_id: ID of the experiment
            metrics: Updated metrics
        """
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id].update_metrics(metrics)

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as completed.

        Args:
            experiment_id: ID of the experiment
        """
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            experiment.mark_completed()
            # Keep completed experiments for a while before cleanup
            asyncio.create_task(self._cleanup_experiment_later(experiment_id))

    async def _cleanup_experiment_later(self, experiment_id: str, delay_seconds: int = 300):
        """Clean up experiment data after a delay.

        Args:
            experiment_id: ID of the experiment
            delay_seconds: Seconds to wait before cleanup
        """
        await asyncio.sleep(delay_seconds)
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
            self.logger.info("experiment_cleaned_up", experiment_id=experiment_id)

    def add_regression_alert(self, alert: RegressionAlert) -> None:
        """Add a regression alert.

        Args:
            alert: Regression alert to add
        """
        self.recent_alerts.append(alert)
        self.logger.warning(
            "regression_alert_added",
            alert_id=alert.alert_id,
            metric=alert.metric_name,
            severity=alert.severity,
            regression_pct=alert.regression_percentage,
        )

    def update_system_health(self, component: str, status: str) -> None:
        """Update system health status.

        Args:
            component: System component name
            status: Health status ('healthy', 'degraded', 'unhealthy')
        """
        self.system_health[component] = status
        if status != "healthy":
            self.logger.warning(
                "system_health_changed",
                component=component,
                status=status,
            )

    def set_processing_queue_size(self, size: int) -> None:
        """Update processing queue size.

        Args:
            size: Current queue size
        """
        self._processing_queue_size = size

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics snapshot.

        Returns:
            DashboardMetrics object with current state
        """
        # Calculate metrics from windows
        tasks_1h = self.task_metrics_1h.get_metrics()
        tasks_processed = len(tasks_1h)
        avg_accuracy = sum(t["accuracy"] for t in tasks_1h) / tasks_processed if tasks_processed else 0.0

        # Calculate error rate
        errors_1h = self.error_metrics_1h.get_count()
        error_rate = errors_1h / tasks_processed if tasks_processed else 0.0

        # Get resource utilization
        resource_metrics = self.resource_metrics_5m.get_metrics()
        if resource_metrics:
            cpu_util = sum(r["cpu_seconds"] for r in resource_metrics) / len(resource_metrics) * 100 / 5
            memory_util = max(r["memory_mb"] for r in resource_metrics) / 16384 * 100  # Assume 16GB RAM
            gpu_util = max(r.get("gpu_memory_mb", 0) for r in resource_metrics) / 8192 * 100 if any(r.get("gpu_memory_mb") for r in resource_metrics) else 0.0
        else:
            cpu_util = memory_util = gpu_util = 0.0

        resource_utilization = {
            "cpu": round(cpu_util, 1),
            "memory": round(memory_util, 1),
            "gpu": round(gpu_util, 1),
        }

        # Calculate top performing strategies
        strategy_performance = []
        for strategy_name, window in self.strategy_metrics.items():
            metrics = window.get_metrics()
            if metrics:
                avg_acc = sum(m["accuracy"] for m in metrics) / len(metrics)
                strategy_performance.append((strategy_name, avg_acc))

        # Sort by accuracy descending
        strategy_performance.sort(key=lambda x: x[1], reverse=True)

        return DashboardMetrics(
            timestamp=datetime.now(),
            active_experiments=len(
                [e for e in self.active_experiments.values() if not e.is_finished]
            ),
            tasks_processed_last_hour=tasks_processed,
            average_accuracy_last_hour=avg_accuracy,
            resource_utilization=resource_utilization,
            processing_queue_size=self._processing_queue_size,
            error_rate_last_hour=error_rate,
            top_performing_strategies=strategy_performance[:10],  # Top 10
            recent_alerts=list(self.recent_alerts)[:20],  # Most recent 20
            system_health=self.system_health.copy(),
        )

    def get_experiment_metrics(self, experiment_id: str) -> ExperimentMetrics | None:
        """Get metrics for a specific experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            ExperimentMetrics or None if not found
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return None

        # Calculate metrics from experiment data
        total_tasks = len(experiment.task_ids)
        successful_tasks = experiment.metrics.get("successful_tasks", 0)
        failed_tasks = experiment.metrics.get("failed_tasks", 0)
        average_accuracy = experiment.metrics.get("average_accuracy", 0.0)
        perfect_matches = experiment.metrics.get("perfect_matches", 0)

        return ExperimentMetrics(
            experiment_id=experiment_id,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            average_accuracy=average_accuracy,
            perfect_matches=perfect_matches,
            total_processing_time_ms=experiment.metrics.get("total_processing_time_ms", 0.0),
            total_resource_cost=experiment.metrics.get("total_cost", 0.0),
            strategy_performance={experiment.strategy_config.get("strategy", "unknown"): experiment.metrics},
            error_distribution=experiment.metrics.get("error_distribution", {}),
        )

    def get_strategy_summary(
        self, time_window: timedelta = timedelta(hours=1)
    ) -> dict[str, dict[str, float]]:
        """Get performance summary for all strategies.

        Args:
            time_window: Time window for metrics

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        summary = {}

        for strategy_name, window in self.strategy_metrics.items():
            metrics = window.get_metrics()
            if not metrics:
                continue

            summary[strategy_name] = {
                "tasks_evaluated": len(metrics),
                "average_accuracy": sum(m["accuracy"] for m in metrics) / len(metrics),
                "perfect_matches": sum(1 for m in metrics if m.get("perfect_match", False)),
                "average_processing_time_ms": sum(m["processing_time_ms"] for m in metrics) / len(metrics),
            }

        return summary


# Global aggregator instance
_dashboard_aggregator: DashboardAggregator | None = None


def get_dashboard_aggregator() -> DashboardAggregator:
    """Get the global dashboard aggregator instance.

    Returns:
        DashboardAggregator instance
    """
    global _dashboard_aggregator
    if _dashboard_aggregator is None:
        _dashboard_aggregator = DashboardAggregator()
    return _dashboard_aggregator
