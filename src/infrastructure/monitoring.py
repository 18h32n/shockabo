"""Metrics collection and Prometheus integration for cross-strategy monitoring.

This module provides shared metrics collection infrastructure for all solving
strategies to enable unified performance tracking and ensemble optimization.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock


@dataclass
class MetricValue:
    """A single metric observation."""

    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and export metrics for strategy performance monitoring.

    Provides Prometheus-compatible metrics for:
    - Strategy solve duration (histogram)
    - Strategy accuracy by task type (gauge)
    - Confidence score distribution (histogram)
    - Resource usage (gauge)
    - API call counts (counter)

    Thread-safe for concurrent strategy execution.

    Example:
        metrics = MetricsCollector()

        start = time.time()
        output = await strategy.solve_task(task)
        duration = time.time() - start

        metrics.record_solve_duration(
            strategy="program_synthesis",
            duration_seconds=duration,
            task_type="transformation"
        )

        metrics.record_accuracy(
            strategy="program_synthesis",
            task_type="transformation",
            correct=True
        )
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = Lock()
        self._solve_durations: dict[str, list[MetricValue]] = defaultdict(list)
        self._accuracy_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
        self._confidence_scores: dict[str, list[MetricValue]] = defaultdict(list)
        self._resource_usage: dict[str, dict[str, MetricValue]] = defaultdict(dict)
        self._api_call_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._start_time = datetime.now()

    def record_solve_duration(
        self,
        strategy: str,
        duration_seconds: float,
        task_type: str | None = None,
        task_difficulty: str | None = None,
    ) -> None:
        """Record strategy solve duration.

        Args:
            strategy: Strategy name (e.g., "program_synthesis")
            duration_seconds: Time taken to solve task
            task_type: Optional task type for granular metrics
            task_difficulty: Optional difficulty level (easy, medium, hard)
        """
        labels = {"strategy": strategy}
        if task_type:
            labels["task_type"] = task_type
        if task_difficulty:
            labels["difficulty"] = task_difficulty

        metric = MetricValue(
            value=duration_seconds, timestamp=datetime.now(), labels=labels
        )

        with self._lock:
            key = self._make_key(strategy, task_type, task_difficulty)
            self._solve_durations[key].append(metric)

    def record_accuracy(
        self,
        strategy: str,
        task_type: str,
        correct: bool,
        task_difficulty: str | None = None,
    ) -> None:
        """Record strategy accuracy result.

        Args:
            strategy: Strategy name
            task_type: Type of task (e.g., "transformation", "pattern")
            correct: Whether prediction was correct
            task_difficulty: Optional difficulty level
        """
        with self._lock:
            key = self._make_key(strategy, task_type, task_difficulty)
            self._accuracy_counts[key]["total"] += 1
            if correct:
                self._accuracy_counts[key]["correct"] += 1

    def record_confidence_score(
        self,
        strategy: str,
        confidence: float,
        task_type: str | None = None,
    ) -> None:
        """Record strategy confidence score.

        Args:
            strategy: Strategy name
            confidence: Confidence score (0.0-1.0)
            task_type: Optional task type
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence {confidence} not in range [0.0, 1.0]")

        labels = {"strategy": strategy}
        if task_type:
            labels["task_type"] = task_type

        metric = MetricValue(value=confidence, timestamp=datetime.now(), labels=labels)

        with self._lock:
            key = self._make_key(strategy, task_type)
            self._confidence_scores[key].append(metric)

    def record_resource_usage(
        self,
        strategy: str,
        resource_type: str,
        value: float,
        task_id: str | None = None,
    ) -> None:
        """Record resource usage metric.

        Args:
            strategy: Strategy name
            resource_type: Type of resource (cpu_seconds, memory_mb, gpu_memory_mb)
            value: Resource value
            task_id: Optional task identifier
        """
        labels = {"strategy": strategy, "resource_type": resource_type}
        if task_id:
            labels["task_id"] = task_id

        metric = MetricValue(value=value, timestamp=datetime.now(), labels=labels)

        with self._lock:
            key = f"{strategy}_{resource_type}"
            self._resource_usage[key] = {resource_type: metric}

    def record_api_call(
        self, strategy: str, tier: str, count: int = 1
    ) -> None:
        """Record API call count.

        Args:
            strategy: Strategy name
            tier: API tier (e.g., "tier1", "tier2")
            count: Number of calls (default: 1)
        """
        with self._lock:
            self._api_call_counts[strategy][tier] += count

    def get_accuracy(
        self, strategy: str, task_type: str | None = None
    ) -> float:
        """Get current accuracy for a strategy.

        Args:
            strategy: Strategy name
            task_type: Optional task type filter

        Returns:
            Accuracy ratio (0.0-1.0), or 0.0 if no data
        """
        with self._lock:
            key = self._make_key(strategy, task_type)
            counts = self._accuracy_counts.get(key, {"correct": 0, "total": 0})
            if counts["total"] == 0:
                return 0.0
            return counts["correct"] / counts["total"]

    def get_average_duration(
        self, strategy: str, task_type: str | None = None
    ) -> float:
        """Get average solve duration for a strategy.

        Args:
            strategy: Strategy name
            task_type: Optional task type filter

        Returns:
            Average duration in seconds, or 0.0 if no data
        """
        with self._lock:
            key = self._make_key(strategy, task_type)
            durations = self._solve_durations.get(key, [])
            if not durations:
                return 0.0
            return sum(m.value for m in durations) / len(durations)

    def get_confidence_distribution(
        self, strategy: str, task_type: str | None = None
    ) -> dict[str, float]:
        """Get confidence score distribution statistics.

        Args:
            strategy: Strategy name
            task_type: Optional task type filter

        Returns:
            Dict with min, max, mean, median, p90, p95 confidence scores
        """
        import numpy as np

        with self._lock:
            key = self._make_key(strategy, task_type)
            scores = self._confidence_scores.get(key, [])
            if not scores:
                return {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                }

            values = np.array([m.value for m in scores])
            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
            }

    def get_api_call_totals(self, strategy: str) -> dict[str, int]:
        """Get total API calls by tier for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dict mapping tier to call count
        """
        with self._lock:
            return dict(self._api_call_counts.get(strategy, {}))

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string

        Note:
            This provides compatibility with Prometheus for monitoring
            and alerting. Metrics are prefixed with 'arc_strategy_'.
        """
        lines = []

        self._export_duration_metrics(lines)
        self._export_accuracy_metrics(lines)
        self._export_confidence_metrics(lines)
        self._export_api_call_metrics(lines)

        return "\n".join(lines)

    def _export_duration_metrics(self, lines: list[str]) -> None:
        """Export solve duration metrics."""
        lines.append("# HELP arc_strategy_solve_duration_seconds Strategy solve duration")
        lines.append("# TYPE arc_strategy_solve_duration_seconds histogram")
        with self._lock:
            for _, metrics in self._solve_durations.items():
                if metrics:
                    strategy = metrics[0].labels.get("strategy", "unknown")
                    task_type = metrics[0].labels.get("task_type", "all")
                    for metric in metrics:
                        labels = f'strategy="{strategy}",task_type="{task_type}"'
                        lines.append(
                            f"arc_strategy_solve_duration_seconds{{{labels}}} {metric.value}"
                        )

    def _export_accuracy_metrics(self, lines: list[str]) -> None:
        """Export accuracy metrics."""
        lines.append("# HELP arc_strategy_accuracy Strategy accuracy ratio")
        lines.append("# TYPE arc_strategy_accuracy gauge")
        with self._lock:
            for key, counts in self._accuracy_counts.items():
                parts = key.split("_")
                if len(parts) >= 2:
                    strategy, task_type = parts[0], parts[1] if len(parts) > 1 else "all"
                    accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                    labels = f'strategy="{strategy}",task_type="{task_type}"'
                    lines.append(f"arc_strategy_accuracy{{{labels}}} {accuracy}")

    def _export_confidence_metrics(self, lines: list[str]) -> None:
        """Export confidence score metrics."""
        lines.append("# HELP arc_strategy_confidence_score Strategy confidence distribution")
        lines.append("# TYPE arc_strategy_confidence_score histogram")
        with self._lock:
            for _, metrics in self._confidence_scores.items():
                if metrics:
                    strategy = metrics[0].labels.get("strategy", "unknown")
                    for metric in metrics:
                        labels = f'strategy="{strategy}"'
                        lines.append(f"arc_strategy_confidence_score{{{labels}}} {metric.value}")

    def _export_api_call_metrics(self, lines: list[str]) -> None:
        """Export API call count metrics."""
        lines.append("# HELP arc_strategy_api_calls_total Total API calls by tier")
        lines.append("# TYPE arc_strategy_api_calls_total counter")
        with self._lock:
            for strategy, tiers in self._api_call_counts.items():
                for tier, count in tiers.items():
                    labels = f'strategy="{strategy}",tier="{tier}"'
                    lines.append(f"arc_strategy_api_calls_total{{{labels}}} {count}")

    def _make_key(self, *parts: str | None) -> str:
        """Create a key from non-None parts."""
        return "_".join(str(p) for p in parts if p is not None)

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._solve_durations.clear()
            self._accuracy_counts.clear()
            self._confidence_scores.clear()
            self._resource_usage.clear()
            self._api_call_counts.clear()
            self._start_time = datetime.now()


_global_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Singleton MetricsCollector instance
    """
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector
