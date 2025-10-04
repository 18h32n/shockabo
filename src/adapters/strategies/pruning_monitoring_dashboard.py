"""Monitoring dashboard for intelligent program pruning metrics.

This module provides real-time monitoring and visualization of pruning
performance metrics.
"""

import logging
from collections import deque
from datetime import datetime, timedelta

from src.adapters.strategies.pruning_ab_controller import ABTestController
from src.adapters.strategies.pruning_metrics_collector import PruningMetricsCollector
from src.domain.models import PruningMetrics

logger = logging.getLogger(__name__)


class PruningMonitoringDashboard:
    """Real-time monitoring dashboard for pruning system performance."""

    def __init__(
        self,
        metrics_collector: PruningMetricsCollector,
        ab_controller: ABTestController | None = None,
        history_window_minutes: int = 60,
    ):
        """Initialize monitoring dashboard.

        Args:
            metrics_collector: Metrics collector instance
            ab_controller: Optional A/B test controller
            history_window_minutes: Minutes of history to maintain
        """
        self.metrics_collector = metrics_collector
        self.ab_controller = ab_controller
        self.history_window = timedelta(minutes=history_window_minutes)

        # Time series data storage
        self.time_series = {
            "pruning_rate": deque(maxlen=1000),
            "false_negative_rate": deque(maxlen=1000),
            "time_saved_ms": deque(maxlen=1000),
            "processing_rate": deque(maxlen=1000),
            "queue_length": deque(maxlen=1000),
        }

        # Alerts configuration
        self.alert_thresholds = {
            "false_negative_rate_max": 0.05,  # 5% max false negative rate
            "min_time_savings_percent": 30,   # Minimum 30% time savings
            "max_pruning_time_ms": 5,         # Max 5ms per pruning decision
            "min_pruning_rate": 0.2,          # Minimum 20% pruning rate
        }

        # Active alerts
        self.active_alerts = []

    def record_metrics(
        self,
        timestamp: datetime,
        strategy_id: str,
        metrics: PruningMetrics,
        queue_length: int = 0,
    ):
        """Record pruning metrics for monitoring.

        Args:
            timestamp: Timestamp of the metrics
            strategy_id: Strategy that generated the metrics
            metrics: Pruning metrics to record
            queue_length: Current evaluation queue length
        """
        # Update time series
        self.time_series["pruning_rate"].append({
            "timestamp": timestamp,
            "value": metrics.pruning_rate,
            "strategy": strategy_id,
        })

        self.time_series["false_negative_rate"].append({
            "timestamp": timestamp,
            "value": metrics.false_negative_rate,
            "strategy": strategy_id,
        })

        self.time_series["time_saved_ms"].append({
            "timestamp": timestamp,
            "value": metrics.time_saved_ms,
            "strategy": strategy_id,
        })

        if metrics.total_programs > 0:
            processing_rate = metrics.total_programs / (
                metrics.avg_pruning_time_ms * metrics.total_programs / 1000
            )
            self.time_series["processing_rate"].append({
                "timestamp": timestamp,
                "value": processing_rate,
                "strategy": strategy_id,
            })

        self.time_series["queue_length"].append({
            "timestamp": timestamp,
            "value": queue_length,
            "strategy": strategy_id,
        })

        # Check for alerts
        self._check_alerts(metrics, strategy_id)

    def _check_alerts(self, metrics: PruningMetrics, strategy_id: str):
        """Check metrics against alert thresholds.

        Args:
            metrics: Metrics to check
            strategy_id: Strategy being monitored
        """
        alerts = []

        # False negative rate alert
        if metrics.false_negative_rate > self.alert_thresholds["false_negative_rate_max"]:
            alerts.append({
                "type": "HIGH_FALSE_NEGATIVE_RATE",
                "severity": "CRITICAL",
                "message": f"False negative rate {metrics.false_negative_rate:.2%} exceeds threshold",
                "strategy": strategy_id,
                "value": metrics.false_negative_rate,
                "threshold": self.alert_thresholds["false_negative_rate_max"],
            })

        # Time savings alert
        time_savings_percent = (
            metrics.time_saved_ms / (metrics.time_saved_ms + metrics.avg_pruning_time_ms * metrics.total_programs) * 100
            if metrics.total_programs > 0 else 0
        )
        if time_savings_percent < self.alert_thresholds["min_time_savings_percent"]:
            alerts.append({
                "type": "LOW_TIME_SAVINGS",
                "severity": "WARNING",
                "message": f"Time savings {time_savings_percent:.1f}% below threshold",
                "strategy": strategy_id,
                "value": time_savings_percent,
                "threshold": self.alert_thresholds["min_time_savings_percent"],
            })

        # Pruning time alert
        if metrics.avg_pruning_time_ms > self.alert_thresholds["max_pruning_time_ms"]:
            alerts.append({
                "type": "SLOW_PRUNING",
                "severity": "WARNING",
                "message": f"Avg pruning time {metrics.avg_pruning_time_ms:.1f}ms exceeds threshold",
                "strategy": strategy_id,
                "value": metrics.avg_pruning_time_ms,
                "threshold": self.alert_thresholds["max_pruning_time_ms"],
            })

        # Low pruning rate alert
        if metrics.pruning_rate < self.alert_thresholds["min_pruning_rate"]:
            alerts.append({
                "type": "LOW_PRUNING_RATE",
                "severity": "INFO",
                "message": f"Pruning rate {metrics.pruning_rate:.2%} below threshold",
                "strategy": strategy_id,
                "value": metrics.pruning_rate,
                "threshold": self.alert_thresholds["min_pruning_rate"],
            })

        # Add new alerts
        for alert in alerts:
            alert["timestamp"] = datetime.now()
            self.active_alerts.append(alert)
            logger.warning(f"Pruning alert: {alert['message']}")

    def get_current_metrics(self) -> dict[str, dict[str, float]]:
        """Get current metrics across all strategies.

        Returns:
            Dictionary of current metrics by strategy
        """
        comparison = self.metrics_collector.get_strategy_comparison()
        efficiency = self.metrics_collector.get_efficiency_metrics()

        # Add real-time metrics
        current_metrics = {}

        for strategy_id, metrics in comparison.items():
            current_metrics[strategy_id] = {
                **metrics,
                "alerts": [
                    alert for alert in self.active_alerts
                    if alert["strategy"] == strategy_id and
                    (datetime.now() - alert["timestamp"]).seconds < 300  # Last 5 minutes
                ],
            }

        # Add system-wide metrics
        current_metrics["_system"] = efficiency

        return current_metrics

    def get_performance_trends(
        self,
        metric: str,
        minutes: int = 30,
    ) -> list[dict]:
        """Get performance trend data for a specific metric.

        Args:
            metric: Metric name to retrieve
            minutes: Number of minutes of history to return

        Returns:
            List of time series data points
        """
        if metric not in self.time_series:
            return []

        cutoff = datetime.now() - timedelta(minutes=minutes)

        # Filter by time window
        return [
            point for point in self.time_series[metric]
            if point["timestamp"] > cutoff
        ]

    def get_strategy_performance_summary(self) -> dict[str, dict]:
        """Get performance summary for all strategies.

        Returns:
            Dictionary mapping strategy_id to performance summary
        """
        summary = {}

        # Get metrics from collector
        comparison = self.metrics_collector.get_strategy_comparison()

        # Add A/B test results if available
        if self.ab_controller:
            ab_summary = self.ab_controller.get_performance_summary()
            allocations = self.ab_controller.get_current_allocations()

            for strategy_id, metrics in comparison.items():
                summary[strategy_id] = {
                    **metrics,
                    "allocation": allocations.get(strategy_id, 0),
                    "ab_metrics": ab_summary.get(strategy_id, {}),
                }
        else:
            summary = comparison

        return summary

    def get_alert_summary(self) -> dict[str, list[dict]]:
        """Get summary of active alerts by type.

        Returns:
            Dictionary grouping alerts by type
        """
        alert_summary = {}
        cutoff = datetime.now() - timedelta(minutes=60)

        for alert in self.active_alerts:
            if alert["timestamp"] > cutoff:
                alert_type = alert["type"]
                if alert_type not in alert_summary:
                    alert_summary[alert_type] = []
                alert_summary[alert_type].append(alert)

        return alert_summary

    def export_dashboard_snapshot(self) -> dict:
        """Export complete dashboard snapshot for reporting.

        Returns:
            Dictionary containing all dashboard data
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": self.get_current_metrics(),
            "performance_summary": self.get_strategy_performance_summary(),
            "alerts": self.get_alert_summary(),
            "trends": {
                "pruning_rate": self.get_performance_trends("pruning_rate", 60),
                "false_negative_rate": self.get_performance_trends("false_negative_rate", 60),
                "time_saved": self.get_performance_trends("time_saved_ms", 60),
            },
            "thresholds": self.alert_thresholds,
        }

    def generate_performance_report(self) -> str:
        """Generate human-readable performance report.

        Returns:
            Formatted performance report
        """
        snapshot = self.export_dashboard_snapshot()

        report = []
        report.append("=== Pruning System Performance Report ===")
        report.append(f"Generated: {snapshot['timestamp']}")
        report.append("")

        # Strategy performance
        report.append("Strategy Performance:")
        for strategy_id, metrics in snapshot["performance_summary"].items():
            if strategy_id == "_system":
                continue

            report.append(f"\n  {strategy_id}:")
            report.append(f"    - Pruning Rate: {metrics.get('pruning_rate', 0):.1%}")
            report.append(f"    - False Negative Rate: {metrics.get('false_negative_rate', 0):.1%}")
            report.append(f"    - Avg Pruning Time: {metrics.get('avg_pruning_time_ms', 0):.1f}ms")
            report.append(f"    - F1 Score: {metrics.get('f1_score', 0):.3f}")

            if "allocation" in metrics:
                report.append(f"    - A/B Test Allocation: {metrics['allocation']:.1%}")

        # System metrics
        system_metrics = snapshot["current_metrics"].get("_system", {})
        if system_metrics:
            report.append("\nSystem-wide Metrics:")
            report.append(f"  - Total Programs Processed: {system_metrics.get('total_programs_processed', 0)}")
            report.append(f"  - Total Time Saved: {system_metrics.get('total_time_saved_ms', 0):.0f}ms")
            report.append(f"  - Efficiency Gain: {system_metrics.get('efficiency_gain_percent', 0):.1f}%")

        # Active alerts
        if snapshot["alerts"]:
            report.append("\nActive Alerts:")
            for alert_type, alerts in snapshot["alerts"].items():
                report.append(f"  - {alert_type}: {len(alerts)} alerts")

        return "\n".join(report)

    def get_grafana_metrics(self) -> list[dict]:
        """Export metrics in Grafana-compatible format.

        Returns:
            List of metric dictionaries for Grafana
        """
        metrics = []
        timestamp = int(datetime.now().timestamp() * 1000)

        # Export current metrics
        for strategy_id, strategy_metrics in self.get_current_metrics().items():
            if strategy_id == "_system":
                continue

            base_labels = {"strategy": strategy_id}

            # Pruning rate
            metrics.append({
                "metric": "pruning_rate",
                "value": strategy_metrics.get("pruning_rate", 0),
                "timestamp": timestamp,
                "labels": base_labels,
            })

            # False negative rate
            metrics.append({
                "metric": "false_negative_rate",
                "value": strategy_metrics.get("false_negative_rate", 0),
                "timestamp": timestamp,
                "labels": base_labels,
            })

            # Average pruning time
            metrics.append({
                "metric": "avg_pruning_time_ms",
                "value": strategy_metrics.get("avg_pruning_time_ms", 0),
                "timestamp": timestamp,
                "labels": base_labels,
            })

            # F1 score
            metrics.append({
                "metric": "pruning_f1_score",
                "value": strategy_metrics.get("f1_score", 0),
                "timestamp": timestamp,
                "labels": base_labels,
            })

        return metrics
