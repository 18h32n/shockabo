"""Queue monitoring and status reporting for experiment queue management.

This module provides real-time monitoring, status reporting, and analytics
for the experiment queue system.
"""

import asyncio
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .experiment_queue import (
    ExperimentJob,
    ExperimentPriority,
    ExperimentStatus,
    get_experiment_queue,
)
from .platform_rotation_orchestrator import get_platform_orchestrator

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QueueAlert:
    """Queue monitoring alert."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    details: dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    timestamp: datetime
    total_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    suspended_jobs: int

    # Performance metrics
    average_queue_time_minutes: float
    average_execution_time_minutes: float
    success_rate: float
    throughput_jobs_per_hour: float

    # Platform distribution
    platform_distribution: dict[str, int]
    priority_distribution: dict[str, int]

    # Resource utilization
    estimated_gpu_utilization: float
    estimated_cost_savings: float


class QueueMonitor:
    """Monitors experiment queue and provides status reporting."""

    def __init__(self, monitoring_interval: int = 30, metrics_retention_days: int = 30):
        """Initialize queue monitor.

        Args:
            monitoring_interval: Monitoring interval in seconds
            metrics_retention_days: How long to retain metrics history
        """
        self.monitoring_interval = monitoring_interval
        self.metrics_retention_days = metrics_retention_days

        # Components
        self.experiment_queue = get_experiment_queue()
        self.orchestrator = get_platform_orchestrator()

        # State
        self.monitoring = False
        self.monitor_task: asyncio.Task | None = None
        self.metrics_history: list[QueueMetrics] = []
        self.alerts: list[QueueAlert] = []

        # Configuration
        self.alert_thresholds = {
            'max_queue_time_hours': 24,
            'max_failed_rate': 0.3,
            'max_queue_size': 100,
            'min_throughput_jobs_per_hour': 0.1
        }

        # Callbacks
        self.callbacks: dict[str, list[Callable]] = {
            'metrics_updated': [],
            'alert_triggered': [],
            'status_changed': []
        }

        self.logger = structlog.get_logger('queue_monitor')

        # Storage
        self.metrics_dir = Path.home() / ".arc-queue-metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Load historical data
        self._load_metrics_history()

    async def start_monitoring(self) -> bool:
        """Start queue monitoring.

        Returns:
            True if started successfully
        """
        if self.monitoring:
            self.logger.warning("monitoring_already_running")
            return False

        try:
            self.monitoring = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("queue_monitoring_started", interval=self.monitoring_interval)
            return True

        except Exception as e:
            self.logger.error("monitoring_start_failed", error=str(e))
            self.monitoring = False
            return False

    async def stop_monitoring(self) -> bool:
        """Stop queue monitoring.

        Returns:
            True if stopped successfully
        """
        if not self.monitoring:
            return True

        try:
            self.monitoring = False

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # Save final metrics
            await self._collect_metrics()
            self._save_metrics_history()

            self.logger.info("queue_monitoring_stopped")
            return True

        except Exception as e:
            self.logger.error("monitoring_stop_failed", error=str(e))
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                await self._collect_metrics()

                # Check for alerts
                self._check_alerts()

                # Clean up old data
                self._cleanup_old_data()

                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 collections
                    self._save_metrics_history()

                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self):
        """Collect current queue metrics."""
        try:
            now = datetime.now()

            # Get queue statistics
            queue_stats = self.experiment_queue.get_queue_stats()
            jobs = self.experiment_queue.list_jobs()

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(jobs)

            # Get platform distribution
            platform_dist = {}
            for job in jobs:
                platform = job.platform or 'unknown'
                platform_dist[platform] = platform_dist.get(platform, 0) + 1

            # Get priority distribution
            priority_dist = {}
            for priority in ExperimentPriority:
                count = len([j for j in jobs if j.priority == priority])
                priority_dist[priority.name] = count

            # Calculate resource utilization
            utilization_metrics = await self._calculate_utilization_metrics()

            metrics = QueueMetrics(
                timestamp=now,
                total_jobs=queue_stats['total_jobs'],
                queued_jobs=queue_stats['queued_jobs'],
                running_jobs=queue_stats['running_jobs'],
                completed_jobs=queue_stats['completed_jobs'],
                failed_jobs=queue_stats['failed_jobs'],
                cancelled_jobs=queue_stats['cancelled_jobs'],
                suspended_jobs=len([j for j in jobs if j.status == ExperimentStatus.SUSPENDED]),

                average_queue_time_minutes=performance_metrics['avg_queue_time'],
                average_execution_time_minutes=performance_metrics['avg_execution_time'],
                success_rate=performance_metrics['success_rate'],
                throughput_jobs_per_hour=performance_metrics['throughput'],

                platform_distribution=platform_dist,
                priority_distribution=priority_dist,

                estimated_gpu_utilization=utilization_metrics['gpu_utilization'],
                estimated_cost_savings=utilization_metrics['cost_savings']
            )

            # Add to history
            self.metrics_history.append(metrics)

            # Trigger callbacks
            self._trigger_callbacks('metrics_updated', metrics)

            self.logger.debug("metrics_collected",
                            total_jobs=metrics.total_jobs,
                            throughput=metrics.throughput_jobs_per_hour,
                            utilization=metrics.estimated_gpu_utilization)

        except Exception as e:
            self.logger.error("metrics_collection_failed", error=str(e))

    def _calculate_performance_metrics(self, jobs: list[ExperimentJob]) -> dict[str, float]:
        """Calculate performance metrics from jobs.

        Args:
            jobs: List of jobs to analyze

        Returns:
            Dictionary with performance metrics
        """
        now = datetime.now()

        # Queue times
        queue_times = []
        for job in jobs:
            if job.started_at and job.created_at:
                queue_time = (job.started_at - job.created_at).total_seconds() / 60
                queue_times.append(queue_time)

        # Execution times
        execution_times = []
        for job in jobs:
            if job.started_at and job.completed_at:
                exec_time = (job.completed_at - job.started_at).total_seconds() / 60
                execution_times.append(exec_time)

        # Success rate
        completed_or_failed = [j for j in jobs if j.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]]
        success_rate = 0.0
        if completed_or_failed:
            completed_count = len([j for j in completed_or_failed if j.status == ExperimentStatus.COMPLETED])
            success_rate = completed_count / len(completed_or_failed)

        # Throughput (jobs per hour in last 24 hours)
        last_24h = now - timedelta(hours=24)
        recent_completed = [
            j for j in jobs
            if j.completed_at and j.completed_at >= last_24h and j.status == ExperimentStatus.COMPLETED
        ]
        throughput = len(recent_completed) / 24  # jobs per hour

        return {
            'avg_queue_time': statistics.mean(queue_times) if queue_times else 0.0,
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0.0,
            'success_rate': success_rate,
            'throughput': throughput
        }

    async def _calculate_utilization_metrics(self) -> dict[str, float]:
        """Calculate resource utilization metrics.

        Returns:
            Dictionary with utilization metrics
        """
        try:
            # Get orchestrator stats
            orchestrator_stats = self.orchestrator.get_orchestration_stats()

            # Estimate GPU utilization based on running jobs and platform capacity
            gpu_utilization = 0.0
            cost_savings = 0.0

            if orchestrator_stats['current_session']:
                session = orchestrator_stats['current_session']

                # Estimate utilization based on current job
                if session['current_job']:
                    gpu_utilization = 0.85  # Assume 85% utilization when running
                else:
                    gpu_utilization = 0.1   # Minimal utilization when idle

                # Estimate cost savings (rough calculation)
                runtime_hours = session['total_runtime_minutes'] / 60
                estimated_cloud_cost = runtime_hours * 2.50  # $2.50/hour typical cloud GPU cost
                cost_savings = estimated_cloud_cost  # Using free tier

            return {
                'gpu_utilization': gpu_utilization,
                'cost_savings': cost_savings
            }

        except Exception as e:
            self.logger.error("utilization_calculation_failed", error=str(e))
            return {'gpu_utilization': 0.0, 'cost_savings': 0.0}

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.metrics_history:
            return

        current_metrics = self.metrics_history[-1]

        # Check queue size
        if current_metrics.queued_jobs > self.alert_thresholds['max_queue_size']:
            self._create_alert(
                AlertSeverity.WARNING,
                f"Queue size exceeded threshold: {current_metrics.queued_jobs} jobs",
                {'queue_size': current_metrics.queued_jobs, 'threshold': self.alert_thresholds['max_queue_size']}
            )

        # Check failure rate
        if current_metrics.success_rate < (1 - self.alert_thresholds['max_failed_rate']):
            self._create_alert(
                AlertSeverity.ERROR,
                f"High failure rate: {(1-current_metrics.success_rate)*100:.1f}%",
                {'failure_rate': 1-current_metrics.success_rate, 'threshold': self.alert_thresholds['max_failed_rate']}
            )

        # Check throughput
        if current_metrics.throughput_jobs_per_hour < self.alert_thresholds['min_throughput_jobs_per_hour']:
            self._create_alert(
                AlertSeverity.WARNING,
                f"Low throughput: {current_metrics.throughput_jobs_per_hour:.2f} jobs/hour",
                {'throughput': current_metrics.throughput_jobs_per_hour, 'threshold': self.alert_thresholds['min_throughput_jobs_per_hour']}
            )

        # Check queue time
        if current_metrics.average_queue_time_minutes > (self.alert_thresholds['max_queue_time_hours'] * 60):
            self._create_alert(
                AlertSeverity.CRITICAL,
                f"Excessive queue time: {current_metrics.average_queue_time_minutes/60:.1f} hours",
                {'queue_time_hours': current_metrics.average_queue_time_minutes/60, 'threshold': self.alert_thresholds['max_queue_time_hours']}
            )

    def _create_alert(self, severity: AlertSeverity, message: str, details: dict[str, Any]):
        """Create a new alert.

        Args:
            severity: Alert severity
            message: Alert message
            details: Alert details
        """
        import uuid

        alert = QueueAlert(
            id=str(uuid.uuid4()),
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            details=details
        )

        self.alerts.append(alert)

        # Log alert
        self.logger.log(
            severity.value.upper(),
            "queue_alert_triggered",
            alert_id=alert.id,
            message=message,
            details=details
        )

        # Trigger callbacks
        self._trigger_callbacks('alert_triggered', alert)

        # Keep only recent alerts
        self.alerts = self.alerts[-100:]  # Keep last 100 alerts

    def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        # Clean old metrics
        cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        # Clean old alerts (keep for 7 days)
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp >= alert_cutoff]

    def get_current_status(self) -> dict[str, Any]:
        """Get current queue status.

        Returns:
            Dictionary with current status
        """
        if not self.metrics_history:
            return {'status': 'no_data'}

        current_metrics = self.metrics_history[-1]
        recent_alerts = [a for a in self.alerts if not a.acknowledged]

        status = {
            'timestamp': current_metrics.timestamp.isoformat(),
            'monitoring_active': self.monitoring,
            'metrics': asdict(current_metrics),
            'alerts': {
                'total': len(recent_alerts),
                'critical': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                'error': len([a for a in recent_alerts if a.severity == AlertSeverity.ERROR]),
                'warning': len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
                'recent': [asdict(a) for a in recent_alerts[-5:]]  # Last 5 alerts
            },
            'health_score': self._calculate_health_score(current_metrics)
        }

        return status

    def _calculate_health_score(self, metrics: QueueMetrics) -> float:
        """Calculate queue health score (0-100).

        Args:
            metrics: Current metrics

        Returns:
            Health score between 0 and 100
        """
        score = 100.0

        # Penalize based on failure rate
        failure_rate = 1 - metrics.success_rate
        score -= failure_rate * 30  # Up to 30 points penalty

        # Penalize based on queue size
        if metrics.queued_jobs > 50:
            score -= min((metrics.queued_jobs - 50) / 10, 20)  # Up to 20 points penalty

        # Penalize based on low throughput
        if metrics.throughput_jobs_per_hour < 1:
            score -= (1 - metrics.throughput_jobs_per_hour) * 15  # Up to 15 points penalty

        # Penalize based on long queue times
        if metrics.average_queue_time_minutes > 60:  # More than 1 hour
            queue_hours = metrics.average_queue_time_minutes / 60
            score -= min(queue_hours * 5, 25)  # Up to 25 points penalty

        # Bonus for high GPU utilization
        if metrics.estimated_gpu_utilization > 0.8:
            score += 10

        return max(0, min(100, score))

    def get_metrics_history(self, hours: int = 24) -> list[QueueMetrics]:
        """Get metrics history for specified time period.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of metrics within the time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_alerts(self, severity: AlertSeverity | None = None,
                   acknowledged: bool | None = None) -> list[QueueAlert]:
        """Get alerts with optional filtering.

        Args:
            severity: Filter by severity
            acknowledged: Filter by acknowledged status

        Returns:
            List of matching alerts
        """
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if acknowledged successfully
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info("alert_acknowledged", alert_id=alert_id)
                return True

        self.logger.warning("alert_not_found", alert_id=alert_id)
        return False

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for monitoring events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for an event.

        Args:
            event_type: Event type
            data: Event data
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error("callback_failed", event_type=event_type, error=str(e))

    def _save_metrics_history(self):
        """Save metrics history to disk."""
        try:
            metrics_file = self.metrics_dir / "metrics_history.json"

            # Convert metrics to serializable format
            serializable_metrics = []
            for metric in self.metrics_history:
                metric_dict = asdict(metric)
                metric_dict['timestamp'] = metric.timestamp.isoformat()
                serializable_metrics.append(metric_dict)

            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)

            self.logger.debug("metrics_history_saved", count=len(self.metrics_history))

        except Exception as e:
            self.logger.error("metrics_save_failed", error=str(e))

    def _load_metrics_history(self):
        """Load metrics history from disk."""
        try:
            metrics_file = self.metrics_dir / "metrics_history.json"

            if not metrics_file.exists():
                return

            with open(metrics_file) as f:
                serializable_metrics = json.load(f)

            # Convert back to QueueMetrics objects
            for metric_dict in serializable_metrics:
                metric_dict['timestamp'] = datetime.fromisoformat(metric_dict['timestamp'])

                try:
                    metric = QueueMetrics(**metric_dict)
                    self.metrics_history.append(metric)
                except Exception as e:
                    self.logger.error("metric_deserialization_failed", error=str(e))

            # Clean up old data
            self._cleanup_old_data()

            self.logger.info("metrics_history_loaded", count=len(self.metrics_history))

        except Exception as e:
            self.logger.error("metrics_load_failed", error=str(e))


# Singleton instance
_queue_monitor = None


def get_queue_monitor() -> QueueMonitor:
    """Get singleton queue monitor instance."""
    global _queue_monitor
    if _queue_monitor is None:
        _queue_monitor = QueueMonitor()
    return _queue_monitor
