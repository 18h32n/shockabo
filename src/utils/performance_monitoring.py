"""
Performance Monitoring and Profiling Tools

Comprehensive monitoring system for tracking training and inference performance
to ensure Story 1.5 requirements are met continuously.
"""
import json
import logging
import os
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil
import torch
from torch.profiler import record_function

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    INFERENCE_TIME = "inference_time"
    TRAINING_STEP_TIME = "training_step_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    LOSS = "loss"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    metric_type: PerformanceMetricType
    value: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert."""
    severity: AlertSeverity
    message: str
    metric_type: PerformanceMetricType
    threshold: float
    actual_value: float
    timestamp: float
    resolved: bool = False


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    start_time: float
    end_time: float
    metrics_summary: dict[str, dict[str, float]]  # metric_type -> {avg, min, max, std}
    alerts: list[PerformanceAlert]
    requirements_status: dict[str, bool]
    recommendations: list[str]
    detailed_metrics: list[PerformanceMetric]


class PerformanceThresholds:
    """Performance thresholds based on Story 1.5 requirements."""

    def __init__(self):
        """Initialize performance thresholds."""
        self.thresholds = {
            # Story 1.5 Requirements
            PerformanceMetricType.INFERENCE_TIME: {
                "max": 432.0,  # 7.2 minutes in seconds
                "warning": 300.0,  # 5 minutes warning
                "target": 240.0,  # 4 minutes target
            },
            PerformanceMetricType.MEMORY_USAGE: {
                "max": 0.95,  # 95% of available memory
                "warning": 0.85,  # 85% warning threshold
                "target": 0.75,  # 75% target
            },
            PerformanceMetricType.ACCURACY: {
                "min": 0.53,  # 53% minimum accuracy
                "warning": 0.55,  # 55% for comfort margin
                "target": 0.60,  # 60% target
            },
            PerformanceMetricType.TRAINING_STEP_TIME: {
                "max": 60.0,  # 1 minute max per step
                "warning": 30.0,  # 30 seconds warning
                "target": 15.0,  # 15 seconds target
            },
            PerformanceMetricType.THROUGHPUT: {
                "min": 8.0,  # Minimum 8 tasks per hour
                "warning": 10.0,  # Warning below 10 tasks/hour
                "target": 15.0,  # Target 15 tasks/hour
            },
        }

    def get_threshold(self, metric_type: PerformanceMetricType, threshold_type: str) -> float | None:
        """Get threshold value for a metric."""
        return self.thresholds.get(metric_type, {}).get(threshold_type)

    def check_violation(self, metric: PerformanceMetric) -> PerformanceAlert | None:
        """Check if metric violates thresholds."""
        thresholds = self.thresholds.get(metric.metric_type)
        if not thresholds:
            return None

        value = metric.value

        # Check maximum thresholds
        if "max" in thresholds and value > thresholds["max"]:
            return PerformanceAlert(
                severity=AlertSeverity.CRITICAL,
                message=f"{metric.metric_type.value} exceeds maximum threshold: {value:.3f} > {thresholds['max']:.3f}",
                metric_type=metric.metric_type,
                threshold=thresholds["max"],
                actual_value=value,
                timestamp=metric.timestamp
            )

        if "warning" in thresholds:
            # For metrics with max thresholds
            if "max" in thresholds and value > thresholds["warning"]:
                return PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"{metric.metric_type.value} exceeds warning threshold: {value:.3f} > {thresholds['warning']:.3f}",
                    metric_type=metric.metric_type,
                    threshold=thresholds["warning"],
                    actual_value=value,
                    timestamp=metric.timestamp
                )
            # For metrics with min thresholds
            elif "min" in thresholds and value < thresholds["warning"]:
                return PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"{metric.metric_type.value} below warning threshold: {value:.3f} < {thresholds['warning']:.3f}",
                    metric_type=metric.metric_type,
                    threshold=thresholds["warning"],
                    actual_value=value,
                    timestamp=metric.timestamp
                )

        # Check minimum thresholds
        if "min" in thresholds and value < thresholds["min"]:
            return PerformanceAlert(
                severity=AlertSeverity.CRITICAL,
                message=f"{metric.metric_type.value} below minimum threshold: {value:.3f} < {thresholds['min']:.3f}",
                metric_type=metric.metric_type,
                threshold=thresholds["min"],
                actual_value=value,
                timestamp=metric.timestamp
            )

        return None


class PerformanceCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_metrics: int = 10000):
        """
        Initialize performance collector.

        Args:
            max_metrics: Maximum number of metrics to store in memory
        """
        self.max_metrics = max_metrics
        self.metrics: list[PerformanceMetric] = []
        self.alerts: list[PerformanceAlert] = []
        self.thresholds = PerformanceThresholds()

        self._lock = threading.Lock()

    def record_metric(
        self,
        metric_type: PerformanceMetricType,
        value: float,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            tags=tags or {}
        )

        with self._lock:
            self.metrics.append(metric)

            # Check for threshold violations
            alert = self.thresholds.check_violation(metric)
            if alert:
                self.alerts.append(alert)
                self._log_alert(alert)

            # Maintain max metrics limit
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]

            # Maintain alerts limit
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

    def _log_alert(self, alert: PerformanceAlert) -> None:
        """Log performance alert."""
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"PERFORMANCE ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"PERFORMANCE ALERT: {alert.message}")
        else:
            logger.info(f"PERFORMANCE ALERT: {alert.message}")

    def get_metrics(
        self,
        metric_type: PerformanceMetricType | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int | None = None
    ) -> list[PerformanceMetric]:
        """Get metrics with optional filtering."""
        with self._lock:
            filtered_metrics = self.metrics.copy()

        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]

        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]

        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]

        if limit:
            filtered_metrics = filtered_metrics[-limit:]

        return filtered_metrics

    def get_metric_statistics(
        self,
        metric_type: PerformanceMetricType,
        start_time: float | None = None,
        end_time: float | None = None
    ) -> dict[str, float]:
        """Get statistical summary of metrics."""
        metrics = self.get_metrics(metric_type, start_time, end_time)

        if not metrics:
            return {"count": 0}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
        }

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]

    def resolve_alert(self, alert: PerformanceAlert) -> None:
        """Mark an alert as resolved."""
        alert.resolved = True
        logger.info(f"Resolved alert: {alert.message}")


class InferenceProfiler:
    """Specialized profiler for inference performance."""

    def __init__(self, collector: PerformanceCollector):
        """Initialize inference profiler."""
        self.collector = collector
        self.current_inference_start: float | None = None

    @contextmanager
    def profile_inference(
        self,
        task_id: str | None = None,
        model_name: str | None = None,
        expected_tokens: int | None = None
    ):
        """Context manager for profiling inference."""
        start_time = time.time()
        start_memory = self._get_gpu_memory_usage()

        # Record start of inference
        self.current_inference_start = start_time

        try:
            with record_function("inference"):
                yield
        finally:
            end_time = time.time()
            end_memory = self._get_gpu_memory_usage()

            # Calculate metrics
            inference_time = end_time - start_time
            memory_delta = end_memory - start_memory

            # Record metrics
            self.collector.record_metric(
                PerformanceMetricType.INFERENCE_TIME,
                inference_time,
                metadata={
                    "task_id": task_id,
                    "model_name": model_name,
                    "expected_tokens": expected_tokens,
                    "memory_delta_mb": memory_delta,
                },
                tags={"operation": "inference", "model": model_name}
            )

            # Record memory usage
            self.collector.record_metric(
                PerformanceMetricType.MEMORY_USAGE,
                end_memory / self._get_total_gpu_memory() if torch.cuda.is_available() else 0.0,
                metadata={"phase": "inference_end"},
                tags={"operation": "inference"}
            )

            # Calculate throughput if tokens provided
            if expected_tokens and inference_time > 0:
                throughput = expected_tokens / inference_time
                self.collector.record_metric(
                    PerformanceMetricType.THROUGHPUT,
                    throughput,
                    metadata={
                        "unit": "tokens_per_second",
                        "tokens": expected_tokens,
                        "time_seconds": inference_time,
                    },
                    tags={"operation": "inference", "metric": "tokens_per_second"}
                )

            self.current_inference_start = None

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 1.0  # Avoid division by zero


class TrainingProfiler:
    """Specialized profiler for training performance."""

    def __init__(self, collector: PerformanceCollector):
        """Initialize training profiler."""
        self.collector = collector
        self.epoch_start_time: float | None = None
        self.step_count = 0

    @contextmanager
    def profile_training_step(
        self,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        epoch: int | None = None
    ):
        """Context manager for profiling training steps."""
        start_time = time.time()
        start_memory = self._get_gpu_memory_usage()

        try:
            with record_function("training_step"):
                yield
        finally:
            end_time = time.time()
            end_memory = self._get_gpu_memory_usage()

            # Calculate metrics
            step_time = end_time - start_time
            memory_delta = end_memory - start_memory

            # Record step time
            self.collector.record_metric(
                PerformanceMetricType.TRAINING_STEP_TIME,
                step_time,
                metadata={
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "step": self.step_count,
                    "memory_delta_mb": memory_delta,
                },
                tags={"operation": "training", "phase": "step"}
            )

            # Record memory usage
            memory_utilization = end_memory / self._get_total_gpu_memory() if torch.cuda.is_available() else 0.0
            self.collector.record_metric(
                PerformanceMetricType.MEMORY_USAGE,
                memory_utilization,
                metadata={"phase": "training_step", "step": self.step_count},
                tags={"operation": "training"}
            )

            self.step_count += 1

    def record_loss(self, loss: float, epoch: int | None = None) -> None:
        """Record training loss."""
        self.collector.record_metric(
            PerformanceMetricType.LOSS,
            loss,
            metadata={"epoch": epoch, "step": self.step_count},
            tags={"operation": "training", "metric": "loss"}
        )

    def record_accuracy(self, accuracy: float, dataset: str = "validation") -> None:
        """Record model accuracy."""
        self.collector.record_metric(
            PerformanceMetricType.ACCURACY,
            accuracy,
            metadata={"dataset": dataset, "step": self.step_count},
            tags={"operation": "evaluation", "metric": "accuracy"}
        )

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 1.0


class SystemMonitor:
    """Monitors system-level performance metrics."""

    def __init__(self, collector: PerformanceCollector, monitoring_interval: float = 10.0):
        """
        Initialize system monitor.

        Args:
            collector: Performance collector instance
            monitoring_interval: Monitoring interval in seconds
        """
        self.collector = collector
        self.monitoring_interval = monitoring_interval
        self.monitoring_thread: threading.Thread | None = None
        self.stop_monitoring = threading.Event()

    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self.monitoring_thread is not None:
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Started system performance monitoring")

    def stop_monitoring(self):
        """Stop system monitoring."""
        if self.monitoring_thread is None:
            return

        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None

        logger.info("Stopped system performance monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                # Record system metrics
                self._record_system_metrics()
                self._record_gpu_metrics()
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")

    def _record_system_metrics(self):
        """Record system-level metrics."""
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Record metrics (using GPU_UTILIZATION as a proxy for system metrics)
        self.collector.record_metric(
            PerformanceMetricType.GPU_UTILIZATION,
            cpu_percent / 100.0,
            metadata={
                "metric": "cpu_utilization",
                "memory_percent": memory_percent,
                "available_gb": memory.available / (1024**3),
            },
            tags={"source": "system", "metric": "cpu"}
        )

    def _record_gpu_metrics(self):
        """Record GPU metrics."""
        if not torch.cuda.is_available():
            return

        # GPU memory utilization
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        utilization = allocated / total

        self.collector.record_metric(
            PerformanceMetricType.MEMORY_USAGE,
            utilization,
            metadata={
                "source": "gpu",
                "allocated_mb": allocated / (1024 * 1024),
                "total_mb": total / (1024 * 1024),
            },
            tags={"source": "gpu", "metric": "memory"}
        )


class PerformanceReporter:
    """Generates comprehensive performance reports."""

    def __init__(self, collector: PerformanceCollector):
        """Initialize performance reporter."""
        self.collector = collector

    def generate_report(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        output_path: str | None = None
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        if end_time is None:
            end_time = time.time()

        if start_time is None:
            start_time = end_time - 3600  # Last hour by default

        # Collect metrics for time period
        all_metrics = self.collector.get_metrics(start_time=start_time, end_time=end_time)

        # Generate summary statistics
        metrics_summary = {}
        for metric_type in PerformanceMetricType:
            stats = self.collector.get_metric_statistics(metric_type, start_time, end_time)
            if stats["count"] > 0:
                metrics_summary[metric_type.value] = stats

        # Get alerts
        alerts = [alert for alert in self.collector.alerts
                 if start_time <= alert.timestamp <= end_time]

        # Check requirements status
        requirements_status = self._check_requirements_status(metrics_summary)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary, alerts)

        report = PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts=alerts,
            requirements_status=requirements_status,
            recommendations=recommendations,
            detailed_metrics=all_metrics
        )

        # Save report if path provided
        if output_path:
            self._save_report(report, output_path)

        return report

    def _check_requirements_status(self, metrics_summary: dict[str, dict[str, float]]) -> dict[str, bool]:
        """Check Story 1.5 requirements status."""
        status = {}

        # AC5: Single task inference under 7.2 minutes
        if PerformanceMetricType.INFERENCE_TIME.value in metrics_summary:
            inference_stats = metrics_summary[PerformanceMetricType.INFERENCE_TIME.value]
            max_inference_time = inference_stats["max"]
            status["inference_time_requirement"] = max_inference_time <= 432.0
        else:
            status["inference_time_requirement"] = False

        # AC4: Achieve 53%+ accuracy on validation set
        if PerformanceMetricType.ACCURACY.value in metrics_summary:
            accuracy_stats = metrics_summary[PerformanceMetricType.ACCURACY.value]
            max_accuracy = accuracy_stats["max"]
            status["accuracy_requirement"] = max_accuracy >= 0.53
        else:
            status["accuracy_requirement"] = False

        # Memory constraint: Stay within GPU memory limits
        if PerformanceMetricType.MEMORY_USAGE.value in metrics_summary:
            memory_stats = metrics_summary[PerformanceMetricType.MEMORY_USAGE.value]
            max_memory = memory_stats["max"]
            status["memory_requirement"] = max_memory <= 0.95
        else:
            status["memory_requirement"] = False

        # Pipeline throughput: Process at least 8 tasks per hour
        if PerformanceMetricType.THROUGHPUT.value in metrics_summary:
            throughput_stats = metrics_summary[PerformanceMetricType.THROUGHPUT.value]
            min_throughput = throughput_stats["min"]
            status["throughput_requirement"] = min_throughput >= 8.0
        else:
            status["throughput_requirement"] = False

        return status

    def _generate_recommendations(
        self,
        metrics_summary: dict[str, dict[str, float]],
        alerts: list[PerformanceAlert]
    ) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Critical alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(f"CRITICAL: Address {len(critical_alerts)} critical performance issues")

        # Inference time recommendations
        if PerformanceMetricType.INFERENCE_TIME.value in metrics_summary:
            stats = metrics_summary[PerformanceMetricType.INFERENCE_TIME.value]
            if stats["max"] > 432.0:
                recommendations.append("Optimize inference time - currently exceeds 7.2 minute requirement")
            elif stats["avg"] > 300.0:
                recommendations.append("Consider inference optimizations - average time approaching limit")

        # Memory usage recommendations
        if PerformanceMetricType.MEMORY_USAGE.value in metrics_summary:
            stats = metrics_summary[PerformanceMetricType.MEMORY_USAGE.value]
            if stats["max"] > 0.9:
                recommendations.append("Implement more aggressive memory optimization")
            elif stats["avg"] > 0.8:
                recommendations.append("Monitor memory usage closely - approaching limits")

        # Accuracy recommendations
        if PerformanceMetricType.ACCURACY.value in metrics_summary:
            stats = metrics_summary[PerformanceMetricType.ACCURACY.value]
            if stats["max"] < 0.53:
                recommendations.append("CRITICAL: Model accuracy below 53% requirement")
            elif stats["max"] < 0.55:
                recommendations.append("Consider model improvements - accuracy near minimum threshold")

        # Training performance
        if PerformanceMetricType.TRAINING_STEP_TIME.value in metrics_summary:
            stats = metrics_summary[PerformanceMetricType.TRAINING_STEP_TIME.value]
            if stats["avg"] > 30.0:
                recommendations.append("Optimize training step time for better efficiency")

        # General recommendations
        if not recommendations:
            recommendations.append("Performance metrics within acceptable ranges")

        recommendations.append("Continue monitoring performance metrics regularly")

        return recommendations

    def _save_report(self, report: PerformanceReport, output_path: str):
        """Save report to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert report to serializable format
            report_dict = {
                "start_time": report.start_time,
                "end_time": report.end_time,
                "duration_hours": (report.end_time - report.start_time) / 3600,
                "metrics_summary": report.metrics_summary,
                "alerts": [asdict(alert) for alert in report.alerts],
                "requirements_status": report.requirements_status,
                "recommendations": report.recommendations,
                "detailed_metrics_count": len(report.detailed_metrics),
                "generation_time": datetime.now().isoformat(),
            }

            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

            logger.info(f"Performance report saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")


def main():
    """Demonstrate performance monitoring capabilities."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing performance monitoring and profiling tools...")

    # Initialize components
    collector = PerformanceCollector()
    inference_profiler = InferenceProfiler(collector)
    training_profiler = TrainingProfiler(collector)
    system_monitor = SystemMonitor(collector, monitoring_interval=2.0)
    reporter = PerformanceReporter(collector)

    # Start system monitoring
    system_monitor.start_monitoring()

    # Simulate some performance measurements
    logger.info("Simulating inference profiling...")
    for i in range(5):
        with inference_profiler.profile_inference(
            task_id=f"test_task_{i}",
            model_name="test_model",
            expected_tokens=200
        ):
            # Simulate inference work
            time.sleep(0.5 + i * 0.1)  # Variable inference times

    logger.info("Simulating training profiling...")
    for epoch in range(3):
        for step in range(5):
            with training_profiler.profile_training_step(
                batch_size=4,
                learning_rate=1e-5,
                epoch=epoch
            ):
                # Simulate training work
                time.sleep(0.2)

            # Record loss
            training_profiler.record_loss(0.8 - epoch * 0.1 - step * 0.02, epoch)

        # Record accuracy
        training_profiler.record_accuracy(0.4 + epoch * 0.05, "validation")

    # Let system monitor collect some data
    time.sleep(5)

    # Stop monitoring
    system_monitor.stop_monitoring()

    # Generate performance report
    logger.info("Generating performance report...")
    report = reporter.generate_report(
        output_path="docs/qa/assessments/performance_monitoring_demo_report.json"
    )

    # Display summary
    print("\n" + "="*70)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*70)
    print(f"Report Period: {datetime.fromtimestamp(report.start_time)} to {datetime.fromtimestamp(report.end_time)}")
    print(f"Total Metrics Collected: {len(report.detailed_metrics)}")
    print(f"Alerts Generated: {len(report.alerts)}")

    print("\nMetrics Summary:")
    for metric_type, stats in report.metrics_summary.items():
        print(f"  {metric_type}: {stats['count']} measurements, avg={stats['avg']:.3f}")

    print("\nRequirements Status:")
    for requirement, status in report.requirements_status.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {requirement}: {status_str}")

    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "="*70)
    print("PERFORMANCE MONITORING SYSTEM READY")
    print("="*70)
    print("Components implemented:")
    print("  ✓ Comprehensive metric collection")
    print("  ✓ Story 1.5 requirements monitoring")
    print("  ✓ Inference performance profiling")
    print("  ✓ Training performance profiling")
    print("  ✓ System resource monitoring")
    print("  ✓ Automated alerting system")
    print("  ✓ Performance report generation")
    print("="*70)


if __name__ == "__main__":
    main()
