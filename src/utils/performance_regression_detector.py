"""
Performance Regression Detection System

Implements comprehensive performance regression detection for DSL operations with:
- Baseline storage and versioned management
- Statistical analysis with percentile tracking (p50, p95, p99)
- Configurable thresholds (20% warning, 50% failure)
- Performance comparison reports across versions
- Integration with existing profiling systems
"""

import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    NONE = "none"
    WARNING = "warning"  # 20% slower than baseline
    CRITICAL = "critical"  # 50% slower than baseline


class PerformanceMetricType(Enum):
    """Types of performance metrics tracked."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    OPERATION_COUNT = "operation_count"


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    operation_name: str
    metric_type: PerformanceMetricType
    value: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "unknown"


@dataclass
class PerformanceBaseline:
    """Performance baseline statistics for an operation."""
    operation_name: str
    metric_type: PerformanceMetricType
    version: str
    sample_count: int
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    min_value: float
    max_value: float
    created_at: float
    last_updated: float
    raw_measurements: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "metric_type": self.metric_type.value,
            "version": self.version,
            "sample_count": self.sample_count,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "raw_measurements": self.raw_measurements
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceBaseline":
        """Create from dictionary."""
        return cls(
            operation_name=data["operation_name"],
            metric_type=PerformanceMetricType(data["metric_type"]),
            version=data["version"],
            sample_count=data["sample_count"],
            mean=data["mean"],
            median=data["median"],
            p95=data["p95"],
            p99=data["p99"],
            std_dev=data["std_dev"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            raw_measurements=data.get("raw_measurements", [])
        )


@dataclass
class RegressionResult:
    """Result of a regression detection analysis."""
    operation_name: str
    metric_type: PerformanceMetricType
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    percentage_change: float
    baseline_version: str
    current_version: str
    message: str
    timestamp: float
    recommendations: list[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance comparison report."""
    baseline_version: str
    current_version: str
    generation_time: float
    total_operations_analyzed: int
    regressions_found: int
    warnings_found: int
    critical_regressions: int
    operations_improved: int
    regression_results: list[RegressionResult]
    summary_statistics: dict[str, Any]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        def serialize_regression_result(result: RegressionResult) -> dict[str, Any]:
            """Convert RegressionResult to serializable dict."""
            return {
                "operation_name": result.operation_name,
                "metric_type": result.metric_type.value,  # Convert enum to string
                "severity": result.severity.value,  # Convert enum to string
                "current_value": result.current_value,
                "baseline_value": result.baseline_value,
                "percentage_change": result.percentage_change,
                "baseline_version": result.baseline_version,
                "current_version": result.current_version,
                "message": result.message,
                "timestamp": result.timestamp,
                "recommendations": result.recommendations
            }

        return {
            "baseline_version": self.baseline_version,
            "current_version": self.current_version,
            "generation_time": self.generation_time,
            "generation_time_iso": datetime.fromtimestamp(self.generation_time).isoformat(),
            "total_operations_analyzed": self.total_operations_analyzed,
            "regressions_found": self.regressions_found,
            "warnings_found": self.warnings_found,
            "critical_regressions": self.critical_regressions,
            "operations_improved": self.operations_improved,
            "regression_results": [serialize_regression_result(r) for r in self.regression_results],
            "summary_statistics": self.summary_statistics,
            "recommendations": self.recommendations
        }


class PerformanceBaselineStorage:
    """Manages storage and retrieval of performance baselines."""

    def __init__(self, storage_dir: str = "performance_baselines"):
        """
        Initialize baseline storage.

        Args:
            storage_dir: Directory to store baseline files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._baselines: dict[str, dict[str, PerformanceBaseline]] = defaultdict(dict)
        self._load_baselines()

    def _get_baseline_key(self, operation_name: str, metric_type: PerformanceMetricType,
                         version: str) -> str:
        """Generate unique key for baseline identification."""
        return f"{operation_name}_{metric_type.value}_{version}"

    def _get_baseline_file(self, version: str) -> Path:
        """Get file path for version-specific baselines."""
        return self.storage_dir / f"baselines_{version}.json"

    def _load_baselines(self) -> None:
        """Load all existing baselines from storage."""
        for baseline_file in self.storage_dir.glob("baselines_*.json"):
            try:
                with open(baseline_file) as f:
                    data = json.load(f)

                version = data.get("version", "unknown")
                for baseline_data in data.get("baselines", []):
                    baseline = PerformanceBaseline.from_dict(baseline_data)
                    key = self._get_baseline_key(
                        baseline.operation_name,
                        baseline.metric_type,
                        baseline.version
                    )
                    self._baselines[version][key] = baseline

                logger.info(f"Loaded {len(data.get('baselines', []))} baselines for version {version}")

            except Exception as e:
                logger.error(f"Failed to load baselines from {baseline_file}: {e}")

    def store_baseline(self, baseline: PerformanceBaseline) -> None:
        """
        Store a performance baseline.

        Args:
            baseline: Performance baseline to store
        """
        key = self._get_baseline_key(
            baseline.operation_name,
            baseline.metric_type,
            baseline.version
        )
        self._baselines[baseline.version][key] = baseline
        self._save_version_baselines(baseline.version)

        logger.debug(f"Stored baseline for {baseline.operation_name} "
                    f"({baseline.metric_type.value}, v{baseline.version})")

    def _save_version_baselines(self, version: str) -> None:
        """Save baselines for a specific version to disk."""
        try:
            baseline_file = self._get_baseline_file(version)
            baselines_data = {
                "version": version,
                "saved_at": time.time(),
                "baselines": [baseline.to_dict() for baseline in self._baselines[version].values()]
            }

            with open(baseline_file, 'w') as f:
                json.dump(baselines_data, f, indent=2)

            logger.debug(f"Saved {len(self._baselines[version])} baselines for version {version}")

        except Exception as e:
            logger.error(f"Failed to save baselines for version {version}: {e}")

    def get_baseline(self, operation_name: str, metric_type: PerformanceMetricType,
                     version: str) -> PerformanceBaseline | None:
        """
        Retrieve a performance baseline.

        Args:
            operation_name: Name of the operation
            metric_type: Type of metric
            version: Version identifier

        Returns:
            Performance baseline or None if not found
        """
        key = self._get_baseline_key(operation_name, metric_type, version)
        return self._baselines.get(version, {}).get(key)

    def get_latest_baseline(self, operation_name: str,
                           metric_type: PerformanceMetricType) -> PerformanceBaseline | None:
        """
        Get the most recent baseline for an operation.

        Args:
            operation_name: Name of the operation
            metric_type: Type of metric

        Returns:
            Most recent baseline or None if not found
        """
        latest_baseline = None
        latest_timestamp = 0

        for version_baselines in self._baselines.values():
            for baseline in version_baselines.values():
                if (baseline.operation_name == operation_name and
                    baseline.metric_type == metric_type and
                    baseline.last_updated > latest_timestamp):
                    latest_baseline = baseline
                    latest_timestamp = baseline.last_updated

        return latest_baseline

    def get_version_baselines(self, version: str) -> list[PerformanceBaseline]:
        """
        Get all baselines for a specific version.

        Args:
            version: Version identifier

        Returns:
            List of baselines for the version
        """
        return list(self._baselines.get(version, {}).values())

    def get_all_versions(self) -> list[str]:
        """Get all available version identifiers."""
        return list(self._baselines.keys())

    def cleanup_old_baselines(self, retention_days: int = 90) -> int:
        """
        Remove baselines older than retention period.

        Args:
            retention_days: Number of days to retain baselines

        Returns:
            Number of baselines removed
        """
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        removed_count = 0

        for version in list(self._baselines.keys()):
            version_baselines = self._baselines[version]
            original_count = len(version_baselines)

            # Remove old baselines
            keys_to_remove = [
                key for key, baseline in version_baselines.items()
                if baseline.last_updated < cutoff_time
            ]

            for key in keys_to_remove:
                del version_baselines[key]
                removed_count += 1

            # If version has no baselines left, remove it entirely
            if not version_baselines and version in self._baselines:
                del self._baselines[version]
                baseline_file = self._get_baseline_file(version)
                if baseline_file.exists():
                    baseline_file.unlink()
            elif original_count != len(version_baselines):
                # Save updated baselines
                self._save_version_baselines(version)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old baselines")

        return removed_count


class PerformanceRegressionDetector:
    """Main regression detection engine."""

    def __init__(self, storage_dir: str = "performance_baselines",
                 warning_threshold: float = 0.20,  # 20% degradation
                 critical_threshold: float = 0.50,  # 50% degradation
                 min_samples: int = 10):
        """
        Initialize regression detector.

        Args:
            storage_dir: Directory for baseline storage
            warning_threshold: Threshold for warning alerts (default 20%)
            critical_threshold: Threshold for critical alerts (default 50%)
            min_samples: Minimum samples required to establish baseline
        """
        self.storage = PerformanceBaselineStorage(storage_dir)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples

        # Track current session metrics
        self._current_metrics: list[PerformanceMetric] = []
        self._current_version = "unknown"

    def set_current_version(self, version: str) -> None:
        """Set the current version for new measurements."""
        self._current_version = version

    def record_metric(self, operation_name: str, metric_type: PerformanceMetricType,
                      value: float, metadata: dict[str, Any] | None = None) -> None:
        """
        Record a performance metric for current session.

        Args:
            operation_name: Name of the operation
            metric_type: Type of metric
            value: Measured value
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            operation_name=operation_name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            version=self._current_version
        )
        self._current_metrics.append(metric)

    def create_baseline(self, operation_name: str, metric_type: PerformanceMetricType,
                       measurements: list[float], version: str,
                       metadata: dict[str, Any] | None = None) -> PerformanceBaseline:
        """
        Create a performance baseline from measurements.

        Args:
            operation_name: Name of the operation
            metric_type: Type of metric
            measurements: List of performance measurements
            version: Version identifier
            metadata: Additional metadata

        Returns:
            Created performance baseline

        Raises:
            ValueError: If insufficient measurements provided
        """
        if len(measurements) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} measurements "
                           f"to create baseline, got {len(measurements)}")

        # Calculate statistics
        sorted_measurements = sorted(measurements)
        n = len(sorted_measurements)

        baseline = PerformanceBaseline(
            operation_name=operation_name,
            metric_type=metric_type,
            version=version,
            sample_count=n,
            mean=statistics.mean(measurements),
            median=statistics.median(measurements),
            p95=sorted_measurements[int(0.95 * n)] if n >= 20 else max(measurements),
            p99=sorted_measurements[int(0.99 * n)] if n >= 100 else max(measurements),
            std_dev=statistics.stdev(measurements) if n > 1 else 0.0,
            min_value=min(measurements),
            max_value=max(measurements),
            created_at=time.time(),
            last_updated=time.time(),
            raw_measurements=measurements[:1000]  # Store max 1000 samples
        )

        self.storage.store_baseline(baseline)
        logger.info(f"Created baseline for {operation_name} ({metric_type.value}, v{version}) "
                   f"with {n} measurements")

        return baseline

    def update_baseline(self, operation_name: str, metric_type: PerformanceMetricType,
                       new_measurements: list[float], version: str) -> PerformanceBaseline:
        """
        Update existing baseline with new measurements.

        Args:
            operation_name: Name of the operation
            metric_type: Type of metric
            new_measurements: New performance measurements
            version: Version identifier

        Returns:
            Updated performance baseline
        """
        existing_baseline = self.storage.get_baseline(operation_name, metric_type, version)

        if existing_baseline:
            # Combine with existing measurements
            all_measurements = existing_baseline.raw_measurements + new_measurements
            # Keep only most recent 1000 measurements
            all_measurements = all_measurements[-1000:]
        else:
            all_measurements = new_measurements

        return self.create_baseline(operation_name, metric_type, all_measurements, version)

    def detect_regressions(self, baseline_version: str,
                          current_metrics: list[PerformanceMetric] | None = None) -> list[RegressionResult]:
        """
        Detect performance regressions by comparing current metrics to baseline.

        Args:
            baseline_version: Version to use as baseline
            current_metrics: Current metrics to analyze (uses session metrics if None)

        Returns:
            List of regression results
        """
        if current_metrics is None:
            current_metrics = self._current_metrics

        results = []

        # Group current metrics by operation and type
        current_by_op = defaultdict(lambda: defaultdict(list))
        for metric in current_metrics:
            current_by_op[metric.operation_name][metric.metric_type].append(metric.value)

        # Compare each operation's performance
        for operation_name, metrics_by_type in current_by_op.items():
            for metric_type, values in metrics_by_type.items():
                baseline = self.storage.get_baseline(operation_name, metric_type, baseline_version)

                if baseline is None:
                    logger.warning(f"No baseline found for {operation_name} "
                                 f"({metric_type.value}, v{baseline_version})")
                    continue

                # Calculate current performance statistics
                if not values:
                    continue

                statistics.mean(values)
                current_p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)

                # Use p95 for comparison to reduce noise from outliers
                baseline_value = baseline.p95
                current_value = current_p95

                # Calculate percentage change
                if baseline_value > 0:
                    percentage_change = (current_value - baseline_value) / baseline_value
                else:
                    percentage_change = 0.0

                # Determine severity
                severity = RegressionSeverity.NONE
                if percentage_change > self.critical_threshold:
                    severity = RegressionSeverity.CRITICAL
                elif percentage_change > self.warning_threshold:
                    severity = RegressionSeverity.WARNING

                # Generate message and recommendations
                message = self._generate_regression_message(
                    operation_name, metric_type, severity, percentage_change,
                    baseline_value, current_value
                )

                recommendations = self._generate_recommendations(
                    operation_name, metric_type, severity, percentage_change
                )

                result = RegressionResult(
                    operation_name=operation_name,
                    metric_type=metric_type,
                    severity=severity,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    percentage_change=percentage_change,
                    baseline_version=baseline_version,
                    current_version=self._current_version,
                    message=message,
                    timestamp=time.time(),
                    recommendations=recommendations
                )

                results.append(result)

        return results

    def _generate_regression_message(self, operation_name: str, metric_type: PerformanceMetricType,
                                   severity: RegressionSeverity, percentage_change: float,
                                   baseline_value: float, current_value: float) -> str:
        """Generate human-readable regression message."""
        if severity == RegressionSeverity.NONE:
            if percentage_change < -0.1:  # 10% improvement
                return (f"{operation_name} ({metric_type.value}) improved by "
                       f"{abs(percentage_change)*100:.1f}%: "
                       f"{baseline_value:.4f} → {current_value:.4f}")
            else:
                return (f"{operation_name} ({metric_type.value}) performance stable: "
                       f"{baseline_value:.4f} → {current_value:.4f} "
                       f"({percentage_change*100:+.1f}%)")
        elif severity == RegressionSeverity.WARNING:
            return (f"WARNING: {operation_name} ({metric_type.value}) degraded by "
                   f"{percentage_change*100:.1f}%: "
                   f"{baseline_value:.4f} → {current_value:.4f}")
        else:  # CRITICAL
            return (f"CRITICAL: {operation_name} ({metric_type.value}) severely degraded by "
                   f"{percentage_change*100:.1f}%: "
                   f"{baseline_value:.4f} → {current_value:.4f}")

    def _generate_recommendations(self, operation_name: str, metric_type: PerformanceMetricType,
                                severity: RegressionSeverity, percentage_change: float) -> list[str]:
        """Generate recommendations based on regression analysis."""
        recommendations = []

        if severity == RegressionSeverity.CRITICAL:
            recommendations.extend([
                f"Immediately investigate {operation_name} implementation changes",
                "Review recent code changes affecting this operation",
                "Consider rolling back recent changes if no improvement found",
                "Profile the operation to identify performance bottlenecks"
            ])
        elif severity == RegressionSeverity.WARNING:
            recommendations.extend([
                f"Monitor {operation_name} performance closely",
                "Review implementation for optimization opportunities",
                "Consider adding operation-specific caching if applicable"
            ])

        if metric_type == PerformanceMetricType.EXECUTION_TIME:
            recommendations.append("Focus on algorithmic optimizations and caching strategies")
        elif metric_type == PerformanceMetricType.MEMORY_USAGE:
            recommendations.append("Review memory allocation patterns and consider object pooling")

        return recommendations

    def generate_report(self, baseline_version: str,
                       current_metrics: list[PerformanceMetric] | None = None) -> PerformanceReport:
        """
        Generate comprehensive performance comparison report.

        Args:
            baseline_version: Version to use as baseline
            current_metrics: Current metrics to analyze

        Returns:
            Comprehensive performance report
        """
        regression_results = self.detect_regressions(baseline_version, current_metrics)

        # Calculate summary statistics
        total_operations = len(regression_results)
        critical_regressions = sum(1 for r in regression_results
                                 if r.severity == RegressionSeverity.CRITICAL)
        warnings = sum(1 for r in regression_results
                      if r.severity == RegressionSeverity.WARNING)
        improvements = sum(1 for r in regression_results
                          if r.percentage_change < -0.1)  # >10% improvement

        regressions_found = critical_regressions + warnings

        # Generate overall recommendations
        overall_recommendations = []
        if critical_regressions > 0:
            overall_recommendations.append(
                f"URGENT: {critical_regressions} critical performance regressions detected. "
                "Immediate investigation required."
            )
        if warnings > 0:
            overall_recommendations.append(
                f"Monitor {warnings} operations showing performance warnings."
            )
        if improvements > 0:
            overall_recommendations.append(
                f"Good news: {improvements} operations showed performance improvements."
            )

        if not overall_recommendations:
            overall_recommendations.append("No significant performance regressions detected.")

        # Calculate summary statistics
        summary_stats = {
            "total_operations_analyzed": total_operations,
            "regression_rate": (regressions_found / total_operations * 100) if total_operations > 0 else 0,
            "improvement_rate": (improvements / total_operations * 100) if total_operations > 0 else 0,
            "average_change_percent": statistics.mean([r.percentage_change * 100 for r in regression_results]) if regression_results else 0,
            "worst_regression_percent": max([r.percentage_change * 100 for r in regression_results]) if regression_results else 0,
            "best_improvement_percent": min([r.percentage_change * 100 for r in regression_results]) if regression_results else 0
        }

        return PerformanceReport(
            baseline_version=baseline_version,
            current_version=self._current_version,
            generation_time=time.time(),
            total_operations_analyzed=total_operations,
            regressions_found=regressions_found,
            warnings_found=warnings,
            critical_regressions=critical_regressions,
            operations_improved=improvements,
            regression_results=regression_results,
            summary_statistics=summary_stats,
            recommendations=overall_recommendations
        )

    def save_report(self, report: PerformanceReport, output_path: str) -> None:
        """
        Save performance report to file.

        Args:
            report: Performance report to save
            output_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)

            logger.info(f"Performance report saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")

    def clear_session_metrics(self) -> None:
        """Clear current session metrics."""
        self._current_metrics.clear()

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for current session metrics."""
        if not self._current_metrics:
            return {"total_metrics": 0}

        by_operation = defaultdict(lambda: defaultdict(list))
        for metric in self._current_metrics:
            by_operation[metric.operation_name][metric.metric_type.value].append(metric.value)

        return {
            "total_metrics": len(self._current_metrics),
            "unique_operations": len(by_operation),
            "operations_tracked": list(by_operation.keys()),
            "current_version": self._current_version,
            "time_span_minutes": (max(m.timestamp for m in self._current_metrics) -
                                min(m.timestamp for m in self._current_metrics)) / 60 if len(self._current_metrics) > 1 else 0
        }
