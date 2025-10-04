"""
Unit tests for performance regression detection system.

Tests cover baseline storage, regression detection, report generation,
and integration with DSL engine profiling.
"""

import json
import os
import shutil
import statistics
import tempfile
import time
import unittest

from src.utils.performance_regression_detector import (
    PerformanceBaseline,
    PerformanceBaselineStorage,
    PerformanceMetricType,
    PerformanceRegressionDetector,
    RegressionSeverity,
)


class TestPerformanceBaselineStorage(unittest.TestCase):
    """Test performance baseline storage functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = PerformanceBaselineStorage(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_baseline(self):
        """Test storing and retrieving baselines."""
        baseline = PerformanceBaseline(
            operation_name="test_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            version="1.0.0",
            sample_count=100,
            mean=0.05,
            median=0.045,
            p95=0.08,
            p99=0.12,
            std_dev=0.015,
            min_value=0.02,
            max_value=0.15,
            created_at=time.time(),
            last_updated=time.time(),
            raw_measurements=[0.05] * 100
        )

        # Store baseline
        self.storage.store_baseline(baseline)

        # Retrieve baseline
        retrieved = self.storage.get_baseline(
            "test_operation",
            PerformanceMetricType.EXECUTION_TIME,
            "1.0.0"
        )

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.operation_name, "test_operation")
        self.assertEqual(retrieved.version, "1.0.0")
        self.assertEqual(retrieved.mean, 0.05)
        self.assertEqual(retrieved.sample_count, 100)

    def test_get_latest_baseline(self):
        """Test retrieving the most recent baseline."""
        # Create baselines with different timestamps
        baseline1 = PerformanceBaseline(
            operation_name="test_op",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            version="1.0.0",
            sample_count=50,
            mean=0.05,
            median=0.045,
            p95=0.08,
            p99=0.12,
            std_dev=0.015,
            min_value=0.02,
            max_value=0.15,
            created_at=time.time() - 3600,  # 1 hour ago
            last_updated=time.time() - 3600,
            raw_measurements=[]
        )

        baseline2 = PerformanceBaseline(
            operation_name="test_op",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            version="2.0.0",
            sample_count=75,
            mean=0.04,
            median=0.035,
            p95=0.07,
            p99=0.10,
            std_dev=0.012,
            min_value=0.01,
            max_value=0.12,
            created_at=time.time(),  # Now
            last_updated=time.time(),
            raw_measurements=[]
        )

        self.storage.store_baseline(baseline1)
        self.storage.store_baseline(baseline2)

        latest = self.storage.get_latest_baseline(
            "test_op",
            PerformanceMetricType.EXECUTION_TIME
        )

        self.assertIsNotNone(latest)
        self.assertEqual(latest.version, "2.0.0")
        self.assertEqual(latest.mean, 0.04)

    def test_get_version_baselines(self):
        """Test retrieving all baselines for a version."""
        baselines = [
            PerformanceBaseline(
                operation_name=f"operation_{i}",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                version="1.0.0",
                sample_count=10,
                mean=0.05 + i * 0.01,
                median=0.045 + i * 0.01,
                p95=0.08 + i * 0.01,
                p99=0.12 + i * 0.01,
                std_dev=0.015,
                min_value=0.02,
                max_value=0.15,
                created_at=time.time(),
                last_updated=time.time(),
                raw_measurements=[]
            )
            for i in range(3)
        ]

        for baseline in baselines:
            self.storage.store_baseline(baseline)

        version_baselines = self.storage.get_version_baselines("1.0.0")

        self.assertEqual(len(version_baselines), 3)
        operation_names = {b.operation_name for b in version_baselines}
        expected_names = {"operation_0", "operation_1", "operation_2"}
        self.assertEqual(operation_names, expected_names)

    def test_cleanup_old_baselines(self):
        """Test cleanup of old baselines."""
        old_time = time.time() - (100 * 24 * 60 * 60)  # 100 days ago
        recent_time = time.time()

        old_baseline = PerformanceBaseline(
            operation_name="old_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            version="0.9.0",
            sample_count=10,
            mean=0.05,
            median=0.045,
            p95=0.08,
            p99=0.12,
            std_dev=0.015,
            min_value=0.02,
            max_value=0.15,
            created_at=old_time,
            last_updated=old_time,
            raw_measurements=[]
        )

        recent_baseline = PerformanceBaseline(
            operation_name="recent_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            version="1.0.0",
            sample_count=10,
            mean=0.04,
            median=0.035,
            p95=0.07,
            p99=0.10,
            std_dev=0.012,
            min_value=0.01,
            max_value=0.12,
            created_at=recent_time,
            last_updated=recent_time,
            raw_measurements=[]
        )

        self.storage.store_baseline(old_baseline)
        self.storage.store_baseline(recent_baseline)

        # Cleanup with 90-day retention
        removed_count = self.storage.cleanup_old_baselines(retention_days=90)

        self.assertEqual(removed_count, 1)

        # Verify old baseline is gone
        old_retrieved = self.storage.get_baseline(
            "old_operation",
            PerformanceMetricType.EXECUTION_TIME,
            "0.9.0"
        )
        self.assertIsNone(old_retrieved)

        # Verify recent baseline remains
        recent_retrieved = self.storage.get_baseline(
            "recent_operation",
            PerformanceMetricType.EXECUTION_TIME,
            "1.0.0"
        )
        self.assertIsNotNone(recent_retrieved)


class TestPerformanceRegressionDetector(unittest.TestCase):
    """Test performance regression detection functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = PerformanceRegressionDetector(
            storage_dir=self.temp_dir,
            warning_threshold=0.20,
            critical_threshold=0.50,
            min_samples=5
        )
        self.detector.set_current_version("2.0.0")

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_baseline(self):
        """Test baseline creation from measurements."""
        measurements = [0.05, 0.06, 0.04, 0.055, 0.045, 0.052, 0.048, 0.062, 0.058, 0.051]

        baseline = self.detector.create_baseline(
            operation_name="test_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=measurements,
            version="1.0.0"
        )

        self.assertEqual(baseline.operation_name, "test_operation")
        self.assertEqual(baseline.version, "1.0.0")
        self.assertEqual(baseline.sample_count, len(measurements))
        self.assertAlmostEqual(baseline.mean, statistics.mean(measurements), places=4)
        self.assertAlmostEqual(baseline.median, statistics.median(measurements), places=4)
        self.assertEqual(baseline.min_value, min(measurements))
        self.assertEqual(baseline.max_value, max(measurements))

    def test_create_baseline_insufficient_samples(self):
        """Test baseline creation with insufficient samples."""
        measurements = [0.05, 0.06, 0.04]  # Less than min_samples (5)

        with self.assertRaises(ValueError):
            self.detector.create_baseline(
                operation_name="test_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                measurements=measurements,
                version="1.0.0"
            )

    def test_record_metric(self):
        """Test recording performance metrics."""
        self.detector.record_metric(
            operation_name="test_op",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=0.05,
            metadata={"test": True}
        )

        stats = self.detector.get_session_stats()
        self.assertEqual(stats["total_metrics"], 1)
        self.assertEqual(stats["unique_operations"], 1)
        self.assertIn("test_op", stats["operations_tracked"])

    def test_detect_no_regression(self):
        """Test detection when no regression occurs."""
        # Create baseline
        baseline_measurements = [0.05] * 10
        self.detector.create_baseline(
            operation_name="stable_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=baseline_measurements,
            version="1.0.0"
        )

        # Record similar current measurements
        for _ in range(5):
            self.detector.record_metric(
                operation_name="stable_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=0.052  # 4% slower - within tolerance
            )

        results = self.detector.detect_regressions("1.0.0")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].severity, RegressionSeverity.NONE)
        self.assertLess(results[0].percentage_change, 0.20)  # Less than warning threshold

    def test_detect_warning_regression(self):
        """Test detection of warning-level regression."""
        # Create baseline
        baseline_measurements = [0.05] * 10
        self.detector.create_baseline(
            operation_name="warning_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=baseline_measurements,
            version="1.0.0"
        )

        # Record degraded current measurements (30% slower)
        for _ in range(5):
            self.detector.record_metric(
                operation_name="warning_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=0.065
            )

        results = self.detector.detect_regressions("1.0.0")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].severity, RegressionSeverity.WARNING)
        self.assertGreaterEqual(results[0].percentage_change, 0.20)  # At least warning threshold
        self.assertLess(results[0].percentage_change, 0.50)  # Less than critical threshold

    def test_detect_critical_regression(self):
        """Test detection of critical regression."""
        # Create baseline
        baseline_measurements = [0.05] * 10
        self.detector.create_baseline(
            operation_name="critical_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=baseline_measurements,
            version="1.0.0"
        )

        # Record severely degraded current measurements (60% slower)
        for _ in range(5):
            self.detector.record_metric(
                operation_name="critical_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=0.08
            )

        results = self.detector.detect_regressions("1.0.0")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].severity, RegressionSeverity.CRITICAL)
        self.assertGreaterEqual(results[0].percentage_change, 0.50)  # At least critical threshold

    def test_detect_improvement(self):
        """Test detection of performance improvements."""
        # Create baseline
        baseline_measurements = [0.10] * 10
        self.detector.create_baseline(
            operation_name="improved_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=baseline_measurements,
            version="1.0.0"
        )

        # Record improved current measurements (50% faster)
        for _ in range(5):
            self.detector.record_metric(
                operation_name="improved_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=0.05
            )

        results = self.detector.detect_regressions("1.0.0")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].severity, RegressionSeverity.NONE)
        self.assertLess(results[0].percentage_change, -0.1)  # Significant improvement
        self.assertIn("improved", results[0].message)

    def test_update_baseline(self):
        """Test updating existing baseline with new measurements."""
        # Create initial baseline
        initial_measurements = [0.05] * 10
        baseline = self.detector.create_baseline(
            operation_name="update_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=initial_measurements,
            version="1.0.0"
        )

        initial_mean = baseline.mean

        # Update with new measurements
        new_measurements = [0.06] * 5
        updated_baseline = self.detector.update_baseline(
            operation_name="update_operation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            new_measurements=new_measurements,
            version="1.0.0"
        )

        # Verify baseline was updated
        self.assertGreater(updated_baseline.mean, initial_mean)
        self.assertEqual(updated_baseline.sample_count, 15)  # 10 + 5

    def test_generate_report(self):
        """Test performance report generation."""
        # Create baselines
        self.detector.create_baseline(
            operation_name="op1",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=[0.05] * 10,
            version="1.0.0"
        )

        self.detector.create_baseline(
            operation_name="op2",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=[0.03] * 10,
            version="1.0.0"
        )

        # Record current metrics with mixed performance
        # op1: critical regression (100% slower)
        for _ in range(5):
            self.detector.record_metric("op1", PerformanceMetricType.EXECUTION_TIME, 0.10)

        # op2: improvement (33% faster)
        for _ in range(5):
            self.detector.record_metric("op2", PerformanceMetricType.EXECUTION_TIME, 0.02)

        report = self.detector.generate_report("1.0.0")

        self.assertEqual(report.baseline_version, "1.0.0")
        self.assertEqual(report.current_version, "2.0.0")
        self.assertEqual(report.total_operations_analyzed, 2)
        self.assertEqual(report.critical_regressions, 1)
        self.assertEqual(report.operations_improved, 1)
        self.assertEqual(report.regressions_found, 1)  # Only critical counts as regression

        # Verify summary statistics
        self.assertIn("regression_rate", report.summary_statistics)
        self.assertIn("improvement_rate", report.summary_statistics)

        # Verify recommendations
        self.assertGreater(len(report.recommendations), 0)
        self.assertIn("URGENT", report.recommendations[0])  # Critical regression warning

    def test_save_and_load_report(self):
        """Test saving and loading performance reports."""
        # Create a simple report
        self.detector.create_baseline(
            operation_name="test_op",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=[0.05] * 10,
            version="1.0.0"
        )

        self.detector.record_metric("test_op", PerformanceMetricType.EXECUTION_TIME, 0.05)

        report = self.detector.generate_report("1.0.0")

        # Save report
        output_path = os.path.join(self.temp_dir, "test_report.json")
        self.detector.save_report(report, output_path)

        # Verify file was created
        self.assertTrue(os.path.exists(output_path))

        # Load and verify report data
        with open(output_path) as f:
            loaded_data = json.load(f)

        self.assertEqual(loaded_data["baseline_version"], "1.0.0")
        self.assertEqual(loaded_data["current_version"], "2.0.0")
        self.assertEqual(loaded_data["total_operations_analyzed"], 1)
        self.assertIn("generation_time_iso", loaded_data)


class TestRegressionDetectorIntegration(unittest.TestCase):
    """Test integration scenarios and edge cases."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = PerformanceRegressionDetector(
            storage_dir=self.temp_dir,
            min_samples=3  # Lower threshold for testing
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_metric_types(self):
        """Test handling multiple metric types for same operation."""
        operation_name = "multi_metric_op"
        version = "1.0.0"

        # Create baselines for different metric types
        self.detector.create_baseline(
            operation_name=operation_name,
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=[0.05, 0.06, 0.04],
            version=version
        )

        self.detector.create_baseline(
            operation_name=operation_name,
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            measurements=[10.0, 12.0, 8.0],
            version=version
        )

        # Record current metrics
        self.detector.set_current_version("2.0.0")
        self.detector.record_metric(operation_name, PerformanceMetricType.EXECUTION_TIME, 0.08)  # Regression
        self.detector.record_metric(operation_name, PerformanceMetricType.MEMORY_USAGE, 8.0)  # Improvement

        results = self.detector.detect_regressions(version)

        # Should detect both metrics
        self.assertEqual(len(results), 2)

        # Find results by metric type
        time_result = next(r for r in results if r.metric_type == PerformanceMetricType.EXECUTION_TIME)
        memory_result = next(r for r in results if r.metric_type == PerformanceMetricType.MEMORY_USAGE)

        # Execution time should show regression
        self.assertEqual(time_result.severity, RegressionSeverity.WARNING)
        self.assertGreater(time_result.percentage_change, 0)

        # Memory usage should show improvement
        self.assertEqual(memory_result.severity, RegressionSeverity.NONE)
        self.assertLess(memory_result.percentage_change, 0)

    def test_missing_baseline(self):
        """Test behavior when baseline is missing."""
        self.detector.set_current_version("2.0.0")
        self.detector.record_metric("new_operation", PerformanceMetricType.EXECUTION_TIME, 0.05)

        # Try to detect regressions against non-existent baseline
        results = self.detector.detect_regressions("1.0.0")

        # Should return empty results (operation skipped due to missing baseline)
        self.assertEqual(len(results), 0)

    def test_empty_current_metrics(self):
        """Test behavior with no current metrics."""
        # Create baseline but don't record any current metrics
        self.detector.create_baseline(
            operation_name="baseline_only",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=[0.05, 0.06, 0.04],
            version="1.0.0"
        )

        results = self.detector.detect_regressions("1.0.0")

        # Should return empty results
        self.assertEqual(len(results), 0)

    def test_session_stats(self):
        """Test session statistics tracking."""
        # Record various metrics
        operations = ["op1", "op2", "op3"]
        for i, op in enumerate(operations):
            for j in range(i + 1):  # Different numbers of measurements
                self.detector.record_metric(
                    operation_name=op,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=0.05 + i * 0.01
                )

        stats = self.detector.get_session_stats()

        self.assertEqual(stats["total_metrics"], 6)  # 1 + 2 + 3
        self.assertEqual(stats["unique_operations"], 3)
        self.assertEqual(set(stats["operations_tracked"]), set(operations))
        self.assertGreater(stats["time_span_minutes"], 0)

    def test_clear_session_metrics(self):
        """Test clearing session metrics."""
        # Record some metrics
        self.detector.record_metric("test_op", PerformanceMetricType.EXECUTION_TIME, 0.05)

        stats_before = self.detector.get_session_stats()
        self.assertEqual(stats_before["total_metrics"], 1)

        # Clear metrics
        self.detector.clear_session_metrics()

        stats_after = self.detector.get_session_stats()
        self.assertEqual(stats_after["total_metrics"], 0)


if __name__ == "__main__":
    unittest.main()
