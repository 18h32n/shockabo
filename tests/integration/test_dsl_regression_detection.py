"""
Integration tests for DSL engine performance regression detection.

Tests the complete integration between DSL engine profiling and 
regression detection system.
"""

import json
import os
import shutil
import tempfile
import unittest

from src.domain.dsl.base import DSLProgram, Operation, OperationResult
from src.domain.dsl.types import Grid
from src.domain.services.dsl_engine import DSLEngine
from src.utils.performance_regression_detector import PerformanceMetricType


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, execution_time: float = 0.05, should_fail: bool = False, **parameters):
        """Initialize mock operation with configurable behavior."""
        self.execution_time = execution_time
        self.should_fail = should_fail
        super().__init__(**parameters)

    def execute(self, grid: Grid, context=None) -> OperationResult:
        """Execute mock operation with simulated timing."""
        import time
        time.sleep(self.execution_time)

        if self.should_fail:
            return OperationResult(
                success=False,
                grid=grid,
                error_message="Mock operation failed"
            )

        # Simple transformation: add 1 to all values
        result_grid = [[cell + 1 for cell in row] for row in grid]

        return OperationResult(
            success=True,
            grid=result_grid,
            execution_time=self.execution_time
        )

    @classmethod
    def get_name(cls) -> str:
        return "mock_operation"

    @classmethod
    def get_description(cls) -> str:
        return "Mock operation for testing"

    @classmethod
    def get_parameter_schema(cls) -> dict:
        return {
            "execution_time": {
                "type": "float",
                "required": False,
                "description": "Simulated execution time"
            }
        }


class FastMockOperation(MockOperation):
    """Fast mock operation for baseline comparison."""

    def __init__(self, **parameters):
        super().__init__(execution_time=0.01, **parameters)

    @classmethod
    def get_name(cls) -> str:
        return "fast_mock_operation"


class SlowMockOperation(MockOperation):
    """Slow mock operation for regression testing."""

    def __init__(self, **parameters):
        super().__init__(execution_time=0.08, **parameters)

    @classmethod
    def get_name(cls) -> str:
        return "slow_mock_operation"


class TestDSLEngineRegressionIntegration(unittest.TestCase):
    """Test DSL engine integration with regression detection."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create DSL engine with regression detection enabled
        self.engine = DSLEngine(
            timeout_seconds=2.0,
            memory_limit_mb=200,
            enable_profiling=True,
            enable_regression_detection=True,
            version="2.0.0"
        )

        # Register mock operations
        self.engine.register_operation(MockOperation)
        self.engine.register_operation(FastMockOperation)
        self.engine.register_operation(SlowMockOperation)

        # Create test grid
        self.test_grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_regression_detection_enabled(self):
        """Test that regression detection is properly enabled."""
        stats = self.engine.get_regression_detector_stats()

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["current_version"], "2.0.0")
        self.assertIsInstance(stats["available_baseline_versions"], list)

    def test_metric_recording_during_execution(self):
        """Test that metrics are recorded during program execution."""
        program = DSLProgram(operations=[
            {"name": "mock_operation", "parameters": {"execution_time": 0.03}},
            {"name": "fast_mock_operation", "parameters": {}}
        ])

        # Execute program
        result = self.engine.execute_program(program, self.test_grid)

        self.assertTrue(result.success)

        # Check that metrics were recorded
        stats = self.engine.get_regression_detector_stats()
        self.assertGreater(stats["session_metrics"], 0)
        self.assertGreater(stats["unique_operations_tracked"], 0)

        # Verify specific operations were tracked
        self.assertIn("mock_operation", stats["operations_tracked"])
        self.assertIn("fast_mock_operation", stats["operations_tracked"])

    def test_baseline_creation_from_profiles(self):
        """Test creating baselines from operation profiles."""
        # Run multiple programs to build up profiles
        for i in range(15):  # Ensure we have enough samples
            program = DSLProgram(operations=[
                {"name": "mock_operation", "parameters": {"execution_time": 0.05 + i * 0.001}}
            ])

            result = self.engine.execute_program(program, self.test_grid)
            self.assertTrue(result.success)

        # Create baseline from profiles
        baseline_result = self.engine.create_performance_baseline("1.0.0")

        self.assertGreater(baseline_result["baselines_created"], 0)
        self.assertEqual(baseline_result["version"], "1.0.0")
        self.assertGreater(baseline_result["total_measurements"], 0)

    def test_regression_detection_no_regression(self):
        """Test regression detection when performance is stable."""
        # Create baseline
        for i in range(10):
            program = DSLProgram(operations=[
                {"name": "mock_operation", "parameters": {"execution_time": 0.05}}
            ])
            self.engine.execute_program(program, self.test_grid)

        self.engine.create_performance_baseline("1.0.0")

        # Clear profiles and run similar performance tests
        self.engine.clear_cache()
        self.engine._operation_profiles.clear()

        for i in range(5):
            program = DSLProgram(operations=[
                {"name": "mock_operation", "parameters": {"execution_time": 0.052}}  # Slightly slower but within tolerance
            ])
            self.engine.execute_program(program, self.test_grid)

        # Detect regressions
        regression_result = self.engine.detect_performance_regressions("1.0.0")

        self.assertEqual(regression_result["critical_regressions"], 0)
        self.assertEqual(regression_result["warnings"], 0)

    def test_regression_detection_warning(self):
        """Test regression detection for warning-level performance degradation."""
        # Create baseline with fast operations
        for i in range(10):
            program = DSLProgram(operations=[
                {"name": "fast_mock_operation", "parameters": {}}
            ])
            self.engine.execute_program(program, self.test_grid)

        self.engine.create_performance_baseline("1.0.0")

        # Clear and run with slower performance (should trigger warning)
        self.engine.regression_detector.clear_session_metrics()

        for i in range(5):
            program = DSLProgram(operations=[
                {"name": "mock_operation", "parameters": {"execution_time": 0.025}}  # ~25% slower than fast operation
            ])
            # Manually record under the same operation name for comparison
            result = self.engine.execute_program(program, self.test_grid)
            self.engine.regression_detector.record_metric(
                "fast_mock_operation",
                PerformanceMetricType.EXECUTION_TIME,
                0.025
            )

        regression_result = self.engine.detect_performance_regressions("1.0.0")

        self.assertGreater(regression_result["warnings"], 0)
        self.assertIn("fast_mock_operation", regression_result["warning_operations"])

    def test_regression_detection_critical(self):
        """Test regression detection for critical performance degradation."""
        # Create baseline with fast operations
        for i in range(10):
            program = DSLProgram(operations=[
                {"name": "fast_mock_operation", "parameters": {}}
            ])
            self.engine.execute_program(program, self.test_grid)

        self.engine.create_performance_baseline("1.0.0")

        # Clear and run with much slower performance (should trigger critical)
        self.engine.regression_detector.clear_session_metrics()

        for i in range(5):
            # Manually record critical regression (>50% slower)
            self.engine.regression_detector.record_metric(
                "fast_mock_operation",
                PerformanceMetricType.EXECUTION_TIME,
                0.08  # 8x slower than baseline (0.01)
            )

        regression_result = self.engine.detect_performance_regressions("1.0.0")

        self.assertGreater(regression_result["critical_regressions"], 0)
        self.assertIn("fast_mock_operation", regression_result["critical_operations"])

    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        # Create baseline
        operations_data = [
            ("fast_mock_operation", 0.01),
            ("mock_operation", 0.05),
        ]

        for op_name, exec_time in operations_data:
            for i in range(10):
                program = DSLProgram(operations=[
                    {"name": op_name, "parameters": {"execution_time": exec_time}}
                ])
                self.engine.execute_program(program, self.test_grid)

        self.engine.create_performance_baseline("1.0.0")

        # Clear and create mixed performance scenario
        self.engine.regression_detector.clear_session_metrics()

        # Record improved performance for one operation
        for i in range(5):
            self.engine.regression_detector.record_metric(
                "fast_mock_operation",
                PerformanceMetricType.EXECUTION_TIME,
                0.005  # 50% improvement
            )

        # Record degraded performance for another
        for i in range(5):
            self.engine.regression_detector.record_metric(
                "mock_operation",
                PerformanceMetricType.EXECUTION_TIME,
                0.08  # 60% degradation
            )

        # Generate report
        output_path = os.path.join(self.temp_dir, "regression_report.json")
        report_summary = self.engine.generate_performance_report("1.0.0", output_path)

        # Verify report summary
        self.assertEqual(report_summary["baseline_version"], "1.0.0")
        self.assertEqual(report_summary["current_version"], "2.0.0")
        self.assertGreater(report_summary["total_operations_analyzed"], 0)
        self.assertGreater(report_summary["critical_regressions"], 0)
        self.assertGreater(report_summary["operations_improved"], 0)
        self.assertTrue(report_summary["report_saved"])

        # Verify report file was created
        self.assertTrue(os.path.exists(output_path))

        # Verify report content
        with open(output_path) as f:
            report_data = json.load(f)

        self.assertIn("regression_results", report_data)
        self.assertIn("recommendations", report_data)
        self.assertIn("summary_statistics", report_data)

    def test_version_management(self):
        """Test version management functionality."""
        # Test initial version
        self.assertEqual(self.engine.version, "2.0.0")

        # Update version
        self.engine.set_version("3.0.0")
        self.assertEqual(self.engine.version, "3.0.0")

        # Verify regression detector was updated
        stats = self.engine.get_regression_detector_stats()
        self.assertEqual(stats["current_version"], "3.0.0")

    def test_cache_performance_tracking(self):
        """Test that cache performance is tracked for regression detection."""
        # Create a program that will benefit from caching
        program = DSLProgram(operations=[
            {"name": "mock_operation", "parameters": {"execution_time": 0.02}}
        ])

        # Run multiple times to populate cache
        for i in range(5):
            result = self.engine.execute_program(program, self.test_grid)
            self.assertTrue(result.success)

        # Verify cache metrics were recorded
        stats = self.engine.get_regression_detector_stats()
        self.assertGreater(stats["session_metrics"], 0)

        # Check that cache performance baseline can be created
        baseline_result = self.engine.create_performance_baseline("1.0.0")
        self.assertGreater(baseline_result["baselines_created"], 0)

    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked in regression detection."""
        # Run a program and verify memory metrics are recorded
        program = DSLProgram(operations=[
            {"name": "mock_operation", "parameters": {}}
        ])

        result = self.engine.execute_program(program, self.test_grid)
        self.assertTrue(result.success)

        # Memory tracking should be included in metadata
        self.assertIn("peak_memory_mb", result.metadata)

        # Memory metrics should be recorded for regression detection
        stats = self.engine.get_regression_detector_stats()
        self.assertGreater(stats["session_metrics"], 0)

    def test_disabled_regression_detection(self):
        """Test engine behavior when regression detection is disabled."""
        # Create engine with regression detection disabled
        engine_no_regression = DSLEngine(
            enable_regression_detection=False,
            version="1.0.0"
        )

        stats = engine_no_regression.get_regression_detector_stats()
        self.assertIn("error", stats)

        baseline_result = engine_no_regression.create_performance_baseline()
        self.assertIn("error", baseline_result)

        regression_result = engine_no_regression.detect_performance_regressions("1.0.0")
        self.assertIn("error", regression_result)

    def test_large_scale_regression_detection(self):
        """Test regression detection with larger scale operations."""
        operation_names = ["mock_operation", "fast_mock_operation", "slow_mock_operation"]

        # Create baselines with many operations
        for op_name in operation_names:
            for i in range(20):  # More samples for statistical significance
                program = DSLProgram(operations=[
                    {"name": op_name, "parameters": {}}
                ])
                result = self.engine.execute_program(program, self.test_grid)
                self.assertTrue(result.success)

        # Create comprehensive baseline
        baseline_result = self.engine.create_performance_baseline("1.0.0")
        self.assertGreaterEqual(baseline_result["baselines_created"], len(operation_names))

        # Clear and simulate mixed performance changes
        self.engine.regression_detector.clear_session_metrics()

        # Simulate various performance patterns
        performance_changes = {
            "mock_operation": 0.06,      # 20% degradation (warning)
            "fast_mock_operation": 0.005, # 50% improvement
            "slow_mock_operation": 0.15   # 87% degradation (critical)
        }

        for op_name, new_time in performance_changes.items():
            for i in range(10):
                self.engine.regression_detector.record_metric(
                    op_name,
                    PerformanceMetricType.EXECUTION_TIME,
                    new_time
                )

        # Detect regressions
        regression_result = self.engine.detect_performance_regressions("1.0.0")

        # Verify comprehensive analysis
        self.assertEqual(regression_result["total_operations_analyzed"], 3)
        self.assertGreater(regression_result["critical_regressions"], 0)
        self.assertGreater(regression_result["warnings"], 0)
        self.assertGreater(regression_result["improvements"], 0)

        # Verify specific operations are categorized correctly
        self.assertIn("slow_mock_operation", regression_result["critical_operations"])

        # Generate and verify comprehensive report
        output_path = os.path.join(self.temp_dir, "large_scale_report.json")
        report_summary = self.engine.generate_performance_report("1.0.0", output_path)

        self.assertTrue(report_summary["report_saved"])
        self.assertGreater(len(report_summary["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main()
