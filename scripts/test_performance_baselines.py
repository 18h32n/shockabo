"""
Test Script for Performance Baseline System

This script validates the complete performance baseline measurement and regression
detection system for Python Function Synthesis. It tests:

1. Baseline creation functionality
2. Performance measurement accuracy
3. Regression detection logic
4. Report generation
5. Target compliance validation
6. End-to-end workflow
"""

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_performance_baselines import PerformanceBaselineCreator
from scripts.performance_comparison import PerformanceComparator
from src.utils.performance_regression_detector import (
    PerformanceMetricType,
    PerformanceRegressionDetector,
    RegressionSeverity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBaselineTestSuite:
    """Comprehensive test suite for performance baseline system."""

    def __init__(self):
        """Initialize test suite with temporary storage."""
        self.temp_dir = tempfile.mkdtemp(prefix="perf_baseline_test_")
        self.test_storage_dir = os.path.join(self.temp_dir, "test_baselines")
        self.results = []

        logger.info(f"Test storage directory: {self.test_storage_dir}")

    def cleanup(self):
        """Clean up temporary test files."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def add_result(self, test_name: str, success: bool, message: str, details: dict[str, Any] = None):
        """Add test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": time.time()
        }
        self.results.append(result)

        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name} - {message}")

    def test_regression_detector_basic(self) -> bool:
        """Test basic regression detector functionality."""
        test_name = "Regression Detector Basic"

        try:
            # Initialize detector
            detector = PerformanceRegressionDetector(
                storage_dir=self.test_storage_dir,
                warning_threshold=0.20,
                critical_threshold=0.50,
                min_samples=5
            )

            # Test version setting
            detector.set_current_version("test_v1.0")

            # Test metric recording
            for i in range(10):
                detector.record_metric(
                    operation_name="test_operation",
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=0.05 + (i * 0.001),  # 50-59ms
                    metadata={"test_iteration": i}
                )

            # Test baseline creation
            measurements = [0.05 + (i * 0.001) for i in range(15)]
            baseline = detector.create_baseline(
                operation_name="test_operation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                measurements=measurements,
                version="baseline_v1.0"
            )

            # Validate baseline properties
            assert baseline.sample_count == 15
            assert 0.05 <= baseline.mean <= 0.06
            assert baseline.p95 > baseline.median
            assert baseline.version == "baseline_v1.0"

            self.add_result(test_name, True, "Basic functionality validated", {
                "baseline_created": True,
                "sample_count": baseline.sample_count,
                "mean": baseline.mean,
                "p95": baseline.p95
            })
            return True

        except Exception as e:
            self.add_result(test_name, False, f"Basic functionality failed: {e}")
            return False

    def test_regression_detection_logic(self) -> bool:
        """Test regression detection logic with known scenarios."""
        test_name = "Regression Detection Logic"

        try:
            detector = PerformanceRegressionDetector(
                storage_dir=self.test_storage_dir,
                warning_threshold=0.20,
                critical_threshold=0.50,
                min_samples=5
            )

            # Create baseline (50ms operations)
            baseline_measurements = [0.048, 0.050, 0.052, 0.049, 0.051] * 3
            detector.create_baseline(
                operation_name="test_regression",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                measurements=baseline_measurements,
                version="regression_baseline"
            )

            # Test scenarios
            scenarios = [
                {
                    "name": "stable_performance",
                    "measurements": [0.049, 0.051, 0.050, 0.052, 0.048],
                    "expected_severity": RegressionSeverity.NONE
                },
                {
                    "name": "warning_regression",
                    "measurements": [0.060, 0.062, 0.061, 0.063, 0.059],  # ~20% slower
                    "expected_severity": RegressionSeverity.WARNING
                },
                {
                    "name": "critical_regression",
                    "measurements": [0.075, 0.078, 0.076, 0.080, 0.074],  # ~50% slower
                    "expected_severity": RegressionSeverity.CRITICAL
                }
            ]

            results = {}
            for scenario in scenarios:
                detector.set_current_version(f"test_{scenario['name']}")
                detector.clear_session_metrics()

                # Record measurements
                for measurement in scenario["measurements"]:
                    detector.record_metric(
                        operation_name="test_regression",
                        metric_type=PerformanceMetricType.EXECUTION_TIME,
                        value=measurement
                    )

                # Detect regressions
                regression_results = detector.detect_regressions("regression_baseline")

                if regression_results:
                    result = regression_results[0]
                    actual_severity = result.severity
                    expected_severity = scenario["expected_severity"]

                    results[scenario["name"]] = {
                        "expected": expected_severity.value,
                        "actual": actual_severity.value,
                        "correct": actual_severity == expected_severity,
                        "percentage_change": result.percentage_change * 100
                    }
                else:
                    results[scenario["name"]] = {
                        "expected": scenario["expected_severity"].value,
                        "actual": "no_result",
                        "correct": False,
                        "percentage_change": 0
                    }

            # Validate all scenarios
            all_correct = all(r["correct"] for r in results.values())

            self.add_result(test_name, all_correct,
                          "Regression detection scenarios validated" if all_correct else "Some scenarios failed",
                          {"scenario_results": results})
            return all_correct

        except Exception as e:
            self.add_result(test_name, False, f"Regression detection failed: {e}")
            return False

    def test_baseline_creator_integration(self) -> bool:
        """Test baseline creator with mocked measurements."""
        test_name = "Baseline Creator Integration"

        try:
            # Create a lightweight baseline creator for testing
            class MockBaselineCreator(PerformanceBaselineCreator):
                def measure_geometric_operations(self):
                    return {
                        "rotate": [0.0025] * 10,
                        "mirror": [0.0020] * 10,
                        "flip": [0.0021] * 10
                    }

                def measure_color_operations(self):
                    return {
                        "map": [0.0055] * 10,
                        "filter": [0.0048] * 10
                    }

                def measure_pattern_operations(self):
                    return {
                        "detect_small": [0.0145] * 8
                    }

                def measure_composition_operations(self):
                    return {
                        "composition_3ops": [0.0680] * 8
                    }

                def measure_system_operations(self):
                    return {
                        "transpilation": [0.0065] * 15,
                        "sandbox_startup": [0.0018] * 20
                    }

            creator = MockBaselineCreator(version="test_2.0", storage_dir=self.test_storage_dir)

            # Run baseline creation
            result = creator.run_baseline_creation()

            # Validate results
            success = result.get("success", False)
            baselines_created = result.get("baselines_created", 0)
            measurements_collected = result.get("measurements_collected", 0)

            # Check that we created baselines for expected operations
            expected_min_baselines = 8  # At least 8 different operations
            expected_min_measurements = 80  # At least 80 total measurements

            validation_passed = (
                success and
                baselines_created >= expected_min_baselines and
                measurements_collected >= expected_min_measurements
            )

            self.add_result(test_name, validation_passed,
                          f"Created {baselines_created} baselines from {measurements_collected} measurements",
                          {
                              "success": success,
                              "baselines_created": baselines_created,
                              "measurements_collected": measurements_collected,
                              "performance_summary": result.get("performance_summary", {})
                          })
            return validation_passed

        except Exception as e:
            self.add_result(test_name, False, f"Baseline creator integration failed: {e}")
            return False

    def test_performance_comparison_workflow(self) -> bool:
        """Test end-to-end performance comparison workflow."""
        test_name = "Performance Comparison Workflow"

        try:
            # Create baseline version data
            detector = PerformanceRegressionDetector(
                storage_dir=self.test_storage_dir,
                min_samples=5
            )

            # Create baseline (version 1.0)
            baseline_ops = {
                "rotate": [0.0025] * 8,
                "filter": [0.0048] * 8,
                "composition": [0.0680] * 8
            }

            for op_name, measurements in baseline_ops.items():
                detector.create_baseline(
                    operation_name=op_name,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    measurements=measurements,
                    version="comparison_v1.0"
                )

            # Create mock comparator that doesn't require actual measurements
            class MockComparator(PerformanceComparator):
                def collect_current_measurements(self):
                    # Simulate some regressions and improvements
                    return {
                        "rotate": [0.0030] * 8,    # Slightly slower (20% regression)
                        "filter": [0.0040] * 8,    # Faster (improvement)
                        "composition": [0.0900] * 8  # Much slower (32% regression)
                    }

            comparator = MockComparator(
                baseline_version="comparison_v1.0",
                current_version="comparison_v2.0",
                storage_dir=self.test_storage_dir
            )

            # Run comparison
            comparison_result = comparator.run_comparison()

            # Validate comparison results
            success = comparison_result.get("success", False)

            if success:
                summary = comparison_result["performance_summary"]
                metrics = summary["performance_metrics"]

                # Check that we detected the expected regressions
                has_regressions = metrics["warning_regressions"] > 0 or metrics["critical_regressions"] > 0
                has_improvements = metrics["improved_operations"] > 0

                validation_passed = has_regressions and has_improvements

                self.add_result(test_name, validation_passed,
                              f"Comparison detected {metrics['warning_regressions']} warnings, "
                              f"{metrics['critical_regressions']} critical, "
                              f"{metrics['improved_operations']} improvements",
                              {
                                  "comparison_successful": success,
                                  "regressions_detected": has_regressions,
                                  "improvements_detected": has_improvements,
                                  "deployment_action": summary["deployment_recommendation"]["action"]
                              })
                return validation_passed
            else:
                self.add_result(test_name, False, f"Comparison failed: {comparison_result.get('error', 'Unknown')}")
                return False

        except Exception as e:
            self.add_result(test_name, False, f"Comparison workflow failed: {e}")
            return False

    def test_performance_target_validation(self) -> bool:
        """Test validation against performance targets."""
        test_name = "Performance Target Validation"

        try:
            creator = PerformanceBaselineCreator(version="target_test", storage_dir=self.test_storage_dir)

            # Test target compliance checking
            test_operations = [
                ("rotate", [0.003] * 10, 5.0, True),      # 3ms < 5ms target = PASS
                ("map", [0.012] * 10, 10.0, False),       # 12ms > 10ms target = FAIL
                ("detect_small", [0.018] * 10, 20.0, True), # 18ms < 20ms target = PASS
                ("fill", [0.035] * 10, 30.0, False),      # 35ms > 30ms target = FAIL
            ]

            target_results = {}

            for op_name, measurements, target_ms, should_pass in test_operations:
                # Create baseline
                baseline = creator.detector.create_baseline(
                    operation_name=op_name,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    measurements=measurements,
                    version="target_test"
                )

                # Check target compliance
                p95_ms = baseline.p95 * 1000 if baseline.p95 < 1 else baseline.p95
                meets_target = p95_ms <= target_ms

                target_results[op_name] = {
                    "p95_ms": p95_ms,
                    "target_ms": target_ms,
                    "meets_target": meets_target,
                    "expected_result": should_pass,
                    "correct_prediction": meets_target == should_pass
                }

            # Validate predictions
            all_correct = all(r["correct_prediction"] for r in target_results.values())

            self.add_result(test_name, all_correct,
                          "Target validation logic working correctly" if all_correct else "Target validation logic failed",
                          {"target_results": target_results})
            return all_correct

        except Exception as e:
            self.add_result(test_name, False, f"Target validation failed: {e}")
            return False

    def test_report_generation(self) -> bool:
        """Test performance report generation."""
        test_name = "Report Generation"

        try:
            detector = PerformanceRegressionDetector(
                storage_dir=self.test_storage_dir,
                min_samples=5
            )

            # Create test data
            detector.create_baseline(
                operation_name="report_test_op",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                measurements=[0.050] * 10,
                version="report_baseline"
            )

            detector.set_current_version("report_current")
            for measurement in [0.065] * 8:  # 30% slower (critical regression)
                detector.record_metric(
                    operation_name="report_test_op",
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=measurement
                )

            # Generate report
            report = detector.generate_report("report_baseline")

            # Validate report structure
            required_fields = [
                "baseline_version", "current_version", "total_operations_analyzed",
                "regressions_found", "critical_regressions", "regression_results",
                "summary_statistics", "recommendations"
            ]

            has_all_fields = all(hasattr(report, field) for field in required_fields)
            has_regression_detected = report.critical_regressions > 0
            has_recommendations = len(report.recommendations) > 0

            # Test report serialization
            report_dict = report.to_dict()
            has_serializable_format = isinstance(report_dict, dict) and "regression_results" in report_dict

            validation_passed = (
                has_all_fields and
                has_regression_detected and
                has_recommendations and
                has_serializable_format
            )

            self.add_result(test_name, validation_passed,
                          f"Report generated with {report.total_operations_analyzed} operations, "
                          f"{report.critical_regressions} critical regressions",
                          {
                              "has_all_fields": has_all_fields,
                              "regression_detected": has_regression_detected,
                              "has_recommendations": has_recommendations,
                              "serializable": has_serializable_format,
                              "operations_analyzed": report.total_operations_analyzed
                          })
            return validation_passed

        except Exception as e:
            self.add_result(test_name, False, f"Report generation failed: {e}")
            return False

    def run_all_tests(self) -> dict[str, Any]:
        """Run all test cases."""
        logger.info("Starting performance baseline system test suite...")
        start_time = time.time()

        # Run test cases
        test_methods = [
            self.test_regression_detector_basic,
            self.test_regression_detection_logic,
            self.test_baseline_creator_integration,
            self.test_performance_comparison_workflow,
            self.test_performance_target_validation,
            self.test_report_generation
        ]

        passed_tests = 0
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_method.__name__} crashed: {e}")

        execution_time = time.time() - start_time

        # Generate summary
        total_tests = len(test_methods)
        success_rate = (passed_tests / total_tests) * 100

        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "execution_time_seconds": execution_time,
                "overall_status": "PASS" if passed_tests == total_tests else "FAIL"
            },
            "test_results": self.results,
            "recommendations": []
        }

        # Add recommendations based on results
        if passed_tests < total_tests:
            failed_tests = [r["test_name"] for r in self.results if not r["success"]]
            summary["recommendations"].append(f"Fix failing tests: {', '.join(failed_tests)}")

        if success_rate >= 80:
            summary["recommendations"].append("Performance baseline system is largely functional")

        if success_rate == 100:
            summary["recommendations"].append("All tests passed - system ready for production use")

        return summary


def main():
    """Main function to run performance baseline tests."""
    print("PERFORMANCE BASELINE SYSTEM TEST SUITE")
    print("=" * 60)
    print("Testing comprehensive performance measurement and regression detection...")
    print()

    # Run test suite
    test_suite = PerformanceBaselineTestSuite()

    try:
        results = test_suite.run_all_tests()

        print("\n" + "=" * 60)
        print("TEST SUITE RESULTS")
        print("=" * 60)

        summary = results["test_summary"]

        print(f"Overall Status: {summary['overall_status']}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time_seconds']:.1f}s")
        print()

        # Show individual test results
        print("Individual Test Results:")
        for result in results["test_results"]:
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"  {status} {result['test_name']}")
            print(f"    {result['message']}")

        print()

        # Show recommendations
        if results["recommendations"]:
            print("Recommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. {rec}")

        # Return appropriate exit code
        return 0 if summary["overall_status"] == "PASS" else 1

    except Exception as e:
        print(f"✗ Test suite execution failed: {e}")
        return 1

    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    exit(main())
