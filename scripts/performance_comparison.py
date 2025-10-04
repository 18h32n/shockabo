"""
Performance Comparison and Regression Detection for Python Function Synthesis

This script compares current performance against established baselines and detects
regressions according to the specified thresholds:
- Warning: 20% over baseline
- Critical: 50% over baseline

It generates comprehensive comparison reports showing:
- P50, P95, P99 latency tracking
- Performance trend analysis
- Regression detection with recommendations
- Version-to-version comparisons
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_performance_baselines import PerformanceBaselineCreator
from src.utils.performance_regression_detector import (
    PerformanceMetricType,
    PerformanceRegressionDetector,
    PerformanceReport,
    RegressionSeverity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceComparator:
    """Comprehensive performance comparison and regression detection."""

    def __init__(self, baseline_version: str, current_version: str, storage_dir: str = "performance_baselines"):
        """
        Initialize performance comparator.
        
        Args:
            baseline_version: Version to use as baseline
            current_version: Current version being tested
            storage_dir: Directory containing baseline files
        """
        self.baseline_version = baseline_version
        self.current_version = current_version

        self.detector = PerformanceRegressionDetector(
            storage_dir=storage_dir,
            warning_threshold=0.20,  # 20% warning threshold
            critical_threshold=0.50,  # 50% critical threshold
            min_samples=5
        )
        self.detector.set_current_version(current_version)

        # Load available baseline versions
        self.available_versions = self.detector.storage.get_all_versions()
        logger.info(f"Available baseline versions: {self.available_versions}")

    def collect_current_measurements(self) -> dict[str, list[float]]:
        """Collect performance measurements for current version."""
        logger.info(f"Collecting performance measurements for version {self.current_version}...")

        # Use the baseline creator to collect fresh measurements
        creator = PerformanceBaselineCreator(version=self.current_version, storage_dir="temp_measurements")

        # Collect measurements for key operations
        measurements = {}

        try:
            # Focus on representative operations for comparison
            measurements.update(creator.measure_geometric_operations())
            measurements.update(creator.measure_color_operations())
            measurements.update(creator.measure_pattern_operations())
            measurements.update(creator.measure_composition_operations())
            measurements.update(creator.measure_system_operations())

            logger.info(f"Collected measurements for {len(measurements)} operation types")
            return measurements

        except Exception as e:
            logger.error(f"Failed to collect current measurements: {e}")
            return {}

    def record_measurements_to_detector(self, measurements: dict[str, list[float]]) -> None:
        """Record measurements to the regression detector."""
        logger.info("Recording measurements to regression detector...")

        for operation_name, values in measurements.items():
            for value in values:
                # Convert to seconds if needed (detector expects seconds for execution time)
                value_seconds = value / 1000 if value > 1 else value

                self.detector.record_metric(
                    operation_name=operation_name,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=value_seconds,
                    metadata={
                        "version": self.current_version,
                        "measurement_source": "performance_comparison"
                    }
                )

        logger.info(f"Recorded {sum(len(v) for v in measurements.values())} measurements to detector")

    def detect_regressions(self) -> list[Any]:
        """Detect performance regressions against baseline."""
        logger.info(f"Detecting regressions against baseline version {self.baseline_version}...")

        if self.baseline_version not in self.available_versions:
            logger.error(f"Baseline version {self.baseline_version} not found in available versions: {self.available_versions}")
            return []

        try:
            results = self.detector.detect_regressions(self.baseline_version)

            logger.info("Regression detection completed:")
            logger.info(f"  Total operations analyzed: {len(results)}")

            # Categorize results
            critical = [r for r in results if r.severity == RegressionSeverity.CRITICAL]
            warnings = [r for r in results if r.severity == RegressionSeverity.WARNING]
            stable = [r for r in results if r.severity == RegressionSeverity.NONE and r.percentage_change >= -0.1]
            improved = [r for r in results if r.severity == RegressionSeverity.NONE and r.percentage_change < -0.1]

            logger.info(f"  Critical regressions: {len(critical)}")
            logger.info(f"  Warning-level regressions: {len(warnings)}")
            logger.info(f"  Stable operations: {len(stable)}")
            logger.info(f"  Improved operations: {len(improved)}")

            return results

        except Exception as e:
            logger.error(f"Regression detection failed: {e}")
            return []

    def generate_detailed_report(self, regression_results: list[Any]) -> PerformanceReport:
        """Generate comprehensive performance comparison report."""
        logger.info("Generating detailed performance comparison report...")

        try:
            report = self.detector.generate_report(self.baseline_version)
            return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Return empty report structure
            from src.utils.performance_regression_detector import PerformanceReport
            return PerformanceReport(
                baseline_version=self.baseline_version,
                current_version=self.current_version,
                generation_time=time.time(),
                total_operations_analyzed=0,
                regressions_found=0,
                warnings_found=0,
                critical_regressions=0,
                operations_improved=0,
                regression_results=[],
                summary_statistics={},
                recommendations=["Error occurred during report generation"]
            )

    def create_comparison_summary(self, report: PerformanceReport) -> dict[str, Any]:
        """Create high-level comparison summary."""

        # Calculate key metrics
        total_ops = report.total_operations_analyzed
        regression_rate = (report.regressions_found / total_ops * 100) if total_ops > 0 else 0
        improvement_rate = (report.operations_improved / total_ops * 100) if total_ops > 0 else 0

        # Determine overall status
        if report.critical_regressions > 0:
            overall_status = "CRITICAL_REGRESSIONS"
            status_message = f"{report.critical_regressions} critical performance regressions detected"
        elif report.warnings_found > 0:
            overall_status = "PERFORMANCE_WARNINGS"
            status_message = f"{report.warnings_found} operations showing performance warnings"
        elif report.operations_improved > 0:
            overall_status = "PERFORMANCE_IMPROVED"
            status_message = f"{report.operations_improved} operations improved"
        else:
            overall_status = "STABLE_PERFORMANCE"
            status_message = "No significant performance changes detected"

        # Identify problematic operations
        critical_operations = [
            r.operation_name for r in report.regression_results
            if r.severity == RegressionSeverity.CRITICAL
        ]

        warning_operations = [
            r.operation_name for r in report.regression_results
            if r.severity == RegressionSeverity.WARNING
        ]

        improved_operations = [
            r.operation_name for r in report.regression_results
            if r.percentage_change < -0.1  # >10% improvement
        ]

        # Calculate worst regression
        worst_regression = 0
        worst_operation = None
        for result in report.regression_results:
            if result.percentage_change > worst_regression:
                worst_regression = result.percentage_change
                worst_operation = result.operation_name

        # Calculate best improvement
        best_improvement = 0
        best_operation = None
        for result in report.regression_results:
            if result.percentage_change < best_improvement:
                best_improvement = result.percentage_change
                best_operation = result.operation_name

        summary = {
            "comparison_summary": {
                "baseline_version": report.baseline_version,
                "current_version": report.current_version,
                "comparison_timestamp": datetime.fromtimestamp(report.generation_time).isoformat(),
                "overall_status": overall_status,
                "status_message": status_message
            },
            "performance_metrics": {
                "total_operations_analyzed": total_ops,
                "regression_rate_percent": round(regression_rate, 1),
                "improvement_rate_percent": round(improvement_rate, 1),
                "critical_regressions": report.critical_regressions,
                "warning_regressions": report.warnings_found,
                "improved_operations": report.operations_improved
            },
            "operation_breakdown": {
                "critical_operations": critical_operations,
                "warning_operations": warning_operations,
                "improved_operations": improved_operations
            },
            "performance_extremes": {
                "worst_regression": {
                    "operation": worst_operation,
                    "percentage": round(worst_regression * 100, 1) if worst_regression else 0
                },
                "best_improvement": {
                    "operation": best_operation,
                    "percentage": round(best_improvement * 100, 1) if best_improvement else 0
                }
            },
            "summary_statistics": report.summary_statistics,
            "recommendations": report.recommendations[:5],  # Top 5 recommendations
            "deployment_recommendation": self._get_deployment_recommendation(report)
        }

        return summary

    def _get_deployment_recommendation(self, report: PerformanceReport) -> dict[str, Any]:
        """Generate deployment recommendation based on regression analysis."""

        if report.critical_regressions > 0:
            return {
                "action": "BLOCK_DEPLOYMENT",
                "reason": f"{report.critical_regressions} critical performance regressions detected",
                "priority": "HIGH",
                "required_actions": [
                    "Investigate critical regressions immediately",
                    "Roll back changes or implement fixes",
                    "Re-run performance tests before deployment"
                ]
            }
        elif report.warnings_found > 3:  # Many warnings
            return {
                "action": "CONDITIONAL_DEPLOYMENT",
                "reason": f"{report.warnings_found} performance warnings detected",
                "priority": "MEDIUM",
                "required_actions": [
                    "Review warning-level regressions",
                    "Monitor performance closely post-deployment",
                    "Plan optimization work for next iteration"
                ]
            }
        elif report.operations_improved > report.warnings_found:
            return {
                "action": "APPROVE_DEPLOYMENT",
                "reason": f"Net performance improvement: {report.operations_improved} improved vs {report.warnings_found} warnings",
                "priority": "LOW",
                "required_actions": [
                    "Document performance improvements",
                    "Update performance baselines"
                ]
            }
        else:
            return {
                "action": "APPROVE_DEPLOYMENT",
                "reason": "No significant performance impact detected",
                "priority": "LOW",
                "required_actions": [
                    "Continue monitoring performance trends"
                ]
            }

    def save_comparison_results(self, summary: dict[str, Any], report: PerformanceReport) -> str:
        """Save comparison results to files."""

        # Create reports directory
        reports_dir = Path("performance_baselines") / "comparison_reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"comparison_{self.baseline_version}_vs_{self.current_version}_{timestamp}"

        # Save summary report
        summary_file = reports_dir / f"{base_filename}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed report
        detailed_file = reports_dir / f"{base_filename}_detailed.json"
        self.detector.save_report(report, str(detailed_file))

        logger.info("Comparison results saved:")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Detailed: {detailed_file}")

        return str(summary_file)

    def run_comparison(self) -> dict[str, Any]:
        """Run complete performance comparison process."""
        logger.info(f"Starting performance comparison: {self.baseline_version} vs {self.current_version}")

        start_time = time.time()

        try:
            # Step 1: Collect current measurements
            current_measurements = self.collect_current_measurements()
            if not current_measurements:
                return {
                    "success": False,
                    "error": "Failed to collect current measurements"
                }

            # Step 2: Record measurements to detector
            self.record_measurements_to_detector(current_measurements)

            # Step 3: Detect regressions
            regression_results = self.detect_regressions()
            if not regression_results:
                return {
                    "success": False,
                    "error": "Failed to detect regressions"
                }

            # Step 4: Generate detailed report
            detailed_report = self.generate_detailed_report(regression_results)

            # Step 5: Create comparison summary
            summary = self.create_comparison_summary(detailed_report)

            # Step 6: Save results
            summary_file = self.save_comparison_results(summary, detailed_report)

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "baseline_version": self.baseline_version,
                "current_version": self.current_version,
                "execution_time_seconds": execution_time,
                "summary_file": summary_file,
                "performance_summary": summary,
                "measurements_analyzed": sum(len(v) for v in current_measurements.values())
            }

            return result

        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "baseline_version": self.baseline_version,
                "current_version": self.current_version
            }


def main():
    """Main function for performance comparison."""
    parser = argparse.ArgumentParser(description="Performance comparison and regression detection")
    parser.add_argument("--baseline", "-b", required=True, help="Baseline version to compare against")
    parser.add_argument("--current", "-c", required=True, help="Current version being tested")
    parser.add_argument("--storage-dir", "-s", default="performance_baselines", help="Baseline storage directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("PERFORMANCE COMPARISON AND REGRESSION DETECTION")
    print("=" * 60)
    print(f"Baseline version: {args.baseline}")
    print(f"Current version:  {args.current}")
    print()

    # Run comparison
    comparator = PerformanceComparator(
        baseline_version=args.baseline,
        current_version=args.current,
        storage_dir=args.storage_dir
    )

    result = comparator.run_comparison()

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)

    if result["success"]:
        summary = result["performance_summary"]

        print("✓ Comparison completed successfully")
        print(f"  Execution time: {result['execution_time_seconds']:.1f}s")
        print(f"  Measurements analyzed: {result['measurements_analyzed']}")
        print()

        # Overall status
        status = summary["comparison_summary"]["overall_status"]
        message = summary["comparison_summary"]["status_message"]

        status_symbols = {
            "CRITICAL_REGRESSIONS": "🚨",
            "PERFORMANCE_WARNINGS": "⚠️",
            "PERFORMANCE_IMPROVED": "✅",
            "STABLE_PERFORMANCE": "🔄"
        }

        print(f"{status_symbols.get(status, '❓')} {message}")
        print()

        # Performance metrics
        metrics = summary["performance_metrics"]
        print("Performance Metrics:")
        print(f"  Operations analyzed: {metrics['total_operations_analyzed']}")
        print(f"  Regression rate: {metrics['regression_rate_percent']}%")
        print(f"  Improvement rate: {metrics['improvement_rate_percent']}%")
        print(f"  Critical regressions: {metrics['critical_regressions']}")
        print(f"  Warning regressions: {metrics['warning_regressions']}")
        print(f"  Improved operations: {metrics['improved_operations']}")
        print()

        # Deployment recommendation
        deployment = summary["deployment_recommendation"]
        action_symbols = {
            "BLOCK_DEPLOYMENT": "🛑",
            "CONDITIONAL_DEPLOYMENT": "⚠️",
            "APPROVE_DEPLOYMENT": "✅"
        }

        print("Deployment Recommendation:")
        print(f"  {action_symbols.get(deployment['action'], '❓')} {deployment['action']}")
        print(f"  Reason: {deployment['reason']}")
        print(f"  Priority: {deployment['priority']}")

        if deployment["required_actions"]:
            print("  Required actions:")
            for action in deployment["required_actions"]:
                print(f"    • {action}")
        print()

        # Key operations
        breakdown = summary["operation_breakdown"]
        if breakdown["critical_operations"]:
            print(f"Critical Operations: {', '.join(breakdown['critical_operations'])}")
        if breakdown["warning_operations"]:
            print(f"Warning Operations: {', '.join(breakdown['warning_operations'])}")
        if breakdown["improved_operations"]:
            print(f"Improved Operations: {', '.join(breakdown['improved_operations'])}")

        # Performance extremes
        extremes = summary["performance_extremes"]
        if extremes["worst_regression"]["operation"]:
            print(f"Worst Regression: {extremes['worst_regression']['operation']} ({extremes['worst_regression']['percentage']:+.1f}%)")
        if extremes["best_improvement"]["operation"]:
            print(f"Best Improvement: {extremes['best_improvement']['operation']} ({extremes['best_improvement']['percentage']:+.1f}%)")

        print()
        print(f"Detailed results: {result['summary_file']}")

        # Return appropriate exit code
        if status == "CRITICAL_REGRESSIONS":
            return 2  # Critical issues
        elif status == "PERFORMANCE_WARNINGS":
            return 1  # Warnings
        else:
            return 0  # Success

    else:
        print(f"✗ Comparison failed: {result['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
