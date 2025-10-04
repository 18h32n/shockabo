"""
Performance Regression Detection System Demonstration

This script demonstrates the capabilities of the performance regression detection
system integrated with DSL operations. It shows how to:

1. Create performance baselines
2. Detect performance regressions
3. Generate performance reports
4. Integrate with DSL engine profiling

Run this script to see the regression detection system in action.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.dsl.base import DSLProgram
from src.domain.dsl.color import ColorReplaceOperation
from src.domain.dsl.geometric import FlipOperation, RotateOperation, TranslateOperation
from src.domain.services.dsl_engine import DSLEngine
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


def create_test_grid(rows: int, cols: int, pattern: str = "random"):
    """Create a test grid with specified pattern."""
    import random

    if pattern == "random":
        return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
    elif pattern == "checkerboard":
        return [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    elif pattern == "gradient":
        return [[min(9, (i + j) % 10) for j in range(cols)] for i in range(rows)]
    else:
        return [[0 for _ in range(cols)] for _ in range(rows)]


def demonstrate_baseline_creation():
    """Demonstrate creating performance baselines."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Performance Baseline Creation")
    print("="*70)

    # Initialize regression detector
    detector = PerformanceRegressionDetector(
        storage_dir="demo_baselines",
        warning_threshold=0.20,
        critical_threshold=0.50,
        min_samples=5
    )
    detector.set_current_version("baseline_v1.0")

    # Test operations for baseline creation
    operations = [
        ("RotateOperation", RotateOperation(angle=90)),
        ("FlipOperation", FlipOperation(direction="horizontal")),
        ("ColorReplaceOperation", ColorReplaceOperation(source_color=1, target_color=8)),
        ("TranslateOperation", TranslateOperation(offset=(2, 2), fill_color=0))
    ]

    test_grid = create_test_grid(10, 10, "random")

    print("Creating performance baselines from operation measurements...")

    # Collect baseline measurements
    for op_name, operation in operations:
        print(f"  Measuring {op_name}...")
        measurements = []

        for i in range(10):
            start_time = time.time()
            result = operation.execute(test_grid)
            exec_time = time.time() - start_time

            if result.success:
                measurements.append(exec_time)

        # Create baseline
        if len(measurements) >= detector.min_samples:
            baseline = detector.create_baseline(
                operation_name=op_name,
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                measurements=measurements,
                version="baseline_v1.0"
            )

            print(f"    Baseline created: mean={baseline.mean:.4f}s, "
                  f"p95={baseline.p95:.4f}s, samples={baseline.sample_count}")
        else:
            print(f"    Warning: Insufficient measurements for {op_name}")

    print(f"\nBaselines stored in: {detector.storage.storage_dir}")
    return detector


def demonstrate_regression_detection(detector):
    """Demonstrate performance regression detection."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Performance Regression Detection")
    print("="*70)

    # Switch to new version for comparison
    detector.set_current_version("test_v2.0")

    print("Simulating different performance scenarios...")

    # Scenario 1: Stable performance (no regression)
    print("\n1. Stable Performance Scenario:")
    for _ in range(3):
        detector.record_metric(
            operation_name="RotateOperation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=0.008,  # Similar to baseline
            metadata={"scenario": "stable"}
        )

    # Scenario 2: Warning-level regression (25% slower)
    print("2. Warning-Level Regression Scenario:")
    for _ in range(3):
        detector.record_metric(
            operation_name="FlipOperation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=0.012,  # ~25% slower than typical baseline
            metadata={"scenario": "warning_regression"}
        )

    # Scenario 3: Critical regression (60% slower)
    print("3. Critical Regression Scenario:")
    for _ in range(3):
        detector.record_metric(
            operation_name="ColorReplaceOperation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=0.016,  # ~60% slower than typical baseline
            metadata={"scenario": "critical_regression"}
        )

    # Scenario 4: Performance improvement (40% faster)
    print("4. Performance Improvement Scenario:")
    for _ in range(3):
        detector.record_metric(
            operation_name="TranslateOperation",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=0.006,  # ~40% faster than typical baseline
            metadata={"scenario": "improvement"}
        )

    # Detect regressions
    print("\nDetecting regressions against baseline_v1.0...")
    results = detector.detect_regressions("baseline_v1.0")

    # Categorize and display results
    critical_regressions = [r for r in results if r.severity == RegressionSeverity.CRITICAL]
    warnings = [r for r in results if r.severity == RegressionSeverity.WARNING]
    stable_or_improved = [r for r in results if r.severity == RegressionSeverity.NONE]

    print("\nRegression Detection Results:")
    print(f"  Total operations analyzed: {len(results)}")
    print(f"  Critical regressions: {len(critical_regressions)}")
    print(f"  Warning-level regressions: {len(warnings)}")
    print(f"  Stable/improved operations: {len(stable_or_improved)}")

    # Display detailed results
    print("\nDetailed Results:")
    for result in results:
        status_icon = {
            RegressionSeverity.CRITICAL: "[CRITICAL]",
            RegressionSeverity.WARNING: "[WARNING]",
            RegressionSeverity.NONE: "[OK]" if result.percentage_change >= -0.1 else "[IMPROVED]"
        }[result.severity]

        print(f"  {status_icon} {result.operation_name}: {result.percentage_change*100:+.1f}%")
        print(f"    {result.message}")

        if result.recommendations:
            print(f"    Recommendations: {', '.join(result.recommendations[:2])}")
        print()

    return results


def demonstrate_dsl_engine_integration():
    """Demonstrate integration with DSL engine."""
    print("\n" + "="*70)
    print("DEMONSTRATION: DSL Engine Integration")
    print("="*70)

    # Create DSL engine with regression detection enabled
    engine = DSLEngine(
        timeout_seconds=2.0,
        enable_profiling=True,
        enable_regression_detection=True,
        version="engine_test_v1.0"
    )

    # Register operations
    engine.register_operation(RotateOperation)
    engine.register_operation(FlipOperation)
    engine.register_operation(ColorReplaceOperation)

    test_grid = create_test_grid(12, 12, "checkerboard")

    print("Running DSL programs with integrated regression detection...")

    # Define test programs
    test_programs = [
        DSLProgram(operations=[
            {"name": "RotateOperation", "parameters": {"angle": 90}},
            {"name": "FlipOperation", "parameters": {"direction": "horizontal"}}
        ]),
        DSLProgram(operations=[
            {"name": "ColorReplaceOperation", "parameters": {"source_color": 1, "target_color": 9}},
            {"name": "RotateOperation", "parameters": {"angle": 180}}
        ]),
        DSLProgram(operations=[
            {"name": "FlipOperation", "parameters": {"direction": "vertical"}},
            {"name": "ColorReplaceOperation", "parameters": {"source_color": 0, "target_color": 5}},
            {"name": "RotateOperation", "parameters": {"angle": 270}}
        ])
    ]

    # Execute programs and collect performance data
    for i, program in enumerate(test_programs, 1):
        print(f"\nExecuting Program {i} ({len(program.operations)} operations)...")

        result = engine.execute_program(program, test_grid)

        if result.success:
            print(f"  Execution time: {result.execution_time:.4f}s")
            print(f"  Peak memory: {result.metadata.get('peak_memory_mb', 0):.2f}MB")
            print(f"  Cache hits: {result.metadata.get('cache_hits', 0)}")
            print(f"  Performance target met: {result.metadata.get('performance_target_met', False)}")
        else:
            print(f"  Program failed: {result.error_message}")

    # Create baseline from engine profiling data
    print("\nCreating performance baseline from engine profiles...")
    baseline_result = engine.create_performance_baseline("engine_baseline_v1.0")

    print(f"  Baselines created: {baseline_result['baselines_created']}")
    print(f"  Operations profiled: {baseline_result['operations_profiled']}")
    print(f"  Total measurements: {baseline_result['total_measurements']}")

    # Run regression detection
    print("\nDetecting regressions using engine integration...")
    regression_result = engine.detect_performance_regressions("engine_baseline_v1.0")

    if "error" not in regression_result:
        print(f"  Operations analyzed: {regression_result['total_operations_analyzed']}")
        print(f"  Critical regressions: {regression_result['critical_regressions']}")
        print(f"  Warnings: {regression_result['warnings']}")
        print(f"  Improvements: {regression_result['improvements']}")

        if regression_result['critical_operations']:
            print(f"  Critical operations: {', '.join(regression_result['critical_operations'])}")
        if regression_result['warning_operations']:
            print(f"  Warning operations: {', '.join(regression_result['warning_operations'])}")
    else:
        print(f"  Error: {regression_result['error']}")

    return engine


def demonstrate_performance_reporting(detector, engine):
    """Demonstrate comprehensive performance reporting."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Performance Reporting")
    print("="*70)

    # Generate comprehensive report from detector
    print("Generating comprehensive performance report...")

    report = detector.generate_report("baseline_v1.0")
    output_dir = "demo_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Save detector report
    detector_report_path = os.path.join(output_dir, "detector_performance_report.json")
    detector.save_report(report, detector_report_path)

    print("\nDetector Report Summary:")
    print(f"  Baseline version: {report.baseline_version}")
    print(f"  Current version: {report.current_version}")
    print(f"  Operations analyzed: {report.total_operations_analyzed}")
    print(f"  Regressions found: {report.regressions_found}")
    print(f"  Critical regressions: {report.critical_regressions}")
    print(f"  Operations improved: {report.operations_improved}")

    if report.summary_statistics:
        print(f"  Regression rate: {report.summary_statistics.get('regression_rate', 0):.1f}%")
        print(f"  Average change: {report.summary_statistics.get('average_change_percent', 0):+.1f}%")

    print("\nTop Recommendations:")
    for i, recommendation in enumerate(report.recommendations[:3], 1):
        print(f"  {i}. {recommendation}")

    print(f"\nDetailed report saved to: {detector_report_path}")

    # Generate engine-integrated report
    print("\nGenerating engine-integrated performance report...")

    engine_report_path = os.path.join(output_dir, "engine_performance_report.json")
    engine_report_summary = engine.generate_performance_report(
        "engine_baseline_v1.0",
        engine_report_path
    )

    if "error" not in engine_report_summary:
        print("\nEngine Report Summary:")
        print(f"  Baseline version: {engine_report_summary['baseline_version']}")
        print(f"  Current version: {engine_report_summary['current_version']}")
        print(f"  Total operations: {engine_report_summary['total_operations_analyzed']}")
        print(f"  Regression rate: {engine_report_summary.get('regression_rate', '0%')}")
        print(f"  Report saved: {engine_report_summary['report_saved']}")

        if engine_report_summary.get('recommendations'):
            print("\nEngine Recommendations:")
            for i, rec in enumerate(engine_report_summary['recommendations'], 1):
                print(f"  {i}. {rec}")

    print(f"\nAll reports saved to: {output_dir}/")


def demonstrate_real_world_scenario():
    """Demonstrate a realistic regression detection scenario."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Real-World Scenario")
    print("="*70)

    print("Simulating a development cycle with performance changes...")

    # Initialize system for a realistic scenario
    detector = PerformanceRegressionDetector(
        storage_dir="real_world_baselines",
        warning_threshold=0.15,  # 15% threshold for stricter monitoring
        critical_threshold=0.30,  # 30% threshold for critical alerts
        min_samples=5
    )

    # Simulate baseline creation (version 1.0)
    print("\n1. Creating baseline from version 1.0...")
    detector.set_current_version("1.0.0")

    baseline_operations = {
        "RotateOperation": [0.008, 0.009, 0.007, 0.008, 0.009, 0.008, 0.007],
        "FlipOperation": [0.005, 0.006, 0.005, 0.006, 0.005, 0.005, 0.006],
        "ColorReplaceOperation": [0.010, 0.011, 0.009, 0.010, 0.011, 0.010, 0.009]
    }

    for op_name, measurements in baseline_operations.items():
        detector.create_baseline(
            operation_name=op_name,
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            measurements=measurements,
            version="1.0.0"
        )
        print(f"  Created baseline for {op_name}: mean={sum(measurements)/len(measurements):.4f}s")

    # Simulate development changes (version 1.1)
    print("\n2. Testing version 1.1 (minor optimizations)...")
    detector.set_current_version("1.1.0")

    # Slight improvements in some operations
    v11_measurements = {
        "RotateOperation": [0.007, 0.008, 0.007, 0.007, 0.008],  # Slightly faster
        "FlipOperation": [0.005, 0.005, 0.006, 0.005, 0.006],   # Stable
        "ColorReplaceOperation": [0.011, 0.012, 0.011, 0.012, 0.011]  # Slightly slower (warning?)
    }

    for op_name, measurements in v11_measurements.items():
        for measurement in measurements:
            detector.record_metric(op_name, PerformanceMetricType.EXECUTION_TIME, measurement)

    results_v11 = detector.detect_regressions("1.0.0")
    print(f"  Analysis: {len(results_v11)} operations checked")

    for result in results_v11:
        change_desc = "improvement" if result.percentage_change < 0 else "regression"
        print(f"    {result.operation_name}: {result.percentage_change*100:+.1f}% ({change_desc})")

    # Simulate problematic release (version 1.2)
    print("\n3. Testing version 1.2 (performance regression introduced)...")
    detector.set_current_version("1.2.0")

    # One operation has significant regression due to algorithm change
    v12_measurements = {
        "RotateOperation": [0.007, 0.008, 0.007, 0.008, 0.007],  # Still good
        "FlipOperation": [0.008, 0.009, 0.008, 0.009, 0.008],   # 50% slower (critical!)
        "ColorReplaceOperation": [0.009, 0.010, 0.009, 0.010, 0.009]  # Improved
    }

    for op_name, measurements in v12_measurements.items():
        for measurement in measurements:
            detector.record_metric(op_name, PerformanceMetricType.EXECUTION_TIME, measurement)

    results_v12 = detector.detect_regressions("1.0.0")
    print(f"  Analysis: {len(results_v12)} operations checked")

    critical_found = False
    for result in results_v12:
        severity_icon = {
            RegressionSeverity.CRITICAL: "[CRITICAL]",
            RegressionSeverity.WARNING: "[WARNING]",
            RegressionSeverity.NONE: "[OK]"
        }[result.severity]

        print(f"    {severity_icon} {result.operation_name}: {result.percentage_change*100:+.1f}%")

        if result.severity == RegressionSeverity.CRITICAL:
            critical_found = True
            print(f"      ACTION REQUIRED: {result.message}")

    # Generate final report
    print("\n4. Generating development cycle report...")
    final_report = detector.generate_report("1.0.0")

    print("\nDevelopment Cycle Summary:")
    print(f"  Baseline version: {final_report.baseline_version}")
    print(f"  Current version: {final_report.current_version}")
    print(f"  Critical regressions detected: {final_report.critical_regressions}")
    print(f"  Recommendation: {'BLOCK RELEASE' if critical_found else 'SAFE TO RELEASE'}")

    if final_report.recommendations:
        print("\nKey Recommendations:")
        for rec in final_report.recommendations[:2]:
            print(f"  • {rec}")


def main():
    """Run the complete performance regression detection demonstration."""
    print("PERFORMANCE REGRESSION DETECTION SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("This demonstration showcases the comprehensive performance regression")
    print("detection system for DSL operations, including:")
    print("• Baseline creation and management")
    print("• Automated regression detection with configurable thresholds")
    print("• Integration with DSL engine profiling")
    print("• Comprehensive performance reporting")
    print("• Real-world development cycle simulation")

    try:
        # Run demonstrations
        detector = demonstrate_baseline_creation()
        demonstrate_regression_detection(detector)
        engine = demonstrate_dsl_engine_integration()
        demonstrate_performance_reporting(detector, engine)
        demonstrate_real_world_scenario()

        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("The performance regression detection system is now fully")
        print("demonstrated and ready for integration into your development")
        print("workflow. Key features verified:")
        print()
        print("* Baseline storage and versioned management")
        print("* Statistical analysis with percentile tracking (p50, p95, p99)")
        print("* Configurable thresholds (20% warning, 50% critical)")
        print("* Integration with DSL engine profiling")
        print("* Comprehensive reporting and trend analysis")
        print("* Real-world development cycle support")
        print()
        print("Check the generated reports in:")
        print("• demo_reports/")
        print("• demo_baselines/")
        print("• real_world_baselines/")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
