"""
Performance Baseline Creation for Python Function Synthesis

This script creates comprehensive performance baseline measurements for the Python
Function Synthesis implementation. It measures the performance of all DSL operations
and creates baseline data according to the story specifications.

Performance Targets (95th percentile):
- Simple operations (rotate, mirror, translate): <5ms
- Color operations (map, filter): <10ms 
- Pattern detection (small patterns <5x5): <20ms
- Pattern detection (large patterns): <50ms
- Flood fill operations: <30ms
- Complex compositions (3-5 operations): <100ms

System-level targets:
- Transpilation time: <10ms per DSL program
- Function execution overhead: <2ms (sandbox startup)
- Memory overhead: <5MB base + grid size
- Batch evaluation: 100 programs in <5 seconds

Regression thresholds:
- Warning: 20% over target
- Critical: 50% over target
"""

import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxExecutor
from src.domain.dsl.base import DSLProgram
from src.utils.performance_regression_detector import (
    PerformanceBaseline,
    PerformanceMetricType,
    PerformanceRegressionDetector,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBaselineCreator:
    """Creates comprehensive performance baselines for DSL operations."""

    def __init__(self, version: str = "2.4.0", storage_dir: str = "performance_baselines"):
        """
        Initialize baseline creator.
        
        Args:
            version: Version identifier for baselines
            storage_dir: Directory to store baseline files
        """
        self.version = version
        self.detector = PerformanceRegressionDetector(
            storage_dir=storage_dir,
            warning_threshold=0.20,  # 20% warning threshold
            critical_threshold=0.50,  # 50% critical threshold
            min_samples=10
        )
        self.detector.set_current_version(version)

        self.transpiler = PythonTranspiler()
        self.executor = SandboxExecutor()

        # Performance targets from story (in milliseconds)
        self.performance_targets = {
            # Simple geometric operations: <5ms
            "rotate": 5.0,
            "mirror": 5.0,
            "flip": 5.0,
            "translate": 5.0,
            "crop": 5.0,
            "pad": 5.0,

            # Color operations: <10ms
            "map": 10.0,
            "filter": 10.0,
            "mask": 10.0,
            "replace": 10.0,
            "invert": 10.0,
            "threshold": 10.0,

            # Pattern detection (small): <20ms
            "detect_small": 20.0,
            "match_small": 20.0,

            # Pattern detection (large): <50ms
            "detect_large": 50.0,
            "match_large": 50.0,

            # Flood fill: <30ms
            "fill": 30.0,
            "flood": 30.0,

            # Complex compositions: <100ms
            "composition_3ops": 100.0,
            "composition_5ops": 100.0,

            # System operations: specialized targets
            "transpilation": 10.0,  # <10ms transpilation
            "sandbox_startup": 2.0,   # <2ms sandbox overhead
        }

        # Test grid configurations
        self.test_grids = {
            "small": (5, 5),
            "medium": (10, 10),
            "large": (20, 20),
            "max": (30, 30)
        }

    def create_test_grid(self, rows: int, cols: int, pattern: str = "random") -> list[list[int]]:
        """Create test grid with specified pattern."""
        if pattern == "random":
            return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
        elif pattern == "checkerboard":
            return [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
        elif pattern == "gradient":
            return [[min(9, (i + j) % 10) for j in range(cols)] for i in range(rows)]
        elif pattern == "border":
            grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if i == 0 or i == rows-1 or j == 0 or j == cols-1:
                        grid[i][j] = 1
            return grid
        elif pattern == "center":
            grid = [[0 for _ in range(cols)] for _ in range(rows)]
            cy, cx = rows // 2, cols // 2
            for i in range(max(0, cy-2), min(rows, cy+3)):
                for j in range(max(0, cx-2), min(cols, cx+3)):
                    grid[i][j] = 1
            return grid
        else:
            return [[0 for _ in range(cols)] for _ in range(rows)]

    def measure_geometric_operations(self) -> dict[str, list[float]]:
        """Measure performance of geometric operations."""
        logger.info("Measuring geometric operations performance...")
        measurements = {}

        # Test configurations for geometric operations
        geometric_ops = [
            ("rotate", {"operation": "rotate", "parameters": {"angle": 90}}),
            ("rotate", {"operation": "rotate", "parameters": {"angle": 180}}),
            ("rotate", {"operation": "rotate", "parameters": {"angle": 270}}),
            ("mirror", {"operation": "mirror", "parameters": {"direction": "horizontal"}}),
            ("mirror", {"operation": "mirror", "parameters": {"direction": "vertical"}}),
            ("flip", {"operation": "flip", "parameters": {"direction": "horizontal"}}),
            ("flip", {"operation": "flip", "parameters": {"direction": "vertical"}}),
            ("flip", {"operation": "flip", "parameters": {"direction": "diagonal"}}),
            ("translate", {"operation": "translate", "parameters": {"dx": 2, "dy": 2, "fill_color": 0}}),
            ("crop", {"operation": "crop", "parameters": {"x1": 1, "y1": 1, "x2": 8, "y2": 8}}),
            ("pad", {"operation": "pad", "parameters": {"top": 1, "bottom": 1, "left": 1, "right": 1, "value": 0}})
        ]

        for op_name, op_def in geometric_ops:
            logger.info(f"  Measuring {op_name}...")
            op_measurements = []

            # Test on different grid sizes
            for grid_name, (rows, cols) in self.test_grids.items():
                test_grid = self.create_test_grid(rows, cols, "random")

                # Multiple measurements for statistical significance
                for _ in range(15):
                    try:
                        # Create DSL program
                        program = DSLProgram(operations=[op_def])

                        # Measure transpilation + execution
                        start_time = time.perf_counter()

                        # Transpile
                        transpile_result = self.transpiler.transpile(program)
                        if not transpile_result.success:
                            continue

                        # Execute
                        exec_result = self.executor.execute_function(
                            transpile_result.source_code,
                            transpile_result.function_name,
                            test_grid,
                            timeout_seconds=1.0
                        )

                        end_time = time.perf_counter()

                        if exec_result.success:
                            execution_time_ms = (end_time - start_time) * 1000
                            op_measurements.append(execution_time_ms)

                    except Exception as e:
                        logger.warning(f"Failed measurement for {op_name}: {e}")
                        continue

            if op_measurements:
                measurements[op_name] = op_measurements
                logger.info(f"    Collected {len(op_measurements)} measurements, "
                           f"mean: {np.mean(op_measurements):.2f}ms, "
                           f"p95: {np.percentile(op_measurements, 95):.2f}ms")

        return measurements

    def measure_color_operations(self) -> dict[str, list[float]]:
        """Measure performance of color operations."""
        logger.info("Measuring color operations performance...")
        measurements = {}

        color_ops = [
            ("map", {"operation": "map", "parameters": {"mapping": {0: 1, 1: 2, 2: 3}}}),
            ("filter", {"operation": "filter", "parameters": {"colors": [1, 2, 3]}}),
            ("replace", {"operation": "replace", "parameters": {"old_color": 1, "new_color": 8}}),
            ("invert", {"operation": "invert", "parameters": {}}),
            ("threshold", {"operation": "threshold", "parameters": {"threshold": 5, "high_color": 9, "low_color": 0}})
        ]

        for op_name, op_def in color_ops:
            logger.info(f"  Measuring {op_name}...")
            op_measurements = []

            for grid_name, (rows, cols) in self.test_grids.items():
                test_grid = self.create_test_grid(rows, cols, "gradient")

                for _ in range(15):
                    try:
                        program = DSLProgram(operations=[op_def])

                        start_time = time.perf_counter()

                        transpile_result = self.transpiler.transpile(program)
                        if not transpile_result.success:
                            continue

                        exec_result = self.executor.execute_function(
                            transpile_result.source_code,
                            transpile_result.function_name,
                            test_grid,
                            timeout_seconds=1.0
                        )

                        end_time = time.perf_counter()

                        if exec_result.success:
                            execution_time_ms = (end_time - start_time) * 1000
                            op_measurements.append(execution_time_ms)

                    except Exception:
                        continue

            if op_measurements:
                measurements[op_name] = op_measurements
                logger.info(f"    Collected {len(op_measurements)} measurements, "
                           f"mean: {np.mean(op_measurements):.2f}ms, "
                           f"p95: {np.percentile(op_measurements, 95):.2f}ms")

        return measurements

    def measure_pattern_operations(self) -> dict[str, list[float]]:
        """Measure performance of pattern operations."""
        logger.info("Measuring pattern operations performance...")
        measurements = {}

        # Small pattern (3x3)
        small_pattern = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

        # Large pattern (7x7)
        large_pattern = [[1 if (i+j) % 2 == 0 else 0 for j in range(7)] for i in range(7)]

        pattern_ops = [
            ("detect_small", {"operation": "detect", "parameters": {"pattern": small_pattern}}),
            ("detect_large", {"operation": "detect", "parameters": {"pattern": large_pattern}}),
            ("fill", {"operation": "fill", "parameters": {"start_x": 5, "start_y": 5, "new_color": 9}}),
        ]

        for op_name, op_def in pattern_ops:
            logger.info(f"  Measuring {op_name}...")
            op_measurements = []

            # Use appropriate grid sizes for pattern operations
            test_grids = self.test_grids if "large" not in op_name else {"medium": (15, 15), "large": (25, 25)}

            for grid_name, (rows, cols) in test_grids.items():
                test_grid = self.create_test_grid(rows, cols, "center")

                for _ in range(10):  # Fewer iterations for pattern ops (they're slower)
                    try:
                        program = DSLProgram(operations=[op_def])

                        start_time = time.perf_counter()

                        transpile_result = self.transpiler.transpile(program)
                        if not transpile_result.success:
                            continue

                        exec_result = self.executor.execute_function(
                            transpile_result.source_code,
                            transpile_result.function_name,
                            test_grid,
                            timeout_seconds=2.0  # Longer timeout for pattern ops
                        )

                        end_time = time.perf_counter()

                        if exec_result.success:
                            execution_time_ms = (end_time - start_time) * 1000
                            op_measurements.append(execution_time_ms)

                    except Exception:
                        continue

            if op_measurements:
                measurements[op_name] = op_measurements
                logger.info(f"    Collected {len(op_measurements)} measurements, "
                           f"mean: {np.mean(op_measurements):.2f}ms, "
                           f"p95: {np.percentile(op_measurements, 95):.2f}ms")

        return measurements

    def measure_composition_operations(self) -> dict[str, list[float]]:
        """Measure performance of complex operation compositions."""
        logger.info("Measuring composition operations performance...")
        measurements = {}

        # 3-operation composition
        comp_3ops = [
            {"operation": "rotate", "parameters": {"angle": 90}},
            {"operation": "mirror", "parameters": {"direction": "horizontal"}},
            {"operation": "replace", "parameters": {"old_color": 1, "new_color": 8}}
        ]

        # 5-operation composition
        comp_5ops = [
            {"operation": "rotate", "parameters": {"angle": 90}},
            {"operation": "mirror", "parameters": {"direction": "horizontal"}},
            {"operation": "replace", "parameters": {"old_color": 1, "new_color": 8}},
            {"operation": "flip", "parameters": {"direction": "vertical"}},
            {"operation": "invert", "parameters": {}}
        ]

        compositions = [
            ("composition_3ops", comp_3ops),
            ("composition_5ops", comp_5ops)
        ]

        for comp_name, ops in compositions:
            logger.info(f"  Measuring {comp_name}...")
            op_measurements = []

            for grid_name, (rows, cols) in self.test_grids.items():
                test_grid = self.create_test_grid(rows, cols, "random")

                for _ in range(10):
                    try:
                        program = DSLProgram(operations=ops)

                        start_time = time.perf_counter()

                        transpile_result = self.transpiler.transpile(program)
                        if not transpile_result.success:
                            continue

                        exec_result = self.executor.execute_function(
                            transpile_result.source_code,
                            transpile_result.function_name,
                            test_grid,
                            timeout_seconds=2.0
                        )

                        end_time = time.perf_counter()

                        if exec_result.success:
                            execution_time_ms = (end_time - start_time) * 1000
                            op_measurements.append(execution_time_ms)

                    except Exception:
                        continue

            if op_measurements:
                measurements[comp_name] = op_measurements
                logger.info(f"    Collected {len(op_measurements)} measurements, "
                           f"mean: {np.mean(op_measurements):.2f}ms, "
                           f"p95: {np.percentile(op_measurements, 95):.2f}ms")

        return measurements

    def measure_system_operations(self) -> dict[str, list[float]]:
        """Measure system-level performance (transpilation, sandbox overhead)."""
        logger.info("Measuring system operations performance...")
        measurements = {}

        # Measure transpilation time separately
        transpilation_times = []
        logger.info("  Measuring transpilation time...")

        test_programs = [
            DSLProgram(operations=[{"operation": "rotate", "parameters": {"angle": 90}}]),
            DSLProgram(operations=[{"operation": "mirror", "parameters": {"direction": "horizontal"}}]),
            DSLProgram(operations=[
                {"operation": "rotate", "parameters": {"angle": 90}},
                {"operation": "replace", "parameters": {"old_color": 1, "new_color": 8}}
            ]),
            DSLProgram(operations=[
                {"operation": "flip", "parameters": {"direction": "horizontal"}},
                {"operation": "invert", "parameters": {}},
                {"operation": "rotate", "parameters": {"angle": 180}}
            ])
        ]

        for _ in range(20):
            for program in test_programs:
                try:
                    start_time = time.perf_counter()
                    result = self.transpiler.transpile(program)
                    end_time = time.perf_counter()

                    if result.success:
                        transpilation_time_ms = (end_time - start_time) * 1000
                        transpilation_times.append(transpilation_time_ms)

                except Exception:
                    continue

        if transpilation_times:
            measurements["transpilation"] = transpilation_times
            logger.info(f"    Collected {len(transpilation_times)} transpilation measurements, "
                       f"mean: {np.mean(transpilation_times):.2f}ms, "
                       f"p95: {np.percentile(transpilation_times, 95):.2f}ms")

        # Measure sandbox startup overhead
        sandbox_overhead_times = []
        logger.info("  Measuring sandbox startup overhead...")

        # Use simple operation to isolate sandbox overhead
        simple_program = DSLProgram(operations=[{"operation": "rotate", "parameters": {"angle": 90}}])
        transpile_result = self.transpiler.transpile(simple_program)

        if transpile_result.success:
            test_grid = self.create_test_grid(5, 5, "random")

            for _ in range(25):
                try:
                    start_time = time.perf_counter()

                    # This includes sandbox startup + minimal execution
                    exec_result = self.executor.execute_function(
                        transpile_result.source_code,
                        transpile_result.function_name,
                        test_grid,
                        timeout_seconds=1.0
                    )

                    end_time = time.perf_counter()

                    if exec_result.success:
                        # This is total time; overhead is the non-computation part
                        total_time_ms = (end_time - start_time) * 1000
                        sandbox_overhead_times.append(total_time_ms)

                except Exception:
                    continue

        if sandbox_overhead_times:
            measurements["sandbox_startup"] = sandbox_overhead_times
            logger.info(f"    Collected {len(sandbox_overhead_times)} sandbox overhead measurements, "
                       f"mean: {np.mean(sandbox_overhead_times):.2f}ms, "
                       f"p95: {np.percentile(sandbox_overhead_times, 95):.2f}ms")

        return measurements

    def create_baselines_from_measurements(self, all_measurements: dict[str, list[float]]) -> list[PerformanceBaseline]:
        """Create performance baselines from collected measurements."""
        logger.info("Creating performance baselines...")
        baselines = []

        for operation_name, measurements in all_measurements.items():
            if len(measurements) >= self.detector.min_samples:
                try:
                    baseline = self.detector.create_baseline(
                        operation_name=operation_name,
                        metric_type=PerformanceMetricType.EXECUTION_TIME,
                        measurements=measurements,
                        version=self.version
                    )
                    baselines.append(baseline)

                    # Check against performance targets
                    target = self.performance_targets.get(operation_name)
                    if target:
                        p95_ms = baseline.p95 * 1000 if baseline.p95 < 1 else baseline.p95
                        target_status = "✓ MEETS TARGET" if p95_ms <= target else "✗ EXCEEDS TARGET"
                        logger.info(f"  {operation_name}: p95={p95_ms:.2f}ms, target={target}ms - {target_status}")
                    else:
                        logger.info(f"  {operation_name}: p95={baseline.p95:.4f}s (no target defined)")

                except Exception as e:
                    logger.error(f"Failed to create baseline for {operation_name}: {e}")
            else:
                logger.warning(f"Insufficient measurements for {operation_name}: {len(measurements)} < {self.detector.min_samples}")

        return baselines

    def generate_baseline_report(self, baselines: list[PerformanceBaseline]) -> dict[str, Any]:
        """Generate comprehensive baseline report."""
        logger.info("Generating baseline report...")

        report = {
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "total_baselines": len(baselines),
            "performance_analysis": {},
            "target_compliance": {},
            "recommendations": []
        }

        # Analyze performance by category
        categories = {
            "geometric": ["rotate", "mirror", "flip", "translate", "crop", "pad"],
            "color": ["map", "filter", "replace", "invert", "threshold"],
            "pattern": ["detect_small", "detect_large", "fill"],
            "composition": ["composition_3ops", "composition_5ops"],
            "system": ["transpilation", "sandbox_startup"]
        }

        for category, operations in categories.items():
            category_baselines = [b for b in baselines if any(op in b.operation_name for op in operations)]

            if category_baselines:
                p95_times = [b.p95 for b in category_baselines]
                report["performance_analysis"][category] = {
                    "operation_count": len(category_baselines),
                    "mean_p95": np.mean(p95_times),
                    "median_p95": np.median(p95_times),
                    "max_p95": np.max(p95_times),
                    "operations": [b.operation_name for b in category_baselines]
                }

        # Check target compliance
        targets_met = 0
        targets_exceeded = 0

        for baseline in baselines:
            target = self.performance_targets.get(baseline.operation_name)
            if target:
                p95_ms = baseline.p95 * 1000 if baseline.p95 < 1 else baseline.p95

                if p95_ms <= target:
                    targets_met += 1
                    compliance = "met"
                else:
                    targets_exceeded += 1
                    compliance = "exceeded"

                report["target_compliance"][baseline.operation_name] = {
                    "target_ms": target,
                    "actual_p95_ms": p95_ms,
                    "compliance": compliance,
                    "deviation_percent": ((p95_ms - target) / target) * 100 if target > 0 else 0
                }

        # Generate recommendations
        if targets_exceeded > 0:
            report["recommendations"].append(f"PERFORMANCE CONCERN: {targets_exceeded} operations exceed performance targets")

            # Identify worst performers
            worst_performers = []
            for op, data in report["target_compliance"].items():
                if data["compliance"] == "exceeded" and data["deviation_percent"] > 50:
                    worst_performers.append(f"{op} ({data['deviation_percent']:.1f}% over target)")

            if worst_performers:
                report["recommendations"].append(f"Priority optimization targets: {', '.join(worst_performers)}")

        if targets_met > 0:
            report["recommendations"].append(f"GOOD: {targets_met} operations meet performance targets")

        # System-level recommendations
        transpilation_baseline = next((b for b in baselines if b.operation_name == "transpilation"), None)
        if transpilation_baseline:
            transpilation_p95_ms = transpilation_baseline.p95 * 1000 if transpilation_baseline.p95 < 1 else transpilation_baseline.p95
            if transpilation_p95_ms > 10:
                report["recommendations"].append(f"Consider transpilation caching: current {transpilation_p95_ms:.1f}ms > 10ms target")

        report["summary"] = {
            "targets_met": targets_met,
            "targets_exceeded": targets_exceeded,
            "overall_status": "MEETS TARGETS" if targets_exceeded == 0 else "NEEDS OPTIMIZATION"
        }

        return report

    def run_baseline_creation(self) -> dict[str, Any]:
        """Run complete baseline creation process."""
        logger.info(f"Starting performance baseline creation for version {self.version}")

        start_time = time.time()
        all_measurements = {}

        try:
            # Measure all operation categories
            all_measurements.update(self.measure_geometric_operations())
            all_measurements.update(self.measure_color_operations())
            all_measurements.update(self.measure_pattern_operations())
            all_measurements.update(self.measure_composition_operations())
            all_measurements.update(self.measure_system_operations())

            # Create baselines
            baselines = self.create_baselines_from_measurements(all_measurements)

            # Generate report
            report = self.generate_baseline_report(baselines)

            # Save detailed report
            report_dir = Path("performance_baselines") / "reports"
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / f"baseline_creation_report_{self.version}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            execution_time = time.time() - start_time

            summary = {
                "success": True,
                "version": self.version,
                "baselines_created": len(baselines),
                "measurements_collected": sum(len(m) for m in all_measurements.values()),
                "execution_time_seconds": execution_time,
                "report_file": str(report_file),
                "performance_summary": report["summary"],
                "recommendations": report["recommendations"]
            }

            logger.info("Baseline creation completed successfully:")
            logger.info(f"  Version: {self.version}")
            logger.info(f"  Baselines created: {len(baselines)}")
            logger.info(f"  Total measurements: {sum(len(m) for m in all_measurements.values())}")
            logger.info(f"  Execution time: {execution_time:.1f}s")
            logger.info(f"  Performance status: {report['summary']['overall_status']}")
            logger.info(f"  Report saved to: {report_file}")

            return summary

        except Exception as e:
            logger.error(f"Baseline creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "version": self.version,
                "measurements_collected": sum(len(m) for m in all_measurements.values()) if all_measurements else 0
            }


def main():
    """Main function to create performance baselines."""
    print("PERFORMANCE BASELINE CREATION FOR PYTHON FUNCTION SYNTHESIS")
    print("=" * 70)
    print("Creating comprehensive performance baselines for DSL operations...")
    print()

    # Create baseline for current version
    current_version = "2.4.0"
    creator = PerformanceBaselineCreator(version=current_version)

    # Run baseline creation
    result = creator.run_baseline_creation()

    print("\n" + "=" * 70)
    print("BASELINE CREATION SUMMARY")
    print("=" * 70)

    if result["success"]:
        print(f"✓ Successfully created performance baselines for version {result['version']}")
        print(f"  Baselines created: {result['baselines_created']}")
        print(f"  Measurements collected: {result['measurements_collected']}")
        print(f"  Execution time: {result['execution_time_seconds']:.1f}s")
        print(f"  Performance status: {result['performance_summary']['overall_status']}")
        print()

        if result["recommendations"]:
            print("Key Recommendations:")
            for rec in result["recommendations"][:3]:
                print(f"  • {rec}")
        print()

        print(f"Detailed report: {result['report_file']}")
        print("Baseline storage: performance_baselines/")
        print()
        print("NEXT STEPS:")
        print("1. Review the detailed performance report for optimization opportunities")
        print("2. Use these baselines for regression detection in future versions")
        print("3. Run performance comparisons using the regression detector")

    else:
        print(f"✗ Baseline creation failed: {result['error']}")
        if result.get("measurements_collected", 0) > 0:
            print(f"  Partial measurements collected: {result['measurements_collected']}")

        return 1

    return 0


if __name__ == "__main__":
    exit(main())
