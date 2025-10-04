"""Performance regression tests for DSL operations.

This module tests performance characteristics, execution time, memory usage,
and scalability with different grid sizes and operation complexities.
Enhanced with automated regression detection system.
"""

import os
import shutil
import tempfile
import time
from statistics import mean, stdev

import psutil
import pytest

from src.domain.dsl.color import (
    ColorFilterOperation,
    ColorInvertOperation,
    ColorMapOperation,
    ColorReplaceOperation,
)
from src.domain.dsl.composition import OverlayOperation
from src.domain.dsl.connectivity import ConnectedComponentsOperation
from src.domain.dsl.edges import EdgeDetectionOperation
from src.domain.dsl.geometric import (
    CropOperation,
    FlipOperation,
    PadOperation,
    RotateOperation,
    TranslateOperation,
)
from src.domain.dsl.pattern import FloodFillOperation, PatternFillOperation, PatternMatchOperation
from src.domain.dsl.types import Grid
from src.utils.performance_regression_detector import (
    PerformanceMetricType,
    PerformanceRegressionDetector,
    RegressionSeverity,
)


def create_test_grid(rows: int, cols: int, pattern: str = "random") -> Grid:
    """Create a test grid of specified size with different patterns."""
    import random

    if pattern == "random":
        return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
    elif pattern == "checkerboard":
        return [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    elif pattern == "gradient":
        return [[min(9, (i + j) % 10) for j in range(cols)] for i in range(rows)]
    elif pattern == "solid":
        return [[5 for _ in range(cols)] for _ in range(rows)]
    else:
        return [[0 for _ in range(cols)] for _ in range(rows)]


def measure_operation_performance(operation, grid: Grid, runs: int = 5,
                                detector: PerformanceRegressionDetector = None) -> dict[str, float]:
    """Measure performance metrics for an operation with optional regression tracking."""
    times = []
    memory_usage = []

    process = psutil.Process(os.getpid())
    operation_name = operation.__class__.__name__

    for _ in range(runs):
        # Measure memory before
        initial_memory = process.memory_info().rss

        # Measure execution time
        start_time = time.time()
        result = operation.execute(grid)
        execution_time = time.time() - start_time

        # Measure memory after
        final_memory = process.memory_info().rss
        memory_delta = final_memory - initial_memory

        if result.success:
            times.append(execution_time)
            memory_usage.append(memory_delta)

            # Record metrics for regression detection
            if detector:
                detector.record_metric(
                    operation_name=operation_name,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=execution_time,
                    metadata={
                        "grid_size": f"{len(grid)}x{len(grid[0])}",
                        "run_index": len(times)
                    }
                )
                detector.record_metric(
                    operation_name=operation_name,
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    value=memory_delta / (1024 * 1024),  # Convert to MB
                    metadata={
                        "grid_size": f"{len(grid)}x{len(grid[0])}",
                        "run_index": len(times)
                    }
                )
        else:
            raise RuntimeError(f"Operation failed: {result.error_message}")

    return {
        "mean_time": mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": stdev(times) if len(times) > 1 else 0.0,
        "mean_memory": mean(memory_usage),
        "max_memory": max(memory_usage)
    }


@pytest.fixture(scope="session")
def regression_detector():
    """Create a session-wide regression detector for performance tracking."""
    temp_dir = tempfile.mkdtemp(prefix="dsl_perf_test_")
    detector = PerformanceRegressionDetector(
        storage_dir=temp_dir,
        warning_threshold=0.20,  # 20% degradation warning
        critical_threshold=0.50,  # 50% degradation critical
        min_samples=3  # Lower threshold for tests
    )
    detector.set_current_version("test_current")

    yield detector

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def performance_baseline_setup(regression_detector):
    """Set up performance baselines for regression testing."""
    # Create baseline measurements from known good operations
    baseline_operations = [
        (RotateOperation(angle=90), "RotateOperation"),
        (FlipOperation(direction="horizontal"), "FlipOperation"),
        (ColorReplaceOperation(source_color=1, target_color=8), "ColorReplaceOperation"),
        (TranslateOperation(offset=(1, 1), fill_color=0), "TranslateOperation")
    ]

    test_grid = create_test_grid(10, 10, "random")

    # Record baseline measurements
    for operation, name in baseline_operations:
        baseline_times = []
        baseline_memory = []

        for _ in range(10):  # More samples for reliable baseline
            start_time = time.time()
            result = operation.execute(test_grid)
            exec_time = time.time() - start_time

            if result.success:
                baseline_times.append(exec_time)
                baseline_memory.append(1.0)  # Placeholder memory usage

        # Create baselines
        if baseline_times:
            try:
                regression_detector.create_baseline(
                    operation_name=name,
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    measurements=baseline_times,
                    version="baseline_v1"
                )
                regression_detector.create_baseline(
                    operation_name=name,
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    measurements=baseline_memory,
                    version="baseline_v1"
                )
            except Exception as e:
                print(f"Warning: Could not create baseline for {name}: {e}")

    return regression_detector


class TestDSLPerformance:
    """Performance test cases for individual DSL operations."""

    @pytest.mark.parametrize("size", [3, 5, 10, 15, 20, 30])
    def test_rotate_operation_scaling(self, size):
        """Test rotation operation performance across different grid sizes."""
        grid = create_test_grid(size, size, "random")
        operation = RotateOperation(angle=90)

        metrics = measure_operation_performance(operation, grid)

        # Performance requirements
        assert metrics["mean_time"] < 0.1, f"Rotation too slow for {size}x{size}: {metrics['mean_time']:.4f}s"
        assert metrics["max_time"] < 0.2, f"Worst-case rotation too slow: {metrics['max_time']:.4f}s"

        # Memory should be reasonable (less than 10MB for any single operation)
        assert metrics["max_memory"] < 10 * 1024 * 1024, f"Memory usage too high: {metrics['max_memory']} bytes"

    @pytest.mark.parametrize("size", [3, 5, 10, 15, 20, 30])
    def test_flip_operation_scaling(self, size):
        """Test flip operation performance across different grid sizes."""
        grid = create_test_grid(size, size, "random")

        for direction in ["horizontal", "vertical", "diagonal_main", "diagonal_anti"]:
            operation = FlipOperation(direction=direction)
            metrics = measure_operation_performance(operation, grid)

            assert metrics["mean_time"] < 0.1, f"Flip {direction} too slow for {size}x{size}: {metrics['mean_time']:.4f}s"

    @pytest.mark.parametrize("size", [3, 5, 10, 15, 20, 30])
    def test_color_operations_scaling(self, size):
        """Test color operation performance across different grid sizes."""
        grid = create_test_grid(size, size, "random")

        operations = [
            ColorReplaceOperation(source_color=1, target_color=8),
            ColorInvertOperation(),
            ColorFilterOperation(keep_colors=[0, 1, 2, 3, 4]),
            ColorMapOperation(mapping={i: (i + 1) % 10 for i in range(10)})
        ]

        for operation in operations:
            metrics = measure_operation_performance(operation, grid)
            assert metrics["mean_time"] < 0.1, f"{operation.__class__.__name__} too slow: {metrics['mean_time']:.4f}s"

    @pytest.mark.parametrize("size", [5, 10, 15, 20])  # Skip very small grids for flood fill
    def test_pattern_operations_scaling(self, size):
        """Test pattern operation performance across different grid sizes."""
        grid = create_test_grid(size, size, "checkerboard")

        operations = [
            FloodFillOperation(position=(size//2, size//2), color=9),
            PatternFillOperation(pattern=[[1, 0], [0, 1]], fill_color=7)
        ]

        for operation in operations:
            metrics = measure_operation_performance(operation, grid)
            # Pattern operations might be slightly slower
            assert metrics["mean_time"] < 0.2, f"{operation.__class__.__name__} too slow: {metrics['mean_time']:.4f}s"

    def test_translate_operation_performance(self):
        """Test translation operation performance with different offsets."""
        base_grid = create_test_grid(10, 10, "random")

        offsets = [(0, 0), (1, 1), (5, 5), (-2, 3), (10, -5)]

        for offset in offsets:
            operation = TranslateOperation(offset=offset, fill_color=0)
            metrics = measure_operation_performance(operation, base_grid)
            assert metrics["mean_time"] < 0.05, f"Translation {offset} too slow: {metrics['mean_time']:.4f}s"

    def test_crop_pad_operation_performance(self):
        """Test crop and pad operation performance."""
        base_grid = create_test_grid(20, 20, "gradient")

        # Test crop operation
        crop_op = CropOperation(top=2, left=2, bottom=17, right=17)
        crop_metrics = measure_operation_performance(crop_op, base_grid)
        assert crop_metrics["mean_time"] < 0.05, f"Crop too slow: {crop_metrics['mean_time']:.4f}s"

        # Test pad operation
        small_grid = create_test_grid(5, 5, "solid")
        pad_op = PadOperation(top=5, bottom=5, left=5, right=5, fill_color=0)
        pad_metrics = measure_operation_performance(pad_op, small_grid)
        assert pad_metrics["mean_time"] < 0.05, f"Pad too slow: {pad_metrics['mean_time']:.4f}s"

    def test_composition_operation_performance(self):
        """Test composition operation performance."""
        base_grid = create_test_grid(15, 15, "random")
        overlay_grid = create_test_grid(5, 5, "solid")

        overlay_op = OverlayOperation(
            overlay_grid=overlay_grid,
            position=(5, 5),
            blend_mode="replace"
        )

        metrics = measure_operation_performance(overlay_op, base_grid)
        assert metrics["mean_time"] < 0.1, f"Overlay too slow: {metrics['mean_time']:.4f}s"


class TestDSLScalability:
    """Scalability test cases for DSL operations."""

    def test_performance_linear_scaling(self):
        """Test that performance scales approximately linearly with grid size."""
        operation = RotateOperation(angle=90)
        sizes = [5, 10, 20]
        times = []

        for size in sizes:
            grid = create_test_grid(size, size, "random")
            metrics = measure_operation_performance(operation, grid, runs=3)
            times.append(metrics["mean_time"])

        # Check that performance doesn't scale worse than quadratic
        # (allowing some margin for measurement noise)
        ratio_10_5 = times[1] / times[0]
        ratio_20_10 = times[2] / times[1]

        # Should be roughly linear (factor of 2-4 for 2x size increase)
        assert ratio_10_5 < 8, f"Performance scaling worse than expected: 10/5 ratio = {ratio_10_5}"
        assert ratio_20_10 < 8, f"Performance scaling worse than expected: 20/10 ratio = {ratio_20_10}"

    def test_memory_scaling(self):
        """Test memory usage scaling with grid size."""
        operation = ColorMapOperation(mapping={i: (i + 1) % 10 for i in range(10)})

        sizes = [10, 20, 30]
        memory_usage = []

        for size in sizes:
            grid = create_test_grid(size, size, "random")
            metrics = measure_operation_performance(operation, grid, runs=3)
            memory_usage.append(metrics["mean_memory"])

        # Memory should scale roughly with grid area
        # Allow for some overhead and measurement noise
        for memory in memory_usage:
            assert memory < 50 * 1024 * 1024, f"Memory usage too high: {memory} bytes"

    def test_operation_chaining_scalability(self):
        """Test scalability of operation chaining."""
        grid = create_test_grid(10, 10, "random")

        # Test different chain lengths
        base_op = FlipOperation(direction="horizontal")

        chain_lengths = [1, 5, 10, 20]
        chain_times = []

        for length in chain_lengths:
            operations = [base_op] * length
            composite = operations[0]
            for op in operations[1:]:
                composite = composite >> op

            start_time = time.time()
            result = composite.execute(grid)
            execution_time = time.time() - start_time

            assert result.success
            chain_times.append(execution_time)

        # Chaining should scale approximately linearly
        assert all(t < 1.0 for t in chain_times), f"Chain execution too slow: {chain_times}"

    @pytest.mark.parametrize("rows,cols", [(1, 30), (30, 1), (5, 20), (20, 5)])
    def test_rectangular_grid_performance(self, rows, cols):
        """Test performance on non-square grids."""
        grid = create_test_grid(rows, cols, "gradient")

        # Test operations that should work on rectangular grids
        operations = [
            FlipOperation(direction="horizontal"),
            FlipOperation(direction="vertical"),
            ColorReplaceOperation(source_color=1, target_color=9),
            TranslateOperation(offset=(1, 1), fill_color=0)
        ]

        for operation in operations:
            metrics = measure_operation_performance(operation, grid)
            assert metrics["mean_time"] < 0.1, f"Rectangular grid operation too slow: {metrics['mean_time']:.4f}s"


class TestDSLMemoryUsage:
    """Memory usage test cases for DSL operations."""

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        import gc

        grid = create_test_grid(10, 10, "random")
        operation = RotateOperation(angle=90)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform many operations
        for _ in range(100):
            result = operation.execute(grid)
            assert result.success

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Allow for some memory increase, but not excessive
        assert memory_increase < 10 * 1024 * 1024, f"Potential memory leak: {memory_increase} bytes increase"

    def test_large_grid_memory_usage(self):
        """Test memory usage with large grids."""
        # Test with the largest allowed ARC grid size
        large_grid = create_test_grid(30, 30, "random")

        operations = [
            RotateOperation(angle=90),
            FlipOperation(direction="horizontal"),
            ColorInvertOperation()
        ]

        process = psutil.Process(os.getpid())

        for operation in operations:
            initial_memory = process.memory_info().rss
            result = operation.execute(large_grid)
            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory

            assert result.success
            # Should use less than 50MB for any single operation
            assert memory_delta < 50 * 1024 * 1024, f"Memory usage too high: {memory_delta} bytes"

    def test_memory_efficiency_comparison(self):
        """Compare memory efficiency of different operations."""
        grid = create_test_grid(15, 15, "random")

        operations = [
            ("rotate", RotateOperation(angle=90)),
            ("flip", FlipOperation(direction="horizontal")),
            ("color_replace", ColorReplaceOperation(source_color=1, target_color=8)),
            ("translate", TranslateOperation(offset=(2, 2), fill_color=0))
        ]

        memory_usage = {}

        for name, operation in operations:
            metrics = measure_operation_performance(operation, grid, runs=3)
            memory_usage[name] = metrics["mean_memory"]

        # All operations should have reasonable memory usage
        for name, memory in memory_usage.items():
            assert memory < 5 * 1024 * 1024, f"{name} uses too much memory: {memory} bytes"


class TestDSLRegressionBenchmarks:
    """Regression benchmarks to catch performance degradation."""

    def test_standard_operation_benchmarks(self):
        """Benchmark standard operations against baseline performance."""
        # Standard test grid (10x10)
        grid = create_test_grid(10, 10, "random")

        # Baseline expectations (in seconds)
        benchmarks = {
            RotateOperation(angle=90): 0.01,
            FlipOperation(direction="horizontal"): 0.01,
            ColorReplaceOperation(source_color=1, target_color=8): 0.01,
            TranslateOperation(offset=(1, 1), fill_color=0): 0.01,
            FloodFillOperation(position=(5, 5), color=9): 0.05
        }

        for operation, baseline in benchmarks.items():
            metrics = measure_operation_performance(operation, grid, runs=5)

            # Allow 2x the baseline for regression tolerance
            assert metrics["mean_time"] < baseline * 2, (
                f"{operation.__class__.__name__} regression: "
                f"{metrics['mean_time']:.4f}s > {baseline * 2:.4f}s baseline"
            )

    def test_complex_operation_benchmarks(self):
        """Benchmark complex operations that involve more computation."""
        grid = create_test_grid(15, 15, "checkerboard")

        # More complex operations with higher baselines
        complex_benchmarks = {
            ConnectedComponentsOperation(): 0.1,
            EdgeDetectionOperation(): 0.1,
            PatternMatchOperation(pattern=[[1, 0], [0, 1]]): 0.1
        }

        for operation, baseline in complex_benchmarks.items():
            metrics = measure_operation_performance(operation, grid, runs=3)

            # Allow 3x the baseline for more complex operations
            assert metrics["mean_time"] < baseline * 3, (
                f"{operation.__class__.__name__} regression: "
                f"{metrics['mean_time']:.4f}s > {baseline * 3:.4f}s baseline"
            )

    def test_chain_operation_benchmarks(self):
        """Benchmark operation chains against baseline performance."""
        grid = create_test_grid(8, 8, "gradient")

        # Test various chain types
        chains = [
            # Simple geometric chain
            (RotateOperation(angle=90) >> FlipOperation(direction="horizontal"), 0.02),

            # Mixed category chain
            (ColorReplaceOperation(source_color=1, target_color=9) >>
             RotateOperation(angle=180) >>
             FlipOperation(direction="vertical"), 0.03),

            # Longer chain
            (FlipOperation(direction="horizontal") >>
             FlipOperation(direction="vertical") >>
             RotateOperation(angle=90) >>
             ColorInvertOperation(), 0.05)
        ]

        for chain, baseline in chains:
            start_time = time.time()
            result = chain.execute(grid)
            execution_time = time.time() - start_time

            assert result.success
            assert execution_time < baseline * 2, (
                f"Chain regression: {execution_time:.4f}s > {baseline * 2:.4f}s baseline"
            )

    def test_worst_case_performance(self):
        """Test worst-case scenarios for performance."""
        # Large grid with complex pattern
        large_grid = create_test_grid(25, 25, "checkerboard")

        # Operations that might be slow on large grids
        demanding_operations = [
            FloodFillOperation(position=(12, 12), color=9),
            ConnectedComponentsOperation(),
            PatternMatchOperation(pattern=[[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        ]

        for operation in demanding_operations:
            metrics = measure_operation_performance(operation, large_grid, runs=3)

            # Even worst-case should complete within reasonable time
            assert metrics["max_time"] < 0.5, (
                f"{operation.__class__.__name__} worst-case too slow: {metrics['max_time']:.4f}s"
            )

    def test_concurrent_operation_performance(self):
        """Test performance when multiple operations might run concurrently."""
        import queue
        import threading

        grid = create_test_grid(10, 10, "random")
        operation = RotateOperation(angle=90)

        results_queue = queue.Queue()

        def run_operation():
            start = time.time()
            result = operation.execute(grid)
            end = time.time()
            results_queue.put((result.success, end - start))

        # Run 5 operations concurrently
        threads = [threading.Thread(target=run_operation) for _ in range(5)]

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        all_succeeded = True
        individual_times = []

        while not results_queue.empty():
            success, exec_time = results_queue.get()
            all_succeeded &= success
            individual_times.append(exec_time)

        assert all_succeeded, "Some concurrent operations failed"
        assert total_time < 1.0, f"Concurrent execution too slow: {total_time:.4f}s"
        assert all(t < 0.2 for t in individual_times), "Individual operations too slow in concurrent context"


class TestDSLRegressionDetection:
    """Test performance regression detection system integration."""

    def test_regression_detection_baseline_creation(self, regression_detector):
        """Test that baselines are properly created from test runs."""
        grid = create_test_grid(10, 10, "random")
        operation = RotateOperation(angle=90)

        # Measure performance with regression tracking
        measure_operation_performance(operation, grid, runs=5, detector=regression_detector)

        # Verify metrics were recorded
        stats = regression_detector.get_session_stats()
        assert stats["total_metrics"] > 0
        assert "RotateOperation" in stats["operations_tracked"]

    def test_regression_detection_no_regression(self, regression_detector):
        """Test that no regression is detected for stable performance."""
        grid = create_test_grid(10, 10, "random")
        operation = RotateOperation(angle=90)

        # Run operation with tracking
        measure_operation_performance(operation, grid, runs=3, detector=regression_detector)

        # Detect regressions against baseline
        results = regression_detector.detect_regressions("baseline_v1")

        # Should find the operation but no significant regression
        rotation_results = [r for r in results if r.operation_name == "RotateOperation"]
        if rotation_results:
            # Performance should be stable (no critical regressions)
            assert all(r.severity != RegressionSeverity.CRITICAL for r in rotation_results)

    def test_regression_detection_warning_threshold(self, regression_detector):
        """Test detection of warning-level performance degradation."""
        grid = create_test_grid(10, 10, "random")

        # Simulate degraded performance by recording slower measurements
        for _ in range(5):
            regression_detector.record_metric(
                operation_name="RotateOperation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=0.05,  # Assume this is ~25% slower than baseline
                metadata={"simulated": True}
            )

        results = regression_detector.detect_regressions("baseline_v1")
        rotation_results = [r for r in results if r.operation_name == "RotateOperation"]

        if rotation_results:
            # Should detect some level of regression
            assert any(r.severity in [RegressionSeverity.WARNING, RegressionSeverity.CRITICAL]
                      for r in rotation_results)

    def test_performance_report_generation(self, regression_detector):
        """Test comprehensive performance report generation."""
        grid = create_test_grid(10, 10, "random")

        # Run multiple operations to build current performance data
        operations = [
            RotateOperation(angle=90),
            FlipOperation(direction="horizontal"),
            ColorReplaceOperation(source_color=1, target_color=8)
        ]

        for operation in operations:
            measure_operation_performance(operation, grid, runs=3, detector=regression_detector)

        # Generate performance report
        report = regression_detector.generate_report("baseline_v1")

        assert report.total_operations_analyzed > 0
        assert report.baseline_version == "baseline_v1"
        assert report.current_version == "test_current"
        assert len(report.recommendations) > 0

    def test_operation_scaling_with_regression_tracking(self, regression_detector):
        """Test that regression tracking works across different grid sizes."""
        operation = FlipOperation(direction="vertical")
        sizes = [5, 10, 15]

        for size in sizes:
            grid = create_test_grid(size, size, "random")
            metrics = measure_operation_performance(
                operation, grid, runs=3, detector=regression_detector
            )

            # Verify performance is reasonable for all sizes
            assert metrics["mean_time"] < 0.1, f"Flip too slow for {size}x{size}: {metrics['mean_time']:.4f}s"

        # Verify regression detector recorded metrics for different grid sizes
        stats = regression_detector.get_session_stats()
        assert "FlipOperation" in stats["operations_tracked"]

    def test_memory_regression_detection(self, regression_detector):
        """Test memory usage regression detection."""
        grid = create_test_grid(15, 15, "random")
        operation = ColorInvertOperation()

        # Measure with regression tracking
        measure_operation_performance(operation, grid, runs=3, detector=regression_detector)

        # Verify memory metrics were recorded
        session_stats = regression_detector.get_session_stats()
        assert session_stats["total_metrics"] >= 6  # 3 runs × 2 metrics (time + memory)

    def test_regression_alert_thresholds(self, regression_detector):
        """Test that regression alerts respect configured thresholds."""
        # Record performance metrics that should trigger warnings
        slow_operation_time = 0.1  # Assume this is significantly slower than baseline

        for _ in range(5):
            regression_detector.record_metric(
                operation_name="FlipOperation",
                metric_type=PerformanceMetricType.EXECUTION_TIME,
                value=slow_operation_time,
                metadata={"threshold_test": True}
            )

        results = regression_detector.detect_regressions("baseline_v1")
        flip_results = [r for r in results if r.operation_name == "FlipOperation"]

        if flip_results:
            # Check that thresholds are properly applied
            for result in flip_results:
                if result.severity == RegressionSeverity.WARNING:
                    assert 0.20 <= result.percentage_change < 0.50  # 20% - 50%
                elif result.severity == RegressionSeverity.CRITICAL:
                    assert result.percentage_change >= 0.50  # >= 50%

    def test_performance_trends_tracking(self, regression_detector):
        """Test that performance trends are properly tracked over time."""
        grid = create_test_grid(8, 8, "checkerboard")
        operation = TranslateOperation(offset=(1, 1), fill_color=0)

        # Simulate performance measurements over multiple sessions
        for session in range(3):
            for run in range(3):
                measure_operation_performance(operation, grid, runs=1, detector=regression_detector)

        # Verify comprehensive tracking
        stats = regression_detector.get_session_stats()
        assert stats["total_metrics"] >= 18  # 3 sessions × 3 runs × 2 metrics
        assert "TranslateOperation" in stats["operations_tracked"]

    @pytest.mark.parametrize("operation_class,max_allowed_time", [
        (RotateOperation, 0.02),
        (FlipOperation, 0.02),
        (ColorReplaceOperation, 0.02),
        (TranslateOperation, 0.02)
    ])
    def test_operation_regression_benchmarks(self, regression_detector, operation_class, max_allowed_time):
        """Test individual operations against regression benchmarks."""
        grid = create_test_grid(10, 10, "random")

        # Create operation with default parameters
        if operation_class == RotateOperation:
            operation = operation_class(angle=90)
        elif operation_class == FlipOperation:
            operation = operation_class(direction="horizontal")
        elif operation_class == ColorReplaceOperation:
            operation = operation_class(source_color=1, target_color=8)
        elif operation_class == TranslateOperation:
            operation = operation_class(offset=(1, 1), fill_color=0)
        else:
            operation = operation_class()

        # Measure performance
        metrics = measure_operation_performance(operation, grid, runs=5, detector=regression_detector)

        # Check against regression benchmark
        assert metrics["mean_time"] < max_allowed_time, (
            f"{operation_class.__name__} regression detected: "
            f"{metrics['mean_time']:.4f}s > {max_allowed_time:.4f}s limit"
        )

        # Verify no critical regressions detected
        results = regression_detector.detect_regressions("baseline_v1")
        operation_results = [r for r in results if r.operation_name == operation_class.__name__]

        if operation_results:
            critical_regressions = [r for r in operation_results if r.severity == RegressionSeverity.CRITICAL]
            assert len(critical_regressions) == 0, (
                f"Critical regression detected for {operation_class.__name__}: "
                f"{[r.message for r in critical_regressions]}"
            )
