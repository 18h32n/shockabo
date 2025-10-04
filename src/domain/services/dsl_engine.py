"""
Domain-Specific Language Engine for ARC transformations.

This module provides the core execution engine for DSL programs that perform
grid transformations. The engine handles operation dispatch, execution timeout,
memory constraints, and result caching.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import signal
import threading
import time
import tracemalloc
from collections import defaultdict
from typing import Any, NamedTuple

from src.domain.dsl.base import DSLProgram, Operation, OperationResult
from src.domain.dsl.types import Grid
from src.infrastructure.config import TranspilerSandboxConfig
from src.utils.performance_regression_detector import (
    PerformanceMetricType,
    PerformanceRegressionDetector,
    RegressionSeverity,
)

logger = logging.getLogger(__name__)


class ExecutionStats(NamedTuple):
    """Statistics for DSL program execution."""
    total_execution_time: float
    operation_count: int
    cache_hits: int
    cache_misses: int
    peak_memory_mb: float
    slow_operations: list[tuple[str, float]]  # (operation_name, execution_time)


class OperationProfile(NamedTuple):
    """Profiling information for a single operation."""
    name: str
    execution_count: int
    total_time: float
    average_time: float
    max_time: float
    min_time: float


class DSLEngine:
    """
    Main execution engine for DSL programs.

    Handles operation dispatch, caching, and resource constraints to execute
    DSL programs efficiently within performance requirements (<100ms per program).

    Features:
    - Pattern-based caching for repeated operation sequences
    - Memory and timeout enforcement
    - Operation profiling and optimization
    - Performance benchmarking and reporting
    """

    def __init__(self, timeout_seconds: float = 1.0, memory_limit_mb: int = 100, enable_profiling: bool = True,
                 config: TranspilerSandboxConfig | None = None, enable_regression_detection: bool = True,
                 version: str = "unknown"):
        """
        Initialize the DSL engine.

        Args:
            timeout_seconds: Maximum execution time per program (default: 1.0s)
            memory_limit_mb: Maximum memory usage per execution (default: 100MB)
            enable_profiling: Whether to enable operation profiling (default: True)
            config: Optional configuration for transpiler and sandbox thresholds
            enable_regression_detection: Whether to enable performance regression detection
            version: Current version identifier for baseline management
        """
        self.config = config or TranspilerSandboxConfig()

        # Use config values as defaults if not explicitly provided
        self.timeout_seconds = timeout_seconds if timeout_seconds != 1.0 else self.config.timeout_seconds
        self.memory_limit_mb = memory_limit_mb if memory_limit_mb != 100 else self.config.memory_limit_mb
        self.memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
        self.enable_profiling = enable_profiling
        self.version = version

        # Operation registry
        self._operation_registry: dict[str, type[Operation]] = {}

        # Multi-level caching system
        self._result_cache: dict[str, tuple[OperationResult, float]] = {}  # (result, timestamp)
        self._pattern_cache: dict[str, Any] = {}  # Pattern-based caching
        self._sequence_cache: dict[str, OperationResult] = {}  # Operation sequence caching

        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._operation_profiles: dict[str, list[float]] = defaultdict(list)
        self._slow_operations_threshold = 0.05  # 50ms threshold

        # Memory tracking
        self._peak_memory = 0
        self._memory_snapshots: list[int] = []

        # Execution context
        self._current_execution = None
        self._execution_lock = threading.Lock()

        # Performance regression detection
        self.enable_regression_detection = enable_regression_detection
        if self.enable_regression_detection:
            self.regression_detector = PerformanceRegressionDetector()
            self.regression_detector.set_current_version(version)
        else:
            self.regression_detector = None

    def register_operation(self, operation_class: type[Operation]) -> None:
        """
        Register a DSL operation with the engine.

        Args:
            operation_class: The operation class to register
        """
        self._operation_registry[operation_class.get_name()] = operation_class

    def execute_program(self, program: DSLProgram, input_grid: Grid) -> OperationResult:
        """
        Execute a complete DSL program on an input grid with comprehensive
        performance monitoring, caching, and resource constraints.

        Args:
            program: The DSL program to execute
            input_grid: The input grid to transform

        Returns:
            OperationResult containing the final transformed grid and enhanced metadata

        Raises:
            TimeoutError: If execution exceeds timeout limit
            MemoryError: If execution exceeds memory limit
            ValueError: If program contains invalid operations
        """
        with self._execution_lock:
            return self._execute_with_monitoring(program, input_grid)

    def _execute_with_monitoring(self, program: DSLProgram, input_grid: Grid) -> OperationResult:
        """Execute program with full monitoring and resource constraints."""
        start_time = time.time()
        program_key = self._get_program_cache_key(program, input_grid)

        # Check sequence cache first
        if program_key in self._sequence_cache:
            self._cache_hits += 1
            cached_result = self._sequence_cache[program_key]

            # Create a new result with updated metadata to avoid mutating cached result
            new_metadata = (cached_result.metadata or {}).copy()
            new_metadata.update({
                "cached": True,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses
            })

            return OperationResult(
                success=cached_result.success,
                grid=cached_result.grid,
                error_message=cached_result.error_message,
                execution_time=0.001,  # Minimal time for cached result
                metadata=new_metadata
            )

        self._cache_misses += 1

        # Start memory monitoring
        if self.enable_profiling:
            tracemalloc.start()

        current_grid = input_grid
        operation_timings = []
        slow_operations = []

        # Set up timeout handling
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Program execution exceeded {self.timeout_seconds}s timeout")

        # Configure timeout (Unix systems only)
        old_handler = None
        try:
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout_seconds) + 1)
        except (AttributeError, OSError):
            # Windows doesn't support SIGALRM, use time-based checking
            pass

        try:
            for i, operation_spec in enumerate(program.operations):
                # Check timeout before each operation
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    raise TimeoutError(f"Program execution exceeded {self.timeout_seconds}s timeout")

                # Also check if next operation would exceed timeout
                # Estimate based on previous operations or use conservative estimate
                estimated_remaining_time = 0.01  # Conservative 10ms estimate per operation
                if elapsed + estimated_remaining_time > self.timeout_seconds:
                    raise TimeoutError(f"Program execution would exceed {self.timeout_seconds}s timeout")

                # Check memory usage
                if self.enable_profiling and i % 5 == 0:  # Check every 5 operations
                    current_memory = self._get_current_memory()
                    if current_memory > self.memory_limit_bytes:
                        gc.collect()  # Try garbage collection
                        current_memory = self._get_current_memory()
                        if current_memory > self.memory_limit_bytes:
                            raise MemoryError(f"Execution exceeded {self.memory_limit_mb}MB memory limit")

                # Execute operation with caching and timeout checking
                operation_start = time.time()
                operation = self._create_operation(operation_spec)

                # Check timeout before operation execution
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError(f"Program execution exceeded {self.timeout_seconds}s timeout")

                result = self._execute_operation_with_cache(operation, current_grid, operation_spec)
                operation_time = time.time() - operation_start

                # Check timeout after operation execution
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError(f"Program execution exceeded {self.timeout_seconds}s timeout")

                # Track performance
                if self.enable_profiling:
                    operation_name = operation_spec.get("name", "unknown")
                    self._operation_profiles[operation_name].append(operation_time)
                    operation_timings.append((operation_name, operation_time))

                    if operation_time > self._slow_operations_threshold:
                        slow_operations.append((operation_name, operation_time))

                    # Record metrics for regression detection
                    if self.regression_detector:
                        self.regression_detector.record_metric(
                            operation_name=operation_name,
                            metric_type=PerformanceMetricType.EXECUTION_TIME,
                            value=operation_time,
                            metadata={
                                "program_index": i,
                                "cached": False,
                                "parameters": operation_spec.get("parameters", {})
                            }
                        )

                if not result.success:
                    execution_time = time.time() - start_time
                    return OperationResult(
                        success=False,
                        grid=current_grid,
                        error_message=result.error_message,
                        execution_time=execution_time,
                        metadata={
                            "failed_operation_index": i,
                            "failed_operation": operation_spec.get("name", "unknown"),
                            "operations_executed": i,
                            "slow_operations": slow_operations
                        }
                    )

                current_grid = result.grid

            # Finalize execution
            execution_time = time.time() - start_time
            peak_memory_mb = 0

            if self.enable_profiling:
                _, peak_memory = tracemalloc.get_traced_memory()
                peak_memory_mb = peak_memory / (1024 * 1024)
                tracemalloc.stop()

            # Record overall program metrics for regression detection
            if self.regression_detector:
                cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
                self.regression_detector.record_metric(
                    operation_name="program_execution",
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    value=execution_time,
                    metadata={
                        "operation_count": len(program.operations),
                        "peak_memory_mb": peak_memory_mb,
                        "slow_operations_count": len(slow_operations)
                    }
                )
                self.regression_detector.record_metric(
                    operation_name="cache_performance",
                    metric_type=PerformanceMetricType.CACHE_HIT_RATE,
                    value=cache_hit_rate,
                    metadata={
                        "cache_hits": self._cache_hits,
                        "cache_misses": self._cache_misses
                    }
                )
                if peak_memory_mb > 0:
                    self.regression_detector.record_metric(
                        operation_name="program_execution",
                        metric_type=PerformanceMetricType.MEMORY_USAGE,
                        value=peak_memory_mb,
                        metadata={"operation_count": len(program.operations)}
                    )

            # Create result with comprehensive metadata
            final_result = OperationResult(
                success=True,
                grid=current_grid,
                execution_time=execution_time,
                metadata={
                    "operation_count": len(program.operations),
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "peak_memory_mb": peak_memory_mb,
                    "slow_operations": slow_operations,
                    "operation_timings": operation_timings,
                    "performance_target_met": execution_time < 0.1,  # <100ms target
                    "cached": False
                }
            )

            # Cache successful results
            if execution_time < 2.0:  # Only cache reasonably fast executions
                self._sequence_cache[program_key] = final_result

            return final_result

        except Exception as e:
            execution_time = time.time() - start_time
            if self.enable_profiling and tracemalloc.is_tracing():
                tracemalloc.stop()

            return OperationResult(
                success=False,
                grid=input_grid,
                error_message=str(e),
                execution_time=execution_time,
                metadata={
                    "error_type": type(e).__name__,
                    "slow_operations": slow_operations
                }
            )

        finally:
            # Clean up timeout handler
            if old_handler is not None and hasattr(signal, 'SIGALRM'):
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except (AttributeError, OSError):
                    pass

    def _create_operation(self, operation_spec: dict[str, Any]) -> Operation:
        """
        Create an operation instance from a specification.

        Args:
            operation_spec: Dictionary containing operation name and parameters

        Returns:
            Configured operation instance

        Raises:
            ValueError: If operation is not registered or spec is invalid
        """
        operation_name = operation_spec.get("name")
        if not operation_name:
            raise ValueError("Operation specification missing 'name' field")

        if operation_name not in self._operation_registry:
            raise ValueError(f"Unknown operation: {operation_name}")

        operation_class = self._operation_registry[operation_name]
        parameters = operation_spec.get("parameters", {})

        return operation_class(**parameters)

    def get_registered_operations(self) -> list[str]:
        """
        Get list of all registered operation names.

        Returns:
            List of operation names available in the engine
        """
        return list(self._operation_registry.keys())

    def _execute_operation_with_cache(self, operation: Operation, grid: Grid, operation_spec: dict[str, Any]) -> OperationResult:
        """Execute an operation with intelligent caching and timeout monitoring."""
        # Create cache key for this specific operation + grid combination
        cache_key = self._get_operation_cache_key(operation_spec, grid)

        # Check cache with timestamp validation
        if cache_key in self._result_cache:
            cached_result, timestamp = self._result_cache[cache_key]
            # Cache valid for 60 seconds
            if time.time() - timestamp < 60:
                return cached_result
            else:
                # Remove expired cache entry
                del self._result_cache[cache_key]

        # Execute operation with timeout monitoring using threading
        operation_start = time.time()
        result = self._execute_with_thread_timeout(operation, grid)
        operation_time = time.time() - operation_start

        # Enhance result with timing info
        if result.execution_time is None:
            result.execution_time = operation_time

        # Cache successful results
        if result.success and operation_time < 1.0:  # Only cache reasonably fast operations
            self._result_cache[cache_key] = (result, time.time())

        return result

    def _execute_with_thread_timeout(self, operation: Operation, grid: Grid) -> OperationResult:
        """Execute operation with thread-based timeout for better control."""
        import queue
        import threading

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def run_operation():
            try:
                result = operation.execute(grid)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        # Start operation in separate thread
        thread = threading.Thread(target=run_operation, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds * 0.8)  # Use 80% of timeout for individual operations

        if thread.is_alive():
            # Operation is still running, consider it timed out
            return OperationResult(
                success=False,
                grid=grid,
                error_message=f"Operation timed out after {self.timeout_seconds * 0.8}s"
            )

        # Check if operation completed with exception
        if not exception_queue.empty():
            exception = exception_queue.get()
            return OperationResult(
                success=False,
                grid=grid,
                error_message=str(exception)
            )

        # Check if operation completed successfully
        if not result_queue.empty():
            return result_queue.get()

        # This shouldn't happen, but handle gracefully
        return OperationResult(
            success=False,
            grid=grid,
            error_message="Operation completed without result"
        )

    def _get_operation_cache_key(self, operation_spec: dict[str, Any], grid: Grid) -> str:
        """Generate a cache key for an operation + grid combination."""
        # Create a hash of the operation spec and grid state
        spec_str = str(sorted(operation_spec.items()))
        grid_str = str(grid)
        combined = f"{spec_str}:{grid_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_program_cache_key(self, program: DSLProgram, input_grid: Grid) -> str:
        """Generate a cache key for a complete program + input combination."""
        program_str = str(program.operations)
        grid_str = str(input_grid)
        combined = f"{program_str}:{grid_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current
        return 0

    def get_execution_stats(self) -> ExecutionStats:
        """Get comprehensive execution statistics."""
        # Calculate slow operations from profiles
        slow_ops = []
        for op_name, times in self._operation_profiles.items():
            if times:
                max_time = max(times)
                if max_time > self._slow_operations_threshold:
                    slow_ops.append((op_name, max_time))

        return ExecutionStats(
            total_execution_time=sum(sum(times) for times in self._operation_profiles.values()),
            operation_count=sum(len(times) for times in self._operation_profiles.values()),
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            peak_memory_mb=self._peak_memory / (1024 * 1024) if self._peak_memory else 0,
            slow_operations=slow_ops
        )

    def get_operation_profiles(self) -> list[OperationProfile]:
        """Get detailed profiling information for all operations."""
        profiles = []
        for op_name, times in self._operation_profiles.items():
            if times:
                profiles.append(OperationProfile(
                    name=op_name,
                    execution_count=len(times),
                    total_time=sum(times),
                    average_time=sum(times) / len(times),
                    max_time=max(times),
                    min_time=min(times)
                ))

        # Sort by average time descending
        return sorted(profiles, key=lambda p: p.average_time, reverse=True)

    def optimize_hot_paths(self) -> dict[str, str]:
        """Analyze and provide optimization recommendations for slow operations."""
        recommendations = {}
        profiles = self.get_operation_profiles()

        for profile in profiles:
            if profile.average_time > self._slow_operations_threshold:
                if profile.average_time > 0.1:  # > 100ms
                    recommendations[profile.name] = "CRITICAL: Operation exceeds 100ms target. Consider algorithmic optimization."
                elif profile.average_time > 0.05:  # > 50ms
                    recommendations[profile.name] = "WARNING: Operation approaching performance limits. Monitor closely."
                elif profile.execution_count > 100 and profile.total_time > 1.0:
                    recommendations[profile.name] = "INFO: High-frequency operation with significant total time. Consider caching."

        return recommendations

    def benchmark_performance(self, test_programs: list[tuple[DSLProgram, Grid]], target_ms: float = 100.0) -> dict[str, Any]:
        """Run performance benchmarks on test programs."""
        results = {
            "programs_tested": len(test_programs),
            "programs_under_target": 0,
            "programs_over_target": 0,
            "average_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float('inf'),
            "target_ms": target_ms,
            "detailed_results": []
        }

        total_time = 0.0

        for i, (program, grid) in enumerate(test_programs):
            result = self.execute_program(program, grid)
            exec_time_ms = (result.execution_time or 0) * 1000

            total_time += exec_time_ms
            results["max_execution_time"] = max(results["max_execution_time"], exec_time_ms)
            results["min_execution_time"] = min(results["min_execution_time"], exec_time_ms)

            if exec_time_ms <= target_ms:
                results["programs_under_target"] += 1
            else:
                results["programs_over_target"] += 1

            results["detailed_results"].append({
                "program_index": i,
                "execution_time_ms": exec_time_ms,
                "success": result.success,
                "meets_target": exec_time_ms <= target_ms,
                "operation_count": len(program.operations)
            })

        if test_programs:
            results["average_execution_time"] = total_time / len(test_programs)

        return results

    def clear_cache(self) -> None:
        """Clear all caches and reset performance tracking."""
        self._result_cache.clear()
        self._pattern_cache.clear()
        self._sequence_cache.clear()
        self._operation_profiles.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._peak_memory = 0
        self._memory_snapshots.clear()
        gc.collect()  # Force garbage collection

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get detailed cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
            "result_cache_size": len(self._result_cache),
            "pattern_cache_size": len(self._pattern_cache),
            "sequence_cache_size": len(self._sequence_cache),
            "total_cache_entries": len(self._result_cache) + len(self._pattern_cache) + len(self._sequence_cache)
        }

    def create_performance_baseline(self, baseline_version: str | None = None) -> dict[str, Any]:
        """
        Create performance baselines from current operation profiles.

        Args:
            baseline_version: Version identifier for the baseline (uses current version if None)

        Returns:
            Dictionary with baseline creation results
        """
        if not self.regression_detector:
            return {"error": "Regression detection not enabled"}

        version = baseline_version or self.version
        baselines_created = 0

        # Create baselines from operation profiles
        for operation_name, execution_times in self._operation_profiles.items():
            if len(execution_times) >= self.regression_detector.min_samples:
                try:
                    self.regression_detector.create_baseline(
                        operation_name=operation_name,
                        metric_type=PerformanceMetricType.EXECUTION_TIME,
                        measurements=execution_times,
                        version=version
                    )
                    baselines_created += 1
                except Exception:
                    # Skip operations that can't create baselines
                    pass

        # Create cache performance baseline
        cache_stats = self.get_cache_statistics()
        if cache_stats["cache_hits"] + cache_stats["cache_misses"] > 0:
            try:
                self.regression_detector.create_baseline(
                    operation_name="cache_performance",
                    metric_type=PerformanceMetricType.CACHE_HIT_RATE,
                    measurements=[cache_stats["hit_rate_percent"] / 100],
                    version=version
                )
                baselines_created += 1
            except Exception:
                # Don't create cache baseline if insufficient samples
                pass

        return {
            "baselines_created": baselines_created,
            "version": version,
            "operations_profiled": len(self._operation_profiles),
            "total_measurements": sum(len(times) for times in self._operation_profiles.values())
        }

    def detect_performance_regressions(self, baseline_version: str) -> dict[str, Any]:
        """
        Detect performance regressions against a baseline version.

        Args:
            baseline_version: Version to compare against

        Returns:
            Dictionary with regression detection results
        """
        if not self.regression_detector:
            return {"error": "Regression detection not enabled"}

        try:
            regression_results = self.regression_detector.detect_regressions(baseline_version)

            # Categorize results
            critical_regressions = [r for r in regression_results if r.severity == RegressionSeverity.CRITICAL]
            warnings = [r for r in regression_results if r.severity == RegressionSeverity.WARNING]
            improvements = [r for r in regression_results if r.percentage_change < -0.1]

            return {
                "total_operations_analyzed": len(regression_results),
                "critical_regressions": len(critical_regressions),
                "warnings": len(warnings),
                "improvements": len(improvements),
                "baseline_version": baseline_version,
                "current_version": self.version,
                "regression_results": [
                    {
                        "operation": r.operation_name,
                        "metric": r.metric_type.value,
                        "severity": r.severity.value,
                        "percentage_change": f"{r.percentage_change * 100:.1f}%",
                        "message": r.message
                    }
                    for r in regression_results
                ],
                "critical_operations": [r.operation_name for r in critical_regressions],
                "warning_operations": [r.operation_name for r in warnings]
            }
        except Exception as e:
            logger.error(f"Failed to detect regressions: {e}")
            return {"error": str(e)}

    def generate_performance_report(self, baseline_version: str, output_path: str | None = None) -> dict[str, Any]:
        """
        Generate comprehensive performance regression report.

        Args:
            baseline_version: Version to compare against
            output_path: Optional path to save detailed report

        Returns:
            Summary of the performance report
        """
        if not self.regression_detector:
            return {"error": "Regression detection not enabled"}

        try:
            report = self.regression_detector.generate_report(baseline_version)

            if output_path:
                self.regression_detector.save_report(report, output_path)

            return {
                "baseline_version": report.baseline_version,
                "current_version": report.current_version,
                "total_operations_analyzed": report.total_operations_analyzed,
                "regressions_found": report.regressions_found,
                "critical_regressions": report.critical_regressions,
                "warnings_found": report.warnings_found,
                "operations_improved": report.operations_improved,
                "regression_rate": f"{(report.regressions_found / report.total_operations_analyzed * 100):.1f}%" if report.total_operations_analyzed > 0 else "0%",
                "recommendations": report.recommendations[:3],  # Top 3 recommendations
                "report_saved": output_path is not None,
                "output_path": output_path
            }
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}

    def set_version(self, version: str) -> None:
        """
        Update the current version identifier.

        Args:
            version: New version identifier
        """
        self.version = version
        if self.regression_detector:
            self.regression_detector.set_current_version(version)

    def get_regression_detector_stats(self) -> dict[str, Any]:
        """
        Get statistics from the regression detector.

        Returns:
            Dictionary with regression detector statistics
        """
        if not self.regression_detector:
            return {"error": "Regression detection not enabled"}

        session_stats = self.regression_detector.get_session_stats()
        available_versions = self.regression_detector.storage.get_all_versions()

        return {
            "enabled": True,
            "current_version": self.version,
            "available_baseline_versions": available_versions,
            "session_metrics": session_stats.get("total_metrics", 0),
            "unique_operations_tracked": session_stats.get("unique_operations", 0),
            "operations_tracked": session_stats.get("operations_tracked", [])
        }


class DSLEngineBuilder:
    """
    Builder pattern for creating configured DSL engines.

    Provides a fluent interface for setting up engines with specific
    operation sets and configuration.
    """

    def __init__(self):
        """Initialize the builder."""
        self._timeout = 1.0
        self._memory_limit = 100
        self._operations: list[type[Operation]] = []

    def with_timeout(self, seconds: float) -> DSLEngineBuilder:
        """
        Set the execution timeout.

        Args:
            seconds: Timeout in seconds

        Returns:
            Builder instance for chaining
        """
        self._timeout = seconds
        return self

    def with_memory_limit(self, mb: int) -> DSLEngineBuilder:
        """
        Set the memory limit.

        Args:
            mb: Memory limit in megabytes

        Returns:
            Builder instance for chaining
        """
        self._memory_limit = mb
        return self

    def with_operations(self, *operations: type[Operation]) -> DSLEngineBuilder:
        """
        Add operations to the engine.

        Args:
            operations: Operation classes to register

        Returns:
            Builder instance for chaining
        """
        self._operations.extend(operations)
        return self

    def build(self) -> DSLEngine:
        """
        Build the configured DSL engine.

        Returns:
            Configured DSL engine instance
        """
        engine = DSLEngine(
            timeout_seconds=self._timeout,
            memory_limit_mb=self._memory_limit
        )

        for operation in self._operations:
            engine.register_operation(operation)

        return engine
