"""
CPU profiling and memory allocation tracking for transpiled Python execution.

This module provides comprehensive profiling capabilities including:
- CPU profiling using cProfile
- Memory allocation tracking using tracemalloc
- Resource usage monitoring
- Profiling data export for analysis
"""

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if sys.platform != "win32":
    import resource

from src.adapters.strategies.python_transpiler import MemoryAllocationData, ProfileData
from src.infrastructure.config import TranspilerSandboxConfig


@dataclass
class ProfilingConfig:
    """Configuration for profiling behavior."""
    cpu_profiling_enabled: bool = False
    memory_tracking_enabled: bool = False
    resource_monitoring_enabled: bool = True

    # CPU profiling settings
    profile_builtin_functions: bool = False
    max_profile_functions: int = 50

    # Memory tracking settings
    tracemalloc_enabled: bool = True
    memory_snapshot_interval: float = 0.1  # seconds
    max_traced_allocations: int = 1000

    # Resource monitoring
    monitor_peak_memory: bool = True
    monitor_cpu_times: bool = True

    # Export settings
    export_raw_profile: bool = False
    export_memory_traces: bool = False


class ProfilingManager:
    """Manages CPU profiling and memory allocation tracking during execution."""

    def __init__(self, config: ProfilingConfig | None = None):
        self.config = config or ProfilingConfig()
        self._profiler: cProfile.Profile | None = None
        self._memory_start_snapshot = None
        self._memory_peak_mb = 0.0
        self._memory_traces: list[tuple[str, int, float]] = []
        self._start_time = 0.0
        self._resource_start = None

    @contextmanager
    def profile_execution(self):
        """Context manager for profiled execution."""
        try:
            self._start_profiling()
            yield self
        finally:
            self._stop_profiling()

    def _start_profiling(self):
        """Start all enabled profiling."""
        self._start_time = time.time()

        # Start CPU profiling
        if self.config.cpu_profiling_enabled:
            self._profiler = cProfile.Profile(
                builtins=self.config.profile_builtin_functions
            )
            self._profiler.enable()

        # Start memory tracking
        if self.config.memory_tracking_enabled and self.config.tracemalloc_enabled:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._memory_start_snapshot = tracemalloc.take_snapshot()

        # Start resource monitoring
        if self.config.resource_monitoring_enabled:
            self._start_resource_monitoring()

    def _stop_profiling(self):
        """Stop all profiling."""
        # Stop CPU profiling
        if self._profiler:
            self._profiler.disable()

        # Stop memory tracking
        if self.config.memory_tracking_enabled and tracemalloc.is_tracing():
            # Take final snapshot but don't stop tracemalloc
            # (might be used by other parts of the system)
            pass

    def _start_resource_monitoring(self):
        """Start resource monitoring."""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self._resource_start = {
                    'memory_rss': process.memory_info().rss,
                    'memory_vms': process.memory_info().vms,
                    'cpu_times': process.cpu_times(),
                    'timestamp': time.time()
                }
            except Exception:
                self._resource_start = None
        elif sys.platform != "win32":
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                self._resource_start = {
                    'memory_kb': usage.ru_maxrss,
                    'user_time': usage.ru_utime,
                    'system_time': usage.ru_stime,
                    'timestamp': time.time()
                }
            except Exception:
                self._resource_start = None

    def get_profile_data(self) -> ProfileData:
        """Extract CPU profiling data."""
        if not self.config.cpu_profiling_enabled or not self._profiler:
            return ProfileData(enabled=False)

        # Get profile statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stats_stream)
        stats.sort_stats('cumulative')

        # Extract summary information
        total_calls = stats.total_calls
        primitive_calls = stats.prim_calls
        total_time = stats.total_tt

        # Get top functions
        top_functions = []
        stats.sort_stats('cumulative')
        for func_key, (_cc, nc, tt, ct, _callers) in list(stats.stats.items())[:self.config.max_profile_functions]:
            if isinstance(func_key, tuple) and len(func_key) >= 3:
                filename, line_no, func_name = func_key
                func_display = f"{filename}:{line_no}({func_name})"
            else:
                func_display = str(func_key)

            top_functions.append((func_display, nc, tt, ct))

        # Prepare raw stats if requested
        raw_stats = stats if self.config.export_raw_profile else None

        return ProfileData(
            enabled=True,
            total_calls=total_calls,
            primitive_calls=primitive_calls,
            total_time=total_time,
            cumulative_time=total_time,  # For CPU profiling, these are the same
            top_functions=top_functions,
            raw_stats=raw_stats
        )

    def get_memory_allocation_data(self) -> MemoryAllocationData:
        """Extract memory allocation tracking data."""
        if not self.config.memory_tracking_enabled:
            return MemoryAllocationData(enabled=False)

        current_memory = self._get_current_memory_mb()
        peak_memory = max(self._memory_peak_mb, current_memory)

        allocation_data = MemoryAllocationData(
            enabled=True,
            peak_memory_mb=peak_memory,
            current_memory_mb=current_memory,
            allocation_count=0,
            deallocation_count=0,
            net_allocations=0
        )

        # Get tracemalloc data if available
        if self.config.tracemalloc_enabled and tracemalloc.is_tracing():
            try:
                current_snapshot = tracemalloc.take_snapshot()

                if self._memory_start_snapshot:
                    # Compare snapshots to get allocation differences
                    top_stats = current_snapshot.compare_to(
                        self._memory_start_snapshot, 'lineno'
                    )

                    # Extract top allocators
                    top_allocators = []
                    traced_allocations = []

                    for stat in top_stats[:self.config.max_traced_allocations]:
                        size_mb = stat.size / (1024 * 1024)
                        location = f"{stat.traceback.format()[0]}" if stat.traceback else "unknown"

                        if size_mb > 0.001:  # Only track allocations > 1KB
                            top_allocators.append((location, size_mb))
                            traced_allocations.append((location, int(stat.size), time.time()))

                    allocation_data.top_allocators = top_allocators[:20]  # Top 20
                    allocation_data.traced_allocations = traced_allocations
                    allocation_data.allocation_count = len([s for s in top_stats if s.size > 0])
                    allocation_data.deallocation_count = len([s for s in top_stats if s.size < 0])
                    allocation_data.net_allocations = allocation_data.allocation_count - allocation_data.deallocation_count

            except Exception:
                # Tracemalloc might fail in some environments
                pass

        return allocation_data

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self._memory_peak_mb = max(self._memory_peak_mb, memory_mb)
                return memory_mb
            except Exception:
                pass

        if sys.platform != "win32":
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                memory_mb = usage.ru_maxrss / 1024  # Convert KB to MB
                self._memory_peak_mb = max(self._memory_peak_mb, memory_mb)
                return memory_mb
            except Exception:
                pass

        # Fallback
        return 10.0

    def export_profile_data(self, filename_prefix: str) -> dict[str, str]:
        """Export profiling data to files for analysis."""
        exported_files = {}

        if self.config.cpu_profiling_enabled and self._profiler and self.config.export_raw_profile:
            # Export raw profile data
            profile_file = f"{filename_prefix}_cpu_profile.prof"
            self._profiler.dump_stats(profile_file)
            exported_files['cpu_profile'] = profile_file

            # Export human-readable stats
            stats_file = f"{filename_prefix}_cpu_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                stats = pstats.Stats(self._profiler, stream=f)
                stats.sort_stats('cumulative')
                stats.print_stats()
            exported_files['cpu_stats'] = stats_file

        if self.config.memory_tracking_enabled and self.config.export_memory_traces:
            # Export memory allocation traces
            memory_file = f"{filename_prefix}_memory_traces.txt"
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write("Memory Allocation Traces\n")
                f.write("=" * 50 + "\n")

                memory_data = self.get_memory_allocation_data()
                f.write(f"Peak Memory: {memory_data.peak_memory_mb:.2f} MB\n")
                f.write(f"Current Memory: {memory_data.current_memory_mb:.2f} MB\n")
                f.write(f"Net Allocations: {memory_data.net_allocations}\n\n")

                f.write("Top Allocators:\n")
                for location, size_mb in memory_data.top_allocators:
                    f.write(f"  {size_mb:.3f} MB - {location}\n")

            exported_files['memory_traces'] = memory_file

        return exported_files


def create_profiling_manager(
    cpu_profiling: bool = False,
    memory_tracking: bool = False,
    resource_monitoring: bool = True,
    transpiler_config: TranspilerSandboxConfig | None = None
) -> ProfilingManager:
    """Create a ProfilingManager with appropriate configuration."""

    config = ProfilingConfig(
        cpu_profiling_enabled=cpu_profiling,
        memory_tracking_enabled=memory_tracking,
        resource_monitoring_enabled=resource_monitoring,
        tracemalloc_enabled=memory_tracking,
        export_raw_profile=cpu_profiling,
        export_memory_traces=memory_tracking
    )

    return ProfilingManager(config)


def get_resource_usage_snapshot() -> dict[str, Any]:
    """Get a snapshot of current resource usage."""
    snapshot = {
        'timestamp': time.time(),
        'memory_mb': 0.0,
        'cpu_percent': 0.0,
        'available': True
    }

    try:
        if HAS_PSUTIL:
            process = psutil.Process()
            snapshot['memory_mb'] = process.memory_info().rss / (1024 * 1024)
            snapshot['cpu_percent'] = process.cpu_percent()
        elif sys.platform != "win32":
            usage = resource.getrusage(resource.RUSAGE_SELF)
            snapshot['memory_mb'] = usage.ru_maxrss / 1024
        else:
            snapshot['available'] = False
    except Exception:
        snapshot['available'] = False

    return snapshot
