"""
Profiling data export functionality for analysis and optimization.

This module provides utilities to export profiling data collected during
transpiled Python execution for detailed analysis and performance optimization.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any

from src.adapters.strategies.python_transpiler import (
    ExecutionMetrics,
    MemoryAllocationData,
    ProfileData,
)


class ProfilingDataExporter:
    """Exports profiling data to various formats for analysis."""

    def __init__(self, output_dir: str = "profiling_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_execution_metrics(
        self,
        metrics: ExecutionMetrics,
        program_id: str,
        timestamp: str | None = None
    ) -> dict[str, str]:
        """Export execution metrics to JSON and CSV files."""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename_base = f"{program_id}_{timestamp}"
        exported_files = {}

        # Export basic metrics to JSON
        json_file = self.output_dir / f"{filename_base}_metrics.json"
        metrics_dict = self._metrics_to_dict(metrics)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        exported_files['metrics_json'] = str(json_file)

        # Export operation timings to CSV
        if metrics.operation_timings:
            csv_file = self.output_dir / f"{filename_base}_timings.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Operation', 'Time_ms', 'Is_Slow'])

                for op_name, time_ms in metrics.operation_timings.items():
                    is_slow = any(slow_op[0] == op_name for slow_op in metrics.slow_operations)
                    writer.writerow([op_name, time_ms, is_slow])

            exported_files['timings_csv'] = str(csv_file)

        # Export profiling data if available
        if metrics.profile_data and metrics.profile_data.enabled:
            profile_files = self._export_cpu_profile_data(
                metrics.profile_data, filename_base
            )
            exported_files.update(profile_files)

        # Export memory allocation data if available
        if metrics.memory_allocation_data and metrics.memory_allocation_data.enabled:
            memory_files = self._export_memory_allocation_data(
                metrics.memory_allocation_data, filename_base
            )
            exported_files.update(memory_files)

        return exported_files

    def export_batch_metrics(
        self,
        metrics_list: list[ExecutionMetrics],
        program_ids: list[str],
        batch_name: str = "batch"
    ) -> dict[str, str]:
        """Export multiple execution metrics for batch analysis."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_base = f"{batch_name}_{timestamp}"
        exported_files = {}

        # Export summary statistics
        summary_file = self.output_dir / f"{filename_base}_summary.json"
        summary_data = self._calculate_batch_summary(metrics_list, program_ids)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)
        exported_files['batch_summary'] = str(summary_file)

        # Export detailed batch data
        detailed_file = self.output_dir / f"{filename_base}_detailed.csv"
        with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'Program_ID', 'Execution_Time_ms', 'Memory_Used_MB',
                'Operation_Count', 'Slow_Operation_Count',
                'CPU_Profiling_Enabled', 'Memory_Tracking_Enabled',
                'Peak_Memory_MB', 'Total_CPU_Calls'
            ])

            # Data rows
            for i, metrics in enumerate(metrics_list):
                program_id = program_ids[i] if i < len(program_ids) else f"program_{i}"

                cpu_enabled = metrics.profile_data and metrics.profile_data.enabled
                memory_enabled = metrics.memory_allocation_data and metrics.memory_allocation_data.enabled
                peak_memory = (metrics.memory_allocation_data.peak_memory_mb
                             if memory_enabled else metrics.memory_used_mb)
                total_calls = (metrics.profile_data.total_calls
                             if cpu_enabled else 0)

                writer.writerow([
                    program_id,
                    metrics.execution_time_ms,
                    metrics.memory_used_mb,
                    len(metrics.operation_timings),
                    len(metrics.slow_operations),
                    cpu_enabled,
                    memory_enabled,
                    peak_memory,
                    total_calls
                ])

        exported_files['batch_detailed'] = str(detailed_file)
        return exported_files

    def _metrics_to_dict(self, metrics: ExecutionMetrics) -> dict[str, Any]:
        """Convert ExecutionMetrics to dictionary for JSON export."""
        result = {
            'execution_time_ms': metrics.execution_time_ms,
            'memory_used_mb': metrics.memory_used_mb,
            'operation_timings': metrics.operation_timings,
            'slow_operations': [
                {'operation': op, 'time_ms': time_ms}
                for op, time_ms in metrics.slow_operations
            ]
        }

        if metrics.profile_data:
            result['profile_data'] = {
                'enabled': metrics.profile_data.enabled,
                'total_calls': metrics.profile_data.total_calls,
                'primitive_calls': metrics.profile_data.primitive_calls,
                'total_time': metrics.profile_data.total_time,
                'cumulative_time': metrics.profile_data.cumulative_time,
                'top_functions': [
                    {
                        'function': func_name,
                        'calls': calls,
                        'total_time': tt,
                        'cumulative_time': ct
                    }
                    for func_name, calls, tt, ct in metrics.profile_data.top_functions
                ]
            }

        if metrics.memory_allocation_data:
            result['memory_allocation_data'] = {
                'enabled': metrics.memory_allocation_data.enabled,
                'peak_memory_mb': metrics.memory_allocation_data.peak_memory_mb,
                'current_memory_mb': metrics.memory_allocation_data.current_memory_mb,
                'allocation_count': metrics.memory_allocation_data.allocation_count,
                'deallocation_count': metrics.memory_allocation_data.deallocation_count,
                'net_allocations': metrics.memory_allocation_data.net_allocations,
                'top_allocators': [
                    {'location': loc, 'size_mb': size}
                    for loc, size in metrics.memory_allocation_data.top_allocators
                ]
            }

        return result

    def _export_cpu_profile_data(
        self,
        profile_data: ProfileData,
        filename_base: str
    ) -> dict[str, str]:
        """Export CPU profiling data to detailed files."""
        exported_files = {}

        # Export top functions to CSV
        if profile_data.top_functions:
            csv_file = self.output_dir / f"{filename_base}_cpu_functions.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Function', 'Calls', 'Total_Time', 'Cumulative_Time', 'Time_Per_Call'])

                for func_name, calls, total_time, cum_time in profile_data.top_functions:
                    time_per_call = total_time / calls if calls > 0 else 0
                    writer.writerow([func_name, calls, total_time, cum_time, time_per_call])

            exported_files['cpu_functions'] = str(csv_file)

        # Export raw profile data if available
        if profile_data.raw_stats:
            try:
                stats_file = self.output_dir / f"{filename_base}_cpu_raw.txt"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    profile_data.raw_stats.print_stats(file=f)
                exported_files['cpu_raw_stats'] = str(stats_file)
            except Exception:
                # Raw stats might not be serializable
                pass

        return exported_files

    def _export_memory_allocation_data(
        self,
        memory_data: MemoryAllocationData,
        filename_base: str
    ) -> dict[str, str]:
        """Export memory allocation data to detailed files."""
        exported_files = {}

        # Export memory summary
        summary_file = self.output_dir / f"{filename_base}_memory_summary.json"
        summary = {
            'peak_memory_mb': memory_data.peak_memory_mb,
            'current_memory_mb': memory_data.current_memory_mb,
            'allocation_count': memory_data.allocation_count,
            'deallocation_count': memory_data.deallocation_count,
            'net_allocations': memory_data.net_allocations
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        exported_files['memory_summary'] = str(summary_file)

        # Export top allocators to CSV
        if memory_data.top_allocators:
            csv_file = self.output_dir / f"{filename_base}_memory_allocators.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Location', 'Size_MB'])

                for location, size_mb in memory_data.top_allocators:
                    writer.writerow([location, size_mb])

            exported_files['memory_allocators'] = str(csv_file)

        # Export allocation traces if available
        if memory_data.traced_allocations:
            trace_file = self.output_dir / f"{filename_base}_memory_traces.csv"
            with open(trace_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Location', 'Size_Bytes', 'Timestamp'])

                for location, size_bytes, timestamp in memory_data.traced_allocations:
                    writer.writerow([location, size_bytes, timestamp])

            exported_files['memory_traces'] = str(trace_file)

        return exported_files

    def _calculate_batch_summary(
        self,
        metrics_list: list[ExecutionMetrics],
        program_ids: list[str]
    ) -> dict[str, Any]:
        """Calculate summary statistics for a batch of metrics."""
        if not metrics_list:
            return {}

        execution_times = [m.execution_time_ms for m in metrics_list]
        memory_usage = [m.memory_used_mb for m in metrics_list]
        operation_counts = [len(m.operation_timings) for m in metrics_list]
        slow_op_counts = [len(m.slow_operations) for m in metrics_list]

        # Calculate basic statistics
        summary = {
            'batch_size': len(metrics_list),
            'program_ids': program_ids[:len(metrics_list)],
            'execution_time_stats': {
                'min_ms': min(execution_times),
                'max_ms': max(execution_times),
                'avg_ms': sum(execution_times) / len(execution_times),
                'total_ms': sum(execution_times)
            },
            'memory_usage_stats': {
                'min_mb': min(memory_usage),
                'max_mb': max(memory_usage),
                'avg_mb': sum(memory_usage) / len(memory_usage),
                'total_mb': sum(memory_usage)
            },
            'operation_stats': {
                'min_operations': min(operation_counts),
                'max_operations': max(operation_counts),
                'avg_operations': sum(operation_counts) / len(operation_counts),
                'total_operations': sum(operation_counts)
            },
            'slow_operation_stats': {
                'min_slow_ops': min(slow_op_counts),
                'max_slow_ops': max(slow_op_counts),
                'avg_slow_ops': sum(slow_op_counts) / len(slow_op_counts),
                'total_slow_ops': sum(slow_op_counts)
            }
        }

        # Add profiling statistics if available
        profiled_count = sum(1 for m in metrics_list
                           if m.profile_data and m.profile_data.enabled)
        memory_tracked_count = sum(1 for m in metrics_list
                                 if m.memory_allocation_data and m.memory_allocation_data.enabled)

        summary['profiling_stats'] = {
            'cpu_profiled_programs': profiled_count,
            'memory_tracked_programs': memory_tracked_count,
            'profiling_coverage': profiled_count / len(metrics_list) if metrics_list else 0,
            'memory_tracking_coverage': memory_tracked_count / len(metrics_list) if metrics_list else 0
        }

        return summary

    def create_performance_report(
        self,
        metrics_list: list[ExecutionMetrics],
        program_ids: list[str],
        report_name: str = "performance_report"
    ) -> str:
        """Create a comprehensive performance report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{report_name}_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Performance Analysis Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Programs Analyzed: {len(metrics_list)}\n\n")

            # Summary statistics
            summary = self._calculate_batch_summary(metrics_list, program_ids)
            f.write("## Summary Statistics\n\n")

            f.write("### Execution Time\n")
            exec_stats = summary['execution_time_stats']
            f.write(f"- Average: {exec_stats['avg_ms']:.2f} ms\n")
            f.write(f"- Min: {exec_stats['min_ms']:.2f} ms\n")
            f.write(f"- Max: {exec_stats['max_ms']:.2f} ms\n")
            f.write(f"- Total: {exec_stats['total_ms']:.2f} ms\n\n")

            f.write("### Memory Usage\n")
            mem_stats = summary['memory_usage_stats']
            f.write(f"- Average: {mem_stats['avg_mb']:.2f} MB\n")
            f.write(f"- Min: {mem_stats['min_mb']:.2f} MB\n")
            f.write(f"- Max: {mem_stats['max_mb']:.2f} MB\n")
            f.write(f"- Total: {mem_stats['total_mb']:.2f} MB\n\n")

            # Profiling coverage
            prof_stats = summary['profiling_stats']
            f.write("### Profiling Coverage\n")
            f.write(f"- CPU Profiled: {prof_stats['cpu_profiled_programs']} programs ({prof_stats['profiling_coverage']:.1%})\n")
            f.write(f"- Memory Tracked: {prof_stats['memory_tracked_programs']} programs ({prof_stats['memory_tracking_coverage']:.1%})\n\n")

            # Top slow operations
            f.write("## Performance Issues\n\n")
            all_slow_ops = []
            for i, metrics in enumerate(metrics_list):
                program_id = program_ids[i] if i < len(program_ids) else f"program_{i}"
                for op_name, time_ms in metrics.slow_operations:
                    all_slow_ops.append((program_id, op_name, time_ms))

            if all_slow_ops:
                f.write("### Slow Operations (Top 10)\n")
                slow_ops_sorted = sorted(all_slow_ops, key=lambda x: x[2], reverse=True)[:10]
                for program_id, op_name, time_ms in slow_ops_sorted:
                    f.write(f"- {program_id}: {op_name} ({time_ms:.2f} ms)\n")
            else:
                f.write("### Slow Operations\nNo slow operations detected.\n")

            f.write("\n---\nReport generated by ProfilingDataExporter\n")

        return str(report_file)
