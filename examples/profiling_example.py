"""
Example demonstrating CPU profiling and memory allocation tracking
for the Python transpiler execution.

This example shows how to:
1. Enable profiling in the transpiler configuration
2. Execute a DSL program with profiling enabled
3. Extract and analyze profiling data
4. Export profiling results for detailed analysis
"""

import json
import time
from pathlib import Path

from src.adapters.strategies.profiling_exporter import ProfilingDataExporter
from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxConfig, SandboxExecutor
from src.infrastructure.config import TranspilerSandboxConfig


def create_sample_dsl_program():
    """Create a sample DSL program for testing profiling."""
    return {
        "operations": [
            {"type": "rotate", "angle": 90},
            {"type": "flip", "axis": "horizontal"},
            {"type": "color_replace", "from_color": 0, "to_color": 5},
            {"type": "pattern_fill", "pattern": [1, 2, 1], "target_color": 0},
            {"type": "crop", "x": 1, "y": 1, "width": 8, "height": 8}
        ]
    }


def create_sample_grid():
    """Create a sample grid for testing."""
    return [
        [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        [1, 0, 2, 0, 1, 0, 2, 0, 1, 0],
        [0, 2, 0, 1, 0, 2, 0, 1, 0, 2],
        [2, 0, 1, 0, 2, 0, 1, 0, 2, 0],
        [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        [1, 0, 2, 0, 1, 0, 2, 0, 1, 0],
        [0, 2, 0, 1, 0, 2, 0, 1, 0, 2],
        [2, 0, 1, 0, 2, 0, 1, 0, 2, 0],
        [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        [1, 0, 2, 0, 1, 0, 2, 0, 1, 0]
    ]


def demonstrate_basic_profiling():
    """Demonstrate basic profiling functionality."""
    print("=== Basic Profiling Example ===")

    # Create configuration with profiling enabled
    config = TranspilerSandboxConfig(
        cpu_profiling_enabled=True,
        memory_tracking_enabled=True,
        resource_monitoring_enabled=True,
        export_profiling_data=True
    )

    # Create transpiler with profiling configuration
    transpiler = PythonTranspiler(config=config)

    # Create and transpile a DSL program
    program = create_sample_dsl_program()
    result = transpiler.transpile(program, "profiled_function")

    print(f"Transpiled function: {result.function_name}")
    print(f"Estimated memory: {result.estimated_memory_mb:.2f} MB")
    print(f"Source lines: {len(result.source_code.splitlines())}")

    # Create sandbox executor with profiling
    sandbox_config = SandboxConfig(
        timeout_seconds=5.0,
        memory_limit_mb=200,
        transpiler_config=config
    )

    executor = SandboxExecutor(sandbox_config)

    # Execute the function with a sample grid
    grid = create_sample_grid()
    execution_result = executor.execute(result.source_code, result.function_name, grid)

    print(f"\nExecution successful: {execution_result.success}")
    if execution_result.success:
        print(f"Result grid size: {len(execution_result.result)}x{len(execution_result.result[0]) if execution_result.result else 0}")
    else:
        print(f"Error: {execution_result.error}")

    # Display execution metrics
    metrics = execution_result.metrics
    print("\n=== Execution Metrics ===")
    print(f"Execution time: {metrics.execution_time_ms:.2f} ms")
    print(f"Memory used: {metrics.memory_used_mb:.2f} MB")
    print(f"Operations executed: {len(metrics.operation_timings)}")
    print(f"Slow operations: {len(metrics.slow_operations)}")

    # Display operation timings
    if metrics.operation_timings:
        print("\n=== Operation Timings ===")
        for op_name, time_ms in metrics.operation_timings.items():
            print(f"  {op_name}: {time_ms:.2f} ms")

    # Display profiling data if available
    if metrics.profile_data and metrics.profile_data.enabled:
        print("\n=== CPU Profiling Data ===")
        profile = metrics.profile_data
        print(f"Total calls: {profile.total_calls}")
        print(f"Primitive calls: {profile.primitive_calls}")
        print(f"Total time: {profile.total_time:.4f} s")

        if profile.top_functions:
            print("\nTop functions by cumulative time:")
            for func_name, calls, total_time, cum_time in profile.top_functions[:5]:
                print(f"  {func_name}: {calls} calls, {cum_time:.4f}s cumulative")

    # Display memory allocation data if available
    if metrics.memory_allocation_data and metrics.memory_allocation_data.enabled:
        print("\n=== Memory Allocation Data ===")
        memory = metrics.memory_allocation_data
        print(f"Peak memory: {memory.peak_memory_mb:.2f} MB")
        print(f"Current memory: {memory.current_memory_mb:.2f} MB")
        print(f"Allocations: {memory.allocation_count}")
        print(f"Deallocations: {memory.deallocation_count}")
        print(f"Net allocations: {memory.net_allocations}")

        if memory.top_allocators:
            print("\nTop memory allocators:")
            for location, size_mb in memory.top_allocators[:5]:
                print(f"  {location}: {size_mb:.3f} MB")

    return execution_result


def demonstrate_batch_profiling():
    """Demonstrate batch profiling and analysis."""
    print("\n\n=== Batch Profiling Example ===")

    # Create configuration with profiling enabled
    config = TranspilerSandboxConfig(
        cpu_profiling_enabled=True,
        memory_tracking_enabled=True,
        resource_monitoring_enabled=True
    )

    transpiler = PythonTranspiler(config=config)

    sandbox_config = SandboxConfig(
        timeout_seconds=5.0,
        memory_limit_mb=200,
        transpiler_config=config
    )
    executor = SandboxExecutor(sandbox_config)

    # Create multiple test programs
    programs = [
        {  # Simple program
            "operations": [
                {"type": "rotate", "angle": 90}
            ]
        },
        {  # Medium complexity program
            "operations": [
                {"type": "flip", "axis": "horizontal"},
                {"type": "color_replace", "from_color": 0, "to_color": 5},
                {"type": "pattern_fill", "pattern": [1, 2], "target_color": 0}
            ]
        },
        {  # Complex program
            "operations": [
                {"type": "rotate", "angle": 180},
                {"type": "flip", "axis": "vertical"},
                {"type": "color_replace", "from_color": 1, "to_color": 3},
                {"type": "crop", "x": 2, "y": 2, "width": 6, "height": 6},
                {"type": "pattern_fill", "pattern": [2, 3, 2], "target_color": 1}
            ]
        }
    ]

    grid = create_sample_grid()
    metrics_list = []
    program_ids = []

    # Execute each program and collect metrics
    for i, program in enumerate(programs):
        program_id = f"program_{i+1}"
        program_ids.append(program_id)

        print(f"Executing {program_id} ({len(program['operations'])} operations)...")

        # Transpile and execute
        result = transpiler.transpile(program, f"func_{i+1}")
        execution_result = executor.execute(result.source_code, result.function_name, grid)

        if execution_result.success:
            metrics_list.append(execution_result.metrics)
            print(f"  Success: {execution_result.metrics.execution_time_ms:.2f} ms")
        else:
            print(f"  Failed: {execution_result.error}")

    # Export batch profiling data
    output_dir = Path("profiling_output")
    output_dir.mkdir(exist_ok=True)

    exporter = ProfilingDataExporter(str(output_dir))

    # Export individual metrics
    for i, metrics in enumerate(metrics_list):
        exported_files = exporter.export_execution_metrics(
            metrics, program_ids[i], f"batch_{int(time.time())}"
        )
        print(f"\nExported {program_ids[i]} profiling data:")
        for file_type, file_path in exported_files.items():
            print(f"  {file_type}: {file_path}")

    # Export batch summary
    batch_files = exporter.export_batch_metrics(
        metrics_list, program_ids, "example_batch"
    )
    print("\nExported batch analysis:")
    for file_type, file_path in batch_files.items():
        print(f"  {file_type}: {file_path}")

    # Create performance report
    report_file = exporter.create_performance_report(
        metrics_list, program_ids, "example_performance_report"
    )
    print(f"\nGenerated performance report: {report_file}")

    return metrics_list


def demonstrate_profiling_export():
    """Demonstrate detailed profiling data export."""
    print("\n\n=== Profiling Export Example ===")

    # Run basic profiling to get data
    execution_result = demonstrate_basic_profiling()

    if execution_result.success and execution_result.metrics:
        # Export detailed profiling data
        output_dir = Path("profiling_output")
        output_dir.mkdir(exist_ok=True)

        exporter = ProfilingDataExporter(str(output_dir))

        # Export with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exported_files = exporter.export_execution_metrics(
            execution_result.metrics,
            "detailed_example",
            timestamp
        )

        print("Exported detailed profiling data:")
        for file_type, file_path in exported_files.items():
            print(f"  {file_type}: {file_path}")

            # Display some content from exported files
            file_path_obj = Path(file_path)
            if file_path_obj.suffix == '.json':
                with open(file_path_obj, encoding='utf-8') as f:
                    data = json.load(f)
                print(f"    Content preview: {list(data.keys())}")
            elif file_path_obj.suffix == '.csv':
                with open(file_path_obj, encoding='utf-8') as f:
                    lines = f.readlines()
                print(f"    Rows: {len(lines)}, Header: {lines[0].strip() if lines else 'Empty'}")


def main():
    """Run profiling examples."""
    print("CPU Profiling and Memory Allocation Tracking Example")
    print("=" * 55)

    try:
        # Demonstrate basic profiling
        demonstrate_basic_profiling()

        # Demonstrate batch profiling
        demonstrate_batch_profiling()

        # Demonstrate profiling export
        demonstrate_profiling_export()

        print("\n\nProfiling examples completed successfully!")
        print("Check the 'profiling_output' directory for exported data files.")

    except Exception as e:
        print(f"Error running profiling examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
