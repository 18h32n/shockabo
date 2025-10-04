# CPU Profiling and Memory Allocation Tracking Guide

This guide explains how to use the CPU profiling and memory allocation tracking features for the Python transpiler execution system.

## Overview

The profiling system provides comprehensive performance analysis capabilities including:

- **CPU Profiling**: Track function calls, execution times, and performance bottlenecks
- **Memory Tracking**: Monitor memory allocations, peak usage, and allocation patterns  
- **Resource Monitoring**: Basic resource usage tracking with minimal overhead
- **Data Export**: Export profiling data for detailed analysis and optimization

## Configuration

### Enabling Profiling

Profiling is controlled through the `TranspilerSandboxConfig` class:

```python
from src.infrastructure.config import TranspilerSandboxConfig

# Enable all profiling features
config = TranspilerSandboxConfig(
    cpu_profiling_enabled=True,        # Enable CPU profiling
    memory_tracking_enabled=True,      # Enable memory allocation tracking
    resource_monitoring_enabled=True,  # Enable basic resource monitoring
    export_profiling_data=True         # Enable detailed data export
)
```

### Profiling Options

| Option | Description | Overhead | Use Case |
|--------|-------------|----------|----------|
| `cpu_profiling_enabled` | Detailed CPU profiling using cProfile | Medium | Performance optimization, bottleneck identification |
| `memory_tracking_enabled` | Memory allocation tracking using tracemalloc | Medium | Memory leak detection, allocation analysis |
| `resource_monitoring_enabled` | Basic resource usage monitoring | Low | General performance monitoring |
| `export_profiling_data` | Export detailed profiling data to files | Low | Detailed analysis, reporting |

## Basic Usage

### 1. Simple Profiling Example

```python
from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxExecutor, SandboxConfig
from src.infrastructure.config import TranspilerSandboxConfig

# Configure profiling
config = TranspilerSandboxConfig(
    cpu_profiling_enabled=True,
    memory_tracking_enabled=True
)

# Create transpiler and executor
transpiler = PythonTranspiler(config=config)
executor = SandboxExecutor(SandboxConfig(transpiler_config=config))

# Transpile and execute DSL program
program = {
    "operations": [
        {"type": "rotate", "angle": 90},
        {"type": "color_replace", "from_color": 0, "to_color": 5}
    ]
}

result = transpiler.transpile(program)
execution_result = executor.execute(result.source_code, result.function_name, grid)

# Access profiling data
metrics = execution_result.metrics
if metrics.profile_data and metrics.profile_data.enabled:
    print(f"Total function calls: {metrics.profile_data.total_calls}")
    print(f"Execution time: {metrics.profile_data.total_time:.4f}s")
```

### 2. Analyzing Execution Metrics

```python
def analyze_execution_metrics(metrics):
    """Analyze execution metrics and identify performance issues."""
    
    # Basic timing analysis
    print(f"Execution time: {metrics.execution_time_ms:.2f} ms")
    print(f"Memory used: {metrics.memory_used_mb:.2f} MB")
    
    # Operation performance
    if metrics.operation_timings:
        slowest_op = max(metrics.operation_timings.items(), key=lambda x: x[1])
        print(f"Slowest operation: {slowest_op[0]} ({slowest_op[1]:.2f} ms)")
    
    # Slow operation analysis
    if metrics.slow_operations:
        print(f"Slow operations detected: {len(metrics.slow_operations)}")
        for op_name, time_ms in metrics.slow_operations:
            print(f"  {op_name}: {time_ms:.2f} ms")
    
    # CPU profiling analysis
    if metrics.profile_data and metrics.profile_data.enabled:
        profile = metrics.profile_data
        print(f"\nCPU Profiling:")
        print(f"  Total calls: {profile.total_calls}")
        print(f"  Time per call: {profile.total_time / profile.total_calls * 1000:.3f} ms")
        
        # Top time-consuming functions
        if profile.top_functions:
            print("  Top functions by cumulative time:")
            for func_name, calls, total_time, cum_time in profile.top_functions[:3]:
                print(f"    {func_name}: {cum_time:.4f}s ({calls} calls)")
    
    # Memory analysis
    if metrics.memory_allocation_data and metrics.memory_allocation_data.enabled:
        memory = metrics.memory_allocation_data
        print(f"\nMemory Analysis:")
        print(f"  Peak memory: {memory.peak_memory_mb:.2f} MB")
        print(f"  Memory efficiency: {metrics.memory_used_mb / memory.peak_memory_mb:.2%}")
        print(f"  Net allocations: {memory.net_allocations}")
        
        # Top allocators
        if memory.top_allocators:
            print("  Top memory allocators:")
            for location, size_mb in memory.top_allocators[:3]:
                print(f"    {location}: {size_mb:.3f} MB")
```

## Advanced Features

### 1. Batch Profiling

```python
from src.adapters.strategies.profiling_exporter import ProfilingDataExporter

def profile_multiple_programs(programs, grids):
    """Profile multiple DSL programs for comparison."""
    
    config = TranspilerSandboxConfig(
        cpu_profiling_enabled=True,
        memory_tracking_enabled=True
    )
    
    transpiler = PythonTranspiler(config=config)
    executor = SandboxExecutor(SandboxConfig(transpiler_config=config))
    
    metrics_list = []
    program_ids = []
    
    for i, (program, grid) in enumerate(zip(programs, grids)):
        program_id = f"program_{i+1}"
        program_ids.append(program_id)
        
        # Execute with profiling
        result = transpiler.transpile(program, f"func_{i+1}")
        execution_result = executor.execute(result.source_code, result.function_name, grid)
        
        if execution_result.success:
            metrics_list.append(execution_result.metrics)
    
    # Export batch analysis
    exporter = ProfilingDataExporter("profiling_output")
    batch_files = exporter.export_batch_metrics(metrics_list, program_ids, "batch_analysis")
    
    # Generate performance report
    report_file = exporter.create_performance_report(metrics_list, program_ids)
    
    return metrics_list, batch_files, report_file
```

### 2. Custom Profiling Analysis

```python
def identify_performance_bottlenecks(metrics_list):
    """Identify common performance bottlenecks across multiple executions."""
    
    # Collect all slow operations
    all_slow_ops = []
    for metrics in metrics_list:
        all_slow_ops.extend(metrics.slow_operations)
    
    # Group by operation type
    from collections import defaultdict
    op_times = defaultdict(list)
    
    for op_name, time_ms in all_slow_ops:
        # Extract operation type (remove instance number)
        op_type = op_name.split('_')[0] if '_' in op_name else op_name
        op_times[op_type].append(time_ms)
    
    # Calculate statistics
    bottlenecks = []
    for op_type, times in op_times.items():
        avg_time = sum(times) / len(times)
        max_time = max(times)
        frequency = len(times)
        
        bottlenecks.append({
            'operation': op_type,
            'frequency': frequency,
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'total_time_ms': sum(times)
        })
    
    # Sort by total impact (frequency * average time)
    bottlenecks.sort(key=lambda x: x['frequency'] * x['avg_time_ms'], reverse=True)
    
    return bottlenecks
```

### 3. Memory Leak Detection

```python
def detect_memory_leaks(metrics_list):
    """Detect potential memory leaks in DSL program execution."""
    
    memory_trends = []
    
    for i, metrics in enumerate(metrics_list):
        if metrics.memory_allocation_data and metrics.memory_allocation_data.enabled:
            memory_data = metrics.memory_allocation_data
            
            # Calculate memory efficiency metrics
            peak_ratio = memory_data.peak_memory_mb / metrics.memory_used_mb if metrics.memory_used_mb > 0 else 1
            allocation_ratio = memory_data.allocation_count / memory_data.deallocation_count if memory_data.deallocation_count > 0 else float('inf')
            
            memory_trends.append({
                'execution': i,
                'peak_memory_mb': memory_data.peak_memory_mb,
                'final_memory_mb': memory_data.current_memory_mb,
                'peak_ratio': peak_ratio,
                'allocation_ratio': allocation_ratio,
                'net_allocations': memory_data.net_allocations
            })
    
    # Identify potential issues
    issues = []
    
    # Check for increasing memory usage
    if len(memory_trends) > 1:
        memory_growth = memory_trends[-1]['final_memory_mb'] - memory_trends[0]['final_memory_mb']
        if memory_growth > 10:  # > 10MB growth
            issues.append(f"Memory usage increased by {memory_growth:.2f} MB across executions")
    
    # Check for high peak ratios (memory spikes)
    high_peak_ratios = [t for t in memory_trends if t['peak_ratio'] > 2.0]
    if high_peak_ratios:
        issues.append(f"{len(high_peak_ratios)} executions had memory spikes > 2x final usage")
    
    # Check for allocation imbalances
    imbalanced = [t for t in memory_trends if t['allocation_ratio'] > 1.5]
    if imbalanced:
        issues.append(f"{len(imbalanced)} executions had significantly more allocations than deallocations")
    
    return issues, memory_trends
```

## Profiling Data Export

### 1. Automatic Export

```python
# Enable automatic export in configuration
config = TranspilerSandboxConfig(
    cpu_profiling_enabled=True,
    memory_tracking_enabled=True,
    export_profiling_data=True  # Enables automatic file export
)

# Profiling data will be automatically exported during execution
```

### 2. Manual Export

```python
from src.adapters.strategies.profiling_exporter import ProfilingDataExporter

# Create exporter
exporter = ProfilingDataExporter("custom_output_directory")

# Export individual execution metrics
exported_files = exporter.export_execution_metrics(
    metrics, 
    program_id="my_program",
    timestamp="20241201_120000"
)

# Export batch analysis
batch_files = exporter.export_batch_metrics(
    metrics_list, 
    program_ids, 
    batch_name="optimization_study"
)

# Generate performance report
report_file = exporter.create_performance_report(
    metrics_list, 
    program_ids, 
    report_name="performance_analysis"
)
```

### 3. Exported File Types

| File Type | Extension | Description |
|-----------|-----------|-------------|
| Metrics JSON | `.json` | Complete execution metrics in structured format |
| Timings CSV | `.csv` | Operation timings for spreadsheet analysis |
| CPU Functions CSV | `.csv` | CPU profiling function statistics |
| Memory Summary JSON | `.json` | Memory allocation summary |
| Memory Allocators CSV | `.csv` | Top memory allocators by size |
| Memory Traces CSV | `.csv` | Detailed allocation traces |
| Performance Report | `.md` | Human-readable performance analysis |

## Performance Considerations

### Profiling Overhead

| Feature | Overhead | Impact | Recommendation |
|---------|----------|--------|----------------|
| CPU Profiling | 10-30% | Significant | Use only during development/debugging |
| Memory Tracking | 5-15% | Moderate | Use for memory optimization studies |
| Resource Monitoring | <5% | Minimal | Safe for production monitoring |

### Best Practices

1. **Development Phase**: Enable all profiling features for comprehensive analysis
2. **Testing Phase**: Use resource monitoring only to minimize overhead
3. **Production**: Disable profiling entirely unless investigating specific issues
4. **Batch Analysis**: Profile representative sample of programs, not all executions

### Optimization Workflow

1. **Baseline Measurement**: Profile without optimizations to establish baseline
2. **Bottleneck Identification**: Use CPU profiling to find slow functions/operations
3. **Memory Analysis**: Use memory tracking to identify inefficient allocations
4. **Targeted Optimization**: Focus on highest-impact issues first
5. **Verification**: Re-profile to confirm improvements

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Check `tracemalloc_enabled` and `max_traced_allocations` settings
2. **Slow Execution**: Reduce profiling features or increase timeout limits
3. **Missing Data**: Ensure profiling is enabled in configuration
4. **Export Failures**: Check write permissions in output directory

### Performance Tuning

```python
# Optimized configuration for minimal overhead
config = TranspilerSandboxConfig(
    cpu_profiling_enabled=False,
    memory_tracking_enabled=False,
    resource_monitoring_enabled=True,  # Keep basic monitoring
    export_profiling_data=False
)

# Configuration for detailed analysis (higher overhead)
config = TranspilerSandboxConfig(
    cpu_profiling_enabled=True,
    memory_tracking_enabled=True,
    resource_monitoring_enabled=True,
    export_profiling_data=True
)
```

## Examples

See `examples/profiling_example.py` for complete working examples demonstrating:

- Basic profiling setup and usage
- Batch profiling and analysis
- Data export and reporting
- Performance bottleneck identification

## API Reference

### Core Classes

- `ProfilingManager`: Main profiling coordination
- `ProfileData`: CPU profiling results
- `MemoryAllocationData`: Memory tracking results
- `ProfilingDataExporter`: Data export functionality
- `TranspilerSandboxConfig`: Configuration management

### Key Methods

- `create_profiling_manager()`: Factory function for profiling manager
- `export_execution_metrics()`: Export individual execution data
- `export_batch_metrics()`: Export batch analysis
- `create_performance_report()`: Generate human-readable report