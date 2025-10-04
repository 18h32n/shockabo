# Configurable Thresholds Implementation

## Overview

This document describes the implementation of configurable thresholds in the Python transpiler and sandbox executor, replacing previously hardcoded values with configurable parameters.

## Changes Made

### 1. New Configuration Class

Added `TranspilerSandboxConfig` in `src/infrastructure/config.py`:

```python
@dataclass
class TranspilerSandboxConfig:
    """Configuration for transpiler and sandbox execution thresholds."""
    # Operation timing thresholds
    slow_operation_threshold_ms: float = 50.0  # milliseconds for marking operations as slow
    
    # Grid size limits
    max_grid_width: int = 30
    max_grid_height: int = 30
    
    # Execution limits
    timeout_seconds: float = 1.0  # Default timeout for program execution
    memory_limit_mb: int = 100  # Default memory limit in MB
    
    # Additional safety thresholds
    max_operation_memory_overhead_factor: float = 0.5  # 50% overhead per operation
```

### 2. Updated Components

#### PythonTranspiler (`src/adapters/strategies/python_transpiler.py`)
- Added `config` parameter to constructor
- Updated input validation to use `config.max_grid_height` and `config.max_grid_width`
- Updated helper functions to use configurable grid limits
- Updated bounds checking to use configurable limits
- Updated memory estimation to use configurable overhead factor

#### SandboxExecutor (`src/adapters/strategies/sandbox_executor.py`)
- Updated `SandboxConfig` to include `transpiler_config`
- Added automatic override of defaults with transpiler config values
- Updated `ExecutionRequest` to include `transpiler_config`
- Updated slow operation threshold detection to use configurable value

#### SandboxWorker (`src/adapters/strategies/sandbox_worker.py`)
- Added `transpiler_config` parameter to `execute_code_in_sandbox`
- Made parameter optional for backward compatibility

#### DSL Types (`src/domain/dsl/types.py`)
- Updated `is_valid_grid` to accept optional `config` parameter
- Grid validation now uses configurable size limits

#### DSL Engine (`src/domain/services/dsl_engine.py`)
- Added `config` parameter to constructor
- Uses config values as defaults for timeout and memory limits

### 3. Replaced Hardcoded Values

The following hardcoded values were made configurable:

| Component | Old Value | Configuration Parameter |
|-----------|-----------|-------------------------|
| Operation timing threshold | 50ms | `slow_operation_threshold_ms` |
| Grid size limits | 30x30 maximum | `max_grid_width`, `max_grid_height` |
| Timeout | 1 second default | `timeout_seconds` |
| Memory limit | 100MB default | `memory_limit_mb` |
| Memory overhead factor | 50% per operation | `max_operation_memory_overhead_factor` |

### 4. Backward Compatibility

All changes maintain backward compatibility:
- Default values match previous hardcoded values
- Existing code without configuration parameters continues to work
- Optional parameters ensure no breaking changes

## Usage Examples

### Basic Usage (Same as Before)
```python
# Uses default values (30x30 grids, 1s timeout, 100MB memory, 50ms slow threshold)
transpiler = PythonTranspiler()
executor = SandboxExecutor()
```

### Custom Configuration
```python
# Create custom configuration
config = TranspilerSandboxConfig(
    slow_operation_threshold_ms=25.0,  # Mark operations >25ms as slow
    max_grid_width=50,                 # Allow 50x50 grids
    max_grid_height=50,
    timeout_seconds=2.0,               # 2 second timeout
    memory_limit_mb=200                # 200MB memory limit
)

# Use with transpiler
transpiler = PythonTranspiler(config=config)

# Use with sandbox
sandbox_config = SandboxConfig(transpiler_config=config)
executor = SandboxExecutor(sandbox_config)

# Grid validation with custom limits
grid = [[1] * 40 for _ in range(40)]  # 40x40 grid
is_valid = is_valid_grid(grid, config)  # True (within 50x50 limit)
```

### Integration with Existing Config System
```python
# Add to PerformanceConfig in existing config system
performance_config = PerformanceConfig(
    transpiler_sandbox=TranspilerSandboxConfig(
        max_grid_width=25,
        max_grid_height=25,
        timeout_seconds=1.5
    )
)
```

## Error Messages

Error messages now reflect the configured values:

```python
# With custom 15x15 limit
config = TranspilerSandboxConfig(max_grid_width=15, max_grid_height=15)

# Generated validation code will contain:
# "Grid dimensions exceed 15x15: {actual_dimensions}"
# "Grid too large: {h}x{w} (max: 15x15)"
```

## Testing

Comprehensive tests were added in `tests/unit/adapters/strategies/test_configurable_thresholds.py`:

- Default configuration values
- Custom configuration application
- Grid size validation with custom limits
- Error message customization
- Backward compatibility
- Integration between components

All existing tests continue to pass, ensuring no regressions.

## Files Modified

1. `src/infrastructure/config.py` - Added `TranspilerSandboxConfig`
2. `src/adapters/strategies/python_transpiler.py` - Updated to use config
3. `src/adapters/strategies/sandbox_executor.py` - Updated to use config
4. `src/adapters/strategies/sandbox_worker.py` - Updated function signature
5. `src/domain/dsl/types.py` - Updated grid validation
6. `src/domain/services/dsl_engine.py` - Added config support
7. `tests/unit/adapters/strategies/test_configurable_thresholds.py` - New tests

## Benefits

1. **Flexibility**: Thresholds can be adjusted based on use case requirements
2. **Testability**: Different configurations can be tested easily
3. **Maintainability**: No more scattered hardcoded values
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Extensibility**: Easy to add new configurable parameters in the future