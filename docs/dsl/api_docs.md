# ARC DSL API Documentation

This document provides comprehensive API documentation for the ARC Domain-Specific Language system, including engine usage, serialization formats, extension points, and integration guidelines.

## Table of Contents

1. [API Overview](#api-overview)
2. [DSL Engine API](#dsl-engine-api)
3. [Operation API](#operation-api)
4. [Serialization Format](#serialization-format)
5. [Extension Points](#extension-points)
6. [Integration Guide](#integration-guide)
7. [Error Handling](#error-handling)
8. [Examples and Use Cases](#examples-and-use-cases)

## API Overview

The ARC DSL API is organized into several main components:

- **DSL Engine**: Core execution engine with caching and performance monitoring
- **Operations**: Individual transformation operations
- **Programs**: Serializable sequences of operations
- **Results**: Standardized result objects with metadata
- **Types**: Core data types (grids, colors, positions)

### Core Concepts

```python
from src.domain.services.dsl_engine import DSLEngine
from src.domain.dsl.base import DSLProgram, OperationResult
from src.domain.dsl.types import Grid, Color

# Basic workflow
engine = DSLEngine()
program = DSLProgram(operations=[...])
result = engine.execute_program(program, input_grid)
```

## DSL Engine API

### Class: DSLEngine

The main execution engine for DSL programs with caching, performance monitoring, and resource management.

#### Constructor

```python
DSLEngine(
    timeout_seconds: float = 1.0,
    memory_limit_mb: int = 100,
    enable_profiling: bool = True
)
```

**Parameters:**
- `timeout_seconds` (float): Maximum execution time per program (default: 1.0)
- `memory_limit_mb` (int): Maximum memory usage per execution (default: 100)
- `enable_profiling` (bool): Enable operation profiling and timing (default: True)

**Example:**
```python
# Create engine with custom settings
engine = DSLEngine(
    timeout_seconds=2.0,
    memory_limit_mb=200,
    enable_profiling=True
)
```

#### Methods

##### execute_program()

Execute a complete DSL program on an input grid.

```python
def execute_program(
    self, 
    program: DSLProgram, 
    input_grid: Grid
) -> OperationResult:
```

**Parameters:**
- `program` (DSLProgram): The DSL program to execute
- `input_grid` (Grid): Input grid to transform

**Returns:**
- `OperationResult`: Contains success status, transformed grid, timing, and metadata

**Raises:**
- `TimeoutError`: If execution exceeds timeout limit
- `MemoryError`: If execution exceeds memory limit
- `ValueError`: If program contains invalid operations

**Example:**
```python
program = DSLProgram(operations=[
    {"name": "rotate", "parameters": {"angle": 90}},
    {"name": "flip", "parameters": {"direction": "horizontal"}}
])

result = engine.execute_program(program, input_grid)

if result.success:
    transformed_grid = result.grid
    print(f"Execution time: {result.execution_time:.3f}s")
else:
    print(f"Error: {result.error_message}")
```

##### register_operation()

Register an operation class with the engine.

```python
def register_operation(self, operation_class: Type[Operation]) -> None:
```

**Parameters:**
- `operation_class` (Type[Operation]): Operation class to register

**Example:**
```python
from src.domain.dsl.geometric import RotateOperation

engine.register_operation(RotateOperation)
```

##### get_execution_stats()

Get comprehensive execution statistics.

```python
def get_execution_stats(self) -> ExecutionStats:
```

**Returns:**
- `ExecutionStats`: Named tuple with timing, cache, and memory statistics

**Example:**
```python
stats = engine.get_execution_stats()
print(f"Total operations: {stats.operation_count}")
print(f"Cache hit rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100:.1f}%")
print(f"Peak memory: {stats.peak_memory_mb:.1f}MB")
```

##### get_operation_profiles()

Get detailed profiling information for all operations.

```python
def get_operation_profiles(self) -> List[OperationProfile]:
```

**Returns:**
- `List[OperationProfile]`: List of operation profiles sorted by average time

**Example:**
```python
profiles = engine.get_operation_profiles()
for profile in profiles:
    print(f"{profile.name}: {profile.average_time * 1000:.2f}ms avg, "
          f"{profile.execution_count} calls")
```

##### benchmark_performance()

Run performance benchmarks on test programs.

```python
def benchmark_performance(
    self, 
    test_programs: List[Tuple[DSLProgram, Grid]], 
    target_ms: float = 100.0
) -> Dict[str, Any]:
```

**Parameters:**
- `test_programs` (List[Tuple[DSLProgram, Grid]]): List of (program, grid) pairs to test
- `target_ms` (float): Target execution time in milliseconds (default: 100.0)

**Returns:**
- `Dict[str, Any]`: Benchmark results with timing statistics and target compliance

**Example:**
```python
test_cases = [
    (simple_program, small_grid),
    (complex_program, large_grid)
]

results = engine.benchmark_performance(test_cases, target_ms=50.0)
print(f"Programs under target: {results['programs_under_target']}")
print(f"Average time: {results['average_execution_time']:.2f}ms")
```

##### clear_cache()

Clear all caches and reset performance tracking.

```python
def clear_cache(self) -> None:
```

**Example:**
```python
# Clear caches periodically in long-running applications
if processed_count > 1000:
    engine.clear_cache()
    processed_count = 0
```

### Class: DSLEngineBuilder

Builder pattern for creating configured DSL engines.

#### Constructor

```python
DSLEngineBuilder()
```

#### Methods

##### with_timeout()

Set the execution timeout.

```python
def with_timeout(self, seconds: float) -> DSLEngineBuilder:
```

##### with_memory_limit()

Set the memory limit.

```python
def with_memory_limit(self, mb: int) -> DSLEngineBuilder:
```

##### with_operations()

Add operations to the engine.

```python
def with_operations(self, *operations: Type[Operation]) -> DSLEngineBuilder:
```

##### build()

Build the configured DSL engine.

```python
def build(self) -> DSLEngine:
```

**Example:**
```python
from src.domain.dsl.geometric import RotateOperation, FlipOperation
from src.domain.dsl.color import ColorMapOperation

engine = (DSLEngineBuilder()
    .with_timeout(2.0)
    .with_memory_limit(200)
    .with_operations(RotateOperation, FlipOperation, ColorMapOperation)
    .build())
```

## Operation API

### Abstract Base Class: Operation

All DSL operations inherit from this base class.

#### Methods

##### execute()

Execute the operation on an input grid.

```python
@abstractmethod
def execute(
    self, 
    grid: Grid, 
    context: Optional[TransformationContext] = None
) -> OperationResult:
```

**Parameters:**
- `grid` (Grid): Input grid to transform
- `context` (Optional[TransformationContext]): Optional execution context

**Returns:**
- `OperationResult`: Result object with success status and transformed grid

##### get_name()

Get the unique name of this operation.

```python
@classmethod
@abstractmethod
def get_name(cls) -> str:
```

##### get_description()

Get a human-readable description.

```python
@classmethod
@abstractmethod
def get_description(cls) -> str:
```

##### get_parameter_schema()

Get the parameter schema for this operation.

```python
@classmethod
@abstractmethod
def get_parameter_schema(cls) -> Dict[str, Any]:
```

#### Operation Composition

Operations can be chained together using the `>>` operator or `compose_with()` method.

```python
# Using >> operator
pipeline = RotateOperation(angle=90) >> FlipOperation(direction="horizontal")

# Using compose_with method
pipeline = RotateOperation(angle=90).compose_with(FlipOperation(direction="horizontal"))

# Execute composed operation
result = pipeline.execute(grid)
```

### Custom Operation Example

```python
from src.domain.dsl.base import Operation, OperationResult
from src.domain.dsl.types import Grid

class CustomOperation(Operation):
    """Example custom operation that doubles all color values."""
    
    def __init__(self, modulo: int = 10):
        super().__init__(modulo=modulo)
        self.modulo = modulo
    
    def execute(self, grid: Grid, context=None) -> OperationResult:
        try:
            result_grid = [
                [(cell * 2) % self.modulo for cell in row]
                for row in grid
            ]
            return OperationResult(success=True, grid=result_grid)
        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))
    
    @classmethod
    def get_name(cls) -> str:
        return "double_colors"
    
    @classmethod
    def get_description(cls) -> str:
        return "Double all color values with modulo"
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "modulo": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Modulo value for color wrapping"
            }
        }

# Register and use
engine.register_operation(CustomOperation)
```

## Serialization Format

### DSLProgram Serialization

DSL programs can be serialized to/from JSON format for storage and transmission.

#### JSON Format

```json
{
    "version": "1.0",
    "operations": [
        {
            "name": "rotate",
            "parameters": {
                "angle": 90
            }
        },
        {
            "name": "color_map",
            "parameters": {
                "mapping": {"0": 1, "1": 0},
                "default_color": null
            }
        }
    ],
    "metadata": {
        "description": "Rotate and swap colors",
        "author": "user",
        "created": "2024-01-15T10:30:00Z"
    }
}
```

#### Serialization API

```python
# To dictionary
program = DSLProgram(operations=[...])
program_dict = program.to_dict()

# To JSON string
import json
json_str = json.dumps(program_dict, indent=2)

# From dictionary
program = DSLProgram.from_dict(program_dict)

# From JSON string
program_dict = json.loads(json_str)
program = DSLProgram.from_dict(program_dict)
```

### Grid Serialization

Grids are represented as nested lists of integers.

```python
# Grid format
grid = [
    [0, 1, 2],  # Row 0
    [3, 4, 5],  # Row 1
    [6, 7, 8]   # Row 2
]

# JSON representation
{
    "grid": [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ],
    "dimensions": [3, 3]
}
```

### OperationResult Serialization

```python
def serialize_result(result: OperationResult) -> Dict[str, Any]:
    """Serialize an operation result."""
    return {
        "success": result.success,
        "grid": result.grid,
        "error_message": result.error_message,
        "execution_time": result.execution_time,
        "metadata": result.metadata or {}
    }
```

## Extension Points

### Creating Custom Operations

1. **Inherit from Base Class**: Choose appropriate base class (Operation, TransformOperation, ColorOperation, etc.)
2. **Implement Required Methods**: `execute()`, `get_name()`, `get_description()`, `get_parameter_schema()`
3. **Add Parameter Validation**: Validate parameters in `__init__()`
4. **Register with Engine**: Use `engine.register_operation()`

#### Example: Pattern Matching Operation

```python
from src.domain.dsl.base import PatternOperation, OperationResult
from src.domain.dsl.types import Grid, Pattern

class PatternMatchOperation(PatternOperation):
    """Find and highlight pattern matches in the grid."""
    
    def __init__(self, pattern: List[List[int]], highlight_color: int = 9):
        super().__init__(pattern=pattern, highlight_color=highlight_color)
        self.pattern = pattern
        self.highlight_color = highlight_color
        
        # Validate pattern
        if not pattern or not pattern[0]:
            raise ValueError("Pattern cannot be empty")
    
    def execute(self, grid: Grid, context=None) -> OperationResult:
        try:
            result_grid = [row.copy() for row in grid]
            matches = self._find_pattern_matches(grid, self.pattern)
            
            # Highlight matches
            for match_row, match_col in matches:
                for i in range(len(self.pattern)):
                    for j in range(len(self.pattern[0])):
                        if (match_row + i < len(result_grid) and 
                            match_col + j < len(result_grid[0])):
                            result_grid[match_row + i][match_col + j] = self.highlight_color
            
            return OperationResult(
                success=True, 
                grid=result_grid,
                metadata={"matches_found": len(matches)}
            )
            
        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))
    
    def _find_pattern_matches(self, grid: Grid, pattern: List[List[int]]) -> List[Tuple[int, int]]:
        """Find all occurrences of pattern in grid."""
        matches = []
        pattern_height, pattern_width = len(pattern), len(pattern[0])
        grid_height, grid_width = len(grid), len(grid[0])
        
        for r in range(grid_height - pattern_height + 1):
            for c in range(grid_width - pattern_width + 1):
                if self._pattern_matches_at(grid, pattern, r, c):
                    matches.append((r, c))
        
        return matches
    
    def _pattern_matches_at(self, grid: Grid, pattern: List[List[int]], row: int, col: int) -> bool:
        """Check if pattern matches at given position."""
        for i in range(len(pattern)):
            for j in range(len(pattern[0])):
                if grid[row + i][col + j] != pattern[i][j]:
                    return False
        return True
    
    @classmethod
    def get_name(cls) -> str:
        return "pattern_match"
    
    @classmethod
    def get_description(cls) -> str:
        return "Find and highlight pattern matches in the grid"
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "list",
                "required": True,
                "description": "2D pattern to search for"
            },
            "highlight_color": {
                "type": "integer",
                "required": False,
                "default": 9,
                "valid_range": [0, 9],
                "description": "Color to highlight matches"
            }
        }
```

### Adding Operation Categories

Create new base classes for operation categories:

```python
from src.domain.dsl.base import Operation

class AnalysisOperation(Operation):
    """Base class for grid analysis operations."""
    
    @classmethod
    def get_category(cls) -> str:
        return "analysis"

class StatisticsOperation(AnalysisOperation):
    """Compute grid statistics."""
    
    def execute(self, grid: Grid, context=None) -> OperationResult:
        stats = {
            "width": len(grid[0]) if grid else 0,
            "height": len(grid),
            "color_counts": self._count_colors(grid)
        }
        
        # Return grid unchanged, but with statistics in metadata
        return OperationResult(
            success=True,
            grid=grid,
            metadata={"statistics": stats}
        )
    
    def _count_colors(self, grid: Grid) -> Dict[int, int]:
        counts = {}
        for row in grid:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        return counts
```

### Engine Extensions

Extend the DSL engine with custom functionality:

```python
class ExtendedDSLEngine(DSLEngine):
    """DSL engine with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_history = []
    
    def execute_program(self, program: DSLProgram, input_grid: Grid) -> OperationResult:
        # Call parent implementation
        result = super().execute_program(program, input_grid)
        
        # Add to history
        self.execution_history.append({
            "timestamp": time.time(),
            "program": program,
            "input_grid": input_grid,
            "result": result
        })
        
        return result
    
    def get_execution_history(self) -> List[Dict]:
        """Get execution history."""
        return self.execution_history.copy()
    
    def export_successful_programs(self) -> List[DSLProgram]:
        """Export all successful programs from history."""
        return [
            entry["program"] for entry in self.execution_history
            if entry["result"].success
        ]
```

## Integration Guide

### Web API Integration

Create a REST API for the DSL system:

```python
from flask import Flask, request, jsonify
from src.domain.services.dsl_engine import DSLEngine
from src.domain.dsl.base import DSLProgram

app = Flask(__name__)
engine = DSLEngine()

@app.route('/execute', methods=['POST'])
def execute_program():
    try:
        data = request.json
        
        # Parse program and grid
        program = DSLProgram.from_dict(data['program'])
        grid = data['grid']
        
        # Execute
        result = engine.execute_program(program, grid)
        
        # Return result
        return jsonify({
            "success": result.success,
            "grid": result.grid,
            "execution_time": result.execution_time,
            "error_message": result.error_message,
            "metadata": result.metadata
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/operations', methods=['GET'])
def list_operations():
    """List available operations."""
    operations = []
    for name in engine.get_registered_operations():
        op_class = engine._operation_registry[name]
        operations.append({
            "name": op_class.get_name(),
            "description": op_class.get_description(),
            "parameters": op_class.get_parameter_schema()
        })
    
    return jsonify({"operations": operations})

@app.route('/benchmark', methods=['POST'])
def run_benchmark():
    """Run performance benchmark."""
    test_cases = request.json['test_cases']
    
    # Convert to programs and grids
    programs_and_grids = []
    for case in test_cases:
        program = DSLProgram.from_dict(case['program'])
        grid = case['grid']
        programs_and_grids.append((program, grid))
    
    # Run benchmark
    results = engine.benchmark_performance(programs_and_grids)
    
    return jsonify(results)
```

### Machine Learning Integration

Integrate with ML frameworks:

```python
import torch
import numpy as np
from typing import List, Tuple

class DSLDatasetGenerator:
    """Generate training data using DSL transformations."""
    
    def __init__(self, engine: DSLEngine):
        self.engine = engine
        self.programs = []
    
    def add_program(self, program: DSLProgram):
        """Add a program to the generation set."""
        self.programs.append(program)
    
    def generate_pairs(
        self, 
        input_grids: List[Grid], 
        max_pairs: int = 1000
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate input-output pairs for training."""
        
        pairs = []
        
        for input_grid in input_grids:
            for program in self.programs:
                if len(pairs) >= max_pairs:
                    break
                
                result = self.engine.execute_program(program, input_grid)
                
                if result.success:
                    # Convert to numpy arrays
                    input_array = np.array(input_grid, dtype=np.int32)
                    output_array = np.array(result.grid, dtype=np.int32)
                    
                    pairs.append((input_array, output_array))
        
        return pairs
    
    def create_pytorch_dataset(self, input_grids: List[Grid]) -> torch.utils.data.Dataset:
        """Create PyTorch dataset."""
        pairs = self.generate_pairs(input_grids)
        
        class DSLDataset(torch.utils.data.Dataset):
            def __init__(self, pairs):
                self.pairs = pairs
            
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                input_grid, output_grid = self.pairs[idx]
                return (
                    torch.tensor(input_grid, dtype=torch.long),
                    torch.tensor(output_grid, dtype=torch.long)
                )
        
        return DSLDataset(pairs)
```

### Batch Processing Integration

Process multiple grids efficiently:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class BatchProcessor:
    """Process multiple grids with DSL programs."""
    
    def __init__(self, engine: DSLEngine, max_workers: int = 4):
        self.engine = engine
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_batch(
        self, 
        program: DSLProgram, 
        grids: List[Grid],
        timeout_per_grid: float = 1.0
    ) -> List[OperationResult]:
        """Process multiple grids with the same program."""
        
        results = [None] * len(grids)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single, program, grid, timeout_per_grid): i
                for i, grid in enumerate(grids)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Grid {index} failed: {e}")
                    results[index] = OperationResult(
                        success=False,
                        grid=grids[index],
                        error_message=str(e)
                    )
        
        return results
    
    def _process_single(
        self, 
        program: DSLProgram, 
        grid: Grid, 
        timeout: float
    ) -> OperationResult:
        """Process a single grid with timeout."""
        
        # Create temporary engine with specific timeout
        temp_engine = DSLEngine(timeout_seconds=timeout)
        
        # Register same operations as main engine
        for name in self.engine.get_registered_operations():
            temp_engine.register_operation(self.engine._operation_registry[name])
        
        return temp_engine.execute_program(program, grid)
    
    def process_multiple_programs(
        self, 
        program_grid_pairs: List[Tuple[DSLProgram, Grid]]
    ) -> List[OperationResult]:
        """Process multiple (program, grid) pairs."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.engine.execute_program, program, grid)
                for program, grid in program_grid_pairs
            ]
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(OperationResult(
                        success=False,
                        grid=None,
                        error_message=str(e)
                    ))
        
        return results
```

## Error Handling

### Standard Error Types

The DSL system uses specific error types for different failure modes:

```python
# Timeout errors
try:
    result = engine.execute_program(program, grid)
except TimeoutError as e:
    print(f"Execution timed out: {e}")

# Memory errors  
except MemoryError as e:
    print(f"Memory limit exceeded: {e}")

# Invalid program errors
except ValueError as e:
    print(f"Invalid program: {e}")
```

### Error Recovery Strategies

```python
def robust_execution(engine: DSLEngine, program: DSLProgram, grid: Grid) -> OperationResult:
    """Execute program with error recovery."""
    
    try:
        # First attempt with normal timeout
        result = engine.execute_program(program, grid)
        
        if result.success:
            return result
        
        # If failed, try with simplified program
        simplified_program = simplify_program(program)
        if simplified_program:
            result = engine.execute_program(simplified_program, grid)
            
            if result.success:
                result.metadata = result.metadata or {}
                result.metadata["simplified"] = True
                return result
        
        # Final fallback: return identity transformation
        return OperationResult(
            success=True,
            grid=grid,
            metadata={"fallback_identity": True}
        )
        
    except TimeoutError:
        # Try with increased timeout
        high_timeout_engine = DSLEngine(timeout_seconds=5.0)
        return high_timeout_engine.execute_program(program, grid)
        
    except MemoryError:
        # Try with cropped grid
        if len(grid) > 10 or len(grid[0]) > 10:
            cropped_grid = crop_to_size(grid, 10, 10)
            return engine.execute_program(program, cropped_grid)
        else:
            return OperationResult(success=False, grid=grid, error_message="Memory limit exceeded")

def simplify_program(program: DSLProgram) -> Optional[DSLProgram]:
    """Create simplified version of program."""
    if len(program.operations) <= 1:
        return None
    
    # Try with just the first operation
    return DSLProgram(operations=[program.operations[0]])
```

## Examples and Use Cases

### Example 1: Image Processing Pipeline

```python
# Create image processing pipeline
from src.domain.dsl.geometric import RotateOperation, FlipOperation
from src.domain.dsl.color import ColorMapOperation, ColorThresholdOperation

engine = DSLEngine()

# Register operations
for op_class in [RotateOperation, FlipOperation, ColorMapOperation, ColorThresholdOperation]:
    engine.register_operation(op_class)

# Create processing pipeline
image_pipeline = DSLProgram(
    operations=[
        {"name": "rotate", "parameters": {"angle": 90}},
        {"name": "color_threshold", "parameters": {"threshold": 5, "low_color": 0, "high_color": 9}},
        {"name": "flip", "parameters": {"direction": "horizontal"}}
    ],
    metadata={"description": "Image preprocessing pipeline"}
)

# Process image
image_grid = load_image_as_grid("input.png")
result = engine.execute_program(image_pipeline, image_grid)

if result.success:
    save_grid_as_image(result.grid, "output.png")
```

### Example 2: Puzzle Solving

```python
# Solve ARC puzzles programmatically
def solve_arc_puzzle(training_pairs, test_input):
    """Attempt to solve ARC puzzle by finding pattern."""
    
    engine = DSLEngine()
    register_all_operations(engine)
    
    # Generate candidate programs
    candidate_programs = generate_candidate_programs()
    
    # Test each program against training data
    for program in candidate_programs:
        correct_count = 0
        
        for input_grid, expected_output in training_pairs:
            result = engine.execute_program(program, input_grid)
            
            if result.success and grids_equal(result.grid, expected_output):
                correct_count += 1
        
        # If program works on all training examples
        if correct_count == len(training_pairs):
            # Apply to test input
            test_result = engine.execute_program(program, test_input)
            
            if test_result.success:
                return test_result.grid, program
    
    return None, None

# Usage
training_data = load_arc_training_data("task_001.json")
test_input = training_data["test"][0]["input"]

solution, program = solve_arc_puzzle(training_data["train"], test_input)
if solution:
    print("Found solution!")
    print(f"Program: {program.operations}")
```

### Example 3: Data Augmentation

```python
# Generate augmented training data
class DataAugmenter:
    def __init__(self):
        self.engine = DSLEngine()
        self._register_augmentation_operations()
        
        # Define augmentation strategies
        self.augmentation_programs = [
            DSLProgram(operations=[{"name": "rotate", "parameters": {"angle": 90}}]),
            DSLProgram(operations=[{"name": "flip", "parameters": {"direction": "horizontal"}}]),
            DSLProgram(operations=[{"name": "flip", "parameters": {"direction": "vertical"}}]),
            DSLProgram(operations=[
                {"name": "rotate", "parameters": {"angle": 90}},
                {"name": "flip", "parameters": {"direction": "horizontal"}}
            ]),
        ]
    
    def augment_dataset(self, original_grids: List[Grid], multiplier: int = 4) -> List[Grid]:
        """Generate augmented dataset."""
        
        augmented = original_grids.copy()
        
        for grid in original_grids:
            augmented_count = 0
            
            for program in self.augmentation_programs:
                if augmented_count >= multiplier:
                    break
                
                result = self.engine.execute_program(program, grid)
                
                if result.success and not self._is_duplicate(result.grid, augmented):
                    augmented.append(result.grid)
                    augmented_count += 1
        
        return augmented
    
    def _is_duplicate(self, grid: Grid, existing_grids: List[Grid]) -> bool:
        """Check if grid already exists in the dataset."""
        return any(grids_equal(grid, existing) for existing in existing_grids)

# Usage
augmenter = DataAugmenter()
original_data = load_training_grids()
augmented_data = augmenter.augment_dataset(original_data, multiplier=3)

print(f"Original dataset: {len(original_data)} grids")
print(f"Augmented dataset: {len(augmented_data)} grids")
```

This comprehensive API documentation provides everything needed to use, extend, and integrate the ARC DSL system into various applications and workflows.