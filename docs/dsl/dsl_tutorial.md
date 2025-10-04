# ARC DSL Tutorial: Step-by-Step Guide

Welcome to the ARC Domain-Specific Language (DSL) tutorial! This guide will take you from basic concepts to advanced techniques for creating powerful grid transformations.

## Table of Contents

1. [Introduction to ARC DSL](#introduction-to-arc-dsl)
2. [Basic Concepts](#basic-concepts)
3. [Your First Operation](#your-first-operation)
4. [Working with the DSL Engine](#working-with-the-dsl-engine)
5. [Basic Operations Walkthrough](#basic-operations-walkthrough)
6. [Creating DSL Programs](#creating-dsl-programs)
7. [Advanced Techniques](#advanced-techniques)
8. [Common Patterns and Solutions](#common-patterns-and-solutions)
9. [Best Practices and Tips](#best-practices-and-tips)
10. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Introduction to ARC DSL

The ARC DSL is a specialized language designed for transforming colored grids, the core data structure in ARC (Abstraction and Reasoning Challenge) tasks. The DSL provides a high-level, composable way to describe complex transformations that can be applied to grids.

### Key Features

- **Type-safe operations**: All operations work with strongly-typed grids and colors
- **Composable transformations**: Operations can be chained together
- **Performance optimized**: Built-in caching and timeout protection
- **Comprehensive coverage**: Operations for geometry, color, patterns, and composition
- **Extensible design**: Easy to add new operations

### Use Cases

- Solving ARC puzzles programmatically
- Generating training data for machine learning models
- Prototyping grid transformation algorithms
- Educational exploration of spatial reasoning

## Basic Concepts

### Grids

A grid is a 2D array of colored cells, where each color is represented by an integer from 0-9:

```python
# A simple 3x3 grid
grid = [
    [0, 1, 2],  # Row 0: black, blue, red
    [3, 4, 5],  # Row 1: green, yellow, gray
    [6, 7, 8]   # Row 2: magenta, orange, sky
]
```

### Colors

Colors in ARC are integers from 0-9, each representing a specific color:
- 0: Black
- 1: Blue
- 2: Red
- 3: Green
- 4: Yellow
- 5: Gray
- 6: Magenta
- 7: Orange
- 8: Sky Blue
- 9: Brown

### Operations

Operations are the building blocks of the DSL. Each operation takes a grid as input and produces a transformed grid as output. Operations are categorized by their function:

- **Geometric**: Spatial transformations (rotate, flip, translate)
- **Color**: Color manipulations (mapping, filtering, replacement)
- **Pattern**: Pattern-based operations (flood fill, matching)
- **Composition**: Grid combination and extraction (crop, overlay)

### Operation Results

Every operation returns an `OperationResult` object containing:

```python
@dataclass
class OperationResult:
    success: bool                    # Whether operation succeeded
    grid: Grid                      # Resulting grid
    error_message: str = None       # Error description if failed
    execution_time: float = None    # Time taken in seconds
    metadata: dict = None          # Additional information
```

## Your First Operation

Let's start with a simple rotation operation:

```python
from src.domain.dsl.geometric import RotateOperation
from src.domain.dsl.types import Color

# Create a sample grid
sample_grid = [
    [Color(0), Color(1), Color(2)],
    [Color(3), Color(4), Color(5)],
    [Color(6), Color(7), Color(8)]
]

# Create and execute a rotation operation
rotate_op = RotateOperation(angle=90)
result = rotate_op.execute(sample_grid)

if result.success:
    print("Rotation successful!")
    print("Execution time:", result.execution_time)
    for row in result.grid:
        print(row)
else:
    print("Operation failed:", result.error_message)
```

Expected output:
```
Rotation successful!
Execution time: 0.001
[6, 3, 0]
[7, 4, 1]
[8, 5, 2]
```

## Working with the DSL Engine

The DSL Engine manages operation execution, caching, and performance monitoring:

### Creating an Engine

```python
from src.domain.services.dsl_engine import DSLEngine, DSLEngineBuilder
from src.domain.dsl.geometric import RotateOperation, FlipOperation
from src.domain.dsl.color import ColorMapOperation

# Method 1: Direct creation
engine = DSLEngine(timeout_seconds=1.0, memory_limit_mb=100)

# Register operations
engine.register_operation(RotateOperation)
engine.register_operation(FlipOperation)
engine.register_operation(ColorMapOperation)

# Method 2: Using builder pattern
engine = (DSLEngineBuilder()
    .with_timeout(1.0)
    .with_memory_limit(100)
    .with_operations(RotateOperation, FlipOperation, ColorMapOperation)
    .build())
```

### Executing Programs

```python
from src.domain.dsl.base import DSLProgram

# Create a program with multiple operations
program = DSLProgram(operations=[
    {"name": "rotate", "parameters": {"angle": 90}},
    {"name": "flip", "parameters": {"direction": "horizontal"}},
    {"name": "color_map", "parameters": {"mapping": {0: 9, 9: 0}}}
])

# Execute the program
result = engine.execute_program(program, sample_grid)

if result.success:
    print("Program executed successfully!")
    print(f"Operations: {result.metadata.get('operation_count')}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Cache hits: {result.metadata.get('cache_hits')}")
else:
    print("Program failed:", result.error_message)
```

## Basic Operations Walkthrough

### Geometric Operations

#### Rotation

```python
# Rotate 90 degrees clockwise
rotate_90 = RotateOperation(angle=90)
result = rotate_90.execute(grid)

# Original:     After 90° rotation:
# [0, 1, 2]  →  [6, 3, 0]
# [3, 4, 5]  →  [7, 4, 1]
# [6, 7, 8]  →  [8, 5, 2]
```

#### Flipping

```python
# Horizontal flip (mirror left-right)
flip_h = FlipOperation(direction="horizontal")
result = flip_h.execute(grid)

# Original:     After horizontal flip:
# [0, 1, 2]  →  [2, 1, 0]
# [3, 4, 5]  →  [5, 4, 3]
# [6, 7, 8]  →  [8, 7, 6]
```

#### Translation

```python
# Shift right by 1, down by 1
translate = TranslateOperation(offset=(1, 1), fill_color=9)
result = translate.execute(grid)

# Original:     After translation:
# [0, 1, 2]  →  [9, 9, 9]
# [3, 4, 5]  →  [9, 0, 1]
# [6, 7, 8]  →  [9, 3, 4]
```

### Color Operations

#### Color Mapping

```python
# Map colors: 0→1, 1→2, 2→0 (shift colors)
color_map = ColorMapOperation(mapping={0: 1, 1: 2, 2: 0})
result = color_map.execute(grid)

# Original:     After color mapping:
# [0, 1, 2]  →  [1, 2, 0]
# [3, 4, 5]  →  [3, 4, 5]  # Unchanged
# [6, 7, 8]  →  [6, 7, 8]  # Unchanged
```

#### Color Filtering

```python
# Keep only red (2) and blue (1), replace others with black (0)
filter_colors = ColorFilterOperation(keep_colors=[1, 2], fill_color=0)
result = filter_colors.execute(grid)

# Original:     After filtering:
# [0, 1, 2]  →  [0, 1, 2]
# [3, 4, 5]  →  [0, 0, 0]
# [6, 7, 8]  →  [0, 0, 0]
```

### Pattern Operations

#### Flood Fill

```python
# Fill connected region starting from (0,0) with red (2)
flood_fill = PatternFillOperation(start_position=(0, 0), target_color=2)

# For a grid with connected black (0) regions:
# [0, 0, 1]     After flood fill:     [2, 2, 1]
# [0, 1, 1]  →  from (0,0) with 2  →  [2, 1, 1]
# [1, 1, 1]                           [1, 1, 1]
```

## Creating DSL Programs

DSL programs are JSON-like structures that define sequences of operations:

### Simple Program

```python
# Simple rotation program
program = DSLProgram(operations=[
    {"name": "rotate", "parameters": {"angle": 90}}
])
```

### Multi-Step Program

```python
# Complex transformation program
program = DSLProgram(
    operations=[
        # Step 1: Add border
        {
            "name": "pad", 
            "parameters": {
                "top": 1, "bottom": 1, "left": 1, "right": 1, 
                "fill_color": 0
            }
        },
        # Step 2: Rotate
        {
            "name": "rotate", 
            "parameters": {"angle": 90}
        },
        # Step 3: Invert colors
        {
            "name": "color_invert", 
            "parameters": {}
        },
        # Step 4: Mirror horizontally
        {
            "name": "flip", 
            "parameters": {"direction": "horizontal"}
        }
    ],
    metadata={
        "description": "Add border, rotate, invert, and mirror",
        "expected_size": (5, 5)
    }
)
```

### Program Execution

```python
# Execute the program
result = engine.execute_program(program, input_grid)

# Check results
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Operations executed: {result.metadata.get('operation_count')}")

if result.success:
    # Use the transformed grid
    transformed_grid = result.grid
else:
    print(f"Error: {result.error_message}")
```

## Advanced Techniques

### Operation Chaining

You can chain operations using the `>>` operator for more readable code:

```python
# Create a transformation pipeline
pipeline = (
    PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0) >>
    RotateOperation(angle=90) >>
    ColorInvertOperation() >>
    FlipOperation(direction="horizontal")
)

# Execute the chained operations
result = pipeline.execute(input_grid)
```

### Conditional Transformations

```python
def conditional_transform(grid):
    """Apply different transformations based on grid properties."""
    
    # Check grid size
    if len(grid) <= 3:
        # Small grids: just rotate
        transform = RotateOperation(angle=90)
    else:
        # Large grids: crop then rotate
        transform = (
            CropOperation(top=1, left=1, bottom=-2, right=-2) >>
            RotateOperation(angle=90)
        )
    
    return transform.execute(grid)
```

### Pattern Detection and Response

```python
def pattern_based_transform(grid):
    """Apply transformations based on detected patterns."""
    
    # Count colors
    color_counts = {}
    for row in grid:
        for color in row:
            color_counts[color] = color_counts.get(color, 0) + 1
    
    # Most common color
    dominant_color = max(color_counts, key=color_counts.get)
    
    if dominant_color == 0:  # Mostly black
        # Invert colors
        return ColorInvertOperation().execute(grid)
    elif dominant_color in [1, 2, 3]:  # Primary colors
        # Apply color mapping
        mapping = {dominant_color: 9}
        return ColorMapOperation(mapping=mapping).execute(grid)
    else:
        # Default: rotate
        return RotateOperation(angle=90).execute(grid)
```

### Error Handling and Recovery

```python
def robust_transform(grid, operations):
    """Execute operations with error handling and recovery."""
    
    current_grid = grid
    executed_ops = []
    
    for i, op_spec in enumerate(operations):
        try:
            # Create and execute operation
            operation = create_operation(op_spec)
            result = operation.execute(current_grid)
            
            if result.success:
                current_grid = result.grid
                executed_ops.append(op_spec)
            else:
                print(f"Operation {i} failed: {result.error_message}")
                # Try alternative operation or skip
                if "fallback" in op_spec:
                    fallback_op = create_operation(op_spec["fallback"])
                    result = fallback_op.execute(current_grid)
                    if result.success:
                        current_grid = result.grid
                        
        except Exception as e:
            print(f"Exception in operation {i}: {e}")
            continue
    
    return current_grid, executed_ops
```

## Common Patterns and Solutions

### Pattern 1: Symmetry Creation

Creating symmetric patterns is a common ARC task:

```python
def create_symmetry(grid, axis="horizontal"):
    """Create symmetric pattern by mirroring."""
    
    if axis == "horizontal":
        # Mirror horizontally and concatenate
        flipped = FlipOperation(direction="horizontal").execute(grid).grid
        # Combine original and flipped (implementation depends on requirements)
        
    elif axis == "vertical":
        # Mirror vertically and concatenate
        flipped = FlipOperation(direction="vertical").execute(grid).grid
        
    elif axis == "both":
        # Create 4-fold symmetry
        h_flip = FlipOperation(direction="horizontal").execute(grid).grid
        v_flip = FlipOperation(direction="vertical").execute(grid).grid
        # Combine all variants
    
    return result_grid
```

### Pattern 2: Color Cycling

Rotating through a sequence of colors:

```python
def cycle_colors(grid, cycle=[0, 1, 2, 0]):
    """Cycle colors according to specified sequence."""
    
    # Create mapping for one step in cycle
    mapping = {}
    for i in range(len(cycle) - 1):
        mapping[cycle[i]] = cycle[i + 1]
    
    return ColorMapOperation(mapping=mapping).execute(grid)
```

### Pattern 3: Region Processing

Processing different regions of a grid differently:

```python
def process_regions(grid):
    """Process different regions with different operations."""
    
    height, width = len(grid), len(grid[0])
    mid_h, mid_w = height // 2, width // 2
    
    # Top-left: rotate 90°
    tl_crop = CropOperation(top=0, left=0, bottom=mid_h-1, right=mid_w-1)
    tl_region = tl_crop.execute(grid).grid
    tl_rotated = RotateOperation(angle=90).execute(tl_region).grid
    
    # Top-right: flip horizontally
    tr_crop = CropOperation(top=0, left=mid_w, bottom=mid_h-1, right=width-1)
    tr_region = tr_crop.execute(grid).grid
    tr_flipped = FlipOperation(direction="horizontal").execute(tr_region).grid
    
    # Combine regions back (implementation depends on requirements)
    # ...
    
    return combined_grid
```

### Pattern 4: Iterative Refinement

Applying operations until a condition is met:

```python
def iterative_fill(grid, max_iterations=10):
    """Apply flood fill iteratively until no changes occur."""
    
    current_grid = grid
    iteration = 0
    
    while iteration < max_iterations:
        # Try flood fill from various positions
        changed = False
        
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if current_grid[r][c] == 0:  # Fill black cells
                    fill_op = PatternFillOperation(
                        start_position=(r, c), 
                        target_color=1
                    )
                    result = fill_op.execute(current_grid)
                    
                    if result.success and result.grid != current_grid:
                        current_grid = result.grid
                        changed = True
                        break
            
            if changed:
                break
        
        if not changed:
            break  # No more changes possible
        
        iteration += 1
    
    return current_grid
```

## Best Practices and Tips

### Performance Optimization

1. **Use Caching**: The DSL engine automatically caches results, but you can help by:
   ```python
   # Reuse operations with same parameters
   rotate_90 = RotateOperation(angle=90)
   result1 = rotate_90.execute(grid1)
   result2 = rotate_90.execute(grid2)  # Same operation instance
   ```

2. **Minimize Grid Copies**: Operations that modify grids in-place are more efficient:
   ```python
   # Prefer operations that work on views when possible
   # Chain operations to avoid intermediate copies
   pipeline = op1 >> op2 >> op3
   ```

3. **Monitor Performance**: Use engine statistics to identify bottlenecks:
   ```python
   stats = engine.get_execution_stats()
   print(f"Cache hit rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100:.1f}%")
   
   # Get operation profiles
   profiles = engine.get_operation_profiles()
   slow_ops = [p for p in profiles if p.average_time > 0.05]  # >50ms
   ```

### Memory Management

1. **Limit Grid Sizes**: Large grids consume significant memory:
   ```python
   # Check grid size before processing
   if len(grid) * len(grid[0]) > 10000:  # 100x100 limit
       print("Warning: Large grid may cause memory issues")
   ```

2. **Clear Caches**: For long-running applications:
   ```python
   # Periodically clear caches
   engine.clear_cache()
   ```

### Error Prevention

1. **Validate Inputs**: Always check grid validity:
   ```python
   def validate_grid(grid):
       if not grid or not grid[0]:
           return False
       
       width = len(grid[0])
       for row in grid:
           if len(row) != width:
               return False
           for cell in row:
               if not (0 <= cell <= 9):
                   return False
       return True
   ```

2. **Handle Edge Cases**: Consider boundary conditions:
   ```python
   # Check bounds before cropping
   if (top < 0 or left < 0 or 
       bottom >= len(grid) or right >= len(grid[0])):
       # Handle error appropriately
   ```

3. **Use Try-Catch**: Wrap operations in exception handling:
   ```python
   try:
       result = operation.execute(grid)
       if not result.success:
           print(f"Operation failed: {result.error_message}")
   except Exception as e:
       print(f"Unexpected error: {e}")
   ```

### Code Organization

1. **Group Related Operations**: Create composite operations for common patterns:
   ```python
   class MirrorAndRotateOperation(CompositeOperation):
       def __init__(self, direction="horizontal", angle=90):
           operations = [
               FlipOperation(direction=direction),
               RotateOperation(angle=angle)
           ]
           super().__init__(operations)
   ```

2. **Use Constants**: Define color constants for readability:
   ```python
   # Color constants
   BLACK, BLUE, RED = 0, 1, 2
   GREEN, YELLOW, GRAY = 3, 4, 5
   
   # Use in operations
   color_map = ColorMapOperation(mapping={BLACK: WHITE, WHITE: BLACK})
   ```

3. **Document Complex Transformations**: Use metadata and comments:
   ```python
   complex_program = DSLProgram(
       operations=[...],
       metadata={
           "description": "Creates checkerboard pattern from solid color",
           "input_requirements": "Single color grid, square dimensions",
           "output_format": "Alternating binary colors"
       }
   )
   ```

## Debugging and Troubleshooting

### Common Issues

1. **Grid Size Mismatch**: 
   ```python
   # Check dimensions after operations
   print(f"Grid size: {len(grid)}x{len(grid[0]) if grid else 0}")
   ```

2. **Color Range Errors**:
   ```python
   # Validate color values
   for row in grid:
       for cell in row:
           assert 0 <= cell <= 9, f"Invalid color: {cell}"
   ```

3. **Operation Chain Failures**:
   ```python
   # Debug each step in chain
   current = input_grid
   for i, op in enumerate(operations):
       result = op.execute(current)
       print(f"Step {i}: {result.success}")
       if not result.success:
           print(f"Error: {result.error_message}")
           break
       current = result.grid
   ```

### Debugging Tools

1. **Visual Grid Display**:
   ```python
   def display_grid(grid, title="Grid"):
       print(f"\n{title}:")
       for row in grid:
           print(" ".join(f"{cell:1d}" for cell in row))
   
   # Use between operations
   display_grid(input_grid, "Input")
   display_grid(result.grid, "Output")
   ```

2. **Operation Timing**:
   ```python
   import time
   
   start = time.time()
   result = operation.execute(grid)
   elapsed = time.time() - start
   print(f"Operation took: {elapsed:.3f}s")
   ```

3. **Grid Comparison**:
   ```python
   def grids_equal(grid1, grid2):
       if len(grid1) != len(grid2):
           return False
       for r1, r2 in zip(grid1, grid2):
           if r1 != r2:
               return False
       return True
   
   # Compare expected vs actual
   if not grids_equal(expected, result.grid):
       print("Grids don't match!")
       display_grid(expected, "Expected")
       display_grid(result.grid, "Actual")
   ```

### Performance Debugging

1. **Profile Operations**:
   ```python
   profiles = engine.get_operation_profiles()
   for profile in sorted(profiles, key=lambda p: p.average_time, reverse=True):
       print(f"{profile.name}: {profile.average_time:.3f}s avg, {profile.execution_count} calls")
   ```

2. **Memory Usage**:
   ```python
   import tracemalloc
   
   tracemalloc.start()
   result = engine.execute_program(program, grid)
   current, peak = tracemalloc.get_traced_memory()
   print(f"Memory usage: {current / 1024 / 1024:.1f}MB current, {peak / 1024 / 1024:.1f}MB peak")
   tracemalloc.stop()
   ```

3. **Cache Analysis**:
   ```python
   cache_stats = engine.get_cache_statistics()
   print(f"Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
   print(f"Cache entries: {cache_stats['total_cache_entries']}")
   ```

## Next Steps

Now that you've learned the basics of the ARC DSL, you can:

1. **Explore Advanced Operations**: Check out the [Operation Reference](operation_reference.md) for complete operation documentation
2. **Optimize Performance**: Read the [Performance Guide](performance_guide.md) for optimization techniques
3. **Build Custom Operations**: Learn to extend the DSL with your own operations
4. **Integrate with ML Models**: Use the DSL to generate training data or as part of neural architectures

Happy coding with the ARC DSL!