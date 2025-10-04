# ARC DSL Operation Reference

This document provides a complete reference for all operations available in the ARC Domain-Specific Language. Operations are organized by category and include detailed descriptions, parameters, return types, and examples.

## Table of Contents

- [Geometric Operations](#geometric-operations)
- [Color Operations](#color-operations)
- [Pattern Operations](#pattern-operations)
- [Composition Operations](#composition-operations)
- [Symmetry Operations](#symmetry-operations)
- [Connectivity Operations](#connectivity-operations)
- [Edge Operations](#edge-operations)
- [Operation Chaining](#operation-chaining)
- [Common Patterns](#common-patterns)

## Geometric Operations

Geometric operations modify the spatial arrangement of grid elements without changing their colors.

### rotate

**Description**: Rotate the grid by the specified angle (90, 180, or 270 degrees clockwise).

**Parameters**:
- `angle` (integer, required): Rotation angle in degrees. Valid values: [90, 180, 270]

**Returns**: Transformed grid with rotated orientation.

**Examples**:
```python
# Rotate 90 degrees clockwise
rotate_90 = RotateOperation(angle=90)

# Rotate 180 degrees
rotate_180 = RotateOperation(angle=180)

# Rotate 270 degrees clockwise (90 degrees counter-clockwise)
rotate_270 = RotateOperation(angle=270)
```

**Usage in DSL Program**:
```json
{
  "name": "rotate",
  "parameters": {"angle": 90}
}
```

### flip

**Description**: Flip/mirror the grid along the specified axis.

**Parameters**:
- `direction` (string, required): Direction to flip the grid. Valid values: ["horizontal", "vertical", "diagonal_main", "diagonal_anti"]

**Returns**: Transformed grid mirrored along the specified axis.

**Examples**:
```python
# Horizontal flip (left-right mirror)
flip_h = FlipOperation(direction="horizontal")

# Vertical flip (up-down mirror)
flip_v = FlipOperation(direction="vertical")

# Flip along main diagonal (transpose)
flip_diag = FlipOperation(direction="diagonal_main")

# Flip along anti-diagonal
flip_anti = FlipOperation(direction="diagonal_anti")
```

**Usage in DSL Program**:
```json
{
  "name": "flip",
  "parameters": {"direction": "horizontal"}
}
```

### translate

**Description**: Shift/translate the grid by the specified offset, filling empty areas with fill_color.

**Parameters**:
- `offset` (tuple, required): Row and column offset (row_offset, col_offset)
- `fill_color` (integer, optional, default: 0): Color to fill empty areas. Valid range: [0, 9]

**Returns**: Translated grid with specified fill color in empty areas.

**Examples**:
```python
# Shift right by 2, down by 1
translate = TranslateOperation(offset=(1, 2), fill_color=0)

# Shift up by 1, left by 1, fill with red
translate_red = TranslateOperation(offset=(-1, -1), fill_color=2)
```

**Usage in DSL Program**:
```json
{
  "name": "translate",
  "parameters": {
    "offset": [1, 2],
    "fill_color": 0
  }
}
```

### crop

**Description**: Crop grid to specified rectangular region.

**Parameters**:
- `top` (integer, required): Top row index (inclusive)
- `left` (integer, required): Left column index (inclusive)
- `bottom` (integer, required): Bottom row index (inclusive)
- `right` (integer, required): Right column index (inclusive)

**Returns**: Cropped grid containing only the specified region.

**Examples**:
```python
# Crop to top-left 3x3 region
crop = CropOperation(top=0, left=0, bottom=2, right=2)

# Crop center region
crop_center = CropOperation(top=1, left=1, bottom=3, right=3)
```

**Usage in DSL Program**:
```json
{
  "name": "crop",
  "parameters": {
    "top": 0,
    "left": 0,
    "bottom": 2,
    "right": 2
  }
}
```

### pad

**Description**: Pad grid by adding rows/columns around edges.

**Parameters**:
- `top` (integer, optional, default: 0): Rows to add at top
- `bottom` (integer, optional, default: 0): Rows to add at bottom
- `left` (integer, optional, default: 0): Columns to add at left
- `right` (integer, optional, default: 0): Columns to add at right
- `fill_color` (integer, optional, default: 0): Color for padding. Valid range: [0, 9]

**Returns**: Padded grid with added border.

**Examples**:
```python
# Add 1-row/column border on all sides
pad = PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0)

# Pad only on right and bottom
pad_rb = PadOperation(top=0, bottom=2, left=0, right=3, fill_color=5)
```

**Usage in DSL Program**:
```json
{
  "name": "pad",
  "parameters": {
    "top": 1,
    "bottom": 1,
    "left": 1,
    "right": 1,
    "fill_color": 0
  }
}
```

## Color Operations

Color operations modify the colors in the grid while typically preserving spatial relationships.

### color_map

**Description**: Map colors in the grid according to the specified mapping.

**Parameters**:
- `mapping` (dict, required): Dictionary mapping source colors to target colors
- `default_color` (integer, optional): Color to use for unmapped colors (None = keep original). Valid range: [0, 9]

**Returns**: Grid with colors mapped according to the specified mapping.

**Examples**:
```python
# Map red (2) to blue (1) and green (3) to yellow (4)
color_map = ColorMapOperation(mapping={2: 1, 3: 4})

# Map with default color for unmapped values
color_map_default = ColorMapOperation(mapping={2: 1}, default_color=0)
```

**Usage in DSL Program**:
```json
{
  "name": "color_map",
  "parameters": {
    "mapping": {"2": 1, "3": 4},
    "default_color": 0
  }
}
```

### color_filter

**Description**: Keep only specified colors, replace others with fill_color.

**Parameters**:
- `keep_colors` (list, required): List of colors to preserve
- `fill_color` (integer, optional, default: 0): Color to use for filtered out colors. Valid range: [0, 9]

**Returns**: Grid with only specified colors preserved.

**Examples**:
```python
# Keep only red (2) and blue (1), fill others with black (0)
color_filter = ColorFilterOperation(keep_colors=[2, 1], fill_color=0)
```

**Usage in DSL Program**:
```json
{
  "name": "color_filter",
  "parameters": {
    "keep_colors": [2, 1],
    "fill_color": 0
  }
}
```

### color_replace

**Description**: Replace all occurrences of source_color with target_color.

**Parameters**:
- `source_color` (integer, required): Color to replace. Valid range: [0, 9]
- `target_color` (integer, required): Color to replace with. Valid range: [0, 9]

**Returns**: Grid with all instances of source_color replaced with target_color.

**Examples**:
```python
# Replace all red (2) with blue (1)
color_replace = ColorReplaceOperation(source_color=2, target_color=1)
```

**Usage in DSL Program**:
```json
{
  "name": "color_replace",
  "parameters": {
    "source_color": 2,
    "target_color": 1
  }
}
```

### color_invert

**Description**: Invert colors using 9-complement (0->9, 1->8, 2->7, etc.).

**Parameters**: None

**Returns**: Grid with all colors inverted using 9-complement.

**Examples**:
```python
# Invert all colors: 0->9, 1->8, 2->7, etc.
invert = ColorInvertOperation()
```

**Usage in DSL Program**:
```json
{
  "name": "color_invert",
  "parameters": {}
}
```

### color_threshold

**Description**: Apply binary threshold: values < threshold -> low_color, values >= threshold -> high_color.

**Parameters**:
- `threshold` (integer, required): Threshold value. Valid range: [0, 9]
- `low_color` (integer, optional, default: 0): Color for values below threshold. Valid range: [0, 9]
- `high_color` (integer, optional, default: 9): Color for values >= threshold. Valid range: [0, 9]

**Returns**: Grid with binary color thresholding applied.

**Examples**:
```python
# Convert to binary: colors 0-4 -> black (0), colors 5-9 -> white (9)
threshold = ColorThresholdOperation(threshold=5, low_color=0, high_color=9)
```

**Usage in DSL Program**:
```json
{
  "name": "color_threshold",
  "parameters": {
    "threshold": 5,
    "low_color": 0,
    "high_color": 9
  }
}
```

## Pattern Operations

Pattern operations work with recurring structures or templates in the grid.

### pattern_fill

**Description**: Fill connected regions of the same color with a new color (flood fill).

**Parameters**:
- `target_color` (integer, required): Color to fill with. Valid range: [0, 9]
- `start_position` (tuple, optional): Starting position for flood fill (if None, fill all matching regions)
- `source_color` (integer, optional): Color to replace (if None, use color at start_position). Valid range: [0, 9]

**Returns**: Grid with connected regions filled.

**Examples**:
```python
# Fill connected black (0) regions with red (2) starting from (0, 0)
fill = PatternFillOperation(start_position=(0, 0), target_color=2)

# Fill all connected regions of specific color
fill_all = PatternFillOperation(source_color=0, target_color=2)
```

**Usage in DSL Program**:
```json
{
  "name": "pattern_fill",
  "parameters": {
    "start_position": [0, 0],
    "target_color": 2
  }
}
```

## Composition Operations

Composition operations combine multiple grids or extract sub-grids.

### crop (Composition Version)

**Description**: Extract a rectangular region from the grid.

**Parameters**:
- `top_left` (tuple, required): Top-left corner of the region (row, col)
- `dimensions` (tuple, optional): (height, width) of the region
- `bottom_right` (tuple, optional): Bottom-right corner of the region

**Returns**: Extracted rectangular region.

**Examples**:
```python
# Extract 3x3 region starting from (1, 1)
crop = CropOperation(top_left=(1, 1), dimensions=(3, 3))

# Extract region defined by corners
crop_corners = CropOperation(top_left=(0, 0), bottom_right=(2, 2))
```

**Usage in DSL Program**:
```json
{
  "name": "crop",
  "parameters": {
    "top_left": [1, 1],
    "dimensions": [3, 3]
  }
}
```

## Operation Chaining

Operations can be chained together using the `>>` operator or the `compose_with` method to create complex transformations.

### Chaining Syntax

```python
# Using >> operator
transformation = RotateOperation(angle=90) >> ColorMapOperation(mapping={0: 1, 1: 0})

# Using compose_with method
transformation = RotateOperation(angle=90).compose_with(ColorMapOperation(mapping={0: 1, 1: 0}))
```

### DSL Program with Chaining

```json
{
  "operations": [
    {
      "name": "rotate",
      "parameters": {"angle": 90}
    },
    {
      "name": "color_map",
      "parameters": {"mapping": {"0": 1, "1": 0}}
    },
    {
      "name": "flip",
      "parameters": {"direction": "horizontal"}
    }
  ]
}
```

### Complex Chaining Example

```python
# Create a complex transformation pipeline
pipeline = (
    PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0) >>
    RotateOperation(angle=90) >>
    ColorMapOperation(mapping={0: 9, 9: 0}) >>
    FlipOperation(direction="horizontal") >>
    CropOperation(top=0, left=0, bottom=3, right=3)
)
```

## Common Patterns

### Mirror and Combine

```python
# Create a mirrored version and combine
original = input_grid
mirrored = FlipOperation(direction="horizontal").execute(original).grid
# Further operations to combine...
```

### Color Swap

```python
# Swap two colors
color_swap = ColorMapOperation(mapping={0: 1, 1: 0})
```

### Rotate and Mirror

```python
# Common transformation: rotate then mirror
transform = RotateOperation(angle=90) >> FlipOperation(direction="vertical")
```

### Extract and Process Region

```python
# Extract region, process it, then place back
region_processor = (
    CropOperation(top=1, left=1, bottom=3, right=3) >>
    ColorInvertOperation() >>
    PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0)
)
```

### Conditional Color Mapping

```python
# Apply different transformations based on colors
binary_threshold = ColorThresholdOperation(threshold=5, low_color=0, high_color=1)
color_specific = ColorFilterOperation(keep_colors=[2, 3, 4], fill_color=0)
```

## Performance Considerations

- **Chaining**: Long chains of operations may impact performance. Consider optimizing by combining operations where possible.
- **Memory**: Large grids with complex operations may consume significant memory. Monitor usage in performance-critical applications.
- **Caching**: The DSL engine automatically caches operation results. Repeated operations on the same input will be faster.
- **Timeout**: Operations have built-in timeout protection (default: 1 second per program).

## Error Handling

All operations return `OperationResult` objects with:
- `success`: Boolean indicating success/failure
- `grid`: Resulting grid (original grid if operation failed)
- `error_message`: Description of any error that occurred
- `execution_time`: Time taken to execute the operation
- `metadata`: Additional information about the operation

Always check the `success` field before using the result grid in production code.