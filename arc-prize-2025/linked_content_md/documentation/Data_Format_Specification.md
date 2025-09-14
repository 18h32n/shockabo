# ARC-AGI-2 Data Format Specification

**Document Type:** Technical Documentation  
**Version:** 2.0  
**Last Updated:** 2025-08-30  
**Audience:** Competition Participants, Researchers, Developers  

## Overview

This document provides the complete specification for the ARC-AGI-2 dataset format used in the ARC Prize 2025 competition. Understanding this format is essential for loading data, developing solutions, and creating valid submissions.

## Task File Structure

### JSON Schema
Each ARC task is stored as a JSON file with the following structure:

```json
{
  "train": [
    {
      "input": [[int, int, ...], [int, int, ...], ...],
      "output": [[int, int, ...], [int, int, ...], ...]
    },
    {
      "input": [[int, int, ...], [int, int, ...], ...], 
      "output": [[int, int, ...], [int, int, ...], ...]
    }
    // Additional training examples (3-5 total)
  ],
  "test": [
    {
      "input": [[int, int, ...], [int, int, ...], ...],
      "output": [[int, int, ...], [int, int, ...], ...] // May be missing in evaluation sets
    }
  ]
}
```

### Field Specifications

#### `train` Array
- **Type**: Array of objects
- **Length**: 3-5 training examples per task
- **Purpose**: Demonstrate the pattern or rule to be learned
- **Required**: Always present and complete
- **Content**: Input/output pairs showing transformation pattern

#### `test` Array  
- **Type**: Array of objects
- **Length**: Exactly 1 test example per task
- **Purpose**: Input requiring output prediction
- **Required**: Always present
- **Content**: Input grid with output to be predicted

#### Input/Output Grids
- **Type**: 2D array of integers (nested lists)
- **Constraints**: All values must be integers 0-9
- **Dimensions**: Variable, typically 1x1 to 30x30
- **Representation**: `grid[row][column]` indexing

## Color Encoding System

### Standard Color Mapping
```json
{
  "0": "Black (Background)",
  "1": "Blue",
  "2": "Red", 
  "3": "Green",
  "4": "Yellow",
  "5": "Gray",
  "6": "Pink",
  "7": "Orange",
  "8": "Light Blue",
  "9": "Brown"
}
```

### Color Properties
- **Range**: Exactly 10 colors (0-9)
- **Background**: Color 0 typically represents background/empty space
- **Usage**: Any color can appear in any position
- **Meaning**: Colors have no inherent semantic meaning beyond the task context

## Grid Specifications

### Size Constraints
- **Minimum**: 1x1 grid
- **Maximum**: 30x30 grid
- **Typical**: 3x3 to 10x10 grids most common
- **Rectangular**: Grids can be non-square (e.g., 5x3, 2x8)

### Coordinate System
- **Origin**: Top-left corner is (0, 0)
- **Indexing**: `grid[row][column]` or `grid[y][x]`
- **Row Direction**: Increases downward (y-axis)
- **Column Direction**: Increases rightward (x-axis)

### Grid Properties
- **Uniformity**: All rows in a grid must have the same length
- **Validity**: All cells must contain integers 0-9
- **Consistency**: Input and output grids can have different dimensions

## File Naming Conventions

### Training Dataset
```
training/
├── task_00001.json
├── task_00002.json
├── ...
└── task_01000.json
```

### Evaluation Datasets
```
evaluation/
├── public/
│   ├── task_00001.json
│   ├── task_00002.json
│   ├── ...
│   └── task_00120.json
├── semi_private/
│   ├── task_00001.json (outputs hidden)
│   ├── ...
│   └── task_00120.json
└── private/
    ├── task_00001.json (outputs hidden)
    ├── ...
    └── task_00120.json
```

## Validation Rules

### Structural Validation
1. **JSON Validity**: File must be valid JSON
2. **Required Fields**: Must contain 'train' and 'test' arrays
3. **Training Examples**: 3-5 examples in 'train' array
4. **Test Examples**: Exactly 1 example in 'test' array
5. **Input/Output Fields**: Each example must have 'input' field

### Content Validation
1. **Grid Structure**: All grids must be 2D arrays
2. **Row Consistency**: All rows in a grid must have same length
3. **Value Range**: All values must be integers 0-9
4. **Non-Empty**: Grids must have at least 1x1 size
5. **Size Limits**: Grids must not exceed 30x30

### Python Validation Example
```python
import json

def validate_arc_task(task_data):
    """Validate ARC task format"""
    # Check required fields
    assert 'train' in task_data, "Missing 'train' field"
    assert 'test' in task_data, "Missing 'test' field"
    
    # Check training examples
    assert 3 <= len(task_data['train']) <= 5, "Must have 3-5 training examples"
    
    # Check test examples  
    assert len(task_data['test']) == 1, "Must have exactly 1 test example"
    
    # Validate each example
    for i, example in enumerate(task_data['train']):
        validate_example(example, f"train[{i}]")
    
    for i, example in enumerate(task_data['test']):
        validate_example(example, f"test[{i}]", require_output=False)
    
    return True

def validate_example(example, context, require_output=True):
    """Validate individual example"""
    assert 'input' in example, f"{context}: Missing 'input'"
    validate_grid(example['input'], f"{context}.input")
    
    if require_output:
        assert 'output' in example, f"{context}: Missing 'output'"
        validate_grid(example['output'], f"{context}.output")

def validate_grid(grid, context):
    """Validate grid structure and content"""
    assert isinstance(grid, list), f"{context}: Must be list"
    assert len(grid) > 0, f"{context}: Must be non-empty"
    assert len(grid) <= 30, f"{context}: Max 30 rows"
    
    row_length = len(grid[0])
    assert row_length > 0, f"{context}: Rows must be non-empty"
    assert row_length <= 30, f"{context}: Max 30 columns"
    
    for i, row in enumerate(grid):
        assert isinstance(row, list), f"{context}[{i}]: Row must be list"
        assert len(row) == row_length, f"{context}[{i}]: Inconsistent row length"
        
        for j, cell in enumerate(row):
            assert isinstance(cell, int), f"{context}[{i}][{j}]: Must be integer"
            assert 0 <= cell <= 9, f"{context}[{i}][{j}]: Must be 0-9"
```

## Example Tasks

### Example 1: Simple Color Fill
```json
{
  "train": [
    {
      "input": [
        [0, 0, 0],
        [0, 1, 0], 
        [0, 0, 0]
      ],
      "output": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
      ]
    },
    {
      "input": [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
      ],
      "output": [
        [2, 2, 2, 2],
        [2, 2, 2, 2], 
        [2, 2, 2, 2],
        [2, 2, 2, 2]
      ]
    }
  ],
  "test": [
    {
      "input": [
        [0, 0, 0],
        [0, 3, 0],
        [0, 0, 0]
      ]
    }
  ]
}
```

### Example 2: Pattern Mirroring
```json
{
  "train": [
    {
      "input": [
        [1, 0],
        [0, 0]
      ],
      "output": [
        [1, 1],
        [1, 1]
      ]
    },
    {
      "input": [
        [2, 0],
        [3, 0]
      ],
      "output": [
        [2, 2],
        [3, 3]
      ]
    }
  ],
  "test": [
    {
      "input": [
        [4, 0],
        [5, 0]
      ]
    }
  ]
}
```

## Loading and Processing

### Python Loading Example
```python
import json
from pathlib import Path

def load_arc_dataset(dataset_path):
    """Load complete ARC dataset"""
    tasks = {}
    dataset_path = Path(dataset_path)
    
    for task_file in dataset_path.glob("*.json"):
        task_id = task_file.stem
        with open(task_file, 'r') as f:
            tasks[task_id] = json.load(f)
    
    return tasks

def load_single_task(task_path):
    """Load single ARC task"""
    with open(task_path, 'r') as f:
        return json.load(f)

# Usage examples
training_tasks = load_arc_dataset("data/training/")
evaluation_tasks = load_arc_dataset("data/evaluation/public/")
single_task = load_single_task("data/training/task_00001.json")
```

### Grid Manipulation Utilities
```python
import numpy as np

def grid_to_numpy(grid):
    """Convert grid to numpy array"""
    return np.array(grid, dtype=np.int32)

def numpy_to_grid(array):
    """Convert numpy array to grid"""
    return array.astype(int).tolist()

def get_grid_dimensions(grid):
    """Get grid dimensions (height, width)"""
    return len(grid), len(grid[0]) if grid else 0

def visualize_grid(grid, title="Grid"):
    """Simple text visualization"""
    print(f"\n{title}:")
    for row in grid:
        print(' '.join(str(cell) for cell in row))
```

## Submission Format

### Submission Structure
Competition submissions must follow this exact format:

```json
{
  "task_00001": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ],
  "task_00002": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ]
  // ... for all evaluation tasks
}
```

### Submission Requirements
- **All Tasks**: Must include predictions for ALL evaluation tasks
- **Two Attempts**: Exactly 2 attempts per task required
- **Grid Format**: Same format as dataset (2D integer arrays)
- **Valid Values**: Only integers 0-9 allowed
- **Matching Dimensions**: Output dimensions should match expected output

## Common Issues and Solutions

### Validation Errors
1. **Inconsistent Row Lengths**: Ensure all rows in a grid have same length
2. **Invalid Color Values**: Only use integers 0-9
3. **Missing Fields**: Include both 'input' and 'output' where required
4. **Wrong Array Nesting**: Grids must be 2D arrays (list of lists)

### Loading Problems
1. **File Encoding**: Use UTF-8 encoding for JSON files
2. **Path Issues**: Use absolute paths or proper relative paths
3. **Memory Usage**: Large datasets may require streaming for memory efficiency
4. **JSON Errors**: Validate JSON syntax before processing

### Performance Optimization
1. **Batch Loading**: Load multiple files simultaneously
2. **Caching**: Cache processed data to avoid recomputation
3. **Validation**: Validate once and cache results
4. **Memory Management**: Use generators for large datasets

## Version History

### Version 2.0 (ARC-AGI-2)
- Enhanced difficulty compared to ARC-AGI-1
- Same format specification maintained for compatibility
- Additional validation rules for competition use
- Improved documentation and examples

### Version 1.0 (ARC-AGI-1)  
- Original format specification
- Basic validation rules
- Initial dataset structure

This specification ensures consistent data handling across all ARC Prize 2025 competition participants and research applications.