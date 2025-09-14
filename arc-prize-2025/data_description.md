# ARC-AGI-2 Dataset Description

## Overview

The ARC-AGI-2 dataset is the foundation of the ARC Prize 2025 competition. This dataset represents a significant advancement over the original ARC-AGI-1, featuring more challenging tasks designed to test artificial general intelligence through novel reasoning capabilities.

## Dataset Structure

### File Organization

The ARC-AGI-2 dataset is organized into several key components:

```
arc-agi-2-dataset/
├── training/
│   ├── task_001.json
│   ├── task_002.json
│   ├── ...
│   └── task_1000.json
├── evaluation/
│   ├── public/
│   │   ├── task_001.json
│   │   ├── ...
│   │   └── task_120.json
│   ├── semi_private/
│   │   ├── task_001.json
│   │   ├── ...
│   │   └── task_120.json
│   └── private/
│       ├── task_001.json
│       ├── ...
│       └── task_120.json
├── submission_template.json
├── dataset_metadata.json
└── task_visualization_tools/
```

### Dataset Splits

#### 1. Public Training Set (1,000 tasks)
- **Purpose**: Algorithm development and training
- **Access**: Fully available with input/output solutions
- **Usage**: Teams can analyze patterns, develop approaches, train models
- **File Format**: JSON files with complete task structure

#### 2. Public Evaluation Set (120 tasks)  
- **Purpose**: Performance testing and validation
- **Access**: Input grids and correct outputs available
- **Usage**: Local testing and algorithm validation
- **Scoring**: Not used for competition leaderboard

#### 3. Semi-Private Evaluation Set (120 tasks)
- **Purpose**: Competition leaderboard rankings
- **Access**: Input grids only, outputs held by Kaggle
- **Usage**: Public leaderboard scoring during competition
- **Submission Limit**: Limited daily submissions

#### 4. Private Evaluation Set (120 tasks)
- **Purpose**: Final competition ranking
- **Access**: Input grids only, outputs completely hidden
- **Usage**: One-time final scoring at competition end
- **Critical**: Determines winners and prize distribution

## Task Format Specification

### JSON Structure

Each ARC-AGI task is stored as a JSON file with the following structure:

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
    // 3-5 training examples total
  ],
  "test": [
    {
      "input": [[int, int, ...], [int, int, ...], ...],
      "output": [[int, int, ...], [int, int, ...], ...] // Hidden in evaluation sets
    }
  ]
}
```

### Grid Representation

- **Data Type**: 2D arrays of integers
- **Color Encoding**: Integers 0-9 represent different colors
  - 0 = Black (background)
  - 1 = Blue
  - 2 = Red  
  - 3 = Green
  - 4 = Yellow
  - 5 = Gray
  - 6 = Pink
  - 7 = Orange
  - 8 = Light Blue
  - 9 = Brown
- **Grid Size**: Variable dimensions, typically 3x3 to 30x30
- **Coordinates**: [row][column] indexing (top-left is [0][0])

### Example Task Breakdown

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
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
      ],
      "output": "PREDICT_THIS"
    }
  ]
}
```

**Pattern**: Fill the entire grid with the color of the single non-black pixel.

## Task Categories and Patterns

### Common Task Types

#### 1. Geometric Transformations
- Rotation (90°, 180°, 270°)
- Reflection (horizontal, vertical, diagonal)
- Scaling (enlargement, reduction)
- Translation (movement)

#### 2. Pattern Completion
- Symmetry completion
- Sequence continuation
- Missing piece identification
- Template matching

#### 3. Object Manipulation
- Object counting
- Size comparison
- Color changes
- Shape recognition

#### 4. Logical Operations
- Boolean operations (AND, OR, XOR)
- Conditional rules
- Set operations
- Constraint satisfaction

#### 5. Spatial Reasoning
- Inside/outside relationships
- Connectivity analysis
- Path finding
- Boundary detection

### Complexity Characteristics

#### ARC-AGI-2 Enhanced Difficulty
- **More Abstract Patterns**: Require deeper conceptual understanding
- **Multi-Step Reasoning**: Cannot be solved with single transformations
- **Compositional Complexity**: Combine multiple simple operations
- **Novel Combinations**: Avoid patterns easily memorized from training data

## Data Quality and Validation

### Task Creation Process
1. **Human Design**: Tasks created by human experts
2. **Multiple Validation**: Verified by multiple human solvers
3. **Difficulty Calibration**: Balanced for appropriate challenge level
4. **Pattern Uniqueness**: Each task requires novel reasoning

### Quality Metrics
- **Human Solvability**: 98.7% of public tasks solvable by at least one human
- **Average Human Performance**: 73.3% - 77.2% accuracy
- **Consistency**: Multiple humans agree on correct solutions
- **Diversity**: Wide range of reasoning types represented

## Submission Format Requirements

### Output Structure

Teams must submit predictions in the following JSON format:

```json
{
  "task_001": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ],
  "task_002": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ]
  // ... for all evaluation tasks
}
```

### Submission Rules
- **Two Attempts**: Exactly 2 prediction attempts per task required
- **Perfect Match**: 100% pixel-perfect accuracy required for scoring
- **Grid Dimensions**: Must match expected output dimensions exactly
- **Color Values**: Must use integers 0-9 only
- **All Tasks**: Must provide predictions for all evaluation tasks

## Data Loading and Processing

### Python Data Loading Example

```python
import json
import os
from pathlib import Path

def load_arc_task(file_path):
    """Load a single ARC task from JSON file"""
    with open(file_path, 'r') as f:
        task = json.load(f)
    return task

def load_arc_dataset(dataset_path):
    """Load all tasks from a dataset directory"""
    tasks = {}
    for task_file in Path(dataset_path).glob('*.json'):
        task_id = task_file.stem
        tasks[task_id] = load_arc_task(task_file)
    return tasks

def visualize_grid(grid):
    """Simple grid visualization"""
    color_map = {0: ' ', 1: '█', 2: '▓', 3: '▒', 4: '░', 
                 5: '●', 6: '○', 7: '■', 8: '□', 9: '▲'}
    for row in grid:
        print(''.join(color_map[cell] for cell in row))
```

### Data Preprocessing Recommendations

1. **Normalization**: Consider grid size normalization strategies
2. **Augmentation**: Rotation and reflection for data augmentation
3. **Encoding**: Experiment with different grid representations
4. **Visualization**: Essential for human pattern recognition
5. **Validation**: Implement pixel-perfect matching for local testing

## Performance Benchmarks

### Historical Performance
- **2020 Competition**: 21% accuracy (best team)
- **2023 Competition**: 30% accuracy (best team)  
- **2024 Competition**: 53% accuracy (best team)
- **Human Baseline**: 73.3% - 77.2% average

### 2025 Target
- **Grand Prize Threshold**: 85% accuracy
- **Gap to Close**: 32+ percentage points from current AI best
- **Challenge Level**: Requires breakthrough advances in AGI

## Tools and Resources

### Visualization Tools
- **Official ARC Testing Interface**: Interactive task solving
- **arcprize.org Task Viewer**: Web-based visualization
- **Kaggle Notebooks**: Community-created visualization tools
- **Custom Tools**: Teams often develop specialized viewers

### Development Resources
- **GitHub Repository**: https://github.com/arcprize/ARC-AGI-2
- **Documentation**: https://arcprize.org/guide
- **Community Discord**: https://discord.gg/9b77dPAmcA
- **Research Papers**: Extensive academic literature available

## Data Licensing and Usage

### Competition Usage Rights
- **Training Data**: Free to use for competition purposes
- **Analysis**: Permitted for research and development
- **Publication**: Academic use encouraged with proper citation
- **Commercial**: Consult competition rules for commercial applications

### Open Source Requirements
- **Solution Code**: Must be open sourced before final scoring
- **Data Derivatives**: Created tools and visualizations encouraged to be shared
- **Community Benefit**: Align with competition's open AGI mission

This comprehensive dataset description provides the foundation for understanding and working with the ARC-AGI-2 dataset in the ARC Prize 2025 competition.