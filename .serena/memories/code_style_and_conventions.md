# ARC Prize 2025 - Code Style and Conventions

## Code Organization
- **Modular Structure**: Main utilities in `data/task_loader.py`
- **Class-Based Design**: `ARCTaskLoader` class encapsulates task operations
- **Function Separation**: Standalone convenience functions provided

## Python Style Conventions

### Imports
- Standard library imports first
- Third-party imports second  
- Absolute imports preferred
- Type hints imported from `typing` module

### Naming Conventions
- **Classes**: PascalCase (`ARCTaskLoader`)
- **Methods/Functions**: snake_case (`load_task`, `visualize_task`)
- **Variables**: snake_case (`data_path`, `color_map`)
- **Constants**: UPPER_CASE (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
- **Private Methods**: Leading underscore (`_validate_task`, `_setup_credentials`)

### Type Hints
- Comprehensive type annotations used
- `Dict[str, Any]` for task data structures
- `Optional[str]` for nullable parameters
- `List` and `Dict` from typing module

### Documentation
- **Docstrings**: Triple-quoted strings for all classes and methods
- **File Headers**: Module-level docstrings explaining purpose
- **Inline Comments**: Descriptive comments for complex logic

### File Organization
- Shebang lines: `#!/usr/bin/env python3`
- Module docstring at top
- Imports grouped and separated
- Classes before standalone functions
- `if __name__ == "__main__":` pattern for script execution

### Error Handling
- Specific exception types (`FileNotFoundError`)
- Descriptive error messages with context
- Input validation in methods (`_validate_task`, `_validate_example`)

### Data Structures
- Dictionaries for task data (JSON compatibility)
- Lists for grid data
- Path objects for file system operations
- Color mapping dictionary for visualization

## Visualization Standards  
- Matplotlib for task visualization
- Consistent color mapping (0=black, 1=blue, etc.)
- Grid-based plotting with proper aspect ratios
- Titles and labels for clarity