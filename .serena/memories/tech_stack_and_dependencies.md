# ARC Prize 2025 - Technology Stack

## Primary Language
- **Python 3** - All code written in Python

## Core Dependencies
Based on imports found in the codebase:

### Standard Library
- `json` - ARC task data format handling
- `os` - Operating system interface
- `pathlib.Path` - File path management
- `typing` - Type hints (Dict, List, Any, Optional)
- `sys` - System-specific parameters
- `subprocess` - Process execution
- `re` - Regular expressions
- `urllib.parse` - URL parsing

### Third-Party Libraries
- `matplotlib.pyplot` - Task visualization and plotting
- `numpy` - Numerical operations on grids
- `requests` - HTTP requests for data fetching

### Kaggle Integration
- Uses Kaggle API for competition data access
- Kaggle credentials embedded for API access
- Competition ID: arc-prize-2025

## Project Structure
```
arc-prize-2025/
├── data/
│   ├── task_loader.py          # Main utility for loading ARC tasks
│   ├── sample_task_001.json    # Example task data
│   ├── sample_task_002.json    # Example task data
│   ├── submission_template.json # Submission format template
│   └── dataset_metadata.json   # Dataset statistics
├── kaggle_processor.py         # Competition data processor
├── description.md              # Competition overview
├── data_description.md         # Dataset format specification
├── rules.md                   # Competition rules
└── linked_content_md/         # External documentation
```

## Development Environment
- Platform: Windows (win32)
- IDE: Claude Code with Serena MCP tools
- No traditional package management files (requirements.txt, pyproject.toml) found
- Dependencies appear to be managed manually