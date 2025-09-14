# ARC Prize 2025 - Suggested Commands

## Windows System Commands
Since this is a Windows environment, use these system commands:

### File Operations
- `dir` - List directory contents (instead of `ls`)
- `type filename` - Display file contents (instead of `cat`)
- `cd directory` - Change directory
- `md directory` - Create directory (instead of `mkdir`)
- `copy source dest` - Copy files
- `move source dest` - Move/rename files
- `del filename` - Delete files

### Search Operations  
- `findstr "pattern" *.py` - Search in Python files (instead of `grep`)
- `dir /s *.json` - Find JSON files recursively

### Git Commands (if repository initialized)
- `git status` - Check repository status
- `git add .` - Stage changes
- `git commit -m "message"` - Commit changes
- `git push` - Push to remote

## Python Execution
- `python script.py` - Run Python scripts
- `python -m module` - Run Python modules
- `python -c "code"` - Execute Python code directly

## Project-Specific Commands

### Task Loading and Analysis
```bash
# Run task loader examples
python arc-prize-2025\data\task_loader.py

# Process competition data
python arc-prize-2025\kaggle_processor.py fetch arc-prize-2025
```

### Development Workflow
```bash
# Navigate to project
cd "C:\Users\Michael\CODING PROJECT\KAGGLE COMPETITIONS\ARC Prize 2025"

# Run task analysis
python arc-prize-2025\data\task_loader.py

# Check task data
type arc-prize-2025\data\sample_task_001.json
```

## No Package Manager Setup
**Important**: This project doesn't use pip/conda requirements files. Dependencies must be installed manually:

```bash
pip install matplotlib numpy requests
```

## IDE Integration
- Use Serena MCP tools for code analysis
- Use Claude Code native tools for file operations
- No specific linting/formatting commands configured yet

## Testing Commands
**Note**: No formal test framework detected. Testing appears to be manual through:
- Running example scripts in `if __name__ == "__main__":` blocks
- Visual inspection of task loading and visualization
- Manual validation of JSON data structures