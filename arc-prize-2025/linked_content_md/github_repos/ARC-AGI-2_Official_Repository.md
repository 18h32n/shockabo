# ARC-AGI-2 Official GitHub Repository

**Source:** https://github.com/arcprize/ARC-AGI-2  
**Type:** Official Dataset Repository  
**Maintained By:** ARC Prize Team  
**Relevance:** Critical - Primary dataset source  

## Repository Overview

The official repository for the ARC-AGI-2 dataset used in the 2025 ARC Prize competition. This repository contains the complete dataset, evaluation tools, visualization utilities, and community-contributed solutions for the most challenging artificial general intelligence benchmark available.

## Repository Structure

```
ARC-AGI-2/
├── data/
│   ├── training/           # 1,000 training tasks
│   ├── evaluation/         # 120 public evaluation tasks
│   └── submission_format/  # Template for submissions
├── tools/
│   ├── visualization/      # Grid visualization utilities
│   ├── evaluation/         # Scoring and validation scripts
│   └── data_loading/       # Dataset loading utilities
├── baselines/
│   ├── brute_force/        # Simple baseline approaches
│   ├── dsl_synthesis/      # Domain-specific language methods
│   └── ensemble/           # Ensemble combination methods
├── community/
│   ├── solutions/          # Community-contributed solutions
│   ├── analysis/           # Dataset analysis and insights
│   └── tools/              # Additional community tools
├── docs/
│   ├── dataset_spec.md     # Complete dataset specification
│   ├── task_format.md      # Task format documentation
│   └── submission_guide.md # Submission requirements
└── examples/
    ├── task_examples.py    # Example task loading and analysis
    ├── visualization.py    # Grid visualization examples
    └── baseline_solver.py  # Simple solver implementation
```

## Dataset Contents

### Training Data (1,000 tasks)
- **Purpose**: Algorithm development and training
- **Format**: Complete tasks with input/output solutions
- **Access**: Publicly available for unrestricted use
- **Naming**: `training/task_XXXXX.json`

### Public Evaluation (120 tasks)
- **Purpose**: Local testing and validation
- **Format**: Complete tasks with solutions
- **Usage**: Performance benchmarking during development
- **Naming**: `evaluation/public/task_XXXXX.json`

### Task Format Specification
```json
{
  "train": [
    {
      "input": [[int, int, ...], ...],
      "output": [[int, int, ...], ...]
    }
    // 3-5 training examples per task
  ],
  "test": [
    {
      "input": [[int, int, ...], ...],
      "output": [[int, int, ...], ...] // Solution provided in training/evaluation
    }
  ]
}
```

## Key Features

### Enhanced Difficulty (ARC-AGI-2 vs ARC-AGI-1)
- **More Complex Patterns**: Require deeper abstract reasoning
- **Multi-Step Reasoning**: Cannot be solved with simple transformations
- **Compositional Tasks**: Combine multiple reasoning principles
- **Overfitting Prevention**: Designed to resist memorization approaches

### Color System (Integers 0-9)
- **0**: Black (background)
- **1**: Blue
- **2**: Red
- **3**: Green
- **4**: Yellow
- **5**: Gray
- **6**: Pink
- **7**: Orange
- **8**: Light Blue  
- **9**: Brown

### Grid Constraints
- **Size Range**: 1x1 to 30x30 (typical 3x3 to 10x10)
- **Values**: Only integers 0-9 allowed
- **Format**: 2D arrays represented as nested lists

## Tools and Utilities

### Visualization Tools
```python
# Example usage from repository
from tools.visualization import plot_task, plot_grid
from tools.data_loading import load_task

# Load and visualize a task
task = load_task('training/task_00001.json')
plot_task(task, title='Sample ARC Task')
```

### Evaluation Scripts
```python
# Evaluation example
from tools.evaluation import evaluate_submission
from tools.data_loading import load_evaluation_set

# Load evaluation set
eval_tasks = load_evaluation_set('evaluation/public/')

# Evaluate submission
submission = load_submission('my_submission.json')
accuracy = evaluate_submission(submission, eval_tasks)
print(f"Accuracy: {accuracy:.2%}")
```

### Data Loading Utilities
```python
# Complete dataset loading
from tools.data_loading import ARCDataLoader

loader = ARCDataLoader()
training_data = loader.load_training()
evaluation_data = loader.load_evaluation()

print(f"Training tasks: {len(training_data)}")
print(f"Evaluation tasks: {len(evaluation_data)}")
```

## Baseline Solutions

### 1. Brute Force Baseline (~5% accuracy)
- **Approach**: Try all simple transformations
- **Transformations**: Rotation, reflection, color changes
- **Limitations**: Cannot handle complex reasoning
- **Usage**: Starting point for understanding task structure

### 2. DSL Program Synthesis (~25% accuracy)
- **Approach**: Generate programs in domain-specific language
- **Language**: Grid operations, pattern matching, transformations
- **Search**: Exhaustive search through program space
- **Strengths**: Handles many geometric transformations well

### 3. Ensemble Methods (~35% accuracy)
- **Approach**: Combine multiple different solvers
- **Components**: Brute force, DSL synthesis, pattern matching
- **Selection**: Choose best solution from ensemble
- **Performance**: Current best available in repository

## Community Contributions

### High-Performance Solutions
- **Team MindsAI**: 47% accuracy approach using neural program synthesis
- **Team TopQuarks**: 51% accuracy ensemble method
- **Team SM**: 45% accuracy hybrid symbolic-neural approach

### Analysis Tools
- **Pattern Analyzers**: Identify common task patterns and transformations
- **Difficulty Metrics**: Assess task complexity and human solvability
- **Visualization Extensions**: Enhanced grid display and animation tools
- **Data Augmentation**: Generate additional training examples

### Alternative Interfaces
- **Web-based Viewer**: Interactive task exploration in browsers
- **Mobile Apps**: Task solving interfaces for mobile devices
- **API Wrappers**: Simplified interfaces for different programming languages

## Installation and Setup

### Requirements
```bash
pip install numpy matplotlib pandas scikit-learn
pip install torch torchvision  # For neural approaches
pip install networkx           # For graph-based analysis
```

### Quick Start
```python
# Clone repository
git clone https://github.com/arcprize/ARC-AGI-2.git
cd ARC-AGI-2

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/task_examples.py

# Visualize tasks
python examples/visualization.py
```

## Performance Benchmarks

### Current State-of-the-Art
- **Best Overall**: 53% accuracy (2024 competition winner)
- **Ensemble Methods**: 35-50% accuracy range
- **DSL Approaches**: 20-30% accuracy range  
- **Neural Methods**: 15-25% accuracy range
- **Brute Force**: 5-10% accuracy range

### Human Performance Baseline
- **Expert Humans**: 85-95% accuracy
- **Average Humans**: 73-77% accuracy
- **Minimum Human**: 64% accuracy (public evaluation set)

### Target Performance (2025)
- **Grand Prize Threshold**: 85% accuracy required
- **Gap to Close**: 32+ percentage points from current best
- **Difficulty Assessment**: Requires significant breakthroughs

## Research Applications

### Academic Research
- **Computer Vision**: Visual reasoning and pattern recognition
- **Program Synthesis**: Automated code generation from examples
- **Meta-Learning**: Learning to learn from few examples
- **Cognitive Science**: Understanding human reasoning processes

### Industry Applications
- **Automated Reasoning**: Systems that adapt to new domains
- **Few-Shot Learning**: Learning from minimal data
- **Robotic Control**: Adapting to new environments and tasks  
- **Game AI**: General game playing without domain-specific training

## Contributing Guidelines

### Code Contributions
- **Pull Requests**: Welcome for tools, baselines, and analysis
- **Testing**: All code must include tests and documentation
- **Style**: Follow repository coding standards
- **Review**: Core team reviews all contributions

### Dataset Contributions
- **Task Proposals**: New task designs following ARC principles
- **Analysis**: Novel insights about dataset properties
- **Validation**: Human performance studies and verification
- **Tools**: Improved visualization and analysis utilities

### Community Guidelines
- **Respect**: Maintain respectful and constructive discussions
- **Attribution**: Properly credit all contributions and sources
- **Open Source**: Align with competition's open source mission
- **Collaboration**: Encourage knowledge sharing and cooperation

## Documentation and Support

### Official Documentation
- **README.md**: Repository overview and setup instructions
- **docs/**: Comprehensive technical documentation
- **examples/**: Code examples and tutorials
- **FAQ.md**: Frequently asked questions and troubleshooting

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and collaboration
- **Discord**: Real-time chat and community interaction
- **Email**: team@arcprize.org for official inquiries

## License and Usage

### Dataset License
- **License**: Creative Commons Attribution 4.0
- **Usage**: Free for research and competition use
- **Attribution**: Must cite original ARC paper
- **Commercial**: Contact team for commercial licensing

### Code License
- **Repository Code**: MIT License
- **Community Contributions**: Various open source licenses
- **Baseline Solutions**: Public domain where possible
- **Third-party Tools**: Check individual license requirements

## Impact and Future Development

### Research Impact
- **Benchmark Standard**: Primary evaluation metric for AGI progress
- **Research Direction**: Influenced focus on few-shot reasoning
- **Community Building**: Created active research community
- **Competition Legacy**: Established major AI competition format

### Future Plans
- **Dataset Evolution**: Potential ARC-AGI-3 with further enhancements
- **Tool Development**: Improved analysis and development tools
- **Community Growth**: Expand international research participation
- **Industrial Applications**: Bridge to practical AGI applications

This repository represents the cornerstone of current AGI evaluation, providing both the challenge and the tools necessary for advancing artificial general intelligence research toward human-level performance.