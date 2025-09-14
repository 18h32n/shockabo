# ARC Prize 2025 Competition Details

**Source:** https://arcprize.org/competitions/2025/  
**Type:** Official Competition Documentation  
**Last Updated:** 2025-08-30  
**Authority:** ARC Prize Organization  

## Competition Overview

The ARC Prize 2025 represents the most ambitious artificial intelligence competition ever created, challenging teams worldwide to develop AI systems capable of novel reasoning to achieve artificial general intelligence. With over $825,000 in total prizes, this competition aims to accelerate open AGI research by incentivizing breakthrough advances in machine reasoning capabilities.

## Competition Mission

### Primary Goals
1. **Advance AGI Research**: Push the boundaries of artificial general intelligence
2. **Open Source Innovation**: Make cutting-edge solutions freely available
3. **Community Building**: Foster global collaboration in AI research
4. **Benchmark Progress**: Establish clear metrics for AGI development

### Success Definition
Success is measured by achieving **85% accuracy** on the ARC-AGI-2 private evaluation dataset - a threshold that represents human-level performance on abstract reasoning tasks requiring novel skill acquisition outside training data.

## Prize Structure: $825,000+ Total

### Grand Prize: $700,000
- **Amount**: $700,000 total (increased from $600,000 in 2024)
- **Recipients**: Top 5 teams achieving ≥85% accuracy on private evaluation
- **Distribution**: Equal split among qualifying teams
- **Rollover Policy**: If no team reaches 85%, prize rolls over to 2026

### Paper Awards: $75,000 Total
Awarded based on submitted research papers evaluated on multiple criteria:

#### Prize Distribution
- **1st Place**: $50,000
- **2nd Place**: $20,000
- **3rd Place**: $5,000

#### Evaluation Criteria
1. **Accuracy**: Performance on ARC-AGI-2 benchmark
2. **Universality**: Generalizability beyond ARC tasks
3. **Progress**: Advancement over previous approaches
4. **Theory**: Theoretical contributions to AGI understanding
5. **Completeness**: Thoroughness of solution and analysis
6. **Novelty**: Innovation and originality of approach

### Top Score Awards: $50,000 Total
Based purely on highest accuracy scores during competition:

- **1st Place**: $25,000
- **2nd Place**: $5,000
- **3rd Place**: $5,000
- **4th Place**: $5,000
- **5th Place**: $5,000

### Additional Prizes: $175,000
Reserved for potential additional prizes to be announced, which may include:
- Innovation awards for novel approaches
- Community contribution recognition
- Special achievement categories
- Educational and outreach initiatives

## Timeline and Key Dates

### Competition Schedule
- **Launch Date**: March 26, 2025
- **Development Period**: March 26 - November 3, 2025 (7+ months)
- **Submission Deadline**: November 3, 2025 (11:59 PM UTC)
- **Paper Submission**: November 9, 2025 (11:59 PM UTC)
- **Winners Announced**: December 5, 2025

### Important Milestones
- **Q2 2025**: Mid-competition progress assessments
- **September 2025**: Final preparation period begins
- **October 2025**: Open source submission requirement reminders
- **November 2025**: Final submissions and evaluation period
- **December 2025**: Results announcement and prize distribution

## Technical Requirements

### Performance Target
- **Accuracy Threshold**: 85% on ARC-AGI-2 private evaluation dataset
- **Task Count**: 120 private evaluation tasks
- **Scoring Method**: Percentage of tasks solved correctly
- **Attempts**: 2 prediction attempts per task allowed

### Platform Constraints
- **Environment**: Kaggle competition platform
- **Hardware**: L4x4 GPUs with 96GB GPU memory (doubled from 2024)
- **Runtime**: Maximum 12 hours total (CPU + GPU combined)
- **Internet**: No internet access during evaluation
- **Storage**: Limited to Kaggle environment constraints

### Submission Format
```json
{
  "task_001": [
    {
      "attempt_1": [[grid_data]],
      "attempt_2": [[grid_data]]
    }
  ]
  // ... for all 120 private evaluation tasks
}
```

## ARC-AGI-2 Dataset

### Dataset Structure
- **Training Set**: 1,000 tasks (fully available)
- **Public Evaluation**: 120 tasks (for local testing)
- **Semi-Private Evaluation**: 120 tasks (for leaderboard)
- **Private Evaluation**: 120 tasks (for final scoring)

### Key Enhancements Over ARC-AGI-1
1. **Increased Difficulty**: More challenging reasoning requirements
2. **Compositional Complexity**: Multi-step reasoning tasks
3. **Overfitting Prevention**: Measures to discourage memorization
4. **Abstract Patterns**: Require deeper conceptual understanding
5. **Novel Combinations**: Avoid easily memorizable patterns

### Task Characteristics
- **Format**: JSON files with input/output grid pairs
- **Grid Representation**: 2D integer arrays (0-9 representing colors)
- **Training Examples**: 3-5 demonstration pairs per task
- **Test Format**: Single input requiring output prediction
- **Pixel-Perfect**: 100% accuracy required for scoring

## Open Source Requirements (CRITICAL)

### Mandatory Open Source Policy
- **Timing**: Teams must open source solutions BEFORE seeing final scores
- **Verification**: Kaggle verifies compliance before final evaluation
- **Disqualification**: Failure to open source disqualifies from ALL prizes
- **No Exceptions**: Policy applies to all prize categories

### Licensing Requirements
- **Submitter Code**: Must use public domain licenses (CC0 or MIT-0)
- **Third-Party Code**: Must allow public sharing (Apache-2.0, GPLv3+, etc.)
- **Documentation**: Comprehensive README and usage instructions required
- **Reproducibility**: Code must be sufficient to reproduce results

### Required Content
1. **Complete Solution**: All code used in final submission
2. **Model Weights**: Any trained models or parameters
3. **Training Code**: Development and training procedures
4. **Data Processing**: Preprocessing and analysis scripts
5. **Dependencies**: Clear specification of all requirements

## Historical Context and Evolution

### Performance Progression
- **2020**: 21% accuracy (ice cuber team - first major success)
- **2022**: ARCathon event with 118 teams from 47 countries
- **2023**: 30% accuracy achieved by teams SM and MindsAI
- **2024**: 53% accuracy reached by top-performing team
- **2025**: Target of 85% for grand prize achievement

### Human Performance Baseline
- **Expert Performance**: 85-95% accuracy
- **Average Human**: 73.3% - 77.2% accuracy
- **Task Solvability**: 98.7% of public tasks solvable by at least one human
- **Performance Gap**: Current 32+ point gap between best AI and target

### Competition Evolution
- **Prize Increases**: $600K (2024) → $825K+ (2025)
- **Enhanced Dataset**: ARC-AGI-1 → ARC-AGI-2
- **Improved Evaluation**: Better overfitting prevention
- **Doubled Resources**: P100 GPUs → L4x4 GPUs
- **Stronger Open Source**: Enhanced requirements for sharing

## Successful Approaches and Strategies

### Current Leading Methods

#### 1. Discrete Program Search
- **Description**: Exhaustive search through program space
- **Strengths**: Systematic exploration of transformations
- **Limitations**: Computationally intensive, limited abstraction
- **Performance**: 15-25% accuracy typical

#### 2. Ensemble Solutions
- **Description**: Combine multiple complementary approaches
- **Strengths**: Leverages diverse solution methods
- **Limitations**: Limited generalization beyond training patterns
- **Performance**: Current best approach (50%+ accuracy)

#### 3. Domain-Specific Language (DSL) Program Synthesis
- **Description**: Specialized language for grid transformations
- **Strengths**: Encodes human-like reasoning patterns
- **Limitations**: Requires extensive domain knowledge
- **Performance**: 20-35% accuracy range

#### 4. Active Inference with LLMs
- **Description**: Fine-tune language models on ARC demonstrations
- **Strengths**: Leverages pre-trained knowledge
- **Limitations**: Constrained by no-internet requirement
- **Performance**: 10-30% accuracy achieved

### Recommended Future Approach (François Chollet)
**Hybrid: Discrete Program Search + Deep Learning Intuition**
- Combine systematic search with neural guidance
- Use deep learning to prune search space intelligently
- Balance symbolic reasoning with pattern recognition
- Most promising path toward 85% target

## Technical Innovation Requirements

### Breakthrough Areas Needed
1. **Abstract Reasoning**: Beyond pattern matching to principle extraction
2. **Few-Shot Learning**: Learn from 3-5 examples effectively
3. **Compositional Understanding**: Combine multiple reasoning steps
4. **Transfer Learning**: Apply learned concepts to novel situations
5. **Efficient Search**: Navigate vast solution spaces intelligently

### Key Technical Challenges
- **No Internet Access**: Solutions must work completely offline
- **Limited Compute**: Must operate within 12-hour constraint
- **Perfect Accuracy**: 100% pixel-perfect matching required
- **Novel Tasks**: Each task designed to require fresh reasoning
- **Resource Efficiency**: ~$0.42 per task computational budget

## Community and Support Resources

### Official Channels
- **Website**: https://arcprize.org/
- **Competition Platform**: https://www.kaggle.com/competitions/arc-prize-2025
- **GitHub**: https://github.com/arcprize/ARC-AGI-2
- **Email**: team@arcprize.org

### Community Platforms
- **Discord Server**: https://discord.gg/9b77dPAmcA (active community)
- **Kaggle Discussions**: Competition-specific forum
- **Research Papers**: Academic publications and preprints
- **Social Media**: @arcprize on Twitter for updates

### Available Resources
- **Official Guide**: Comprehensive participation documentation
- **Baseline Solutions**: Starting point implementations
- **Visualization Tools**: Task analysis and development aids
- **Community Contributions**: Shared solutions and insights

## Impact and Vision

### Immediate Goals (2025)
- Achieve 85% accuracy threshold for first time in competition history
- Demonstrate breakthrough in artificial general intelligence capabilities
- Create substantial open source repository of AGI research
- Establish new standard for AI reasoning evaluation

### Long-term Vision
- **AGI Advancement**: Accelerate progress toward artificial general intelligence
- **Open Research**: Maintain commitment to open source innovation
- **Global Collaboration**: Build international research community
- **Practical Applications**: Bridge research to real-world AGI systems

### Success Indicators
- Teams achieving 85%+ accuracy (grand prize qualification)
- Innovative approaches advancing beyond current 53% best
- Strong community engagement and collaboration
- Significant open source contributions to AGI research

## Participation Recommendations

### For New Participants
1. **Start with Basics**: Understand ARC-AGI task structure and principles
2. **Study Baselines**: Analyze existing solutions for learning
3. **Use Tools**: Leverage visualization and analysis utilities
4. **Join Community**: Engage with Discord and Kaggle discussions
5. **Plan Early**: Allow time for iterative development and testing

### For Experienced Teams
1. **Novel Approaches**: Focus on breakthrough methods beyond incremental improvements
2. **Hybrid Strategies**: Combine multiple complementary techniques
3. **Efficiency Optimization**: Balance accuracy with computational constraints
4. **Open Source Planning**: Prepare for public release requirements
5. **Community Contribution**: Share insights and collaborate with others

### Critical Success Factors
- **Innovation**: Develop genuinely novel approaches to reasoning
- **Persistence**: Expect multiple iterations and continuous improvement
- **Collaboration**: Engage with community for shared learning
- **Preparation**: Plan for open source requirements from the beginning
- **Focus**: Maintain focus on the core challenge of abstract reasoning

The ARC Prize 2025 represents humanity's most ambitious attempt to create artificial general intelligence through competitive innovation, community collaboration, and open source advancement. Success will require breakthrough thinking, technical excellence, and commitment to shared progress in AGI research.