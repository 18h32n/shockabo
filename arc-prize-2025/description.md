# ARC Prize 2025: Create an AI Capable of Novel Reasoning

## Competition Overview

The ARC Prize 2025 is a $1M+ Kaggle competition aimed at creating an AI capable of novel reasoning to achieve artificial general intelligence (AGI). The competition runs from March 26 to November 3, 2025, with the ambitious goal of reaching **85% accuracy** on the ARC-AGI-2 private evaluation dataset.

## The Challenge

The Abstract and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) was introduced by François Chollet in his influential 2019 paper "On the Measure of Intelligence". Unlike traditional AI benchmarks that test accumulated knowledge, ARC-AGI tests **fluid intelligence** - the ability to efficiently acquire new skills outside of training data.

### What Makes ARC-AGI Unique

- **Intelligence Redefined**: AGI is measured as "a system that can efficiently acquire new skills outside of its training data"
- **Universal Cognitive Primitives**: Uses only core knowledge available to all humans (object cohesion, goal-directedness, counting, etc.)
- **Easy for Humans, Hard for AI**: Tasks that humans solve effortlessly (~73-77% average) but current AI finds extremely challenging (53% best AI performance)
- **Novel Reasoning Required**: Each task requires understanding abstract patterns and applying them to new situations

## Dataset Structure: ARC-AGI-2

The 2025 competition uses the new **ARC-AGI-2 dataset**, which is significantly more challenging than the original ARC-AGI-1:

- **Public Training Set**: 1,000 tasks for algorithm development
- **Public Evaluation Set**: 120 tasks for performance testing
- **Semi-Private Evaluation Set**: 120 tasks for leaderboard rankings
- **Private Evaluation Set**: 120 tasks for final competition scoring

### Task Format

Each ARC-AGI task consists of:
- **Training Examples**: 3-5 input/output grid pairs demonstrating a pattern
- **Test Input**: A single grid requiring output prediction
- **Goal**: Generate the correct output grid with 100% pixel-perfect accuracy

Example task structure:
```json
{
  "train": [
    {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
    {"input": [[0, 0], [4, 0]], "output": [[4, 4], [4, 4]]},
    {"input": [[0, 0], [6, 0]], "output": [[6, 6], [6, 6]]}
  ],
  "test": [
    {"input": [[0, 0], [0, 8]], "output": "PREDICT_THIS"}
  ]
}
```

## Prize Structure: $825,000+ Total

### Grand Prize: $700,000
- Awarded to **top 5 teams** scoring at least 85% on private evaluation set
- Increased by $100,000 from 2024
- **If not achieved, rolls over to next year**

### Paper Awards: $75,000
- **1st Place**: $50,000
- **2nd Place**: $20,000  
- **3rd Place**: $5,000
- Evaluated on: Accuracy, Universality, Progress, Theory, Completeness, Novelty

### Top Score Awards: $50,000
- **1st Place**: $25,000
- **2nd-5th Place**: $5,000 each
- Based on highest scores during competition

### Additional Prizes: $175,000
- Reserved for potential additional prizes to be announced

## Technical Requirements

### Submission Constraints
- **Runtime Limit**: Maximum 12 hours (CPU or GPU combined)
- **Hardware**: Kaggle L4x4 GPUs with 96GB GPU memory
- **Internet Access**: None during evaluation
- **External Data**: Pre-trained models and freely available data allowed
- **Output Format**: submission.json file

### Scoring Methodology
- Evaluated on percentage of correct predictions on private set
- **2 prediction attempts per task**
- **100% pixel-perfect match required** for scoring
- Final score = correct predictions / total task outputs

## Key 2025 Competition Changes

### Major Updates
- **ARC-AGI-2 Dataset**: More difficult tasks than ARC-AGI-1
- **Enhanced Open Source Requirements**: Teams must open source solutions before seeing final scores
- **Improved Leaderboard**: Semi-private scoring with one-time private evaluation
- **Doubled Compute**: L4x4 GPUs (~$50 value) vs previous P100s
- **Overfitting Prevention**: Additional measures to reduce data mining and encourage conceptual progress

## Open Source Requirements

### Licensing Requirements
- **Submitter Code**: Must be public domain (CC0 or MIT-0)
- **Third-party Code**: Must allow public sharing (Apache-2.0, GPLv3+)
- **Pre-submission Requirement**: Teams must open source before seeing final scores

### Competition Philosophy
The primary mission is to **accelerate open AGI progress** by making cutting-edge solutions freely available to the research community. All leading participants are expected to open source their solutions if prize-eligible.

## Successful Approaches from Previous Years

### 1. Discrete Program Search
- First approach to achieve meaningful success
- Searches through massive program space systematically
- Good baseline performance but computationally intensive

### 2. Ensemble Solutions
- Combines multiple existing solutions
- Current approach for highest scores
- Limited generalization to unseen tasks

### 3. Domain-Specific Language (DSL) Program Synthesis
- Develops specialized language for grid transformations
- Encapsulates common concepts (rotation, mirroring, pattern completion)
- Synthesizes programs by composing primitives

### 4. Active Inference with LLMs
- Fine-tunes LLMs on task demonstration examples
- Expands limited examples artificially
- Achieved current state-of-the-art (34% accuracy)

### Promising Future Approach (François Chollet's Recommendation)
**Hybrid: Discrete Program Search + Deep Learning Intuition**
- Use deep learning to guide program search
- Leverage LLM intuition to prune search space
- Combines symbolic reasoning with neural guidance

## Competition Strategy

### Getting Started
1. **Study ARC-AGI-1**: Start with simpler dataset for learning
2. **Use Visualization Tools**: 
   - Official ARC-AGI testing interface
   - arcprize.org task viewer
   - Community-created apps
3. **Join Community Resources**:
   - Discord server: https://discord.gg/9b77dPAmcA
   - Kaggle discussions
   - Newsletter subscription

### Development Tips
1. **Focus on Skill Acquisition**: Emphasize generalization over memorization
2. **Human-Inspired Approaches**: Study cognitive science and developmental psychology
3. **Hybrid Methods**: Combine symbolic and neural approaches
4. **Start Small**: Build narrow solutions and scale up
5. **Novel Ideas**: Focus on genuinely new approaches over incremental improvements

## Historical Performance Context

### Performance Evolution
- **2020**: 21% success rate (ice cuber team)
- **2022**: ARCathon with 118 teams from 47 countries
- **2023**: 30% success rate (teams SM and MindsAI)
- **2024**: 53% top score on private evaluation
- **2025**: Target of 85% for Grand Prize

### Human vs. AI Performance
- **Human Performance**: 73.3% - 77.2% average
- **Best AI Performance**: 53% (2024)
- **Target AI Performance**: 85% (2025 Grand Prize threshold)
- **Gap to Close**: 32 percentage points minimum

## Critical Success Factors

### To Win the Grand Prize ($700,000)
1. **Achieve 85%+ accuracy** on ARC-AGI-2 private evaluation set
2. **Stay within efficiency limits** (~$0.42/task computational budget)
3. **Open source your solution** before receiving final scores
4. **Submit by November 3, 2025**

### Key Challenges
- **No Internet Access**: Solutions must work completely offline
- **Limited Compute**: Must work within Kaggle L4x4 constraints
- **Novel Reasoning**: Tasks designed to be unseen and unpredictable
- **Perfect Accuracy**: Requires 100% pixel-perfect matches

## Timeline

- **March 26, 2025**: Competition Launch
- **November 3, 2025**: Final Submission Deadline  
- **November 9, 2025**: Paper Submission Deadline
- **December 5, 2025**: Winners Announced

## Resources

### Official Documentation
- **Competition Details**: https://arcprize.org/competitions/2025/
- **Official Guide**: https://arcprize.org/guide
- **ARC-AGI Overview**: https://arcprize.org/arc-agi
- **Kaggle Competition**: https://www.kaggle.com/competitions/arc-prize-2025

### Research Papers
- **"On the Measure of Intelligence"**: https://arxiv.org/abs/1911.01547
- **2024 NYU Human Performance Study**: https://arxiv.org/abs/2409.01374

This competition represents one of the most ambitious challenges in AI research, requiring breakthrough advances in artificial general intelligence to achieve the target 85% accuracy threshold.