# On the Measure of Intelligence

**Source:** https://arxiv.org/abs/1911.01547  
**Author:** François Chollet  
**Published:** 2019  
**Type:** Foundational Research Paper  
**Relevance:** Critical - Introduces ARC-AGI benchmark  

## Abstract

This paper introduces the Abstract and Reasoning Corpus (ARC), a new benchmark for artificial general intelligence that focuses on measuring skill-acquisition efficiency rather than skill itself. The work argues that intelligence should be defined as the ability to efficiently acquire new skills outside of one's training data, emphasizing fluid intelligence over crystallized intelligence.

## Key Contributions

### 1. Intelligence Redefinition
- **Traditional View**: Intelligence as accumulated knowledge and skills
- **Proposed View**: Intelligence as skill-acquisition efficiency  
- **Focus**: Ability to adapt to novel situations with minimal prior experience
- **Measurement**: Performance on tasks outside training distribution

### 2. ARC Benchmark Design
- **Task Structure**: Visual reasoning tasks with 3-5 demonstration examples
- **Core Knowledge**: Uses only universal human cognitive primitives
- **Primitives Used**:
  - Object cohesion and persistence
  - Goal-directedness and intentionality  
  - Numbers and counting
  - Basic geometry and topology
  - Basic physics (intuitive)

### 3. Fluid vs Crystallized Intelligence
- **Fluid Intelligence**: Pattern recognition, abstract reasoning, novel problem solving
- **Crystallized Intelligence**: Accumulated knowledge, learned skills, memorized facts
- **ARC Focus**: Tests fluid intelligence specifically
- **AI Challenge**: Current systems excel at crystallized but struggle with fluid

## ARC Benchmark Specifications

### Task Format
```json
{
  "train": [
    {"input": [[grid]], "output": [[grid]]},
    {"input": [[grid]], "output": [[grid]]},
    // 3-5 training examples
  ],
  "test": [
    {"input": [[grid]], "output": "TO_BE_PREDICTED"}
  ]
}
```

### Design Principles
1. **Simplicity**: Tasks should be simple for humans to understand
2. **Generality**: Test broad reasoning capabilities, not specific skills
3. **Fairness**: Avoid cultural or educational biases
4. **Efficiency**: Minimal training data required
5. **Novelty**: Each task requires fresh reasoning

### Human vs AI Performance (2019)
- **Human Performance**: ~80% accuracy on average
- **AI Performance**: <20% accuracy with contemporary methods
- **Performance Gap**: Demonstrates fundamental limitations in current AI

## Theoretical Framework

### Intelligence Measurement Criteria
1. **Skill Acquisition Efficiency**: How quickly can new skills be learned?
2. **Generalization Scope**: How broadly do learned skills transfer?
3. **Prior Knowledge Requirements**: How much background knowledge is needed?
4. **Data Efficiency**: How few examples are required for learning?

### Core Cognitive Primitives
The paper identifies minimal knowledge assumed for ARC tasks:

#### Object System
- Object cohesion (objects are distinct entities)
- Object persistence (objects continue to exist)
- Object influence (objects can affect each other)

#### Goal System  
- Goal-directedness (actions serve purposes)
- Intentionality (agents have intentions)
- Efficiency (prefer simpler solutions)

#### Counting and Numbers
- Small number recognition (1-10)
- Basic arithmetic operations
- Cardinality and ordinality concepts

#### Geometry and Topology
- Basic shapes (lines, rectangles, etc.)
- Spatial relationships (inside, outside, adjacent)
- Symmetry and reflection
- Rotation and translation

## Implications for AI Research

### Current AI Limitations
- **Memorization vs Understanding**: Current systems memorize patterns rather than understand principles
- **Data Hunger**: Require massive datasets for narrow capabilities
- **Transfer Failure**: Poor performance on novel tasks outside training
- **Brittleness**: Small changes can cause dramatic performance drops

### Required Breakthroughs
1. **Program Synthesis**: Generate programs from few examples
2. **Abstract Reasoning**: Extract general principles from specific instances
3. **Flexible Representation**: Adapt internal representations to new domains
4. **Meta-Learning**: Learn how to learn more effectively

### Evaluation Framework
- **Developer Set**: For algorithm development (public)
- **Evaluation Set**: For performance assessment (public)
- **Test Set**: For final benchmarking (private)
- **Scoring**: Simple accuracy - percentage of correctly solved tasks

## ARC Task Examples

### Example 1: Color Filling
**Pattern**: Fill the entire grid with the color of a single colored pixel
- Input: Grid with one colored pixel surrounded by black
- Output: Entire grid filled with that color
- Reasoning: Identify unique non-black color, apply to all positions

### Example 2: Symmetry Completion  
**Pattern**: Complete symmetrical patterns
- Input: Partially filled symmetrical pattern
- Output: Complete symmetrical pattern
- Reasoning: Identify axis of symmetry, mirror incomplete sections

### Example 3: Object Counting
**Pattern**: Output depends on number of objects in input
- Input: Grid with various colored objects
- Output: Single object repeated N times where N = input object count
- Reasoning: Count distinct objects, generate corresponding output

## Research Impact and Legacy

### Influence on AGI Research
- **Benchmark Adoption**: Widely adopted in AI research community
- **Competition Creation**: Led to ARC Prize competitions with substantial prizes
- **Research Direction**: Shifted focus toward few-shot learning and reasoning
- **Evaluation Standards**: Established new criteria for AI capability assessment

### Follow-up Research
- **Program Synthesis Approaches**: Attempts to solve ARC through code generation
- **Neural Architecture Search**: Custom architectures for visual reasoning
- **Meta-Learning Methods**: Learning to learn from few examples
- **Hybrid Systems**: Combining symbolic and neural approaches

## Technical Challenges

### Core Difficulties
1. **Pattern Extraction**: Identifying underlying rules from few examples
2. **Abstraction**: Moving from specific instances to general principles  
3. **Combinatorial Explosion**: Vast space of possible rules and transformations
4. **Noise Resistance**: Distinguishing signal from noise in limited data

### Solution Approaches
- **Discrete Program Search**: Exhaustive search through program space
- **Neural Program Synthesis**: Use neural networks to generate programs
- **Ensemble Methods**: Combine multiple approaches for better coverage
- **Domain-Specific Languages**: Create specialized languages for grid operations

## Competition Evolution

### ARC Prize 2020-2024
- **2020**: First competition, ~21% best performance
- **2022**: ARCathon with 118 teams from 47 countries
- **2024**: 53% best performance, significant progress but still below human level

### ARC-AGI-2 (2025)
- **Increased Difficulty**: More challenging tasks than original ARC
- **Enhanced Evaluation**: Better measures to prevent overfitting
- **Higher Stakes**: $700,000+ prizes for 85% accuracy achievement

## Philosophical Implications

### Definition of Intelligence
- **Beyond Task Performance**: Intelligence as learning ability, not just execution
- **Universal vs Specific**: Focus on general reasoning over domain expertise  
- **Efficiency Emphasis**: Prefer solutions requiring minimal data and computation
- **Human-Centric**: Use human cognitive primitives as baseline

### AGI Development Path
- **Current Limitations**: Massive-data-dependent systems lack true understanding
- **Required Shift**: Move from pattern matching to principle extraction
- **Success Metrics**: Measure learning efficiency, not just final performance
- **Evaluation Standards**: Use novel tasks requiring fresh reasoning

## Conclusion

"On the Measure of Intelligence" fundamentally redefined how we should evaluate and develop artificial intelligence systems. By introducing ARC and emphasizing skill-acquisition efficiency over accumulated performance, Chollet provided a blueprint for assessing progress toward artificial general intelligence.

The paper's impact extends beyond the specific benchmark to influence broader AI research directions, competition design, and philosophical discussions about the nature of intelligence. The ongoing ARC Prize competitions represent direct applications of these ideas, challenging the AI community to develop systems capable of human-like fluid reasoning.

**Critical Takeaways:**
1. Intelligence should be measured by learning efficiency, not just performance
2. Current AI systems rely too heavily on memorization rather than understanding  
3. ARC provides a concrete benchmark for evaluating progress toward AGI
4. Success on ARC requires fundamental advances in reasoning and abstraction
5. The path to AGI lies in developing systems that can quickly adapt to novel situations with minimal prior knowledge