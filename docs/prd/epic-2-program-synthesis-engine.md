# Epic 2: Program Synthesis Engine

Goal: Build an evolutionary program generation system that creates and evaluates 500+ Python transformation functions per task. This epic implements Jeremy Berman's proven approach (53.6% accuracy) using LLM-guided search with genetic algorithms and diversity preservation.

## Innovation Tournament Results: Pure LLM Generation Approach
Based on competitive analysis with user input, we're using Pure LLM Generation:
- GPT-5/Gemini direct code generation for high-quality solutions (~70% first-try success)
- Multi-tier API fallbacks (GPT-5 → Gemini → GLM-4.5 → Local models)
- Aggressive caching to minimize API costs (<$0.20/task target)
- Fast iteration with 2-5 second response times per program

## Parallel Work Streams

**Stream A (DSL & Evolution) - Team Member A:**
- Days 4-7 (Early Start): DSL design, genetic algorithm framework  
- Days 10-14: Evolutionary search, fitness functions, diversity
- Days 15-21: LLM integration, optimization

**Stream B (Generation & Testing) - Team Member B:**
- Days 7-10 (Early Start): Program templates, basic synthesis
- Days 10-14: Python generator, GPT-5/Gemini integration, caching
- Days 15-21: Full pipeline, 500+ programs per task

## Story 2.1: Domain-Specific Language Design

As a developer,
I want a comprehensive DSL for ARC transformations,
so that programs can express any grid manipulation concisely.

**Acceptance Criteria:**
1: Minimum 50 core operations (rotation, mirroring, color mapping, etc.)
2: Composable operations supporting function chaining
3: Type-safe grid operations with error handling
4: Efficient execution engine (<100ms per program)
5: Serializable program representation for caching
6: Documentation with examples for each operation
7: Unit tests achieving 100% coverage of DSL operations

**Critique & Refinement**: The DSL should prioritize operations discovered from analyzing the 1000 training tasks. Consider implementing a usage tracking system to identify which operations contribute most to successful solutions, allowing dynamic DSL expansion.

## Story 2.2: Genetic Algorithm Framework

As a developer,
I want a flexible genetic programming system,
so that I can evolve programs that solve ARC tasks.

**Acceptance Criteria:**
1: Population management supporting 1000+ programs
2: Crossover operations preserving program validity
3: Mutation operators with configurable rates
4: Fitness evaluation based on output similarity
5: Diversity preservation mechanisms (niching)
6: Parallel evaluation across population
7: Convergence detection and early stopping

## Story 2.3: Smart Model Routing for Program Generation

As a developer,
I want to use tiered LLM routing based on task complexity,
so that we minimize costs while maintaining generation quality.

**Acceptance Criteria:**
1: Implement SmartModelRouter with complexity detection
2: Tier 1 (Simple, 60%): Qwen2.5-Coder ($0.15/M tokens)
3: Tier 2 (Medium, 25%): Gemini 2.5 Flash ($0.31/$2.62)
4: Tier 3 (Complex, 10%): GLM-4.5 ($0.59/$2.19)
5: Tier 4 (Breakthrough, 5%): GPT-5 ($1.25/$10)
6: Local fallback: Falcon Mamba 7B (free)
7: Total cost target: <$100 (vs $1000+ pure GPT-5)

## Story 2.4: Python Function Synthesis

As a developer,
I want to generate executable Python functions from DSL programs,
so that solutions can be evaluated efficiently.

**Acceptance Criteria:**
1: DSL-to-Python transpiler with 100% operation coverage
2: Runtime safety checks (bounds, types)
3: Execution timeout of 1 second per function
4: Memory limit of 100MB per execution
5: Sandboxed execution environment
6: Performance profiling for optimization
7: Error messages mapping back to DSL operations

## Story 2.5: Evolutionary Search Pipeline

As a developer,
I want a complete evolutionary search system,
so that I can find programs solving each ARC task.

**Acceptance Criteria:**
1: Generate and evaluate 500+ programs per task
2: Achieve 45%+ standalone accuracy
3: Complete search in under 5 minutes per task
4: Track genealogy and successful mutations
5: Export top 10 programs per task
6: Integration with evaluation framework
7: Reproducible results with seed control

## Story 2.6: Program Caching & Analysis

As a developer,
I want to cache and analyze successful programs,
so that I can reuse solutions and understand patterns.

**Acceptance Criteria:**
1: Persistent cache of all evaluated programs
2: Similarity detection for program deduplication
3: Pattern mining across successful programs
4: Performance analytics dashboard
5: Export programs in readable format
6: Integration with ensemble voting system
7: Cache size management (stay under 1GB)

## Story 2.7: GPU-Accelerated Program Evaluation

As a developer,
I want GPU-accelerated batch evaluation of programs,
so that I can evaluate 500+ programs efficiently within time constraints.

**Acceptance Criteria:**
1: Batch evaluation of 100 programs in parallel on GPU
2: Vectorized grid operations using PyTorch/JAX
3: Memory-efficient batching (stay under 8GB VRAM)
4: 10x speedup over CPU evaluation
5: Automatic CPU fallback if GPU unavailable
6: Profile and optimize bottlenecks
7: Integration with existing evaluation framework

## Story 2.8: Intelligent Program Pruning

As a developer,
I want early termination of obviously wrong programs,
so that I can focus compute on promising candidates.

**Acceptance Criteria:**
1: Quick pre-checks before full evaluation
2: Pattern-based early rejection rules
3: Partial execution with confidence scoring
4: Save 40% of evaluation time
5: Adjustable pruning aggressiveness
6: Track false negative rate (<5%)
7: A/B testing framework for pruning strategies

## Story 2.9: Distributed Evolution Across Platforms

As a developer,
I want to distribute evolutionary search across multiple platforms,
so that I can utilize all available compute resources.

**Acceptance Criteria:**
1: Population sharding across Kaggle/Colab/Local
2: Asynchronous population exchange
3: Platform-aware load balancing
4: Checkpoint synchronization every generation
5: Handle platform disconnections gracefully
6: Merge results without duplication
7: 2.5x throughput improvement over single platform

## Story 2.10: End-to-End Program Synthesis Validation

As a developer,
I want comprehensive integration tests for the synthesis pipeline,
so that I can ensure all components work together correctly.

**Acceptance Criteria:**
1: Test with 20 representative ARC tasks
2: Verify DSL→Python→Execution chain
3: Validate API integration with rate limiting
4: Test cache hit/miss scenarios
5: Verify evolutionary convergence
6: Performance benchmarks met (5 min/task)
7: Memory usage within limits

## Story 2.11: Cross-Strategy Integration Points

As a developer,
I want well-defined interfaces with other strategies,
so that program synthesis integrates seamlessly with the ensemble.

**Acceptance Criteria:**
1: Standardized output format for programs
2: Confidence scoring compatible with ensemble
3: Timing synchronization with TTT strategy
4: Shared evaluation metrics
5: Unified logging and monitoring
6: Integration tests with mock strategies
7: API documentation complete

## Story 2.12: Multi-Armed Bandit Controller

As a developer,
I want a MAB system for optimal strategy selection,
so that we dynamically allocate resources to best-performing approaches.

**Acceptance Criteria:**
1: Thompson sampling implementation for exploration
2: Real-time reward tracking per generation strategy
3: Contextual bandits using task features
4: Smooth handling of strategy failures
5: Interpretable selection decisions
6: Cost-aware reward functions
7: Performance improvement of 20%+ over fixed allocation
