# Genetic Algorithm API Documentation

## Overview

The Genetic Algorithm Framework provides an evolutionary approach to generating DSL programs that solve ARC tasks. It uses population-based search with crossover, mutation, and diversity preservation mechanisms to evolve effective solutions.

## Quick Start

```python
import asyncio
from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.infrastructure.config import GeneticAlgorithmConfig
from src.domain.models import ARCTask

# Configure evolution
config = GeneticAlgorithmConfig()
config.population.size = 100
config.convergence.max_generations = 50

# Create engine
engine = EvolutionEngine(config=config, dsl_engine=your_dsl_engine)

# Run evolution
async def evolve_solution(task: ARCTask):
    best_individual, stats = await engine.evolve(task)
    print(f"Best fitness: {best_individual.fitness}")
    return best_individual

# Execute
asyncio.run(evolve_solution(your_arc_task))
```

## Core Components

### 1. Individual

Represents a single DSL program in the population.

```python
from src.adapters.strategies.evolution_engine import Individual

# Individual attributes
individual = Individual(operations=list_of_operations)
individual.fitness      # Fitness score (0.0 to 1.0)
individual.age         # Number of generations survived
individual.parent_ids  # Set of parent IDs (genealogy)
individual.id         # Unique identifier
```

### 2. Population

Manages a collection of individuals.

```python
from src.adapters.strategies.evolution_engine import Population

population = Population()
population.add_individual(individual)
population.get_elite(10)  # Get top 10 individuals
population.average_fitness()  # Population statistics
```

### 3. EvolutionEngine

Main engine that orchestrates the evolution process.

```python
engine = EvolutionEngine(
    config=genetic_algorithm_config,
    dsl_engine=dsl_engine,
    operation_templates=optional_templates
)

# Run evolution asynchronously
best_individual, stats = await engine.evolve(
    task=arc_task,
    callbacks=[progress_callback]
)
```

## Genetic Operators

### Crossover Operators

```python
from src.adapters.strategies.genetic_operators import (
    SinglePointCrossover,
    UniformCrossover,
    SubtreeCrossover
)

# Single-point crossover
crossover = SinglePointCrossover()
offspring = crossover.apply(parent1, parent2)

# Uniform crossover with custom swap probability
crossover = UniformCrossover(swap_probability=0.5)
offspring = crossover.apply(parent1, parent2)

# Subtree crossover for hierarchical programs
crossover = SubtreeCrossover()
offspring = crossover.apply(parent1, parent2)
```

### Mutation Operators

```python
from src.adapters.strategies.genetic_operators import (
    OperationReplacementMutation,
    ParameterMutation,
    InsertDeleteMutation,
    ReorderMutation,
    AdaptiveMutation
)

# Operation replacement
mutation = OperationReplacementMutation(mutation_rate=0.1)
mutated = mutation.apply(individual)

# Parameter mutation
mutation = ParameterMutation(mutation_rate=0.2, mutation_strength=0.3)
mutated = mutation.apply(individual)

# Adaptive mutation (adjusts rates based on fitness stagnation)
mutation = AdaptiveMutation(base_rate=0.1, max_rate=0.3)
mutation.adapt_rate(fitness_improvement=0.01)
```

## Diversity Mechanisms

### Fitness Sharing

```python
from src.adapters.strategies.diversity_mechanisms import FitnessSharing

diversity = FitnessSharing(niche_radius=0.15, alpha=1.0)
adjusted_fitness = diversity.apply_pressure(individual, population)
```

### Speciation

```python
from src.adapters.strategies.diversity_mechanisms import Speciation

speciation = Speciation(compatibility_threshold=0.3)
speciation.speciate_population(population)
# Individuals are now assigned to species
```

### Novelty Search

```python
from src.adapters.strategies.diversity_mechanisms import NoveltySearch

novelty = NoveltySearch(archive_size=100, k_nearest=15)
novelty_fitness = novelty.apply_pressure(individual, population)
```

## Configuration

### YAML Configuration

```yaml
# configs/strategies/evolution.yaml
evolution:
  population:
    size: 1000
    initialization:
      method: "hybrid"
      llm_seed_ratio: 0.2
      template_ratio: 0.5
    elite_size: 50

  genetic_operators:
    crossover:
      rate: 0.7
      methods:
        single_point: 0.4
        uniform: 0.3
        subtree: 0.3
    mutation:
      base_rate: 0.1
      adaptive: true
      max_rate: 0.3

  fitness:
    metrics:
      grid_similarity: 0.7
      program_length: 0.2
      execution_time: 0.1
    cache_enabled: true
    early_termination:
      threshold: 0.95

  diversity:
    method: "fitness_sharing"
    niche_radius: 0.15

  parallelization:
    backend: "multiprocessing"
    workers: 4
    batch_size: 250

  convergence:
    max_generations: 200
    stagnation_patience: 20
    min_fitness_improvement: 0.001
```

### Programmatic Configuration

```python
from src.infrastructure.config import (
    GeneticAlgorithmConfig,
    PopulationConfig,
    FitnessConfig,
    ConvergenceConfig
)

config = GeneticAlgorithmConfig()

# Population settings
config.population.size = 500
config.population.elite_size = 25

# Fitness evaluation
config.fitness.metrics = {
    "grid_similarity": 0.8,
    "program_length": 0.1,
    "execution_time": 0.1
}

# Convergence criteria
config.convergence.max_generations = 100
config.convergence.stagnation_patience = 15
```

## Integration with Program Synthesis

### Enable Evolution in Synthesis

```python
from src.adapters.strategies.program_synthesis import (
    ProgramSynthesisAdapter,
    ProgramSynthesisConfig,
    ProgramGenerationStrategy
)

# Configure synthesis to use evolution
config = ProgramSynthesisConfig(
    generation_strategy=ProgramGenerationStrategy.SEARCH_BASED,
    use_evolution=True,
    evolution_config_path=Path("configs/strategies/evolution.yaml"),
    max_evolution_time=30.0  # seconds
)

adapter = ProgramSynthesisAdapter(config)
```

## Parallel Evaluation

### Using ParallelEvaluator

```python
from src.adapters.strategies.parallel_evaluation import ParallelEvaluator

async with ParallelEvaluator(
    num_workers=4,
    batch_size=250,
    timeout_per_individual=1.0
) as evaluator:
    results = await evaluator.evaluate_population(
        individuals=population.individuals,
        task=arc_task,
        progress_callback=lambda done, total: print(f"{done}/{total}")
    )
```

### GPU Acceleration (Experimental)

```python
from src.adapters.strategies.parallel_evaluation import GPUAcceleratedEvaluator

async with GPUAcceleratedEvaluator(
    num_workers=2,
    gpu_batch_size=100
) as evaluator:
    results = await evaluator.evaluate_population(
        individuals=population.individuals,
        task=arc_task
    )
```

## Callbacks and Monitoring

### Generation Callbacks

```python
def generation_callback(population: Population):
    """Called after each generation."""
    print(f"Generation {population.generation}")
    print(f"Best fitness: {population.best_individual.fitness}")
    print(f"Average fitness: {population.average_fitness()}")
    print(f"Diversity: {population.diversity_metrics}")

# Use in evolution
best_individual, stats = await engine.evolve(
    task=arc_task,
    callbacks=[generation_callback]
)
```

### Progress Tracking

```python
class EvolutionMonitor:
    def __init__(self):
        self.fitness_history = []
        self.diversity_history = []
    
    def __call__(self, population):
        if population.best_individual:
            self.fitness_history.append(population.best_individual.fitness)
        self.diversity_history.append(
            population.diversity_metrics.get('unique_programs', 0)
        )

monitor = EvolutionMonitor()
best_individual, stats = await engine.evolve(task, callbacks=[monitor])

# Plot results
import matplotlib.pyplot as plt
plt.plot(monitor.fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()
```

## Best Practices

### 1. Population Size Selection

- Small tasks (< 10x10 grids): 100-200 individuals
- Medium tasks (10x10 to 20x20): 200-500 individuals
- Large tasks (> 20x20): 500-1000 individuals

### 2. Operator Selection

- Use multiple crossover types for diversity
- Start with higher mutation rates (0.1-0.2)
- Enable adaptive mutation for long runs

### 3. Diversity Preservation

- Fitness sharing: Good for maintaining solution diversity
- Speciation: Better for complex multi-modal problems
- Novelty search: Use when fitness landscape is deceptive

### 4. Performance Optimization

```python
# Use caching
config.fitness.cache_enabled = True

# Limit program length
config.max_program_length = 10

# Use parallel evaluation
config.parallelization.workers = min(4, multiprocessing.cpu_count())

# Set reasonable timeouts
config.performance.program_timeout = 1.0  # seconds
config.performance.generation_timeout = 30  # seconds
```

### 5. Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Track specific individuals
def track_individual(population):
    for ind in population.individuals:
        if ind.fitness > 0.9:
            print(f"High fitness individual {ind.id}:")
            for op in ind.operations:
                print(f"  - {op.get_name()} {op.parameters}")

# Save best programs
best_programs = []
def save_best(population):
    if population.best_individual:
        best_programs.append({
            'generation': population.generation,
            'fitness': population.best_individual.fitness,
            'operations': [
                {'name': op.get_name(), 'params': op.parameters}
                for op in population.best_individual.operations
            ]
        })
```

## Common Issues and Solutions

### Issue: Premature Convergence

```python
# Increase diversity pressure
config.diversity.niche_radius = 0.2  # Larger niches

# Increase mutation rate
config.genetic_operators.mutation.base_rate = 0.15

# Use speciation
config.diversity.method = "speciation"
```

### Issue: Slow Evaluation

```python
# Reduce population size
config.population.size = 100

# Increase parallelization
config.parallelization.workers = 8
config.parallelization.batch_size = 500

# Enable early termination
config.fitness.early_termination.threshold = 0.9
```

### Issue: Memory Usage

```python
# Limit population size
config.population.size = 200

# Reduce elite size
config.population.elite_size = 10

# Clear fitness cache periodically
if engine.fitness_evaluator:
    engine.fitness_evaluator.clear_cache()
```

## Example: Complete Evolution Setup

```python
import asyncio
from pathlib import Path
from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.infrastructure.config import GeneticAlgorithmConfig
from src.domain.services.dsl_engine import DSLEngine

async def solve_arc_task(task):
    # Configure evolution
    config = GeneticAlgorithmConfig()
    config.population.size = 200
    config.genetic_operators.crossover.rate = 0.7
    config.genetic_operators.mutation.base_rate = 0.1
    config.diversity.method = "fitness_sharing"
    config.convergence.max_generations = 50
    config.convergence.stagnation_patience = 10
    
    # Create DSL engine
    dsl_engine = DSLEngine()
    
    # Create evolution engine
    engine = EvolutionEngine(config=config, dsl_engine=dsl_engine)
    
    # Track progress
    best_fitness_history = []
    
    def progress_callback(population):
        if population.best_individual:
            best_fitness_history.append(population.best_individual.fitness)
            print(f"Gen {population.generation}: "
                  f"Best={population.best_individual.fitness:.3f}, "
                  f"Avg={population.average_fitness():.3f}")
    
    try:
        # Run evolution
        best_individual, stats = await engine.evolve(
            task=task,
            callbacks=[progress_callback]
        )
        
        # Extract solution
        if best_individual:
            print(f"\nBest solution found with fitness: {best_individual.fitness}")
            print(f"Program length: {best_individual.program_length()}")
            print(f"Generations: {stats['generations']}")
            
            # Convert to executable program
            operations = []
            for op in best_individual.operations:
                operations.append({
                    "name": op.get_name(),
                    "parameters": op.parameters
                })
            
            return operations
        else:
            print("No solution found")
            return None
            
    finally:
        engine.cleanup()

# Run the solver
if __name__ == "__main__":
    from src.domain.models import ARCTask
    
    # Example task
    task = ARCTask(
        task_id="example_001",
        task_source="test",
        train_examples=[
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
        ],
        test_input=[[3, 0], [0, 3]]
    )
    
    solution = asyncio.run(solve_arc_task(task))
    print(f"\nSolution: {solution}")
```