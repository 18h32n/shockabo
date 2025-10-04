# Genetic Algorithm Troubleshooting Guide

## Common Issues and Solutions

### 1. Premature Convergence

**Symptoms:**
- Population converges to suboptimal solution quickly
- Diversity metrics drop rapidly
- Best fitness plateaus early

**Solutions:**

```python
# Increase diversity pressure
config.diversity.method = "speciation"  # Better than fitness_sharing for this
config.diversity.species_threshold = 0.4  # Higher threshold = more species

# Increase mutation rates
config.genetic_operators.mutation.base_rate = 0.15  # Higher base rate
config.genetic_operators.mutation.adaptive = True  # Enable adaptive mutation

# Reduce selection pressure
config.population.elite_size = 20  # Smaller elite (2% instead of 5%)

# Use novelty search
config.diversity.method = "novelty"  # Rewards behavioral uniqueness
```

### 2. Slow Convergence

**Symptoms:**
- Fitness improves very slowly
- Many generations needed for simple tasks
- Low success rate

**Solutions:**

```python
# Increase selection pressure
config.population.elite_size = 100  # Larger elite (10% of population)

# Use better initialization
config.population.initialization["llm_seed_ratio"] = 0.3  # More LLM seeds
config.population.initialization["template_ratio"] = 0.7  # More templates

# Optimize operators
config.genetic_operators.crossover.rate = 0.8  # Higher crossover rate

# Reduce population size for faster generations
config.population.size = 200  # Smaller, focused population
```

### 3. Out of Memory Errors

**Symptoms:**
- Process killed due to memory usage
- Slow performance with swapping
- Memory errors during evaluation

**Solutions:**

```python
# Reduce population size
config.population.size = 100  # Minimal viable population

# Limit program complexity
config.max_program_length = 8  # Shorter programs

# Enable aggressive cleanup
config.fitness.cache_enabled = False  # Disable caching

# Batch processing
config.parallelization.batch_size = 50  # Smaller batches
config.parallelization.workers = 2  # Fewer parallel workers

# In code:
if generation % 10 == 0:
    engine.fitness_evaluator.clear_cache()
    import gc
    gc.collect()
```

### 4. Timeout Errors

**Symptoms:**
- Evolution times out before completing
- Individual evaluations timeout
- Parallel processing hangs

**Solutions:**

```python
# Increase timeouts
config.performance.generation_timeout = 60  # More time per generation
config.performance.program_timeout = 2  # More time per program

# Reduce computation
config.convergence.max_generations = 50  # Fewer generations
config.population.size = 100  # Smaller population

# Optimize evaluation
config.fitness.early_termination.threshold = 0.9  # Stop at 90% match
```

### 5. Poor Parallel Performance

**Symptoms:**
- No speedup with multiple workers
- Worse performance with parallelization
- Deadlocks or hangs

**Solutions:**

```python
# Optimize worker configuration
import multiprocessing
config.parallelization.workers = min(4, multiprocessing.cpu_count() - 1)

# Increase batch size
config.parallelization.batch_size = 500  # Larger batches

# Use sequential for small populations
if config.population.size < 100:
    config.parallelization.backend = "sequential"

# Debug parallel issues
import logging
logging.getLogger("multiprocessing").setLevel(logging.DEBUG)
```

### 6. Invalid Programs Generated

**Symptoms:**
- Programs fail to execute
- Many individuals have 0 fitness
- Crossover/mutation creates invalid operations

**Solutions:**

```python
# Add validation in operators
def validate_program(operations):
    """Ensure program is valid before evaluation."""
    if not operations:
        return False
    
    for op in operations:
        if not hasattr(op, 'get_name') or not hasattr(op, 'parameters'):
            return False
    
    return True

# In fitness evaluator
def evaluate(self, individual):
    if not validate_program(individual.operations):
        return 0.0  # Invalid program
    
    # Normal evaluation...

# Use safer crossover
config.genetic_operators.crossover.methods = {
    "single_point": 0.6,  # More single-point
    "uniform": 0.3,
    "subtree": 0.1  # Less subtree (more complex)
}
```

### 7. Stagnation Despite Diversity

**Symptoms:**
- Population is diverse but not improving
- Many unique programs with similar low fitness
- No breakthrough solutions

**Solutions:**

```python
# Switch strategies
# Try different diversity mechanism
if generation > 50 and best_fitness < 0.5:
    config.diversity.method = "crowding"  # Focus on objective space

# Increase mutation strength
mutation = AdaptiveMutation()
mutation.current_rate = 0.25  # Aggressive mutation

# Add guided search
# Use LLM to suggest mutations when stuck
if stagnation_counter > 10:
    # Trigger LLM-guided mutation
    config.genetic_operators.mutation.llm_guided["trigger"] = "always"

# Reset population partially
if stagnation_counter > 20:
    # Keep only top 10%, regenerate rest
    elite = population.get_elite(10)
    population.individuals = elite
    # Reinitialize rest of population
```

## Debugging Techniques

### 1. Enable Detailed Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='evolution_debug.log'
)

# Log specific components
logging.getLogger("evolution_engine").setLevel(logging.DEBUG)
logging.getLogger("fitness_evaluator").setLevel(logging.INFO)
```

### 2. Track Individual Programs

```python
def debug_callback(population):
    """Debug callback to track specific individuals."""
    
    # Log best individual's program
    if population.best_individual:
        print(f"\nGen {population.generation} Best Program:")
        for i, op in enumerate(population.best_individual.operations):
            print(f"  {i}: {op.get_name()} - {op.parameters}")
    
    # Find problematic individuals
    for ind in population.individuals:
        if ind.fitness == 0.0 and ind.age > 5:
            print(f"Long-lived zero fitness: {ind.id}")
    
    # Check species distribution
    if hasattr(population, 'species'):
        print(f"Species: {len(population.species)}")
        for species_id, members in population.species.items():
            print(f"  Species {species_id}: {len(members)} members")
```

### 3. Visualize Evolution Progress

```python
from src.adapters.strategies.evolution_visualization import (
    EvolutionVisualizer,
    create_evolution_monitor
)

# Set up visualization
visualizer = EvolutionVisualizer(output_dir=Path("debug_output"))
monitor = create_evolution_monitor(visualizer)

# Run with monitoring
best_individual, stats = await engine.evolve(
    task,
    callbacks=[monitor, debug_callback]
)

# Generate report
visualizer.generate_html_report()
visualizer.export_fitness_history()

# Analyze convergence
convergence = visualizer.analyze_convergence()
print(f"Convergence analysis: {convergence}")
```

### 4. Profile Performance

```python
import cProfile
import pstats

def profile_evolution():
    """Profile evolution performance."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    
    # Run evolution
    asyncio.run(engine.evolve(task))
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_operation():
    population = Population()
    # ... operations
```

### 5. Test Individual Components

```python
# Test operators independently
def test_crossover():
    """Test crossover operator."""
    from src.adapters.strategies.genetic_operators import SinglePointCrossover
    
    # Create test individuals
    ind1 = Individual(operations=[...])
    ind2 = Individual(operations=[...])
    
    crossover = SinglePointCrossover()
    offspring = crossover.apply(ind1, ind2)
    
    print(f"Parent 1 length: {ind1.program_length()}")
    print(f"Parent 2 length: {ind2.program_length()}")
    print(f"Offspring lengths: {[o.program_length() for o in offspring]}")
    
    # Validate offspring
    for child in offspring:
        assert len(child.operations) > 0
        assert all(hasattr(op, 'get_name') for op in child.operations)

# Test fitness evaluation
def test_fitness():
    """Test fitness evaluator."""
    evaluator = FitnessEvaluator(task=task, dsl_engine=engine)
    
    # Create test individual
    individual = Individual(operations=[...])
    
    fitness = evaluator.evaluate(individual)
    print(f"Fitness: {fitness}")
    print(f"Cached execution: {individual.cached_execution}")
```

## Performance Optimization Tips

### 1. Optimal Configuration by Task Size

**Small Tasks (< 10x10 grids):**
```python
config.population.size = 100
config.parallelization.workers = 2
config.convergence.max_generations = 50
```

**Medium Tasks (10x10 - 20x20):**
```python
config.population.size = 200-300
config.parallelization.workers = 4
config.convergence.max_generations = 100
```

**Large Tasks (> 20x20):**
```python
config.population.size = 500
config.parallelization.workers = 6
config.convergence.max_generations = 150
config.fitness.early_termination.threshold = 0.85  # Lower threshold
```

### 2. Platform-Specific Settings

**Kaggle Notebooks:**
```python
config.parallelization.workers = 2  # Limited CPUs
config.performance.memory_limit = 4096  # Use available memory
config.population.size = 200  # Balance speed and diversity
```

**Google Colab:**
```python
config.parallelization.gpu_acceleration = True
config.parallelization.gpu_batch_size = 200
config.population.size = 300
```

**Local Development:**
```python
import multiprocessing
config.parallelization.workers = multiprocessing.cpu_count() - 1
config.performance.memory_limit = 8192  # Adjust based on system
```

### 3. Emergency Recovery

If evolution gets stuck or crashes:

```python
# Save population periodically
def checkpoint_callback(population):
    if population.generation % 10 == 0:
        import pickle
        with open(f"checkpoint_gen_{population.generation}.pkl", 'wb') as f:
            pickle.dump(population, f)

# Recovery function
def recover_from_checkpoint(checkpoint_file):
    import pickle
    with open(checkpoint_file, 'rb') as f:
        population = pickle.load(f)
    
    # Recreate engine with recovered population
    engine = EvolutionEngine(config, dsl_engine)
    engine.population = population
    
    # Continue evolution
    return engine
```

## When to Use Alternative Approaches

Consider alternatives to genetic algorithms when:

1. **Task has clear pattern**: Use template-based approach
2. **Small search space**: Use exhaustive search
3. **Need guaranteed optimal**: Use constraint solvers
4. **Very limited time**: Use greedy heuristics
5. **High-dimensional space**: Consider gradient-based methods

## Getting Help

1. Check logs in `.ai/debug-log.md`
2. Run performance profiler: `python -m tests.performance.test_evolution_performance`
3. Generate visualization report for analysis
4. Reduce problem size to create minimal test case
5. Check system resources (CPU, memory, disk space)

Remember: Evolution is a stochastic process. Some variation in results is normal. Run multiple times with different seeds to assess true performance.