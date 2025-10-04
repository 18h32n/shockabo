"""
End-to-end example of using the genetic algorithm framework to solve an ARC task.

This example demonstrates:
1. Setting up the evolution engine
2. Configuring parameters for a specific task
3. Running evolution with monitoring
4. Analyzing results
5. Using the best solution
"""

import asyncio
import json
from pathlib import Path
from typing import Any

# Import required components
from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.adapters.strategies.evolution_visualization import (
    EvolutionVisualizer,
    create_evolution_monitor,
)
from src.domain.models import ARCTask
from src.domain.services.dsl_engine import DSLEngine
from src.infrastructure.config import GeneticAlgorithmConfig

# Example ARC task: Color transformation with pattern
EXAMPLE_TASK = {
    "task_id": "example_color_pattern",
    "train": [
        {
            "input": [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ],
            "output": [
                [2, 0, 2, 0],
                [0, 2, 0, 2],
                [2, 0, 2, 0],
                [0, 2, 0, 2]
            ]
        },
        {
            "input": [
                [3, 3, 0],
                [3, 3, 0],
                [0, 0, 0]
            ],
            "output": [
                [4, 4, 0],
                [4, 4, 0],
                [0, 0, 0]
            ]
        }
    ],
    "test": [
        {
            "input": [
                [5, 0, 5],
                [0, 0, 0],
                [5, 0, 5]
            ]
        }
    ]
}


def create_arc_task_from_json(task_data: dict) -> ARCTask:
    """Convert JSON task data to ARCTask object."""
    return ARCTask(
        task_id=task_data["task_id"],
        task_source="example",
        train_examples=task_data["train"],
        test_input=task_data["test"][0]["input"]
    )


async def solve_arc_task_with_evolution(
    task: ARCTask,
    output_dir: Path = Path("evolution_example_output")
) -> dict[str, Any]:
    """
    Solve an ARC task using genetic algorithm evolution.
    
    Args:
        task: The ARC task to solve
        output_dir: Directory for output files
        
    Returns:
        Dictionary with solution and statistics
    """

    # 1. Configure the genetic algorithm
    print("Step 1: Configuring genetic algorithm...")

    config = GeneticAlgorithmConfig()

    # Population settings
    config.population.size = 200  # Medium-sized population
    config.population.elite_size = 20  # Preserve top 10%
    config.population.initialization = {
        "method": "hybrid",
        "llm_seed_ratio": 0.0,  # No LLM for this example
        "template_ratio": 0.6   # 60% template-based
    }

    # Genetic operators
    config.genetic_operators.crossover.rate = 0.7
    config.genetic_operators.mutation.base_rate = 0.1
    config.genetic_operators.mutation.adaptive = True

    # Fitness evaluation
    config.fitness.metrics = {
        "grid_similarity": 0.8,  # High weight on accuracy
        "program_length": 0.15,  # Prefer shorter programs
        "execution_time": 0.05   # Small penalty for slow programs
    }
    config.fitness.cache_enabled = True
    config.fitness.early_termination.threshold = 0.95

    # Diversity preservation
    config.diversity.method = "fitness_sharing"
    config.diversity.niche_radius = 0.15

    # Parallelization
    config.parallelization.backend = "multiprocessing"
    config.parallelization.workers = 4
    config.parallelization.batch_size = 50

    # Convergence criteria
    config.convergence.max_generations = 50
    config.convergence.stagnation_patience = 15
    config.convergence.min_fitness_improvement = 0.001

    print(f"  Population size: {config.population.size}")
    print(f"  Max generations: {config.convergence.max_generations}")
    print(f"  Parallelization: {config.parallelization.workers} workers")

    # 2. Create DSL engine and evolution engine
    print("\nStep 2: Creating engines...")

    dsl_engine = DSLEngine()  # In real usage, would be properly configured
    evolution_engine = EvolutionEngine(
        config=config,
        dsl_engine=dsl_engine
    )

    # 3. Set up monitoring and visualization
    print("\nStep 3: Setting up monitoring...")

    output_dir.mkdir(exist_ok=True)
    visualizer = EvolutionVisualizer(output_dir=output_dir)
    monitor_callback = create_evolution_monitor(visualizer)

    # Custom progress callback
    generation_stats = []

    def progress_callback(population):
        """Track detailed statistics per generation."""
        stats = {
            "generation": population.generation,
            "best_fitness": population.best_individual.fitness if population.best_individual else 0,
            "average_fitness": population.average_fitness(),
            "population_size": population.size(),
            "unique_programs": population.diversity_metrics.get('unique_programs', 0),
            "best_program_length": population.best_individual.program_length() if population.best_individual else 0
        }
        generation_stats.append(stats)

        # Print progress every 5 generations
        if population.generation % 5 == 0:
            print(f"  Gen {stats['generation']:3d}: "
                  f"Best={stats['best_fitness']:.3f}, "
                  f"Avg={stats['average_fitness']:.3f}, "
                  f"Unique={stats['unique_programs']:.1%}")

    # 4. Run evolution
    print("\nStep 4: Running evolution...")
    print("  This may take a few minutes...\n")

    try:
        best_individual, evolution_stats = await evolution_engine.evolve(
            task=task,
            callbacks=[monitor_callback, progress_callback]
        )

        print("\nEvolution completed!")
        print(f"  Total generations: {evolution_stats['generations']}")
        print(f"  Best fitness achieved: {best_individual.fitness:.3f}")
        print(f"  Best program length: {best_individual.program_length()} operations")

    except Exception as e:
        print(f"Evolution failed: {e}")
        raise
    finally:
        evolution_engine.cleanup()

    # 5. Analyze results
    print("\nStep 5: Analyzing results...")

    # Convert best individual to executable program
    best_program = []
    for op in best_individual.operations:
        best_program.append({
            "name": op.get_name(),
            "parameters": op.parameters
        })

    # Test on the test input
    print("\nTesting best program on test input...")
    test_result = dsl_engine.execute_program(
        best_individual.operations,
        task.test_input
    )

    if test_result['success']:
        print("  Test execution successful!")
        print(f"  Test input shape: {len(task.test_input)}x{len(task.test_input[0])}")
        print(f"  Output shape: {len(test_result['output'])}x{len(test_result['output'][0])}")
    else:
        print(f"  Test execution failed: {test_result.get('error')}")

    # Generate visualizations
    print("\nGenerating visualization report...")
    visualizer.generate_html_report("evolution_report.html")
    visualizer.export_fitness_history("fitness_history.csv")

    # Save detailed results
    results = {
        "task_id": task.task_id,
        "config": {
            "population_size": config.population.size,
            "max_generations": config.convergence.max_generations,
            "diversity_method": config.diversity.method
        },
        "evolution_stats": evolution_stats,
        "best_solution": {
            "fitness": best_individual.fitness,
            "program_length": best_individual.program_length(),
            "program": best_program,
            "parent_ids": list(best_individual.parent_ids),
            "age": best_individual.age
        },
        "test_result": {
            "success": test_result['success'],
            "output": test_result.get('output', []) if test_result['success'] else None
        },
        "generation_history": generation_stats,
        "convergence_analysis": visualizer.analyze_convergence()
    }

    # Save results to file
    results_file = output_dir / "evolution_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print(f"Visualization report: {output_dir / 'evolution_report.html'}")

    return results


def analyze_solution_quality(results: dict[str, Any]) -> None:
    """Analyze the quality of the evolved solution."""
    print("\n" + "="*60)
    print("SOLUTION ANALYSIS")
    print("="*60)

    best_solution = results["best_solution"]

    # Program complexity
    print("\n1. Program Complexity:")
    print(f"   - Length: {best_solution['program_length']} operations")
    print(f"   - Unique operations: {len(set(op['name'] for op in best_solution['program']))}")

    # Evolution efficiency
    evolution_stats = results["evolution_stats"]
    print("\n2. Evolution Efficiency:")
    print(f"   - Generations used: {evolution_stats['generations']}")
    print(f"   - Final population size: {evolution_stats['population_size']}")

    # Convergence analysis
    convergence = results["convergence_analysis"]
    print("\n3. Convergence Analysis:")
    print(f"   - Total generations: {convergence.get('total_generations', 'N/A')}")
    print(f"   - Plateau generation: {convergence.get('plateau_generation', 'Not detected')}")
    print(f"   - Improvement rate: {convergence.get('improvement_rate', 0):.4f}/generation")

    # Solution quality
    print("\n4. Solution Quality:")
    print(f"   - Training fitness: {best_solution['fitness']:.3f}")
    print(f"   - Test success: {'Yes' if results['test_result']['success'] else 'No'}")

    # Program operations
    print("\n5. Program Operations:")
    for i, op in enumerate(best_solution['program']):
        print(f"   {i+1}. {op['name']} with parameters: {op['parameters']}")


async def main():
    """Main function to run the example."""
    print("Genetic Algorithm Evolution Example for ARC Tasks")
    print("=" * 60)

    # Create task from example data
    task = create_arc_task_from_json(EXAMPLE_TASK)
    print(f"\nTask: {task.task_id}")
    print(f"Training examples: {len(task.train_examples)}")
    print(f"Test input size: {len(task.test_input)}x{len(task.test_input[0])}")

    # Solve the task
    try:
        results = await solve_arc_task_with_evolution(task)

        # Analyze the solution
        analyze_solution_quality(results)

        # Display test output if successful
        if results['test_result']['success']:
            print("\n6. Test Output:")
            output = results['test_result']['output']
            for row in output:
                print(f"   {row}")

    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Example completed!")


if __name__ == "__main__":
    # Note: In a real scenario, the DSL engine would need to be properly
    # initialized with actual DSL operations. This example uses a mock
    # implementation for demonstration purposes.

    # Run the async main function
    asyncio.run(main())
