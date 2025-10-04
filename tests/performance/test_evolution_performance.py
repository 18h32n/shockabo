"""
Performance benchmarks for genetic algorithm framework.

This module tests and benchmarks the performance characteristics
of the evolution engine under various configurations.
"""

import asyncio
import os
import time
from statistics import mean, stdev

import psutil
import pytest

from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.domain.models import ARCTask
from src.infrastructure.config import GeneticAlgorithmConfig


class MockDSLEngine:
    """Mock DSL engine for performance testing."""

    def __init__(self, execution_time: float = 0.001):
        self.execution_time = execution_time
        self.execution_count = 0

    def execute_program(self, operations, grid):
        """Simulate program execution."""
        self.execution_count += 1
        time.sleep(self.execution_time)  # Simulate work
        return {
            "success": True,
            "output": grid  # Return input as output for simplicity
        }


def create_test_task(grid_size: int = 5) -> ARCTask:
    """Create a test ARC task of specified size."""
    grid = [[i % 10 for i in range(grid_size)] for _ in range(grid_size)]
    return ARCTask(
        task_id=f"perf_test_{grid_size}x{grid_size}",
        task_source="benchmark",
        train_examples=[
            {"input": grid, "output": grid},
            {"input": grid, "output": grid}
        ],
        test_input=grid
    )


class TestEvolutionPerformance:
    """Performance benchmarks for evolution engine."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_population_initialization_speed(self):
        """Benchmark population initialization speed."""
        config = GeneticAlgorithmConfig()
        mock_engine = MockDSLEngine()
        task = create_test_task()

        # Test different population sizes
        population_sizes = [100, 500, 1000]
        results = {}

        for pop_size in population_sizes:
            config.population.size = pop_size
            config.convergence.max_generations = 0  # Only initialization

            engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

            start_time = time.time()
            await engine._initialize_population(task)
            init_time = time.time() - start_time

            results[pop_size] = {
                "time": init_time,
                "time_per_individual": init_time / pop_size
            }

            engine.cleanup()

        # Print results
        print("\nPopulation Initialization Performance:")
        print("Size | Total Time | Time/Individual")
        print("-" * 40)
        for size, metrics in results.items():
            print(f"{size:4d} | {metrics['time']:10.3f}s | {metrics['time_per_individual']*1000:10.3f}ms")

        # Assert reasonable performance
        assert all(r["time_per_individual"] < 0.01 for r in results.values()), \
            "Population initialization too slow"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_fitness_evaluation_performance(self):
        """Benchmark fitness evaluation performance."""
        config = GeneticAlgorithmConfig()
        config.population.size = 100
        config.convergence.max_generations = 1

        # Test with different execution times
        execution_times = [0.001, 0.01, 0.1]
        results = {}

        for exec_time in execution_times:
            mock_engine = MockDSLEngine(execution_time=exec_time)
            task = create_test_task()

            engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

            # Initialize population
            await engine._initialize_population(task)

            # Initialize fitness evaluator
            engine.fitness_evaluator = engine.FitnessEvaluator(
                task=task,
                dsl_engine=mock_engine
            )

            # Time evaluation
            start_time = time.time()
            await engine._evaluate_population()
            eval_time = time.time() - start_time

            results[exec_time] = {
                "total_time": eval_time,
                "evaluations": mock_engine.execution_count,
                "time_per_eval": eval_time / mock_engine.execution_count if mock_engine.execution_count > 0 else 0
            }

            engine.cleanup()

        # Print results
        print("\nFitness Evaluation Performance:")
        print("Exec Time | Total Time | Evaluations | Time/Eval")
        print("-" * 50)
        for exec_time, metrics in results.items():
            print(f"{exec_time:9.3f}s | {metrics['total_time']:10.3f}s | "
                  f"{metrics['evaluations']:11d} | {metrics['time_per_eval']:9.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_parallel_evaluation_scaling(self):
        """Test how well parallel evaluation scales with workers."""
        config = GeneticAlgorithmConfig()
        config.population.size = 200
        config.convergence.max_generations = 1
        config.parallelization.backend = "multiprocessing"

        mock_engine = MockDSLEngine(execution_time=0.01)
        task = create_test_task()

        # Test with different worker counts
        worker_counts = [1, 2, 4]
        results = {}

        for workers in worker_counts:
            config.parallelization.workers = workers

            engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

            start_time = time.time()

            # Run one generation
            await engine._initialize_population(task)
            engine.fitness_evaluator = engine.FitnessEvaluator(
                task=task,
                dsl_engine=mock_engine
            )
            await engine._evaluate_population()

            total_time = time.time() - start_time

            results[workers] = {
                "time": total_time,
                "speedup": results[1]["time"] / total_time if workers > 1 and 1 in results else 1.0
            }

            engine.cleanup()

        # Print results
        print("\nParallel Evaluation Scaling:")
        print("Workers | Time     | Speedup")
        print("-" * 30)
        for workers, metrics in results.items():
            print(f"{workers:7d} | {metrics['time']:8.3f}s | {metrics['speedup']:7.2f}x")

        # Check that we get some speedup
        if len(results) > 1:
            assert results[max(worker_counts)]["speedup"] > 1.2, \
                "Parallel evaluation not providing speedup"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_usage_scaling(self):
        """Test memory usage with different population sizes."""
        process = psutil.Process(os.getpid())
        mock_engine = MockDSLEngine()
        task = create_test_task()

        population_sizes = [100, 500, 1000]
        results = {}

        for pop_size in population_sizes:
            config = GeneticAlgorithmConfig()
            config.population.size = pop_size
            config.convergence.max_generations = 2

            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

            # Run evolution
            await engine._initialize_population(task)
            engine.fitness_evaluator = engine.FitnessEvaluator(
                task=task,
                dsl_engine=mock_engine
            )
            await engine._evaluate_population()
            await engine._create_next_generation()

            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            results[pop_size] = {
                "memory_increase": memory_increase,
                "memory_per_individual": memory_increase / pop_size
            }

            engine.cleanup()

            # Force garbage collection
            import gc
            gc.collect()
            time.sleep(0.1)

        # Print results
        print("\nMemory Usage Scaling:")
        print("Pop Size | Memory Increase | Per Individual")
        print("-" * 45)
        for size, metrics in results.items():
            print(f"{size:8d} | {metrics['memory_increase']:14.2f}MB | "
                  f"{metrics['memory_per_individual']*1000:13.2f}KB")

        # Check reasonable memory usage
        assert all(r["memory_per_individual"] < 1.0 for r in results.values()), \
            "Excessive memory usage per individual"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_generation_throughput(self):
        """Test generations per second throughput."""
        config = GeneticAlgorithmConfig()
        config.population.size = 100
        config.convergence.max_generations = 20
        config.parallelization.workers = 2

        mock_engine = MockDSLEngine(execution_time=0.001)
        task = create_test_task()

        engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

        generation_times = []

        def timing_callback(population):
            generation_times.append(time.time())

        try:
            start_time = time.time()

            # Run evolution
            best_individual, stats = await asyncio.wait_for(
                engine.evolve(task, callbacks=[timing_callback]),
                timeout=60.0
            )

            total_time = time.time() - start_time

            # Calculate throughput
            generations = stats['generations']
            throughput = generations / total_time

            # Calculate per-generation times
            if len(generation_times) > 1:
                gen_durations = [
                    generation_times[i+1] - generation_times[i]
                    for i in range(len(generation_times)-1)
                ]
                avg_gen_time = mean(gen_durations)
                std_gen_time = stdev(gen_durations) if len(gen_durations) > 1 else 0
            else:
                avg_gen_time = total_time / generations if generations > 0 else 0
                std_gen_time = 0

            print("\nGeneration Throughput:")
            print(f"Total generations: {generations}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Throughput: {throughput:.2f} gen/s")
            print(f"Avg generation time: {avg_gen_time:.3f}s ± {std_gen_time:.3f}s")

            # Assert reasonable throughput
            assert throughput > 0.5, "Generation throughput too low"

        finally:
            engine.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_convergence_speed(self):
        """Test how quickly the algorithm converges."""
        # Simple task that should converge quickly
        task = ARCTask(
            task_id="convergence_test",
            task_source="benchmark",
            train_examples=[
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
            ],
            test_input=[[3, 0], [0, 3]]
        )

        config = GeneticAlgorithmConfig()
        config.population.size = 50
        config.convergence.max_generations = 50
        config.convergence.stagnation_patience = 10

        mock_engine = MockDSLEngine()
        # Mock will return perfect output for "flip" operation

        engine = EvolutionEngine(config=config, dsl_engine=mock_engine)

        fitness_history = []
        convergence_generation = None

        def convergence_callback(population):
            nonlocal convergence_generation
            if population.best_individual:
                fitness = population.best_individual.fitness
                fitness_history.append(fitness)

                # Check if converged
                if fitness > 0.95 and convergence_generation is None:
                    convergence_generation = population.generation

        try:
            # Run evolution
            best_individual, stats = await engine.evolve(
                task,
                callbacks=[convergence_callback]
            )

            print("\nConvergence Speed:")
            print(f"Final best fitness: {best_individual.fitness:.3f}")
            print(f"Generations to 95% fitness: {convergence_generation or 'Not reached'}")
            print(f"Total generations: {stats['generations']}")

            # Plot fitness curve
            if fitness_history:
                print("\nFitness progression:")
                for i in range(0, len(fitness_history), 5):
                    print(f"  Gen {i:3d}: {fitness_history[i]:.3f}")

        finally:
            engine.cleanup()


def run_performance_profile():
    """Run a complete performance profile of the evolution engine."""
    print("Running Evolution Engine Performance Profile...")
    print("=" * 60)

    # Run all benchmarks
    test = TestEvolutionPerformance()

    async def run_all():
        await test.test_population_initialization_speed()
        await test.test_fitness_evaluation_performance()
        await test.test_parallel_evaluation_scaling()
        await test.test_memory_usage_scaling()
        await test.test_generation_throughput()
        await test.test_convergence_speed()

    asyncio.run(run_all())

    print("\n" + "=" * 60)
    print("Performance profile complete!")


if __name__ == "__main__":
    # Run performance profile
    run_performance_profile()

    # Or run with pytest
    # pytest.main([__file__, "-v", "-s", "-m", "benchmark"])
