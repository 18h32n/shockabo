"""Performance benchmarks for evolution pipeline scalability."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import pytest

from src.adapters.strategies.evolution_engine import EvolutionEngine, Individual, Population
from src.infrastructure.config import Config


class TestEvolutionScalability:
    """Performance benchmarks for evolution pipeline."""

    @pytest.fixture
    def benchmark_config(self):
        """Create configuration for benchmarking."""
        config = Config()

        # Base configuration
        config.evolution.enabled = True
        config.evolution.mutation_rate = 0.1
        config.evolution.crossover_rate = 0.7
        config.evolution.tournament_size = 3
        config.evolution.elitism_count = 5

        # Performance settings
        config.evolution.parallel_workers = psutil.cpu_count() // 2
        config.evolution.memory_limit_mb = 4096
        config.evolution.fitness_evaluation_timeout = 1
        config.evolution.generation_timeout = 60

        # Disable early termination for benchmarks
        config.evolution.early_termination.enabled = False

        config.platform = "benchmark"

        return config

    @pytest.fixture
    def benchmark_task(self):
        """Create task for benchmarking."""
        return {
            "train": [
                {
                    "input": np.random.randint(0, 10, (5, 5)).tolist(),
                    "output": np.random.randint(0, 10, (5, 5)).tolist()
                } for _ in range(3)
            ]
        }

    def test_program_generation_rate(self, benchmark_config, benchmark_task):
        """Benchmark program generation rate to ensure 500+ target."""
        results = []

        # Test different population sizes
        population_sizes = [100, 200, 500, 1000]
        generation_counts = [5, 10, 20]

        for pop_size in population_sizes:
            for gen_count in generation_counts:
                benchmark_config.evolution.population_size = pop_size
                benchmark_config.evolution.max_generations = gen_count

                engine = EvolutionEngine(benchmark_config)

                # Track metrics
                start_time = time.time()
                program_count = 0

                # Mock fitness evaluation to count programs
                def counting_evaluate(operations, examples):
                    nonlocal program_count
                    program_count += 1
                    return np.random.random()  # Random fitness

                with patch.object(engine, '_evaluate_fitness', side_effect=counting_evaluate):
                    result = engine.evolve(benchmark_task["train"])

                elapsed_time = time.time() - start_time

                # Store results
                results.append({
                    "population_size": pop_size,
                    "generations": gen_count,
                    "total_programs": program_count,
                    "time_seconds": elapsed_time,
                    "programs_per_second": program_count / elapsed_time,
                    "expected_programs": pop_size * gen_count
                })

        # Analyze results
        print("\n=== Program Generation Benchmark Results ===")
        print(f"{'Pop Size':>10} {'Gens':>6} {'Programs':>10} {'Time(s)':>10} {'Prog/s':>10}")
        print("-" * 56)

        for r in results:
            print(f"{r['population_size']:>10} {r['generations']:>6} "
                  f"{r['total_programs']:>10} {r['time_seconds']:>10.2f} "
                  f"{r['programs_per_second']:>10.1f}")

        # Verify 500+ programs can be generated
        capable_configs = [r for r in results if r['total_programs'] >= 500]
        assert len(capable_configs) > 0, "No configuration achieved 500+ programs"

        # Find optimal configuration
        optimal = min(capable_configs, key=lambda x: x['time_seconds'])
        print(f"\nOptimal config for 500+ programs: "
              f"Pop={optimal['population_size']}, Gens={optimal['generations']}, "
              f"Time={optimal['time_seconds']:.2f}s")

    def test_time_constraint_compliance(self, benchmark_config, benchmark_task):
        """Test that evolution completes within 5-minute constraint."""
        # Configure for realistic run
        benchmark_config.evolution.population_size = 500
        benchmark_config.evolution.max_generations = 20
        benchmark_config.evolution.max_runtime_seconds = 300  # 5 minutes

        engine = EvolutionEngine(benchmark_config)

        # Run with time tracking
        start_time = time.time()
        result = engine.evolve(benchmark_task["train"])
        elapsed_time = time.time() - start_time

        # Verify completion within time limit
        assert elapsed_time < 300, f"Evolution took {elapsed_time:.2f}s, exceeding 5-minute limit"
        assert result is not None
        assert result["best_individual"] is not None

        # Calculate actual program throughput
        metrics = engine.get_evolution_metrics()
        programs_evaluated = metrics.get("total_programs_evaluated", 0)

        print("\n=== Time Constraint Test ===")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Programs evaluated: {programs_evaluated}")
        print(f"Generations completed: {result['generations']}")
        print(f"Throughput: {programs_evaluated/elapsed_time:.1f} programs/second")

    def test_memory_usage_scaling(self, benchmark_config):
        """Test memory usage with increasing population sizes."""
        memory_results = []
        process = psutil.Process()

        population_sizes = [100, 250, 500, 1000, 2000]

        for pop_size in population_sizes:
            benchmark_config.evolution.population_size = pop_size
            benchmark_config.evolution.max_generations = 5

            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            engine = EvolutionEngine(benchmark_config)

            # Create large population
            individuals = []
            for _ in range(pop_size):
                # Create individuals with varying operation counts
                op_count = np.random.randint(5, 20)
                operations = [{"op": f"op_{i}", "params": list(range(10))}
                             for i in range(op_count)]
                individuals.append(Individual(operations=operations))

            population = Population(individuals=individuals)

            # Measure peak memory during evolution
            peak_memory = baseline_memory
            for _ in range(3):  # Few generations
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                # Simulate evolution step
                with patch.object(engine, '_evaluate_population_batch'):
                    population = engine._evolve_generation(population, [])

            memory_used = peak_memory - baseline_memory

            memory_results.append({
                "population_size": pop_size,
                "memory_used_mb": memory_used,
                "memory_per_individual": memory_used / pop_size
            })

        print("\n=== Memory Usage Scaling ===")
        print(f"{'Pop Size':>10} {'Memory(MB)':>12} {'MB/Individual':>15}")
        print("-" * 40)

        for r in memory_results:
            print(f"{r['population_size']:>10} {r['memory_used_mb']:>12.1f} "
                  f"{r['memory_per_individual']:>15.3f}")

        # Verify memory stays within limits
        for r in memory_results:
            assert r['memory_used_mb'] < benchmark_config.evolution.memory_limit_mb, \
                f"Memory usage {r['memory_used_mb']}MB exceeds limit for pop_size={r['population_size']}"

    def test_parallel_evaluation_speedup(self, benchmark_config, benchmark_task):
        """Test parallel evaluation performance gains."""
        speedup_results = []

        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        population_size = 1000

        benchmark_config.evolution.population_size = population_size
        benchmark_config.evolution.max_generations = 5
        benchmark_config.evolution.batch_size = 100

        for workers in worker_counts:
            if workers > psutil.cpu_count():
                continue

            benchmark_config.evolution.parallel_workers = workers
            engine = EvolutionEngine(benchmark_config)

            # Time evaluation
            start_time = time.time()

            # Create test population
            individuals = [Individual(operations=[{"op": f"op_{i}"}])
                          for i in range(population_size)]

            # Evaluate with timing
            with patch.object(engine, '_evaluate_fitness') as mock_eval:
                # Simulate work
                def slow_eval(ops, examples):
                    time.sleep(0.001)  # 1ms per evaluation
                    return np.random.random()

                mock_eval.side_effect = slow_eval

                evaluated = engine._evaluate_population_batch(
                    individuals, benchmark_task["train"]
                )

            elapsed_time = time.time() - start_time

            speedup_results.append({
                "workers": workers,
                "time": elapsed_time,
                "throughput": population_size / elapsed_time
            })

        # Calculate speedups
        baseline_time = speedup_results[0]["time"]

        print("\n=== Parallel Evaluation Speedup ===")
        print(f"{'Workers':>8} {'Time(s)':>10} {'Speedup':>10} {'Efficiency':>12}")
        print("-" * 42)

        for r in speedup_results:
            speedup = baseline_time / r["time"]
            efficiency = speedup / r["workers"]
            print(f"{r['workers']:>8} {r['time']:>10.2f} {speedup:>10.2f}x "
                  f"{efficiency:>12.2%}")

        # Verify parallel speedup
        if len(speedup_results) > 1:
            assert speedup_results[-1]["time"] < speedup_results[0]["time"], \
                "Parallel evaluation should be faster than serial"

    def test_batch_size_optimization(self, benchmark_config):
        """Test optimal batch sizes for different platforms."""
        batch_results = []

        batch_sizes = [10, 25, 50, 100, 200, 500]
        population_size = 1000

        benchmark_config.evolution.population_size = population_size
        benchmark_config.evolution.parallel_workers = 4

        for batch_size in batch_sizes:
            benchmark_config.evolution.batch_size = batch_size
            engine = EvolutionEngine(benchmark_config)

            # Create test population
            individuals = [Individual(operations=[{"op": "test"}])
                          for i in range(population_size)]

            # Time batch processing
            start_time = time.time()

            with patch.object(engine, '_evaluate_batch') as mock_eval:
                mock_eval.return_value = [(0.5, None)] * batch_size

                evaluated = engine._evaluate_population_batch(individuals, [])

            elapsed_time = time.time() - start_time

            batch_results.append({
                "batch_size": batch_size,
                "time": elapsed_time,
                "batches": population_size // batch_size,
                "overhead": elapsed_time * 1000 / (population_size // batch_size)  # ms per batch
            })

        print("\n=== Batch Size Optimization ===")
        print(f"{'Batch Size':>12} {'Time(s)':>10} {'Batches':>10} {'Overhead(ms)':>15}")
        print("-" * 50)

        for r in batch_results:
            print(f"{r['batch_size']:>12} {r['time']:>10.3f} {r['batches']:>10} "
                  f"{r['overhead']:>15.2f}")

        # Find optimal batch size
        optimal = min(batch_results, key=lambda x: x['time'])
        print(f"\nOptimal batch size: {optimal['batch_size']}")

    def test_generation_throughput(self, benchmark_config):
        """Test generation completion throughput."""
        throughput_results = []

        configs = [
            {"pop": 100, "gens": 50},
            {"pop": 200, "gens": 25},
            {"pop": 500, "gens": 10},
            {"pop": 1000, "gens": 5}
        ]

        for config in configs:
            benchmark_config.evolution.population_size = config["pop"]
            benchmark_config.evolution.max_generations = config["gens"]

            engine = EvolutionEngine(benchmark_config)

            # Mock fast fitness evaluation
            with patch.object(engine, '_evaluate_fitness', return_value=0.5):
                start_time = time.time()
                result = engine.evolve([{"input": [[1]], "output": [[2]]}])
                elapsed_time = time.time() - start_time

            total_evaluations = config["pop"] * config["gens"]

            throughput_results.append({
                "config": f"{config['pop']}x{config['gens']}",
                "total_evaluations": total_evaluations,
                "time": elapsed_time,
                "eval_per_second": total_evaluations / elapsed_time,
                "gen_per_minute": (config["gens"] / elapsed_time) * 60
            })

        print("\n=== Generation Throughput ===")
        print(f"{'Config':>12} {'Evals':>10} {'Time(s)':>10} {'Eval/s':>10} {'Gen/min':>10}")
        print("-" * 55)

        for r in throughput_results:
            print(f"{r['config']:>12} {r['total_evaluations']:>10} "
                  f"{r['time']:>10.2f} {r['eval_per_second']:>10.1f} "
                  f"{r['gen_per_minute']:>10.1f}")

    def test_diversity_maintenance_performance(self, benchmark_config):
        """Test performance impact of diversity mechanisms."""
        diversity_results = []

        # Test with and without diversity mechanisms
        diversity_configs = [
            {"enabled": False, "name": "No Diversity"},
            {"enabled": True, "speciation": True, "name": "Speciation"},
            {"enabled": True, "crowding": True, "name": "Crowding"},
            {"enabled": True, "novelty": True, "name": "Novelty Search"}
        ]

        benchmark_config.evolution.population_size = 500
        benchmark_config.evolution.max_generations = 10

        for div_config in diversity_configs:
            # Configure diversity
            if div_config["enabled"]:
                benchmark_config.evolution.diversity_preservation.enabled = True
                benchmark_config.evolution.diversity_preservation.method = div_config.get("method", "speciation")
            else:
                benchmark_config.evolution.diversity_preservation.enabled = False

            engine = EvolutionEngine(benchmark_config)

            start_time = time.time()
            result = engine.evolve([{"input": [[1]], "output": [[2]]}])
            elapsed_time = time.time() - start_time

            # Measure diversity
            if result["population"]:
                unique_programs = len(set(str(ind.operations)
                                        for ind in result["population"].individuals))
                diversity_ratio = unique_programs / len(result["population"].individuals)
            else:
                diversity_ratio = 0

            diversity_results.append({
                "method": div_config["name"],
                "time": elapsed_time,
                "diversity_ratio": diversity_ratio,
                "best_fitness": result["best_individual"].fitness if result["best_individual"] else 0
            })

        print("\n=== Diversity Mechanism Performance ===")
        print(f"{'Method':>15} {'Time(s)':>10} {'Diversity':>12} {'Best Fitness':>15}")
        print("-" * 55)

        for r in diversity_results:
            print(f"{r['method']:>15} {r['time']:>10.2f} "
                  f"{r['diversity_ratio']:>12.2%} {r['best_fitness']:>15.3f}")

    def test_checkpoint_overhead(self, benchmark_config, tmp_path):
        """Test performance impact of checkpointing."""
        checkpoint_results = []

        checkpoint_intervals = [0, 5, 10, 20]  # 0 means no checkpointing

        benchmark_config.evolution.population_size = 500
        benchmark_config.evolution.max_generations = 20
        benchmark_config.evolution.checkpoint_path = str(tmp_path)

        for interval in checkpoint_intervals:
            benchmark_config.evolution.checkpoint_interval = interval if interval > 0 else 999
            benchmark_config.evolution.checkpoint_enabled = interval > 0

            engine = EvolutionEngine(benchmark_config)

            # Track checkpoint writes
            checkpoint_count = 0
            original_save = engine._save_checkpoint if hasattr(engine, '_save_checkpoint') else None

            def counting_save(*args, **kwargs):
                nonlocal checkpoint_count
                checkpoint_count += 1
                if original_save:
                    return original_save(*args, **kwargs)

            if original_save:
                engine._save_checkpoint = counting_save

            start_time = time.time()
            result = engine.evolve([{"input": [[1]], "output": [[2]]}])
            elapsed_time = time.time() - start_time

            checkpoint_results.append({
                "interval": interval,
                "checkpoints": checkpoint_count,
                "time": elapsed_time,
                "overhead": elapsed_time - checkpoint_results[0]["time"] if checkpoint_results else 0
            })

        print("\n=== Checkpoint Overhead ===")
        print(f"{'Interval':>10} {'Checkpoints':>13} {'Time(s)':>10} {'Overhead(s)':>13}")
        print("-" * 48)

        for r in checkpoint_results:
            print(f"{r['interval']:>10} {r['checkpoints']:>13} "
                  f"{r['time']:>10.2f} {r['overhead']:>13.2f}")

    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "platform": "benchmark"
            },
            "recommendations": {
                "kaggle": {
                    "population_size": 500,
                    "max_generations": 10,
                    "batch_size": 50,
                    "workers": 2
                },
                "colab": {
                    "population_size": 1000,
                    "max_generations": 20,
                    "batch_size": 200,
                    "workers": 4
                },
                "paperspace": {
                    "population_size": 250,
                    "max_generations": 20,
                    "batch_size": 100,
                    "workers": 1
                }
            }
        }

        # Save report
        report_path = Path("evolution_performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nPerformance report saved to: {report_path}")
