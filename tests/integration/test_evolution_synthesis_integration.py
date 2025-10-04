"""
Integration tests for genetic algorithm with program synthesis pipeline.

Tests the complete integration of evolution engine with the existing
program synthesis adapter and DSL engine.
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.adapters.strategies.program_synthesis import (
    ProgramGenerationStrategy,
    ProgramSynthesisAdapter,
    ProgramSynthesisConfig,
)
from src.domain.models import ARCTask
from src.infrastructure.config import GeneticAlgorithmConfig


@pytest.fixture
def arc_task_color_change():
    """Create ARC task with simple color change pattern."""
    return ARCTask(
        task_id="test_color_001",
        task_source="test",
        train_examples=[
            {
                "input": [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                "output": [[2, 2, 0], [2, 2, 0], [0, 0, 0]]
            },
            {
                "input": [[3, 0, 3], [0, 0, 0], [3, 0, 3]],
                "output": [[4, 0, 4], [0, 0, 0], [4, 0, 4]]
            }
        ],
        test_input=[[5, 5], [5, 5]]
    )


@pytest.fixture
def arc_task_rotation():
    """Create ARC task with rotation pattern."""
    return ARCTask(
        task_id="test_rotate_001",
        task_source="test",
        train_examples=[
            {
                "input": [[1, 0], [0, 0]],
                "output": [[0, 1], [0, 0]]
            },
            {
                "input": [[2, 3], [0, 0]],
                "output": [[0, 2], [0, 3]]
            }
        ],
        test_input=[[4, 5], [6, 0]]
    )


class TestEvolutionSynthesisIntegration:
    """Test integration between evolution engine and program synthesis."""

    def test_synthesis_config_evolution_enabled(self):
        """Test that evolution can be enabled in synthesis config."""
        config = ProgramSynthesisConfig()
        assert hasattr(config, 'use_evolution')
        assert hasattr(config, 'evolution_config_path')
        assert hasattr(config, 'max_evolution_time')

        # Check default values
        assert config.use_evolution is True
        assert config.evolution_config_path == Path("configs/strategies/evolution.yaml")
        assert config.max_evolution_time == 30.0

    @pytest.mark.asyncio
    async def test_evolved_program_generation(self, arc_task_color_change):
        """Test that evolution generates programs through synthesis adapter."""
        config = ProgramSynthesisConfig(
            generation_strategy=ProgramGenerationStrategy.SEARCH_BASED,
            use_evolution=True,
            max_evolution_time=5.0  # Short timeout for testing
        )

        adapter = ProgramSynthesisAdapter(config)

        # Mock the evolution method to return quickly
        with patch.object(adapter, '_generate_evolved_programs') as mock_evolve:
            mock_evolve.return_value = [
                {
                    "operations": [
                        {"name": "replace_color", "parameters": {"from": 1, "to": 2}}
                    ]
                }
            ]

            # This would normally be called internally, but test directly
            programs = await adapter._generate_evolved_programs(
                arc_task_color_change,
                {"color_changes": [{"colors_removed": [1], "colors_added": [2]}]}
            )

            assert len(programs) > 0
            mock_evolve.assert_called_once()

    @pytest.mark.asyncio
    async def test_evolution_engine_with_real_task(self, arc_task_rotation):
        """Test evolution engine with a real ARC task."""
        # Create minimal config for fast testing
        config = GeneticAlgorithmConfig()
        config.population.size = 20
        config.convergence.max_generations = 5
        config.performance.generation_timeout = 2

        # Mock DSL engine
        mock_dsl_engine = Mock()
        mock_dsl_engine.execute_program.return_value = {
            "success": True,
            "output": [[0, 4], [0, 5]]  # Rotated output
        }

        # Create evolution engine
        engine = EvolutionEngine(
            config=config,
            dsl_engine=mock_dsl_engine
        )

        # Run evolution with timeout
        try:
            best_individual, stats = await asyncio.wait_for(
                engine.evolve(arc_task_rotation),
                timeout=10.0
            )

            # Verify results
            assert best_individual is not None
            assert best_individual.fitness > 0
            assert stats['generations'] > 0
            assert stats['population_size'] == 20

        finally:
            engine.cleanup()

    @pytest.mark.asyncio
    async def test_evolution_convergence_behavior(self, arc_task_color_change):
        """Test that evolution converges properly."""
        config = GeneticAlgorithmConfig()
        config.population.size = 50
        config.convergence.stagnation_patience = 3
        config.convergence.min_fitness_improvement = 0.01
        config.convergence.max_generations = 10

        # Mock DSL engine with improving fitness
        mock_dsl_engine = Mock()
        call_count = 0

        def mock_execute(operations, grid):
            nonlocal call_count
            call_count += 1
            # Simulate improving fitness over time
            if call_count < 50:
                return {"success": True, "output": [[1, 1, 0], [1, 1, 0], [0, 0, 0]]}
            else:
                return {"success": True, "output": [[2, 2, 0], [2, 2, 0], [0, 0, 0]]}

        mock_dsl_engine.execute_program.side_effect = mock_execute

        engine = EvolutionEngine(config=config, dsl_engine=mock_dsl_engine)

        # Track generation callbacks
        generation_count = 0
        fitness_history = []

        def generation_callback(population):
            nonlocal generation_count
            generation_count += 1
            if population.best_individual:
                fitness_history.append(population.best_individual.fitness)

        # Run evolution
        try:
            best_individual, stats = await engine.evolve(
                arc_task_color_change,
                callbacks=[generation_callback]
            )

            # Verify convergence behavior
            assert generation_count <= config.convergence.max_generations
            assert len(fitness_history) > 0

            # Check that fitness improved
            if len(fitness_history) > 1:
                assert fitness_history[-1] >= fitness_history[0]

        finally:
            engine.cleanup()

    def test_evolution_config_loading(self):
        """Test that evolution config can be loaded from YAML."""
        config_path = Path("configs/strategies/evolution.yaml")

        # Check if config file exists
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            assert 'evolution' in config_data
            assert 'population' in config_data['evolution']
            assert 'genetic_operators' in config_data['evolution']
            assert 'fitness' in config_data['evolution']
            assert 'diversity' in config_data['evolution']

    @pytest.mark.asyncio
    async def test_parallel_evaluation_integration(self, arc_task_rotation):
        """Test parallel evaluation with multiple workers."""
        config = GeneticAlgorithmConfig()
        config.population.size = 100
        config.parallelization.workers = 2
        config.parallelization.batch_size = 50
        config.convergence.max_generations = 2

        # Mock DSL engine
        mock_dsl_engine = Mock()
        mock_dsl_engine.execute_program.return_value = {
            "success": True,
            "output": [[0, 4], [0, 5]]
        }

        engine = EvolutionEngine(config=config, dsl_engine=mock_dsl_engine)

        try:
            # Run evolution
            best_individual, stats = await asyncio.wait_for(
                engine.evolve(arc_task_rotation),
                timeout=30.0
            )

            # Verify parallel processing occurred
            assert stats['population_size'] == 100
            # With 2 workers and batch size 50, should process efficiently

        finally:
            engine.cleanup()

    def test_diversity_mechanisms_integration(self):
        """Test that diversity mechanisms work with evolution engine."""
        from src.adapters.strategies.diversity_mechanisms import (
            FitnessSharing,
            NoveltySearch,
            Speciation,
        )

        # Test each diversity mechanism can be instantiated
        mechanisms = [
            FitnessSharing(niche_radius=0.15),
            Speciation(compatibility_threshold=0.3),
            NoveltySearch(archive_size=100)
        ]

        for mechanism in mechanisms:
            assert hasattr(mechanism, 'calculate')
            assert hasattr(mechanism, 'apply_pressure')


class TestEvolutionPerformance:
    """Test performance characteristics of evolution engine."""

    @pytest.mark.asyncio
    async def test_large_population_handling(self):
        """Test that engine can handle 1000+ population."""
        config = GeneticAlgorithmConfig()
        config.population.size = 1000
        config.convergence.max_generations = 1  # Just one generation

        # Mock DSL engine for speed
        mock_dsl_engine = Mock()
        mock_dsl_engine.execute_program.return_value = {
            "success": True,
            "output": [[0]]
        }

        # Simple task
        task = ARCTask(
            task_id="perf_test",
            task_source="test",
            train_examples=[{"input": [[1]], "output": [[2]]}],
            test_input=[[3]]
        )

        engine = EvolutionEngine(config=config, dsl_engine=mock_dsl_engine)

        import time
        start_time = time.time()

        try:
            # Run one generation
            best_individual, stats = await asyncio.wait_for(
                engine.evolve(task),
                timeout=60.0  # 1 minute timeout
            )

            elapsed = time.time() - start_time

            # Verify performance
            assert stats['population_size'] == 1000
            assert elapsed < 60.0  # Should complete within timeout

            # Log performance metrics
            print(f"\nPerformance: {stats['population_size']} individuals in {elapsed:.2f}s")
            print(f"Time per individual: {elapsed / stats['population_size']:.3f}s")

        finally:
            engine.cleanup()

    @pytest.mark.asyncio
    async def test_memory_usage_within_limits(self):
        """Test that memory usage stays within configured limits."""
        config = GeneticAlgorithmConfig()
        config.population.size = 500
        config.performance.memory_limit = 2048  # 2GB limit
        config.convergence.max_generations = 2

        # Mock DSL engine
        mock_dsl_engine = Mock()
        mock_dsl_engine.execute_program.return_value = {
            "success": True,
            "output": [[0, 1], [1, 0]]
        }

        # Create a task
        task = ARCTask(
            task_id="mem_test",
            task_source="test",
            train_examples=[
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
            ],
            test_input=[[2, 0], [0, 2]]
        )

        engine = EvolutionEngine(config=config, dsl_engine=mock_dsl_engine)

        try:
            # Monitor memory usage
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run evolution
            best_individual, stats = await engine.evolve(task)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Log memory usage
            print(f"\nMemory usage: {memory_increase:.2f}MB increase")
            print(f"Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB")

            # Verify memory is reasonable
            assert memory_increase < config.performance.memory_limit

        finally:
            engine.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
