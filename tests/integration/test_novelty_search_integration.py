"""
Integration tests for novelty search with evolution engine.

Tests Task 7.5: Implement novelty search as alternative to fitness optimization.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.adapters.strategies.evolution_config import (
    ConvergenceConfig,
    DiversityConfig,
    EvolutionConfig,
    FitnessConfig,
    GeneticOperatorConfig,
    ParallelizationConfig,
    PerformanceConfig,
    PopulationConfig,
)
from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.domain.models import ARCTask, Grid, InputOutputPair


class TestNoveltySearchIntegration:
    """Test novelty search integration with evolution engine."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock ARC task."""
        task = MagicMock(spec=ARCTask)
        task.task_id = "test_task_001"

        # Create proper grid-like objects
        class MockGrid:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data), len(data[0]) if data else 0)

        input_grid = MockGrid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output_grid = MockGrid([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

        # Create training pair
        train_pair = MagicMock(spec=InputOutputPair)
        train_pair.input = input_grid
        train_pair.output = output_grid

        task.train_pairs = [train_pair]
        task.test_pairs = []

        return task

    @pytest.fixture
    def evolution_config_with_novelty(self):
        """Create evolution config with novelty search enabled."""
        config = EvolutionConfig()

        # Enable novelty search (dictionary that supports .get() method)
        config.novelty_search = {
            'enabled': True,
            'archive_size': 100,
            'min_distance': 0.1,
            'novelty_weight': 0.5,
            'k_neighbors': 5
        }

        # Basic evolution settings
        config.population = PopulationConfig()
        config.population.size = 20
        config.population.elite_size = 2
        config.population.initialization = {"method": "random"}

        config.convergence = ConvergenceConfig()
        config.convergence.max_generations = 5
        config.convergence.stagnation_patience = 3

        config.parallelization = ParallelizationConfig()
        config.parallelization.backend = "sequential"
        config.parallelization.workers = 1
        config.parallelization.batch_size = 10

        config.performance = PerformanceConfig()
        config.performance.memory_limit = 1024
        config.performance.generation_timeout = 30
        config.performance.program_timeout = 1

        # Set other required configs
        config.genetic_operators = GeneticOperatorConfig()
        config.fitness = FitnessConfig()
        config.diversity = DiversityConfig()
        config.diversity.method = "fitness_sharing"
        config.diversity.niche_radius = 0.15

        # Disable other advanced features
        class MockConfig:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config.island_model = MockConfig({'enabled': False})
        config.coevolution = MockConfig({'enabled': False})

        # Reproducibility
        from src.adapters.strategies.evolution_config import ReproducibilityConfig
        config.reproducibility = ReproducibilityConfig()
        config.reproducibility.checkpoint_enabled = False
        config.reproducibility.deterministic = False
        config.reproducibility.seed = None  # Use None for random seed

        return config

    @pytest.mark.asyncio
    async def test_novelty_search_initialization(self, evolution_config_with_novelty):
        """Test that novelty search is properly initialized."""
        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        engine = EvolutionEngine(evolution_config_with_novelty, mock_dsl)

        assert engine.novelty_search_enabled is True
        assert engine.novelty_search_engine is not None
        assert engine.novelty_search_engine.novelty_weight == 0.5
        assert engine.novelty_search_engine.archive.max_size == 100

    @pytest.mark.asyncio
    async def test_novelty_evaluation_integration(self, evolution_config_with_novelty, mock_task):
        """Test that novelty search is integrated into fitness evaluation."""
        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        engine = EvolutionEngine(evolution_config_with_novelty, mock_dsl)

        # Mock fitness evaluator
        class MockFitnessEvaluator:
            def evaluate(self, individual):
                return 0.5
        
        engine.fitness_evaluator = MockFitnessEvaluator()

        # Initialize population
        await engine._initialize_population(mock_task)
        engine.current_task = mock_task

        # Mock execution results
        for i, ind in enumerate(engine.population.individuals):
            # Create varying outputs for behavioral diversity
            class MockGrid:
                def __init__(self, data):
                    self.data = data
                    self.shape = (len(data), len(data[0]) if data else 0)
            
            # Different patterns for different individuals
            if i == 0:
                grid_data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # Uniform
            elif i == 1:
                grid_data = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]  # Repeating
            else:
                # Random pattern
                grid_data = [[j % 9, (j+1) % 9, (j+2) % 9] for j in range(3)]
            
            output = MockGrid(grid_data)
            ind.metadata['execution_result'] = output

        # Evaluate population
        await engine._evaluate_population()

        # Check that novelty scores were calculated
        for ind in engine.population.individuals:
            assert 'novelty_score' in ind.metadata
            assert 'behavior_features' in ind.metadata
            assert isinstance(ind.metadata['behavior_features'], list)
            assert len(ind.metadata['behavior_features']) == 50  # Feature vector size

        # Check that archive was populated
        assert len(engine.novelty_search_engine.archive.behaviors) > 0

        # Check diversity metrics
        assert 'novelty_archive_size' in engine.population.diversity_metrics
        assert 'average_novelty' in engine.population.diversity_metrics

    @pytest.mark.asyncio
    async def test_fitness_novelty_combination(self, evolution_config_with_novelty, mock_task):
        """Test that fitness and novelty are properly combined."""
        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        engine = EvolutionEngine(evolution_config_with_novelty, mock_dsl)

        # Set novelty weight to 0.3 for testing
        engine.novelty_search_engine.novelty_weight = 0.3

        # Create test population
        from src.adapters.strategies.evolution_engine import Individual, Population
        engine.population = Population()

        # Create individuals with known objective fitness
        for i in range(5):
            ind = Individual(operations=[])
            ind.id = f"test_ind_{i}"
            ind.fitness = 0.8  # Objective fitness
            engine.population.add_individual(ind)

        # Mock execution results with different behaviors
        execution_results = {}
        for i, ind in enumerate(engine.population.individuals):
            grid = MagicMock(spec=Grid)
            grid.shape = (2, 2)
            grid.data = [[i, i], [i, i]]  # Different uniform patterns
            execution_results[ind.id] = grid

        engine.current_task = mock_task

        # Calculate novelty and update fitness
        objective_fitness = {ind.id: ind.fitness for ind in engine.population.individuals}
        novelty_scores = engine.novelty_search_engine.evaluate_novelty(
            engine.population, mock_task, execution_results
        )

        engine.novelty_search_engine.update_population_fitness(
            engine.population, objective_fitness, novelty_scores
        )

        # Check combined fitness
        for ind in engine.population.individuals:
            # Combined should be weighted average
            obj_fit = ind.metadata['objective_fitness']
            nov_fit = ind.metadata['novelty_fitness']
            expected = 0.7 * obj_fit + 0.3 * nov_fit

            assert pytest.approx(ind.fitness) == expected
            assert ind.metadata['combined_fitness'] == ind.fitness

    @pytest.mark.asyncio
    async def test_novelty_archive_management(self, evolution_config_with_novelty, mock_task):
        """Test novelty archive size management."""
        # Set small archive for testing
        evolution_config_with_novelty.novelty_search['archive_size'] = 10
        evolution_config_with_novelty.novelty_search['min_distance'] = 0.01

        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        engine = EvolutionEngine(evolution_config_with_novelty, mock_dsl)
        engine.current_task = mock_task

        # Generate many diverse behaviors
        from src.adapters.strategies.evolution_engine import Individual, Population

        for generation in range(3):
            engine.population = Population()
            engine.population.generation = generation

            # Create 20 individuals per generation
            for i in range(20):
                ind = Individual(operations=[])
                ind.id = f"gen_{generation}_ind_{i}"
                ind.fitness = 0.5
                engine.population.add_individual(ind)

            # Create diverse execution results
            execution_results = {}
            for i, ind in enumerate(engine.population.individuals):
                grid = MagicMock(spec=Grid)
                grid.shape = (3, 3)
                # Create unique pattern for each
                grid.data = [[
                    (generation * 20 + i + j + k) % 10
                    for k in range(3)
                ] for j in range(3)]
                execution_results[ind.id] = grid

            # Evaluate novelty
            engine.novelty_search_engine.evaluate_novelty(
                engine.population, mock_task, execution_results
            )

        # Archive should respect size limit
        assert len(engine.novelty_search_engine.archive.behaviors) <= 10

    @pytest.mark.asyncio
    async def test_evolution_with_novelty_search(self, evolution_config_with_novelty, mock_task):
        """Test complete evolution run with novelty search."""
        # Reduce generations for faster test
        evolution_config_with_novelty.convergence.max_generations = 3

        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        engine = EvolutionEngine(evolution_config_with_novelty, mock_dsl)

        # Mock components are already set

        mock_evaluator = MagicMock()
        # Return varying fitness to simulate evolution
        mock_evaluator.evaluate.side_effect = lambda ind: np.random.rand() * 0.5 + 0.3
        engine.fitness_evaluator = mock_evaluator

        # Mock execution for each evaluation
        original_eval = engine._evaluate_batch_sequential

        async def mock_eval_batch(batch):
            await original_eval(batch)
            # Add execution results
            for ind in batch:
                grid = MagicMock(spec=Grid)
                grid.shape = (3, 3)
                # Create somewhat random patterns
                grid.data = [
                    [int(ind.fitness * 10) % 10 for _ in range(3)]
                    for _ in range(3)
                ]
                ind.metadata['execution_result'] = grid

        engine._evaluate_batch_sequential = mock_eval_batch

        # Run evolution
        best_individual, stats = await engine.evolve(mock_task)

        # Verify novelty search was used
        assert stats['novelty_search_enabled'] is True
        assert stats['novelty_search_stats'] is not None
        assert stats['novelty_search_stats']['archive_size'] > 0
        assert stats['novelty_weight'] == 0.5

        # Check that behaviors were tracked
        assert 'average_novelty' in stats['diversity_metrics']
        assert stats['diversity_metrics']['novelty_archive_size'] > 0

    def test_novelty_disabled_by_default(self):
        """Test that novelty search is disabled when not configured."""
        config = EvolutionConfig()
        # Don't set novelty_search config
        
        # Mock DSL engine for testing
        mock_dsl = MagicMock()
        
        engine = EvolutionEngine(config, mock_dsl)

        assert engine.novelty_search_enabled is False
        assert engine.novelty_search_engine is None
