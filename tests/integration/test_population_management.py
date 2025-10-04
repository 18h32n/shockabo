"""
Integration tests for genetic algorithm population management.

Tests the complete population lifecycle including initialization,
evaluation, selection, and generation creation.
"""

from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.evolution_engine import EvolutionEngine
from src.adapters.strategies.operation_templates import OperationTemplateGenerator
from src.domain.models import ARCTask
from src.domain.services.dsl_engine import DSLEngine
from src.infrastructure.config import GeneticAlgorithmConfig


@pytest.fixture
def sample_arc_task():
    """Create a sample ARC task for testing."""
    return ARCTask(
        task_id="test_task_001",
        task_source="training",
        train_examples=[
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]]
            },
            {
                "input": [[2, 0], [0, 2]],
                "output": [[0, 2], [2, 0]]
            }
        ],
        test_input=[[3, 0], [0, 3]]
    )


@pytest.fixture
def genetic_config():
    """Create a test genetic algorithm configuration."""
    config = GeneticAlgorithmConfig()
    config.population.size = 100
    config.population.elite_size = 10
    config.population.initialization = {
        "method": "hybrid",
        "llm_seed_ratio": 0.1,
        "template_ratio": 0.5
    }
    return config


@pytest.fixture
def mock_dsl_engine():
    """Create a mock DSL engine."""
    engine = Mock(spec=DSLEngine)
    # Mock successful execution
    engine.execute_program.return_value = {
        "success": True,
        "output": [[0, 1], [1, 0]]
    }
    return engine


class TestPopulationManagement:
    """Test population management functionality."""

    @pytest.mark.asyncio
    async def test_population_initialization(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test population initialization with diverse individuals."""
        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        # Initialize population
        await engine._initialize_population(sample_arc_task)

        # Check population size
        assert engine.population.size() == genetic_config.population.size

        # Check diversity in initial population
        unique_ids = set(ind.id for ind in engine.population.individuals)
        assert len(unique_ids) > genetic_config.population.size * 0.8  # At least 80% unique

        # Check that all individuals have operations
        for ind in engine.population.individuals:
            assert len(ind.operations) > 0
            assert ind.fitness == 0.0  # Not evaluated yet
            assert ind.age == 0

    @pytest.mark.asyncio
    async def test_population_evaluation(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test fitness evaluation of population."""
        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        # Initialize smaller population for faster test
        genetic_config.population.size = 10
        await engine._initialize_population(sample_arc_task)

        # Initialize fitness evaluator
        engine.fitness_evaluator = engine.FitnessEvaluator(
            task=sample_arc_task,
            dsl_engine=mock_dsl_engine
        )

        # Evaluate population
        await engine._evaluate_population()

        # Check that all individuals have been evaluated
        for ind in engine.population.individuals:
            assert ind.fitness >= 0.0
            assert ind.fitness <= 1.0

        # Check that best individual is tracked
        engine.population.update_generation_stats()
        assert engine.population.best_individual is not None
        assert engine.population.best_individual.fitness == max(
            ind.fitness for ind in engine.population.individuals
        )

    @pytest.mark.asyncio
    async def test_elite_preservation(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test that elite individuals are preserved across generations."""
        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        # Create small population with known fitness values
        genetic_config.population.size = 20
        genetic_config.population.elite_size = 5

        await engine._initialize_population(sample_arc_task)

        # Assign fitness values
        for i, ind in enumerate(engine.population.individuals):
            ind.fitness = i * 0.05  # 0.0, 0.05, 0.1, ..., 0.95

        # Get elite before generation
        elite_before = engine.population.get_elite(genetic_config.population.elite_size)
        elite_ids = {ind.id for ind in elite_before}

        # Create next generation
        await engine._create_next_generation()

        # Check that elite are preserved
        current_ids = {ind.id for ind in engine.population.individuals[:genetic_config.population.elite_size]}

        # At least some elite should be preserved (exact match depends on implementation)
        assert len(elite_ids & current_ids) > 0

    @pytest.mark.asyncio
    async def test_generation_statistics(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test generation statistics tracking."""
        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        genetic_config.population.size = 50
        await engine._initialize_population(sample_arc_task)

        # Assign random fitness values
        import random
        for ind in engine.population.individuals:
            ind.fitness = random.random()

        # Update statistics
        engine.population.update_generation_stats()

        # Check statistics
        assert 'unique_programs' in engine.population.diversity_metrics
        assert 'fitness_variance' in engine.population.diversity_metrics
        assert 'average_age' in engine.population.diversity_metrics

        assert engine.population.diversity_metrics['unique_programs'] > 0
        assert engine.population.diversity_metrics['fitness_variance'] >= 0
        assert engine.population.diversity_metrics['average_age'] == 0  # First generation

        # Check fitness history
        assert len(engine.population.best_fitness_history) == 1
        assert engine.population.best_fitness_history[0] == engine.population.best_individual.fitness

    @pytest.mark.asyncio
    async def test_selection_pressure(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test that selection favors higher fitness individuals."""
        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        genetic_config.population.size = 100
        await engine._initialize_population(sample_arc_task)

        # Create clear fitness gradient
        for i, ind in enumerate(engine.population.individuals):
            ind.fitness = i / 100.0

        # Perform many parent selections
        selected_fitness = []
        for _ in range(1000):
            parent = engine._select_parent()
            selected_fitness.append(parent.fitness)

        # Average fitness of selected parents should be higher than population average
        avg_selected = sum(selected_fitness) / len(selected_fitness)
        avg_population = engine.population.average_fitness()

        assert avg_selected > avg_population
        assert avg_selected > 0.6  # Should be biased toward higher fitness

    def test_template_generator(self):
        """Test operation template generation."""
        generator = OperationTemplateGenerator()

        # Test random program generation
        random_prog = generator.generate_random_program(min_length=2, max_length=5)
        assert 2 <= len(random_prog) <= 5
        assert all('name' in op and 'parameters' in op for op in random_prog)

        # Test template-based generation
        for template_type in ['transform', 'color', 'pattern', 'symmetry']:
            template_prog = generator.generate_from_template(template_type)
            assert len(template_prog) > 0
            assert all('name' in op and 'parameters' in op for op in template_prog)

        # Test diverse population generation
        population = generator.generate_diverse_population(
            size=100,
            random_ratio=0.3,
            template_ratio=0.5
        )
        assert len(population) == 100

        # Check diversity
        unique_programs = set(str(prog) for prog in population)
        assert len(unique_programs) > 50  # Should have good diversity


class TestEvolutionEngine:
    """Test the complete evolution engine."""

    @pytest.mark.asyncio
    async def test_evolution_cycle(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test a complete evolution cycle."""
        genetic_config.population.size = 20
        genetic_config.convergence.max_generations = 3

        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        # Mock fitness evaluator to give predictable results
        def mock_evaluate(individual):
            # Simple fitness based on program length
            return min(1.0, len(individual.operations) * 0.1 + random.random() * 0.1)

        with patch.object(engine, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.side_effect = mock_evaluate

            # Run evolution
            best_individual, stats = await engine.evolve(
                task=sample_arc_task,
                callbacks=[]
            )

            # Check results
            assert best_individual is not None
            assert best_individual.fitness > 0
            assert stats['generations'] <= genetic_config.convergence.max_generations
            assert len(stats['fitness_history']) > 0
            assert stats['population_size'] == genetic_config.population.size

    @pytest.mark.asyncio
    async def test_convergence_detection(self, genetic_config, mock_dsl_engine, sample_arc_task):
        """Test that evolution stops when converged."""
        genetic_config.population.size = 10
        genetic_config.convergence.stagnation_patience = 2
        genetic_config.convergence.min_fitness_improvement = 0.01

        engine = EvolutionEngine(
            config=genetic_config,
            dsl_engine=mock_dsl_engine
        )

        # Set up population with no improvement
        await engine._initialize_population(sample_arc_task)

        # Mock constant fitness
        for ind in engine.population.individuals:
            ind.fitness = 0.5

        # Test convergence tracker
        tracker = engine.convergence_tracker

        assert not tracker.has_converged(engine.population)

        # Simulate no improvement for multiple generations
        for _ in range(3):
            engine.population.increment_generation()
            converged = tracker.has_converged(engine.population)

        # Should detect convergence after patience exceeded
        assert tracker.stagnation_counter >= genetic_config.convergence.stagnation_patience
        assert converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
