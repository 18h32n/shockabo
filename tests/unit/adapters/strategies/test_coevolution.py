"""
Unit tests for co-evolution of programs and fitness functions.

Tests Task 7.4: Add co-evolution of programs and fitness functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.adapters.strategies.coevolution import (
    CoevolutionEngine,
    FitnessComponent,
    FitnessComponentLibrary,
    FitnessFunction,
)
from src.adapters.strategies.evolution_engine import Individual, Population
from src.domain.models import Grid


class TestFitnessComponent:
    """Test fitness component functionality."""

    def test_component_evaluation(self):
        """Test basic component evaluation."""
        def simple_evaluator(pred, target):
            return 1.0 if pred == target else 0.0

        component = FitnessComponent(
            name="test",
            weight=0.5,
            evaluator=simple_evaluator
        )

        grid1 = MagicMock()
        grid2 = MagicMock()

        # Test weighted evaluation
        score = component.evaluate(grid1, grid1)
        assert score == 0.5  # weight * 1.0

        score = component.evaluate(grid1, grid2)
        assert score == 0.0  # weight * 0.0

    def test_component_error_handling(self):
        """Test component handles evaluation errors."""
        def error_evaluator(pred, target):
            raise ValueError("Test error")

        component = FitnessComponent(
            name="error_test",
            weight=1.0,
            evaluator=error_evaluator
        )

        score = component.evaluate(MagicMock(), MagicMock())
        assert score == 0.0


class TestFitnessFunction:
    """Test fitness function functionality."""

    def test_fitness_function_creation(self):
        """Test creating fitness function."""
        components = [
            FitnessComponent("comp1", 0.3, lambda p, t: 0.8),
            FitnessComponent("comp2", 0.7, lambda p, t: 0.6)
        ]

        fitness_func = FitnessFunction(components=components)

        assert len(fitness_func.components) == 2
        assert fitness_func.id != ""  # ID should be generated
        assert fitness_func.age == 0

    def test_fitness_evaluation(self):
        """Test evaluating fitness with multiple components."""
        components = [
            FitnessComponent("exact", 0.5, lambda p, t: 1.0),
            FitnessComponent("partial", 0.5, lambda p, t: 0.5)
        ]

        fitness_func = FitnessFunction(components=components)

        score = fitness_func.evaluate(MagicMock(), MagicMock())
        assert score == 0.75  # (0.5 * 1.0 + 0.5 * 0.5) / 1.0

    def test_weight_normalization(self):
        """Test weight normalization."""
        components = [
            FitnessComponent("c1", 2.0, lambda p, t: 1.0),
            FitnessComponent("c2", 3.0, lambda p, t: 1.0)
        ]

        fitness_func = FitnessFunction(components=components)
        fitness_func.normalize_weights()

        # Weights should sum to 1.0
        total_weight = sum(c.weight for c in fitness_func.components)
        assert pytest.approx(total_weight) == 1.0
        assert pytest.approx(fitness_func.components[0].weight) == 0.4
        assert pytest.approx(fitness_func.components[1].weight) == 0.6


class TestFitnessComponentLibrary:
    """Test fitness component library."""

    def test_available_components(self):
        """Test getting available components."""
        components = FitnessComponentLibrary.get_available_components()

        assert len(components) > 0
        component_names = [c.name for c in components]

        # Check essential components exist
        assert "exact_match" in component_names
        assert "pixel_accuracy" in component_names
        assert "shape_match" in component_names
        assert "color_distribution" in component_names

    def test_exact_match(self):
        """Test exact match component."""
        # Create mock grids
        grid1 = MagicMock(spec=Grid)
        grid1.shape = (2, 2)
        grid1.data = [[1, 2], [3, 4]]

        grid2 = MagicMock(spec=Grid)
        grid2.shape = (2, 2)
        grid2.data = [[1, 2], [3, 4]]

        grid3 = MagicMock(spec=Grid)
        grid3.shape = (2, 2)
        grid3.data = [[1, 2], [3, 5]]  # Different

        # Test exact match
        assert FitnessComponentLibrary._exact_match(grid1, grid2) == 1.0
        assert FitnessComponentLibrary._exact_match(grid1, grid3) == 0.0

    def test_pixel_accuracy(self):
        """Test pixel accuracy component."""
        grid1 = MagicMock(spec=Grid)
        grid1.shape = (2, 2)
        grid1.data = [[1, 2], [3, 4]]

        grid2 = MagicMock(spec=Grid)
        grid2.shape = (2, 2)
        grid2.data = [[1, 2], [3, 5]]  # 3/4 match

        accuracy = FitnessComponentLibrary._pixel_accuracy(grid1, grid2)
        assert pytest.approx(accuracy) == 0.75

    def test_shape_match(self):
        """Test shape matching component."""
        grid1 = MagicMock(spec=Grid)
        grid1.shape = (3, 3)

        grid2 = MagicMock(spec=Grid)
        grid2.shape = (3, 3)

        grid3 = MagicMock(spec=Grid)
        grid3.shape = (6, 6)

        # Exact match
        assert FitnessComponentLibrary._shape_match(grid1, grid2) == 1.0

        # Different size
        score = FitnessComponentLibrary._shape_match(grid1, grid3)
        assert 0 < score < 1  # Partial credit

    def test_color_distribution(self):
        """Test color distribution component."""
        grid1 = MagicMock(spec=Grid)
        grid1.data = [[1, 1, 2], [2, 3, 3]]  # 2 ones, 2 twos, 2 threes

        grid2 = MagicMock(spec=Grid)
        grid2.data = [[1, 2, 3], [1, 2, 3]]  # Same distribution

        grid3 = MagicMock(spec=Grid)
        grid3.data = [[1, 1, 1], [1, 1, 1]]  # All ones

        # Same distribution
        score1 = FitnessComponentLibrary._color_distribution(grid1, grid2)
        assert score1 == 1.0

        # Different distribution
        score2 = FitnessComponentLibrary._color_distribution(grid1, grid3)
        assert score2 < 1.0

    def test_symmetry_detection(self):
        """Test symmetry detection."""
        # Create a mock Grid class that behaves like the expected interface
        class MockGrid:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data), len(data[0]))
        
        # Horizontally symmetric grid
        h_sym_grid = MockGrid([[1, 2, 1], [3, 4, 3], [1, 2, 1]])

        # Vertically symmetric grid
        v_sym_grid = MockGrid([[1, 2, 1], [2, 4, 2], [3, 2, 3]])

        # Non-symmetric grid
        no_sym_grid = MockGrid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert FitnessComponentLibrary._has_horizontal_symmetry(h_sym_grid) is True
        assert FitnessComponentLibrary._has_vertical_symmetry(v_sym_grid) is True
        assert FitnessComponentLibrary._has_horizontal_symmetry(no_sym_grid) is False
        assert FitnessComponentLibrary._has_vertical_symmetry(no_sym_grid) is False


class TestCoevolutionEngine:
    """Test co-evolution engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create co-evolution engine."""
        return CoevolutionEngine(
            program_pop_size=10,
            fitness_pop_size=5,
            elite_ratio=0.2
        )

    @pytest.fixture
    def mock_population(self):
        """Create mock program population."""
        population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.fitness = i * 0.1
            ind.id = f"ind_{i}"
            population.add_individual(ind)
        return population

    @pytest.fixture
    def mock_task(self):
        """Create mock ARC task."""
        task = MagicMock()

        # Create mock grids
        input_grid = MagicMock(spec=Grid)
        input_grid.shape = (3, 3)
        input_grid.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        output_grid = MagicMock(spec=Grid)
        output_grid.shape = (3, 3)
        output_grid.data = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

        train_pair = MagicMock()
        train_pair.input = input_grid
        train_pair.output = output_grid

        task.train_pairs = [train_pair]
        return task

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.program_pop_size == 10
        assert engine.fitness_pop_size == 5
        assert engine.elite_ratio == 0.2
        assert engine.generation == 0

    def test_fitness_population_initialization(self, engine):
        """Test initializing fitness function population."""
        engine.initialize_populations()

        assert len(engine.fitness_population) == 5

        # Check diversity of fitness functions
        component_sets = []
        for ff in engine.fitness_population:
            comp_names = sorted([c.name for c in ff.components])
            component_sets.append(tuple(comp_names))

        # Should have some diversity
        unique_sets = len(set(component_sets))
        assert unique_sets >= 3  # At least 3 different combinations

        # First two should be standard (exact match, pixel accuracy)
        assert engine.fitness_population[0].components[0].name == "exact_match"
        assert engine.fitness_population[1].components[0].name == "pixel_accuracy"

    def test_coevolve_generation(self, engine, mock_population, mock_task):
        """Test one generation of co-evolution."""
        engine.initialize_populations()

        # Add execution results to some individuals
        for i, ind in enumerate(mock_population.individuals[:5]):
            ind.metadata["execution_result"] = MagicMock(spec=Grid)
            ind.metadata["execution_result"].shape = (3, 3)
            ind.metadata["execution_result"].data = [[i, i, i], [i, i, i], [i, i, i]]

        # Run one generation
        updated_pop, best_fitness_func = engine.coevolve_generation(
            mock_population, mock_task
        )

        assert engine.generation == 1
        assert best_fitness_func is not None
        assert isinstance(best_fitness_func, FitnessFunction)

        # Check that fitness evaluations were performed
        for ind in mock_population.individuals[:5]:
            assert "fitness_evaluations" in ind.metadata

    def test_fitness_function_evolution(self, engine):
        """Test evolution of fitness functions."""
        engine.initialize_populations()
        initial_pop_size = len(engine.fitness_population)

        # Set some meta-fitness scores
        for i, ff in enumerate(engine.fitness_population):
            ff.fitness_score = i * 0.2

        # Evolve fitness population
        engine._evolve_fitness_population()

        # Population size should be maintained
        assert len(engine.fitness_population) == initial_pop_size

        # Elite should be preserved
        elite_count = int(initial_pop_size * engine.elite_ratio)
        # Check that high-fitness individuals are in new population
        high_fitness_ids = [ff.id for ff in sorted(
            engine.fitness_population,
            key=lambda f: f.fitness_score,
            reverse=True
        )[:elite_count]]

        new_ids = [ff.id for ff in engine.fitness_population]
        for elite_id in high_fitness_ids[:elite_count]:
            assert elite_id in new_ids

    def test_mutation_operations(self, engine):
        """Test fitness function mutations."""
        engine.initialize_populations()

        # Test add mutation
        ff = engine.fitness_population[0]
        initial_count = len(ff.components)
        engine._mutate_fitness_function(ff)
        # Components might change
        assert len(ff.components) >= 1  # At least one component

        # Test weight mutation
        ff2 = FitnessFunction(components=[
            FitnessComponent("test", 0.5, lambda p, t: 1.0)
        ])
        initial_weight = ff2.components[0].weight

        # Force weight mutation
        with patch('random.choice', return_value="weight"):
            with patch('random.random', return_value=0.0):
                engine._mutate_fitness_function(ff2)

        # Weight should have changed
        assert ff2.components[0].weight != initial_weight

    def test_crossover_operations(self, engine):
        """Test fitness function crossover."""
        parent1 = FitnessFunction(components=[
            FitnessComponent("comp1", 0.5, lambda p, t: 1.0),
            FitnessComponent("comp2", 0.5, lambda p, t: 0.5)
        ])

        parent2 = FitnessFunction(components=[
            FitnessComponent("comp3", 0.3, lambda p, t: 0.8),
            FitnessComponent("comp4", 0.7, lambda p, t: 0.2)
        ])

        offspring = engine._crossover_fitness_functions(parent1, parent2)

        # Offspring should have components from both parents
        offspring_names = [c.name for c in offspring.components]
        parent_names = ["comp1", "comp2", "comp3", "comp4"]

        # At least one component from parents
        assert any(name in parent_names for name in offspring_names)

        # Should be normalized
        total_weight = sum(c.weight for c in offspring.components)
        assert pytest.approx(total_weight) == 1.0

    def test_meta_fitness_evaluation(self, engine, mock_task):
        """Test meta-fitness calculation for fitness functions."""
        engine.initialize_populations()

        # Create mock programs with known results
        mock_programs = []
        for i in range(5):
            ind = Individual(operations=[])
            ind.id = f"prog_{i}"

            # Create execution result
            result = MagicMock(spec=Grid)
            result.shape = (3, 3)
            if i == 0:
                # Perfect match
                result.data = mock_task.train_pairs[0].output.data
            else:
                # Partial matches
                result.data = [[i, i, i], [i, i, i], [i, i, i]]

            ind.metadata["execution_result"] = result
            ind.metadata["fitness_evaluations"] = {}
            mock_programs.append(ind)

        engine.program_population = mock_programs

        # Evaluate fitness functions
        engine._evaluate_fitness_functions(mock_task)

        # All fitness functions should have meta-fitness scores
        for ff in engine.fitness_population:
            assert hasattr(ff, 'fitness_score')
            assert 0 <= ff.fitness_score <= 1

    def test_coevolution_statistics(self, engine):
        """Test statistics collection."""
        engine.initialize_populations()
        engine.generation = 5

        # Add some mock data
        for ind in range(3):
            i = Individual(operations=[])
            i.fitness = 0.5 + ind * 0.1
            engine.program_population.append(i)

        stats = engine.get_coevolution_stats()

        assert stats['generation'] == 5
        assert 'best_program_fitness' in stats
        assert 'best_fitness_function_score' in stats
        assert 'fitness_diversity' in stats
        assert 'component_usage' in stats

        # Check component usage tracking
        assert isinstance(stats['component_usage'], dict)
