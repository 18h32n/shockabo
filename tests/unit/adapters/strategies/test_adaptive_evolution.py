"""
Unit tests for enhanced adaptive mutation implementation.

Tests Task 7.2: Adaptive mutation rates based on fitness progress.
"""

from unittest.mock import Mock

import pytest

from src.adapters.strategies.adaptive_evolution import (
    AdaptiveMutationController,
    EnhancedAdaptiveMutation,
    FitnessHistory,
    MutationRateSchedule,
    create_adaptive_mutation_schedule,
)
from src.adapters.strategies.evolution_engine import Individual, Population
from src.adapters.strategies.genetic_operators import MutationOperator


class TestFitnessHistory:
    """Test FitnessHistory tracking functionality."""

    def test_add_and_trend_calculation(self):
        """Test adding fitness values and trend calculation."""
        history = FitnessHistory(max_history=10)

        # Add increasing fitness values
        for i in range(5):
            history.add(0.5 + i * 0.1)

        # Check positive trend
        trend = history.get_trend(window=5)
        assert trend > 0, "Should detect positive trend"

        # Add decreasing values
        history = FitnessHistory()
        for i in range(5):
            history.add(0.9 - i * 0.1)

        # Check negative trend
        trend = history.get_trend(window=5)
        assert trend < 0, "Should detect negative trend"

    def test_variance_calculation(self):
        """Test fitness variance calculation."""
        history = FitnessHistory()

        # No variance with single value
        history.add(0.5)
        assert history.get_variance() == 0.0

        # Add varied values
        for val in [0.1, 0.5, 0.9, 0.3, 0.7]:
            history.add(val)

        variance = history.get_variance()
        assert variance > 0, "Should have positive variance"
        assert variance == pytest.approx(0.0667, rel=1e-3)

    def test_stagnation_detection(self):
        """Test stagnation detection."""
        history = FitnessHistory()

        # Not stagnant with insufficient history
        assert not history.is_stagnant(window=5)

        # Add stagnant values
        for _ in range(10):
            history.add(0.5)

        assert history.is_stagnant(threshold=0.001, window=5)

        # Add improving values
        history.add(0.6)
        history.add(0.7)

        assert not history.is_stagnant(threshold=0.1, window=5)


class TestMutationRateSchedule:
    """Test mutation rate schedule configuration."""

    def test_default_schedule(self):
        """Test default schedule parameters."""
        schedule = MutationRateSchedule()

        assert schedule.base_rate == 0.1
        assert schedule.min_rate == 0.01
        assert schedule.max_rate == 0.5
        assert schedule.strategy == "fitness_based"
        assert schedule.adaptation_speed == 0.1

    def test_custom_schedule(self):
        """Test custom schedule parameters."""
        schedule = MutationRateSchedule(
            base_rate=0.2,
            min_rate=0.05,
            max_rate=0.6,
            strategy="time_decay",
            decay_factor=0.95
        )

        assert schedule.base_rate == 0.2
        assert schedule.min_rate == 0.05
        assert schedule.max_rate == 0.6
        assert schedule.strategy == "time_decay"
        assert schedule.decay_factor == 0.95


class TestAdaptiveMutationController:
    """Test adaptive mutation controller functionality."""

    @pytest.fixture
    def controller(self):
        """Create a controller with default settings."""
        return AdaptiveMutationController()

    @pytest.fixture
    def mock_population(self):
        """Create a mock population."""
        population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.fitness = 0.5 + i * 0.05
            ind.id = f"ind_{i}"
            population.add_individual(ind)
        return population

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.generation_count == 0
        assert controller.schedule is not None
        assert len(controller.individual_histories) == 0
        assert controller.fitness_landscape_ruggedness == 0.0

    def test_update_population_stats(self, controller, mock_population):
        """Test updating population statistics."""
        controller.update_population_stats(mock_population)

        assert controller.generation_count == 1
        assert len(controller.population_history.history) == 1
        assert len(controller.diversity_history) == 1

        # Check average fitness recorded
        avg_fitness = controller.population_history.history[0]
        assert avg_fitness == pytest.approx(0.725, rel=1e-3)

    def test_fitness_based_mutation_rate(self, controller, mock_population):
        """Test fitness-based mutation rate calculation."""
        # Set up individual with stagnant history
        individual = mock_population.individuals[0]

        # Add stagnant fitness history
        controller.individual_histories[individual.id] = FitnessHistory()
        for _ in range(10):
            controller.individual_histories[individual.id].add(0.5)

        # Stagnant individual should get higher rate
        rate = controller.get_mutation_rate(individual)
        assert rate > controller.schedule.base_rate

        # High fitness individual
        high_fitness_ind = Individual(operations=[])
        high_fitness_ind.fitness = 0.9
        high_fitness_ind.id = "high_fitness"

        rate = controller.get_mutation_rate(high_fitness_ind)
        assert rate < controller.schedule.base_rate

    def test_time_decay_mutation_rate(self, controller):
        """Test time-decay mutation rate."""
        controller.schedule.strategy = "time_decay"
        controller.schedule.decay_factor = 0.9

        individual = Individual(operations=[])
        individual.id = "test"

        # Initial rate
        rate1 = controller.get_mutation_rate(individual)

        # Advance generations
        controller.generation_count = 10
        rate2 = controller.get_mutation_rate(individual)

        assert rate2 < rate1
        assert rate2 == pytest.approx(
            controller.schedule.base_rate * (0.9 ** 10), rel=1e-3
        )

    def test_cyclic_mutation_rate(self, controller):
        """Test cyclic mutation rate."""
        controller.schedule.strategy = "cyclic"
        controller.schedule.cycle_period = 10
        controller.schedule.min_rate = 0.1
        controller.schedule.max_rate = 0.3

        individual = Individual(operations=[])
        individual.id = "test"

        rates = []
        for gen in range(20):
            controller.generation_count = gen
            rate = controller.get_mutation_rate(individual)
            rates.append(rate)

        # Check cyclic pattern - at least should return valid rates
        assert all(isinstance(rate, float) for rate in rates)
        assert all(rate >= controller.schedule.min_rate for rate in rates)
        assert all(rate <= controller.schedule.max_rate for rate in rates)
        assert min(rates) >= controller.schedule.min_rate
        assert max(rates) <= controller.schedule.max_rate

    def test_landscape_analysis(self, controller, mock_population):
        """Test fitness landscape analysis."""
        controller._analyze_fitness_landscape(mock_population)

        # Should detect some ruggedness
        assert controller.fitness_landscape_ruggedness > 0

        # Create smooth landscape
        smooth_population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.fitness = i * 0.1  # Linear progression
            smooth_population.add_individual(ind)

        controller._analyze_fitness_landscape(smooth_population)
        # Smooth landscape should have low ruggedness
        assert controller.fitness_landscape_ruggedness < 0.5

    def test_operator_performance_tracking(self, controller):
        """Test tracking operator performance."""
        # Track some operations
        controller.adapt_operator_rates("mutation1", True)
        controller.adapt_operator_rates("mutation1", True)
        controller.adapt_operator_rates("mutation1", False)

        controller.adapt_operator_rates("mutation2", False)
        controller.adapt_operator_rates("mutation2", False)

        # Need enough samples
        for _ in range(10):
            controller.adapt_operator_rates("mutation1", True)
            controller.adapt_operator_rates("mutation2", False)

        # Check success rates
        assert "mutation1" in controller.mutation_success_rates
        assert controller.mutation_success_rates["mutation1"] > 0.5
        assert controller.mutation_success_rates["mutation2"] < 0.5

        # Get operator weights
        weights = controller.get_operator_weights()
        assert weights["mutation1"] > weights["mutation2"]


class TestEnhancedAdaptiveMutation:
    """Test enhanced adaptive mutation operator."""

    @pytest.fixture
    def mock_operators(self):
        """Create mock mutation operators."""
        ops = []
        for i in range(3):
            op = Mock(spec=MutationOperator)
            op.get_name.return_value = f"mutation_{i}"
            op.apply.return_value = [Individual(operations=[])]
            ops.append(op)
        return ops

    @pytest.fixture
    def mutation(self, mock_operators):
        """Create enhanced adaptive mutation."""
        controller = AdaptiveMutationController()
        return EnhancedAdaptiveMutation(
            controller=controller,
            base_operators=mock_operators
        )

    def test_initialization(self, mutation, mock_operators):
        """Test mutation initialization."""
        assert mutation.controller is not None
        assert len(mutation.base_operators) == 3
        assert mutation.last_fitness_by_individual == {}

    def test_apply_mutation(self, mutation):
        """Test applying mutation."""
        individual = Individual(operations=[])
        individual.fitness = 0.5
        individual.id = "test_ind"

        # Mock controller to return high mutation rate
        mutation.controller.get_mutation_rate = Mock(return_value=0.9)

        result = mutation.apply(individual)

        assert len(result) == 1
        assert result[0] != individual  # Should be mutated
        assert "mutation_rate" in result[0].metadata
        assert "mutation_operator" in result[0].metadata

    def test_no_mutation_low_rate(self, mutation):
        """Test no mutation with low rate."""
        individual = Individual(operations=[])
        individual.id = "test_ind"

        # Mock controller to return very low mutation rate
        mutation.controller.get_mutation_rate = Mock(return_value=0.0)

        result = mutation.apply(individual)

        # Should return copy without mutation
        assert len(result) == 1
        # Check no mutation metadata
        assert "mutation_rate" not in result[0].metadata

    def test_operator_selection_by_performance(self, mutation, mock_operators):
        """Test operator selection based on performance."""
        # Set up operator performance
        mutation.controller.mutation_success_rates = {
            "mutation_0": 0.8,
            "mutation_1": 0.2,
            "mutation_2": 0.5
        }

        # Track selections over multiple calls
        selected_counts = {f"mutation_{i}": 0 for i in range(3)}

        for _ in range(100):
            op = mutation._select_operator()
            selected_counts[op.get_name()] += 1

        # mutation_0 should be selected most often
        assert selected_counts["mutation_0"] > selected_counts["mutation_1"]
        assert selected_counts["mutation_0"] > selected_counts["mutation_2"]

    def test_success_tracking(self, mutation):
        """Test mutation success tracking."""
        # Create parent and child
        parent = Individual(operations=[])
        parent.fitness = 0.5
        parent.id = "parent"

        child = Individual(operations=[])
        child.fitness = 0.7  # Improved
        child.id = "child"
        child.metadata["mutation_operator"] = "test_mutation"

        # Track parent fitness
        mutation.last_fitness_by_individual["child"] = 0.5

        # Mock controller tracking
        mutation.controller.adapt_operator_rates = Mock()

        # Update success
        mutation.update_success(child)

        # Should track success
        mutation.controller.adapt_operator_rates.assert_called_with("test_mutation", True)

        # Parent tracking should be cleaned up
        assert "child" not in mutation.last_fitness_by_individual


class TestFactoryFunctions:
    """Test factory and helper functions."""

    def test_create_adaptive_mutation_schedule(self):
        """Test creating mutation schedules."""
        # Fitness-based schedule
        schedule = create_adaptive_mutation_schedule(
            strategy="fitness_based",
            base_rate=0.15,
            max_rate=0.6
        )

        assert schedule.strategy == "fitness_based"
        assert schedule.base_rate == 0.15
        assert schedule.max_rate == 0.6

        # Time decay schedule
        schedule = create_adaptive_mutation_schedule(
            strategy="time_decay",
            decay_factor=0.95
        )

        assert schedule.strategy == "time_decay"
        assert schedule.decay_factor == 0.95

        # Cyclic schedule
        schedule = create_adaptive_mutation_schedule(
            strategy="cyclic",
            cycle_period=30
        )

        assert schedule.strategy == "cyclic"
        assert schedule.cycle_period == 30
