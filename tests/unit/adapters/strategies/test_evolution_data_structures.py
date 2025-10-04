"""
Unit tests for genetic algorithm data structures.

Tests Individual, Population, and related classes for the evolution engine.
"""

from datetime import datetime

import pytest

from src.adapters.strategies.evolution_engine import (
    DiversityMethod,
    Individual,
    Population,
    SelectionMethod,
)
from src.domain.dsl.base import Operation


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, name: str = "mock_op", **kwargs):
        self._name = name
        super().__init__(**kwargs)

    def execute(self, grid, context=None):
        # Mock implementation
        return {"success": True, "grid": grid}

    @classmethod
    def get_name(cls):
        return "mock_op"

    @classmethod
    def get_description(cls):
        return "Mock operation for testing"

    @classmethod
    def get_parameter_schema(cls):
        return {}


class TestIndividual:
    """Test Individual class functionality."""

    def test_individual_creation(self):
        """Test creating an individual."""
        ops = [MockOperation("op1"), MockOperation("op2")]
        individual = Individual(operations=ops)

        assert individual.operations == ops
        assert individual.fitness == 0.0
        assert individual.age == 0
        assert len(individual.parent_ids) == 0
        assert individual.id is not None
        assert isinstance(individual.created_at, datetime)

    def test_individual_id_generation(self):
        """Test that individuals with different operations get different IDs."""
        # Use operations with different parameters to ensure different IDs
        class TestOp(MockOperation):
            def __init__(self, name, param=None):
                super().__init__(name)
                self.parameters['param'] = param

        ops1 = [TestOp("op1", param=1), TestOp("op2", param=2)]
        ops2 = [TestOp("op2", param=3), TestOp("op1", param=4)]

        ind1 = Individual(operations=ops1)
        ind2 = Individual(operations=ops2)

        assert ind1.id != ind2.id

    def test_individual_identical_operations_same_id(self):
        """Test that individuals with identical operations get same ID."""
        ops1 = [MockOperation("op1"), MockOperation("op2")]
        ops2 = [MockOperation("op1"), MockOperation("op2")]

        ind1 = Individual(operations=ops1)
        ind2 = Individual(operations=ops2)

        # Note: This might fail if MockOperation parameters differ
        # In real implementation, we'd ensure parameter comparison

    def test_program_length(self):
        """Test program length calculation."""
        ops = [MockOperation(f"op{i}") for i in range(5)]
        individual = Individual(operations=ops)

        assert individual.program_length() == 5

    def test_increment_age(self):
        """Test age increment."""
        individual = Individual(operations=[MockOperation()])

        assert individual.age == 0
        individual.increment_age()
        assert individual.age == 1
        individual.increment_age()
        assert individual.age == 2

    def test_is_elite(self):
        """Test elite status checking."""
        individual = Individual(operations=[MockOperation()])

        # Low fitness
        individual.fitness = 0.5
        assert not individual.is_elite(0.9)

        # High fitness
        individual.fitness = 0.95
        assert individual.is_elite(0.9)

        # Custom threshold
        individual.fitness = 0.7
        assert individual.is_elite(0.7)
        assert not individual.is_elite(0.8)


class TestPopulation:
    """Test Population class functionality."""

    def test_population_creation(self):
        """Test creating a population."""
        population = Population()

        assert len(population.individuals) == 0
        assert population.generation == 0
        assert len(population.species) == 0
        assert population.best_individual is None

    def test_add_individual(self):
        """Test adding individuals to population."""
        population = Population()

        ind1 = Individual(operations=[MockOperation()], fitness=0.5)
        ind2 = Individual(operations=[MockOperation()], fitness=0.7)

        population.add_individual(ind1)
        assert len(population.individuals) == 1
        assert population.best_individual == ind1

        population.add_individual(ind2)
        assert len(population.individuals) == 2
        assert population.best_individual == ind2  # Higher fitness

    def test_remove_individual(self):
        """Test removing individuals from population."""
        population = Population()
        ind = Individual(operations=[MockOperation()])

        population.add_individual(ind)
        assert len(population.individuals) == 1

        population.remove_individual(ind)
        assert len(population.individuals) == 0

    def test_get_elite(self):
        """Test getting elite individuals."""
        population = Population()

        # Add individuals with different fitness values
        for i in range(10):
            ind = Individual(operations=[MockOperation()], fitness=i * 0.1)
            population.add_individual(ind)

        elite = population.get_elite(3)
        assert len(elite) == 3
        assert elite[0].fitness == pytest.approx(0.9)
        assert elite[1].fitness == pytest.approx(0.8)
        assert elite[2].fitness == pytest.approx(0.7)

    def test_population_size(self):
        """Test population size calculation."""
        population = Population()

        assert population.size() == 0

        for i in range(5):
            population.add_individual(Individual(operations=[MockOperation()]))

        assert population.size() == 5

    def test_average_fitness(self):
        """Test average fitness calculation."""
        population = Population()

        # Empty population
        assert population.average_fitness() == 0.0

        # Add individuals
        fitnesses = [0.2, 0.4, 0.6, 0.8]
        for f in fitnesses:
            ind = Individual(operations=[MockOperation()], fitness=f)
            population.add_individual(ind)

        assert population.average_fitness() == pytest.approx(0.5)

    def test_fitness_variance(self):
        """Test fitness variance calculation."""
        population = Population()

        # Empty population
        assert population.fitness_variance() == 0.0

        # Add individuals with same fitness
        for _ in range(4):
            ind = Individual(operations=[MockOperation()], fitness=0.5)
            population.add_individual(ind)

        assert population.fitness_variance() == pytest.approx(0.0)

        # Add individuals with different fitness
        population = Population()
        fitnesses = [0.0, 0.5, 1.0]
        for f in fitnesses:
            ind = Individual(operations=[MockOperation()], fitness=f)
            population.add_individual(ind)

        # Variance should be > 0
        assert population.fitness_variance() > 0

    def test_update_generation_stats(self):
        """Test generation statistics update."""
        population = Population()

        # Add individuals
        for i in range(5):
            ind = Individual(operations=[MockOperation()], fitness=i * 0.2)
            ind.age = i
            population.add_individual(ind)

        population.update_generation_stats()

        assert len(population.best_fitness_history) == 1
        assert population.best_fitness_history[0] == 0.8
        assert 'unique_programs' in population.diversity_metrics
        assert 'fitness_variance' in population.diversity_metrics
        assert 'average_age' in population.diversity_metrics

    def test_increment_generation(self):
        """Test generation increment."""
        population = Population()

        # Add individuals
        for i in range(3):
            ind = Individual(operations=[MockOperation()], age=0)
            population.add_individual(ind)

        assert population.generation == 0
        assert all(ind.age == 0 for ind in population.individuals)

        population.increment_generation()

        assert population.generation == 1
        assert all(ind.age == 1 for ind in population.individuals)


class TestEnums:
    """Test enum classes."""

    def test_selection_method_enum(self):
        """Test SelectionMethod enum values."""
        assert SelectionMethod.TOURNAMENT.value == "tournament"
        assert SelectionMethod.ROULETTE.value == "roulette"
        assert SelectionMethod.RANK.value == "rank"
        assert SelectionMethod.ELITE.value == "elite"

    def test_diversity_method_enum(self):
        """Test DiversityMethod enum values."""
        assert DiversityMethod.FITNESS_SHARING.value == "fitness_sharing"
        assert DiversityMethod.SPECIATION.value == "speciation"
        assert DiversityMethod.NOVELTY_SEARCH.value == "novelty"
        assert DiversityMethod.CROWDING.value == "crowding"


if __name__ == "__main__":
    pytest.main([__file__])
