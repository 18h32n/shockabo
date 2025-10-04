"""
Unit tests for genetic operators (crossover and mutation).

Tests all crossover and mutation operators to ensure they maintain
program validity and behave according to specifications.
"""

from unittest.mock import patch

import pytest

from src.adapters.strategies.evolution_engine import Individual
from src.adapters.strategies.genetic_operators import (
    AdaptiveMutation,
    InsertDeleteMutation,
    OperationReplacementMutation,
    ParameterMutation,
    ReorderMutation,
    SinglePointCrossover,
    SubtreeCrossover,
    UniformCrossover,
)
from src.domain.dsl.base import Operation


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, name: str, category: str = "test", **params):
        self._op_name = name
        self._op_category = category
        # Call parent init with params
        super().__init__(**params)

    def execute(self, grid, context=None):
        from src.domain.dsl.base import OperationResult
        return OperationResult(success=True, grid=grid)

    def get_name(self):
        return self._op_name

    @classmethod
    def get_description(cls):
        return "Mock operation for testing"

    @classmethod
    def get_parameter_schema(cls):
        return {
            "value": {"type": "int", "min": 0, "max": 10, "required": True},
            "color": {"type": "color", "min": 0, "max": 9}
        }

    def _validate_parameters(self):
        """Override parameter validation for testing."""
        pass


@pytest.fixture
def mock_operations():
    """Create a list of mock operations."""
    return [
        MockOperation("rotate", "geometric", angle=90),
        MockOperation("flip", "geometric", direction="horizontal"),
        MockOperation("fill", "color", color=3),
        MockOperation("replace", "color", from_color=1, to_color=5),
        MockOperation("find", "pattern", pattern_type="object")
    ]


@pytest.fixture
def parent1(mock_operations):
    """Create first parent individual."""
    return Individual(operations=mock_operations[:3])


@pytest.fixture
def parent2(mock_operations):
    """Create second parent individual."""
    return Individual(operations=mock_operations[2:])


class TestSinglePointCrossover:
    """Test single-point crossover operator."""

    def test_crossover_produces_two_offspring(self, parent1, parent2):
        """Test that crossover produces exactly two offspring."""
        crossover = SinglePointCrossover()
        offspring = crossover.apply(parent1, parent2)

        assert len(offspring) == 2
        assert all(isinstance(o, Individual) for o in offspring)

    def test_crossover_preserves_parent_ids(self, parent1, parent2):
        """Test that offspring track both parent IDs."""
        crossover = SinglePointCrossover()
        offspring = crossover.apply(parent1, parent2)

        for child in offspring:
            assert parent1.id in child.parent_ids
            assert parent2.id in child.parent_ids

    def test_crossover_with_short_sequences(self):
        """Test crossover with sequences too short to cross."""
        parent1 = Individual(operations=[MockOperation("op1", "test")])
        parent2 = Individual(operations=[MockOperation("op2", "test")])

        crossover = SinglePointCrossover()
        offspring = crossover.apply(parent1, parent2)

        # Should return copies of parents
        assert len(offspring) == 2
        assert len(offspring[0].operations) == 1
        assert len(offspring[1].operations) == 1

    def test_crossover_maintains_valid_operations(self, parent1, parent2):
        """Test that offspring contain valid operations."""
        crossover = SinglePointCrossover()
        offspring = crossover.apply(parent1, parent2)

        for child in offspring:
            assert len(child.operations) > 0
            for op in child.operations:
                assert isinstance(op, Operation)
                assert hasattr(op, 'get_name')
                assert hasattr(op, 'execute')


class TestUniformCrossover:
    """Test uniform crossover operator."""

    def test_uniform_crossover_basic(self, parent1, parent2):
        """Test basic uniform crossover functionality."""
        crossover = UniformCrossover(swap_probability=0.5)
        offspring = crossover.apply(parent1, parent2)

        assert len(offspring) == 2
        assert all(isinstance(o, Individual) for o in offspring)

    def test_uniform_crossover_with_different_lengths(self):
        """Test uniform crossover with different length parents."""
        parent1 = Individual(operations=[
            MockOperation(f"op{i}", "test") for i in range(5)
        ])
        parent2 = Individual(operations=[
            MockOperation(f"op{i+5}", "test") for i in range(3)
        ])

        crossover = UniformCrossover(swap_probability=0.5)
        offspring = crossover.apply(parent1, parent2)

        # Offspring should have operations from both parents
        assert len(offspring) == 2
        for child in offspring:
            assert len(child.operations) > 0
            assert len(child.operations) <= max(5, 3)

    def test_swap_probability_extremes(self, parent1, parent2):
        """Test uniform crossover with extreme swap probabilities."""
        # Test with 0 probability (no swapping)
        crossover = UniformCrossover(swap_probability=0.0)
        offspring = crossover.apply(parent1, parent2)

        # First child should be similar to first parent
        assert len(offspring[0].operations) == len(parent1.operations)

        # Test with 1.0 probability (always swap)
        crossover = UniformCrossover(swap_probability=1.0)
        offspring = crossover.apply(parent1, parent2)

        assert len(offspring) == 2


class TestSubtreeCrossover:
    """Test subtree crossover operator."""

    def test_subtree_identification(self, parent1, parent2):
        """Test that subtrees are correctly identified."""
        crossover = SubtreeCrossover()
        subtrees = crossover._identify_subtrees(parent1.operations)

        # Should identify consecutive operations of same category
        assert len(subtrees) > 0
        for start, end in subtrees:
            assert 0 <= start <= end < len(parent1.operations)

    def test_subtree_crossover_basic(self, parent1, parent2):
        """Test basic subtree crossover functionality."""
        crossover = SubtreeCrossover()
        offspring = crossover.apply(parent1, parent2)

        assert len(offspring) == 2
        assert all(isinstance(o, Individual) for o in offspring)

    def test_operation_category_detection(self):
        """Test operation category detection."""
        crossover = SubtreeCrossover()

        geometric_op = MockOperation("rotate", "geometric")
        color_op = MockOperation("fill", "color")
        pattern_op = MockOperation("find", "pattern")

        assert crossover._get_operation_category(geometric_op) == "geometric"
        assert crossover._get_operation_category(color_op) == "color"
        assert crossover._get_operation_category(pattern_op) == "pattern"

    def test_fallback_to_single_point(self):
        """Test fallback when no subtrees found."""
        # Create parents with single operations
        parent1 = Individual(operations=[MockOperation("op1", "test")])
        parent2 = Individual(operations=[MockOperation("op2", "test")])

        crossover = SubtreeCrossover()
        offspring = crossover.apply(parent1, parent2)

        # Should fall back to single point crossover
        assert len(offspring) == 2


class TestOperationReplacementMutation:
    """Test operation replacement mutation."""

    def test_mutation_basic(self, parent1):
        """Test basic mutation functionality."""
        mutation = OperationReplacementMutation(mutation_rate=0.5)
        offspring = mutation.apply(parent1)

        assert len(offspring) == 1
        assert isinstance(offspring[0], Individual)
        assert parent1.id in offspring[0].parent_ids

    def test_mutation_rate_respected(self, parent1):
        """Test that mutation rate is approximately respected."""
        # Test with 0 mutation rate
        mutation = OperationReplacementMutation(mutation_rate=0.0)
        offspring = mutation.apply(parent1)

        # Should force at least one mutation
        assert len(offspring[0].operations) == len(parent1.operations)

    def test_operation_category_preservation(self, parent1):
        """Test that replacements are from same category."""
        mutation = OperationReplacementMutation(mutation_rate=1.0)

        # Mock the category detection
        with patch.object(mutation, '_get_operation_category') as mock_category:
            mock_category.return_value = 'geometric'

            offspring = mutation.apply(parent1)

            # Should have called category detection
            assert mock_category.called


class TestParameterMutation:
    """Test parameter mutation operator."""

    def test_parameter_mutation_basic(self, parent1):
        """Test basic parameter mutation."""
        mutation = ParameterMutation(mutation_rate=0.5)
        offspring = mutation.apply(parent1)

        assert len(offspring) == 1
        assert isinstance(offspring[0], Individual)
        assert len(offspring[0].operations) == len(parent1.operations)

    def test_parameter_bounds_respected(self):
        """Test that mutated parameters stay within bounds."""
        op = MockOperation("test", "test", value=5, color=3)
        individual = Individual(operations=[op])

        mutation = ParameterMutation(mutation_rate=1.0, mutation_strength=0.5)

        # Run multiple mutations to test bounds
        for _ in range(10):
            offspring = mutation.apply(individual)
            mutated_op = offspring[0].operations[0]

            # Check that parameters are within schema bounds
            if hasattr(mutated_op, '_params'):
                if 'value' in mutated_op._params:
                    assert 0 <= mutated_op._params['value'] <= 10
                if 'color' in mutated_op._params:
                    assert 0 <= mutated_op._params['color'] <= 9


class TestInsertDeleteMutation:
    """Test insert/delete mutation operator."""

    def test_insertion(self):
        """Test operation insertion."""
        ops = [MockOperation(f"op{i}", "test") for i in range(3)]
        individual = Individual(operations=ops)

        mutation = InsertDeleteMutation(
            insertion_rate=1.0,
            deletion_rate=0.0,
            max_length=10
        )
        offspring = mutation.apply(individual)

        # Should have inserted an operation
        assert len(offspring[0].operations) == 4

    def test_deletion(self):
        """Test operation deletion."""
        ops = [MockOperation(f"op{i}", "test") for i in range(5)]
        individual = Individual(operations=ops)

        mutation = InsertDeleteMutation(
            insertion_rate=0.0,
            deletion_rate=1.0,
            min_length=2
        )
        offspring = mutation.apply(individual)

        # Should have deleted an operation
        assert len(offspring[0].operations) == 4

    def test_length_bounds(self):
        """Test that length bounds are respected."""
        # Test minimum length
        ops = [MockOperation("op1", "test"), MockOperation("op2", "test")]
        individual = Individual(operations=ops)

        mutation = InsertDeleteMutation(
            insertion_rate=0.0,
            deletion_rate=1.0,
            min_length=2
        )
        offspring = mutation.apply(individual)

        # Should not delete below minimum
        assert len(offspring[0].operations) >= 2

        # Test maximum length
        ops = [MockOperation(f"op{i}", "test") for i in range(20)]
        individual = Individual(operations=ops)

        mutation = InsertDeleteMutation(
            insertion_rate=1.0,
            deletion_rate=0.0,
            max_length=20
        )
        offspring = mutation.apply(individual)

        # Should not insert above maximum
        assert len(offspring[0].operations) <= 20


class TestReorderMutation:
    """Test reorder mutation operator."""

    def test_reorder_preserves_operations(self, parent1):
        """Test that reordering preserves all operations."""
        mutation = ReorderMutation(shuffle_segments=False)
        offspring = mutation.apply(parent1)

        assert len(offspring) == 1
        assert len(offspring[0].operations) == len(parent1.operations)

        # Check that all operations are preserved (though order may differ)
        parent_names = [op.get_name() for op in parent1.operations]
        offspring_names = [op.get_name() for op in offspring[0].operations]
        assert sorted(parent_names) == sorted(offspring_names)

    def test_segment_shuffling(self):
        """Test segment-based shuffling."""
        ops = [MockOperation(f"op{i}", f"cat{i//2}") for i in range(6)]
        individual = Individual(operations=ops)

        mutation = ReorderMutation(shuffle_segments=True, segment_size=2)
        offspring = mutation.apply(individual)

        assert len(offspring[0].operations) == 6


class TestAdaptiveMutation:
    """Test adaptive mutation operator."""

    def test_adaptive_mutation_basic(self, parent1):
        """Test basic adaptive mutation."""
        mutation = AdaptiveMutation(base_rate=0.1, max_rate=0.3)
        offspring = mutation.apply(parent1)

        assert len(offspring) == 1
        assert isinstance(offspring[0], Individual)

    def test_rate_adaptation(self):
        """Test mutation rate adaptation."""
        mutation = AdaptiveMutation(base_rate=0.1, max_rate=0.3)

        # Test rate increase on stagnation
        initial_rate = mutation.current_rate
        mutation.adapt_rate(fitness_improvement=0.0)  # No improvement
        assert mutation.current_rate > initial_rate

        # Test rate decrease on improvement
        mutation.adapt_rate(fitness_improvement=0.1)  # Good improvement
        assert mutation.current_rate < mutation.max_rate

        # Test bounds
        for _ in range(10):
            mutation.adapt_rate(fitness_improvement=0.0)
        assert mutation.current_rate <= mutation.max_rate

        for _ in range(10):
            mutation.adapt_rate(fitness_improvement=1.0)
        assert mutation.current_rate >= mutation.base_rate

    def test_sub_mutation_selection(self, parent1):
        """Test that adaptive mutation uses different sub-mutations."""
        mutation = AdaptiveMutation()

        # Test that it has multiple sub-mutations initialized
        assert len(mutation.mutations) > 0

        # Run the mutation and verify it works
        offspring = mutation.apply(parent1)
        assert len(offspring) == 1
        assert isinstance(offspring[0], Individual)
        assert parent1.id in offspring[0].parent_ids


# Integration tests for crossover and mutation combinations
class TestGeneticOperatorIntegration:
    """Test genetic operators working together."""

    def test_crossover_then_mutation(self, parent1, parent2):
        """Test applying crossover followed by mutation."""
        crossover = SinglePointCrossover()
        mutation = OperationReplacementMutation(mutation_rate=0.5)

        # Apply crossover
        offspring = crossover.apply(parent1, parent2)

        # Apply mutation to offspring
        mutated_offspring = []
        for child in offspring:
            mutated = mutation.apply(child)
            mutated_offspring.extend(mutated)

        assert len(mutated_offspring) == 2
        for child in mutated_offspring:
            assert isinstance(child, Individual)
            assert len(child.operations) > 0

    def test_multiple_mutations(self, parent1):
        """Test applying multiple mutations in sequence."""
        mutations = [
            ParameterMutation(mutation_rate=0.5),
            InsertDeleteMutation(insertion_rate=0.3, deletion_rate=0.1),
            ReorderMutation()
        ]

        current = parent1
        for mutation in mutations:
            offspring = mutation.apply(current)
            current = offspring[0]

        # Final individual should be valid
        assert isinstance(current, Individual)
        assert len(current.operations) > 0
        # After multiple mutations, parent_ids contains immediate parent, not original
        assert len(current.parent_ids) > 0

    def test_population_genetic_operations(self):
        """Test genetic operations on a population."""
        # Create a small population
        population = [
            Individual(operations=[
                MockOperation(f"op{j}", f"cat{j%3}") for j in range(3)
            ])
            for i in range(10)
        ]

        crossover = UniformCrossover(swap_probability=0.5)
        mutation = AdaptiveMutation(base_rate=0.1)

        # Apply genetic operations
        new_population = []

        # Crossover pairs
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                offspring = crossover.apply(population[i], population[i+1])
                new_population.extend(offspring)

        # Mutation
        mutated_population = []
        for individual in new_population:
            mutated = mutation.apply(individual)
            mutated_population.extend(mutated)

        # Verify population properties
        assert len(mutated_population) > 0
        for individual in mutated_population:
            assert isinstance(individual, Individual)
            assert len(individual.operations) > 0
            assert len(individual.parent_ids) > 0
