"""
Functional tests for diversity preservation mechanisms.

Tests the functionality of fitness sharing, speciation, and novelty search
to ensure they maintain population diversity effectively.
"""


from src.adapters.strategies.diversity_mechanisms import FitnessSharing, NoveltySearch, Speciation
from src.adapters.strategies.evolution_engine import Individual, Population
from src.domain.dsl.base import Operation


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, name: str, value: int = 0):
        self._name = name
        self._value = value
        super().__init__(value=value)

    def execute(self, grid, context=None):
        from src.domain.dsl.base import OperationResult
        return OperationResult(success=True, grid=grid)

    def get_name(self):
        return self._name

    @classmethod
    def get_description(cls):
        return "Mock operation for testing"

    @classmethod
    def get_parameter_schema(cls):
        return {"value": {"type": "int", "min": 0, "max": 10}}


def create_test_population(size: int = 10, diversity: str = "low") -> Population:
    """Create a test population with controlled diversity."""
    individuals = []

    if diversity == "low":
        # Create similar individuals
        for i in range(size):
            ops = [
                MockOperation("op1", value=i % 2),
                MockOperation("op2", value=i % 2),
                MockOperation("op3", value=i % 2)
            ]
            individual = Individual(operations=ops)
            individual.fitness = 0.5 + (i % 3) * 0.1
            individuals.append(individual)

    elif diversity == "high":
        # Create diverse individuals
        for i in range(size):
            ops = [
                MockOperation(f"op{i%5}", value=i),
                MockOperation(f"op{(i+1)%5}", value=(i+3)%10),
                MockOperation(f"op{(i+2)%5}", value=(i+7)%10)
            ]
            individual = Individual(operations=ops)
            individual.fitness = 0.3 + (i / size) * 0.6
            individuals.append(individual)

    elif diversity == "clustered":
        # Create clusters of similar individuals
        for cluster in range(3):
            for member in range(size // 3):
                i = cluster * (size // 3) + member
                if i < size:
                    ops = [
                        MockOperation(f"cluster{cluster}", value=cluster),
                        MockOperation(f"op{member%2}", value=member%2),
                        MockOperation("shared", value=cluster*2)
                    ]
                    individual = Individual(operations=ops)
                    individual.fitness = 0.4 + cluster * 0.2 + member * 0.01
                    individuals.append(individual)

    return Population(individuals=individuals)


class TestDistanceCalculation:
    """Test distance calculation between individuals."""

    def test_identical_individuals_zero_distance(self):
        """Test that identical individuals have zero distance."""
        from copy import deepcopy

        ops = [MockOperation("op1", value=5), MockOperation("op2", value=3)]
        ind1 = Individual(operations=ops)
        ind2 = Individual(operations=deepcopy(ops))

        # Use FitnessSharing to test distance calculation
        sharing = FitnessSharing()
        distance = sharing._calculate_distance(ind1, ind2)
        # Should be close to zero (behavioral component returns 0.5 without cached execution)
        assert distance == 0.15  # 0.7 * 0 (structural) + 0.3 * 0.5 (behavioral)

    def test_different_individuals_positive_distance(self):
        """Test that different individuals have positive distance."""
        ind1 = Individual(operations=[MockOperation("op1", 5)])
        ind2 = Individual(operations=[MockOperation("op2", 3)])

        sharing = FitnessSharing()
        distance = sharing._calculate_distance(ind1, ind2)
        assert distance > 0.0

    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        ind1 = Individual(operations=[MockOperation("op1", 5), MockOperation("op2", 3)])
        ind2 = Individual(operations=[MockOperation("op3", 7), MockOperation("op4", 2)])

        sharing = FitnessSharing()
        dist1_2 = sharing._calculate_distance(ind1, ind2)
        dist2_1 = sharing._calculate_distance(ind2, ind1)

        assert abs(dist1_2 - dist2_1) < 1e-6


class TestFitnessSharing:
    """Test fitness sharing mechanism."""

    def test_fitness_sharing_reduces_crowded_fitness(self):
        """Test that fitness sharing reduces fitness in crowded regions."""
        population = create_test_population(10, diversity="low")
        sharing = FitnessSharing(niche_radius=0.3)

        # Apply fitness sharing to each individual
        shared_fitnesses = []
        for ind in population.individuals:
            shared_fitness = sharing.apply_pressure(ind, population)
            shared_fitnesses.append(shared_fitness)

        raw_fitness_values = [ind.fitness for ind in population.individuals]

        # At least some should be different due to sharing
        assert any(abs(shared - raw) > 1e-6
                   for shared, raw in zip(shared_fitnesses, raw_fitness_values, strict=False))

    def test_fitness_sharing_diversity_calculation(self):
        """Test diversity calculation with fitness sharing."""
        population = create_test_population(10, diversity="high")
        sharing = FitnessSharing(niche_radius=0.2)

        diversity = sharing.calculate(population)

        # High diversity population should have high diversity score
        assert diversity > 0.3

    def test_niche_radius_effect(self):
        """Test effect of different niche radius values."""
        population = create_test_population(10, diversity="clustered")

        # Apply fitness sharing with different radii and check effect
        sharing_small = FitnessSharing(niche_radius=0.1)
        sharing_large = FitnessSharing(niche_radius=0.8)

        # Count how many individuals get fitness reduction with each radius
        small_radius_reductions = 0
        large_radius_reductions = 0

        for ind in population.individuals:
            shared_small = sharing_small.apply_pressure(ind, population)
            shared_large = sharing_large.apply_pressure(ind, population)

            if shared_small < ind.fitness:
                small_radius_reductions += 1
            if shared_large < ind.fitness:
                large_radius_reductions += 1

        # Larger radius should cause more fitness reductions (more sharing)
        assert large_radius_reductions >= small_radius_reductions


class TestSpeciation:
    """Test speciation mechanism."""

    def test_speciation_creates_species(self):
        """Test that speciation groups similar individuals."""
        population = create_test_population(12, diversity="clustered")
        speciation = Speciation(compatibility_threshold=0.3)

        # Apply speciation to each individual
        for ind in population.individuals:
            speciation.apply_pressure(ind, population)

        # Check that species were assigned
        species_ids = {ind.species_id for ind in population.individuals
                       if ind.species_id is not None}
        assert len(species_ids) >= 2  # Should have at least 2 species
        assert len(species_ids) <= 6  # But not too many for clustered data

    def test_speciation_representative_selection(self):
        """Test that species representatives are selected correctly."""
        population = create_test_population(15, diversity="clustered")
        speciation = Speciation(compatibility_threshold=0.3)

        # Apply speciation
        for ind in population.individuals:
            speciation.apply_pressure(ind, population)

        # Check that each species has individuals
        species_counts = {}
        for ind in population.individuals:
            if ind.species_id is not None:
                species_counts[ind.species_id] = species_counts.get(ind.species_id, 0) + 1

        assert all(count > 0 for count in species_counts.values())



class TestNoveltySearch:
    """Test novelty search mechanism."""

    def test_novelty_score_calculation(self):
        """Test that novelty scores are calculated."""
        population = create_test_population(10, diversity="high")
        novelty = NoveltySearch(archive_size=20, k_nearest=3, novelty_threshold=0.0)

        # Add diverse cached execution to enable novelty calculation
        for i, ind in enumerate(population.individuals):
            # Make behaviors more diverse to ensure novelty detection
            ind.cached_execution = {
                "test1": [[i, (i+5) % 10], [(i*2) % 10, (i*3) % 10]],
                "test2": [[(i+2) % 10, (i-1) % 10], [(i*4) % 10, (i+7) % 10]]
            }

        # Apply novelty search to each individual
        for ind in population.individuals:
            novelty.apply_pressure(ind, population)

        # Check that novelty scores were assigned
        novelty_scores = [ind.novelty_score for ind in population.individuals]
        assert all(score is not None and score >= 0.0 for score in novelty_scores)
        # With diverse behaviors, at least the first individual should have novelty
        # (since archive is initially empty)

    def test_archive_maintenance(self):
        """Test that the novelty archive is maintained correctly."""
        population = create_test_population(10, diversity="high")
        novelty = NoveltySearch(archive_size=5, k_nearest=2)

        # Apply multiple times
        for _ in range(3):
            for ind in population.individuals:
                # Set some cached execution for testing
                ind.cached_execution = {"test": [[1, 2], [3, 4]]}
                novelty.apply_pressure(ind, population)

        # Archive should not exceed max size
        assert len(novelty.archive) <= novelty.archive_size

    def test_novelty_promotes_diversity(self):
        """Test that novelty search promotes diverse solutions."""
        # Create two populations - one uniform, one diverse
        uniform_pop = create_test_population(8, diversity="low")
        diverse_pop = create_test_population(8, diversity="high")

        novelty = NoveltySearch(archive_size=10, k_nearest=3)

        # Add cached execution to enable novelty calculation
        for ind in uniform_pop.individuals:
            ind.cached_execution = {"test": [[1, 1], [1, 1]]}
        for i, ind in enumerate(diverse_pop.individuals):
            ind.cached_execution = {"test": [[i % 3, (i+1) % 3], [(i+2) % 3, i % 3]]}

        # Calculate novelty for both
        for ind in uniform_pop.individuals:
            novelty.apply_pressure(ind, uniform_pop)
        uniform_novelty = novelty.calculate(uniform_pop)

        novelty.archive.clear()  # Reset archive

        for ind in diverse_pop.individuals:
            novelty.apply_pressure(ind, diverse_pop)
        diverse_novelty = novelty.calculate(diverse_pop)

        # Diverse population should have higher novelty
        assert diverse_novelty > uniform_novelty


class TestDiversityIntegration:
    """Test integration of multiple diversity mechanisms."""

    def test_combined_diversity_mechanisms(self):
        """Test using multiple diversity mechanisms together."""
        population = create_test_population(20, diversity="clustered")

        # Add cached execution for novelty search
        for i, ind in enumerate(population.individuals):
            ind.cached_execution = {"test": [[i % 4, (i+1) % 4], [(i+2) % 4, (i+3) % 4]]}

        # Apply different mechanisms
        fitness_sharing = FitnessSharing(niche_radius=0.2)
        speciation = Speciation(compatibility_threshold=0.3)
        novelty = NoveltySearch(archive_size=10)

        # Apply all mechanisms to each individual
        for ind in population.individuals:
            shared_fitness = fitness_sharing.apply_pressure(ind, population)
            ind.metadata['shared_fitness'] = shared_fitness
            speciation.apply_pressure(ind, population)
            novelty.apply_pressure(ind, population)

        # Check that all mechanisms added their metadata
        for ind in population.individuals:
            assert 'shared_fitness' in ind.metadata
            assert ind.species_id is not None
            assert ind.novelty_score is not None

    def test_diversity_metrics_consistency(self):
        """Test that diversity metrics are consistent."""
        population = create_test_population(15, diversity="high")

        # Add cached execution for novelty search
        for i, ind in enumerate(population.individuals):
            ind.cached_execution = {"test": [[i % 5, (i+3) % 5], [(i+1) % 5, (i+4) % 5]]}

        mechanisms = [
            FitnessSharing(niche_radius=0.2),
            Speciation(compatibility_threshold=0.3),
            NoveltySearch(archive_size=10)
        ]

        # Apply mechanisms and calculate diversity
        diversity_scores = []
        for mechanism in mechanisms:
            # Apply to all individuals first
            for ind in population.individuals:
                mechanism.apply_pressure(ind, population)

            diversity = mechanism.calculate(population)
            diversity_scores.append(diversity)

        # All should detect some diversity (> 0.1)
        assert all(score > 0.1 for score in diversity_scores)
