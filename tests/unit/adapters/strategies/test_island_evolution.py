"""
Unit tests for island evolution model implementation.

Tests Task 7.1: Island model for parallel population evolution.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.strategies.evolution_engine import Individual, Population
from src.adapters.strategies.island_evolution import Island, IslandEvolutionEngine, MigrationPolicy
from src.domain.models import ARCTask
from src.infrastructure.config import GeneticAlgorithmConfig


class TestIsland:
    """Test Island class functionality."""

    def test_island_initialization(self):
        """Test island initialization."""
        population = Population()
        mock_engine = MagicMock()

        island = Island(
            id=1,
            population=population,
            evolution_engine=mock_engine
        )

        assert island.id == 1
        assert island.population == population
        assert island.evolution_engine == mock_engine
        assert island.migration_history == []
        assert island.fitness_trend == []

    def test_update_fitness_trend(self):
        """Test fitness trend tracking."""
        population = Population()
        # Add some individuals with fitness
        for i in range(5):
            ind = Individual(operations=[])
            ind.fitness = 0.5 + i * 0.1
            population.add_individual(ind)

        island = Island(
            id=1,
            population=population,
            evolution_engine=MagicMock()
        )

        # Update fitness trend
        island.update_fitness_trend()

        assert len(island.fitness_trend) == 1
        assert island.fitness_trend[0] == pytest.approx(0.7)  # Average of 0.5, 0.6, 0.7, 0.8, 0.9

    def test_stagnation_detection(self):
        """Test island stagnation detection."""
        island = Island(
            id=1,
            population=Population(),
            evolution_engine=MagicMock()
        )

        # Not stagnant with insufficient history
        assert not island.is_stagnant(lookback=5)

        # Add flat fitness trend
        island.fitness_trend = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert island.is_stagnant(lookback=5, threshold=0.001)

        # Add improving fitness trend
        island.fitness_trend = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
        assert not island.is_stagnant(lookback=5, threshold=0.05)


class TestMigrationPolicy:
    """Test MigrationPolicy configuration."""

    def test_default_policy(self):
        """Test default migration policy settings."""
        policy = MigrationPolicy()

        assert policy.frequency == 10
        assert policy.migration_rate == 0.1
        assert policy.selection_method == "tournament"
        assert policy.topology == "ring"
        assert policy.adaptive == True

    def test_custom_policy(self):
        """Test custom migration policy."""
        policy = MigrationPolicy(
            frequency=20,
            migration_rate=0.2,
            selection_method="best",
            topology="fully_connected",
            adaptive=False
        )

        assert policy.frequency == 20
        assert policy.migration_rate == 0.2
        assert policy.selection_method == "best"
        assert policy.topology == "fully_connected"
        assert policy.adaptive == False


class TestIslandEvolutionEngine:
    """Test IslandEvolutionEngine functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        
        # Population configuration
        config.population = MagicMock()
        config.population.size = 100
        
        # Parallelization configuration
        config.parallelization = MagicMock()
        config.parallelization.workers = 4
        
        # Diversity configuration
        config.diversity = MagicMock()
        config.diversity.method = "fitness_sharing"
        
        # Genetic operators configuration
        config.genetic_operators = MagicMock()
        config.genetic_operators.mutation = MagicMock()
        config.genetic_operators.mutation.base_rate = 0.1
        config.genetic_operators.crossover = MagicMock()
        config.genetic_operators.crossover.rate = 0.7
        
        return config

    @pytest.fixture
    def mock_dsl_engine(self):
        """Create mock DSL engine."""
        return MagicMock()

    def test_initialization(self, mock_config, mock_dsl_engine):
        """Test island evolution engine initialization."""
        engine = IslandEvolutionEngine(
            num_islands=4,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        assert engine.num_islands == 4
        assert engine.island_pop_size == 25  # 100 / 4
        assert len(engine.island_configs) == 4
        assert engine.migration_policy is not None
        assert engine.islands == []
        assert engine.global_best_individual is None

    def test_island_config_variations(self, mock_config, mock_dsl_engine):
        """Test that island configurations vary properly."""
        engine = IslandEvolutionEngine(
            num_islands=4,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        configs = engine.island_configs

        # Check population sizes
        for config in configs:
            assert config.population.size == 25

        # Check parameter variations
        # Island 0: High mutation
        assert configs[0].genetic_operators.mutation.base_rate > mock_config.genetic_operators.mutation.base_rate
        assert configs[0].genetic_operators.crossover.rate < mock_config.genetic_operators.crossover.rate

        # Island 1: High crossover
        assert configs[1].genetic_operators.crossover.rate > mock_config.genetic_operators.crossover.rate
        assert configs[1].genetic_operators.mutation.base_rate < mock_config.genetic_operators.mutation.base_rate

        # Check diversity methods vary
        diversity_methods = [c.diversity.method for c in configs]
        assert len(set(diversity_methods)) > 1  # At least 2 different methods

    @pytest.mark.asyncio
    async def test_island_initialization(self, mock_config, mock_dsl_engine):
        """Test async island initialization."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        # Mock task
        task = MagicMock(spec=ARCTask)

        # Patch evolution engine initialization
        with patch.object(engine, '_copy_config', return_value=mock_config):
            # Create mock evolution engines
            mock_evolution_engine = MagicMock()
            mock_evolution_engine._initialize_population = asyncio.coroutine(MagicMock())
            mock_evolution_engine.population = Population()

            with patch('src.adapters.strategies.island_evolution.EvolutionEngine', return_value=mock_evolution_engine):
                await engine.initialize_islands(task)

        assert len(engine.islands) == 2
        for island in engine.islands:
            assert isinstance(island, Island)
            assert island.evolution_engine is not None
            assert island.population is not None

    def test_select_migrants_best(self, mock_config, mock_dsl_engine):
        """Test migrant selection using best method."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        engine.migration_policy.selection_method = "best"
        engine.migration_policy.migration_rate = 0.2

        # Create island with population
        population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.fitness = i * 0.1
            population.add_individual(ind)

        island = Island(1, population, MagicMock())

        # Select migrants (20% of 10 = 2)
        migrants = engine._select_migrants(island)

        assert len(migrants) == 2
        # Should select best individuals
        assert migrants[0].fitness == 0.9
        assert migrants[1].fitness == 0.8

    def test_select_migrants_tournament(self, mock_config, mock_dsl_engine):
        """Test migrant selection using tournament method."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        engine.migration_policy.selection_method = "tournament"
        engine.migration_policy.migration_rate = 0.2

        # Create island with population
        population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.fitness = i * 0.1
            population.add_individual(ind)

        island = Island(1, population, MagicMock())

        # Select migrants
        migrants = engine._select_migrants(island)

        assert len(migrants) == 2
        # Tournament should tend to select higher fitness individuals
        assert all(m.fitness >= 0.0 for m in migrants)

    def test_integrate_migrants(self, mock_config, mock_dsl_engine):
        """Test migrant integration into target island."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        # Create target island
        population = Population()
        for i in range(5):
            ind = Individual(operations=[])
            ind.fitness = i * 0.1
            population.add_individual(ind)

        target_island = Island(1, population, MagicMock())

        # Create migrants
        migrants = []
        for i in range(2):
            migrant = Individual(operations=[])
            migrant.fitness = 0.8 + i * 0.1
            migrant.metadata['island_id'] = 0
            migrants.append(migrant)

        # Integrate migrants
        engine._integrate_migrants(target_island, migrants)

        # Check population size maintained
        assert target_island.population.size() == 5

        # Check worst individuals replaced
        fitness_values = [ind.fitness for ind in target_island.population.individuals]
        assert 0.8 in fitness_values  # First migrant
        assert 0.9 in fitness_values  # Second migrant

        # Check migration metadata added
        migrated_individuals = [
            ind for ind in target_island.population.individuals
            if 'migration_generation' in ind.metadata
        ]
        assert len(migrated_individuals) == 2

    def test_migration_ring_topology(self, mock_config, mock_dsl_engine):
        """Test ring topology migration pattern."""
        engine = IslandEvolutionEngine(
            num_islands=4,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        engine.migration_policy.topology = "ring"

        # Create islands
        for i in range(4):
            population = Population()
            island = Island(i, population, MagicMock())
            engine.islands.append(island)

        # Mock select and integrate methods
        engine._select_migrants = MagicMock(return_value=[])
        engine._integrate_migrants = MagicMock()

        # Perform migration
        asyncio.run(engine._perform_migration())

        # Check ring pattern: 0->1, 1->2, 2->3, 3->0
        assert engine._select_migrants.call_count == 4
        assert engine._integrate_migrants.call_count == 4

    def test_adaptive_island_parameters(self, mock_config, mock_dsl_engine):
        """Test adaptive parameter adjustment."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        engine.migration_policy.adaptive = True

        # Create stagnant island
        population = Population()
        for i in range(5):
            ind = Individual(operations=[])
            ind.fitness = 0.5  # All same fitness
            population.add_individual(ind)

        mock_evolution_engine = MagicMock()
        mock_evolution_engine.config.genetic_operators.mutation.base_rate = 0.1
        mock_evolution_engine.config.genetic_operators.crossover.rate = 0.7

        island = Island(1, population, mock_evolution_engine)
        island.fitness_trend = [0.5] * 10  # Stagnant trend
        engine.islands.append(island)

        # Low diversity score
        engine.island_diversity_scores[1] = 0.2

        # Apply adaptation
        engine._adapt_island_parameters()

        # Check mutation rate increased due to low diversity and stagnation
        assert mock_evolution_engine.config.genetic_operators.mutation.base_rate > 0.1

    def test_convergence_check(self, mock_config, mock_dsl_engine):
        """Test convergence checking."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        # No convergence without best individual
        assert not engine._check_convergence()

        # Create high-fitness individual
        best = Individual(operations=[])
        best.fitness = 0.96
        engine.global_best_individual = best

        # Should converge with high fitness
        assert engine._check_convergence()

        # Test stagnation-based convergence
        engine.global_best_individual.fitness = 0.5

        # Create stagnant islands
        for i in range(2):
            population = Population()
            island = Island(i, population, MagicMock())
            island.fitness_trend = [0.5] * 10
            engine.islands.append(island)

        # Should converge when most islands stagnant
        assert engine._check_convergence()

    def test_evolution_stats_collection(self, mock_config, mock_dsl_engine):
        """Test collection of evolution statistics."""
        engine = IslandEvolutionEngine(
            num_islands=2,
            config=mock_config,
            dsl_engine=mock_dsl_engine
        )

        engine.generation = 50
        engine.migration_count = 5

        # Create islands with data
        for i in range(2):
            population = Population()
            ind = Individual(operations=[])
            ind.fitness = 0.7 + i * 0.1
            population.add_individual(ind)
            population.best_individual = ind

            island = Island(i, population, MagicMock())
            island.fitness_trend = [0.5, 0.6, 0.7]
            engine.islands.append(island)
            engine.island_diversity_scores[i] = 0.5
            engine.island_specialization[i] = "balanced"

        # Set global best
        engine.global_best_individual = engine.islands[1].population.best_individual

        # Collect stats
        stats = engine._collect_evolution_stats()

        assert stats['num_islands'] == 2
        assert stats['total_generations'] == 50
        assert stats['migration_count'] == 5
        assert stats['best_fitness'] == 0.8
        assert len(stats['island_stats']) == 2
        assert stats['island_stats'][0]['best_fitness'] == 0.7
        assert stats['island_stats'][1]['best_fitness'] == 0.8
        assert stats['island_stats'][0]['diversity_score'] == 0.5
        assert stats['island_stats'][0]['specialization'] == "balanced"
