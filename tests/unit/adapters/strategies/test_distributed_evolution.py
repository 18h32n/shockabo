"""Unit tests for DistributedEvolutionCoordinator."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.adapters.strategies.distributed_evolution import (
    DistributedEvolutionCoordinator,
)
from src.infrastructure.components.platform_detector import Platform


class MigrationScheduler:
    """Mock migration scheduler for testing."""
    def __init__(self, interval=5):
        self.interval = interval

    def should_migrate(self, generation):
        return generation > 0 and generation % self.interval == 0


class PopulationMigrator:
    """Mock population migrator for testing."""
    def select_migrants(self, population, count):
        return sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:count]

    def integrate_migrants(self, target_pop, migrants, max_size):
        combined = list(target_pop.individuals) + list(migrants)
        sorted_combined = sorted(combined, key=lambda x: x.fitness, reverse=True)[:max_size]
        target_pop.individuals = sorted_combined
        return target_pop


class TestDistributedEvolutionCoordinator:
    """Test suite for DistributedEvolutionCoordinator."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.distributed.enabled = True
        config.distributed.platforms = ["kaggle", "colab", "paperspace"]
        config.distributed.coordinator_platform = "colab"
        config.distributed.migration_interval = 10
        config.distributed.migration_size = 10
        config.distributed.sync_interval = 5
        config.distributed.checkpoint_path = "distributed_checkpoints"
        return config

    @pytest.fixture
    def coordinator(self, mock_config):
        """Create coordinator instance."""
        return DistributedEvolutionCoordinator()

    def test_initialization(self, mock_config):
        """Test coordinator initialization."""
        coordinator = DistributedEvolutionCoordinator()

        assert coordinator.checkpoint_dir.exists()
        assert coordinator.active_plans == {}
        assert coordinator.platform_metrics == {}


class TestPopulationSharding:
    """Test suite for population sharding functionality."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for sharding tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return DistributedEvolutionCoordinator(checkpoint_dir=Path(temp_dir))

    def test_calculate_platform_shard_sizes_basic(self, coordinator):
        """Test basic shard size calculation."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]
        total_population = 1000

        shard_sizes = coordinator.calculate_platform_shard_sizes(
            platforms=platforms,
            total_population_size=total_population
        )

        # Should return dict with all platforms
        assert len(shard_sizes) == 3
        assert all(platform in shard_sizes for platform in platforms)

        # Total should match input
        assert sum(shard_sizes.values()) == total_population

        # All shards should be >= minimum size (10)
        assert all(size >= 10 for size in shard_sizes.values())

    def test_calculate_platform_shard_sizes_weighted(self, coordinator):
        """Test shard sizes respect platform capabilities."""
        platforms = [Platform.KAGGLE, Platform.PAPERSPACE, Platform.LOCAL]
        total_population = 1000

        shard_sizes = coordinator.calculate_platform_shard_sizes(
            platforms=platforms,
            total_population_size=total_population
        )

        # Local should get most (best resources)
        # Kaggle should get more than Paperspace
        assert shard_sizes[Platform.LOCAL] > shard_sizes[Platform.KAGGLE]
        assert shard_sizes[Platform.KAGGLE] > shard_sizes[Platform.PAPERSPACE]

    def test_calculate_platform_shard_sizes_with_load(self, coordinator):
        """Test shard sizes adjust for current platform load."""
        platforms = [Platform.KAGGLE, Platform.COLAB]
        total_population = 1000

        # Kaggle at high load, Colab available
        platform_capabilities = {
            Platform.KAGGLE: {
                'current_load': 0.9,
                'memory_available': 0.2
            },
            Platform.COLAB: {
                'current_load': 0.1,
                'memory_available': 0.9
            }
        }

        shard_sizes = coordinator.calculate_platform_shard_sizes(
            platforms=platforms,
            total_population_size=total_population,
            platform_capabilities=platform_capabilities
        )

        # Colab should get more due to lower load
        assert shard_sizes[Platform.COLAB] > shard_sizes[Platform.KAGGLE]

    def test_calculate_platform_shard_sizes_empty_platforms(self, coordinator):
        """Test error handling for empty platform list."""
        with pytest.raises(ValueError, match="No platforms provided"):
            coordinator.calculate_platform_shard_sizes(
                platforms=[],
                total_population_size=1000
            )

    def test_assign_islands_to_platforms_basic(self, coordinator):
        """Test basic island assignment."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]
        total_islands = 9

        assignments = coordinator.assign_islands_to_platforms(
            platforms=platforms,
            total_islands=total_islands
        )

        # All platforms should be assigned
        assert len(assignments) == 3
        assert all(platform in assignments for platform in platforms)

        # All islands should be assigned
        all_islands = []
        for islands in assignments.values():
            all_islands.extend(islands)
        assert sorted(all_islands) == list(range(total_islands))

        # No duplicate islands
        assert len(all_islands) == len(set(all_islands))

    def test_assign_islands_with_shard_sizes(self, coordinator):
        """Test island assignment proportional to shard sizes."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]
        total_islands = 10

        # Local gets 50%, Kaggle 30%, Colab 20%
        shard_sizes = {
            Platform.LOCAL: 500,
            Platform.KAGGLE: 300,
            Platform.COLAB: 200
        }

        assignments = coordinator.assign_islands_to_platforms(
            platforms=platforms,
            total_islands=total_islands,
            shard_sizes=shard_sizes
        )

        # Local should get most islands
        assert len(assignments[Platform.LOCAL]) >= len(assignments[Platform.KAGGLE])
        assert len(assignments[Platform.KAGGLE]) >= len(assignments[Platform.COLAB])

        # All islands assigned
        total_assigned = sum(len(islands) for islands in assignments.values())
        assert total_assigned == total_islands

    def test_assign_islands_insufficient_islands(self, coordinator):
        """Test error when islands < platforms."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]
        total_islands = 2

        with pytest.raises(ValueError, match="Total islands.*must be >="):
            coordinator.assign_islands_to_platforms(
                platforms=platforms,
                total_islands=total_islands
            )

    def test_assign_islands_empty_platforms(self, coordinator):
        """Test error handling for empty platform list."""
        with pytest.raises(ValueError, match="No platforms provided"):
            coordinator.assign_islands_to_platforms(
                platforms=[],
                total_islands=10
            )

    def test_assign_islands_single_platform(self, coordinator):
        """Test island assignment with single platform."""
        platforms = [Platform.LOCAL]
        total_islands = 5

        assignments = coordinator.assign_islands_to_platforms(
            platforms=platforms,
            total_islands=total_islands
        )

        assert len(assignments) == 1
        assert assignments[Platform.LOCAL] == list(range(5))

    def test_assign_islands_minimum_per_platform(self, coordinator):
        """Test each platform gets at least 1 island."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]
        total_islands = 3

        assignments = coordinator.assign_islands_to_platforms(
            platforms=platforms,
            total_islands=total_islands
        )

        # Each platform should get exactly 1 island
        assert all(len(islands) == 1 for islands in assignments.values())
