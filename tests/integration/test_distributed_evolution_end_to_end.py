"""
End-to-end integration tests for distributed evolution.

Tests complete workflow from configuration to population merge.
"""

import asyncio
from pathlib import Path

import pytest

from src.adapters.strategies.distributed_checkpoint_manager import AsyncCheckpointManager
from src.adapters.strategies.distributed_evolution import DistributedEvolutionCoordinator
from src.adapters.strategies.distributed_evolution_config import (
    CoordinatorConfig,
    DistributedEvolutionConfig,
    PlatformConfig,
)
from src.adapters.strategies.platform_health_monitor import PlatformHealthMonitor, PlatformStatus
from src.adapters.strategies.population_merger import PopulationMerger
from src.domain.models import ARCTask
from src.infrastructure.components.platform_detector import Platform


class TestDistributedEvolutionEndToEnd:
    """End-to-end tests for distributed evolution system."""

    @pytest.fixture
    def test_config(self, tmp_path: Path) -> DistributedEvolutionConfig:
        """Create test configuration."""
        return DistributedEvolutionConfig(
            enabled=True,
            checkpoint_frequency=1,
            heartbeat_timeout=30,
            heartbeat_interval=10,
            recovery_timeout=60,
            coordinator=CoordinatorConfig(
                api_host="localhost",
                api_port=8000,
                coordinator_id="test-coordinator"
            ),
            platforms=[
                PlatformConfig(
                    id="test-kaggle",
                    role="coordinator",
                    memory_limit_mb=4096,
                    worker_count=2,
                    batch_size=500,
                    platform_type=Platform.KAGGLE
                ),
                PlatformConfig(
                    id="test-colab",
                    role="worker",
                    memory_limit_mb=12288,
                    worker_count=2,
                    batch_size=250,
                    platform_type=Platform.COLAB
                ),
                PlatformConfig(
                    id="test-local",
                    role="worker",
                    memory_limit_mb=16384,
                    worker_count=4,
                    batch_size=1000,
                    platform_type=Platform.LOCAL
                )
            ],
            checkpoint_dir=tmp_path / "checkpoints",
            enable_compression=True,
            compression_level=6
        )

    @pytest.fixture
    def test_task(self) -> ARCTask:
        """Create test ARC task."""
        return ARCTask(
            task_id="test_e2e",
            task_source="test",
            train_examples=[
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            test_input=[[1, 0], [0, 1]],
            test_output=[[0, 1], [1, 0]]
        )

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        test_config: DistributedEvolutionConfig,
        test_task: ARCTask
    ):
        """Test complete distributed evolution workflow."""
        coordinator = DistributedEvolutionCoordinator(
            checkpoint_dir=test_config.checkpoint_dir,
            checkpoint_frequency=test_config.checkpoint_frequency
        )
        health_monitor = PlatformHealthMonitor(
            heartbeat_timeout=test_config.heartbeat_timeout,
            heartbeat_interval=test_config.heartbeat_interval
        )
        merger = PopulationMerger()

        for platform_config in test_config.platforms:
            health_monitor.register_platform(
                platform_id=platform_config.id,
                capabilities={
                    "memory_mb": platform_config.memory_limit_mb,
                    "workers": platform_config.worker_count,
                    "batch_size": platform_config.batch_size
                }
            )

        plan = coordinator.create_distribution_plan(
            task=test_task,
            available_platforms=[p.platform_type for p in test_config.platforms],
            total_generations=50
        )

        assert plan.total_islands > 0
        assert len(plan.platform_tasks) == 3

        platform_populations = {}
        for platform_task in plan.platform_tasks:
            await health_monitor.process_heartbeat(
                platform_id=platform_task.platform.value,
                status="healthy",
                metrics={"generation": 10}
            )

            population = self._create_mock_population(
                platform_id=platform_task.platform.value,
                size=20
            )
            platform_populations[platform_task.platform.value] = population

        merge_result = merger.merge_populations(platform_populations)

        assert len(merge_result.merged_population) > 0
        assert merge_result.platforms_merged == 3
        assert merge_result.duplicates_removed >= 0

    @pytest.mark.asyncio
    async def test_configuration_validation(self, test_config: DistributedEvolutionConfig):
        """Test configuration validation."""
        errors = test_config.validate()

        assert any("GCS bucket required" in error for error in errors)

        test_config.gcs_bucket = "test-bucket"
        test_config.gcs_credentials_path = Path("nonexistent.json")

        errors = test_config.validate()
        assert any("not found" in error for error in errors)

    @pytest.mark.asyncio
    async def test_platform_registration(self, test_config: DistributedEvolutionConfig):
        """Test platform registration and health monitoring."""
        health_monitor = PlatformHealthMonitor(
            heartbeat_timeout=test_config.heartbeat_timeout
        )

        for platform_config in test_config.platforms:
            health_monitor.register_platform(
                platform_id=platform_config.id,
                capabilities={
                    "memory_mb": platform_config.memory_limit_mb,
                    "workers": platform_config.worker_count
                }
            )

        all_status = health_monitor.get_all_platform_status()
        assert len(all_status) == 3

        for platform_id, status in all_status.items():
            assert status.platform_id == platform_id
            assert status.is_healthy

    @pytest.mark.asyncio
    async def test_checkpoint_cycle(
        self,
        test_config: DistributedEvolutionConfig,
        tmp_path: Path
    ):
        """Test checkpoint save and load cycle."""
        checkpoint_manager = AsyncCheckpointManager(
            platform_id="test-platform",
            checkpoint_dir=tmp_path,
            enable_compression=test_config.enable_compression,
            compression_level=test_config.compression_level
        )

        population = self._create_mock_population("test", 10)
        checkpoint = checkpoint_manager.create_checkpoint(
            generation=5,
            population=population
        )

        serialized = checkpoint_manager.serialize_checkpoint(checkpoint)
        restored = checkpoint_manager.deserialize_checkpoint(serialized)

        assert restored.generation == checkpoint.generation
        assert len(restored.population) == len(checkpoint.population)
        assert restored.metadata.platform_id == checkpoint.metadata.platform_id

    @pytest.mark.asyncio
    async def test_distribution_plan_creation(
        self,
        test_config: DistributedEvolutionConfig,
        test_task: ARCTask
    ):
        """Test creation of distribution plan."""
        coordinator = DistributedEvolutionCoordinator(
            checkpoint_dir=test_config.checkpoint_dir
        )

        plan = coordinator.create_distribution_plan(
            task=test_task,
            available_platforms=[p.platform_type for p in test_config.platforms],
            total_generations=100
        )

        assert plan.task_id == test_task.task_id
        assert plan.total_generations == 100
        assert len(plan.platform_tasks) == 3

        coordinator_count = sum(
            1 for pt in plan.platform_tasks
            if pt.role.value == "coordinator"
        )
        assert coordinator_count == 1

        worker_count = sum(
            1 for pt in plan.platform_tasks
            if pt.role.value == "worker"
        )
        assert worker_count == 2

    @pytest.mark.asyncio
    async def test_population_merge_deduplication(self):
        """Test population merge correctly deduplicates."""
        merger = PopulationMerger()

        pop1 = [
            {"program": "prog1", "fitness": 0.9, "hash": "hash1"},
            {"program": "prog2", "fitness": 0.8, "hash": "hash2"},
        ]

        pop2 = [
            {"program": "prog1", "fitness": 0.85, "hash": "hash1"},
            {"program": "prog3", "fitness": 0.75, "hash": "hash3"},
        ]

        merged_result = merger.merge_populations({"platform1": pop1, "platform2": pop2})
        merged = merged_result.merged_population

        assert len(merged) == 3
        hashes = {ind["hash"] for ind in merged}
        assert len(hashes) == 3

        prog1_in_merged = [ind for ind in merged if ind["hash"] == "hash1"]
        assert len(prog1_in_merged) == 1
        assert prog1_in_merged[0]["fitness"] == 0.9

    @pytest.mark.asyncio
    async def test_platform_failover(self, test_config: DistributedEvolutionConfig):
        """Test platform disconnection and task redistribution."""
        health_monitor = PlatformHealthMonitor(
            heartbeat_timeout=test_config.heartbeat_timeout
        )

        for platform_config in test_config.platforms:
            health_monitor.register_platform(
                platform_id=platform_config.id,
                capabilities={
                    "memory_mb": platform_config.memory_limit_mb,
                    "workers": platform_config.worker_count
                }
            )

        await health_monitor.process_heartbeat(
            platform_id="test-kaggle",
            status="healthy",
            metrics={}
        )

        await asyncio.sleep(0.1)

        status = health_monitor.get_platform_status("test-kaggle")
        assert status == PlatformStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_coordinator_config_helpers(self, test_config: DistributedEvolutionConfig):
        """Test configuration helper methods."""
        coordinator_url = test_config.get_coordinator_url()
        assert coordinator_url == "http://localhost:8000"

        coordinator_platform = test_config.get_coordinator_platform()
        assert coordinator_platform is not None
        assert coordinator_platform.role == "coordinator"
        assert coordinator_platform.id == "test-kaggle"

        worker_platforms = test_config.get_worker_platforms()
        assert len(worker_platforms) == 2
        assert all(p.role == "worker" for p in worker_platforms)

        platform_config = test_config.get_platform_config("test-colab")
        assert platform_config is not None
        assert platform_config.memory_limit_mb == 12288

    @pytest.mark.asyncio
    async def test_multi_generation_evolution(
        self,
        test_config: DistributedEvolutionConfig,
        test_task: ARCTask
    ):
        """Test evolution across multiple generations with checkpoints."""
        coordinator = DistributedEvolutionCoordinator(
            checkpoint_dir=test_config.checkpoint_dir,
            checkpoint_frequency=5
        )

        plan = coordinator.create_distribution_plan(
            task=test_task,
            available_platforms=[p.platform_type for p in test_config.platforms],
            total_generations=20
        )

        generations_to_simulate = [5, 10, 15, 20]
        for generation in generations_to_simulate:
            platform_populations = {}

            for platform_task in plan.platform_tasks:
                population = self._create_mock_population(
                    platform_id=platform_task.platform.value,
                    size=15,
                    generation=generation
                )
                platform_populations[platform_task.platform.value] = population

            if generation % test_config.checkpoint_frequency == 0:
                checkpoint_manager = AsyncCheckpointManager(
                    platform_id="test-platform",
                    checkpoint_dir=test_config.checkpoint_dir
                )

                checkpoint = checkpoint_manager.create_checkpoint(
                    generation=generation,
                    population=platform_populations[list(platform_populations.keys())[0]]
                )

                assert checkpoint.generation == generation

        assert test_config.checkpoint_dir.exists()

    def _create_mock_population(
        self,
        platform_id: str,
        size: int,
        generation: int = 0
    ) -> list[dict]:
        """Create mock population for testing."""
        population = []
        for i in range(size):
            population.append({
                "program": f"program_{platform_id}_{generation}_{i}",
                "fitness": 0.5 + (i * 0.01),
                "hash": f"hash_{platform_id}_{generation}_{i}"
            })
        return population
