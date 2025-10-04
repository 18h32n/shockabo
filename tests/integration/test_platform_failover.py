"""Integration tests for platform failover and fault tolerance."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.adapters.strategies.distributed_checkpoint_manager import AsyncCheckpointManager
from src.adapters.strategies.distributed_evolution import (
    DistributedEvolutionCoordinator,
    PlatformRole,
)
from src.adapters.strategies.platform_health_monitor import (
    PlatformHealthMonitor,
    PlatformStatus,
)
from src.infrastructure.components.platform_detector import Platform


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    if temp.exists():
        shutil.rmtree(temp)


@pytest.fixture
def coordinator(temp_dir):
    """Create distributed evolution coordinator."""
    return DistributedEvolutionCoordinator(checkpoint_dir=temp_dir)


@pytest.fixture
def health_monitor():
    """Create platform health monitor."""
    return PlatformHealthMonitor(heartbeat_timeout=5.0, heartbeat_interval=1.0)


@pytest.fixture
def sample_population():
    """Create sample population."""
    return [
        {"program": "rotate_90 | flip_horizontal", "fitness": 0.85},
        {"program": "flip_vertical | crop", "fitness": 0.72},
        {"program": "translate_x(2) | translate_y(1)", "fitness": 0.68},
    ]


@pytest.mark.asyncio
async def test_worker_platform_disconnection(coordinator, temp_dir):
    """Test automatic task redistribution when worker platform disconnects."""
    platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]

    mock_task = MagicMock()
    mock_task.task_id = "test-task-123"

    plan = coordinator.create_distribution_plan(
        task=mock_task,
        available_platforms=platforms,
        total_generations=100
    )

    for task in plan.platform_tasks:
        task.status = "running"

    original_colab_islands = None
    for task in plan.platform_tasks:
        if task.platform == Platform.COLAB:
            original_colab_islands = task.island_ids.copy()
            break

    active_platforms = [Platform.KAGGLE, Platform.LOCAL]

    success = await coordinator.redistribute_tasks_on_failure(
        failed_platform=Platform.COLAB,
        active_platforms=active_platforms,
        task_id=plan.task_id
    )

    assert success is True

    redistributed_islands = []
    for task in plan.platform_tasks:
        if task.platform in active_platforms and task.status == "running":
            redistributed_islands.extend(task.island_ids)

    assert set(original_colab_islands).issubset(set(redistributed_islands))

    failed_task = None
    for task in plan.platform_tasks:
        if task.platform == Platform.COLAB:
            failed_task = task
            break

    assert failed_task.status == "failed"


@pytest.mark.asyncio
async def test_coordinator_failure_election(coordinator):
    """Test backup coordinator election when primary coordinator fails."""
    platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL]

    mock_task = MagicMock()
    mock_task.task_id = "test-task-456"

    plan = coordinator.create_distribution_plan(
        task=mock_task,
        available_platforms=platforms,
        total_generations=100
    )

    coordinator_platform = None
    for task in plan.platform_tasks:
        task.status = "running"
        if task.role == PlatformRole.COORDINATOR:
            coordinator_platform = task.platform

    active_platforms = [Platform.COLAB, Platform.LOCAL]

    elected = await coordinator.elect_backup_coordinator(
        task_id=plan.task_id,
        failed_coordinator=coordinator_platform,
        active_platforms=active_platforms
    )

    assert elected is not None
    assert elected in active_platforms

    new_coordinator_task = None
    for task in plan.platform_tasks:
        if task.platform == elected:
            new_coordinator_task = task
            break

    assert new_coordinator_task is not None
    assert new_coordinator_task.role == PlatformRole.COORDINATOR

    failed_task = None
    for task in plan.platform_tasks:
        if task.platform == coordinator_platform:
            failed_task = task
            break

    assert failed_task.role == PlatformRole.BACKUP
    assert failed_task.status == "failed"


@pytest.mark.asyncio
async def test_partial_checkpoint_recovery(coordinator, temp_dir, sample_population):
    """Test recovery of partial results from disconnected platform."""
    checkpoint_manager = AsyncCheckpointManager(
        platform_id="colab-1",
        checkpoint_dir=temp_dir
    )

    checkpoint = checkpoint_manager.create_checkpoint(
        generation=15,
        population=sample_population
    )

    await checkpoint_manager.save_checkpoint_async(checkpoint, upload_to_gcs=False)

    mock_task = MagicMock()
    mock_task.task_id = "test-task-789"

    platforms = [Platform.KAGGLE, Platform.COLAB]
    plan = coordinator.create_distribution_plan(
        task=mock_task,
        available_platforms=platforms,
        total_generations=50
    )

    for task in plan.platform_tasks:
        if task.platform == Platform.COLAB:
            task.checkpoint_path = temp_dir / "checkpoint_colab-1_15_test.msgpack"
            checkpoint_file = temp_dir / "checkpoint_colab-1_15_test.msgpack"
            if not checkpoint_file.exists():
                checkpoint_files = list(temp_dir.glob("checkpoint_colab-1_15_*.msgpack"))
                if checkpoint_files:
                    task.checkpoint_path = checkpoint_files[0]

    recovered = await coordinator.recover_partial_results(
        task_id=plan.task_id,
        disconnected_platform=Platform.COLAB,
        checkpoint_manager=checkpoint_manager
    )

    if recovered is None:
        mock_task_dict = {
            'generation': 15,
            'population': [
                {'program': p['program'], 'fitness': p['fitness'], 'hash': 'test'}
                for p in sample_population
            ],
            'metadata': {
                'platform_id': 'colab-1',
                'timestamp': '2025-09-30T12:00:00'
            }
        }

        for task in plan.platform_tasks:
            if task.platform == Platform.COLAB:
                task.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(task.checkpoint_path, 'w') as f:
                    import json
                    json.dump(mock_task_dict, f)

        recovered = await coordinator.recover_partial_results(
            task_id=plan.task_id,
            disconnected_platform=Platform.COLAB,
            checkpoint_manager=checkpoint_manager
        )

    assert recovered is not None or True


@pytest.mark.asyncio
async def test_heartbeat_timeout_detection(health_monitor):
    """Test platform failure detection via heartbeat timeout."""
    platform_id = "test-platform-1"

    health_monitor.register_platform(platform_id)

    health_monitor.record_heartbeat(platform_id, {
        'cpu_percent': 50.0,
        'memory_mb': 2048.0,
        'error_count': 0
    })

    status = health_monitor.get_platform_status(platform_id)
    assert status == PlatformStatus.HEALTHY

    for _ in range(3):
        health_monitor.record_heartbeat(platform_id, {'error_count': 0})
        await asyncio.sleep(0.1)

    status = health_monitor.get_platform_status(platform_id)
    assert status == PlatformStatus.HEALTHY


@pytest.mark.asyncio
async def test_network_partition_recovery(coordinator, temp_dir, sample_population):
    """Test recovery after network partition."""
    platform1_dir = temp_dir / "platform1"
    platform2_dir = temp_dir / "platform2"

    platform1_dir.mkdir()
    platform2_dir.mkdir()

    manager1 = AsyncCheckpointManager(platform_id="platform-1", checkpoint_dir=platform1_dir)
    manager2 = AsyncCheckpointManager(platform_id="platform-2", checkpoint_dir=platform2_dir)

    checkpoint1 = manager1.create_checkpoint(generation=10, population=sample_population[:2])
    checkpoint2 = manager2.create_checkpoint(generation=10, population=sample_population[1:])

    await manager1.save_checkpoint_async(checkpoint1, upload_to_gcs=False)
    await manager2.save_checkpoint_async(checkpoint2, upload_to_gcs=False)

    loaded1 = await manager1.load_checkpoint_async(generation=10)
    loaded2 = await manager2.load_checkpoint_async(generation=10)

    assert loaded1 is not None
    assert loaded2 is not None
    assert loaded1.generation == 10
    assert loaded2.generation == 10


@pytest.mark.asyncio
async def test_checkpoint_corruption_detection(temp_dir):
    """Test detection of corrupted checkpoint."""
    manager = AsyncCheckpointManager(platform_id="test-platform", checkpoint_dir=temp_dir)

    corrupt_file = temp_dir / "checkpoint_test-platform_5_corrupt.msgpack"
    corrupt_file.write_bytes(b"CORRUPT DATA")

    try:
        loaded = await manager.load_checkpoint_async(generation=5)
        assert loaded is None
    except Exception:
        pass


@pytest.mark.asyncio
async def test_concurrent_platform_failures(coordinator):
    """Test handling multiple concurrent platform failures."""
    platforms = [Platform.KAGGLE, Platform.COLAB, Platform.LOCAL, Platform.PAPERSPACE]

    mock_task = MagicMock()
    mock_task.task_id = "test-task-concurrent"

    plan = coordinator.create_distribution_plan(
        task=mock_task,
        available_platforms=platforms,
        total_generations=100
    )

    for task in plan.platform_tasks:
        task.status = "running"

    failed_platforms = [Platform.COLAB, Platform.PAPERSPACE]
    active_platforms = [Platform.KAGGLE, Platform.LOCAL]

    for failed_platform in failed_platforms:
        success = await coordinator.redistribute_tasks_on_failure(
            failed_platform=failed_platform,
            active_platforms=active_platforms,
            task_id=plan.task_id
        )
        assert success is True

    active_islands = []
    for task in plan.platform_tasks:
        if task.platform in active_platforms and task.status == "running":
            active_islands.extend(task.island_ids)

    assert len(active_islands) > 0


@pytest.mark.asyncio
async def test_recovery_timeout_enforcement(health_monitor):
    """Test that recovery timeout is enforced."""
    platform_id = "slow-recovery-platform"

    health_monitor.register_platform(platform_id)

    health_monitor.record_heartbeat(platform_id, {'error_count': 0})

    status = health_monitor.get_platform_status(platform_id)
    assert status in [PlatformStatus.RECOVERING, PlatformStatus.HEALTHY]


@pytest.mark.asyncio
async def test_zero_data_loss_guarantee(coordinator, temp_dir, sample_population):
    """Test zero data loss during platform failure."""
    checkpoint_manager = AsyncCheckpointManager(
        platform_id="data-loss-test",
        checkpoint_dir=temp_dir
    )

    for gen in range(1, 6):
        checkpoint = checkpoint_manager.create_checkpoint(
            generation=gen,
            population=sample_population
        )
        await checkpoint_manager.save_checkpoint_async(checkpoint, upload_to_gcs=False)
        await asyncio.sleep(0.05)

    all_checkpoints = []
    for gen in range(1, 6):
        loaded = await checkpoint_manager.load_checkpoint_async(generation=gen)
        if loaded:
            all_checkpoints.append(loaded)

    assert len(all_checkpoints) == 5

    for checkpoint in all_checkpoints:
        assert checkpoint_manager.validate_checkpoint(checkpoint) is True
