"""Integration tests for distributed checkpoint synchronization."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.adapters.strategies.distributed_checkpoint_manager import (
    AsyncCheckpointManager,
    Checkpoint,
    CheckpointMetadata,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoint tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_population():
    """Create sample population for testing."""
    return [
        {"program": "rotate_90 | flip_horizontal", "fitness": 0.85},
        {"program": "flip_vertical | crop", "fitness": 0.72},
        {"program": "translate_x(2) | translate_y(1)", "fitness": 0.68},
        {"program": "scale(2) | filter_color(1)", "fitness": 0.91},
        {"program": "pad(1) | crop", "fitness": 0.55},
    ]


@pytest.fixture
def mock_gcs_client():
    """Create mock GCS client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_bucket.list_blobs.return_value = []

    return mock_client


@pytest.mark.asyncio
async def test_checkpoint_create_and_serialize(temp_checkpoint_dir, sample_population):
    """Test checkpoint creation and serialization."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    checkpoint = manager.create_checkpoint(generation=5, population=sample_population)

    assert checkpoint.version == "1.0"
    assert checkpoint.generation == 5
    assert len(checkpoint.population) == 5
    assert checkpoint.metadata.platform_id == "test-platform-1"

    for i, program_checkpoint in enumerate(checkpoint.population):
        assert program_checkpoint.program == sample_population[i]["program"]
        assert program_checkpoint.fitness == sample_population[i]["fitness"]
        assert len(program_checkpoint.hash) == 64

    serialized = manager.serialize_checkpoint(checkpoint)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    deserialized = manager.deserialize_checkpoint(serialized)
    assert deserialized.generation == checkpoint.generation
    assert len(deserialized.population) == len(checkpoint.population)


@pytest.mark.asyncio
async def test_checkpoint_save_and_load_local(temp_checkpoint_dir, sample_population):
    """Test saving and loading checkpoint locally."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    checkpoint = manager.create_checkpoint(generation=10, population=sample_population)

    saved_path = await manager.save_checkpoint_async(checkpoint, upload_to_gcs=False)

    assert saved_path.exists()
    assert "checkpoint_test-platform-1_10_" in saved_path.name
    assert saved_path.suffix == ".msgpack"

    loaded_checkpoint = await manager.load_checkpoint_async(generation=10)

    assert loaded_checkpoint is not None
    assert loaded_checkpoint.generation == 10
    assert len(loaded_checkpoint.population) == 5
    assert loaded_checkpoint.metadata.platform_id == "test-platform-1"


@pytest.mark.asyncio
async def test_checkpoint_validation(temp_checkpoint_dir, sample_population):
    """Test checkpoint validation logic."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    checkpoint = manager.create_checkpoint(generation=5, population=sample_population)

    assert manager.validate_checkpoint(checkpoint) is True

    invalid_checkpoint = Checkpoint(
        version="2.0",
        generation=5,
        population=[],
        metadata=CheckpointMetadata(
            platform_id="test-platform-1",
            timestamp="2025-09-30T12:00:00"
        )
    )

    assert manager.validate_checkpoint(invalid_checkpoint) is False


@pytest.mark.asyncio
async def test_checkpoint_hash_verification(temp_checkpoint_dir):
    """Test hash-based integrity verification."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    population = [{"program": "test_program", "fitness": 0.5}]
    checkpoint = manager.create_checkpoint(generation=1, population=population)

    assert manager.validate_checkpoint(checkpoint) is True

    checkpoint.population[0].hash = "invalid_hash"

    assert manager.validate_checkpoint(checkpoint) is False


@pytest.mark.asyncio
async def test_checkpoint_load_latest(temp_checkpoint_dir, sample_population):
    """Test loading latest checkpoint when generation not specified."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    await manager.save_checkpoint_async(
        manager.create_checkpoint(generation=1, population=sample_population),
        upload_to_gcs=False
    )
    await asyncio.sleep(0.1)

    await manager.save_checkpoint_async(
        manager.create_checkpoint(generation=2, population=sample_population),
        upload_to_gcs=False
    )
    await asyncio.sleep(0.1)

    await manager.save_checkpoint_async(
        manager.create_checkpoint(generation=3, population=sample_population),
        upload_to_gcs=False
    )

    latest_checkpoint = await manager.load_checkpoint_async()

    assert latest_checkpoint is not None
    assert latest_checkpoint.generation == 3


@pytest.mark.asyncio
async def test_checkpoint_background_queue(temp_checkpoint_dir, sample_population):
    """Test background checkpoint queue processing."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    await manager.start_background_checkpointing()

    checkpoint1 = manager.create_checkpoint(generation=1, population=sample_population)
    checkpoint2 = manager.create_checkpoint(generation=2, population=sample_population)
    checkpoint3 = manager.create_checkpoint(generation=3, population=sample_population)

    await manager.queue_checkpoint(checkpoint1, upload_to_gcs=False)
    await manager.queue_checkpoint(checkpoint2, upload_to_gcs=False)
    await manager.queue_checkpoint(checkpoint3, upload_to_gcs=False)

    await manager.stop_background_checkpointing()

    checkpoint_files = list(temp_checkpoint_dir.glob("checkpoint_test-platform-1_*.msgpack"))
    assert len(checkpoint_files) == 3


@pytest.mark.asyncio
async def test_checkpoint_gcs_upload(temp_checkpoint_dir, sample_population):
    """Test GCS upload functionality."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir,
        gcs_bucket="test-bucket",
        gcs_credentials="fake-credentials.json"
    )

    manager.gcs_client = mock_client

    checkpoint = manager.create_checkpoint(generation=5, population=sample_population)

    await manager.save_checkpoint_async(checkpoint, upload_to_gcs=True)

    mock_client.bucket.assert_called_once_with("test-bucket")
    mock_blob.upload_from_filename.assert_called_once()


@pytest.mark.asyncio
async def test_checkpoint_gcs_download(temp_checkpoint_dir, sample_population):
    """Test GCS download functionality."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir,
        gcs_bucket="test-bucket",
        gcs_credentials="fake-credentials.json"
    )

    checkpoint = manager.create_checkpoint(generation=5, population=sample_population)
    serialized_data = manager.serialize_checkpoint(checkpoint)

    mock_blob.download_as_bytes.return_value = serialized_data
    mock_blob.time_created = "2025-09-30T12:00:00"
    mock_bucket.list_blobs.return_value = [mock_blob]

    mock_client.bucket.return_value = mock_bucket

    manager.gcs_client = mock_client

    loaded_checkpoint = await manager.load_checkpoint_async(generation=5, from_gcs=True)

    assert loaded_checkpoint is not None
    assert loaded_checkpoint.generation == 5
    assert len(loaded_checkpoint.population) == 5


@pytest.mark.asyncio
async def test_checkpoint_multi_platform_sync(temp_checkpoint_dir, sample_population):
    """Test synchronization across multiple platforms."""
    platform1_dir = temp_checkpoint_dir / "platform1"
    platform2_dir = temp_checkpoint_dir / "platform2"
    platform3_dir = temp_checkpoint_dir / "platform3"

    platform1_dir.mkdir()
    platform2_dir.mkdir()
    platform3_dir.mkdir()

    manager1 = AsyncCheckpointManager(platform_id="platform-1", checkpoint_dir=platform1_dir)
    manager2 = AsyncCheckpointManager(platform_id="platform-2", checkpoint_dir=platform2_dir)
    manager3 = AsyncCheckpointManager(platform_id="platform-3", checkpoint_dir=platform3_dir)

    checkpoint1 = manager1.create_checkpoint(generation=10, population=sample_population[:2])
    checkpoint2 = manager2.create_checkpoint(generation=10, population=sample_population[2:4])
    checkpoint3 = manager3.create_checkpoint(generation=10, population=sample_population[4:])

    await manager1.save_checkpoint_async(checkpoint1, upload_to_gcs=False)
    await manager2.save_checkpoint_async(checkpoint2, upload_to_gcs=False)
    await manager3.save_checkpoint_async(checkpoint3, upload_to_gcs=False)

    loaded1 = await manager1.load_checkpoint_async(generation=10)
    loaded2 = await manager2.load_checkpoint_async(generation=10)
    loaded3 = await manager3.load_checkpoint_async(generation=10)

    assert loaded1 is not None and len(loaded1.population) == 2
    assert loaded2 is not None and len(loaded2.population) == 2
    assert loaded3 is not None and len(loaded3.population) == 1

    all_programs = (
        [p.program for p in loaded1.population] +
        [p.program for p in loaded2.population] +
        [p.program for p in loaded3.population]
    )

    assert len(all_programs) == 5
    assert all(prog in [p["program"] for p in sample_population] for prog in all_programs)


@pytest.mark.asyncio
async def test_checkpoint_resource_usage_tracking(temp_checkpoint_dir, sample_population):
    """Test resource usage tracking in checkpoints."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    checkpoint = manager.create_checkpoint(generation=5, population=sample_population)

    assert "cpu_percent" in checkpoint.metadata.resource_usage
    assert "memory_mb" in checkpoint.metadata.resource_usage
    assert isinstance(checkpoint.metadata.resource_usage["cpu_percent"], float)
    assert isinstance(checkpoint.metadata.resource_usage["memory_mb"], float)


@pytest.mark.asyncio
async def test_checkpoint_generation_based_sync(temp_checkpoint_dir, sample_population):
    """Test generation-based checkpoint synchronization."""
    manager = AsyncCheckpointManager(
        platform_id="test-platform-1",
        checkpoint_dir=temp_checkpoint_dir
    )

    for generation in range(1, 6):
        checkpoint = manager.create_checkpoint(generation=generation, population=sample_population)
        await manager.save_checkpoint_async(checkpoint, upload_to_gcs=False)
        await asyncio.sleep(0.05)

    for generation in range(1, 6):
        loaded = await manager.load_checkpoint_async(generation=generation)
        assert loaded is not None
        assert loaded.generation == generation

    latest = await manager.load_checkpoint_async()
    assert latest is not None
    assert latest.generation == 5
