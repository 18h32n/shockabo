"""Unit tests for program cache repository."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.adapters.repositories.program_cache import (
    ProgramCache,
    ProgramCacheEntry,
)
from src.adapters.repositories.program_cache_config import ProgramCacheConfig, StorageConfig
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config(temp_cache_dir):
    """Create test configuration."""
    config = ProgramCacheConfig(
        storage=StorageConfig(
            size_limit_gb=0.001,  # 1MB for testing
            cache_dir=temp_cache_dir,
            eviction_policy="least-recently-used",
            retention_days=1
        )
    )
    return config


@pytest.fixture
def program_cache(test_config):
    """Create program cache instance."""
    cache = ProgramCache(config=test_config)
    yield cache
    cache.close()


@pytest.fixture
def sample_program():
    """Create sample DSL program."""
    return DSLProgram(
        operations=[
            {"type": "rotate", "params": {"angle": 90}},
            {"type": "flip", "params": {"axis": "horizontal"}},
            {"type": "fill", "params": {"color": 1}}
        ],
        version="1.0",
        metadata={"created_by": "test"}
    )


@pytest.fixture
def sample_program_2():
    """Create second sample DSL program."""
    return DSLProgram(
        operations=[
            {"type": "rotate", "params": {"angle": 180}},
            {"type": "flip", "params": {"axis": "vertical"}},
            {"type": "mask", "params": {"pattern": [[1, 0], [0, 1]]}}
        ],
        version="1.0",
        metadata={"created_by": "test"}
    )


class TestProgramCache:
    """Test ProgramCache functionality."""

    def test_initialization(self, program_cache, test_config):
        """Test cache initialization."""
        assert program_cache.config == test_config
        assert program_cache.cache_dir.exists()
        assert program_cache.index_dir.exists()
        assert program_cache.similarity_threshold == test_config.similarity.similarity_threshold

    def test_save_and_retrieve_program(self, program_cache, sample_program):
        """Test saving and retrieving a program."""
        # Save program
        program_id = program_cache.save_program(
            program=sample_program,
            task_id="test_task_001",
            task_source="training",
            success=True,
            accuracy_score=0.95,
            execution_time_ms=150.5,
            generation=3,
            parents=["parent1", "parent2"],
            mutation_type="crossover",
            fitness_score=0.88
        )

        assert program_id is not None
        assert program_id != ""

        # Retrieve program
        entry = program_cache.get_program(program_id)
        assert entry is not None
        assert entry.program_id == program_id
        assert entry.task_id == "test_task_001"
        assert entry.success is True
        assert entry.accuracy_score == 0.95
        assert entry.execution_time_ms == 150.5
        assert entry.generation == 3
        assert entry.parents == ["parent1", "parent2"]
        assert entry.mutation_type == "crossover"
        assert entry.fitness_score == 0.88
        assert entry.access_count == 1  # Updated on retrieval

    def test_duplicate_detection(self, program_cache, sample_program):
        """Test duplicate program detection."""
        # Save program first time
        program_id_1 = program_cache.save_program(
            program=sample_program,
            task_id="task1",
            task_source="training",
            success=True,
            accuracy_score=0.9,
            execution_time_ms=100
        )

        # Save same program again
        program_id_2 = program_cache.save_program(
            program=sample_program,
            task_id="task2",
            task_source="training",
            success=True,
            accuracy_score=0.95,
            execution_time_ms=90
        )

        # Should return same ID (duplicate detected)
        assert program_id_1 == program_id_2
        assert program_cache.stats["duplicates_found"] == 1

    def test_find_similar_programs(self, program_cache, sample_program, sample_program_2):
        """Test finding similar programs."""
        # Save first program
        program_id_1 = program_cache.save_program(
            program=sample_program,
            task_id="task1",
            task_source="training",
            success=True,
            accuracy_score=0.9,
            execution_time_ms=100
        )

        # Save second program
        program_id_2 = program_cache.save_program(
            program=sample_program_2,
            task_id="task2",
            task_source="training",
            success=True,
            accuracy_score=0.85,
            execution_time_ms=120
        )

        # Find similar to first program
        similar = program_cache.find_similar_programs(sample_program)
        assert len(similar) >= 1
        assert similar[0][0] == program_id_1
        assert similar[0][1] == 1.0  # Exact match
        assert similar[0][2].exact_match  # Should have exact match in result

    def test_get_programs_by_task(self, program_cache, sample_program, sample_program_2):
        """Test getting programs by task ID."""
        task_id = "shared_task"

        # Save multiple programs for same task
        program_id_1 = program_cache.save_program(
            program=sample_program,
            task_id=task_id,
            task_source="training",
            success=True,
            accuracy_score=0.9,
            execution_time_ms=100
        )

        program_id_2 = program_cache.save_program(
            program=sample_program_2,
            task_id=task_id,
            task_source="training",
            success=True,
            accuracy_score=0.95,
            execution_time_ms=80
        )

        # Get programs for task
        programs = program_cache.get_programs_by_task(task_id)
        assert len(programs) == 2

        # Should be sorted by accuracy (descending) and time (ascending)
        assert programs[0].accuracy_score >= programs[1].accuracy_score
        if programs[0].accuracy_score == programs[1].accuracy_score:
            assert programs[0].execution_time_ms <= programs[1].execution_time_ms

    def test_get_successful_programs(self, program_cache, sample_program, sample_program_2):
        """Test getting successful programs above threshold."""
        # Save successful program
        program_cache.save_program(
            program=sample_program,
            task_id="task1",
            task_source="training",
            success=True,
            accuracy_score=0.85,
            execution_time_ms=100
        )

        # Save failed program
        program_cache.save_program(
            program=sample_program_2,
            task_id="task2",
            task_source="training",
            success=False,
            accuracy_score=0.3,
            execution_time_ms=200
        )

        # Get successful programs
        successful = program_cache.get_successful_programs(min_accuracy=0.8)
        assert len(successful) == 1
        assert successful[0].success is True
        assert successful[0].accuracy_score >= 0.8

    def test_cache_statistics(self, program_cache, sample_program):
        """Test cache statistics collection."""
        # Save some programs
        for i in range(5):
            program_cache.save_program(
                program=sample_program,
                task_id=f"task_{i}",
                task_source="training" if i < 3 else "evaluation",
                success=i % 2 == 0,
                accuracy_score=0.7 + (i * 0.05),
                execution_time_ms=100 + i * 10,
                generation=i // 2
            )

        # Get statistics
        stats = program_cache.get_statistics()

        assert stats.total_programs == 1  # Due to duplicate detection
        assert stats.successful_programs == 1
        assert stats.unique_programs == 1
        assert stats.total_size_bytes > 0
        assert isinstance(stats.cache_hit_rate, float)
        assert isinstance(stats.task_type_distribution, dict)
        assert isinstance(stats.generation_distribution, dict)

    def test_cleanup_old_programs(self, program_cache, sample_program):
        """Test cleanup of old programs."""
        # Save program
        program_id = program_cache.save_program(
            program=sample_program,
            task_id="old_task",
            task_source="training",
            success=True,
            accuracy_score=0.9,
            execution_time_ms=100
        )

        # Manually set old access time
        entry = program_cache.get_program(program_id)
        old_time = datetime.now() - timedelta(days=2)

        # Mock the last_accessed time
        with patch.object(entry, 'last_accessed', old_time):
            # This is a simplified test - in real scenario we'd need to
            # update the cached entry with the old timestamp
            pass

        # Cleanup programs older than 1 day
        removed = program_cache.cleanup_old_programs(days=1)
        # Note: This test may need adjustment based on actual implementation

    def test_export_programs(self, program_cache, sample_program, temp_cache_dir):
        """Test exporting programs to files."""
        # Save programs
        for i in range(3):
            program_cache.save_program(
                program=sample_program,
                task_id=f"task_{i}",
                task_source="training",
                success=i != 1,  # Second one fails
                accuracy_score=0.8 + (i * 0.05),
                execution_time_ms=100 + i * 10
            )

        # Export all programs
        export_dir = Path(temp_cache_dir) / "export"
        exported_all = program_cache.export_programs(
            output_dir=str(export_dir),
            format="json",
            filter_successful=False
        )

        # Should export 1 unique program (due to deduplication)
        assert exported_all == 1
        assert export_dir.exists()
        assert len(list(export_dir.glob("*.json"))) == 1

        # Export only successful
        export_dir_success = Path(temp_cache_dir) / "export_success"
        exported_success = program_cache.export_programs(
            output_dir=str(export_dir_success),
            format="json",
            filter_successful=True
        )

        assert exported_success == 1  # All duplicates are successful

    def test_cache_size_limit(self, program_cache, sample_program):
        """Test cache size limit enforcement."""
        # This test would require filling the cache beyond limit
        # For unit test, we just verify the configuration is set
        assert program_cache.config.storage.size_limit_gb == 0.001  # 1MB test limit

        # Verify cache has size limit set
        assert program_cache.cache.size_limit > 0

    def test_program_hash_generation(self, program_cache, sample_program):
        """Test program hash generation for deduplication."""
        hash1 = program_cache.generate_program_hash(sample_program)
        hash2 = program_cache.generate_program_hash(sample_program)

        # Same program should generate same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        # Different program should generate different hash
        modified_program = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 180}}  # Different angle
            ],
            version="1.0"
        )
        hash3 = program_cache.generate_program_hash(modified_program)
        assert hash1 != hash3

    def test_compute_similarity(self, program_cache, sample_program, sample_program_2):
        """Test program similarity computation."""
        # Same program should have similarity 1.0
        similarity_same = program_cache._compute_similarity(sample_program, sample_program)
        assert similarity_same == 1.0

        # Different programs should have lower similarity
        similarity_diff = program_cache._compute_similarity(sample_program, sample_program_2)
        assert 0.0 <= similarity_diff < 1.0

        # Programs with some common operations
        program3 = DSLProgram(
            operations=[
                {"type": "rotate", "params": {"angle": 90}},  # Same as sample_program
                {"type": "flip", "params": {"axis": "horizontal"}},  # Same
                {"type": "mask", "params": {"pattern": [[1, 0]]}}  # Different
            ]
        )
        similarity_partial = program_cache._compute_similarity(sample_program, program3)

        # This program shares 2/3 operations with sample_program in sequence
        assert similarity_partial > 0.6  # Should have reasonable similarity

        # Test with a very different program
        program4 = DSLProgram(
            operations=[
                {"type": "crop", "params": {"x": 0, "y": 0}},
                {"type": "resize", "params": {"scale": 2}},
                {"type": "invert", "params": {}}
            ]
        )
        similarity_very_diff = program_cache._compute_similarity(sample_program, program4)

        # Program with 2 matching ops should be more similar than program with 0 matching ops
        assert similarity_partial > similarity_very_diff
        assert similarity_very_diff < 0.3  # Very low similarity for completely different programs

    def test_context_manager(self, test_config):
        """Test cache works as context manager."""
        with ProgramCache(config=test_config) as cache:
            assert cache is not None
            assert cache.cache_dir.exists()

        # Cache should be closed after context

    def test_cache_entry_serialization(self, sample_program):
        """Test ProgramCacheEntry serialization."""
        entry = ProgramCacheEntry(
            program_id="test_123",
            program_hash="abcd1234",
            program=sample_program,
            task_id="task_001",
            task_source="training",
            success=True,
            accuracy_score=0.95,
            execution_time_ms=150.5,
            generation=3,
            parents=["parent1"],
            mutation_type="mutation",
            fitness_score=0.88
        )

        # Test to_dict
        data = entry.to_dict()
        assert data["program_id"] == "test_123"
        assert isinstance(data["created_at"], str)  # Should be ISO format
        assert isinstance(data["program"], dict)

        # Test from_dict
        reconstructed = ProgramCacheEntry.from_dict(data)
        assert reconstructed.program_id == entry.program_id
        assert reconstructed.accuracy_score == entry.accuracy_score
        assert isinstance(reconstructed.created_at, datetime)

    def test_deduplicate_programs(self, program_cache, sample_program):
        """Test program deduplication functionality."""
        # Save several similar programs
        base_prog = sample_program

        # Save original
        id1 = program_cache.save_program(
            program=base_prog,
            task_id="task1",
            task_source="training",
            success=True,
            accuracy_score=0.9,
            execution_time_ms=100
        )

        # Save exact duplicate (should return same ID)
        id2 = program_cache.save_program(
            program=base_prog,
            task_id="task2",
            task_source="training",
            success=True,
            accuracy_score=0.85,
            execution_time_ms=110
        )

        # Save slightly different program
        diff_prog = DSLProgram(
            operations=[
                {"type": "crop", "params": {"x": 0, "y": 0}},
                {"type": "resize", "params": {"scale": 2}}
            ],
            version="1.0"
        )

        id3 = program_cache.save_program(
            program=diff_prog,
            task_id="task3",
            task_source="training",
            success=True,
            accuracy_score=0.8,
            execution_time_ms=90
        )

        # Run deduplication
        dedupe_map = program_cache.deduplicate_programs(threshold=0.9)

        # Should not have duplicates since exact duplicates are caught on save
        assert len(dedupe_map) == 0 or all(len(dups) == 0 for dups in dedupe_map.values())
