"""Unit tests for ARC data repository performance and functionality."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.adapters.repositories.cache_repository import CacheRepository


class TestARCDataRepository:
    """Test suite for ARC data repository."""

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {
                    "input": [[0, 1, 2], [1, 0, 2], [2, 2, 0]]
                }
            ]
        }

    @pytest.fixture
    def temp_data_dir(self, sample_task_data):
        """Create temporary data directory with sample tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create training directory
            training_dir = data_path / "training"
            training_dir.mkdir()

            # Create sample task files
            for i in range(10):
                task_file = training_dir / f"task_{i:03d}.json"
                with open(task_file, 'w') as f:
                    json.dump(sample_task_data, f)

            yield str(data_path)

    @pytest.fixture
    def repository(self, temp_data_dir):
        """ARC data repository instance."""
        return ARCDataRepository(data_path=temp_data_dir)

    @pytest.fixture
    def repository_with_cache(self, temp_data_dir):
        """ARC data repository with cache."""
        cache_repo = CacheRepository(cache_dir=str(Path(temp_data_dir) / "cache"))
        repository = ARCDataRepository(data_path=temp_data_dir, cache_repository=cache_repo)
        yield repository
        # Ensure proper cleanup
        try:
            cache_repo.close()
        except Exception as e:
            print(f"Warning: Error during cache cleanup: {e}")

    def test_load_single_task(self, repository):
        """Test loading a single task."""
        task = repository.load_task("task_001")

        assert task is not None
        assert task.task_id == "task_001"
        assert task.task_source == "training"
        assert len(task.train_examples) == 1
        assert task.train_examples[0]["input"] == [[0, 1], [1, 0]]
        assert task.train_examples[0]["output"] == [[1, 0], [0, 1]]

    def test_load_nonexistent_task(self, repository):
        """Test loading a nonexistent task."""
        task = repository.load_task("nonexistent_task")
        assert task is None

    def test_load_all_tasks_performance(self, repository):
        """Test bulk loading performance meets AC requirement (<10s for 1000 tasks)."""
        start_time = time.perf_counter()
        tasks = repository.load_all_tasks("training")
        load_time = time.perf_counter() - start_time

        assert len(tasks) == 10  # We created 10 sample tasks
        assert load_time < 1.0  # Should be much faster for 10 tasks

        # Extrapolate performance for 1000 tasks
        estimated_time_1000 = (load_time / 10) * 1000
        assert estimated_time_1000 < 10.0, f"Estimated time for 1000 tasks: {estimated_time_1000:.2f}s"

    def test_caching_performance(self, repository_with_cache):
        """Test caching mechanism improves performance."""
        task_id = "task_001"

        # First load (cache miss)
        start_time = time.perf_counter()
        task1 = repository_with_cache.load_task(task_id)
        first_load_time = time.perf_counter() - start_time

        # Second load (cache hit)
        start_time = time.perf_counter()
        task2 = repository_with_cache.load_task(task_id)
        second_load_time = time.perf_counter() - start_time

        assert task1 is not None
        assert task2 is not None
        assert task1.task_id == task2.task_id
        assert second_load_time < first_load_time  # Cache should be faster

        stats = repository_with_cache.get_load_statistics()
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1

    def test_parallel_loading(self, repository):
        """Test parallel loading works correctly."""
        tasks = repository.load_all_tasks("training", limit=5)

        assert len(tasks) == 5
        for task_id, task in tasks.items():
            assert task.task_id == task_id
            assert len(task.train_examples) == 1

    def test_memory_usage_estimation(self, repository):
        """Test memory usage estimation."""
        task_ids = repository.get_task_ids("training")
        memory_estimate = repository.estimate_memory_usage(task_ids[:5])

        assert memory_estimate > 0
        assert isinstance(memory_estimate, int)

    def test_data_integrity_validation(self, repository):
        """Test data integrity validation."""
        issues = repository.validate_data_integrity("training")

        # Should have no issues with our sample data
        assert len(issues["corrupted_tasks"]) == 0
        assert len(issues["validation_errors"]) == 0

    def test_load_statistics(self, repository):
        """Test load statistics tracking."""
        # Load some tasks
        repository.load_task("task_001")
        repository.load_task("task_002")

        stats = repository.get_load_statistics()

        assert stats["total_loaded"] == 2
        assert stats["total_load_time"] > 0
        assert stats["avg_load_time"] > 0

    def test_task_ids_retrieval(self, repository):
        """Test getting list of task IDs."""
        task_ids = repository.get_task_ids("training")

        assert len(task_ids) == 10
        assert "task_001" in task_ids
        assert "task_009" in task_ids

    def test_performance_requirements(self, repository):
        """Test that performance requirements are met."""
        # Test loading 10 tasks (will extrapolate to 1000)
        start_time = time.perf_counter()
        tasks = repository.load_all_tasks("training")
        load_time = time.perf_counter() - start_time

        # Calculate tasks per second
        tasks_per_second = len(tasks) / load_time

        # Should be able to load at least 100 tasks per second to meet 10s requirement for 1000 tasks
        assert tasks_per_second >= 100, f"Loading rate: {tasks_per_second:.2f} tasks/s, need >=100 tasks/s"

    def test_repository_with_max_workers(self, temp_data_dir):
        """Test repository with different worker configurations."""
        # Test with single worker
        repo_single = ARCDataRepository(data_path=temp_data_dir, max_workers=1)
        start_time = time.perf_counter()
        tasks_single = repo_single.load_all_tasks("training")
        time_single = time.perf_counter() - start_time

        # Test with multiple workers
        repo_multi = ARCDataRepository(data_path=temp_data_dir, max_workers=4)
        start_time = time.perf_counter()
        tasks_multi = repo_multi.load_all_tasks("training")
        time_multi = time.perf_counter() - start_time

        assert len(tasks_single) == len(tasks_multi)
        # Multi-worker should be faster or at least not significantly slower
        assert time_multi <= time_single * 1.5  # Allow some overhead for small dataset

    @pytest.mark.parametrize("task_source", ["training", "evaluation"])
    def test_different_task_sources(self, temp_data_dir, sample_task_data, task_source):
        """Test loading from different task sources."""
        # Create source directory
        source_dir = Path(temp_data_dir) / task_source
        source_dir.mkdir(exist_ok=True)

        # Add a test file
        task_file = source_dir / "test_task.json"
        with open(task_file, 'w') as f:
            json.dump(sample_task_data, f)

        repository = ARCDataRepository(data_path=temp_data_dir)
        task = repository.load_task("test_task", task_source)

        assert task is not None
        assert task.task_source == task_source
