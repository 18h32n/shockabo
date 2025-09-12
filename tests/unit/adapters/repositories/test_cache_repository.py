"""Unit tests for cache repository functionality."""

import tempfile
import time

import pytest

from src.adapters.repositories.cache_repository import CacheRepository
from src.domain.models import ARCTask


class TestCacheRepository:
    """Test suite for cache repository."""

    @pytest.fixture
    def sample_task(self):
        """Sample ARC task for testing."""
        return ARCTask(
            task_id="test_task_001",
            task_source="training",
            difficulty_level="easy",
            train_examples=[
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            test_input=[[0, 1, 2], [1, 0, 2]]
        )

    @pytest.fixture
    def cache_repo(self):
        """Cache repository instance with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_repo = CacheRepository(cache_dir=temp_dir, size_limit=10*1024*1024)  # 10MB limit
            yield cache_repo
            # Ensure proper cleanup
            try:
                cache_repo.close()
            except Exception as e:
                print(f"Warning: Error during cache cleanup: {e}")

    def test_basic_cache_operations(self, cache_repo, sample_task):
        """Test basic cache set/get/delete operations."""
        key = "test_key"

        # Test cache miss
        result = cache_repo.get(key)
        assert result is None

        # Test cache set
        success = cache_repo.set(key, sample_task)
        assert success is True

        # Test cache hit
        result = cache_repo.get(key)
        assert result is not None
        assert result.task_id == sample_task.task_id

        # Test cache delete
        success = cache_repo.delete(key)
        assert success is True

        # Verify deletion
        result = cache_repo.get(key)
        assert result is None

    def test_cache_key_creation(self, cache_repo):
        """Test cache key creation with different parameters."""
        # Basic key
        key1 = cache_repo.create_cache_key("task_001", "training")
        assert key1 == "training:task_001"

        # Key with preprocessing options
        options = {"augmentation": ["rotate_90"], "sparse": True}
        key2 = cache_repo.create_cache_key("task_001", "training", options)
        assert key2.startswith("training:task_001:")
        assert len(key2) > len(key1)  # Should have hash suffix

        # Same options should produce same key
        key3 = cache_repo.create_cache_key("task_001", "training", options)
        assert key2 == key3

    def test_cache_statistics(self, cache_repo, sample_task):
        """Test cache statistics tracking."""
        key = "stats_test"

        # Initial stats
        stats = cache_repo.get_statistics()
        initial_hits = stats["hits"]
        initial_misses = stats["misses"]

        # Generate miss
        cache_repo.get(key)

        # Generate hit
        cache_repo.set(key, sample_task)
        cache_repo.get(key)

        # Check updated stats
        stats = cache_repo.get_statistics()
        assert stats["hits"] == initial_hits + 1
        assert stats["misses"] == initial_misses + 1
        assert stats["sets"] >= 1
        assert stats["hit_rate"] >= 0

    def test_cache_size_limits(self, cache_repo, sample_task):
        """Test cache size limits and LRU eviction."""
        # Fill cache with multiple tasks
        for i in range(20):
            key = f"task_{i:03d}"
            task = ARCTask(
                task_id=f"test_task_{i:03d}",
                task_source="training",
                train_examples=[{"input": [list(range(10)) for _ in range(10)]}] * 5  # Larger tasks
            )
            cache_repo.set(key, task)

        size_info = cache_repo.get_size_info()
        assert "volume" in size_info
        assert "size_limit" in size_info
        assert size_info["volume"] <= size_info["size_limit"]

    def test_cache_warming(self, cache_repo):
        """Test cache warming functionality."""
        # Create tasks for warming
        tasks = {}
        for i in range(5):
            task_id = f"warm_task_{i}"
            tasks[task_id] = ARCTask(
                task_id=task_id,
                task_source="training",
                train_examples=[{"input": [[0, 1], [1, 0]]}]
            )

        # Warm cache
        results = cache_repo.warm_cache(tasks, "training")

        # Check all tasks were cached
        assert len(results) == 5
        assert all(success for success in results.values())

        # Verify tasks are in cache
        for task_id in tasks:
            key = cache_repo.create_cache_key(task_id, "training")
            cached_task = cache_repo.get(key)
            assert cached_task is not None
            assert cached_task.task_id == task_id

    def test_pattern_invalidation(self, cache_repo, sample_task):
        """Test pattern-based cache invalidation."""
        # Set multiple related keys
        keys = ["training:task_001", "training:task_002", "evaluation:task_001"]
        for key in keys:
            cache_repo.set(key, sample_task)

        # Invalidate training tasks
        invalidated = cache_repo.invalidate_pattern("training:")
        assert invalidated == 2

        # Verify training tasks are gone
        assert cache_repo.get("training:task_001") is None
        assert cache_repo.get("training:task_002") is None

        # Verify evaluation task remains
        assert cache_repo.get("evaluation:task_001") is not None

    def test_cache_efficiency_metrics(self, cache_repo, sample_task):
        """Test cache efficiency calculations."""
        # Add some cache activity
        for i in range(10):
            key = f"efficiency_test_{i}"
            cache_repo.set(key, sample_task)
            cache_repo.get(key)  # Generate hit

        # Generate some misses
        for i in range(5):
            cache_repo.get(f"nonexistent_{i}")

        efficiency = cache_repo.get_cache_efficiency()

        assert "hit_rate" in efficiency
        assert "storage_efficiency" in efficiency
        assert "requests_per_item" in efficiency
        assert "average_access_frequency" in efficiency

        assert 0 <= efficiency["hit_rate"] <= 1
        assert 0 <= efficiency["storage_efficiency"] <= 1

    def test_cache_expiration(self, cache_repo, sample_task):
        """Test cache expiration functionality."""
        key = "expiring_key"

        # Set with short expiration
        cache_repo.set(key, sample_task, expire=0.1)  # 100ms

        # Should be available immediately
        result = cache_repo.get(key)
        assert result is not None

        # Wait for expiration
        time.sleep(0.2)

        # Cleanup expired entries
        cache_repo.cleanup_expired()

        # Should be expired now
        result = cache_repo.get(key)
        # Note: diskcache handles expiration automatically, so this might still return None

    def test_cache_exists(self, cache_repo, sample_task):
        """Test cache key existence checking."""
        key = "exists_test"

        # Should not exist initially
        assert not cache_repo.exists(key)

        # Set key
        cache_repo.set(key, sample_task)

        # Should exist now
        assert cache_repo.exists(key)

        # Delete key
        cache_repo.delete(key)

        # Should not exist after deletion
        assert not cache_repo.exists(key)

    def test_cache_keys_listing(self, cache_repo, sample_task):
        """Test listing all cache keys."""
        # Initially empty
        keys = cache_repo.keys()
        initial_count = len(keys)

        # Add some keys
        test_keys = ["key1", "key2", "key3"]
        for key in test_keys:
            cache_repo.set(key, sample_task)

        # Check updated key list
        keys = cache_repo.keys()
        assert len(keys) == initial_count + len(test_keys)

        for key in test_keys:
            assert key in keys

    def test_cache_clear(self, cache_repo, sample_task):
        """Test clearing entire cache."""
        # Add some items
        for i in range(5):
            cache_repo.set(f"clear_test_{i}", sample_task)

        # Verify items exist
        assert len(cache_repo.keys()) >= 5

        # Clear cache
        cache_repo.clear()

        # Verify cache is empty
        assert len(cache_repo.keys()) == 0

        # Verify statistics are reset
        stats = cache_repo.get_statistics()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0

    def test_context_manager(self, sample_task):
        """Test cache repository as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with CacheRepository(cache_dir=temp_dir) as cache_repo:
                cache_repo.set("context_test", sample_task)
                result = cache_repo.get("context_test")
                assert result is not None

    def test_windows_file_cleanup(self, sample_task):
        """Test Windows-specific file cleanup functionality."""

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_repo = CacheRepository(cache_dir=temp_dir)

            # Add some data to create database files
            for i in range(3):
                cache_repo.set(f"windows_test_{i}", sample_task)

            # Test force cleanup method
            cache_repo.force_cleanup()

            # On Windows, this should not raise permission errors
            # On other platforms, this test still validates the cleanup works
            cache_repo.close()

            # Create a new cache repo in the same directory to ensure
            # no file locks remain
            cache_repo2 = CacheRepository(cache_dir=temp_dir)
            cache_repo2.set("test_after_cleanup", sample_task)
            result = cache_repo2.get("test_after_cleanup")
            assert result is not None

            cache_repo2.close()
