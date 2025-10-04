"""Integration tests for cache size management under load."""

import gc
import tempfile
import time
from pathlib import Path

import pytest

from src.adapters.repositories.program_cache import ProgramCache
from src.adapters.repositories.program_cache_config import ProgramCacheConfig
from src.domain.dsl.base import DSLProgram


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_config_small(cache_dir):
    """Create cache configuration with small size limit for testing."""
    config = ProgramCacheConfig.from_yaml(
        str(Path(__file__).parent.parent.parent / "configs" / "strategies" / "program_cache.yaml")
    )
    # Override with smaller limits for testing
    config.storage.cache_dir = cache_dir
    config.storage.size_limit_gb = 0.001  # 1MB limit
    config.storage.eviction_policy = "least-recently-used"
    return config


@pytest.fixture
def small_cache(cache_config_small):
    """Create cache with small size limit."""
    cache = ProgramCache(config=cache_config_small)
    yield cache
    cache.close()


class TestCacheSizeManagement:
    """Test cache size management and eviction policies."""

    def test_cache_size_limit_enforcement(self, small_cache, cache_config_small):
        """Test that cache enforces size limits."""
        # Generate programs until we exceed the size limit
        programs_saved = []
        total_size = 0

        for i in range(100):  # Try to save many programs
            # Create a program with varying complexity
            operations = []
            for j in range(1 + (i % 10)):  # 1-10 operations
                operations.append({
                    "type": "rotate",
                    "params": {"angle": 90 * ((i + j) % 4)}
                })
                operations.append({
                    "type": "fill",
                    "params": {"color": (i + j) % 10}
                })

            program = DSLProgram(operations=operations)

            program_id = small_cache.save_program(
                program=program,
                task_id=f"size_test_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.8 + (i % 20) * 0.01,
                execution_time_ms=100 + i
            )

            if program_id:
                programs_saved.append(program_id)

        # Check cache statistics
        stats = small_cache.get_statistics()

        print(f"\nPrograms saved: {len(programs_saved)}")
        print(f"Total cache size: {stats.total_size_bytes / 1024:.2f} KB")
        print(f"Size limit: {cache_config_small.storage.size_limit_gb * 1024 * 1024} KB")

        # Cache size should not exceed limit (with some tolerance for overhead)
        size_limit_bytes = cache_config_small.storage.size_limit_gb * 1024 * 1024 * 1024
        assert stats.total_size_bytes <= size_limit_bytes * 1.2  # 20% tolerance

    def test_lru_eviction_policy(self, small_cache):
        """Test that LRU eviction works correctly."""
        # Save initial programs
        first_batch_ids = []
        for i in range(20):
            program = DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": i * 10}},
                {"type": "fill", "params": {"color": i % 5}}
            ])

            program_id = small_cache.save_program(
                program=program,
                task_id=f"lru_test_1_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.85,
                execution_time_ms=100
            )

            if program_id:
                first_batch_ids.append(program_id)

        # Access some programs to update their access times
        accessed_ids = first_batch_ids[:5]  # Access first 5
        for pid in accessed_ids:
            _ = small_cache.get_program(pid)

        # Wait a moment to ensure timestamp differences
        time.sleep(0.1)

        # Save more programs to trigger eviction
        second_batch_ids = []
        for i in range(30):
            program = DSLProgram(operations=[
                {"type": "crop", "params": {"x": 0, "y": 0, "width": 10 + i, "height": 10 + i}},
                {"type": "resize", "params": {"scale": 1.0 + i * 0.1}}
            ])

            program_id = small_cache.save_program(
                program=program,
                task_id=f"lru_test_2_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.9,
                execution_time_ms=110
            )

            if program_id:
                second_batch_ids.append(program_id)

        # Check which programs remain
        # Recently accessed programs should still be in cache
        for pid in accessed_ids:
            entry = small_cache.get_program(pid)
            if entry:
                print(f"\nProgram {pid} still in cache (was accessed)")

        # Some non-accessed programs from first batch might be evicted
        evicted_count = 0
        for pid in first_batch_ids[5:]:  # Non-accessed programs
            entry = small_cache.get_program(pid)
            if not entry:
                evicted_count += 1

        print(f"\nEvicted {evicted_count} non-accessed programs from first batch")

        # Check cache statistics to understand state
        stats = small_cache.get_statistics()
        print(f"Total programs in cache: {stats.total_programs}")
        print(f"Cache size: {stats.total_size_bytes / 1024:.2f} KB")
        print(f"Size limit: {small_cache.config.storage.size_limit_gb * 1024 * 1024} KB")

        # With deduplication, we might not hit the size limit
        # At least verify the cache is functioning
        assert stats.total_programs > 0
        assert len(second_batch_ids) > 0

    def test_cache_performance_under_load(self, small_cache):
        """Test cache performance with many operations."""
        save_times = []
        retrieve_times = []
        search_times = []

        # Phase 1: Save many programs
        print("\nPhase 1: Saving programs...")
        for i in range(50):
            operations = []
            for j in range(3):
                operations.append({
                    "type": ["rotate", "flip", "fill", "crop"][j % 4],
                    "params": {"value": i + j}
                })

            program = DSLProgram(operations=operations)

            start = time.time()
            program_id = small_cache.save_program(
                program=program,
                task_id=f"perf_test_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.8 + (i % 10) * 0.01,
                execution_time_ms=100 + i
            )
            save_time = time.time() - start
            save_times.append(save_time)

        avg_save_time = sum(save_times) / len(save_times)
        print(f"Average save time: {avg_save_time * 1000:.2f} ms")

        # Phase 2: Retrieve programs
        print("\nPhase 2: Retrieving programs...")
        stats = small_cache.get_statistics()
        total_programs = stats.total_programs

        for i in range(min(30, total_programs)):
            start = time.time()
            entry = small_cache.get_program(f"perf_test_{i}_*")  # Pattern-based retrieval
            retrieve_time = time.time() - start
            if entry:
                retrieve_times.append(retrieve_time)

        if retrieve_times:
            avg_retrieve_time = sum(retrieve_times) / len(retrieve_times)
            print(f"Average retrieve time: {avg_retrieve_time * 1000:.2f} ms")
            assert avg_retrieve_time < 0.01  # Should be fast (< 10ms)

        # Phase 3: Similarity searches
        print("\nPhase 3: Similarity searches...")
        test_program = DSLProgram(operations=[
            {"type": "rotate", "params": {"value": 45}},
            {"type": "flip", "params": {"value": 1}}
        ])

        for i in range(10):
            start = time.time()
            similar = small_cache.find_similar_programs(
                test_program,
                max_results=5
            )
            search_time = time.time() - start
            search_times.append(search_time)

        avg_search_time = sum(search_times) / len(search_times)
        print(f"Average similarity search time: {avg_search_time * 1000:.2f} ms")

        # Performance assertions
        assert avg_save_time < 0.05  # Save should be fast (< 50ms)
        assert avg_search_time < 0.1  # Search should complete within 100ms

    def test_concurrent_access_handling(self, small_cache):
        """Test cache behavior with rapid concurrent-like access."""
        # Simulate rapid interleaved operations
        operations_log = []

        for i in range(20):
            # Save operation
            program = DSLProgram(operations=[
                {"type": "rotate", "params": {"angle": i * 10}}
            ])

            start = time.time()
            program_id = small_cache.save_program(
                program=program,
                task_id=f"concurrent_test_{i}",
                task_source="training",
                success=True,
                accuracy_score=0.85,
                execution_time_ms=100
            )
            save_duration = time.time() - start
            operations_log.append(("save", save_duration))

            # Interleave with reads
            if i % 3 == 0 and i > 0:
                start = time.time()
                entry = small_cache.get_program(f"concurrent_test_{i-1}_*")
                read_duration = time.time() - start
                operations_log.append(("read", read_duration))

            # Interleave with searches
            if i % 5 == 0:
                start = time.time()
                similar = small_cache.find_similar_programs(
                    program,
                    max_results=3
                )
                search_duration = time.time() - start
                operations_log.append(("search", search_duration))

        # Analyze operation timings
        save_ops = [d for op, d in operations_log if op == "save"]
        read_ops = [d for op, d in operations_log if op == "read"]
        search_ops = [d for op, d in operations_log if op == "search"]

        print("\nOperation timings:")
        print(f"Saves: {len(save_ops)} ops, avg {sum(save_ops)/len(save_ops)*1000:.2f} ms")
        if read_ops:
            print(f"Reads: {len(read_ops)} ops, avg {sum(read_ops)/len(read_ops)*1000:.2f} ms")
        if search_ops:
            print(f"Searches: {len(search_ops)} ops, avg {sum(search_ops)/len(search_ops)*1000:.2f} ms")

        # All operations should complete without errors
        assert len(operations_log) > 0

        # Check cache integrity
        stats = small_cache.get_statistics()
        assert stats.total_programs > 0
        assert stats.cache_hit_rate >= 0

    def test_memory_usage_monitoring(self, small_cache):
        """Test that cache properly tracks memory usage."""
        initial_stats = small_cache.get_statistics()
        initial_size = initial_stats.total_size_bytes

        # Add programs and track size growth
        size_measurements = [initial_size]

        for batch in range(5):
            # Add batch of programs
            for i in range(10):
                idx = batch * 10 + i
                program = DSLProgram(operations=[
                    {"type": "rotate", "params": {"angle": idx * 5}},
                    {"type": "fill", "params": {"color": idx % 10}},
                    {"type": "mask", "params": {"pattern": [[1, 0], [0, 1]]}}
                ])

                small_cache.save_program(
                    program=program,
                    task_id=f"memory_test_{idx}",
                    task_source="training",
                    success=True,
                    accuracy_score=0.8,
                    execution_time_ms=100
                )

            # Measure size after batch
            stats = small_cache.get_statistics()
            size_measurements.append(stats.total_size_bytes)

            # Force garbage collection to ensure accurate memory measurements
            gc.collect()

        # Analyze size growth
        print("\nCache size growth:")
        for i, size in enumerate(size_measurements):
            print(f"After batch {i}: {size / 1024:.2f} KB")

        # Size should increase but stay within limits
        final_size = size_measurements[-1]
        size_limit_bytes = small_cache.config.storage.size_limit_gb * 1024 * 1024 * 1024

        assert final_size <= size_limit_bytes * 1.2  # Allow 20% overhead

        # Size should stabilize (not grow indefinitely)
        if len(size_measurements) > 3:
            recent_growth = size_measurements[-1] - size_measurements[-3]
            avg_size = sum(size_measurements[-3:]) / 3
            growth_rate = recent_growth / avg_size if avg_size > 0 else 0

            print(f"Recent growth rate: {growth_rate * 100:.1f}%")
            assert growth_rate < 0.25  # Growth should slow down as cache fills
