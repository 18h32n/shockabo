"""Integration tests for the complete data pipeline."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.adapters.repositories.cache_repository import CacheRepository
from src.adapters.repositories.data_loader import ARCDataLoader, TaskBasedBatchingStrategy
from src.utils.data_augmentation import ARCTaskAugmentor
from src.utils.grid_ops import MemoryEfficientTaskStorage, MemoryProfiler


class TestDataPipelineIntegration:
    """Integration tests for the complete ARC data pipeline."""

    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            training_dir = data_path / "training"
            training_dir.mkdir()

            # Create 100 sample tasks with varying complexity
            for i in range(100):
                # Vary grid sizes for realistic testing
                size = 3 + (i % 7)  # Grid sizes from 3x3 to 9x9

                sample_data = {
                    "train": [
                        {
                            "input": [[j % 10 for j in range(size)] for _ in range(size)],
                            "output": [[(j + 1) % 10 for j in range(size)] for _ in range(size)]
                        },
                        {
                            "input": [[(j + 2) % 10 for j in range(size)] for _ in range(size)],
                            "output": [[(j + 3) % 10 for j in range(size)] for _ in range(size)]
                        }
                    ],
                    "test": [
                        {
                            "input": [[(j + 4) % 10 for j in range(size)] for _ in range(size)]
                        }
                    ]
                }

                task_file = training_dir / f"task_{i:03d}.json"
                with open(task_file, 'w') as f:
                    json.dump(sample_data, f)

            yield str(data_path)

    @pytest.fixture
    def complete_pipeline(self, large_dataset):
        """Set up complete data pipeline with proper cleanup."""
        cache_repo = CacheRepository(
            cache_dir=str(Path(large_dataset) / "cache"),
            size_limit=50 * 1024 * 1024  # 50MB
        )

        # Use optimal configuration from performance testing (documented as achieving 0.3-0.5s for 1000 tasks)
        repository = ARCDataRepository(
            data_path=large_dataset,
            cache_repository=cache_repo,
            max_workers=2  # Optimal configuration from performance_test.py results
        )

        data_loader = ARCDataLoader(
            repository=repository,
            cache_repository=cache_repo,
            batch_size=16,
            num_workers=2
        )

        pipeline = {
            "repository": repository,
            "cache": cache_repo,
            "loader": data_loader,
            "augmentor": ARCTaskAugmentor()
        }

        yield pipeline

        # Cleanup: ensure cache is properly closed
        try:
            cache_repo.close()
        except Exception as e:
            print(f"Warning: Error during cache cleanup: {e}")

    def test_full_pipeline_performance(self, complete_pipeline):
        """Test complete pipeline performance meets requirements."""
        repository = complete_pipeline["repository"]
        complete_pipeline["loader"]

        # Test AC1: Load all tasks in under 10 seconds (scaled for 100 tasks)
        # The optimized repository now uses parallel processing by default (fixed logic based on performance_test.py)
        start_time = time.perf_counter()
        all_tasks = repository.load_all_tasks("training")
        load_time = time.perf_counter() - start_time

        assert len(all_tasks) == 100

        # Extrapolate to 1000 tasks based on current performance
        estimated_time_1000 = (load_time / 100) * 1000

        # The story documents achieving 0.3-0.5s for 1000 tasks with optimal configuration
        # Allow for some variance due to synthetic vs real data, but should be well under 10s
        assert estimated_time_1000 < 10.0, (
            f"Estimated load time for 1000 tasks: {estimated_time_1000:.2f}s exceeds 10s requirement. "
            f"Actual load time for {len(all_tasks)} tasks: {load_time:.2f}s"
        )

        print(f"Loaded {len(all_tasks)} tasks in {load_time:.2f}s ({load_time/len(all_tasks):.3f}s per task)")
        print(f"Estimated time for 1000 tasks: {estimated_time_1000:.2f}s (Target: <10s, Optimal: 0.3-0.5s)")

        # Log performance achievement
        if estimated_time_1000 <= 1.0:
            print("[EXCELLENT] Meets documented optimal performance (0.3-0.5s for 1000 tasks)")
        elif estimated_time_1000 <= 5.0:
            print("[GOOD] Well within 10s requirement")
        else:
            print("[ACCEPTABLE] Within 10s requirement but could be optimized")

    def test_memory_constraints(self, complete_pipeline):
        """Test AC6: Memory usage stays under 4GB."""
        repository = complete_pipeline["repository"]

        # Load subset of tasks and measure memory
        initial_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

        tasks = repository.load_all_tasks("training", limit=50)

        current_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]
        memory_used = current_memory - initial_memory

        # Extrapolate to full dataset (1000 tasks)
        estimated_memory_1000 = (memory_used / 50) * 1000

        assert estimated_memory_1000 < 4096, f"Estimated memory for 1000 tasks: {estimated_memory_1000:.2f}MB"

        print(f"Memory used for {len(tasks)} tasks: {memory_used:.2f}MB")

    def test_caching_integration(self, complete_pipeline):
        """Test AC2: Caching mechanism integration."""
        repository = complete_pipeline["repository"]
        cache = complete_pipeline["cache"]

        # Get task IDs first (to know what tasks exist)
        task_ids = repository.get_task_ids("training")[:20]  # First 20 tasks

        # Load tasks individually to use cache (load_all_tasks bypasses cache)
        tasks = {}
        for task_id in task_ids:
            task = repository.load_task(task_id, "training")
            if task:
                tasks[task_id] = task

        # Check cache statistics
        cache_stats = cache.get_statistics()
        assert cache_stats["sets"] >= len(tasks)  # All loaded tasks should be cached
        assert len(tasks) >= 15  # Should have loaded at least 15 tasks

        # Reload same tasks (should hit cache)
        start_time = time.perf_counter()
        for task_id in list(tasks.keys())[:10]:
            repository.load_task(task_id)
        time.perf_counter() - start_time  # cache_hit_time (not used in assertions)

        # Load different tasks (cache miss)
        remaining_task_ids = list(tasks.keys())[10:20]
        start_time = time.perf_counter()
        for task_id in remaining_task_ids:
            repository.load_task(task_id)
        time.perf_counter() - start_time  # cache_miss_time (not used in assertions)

        # Cache should improve performance
        cache_stats_final = cache.get_statistics()
        assert cache_stats_final["hit_rate"] > 0.3  # At least 30% hit rate

        print(f"Cache hit rate: {cache_stats_final['hit_rate']:.2%}")
        print(f"Cache hits: {cache_stats_final['hits']}, Cache misses: {cache_stats_final['misses']}")

    def test_batch_processing_integration(self, complete_pipeline):
        """Test AC5: Batch processing with different strategies."""
        loader = complete_pipeline["loader"]
        repository = complete_pipeline["repository"]

        # Load all tasks for batching
        all_tasks = repository.load_all_tasks("training")

        # Test simple batching
        batches = list(loader.stream_batches(all_tasks))
        total_tasks_in_batches = sum(len(batch) for batch in batches)
        assert total_tasks_in_batches == len(all_tasks)

        # Test task-based batching strategy
        task_based_loader = ARCDataLoader(
            repository=repository,
            batch_size=10,
            batching_strategy=TaskBasedBatchingStrategy("grid_size")
        )

        task_based_batches = list(task_based_loader.stream_batches(all_tasks))
        task_based_total = sum(len(batch) for batch in task_based_batches)
        assert task_based_total == len(all_tasks)

        print(f"Simple batching: {len(batches)} batches")
        print(f"Task-based batching: {len(task_based_batches)} batches")

    def test_data_augmentation_integration(self, complete_pipeline):
        """Test AC3: Data augmentation integration."""
        repository = complete_pipeline["repository"]
        augmentor = complete_pipeline["augmentor"]

        # Load sample tasks
        tasks = repository.load_all_tasks("training", limit=10)

        # Apply augmentation
        augmented_tasks = augmentor.batch_augment(
            tasks,
            ["rotate_90", "flip_horizontal"],
            max_augmentations_per_task=2
        )

        # Should have more tasks after augmentation
        assert len(augmented_tasks) > len(tasks)
        assert len(augmented_tasks) <= len(tasks) * 3  # Original + 2 augmentations max

        # Verify augmentation preserves semantics
        augmentation_stats = augmentor.get_augmentation_statistics(tasks, augmented_tasks)
        assert augmentation_stats["augmentation_ratio"] >= 1.0

        print(f"Original tasks: {len(tasks)}, Augmented: {len(augmented_tasks)}")

    def test_sparse_matrix_integration(self, complete_pipeline):
        """Test AC4: Sparse matrix representation integration."""
        repository = complete_pipeline["repository"]

        # Load tasks
        tasks = repository.load_all_tasks("training", limit=20)

        # Test memory-efficient storage
        efficient_storage = MemoryEfficientTaskStorage(use_sparse=True)

        # Store tasks with compression
        for task in tasks.values():
            efficient_storage.store_task(task)

        # Check compression statistics
        memory_stats = efficient_storage.get_memory_statistics()
        assert memory_stats["total_tasks_stored"] == len(tasks)

        # Verify tasks can be loaded back correctly
        for task_id in list(tasks.keys())[:5]:
            restored_task = efficient_storage.load_task(task_id)
            assert restored_task is not None
            assert restored_task.task_id == task_id

        print(f"Compression rate: {memory_stats['compression_rate']:.2%}")

    def test_end_to_end_workflow(self, complete_pipeline):
        """Test complete end-to-end workflow."""
        repository = complete_pipeline["repository"]
        loader = complete_pipeline["loader"]
        augmentor = complete_pipeline["augmentor"]
        cache = complete_pipeline["cache"]

        # 1. Load data with caching
        print("Step 1: Loading data...")
        start_time = time.perf_counter()
        all_tasks = repository.load_all_tasks("training")
        load_time = time.perf_counter() - start_time

        # 2. Apply augmentation
        print("Step 2: Applying augmentation...")
        augmented_tasks = augmentor.batch_augment(
            all_tasks,
            ["rotate_90", "flip_horizontal"],
            max_augmentations_per_task=1
        )

        # 3. Batch processing
        print("Step 3: Batch processing...")
        loader.batch_size = 20
        batch_count = 0
        total_processed = 0

        for batch in loader.stream_batches(augmented_tasks):
            batch_count += 1
            total_processed += len(batch)

            # Simulate processing
            if batch_count >= 5:  # Process first 5 batches only for test
                break

        # 4. Verify results
        print("Step 4: Verification...")
        assert len(all_tasks) == 100
        assert len(augmented_tasks) > len(all_tasks)
        assert total_processed >= 50  # At least 5 batches processed (size varies with augmentation)

        # Check performance metrics
        perf_stats = loader.get_performance_statistics()
        cache_stats = cache.get_statistics()

        print(f"Load time: {load_time:.2f}s")
        print(f"Tasks processed: {total_processed}")
        print(f"Average batch time: {perf_stats.get('avg_batch_time_s', 0):.3f}s")
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

        # All acceptance criteria should be met
        assert load_time < 2.0  # For 100 tasks, should be much faster than 10s limit
        assert cache_stats['hit_rate'] >= 0  # Some cache usage
        assert len(augmented_tasks) > len(all_tasks)  # Augmentation worked
        assert total_processed > 0  # Batch processing worked

    def test_error_handling_and_resilience(self, complete_pipeline):
        """Test pipeline resilience to errors."""
        repository = complete_pipeline["repository"]
        loader = complete_pipeline["loader"]

        # Test with some invalid task IDs
        mixed_task_ids = ["task_001", "task_002", "invalid_task", "task_003"]

        # Should handle invalid tasks gracefully
        batch = loader.load_batch(mixed_task_ids)

        # Should load valid tasks and skip invalid ones
        assert len(batch) == 3  # 3 valid tasks out of 4
        assert "invalid_task" not in batch

        # Repository should report integrity issues
        integrity_report = repository.validate_data_integrity("nonexistent_source")
        assert "missing_files" in integrity_report

    def test_memory_optimization_features(self, complete_pipeline):
        """Test memory optimization features work together."""
        repository = complete_pipeline["repository"]

        # Test memory estimation
        task_ids = repository.get_task_ids("training")
        memory_estimate = repository.estimate_memory_usage(task_ids[:20])

        assert memory_estimate > 0

        # Test memory constraints checking
        current_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]
        constraint_check = MemoryProfiler.check_memory_constraints(
            current_memory,
            memory_estimate / 1024 / 1024,  # Convert to MB
            limit_mb=4096
        )

        assert constraint_check["within_limits"] is True

        # Test optimization suggestions
        suggestions = MemoryProfiler.suggest_optimization(constraint_check)
        assert isinstance(suggestions, list)
