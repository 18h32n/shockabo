"""DataLoader for batch processing with configurable batching strategies."""

import random
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...domain.models import ARCTask
from ...utils.grid_ops import MemoryProfiler
from .arc_data_repository import ARCDataRepository
from .cache_repository import CacheRepository


class BatchingStrategy:
    """Base class for batching strategies."""

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        """Create batches of task IDs."""
        raise NotImplementedError


class SimpleBatchingStrategy(BatchingStrategy):
    """Simple sequential batching strategy."""

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        task_ids = list(tasks.keys())
        return [task_ids[i:i + batch_size] for i in range(0, len(task_ids), batch_size)]


class TaskBasedBatchingStrategy(BatchingStrategy):
    """Batching strategy based on task characteristics."""

    def __init__(self, grouping_key: str = "difficulty_level"):
        """Initialize with grouping key."""
        self.grouping_key = grouping_key

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        # Group tasks by characteristic
        groups = defaultdict(list)

        for task_id, task in tasks.items():
            if self.grouping_key == "difficulty_level":
                key = task.difficulty_level
            elif self.grouping_key == "family_id":
                key = task.family_id or "unknown"
            elif self.grouping_key == "grid_size":
                dims = task.get_grid_dimensions()
                if dims["train_input"]:
                    avg_size = sum(h * w for h, w in dims["train_input"]) / len(dims["train_input"])
                    key = "small" if avg_size < 100 else "medium" if avg_size < 400 else "large"
                else:
                    key = "unknown"
            else:
                key = "default"

            groups[key].append(task_id)

        # Create batches within each group
        batches = []
        for group_tasks in groups.values():
            group_batches = [group_tasks[i:i + batch_size]
                           for i in range(0, len(group_tasks), batch_size)]
            batches.extend(group_batches)

        return batches


class ExampleBasedBatchingStrategy(BatchingStrategy):
    """Batching strategy based on individual examples."""

    def __init__(self, max_examples_per_batch: int = 50):
        """Initialize with maximum examples per batch."""
        self.max_examples_per_batch = max_examples_per_batch

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        # Count examples per task
        task_example_counts = {}
        for task_id, task in tasks.items():
            example_count = len(task.train_examples) + (1 if task.test_input else 0)
            task_example_counts[task_id] = example_count

        # Create batches based on example count limits
        batches = []
        current_batch = []
        current_example_count = 0

        for task_id, example_count in task_example_counts.items():
            if (len(current_batch) >= batch_size or
                current_example_count + example_count > self.max_examples_per_batch):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task_id]
                current_example_count = example_count
            else:
                current_batch.append(task_id)
                current_example_count += example_count

        if current_batch:
            batches.append(current_batch)

        return batches


class AdaptiveBatchingStrategy(BatchingStrategy):
    """Adaptive batching based on memory and processing time."""

    def __init__(self, memory_limit_mb: float = 1024, target_batch_time_s: float = 5.0):
        """Initialize with memory and time constraints."""
        self.memory_limit_mb = memory_limit_mb
        self.target_batch_time_s = target_batch_time_s
        self.processing_history = []

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        # Estimate memory usage per task
        task_memory_estimates = {}
        for task_id, task in tasks.items():
            memory_estimate = task.get_memory_usage_estimate() / 1024 / 1024  # Convert to MB
            task_memory_estimates[task_id] = memory_estimate

        # Adjust batch size based on memory constraints
        avg_task_memory = sum(task_memory_estimates.values()) / len(task_memory_estimates)
        memory_constrained_batch_size = max(1, int(self.memory_limit_mb / avg_task_memory))

        # Use the more conservative batch size
        effective_batch_size = min(batch_size, memory_constrained_batch_size)

        # Create batches with memory awareness
        batches = []
        current_batch = []
        current_memory = 0

        for task_id, memory_estimate in task_memory_estimates.items():
            if (len(current_batch) >= effective_batch_size or
                current_memory + memory_estimate > self.memory_limit_mb):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task_id]
                current_memory = memory_estimate
            else:
                current_batch.append(task_id)
                current_memory += memory_estimate

        if current_batch:
            batches.append(current_batch)

        return batches


class ARCDataLoader:
    """High-performance data loader with configurable batching and streaming."""

    def __init__(
        self,
        repository: ARCDataRepository,
        cache_repository: CacheRepository | None = None,
        batch_size: int = 32,
        batching_strategy: BatchingStrategy | None = None,
        shuffle: bool = False,
        num_workers: int = 1,
        prefetch_factor: int = 2
    ):
        """Initialize data loader."""
        self.repository = repository
        self.cache_repository = cache_repository
        self.batch_size = batch_size
        self.batching_strategy = batching_strategy or SimpleBatchingStrategy()
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Performance tracking
        self.batch_times = []
        self.memory_usage = []
        self.current_epoch = 0

    def load_batch(self, task_ids: list[str], task_source: str = "training") -> dict[str, ARCTask]:
        """Load a batch of tasks."""
        start_time = time.perf_counter()
        start_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

        batch_tasks = {}

        if self.num_workers == 1:
            # Sequential loading
            for task_id in task_ids:
                task = self.repository.load_task(task_id, task_source)
                if task:
                    batch_tasks[task_id] = task
        else:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_task_id = {
                    executor.submit(self.repository.load_task, task_id, task_source): task_id
                    for task_id in task_ids
                }

                for future in future_to_task_id:
                    task_id = future_to_task_id[future]
                    try:
                        task = future.result()
                        if task:
                            batch_tasks[task_id] = task
                    except Exception as e:
                        print(f"Error loading task {task_id}: {e}")

        # Record performance metrics
        batch_time = time.perf_counter() - start_time
        end_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

        self.batch_times.append(batch_time)
        self.memory_usage.append(end_memory - start_memory)

        return batch_tasks

    def stream_batches(
        self,
        tasks: dict[str, ARCTask],
        task_source: str = "training"
    ) -> Iterator[dict[str, ARCTask]]:
        """Stream batches of tasks for memory-efficient processing."""
        task_ids = list(tasks.keys())

        if self.shuffle:
            random.shuffle(task_ids)

        # Create batches using strategy
        batches = self.batching_strategy.create_batches(tasks, self.batch_size)

        # Stream batches
        for batch_task_ids in batches:
            yield self.load_batch(batch_task_ids, task_source)

    def stream_all_tasks(
        self,
        task_source: str = "training",
        limit: int | None = None
    ) -> Iterator[dict[str, ARCTask]]:
        """Stream all tasks from repository in batches."""
        all_task_ids = self.repository.get_task_ids(task_source)

        if limit:
            all_task_ids = all_task_ids[:limit]

        if self.shuffle:
            random.shuffle(all_task_ids)

        # Create dummy tasks dict for batching strategy
        dummy_tasks = dict.fromkeys(all_task_ids)
        batches = self.batching_strategy.create_batches(dummy_tasks, self.batch_size)

        for batch_task_ids in batches:
            yield self.load_batch(batch_task_ids, task_source)

    def get_data_statistics(self, tasks: dict[str, ARCTask]) -> dict[str, Any]:
        """Get statistics about the dataset."""
        if not tasks:
            return {}

        # Basic statistics
        total_tasks = len(tasks)
        total_train_examples = sum(len(task.train_examples) for task in tasks.values())

        # Memory analysis
        memory_analysis = MemoryProfiler.estimate_task_memory_usage(tasks)

        # Dimension analysis
        input_dimensions = []
        output_dimensions = []

        for task in tasks.values():
            dims = task.get_grid_dimensions()
            input_dimensions.extend(dims["train_input"])
            output_dimensions.extend(dims["train_output"])

        # Difficulty distribution
        difficulty_counts = defaultdict(int)
        for task in tasks.values():
            difficulty_counts[task.difficulty_level] += 1

        return {
            "total_tasks": total_tasks,
            "total_train_examples": total_train_examples,
            "avg_examples_per_task": total_train_examples / total_tasks if total_tasks > 0 else 0,
            "memory_analysis": memory_analysis,
            "difficulty_distribution": dict(difficulty_counts),
            "input_dimension_stats": {
                "unique_dimensions": len(set(input_dimensions)),
                "most_common": max(set(input_dimensions), key=input_dimensions.count) if input_dimensions else None,
                "size_range": (min(h*w for h,w in input_dimensions), max(h*w for h,w in input_dimensions)) if input_dimensions else None
            }
        }

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get loader performance statistics."""
        if not self.batch_times:
            return {"no_data": True}

        return {
            "total_batches_loaded": len(self.batch_times),
            "avg_batch_time_s": sum(self.batch_times) / len(self.batch_times),
            "min_batch_time_s": min(self.batch_times),
            "max_batch_time_s": max(self.batch_times),
            "total_time_s": sum(self.batch_times),
            "avg_memory_per_batch_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "current_epoch": self.current_epoch,
            "repository_stats": self.repository.get_load_statistics()
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self.batch_times.clear()
        self.memory_usage.clear()
        self.current_epoch = 0

    def optimize_batch_size(
        self,
        tasks: dict[str, ARCTask],
        target_memory_mb: float = 1024,
        target_time_s: float = 5.0
    ) -> int:
        """Automatically optimize batch size based on constraints."""
        # Test different batch sizes
        test_sizes = [1, 4, 8, 16, 32, 64, 128]
        results = {}

        sample_tasks = dict(list(tasks.items())[:min(100, len(tasks))])  # Use sample for testing

        for test_size in test_sizes:
            # Create test batches
            test_batches = self.batching_strategy.create_batches(sample_tasks, test_size)
            if not test_batches:
                continue

            # Test first batch
            start_time = time.perf_counter()
            start_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

            test_batch = self.load_batch(test_batches[0])

            batch_time = time.perf_counter() - start_time
            memory_usage = MemoryProfiler.get_current_memory_usage()["rss_mb"] - start_memory

            results[test_size] = {
                "batch_time": batch_time,
                "memory_usage": memory_usage,
                "tasks_per_second": len(test_batch) / batch_time if batch_time > 0 else 0
            }

            # Clean up
            del test_batch

        # Find optimal batch size
        optimal_size = self.batch_size
        best_score = 0

        for size, metrics in results.items():
            # Score based on meeting constraints and throughput
            memory_score = 1.0 if metrics["memory_usage"] < target_memory_mb else target_memory_mb / metrics["memory_usage"]
            time_score = 1.0 if metrics["batch_time"] < target_time_s else target_time_s / metrics["batch_time"]
            throughput_score = metrics["tasks_per_second"] / 10.0  # Normalize throughput

            combined_score = memory_score * time_score * throughput_score

            if combined_score > best_score:
                best_score = combined_score
                optimal_size = size

        return optimal_size
