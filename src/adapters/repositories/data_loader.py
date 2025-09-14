"""DataLoader for batch processing with configurable batching strategies."""

import random
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from ...domain.models import ARCTask
from ...utils.error_handling import (
    ARCBaseException,
    DataNotFoundException,
    ErrorCode,
    ErrorContext,
    ErrorRecovery,
    ErrorSeverity,
)
from ...utils.grid_ops import MemoryProfiler
from .arc_data_repository import ARCDataRepository
from .cache_repository import CacheRepository

logger = structlog.get_logger(__name__)


class BatchingStrategy:
    """Base class for batching strategies."""

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        """Create batches of task IDs."""
        raise NotImplementedError("Batching strategy must implement create_batches method")


class SimpleBatchingStrategy(BatchingStrategy):
    """Simple sequential batching strategy."""

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        """Create simple sequential batches."""
        if not tasks:
            logger.warning("create_batches_empty_tasks", strategy="simple")
            return []

        if batch_size <= 0:
            raise ARCBaseException(
                message=f"Invalid batch size: {batch_size}",
                error_code=ErrorCode.VALIDATION_ERROR,
                suggestions=["Batch size must be a positive integer"]
            )

        task_ids = list(tasks.keys())
        batches = [task_ids[i:i + batch_size] for i in range(0, len(task_ids), batch_size)]

        logger.info(
            "batches_created",
            strategy="simple",
            total_tasks=len(task_ids),
            batch_size=batch_size,
            num_batches=len(batches)
        )

        return batches


class TaskBasedBatchingStrategy(BatchingStrategy):
    """Batching strategy based on task characteristics."""

    def __init__(self, grouping_key: str = "difficulty_level"):
        """Initialize with grouping key."""
        self.grouping_key = grouping_key

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        """Create batches grouped by task characteristics."""
        if not tasks:
            logger.warning("create_batches_empty_tasks", strategy="task_based")
            return []

        if batch_size <= 0:
            raise ARCBaseException(
                message=f"Invalid batch size: {batch_size}",
                error_code=ErrorCode.VALIDATION_ERROR,
                suggestions=["Batch size must be a positive integer"]
            )

        try:
            # Group tasks by characteristic
            groups = defaultdict(list)

            for task_id, task in tasks.items():
                try:
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
                except Exception as e:
                    logger.warning(
                        "task_grouping_error",
                        task_id=task_id,
                        grouping_key=self.grouping_key,
                        error=str(e)
                    )
                    groups["error"].append(task_id)

            # Create batches within each group
            batches = []
            for group_name, group_tasks in groups.items():
                group_batches = [group_tasks[i:i + batch_size]
                               for i in range(0, len(group_tasks), batch_size)]
                batches.extend(group_batches)

                logger.debug(
                    "group_batched",
                    group_name=group_name,
                    group_size=len(group_tasks),
                    group_batches=len(group_batches)
                )

            logger.info(
                "batches_created",
                strategy="task_based",
                grouping_key=self.grouping_key,
                total_tasks=len(tasks),
                num_groups=len(groups),
                num_batches=len(batches)
            )

            return batches

        except Exception as e:
            raise ARCBaseException(
                message=f"Failed to create task-based batches: {str(e)}",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Check task data integrity",
                    "Verify grouping key is valid",
                    "Try using SimpleBatchingStrategy as fallback"
                ]
            ) from e


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
        current_batch: list[str] = []
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
        self.processing_history: list[dict[str, Any]] = []

    def create_batches(self, tasks: dict[str, ARCTask], batch_size: int) -> list[list[str]]:
        # Estimate memory usage per task
        task_memory_estimates = {}
        for task_id, task in tasks.items():
            memory_estimate = task.get_memory_usage_estimate() / 1024 / 1024  # Convert to MB
            task_memory_estimates[task_id] = memory_estimate

        # Adjust batch size based on memory constraints
        avg_task_memory = sum(task_memory_estimates.values()) / len(task_memory_estimates)
        memory_constrained_batch_size: int = max(1, int(self.memory_limit_mb / avg_task_memory))

        # Use the more conservative batch size
        effective_batch_size = min(batch_size, memory_constrained_batch_size)

        # Create batches with memory awareness
        batches = []
        current_batch: list[str] = []
        current_memory: float = 0.0

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
        self.batch_times: list[float] = []
        self.memory_usage: list[float] = []
        self.current_epoch = 0

    def load_batch(self, task_ids: list[str], task_source: str = "training") -> dict[str, ARCTask]:
        """Load a batch of tasks."""
        start_time = time.perf_counter()
        start_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

        batch_tasks = {}

        failed_tasks = []

        @ErrorRecovery.with_retry(
            max_retries=3,
            delay=0.5,
            retryable_exceptions=(ARCBaseException, IOError)
        )
        def load_single_task(task_id: str) -> tuple[str, ARCTask | None]:
            """Load a single task with retry logic."""
            try:
                task = self.repository.load_task(task_id, task_source)
                if task is None:
                    raise DataNotFoundException("task", task_id)
                return task_id, task
            except Exception as e:
                logger.warning(
                    "task_load_failed",
                    task_id=task_id,
                    error=str(e),
                    attempt="retry"
                )
                raise

        if self.num_workers == 1:
            # Sequential loading with error handling
            for task_id in task_ids:
                try:
                    task_id_result, task = load_single_task(task_id)
                    if task:
                        batch_tasks[task_id_result] = task
                except Exception as e:
                    failed_tasks.append((task_id, str(e)))
                    logger.error(
                        "sequential_task_load_failed",
                        task_id=task_id,
                        error=str(e),
                        context=ErrorContext(task_id=task_id)
                    )
        else:
            # Parallel loading with error handling
            try:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_task_id = {
                        executor.submit(load_single_task, task_id): task_id
                        for task_id in task_ids
                    }

                    for future in future_to_task_id:
                        task_id = future_to_task_id[future]
                        try:
                            task_id_result, task = future.result(timeout=30)  # 30s timeout
                            if task:
                                batch_tasks[task_id_result] = task
                        except Exception as e:
                            failed_tasks.append((task_id, str(e)))
                            logger.error(
                                "parallel_task_load_failed",
                                task_id=task_id,
                                error=str(e),
                                context=ErrorContext(task_id=task_id)
                            )
            except Exception as e:
                raise ARCBaseException(
                    message=f"Parallel task loading failed: {str(e)}",
                    error_code=ErrorCode.DATA_LOADING_ERROR,
                    severity=ErrorSeverity.HIGH,
                    suggestions=[
                        "Try reducing num_workers",
                        "Check system resources",
                        "Use sequential loading as fallback"
                    ]
                ) from e

        # Log batch loading results
        if failed_tasks:
            logger.warning(
                "batch_partial_failure",
                requested_tasks=len(task_ids),
                loaded_tasks=len(batch_tasks),
                failed_tasks=len(failed_tasks),
                failure_details=failed_tasks[:5]  # Log first 5 failures
            )

        if not batch_tasks:
            raise ARCBaseException(
                message=f"Failed to load any tasks from batch of {len(task_ids)} tasks",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Check data source availability",
                    "Verify task IDs are correct",
                    "Check repository configuration"
                ]
            )

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
        try:
            all_task_ids = self.repository.get_task_ids(task_source)

            if not all_task_ids:
                logger.warning(
                    "no_tasks_found",
                    task_source=task_source,
                    limit=limit
                )
                return

            if limit:
                all_task_ids = all_task_ids[:limit]

            if self.shuffle:
                random.shuffle(all_task_ids)

            # Create simple batches since we don't have actual task objects yet
            batches = [all_task_ids[i:i + self.batch_size] for i in range(0, len(all_task_ids), self.batch_size)]

            logger.info(
                "streaming_tasks_started",
                task_source=task_source,
                total_tasks=len(all_task_ids),
                num_batches=len(batches),
                batch_size=self.batch_size,
                limit=limit
            )

            for batch_idx, batch_task_ids in enumerate(batches):
                try:
                    batch_tasks = self.load_batch(batch_task_ids, task_source)
                    logger.debug(
                        "batch_streamed",
                        batch_idx=batch_idx,
                        batch_size=len(batch_tasks),
                        requested_size=len(batch_task_ids)
                    )
                    yield batch_tasks
                except Exception as e:
                    logger.error(
                        "batch_streaming_failed",
                        batch_idx=batch_idx,
                        batch_task_ids=batch_task_ids,
                        error=str(e)
                    )
                    # Continue with next batch instead of failing completely
                    continue

        except Exception as e:
            raise ARCBaseException(
                message=f"Failed to stream tasks from {task_source}: {str(e)}",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(additional_data={"task_source": task_source, "limit": limit}),
                suggestions=[
                    "Check repository connectivity",
                    "Verify task source exists",
                    "Try reducing batch size"
                ]
            ) from e

    def get_data_statistics(self, tasks: dict[str, ARCTask]) -> dict[str, Any]:
        """Get statistics about the dataset with error handling."""
        if not tasks:
            logger.warning("get_data_statistics_empty_tasks")
            return {"error": "No tasks provided", "total_tasks": 0}

        try:
            # Basic statistics
            total_tasks = len(tasks)
            total_train_examples = 0
            corrupted_tasks = []

            for task_id, task in tasks.items():
                try:
                    total_train_examples += len(task.train_examples)
                except Exception as e:
                    corrupted_tasks.append((task_id, str(e)))
                    logger.warning(
                        "task_statistics_error",
                        task_id=task_id,
                        error=str(e)
                    )

            # Memory analysis with error handling
            memory_analysis = {}
            try:
                memory_analysis = MemoryProfiler.estimate_task_memory_usage(tasks)
            except Exception as e:
                logger.error("memory_analysis_failed", error=str(e))
                memory_analysis = {"error": f"Memory analysis failed: {str(e)}"}

            # Dimension analysis with error handling
            input_dimensions = []
            output_dimensions = []
            dimension_errors = []

            for task_id, task in tasks.items():
                try:
                    dims = task.get_grid_dimensions()
                    input_dimensions.extend(dims.get("train_input", []))
                    output_dimensions.extend(dims.get("train_output", []))
                except Exception as e:
                    dimension_errors.append((task_id, str(e)))

            # Difficulty distribution with error handling
            difficulty_counts: dict[str, int] = defaultdict(int)
            difficulty_errors = []

            for task_id, task in tasks.items():
                try:
                    difficulty_counts[task.difficulty_level] += 1
                except Exception as e:
                    difficulty_errors.append((task_id, str(e)))
                    difficulty_counts["unknown"] += 1

            # Build statistics with error information
            stats = {
                "total_tasks": total_tasks,
                "total_train_examples": total_train_examples,
                "avg_examples_per_task": total_train_examples / total_tasks if total_tasks > 0 else 0,
                "memory_analysis": memory_analysis,
                "difficulty_distribution": dict(difficulty_counts),
                "input_dimension_stats": {
                    "unique_dimensions": len(set(input_dimensions)) if input_dimensions else 0,
                    "most_common": max(set(input_dimensions), key=input_dimensions.count) if input_dimensions else None,
                    "size_range": (
                        min(h*w for h,w in input_dimensions),
                        max(h*w for h,w in input_dimensions)
                    ) if input_dimensions else None
                },
                "data_quality": {
                    "corrupted_tasks": len(corrupted_tasks),
                    "dimension_errors": len(dimension_errors),
                    "difficulty_errors": len(difficulty_errors),
                    "data_integrity_score": 1.0 - (len(corrupted_tasks) + len(dimension_errors) + len(difficulty_errors)) / (3 * total_tasks) if total_tasks > 0 else 0
                }
            }

            # Log data quality issues
            if corrupted_tasks or dimension_errors or difficulty_errors:
                logger.warning(
                    "data_quality_issues",
                    corrupted_tasks=len(corrupted_tasks),
                    dimension_errors=len(dimension_errors),
                    difficulty_errors=len(difficulty_errors),
                    integrity_score=stats["data_quality"]["data_integrity_score"]
                )

            return stats

        except Exception as e:
            logger.error("statistics_generation_failed", error=str(e), exc_info=True)
            raise ARCBaseException(
                message=f"Failed to generate data statistics: {str(e)}",
                error_code=ErrorCode.DATA_CORRUPTION,
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "Check task data integrity",
                    "Validate data format",
                    "Try with a smaller subset of tasks"
                ]
            ) from e

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
            memory_usage: float = MemoryProfiler.get_current_memory_usage()["rss_mb"] - start_memory

            results[test_size] = {
                "batch_time": batch_time,
                "memory_usage": memory_usage,
                "tasks_per_second": len(test_batch) / batch_time if batch_time > 0 else 0
            }

            # Clean up
            del test_batch

        # Find optimal batch size
        optimal_size = self.batch_size
        best_score: float = 0.0

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
