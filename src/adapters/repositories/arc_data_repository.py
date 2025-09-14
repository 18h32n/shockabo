"""ARC Data Repository implementation with performance optimizations."""

import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import structlog

# Add source root to path for absolute imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Add arc-prize-2025 data directory to path
_arc_data_path = _src_root.parent / "arc-prize-2025" / "data"
if str(_arc_data_path) not in sys.path:
    sys.path.append(str(_arc_data_path))

try:
    from task_loader import ARCTaskLoader
except ImportError:
    # Fallback if task_loader is not available
    ARCTaskLoader = None

try:
    from domain.models import ARCTask
except ImportError:
    # Fallback to relative import
    from ...domain.models import ARCTask

from ...utils.error_handling import (
    ARCBaseException,
    ErrorCode,
    ErrorContext,
    ErrorRecovery,
    ErrorSeverity,
)

logger = structlog.get_logger(__name__)


class ARCDataRepository:
    """High-performance ARC task data repository with caching and bulk loading."""

    def __init__(
        self,
        data_path: str | None = None,
        cache_repository: Any | None = None,
        max_workers: int | None = None,
        use_real_dataset: bool = True
    ):
        """Initialize repository with data path and optional cache."""
        try:
            if data_path is None:
                if use_real_dataset:
                    # Use real ARC Prize 2025 dataset
                    data_path_obj = Path(__file__).parent.parent.parent.parent / "arc-prize-2025"
                else:
                    # Use legacy data structure
                    data_path_obj = Path(__file__).parent.parent.parent.parent / "data" / "arc-agi"
                data_path = str(data_path_obj)

            self.data_path = Path(data_path)

            # Validate data path exists
            if not self.data_path.exists():
                raise ARCBaseException(
                    message=f"Data path does not exist: {data_path}",
                    error_code=ErrorCode.DATA_NOT_FOUND,
                    severity=ErrorSeverity.CRITICAL,
                    suggestions=[
                        "Check that the data directory is correctly configured",
                        "Ensure ARC dataset is downloaded and extracted",
                        "Verify file system permissions"
                    ]
                )

            self.cache_repository = cache_repository
            self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
            self.use_real_dataset = use_real_dataset

            # Initialize task loader if available
            if ARCTaskLoader:
                if use_real_dataset:
                    # For real dataset, we don't need task loader as we load from JSON files directly
                    self.task_loader = None
                else:
                    training_path = self.data_path / "training"
                    if training_path.exists():
                        self.task_loader = ARCTaskLoader(str(training_path))
                    else:
                        logger.warning(
                            "training_path_not_found",
                            path=str(training_path),
                            message="Task loader will not be available for validation"
                        )
                        self.task_loader = None
            else:
                logger.info("task_loader_not_available", message="Using fallback task loading")
                self.task_loader = None

            # Performance metrics
            self.load_stats = {
                "total_loaded": 0,
                "total_load_time": 0.0,
                "cache_hits": 0,
                "cache_misses": 0
            }

            # Load real dataset if using new format
            self._real_data_cache = {}
            if self.use_real_dataset:
                self._load_real_dataset_cache()

        except Exception as e:
            logger.error(
                "repository_initialization_failed",
                error=str(e),
                data_path=data_path,
                use_real_dataset=use_real_dataset
            )
            raise ARCBaseException(
                message=f"Failed to initialize ARC data repository: {e}",
                error_code=ErrorCode.INITIALIZATION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                context=ErrorContext(
                    operation="ARCDataRepository.__init__",
                    data_path=data_path,
                    use_real_dataset=use_real_dataset
                ),
                recovery_action=ErrorRecovery.FAIL_FAST
            ) from e

    def _load_real_dataset_cache(self):
        """Load real ARC dataset files into memory for fast access."""
        try:
            # Load training challenges and solutions
            training_challenges_file = self.data_path / "arc-agi_training_challenges.json"
            training_solutions_file = self.data_path / "arc-agi_training_solutions.json"

            if training_challenges_file.exists() and training_solutions_file.exists():
                with open(training_challenges_file) as f:
                    training_challenges = json.load(f)
                with open(training_solutions_file) as f:
                    training_solutions = json.load(f)

                self._real_data_cache["training"] = {
                    "challenges": training_challenges,
                    "solutions": training_solutions
                }

            # Load evaluation challenges and solutions
            eval_challenges_file = self.data_path / "arc-agi_evaluation_challenges.json"
            eval_solutions_file = self.data_path / "arc-agi_evaluation_solutions.json"

            if eval_challenges_file.exists() and eval_solutions_file.exists():
                with open(eval_challenges_file) as f:
                    eval_challenges = json.load(f)
                with open(eval_solutions_file) as f:
                    eval_solutions = json.load(f)

                self._real_data_cache["evaluation"] = {
                    "challenges": eval_challenges,
                    "solutions": eval_solutions
                }

            # Load test challenges (no solutions available)
            test_challenges_file = self.data_path / "arc-agi_test_challenges.json"
            if test_challenges_file.exists():
                with open(test_challenges_file) as f:
                    test_challenges = json.load(f)

                self._real_data_cache["test"] = {
                    "challenges": test_challenges,
                    "solutions": {}
                }

            print(f"Loaded real ARC dataset: {len(self._real_data_cache)} data sources")

        except Exception as e:
            print(f"Warning: Could not load real dataset cache: {e}")
            self._real_data_cache = {}

    def load_task(self, task_id: str, task_source: str = "training") -> ARCTask | None:
        """Load a single ARC task by ID with optimized performance."""
        start_time = time.perf_counter()

        # Try cache first
        if self.cache_repository:
            cache_key = f"{task_source}:{task_id}"
            cached_task = self.cache_repository.get(cache_key)
            if cached_task:
                self.load_stats["cache_hits"] += 1
                return cached_task
            self.load_stats["cache_misses"] += 1

        try:
            if self.use_real_dataset and task_source in self._real_data_cache:
                # Load from real dataset cache
                challenges = self._real_data_cache[task_source]["challenges"]
                solutions = self._real_data_cache[task_source]["solutions"]

                if task_id not in challenges:
                    return None

                challenge_data = challenges[task_id]
                solution_data = solutions.get(task_id, [])

                # Extract test output from solutions if available
                test_output = None
                if solution_data and len(solution_data) > 0:
                    test_output = solution_data[0]  # First solution

                # Extract test input from challenge
                test_input = []
                if challenge_data.get("test") and len(challenge_data["test"]) > 0:
                    test_input = challenge_data["test"][0].get("input", [])

                # Create ARCTask directly
                arc_task = ARCTask(
                    task_id=task_id,
                    task_source=task_source,
                    train_examples=challenge_data.get("train", []),
                    test_input=test_input,
                    test_output=test_output
                )

            else:
                # Load from legacy file structure
                task_file = f"{task_id}.json"
                source_path = self.data_path / task_source
                task_file_path = source_path / task_file

                if not task_file_path.exists():
                    return None

                # Direct JSON loading without ARCTaskLoader validation overhead
                with open(task_file_path) as f:
                    raw_task = json.load(f)

                # Convert to domain model
                arc_task = ARCTask.from_dict(raw_task, task_id, task_source)

            # Cache the result
            if self.cache_repository:
                cache_key = f"{task_source}:{task_id}"
                self.cache_repository.set(cache_key, arc_task)

            # Update metrics
            load_time = time.perf_counter() - start_time
            self.load_stats["total_loaded"] += 1
            self.load_stats["total_load_time"] += load_time

            return arc_task

        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            return None

    def load_all_tasks(
        self,
        task_source: str = "training",
        limit: int | None = None
    ) -> dict[str, ARCTask]:
        """Load all tasks from specified source with optimized performance."""
        start_time = time.perf_counter()

        if self.use_real_dataset and task_source in self._real_data_cache:
            # Load from real dataset cache
            challenges = self._real_data_cache[task_source]["challenges"]
            solutions = self._real_data_cache[task_source]["solutions"]

            task_ids = list(challenges.keys())
            if limit:
                task_ids = task_ids[:limit]

            print(f"Loading {len(task_ids)} tasks from real {task_source} dataset...")

            tasks = {}
            for task_id in task_ids:
                task = self.load_task(task_id, task_source)
                if task:
                    tasks[task_id] = task
        else:
            # Load from legacy file structure
            source_path = self.data_path / task_source
            if not source_path.exists():
                raise FileNotFoundError(f"Data source path not found: {source_path}")

            # Get all task files
            task_files = list(source_path.glob("*.json"))
            if limit:
                task_files = task_files[:limit]

            print(f"Loading {len(task_files)} tasks from {task_source}...")

            # Use sequential loading for optimal performance
            # ProcessPoolExecutor has significant overhead on Windows for lightweight JSON parsing tasks
            # Sequential loading with optimized JSON parsing is faster for this use case
            tasks = self._load_tasks_sequential(task_files, task_source)

        load_time = time.perf_counter() - start_time
        self.load_stats["total_loaded"] += len(tasks)
        self.load_stats["total_load_time"] += load_time

        if len(tasks) > 0:
            print(f"Loaded {len(tasks)} tasks in {load_time:.2f} seconds ({load_time/len(tasks):.3f}s per task)")
        else:
            print(f"No tasks loaded in {load_time:.2f} seconds")
        return tasks

    def _load_tasks_sequential(self, task_files: list[Path], task_source: str) -> dict[str, ARCTask]:
        """Load tasks sequentially with maximum performance optimization for small-medium datasets."""
        tasks = {}

        for task_file in task_files:
            try:
                # Ultra-fast JSON loading with minimal validation
                with open(task_file, 'rb') as f:
                    raw_data = f.read()

                # Parse JSON directly
                raw_task = json.loads(raw_data)
                task_id = task_file.stem

                # Direct instantiation without validation - maximum speed
                train_examples = raw_task.get("train", [])
                test_examples = raw_task.get("test", [])

                # Minimal processing for test data
                if test_examples:
                    test_input = test_examples[0].get("input", [])
                    test_output = test_examples[0].get("output")
                else:
                    test_input = []
                    test_output = None

                # Direct object creation bypassing from_dict validation
                arc_task = ARCTask(
                    task_id=task_id,
                    task_source=task_source,
                    train_examples=train_examples,
                    test_input=test_input,
                    test_output=test_output
                )
                tasks[task_id] = arc_task

            except Exception as e:
                # Minimal error handling to maintain speed
                print(f"Error loading {task_file.name}: {e}")
                continue

        return tasks

    def _load_tasks_parallel(self, task_files: list[Path], task_source: str) -> dict[str, ARCTask]:
        """Load tasks in parallel for large datasets."""
        tasks = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task_id = {
                executor.submit(self._load_task_worker, str(task_file), task_source):
                task_file.stem for task_file in task_files
            }

            # Collect results
            for future in as_completed(future_to_task_id):
                task_id = future_to_task_id[future]
                try:
                    task = future.result()
                    if task:
                        tasks[task_id] = task
                except Exception as e:
                    print(f"Error loading task {task_id}: {e}")

        return tasks

    @staticmethod
    def _load_task_worker(task_file_path: str, task_source: str) -> ARCTask | None:
        """Worker function for parallel task loading."""
        try:
            with open(task_file_path) as f:
                raw_task = json.load(f)

            task_id = Path(task_file_path).stem
            return ARCTask.from_dict(raw_task, task_id, task_source)

        except Exception:
            return None

    def get_task_ids(self, task_source: str = "training") -> list[str]:
        """Get list of all available task IDs for a source."""
        if self.use_real_dataset and task_source in self._real_data_cache:
            # Get task IDs from real dataset cache
            challenges = self._real_data_cache[task_source]["challenges"]
            return list(challenges.keys())
        else:
            # Get task IDs from legacy file structure
            source_path = self.data_path / task_source
            if not source_path.exists():
                return []

            return [f.stem for f in source_path.glob("*.json")]

    def get_load_statistics(self) -> dict[str, Any]:
        """Get repository performance statistics."""
        stats = self.load_stats.copy()
        if stats["total_loaded"] > 0:
            stats["avg_load_time"] = stats["total_load_time"] / stats["total_loaded"]
        else:
            stats["avg_load_time"] = 0.0

        if self.cache_repository:
            total_requests = stats["cache_hits"] + stats["cache_misses"]
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests if total_requests > 0 else 0.0

        return stats

    def estimate_memory_usage(self, task_ids: list[str], task_source: str = "training") -> int:
        """Estimate memory usage for loading specified tasks."""
        sample_size = min(10, len(task_ids))
        sample_ids = task_ids[:sample_size]

        total_memory = 0
        for task_id in sample_ids:
            task = self.load_task(task_id, task_source)
            if task:
                total_memory += task.get_memory_usage_estimate()

        if sample_size > 0:
            avg_memory = total_memory / sample_size
            return int(avg_memory * len(task_ids))
        return 0

    def validate_data_integrity(self, task_source: str = "training") -> dict[str, list[str]]:
        """Validate data integrity for all tasks in source."""
        issues: dict[str, list[str]] = {"corrupted_tasks": [], "missing_files": [], "validation_errors": []}

        source_path = self.data_path / task_source
        if not source_path.exists():
            issues["missing_files"].append(str(source_path))
            return issues

        task_files = list(source_path.glob("*.json"))

        for task_file in task_files:
            try:
                # Try loading with task_loader validation
                self.task_loader.data_path = source_path
                raw_task = self.task_loader.load_task(task_file.name)

                # Additional domain model validation
                task_id = task_file.stem
                ARCTask.from_dict(raw_task, task_id, task_source)

            except Exception as e:
                issues["validation_errors"].append(f"{task_file.name}: {str(e)}")

        return issues
