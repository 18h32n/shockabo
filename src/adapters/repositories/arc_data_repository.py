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
    DataNotFoundException,
    DataCorruptionException,
    DataFormatException,
    ARCBaseException,
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    ErrorRecovery,
)

logger = structlog.get_logger(__name__)


class ARCDataRepository:
    """High-performance ARC task data repository with caching and bulk loading."""

    def __init__(
        self,
        data_path: str | None = None,
        cache_repository: Any | None = None,
        max_workers: int | None = None
    ):
        """Initialize repository with data path and optional cache."""
        try:
            if data_path is None:
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
            
            # Initialize task loader if available
            if ARCTaskLoader:
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
                "cache_misses": 0,
                "load_errors": 0,
                "validation_errors": 0
            }
            
            logger.info(
                "arc_repository_initialized",
                data_path=str(self.data_path),
                max_workers=self.max_workers,
                cache_enabled=self.cache_repository is not None,
                task_loader_available=self.task_loader is not None
            )
            
        except Exception as e:
            if isinstance(e, ARCBaseException):
                raise
            raise ARCBaseException(
                message=f"Failed to initialize ARC data repository: {str(e)}",
                error_code=ErrorCode.DATABASE_ERROR,
                severity=ErrorSeverity.CRITICAL,
                suggestions=[
                    "Check data path configuration",
                    "Verify file system permissions",
                    "Ensure all dependencies are installed"
                ]
            ) from e

    def load_task(self, task_id: str, task_source: str = "training") -> ARCTask | None:
        """Load a single ARC task by ID with optimized performance and error handling."""
        start_time = time.perf_counter()
        context = ErrorContext(
            task_id=task_id,
            additional_data={"task_source": task_source}
        )

        try:
            # Validate inputs
            if not task_id or not task_id.strip():
                raise ARCBaseException(
                    message="Task ID cannot be empty",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    context=context,
                    suggestions=["Provide a valid task ID"]
                )

            # Try cache first
            if self.cache_repository:
                cache_key = f"{task_source}:{task_id}"
                try:
                    cached_task = self.cache_repository.get(cache_key)
                    if cached_task:
                        self.load_stats["cache_hits"] += 1
                        logger.debug(
                            "task_loaded_from_cache",
                            task_id=task_id,
                            task_source=task_source
                        )
                        return cached_task
                    self.load_stats["cache_misses"] += 1
                except Exception as e:
                    logger.warning(
                        "cache_access_failed",
                        task_id=task_id,
                        error=str(e)
                    )

            # Load from disk with error handling
            task_file = f"{task_id}.json"
            source_path = self.data_path / task_source
            task_file_path = source_path / task_file

            # Check if source path exists
            if not source_path.exists():
                raise ARCBaseException(
                    message=f"Task source '{task_source}' not found",
                    error_code=ErrorCode.DATA_NOT_FOUND,
                    context=context,
                    suggestions=[
                        f"Check if '{task_source}' directory exists in {self.data_path}",
                        "Verify data source name is correct"
                    ]
                )

            # Check if task file exists
            if not task_file_path.exists():
                raise DataNotFoundException(
                    "task",
                    task_id,
                    context=context,
                    suggestions=[
                        f"Check if task '{task_id}' exists in source '{task_source}'",
                        "Verify task ID format and spelling"
                    ]
                )

            # Load and parse JSON with error handling
            try:
                with open(task_file_path, 'r', encoding='utf-8') as f:
                    raw_task = json.load(f)
            except json.JSONDecodeError as e:
                raise DataCorruptionException(
                    "task file",
                    f"Invalid JSON format: {str(e)}",
                    context=context,
                    suggestions=[
                        "Check file integrity",
                        "Verify JSON format is valid",
                        "Re-download the dataset if corrupted"
                    ]
                ) from e
            except (IOError, OSError) as e:
                raise ARCBaseException(
                    message=f"Failed to read task file: {str(e)}",
                    error_code=ErrorCode.DATA_LOADING_ERROR,
                    context=context,
                    suggestions=[
                        "Check file permissions",
                        "Verify disk space availability",
                        "Check file system integrity"
                    ]
                ) from e

            # Convert to domain model with validation
            try:
                arc_task = ARCTask.from_dict(raw_task, task_id, task_source)
            except Exception as e:
                self.load_stats["validation_errors"] += 1
                raise DataFormatException(
                    "ARCTask format",
                    f"Task validation failed: {str(e)}",
                    context=context,
                    suggestions=[
                        "Check task structure matches ARCTask schema",
                        "Verify train/test examples format",
                        "Check grid data integrity"
                    ]
                ) from e

            # Cache the result with error handling
            if self.cache_repository:
                try:
                    cache_key = f"{task_source}:{task_id}"
                    self.cache_repository.set(cache_key, arc_task)
                except Exception as e:
                    logger.warning(
                        "cache_storage_failed",
                        task_id=task_id,
                        error=str(e)
                    )

            # Update metrics
            load_time = time.perf_counter() - start_time
            self.load_stats["total_loaded"] += 1
            self.load_stats["total_load_time"] += load_time

            logger.debug(
                "task_loaded_successfully",
                task_id=task_id,
                task_source=task_source,
                load_time_ms=load_time * 1000,
                train_examples=len(arc_task.train_examples),
                has_test_output=arc_task.test_output is not None
            )

            return arc_task

        except ARCBaseException:
            self.load_stats["load_errors"] += 1
            raise
        except Exception as e:
            self.load_stats["load_errors"] += 1
            logger.error(
                "unexpected_task_load_error",
                task_id=task_id,
                task_source=task_source,
                error=str(e),
                exc_info=True
            )
            raise ARCBaseException(
                message=f"Unexpected error loading task '{task_id}': {str(e)}",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                severity=ErrorSeverity.HIGH,
                context=context,
                suggestions=[
                    "Check system resources and permissions",
                    "Verify data integrity",
                    "Contact support if problem persists"
                ]
            ) from e

    def load_all_tasks(
        self,
        task_source: str = "training",
        limit: int | None = None
    ) -> dict[str, ARCTask]:
        """Load all tasks from specified source with optimized performance and error handling."""
        start_time = time.perf_counter()
        context = ErrorContext(additional_data={
            "task_source": task_source,
            "limit": limit
        })

        try:
            # Validate source path
            source_path = self.data_path / task_source
            if not source_path.exists():
                raise DataNotFoundException(
                    "task source",
                    task_source,
                    context=context,
                    suggestions=[
                        f"Check if '{task_source}' directory exists in {self.data_path}",
                        "Verify task source name is correct",
                        "Ensure dataset is properly downloaded"
                    ]
                )

            # Get all task files
            try:
                task_files = list(source_path.glob("*.json"))
            except Exception as e:
                raise ARCBaseException(
                    message=f"Failed to list task files in {source_path}: {str(e)}",
                    error_code=ErrorCode.DATA_LOADING_ERROR,
                    context=context,
                    suggestions=[
                        "Check directory permissions",
                        "Verify file system integrity"
                    ]
                ) from e

            if not task_files:
                logger.warning(
                    "no_task_files_found",
                    task_source=task_source,
                    source_path=str(source_path)
                )
                return {}

            if limit and limit > 0:
                task_files = task_files[:limit]

            logger.info(
                "bulk_loading_started",
                task_source=task_source,
                total_files=len(task_files),
                limit=limit
            )

            # Choose loading strategy based on dataset size and system resources
            if len(task_files) > 100 and self.max_workers > 1:
                logger.info("using_parallel_loading", files=len(task_files))
                tasks = self._load_tasks_parallel(task_files, task_source)
            else:
                logger.info("using_sequential_loading", files=len(task_files))
                tasks = self._load_tasks_sequential(task_files, task_source)

            load_time = time.perf_counter() - start_time
            self.load_stats["total_loaded"] += len(tasks)
            self.load_stats["total_load_time"] += load_time

            # Calculate success rate
            success_rate = len(tasks) / len(task_files) if task_files else 0
            failed_count = len(task_files) - len(tasks)

            logger.info(
                "bulk_loading_completed",
                task_source=task_source,
                total_files=len(task_files),
                loaded_tasks=len(tasks),
                failed_tasks=failed_count,
                success_rate=success_rate,
                load_time_s=load_time,
                avg_time_per_task_ms=(load_time / len(tasks) * 1000) if tasks else 0
            )

            if failed_count > 0:
                logger.warning(
                    "bulk_loading_partial_failure",
                    failed_tasks=failed_count,
                    success_rate=success_rate
                )

            return tasks

        except ARCBaseException:
            raise
        except Exception as e:
            logger.error(
                "bulk_loading_failed",
                task_source=task_source,
                error=str(e),
                exc_info=True
            )
            raise ARCBaseException(
                message=f"Failed to load tasks from '{task_source}': {str(e)}",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                severity=ErrorSeverity.HIGH,
                context=context,
                suggestions=[
                    "Check data source integrity",
                    "Verify system resources",
                    "Try loading with a smaller limit"
                ]
            ) from e

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
                # Log error but continue with other tasks
                logger.warning(
                    "sequential_task_load_failed",
                    task_file=task_file.name,
                    error=str(e)
                )
                self.load_stats["load_errors"] += 1
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
                    logger.error(
                        "parallel_task_load_failed",
                        task_id=task_id,
                        error=str(e)
                    )
                    self.load_stats["load_errors"] += 1

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
        """Get list of all available task IDs for a source with error handling."""
        try:
            source_path = self.data_path / task_source
            if not source_path.exists():
                logger.warning(
                    "task_source_not_found",
                    task_source=task_source,
                    path=str(source_path)
                )
                return []

            task_ids = [f.stem for f in source_path.glob("*.json")]
            logger.debug(
                "task_ids_retrieved",
                task_source=task_source,
                count=len(task_ids)
            )
            return task_ids
            
        except Exception as e:
            logger.error(
                "get_task_ids_failed",
                task_source=task_source,
                error=str(e),
                exc_info=True
            )
            raise ARCBaseException(
                message=f"Failed to get task IDs from '{task_source}': {str(e)}",
                error_code=ErrorCode.DATA_LOADING_ERROR,
                context=ErrorContext(additional_data={"task_source": task_source}),
                suggestions=[
                    "Check directory permissions",
                    "Verify data source exists",
                    "Check file system integrity"
                ]
            ) from e

    def get_load_statistics(self) -> dict[str, Any]:
        """Get repository performance statistics with error handling."""
        try:
            stats = self.load_stats.copy()
            
            # Calculate derived metrics
            if stats["total_loaded"] > 0:
                stats["avg_load_time_ms"] = (stats["total_load_time"] / stats["total_loaded"]) * 1000
                stats["error_rate"] = stats["load_errors"] / (stats["total_loaded"] + stats["load_errors"])
                stats["validation_error_rate"] = stats["validation_errors"] / (stats["total_loaded"] + stats["validation_errors"])
            else:
                stats["avg_load_time_ms"] = 0.0
                stats["error_rate"] = 0.0
                stats["validation_error_rate"] = 0.0

            # Cache statistics
            if self.cache_repository:
                total_requests = stats["cache_hits"] + stats["cache_misses"]
                stats["cache_hit_rate"] = stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
                stats["cache_enabled"] = True
            else:
                stats["cache_enabled"] = False
                stats["cache_hit_rate"] = 0.0
            
            # Repository health metrics
            stats["repository_health"] = {
                "healthy": stats["error_rate"] < 0.05,  # Less than 5% error rate
                "data_quality_score": 1.0 - stats["validation_error_rate"],
                "performance_score": min(1.0, 100.0 / (stats["avg_load_time_ms"] + 1)),  # Faster = higher score
            }

            return stats
            
        except Exception as e:
            logger.error("statistics_calculation_failed", error=str(e))
            return {
                "error": f"Failed to calculate statistics: {str(e)}",
                "total_loaded": self.load_stats.get("total_loaded", 0)
            }

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
        """Validate data integrity for all tasks in source with comprehensive error handling."""
        issues: dict[str, list[str]] = {
            "corrupted_tasks": [],
            "missing_files": [],
            "validation_errors": [],
            "json_errors": [],
            "permission_errors": []
        }
        
        try:
            source_path = self.data_path / task_source
            if not source_path.exists():
                issues["missing_files"].append(str(source_path))
                logger.error(
                    "validation_source_missing",
                    task_source=task_source,
                    path=str(source_path)
                )
                return issues

            # Get task files with error handling
            try:
                task_files = list(source_path.glob("*.json"))
            except Exception as e:
                issues["permission_errors"].append(f"Cannot access directory {source_path}: {str(e)}")
                return issues

            logger.info(
                "data_integrity_validation_started",
                task_source=task_source,
                total_files=len(task_files)
            )

            validated_count = 0
            for task_file in task_files:
                try:
                    # Basic file access check
                    if not task_file.is_file():
                        issues["missing_files"].append(str(task_file))
                        continue
                    
                    # JSON parsing validation
                    try:
                        with open(task_file, 'r', encoding='utf-8') as f:
                            raw_task = json.load(f)
                    except json.JSONDecodeError as e:
                        issues["json_errors"].append(f"{task_file.name}: Invalid JSON - {str(e)}")
                        continue
                    except (IOError, OSError) as e:
                        issues["permission_errors"].append(f"{task_file.name}: File access error - {str(e)}")
                        continue

                    # Task loader validation if available
                    if self.task_loader:
                        try:
                            self.task_loader.data_path = source_path
                            loader_task = self.task_loader.load_task(task_file.name)
                            # Compare with raw loaded data
                            if not loader_task or loader_task != raw_task:
                                issues["corrupted_tasks"].append(f"{task_file.name}: Task loader validation mismatch")
                        except Exception as e:
                            issues["corrupted_tasks"].append(f"{task_file.name}: Task loader error - {str(e)}")

                    # Domain model validation
                    try:
                        task_id = task_file.stem
                        ARCTask.from_dict(raw_task, task_id, task_source)
                        validated_count += 1
                    except Exception as e:
                        issues["validation_errors"].append(f"{task_file.name}: Domain validation - {str(e)}")

                except Exception as e:
                    issues["validation_errors"].append(f"{task_file.name}: Unexpected error - {str(e)}")
                    logger.error(
                        "task_validation_unexpected_error",
                        task_file=str(task_file),
                        error=str(e)
                    )

            # Calculate validation summary
            total_issues = sum(len(issue_list) for issue_list in issues.values())
            validation_score = validated_count / len(task_files) if task_files else 0
            
            logger.info(
                "data_integrity_validation_completed",
                task_source=task_source,
                total_files=len(task_files),
                validated_count=validated_count,
                total_issues=total_issues,
                validation_score=validation_score,
                corrupted_tasks=len(issues["corrupted_tasks"]),
                json_errors=len(issues["json_errors"]),
                validation_errors=len(issues["validation_errors"])
            )
            
            # Add summary information
            issues["summary"] = [
                f"Validated {validated_count}/{len(task_files)} tasks successfully",
                f"Data integrity score: {validation_score:.2%}",
                f"Total issues found: {total_issues}"
            ]

            return issues
            
        except Exception as e:
            logger.error(
                "data_integrity_validation_failed",
                task_source=task_source,
                error=str(e),
                exc_info=True
            )
            issues["validation_errors"].append(f"Validation process failed: {str(e)}")
            return issues
