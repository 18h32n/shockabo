"""Weights & Biases integration client for experiment tracking and monitoring.

This module provides secure integration with Weights & Biases for tracking experiments,
logging metrics, and monitoring resource usage within the 100GB free tier limit.
"""

import asyncio
import json
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. W&B integration will be disabled.", ImportWarning, stacklevel=2)

from src.domain.evaluation_models import (
    ExperimentMetrics,
    ExperimentRun,
    ResourceUsage,
)
from src.domain.services.evaluation_service import EvaluationResult
from src.utils.secure_credentials import get_credential_manager

logger = structlog.get_logger(__name__)


@dataclass
class BatchLogEntry:
    """Entry in the batch logging queue."""

    entry_type: str  # 'evaluation', 'resource', 'summary'
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class BatchProcessor:
    """Async batch processor for W&B logging operations."""

    def __init__(self,
                 batch_size: int = 10,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of entries to batch together
            flush_interval: Seconds between automatic flushes
            max_queue_size: Maximum queue size before forced flush
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size

        self.queue: deque[BatchLogEntry] = deque()
        self.processing_task: asyncio.Task | None = None
        self.is_running = False

        # Performance metrics
        self.batches_processed = 0
        self.entries_processed = 0
        self.failures = 0
        self.last_flush_time = datetime.now()

        self.logger = structlog.get_logger(__name__).bind(component="batch_processor")

    async def start(self):
        """Start the batch processing task."""
        if self.is_running:
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_queue())
        self.logger.info("batch_processor_started",
                        batch_size=self.batch_size,
                        flush_interval=self.flush_interval)

    async def stop(self):
        """Stop the batch processor and flush remaining entries."""
        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Flush remaining entries
        if self.queue:
            await self._flush_batch(list(self.queue))
            self.queue.clear()

        self.logger.info("batch_processor_stopped",
                        batches_processed=self.batches_processed,
                        entries_processed=self.entries_processed)

    def add_entry(self, entry_type: str, data: dict[str, Any]) -> bool:
        """Add an entry to the batch queue.
        
        Args:
            entry_type: Type of log entry
            data: Data to log
            
        Returns:
            True if added successfully
        """
        if len(self.queue) >= self.max_queue_size:
            self.logger.warning("batch_queue_full",
                              queue_size=len(self.queue),
                              dropping_entry=True)
            # Force flush to make room
            asyncio.create_task(self._flush_current_batch())
            return False

        entry = BatchLogEntry(entry_type=entry_type, data=data)
        self.queue.append(entry)

        # Force flush if batch size reached
        if len(self.queue) >= self.batch_size:
            asyncio.create_task(self._flush_current_batch())

        return True

    async def _process_queue(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Check if it's time to flush
                time_since_flush = (datetime.now() - self.last_flush_time).total_seconds()

                if (len(self.queue) >= self.batch_size or
                    (len(self.queue) > 0 and time_since_flush >= self.flush_interval)):
                    await self._flush_current_batch()

                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error("batch_processing_error", error=str(e), exc_info=True)
                await asyncio.sleep(1)  # Back off on errors

    async def _flush_current_batch(self):
        """Flush the current batch of entries."""
        if not self.queue:
            return

        # Extract batch entries
        batch_entries = []
        batch_size = min(len(self.queue), self.batch_size)

        for _ in range(batch_size):
            if self.queue:
                batch_entries.append(self.queue.popleft())

        if batch_entries:
            await self._flush_batch(batch_entries)

    async def _flush_batch(self, entries: list[BatchLogEntry]):
        """Flush a batch of entries to W&B.
        
        Args:
            entries: List of entries to flush
        """
        if not entries:
            return

        start_time = datetime.now()

        try:
            # Group entries by type for efficient processing
            grouped_entries = defaultdict(list)
            for entry in entries:
                grouped_entries[entry.entry_type].append(entry.data)

            # Process each group
            for entry_type, data_list in grouped_entries.items():
                await self._process_entry_group(entry_type, data_list)

            # Update metrics
            self.batches_processed += 1
            self.entries_processed += len(entries)
            self.last_flush_time = datetime.now()

            processing_time = (self.last_flush_time - start_time).total_seconds()

            self.logger.info("batch_flushed",
                           entries_count=len(entries),
                           processing_time_ms=processing_time * 1000,
                           entry_types=list(grouped_entries.keys()))

        except Exception as e:
            self.failures += 1
            self.logger.error("batch_flush_failed",
                            entries_count=len(entries),
                            error=str(e),
                            exc_info=True)

    async def _process_entry_group(self, entry_type: str, data_list: list[dict[str, Any]]):
        """Process a group of entries of the same type.
        
        Args:
            entry_type: Type of entries
            data_list: List of data to process
        """
        if not WANDB_AVAILABLE or not wandb.run:
            return

        try:
            if entry_type == "evaluation":
                # Batch evaluation results
                combined_metrics = {}
                for data in data_list:
                    for key, value in data.items():
                        if key not in combined_metrics:
                            combined_metrics[key] = []
                        combined_metrics[key].append(value)

                # Log combined metrics
                wandb.log(combined_metrics)

            elif entry_type == "resource":
                # Batch resource usage
                for data in data_list:
                    wandb.log(data)

            elif entry_type == "summary":
                # Update summary (last one wins)
                for data in data_list:
                    wandb.run.summary.update(data)

        except Exception as e:
            self.logger.error("entry_group_processing_failed",
                            entry_type=entry_type,
                            count=len(data_list),
                            error=str(e))
            raise

    def get_statistics(self) -> dict[str, Any]:
        """Get batch processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            "queue_size": len(self.queue),
            "batches_processed": self.batches_processed,
            "entries_processed": self.entries_processed,
            "failures": self.failures,
            "is_running": self.is_running,
            "config": {
                "batch_size": self.batch_size,
                "flush_interval": self.flush_interval,
                "max_queue_size": self.max_queue_size
            }
        }


class BatchOperationType(Enum):
    """Types of batch operations supported by the W&B client."""

    EVALUATION_RESULT = "evaluation_result"
    RESOURCE_USAGE = "resource_usage"
    EXPERIMENT_SUMMARY = "experiment_summary"
    CUSTOM_METRICS = "custom_metrics"


@dataclass
class BatchOperation:
    """Represents a single operation to be batched."""

    operation_type: BatchOperationType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    operation_id: str = field(default_factory=lambda: f"op_{int(time.time() * 1000000)}")


@dataclass
class BatchMetrics:
    """Metrics for tracking batch processing performance."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    total_flush_time_ms: float = 0.0
    average_flush_time_ms: float = 0.0
    partial_failures: int = 0
    retry_operations: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_batch_stats(self, batch_size: int, flush_time_ms: float,
                          successful: int, failed: int, partial_failure: bool = False):
        """Update batch processing statistics."""
        self.total_batches += 1
        self.total_operations += batch_size
        self.successful_operations += successful
        self.failed_operations += failed
        self.total_flush_time_ms += flush_time_ms

        if partial_failure:
            self.partial_failures += 1

        self.average_batch_size = self.total_operations / self.total_batches
        self.average_flush_time_ms = self.total_flush_time_ms / self.total_batches
        self.last_updated = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / max(1, self.total_operations),
            "total_batches": self.total_batches,
            "average_batch_size": round(self.average_batch_size, 2),
            "average_flush_time_ms": round(self.average_flush_time_ms, 2),
            "partial_failures": self.partial_failures,
            "retry_operations": self.retry_operations,
        }


class WandBConfig:
    """Configuration for Weights & Biases integration."""

    def __init__(self):
        """Initialize W&B configuration from secure storage or environment variables."""
        # Get credential manager
        self._cred_manager = get_credential_manager()

        # Try to get API key from secure storage first, then environment
        self.api_key = self._cred_manager.get_credential_with_fallback(
            'WANDB_API_KEY',
            env_var='WANDB_API_KEY'
        )

        self.project_name = os.environ.get("WANDB_PROJECT", "arc-prize-2025")
        self.entity = os.environ.get("WANDB_ENTITY")  # Optional: organization/team name
        self.base_dir = os.environ.get("WANDB_DIR", "./wandb")
        self.mode = os.environ.get("WANDB_MODE", "online")  # online, offline, or disabled
        self.tags = os.environ.get("WANDB_TAGS", "evaluation,arc").split(",")

        # Free tier limits
        self.storage_limit_gb = 100
        self.storage_warning_threshold = 0.8  # Warn at 80% usage
        self.storage_critical_threshold = 0.95  # Critical at 95% usage

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid
        """
        if not self.api_key and self.mode == "online":
            logger.error("wandb_config_invalid", reason="WANDB_API_KEY not set")
            return False

        if not WANDB_AVAILABLE and self.mode != "disabled":
            logger.error("wandb_config_invalid", reason="wandb package not installed")
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "project_name": self.project_name,
            "entity": self.entity,
            "base_dir": self.base_dir,
            "mode": self.mode,
            "tags": self.tags,
            "storage_limit_gb": self.storage_limit_gb,
        }


class UsageMonitor:
    """Monitors W&B storage usage to stay within free tier limits."""

    def __init__(self, config: WandBConfig):
        """Initialize usage monitor.

        Args:
            config: W&B configuration
        """
        self.config = config
        self.logger = structlog.get_logger(__name__).bind(component="usage_monitor")
        self._usage_cache_file = Path(self.config.base_dir) / ".wandb_usage_cache.json"
        self._last_check = datetime.min
        self._cached_usage_gb = 0.0

    def get_current_usage_gb(self) -> float:
        """Get current storage usage in GB.

        Returns:
            Current usage in GB
        """
        if not WANDB_AVAILABLE or not wandb.run:
            return self._get_cached_usage()

        try:
            # Get usage from W&B API
            api = wandb.Api()
            if self.config.entity:
                runs = api.runs(f"{self.config.entity}/{self.config.project_name}")
            else:
                runs = api.runs(self.config.project_name)

            total_size_bytes = 0
            for run in runs:
                # Sum up artifact sizes
                for artifact in run.logged_artifacts():
                    total_size_bytes += artifact.size or 0

                # Estimate log file sizes (rough approximation)
                total_size_bytes += run.summary.get("_runtime", 0) * 1000  # ~1KB per second

            total_gb = total_size_bytes / (1024**3)
            self._cache_usage(total_gb)
            return total_gb

        except Exception as e:
            self.logger.warning("usage_check_failed", error=str(e))
            return self._get_cached_usage()

    def _get_cached_usage(self) -> float:
        """Get cached usage value.

        Returns:
            Cached usage in GB
        """
        try:
            if self._usage_cache_file.exists():
                with open(self._usage_cache_file) as f:
                    data = json.load(f)
                    return data.get("usage_gb", 0.0)
        except Exception:
            pass
        return 0.0

    def _cache_usage(self, usage_gb: float):
        """Cache usage value.

        Args:
            usage_gb: Usage in GB to cache
        """
        try:
            self._usage_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._usage_cache_file, "w") as f:
                json.dump(
                    {
                        "usage_gb": usage_gb,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                )
            self._cached_usage_gb = usage_gb
        except Exception as e:
            self.logger.warning("usage_cache_failed", error=str(e))

    def check_usage_limits(self) -> tuple[bool, str | None]:
        """Check if usage is within limits.

        Returns:
            Tuple of (is_ok, warning_message)
        """
        usage_gb = self.get_current_usage_gb()
        usage_percentage = usage_gb / self.config.storage_limit_gb

        if usage_percentage >= self.config.storage_critical_threshold:
            message = (
                f"CRITICAL: W&B storage at {usage_percentage:.1%} "
                f"({usage_gb:.1f}/{self.config.storage_limit_gb}GB). "
                "Consider cleaning up old runs or upgrading plan."
            )
            self.logger.error("wandb_storage_critical", usage_gb=usage_gb, percentage=usage_percentage)
            return False, message

        if usage_percentage >= self.config.storage_warning_threshold:
            message = (
                f"WARNING: W&B storage at {usage_percentage:.1%} "
                f"({usage_gb:.1f}/{self.config.storage_limit_gb}GB). "
                "Approaching free tier limit."
            )
            self.logger.warning("wandb_storage_warning", usage_gb=usage_gb, percentage=usage_percentage)
            return True, message

        return True, None


class WandBClient:
    """Client for Weights & Biases integration."""

    def __init__(self, config: WandBConfig | None = None, enable_batching: bool = True):
        """Initialize W&B client.

        Args:
            config: W&B configuration (uses environment if not provided)
            enable_batching: Whether to enable batch processing for performance
        """
        self.config = config or WandBConfig()
        self.logger = structlog.get_logger(__name__).bind(service="wandb_client")
        self.usage_monitor = UsageMonitor(self.config)
        self._initialized = False
        self._current_run: Any | None = None

        # Batch processing
        self.enable_batching = enable_batching
        self.batch_processor: BatchProcessor | None = None
        if enable_batching:
            batch_size = int(os.environ.get("WANDB_BATCH_SIZE", "10"))
            flush_interval = float(os.environ.get("WANDB_FLUSH_INTERVAL", "5.0"))
            max_queue_size = int(os.environ.get("WANDB_MAX_QUEUE_SIZE", "1000"))

            self.batch_processor = BatchProcessor(
                batch_size=batch_size,
                flush_interval=flush_interval,
                max_queue_size=max_queue_size
            )

        # Validate configuration
        if not self.config.validate():
            self.logger.warning("wandb_disabled", reason="Invalid configuration")
            self.config.mode = "disabled"

    def initialize(self) -> bool:
        """Initialize W&B connection.

        Returns:
            True if initialization successful
        """
        if self._initialized or self.config.mode == "disabled":
            return self._initialized

        try:
            # Check usage before initializing
            is_ok, warning = self.usage_monitor.check_usage_limits()
            if warning:
                self.logger.warning("wandb_usage_warning", message=warning)
            if not is_ok:
                self.logger.error("wandb_init_blocked", reason="Storage limit exceeded")
                return False

            # Initialize W&B with secure authentication
            if WANDB_AVAILABLE:
                # Set API key in environment for wandb to pick up securely
                if self.config.api_key:
                    os.environ["WANDB_API_KEY"] = self.config.api_key
                    # Call login without key parameter - wandb will use env var
                    wandb.login()
                    # Clear from environment after login
                    if "WANDB_API_KEY" in os.environ:
                        del os.environ["WANDB_API_KEY"]
                else:
                    # Try login without key - may prompt or use existing auth
                    wandb.login()

                self._initialized = True
                self.logger.info(
                    "wandb_initialized",
                    project=self.config.project_name,
                    mode=self.config.mode,
                )
                return True

        except Exception as e:
            self.logger.error("wandb_init_failed", error=str(e), exc_info=True)
            self.config.mode = "disabled"

        return False

    def start_experiment(
        self,
        experiment: ExperimentRun,
        config: dict[str, Any],
        resume: str | None = None,
    ) -> str | None:
        """Start a new W&B experiment run.

        Args:
            experiment: Experiment run details
            config: Experiment configuration
            resume: Optional run ID to resume

        Returns:
            Run ID if successful, None otherwise
        """
        if not self.initialize() or self.config.mode == "disabled":
            return None

        try:
            # End any existing run
            if self._current_run:
                self.end_experiment()

            # Start new run
            run_config = {
                "experiment_name": experiment.experiment_name,
                "num_tasks": len(experiment.task_ids),
                "strategy": experiment.strategy_config,
                **config,
            }

            self._current_run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=experiment.experiment_name,
                config=run_config,
                tags=self.config.tags + [experiment.strategy_config.get("strategy", "unknown")],
                resume=resume,
                mode=self.config.mode,
            )

            # Log experiment metadata
            wandb.config.update(
                {
                    "run_id": experiment.run_id,
                    "started_at": experiment.started_at.isoformat(),
                    "task_ids": experiment.task_ids[:10],  # Sample to avoid too much data
                }
            )

            self.logger.info(
                "wandb_experiment_started",
                run_id=self._current_run.id,
                experiment_id=experiment.run_id,
            )

            return self._current_run.id

        except Exception as e:
            self.logger.error(
                "wandb_start_experiment_failed",
                experiment_id=experiment.run_id,
                error=str(e),
                exc_info=True,
            )
            return None

    def log_evaluation_result(self, result: EvaluationResult) -> bool:
        """Log evaluation result to W&B.

        Args:
            result: Evaluation result to log

        Returns:
            True if logging successful
        """
        if not self._current_run or self.config.mode == "disabled":
            return False

        try:
            # Log main metrics
            metrics = {
                "task_id": result.task_id,
                "accuracy": result.final_accuracy,
                "strategy": result.strategy_used,
                "num_attempts": len(result.attempts),
                "processing_time_ms": result.total_processing_time_ms,
            }

            # Add attempt-specific metrics
            for i, attempt in enumerate(result.attempts):
                prefix = f"attempt_{i+1}"
                metrics.update(
                    {
                        f"{prefix}/accuracy": attempt.pixel_accuracy.accuracy,
                        f"{prefix}/perfect_match": attempt.pixel_accuracy.perfect_match,
                        f"{prefix}/confidence": attempt.confidence_score,
                        f"{prefix}/processing_time_ms": attempt.processing_time_ms,
                    }
                )

                if attempt.error_category:
                    metrics[f"{prefix}/error_category"] = attempt.error_category.value

            # Log to W&B
            wandb.log(metrics)

            # Log detailed error analysis if available
            if result.best_attempt and result.best_attempt.error_details:
                wandb.log(
                    {
                        "error_analysis": wandb.Table(
                            columns=["task_id", "error_type", "details"],
                            data=[
                                [
                                    result.task_id,
                                    result.best_attempt.error_category.value if result.best_attempt.error_category else "none",
                                    json.dumps(result.best_attempt.error_details),
                                ]
                            ],
                        )
                    }
                )

            return True

        except Exception as e:
            self.logger.error(
                "wandb_log_evaluation_failed",
                task_id=result.task_id,
                error=str(e),
                exc_info=True,
            )
            return False

    def log_resource_usage(self, usage: ResourceUsage) -> bool:
        """Log resource usage metrics to W&B.

        Args:
            usage: Resource usage data

        Returns:
            True if logging successful
        """
        if not self._current_run or self.config.mode == "disabled":
            return False

        try:
            metrics = {
                "resource/cpu_seconds": usage.cpu_seconds,
                "resource/memory_mb": usage.memory_mb,
                "resource/total_tokens": usage.total_tokens,
                "resource/estimated_cost": usage.estimated_cost,
            }

            if usage.gpu_memory_mb is not None:
                metrics["resource/gpu_memory_mb"] = usage.gpu_memory_mb

            # Log API calls breakdown
            for api_name, count in usage.api_calls.items():
                metrics[f"resource/api_calls/{api_name}"] = count

            wandb.log(metrics)
            return True

        except Exception as e:
            self.logger.error(
                "wandb_log_resource_failed",
                task_id=usage.task_id,
                error=str(e),
                exc_info=True,
            )
            return False

    def log_experiment_summary(self, metrics: ExperimentMetrics) -> bool:
        """Log experiment summary metrics to W&B.

        Args:
            metrics: Experiment summary metrics

        Returns:
            True if logging successful
        """
        if not self._current_run or self.config.mode == "disabled":
            return False

        try:
            # Log summary metrics
            summary = {
                "total_tasks": metrics.total_tasks,
                "successful_tasks": metrics.successful_tasks,
                "failed_tasks": metrics.failed_tasks,
                "average_accuracy": metrics.average_accuracy,
                "perfect_matches": metrics.perfect_matches,
                "success_rate": metrics.success_rate,
                "perfect_match_rate": metrics.perfect_match_rate,
                "total_processing_time_ms": metrics.total_processing_time_ms,
                "average_processing_time_ms": metrics.average_processing_time_ms,
                "total_resource_cost": metrics.total_resource_cost,
            }

            wandb.run.summary.update(summary)

            # Log strategy performance table
            if metrics.strategy_performance:
                strategy_data = []
                for strategy, perf in metrics.strategy_performance.items():
                    strategy_data.append(
                        [
                            strategy,
                            perf.get("tasks_evaluated", 0),
                            perf.get("average_accuracy", 0),
                            perf.get("perfect_matches", 0),
                        ]
                    )

                wandb.log(
                    {
                        "strategy_performance": wandb.Table(
                            columns=["strategy", "tasks", "avg_accuracy", "perfect_matches"],
                            data=strategy_data,
                        )
                    }
                )

            # Log error distribution
            if metrics.error_distribution:
                error_data = [[error, count] for error, count in metrics.error_distribution.items()]
                wandb.log(
                    {
                        "error_distribution": wandb.Table(
                            columns=["error_type", "count"],
                            data=error_data,
                        )
                    }
                )

            return True

        except Exception as e:
            self.logger.error(
                "wandb_log_summary_failed",
                experiment_id=metrics.experiment_id,
                error=str(e),
                exc_info=True,
            )
            return False

    def save_model_artifact(
        self,
        model_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save a model or checkpoint as W&B artifact.

        Args:
            model_path: Path to model file or directory
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            metadata: Optional metadata to attach

        Returns:
            True if save successful
        """
        if not self._current_run or self.config.mode == "disabled":
            return False

        try:
            # Check usage before saving
            is_ok, warning = self.usage_monitor.check_usage_limits()
            if not is_ok:
                self.logger.error("wandb_artifact_blocked", reason="Storage limit exceeded")
                return False

            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {},
            )

            # Add file or directory
            if os.path.isdir(model_path):
                artifact.add_dir(model_path)
            else:
                artifact.add_file(model_path)

            # Log artifact
            wandb.log_artifact(artifact)

            self.logger.info(
                "wandb_artifact_saved",
                artifact_name=artifact_name,
                artifact_type=artifact_type,
            )
            return True

        except Exception as e:
            self.logger.error(
                "wandb_save_artifact_failed",
                artifact_name=artifact_name,
                error=str(e),
                exc_info=True,
            )
            return False

    def end_experiment(self) -> bool:
        """End the current W&B run.

        Returns:
            True if successful
        """
        if not self._current_run:
            return True

        try:
            # Final usage check
            is_ok, warning = self.usage_monitor.check_usage_limits()
            if warning:
                wandb.alert(
                    title="Storage Usage Warning",
                    text=warning,
                    level=wandb.AlertLevel.WARN,
                )

            # Finish the run
            wandb.finish()
            self._current_run = None

            self.logger.info("wandb_experiment_ended")
            return True

        except Exception as e:
            self.logger.error("wandb_end_experiment_failed", error=str(e), exc_info=True)
            return False

    def download_artifact(
        self, artifact_name: str, artifact_type: str = "model", version: str = "latest"
    ) -> str | None:
        """Download a W&B artifact.

        Args:
            artifact_name: Name of the artifact
            artifact_type: Type of artifact
            version: Version to download (default: latest)

        Returns:
            Local path to downloaded artifact, or None if failed
        """
        if not self.initialize() or self.config.mode == "disabled":
            return None

        try:
            api = wandb.Api()
            artifact_path = f"{self.config.project_name}/{artifact_name}:{version}"
            if self.config.entity:
                artifact_path = f"{self.config.entity}/{artifact_path}"

            artifact = api.artifact(artifact_path, type=artifact_type)
            local_path = artifact.download()

            self.logger.info(
                "wandb_artifact_downloaded",
                artifact_name=artifact_name,
                version=version,
                local_path=local_path,
            )
            return local_path

        except Exception as e:
            self.logger.error(
                "wandb_download_artifact_failed",
                artifact_name=artifact_name,
                error=str(e),
                exc_info=True,
            )
            return None


class BatchedWandBClient(WandBClient):
    """W&B client with async batch processing capabilities for improved performance."""

    def __init__(
        self,
        config: WandBConfig | None = None,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
        max_retry_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
        enable_batching: bool = True,
    ):
        """Initialize batched W&B client.

        Args:
            config: W&B configuration
            batch_size: Maximum number of operations per batch
            flush_interval_seconds: Time interval between automatic flushes
            max_retry_attempts: Maximum retry attempts for failed operations
            retry_delay_seconds: Delay between retry attempts
            enable_batching: Whether to enable batching (can be disabled for debugging)
        """
        super().__init__(config)
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.enable_batching = enable_batching

        # Batch processing state
        self._batch_queue: deque[BatchOperation] = deque()
        self._retry_queue: deque[BatchOperation] = deque()
        self._batch_metrics = BatchMetrics()
        self._batch_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._running = False

        # Operation handlers for different batch types
        self._operation_handlers = {
            BatchOperationType.EVALUATION_RESULT: self._process_evaluation_results_batch,
            BatchOperationType.RESOURCE_USAGE: self._process_resource_usage_batch,
            BatchOperationType.EXPERIMENT_SUMMARY: self._process_experiment_summary_batch,
            BatchOperationType.CUSTOM_METRICS: self._process_custom_metrics_batch,
        }

        self.logger = self.logger.bind(component="batched_wandb_client")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_batch_processing()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_batch_processing()

    async def start_batch_processing(self) -> None:
        """Start the batch processing task."""
        if self._running or not self.enable_batching:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._batch_flush_worker())
        self.logger.info(
            "batch_processing_started",
            batch_size=self.batch_size,
            flush_interval=self.flush_interval_seconds,
        )

    async def stop_batch_processing(self) -> None:
        """Stop batch processing and flush remaining operations."""
        if not self._running:
            return

        self._running = False

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining operations
        await self._flush_batches(force=True)
        self.logger.info(
            "batch_processing_stopped",
            final_metrics=self._batch_metrics.to_dict(),
        )

    async def _batch_flush_worker(self) -> None:
        """Background worker that periodically flushes batches."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                if self._running:  # Check again after sleep
                    await self._flush_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("batch_flush_worker_error", error=str(e), exc_info=True)

    async def _add_to_batch(self, operation: BatchOperation) -> None:
        """Add an operation to the batch queue."""
        async with self._batch_lock:
            self._batch_queue.append(operation)

            # Check if we should flush immediately
            if len(self._batch_queue) >= self.batch_size:
                await self._flush_batches()

    async def _flush_batches(self, force: bool = False) -> None:
        """Flush all pending batches."""
        async with self._batch_lock:
            if not self._batch_queue and not self._retry_queue:
                return

            # Process retry queue first
            if self._retry_queue:
                await self._process_retry_queue()

            # Group operations by type
            operations_by_type = defaultdict(list)
            operations_to_process = list(self._batch_queue)

            # Only process a batch if we have enough operations or are forcing
            if not force and len(operations_to_process) < self.batch_size:
                return

            self._batch_queue.clear()

            for op in operations_to_process:
                operations_by_type[op.operation_type].append(op)

            # Process each operation type
            for op_type, operations in operations_by_type.items():
                await self._process_operations_batch(op_type, operations)

    async def _process_operations_batch(
        self, operation_type: BatchOperationType, operations: list[BatchOperation]
    ) -> None:
        """Process a batch of operations of the same type."""
        if not operations or not self._current_run or self.config.mode == "disabled":
            return

        start_time = time.time()
        successful_ops = 0
        failed_ops = 0
        partial_failure = False

        try:
            handler = self._operation_handlers.get(operation_type)
            if not handler:
                self.logger.error("unknown_operation_type", operation_type=operation_type.value)
                failed_ops = len(operations)
                return

            # Execute the batch operation
            success_results = await handler(operations)

            # Count successes and failures
            for i, success in enumerate(success_results):
                if success:
                    successful_ops += 1
                else:
                    failed_ops += 1
                    # Add failed operations to retry queue
                    if operations[i].retry_count < self.max_retry_attempts:
                        operations[i].retry_count += 1
                        self._retry_queue.append(operations[i])
                        self._batch_metrics.retry_operations += 1

            if failed_ops > 0 and successful_ops > 0:
                partial_failure = True

            self.logger.info(
                "batch_processed",
                operation_type=operation_type.value,
                total_operations=len(operations),
                successful=successful_ops,
                failed=failed_ops,
                partial_failure=partial_failure,
            )

        except Exception as e:
            failed_ops = len(operations)
            partial_failure = False
            self.logger.error(
                "batch_processing_error",
                operation_type=operation_type.value,
                error=str(e),
                exc_info=True,
            )

        finally:
            flush_time_ms = (time.time() - start_time) * 1000
            self._batch_metrics.update_batch_stats(
                len(operations), flush_time_ms, successful_ops, failed_ops, partial_failure
            )

    async def _process_retry_queue(self) -> None:
        """Process operations in the retry queue."""
        retry_operations = list(self._retry_queue)
        self._retry_queue.clear()

        if not retry_operations:
            return

        # Group retry operations by type and process them
        operations_by_type = defaultdict(list)
        for op in retry_operations:
            operations_by_type[op.operation_type].append(op)

        for op_type, operations in operations_by_type.items():
            await asyncio.sleep(self.retry_delay_seconds)  # Delay before retry
            await self._process_operations_batch(op_type, operations)

    async def _process_evaluation_results_batch(self, operations: list[BatchOperation]) -> list[bool]:
        """Process a batch of evaluation result operations."""
        results = []

        try:
            # Aggregate all metrics for batch logging
            batch_metrics = {}

            for i, operation in enumerate(operations):
                result: EvaluationResult = operation.data

                try:
                    # Create metrics for this result with batch prefix
                    prefix = f"batch_{i}_"
                    metrics = {
                        f"{prefix}task_id": result.task_id,
                        f"{prefix}accuracy": result.final_accuracy,
                        f"{prefix}strategy": result.strategy_used,
                        f"{prefix}num_attempts": len(result.attempts),
                        f"{prefix}processing_time_ms": result.total_processing_time_ms,
                    }

                    # Add attempt-specific metrics
                    for j, attempt in enumerate(result.attempts):
                        attempt_prefix = f"{prefix}attempt_{j+1}"
                        metrics.update({
                            f"{attempt_prefix}/accuracy": attempt.pixel_accuracy.accuracy,
                            f"{attempt_prefix}/perfect_match": attempt.pixel_accuracy.perfect_match,
                            f"{attempt_prefix}/confidence": attempt.confidence_score,
                            f"{attempt_prefix}/processing_time_ms": attempt.processing_time_ms,
                        })

                        if attempt.error_category:
                            metrics[f"{attempt_prefix}/error_category"] = attempt.error_category.value

                    batch_metrics.update(metrics)
                    results.append(True)

                except Exception as e:
                    self.logger.error(
                        "batch_evaluation_result_error",
                        task_id=result.task_id,
                        error=str(e),
                    )
                    results.append(False)

            # Log all metrics in a single W&B call
            if batch_metrics:
                wandb.log(batch_metrics)

        except Exception as e:
            self.logger.error("batch_evaluation_results_error", error=str(e), exc_info=True)
            results = [False] * len(operations)

        return results

    async def _process_resource_usage_batch(self, operations: list[BatchOperation]) -> list[bool]:
        """Process a batch of resource usage operations."""
        results = []

        try:
            batch_metrics = {}

            for i, operation in enumerate(operations):
                usage: ResourceUsage = operation.data

                try:
                    prefix = f"batch_{i}_"
                    metrics = {
                        f"{prefix}resource/cpu_seconds": usage.cpu_seconds,
                        f"{prefix}resource/memory_mb": usage.memory_mb,
                        f"{prefix}resource/total_tokens": usage.total_tokens,
                        f"{prefix}resource/estimated_cost": usage.estimated_cost,
                        f"{prefix}resource/task_id": usage.task_id,
                    }

                    if usage.gpu_memory_mb is not None:
                        metrics[f"{prefix}resource/gpu_memory_mb"] = usage.gpu_memory_mb

                    # Log API calls breakdown
                    for api_name, count in usage.api_calls.items():
                        metrics[f"{prefix}resource/api_calls/{api_name}"] = count

                    batch_metrics.update(metrics)
                    results.append(True)

                except Exception as e:
                    self.logger.error(
                        "batch_resource_usage_error",
                        task_id=usage.task_id,
                        error=str(e),
                    )
                    results.append(False)

            # Log all metrics in a single W&B call
            if batch_metrics:
                wandb.log(batch_metrics)

        except Exception as e:
            self.logger.error("batch_resource_usage_error", error=str(e), exc_info=True)
            results = [False] * len(operations)

        return results

    async def _process_experiment_summary_batch(self, operations: list[BatchOperation]) -> list[bool]:
        """Process a batch of experiment summary operations."""
        # Experiment summaries are typically singular per experiment,
        # so we process them individually but within the same batch
        results = []

        for operation in operations:
            try:
                metrics: ExperimentMetrics = operation.data
                success = self.log_experiment_summary(metrics)  # Use sync method
                results.append(success)
            except Exception as e:
                self.logger.error("batch_experiment_summary_error", error=str(e))
                results.append(False)

        return results

    async def _process_custom_metrics_batch(self, operations: list[BatchOperation]) -> list[bool]:
        """Process a batch of custom metrics operations."""
        results = []

        try:
            batch_metrics = {}

            for i, operation in enumerate(operations):
                custom_metrics: dict[str, Any] = operation.data

                try:
                    # Add batch prefix to avoid key conflicts
                    prefixed_metrics = {f"batch_{i}_{k}": v for k, v in custom_metrics.items()}
                    batch_metrics.update(prefixed_metrics)
                    results.append(True)
                except Exception as e:
                    self.logger.error("batch_custom_metrics_error", error=str(e))
                    results.append(False)

            # Log all custom metrics in a single W&B call
            if batch_metrics:
                wandb.log(batch_metrics)

        except Exception as e:
            self.logger.error("batch_custom_metrics_error", error=str(e), exc_info=True)
            results = [False] * len(operations)

        return results

    # Async batch methods
    async def log_evaluation_result_async(self, result: EvaluationResult) -> bool:
        """Async version of log_evaluation_result that uses batching."""
        if not self.enable_batching:
            return self.log_evaluation_result(result)  # Fall back to sync

        operation = BatchOperation(
            operation_type=BatchOperationType.EVALUATION_RESULT,
            data=result,
        )
        await self._add_to_batch(operation)
        return True  # Queued successfully

    async def log_resource_usage_async(self, usage: ResourceUsage) -> bool:
        """Async version of log_resource_usage that uses batching."""
        if not self.enable_batching:
            return self.log_resource_usage(usage)  # Fall back to sync

        operation = BatchOperation(
            operation_type=BatchOperationType.RESOURCE_USAGE,
            data=usage,
        )
        await self._add_to_batch(operation)
        return True  # Queued successfully

    async def log_experiment_summary_async(self, metrics: ExperimentMetrics) -> bool:
        """Async version of log_experiment_summary that uses batching."""
        if not self.enable_batching:
            return self.log_experiment_summary(metrics)  # Fall back to sync

        operation = BatchOperation(
            operation_type=BatchOperationType.EXPERIMENT_SUMMARY,
            data=metrics,
        )
        await self._add_to_batch(operation)
        return True  # Queued successfully

    async def log_custom_metrics_async(self, metrics: dict[str, Any]) -> bool:
        """Log custom metrics using batching."""
        if not self.enable_batching:
            if self._current_run and self.config.mode != "disabled":
                try:
                    wandb.log(metrics)
                    return True
                except Exception as e:
                    self.logger.error("custom_metrics_log_error", error=str(e))
                    return False
            return False

        operation = BatchOperation(
            operation_type=BatchOperationType.CUSTOM_METRICS,
            data=metrics,
        )
        await self._add_to_batch(operation)
        return True  # Queued successfully

    def get_batch_metrics(self) -> dict[str, Any]:
        """Get current batch processing metrics."""
        return self._batch_metrics.to_dict()

    def get_queue_sizes(self) -> dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            "batch_queue": len(self._batch_queue),
            "retry_queue": len(self._retry_queue),
        }


# Global client instances
_wandb_client: WandBClient | None = None
_batched_wandb_client: BatchedWandBClient | None = None


def get_wandb_client() -> WandBClient:
    """Get the global W&B client instance.

    Returns:
        WandBClient instance
    """
    global _wandb_client
    if _wandb_client is None:
        _wandb_client = WandBClient()
    return _wandb_client


def get_batched_wandb_client(
    batch_size: int = 50,
    flush_interval_seconds: float = 5.0,
    **kwargs
) -> BatchedWandBClient:
    """Get the global batched W&B client instance.

    Args:
        batch_size: Maximum number of operations per batch
        flush_interval_seconds: Time interval between automatic flushes
        **kwargs: Additional arguments for BatchedWandBClient

    Returns:
        BatchedWandBClient instance
    """
    global _batched_wandb_client
    if _batched_wandb_client is None:
        _batched_wandb_client = BatchedWandBClient(
            batch_size=batch_size,
            flush_interval_seconds=flush_interval_seconds,
            **kwargs
        )
    return _batched_wandb_client
