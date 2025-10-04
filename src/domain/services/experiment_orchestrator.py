"""Experiment queue management system with priority scheduling and platform persistence.

This module implements a comprehensive experiment orchestration system that manages
experiment queues with priority-based scheduling, persists queue state across platform
switches, monitors queue status, and provides retry logic for failed experiments.

Features:
- Priority-based experiment scheduling (critical, high, normal, low, background)
- Queue persistence for platform rotation continuity
- Experiment status tracking and monitoring
- Exponential backoff retry logic for failed experiments
- Async patterns for efficient queue management
- Resource-aware experiment allocation
- Platform preference handling
"""

import asyncio
import heapq
import json
import logging
import random
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from src.domain.models import ResourceUsage, StrategyType

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment execution status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    SUSPENDED = "suspended"


class ExperimentPriority(IntEnum):
    """Experiment priority levels (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class RetryStrategy(Enum):
    """Retry strategy options for failed experiments."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"
    NONE = "none"


# Alias for backward compatibility
ExperimentRetryStrategy = RetryStrategy


@dataclass
class ExperimentResources:
    """Resource requirements and constraints for an experiment."""
    memory_gb: float = 8.0
    gpu_memory_gb: float = 12.0
    cpu_cores: int = 2
    max_runtime_hours: float = 4.0
    disk_space_gb: float = 10.0
    gpu_required: bool = True
    platform_preference: str | None = None  # kaggle, colab, paperspace
    min_python_version: str = "3.12"


@dataclass
class ExperimentConfig:
    """Configuration settings for an experiment."""
    name: str
    description: str = ""
    strategy_type: StrategyType = StrategyType.TEST_TIME_TRAINING
    model_size: str = "1B"  # 1B, 8B, etc.
    dataset_tasks: list[str] = field(default_factory=list)  # ARC task IDs
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 5
    checkpoint_interval_minutes: int = 30
    early_stopping_patience: int = 3
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    platform_preferences: list[str] = field(default_factory=list)  # Platform rotation preferences
    retry_strategy: "RetryStrategy" = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3


@dataclass
class ExperimentRetryConfig:
    """Retry configuration for failed experiments."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay_minutes: float = 5.0
    max_delay_minutes: float = 120.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_statuses: set[ExperimentStatus] = field(
        default_factory=lambda: {ExperimentStatus.FAILED}
    )


@dataclass
class ExperimentProgress:
    """Track experiment progress and metrics."""
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    elapsed_time_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    last_checkpoint_path: str | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def get_completion_percentage(self) -> float:
        """Calculate experiment completion percentage."""
        if self.total_steps > 0:
            return min(100.0, (self.current_step / self.total_steps) * 100.0)
        elif self.total_epochs > 0:
            return min(100.0, (self.current_epoch / self.total_epochs) * 100.0)
        return 0.0


@dataclass
class ExperimentResults:
    """Store experiment results and artifacts."""
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    final_loss: float = 0.0
    convergence_achieved: bool = False
    model_checkpoint_path: str | None = None
    training_history: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    resource_usage: ResourceUsage | None = None
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    error_details: str | None = None


@dataclass
class Experiment:
    """Complete experiment definition with tracking and results."""
    id: str
    config: ExperimentConfig
    resources: ExperimentResources
    retry_config: ExperimentRetryConfig
    priority: ExperimentPriority
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    platform: str | None = None
    retry_count: int = 0
    next_retry_at: datetime | None = None
    progress: ExperimentProgress = field(default_factory=ExperimentProgress)
    results: ExperimentResults = field(default_factory=ExperimentResults)
    created_by: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other) -> bool:
        """Support priority queue ordering."""
        if not isinstance(other, Experiment):
            return NotImplemented
        # Lower priority number = higher priority in queue
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)

    def can_retry(self) -> bool:
        """Check if experiment can be retried."""
        return (
            self.retry_count < self.retry_config.max_retries
            and self.status in self.retry_config.retry_on_statuses
            and (self.next_retry_at is None or datetime.now() >= self.next_retry_at)
        )

    def get_runtime_seconds(self) -> float | None:
        """Get experiment runtime in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return (end_time - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment to dictionary for serialization."""
        return {
            "id": self.id,
            "config": asdict(self.config),
            "resources": asdict(self.resources),
            "retry_config": asdict(self.retry_config),
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "platform": self.platform,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "progress": asdict(self.progress),
            "results": asdict(self.results),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Create experiment from dictionary."""
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])
        scheduled_at = datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        next_retry_at = datetime.fromisoformat(data["next_retry_at"]) if data.get("next_retry_at") else None

        # Parse nested objects
        config_data = data["config"]
        config = ExperimentConfig(
            name=config_data["name"],
            description=config_data.get("description", ""),
            strategy_type=StrategyType(config_data.get("strategy_type", "ttt")),
            model_size=config_data.get("model_size", "1B"),
            dataset_tasks=config_data.get("dataset_tasks", []),
            batch_size=config_data.get("batch_size", 4),
            learning_rate=config_data.get("learning_rate", 1e-4),
            max_epochs=config_data.get("max_epochs", 5),
            checkpoint_interval_minutes=config_data.get("checkpoint_interval_minutes", 30),
            early_stopping_patience=config_data.get("early_stopping_patience", 3),
            mixed_precision=config_data.get("mixed_precision", True),
            gradient_checkpointing=config_data.get("gradient_checkpointing", True),
            parameters=config_data.get("parameters", {}),
            tags=config_data.get("tags", []),
        )

        resources_data = data["resources"]
        resources = ExperimentResources(
            memory_gb=resources_data.get("memory_gb", 8.0),
            gpu_memory_gb=resources_data.get("gpu_memory_gb", 12.0),
            cpu_cores=resources_data.get("cpu_cores", 2),
            max_runtime_hours=resources_data.get("max_runtime_hours", 4.0),
            disk_space_gb=resources_data.get("disk_space_gb", 10.0),
            gpu_required=resources_data.get("gpu_required", True),
            platform_preference=resources_data.get("platform_preference"),
            min_python_version=resources_data.get("min_python_version", "3.12"),
        )

        retry_data = data["retry_config"]
        retry_config = ExperimentRetryConfig(
            strategy=RetryStrategy(retry_data.get("strategy", "exponential_backoff")),
            max_retries=retry_data.get("max_retries", 3),
            base_delay_minutes=retry_data.get("base_delay_minutes", 5.0),
            max_delay_minutes=retry_data.get("max_delay_minutes", 120.0),
            backoff_multiplier=retry_data.get("backoff_multiplier", 2.0),
            jitter=retry_data.get("jitter", True),
            retry_on_statuses={
                ExperimentStatus(status) for status in retry_data.get("retry_on_statuses", ["failed"])
            },
        )

        progress_data = data["progress"]
        progress = ExperimentProgress(
            current_epoch=progress_data.get("current_epoch", 0),
            total_epochs=progress_data.get("total_epochs", 0),
            current_step=progress_data.get("current_step", 0),
            total_steps=progress_data.get("total_steps", 0),
            loss=progress_data.get("loss", 0.0),
            accuracy=progress_data.get("accuracy", 0.0),
            learning_rate=progress_data.get("learning_rate", 0.0),
            elapsed_time_seconds=progress_data.get("elapsed_time_seconds", 0.0),
            estimated_remaining_seconds=progress_data.get("estimated_remaining_seconds"),
            memory_usage_mb=progress_data.get("memory_usage_mb", 0.0),
            gpu_memory_usage_mb=progress_data.get("gpu_memory_usage_mb", 0.0),
            last_checkpoint_path=progress_data.get("last_checkpoint_path"),
            custom_metrics=progress_data.get("custom_metrics", {}),
        )

        results_data = data["results"]
        results = ExperimentResults(
            final_accuracy=results_data.get("final_accuracy", 0.0),
            best_accuracy=results_data.get("best_accuracy", 0.0),
            final_loss=results_data.get("final_loss", 0.0),
            convergence_achieved=results_data.get("convergence_achieved", False),
            model_checkpoint_path=results_data.get("model_checkpoint_path"),
            training_history=results_data.get("training_history", []),
            predictions=results_data.get("predictions", []),
            resource_usage=results_data.get("resource_usage"),
            artifacts=results_data.get("artifacts", {}),
            evaluation_metrics=results_data.get("evaluation_metrics", {}),
            error_details=results_data.get("error_details"),
        )

        return cls(
            id=data["id"],
            config=config,
            resources=resources,
            retry_config=retry_config,
            priority=ExperimentPriority(data["priority"]),
            status=ExperimentStatus(data["status"]),
            created_at=created_at,
            updated_at=updated_at,
            scheduled_at=scheduled_at,
            started_at=started_at,
            completed_at=completed_at,
            platform=data.get("platform"),
            retry_count=data.get("retry_count", 0),
            next_retry_at=next_retry_at,
            progress=progress,
            results=results,
            created_by=data.get("created_by", "system"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueueStatistics:
    """Statistics about the experiment queue."""
    total_experiments: int
    pending_experiments: int
    queued_experiments: int
    running_experiments: int
    completed_experiments: int
    failed_experiments: int
    cancelled_experiments: int
    priority_distribution: dict[str, int]
    platform_distribution: dict[str, int]
    average_wait_time_minutes: float
    average_runtime_minutes: float
    success_rate: float
    retry_rate: float
    resource_utilization: dict[str, float]


class ExperimentOrchestrator:
    """
    Experiment queue management system with priority scheduling and persistence.

    This orchestrator manages a priority-based experiment queue with support for:
    - Priority scheduling (critical, high, normal, low, background)
    - Queue persistence across platform switches
    - Experiment monitoring and status reporting
    - Retry logic with exponential backoff
    - Resource-aware experiment allocation
    - Platform preference handling
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        auto_save_interval: float = 60.0,  # seconds
        cleanup_interval: float = 3600.0,  # seconds
        state_dir: str | Path | None = None,  # Alias for persistence_path
    ):
        """
        Initialize experiment orchestrator.

        Args:
            persistence_path: Path to persist queue state (default: ~/.arc-experiments)
            auto_save_interval: Interval for automatic queue state saving
            cleanup_interval: Interval for cleaning up old completed experiments
            state_dir: Alias for persistence_path for backward compatibility
        """
        # Use state_dir if provided, otherwise use persistence_path
        if state_dir is not None:
            self.persistence_path = Path(state_dir)
        else:
            self.persistence_path = persistence_path or Path.home() / ".arc-experiments"
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Queue management
        self._queue: list[Experiment] = []
        self._experiments: dict[str, Experiment] = {}
        self._lock = threading.RLock()

        # Event callbacks
        self._callbacks: dict[str, list[Callable[[Experiment], None]]] = {
            "experiment_added": [],
            "experiment_started": [],
            "experiment_progress": [],
            "experiment_completed": [],
            "experiment_failed": [],
            "experiment_cancelled": [],
            "experiment_retry": [],
        }

        # Background tasks
        self._auto_save_interval = auto_save_interval
        self._cleanup_interval = cleanup_interval
        self._running = False
        self._background_tasks: list[asyncio.Task] = []

        logger.info(f"Experiment orchestrator initialized with persistence at {self.persistence_path}")

        # Load persisted state
        self._load_queue_state()

    async def start(self):
        """Start the experiment orchestrator and background tasks."""
        if self._running:
            logger.warning("Experiment orchestrator is already running")
            return

        self._running = True
        logger.info("Starting experiment orchestrator...")

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._auto_save_task()),
            asyncio.create_task(self._cleanup_task()),
            asyncio.create_task(self._retry_monitor_task()),
        ]

        logger.info("Experiment orchestrator started with background tasks")

    async def stop(self):
        """Stop the experiment orchestrator and background tasks."""
        if not self._running:
            return

        logger.info("Stopping experiment orchestrator...")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Save final state
        await self._save_queue_state()

        logger.info("Experiment orchestrator stopped")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        strategy_type: StrategyType = StrategyType.TEST_TIME_TRAINING,
        model_size: str = "1B",
        dataset_tasks: list[str] | None = None,
        priority: ExperimentPriority = ExperimentPriority.NORMAL,
        resources: ExperimentResources | None = None,
        retry_config: ExperimentRetryConfig | None = None,
        command: str | None = None,  # For backward compatibility
        **kwargs,
    ) -> Experiment | str:
        """
        Create a new experiment with the given configuration.

        Args:
            name: Experiment name
            description: Experiment description
            strategy_type: Strategy to use for solving
            model_size: Model size (1B, 8B, etc.)
            dataset_tasks: List of ARC task IDs to train on
            priority: Experiment priority level
            resources: Resource requirements
            retry_config: Retry configuration
            **kwargs: Additional configuration parameters

        Returns:
            Created experiment instance
        """
        experiment_id = str(uuid.uuid4())

        # Handle command parameter for backward compatibility
        parameters = kwargs.get("parameters", {})
        if command is not None:
            parameters["command"] = command

        config = ExperimentConfig(
            name=name,
            description=description,
            strategy_type=strategy_type,
            model_size=model_size,
            dataset_tasks=dataset_tasks or [],
            batch_size=kwargs.get("batch_size", 4),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            max_epochs=kwargs.get("max_epochs", 5),
            checkpoint_interval_minutes=kwargs.get("checkpoint_interval_minutes", 30),
            early_stopping_patience=kwargs.get("early_stopping_patience", 3),
            mixed_precision=kwargs.get("mixed_precision", True),
            gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
            parameters=parameters,
            tags=kwargs.get("tags", []),
        )

        if resources is None:
            resources = ExperimentResources(
                memory_gb=kwargs.get("memory_gb", 8.0),
                gpu_memory_gb=kwargs.get("gpu_memory_gb", 12.0),
                cpu_cores=kwargs.get("cpu_cores", 2),
                max_runtime_hours=kwargs.get("max_runtime_hours", 4.0),
                disk_space_gb=kwargs.get("disk_space_gb", 10.0),
                gpu_required=kwargs.get("gpu_required", True),
                platform_preference=kwargs.get("platform_preference"),
                min_python_version=kwargs.get("min_python_version", "3.12"),
            )

        if retry_config is None:
            retry_config = ExperimentRetryConfig(
                max_retries=kwargs.get("max_retries", 3),
                base_delay_minutes=kwargs.get("base_delay_minutes", 5.0),
            )

        experiment = Experiment(
            id=experiment_id,
            config=config,
            resources=resources,
            retry_config=retry_config,
            priority=priority,
            created_by=kwargs.get("created_by", "system"),
            metadata=kwargs.get("metadata", {}),
        )

        logger.info(f"Created experiment {experiment_id}: {name}")

        # Store experiment in the experiments dictionary for legacy API support
        self._experiments[experiment_id] = experiment

        # Return ID for backward compatibility when command is provided, otherwise return experiment
        if command is not None:
            return experiment_id
        return experiment

    async def add_experiment(self, experiment: Experiment | str) -> bool:
        """
        Add experiment to the queue.

        Args:
            experiment: Experiment to add to queue, or experiment ID string

        Returns:
            True if added successfully, False otherwise
        """
        try:
            with self._lock:
                # Handle both Experiment objects and string IDs
                if isinstance(experiment, str):
                    experiment_id = experiment
                    if experiment_id not in self._experiments:
                        logger.error(f"Experiment {experiment_id} not found")
                        return False
                    experiment = self._experiments[experiment_id]
                else:
                    experiment_id = experiment.id

                if experiment_id in self._experiments and experiment.status != ExperimentStatus.PENDING:
                    logger.warning(f"Experiment {experiment_id} already exists in queue")
                    return False

                # Add to priority queue and experiments dictionary
                heapq.heappush(self._queue, experiment)
                self._experiments[experiment_id] = experiment

                # Update status
                experiment.status = ExperimentStatus.QUEUED
                experiment.updated_at = datetime.now()

                logger.info(
                    f"Added experiment {experiment_id} to queue with priority {experiment.priority.name}"
                )

                # Trigger callbacks
                await self._trigger_callbacks("experiment_added", experiment)

                return True

        except Exception as e:
            experiment_id = experiment.id if hasattr(experiment, 'id') else str(experiment)
            logger.error(f"Failed to add experiment {experiment_id}: {str(e)}")
            return False

    async def get_next_experiment(
        self,
        platform_filter: str | None = None,
        resource_filter: ExperimentResources | None = None,
    ) -> Experiment | None:
        """
        Get the next experiment from the queue based on priority and filters.

        Args:
            platform_filter: Filter by platform preference
            resource_filter: Filter by available resources

        Returns:
            Next experiment to run, or None if no suitable experiments
        """
        with self._lock:
            if not self._queue:
                return None

            # Find suitable experiments
            suitable_experiments = []
            temp_queue = []

            while self._queue:
                experiment = heapq.heappop(self._queue)

                # Check if experiment is suitable
                if self._is_experiment_suitable(experiment, platform_filter, resource_filter):
                    suitable_experiments.append(experiment)
                else:
                    temp_queue.append(experiment)

            # Restore non-suitable experiments to queue
            for experiment in temp_queue:
                heapq.heappush(self._queue, experiment)

            # Return highest priority suitable experiment
            if suitable_experiments:
                next_experiment = min(
                    suitable_experiments, key=lambda e: (e.priority.value, e.created_at)
                )

                # Put back non-selected suitable experiments
                for experiment in suitable_experiments:
                    if experiment.id != next_experiment.id:
                        heapq.heappush(self._queue, experiment)

                # Update experiment status
                next_experiment.status = ExperimentStatus.RUNNING
                next_experiment.scheduled_at = datetime.now()
                next_experiment.started_at = datetime.now()
                next_experiment.updated_at = datetime.now()

                logger.info(f"Scheduled experiment {next_experiment.id} for execution")

                # Trigger callbacks
                await self._trigger_callbacks("experiment_started", next_experiment)

                return next_experiment

            return None

    def _is_experiment_suitable(
        self,
        experiment: Experiment,
        platform_filter: str | None = None,
        resource_filter: ExperimentResources | None = None,
    ) -> bool:
        """Check if experiment is suitable based on filters."""
        # Check status
        if experiment.status not in [ExperimentStatus.QUEUED, ExperimentStatus.RETRYING]:
            return False

        # Check retry timing
        if experiment.status == ExperimentStatus.RETRYING and not experiment.can_retry():
            return False

        # Check platform preference
        if platform_filter and experiment.resources.platform_preference:
            if experiment.resources.platform_preference != platform_filter:
                return False

        # Check resource requirements
        if resource_filter:
            if (
                experiment.resources.memory_gb > resource_filter.memory_gb
                or experiment.resources.gpu_memory_gb > resource_filter.gpu_memory_gb
                or experiment.resources.cpu_cores > resource_filter.cpu_cores
            ):
                return False

        return True

    async def update_experiment_progress(
        self,
        experiment_id: str,
        progress: ExperimentProgress,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update experiment progress and metrics.

        Args:
            experiment_id: Experiment ID
            progress: Updated progress information
            metadata: Additional metadata updates

        Returns:
            True if updated successfully
        """
        try:
            with self._lock:
                if experiment_id not in self._experiments:
                    logger.error(f"Experiment {experiment_id} not found for progress update")
                    return False

                experiment = self._experiments[experiment_id]
                experiment.progress = progress
                experiment.updated_at = datetime.now()

                if metadata:
                    experiment.metadata.update(metadata)

                logger.debug(
                    f"Updated progress for experiment {experiment_id}: "
                    f"{progress.get_completion_percentage():.1f}% complete"
                )

                # Trigger callbacks
                await self._trigger_callbacks("experiment_progress", experiment)

                return True

        except Exception as e:
            logger.error(f"Failed to update progress for experiment {experiment_id}: {str(e)}")
            return False

    async def complete_experiment(
        self,
        experiment_id: str,
        results: ExperimentResults,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> bool:
        """
        Mark experiment as completed with results.

        Args:
            experiment_id: Experiment ID
            results: Final experiment results
            status: Final status (COMPLETED, FAILED, CANCELLED)

        Returns:
            True if completed successfully
        """
        try:
            with self._lock:
                if experiment_id not in self._experiments:
                    logger.error(f"Experiment {experiment_id} not found for completion")
                    return False

                experiment = self._experiments[experiment_id]
                experiment.status = status
                experiment.results = results
                experiment.completed_at = datetime.now()
                experiment.updated_at = datetime.now()

                logger.info(
                    f"Experiment {experiment_id} completed with status {status.value}"
                )

                # Trigger callbacks based on final status
                if status == ExperimentStatus.COMPLETED:
                    await self._trigger_callbacks("experiment_completed", experiment)
                elif status == ExperimentStatus.FAILED:
                    await self._trigger_callbacks("experiment_failed", experiment)
                elif status == ExperimentStatus.CANCELLED:
                    await self._trigger_callbacks("experiment_cancelled", experiment)

                return True

        except Exception as e:
            logger.error(f"Failed to complete experiment {experiment_id}: {str(e)}")
            return False

    async def retry_experiment(self, experiment_id: str, delay_override: float | None = None) -> bool:
        """
        Retry a failed experiment if retries are available.

        Args:
            experiment_id: Experiment ID to retry
            delay_override: Override default retry delay (minutes)

        Returns:
            True if retry was scheduled successfully
        """
        try:
            with self._lock:
                if experiment_id not in self._experiments:
                    logger.error(f"Experiment {experiment_id} not found for retry")
                    return False

                experiment = self._experiments[experiment_id]

                if not experiment.can_retry():
                    logger.warning(
                        f"Experiment {experiment_id} cannot be retried "
                        f"(retries: {experiment.retry_count}/{experiment.retry_config.max_retries})"
                    )
                    return False

                # Calculate retry delay
                if delay_override is not None:
                    delay_minutes = delay_override
                else:
                    delay_minutes = self._calculate_retry_delay(experiment)

                # Schedule retry
                experiment.retry_count += 1
                experiment.status = ExperimentStatus.RETRYING
                experiment.next_retry_at = datetime.now() + timedelta(minutes=delay_minutes)
                experiment.started_at = None
                experiment.completed_at = None
                experiment.updated_at = datetime.now()

                # Reset progress and results
                experiment.progress = ExperimentProgress()
                experiment.results.error_details = None

                # Add back to queue
                heapq.heappush(self._queue, experiment)

                logger.info(
                    f"Scheduled retry {experiment.retry_count} for experiment {experiment_id} "
                    f"in {delay_minutes:.1f} minutes"
                )

                # Trigger callbacks
                await self._trigger_callbacks("experiment_retry", experiment)

                return True

        except Exception as e:
            logger.error(f"Failed to retry experiment {experiment_id}: {str(e)}")
            return False

    def _calculate_retry_delay(self, experiment: Experiment) -> float:
        """Calculate retry delay based on retry strategy."""
        config = experiment.retry_config

        if config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = config.base_delay_minutes
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay_minutes * (experiment.retry_count + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay_minutes * (config.backoff_multiplier ** experiment.retry_count)
        else:
            delay = config.base_delay_minutes

        # Apply maximum delay limit
        delay = min(delay, config.max_delay_minutes)

        # Add jitter if enabled
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)

    async def cancel_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """
        Cancel a queued or running experiment.

        Args:
            experiment_id: Experiment ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        try:
            with self._lock:
                if experiment_id not in self._experiments:
                    logger.error(f"Experiment {experiment_id} not found for cancellation")
                    return False

                experiment = self._experiments[experiment_id]

                if experiment.status in [
                    ExperimentStatus.COMPLETED,
                    ExperimentStatus.CANCELLED,
                ]:
                    logger.warning(
                        f"Experiment {experiment_id} cannot be cancelled "
                        f"(current status: {experiment.status.value})"
                    )
                    return False

                # Remove from queue if queued
                if experiment.status in [ExperimentStatus.QUEUED, ExperimentStatus.RETRYING]:
                    self._queue = [e for e in self._queue if e.id != experiment_id]
                    heapq.heapify(self._queue)

                # Update experiment
                experiment.status = ExperimentStatus.CANCELLED
                experiment.completed_at = datetime.now()
                experiment.updated_at = datetime.now()

                if reason:
                    experiment.results.error_details = f"Cancelled: {reason}"

                logger.info(f"Cancelled experiment {experiment_id}: {reason}")

                # Trigger callbacks
                await self._trigger_callbacks("experiment_cancelled", experiment)

                return True

        except Exception as e:
            logger.error(f"Failed to cancel experiment {experiment_id}: {str(e)}")
            return False

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment if found, None otherwise
        """
        with self._lock:
            return self._experiments.get(experiment_id)

    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus | None:
        """
        Get experiment status by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment status if found, None otherwise
        """
        experiment = self.get_experiment(experiment_id)
        return experiment.status if experiment else None

    def list_experiments(
        self,
        status_filter: ExperimentStatus | None = None,
        priority_filter: ExperimentPriority | None = None,
        platform_filter: str | None = None,
        limit: int | None = None,
    ) -> list[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            status_filter: Filter by experiment status
            priority_filter: Filter by experiment priority
            platform_filter: Filter by platform
            limit: Maximum number of experiments to return

        Returns:
            List of matching experiments
        """
        with self._lock:
            experiments = list(self._experiments.values())

            # Apply filters
            if status_filter:
                experiments = [e for e in experiments if e.status == status_filter]

            if priority_filter:
                experiments = [e for e in experiments if e.priority == priority_filter]

            if platform_filter:
                experiments = [e for e in experiments if e.platform == platform_filter]

            # Sort by priority and creation time
            experiments.sort(key=lambda e: (e.priority.value, e.created_at))

            # Apply limit
            if limit:
                experiments = experiments[:limit]

            return experiments

    def get_queue_statistics(self) -> QueueStatistics:
        """
        Get comprehensive queue statistics.

        Returns:
            Queue statistics
        """
        with self._lock:
            experiments = list(self._experiments.values())

            # Count by status
            status_counts = {}
            for status in ExperimentStatus:
                status_counts[status] = len([e for e in experiments if e.status == status])

            # Count by priority
            priority_counts = {}
            for priority in ExperimentPriority:
                priority_counts[priority.name] = len([e for e in experiments if e.priority == priority])

            # Count by platform
            platform_counts = {}
            for experiment in experiments:
                platform = experiment.platform or "unassigned"
                platform_counts[platform] = platform_counts.get(platform, 0) + 1

            # Calculate timing statistics
            completed_experiments = [
                e for e in experiments if e.status == ExperimentStatus.COMPLETED
            ]

            wait_times = []
            runtimes = []

            for experiment in completed_experiments:
                if experiment.scheduled_at and experiment.created_at:
                    wait_time = (experiment.scheduled_at - experiment.created_at).total_seconds() / 60
                    wait_times.append(wait_time)

                runtime = experiment.get_runtime_seconds()
                if runtime:
                    runtimes.append(runtime / 60)  # Convert to minutes

            avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
            avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0

            # Calculate success and retry rates
            total_completed = (
                status_counts[ExperimentStatus.COMPLETED]
                + status_counts[ExperimentStatus.FAILED]
                + status_counts[ExperimentStatus.CANCELLED]
            )

            success_rate = (
                status_counts[ExperimentStatus.COMPLETED] / total_completed
                if total_completed > 0
                else 0.0
            )

            retry_experiments = [e for e in experiments if e.retry_count > 0]
            retry_rate = len(retry_experiments) / len(experiments) if experiments else 0.0

            # Resource utilization (simplified)
            resource_utilization = {
                "memory_gb": sum(e.resources.memory_gb for e in experiments if e.status == ExperimentStatus.RUNNING),
                "gpu_memory_gb": sum(e.resources.gpu_memory_gb for e in experiments if e.status == ExperimentStatus.RUNNING),
                "cpu_cores": sum(e.resources.cpu_cores for e in experiments if e.status == ExperimentStatus.RUNNING),
            }

            return QueueStatistics(
                total_experiments=len(experiments),
                pending_experiments=status_counts[ExperimentStatus.PENDING],
                queued_experiments=status_counts[ExperimentStatus.QUEUED],
                running_experiments=status_counts[ExperimentStatus.RUNNING],
                completed_experiments=status_counts[ExperimentStatus.COMPLETED],
                failed_experiments=status_counts[ExperimentStatus.FAILED],
                cancelled_experiments=status_counts[ExperimentStatus.CANCELLED],
                priority_distribution=priority_counts,
                platform_distribution=platform_counts,
                average_wait_time_minutes=avg_wait_time,
                average_runtime_minutes=avg_runtime,
                success_rate=success_rate,
                retry_rate=retry_rate,
                resource_utilization=resource_utilization,
            )

    def register_callback(self, event_type: str, callback: Callable[[Experiment], None]) -> bool:
        """
        Register callback for experiment events.

        Args:
            event_type: Event type to listen for
            callback: Callback function to register

        Returns:
            True if registered successfully
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
            logger.debug(f"Registered callback for {event_type} events")
            return True
        else:
            logger.error(f"Unknown event type: {event_type}")
            return False

    async def _trigger_callbacks(self, event_type: str, experiment: Experiment):
        """Trigger callbacks for an event type."""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(experiment)
                else:
                    callback(experiment)
            except Exception as e:
                logger.error(
                    f"Callback failed for {event_type} event on experiment {experiment.id}: {str(e)}"
                )

    async def _auto_save_task(self):
        """Background task for automatic queue state saving."""
        while self._running:
            try:
                await asyncio.sleep(self._auto_save_interval)
                await self._save_queue_state()
                logger.debug("Auto-saved queue state")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save task failed: {str(e)}")

    async def _cleanup_task(self):
        """Background task for cleaning up old experiments."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_old_experiments()
                logger.debug("Performed experiment cleanup")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")

    async def _retry_monitor_task(self):
        """Background task for monitoring and processing retries."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                await self._process_pending_retries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry monitor task failed: {str(e)}")

    async def _process_pending_retries(self):
        """Process experiments that are ready to be retried."""
        current_time = datetime.now()

        with self._lock:
            for experiment in list(self._experiments.values()):
                if (
                    experiment.status == ExperimentStatus.RETRYING
                    and experiment.next_retry_at
                    and current_time >= experiment.next_retry_at
                ):
                    # Move experiment back to queued status
                    experiment.status = ExperimentStatus.QUEUED
                    experiment.next_retry_at = None
                    experiment.updated_at = current_time

                    # Add back to priority queue
                    heapq.heappush(self._queue, experiment)

                    logger.info(f"Experiment {experiment.id} retry is now available")

    async def _cleanup_old_experiments(self, older_than_days: int = 30):
        """Clean up old completed experiments."""
        cleanup_threshold = datetime.now() - timedelta(days=older_than_days)
        cleaned_count = 0

        with self._lock:
            experiments_to_remove = []

            for experiment_id, experiment in self._experiments.items():
                if (
                    experiment.status
                    in [
                        ExperimentStatus.COMPLETED,
                        ExperimentStatus.CANCELLED,
                    ]
                    and experiment.completed_at
                    and experiment.completed_at < cleanup_threshold
                ):
                    experiments_to_remove.append(experiment_id)

            # Remove old experiments
            for experiment_id in experiments_to_remove:
                del self._experiments[experiment_id]
                cleaned_count += 1

            # Rebuild priority queue
            self._queue = [e for e in self._queue if e.id in self._experiments]
            heapq.heapify(self._queue)

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old experiments")

        return cleaned_count

    async def _save_queue_state(self):
        """Save queue state to persistent storage."""
        try:
            state_file = self.persistence_path / "orchestrator_state.json"

            # Create serializable state
            with self._lock:
                state = {
                    "version": "1.0",
                    "saved_at": datetime.now().isoformat(),
                    "experiments": {
                        exp_id: experiment.to_dict()
                        for exp_id, experiment in self._experiments.items()
                    },
                }

            # Write to temporary file first, then rename for atomicity
            temp_file = state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            temp_file.replace(state_file)
            logger.debug(f"Saved queue state with {len(self._experiments)} experiments")

        except Exception as e:
            logger.error(f"Failed to save queue state: {str(e)}")

    def _load_queue_state(self):
        """Load queue state from persistent storage."""
        try:
            state_file = self.persistence_path / "orchestrator_state.json"

            if not state_file.exists():
                logger.info("No persisted queue state found")
                return

            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)

            # Load experiments
            loaded_count = 0
            with self._lock:
                for exp_id, exp_data in state.get("experiments", {}).items():
                    try:
                        experiment = Experiment.from_dict(exp_data)
                        self._experiments[exp_id] = experiment

                        # Add queued experiments back to priority queue
                        if experiment.status in [
                            ExperimentStatus.QUEUED,
                            ExperimentStatus.RETRYING,
                        ]:
                            heapq.heappush(self._queue, experiment)

                        loaded_count += 1

                    except Exception as e:
                        logger.error(f"Failed to load experiment {exp_id}: {str(e)}")

            logger.info(f"Loaded {loaded_count} experiments from persistent state")

        except Exception as e:
            logger.error(f"Failed to load queue state: {str(e)}")


# Singleton instance for global access
_orchestrator_instance: ExperimentOrchestrator | None = None


def get_experiment_orchestrator() -> ExperimentOrchestrator:
    """Get singleton experiment orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ExperimentOrchestrator()
    return _orchestrator_instance


# Convenience functions for common operations
def create_quick_experiment(
    name: str,
    model_size: str = "1B",
    dataset_tasks: list[str] | None = None,
    priority: ExperimentPriority = ExperimentPriority.NORMAL,
    **kwargs,
) -> Experiment:
    """Create an experiment with sensible defaults."""
    orchestrator = get_experiment_orchestrator()
    return orchestrator.create_experiment(
        name=name,
        model_size=model_size,
        dataset_tasks=dataset_tasks or [],
        priority=priority,
        **kwargs,
    )


async def submit_experiment(experiment: Experiment) -> bool:
    """Submit an experiment to the queue."""
    orchestrator = get_experiment_orchestrator()
    return await orchestrator.add_experiment(experiment)


async def get_queue_status() -> QueueStatistics:
    """Get current queue statistics."""
    orchestrator = get_experiment_orchestrator()
    return orchestrator.get_queue_statistics()
