"""Experiment queue management with priority scheduling.

This module provides experiment queuing, priority scheduling, and queue persistence
for platform rotation automation.
"""

import heapq
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    RETRYING = "retrying"


class ExperimentPriority(IntEnum):
    """Experiment priority levels (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_id: str
    name: str
    model_size: str  # e.g., "1B", "8B"
    dataset_tasks: list[int]  # ARC task IDs
    max_runtime_minutes: int
    checkpoint_interval_minutes: int
    max_memory_gb: float
    gpu_required: bool = True
    platform_preference: str | None = None  # "kaggle", "colab", "paperspace"
    parameters: dict[str, Any] = None
    retry_config: dict[str, Any] = None


@dataclass
class ExperimentMetadata:
    """Experiment metadata."""
    created_at: datetime
    updated_at: datetime
    submitted_by: str = "system"
    tags: list[str] = None
    description: str = ""
    estimated_duration_minutes: int | None = None
    resource_requirements: dict[str, Any] = None


@dataclass
class ExperimentJob:
    """Complete experiment job definition."""
    id: str
    config: ExperimentConfig
    priority: ExperimentPriority
    status: ExperimentStatus
    metadata: ExperimentMetadata
    created_at: datetime
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    platform: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    last_error: str | None = None
    progress: dict[str, Any] = None
    results: dict[str, Any] = None

    def __lt__(self, other):
        """Support priority queue ordering."""
        if not isinstance(other, ExperimentJob):
            return NotImplemented
        # Lower priority number = higher priority in queue
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'config': asdict(self.config),
            'priority': self.priority.value,
            'status': self.status.value,
            'metadata': asdict(self.metadata),
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'platform': self.platform,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'last_error': self.last_error,
            'progress': self.progress or {},
            'results': self.results or {}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ExperimentJob':
        """Create instance from dictionary."""
        # Parse config
        config_data = data['config']
        config = ExperimentConfig(
            experiment_id=config_data['experiment_id'],
            name=config_data['name'],
            model_size=config_data['model_size'],
            dataset_tasks=config_data['dataset_tasks'],
            max_runtime_minutes=config_data['max_runtime_minutes'],
            checkpoint_interval_minutes=config_data['checkpoint_interval_minutes'],
            max_memory_gb=config_data['max_memory_gb'],
            gpu_required=config_data.get('gpu_required', True),
            platform_preference=config_data.get('platform_preference'),
            parameters=config_data.get('parameters', {}),
            retry_config=config_data.get('retry_config', {})
        )

        # Parse metadata
        metadata_data = data['metadata']
        metadata = ExperimentMetadata(
            created_at=datetime.fromisoformat(metadata_data['created_at']),
            updated_at=datetime.fromisoformat(metadata_data['updated_at']),
            submitted_by=metadata_data.get('submitted_by', 'system'),
            tags=metadata_data.get('tags', []),
            description=metadata_data.get('description', ''),
            estimated_duration_minutes=metadata_data.get('estimated_duration_minutes'),
            resource_requirements=metadata_data.get('resource_requirements', {})
        )

        return cls(
            id=data['id'],
            config=config,
            priority=ExperimentPriority(data['priority']),
            status=ExperimentStatus(data['status']),
            metadata=metadata,
            created_at=datetime.fromisoformat(data['created_at']),
            scheduled_at=datetime.fromisoformat(data['scheduled_at']) if data['scheduled_at'] else None,
            started_at=datetime.fromisoformat(data['started_at']) if data['started_at'] else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
            platform=data.get('platform'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            last_error=data.get('last_error'),
            progress=data.get('progress', {}),
            results=data.get('results', {})
        )


class ExperimentQueue:
    """Priority queue for experiment scheduling."""

    def __init__(self, persistence_path: Path | None = None):
        """Initialize experiment queue.

        Args:
            persistence_path: Path to persist queue state
        """
        self.persistence_path = persistence_path or Path.home() / ".arc-queue"
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        self._queue: list[ExperimentJob] = []
        self._jobs: dict[str, ExperimentJob] = {}
        self._lock = Lock()
        self._callbacks: dict[str, list[Callable]] = {
            'job_added': [],
            'job_updated': [],
            'job_completed': [],
            'job_failed': []
        }

        self.logger = structlog.get_logger('experiment_queue')

        # Load persisted state
        self._load_queue()

    def add_job(self, job: ExperimentJob) -> bool:
        """Add experiment job to queue.

        Args:
            job: Experiment job to add

        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                if job.id in self._jobs:
                    self.logger.warning("job_already_exists", job_id=job.id)
                    return False

                # Add to priority queue
                heapq.heappush(self._queue, job)
                self._jobs[job.id] = job

                # Update status
                job.status = ExperimentStatus.QUEUED
                job.metadata.updated_at = datetime.now()

                self.logger.info("job_added", job_id=job.id, priority=job.priority.name)

                # Persist state
                self._persist_queue()

                # Notify callbacks
                self._trigger_callbacks('job_added', job)

                return True

        except Exception as e:
            self.logger.error("job_add_failed", job_id=job.id, error=str(e))
            return False

    def get_next_job(self, platform_filter: str | None = None) -> ExperimentJob | None:
        """Get next job from queue based on priority.

        Args:
            platform_filter: Optional platform filter

        Returns:
            Next experiment job or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None

            # Find suitable job considering platform preference
            suitable_jobs = []
            temp_queue = []

            while self._queue:
                job = heapq.heappop(self._queue)

                # Check if job is suitable
                if (job.status == ExperimentStatus.QUEUED and
                    (not platform_filter or
                     job.config.platform_preference == platform_filter or
                     job.config.platform_preference is None)):
                    suitable_jobs.append(job)
                else:
                    temp_queue.append(job)

            # Restore remaining jobs to queue
            for job in temp_queue:
                heapq.heappush(self._queue, job)

            # Return highest priority suitable job
            if suitable_jobs:
                # Sort by priority and return best match
                next_job = min(suitable_jobs, key=lambda j: (j.priority.value, j.created_at))

                # Put back non-selected suitable jobs
                for job in suitable_jobs:
                    if job.id != next_job.id:
                        heapq.heappush(self._queue, job)

                return next_job

            return None

    def update_job_status(self, job_id: str, status: ExperimentStatus,
                         metadata: dict[str, Any] | None = None) -> bool:
        """Update job status and metadata.

        Args:
            job_id: Job ID to update
            status: New status
            metadata: Additional metadata updates

        Returns:
            True if updated successfully
        """
        try:
            with self._lock:
                if job_id not in self._jobs:
                    self.logger.error("job_not_found", job_id=job_id)
                    return False

                job = self._jobs[job_id]
                old_status = job.status
                job.status = status
                job.metadata.updated_at = datetime.now()

                # Update timestamps based on status
                if status == ExperimentStatus.RUNNING and old_status != ExperimentStatus.RUNNING:
                    job.started_at = datetime.now()
                elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                    job.completed_at = datetime.now()

                # Update metadata if provided
                if metadata:
                    if 'progress' in metadata:
                        job.progress = metadata['progress']
                    if 'results' in metadata:
                        job.results = metadata['results']
                    if 'last_error' in metadata:
                        job.last_error = metadata['last_error']
                    if 'platform' in metadata:
                        job.platform = metadata['platform']

                self.logger.info("job_status_updated",
                                job_id=job_id,
                                old_status=old_status.value,
                                new_status=status.value)

                # Persist changes
                self._persist_queue()

                # Notify callbacks
                callback_type = 'job_updated'
                if status == ExperimentStatus.COMPLETED:
                    callback_type = 'job_completed'
                elif status == ExperimentStatus.FAILED:
                    callback_type = 'job_failed'

                self._trigger_callbacks(callback_type, job)

                return True

        except Exception as e:
            self.logger.error("job_status_update_failed", job_id=job_id, error=str(e))
            return False

    def get_job(self, job_id: str) -> ExperimentJob | None:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Experiment job or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status_filter: ExperimentStatus | None = None,
                  platform_filter: str | None = None) -> list[ExperimentJob]:
        """List jobs with optional filtering.

        Args:
            status_filter: Optional status filter
            platform_filter: Optional platform filter

        Returns:
            List of matching jobs
        """
        with self._lock:
            jobs = list(self._jobs.values())

            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]

            if platform_filter:
                jobs = [job for job in jobs if job.platform == platform_filter]

            # Sort by priority and creation time
            jobs.sort(key=lambda j: (j.priority.value, j.created_at))
            return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        return self.update_job_status(job_id, ExperimentStatus.CANCELLED)

    def retry_job(self, job_id: str) -> bool:
        """Retry a failed job if retries are available.

        Args:
            job_id: Job ID to retry

        Returns:
            True if retry initiated successfully
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                self.logger.error("job_not_found_for_retry", job_id=job_id)
                return False

            if job.retry_count >= job.max_retries:
                self.logger.warning("max_retries_exceeded", job_id=job_id)
                return False

            if job.status not in [ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                self.logger.warning("job_not_retryable", job_id=job_id, status=job.status.value)
                return False

            # Increment retry count and reset job
            job.retry_count += 1
            job.status = ExperimentStatus.QUEUED
            job.started_at = None
            job.completed_at = None
            job.last_error = None
            job.metadata.updated_at = datetime.now()

            # Add back to priority queue
            heapq.heappush(self._queue, job)

            self.logger.info("job_retry_initiated", job_id=job_id, retry_count=job.retry_count)

            # Persist changes
            self._persist_queue()

            return True

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            stats = {
                'total_jobs': len(self._jobs),
                'queued_jobs': len([j for j in self._jobs.values() if j.status == ExperimentStatus.QUEUED]),
                'running_jobs': len([j for j in self._jobs.values() if j.status == ExperimentStatus.RUNNING]),
                'completed_jobs': len([j for j in self._jobs.values() if j.status == ExperimentStatus.COMPLETED]),
                'failed_jobs': len([j for j in self._jobs.values() if j.status == ExperimentStatus.FAILED]),
                'cancelled_jobs': len([j for j in self._jobs.values() if j.status == ExperimentStatus.CANCELLED]),
                'priority_distribution': {},
                'platform_distribution': {}
            }

            # Priority distribution
            for priority in ExperimentPriority:
                count = len([j for j in self._jobs.values() if j.priority == priority])
                stats['priority_distribution'][priority.name] = count

            # Platform distribution
            platforms = {j.platform for j in self._jobs.values() if j.platform}
            for platform in platforms:
                count = len([j for j in self._jobs.values() if j.platform == platform])
                stats['platform_distribution'][platform] = count

            return stats

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for queue events.

        Args:
            event_type: Event type (job_added, job_updated, job_completed, job_failed)
            callback: Callback function
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, job: ExperimentJob):
        """Trigger callbacks for an event.

        Args:
            event_type: Event type
            job: Job that triggered the event
        """
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(job)
            except Exception as e:
                self.logger.error("callback_failed",
                                event_type=event_type,
                                job_id=job.id,
                                error=str(e))

    def _persist_queue(self):
        """Persist queue state to disk."""
        try:
            queue_file = self.persistence_path / "queue_state.json"

            # Serialize all jobs
            serialized_jobs = {job_id: job.to_dict() for job_id, job in self._jobs.items()}

            with open(queue_file, 'w') as f:
                json.dump(serialized_jobs, f, indent=2)

            self.logger.debug("queue_persisted", job_count=len(self._jobs))

        except Exception as e:
            self.logger.error("queue_persistence_failed", error=str(e))

    def _load_queue(self):
        """Load queue state from disk."""
        try:
            queue_file = self.persistence_path / "queue_state.json"

            if not queue_file.exists():
                self.logger.info("no_persisted_queue_found")
                return

            with open(queue_file) as f:
                serialized_jobs = json.load(f)

            # Deserialize jobs
            for job_id, job_data in serialized_jobs.items():
                try:
                    job = ExperimentJob.from_dict(job_data)
                    self._jobs[job_id] = job

                    # Add queued jobs back to priority queue
                    if job.status == ExperimentStatus.QUEUED:
                        heapq.heappush(self._queue, job)

                except Exception as e:
                    self.logger.error("job_deserialization_failed",
                                    job_id=job_id, error=str(e))

            self.logger.info("queue_loaded", job_count=len(self._jobs))

        except Exception as e:
            self.logger.error("queue_load_failed", error=str(e))

    def cleanup_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clean up old completed jobs.

        Args:
            older_than_days: Remove jobs completed more than this many days ago

        Returns:
            Number of jobs cleaned up
        """
        cleanup_threshold = datetime.now() - timedelta(days=older_than_days)
        cleaned_count = 0

        with self._lock:
            jobs_to_remove = []

            for job_id, job in self._jobs.items():
                if (job.status in [ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cleanup_threshold):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                cleaned_count += 1

            # Rebuild priority queue (remove any cleaned jobs)
            self._queue = [job for job in self._queue if job.id in self._jobs]
            heapq.heapify(self._queue)

            if cleaned_count > 0:
                self._persist_queue()
                self.logger.info("jobs_cleaned_up", count=cleaned_count)

        return cleaned_count


# Singleton instance
_experiment_queue = None


def get_experiment_queue() -> ExperimentQueue:
    """Get singleton experiment queue instance."""
    global _experiment_queue
    if _experiment_queue is None:
        _experiment_queue = ExperimentQueue()
    return _experiment_queue


def create_experiment_job(name: str,
                         model_size: str,
                         dataset_tasks: list[int],
                         priority: ExperimentPriority = ExperimentPriority.NORMAL,
                         max_runtime_minutes: int = 60,
                         **kwargs) -> ExperimentJob:
    """Convenience function to create an experiment job.

    Args:
        name: Experiment name
        model_size: Model size (e.g., "1B", "8B")
        dataset_tasks: List of ARC task IDs
        priority: Job priority
        max_runtime_minutes: Maximum runtime in minutes
        **kwargs: Additional configuration parameters

    Returns:
        Created experiment job
    """
    job_id = str(uuid.uuid4())
    now = datetime.now()

    config = ExperimentConfig(
        experiment_id=job_id,
        name=name,
        model_size=model_size,
        dataset_tasks=dataset_tasks,
        max_runtime_minutes=max_runtime_minutes,
        checkpoint_interval_minutes=kwargs.get('checkpoint_interval_minutes', 10),
        max_memory_gb=kwargs.get('max_memory_gb', 14.0),
        gpu_required=kwargs.get('gpu_required', True),
        platform_preference=kwargs.get('platform_preference'),
        parameters=kwargs.get('parameters', {}),
        retry_config=kwargs.get('retry_config', {})
    )

    metadata = ExperimentMetadata(
        created_at=now,
        updated_at=now,
        submitted_by=kwargs.get('submitted_by', 'system'),
        tags=kwargs.get('tags', []),
        description=kwargs.get('description', ''),
        estimated_duration_minutes=kwargs.get('estimated_duration_minutes'),
        resource_requirements=kwargs.get('resource_requirements', {})
    )

    return ExperimentJob(
        id=job_id,
        config=config,
        priority=priority,
        status=ExperimentStatus.QUEUED,
        metadata=metadata,
        created_at=now,
        max_retries=kwargs.get('max_retries', 3)
    )
