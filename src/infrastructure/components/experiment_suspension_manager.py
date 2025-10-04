"""Experiment suspension management for graceful timeout handling.

This module provides graceful experiment suspension and resumption capabilities
for handling platform session timeouts and other interruptions.
"""

import asyncio
import json
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .experiment_queue import ExperimentJob, ExperimentStatus, get_experiment_queue
from .graceful_shutdown import get_shutdown_manager
from .session_timeout_manager import SessionInfo, get_session_timeout_manager

logger = structlog.get_logger(__name__)


class SuspensionReason(Enum):
    """Reasons for experiment suspension."""
    SESSION_TIMEOUT_WARNING = "session_timeout_warning"
    PLATFORM_ROTATION = "platform_rotation"
    RESOURCE_CONSTRAINT = "resource_constraint"
    USER_REQUEST = "user_request"
    SYSTEM_MAINTENANCE = "system_maintenance"
    ERROR_RECOVERY = "error_recovery"


class SuspensionStrategy(Enum):
    """Strategies for experiment suspension."""
    IMMEDIATE = "immediate"           # Stop immediately
    CHECKPOINT_SAVE = "checkpoint_save"   # Save checkpoint then stop
    GRACEFUL = "graceful"            # Complete current batch/iteration
    DELAYED = "delayed"              # Complete current phase then stop


@dataclass
class SuspensionPoint:
    """Information about where an experiment was suspended."""
    job_id: str
    suspension_time: datetime
    reason: SuspensionReason
    strategy_used: SuspensionStrategy
    platform: str
    checkpoint_path: str | None = None
    experiment_state: dict[str, Any] = None
    progress_info: dict[str, Any] = None
    resource_usage: dict[str, Any] = None
    estimated_completion_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'suspension_time': self.suspension_time.isoformat(),
            'reason': self.reason.value,
            'strategy_used': self.strategy_used.value,
            'platform': self.platform,
            'checkpoint_path': self.checkpoint_path,
            'experiment_state': self.experiment_state or {},
            'progress_info': self.progress_info or {},
            'resource_usage': self.resource_usage or {},
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SuspensionPoint':
        """Create from dictionary."""
        return cls(
            job_id=data['job_id'],
            suspension_time=datetime.fromisoformat(data['suspension_time']),
            reason=SuspensionReason(data['reason']),
            strategy_used=SuspensionStrategy(data['strategy_used']),
            platform=data['platform'],
            checkpoint_path=data.get('checkpoint_path'),
            experiment_state=data.get('experiment_state', {}),
            progress_info=data.get('progress_info', {}),
            resource_usage=data.get('resource_usage', {}),
            estimated_completion_time=datetime.fromisoformat(data['estimated_completion_time']) if data.get('estimated_completion_time') else None
        )


@dataclass
class ResumptionConfig:
    """Configuration for resuming suspended experiments."""
    auto_resume: bool = True
    platform_preference: str | None = None
    resource_requirements: dict[str, Any] = None
    priority_boost: bool = True
    checkpoint_validation: bool = True
    max_resume_attempts: int = 3


class ExperimentSuspensionManager:
    """Manages graceful experiment suspension and resumption."""

    def __init__(self, suspension_dir: Path | None = None):
        """Initialize experiment suspension manager.

        Args:
            suspension_dir: Directory to store suspension state
        """
        self.suspension_dir = suspension_dir or Path.home() / ".arc-suspension"
        self.suspension_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.experiment_queue = get_experiment_queue()
        self.session_manager = get_session_timeout_manager()
        self.shutdown_manager = get_shutdown_manager()

        # State
        self.active_suspensions: dict[str, SuspensionPoint] = {}
        self.suspension_history: list[SuspensionPoint] = []
        self.suspension_callbacks: dict[str, list[Callable]] = {
            'suspension_started': [],
            'suspension_completed': [],
            'resumption_started': [],
            'resumption_completed': [],
            'suspension_failed': []
        }

        # Configuration
        self.default_strategy = SuspensionStrategy.CHECKPOINT_SAVE
        self.suspension_timeout_seconds = 300  # 5 minutes max for suspension

        self.logger = structlog.get_logger('suspension_manager')

        # Register callbacks
        self.session_manager.register_callback('session_warning', self._on_session_warning)
        self.shutdown_manager.register_shutdown_hook(self._on_shutdown_initiated, async_hook=True)

        # Load persisted state
        self._load_suspension_state()

    async def suspend_experiment(self,
                                job_id: str,
                                reason: SuspensionReason,
                                strategy: SuspensionStrategy | None = None,
                                metadata: dict[str, Any] | None = None) -> bool:
        """Suspend an experiment gracefully.

        Args:
            job_id: Job ID to suspend
            reason: Reason for suspension
            strategy: Suspension strategy to use
            metadata: Additional metadata

        Returns:
            True if suspended successfully
        """
        try:
            job = self.experiment_queue.get_job(job_id)
            if not job:
                self.logger.error("job_not_found_for_suspension", job_id=job_id)
                return False

            if job.status != ExperimentStatus.RUNNING:
                self.logger.warning("job_not_running", job_id=job_id, status=job.status.value)
                return False

            strategy = strategy or self._determine_suspension_strategy(job, reason)

            self.logger.info("experiment_suspension_started",
                           job_id=job_id,
                           reason=reason.value,
                           strategy=strategy.value)

            # Trigger callbacks
            self._trigger_callbacks('suspension_started', {'job': job, 'reason': reason, 'strategy': strategy})

            # Execute suspension based on strategy
            suspension_point = await self._execute_suspension(job, reason, strategy, metadata or {})

            if suspension_point:
                # Store suspension point
                self.active_suspensions[job_id] = suspension_point
                self.suspension_history.append(suspension_point)

                # Update job status
                self.experiment_queue.update_job_status(
                    job_id,
                    ExperimentStatus.SUSPENDED,
                    {
                        'suspension_reason': reason.value,
                        'suspension_time': suspension_point.suspension_time.isoformat(),
                        'checkpoint_path': suspension_point.checkpoint_path,
                        'suspension_strategy': strategy.value
                    }
                )

                # Persist state
                self._save_suspension_state()

                # Trigger completion callback
                self._trigger_callbacks('suspension_completed', suspension_point)

                self.logger.info("experiment_suspended_successfully",
                               job_id=job_id,
                               checkpoint_path=suspension_point.checkpoint_path)

                return True
            else:
                self.logger.error("experiment_suspension_failed", job_id=job_id)
                self._trigger_callbacks('suspension_failed', {'job': job, 'reason': reason})
                return False

        except Exception as e:
            self.logger.error("suspension_error", job_id=job_id, error=str(e))
            self._trigger_callbacks('suspension_failed', {'job_id': job_id, 'error': str(e)})
            return False

    def _determine_suspension_strategy(self, job: ExperimentJob, reason: SuspensionReason) -> SuspensionStrategy:
        """Determine appropriate suspension strategy.

        Args:
            job: Job to suspend
            reason: Suspension reason

        Returns:
            Recommended suspension strategy
        """
        # Strategy based on reason
        strategy_map = {
            SuspensionReason.SESSION_TIMEOUT_WARNING: SuspensionStrategy.CHECKPOINT_SAVE,
            SuspensionReason.PLATFORM_ROTATION: SuspensionStrategy.CHECKPOINT_SAVE,
            SuspensionReason.RESOURCE_CONSTRAINT: SuspensionStrategy.GRACEFUL,
            SuspensionReason.USER_REQUEST: SuspensionStrategy.GRACEFUL,
            SuspensionReason.SYSTEM_MAINTENANCE: SuspensionStrategy.CHECKPOINT_SAVE,
            SuspensionReason.ERROR_RECOVERY: SuspensionStrategy.IMMEDIATE
        }

        base_strategy = strategy_map.get(reason, self.default_strategy)

        # Adjust based on job characteristics
        if job.config.max_runtime_minutes < 30:  # Short jobs
            return SuspensionStrategy.GRACEFUL
        elif job.config.checkpoint_interval_minutes <= 5:  # Frequent checkpointing
            return SuspensionStrategy.CHECKPOINT_SAVE

        return base_strategy

    async def _execute_suspension(self,
                                job: ExperimentJob,
                                reason: SuspensionReason,
                                strategy: SuspensionStrategy,
                                metadata: dict[str, Any]) -> SuspensionPoint | None:
        """Execute experiment suspension based on strategy.

        Args:
            job: Job to suspend
            reason: Suspension reason
            strategy: Suspension strategy
            metadata: Additional metadata

        Returns:
            Suspension point or None if failed
        """
        try:
            suspension_start = datetime.now()

            # Collect current experiment state
            experiment_state = await self._collect_experiment_state(job)
            progress_info = await self._collect_progress_info(job)
            resource_usage = await self._collect_resource_usage(job)

            checkpoint_path = None

            # Execute strategy-specific suspension
            if strategy == SuspensionStrategy.IMMEDIATE:
                checkpoint_path = await self._immediate_suspension(job)
            elif strategy == SuspensionStrategy.CHECKPOINT_SAVE:
                checkpoint_path = await self._checkpoint_save_suspension(job)
            elif strategy == SuspensionStrategy.GRACEFUL:
                checkpoint_path = await self._graceful_suspension(job)
            elif strategy == SuspensionStrategy.DELAYED:
                checkpoint_path = await self._delayed_suspension(job)

            # Estimate completion time if job were to continue
            estimated_completion = self._estimate_completion_time(job, progress_info)

            return SuspensionPoint(
                job_id=job.id,
                suspension_time=suspension_start,
                reason=reason,
                strategy_used=strategy,
                platform=job.platform or 'unknown',
                checkpoint_path=checkpoint_path,
                experiment_state=experiment_state,
                progress_info=progress_info,
                resource_usage=resource_usage,
                estimated_completion_time=estimated_completion
            )

        except Exception as e:
            self.logger.error("suspension_execution_failed", job_id=job.id, error=str(e))
            return None

    async def _immediate_suspension(self, job: ExperimentJob) -> str | None:
        """Execute immediate suspension."""
        self.logger.info("executing_immediate_suspension", job_id=job.id)

        # Try to save a quick checkpoint
        try:
            return await self._save_emergency_checkpoint(job)
        except Exception as e:
            self.logger.warning("emergency_checkpoint_failed", job_id=job.id, error=str(e))
            return None

    async def _checkpoint_save_suspension(self, job: ExperimentJob) -> str | None:
        """Execute suspension with checkpoint save."""
        self.logger.info("executing_checkpoint_save_suspension", job_id=job.id)

        try:
            # Request checkpoint save
            checkpoint_path = await self._request_checkpoint_save(job)

            # Wait for checkpoint completion (with timeout)
            if checkpoint_path:
                await self._wait_for_checkpoint_completion(job, checkpoint_path)
                return checkpoint_path

            return None

        except TimeoutError:
            self.logger.warning("checkpoint_save_timeout", job_id=job.id)
            # Fallback to emergency checkpoint
            return await self._save_emergency_checkpoint(job)
        except Exception as e:
            self.logger.error("checkpoint_save_failed", job_id=job.id, error=str(e))
            return await self._save_emergency_checkpoint(job)

    async def _graceful_suspension(self, job: ExperimentJob) -> str | None:
        """Execute graceful suspension."""
        self.logger.info("executing_graceful_suspension", job_id=job.id)

        try:
            # Signal graceful stop to experiment
            await self._signal_graceful_stop(job)

            # Wait for current iteration/batch to complete
            await self._wait_for_graceful_completion(job)

            # Save checkpoint
            return await self._save_checkpoint_after_graceful_stop(job)

        except TimeoutError:
            self.logger.warning("graceful_suspension_timeout", job_id=job.id)
            return await self._checkpoint_save_suspension(job)
        except Exception as e:
            self.logger.error("graceful_suspension_failed", job_id=job.id, error=str(e))
            return await self._save_emergency_checkpoint(job)

    async def _delayed_suspension(self, job: ExperimentJob) -> str | None:
        """Execute delayed suspension."""
        self.logger.info("executing_delayed_suspension", job_id=job.id)

        try:
            # Wait for current phase to complete
            await self._wait_for_phase_completion(job)

            # Then execute graceful suspension
            return await self._graceful_suspension(job)

        except TimeoutError:
            self.logger.warning("delayed_suspension_timeout", job_id=job.id)
            return await self._graceful_suspension(job)
        except Exception as e:
            self.logger.error("delayed_suspension_failed", job_id=job.id, error=str(e))
            return await self._graceful_suspension(job)

    async def _collect_experiment_state(self, job: ExperimentJob) -> dict[str, Any]:
        """Collect current experiment state."""
        try:
            # This would integrate with the actual training process
            # For now, return basic information
            return {
                'job_id': job.id,
                'current_status': job.status.value,
                'start_time': job.started_at.isoformat() if job.started_at else None,
                'runtime_minutes': (datetime.now() - job.started_at).total_seconds() / 60 if job.started_at else 0,
                'model_config': job.config.parameters,
                'platform': job.platform
            }
        except Exception as e:
            self.logger.error("state_collection_failed", job_id=job.id, error=str(e))
            return {}

    async def _collect_progress_info(self, job: ExperimentJob) -> dict[str, Any]:
        """Collect experiment progress information."""
        try:
            # Extract progress from job if available
            progress = job.progress or {}

            return {
                'completion_percentage': progress.get('completion_percentage', 0),
                'current_epoch': progress.get('current_epoch', 0),
                'current_batch': progress.get('current_batch', 0),
                'total_epochs': progress.get('total_epochs', 0),
                'tasks_completed': progress.get('tasks_completed', 0),
                'tasks_total': len(job.config.dataset_tasks) if job.config.dataset_tasks else 0,
                'last_checkpoint': progress.get('last_checkpoint'),
                'metrics': progress.get('metrics', {})
            }
        except Exception as e:
            self.logger.error("progress_collection_failed", job_id=job.id, error=str(e))
            return {}

    async def _collect_resource_usage(self, job: ExperimentJob) -> dict[str, Any]:
        """Collect resource usage information."""
        try:
            # This would integrate with system monitoring
            # For now, return placeholder information
            return {
                'memory_usage_mb': 8000,  # Placeholder
                'gpu_utilization': 0.85,  # Placeholder
                'disk_usage_mb': 2000,    # Placeholder
                'estimated_memory_mb': job.config.max_memory_gb * 1024,
                'platform': job.platform
            }
        except Exception as e:
            self.logger.error("resource_collection_failed", job_id=job.id, error=str(e))
            return {}

    def _estimate_completion_time(self, job: ExperimentJob, progress_info: dict[str, Any]) -> datetime | None:
        """Estimate completion time based on progress."""
        try:
            completion_pct = progress_info.get('completion_percentage', 0)

            if completion_pct > 0 and job.started_at:
                elapsed_time = datetime.now() - job.started_at
                total_estimated_time = elapsed_time / (completion_pct / 100)
                return job.started_at + total_estimated_time

            # Fallback to max runtime
            if job.started_at:
                return job.started_at + timedelta(minutes=job.config.max_runtime_minutes)

            return None

        except Exception as e:
            self.logger.error("completion_estimation_failed", job_id=job.id, error=str(e))
            return None

    async def _save_emergency_checkpoint(self, job: ExperimentJob) -> str | None:
        """Save emergency checkpoint."""
        try:
            checkpoint_dir = self.suspension_dir / "emergency_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"{job.id}_emergency_{int(datetime.now().timestamp())}.pkl"

            checkpoint_data = {
                'job_id': job.id,
                'job_config': asdict(job.config),
                'suspension_time': datetime.now().isoformat(),
                'experiment_state': await self._collect_experiment_state(job),
                'progress_info': await self._collect_progress_info(job)
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            self.logger.info("emergency_checkpoint_saved",
                           job_id=job.id,
                           checkpoint_path=str(checkpoint_path))

            return str(checkpoint_path)

        except Exception as e:
            self.logger.error("emergency_checkpoint_failed", job_id=job.id, error=str(e))
            return None

    async def _request_checkpoint_save(self, job: ExperimentJob) -> str | None:
        """Request checkpoint save from running experiment."""
        # This would send a signal to the running experiment to save a checkpoint
        # For now, simulate checkpoint save
        await asyncio.sleep(2)  # Simulate checkpoint save time

        checkpoint_dir = self.suspension_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{job.id}_checkpoint_{int(datetime.now().timestamp())}.pkl"

        # Simulate checkpoint creation
        checkpoint_data = {
            'job_id': job.id,
            'timestamp': datetime.now().isoformat(),
            'type': 'regular_checkpoint'
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        return str(checkpoint_path)

    async def _wait_for_checkpoint_completion(self, job: ExperimentJob, checkpoint_path: str):
        """Wait for checkpoint to complete."""
        # Wait with timeout
        timeout = self.suspension_timeout_seconds
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            if Path(checkpoint_path).exists():
                # Additional verification that checkpoint is complete
                await asyncio.sleep(1)  # Brief wait for file completion
                return

            await asyncio.sleep(1)

        raise TimeoutError("Checkpoint completion timeout")

    async def _signal_graceful_stop(self, job: ExperimentJob):
        """Signal graceful stop to experiment."""
        # This would send a signal to the running process
        # For now, just log the action
        self.logger.info("graceful_stop_signaled", job_id=job.id)
        await asyncio.sleep(1)  # Simulate signal processing

    async def _wait_for_graceful_completion(self, job: ExperimentJob):
        """Wait for graceful completion."""
        # Wait for current iteration to complete
        timeout = min(self.suspension_timeout_seconds, 120)  # Max 2 minutes for graceful
        await asyncio.sleep(min(timeout, 10))  # Simulate graceful completion

    async def _save_checkpoint_after_graceful_stop(self, job: ExperimentJob) -> str | None:
        """Save checkpoint after graceful stop."""
        return await self._request_checkpoint_save(job)

    async def _wait_for_phase_completion(self, job: ExperimentJob):
        """Wait for current phase to complete."""
        # This would wait for current training phase (epoch, batch, etc.)
        timeout = min(self.suspension_timeout_seconds, 300)  # Max 5 minutes for phase
        await asyncio.sleep(min(timeout, 30))  # Simulate phase completion

    def _on_session_warning(self, session_info: SessionInfo):
        """Handle session warning callback."""
        self.logger.warning("session_timeout_warning_received",
                          platform=session_info.platform.value,
                          remaining_minutes=session_info.get_remaining_minutes())

        # Find running jobs and suspend them
        asyncio.create_task(self._suspend_running_jobs_for_session_warning())

    async def _suspend_running_jobs_for_session_warning(self):
        """Suspend all running jobs due to session warning."""
        try:
            running_jobs = self.experiment_queue.list_jobs(status_filter=ExperimentStatus.RUNNING)

            for job in running_jobs:
                self.logger.info("suspending_job_for_session_warning", job_id=job.id)

                await self.suspend_experiment(
                    job.id,
                    SuspensionReason.SESSION_TIMEOUT_WARNING,
                    strategy=SuspensionStrategy.CHECKPOINT_SAVE
                )

        except Exception as e:
            self.logger.error("session_warning_suspension_failed", error=str(e))

    async def _on_shutdown_initiated(self, shutdown_event):
        """Handle shutdown initiation."""
        reason_map = {
            'platform_rotation': SuspensionReason.PLATFORM_ROTATION,
            'session_timeout': SuspensionReason.SESSION_TIMEOUT_WARNING,
            'user_interrupt': SuspensionReason.USER_REQUEST
        }

        suspension_reason = reason_map.get(
            shutdown_event.reason.value,
            SuspensionReason.SYSTEM_MAINTENANCE
        )

        # Suspend all running jobs
        running_jobs = self.experiment_queue.list_jobs(status_filter=ExperimentStatus.RUNNING)

        for job in running_jobs:
            await self.suspend_experiment(job.id, suspension_reason)

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for suspension events."""
        if event_type in self.suspension_callbacks:
            self.suspension_callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for an event."""
        for callback in self.suspension_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error("callback_failed", event_type=event_type, error=str(e))

    def _save_suspension_state(self):
        """Save suspension state to disk."""
        try:
            state_file = self.suspension_dir / "suspension_state.json"

            # Serialize active suspensions
            active_suspensions = {
                job_id: suspension.to_dict()
                for job_id, suspension in self.active_suspensions.items()
            }

            state = {
                'active_suspensions': active_suspensions,
                'suspension_count': len(self.suspension_history),
                'last_saved': datetime.now().isoformat()
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error("suspension_state_save_failed", error=str(e))

    def _load_suspension_state(self):
        """Load suspension state from disk."""
        try:
            state_file = self.suspension_dir / "suspension_state.json"

            if not state_file.exists():
                return

            with open(state_file) as f:
                state = json.load(f)

            # Load active suspensions
            for job_id, suspension_data in state.get('active_suspensions', {}).items():
                try:
                    suspension = SuspensionPoint.from_dict(suspension_data)
                    self.active_suspensions[job_id] = suspension
                except Exception as e:
                    self.logger.error("suspension_load_failed", job_id=job_id, error=str(e))

            self.logger.info("suspension_state_loaded",
                           active_suspensions=len(self.active_suspensions))

        except Exception as e:
            self.logger.error("suspension_state_load_failed", error=str(e))

    def get_suspension_status(self) -> dict[str, Any]:
        """Get suspension status."""
        return {
            'active_suspensions': len(self.active_suspensions),
            'total_suspensions': len(self.suspension_history),
            'suspension_reasons': {
                reason.value: len([s for s in self.suspension_history if s.reason == reason])
                for reason in SuspensionReason
            },
            'suspension_strategies': {
                strategy.value: len([s for s in self.suspension_history if s.strategy_used == strategy])
                for strategy in SuspensionStrategy
            }
        }


# Singleton instance
_suspension_manager = None


def get_suspension_manager() -> ExperimentSuspensionManager:
    """Get singleton experiment suspension manager instance."""
    global _suspension_manager
    if _suspension_manager is None:
        _suspension_manager = ExperimentSuspensionManager()
    return _suspension_manager
