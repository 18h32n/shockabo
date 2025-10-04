"""Automated recovery logic for session restoration and experiment resumption.

This module provides comprehensive recovery mechanisms for resuming experiments
after platform switches, session timeouts, and other interruptions.
"""

import asyncio
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .email_notification_system import get_email_notification_system
from .experiment_queue import (
    ExperimentJob,
    ExperimentStatus,
    get_experiment_queue,
)
from .experiment_suspension_manager import SuspensionPoint, SuspensionReason, get_suspension_manager
from .platform_detector import get_platform_detector
from .session_timeout_manager import get_session_timeout_manager

logger = structlog.get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    IMMEDIATE = "immediate"           # Resume immediately
    CHECKPOINT_RESUME = "checkpoint_resume"  # Resume from checkpoint
    FRESH_START = "fresh_start"      # Start from beginning with same config
    PLATFORM_SWITCH = "platform_switch"     # Try different platform
    DELAYED_RESUME = "delayed_resume"        # Wait before resuming
    INTELLIGENT = "intelligent"      # Use ML to determine best strategy


class RecoveryOutcome(Enum):
    """Recovery attempt outcomes."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    DEFERRED = "deferred"


@dataclass
class RecoveryAttempt:
    """Information about a recovery attempt."""
    job_id: str
    suspension_point: SuspensionPoint
    strategy: RecoveryStrategy
    attempt_time: datetime
    outcome: RecoveryOutcome
    new_platform: str | None = None
    error_message: str | None = None
    recovery_metadata: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'suspension_point': self.suspension_point.to_dict(),
            'strategy': self.strategy.value,
            'attempt_time': self.attempt_time.isoformat(),
            'outcome': self.outcome.value,
            'new_platform': self.new_platform,
            'error_message': self.error_message,
            'recovery_metadata': self.recovery_metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RecoveryAttempt':
        """Create from dictionary."""
        return cls(
            job_id=data['job_id'],
            suspension_point=SuspensionPoint.from_dict(data['suspension_point']),
            strategy=RecoveryStrategy(data['strategy']),
            attempt_time=datetime.fromisoformat(data['attempt_time']),
            outcome=RecoveryOutcome(data['outcome']),
            new_platform=data.get('new_platform'),
            error_message=data.get('error_message'),
            recovery_metadata=data.get('recovery_metadata', {})
        )


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    auto_recovery_enabled: bool = True
    max_recovery_attempts: int = 3
    recovery_delay_minutes: int = 5
    checkpoint_validation: bool = True
    platform_rotation_on_failure: bool = True
    priority_boost_on_resume: bool = True
    max_recovery_time_hours: int = 24

    # Strategy preferences
    preferred_strategies: list[RecoveryStrategy] = None

    def __post_init__(self):
        if self.preferred_strategies is None:
            self.preferred_strategies = [
                RecoveryStrategy.CHECKPOINT_RESUME,
                RecoveryStrategy.PLATFORM_SWITCH,
                RecoveryStrategy.FRESH_START
            ]


class CheckpointValidator:
    """Validates checkpoints before recovery."""

    def __init__(self):
        self.logger = structlog.get_logger('checkpoint_validator')

    async def validate_checkpoint(self, checkpoint_path: str,
                                job: ExperimentJob) -> dict[str, Any]:
        """Validate checkpoint file and return validation results.

        Args:
            checkpoint_path: Path to checkpoint file
            job: Original job configuration

        Returns:
            Validation results dictionary
        """
        try:
            checkpoint_file = Path(checkpoint_path)

            validation_result = {
                'valid': False,
                'file_exists': False,
                'file_size_mb': 0,
                'file_age_hours': 0,
                'content_valid': False,
                'config_matches': False,
                'progress_extractable': False,
                'error': None,
                'metadata': {}
            }

            # Check file existence
            if not checkpoint_file.exists():
                validation_result['error'] = "Checkpoint file does not exist"
                return validation_result

            validation_result['file_exists'] = True

            # Check file size
            file_size = checkpoint_file.stat().st_size
            validation_result['file_size_mb'] = file_size / (1024 * 1024)

            if file_size == 0:
                validation_result['error'] = "Checkpoint file is empty"
                return validation_result

            # Check file age
            file_mtime = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
            file_age = datetime.now() - file_mtime
            validation_result['file_age_hours'] = file_age.total_seconds() / 3600

            # Validate content
            try:
                content_valid = await self._validate_checkpoint_content(checkpoint_path)
                validation_result['content_valid'] = content_valid

                if content_valid:
                    # Check config compatibility
                    config_valid = await self._validate_config_compatibility(checkpoint_path, job)
                    validation_result['config_matches'] = config_valid

                    # Check if progress is extractable
                    progress_valid = await self._validate_progress_extractable(checkpoint_path)
                    validation_result['progress_extractable'] = progress_valid

                    validation_result['valid'] = config_valid and progress_valid

            except Exception as e:
                validation_result['error'] = f"Content validation failed: {str(e)}"

            self.logger.info("checkpoint_validated",
                           checkpoint_path=checkpoint_path,
                           valid=validation_result['valid'],
                           file_size_mb=validation_result['file_size_mb'],
                           file_age_hours=validation_result['file_age_hours'])

            return validation_result

        except Exception as e:
            self.logger.error("checkpoint_validation_failed",
                            checkpoint_path=checkpoint_path,
                            error=str(e))
            return {
                'valid': False,
                'error': str(e),
                'file_exists': False,
                'file_size_mb': 0,
                'file_age_hours': 0,
                'content_valid': False,
                'config_matches': False,
                'progress_extractable': False,
                'metadata': {}
            }

    async def _validate_checkpoint_content(self, checkpoint_path: str) -> bool:
        """Validate checkpoint content structure."""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            # Check for required fields
            required_fields = ['job_id', 'timestamp']
            return all(field in data for field in required_fields)

        except Exception as e:
            self.logger.error("checkpoint_content_validation_failed", error=str(e))
            return False

    async def _validate_config_compatibility(self, checkpoint_path: str,
                                           job: ExperimentJob) -> bool:
        """Validate checkpoint compatibility with job config."""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            checkpoint_job_id = data.get('job_id')
            return checkpoint_job_id == job.id

        except Exception as e:
            self.logger.error("config_compatibility_validation_failed", error=str(e))
            return False

    async def _validate_progress_extractable(self, checkpoint_path: str) -> bool:
        """Validate that progress information can be extracted."""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            # Check for progress information
            return 'progress_info' in data or 'experiment_state' in data

        except Exception as e:
            self.logger.error("progress_extraction_validation_failed", error=str(e))
            return False


class RecoveryManager:
    """Manages automated recovery of suspended experiments."""

    def __init__(self,
                 config: RecoveryConfig | None = None,
                 recovery_dir: Path | None = None):
        """Initialize recovery manager.

        Args:
            config: Recovery configuration
            recovery_dir: Directory to store recovery state
        """
        self.config = config or RecoveryConfig()
        self.recovery_dir = recovery_dir or Path.home() / ".arc-recovery"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.experiment_queue = get_experiment_queue()
        self.suspension_manager = get_suspension_manager()
        self.platform_detector = get_platform_detector()
        self.session_manager = get_session_timeout_manager()
        self.notification_system = get_email_notification_system()
        self.checkpoint_validator = CheckpointValidator()

        # State
        self.recovery_attempts: dict[str, list[RecoveryAttempt]] = {}
        self.active_recoveries: dict[str, asyncio.Task] = {}
        self.recovery_queue: list[str] = []  # Job IDs pending recovery

        # Background tasks
        self.recovery_task: asyncio.Task | None = None
        self.running = False

        self.logger = structlog.get_logger('recovery_manager')

        # Register callbacks
        self.session_manager.register_callback('session_detected', self._on_session_restored)
        self.experiment_queue.register_callback('job_failed', self._on_job_failed)

        # Load persisted state
        self._load_recovery_state()

    async def start_recovery_management(self) -> bool:
        """Start automated recovery management.

        Returns:
            True if started successfully
        """
        if self.running:
            self.logger.warning("recovery_management_already_running")
            return False

        if not self.config.auto_recovery_enabled:
            self.logger.info("recovery_management_disabled")
            return False

        try:
            self.running = True

            # Start recovery processing loop
            self.recovery_task = asyncio.create_task(self._recovery_processing_loop())

            # Check for suspended jobs that need recovery
            await self._check_for_suspended_jobs()

            self.logger.info("recovery_management_started")
            return True

        except Exception as e:
            self.logger.error("recovery_management_start_failed", error=str(e))
            self.running = False
            return False

    async def stop_recovery_management(self) -> bool:
        """Stop recovery management.

        Returns:
            True if stopped successfully
        """
        if not self.running:
            return True

        try:
            self.running = False

            # Cancel recovery task
            if self.recovery_task:
                self.recovery_task.cancel()
                try:
                    await self.recovery_task
                except asyncio.CancelledError:
                    pass

            # Cancel active recovery tasks
            for task in self.active_recoveries.values():
                task.cancel()

            if self.active_recoveries:
                await asyncio.gather(*self.active_recoveries.values(), return_exceptions=True)

            self.active_recoveries.clear()

            # Save state
            self._save_recovery_state()

            self.logger.info("recovery_management_stopped")
            return True

        except Exception as e:
            self.logger.error("recovery_management_stop_failed", error=str(e))
            return False

    async def recover_experiment(self, job_id: str,
                                strategy: RecoveryStrategy | None = None) -> RecoveryOutcome:
        """Recover a specific experiment.

        Args:
            job_id: Job ID to recover
            strategy: Specific recovery strategy to use

        Returns:
            Recovery outcome
        """
        try:
            job = self.experiment_queue.get_job(job_id)
            if not job:
                self.logger.error("job_not_found_for_recovery", job_id=job_id)
                return RecoveryOutcome.FAILED

            if job.status != ExperimentStatus.SUSPENDED:
                self.logger.warning("job_not_suspended", job_id=job_id, status=job.status.value)
                return RecoveryOutcome.SKIPPED

            # Get suspension point
            suspension_point = self.suspension_manager.active_suspensions.get(job_id)
            if not suspension_point:
                self.logger.error("suspension_point_not_found", job_id=job_id)
                return RecoveryOutcome.FAILED

            # Determine recovery strategy
            if not strategy:
                strategy = await self._determine_recovery_strategy(job, suspension_point)

            # Execute recovery
            return await self._execute_recovery(job, suspension_point, strategy)

        except Exception as e:
            self.logger.error("recovery_failed", job_id=job_id, error=str(e))
            return RecoveryOutcome.FAILED

    async def _recovery_processing_loop(self):
        """Main recovery processing loop."""
        while self.running:
            try:
                if self.recovery_queue:
                    job_id = self.recovery_queue.pop(0)

                    if job_id not in self.active_recoveries:
                        # Start recovery task
                        recovery_task = asyncio.create_task(self._process_recovery(job_id))
                        self.active_recoveries[job_id] = recovery_task

                # Clean up completed tasks
                completed_tasks = []
                for job_id, task in self.active_recoveries.items():
                    if task.done():
                        completed_tasks.append(job_id)

                for job_id in completed_tasks:
                    del self.active_recoveries[job_id]

                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("recovery_processing_loop_error", error=str(e))
                await asyncio.sleep(60)

    async def _process_recovery(self, job_id: str):
        """Process recovery for a specific job."""
        try:
            # Wait for initial delay
            await asyncio.sleep(self.config.recovery_delay_minutes * 60)

            # Attempt recovery
            outcome = await self.recover_experiment(job_id)

            if outcome == RecoveryOutcome.FAILED:
                # Check if we should retry
                attempts = self.recovery_attempts.get(job_id, [])
                if len(attempts) < self.config.max_recovery_attempts:
                    # Schedule retry
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
                    self.recovery_queue.append(job_id)
                    self.logger.info("recovery_retry_scheduled", job_id=job_id,
                                   attempt=len(attempts) + 1)
                else:
                    self.logger.error("recovery_max_attempts_reached", job_id=job_id)
                    await self._handle_recovery_exhausted(job_id)

        except asyncio.CancelledError:
            self.logger.info("recovery_task_cancelled", job_id=job_id)
        except Exception as e:
            self.logger.error("recovery_processing_failed", job_id=job_id, error=str(e))

    async def _determine_recovery_strategy(self, job: ExperimentJob,
                                         suspension_point: SuspensionPoint) -> RecoveryStrategy:
        """Determine the best recovery strategy for a job.

        Args:
            job: Suspended job
            suspension_point: Suspension information

        Returns:
            Recommended recovery strategy
        """
        # Check if intelligent strategy is enabled
        if RecoveryStrategy.INTELLIGENT in self.config.preferred_strategies:
            return await self._intelligent_strategy_selection(job, suspension_point)

        # Use rule-based strategy selection
        return await self._rule_based_strategy_selection(job, suspension_point)

    async def _intelligent_strategy_selection(self, job: ExperimentJob,
                                            suspension_point: SuspensionPoint) -> RecoveryStrategy:
        """Use intelligent analysis to select recovery strategy."""
        # This would use ML models to predict best recovery strategy
        # For now, fall back to rule-based selection
        return await self._rule_based_strategy_selection(job, suspension_point)

    async def _rule_based_strategy_selection(self, job: ExperimentJob,
                                           suspension_point: SuspensionPoint) -> RecoveryStrategy:
        """Use rules to select recovery strategy."""
        # Strategy based on suspension reason
        if suspension_point.reason == SuspensionReason.SESSION_TIMEOUT_WARNING:
            return RecoveryStrategy.CHECKPOINT_RESUME
        elif suspension_point.reason == SuspensionReason.PLATFORM_ROTATION:
            return RecoveryStrategy.PLATFORM_SWITCH
        elif suspension_point.reason == SuspensionReason.RESOURCE_CONSTRAINT:
            return RecoveryStrategy.DELAYED_RESUME
        elif suspension_point.reason == SuspensionReason.ERROR_RECOVERY:
            return RecoveryStrategy.FRESH_START

        # Check checkpoint availability
        if suspension_point.checkpoint_path:
            checkpoint_valid = await self.checkpoint_validator.validate_checkpoint(
                suspension_point.checkpoint_path, job
            )
            if checkpoint_valid['valid']:
                return RecoveryStrategy.CHECKPOINT_RESUME

        # Check if platform switch might help
        current_platform = self.platform_detector.detect_platform().platform
        if current_platform.value != suspension_point.platform:
            return RecoveryStrategy.PLATFORM_SWITCH

        # Default to fresh start
        return RecoveryStrategy.FRESH_START

    async def _execute_recovery(self, job: ExperimentJob,
                              suspension_point: SuspensionPoint,
                              strategy: RecoveryStrategy) -> RecoveryOutcome:
        """Execute recovery based on strategy.

        Args:
            job: Job to recover
            suspension_point: Suspension information
            strategy: Recovery strategy to use

        Returns:
            Recovery outcome
        """
        recovery_attempt = RecoveryAttempt(
            job_id=job.id,
            suspension_point=suspension_point,
            strategy=strategy,
            attempt_time=datetime.now(),
            outcome=RecoveryOutcome.FAILED  # Will be updated
        )

        try:
            self.logger.info("recovery_attempt_started",
                           job_id=job.id,
                           strategy=strategy.value,
                           suspension_reason=suspension_point.reason.value)

            if strategy == RecoveryStrategy.IMMEDIATE:
                outcome = await self._immediate_recovery(job, suspension_point, recovery_attempt)
            elif strategy == RecoveryStrategy.CHECKPOINT_RESUME:
                outcome = await self._checkpoint_resume_recovery(job, suspension_point, recovery_attempt)
            elif strategy == RecoveryStrategy.FRESH_START:
                outcome = await self._fresh_start_recovery(job, suspension_point, recovery_attempt)
            elif strategy == RecoveryStrategy.PLATFORM_SWITCH:
                outcome = await self._platform_switch_recovery(job, suspension_point, recovery_attempt)
            elif strategy == RecoveryStrategy.DELAYED_RESUME:
                outcome = await self._delayed_resume_recovery(job, suspension_point, recovery_attempt)
            else:
                outcome = RecoveryOutcome.FAILED
                recovery_attempt.error_message = f"Unknown strategy: {strategy.value}"

            recovery_attempt.outcome = outcome

            # Record recovery attempt
            if job.id not in self.recovery_attempts:
                self.recovery_attempts[job.id] = []
            self.recovery_attempts[job.id].append(recovery_attempt)

            # Save state
            self._save_recovery_state()

            # Send notification if recovery succeeded
            if outcome == RecoveryOutcome.SUCCESS:
                await self._send_recovery_notification(job, recovery_attempt)

            self.logger.info("recovery_attempt_completed",
                           job_id=job.id,
                           strategy=strategy.value,
                           outcome=outcome.value)

            return outcome

        except Exception as e:
            recovery_attempt.outcome = RecoveryOutcome.FAILED
            recovery_attempt.error_message = str(e)

            self.logger.error("recovery_execution_failed",
                            job_id=job.id,
                            strategy=strategy.value,
                            error=str(e))

            return RecoveryOutcome.FAILED

    async def _immediate_recovery(self, job: ExperimentJob,
                                suspension_point: SuspensionPoint,
                                recovery_attempt: RecoveryAttempt) -> RecoveryOutcome:
        """Execute immediate recovery."""
        try:
            # Simply resume the job as queued
            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'recovery_strategy': RecoveryStrategy.IMMEDIATE.value,
                    'recovery_time': datetime.now().isoformat(),
                    'resumed_from_platform': suspension_point.platform
                }
            )

            if success:
                # Remove from suspension manager
                if job.id in self.suspension_manager.active_suspensions:
                    del self.suspension_manager.active_suspensions[job.id]

                return RecoveryOutcome.SUCCESS
            else:
                return RecoveryOutcome.FAILED

        except Exception as e:
            recovery_attempt.error_message = str(e)
            return RecoveryOutcome.FAILED

    async def _checkpoint_resume_recovery(self, job: ExperimentJob,
                                        suspension_point: SuspensionPoint,
                                        recovery_attempt: RecoveryAttempt) -> RecoveryOutcome:
        """Execute checkpoint resume recovery."""
        try:
            if not suspension_point.checkpoint_path:
                recovery_attempt.error_message = "No checkpoint path available"
                return RecoveryOutcome.FAILED

            # Validate checkpoint
            if self.config.checkpoint_validation:
                validation_result = await self.checkpoint_validator.validate_checkpoint(
                    suspension_point.checkpoint_path, job
                )

                if not validation_result['valid']:
                    recovery_attempt.error_message = f"Checkpoint validation failed: {validation_result.get('error', 'Unknown error')}"
                    return RecoveryOutcome.FAILED

            # Update job with checkpoint information
            job.progress = job.progress or {}
            job.progress['checkpoint_path'] = suspension_point.checkpoint_path
            job.progress['resumed_from_suspension'] = True
            job.progress['suspension_reason'] = suspension_point.reason.value

            # Priority boost if configured
            if self.config.priority_boost_on_resume:
                if job.priority.value > 0:  # Don't boost critical priority
                    job.priority = type(job.priority)(job.priority.value - 1)

            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'recovery_strategy': RecoveryStrategy.CHECKPOINT_RESUME.value,
                    'recovery_time': datetime.now().isoformat(),
                    'checkpoint_path': suspension_point.checkpoint_path,
                    'progress': job.progress
                }
            )

            if success:
                recovery_attempt.recovery_metadata = {
                    'checkpoint_path': suspension_point.checkpoint_path,
                    'checkpoint_validated': self.config.checkpoint_validation
                }

                # Remove from suspension manager
                if job.id in self.suspension_manager.active_suspensions:
                    del self.suspension_manager.active_suspensions[job.id]

                return RecoveryOutcome.SUCCESS
            else:
                return RecoveryOutcome.FAILED

        except Exception as e:
            recovery_attempt.error_message = str(e)
            return RecoveryOutcome.FAILED

    async def _fresh_start_recovery(self, job: ExperimentJob,
                                  suspension_point: SuspensionPoint,
                                  recovery_attempt: RecoveryAttempt) -> RecoveryOutcome:
        """Execute fresh start recovery."""
        try:
            # Reset job state for fresh start
            job.retry_count = 0
            job.started_at = None
            job.completed_at = None
            job.progress = {}
            job.results = {}

            # Priority boost if configured
            if self.config.priority_boost_on_resume:
                if job.priority.value > 0:
                    job.priority = type(job.priority)(job.priority.value - 1)

            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'recovery_strategy': RecoveryStrategy.FRESH_START.value,
                    'recovery_time': datetime.now().isoformat(),
                    'fresh_start': True,
                    'original_suspension_reason': suspension_point.reason.value
                }
            )

            if success:
                recovery_attempt.recovery_metadata = {
                    'fresh_start': True,
                    'original_suspension_time': suspension_point.suspension_time.isoformat()
                }

                # Remove from suspension manager
                if job.id in self.suspension_manager.active_suspensions:
                    del self.suspension_manager.active_suspensions[job.id]

                return RecoveryOutcome.SUCCESS
            else:
                return RecoveryOutcome.FAILED

        except Exception as e:
            recovery_attempt.error_message = str(e)
            return RecoveryOutcome.FAILED

    async def _platform_switch_recovery(self, job: ExperimentJob,
                                      suspension_point: SuspensionPoint,
                                      recovery_attempt: RecoveryAttempt) -> RecoveryOutcome:
        """Execute platform switch recovery."""
        try:
            # Determine best platform to switch to
            current_platform = self.platform_detector.detect_platform().platform
            available_platforms = ['kaggle', 'colab', 'paperspace']

            # Remove current platform and suspension platform
            if current_platform.value in available_platforms:
                available_platforms.remove(current_platform.value)
            if suspension_point.platform in available_platforms:
                available_platforms.remove(suspension_point.platform)

            if not available_platforms:
                recovery_attempt.error_message = "No alternative platforms available"
                return RecoveryOutcome.FAILED

            # Select best available platform (simple round-robin for now)
            new_platform = available_platforms[0]

            # Update job platform preference
            job.config.platform_preference = new_platform
            job.platform = None  # Clear current platform

            # Priority boost if configured
            if self.config.priority_boost_on_resume:
                if job.priority.value > 0:
                    job.priority = type(job.priority)(job.priority.value - 1)

            # Include checkpoint if available
            recovery_metadata = {
                'platform_switched': True,
                'from_platform': suspension_point.platform,
                'to_platform': new_platform
            }

            if suspension_point.checkpoint_path:
                job.progress = job.progress or {}
                job.progress['checkpoint_path'] = suspension_point.checkpoint_path
                recovery_metadata['checkpoint_available'] = True

            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'recovery_strategy': RecoveryStrategy.PLATFORM_SWITCH.value,
                    'recovery_time': datetime.now().isoformat(),
                    'platform': new_platform,
                    'progress': job.progress
                }
            )

            if success:
                recovery_attempt.new_platform = new_platform
                recovery_attempt.recovery_metadata = recovery_metadata

                # Remove from suspension manager
                if job.id in self.suspension_manager.active_suspensions:
                    del self.suspension_manager.active_suspensions[job.id]

                return RecoveryOutcome.SUCCESS
            else:
                return RecoveryOutcome.FAILED

        except Exception as e:
            recovery_attempt.error_message = str(e)
            return RecoveryOutcome.FAILED

    async def _delayed_resume_recovery(self, job: ExperimentJob,
                                     suspension_point: SuspensionPoint,
                                     recovery_attempt: RecoveryAttempt) -> RecoveryOutcome:
        """Execute delayed resume recovery."""
        try:
            # Calculate delay based on suspension reason and system load
            delay_minutes = self._calculate_resume_delay(suspension_point)

            # Schedule for delayed resume
            resume_time = datetime.now() + timedelta(minutes=delay_minutes)

            job.progress = job.progress or {}
            job.progress['delayed_resume'] = True
            job.progress['resume_time'] = resume_time.isoformat()
            job.progress['delay_reason'] = suspension_point.reason.value

            # For now, we'll queue it immediately but mark it as delayed
            # In a full implementation, this would use a scheduler
            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'recovery_strategy': RecoveryStrategy.DELAYED_RESUME.value,
                    'recovery_time': datetime.now().isoformat(),
                    'delayed_resume_time': resume_time.isoformat(),
                    'delay_minutes': delay_minutes,
                    'progress': job.progress
                }
            )

            if success:
                recovery_attempt.recovery_metadata = {
                    'delayed_resume': True,
                    'delay_minutes': delay_minutes,
                    'resume_time': resume_time.isoformat()
                }

                # Remove from suspension manager
                if job.id in self.suspension_manager.active_suspensions:
                    del self.suspension_manager.active_suspensions[job.id]

                return RecoveryOutcome.SUCCESS
            else:
                return RecoveryOutcome.FAILED

        except Exception as e:
            recovery_attempt.error_message = str(e)
            return RecoveryOutcome.FAILED

    def _calculate_resume_delay(self, suspension_point: SuspensionPoint) -> int:
        """Calculate appropriate delay for resume."""
        base_delay = self.config.recovery_delay_minutes

        # Adjust based on suspension reason
        if suspension_point.reason == SuspensionReason.RESOURCE_CONSTRAINT:
            return base_delay * 3  # Wait longer for resource issues
        elif suspension_point.reason == SuspensionReason.PLATFORM_ROTATION:
            return base_delay // 2  # Shorter delay for rotation
        elif suspension_point.reason == SuspensionReason.SESSION_TIMEOUT_WARNING:
            return base_delay * 2  # Medium delay for timeouts

        return base_delay

    async def _check_for_suspended_jobs(self):
        """Check for suspended jobs that need recovery."""
        try:
            suspended_jobs = self.experiment_queue.list_jobs(status_filter=ExperimentStatus.SUSPENDED)

            for job in suspended_jobs:
                if job.id not in self.recovery_queue and job.id not in self.active_recoveries:
                    # Check if job is eligible for recovery
                    if await self._is_job_eligible_for_recovery(job):
                        self.recovery_queue.append(job.id)
                        self.logger.info("suspended_job_queued_for_recovery", job_id=job.id)

        except Exception as e:
            self.logger.error("suspended_jobs_check_failed", error=str(e))

    async def _is_job_eligible_for_recovery(self, job: ExperimentJob) -> bool:
        """Check if job is eligible for recovery."""
        # Check if too much time has passed
        if job.metadata.updated_at:
            time_since_update = datetime.now() - job.metadata.updated_at
            if time_since_update > timedelta(hours=self.config.max_recovery_time_hours):
                return False

        # Check if max attempts reached
        attempts = self.recovery_attempts.get(job.id, [])
        if len(attempts) >= self.config.max_recovery_attempts:
            return False

        return True

    async def _handle_recovery_exhausted(self, job_id: str):
        """Handle case where recovery attempts are exhausted."""
        try:
            # Mark job as failed
            self.experiment_queue.update_job_status(
                job_id,
                ExperimentStatus.FAILED,
                {'recovery_exhausted': True, 'max_recovery_attempts_reached': True}
            )

            # Send notification
            job = self.experiment_queue.get_job(job_id)
            if job:
                await self.notification_system.send_experiment_failed(job)

            self.logger.warning("recovery_exhausted", job_id=job_id)

        except Exception as e:
            self.logger.error("recovery_exhausted_handling_failed", job_id=job_id, error=str(e))

    async def _send_recovery_notification(self, job: ExperimentJob, recovery_attempt: RecoveryAttempt):
        """Send recovery success notification."""
        try:
            body = f"Experiment '{job.config.name}' has been successfully recovered using {recovery_attempt.strategy.value} strategy."

            if recovery_attempt.new_platform:
                body += f" Switched to platform: {recovery_attempt.new_platform}"

            # For now, just log the notification
            # In a full implementation, this would use the notification system
            self.logger.info("recovery_notification_sent",
                           job_id=job.id,
                           strategy=recovery_attempt.strategy.value)

        except Exception as e:
            self.logger.error("recovery_notification_failed", job_id=job.id, error=str(e))

    def _on_session_restored(self, session_info):
        """Handle session restoration callback."""
        self.logger.info("session_restored_detected", platform=session_info.platform.value)

        # Trigger recovery check
        asyncio.create_task(self._check_for_suspended_jobs())

    def _on_job_failed(self, job: ExperimentJob):
        """Handle job failure callback."""
        # If job was in recovery, this might indicate recovery failure
        if job.id in self.active_recoveries:
            self.logger.warning("job_failed_during_recovery", job_id=job.id)

    def _save_recovery_state(self):
        """Save recovery state to disk."""
        try:
            state_file = self.recovery_dir / "recovery_state.json"

            # Serialize recovery attempts
            serialized_attempts = {}
            for job_id, attempts in self.recovery_attempts.items():
                serialized_attempts[job_id] = [attempt.to_dict() for attempt in attempts]

            state = {
                'recovery_attempts': serialized_attempts,
                'recovery_queue': self.recovery_queue,
                'last_saved': datetime.now().isoformat()
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error("recovery_state_save_failed", error=str(e))

    def _load_recovery_state(self):
        """Load recovery state from disk."""
        try:
            state_file = self.recovery_dir / "recovery_state.json"

            if not state_file.exists():
                return

            with open(state_file) as f:
                state = json.load(f)

            # Load recovery attempts
            for job_id, attempts_data in state.get('recovery_attempts', {}).items():
                attempts = []
                for attempt_data in attempts_data:
                    try:
                        attempt = RecoveryAttempt.from_dict(attempt_data)
                        attempts.append(attempt)
                    except Exception as e:
                        self.logger.error("recovery_attempt_load_failed",
                                        job_id=job_id, error=str(e))

                if attempts:
                    self.recovery_attempts[job_id] = attempts

            # Load recovery queue
            self.recovery_queue = state.get('recovery_queue', [])

            self.logger.info("recovery_state_loaded",
                           jobs_with_attempts=len(self.recovery_attempts),
                           queued_recoveries=len(self.recovery_queue))

        except Exception as e:
            self.logger.error("recovery_state_load_failed", error=str(e))

    def get_recovery_statistics(self) -> dict[str, Any]:
        """Get recovery statistics."""
        total_attempts = sum(len(attempts) for attempts in self.recovery_attempts.values())

        outcome_counts = {}
        strategy_counts = {}

        for attempts in self.recovery_attempts.values():
            for attempt in attempts:
                outcome = attempt.outcome.value
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

                strategy = attempt.strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            'auto_recovery_enabled': self.config.auto_recovery_enabled,
            'running': self.running,
            'total_jobs_with_attempts': len(self.recovery_attempts),
            'total_recovery_attempts': total_attempts,
            'active_recoveries': len(self.active_recoveries),
            'queued_recoveries': len(self.recovery_queue),
            'outcome_distribution': outcome_counts,
            'strategy_distribution': strategy_counts,
            'success_rate': outcome_counts.get('success', 0) / max(total_attempts, 1)
        }


# Singleton instance
_recovery_manager = None


def get_recovery_manager(config: RecoveryConfig | None = None) -> RecoveryManager:
    """Get singleton recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager(config)
    return _recovery_manager
