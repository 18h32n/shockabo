"""Platform rotation orchestrator for automated experiment execution.

This module manages platform rotation, queue persistence across switches,
and coordinated experiment execution across multiple platforms.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from ..services.training_orchestrator import get_training_orchestrator
from .experiment_queue import (
    ExperimentJob,
    ExperimentStatus,
    get_experiment_queue,
)
from .graceful_shutdown import ShutdownReason, get_shutdown_manager
from .platform_availability import AvailabilityStatus, get_availability_checker
from .platform_detector import Platform, get_platform_detector

logger = structlog.get_logger(__name__)


class RotationStrategy(Enum):
    """Platform rotation strategies."""
    ROUND_ROBIN = "round_robin"
    QUOTA_BASED = "quota_based"
    AVAILABILITY_BASED = "availability_based"
    INTELLIGENT = "intelligent"


@dataclass
class PlatformSession:
    """Platform session information."""
    platform: Platform
    start_time: datetime
    estimated_end_time: datetime
    current_job: str | None = None
    jobs_completed: int = 0
    total_runtime_minutes: int = 0
    gpu_utilization: float = 0.0
    quota_remaining_hours: float = 0.0

    def is_near_timeout(self, warning_minutes: int = 30) -> bool:
        """Check if session is near timeout."""
        if not self.estimated_end_time:
            return False

        warning_time = datetime.now() + timedelta(minutes=warning_minutes)
        return warning_time >= self.estimated_end_time

    def get_remaining_minutes(self) -> float:
        """Get remaining session time in minutes."""
        if not self.estimated_end_time:
            return float('inf')

        remaining = self.estimated_end_time - datetime.now()
        return max(0, remaining.total_seconds() / 60)


class PlatformRotationOrchestrator:
    """Orchestrates experiment execution across multiple platforms with rotation."""

    def __init__(self,
                 rotation_strategy: RotationStrategy = RotationStrategy.INTELLIGENT,
                 state_dir: Path | None = None):
        """Initialize platform rotation orchestrator.

        Args:
            rotation_strategy: Strategy for platform rotation
            state_dir: Directory to persist orchestrator state
        """
        self.rotation_strategy = rotation_strategy
        self.state_dir = state_dir or Path.home() / ".arc-orchestrator"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.platform_detector = get_platform_detector()
        self.availability_checker = get_availability_checker()
        self.shutdown_manager = get_shutdown_manager()
        self.experiment_queue = get_experiment_queue()
        self.training_orchestrator = get_training_orchestrator()

        # State
        self.current_session: PlatformSession | None = None
        self.platform_sessions: dict[str, PlatformSession] = {}
        self.running = False
        self.orchestration_task: asyncio.Task | None = None

        # Configuration
        self.platform_configs = {
            Platform.KAGGLE: {
                'max_session_hours': 12,
                'quota_hours_weekly': 30,
                'warning_minutes': 30,
                'cooldown_minutes': 60
            },
            Platform.COLAB: {
                'max_session_hours': 12,
                'quota_hours_daily': 12,
                'warning_minutes': 30,
                'cooldown_minutes': 30
            },
            Platform.PAPERSPACE: {
                'max_session_hours': 6,
                'quota_hours_daily': 6,
                'warning_minutes': 20,
                'cooldown_minutes': 120
            }
        }

        # Callbacks
        self.callbacks: dict[str, list[Callable]] = {
            'rotation_started': [],
            'rotation_completed': [],
            'job_suspended': [],
            'job_resumed': [],
            'platform_switched': []
        }

        self.logger = structlog.get_logger('platform_orchestrator')

        # Register shutdown hook
        self.shutdown_manager.register_shutdown_hook(
            self._shutdown_hook, async_hook=True
        )

        # Register queue callbacks
        self.experiment_queue.register_callback('job_added', self._on_job_added)
        self.experiment_queue.register_callback('job_completed', self._on_job_completed)
        self.experiment_queue.register_callback('job_failed', self._on_job_failed)

        # Load state
        self._load_state()

    async def start_orchestration(self) -> bool:
        """Start the platform rotation orchestration.

        Returns:
            True if started successfully
        """
        if self.running:
            self.logger.warning("orchestration_already_running")
            return False

        try:
            self.running = True

            # Detect current platform
            platform_info = self.platform_detector.detect_platform()
            current_platform = platform_info.platform

            # Initialize session
            await self._initialize_session(current_platform)

            # Start orchestration loop
            self.orchestration_task = asyncio.create_task(self._orchestration_loop())

            self.logger.info("orchestration_started", platform=current_platform.value)
            return True

        except Exception as e:
            self.logger.error("orchestration_start_failed", error=str(e))
            self.running = False
            return False

    async def stop_orchestration(self) -> bool:
        """Stop the platform rotation orchestration.

        Returns:
            True if stopped successfully
        """
        if not self.running:
            return True

        try:
            self.running = False

            if self.orchestration_task:
                self.orchestration_task.cancel()
                try:
                    await self.orchestration_task
                except asyncio.CancelledError:
                    pass

            # Suspend current job if running
            if self.current_session and self.current_session.current_job:
                await self._suspend_current_job("orchestration_stopped")

            # Save state
            self._save_state()

            self.logger.info("orchestration_stopped")
            return True

        except Exception as e:
            self.logger.error("orchestration_stop_failed", error=str(e))
            return False

    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.running:
            try:
                # Check current session status
                if self.current_session:
                    # Check for timeout warning
                    if self.current_session.is_near_timeout():
                        await self._handle_session_timeout_warning()

                    # Check if current job is still running
                    if self.current_session.current_job:
                        job = self.experiment_queue.get_job(self.current_session.current_job)
                        if job and job.status != ExperimentStatus.RUNNING:
                            # Job finished, update session
                            self.current_session.current_job = None
                            self.current_session.jobs_completed += 1

                            # Update total runtime
                            if job.started_at and job.completed_at:
                                runtime = job.completed_at - job.started_at
                                self.current_session.total_runtime_minutes += runtime.total_seconds() / 60

                # Try to get next job
                next_job = await self._get_next_suitable_job()

                if next_job:
                    # Execute job
                    await self._execute_job(next_job)
                else:
                    # No suitable jobs, check if we should rotate platforms
                    if await self._should_rotate_platform():
                        await self._rotate_to_best_platform()

                # Save state periodically
                self._save_state()

                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("orchestration_loop_error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error

    async def _initialize_session(self, platform: Platform):
        """Initialize session for current platform.

        Args:
            platform: Current platform
        """
        config = self.platform_configs.get(platform, {})
        max_hours = config.get('max_session_hours', 12)

        # Check availability
        availability = await self.availability_checker.check_platform_availability(platform)

        self.current_session = PlatformSession(
            platform=platform,
            start_time=datetime.now(),
            estimated_end_time=datetime.now() + timedelta(hours=max_hours),
            quota_remaining_hours=availability.quota.remaining_hours if availability.quota else max_hours
        )

        self.platform_sessions[platform.value] = self.current_session

        self.logger.info("session_initialized",
                        platform=platform.value,
                        max_hours=max_hours,
                        quota_remaining=self.current_session.quota_remaining_hours)

    async def _get_next_suitable_job(self) -> ExperimentJob | None:
        """Get next suitable job considering platform constraints.

        Returns:
            Next suitable job or None
        """
        if not self.current_session:
            return None

        if self.current_session.current_job:
            return None  # Already running a job

        platform_name = self.current_session.platform.value
        remaining_minutes = self.current_session.get_remaining_minutes()

        # Get next job with platform preference
        job = self.experiment_queue.get_next_job(platform_filter=platform_name)

        # If no platform-specific job, try any job that can fit
        if not job:
            job = self.experiment_queue.get_next_job()

            # Check if job can fit in remaining time
            if job and job.config.max_runtime_minutes > remaining_minutes - 10:  # 10 min buffer
                return None  # Job too long for remaining session time

        return job

    async def _execute_job(self, job: ExperimentJob):
        """Execute experiment job.

        Args:
            job: Job to execute
        """
        try:
            # Update job status
            self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.RUNNING,
                {'platform': self.current_session.platform.value}
            )

            # Update session
            self.current_session.current_job = job.id

            self.logger.info("job_execution_started",
                            job_id=job.id,
                            platform=self.current_session.platform.value)

            # Execute job asynchronously (don't await here)
            asyncio.create_task(self._run_job_async(job))

        except Exception as e:
            self.logger.error("job_execution_failed", job_id=job.id, error=str(e))
            self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.FAILED,
                {'last_error': str(e)}
            )

    async def _run_job_async(self, job: ExperimentJob):
        """Run job asynchronously.

        Args:
            job: Job to run
        """
        try:
            # Use training orchestrator to run the experiment
            success = await self.training_orchestrator.run_experiment_async(job)

            if success:
                self.experiment_queue.update_job_status(job.id, ExperimentStatus.COMPLETED)
            else:
                self.experiment_queue.update_job_status(job.id, ExperimentStatus.FAILED)

        except Exception as e:
            self.logger.error("async_job_execution_failed", job_id=job.id, error=str(e))
            self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.FAILED,
                {'last_error': str(e)}
            )

    async def _should_rotate_platform(self) -> bool:
        """Determine if we should rotate to a different platform.

        Returns:
            True if rotation is recommended
        """
        if not self.current_session:
            return False

        # Check if current session is near timeout
        if self.current_session.is_near_timeout():
            return True

        # Check if no jobs are queued for current platform
        queued_jobs = self.experiment_queue.list_jobs(
            status_filter=ExperimentStatus.QUEUED
        )

        platform_jobs = [
            job for job in queued_jobs
            if job.config.platform_preference == self.current_session.platform.value
        ]

        if not platform_jobs and not queued_jobs:
            return False  # No jobs to process

        # Use rotation strategy
        if self.rotation_strategy == RotationStrategy.INTELLIGENT:
            return await self._intelligent_rotation_decision()
        elif self.rotation_strategy == RotationStrategy.QUOTA_BASED:
            return await self._quota_based_rotation_decision()
        elif self.rotation_strategy == RotationStrategy.AVAILABILITY_BASED:
            return await self._availability_based_rotation_decision()

        return False

    async def _intelligent_rotation_decision(self) -> bool:
        """Make intelligent rotation decision based on multiple factors.

        Returns:
            True if rotation is recommended
        """
        # Check all platforms for availability and queue fit
        best_platform = await self._find_best_platform()

        if not best_platform:
            return False

        current_platform = self.current_session.platform

        # Rotate if better platform is available and current is suboptimal
        if (best_platform != current_platform and
            self.current_session.get_remaining_minutes() < 120):  # Less than 2 hours
            return True

        return False

    async def _quota_based_rotation_decision(self) -> bool:
        """Make rotation decision based on quota utilization.

        Returns:
            True if rotation is recommended
        """
        if self.current_session.quota_remaining_hours < 1:  # Less than 1 hour quota
            return True

        return False

    async def _availability_based_rotation_decision(self) -> bool:
        """Make rotation decision based on platform availability.

        Returns:
            True if rotation is recommended
        """
        availability = await self.availability_checker.check_platform_availability(
            self.current_session.platform
        )

        return availability.status != AvailabilityStatus.AVAILABLE

    async def _find_best_platform(self) -> Platform | None:
        """Find the best available platform for next jobs.

        Returns:
            Best platform or None if none available
        """
        platform_scores = {}

        for platform in Platform:
            if platform == Platform.LOCAL:
                continue  # Skip local for rotation

            try:
                availability = await self.availability_checker.check_platform_availability(platform)

                if availability.status != AvailabilityStatus.AVAILABLE:
                    continue

                # Score based on quota, availability, and queue fit
                score = 0

                # Quota score (0-100)
                if availability.quota:
                    quota_ratio = availability.quota.remaining_hours / availability.quota.total_hours
                    score += quota_ratio * 100

                # Session time score (0-50)
                config = self.platform_configs.get(platform, {})
                max_hours = config.get('max_session_hours', 12)
                score += min(max_hours / 12, 1) * 50

                # Queue fit score (0-50)
                queued_jobs = self.experiment_queue.list_jobs(ExperimentStatus.QUEUED)
                suitable_jobs = sum(
                    1 for job in queued_jobs
                    if job.config.max_runtime_minutes <= max_hours * 60 - 60  # 1 hour buffer
                )
                score += min(suitable_jobs / 10, 1) * 50

                platform_scores[platform] = score

            except Exception as e:
                self.logger.error("platform_scoring_failed", platform=platform.value, error=str(e))

        if not platform_scores:
            return None

        # Return platform with highest score
        return max(platform_scores, key=platform_scores.get)

    async def _rotate_to_best_platform(self):
        """Rotate to the best available platform."""
        best_platform = await self._find_best_platform()

        if not best_platform:
            self.logger.warning("no_suitable_platform_for_rotation")
            return

        if best_platform == self.current_session.platform:
            return  # Already on best platform

        await self._rotate_to_platform(best_platform)

    async def _rotate_to_platform(self, target_platform: Platform):
        """Rotate to specified platform.

        Args:
            target_platform: Platform to rotate to
        """
        try:
            self.logger.info("platform_rotation_started",
                            from_platform=self.current_session.platform.value,
                            to_platform=target_platform.value)

            # Suspend current job if running
            if self.current_session.current_job:
                await self._suspend_current_job("platform_rotation")

            # Trigger callbacks
            self._trigger_callbacks('rotation_started', {
                'from_platform': self.current_session.platform.value,
                'to_platform': target_platform.value
            })

            # Save current session state
            self._save_state()

            # Initiate graceful shutdown for rotation
            self.shutdown_manager.initiate_shutdown(
                reason=ShutdownReason.PLATFORM_ROTATION,
                metadata={'target_platform': target_platform.value}
            )

        except Exception as e:
            self.logger.error("platform_rotation_failed", error=str(e))

    async def _suspend_current_job(self, reason: str):
        """Suspend the currently running job.

        Args:
            reason: Reason for suspension
        """
        if not self.current_session.current_job:
            return

        job_id = self.current_session.current_job

        try:
            # Update job status
            self.experiment_queue.update_job_status(
                job_id,
                ExperimentStatus.SUSPENDED,
                {'suspension_reason': reason, 'suspended_at': datetime.now().isoformat()}
            )

            # Clear current job
            self.current_session.current_job = None

            # Trigger callbacks
            job = self.experiment_queue.get_job(job_id)
            if job:
                self._trigger_callbacks('job_suspended', job)

            self.logger.info("job_suspended", job_id=job_id, reason=reason)

        except Exception as e:
            self.logger.error("job_suspension_failed", job_id=job_id, error=str(e))

    async def _handle_session_timeout_warning(self):
        """Handle session timeout warning."""
        if not self.current_session:
            return

        self.logger.warning("session_timeout_approaching",
                          platform=self.current_session.platform.value,
                          remaining_minutes=self.current_session.get_remaining_minutes())

        # Suspend current job
        if self.current_session.current_job:
            await self._suspend_current_job("session_timeout_warning")

        # Try to rotate to another platform
        await self._rotate_to_best_platform()

    def _on_job_added(self, job: ExperimentJob):
        """Handle job added to queue."""
        self.logger.info("job_added_to_queue", job_id=job.id, priority=job.priority.name)

    def _on_job_completed(self, job: ExperimentJob):
        """Handle job completion."""
        self.logger.info("job_completed", job_id=job.id, platform=job.platform)

        # Update session stats if it's our current session
        if (self.current_session and
            self.current_session.current_job == job.id):
            self.current_session.current_job = None
            self.current_session.jobs_completed += 1

    def _on_job_failed(self, job: ExperimentJob):
        """Handle job failure."""
        self.logger.error("job_failed", job_id=job.id, error=job.last_error)

        # Try to retry if retries are available
        if job.retry_count < job.max_retries:
            self.experiment_queue.retry_job(job.id)

    async def _shutdown_hook(self, shutdown_event):
        """Handle shutdown event."""
        self.logger.info("orchestrator_shutdown_initiated", reason=shutdown_event.reason.value)
        await self.stop_orchestration()

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for orchestration events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for an event.

        Args:
            event_type: Event type
            data: Event data
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error("callback_failed", event_type=event_type, error=str(e))

    def _save_state(self):
        """Save orchestrator state."""
        try:
            state_file = self.state_dir / "orchestrator_state.json"

            state = {
                'current_session': None,
                'platform_sessions': {},
                'rotation_strategy': self.rotation_strategy.value,
                'last_saved': datetime.now().isoformat()
            }

            if self.current_session:
                state['current_session'] = {
                    'platform': self.current_session.platform.value,
                    'start_time': self.current_session.start_time.isoformat(),
                    'estimated_end_time': self.current_session.estimated_end_time.isoformat(),
                    'current_job': self.current_session.current_job,
                    'jobs_completed': self.current_session.jobs_completed,
                    'total_runtime_minutes': self.current_session.total_runtime_minutes,
                    'quota_remaining_hours': self.current_session.quota_remaining_hours
                }

            # Save platform sessions
            for platform_name, session in self.platform_sessions.items():
                state['platform_sessions'][platform_name] = {
                    'platform': session.platform.value,
                    'start_time': session.start_time.isoformat(),
                    'estimated_end_time': session.estimated_end_time.isoformat(),
                    'jobs_completed': session.jobs_completed,
                    'total_runtime_minutes': session.total_runtime_minutes,
                    'quota_remaining_hours': session.quota_remaining_hours
                }

            import json
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error("state_save_failed", error=str(e))

    def _load_state(self):
        """Load orchestrator state."""
        try:
            state_file = self.state_dir / "orchestrator_state.json"

            if not state_file.exists():
                return

            import json
            with open(state_file) as f:
                state = json.load(f)

            # Load rotation strategy
            if 'rotation_strategy' in state:
                try:
                    self.rotation_strategy = RotationStrategy(state['rotation_strategy'])
                except ValueError:
                    pass

            # Load platform sessions
            for platform_name, session_data in state.get('platform_sessions', {}).items():
                try:
                    platform = Platform(session_data['platform'])
                    session = PlatformSession(
                        platform=platform,
                        start_time=datetime.fromisoformat(session_data['start_time']),
                        estimated_end_time=datetime.fromisoformat(session_data['estimated_end_time']),
                        jobs_completed=session_data.get('jobs_completed', 0),
                        total_runtime_minutes=session_data.get('total_runtime_minutes', 0),
                        quota_remaining_hours=session_data.get('quota_remaining_hours', 0)
                    )
                    self.platform_sessions[platform_name] = session
                except Exception as e:
                    self.logger.error("session_load_failed", platform=platform_name, error=str(e))

            self.logger.info("orchestrator_state_loaded")

        except Exception as e:
            self.logger.error("state_load_failed", error=str(e))

    def get_orchestration_stats(self) -> dict[str, Any]:
        """Get orchestration statistics.

        Returns:
            Dictionary with orchestration statistics
        """
        stats = {
            'running': self.running,
            'rotation_strategy': self.rotation_strategy.value,
            'current_session': None,
            'platform_sessions': {},
            'queue_stats': self.experiment_queue.get_queue_stats()
        }

        if self.current_session:
            stats['current_session'] = {
                'platform': self.current_session.platform.value,
                'remaining_minutes': self.current_session.get_remaining_minutes(),
                'current_job': self.current_session.current_job,
                'jobs_completed': self.current_session.jobs_completed,
                'total_runtime_minutes': self.current_session.total_runtime_minutes,
                'quota_remaining_hours': self.current_session.quota_remaining_hours
            }

        for platform_name, session in self.platform_sessions.items():
            stats['platform_sessions'][platform_name] = {
                'jobs_completed': session.jobs_completed,
                'total_runtime_minutes': session.total_runtime_minutes,
                'quota_remaining_hours': session.quota_remaining_hours
            }

        return stats


# Singleton instance
_platform_orchestrator = None


def get_platform_orchestrator() -> PlatformRotationOrchestrator:
    """Get singleton platform rotation orchestrator instance."""
    global _platform_orchestrator
    if _platform_orchestrator is None:
        _platform_orchestrator = PlatformRotationOrchestrator()
    return _platform_orchestrator
