"""Experiment retry management with intelligent retry strategies.

This module provides sophisticated retry logic for failed experiment runs,
including backoff strategies, failure analysis, and platform-aware retries.
"""

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .experiment_queue import (
    ExperimentJob,
    ExperimentStatus,
    get_experiment_queue,
)

logger = structlog.get_logger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    IMMEDIATE = "immediate"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    PLATFORM_AWARE = "platform_aware"
    INTELLIGENT = "intelligent"


class FailureCategory(Enum):
    """Failure categories for retry decisions."""
    PLATFORM_ERROR = "platform_error"        # Platform/infrastructure issues
    RESOURCE_ERROR = "resource_error"        # Memory, GPU, storage issues
    TIMEOUT_ERROR = "timeout_error"          # Session or job timeouts
    DATA_ERROR = "data_error"               # Data loading/processing issues
    MODEL_ERROR = "model_error"             # Model loading/inference issues
    NETWORK_ERROR = "network_error"         # Network connectivity issues
    UNKNOWN_ERROR = "unknown_error"         # Unknown/unclassified errors


@dataclass
class RetryConfig:
    """Retry configuration for different failure types."""
    strategy: RetryStrategy
    max_retries: int
    base_delay_seconds: int
    max_delay_seconds: int
    backoff_multiplier: float
    jitter_enabled: bool
    platform_rotation_on_failure: bool

    @classmethod
    def default(cls) -> 'RetryConfig':
        """Create default retry configuration."""
        return cls(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=3,
            base_delay_seconds=60,
            max_delay_seconds=3600,  # 1 hour
            backoff_multiplier=2.0,
            jitter_enabled=True,
            platform_rotation_on_failure=True
        )

    @classmethod
    def for_failure_category(cls, category: FailureCategory) -> 'RetryConfig':
        """Create retry configuration for specific failure category."""
        configs = {
            FailureCategory.PLATFORM_ERROR: cls(
                strategy=RetryStrategy.PLATFORM_AWARE,
                max_retries=5,
                base_delay_seconds=300,  # 5 minutes
                max_delay_seconds=1800,  # 30 minutes
                backoff_multiplier=1.5,
                jitter_enabled=True,
                platform_rotation_on_failure=True
            ),
            FailureCategory.RESOURCE_ERROR: cls(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_retries=3,
                base_delay_seconds=600,  # 10 minutes
                max_delay_seconds=3600,  # 1 hour
                backoff_multiplier=1.0,
                jitter_enabled=False,
                platform_rotation_on_failure=True
            ),
            FailureCategory.TIMEOUT_ERROR: cls(
                strategy=RetryStrategy.PLATFORM_AWARE,
                max_retries=2,
                base_delay_seconds=1800,  # 30 minutes
                max_delay_seconds=7200,   # 2 hours
                backoff_multiplier=2.0,
                jitter_enabled=True,
                platform_rotation_on_failure=True
            ),
            FailureCategory.DATA_ERROR: cls(
                strategy=RetryStrategy.FIXED_DELAY,
                max_retries=2,
                base_delay_seconds=120,  # 2 minutes
                max_delay_seconds=300,   # 5 minutes
                backoff_multiplier=1.0,
                jitter_enabled=False,
                platform_rotation_on_failure=False
            ),
            FailureCategory.MODEL_ERROR: cls(
                strategy=RetryStrategy.FIXED_DELAY,
                max_retries=2,
                base_delay_seconds=180,  # 3 minutes
                max_delay_seconds=300,   # 5 minutes
                backoff_multiplier=1.0,
                jitter_enabled=False,
                platform_rotation_on_failure=False
            ),
            FailureCategory.NETWORK_ERROR: cls(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=4,
                base_delay_seconds=30,
                max_delay_seconds=600,   # 10 minutes
                backoff_multiplier=2.0,
                jitter_enabled=True,
                platform_rotation_on_failure=False
            ),
            FailureCategory.UNKNOWN_ERROR: cls(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=2,
                base_delay_seconds=300,  # 5 minutes
                max_delay_seconds=1800,  # 30 minutes
                backoff_multiplier=2.0,
                jitter_enabled=True,
                platform_rotation_on_failure=False
            )
        }

        return configs.get(category, cls.default())


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    scheduled_time: datetime
    failure_category: FailureCategory
    retry_config: RetryConfig
    platform_preference: str | None = None
    metadata: dict[str, Any] = None


class FailureAnalyzer:
    """Analyzes job failures to determine retry strategy."""

    def __init__(self):
        self.error_patterns = {
            FailureCategory.PLATFORM_ERROR: [
                'platform', 'session', 'kernel', 'runtime', 'notebook',
                'connection', 'authentication', 'quota', 'limit'
            ],
            FailureCategory.RESOURCE_ERROR: [
                'memory', 'gpu', 'cuda', 'storage', 'disk', 'space',
                'allocation', 'oom', 'out of memory'
            ],
            FailureCategory.TIMEOUT_ERROR: [
                'timeout', 'time out', 'timed out', 'expired',
                'deadline', 'killed', 'terminated'
            ],
            FailureCategory.DATA_ERROR: [
                'data', 'dataset', 'loading', 'file not found',
                'corrupt', 'invalid', 'parse', 'format'
            ],
            FailureCategory.MODEL_ERROR: [
                'model', 'checkpoint', 'weights', 'layer',
                'forward', 'backward', 'gradient'
            ],
            FailureCategory.NETWORK_ERROR: [
                'network', 'connection', 'http', 'ssl', 'certificate',
                'dns', 'resolve', 'download', 'upload'
            ]
        }

        self.logger = structlog.get_logger('failure_analyzer')

    def analyze_failure(self, job: ExperimentJob) -> FailureCategory:
        """Analyze job failure and categorize it.

        Args:
            job: Failed experiment job

        Returns:
            Failure category
        """
        if not job.last_error:
            return FailureCategory.UNKNOWN_ERROR

        error_text = job.last_error.lower()

        # Score each category based on keyword matches
        category_scores = {}
        for category, keywords in self.error_patterns.items():
            score = sum(1 for keyword in keywords if keyword in error_text)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            self.logger.info("failure_categorized",
                           job_id=job.id,
                           category=best_category.value,
                           error=job.last_error)
            return best_category

        return FailureCategory.UNKNOWN_ERROR


class ExperimentRetryManager:
    """Manages experiment retries with intelligent strategies."""

    def __init__(self):
        """Initialize retry manager."""
        self.experiment_queue = get_experiment_queue()
        self.failure_analyzer = FailureAnalyzer()

        # State
        self.retry_schedules: dict[str, RetryAttempt] = {}  # job_id -> retry_attempt
        self.retry_history: dict[str, list[RetryAttempt]] = {}  # job_id -> attempts

        # Retry task management
        self.retry_tasks: dict[str, asyncio.Task] = {}
        self.running = False

        self.logger = structlog.get_logger('retry_manager')

        # Register queue callbacks
        self.experiment_queue.register_callback('job_failed', self._on_job_failed)
        self.experiment_queue.register_callback('job_completed', self._on_job_completed)

    async def start_retry_management(self) -> bool:
        """Start retry management.

        Returns:
            True if started successfully
        """
        if self.running:
            self.logger.warning("retry_management_already_running")
            return False

        try:
            self.running = True
            self.logger.info("retry_management_started")
            return True

        except Exception as e:
            self.logger.error("retry_management_start_failed", error=str(e))
            self.running = False
            return False

    async def stop_retry_management(self) -> bool:
        """Stop retry management.

        Returns:
            True if stopped successfully
        """
        if not self.running:
            return True

        try:
            self.running = False

            # Cancel all pending retry tasks
            for task in self.retry_tasks.values():
                task.cancel()

            # Wait for tasks to complete
            if self.retry_tasks:
                await asyncio.gather(*self.retry_tasks.values(), return_exceptions=True)

            self.retry_tasks.clear()

            self.logger.info("retry_management_stopped")
            return True

        except Exception as e:
            self.logger.error("retry_management_stop_failed", error=str(e))
            return False

    def _on_job_failed(self, job: ExperimentJob):
        """Handle job failure event.

        Args:
            job: Failed job
        """
        if not self.running:
            return

        self.logger.info("job_failure_detected", job_id=job.id, error=job.last_error)

        # Schedule retry if eligible
        asyncio.create_task(self._schedule_retry(job))

    def _on_job_completed(self, job: ExperimentJob):
        """Handle job completion event.

        Args:
            job: Completed job
        """
        # Clean up retry state for completed job
        if job.id in self.retry_schedules:
            del self.retry_schedules[job.id]

        if job.id in self.retry_tasks:
            self.retry_tasks[job.id].cancel()
            del self.retry_tasks[job.id]

    async def _schedule_retry(self, job: ExperimentJob):
        """Schedule retry for failed job.

        Args:
            job: Failed job to retry
        """
        try:
            # Check if job is eligible for retry
            if job.retry_count >= job.max_retries:
                self.logger.info("job_retry_limit_reached", job_id=job.id)
                return

            # Analyze failure
            failure_category = self.failure_analyzer.analyze_failure(job)
            retry_config = RetryConfig.for_failure_category(failure_category)

            # Override max retries if job-specific limit is lower
            if job.max_retries < retry_config.max_retries:
                retry_config.max_retries = job.max_retries

            # Check if we've exceeded category-specific retry limit
            if job.retry_count >= retry_config.max_retries:
                self.logger.info("job_category_retry_limit_reached",
                               job_id=job.id,
                               category=failure_category.value)
                return

            # Calculate retry delay
            delay_seconds = self._calculate_retry_delay(job, retry_config)
            scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)

            # Determine platform preference for retry
            platform_preference = self._determine_retry_platform(job, failure_category, retry_config)

            # Create retry attempt
            retry_attempt = RetryAttempt(
                attempt_number=job.retry_count + 1,
                scheduled_time=scheduled_time,
                failure_category=failure_category,
                retry_config=retry_config,
                platform_preference=platform_preference,
                metadata={
                    'original_platform': job.platform,
                    'failure_time': datetime.now().isoformat(),
                    'delay_seconds': delay_seconds
                }
            )

            # Store retry schedule
            self.retry_schedules[job.id] = retry_attempt

            # Add to retry history
            if job.id not in self.retry_history:
                self.retry_history[job.id] = []
            self.retry_history[job.id].append(retry_attempt)

            # Schedule retry task
            retry_task = asyncio.create_task(
                self._execute_retry(job, retry_attempt, delay_seconds)
            )
            self.retry_tasks[job.id] = retry_task

            self.logger.info("retry_scheduled",
                           job_id=job.id,
                           attempt=retry_attempt.attempt_number,
                           category=failure_category.value,
                           delay_seconds=delay_seconds,
                           scheduled_time=scheduled_time.isoformat(),
                           platform_preference=platform_preference)

        except Exception as e:
            self.logger.error("retry_scheduling_failed", job_id=job.id, error=str(e))

    def _calculate_retry_delay(self, job: ExperimentJob, config: RetryConfig) -> int:
        """Calculate delay before retry attempt.

        Args:
            job: Job to retry
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        base_delay = config.base_delay_seconds

        if config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (job.retry_count + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (config.backoff_multiplier ** job.retry_count)
        elif config.strategy == RetryStrategy.PLATFORM_AWARE:
            # Longer delays for platform-specific issues
            delay = base_delay * (config.backoff_multiplier ** job.retry_count)
            # Add extra delay for platform rotation
            if config.platform_rotation_on_failure:
                delay += 300  # Additional 5 minutes for platform switch
        elif config.strategy == RetryStrategy.INTELLIGENT:
            # Use failure history to adjust delay
            delay = self._calculate_intelligent_delay(job, config)
        else:
            delay = base_delay

        # Apply maximum delay limit
        delay = min(delay, config.max_delay_seconds)

        # Add jitter if enabled
        if config.jitter_enabled:
            jitter = random.uniform(0.8, 1.2)
            delay = int(delay * jitter)

        return max(delay, 0)

    def _calculate_intelligent_delay(self, job: ExperimentJob, config: RetryConfig) -> int:
        """Calculate intelligent delay based on failure history.

        Args:
            job: Job to retry
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        base_delay = config.base_delay_seconds

        # Get retry history for this job
        history = self.retry_history.get(job.id, [])

        if not history:
            return base_delay

        # Analyze failure patterns
        recent_failures = len([h for h in history if
                             datetime.now() - h.scheduled_time < timedelta(hours=1)])

        # Increase delay for rapid consecutive failures
        if recent_failures > 1:
            delay_multiplier = math.pow(1.5, recent_failures - 1)
            return min(int(base_delay * delay_multiplier), config.max_delay_seconds)

        return base_delay

    def _determine_retry_platform(self, job: ExperimentJob,
                                category: FailureCategory,
                                config: RetryConfig) -> str | None:
        """Determine platform preference for retry.

        Args:
            job: Job to retry
            category: Failure category
            config: Retry configuration

        Returns:
            Platform preference or None
        """
        if not config.platform_rotation_on_failure:
            return job.config.platform_preference

        # For platform-specific failures, try different platform
        if category in [FailureCategory.PLATFORM_ERROR, FailureCategory.TIMEOUT_ERROR,
                       FailureCategory.RESOURCE_ERROR]:

            # Get available platforms (exclude current if it failed)
            available_platforms = ['kaggle', 'colab', 'paperspace']
            if job.platform in available_platforms:
                available_platforms.remove(job.platform)

            if available_platforms:
                # Simple round-robin for now
                platform_index = job.retry_count % len(available_platforms)
                return available_platforms[platform_index]

        return job.config.platform_preference

    async def _execute_retry(self, job: ExperimentJob, retry_attempt: RetryAttempt, delay_seconds: int):
        """Execute retry after delay.

        Args:
            job: Job to retry
            retry_attempt: Retry attempt information
            delay_seconds: Delay before retry
        """
        try:
            # Wait for retry delay
            await asyncio.sleep(delay_seconds)

            # Check if job is still eligible for retry
            current_job = self.experiment_queue.get_job(job.id)
            if not current_job or current_job.status != ExperimentStatus.FAILED:
                self.logger.info("job_retry_cancelled_status_changed", job_id=job.id)
                return

            # Update job for retry
            current_job.retry_count += 1
            current_job.status = ExperimentStatus.QUEUED
            current_job.started_at = None
            current_job.completed_at = None
            current_job.last_error = None
            current_job.metadata.updated_at = datetime.now()

            # Update platform preference if needed
            if retry_attempt.platform_preference:
                current_job.config.platform_preference = retry_attempt.platform_preference
                current_job.platform = None  # Clear previous platform

            # Add retry metadata
            if not current_job.progress:
                current_job.progress = {}
            current_job.progress['retry_info'] = {
                'attempt': retry_attempt.attempt_number,
                'failure_category': retry_attempt.failure_category.value,
                'retry_strategy': retry_attempt.retry_config.strategy.value,
                'original_platform': retry_attempt.metadata.get('original_platform'),
                'new_platform_preference': retry_attempt.platform_preference
            }

            # Re-queue the job
            success = self.experiment_queue.update_job_status(
                job.id,
                ExperimentStatus.QUEUED,
                {
                    'retry_attempt': retry_attempt.attempt_number,
                    'retry_category': retry_attempt.failure_category.value,
                    'platform': retry_attempt.platform_preference,
                    'progress': current_job.progress
                }
            )

            if success:
                self.logger.info("job_retry_executed",
                               job_id=job.id,
                               attempt=retry_attempt.attempt_number,
                               category=retry_attempt.failure_category.value,
                               platform_preference=retry_attempt.platform_preference)
            else:
                self.logger.error("job_retry_failed", job_id=job.id)

            # Clean up retry state
            if job.id in self.retry_schedules:
                del self.retry_schedules[job.id]
            if job.id in self.retry_tasks:
                del self.retry_tasks[job.id]

        except asyncio.CancelledError:
            self.logger.info("retry_task_cancelled", job_id=job.id)
        except Exception as e:
            self.logger.error("retry_execution_failed", job_id=job.id, error=str(e))

    def get_retry_status(self, job_id: str) -> dict[str, Any] | None:
        """Get retry status for a job.

        Args:
            job_id: Job ID

        Returns:
            Retry status information or None
        """
        if job_id not in self.retry_schedules:
            return None

        retry_attempt = self.retry_schedules[job_id]
        history = self.retry_history.get(job_id, [])

        return {
            'next_attempt': retry_attempt.attempt_number,
            'scheduled_time': retry_attempt.scheduled_time.isoformat(),
            'failure_category': retry_attempt.failure_category.value,
            'strategy': retry_attempt.retry_config.strategy.value,
            'platform_preference': retry_attempt.platform_preference,
            'total_attempts': len(history),
            'retry_active': job_id in self.retry_tasks
        }

    def get_retry_statistics(self) -> dict[str, Any]:
        """Get retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        total_retries = sum(len(history) for history in self.retry_history.values())
        active_retries = len(self.retry_schedules)

        # Category breakdown
        category_counts = {}
        for history in self.retry_history.values():
            for attempt in history:
                category = attempt.failure_category.value
                category_counts[category] = category_counts.get(category, 0) + 1

        # Strategy usage
        strategy_counts = {}
        for history in self.retry_history.values():
            for attempt in history:
                strategy = attempt.retry_config.strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            'total_retries': total_retries,
            'active_retries': active_retries,
            'jobs_with_retries': len(self.retry_history),
            'category_breakdown': category_counts,
            'strategy_usage': strategy_counts,
            'average_retries_per_job': total_retries / max(len(self.retry_history), 1)
        }

    def cancel_retry(self, job_id: str) -> bool:
        """Cancel scheduled retry for a job.

        Args:
            job_id: Job ID to cancel retry for

        Returns:
            True if cancelled successfully
        """
        if job_id not in self.retry_schedules:
            return False

        try:
            # Cancel retry task
            if job_id in self.retry_tasks:
                self.retry_tasks[job_id].cancel()
                del self.retry_tasks[job_id]

            # Remove from schedule
            del self.retry_schedules[job_id]

            self.logger.info("retry_cancelled", job_id=job_id)
            return True

        except Exception as e:
            self.logger.error("retry_cancellation_failed", job_id=job_id, error=str(e))
            return False


# Singleton instance
_retry_manager = None


def get_retry_manager() -> ExperimentRetryManager:
    """Get singleton retry manager instance."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = ExperimentRetryManager()
    return _retry_manager
