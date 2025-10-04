"""Graceful shutdown management for clean platform transitions."""

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .platform_detector import Platform, get_platform_detector


class ShutdownReason(Enum):
    """Reasons for shutdown."""
    USER_INTERRUPT = "user_interrupt"
    PLATFORM_ROTATION = "platform_rotation"
    QUOTA_EXHAUSTED = "quota_exhausted"
    SESSION_TIMEOUT = "session_timeout"
    SYSTEM_ERROR = "system_error"
    SCHEDULED_SHUTDOWN = "scheduled_shutdown"


@dataclass
class ShutdownEvent:
    """Shutdown event information."""
    reason: ShutdownReason
    timestamp: datetime
    platform: Platform
    metadata: dict[str, Any] = None
    graceful: bool = True


class GracefulShutdownManager:
    """Manages graceful shutdown hooks for clean platform transitions."""

    def __init__(self):
        self.logger = self._setup_logging()
        self._shutdown_hooks: list[Callable] = []
        self._async_shutdown_hooks: list[Callable] = []
        self._shutdown_in_progress = False
        self._shutdown_event: ShutdownEvent | None = None
        self._lock = threading.Lock()
        self._detector = get_platform_detector()
        self._register_signal_handlers()
        self._register_atexit_handler()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for shutdown manager."""
        logger = logging.getLogger('graceful_shutdown')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Handle SIGTERM (termination signal)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # On Windows, also handle SIGBREAK
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _register_atexit_handler(self):
        """Register atexit handler for cleanup."""
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        signal_names = {
            signal.SIGINT: "SIGINT",
            signal.SIGTERM: "SIGTERM"
        }
        if sys.platform == "win32":
            signal_names[signal.SIGBREAK] = "SIGBREAK"

        signal_name = signal_names.get(signum, f"Signal {signum}")
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")

        self.initiate_shutdown(
            reason=ShutdownReason.USER_INTERRUPT,
            metadata={'signal': signal_name, 'signal_number': signum}
        )

    def _atexit_handler(self):
        """Handle atexit cleanup."""
        if not self._shutdown_in_progress:
            self.logger.info("Process exiting, running cleanup hooks...")
            self.initiate_shutdown(
                reason=ShutdownReason.SYSTEM_ERROR,
                metadata={'source': 'atexit'},
                force_sync=True
            )

    def register_shutdown_hook(self, hook: Callable, async_hook: bool = False):
        """
        Register a shutdown hook function.

        Args:
            hook: Function to call during shutdown
            async_hook: True if hook is async, False for sync
        """
        with self._lock:
            if async_hook:
                self._async_shutdown_hooks.append(hook)
            else:
                self._shutdown_hooks.append(hook)

        self.logger.debug(f"Registered {'async' if async_hook else 'sync'} shutdown hook: {hook.__name__}")

    def unregister_shutdown_hook(self, hook: Callable, async_hook: bool = False):
        """
        Unregister a shutdown hook function.

        Args:
            hook: Function to remove from shutdown hooks
            async_hook: True if hook is async, False for sync
        """
        with self._lock:
            if async_hook and hook in self._async_shutdown_hooks:
                self._async_shutdown_hooks.remove(hook)
            elif not async_hook and hook in self._shutdown_hooks:
                self._shutdown_hooks.remove(hook)

    def initiate_shutdown(self, reason: ShutdownReason,
                         metadata: dict[str, Any] | None = None,
                         force_sync: bool = False):
        """
        Initiate graceful shutdown.

        Args:
            reason: Reason for shutdown
            metadata: Additional shutdown metadata
            force_sync: Force synchronous execution (for atexit)
        """
        with self._lock:
            if self._shutdown_in_progress:
                self.logger.warning("Shutdown already in progress, ignoring duplicate request")
                return

            self._shutdown_in_progress = True
            self._shutdown_event = ShutdownEvent(
                reason=reason,
                timestamp=datetime.now(),
                platform=self._detector.detect_platform().platform,
                metadata=metadata or {},
                graceful=True
            )

        self.logger.info(f"Initiating graceful shutdown: {reason.value}")

        try:
            if force_sync:
                self._execute_sync_shutdown()
            else:
                # Try to run async shutdown, fall back to sync if needed
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._execute_async_shutdown())
                except RuntimeError:
                    # No running event loop, execute synchronously
                    asyncio.run(self._execute_async_shutdown())
        except Exception as e:
            self.logger.error(f"Error during shutdown execution: {e}")
            self._shutdown_event.graceful = False

    async def _execute_async_shutdown(self):
        """Execute async shutdown hooks."""
        self.logger.info("Executing async shutdown hooks...")

        # Execute async hooks
        for hook in self._async_shutdown_hooks:
            try:
                self.logger.debug(f"Executing async hook: {hook.__name__}")
                if asyncio.iscoroutinefunction(hook):
                    await hook(self._shutdown_event)
                else:
                    hook(self._shutdown_event)
            except Exception as e:
                self.logger.error(f"Error in async shutdown hook {hook.__name__}: {e}")
                self._shutdown_event.graceful = False

        # Execute sync hooks
        self._execute_sync_hooks()

        self.logger.info("Async shutdown complete")

    def _execute_sync_shutdown(self):
        """Execute synchronous shutdown hooks."""
        self.logger.info("Executing sync shutdown hooks...")
        self._execute_sync_hooks()
        self.logger.info("Sync shutdown complete")

    def _execute_sync_hooks(self):
        """Execute synchronous shutdown hooks."""
        for hook in self._shutdown_hooks:
            try:
                self.logger.debug(f"Executing sync hook: {hook.__name__}")
                hook(self._shutdown_event)
            except Exception as e:
                self.logger.error(f"Error in sync shutdown hook {hook.__name__}: {e}")
                self._shutdown_event.graceful = False

    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is currently in progress."""
        return self._shutdown_in_progress

    def get_shutdown_event(self) -> ShutdownEvent | None:
        """Get current shutdown event information."""
        return self._shutdown_event


class ExperimentShutdownManager:
    """Specialized shutdown manager for experiment cleanup."""

    def __init__(self, shutdown_manager: GracefulShutdownManager | None = None):
        self.shutdown_manager = shutdown_manager or get_shutdown_manager()
        self.logger = logging.getLogger('experiment_shutdown')
        self._experiment_state: dict[str, Any] = {}
        self._checkpoint_callbacks: list[Callable] = []
        self._cleanup_callbacks: list[Callable] = []

        # Register with main shutdown manager
        self.shutdown_manager.register_shutdown_hook(
            self._experiment_shutdown_hook, async_hook=True
        )

    def set_experiment_state(self, state: dict[str, Any]):
        """Set current experiment state for shutdown handling."""
        self._experiment_state = state.copy()

    def register_checkpoint_callback(self, callback: Callable):
        """Register callback for saving checkpoints during shutdown."""
        self._checkpoint_callbacks.append(callback)

    def register_cleanup_callback(self, callback: Callable):
        """Register callback for cleanup during shutdown."""
        self._cleanup_callbacks.append(callback)

    async def _experiment_shutdown_hook(self, shutdown_event: ShutdownEvent):
        """Handle experiment-specific shutdown tasks."""
        self.logger.info("Starting experiment shutdown procedures...")

        try:
            # Save checkpoints
            await self._save_checkpoints(shutdown_event)

            # Clean up resources
            await self._cleanup_resources(shutdown_event)

            # Log experiment completion
            self._log_experiment_shutdown(shutdown_event)

        except Exception as e:
            self.logger.error(f"Error during experiment shutdown: {e}")
            raise

    async def _save_checkpoints(self, shutdown_event: ShutdownEvent):
        """Save experiment checkpoints during shutdown."""
        self.logger.info("Saving experiment checkpoints...")

        for callback in self._checkpoint_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._experiment_state, shutdown_event)
                else:
                    callback(self._experiment_state, shutdown_event)
                self.logger.debug(f"Checkpoint callback completed: {callback.__name__}")
            except Exception as e:
                self.logger.error(f"Error in checkpoint callback {callback.__name__}: {e}")

    async def _cleanup_resources(self, shutdown_event: ShutdownEvent):
        """Clean up experiment resources during shutdown."""
        self.logger.info("Cleaning up experiment resources...")

        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._experiment_state, shutdown_event)
                else:
                    callback(self._experiment_state, shutdown_event)
                self.logger.debug(f"Cleanup callback completed: {callback.__name__}")
            except Exception as e:
                self.logger.error(f"Error in cleanup callback {callback.__name__}: {e}")

    def _log_experiment_shutdown(self, shutdown_event: ShutdownEvent):
        """Log experiment shutdown information."""
        shutdown_info = {
            'reason': shutdown_event.reason.value,
            'timestamp': shutdown_event.timestamp.isoformat(),
            'platform': shutdown_event.platform.value,
            'graceful': shutdown_event.graceful,
            'experiment_state': self._experiment_state,
            'metadata': shutdown_event.metadata
        }

        self.logger.info(f"Experiment shutdown completed: {shutdown_info}")


# Timeout detection for different platforms
class PlatformTimeoutDetector:
    """Detects approaching platform timeouts and triggers graceful shutdown."""

    def __init__(self, shutdown_manager: GracefulShutdownManager | None = None):
        self.shutdown_manager = shutdown_manager or get_shutdown_manager()
        self.logger = logging.getLogger('timeout_detector')
        self._detector = get_platform_detector()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._session_start_time = datetime.now()
        self._last_activity_time = datetime.now()
        self._state_file = Path.home() / '.arc_session_state.json'
        self._experiment_state: dict[str, Any] = {}
        self._recovery_callbacks: list[Callable] = []

        # Load previous session state if exists
        self._load_session_state()

    async def start_monitoring(self, warning_minutes: int = 30):
        """
        Start monitoring for platform timeouts.

        Args:
            warning_minutes: Minutes before timeout to trigger warning
        """
        if self._monitoring:
            return

        self._monitoring = True
        self.logger.info(f"Starting timeout monitoring (warning: {warning_minutes} min)")

        # Save initial session state
        self._save_session_state()

        self._monitor_task = asyncio.create_task(
            self._monitor_timeout_loop(warning_minutes)
        )

    def stop_monitoring(self):
        """Stop timeout monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    def set_experiment_state(self, state: dict[str, Any]):
        """Set current experiment state for recovery purposes."""
        self._experiment_state = state.copy()
        self._save_session_state()

    def register_recovery_callback(self, callback: Callable):
        """Register callback for session recovery after restoration."""
        self._recovery_callbacks.append(callback)

    def update_activity(self):
        """Update last activity timestamp for idle detection."""
        self._last_activity_time = datetime.now()

    async def check_and_recover_session(self) -> bool:
        """
        Check if this is a session recovery and execute recovery logic.

        Returns:
            True if recovery was performed, False otherwise
        """
        if not self._state_file.exists():
            return False

        try:
            with open(self._state_file) as f:
                state = json.load(f)

            # Check if this is a recovery scenario
            last_session_time = datetime.fromisoformat(state.get('last_save_time', ''))
            time_since_last = (datetime.now() - last_session_time).total_seconds() / 60

            # If more than 15 minutes since last save, likely a recovery
            if time_since_last > 15:
                self.logger.info(f"Detected session recovery after {time_since_last:.1f} minutes")
                await self._execute_recovery(state)
                return True

        except Exception as e:
            self.logger.error(f"Error checking session recovery: {e}")

        return False

    async def _monitor_timeout_loop(self, warning_minutes: int):
        """Monitor loop for platform timeouts."""
        while self._monitoring:
            try:
                platform_info = self._detector.detect_platform()

                # Update activity tracking
                self.update_activity()

                # Check platform-specific timeout logic
                if await self._check_timeout_approaching(platform_info.platform, warning_minutes):
                    self.logger.warning("Platform timeout approaching, initiating graceful shutdown")

                    # Save state before shutdown
                    await self._save_experiment_state()

                    self.shutdown_manager.initiate_shutdown(
                        reason=ShutdownReason.SESSION_TIMEOUT,
                        metadata={
                            'warning_minutes': warning_minutes,
                            'platform': platform_info.platform.value,
                            'session_duration_hours': self._get_session_duration_hours()
                        }
                    )
                    break

                # Save session state periodically
                self._save_session_state()

                # Check every minute
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in timeout monitoring: {e}")
                await asyncio.sleep(60)

    async def _check_timeout_approaching(self, platform: Platform, warning_minutes: int) -> bool:
        """Check if platform timeout is approaching."""
        try:
            if platform == Platform.KAGGLE:
                return await self._check_kaggle_timeout(warning_minutes)
            elif platform == Platform.COLAB:
                return await self._check_colab_timeout(warning_minutes)
            elif platform == Platform.PAPERSPACE:
                return await self._check_paperspace_timeout(warning_minutes)
            else:
                # For local or unknown platforms, no timeout
                return False

        except Exception as e:
            self.logger.error(f"Error checking timeout for {platform.value}: {e}")
            # Fall back to time-based estimation
            return self._check_time_based_timeout(platform, warning_minutes)

    async def _check_kaggle_timeout(self, warning_minutes: int) -> bool:
        """Check Kaggle session timeout (12 hours)."""
        try:
            # Try to get session info from Kaggle API if available
            if 'KAGGLE_USER_SECRETS_TOKEN' in os.environ:
                # API approach (if implemented by Kaggle)
                session_info = await self._get_kaggle_session_info()
                if session_info and 'remaining_minutes' in session_info:
                    return session_info['remaining_minutes'] <= warning_minutes

            # Fall back to time-based estimation
            session_hours = self._get_session_duration_hours()
            kaggle_limit_hours = 12
            remaining_hours = kaggle_limit_hours - session_hours
            remaining_minutes = remaining_hours * 60

            self.logger.debug(f"Kaggle session: {session_hours:.1f}h elapsed, {remaining_minutes:.1f}m remaining")
            return remaining_minutes <= warning_minutes

        except Exception as e:
            self.logger.error(f"Error checking Kaggle timeout: {e}")
            return self._check_time_based_timeout(Platform.KAGGLE, warning_minutes)

    async def _check_colab_timeout(self, warning_minutes: int) -> bool:
        """Check Colab session timeout (12 hours with idle detection)."""
        try:
            # Check for Colab-specific indicators
            idle_hours = self._get_idle_duration_hours()
            session_hours = self._get_session_duration_hours()

            # Colab typically disconnects after 12 hours or extended idle time
            colab_limit_hours = 12
            idle_limit_hours = 1.5  # Disconnect after 90 minutes of idle

            # Check session time limit
            remaining_session_minutes = (colab_limit_hours - session_hours) * 60

            # Check idle time limit
            remaining_idle_minutes = (idle_limit_hours - idle_hours) * 60

            # Use the more restrictive limit
            effective_remaining = min(remaining_session_minutes, remaining_idle_minutes)

            self.logger.debug(f"Colab: {session_hours:.1f}h session, {idle_hours:.1f}h idle, {effective_remaining:.1f}m remaining")
            return effective_remaining <= warning_minutes

        except Exception as e:
            self.logger.error(f"Error checking Colab timeout: {e}")
            return self._check_time_based_timeout(Platform.COLAB, warning_minutes)

    async def _check_paperspace_timeout(self, warning_minutes: int) -> bool:
        """Check Paperspace session timeout (6 hours for free tier)."""
        try:
            # Try to get session info from Paperspace API if available
            if 'PAPERSPACE_API_KEY' in os.environ:
                session_info = await self._get_paperspace_session_info()
                if session_info and 'remaining_minutes' in session_info:
                    return session_info['remaining_minutes'] <= warning_minutes

            # Fall back to time-based estimation
            session_hours = self._get_session_duration_hours()
            paperspace_limit_hours = 6  # Free tier limit
            remaining_hours = paperspace_limit_hours - session_hours
            remaining_minutes = remaining_hours * 60

            self.logger.debug(f"Paperspace session: {session_hours:.1f}h elapsed, {remaining_minutes:.1f}m remaining")
            return remaining_minutes <= warning_minutes

        except Exception as e:
            self.logger.error(f"Error checking Paperspace timeout: {e}")
            return self._check_time_based_timeout(Platform.PAPERSPACE, warning_minutes)

    def _check_time_based_timeout(self, platform: Platform, warning_minutes: int) -> bool:
        """Fall back to time-based timeout estimation."""
        session_hours = self._get_session_duration_hours()

        # Platform time limits
        time_limits = {
            Platform.KAGGLE: 12,
            Platform.COLAB: 12,
            Platform.PAPERSPACE: 6,
        }

        limit_hours = time_limits.get(platform, 24)  # Default 24h for unknown
        remaining_hours = limit_hours - session_hours
        remaining_minutes = remaining_hours * 60

        return remaining_minutes <= warning_minutes

    async def _get_kaggle_session_info(self) -> dict[str, Any] | None:
        """Get Kaggle session information via API (if available)."""
        try:
            # This would use Kaggle's API if they provide session info
            # Currently this is a placeholder for future implementation
            self.logger.debug("Kaggle session API not implemented yet")
            return None
        except Exception as e:
            self.logger.debug(f"Could not get Kaggle session info: {e}")
            return None

    async def _get_paperspace_session_info(self) -> dict[str, Any] | None:
        """Get Paperspace session information via API."""
        try:
            api_key = os.environ.get('PAPERSPACE_API_KEY')
            if not api_key:
                return None

            # Placeholder for Paperspace API call
            # The actual implementation would depend on Paperspace's API
            self.logger.debug("Paperspace session API not implemented yet")
            return None

        except Exception as e:
            self.logger.debug(f"Could not get Paperspace session info: {e}")
            return None

    def _get_session_duration_hours(self) -> float:
        """Get current session duration in hours."""
        duration = datetime.now() - self._session_start_time
        return duration.total_seconds() / 3600

    def _get_idle_duration_hours(self) -> float:
        """Get current idle duration in hours."""
        idle_duration = datetime.now() - self._last_activity_time
        return idle_duration.total_seconds() / 3600

    def _load_session_state(self):
        """Load session state from file."""
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    state = json.load(f)

                # Load session start time
                if 'session_start_time' in state:
                    self._session_start_time = datetime.fromisoformat(state['session_start_time'])

                # Load experiment state
                if 'experiment_state' in state:
                    self._experiment_state = state['experiment_state']

                self.logger.debug("Loaded session state from file")

        except Exception as e:
            self.logger.error(f"Error loading session state: {e}")

    def _save_session_state(self):
        """Save session state to file."""
        try:
            state = {
                'session_start_time': self._session_start_time.isoformat(),
                'last_activity_time': self._last_activity_time.isoformat(),
                'last_save_time': datetime.now().isoformat(),
                'experiment_state': self._experiment_state,
                'platform': self._detector.detect_platform().platform.value
            }

            # Ensure directory exists
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving session state: {e}")

    async def _save_experiment_state(self):
        """Save experiment state before suspension."""
        self.logger.info("Saving experiment state before session timeout...")

        try:
            # Update experiment state in session file
            self._save_session_state()

            # Additional checkpoint creation could be added here
            # This would integrate with the experiment's checkpoint system

            self.logger.info("Experiment state saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving experiment state: {e}")

    async def _execute_recovery(self, state: dict[str, Any]):
        """Execute recovery logic after session restoration."""
        self.logger.info("Executing session recovery procedures...")

        try:
            # Restore experiment state
            self._experiment_state = state.get('experiment_state', {})

            # Execute recovery callbacks
            for callback in self._recovery_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self._experiment_state)
                    else:
                        callback(self._experiment_state)
                    self.logger.debug(f"Recovery callback completed: {callback.__name__}")
                except Exception as e:
                    self.logger.error(f"Error in recovery callback {callback.__name__}: {e}")

            self.logger.info("Session recovery completed successfully")

        except Exception as e:
            self.logger.error(f"Error during session recovery: {e}")

    def cleanup_session_state(self):
        """Clean up session state file."""
        try:
            if self._state_file.exists():
                self._state_file.unlink()
                self.logger.debug("Session state file cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up session state: {e}")


# Singleton instances
_shutdown_manager_instance = None
_timeout_detector_instance = None


def get_shutdown_manager() -> GracefulShutdownManager:
    """Get singleton shutdown manager instance."""
    global _shutdown_manager_instance
    if _shutdown_manager_instance is None:
        _shutdown_manager_instance = GracefulShutdownManager()
    return _shutdown_manager_instance


def get_timeout_detector() -> PlatformTimeoutDetector:
    """Get singleton timeout detector instance."""
    global _timeout_detector_instance
    if _timeout_detector_instance is None:
        _timeout_detector_instance = PlatformTimeoutDetector()
    return _timeout_detector_instance


# Convenience functions for common shutdown hooks
def register_checkpoint_save_hook(save_function: Callable):
    """Register a checkpoint saving function as shutdown hook."""
    manager = get_shutdown_manager()
    manager.register_shutdown_hook(save_function, async_hook=asyncio.iscoroutinefunction(save_function))


def register_resource_cleanup_hook(cleanup_function: Callable):
    """Register a resource cleanup function as shutdown hook."""
    manager = get_shutdown_manager()
    manager.register_shutdown_hook(cleanup_function, async_hook=asyncio.iscoroutinefunction(cleanup_function))


def initiate_platform_rotation_shutdown(metadata: dict[str, Any] | None = None):
    """Initiate shutdown for platform rotation."""
    manager = get_shutdown_manager()
    manager.initiate_shutdown(
        reason=ShutdownReason.PLATFORM_ROTATION,
        metadata=metadata
    )


async def start_timeout_monitoring(warning_minutes: int = 30):
    """Start platform timeout monitoring with specified warning time."""
    detector = get_timeout_detector()
    await detector.start_monitoring(warning_minutes)


def stop_timeout_monitoring():
    """Stop platform timeout monitoring."""
    detector = get_timeout_detector()
    detector.stop_monitoring()


async def check_session_recovery() -> bool:
    """Check if this is a session recovery and execute recovery logic."""
    detector = get_timeout_detector()
    return await detector.check_and_recover_session()


def register_session_recovery_callback(callback: Callable):
    """Register callback for session recovery after restoration."""
    detector = get_timeout_detector()
    detector.register_recovery_callback(callback)


def update_session_activity():
    """Update session activity timestamp for idle detection."""
    detector = get_timeout_detector()
    detector.update_activity()


def set_experiment_state_for_recovery(state: dict[str, Any]):
    """Set experiment state for session recovery purposes."""
    detector = get_timeout_detector()
    detector.set_experiment_state(state)
