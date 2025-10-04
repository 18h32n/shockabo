"""Session timeout management for platform rotation automation.

This module provides comprehensive session timeout detection, monitoring,
and handling for different platforms (Kaggle, Colab, Paperspace).
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .graceful_shutdown import ShutdownReason, get_shutdown_manager
from .platform_detector import Platform, get_platform_detector

logger = structlog.get_logger(__name__)


class SessionState(Enum):
    """Session state enumeration."""
    ACTIVE = "active"
    WARNING = "warning"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class SessionInfo:
    """Session information for a platform."""
    platform: Platform
    session_id: str | None
    start_time: datetime
    last_activity: datetime
    estimated_end_time: datetime | None
    warning_time: datetime | None
    max_duration_hours: float
    idle_timeout_hours: float
    state: SessionState
    metadata: dict[str, Any]

    def get_remaining_minutes(self) -> float:
        """Get remaining session time in minutes."""
        if not self.estimated_end_time:
            return float('inf')

        remaining = self.estimated_end_time - datetime.now()
        return max(0, remaining.total_seconds() / 60)

    def is_near_timeout(self, warning_minutes: int = 30) -> bool:
        """Check if session is approaching timeout."""
        if not self.estimated_end_time:
            return False

        warning_time = datetime.now() + timedelta(minutes=warning_minutes)
        return warning_time >= self.estimated_end_time

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class PlatformSessionDetector:
    """Detects session information for different platforms."""

    def __init__(self):
        self.logger = structlog.get_logger('session_detector')

        # Platform-specific session limits
        self.platform_limits = {
            Platform.KAGGLE: {
                'max_duration_hours': 12,
                'idle_timeout_hours': 1,
                'warning_minutes': 30,
                'check_methods': ['api', 'runtime']
            },
            Platform.COLAB: {
                'max_duration_hours': 12,
                'idle_timeout_hours': 1.5,
                'warning_minutes': 30,
                'check_methods': ['runtime', 'api']
            },
            Platform.PAPERSPACE: {
                'max_duration_hours': 6,
                'idle_timeout_hours': 0.5,
                'warning_minutes': 20,
                'check_methods': ['api']
            }
        }

    async def detect_session_info(self, platform: Platform) -> SessionInfo | None:
        """Detect session information for platform.

        Args:
            platform: Platform to check

        Returns:
            Session information or None if not detectable
        """
        try:
            if platform == Platform.KAGGLE:
                return await self._detect_kaggle_session()
            elif platform == Platform.COLAB:
                return await self._detect_colab_session()
            elif platform == Platform.PAPERSPACE:
                return await self._detect_paperspace_session()
            elif platform == Platform.LOCAL:
                return self._detect_local_session()

            return None

        except Exception as e:
            self.logger.error("session_detection_failed", platform=platform.value, error=str(e))
            return None

    async def _detect_kaggle_session(self) -> SessionInfo | None:
        """Detect Kaggle session information."""
        try:
            # Try to get session info from Kaggle API
            session_data = await self._check_kaggle_api()

            if session_data:
                limits = self.platform_limits[Platform.KAGGLE]
                start_time = datetime.now() - timedelta(hours=session_data.get('runtime_hours', 0))

                return SessionInfo(
                    platform=Platform.KAGGLE,
                    session_id=session_data.get('session_id'),
                    start_time=start_time,
                    last_activity=datetime.now(),
                    estimated_end_time=start_time + timedelta(hours=limits['max_duration_hours']),
                    warning_time=start_time + timedelta(hours=limits['max_duration_hours'] - limits['warning_minutes']/60),
                    max_duration_hours=limits['max_duration_hours'],
                    idle_timeout_hours=limits['idle_timeout_hours'],
                    state=SessionState.ACTIVE,
                    metadata=session_data
                )

            # Fallback: detect from runtime environment
            return self._detect_kaggle_runtime_session()

        except Exception as e:
            self.logger.error("kaggle_session_detection_failed", error=str(e))
            return None

    async def _detect_colab_session(self) -> SessionInfo | None:
        """Detect Google Colab session information."""
        try:
            # Try to get session info from Colab runtime
            session_data = await self._check_colab_runtime()

            if session_data:
                limits = self.platform_limits[Platform.COLAB]
                start_time = datetime.now() - timedelta(hours=session_data.get('runtime_hours', 0))

                return SessionInfo(
                    platform=Platform.COLAB,
                    session_id=session_data.get('session_id'),
                    start_time=start_time,
                    last_activity=datetime.now(),
                    estimated_end_time=start_time + timedelta(hours=limits['max_duration_hours']),
                    warning_time=start_time + timedelta(hours=limits['max_duration_hours'] - limits['warning_minutes']/60),
                    max_duration_hours=limits['max_duration_hours'],
                    idle_timeout_hours=limits['idle_timeout_hours'],
                    state=SessionState.ACTIVE,
                    metadata=session_data
                )

            return None

        except Exception as e:
            self.logger.error("colab_session_detection_failed", error=str(e))
            return None

    async def _detect_paperspace_session(self) -> SessionInfo | None:
        """Detect Paperspace session information."""
        try:
            # Try to get session info from Paperspace API
            session_data = await self._check_paperspace_api()

            if session_data:
                limits = self.platform_limits[Platform.PAPERSPACE]
                start_time = datetime.fromisoformat(session_data.get('created_at', datetime.now().isoformat()))

                return SessionInfo(
                    platform=Platform.PAPERSPACE,
                    session_id=session_data.get('machine_id'),
                    start_time=start_time,
                    last_activity=datetime.now(),
                    estimated_end_time=start_time + timedelta(hours=limits['max_duration_hours']),
                    warning_time=start_time + timedelta(hours=limits['max_duration_hours'] - limits['warning_minutes']/60),
                    max_duration_hours=limits['max_duration_hours'],
                    idle_timeout_hours=limits['idle_timeout_hours'],
                    state=SessionState.ACTIVE,
                    metadata=session_data
                )

            return None

        except Exception as e:
            self.logger.error("paperspace_session_detection_failed", error=str(e))
            return None

    def _detect_local_session(self) -> SessionInfo:
        """Create session info for local environment."""
        limits = {
            'max_duration_hours': 24,  # No real limit for local
            'idle_timeout_hours': 24,
            'warning_minutes': 60
        }

        return SessionInfo(
            platform=Platform.LOCAL,
            session_id="local",
            start_time=datetime.now(),
            last_activity=datetime.now(),
            estimated_end_time=None,  # No timeout for local
            warning_time=None,
            max_duration_hours=limits['max_duration_hours'],
            idle_timeout_hours=limits['idle_timeout_hours'],
            state=SessionState.ACTIVE,
            metadata={'type': 'local_environment'}
        )

    def _detect_kaggle_runtime_session(self) -> SessionInfo | None:
        """Detect Kaggle session from runtime environment."""
        try:
            import os

            # Check for Kaggle environment variables
            if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                limits = self.platform_limits[Platform.KAGGLE]

                # Estimate start time (rough approximation)
                start_time = datetime.now() - timedelta(minutes=5)  # Assume started recently

                return SessionInfo(
                    platform=Platform.KAGGLE,
                    session_id=os.environ.get('KAGGLE_KERNEL_ID', 'unknown'),
                    start_time=start_time,
                    last_activity=datetime.now(),
                    estimated_end_time=start_time + timedelta(hours=limits['max_duration_hours']),
                    warning_time=start_time + timedelta(hours=limits['max_duration_hours'] - limits['warning_minutes']/60),
                    max_duration_hours=limits['max_duration_hours'],
                    idle_timeout_hours=limits['idle_timeout_hours'],
                    state=SessionState.ACTIVE,
                    metadata={'detection_method': 'runtime_environment'}
                )

            return None

        except Exception as e:
            self.logger.error("kaggle_runtime_detection_failed", error=str(e))
            return None

    async def _check_kaggle_api(self) -> dict[str, Any] | None:
        """Check Kaggle session via API."""
        try:

            # This is a placeholder - actual implementation would depend on
            # available Kaggle API endpoints for session information
            # For now, return None to fall back to runtime detection
            return None

        except Exception:
            return None

    async def _check_colab_runtime(self) -> dict[str, Any] | None:
        """Check Colab session via runtime."""
        try:
            # Try to check Colab runtime information
            # This would use Colab-specific APIs if available

            # Check for Colab environment
            try:
                import google.colab  # noqa: F401

                # If we can import google.colab, we're in Colab
                return {
                    'session_id': 'colab_session',
                    'runtime_hours': 0,  # Would need actual runtime tracking
                    'detection_method': 'runtime_import'
                }
            except ImportError:
                return None

        except Exception as e:
            self.logger.error("colab_runtime_check_failed", error=str(e))
            return None

    async def _check_paperspace_api(self) -> dict[str, Any] | None:
        """Check Paperspace session via API."""
        try:
            # This would use Paperspace API to get machine/session information
            # Placeholder implementation
            return None

        except Exception as e:
            self.logger.error("paperspace_api_check_failed", error=str(e))
            return None


class SessionTimeoutManager:
    """Manages session timeouts across platforms."""

    def __init__(self,
                 check_interval_minutes: int = 5,
                 state_dir: Path | None = None):
        """Initialize session timeout manager.

        Args:
            check_interval_minutes: How often to check session status
            state_dir: Directory to persist session state
        """
        self.check_interval_minutes = check_interval_minutes
        self.state_dir = state_dir or Path.home() / ".arc-session-state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.platform_detector = get_platform_detector()
        self.session_detector = PlatformSessionDetector()
        self.shutdown_manager = get_shutdown_manager()

        # State
        self.current_session: SessionInfo | None = None
        self.session_history: list[SessionInfo] = []
        self.monitoring = False
        self.monitor_task: asyncio.Task | None = None

        # Callbacks
        self.callbacks: dict[str, list[Callable]] = {
            'session_warning': [],
            'session_expired': [],
            'session_detected': [],
            'session_lost': []
        }

        self.logger = structlog.get_logger('session_timeout_manager')

        # Load persisted state
        self._load_session_state()

    async def start_monitoring(self) -> bool:
        """Start session timeout monitoring.

        Returns:
            True if started successfully
        """
        if self.monitoring:
            self.logger.warning("session_monitoring_already_running")
            return False

        try:
            self.monitoring = True

            # Detect current session
            await self._detect_current_session()

            # Start monitoring loop
            self.monitor_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("session_timeout_monitoring_started",
                           check_interval=self.check_interval_minutes)
            return True

        except Exception as e:
            self.logger.error("session_monitoring_start_failed", error=str(e))
            self.monitoring = False
            return False

    async def stop_monitoring(self) -> bool:
        """Stop session timeout monitoring.

        Returns:
            True if stopped successfully
        """
        if not self.monitoring:
            return True

        try:
            self.monitoring = False

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # Save session state
            self._save_session_state()

            self.logger.info("session_timeout_monitoring_stopped")
            return True

        except Exception as e:
            self.logger.error("session_monitoring_stop_failed", error=str(e))
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Check current session status
                await self._check_session_status()

                # Save state periodically
                self._save_session_state()

                # Wait for next check
                await asyncio.sleep(self.check_interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("session_monitoring_loop_error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _detect_current_session(self):
        """Detect current platform session."""
        try:
            platform_info = self.platform_detector.detect_platform()
            platform = platform_info.platform

            session_info = await self.session_detector.detect_session_info(platform)

            if session_info:
                self.current_session = session_info
                self.session_history.append(session_info)

                self.logger.info("session_detected",
                               platform=platform.value,
                               session_id=session_info.session_id,
                               remaining_minutes=session_info.get_remaining_minutes())

                # Trigger callbacks
                self._trigger_callbacks('session_detected', session_info)
            else:
                self.logger.warning("session_detection_failed", platform=platform.value)

        except Exception as e:
            self.logger.error("session_detection_error", error=str(e))

    async def _check_session_status(self):
        """Check current session status for timeouts."""
        if not self.current_session:
            await self._detect_current_session()
            return

        try:
            now = datetime.now()
            session = self.current_session

            # Update activity timestamp
            session.update_activity()

            # Check for session expiration
            if session.estimated_end_time and now >= session.estimated_end_time:
                session.state = SessionState.EXPIRED

                self.logger.critical("session_expired",
                                   platform=session.platform.value,
                                   session_id=session.session_id)

                # Trigger callbacks
                self._trigger_callbacks('session_expired', session)

                # Initiate graceful shutdown
                self.shutdown_manager.initiate_shutdown(
                    reason=ShutdownReason.SESSION_TIMEOUT,
                    metadata={'session_id': session.session_id}
                )

            # Check for timeout warning
            elif session.is_near_timeout():
                if session.state != SessionState.WARNING:
                    session.state = SessionState.WARNING

                    remaining = session.get_remaining_minutes()
                    self.logger.warning("session_timeout_warning",
                                      platform=session.platform.value,
                                      remaining_minutes=remaining)

                    # Trigger callbacks
                    self._trigger_callbacks('session_warning', session)

            # Try to refresh session information periodically
            if now - session.last_activity > timedelta(minutes=10):
                refreshed_session = await self.session_detector.detect_session_info(session.platform)
                if refreshed_session:
                    # Update with fresh information
                    session.estimated_end_time = refreshed_session.estimated_end_time
                    session.metadata.update(refreshed_session.metadata)
                else:
                    # Session may be lost
                    session.state = SessionState.UNKNOWN
                    self.logger.warning("session_status_unknown",
                                      platform=session.platform.value)

                    self._trigger_callbacks('session_lost', session)

        except Exception as e:
            self.logger.error("session_status_check_failed", error=str(e))

    def get_current_session_info(self) -> dict[str, Any] | None:
        """Get current session information.

        Returns:
            Dictionary with session information or None
        """
        if not self.current_session:
            return None

        session = self.current_session
        return {
            'platform': session.platform.value,
            'session_id': session.session_id,
            'state': session.state.value,
            'start_time': session.start_time.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'remaining_minutes': session.get_remaining_minutes(),
            'estimated_end_time': session.estimated_end_time.isoformat() if session.estimated_end_time else None,
            'max_duration_hours': session.max_duration_hours,
            'idle_timeout_hours': session.idle_timeout_hours,
            'is_near_timeout': session.is_near_timeout(),
            'metadata': session.metadata
        }

    def get_session_statistics(self) -> dict[str, Any]:
        """Get session statistics.

        Returns:
            Dictionary with session statistics
        """
        stats = {
            'monitoring_active': self.monitoring,
            'current_session': self.get_current_session_info(),
            'total_sessions': len(self.session_history),
            'platform_distribution': {},
            'average_session_duration_hours': 0.0
        }

        if self.session_history:
            # Platform distribution
            for session in self.session_history:
                platform = session.platform.value
                stats['platform_distribution'][platform] = stats['platform_distribution'].get(platform, 0) + 1

            # Average session duration
            total_duration = 0
            completed_sessions = 0

            for session in self.session_history:
                if session.estimated_end_time:
                    duration = (session.estimated_end_time - session.start_time).total_seconds() / 3600
                    total_duration += duration
                    completed_sessions += 1

            if completed_sessions > 0:
                stats['average_session_duration_hours'] = total_duration / completed_sessions

        return stats

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for session events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, session_info: SessionInfo):
        """Trigger callbacks for a session event.

        Args:
            event_type: Event type
            session_info: Session information
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(session_info)
            except Exception as e:
                self.logger.error("callback_failed",
                                event_type=event_type,
                                error=str(e))

    def _save_session_state(self):
        """Save session state to disk."""
        try:
            state_file = self.state_dir / "session_state.json"

            state = {
                'monitoring': self.monitoring,
                'current_session': None,
                'last_saved': datetime.now().isoformat()
            }

            if self.current_session:
                session = self.current_session
                state['current_session'] = {
                    'platform': session.platform.value,
                    'session_id': session.session_id,
                    'start_time': session.start_time.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'estimated_end_time': session.estimated_end_time.isoformat() if session.estimated_end_time else None,
                    'warning_time': session.warning_time.isoformat() if session.warning_time else None,
                    'max_duration_hours': session.max_duration_hours,
                    'idle_timeout_hours': session.idle_timeout_hours,
                    'state': session.state.value,
                    'metadata': session.metadata
                }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error("session_state_save_failed", error=str(e))

    def _load_session_state(self):
        """Load session state from disk."""
        try:
            state_file = self.state_dir / "session_state.json"

            if not state_file.exists():
                return

            with open(state_file) as f:
                state = json.load(f)

            # Load current session if exists and recent
            if state.get('current_session'):
                session_data = state['current_session']

                # Only restore if session is recent (within last hour)
                last_activity = datetime.fromisoformat(session_data['last_activity'])
                if datetime.now() - last_activity < timedelta(hours=1):

                    platform = Platform(session_data['platform'])

                    self.current_session = SessionInfo(
                        platform=platform,
                        session_id=session_data['session_id'],
                        start_time=datetime.fromisoformat(session_data['start_time']),
                        last_activity=last_activity,
                        estimated_end_time=datetime.fromisoformat(session_data['estimated_end_time']) if session_data['estimated_end_time'] else None,
                        warning_time=datetime.fromisoformat(session_data['warning_time']) if session_data['warning_time'] else None,
                        max_duration_hours=session_data['max_duration_hours'],
                        idle_timeout_hours=session_data['idle_timeout_hours'],
                        state=SessionState(session_data['state']),
                        metadata=session_data['metadata']
                    )

                    self.logger.info("session_state_restored",
                                   platform=platform.value,
                                   session_id=session_data['session_id'])

        except Exception as e:
            self.logger.error("session_state_load_failed", error=str(e))


# Singleton instance
_session_timeout_manager = None


def get_session_timeout_manager() -> SessionTimeoutManager:
    """Get singleton session timeout manager instance."""
    global _session_timeout_manager
    if _session_timeout_manager is None:
        _session_timeout_manager = SessionTimeoutManager()
    return _session_timeout_manager
