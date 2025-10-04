"""
Platform health monitoring for distributed evolution.

Tracks platform availability via REST API heartbeats and handles
automatic task redistribution on platform failures.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class PlatformStatus(Enum):
    """Platform health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISCONNECTED = "disconnected"


@dataclass
class PlatformMetrics:
    """Platform performance metrics."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    tasks_completed: int = 0
    avg_task_duration: float = 0.0
    error_count: int = 0
    last_heartbeat: datetime | None = None


@dataclass
class PlatformHealth:
    """Platform health information."""
    platform_id: str
    status: PlatformStatus = PlatformStatus.DISCONNECTED
    metrics: PlatformMetrics = field(default_factory=PlatformMetrics)
    consecutive_failures: int = 0
    last_status_change: datetime = field(default_factory=datetime.now)
    recovery_attempts: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if platform is healthy."""
        return self.status == PlatformStatus.HEALTHY


class PlatformHealthMonitor:
    """Monitors platform health and handles failures."""

    DEFAULT_HEARTBEAT_TIMEOUT = 30.0  # seconds
    DEFAULT_HEARTBEAT_INTERVAL = 10.0  # seconds
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_TIMEOUT = 60.0  # seconds

    def __init__(
        self,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        recovery_timeout: float = RECOVERY_TIMEOUT
    ):
        """
        Initialize platform health monitor.

        Args:
            heartbeat_timeout: Seconds before platform considered failed
            heartbeat_interval: Seconds between heartbeat checks
            recovery_timeout: Maximum seconds for platform recovery
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.heartbeat_interval = heartbeat_interval
        self.recovery_timeout = recovery_timeout

        self.platforms: dict[str, PlatformHealth] = {}
        self._monitoring_task: asyncio.Task | None = None
        self._running = False
        self._failure_callbacks: list = []
        self._recovery_callbacks: list = []

    def register_platform(self, platform_id: str, capabilities: dict | None = None) -> None:
        """Register a platform for monitoring with optional capabilities."""
        if platform_id not in self.platforms:
            self.platforms[platform_id] = PlatformHealth(
                platform_id=platform_id,
                status=PlatformStatus.HEALTHY
            )
            logger.info("platform_registered", platform_id=platform_id, capabilities=capabilities)

    def unregister_platform(self, platform_id: str) -> None:
        """Unregister a platform from monitoring."""
        if platform_id in self.platforms:
            del self.platforms[platform_id]
            logger.info("platform_unregistered", platform_id=platform_id)

    def record_heartbeat(
        self,
        platform_id: str,
        metrics: dict[str, float] | None = None
    ) -> None:
        """
        Record heartbeat from platform.

        Args:
            platform_id: Platform identifier
            metrics: Optional performance metrics
        """
        if platform_id not in self.platforms:
            self.register_platform(platform_id)

        health = self.platforms[platform_id]
        health.metrics.last_heartbeat = datetime.now()

        # Update metrics if provided
        if metrics:
            health.metrics.cpu_percent = metrics.get('cpu_percent', 0.0)
            health.metrics.memory_mb = metrics.get('memory_mb', 0.0)
            health.metrics.memory_percent = metrics.get('memory_percent', 0.0)
            health.metrics.tasks_completed = metrics.get('tasks_completed', 0)
            health.metrics.avg_task_duration = metrics.get('avg_task_duration', 0.0)
            health.metrics.error_count = metrics.get('error_count', 0)

        # Update status if platform was failed/degraded
        old_status = health.status
        if health.status in [PlatformStatus.FAILED, PlatformStatus.DISCONNECTED]:
            health.status = PlatformStatus.RECOVERING
            health.consecutive_failures = 0
            logger.info(
                "platform_recovering",
                platform_id=platform_id,
                old_status=old_status.value
            )
        elif health.status == PlatformStatus.RECOVERING:
            # Check if fully recovered
            if health.metrics.error_count == 0:
                health.status = PlatformStatus.HEALTHY
                health.recovery_attempts = 0
                logger.info("platform_recovered", platform_id=platform_id)
                self._trigger_recovery_callbacks(platform_id)
        elif health.status == PlatformStatus.DEGRADED:
            # Check if back to healthy
            if (health.metrics.cpu_percent < 80.0 and
                health.metrics.memory_percent < 85.0 and
                health.metrics.error_count == 0):
                health.status = PlatformStatus.HEALTHY
                logger.info("platform_healthy", platform_id=platform_id)

    def get_platform_status(self, platform_id: str) -> PlatformStatus | None:
        """Get current platform status."""
        health = self.platforms.get(platform_id)
        return health.status if health else None

    def get_platform_metrics(self, platform_id: str) -> PlatformMetrics | None:
        """Get current platform metrics."""
        health = self.platforms.get(platform_id)
        return health.metrics if health else None

    def get_healthy_platforms(self) -> list[str]:
        """Get list of healthy platform IDs."""
        return [
            platform_id
            for platform_id, health in self.platforms.items()
            if health.status == PlatformStatus.HEALTHY
        ]

    def get_failed_platforms(self) -> list[str]:
        """Get list of failed platform IDs."""
        return [
            platform_id
            for platform_id, health in self.platforms.items()
            if health.status in [PlatformStatus.FAILED, PlatformStatus.DISCONNECTED]
        ]

    def get_all_platform_status(self) -> dict[str, PlatformHealth]:
        """Get status for all platforms."""
        return self.platforms.copy()

    async def process_heartbeat(
        self,
        platform_id: str,
        metrics: dict[str, float] | None = None,
        status: str | None = None
    ) -> None:
        """
        Process heartbeat asynchronously (alias for record_heartbeat).

        Args:
            platform_id: Platform identifier
            metrics: Optional performance metrics
            status: Optional status string (ignored, status derived from metrics)
        """
        self.record_heartbeat(platform_id, metrics)

    async def start_monitoring(self) -> None:
        """Start background monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("health_monitoring_started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("health_monitoring_stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_platforms()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e))

    async def _check_all_platforms(self) -> None:
        """Check health of all registered platforms."""
        now = datetime.now()

        for platform_id, health in self.platforms.items():
            if health.metrics.last_heartbeat is None:
                continue

            time_since_heartbeat = (now - health.metrics.last_heartbeat).total_seconds()

            # Check for timeout
            if time_since_heartbeat > self.heartbeat_timeout:
                self._handle_platform_timeout(platform_id, health, time_since_heartbeat)

            # Check for degraded performance
            elif health.status == PlatformStatus.HEALTHY:
                if (health.metrics.cpu_percent > 90.0 or
                    health.metrics.memory_percent > 90.0 or
                    health.metrics.error_count > 5):
                    health.status = PlatformStatus.DEGRADED
                    health.last_status_change = now
                    logger.warning(
                        "platform_degraded",
                        platform_id=platform_id,
                        cpu=health.metrics.cpu_percent,
                        memory=health.metrics.memory_percent,
                        errors=health.metrics.error_count
                    )

    def _handle_platform_timeout(
        self,
        platform_id: str,
        health: PlatformHealth,
        time_since_heartbeat: float
    ) -> None:
        """Handle platform timeout."""
        health.consecutive_failures += 1

        if health.status not in [PlatformStatus.FAILED, PlatformStatus.DISCONNECTED]:
            old_status = health.status
            health.status = PlatformStatus.FAILED
            health.last_status_change = datetime.now()

            logger.error(
                "platform_timeout",
                platform_id=platform_id,
                old_status=old_status.value,
                time_since_heartbeat=time_since_heartbeat,
                consecutive_failures=health.consecutive_failures
            )

            self._trigger_failure_callbacks(platform_id)

    def on_platform_failure(self, callback) -> None:
        """Register callback for platform failures."""
        self._failure_callbacks.append(callback)

    def on_platform_recovery(self, callback) -> None:
        """Register callback for platform recoveries."""
        self._recovery_callbacks.append(callback)

    def _trigger_failure_callbacks(self, platform_id: str) -> None:
        """Trigger all registered failure callbacks."""
        for callback in self._failure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(platform_id))
                else:
                    callback(platform_id)
            except Exception as e:
                logger.error(
                    "failure_callback_error",
                    platform_id=platform_id,
                    error=str(e)
                )

    def _trigger_recovery_callbacks(self, platform_id: str) -> None:
        """Trigger all registered recovery callbacks."""
        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(platform_id))
                else:
                    callback(platform_id)
            except Exception as e:
                logger.error(
                    "recovery_callback_error",
                    platform_id=platform_id,
                    error=str(e)
                )

    def get_monitoring_summary(self) -> dict[str, any]:
        """Get summary of platform health status."""
        status_counts = {
            status.value: 0 for status in PlatformStatus
        }

        for health in self.platforms.values():
            status_counts[health.status.value] += 1

        return {
            'total_platforms': len(self.platforms),
            'status_counts': status_counts,
            'healthy_platforms': self.get_healthy_platforms(),
            'failed_platforms': self.get_failed_platforms(),
            'monitoring_active': self._running
        }
