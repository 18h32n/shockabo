"""Platform availability checker with quota monitoring."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .platform_detector import Platform, PlatformDetector, get_platform_detector


class AvailabilityStatus(Enum):
    """Platform availability status."""
    AVAILABLE = "available"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT_APPROACHING = "timeout_approaching"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class QuotaInfo:
    """Platform quota information."""
    platform: Platform
    gpu_hours_used: float
    gpu_hours_total: float
    gpu_hours_remaining: float
    reset_time: datetime | None = None
    session_start_time: datetime | None = None
    session_duration_hours: float = 0.0
    session_timeout_hours: float = 12.0
    last_updated: datetime = None


@dataclass
class AvailabilityCheck:
    """Platform availability check result."""
    platform: Platform
    status: AvailabilityStatus
    quota_info: QuotaInfo
    can_start_experiment: bool
    estimated_runtime_hours: float = 0.0
    next_available_time: datetime | None = None
    warnings: list[str] = None
    metadata: dict[str, any] = None


class PlatformAvailabilityChecker:
    """Service for checking platform availability and monitoring quotas."""

    def __init__(self, detector: PlatformDetector | None = None):
        self.detector = detector or get_platform_detector()
        self._quota_cache: dict[Platform, QuotaInfo] = {}
        self._session_trackers: dict[Platform, datetime] = {}

    async def check_availability(self, platform: Platform,
                               estimated_runtime_hours: float = 1.0) -> AvailabilityCheck:
        """
        Check if a platform is available for starting an experiment.

        Args:
            platform: Platform to check
            estimated_runtime_hours: Expected experiment runtime

        Returns:
            AvailabilityCheck with current status and recommendations
        """
        quota_info = await self._get_quota_info(platform)
        status = self._determine_availability_status(quota_info, estimated_runtime_hours)
        can_start = self._can_start_experiment(quota_info, estimated_runtime_hours)
        next_available = self._calculate_next_available_time(quota_info, platform)
        warnings = self._generate_warnings(quota_info, estimated_runtime_hours)

        return AvailabilityCheck(
            platform=platform,
            status=status,
            quota_info=quota_info,
            can_start_experiment=can_start,
            estimated_runtime_hours=estimated_runtime_hours,
            next_available_time=next_available,
            warnings=warnings,
            metadata=self._get_platform_metadata(platform)
        )

    async def check_all_platforms(self,
                                estimated_runtime_hours: float = 1.0) -> list[AvailabilityCheck]:
        """Check availability for all supported platforms."""
        platforms = [Platform.KAGGLE, Platform.COLAB, Platform.PAPERSPACE, Platform.LOCAL]
        checks = []

        for platform in platforms:
            try:
                check = await self.check_availability(platform, estimated_runtime_hours)
                checks.append(check)
            except Exception as e:
                # Create failed check
                checks.append(AvailabilityCheck(
                    platform=platform,
                    status=AvailabilityStatus.UNKNOWN,
                    quota_info=QuotaInfo(
                        platform=platform,
                        gpu_hours_used=0,
                        gpu_hours_total=0,
                        gpu_hours_remaining=0,
                        last_updated=datetime.now()
                    ),
                    can_start_experiment=False,
                    warnings=[f"Failed to check availability: {str(e)}"]
                ))

        return checks

    async def get_best_platform(self,
                              estimated_runtime_hours: float = 1.0,
                              prefer_current: bool = True) -> AvailabilityCheck | None:
        """
        Get the best available platform for starting an experiment.

        Args:
            estimated_runtime_hours: Expected experiment runtime
            prefer_current: Prefer current platform if available

        Returns:
            Best available platform or None if none available
        """
        checks = await self.check_all_platforms(estimated_runtime_hours)
        available_checks = [c for c in checks if c.can_start_experiment]

        if not available_checks:
            return None

        # If prefer_current is True, check if current platform is available
        if prefer_current:
            current_platform = self.detector.detect_platform().platform
            current_check = next((c for c in available_checks if c.platform == current_platform), None)
            if current_check:
                return current_check

        # Sort by preference: most remaining hours, then by platform priority
        def platform_score(check: AvailabilityCheck) -> tuple[float, int]:
            # Higher remaining hours is better
            remaining_hours = check.quota_info.gpu_hours_remaining

            # Platform priority (lower is better)
            priority_map = {
                Platform.KAGGLE: 1,    # 30 hours weekly
                Platform.COLAB: 2,     # 12 hours daily
                Platform.PAPERSPACE: 3, # 6 hours daily
                Platform.LOCAL: 0       # Unlimited (best if available)
            }
            priority = priority_map.get(check.platform, 999)

            return (-remaining_hours, priority)  # Negative for descending sort

        available_checks.sort(key=platform_score)
        return available_checks[0]

    async def _get_quota_info(self, platform: Platform) -> QuotaInfo:
        """Get quota information for a platform."""
        # Check cache first
        cached = self._quota_cache.get(platform)
        if cached and self._is_cache_valid(cached):
            return cached

        # Get fresh quota info
        quota_info = await self._fetch_quota_info(platform)
        self._quota_cache[platform] = quota_info
        return quota_info

    def _is_cache_valid(self, quota_info: QuotaInfo, max_age_minutes: int = 5) -> bool:
        """Check if cached quota info is still valid."""
        if not quota_info.last_updated:
            return False

        age = datetime.now() - quota_info.last_updated
        return age < timedelta(minutes=max_age_minutes)

    async def _fetch_quota_info(self, platform: Platform) -> QuotaInfo:
        """Fetch fresh quota information for a platform."""
        now = datetime.now()

        # Get platform info
        platform_info = self.detector.detect_platform()
        limits = platform_info.runtime_limits or {}

        if platform == Platform.KAGGLE:
            return await self._get_kaggle_quota(now, limits)
        elif platform == Platform.COLAB:
            return await self._get_colab_quota(now, limits)
        elif platform == Platform.PAPERSPACE:
            return await self._get_paperspace_quota(now, limits)
        elif platform == Platform.LOCAL:
            return await self._get_local_quota(now, limits)
        else:
            # Unknown platform
            return QuotaInfo(
                platform=platform,
                gpu_hours_used=0,
                gpu_hours_total=0,
                gpu_hours_remaining=0,
                last_updated=now
            )

    async def _get_kaggle_quota(self, now: datetime, limits: dict) -> QuotaInfo:
        """Get Kaggle-specific quota information."""
        # TODO: Implement actual Kaggle API calls
        total_hours = limits.get('gpu_hours_weekly', 30)

        # Placeholder implementation - in real implementation, would call Kaggle API
        # For now, estimate based on session tracking
        session_start = self._session_trackers.get(Platform.KAGGLE, now)
        session_duration = (now - session_start).total_seconds() / 3600

        # Estimate usage (this would come from API in real implementation)
        used_hours = min(session_duration, total_hours * 0.8)  # Cap at 80% for safety
        remaining_hours = max(0, total_hours - used_hours)

        # Reset time is weekly (Monday 00:00 UTC)
        reset_time = self._get_next_weekly_reset()

        return QuotaInfo(
            platform=Platform.KAGGLE,
            gpu_hours_used=used_hours,
            gpu_hours_total=total_hours,
            gpu_hours_remaining=remaining_hours,
            reset_time=reset_time,
            session_start_time=session_start,
            session_duration_hours=session_duration,
            session_timeout_hours=limits.get('session_timeout_hours', 12),
            last_updated=now
        )

    async def _get_colab_quota(self, now: datetime, limits: dict) -> QuotaInfo:
        """Get Colab-specific quota information."""
        # TODO: Implement actual Colab quota checking
        total_hours = limits.get('gpu_hours_daily', 12)

        session_start = self._session_trackers.get(Platform.COLAB, now)
        session_duration = (now - session_start).total_seconds() / 3600

        # Estimate usage
        used_hours = min(session_duration, total_hours * 0.9)
        remaining_hours = max(0, total_hours - used_hours)

        # Reset time is daily at midnight UTC
        reset_time = self._get_next_daily_reset()

        return QuotaInfo(
            platform=Platform.COLAB,
            gpu_hours_used=used_hours,
            gpu_hours_total=total_hours,
            gpu_hours_remaining=remaining_hours,
            reset_time=reset_time,
            session_start_time=session_start,
            session_duration_hours=session_duration,
            session_timeout_hours=limits.get('session_timeout_hours', 12),
            last_updated=now
        )

    async def _get_paperspace_quota(self, now: datetime, limits: dict) -> QuotaInfo:
        """Get Paperspace-specific quota information."""
        # TODO: Implement actual Paperspace API calls
        total_hours = limits.get('gpu_hours_daily', 6)

        session_start = self._session_trackers.get(Platform.PAPERSPACE, now)
        session_duration = (now - session_start).total_seconds() / 3600

        # Estimate usage
        used_hours = min(session_duration, total_hours * 0.8)
        remaining_hours = max(0, total_hours - used_hours)

        # Reset time is daily at midnight UTC
        reset_time = self._get_next_daily_reset()

        return QuotaInfo(
            platform=Platform.PAPERSPACE,
            gpu_hours_used=used_hours,
            gpu_hours_total=total_hours,
            gpu_hours_remaining=remaining_hours,
            reset_time=reset_time,
            session_start_time=session_start,
            session_duration_hours=session_duration,
            session_timeout_hours=limits.get('session_timeout_hours', 6),
            last_updated=now
        )

    async def _get_local_quota(self, now: datetime, limits: dict) -> QuotaInfo:
        """Get local platform quota (unlimited)."""
        return QuotaInfo(
            platform=Platform.LOCAL,
            gpu_hours_used=0,
            gpu_hours_total=-1,  # Unlimited
            gpu_hours_remaining=-1,  # Unlimited
            session_start_time=now,
            session_duration_hours=0,
            session_timeout_hours=-1,  # No timeout
            last_updated=now
        )

    def _determine_availability_status(self, quota_info: QuotaInfo,
                                     estimated_runtime: float) -> AvailabilityStatus:
        """Determine platform availability status."""
        if quota_info.gpu_hours_remaining == -1:  # Unlimited (local)
            return AvailabilityStatus.AVAILABLE

        if quota_info.gpu_hours_remaining <= 0:
            return AvailabilityStatus.QUOTA_EXCEEDED

        if quota_info.gpu_hours_remaining < estimated_runtime:
            return AvailabilityStatus.QUOTA_EXCEEDED

        # Check session timeout
        if quota_info.session_timeout_hours > 0:
            time_until_timeout = quota_info.session_timeout_hours - quota_info.session_duration_hours
            if time_until_timeout < estimated_runtime:
                return AvailabilityStatus.TIMEOUT_APPROACHING

        return AvailabilityStatus.AVAILABLE

    def _can_start_experiment(self, quota_info: QuotaInfo,
                            estimated_runtime: float) -> bool:
        """Check if an experiment can be started."""
        status = self._determine_availability_status(quota_info, estimated_runtime)
        return status in [AvailabilityStatus.AVAILABLE, AvailabilityStatus.TIMEOUT_APPROACHING]

    def _calculate_next_available_time(self, quota_info: QuotaInfo,
                                     platform: Platform) -> datetime | None:
        """Calculate when platform will next be available."""
        if quota_info.gpu_hours_remaining > 0:
            return None  # Already available

        return quota_info.reset_time

    def _generate_warnings(self, quota_info: QuotaInfo,
                         estimated_runtime: float) -> list[str]:
        """Generate warnings about quota or timing issues."""
        warnings = []

        if quota_info.gpu_hours_remaining > 0:
            if quota_info.gpu_hours_remaining < estimated_runtime * 1.2:  # 20% buffer
                warnings.append(
                    f"Low quota: {quota_info.gpu_hours_remaining:.1f}h remaining, "
                    f"need {estimated_runtime:.1f}h"
                )

        if quota_info.session_timeout_hours > 0:
            time_until_timeout = quota_info.session_timeout_hours - quota_info.session_duration_hours
            if time_until_timeout < estimated_runtime * 1.1:  # 10% buffer
                warnings.append(
                    f"Session timeout approaching: {time_until_timeout:.1f}h remaining"
                )

        return warnings

    def _get_platform_metadata(self, platform: Platform) -> dict[str, any]:
        """Get additional platform metadata."""
        return {
            'check_time': datetime.now().isoformat(),
            'platform_name': platform.value
        }

    def _get_next_weekly_reset(self) -> datetime:
        """Get next Monday 00:00 UTC (weekly reset)."""
        now = datetime.utcnow()
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0 and now.hour > 0:  # If it's Monday but after midnight
            days_until_monday = 7

        next_monday = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        return next_monday

    def _get_next_daily_reset(self) -> datetime:
        """Get next 00:00 UTC (daily reset)."""
        now = datetime.utcnow()
        next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return next_reset

    def start_session_tracking(self, platform: Platform):
        """Start tracking session for a platform."""
        self._session_trackers[platform] = datetime.now()

    def stop_session_tracking(self, platform: Platform):
        """Stop tracking session for a platform."""
        self._session_trackers.pop(platform, None)

    def clear_quota_cache(self, platform: Platform | None = None):
        """Clear quota cache for a platform or all platforms."""
        if platform:
            self._quota_cache.pop(platform, None)
        else:
            self._quota_cache.clear()


# Singleton instance
_availability_checker_instance = None


def get_availability_checker() -> PlatformAvailabilityChecker:
    """Get singleton availability checker instance."""
    global _availability_checker_instance
    if _availability_checker_instance is None:
        _availability_checker_instance = PlatformAvailabilityChecker()
    return _availability_checker_instance
