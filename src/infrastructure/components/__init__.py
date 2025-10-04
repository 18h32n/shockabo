"""Infrastructure components package."""

from .email_notifier import (
    EmailConfig,
    EmailNotifier,
    NotificationType,
    Priority,
    RateLimitConfig,
    get_email_notifier,
    send_experiment_completion_notification,
    send_experiment_failure_notification,
    send_platform_failure_notification,
    send_quota_warning_notification,
)
from .gpu_monitor import get_gpu_monitor
from .graceful_shutdown import (
    ExperimentShutdownManager,
    GracefulShutdownManager,
    PlatformTimeoutDetector,
    ShutdownEvent,
    ShutdownReason,
    get_shutdown_manager,
    initiate_platform_rotation_shutdown,
    register_checkpoint_save_hook,
    register_resource_cleanup_hook,
)
from .platform_availability import (
    AvailabilityCheck,
    AvailabilityStatus,
    PlatformAvailabilityChecker,
    QuotaInfo,
    get_availability_checker,
)
from .platform_detector import Platform, PlatformDetector, PlatformInfo, get_platform_detector

__all__ = [
    'PlatformDetector', 'Platform', 'PlatformInfo', 'get_platform_detector',
    'PlatformAvailabilityChecker', 'AvailabilityStatus', 'QuotaInfo', 'AvailabilityCheck',
    'get_availability_checker',
    'GracefulShutdownManager', 'ExperimentShutdownManager', 'PlatformTimeoutDetector',
    'ShutdownReason', 'ShutdownEvent', 'get_shutdown_manager',
    'register_checkpoint_save_hook', 'register_resource_cleanup_hook',
    'initiate_platform_rotation_shutdown',
    'EmailNotifier', 'EmailConfig', 'RateLimitConfig', 'NotificationType', 'Priority',
    'get_email_notifier', 'send_experiment_completion_notification',
    'send_experiment_failure_notification', 'send_platform_failure_notification',
    'send_quota_warning_notification', 'get_gpu_monitor'
]
