"""Health check models and data structures for system monitoring."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HealthStatus(Enum):
    """Health status levels for components and overall system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components that can be monitored."""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    SERVICE = "service"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    last_checked: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "details": self.details,
            "error_message": self.error_message,
            "last_checked": self.last_checked.isoformat()
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for health monitoring."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    active_connections: int = 0
    request_rate_per_minute: float = 0.0
    error_rate_percent: float = 0.0
    average_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "active_connections": self.active_connections,
            "request_rate_per_minute": self.request_rate_per_minute,
            "error_rate_percent": self.error_rate_percent,
            "average_response_time_ms": self.average_response_time_ms,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class DatabaseHealth:
    """Database-specific health information."""
    connection_pool_size: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    query_response_time_ms: float = 0.0
    total_queries: int = 0
    failed_queries: int = 0
    database_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "connection_pool_size": self.connection_pool_size,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "query_response_time_ms": self.query_response_time_ms,
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "database_size_mb": self.database_size_mb
        }


@dataclass
class CacheHealth:
    """Cache-specific health information."""
    hit_rate_percent: float = 0.0
    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0
    cache_size_mb: float = 0.0
    max_size_mb: float = 0.0
    usage_percent: float = 0.0
    evictions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hit_rate_percent": self.hit_rate_percent,
            "total_requests": self.total_requests,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "cache_size_mb": self.cache_size_mb,
            "max_size_mb": self.max_size_mb,
            "usage_percent": self.usage_percent,
            "evictions": self.evictions
        }


@dataclass
class ExternalServiceHealth:
    """External service health information."""
    service_name: str
    endpoint_url: str
    last_successful_call: datetime | None = None
    consecutive_failures: int = 0
    total_calls: int = 0
    success_rate_percent: float = 0.0
    average_response_time_ms: float = 0.0
    rate_limit_remaining: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "endpoint_url": self.endpoint_url,
            "last_successful_call": self.last_successful_call.isoformat() if self.last_successful_call else None,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "success_rate_percent": self.success_rate_percent,
            "average_response_time_ms": self.average_response_time_ms,
            "rate_limit_remaining": self.rate_limit_remaining
        }


@dataclass
class SystemHealthSummary:
    """Overall system health summary."""
    overall_status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.3.0"
    environment: str = "development"
    uptime_seconds: float = 0.0
    components: list[HealthCheckResult] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    detailed_health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "environment": self.environment,
            "uptime_seconds": self.uptime_seconds,
            "components": [comp.to_dict() for comp in self.components],
            "performance_metrics": self.performance_metrics.to_dict(),
            "detailed_health": self.detailed_health
        }

    def add_component_result(self, result: HealthCheckResult) -> None:
        """Add a component health check result."""
        self.components.append(result)

    def get_component_by_name(self, name: str) -> HealthCheckResult | None:
        """Get component health result by name."""
        for component in self.components:
            if component.component_name == name:
                return component
        return None

    def get_components_by_type(self, component_type: ComponentType) -> list[HealthCheckResult]:
        """Get all components of a specific type."""
        return [comp for comp in self.components if comp.component_type == component_type]

    def calculate_overall_status(self) -> HealthStatus:
        """Calculate overall status based on component statuses."""
        if not self.components:
            return HealthStatus.UNKNOWN

        # Count status types
        status_counts = {}
        for component in self.components:
            status = component.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Determine overall status
        total_components = len(self.components)
        unhealthy_count = status_counts.get(HealthStatus.UNHEALTHY, 0)
        degraded_count = status_counts.get(HealthStatus.DEGRADED, 0)
        healthy_count = status_counts.get(HealthStatus.HEALTHY, 0)
        unknown_count = status_counts.get(HealthStatus.UNKNOWN, 0)

        # If any component is unhealthy, system is unhealthy
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY

        # If more than 50% are degraded or unknown, system is degraded
        if (degraded_count + unknown_count) > (total_components * 0.5):
            return HealthStatus.DEGRADED

        # If any component is degraded, system is degraded
        if degraded_count > 0:
            return HealthStatus.DEGRADED

        # If all components are healthy
        if healthy_count == total_components:
            return HealthStatus.HEALTHY

        # Default to degraded if we can't determine
        return HealthStatus.DEGRADED


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    enabled: bool = True
    check_interval_seconds: int = 30
    timeout_seconds: int = 5
    max_failures: int = 3
    degraded_threshold_ms: float = 1000.0
    unhealthy_threshold_ms: float = 5000.0
    components_to_check: list[str] = field(default_factory=lambda: [
        "database", "cache", "wandb", "storage", "memory", "cpu"
    ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "check_interval_seconds": self.check_interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "max_failures": self.max_failures,
            "degraded_threshold_ms": self.degraded_threshold_ms,
            "unhealthy_threshold_ms": self.unhealthy_threshold_ms,
            "components_to_check": self.components_to_check
        }


class HealthCheckRegistry:
    """Registry for health check functions."""

    def __init__(self):
        """Initialize the registry."""
        self._checks = {}

    def register(self, name: str, check_function, component_type: ComponentType):
        """Register a health check function.

        Args:
            name: Name of the health check
            check_function: Function that returns HealthCheckResult
            component_type: Type of component being checked
        """
        self._checks[name] = {
            'function': check_function,
            'component_type': component_type
        }

    def get_check(self, name: str):
        """Get a registered health check function."""
        return self._checks.get(name)

    def get_all_checks(self) -> dict[str, Any]:
        """Get all registered health checks."""
        return self._checks.copy()

    def get_checks_by_type(self, component_type: ComponentType) -> dict[str, Any]:
        """Get all health checks of a specific type."""
        return {
            name: check_info
            for name, check_info in self._checks.items()
            if check_info['component_type'] == component_type
        }


# Global health check registry
health_check_registry = HealthCheckRegistry()
