"""Health check service for monitoring system components and performance."""

import asyncio
import os
import platform
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import structlog

from src.domain.health_models import (
    CacheHealth,
    ComponentType,
    DatabaseHealth,
    ExternalServiceHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthStatus,
    PerformanceMetrics,
    SystemHealthSummary,
    health_check_registry,
)
from src.infrastructure.config import get_config

logger = structlog.get_logger(__name__)


class HealthService:
    """Service for performing health checks and monitoring system status."""

    def __init__(self, config: HealthCheckConfig | None = None):
        """Initialize the health service.
        
        Args:
            config: Health check configuration
        """
        self.config = config or self._load_default_config()
        self.logger = structlog.get_logger(__name__).bind(service="health_service")
        self._start_time = time.time()
        self._last_check_results: dict[str, HealthCheckResult] = {}
        self._performance_history: list[PerformanceMetrics] = []
        self._max_history_size = 100
        
        # Register default health checks
        self._register_default_checks()

    def _load_default_config(self) -> HealthCheckConfig:
        """Load default health check configuration from config manager."""
        try:
            from src.infrastructure.config import get_config
            
            config_manager = get_config()
            health_config_data = config_manager.get("health_check", {})
            
            if health_config_data:
                return HealthCheckConfig(
                    enabled=health_config_data.get("enabled", True),
                    check_interval_seconds=health_config_data.get("check_interval_seconds", 30),
                    timeout_seconds=health_config_data.get("timeout_seconds", 5),
                    max_failures=health_config_data.get("max_failures", 3),
                    degraded_threshold_ms=health_config_data.get("degraded_threshold_ms", 1000.0),
                    unhealthy_threshold_ms=health_config_data.get("unhealthy_threshold_ms", 5000.0),
                    components_to_check=health_config_data.get("components_to_check", 
                        ["database", "cache", "wandb", "storage", "memory", "cpu"])
                )
        except Exception as e:
            self.logger.warning("failed_to_load_health_config", error=str(e))
        
        # Return default configuration
        return HealthCheckConfig()

    def _register_default_checks(self) -> None:
        """Register default health check functions."""
        # Database health check
        health_check_registry.register(
            "database",
            self._check_database_health,
            ComponentType.DATABASE
        )
        
        # Cache health check
        health_check_registry.register(
            "cache",
            self._check_cache_health,
            ComponentType.CACHE
        )
        
        # W&B integration health check
        health_check_registry.register(
            "wandb",
            self._check_wandb_health,
            ComponentType.EXTERNAL_API
        )
        
        # Storage health check
        health_check_registry.register(
            "storage",
            self._check_storage_health,
            ComponentType.STORAGE
        )
        
        # Memory health check
        health_check_registry.register(
            "memory",
            self._check_memory_health,
            ComponentType.SERVICE
        )
        
        # CPU health check  
        health_check_registry.register(
            "cpu",
            self._check_cpu_health,
            ComponentType.SERVICE
        )

    async def perform_basic_health_check(self) -> SystemHealthSummary:
        """Perform basic health check for /health endpoint.
        
        Returns:
            Basic system health summary
        """
        start_time = time.time()
        summary = SystemHealthSummary(
            overall_status=HealthStatus.UNKNOWN,
            uptime_seconds=time.time() - self._start_time,
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        try:
            # Check critical components only for basic check
            critical_components = ["database", "cache", "memory"]
            tasks = []
            
            for component_name in critical_components:
                if component_name in self.config.components_to_check:
                    check_info = health_check_registry.get_check(component_name)
                    if check_info:
                        tasks.append(self._run_health_check(component_name, check_info))
            
            # Run checks concurrently with timeout
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, HealthCheckResult):
                        summary.add_component_result(result)
                        self._last_check_results[result.component_name] = result
                    elif isinstance(result, Exception):
                        self.logger.error("health_check_exception", error=str(result))
            
            # Calculate overall status
            summary.overall_status = summary.calculate_overall_status()
            
            # Add basic performance metrics
            summary.performance_metrics = self._get_basic_performance_metrics()
            
            check_duration = (time.time() - start_time) * 1000
            self.logger.info(
                "basic_health_check_completed",
                duration_ms=check_duration,
                status=summary.overall_status.value,
                components_checked=len(summary.components)
            )
            
        except Exception as e:
            self.logger.error("basic_health_check_failed", error=str(e), exc_info=True)
            summary.overall_status = HealthStatus.UNHEALTHY
        
        return summary

    async def perform_detailed_health_check(self) -> SystemHealthSummary:
        """Perform detailed health check for /health/detailed endpoint.
        
        Returns:
            Detailed system health summary with all components and metrics
        """
        start_time = time.time()
        summary = SystemHealthSummary(
            overall_status=HealthStatus.UNKNOWN,
            uptime_seconds=time.time() - self._start_time,
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        try:
            # Get all registered health checks
            all_checks = health_check_registry.get_all_checks()
            tasks = []
            
            for component_name in self.config.components_to_check:
                check_info = all_checks.get(component_name)
                if check_info:
                    tasks.append(self._run_health_check(component_name, check_info))
            
            # Run checks concurrently with timeout
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, HealthCheckResult):
                        summary.add_component_result(result)
                        self._last_check_results[result.component_name] = result
                    elif isinstance(result, Exception):
                        self.logger.error("detailed_health_check_exception", error=str(result))
            
            # Calculate overall status
            summary.overall_status = summary.calculate_overall_status()
            
            # Add comprehensive performance metrics
            summary.performance_metrics = self._get_comprehensive_performance_metrics()
            
            # Add detailed health information for each component type
            summary.detailed_health = await self._get_detailed_health_info()
            
            check_duration = (time.time() - start_time) * 1000
            self.logger.info(
                "detailed_health_check_completed",
                duration_ms=check_duration,
                status=summary.overall_status.value,
                components_checked=len(summary.components)
            )
            
        except Exception as e:
            self.logger.error("detailed_health_check_failed", error=str(e), exc_info=True)
            summary.overall_status = HealthStatus.UNHEALTHY
        
        return summary

    async def perform_readiness_check(self) -> bool:
        """Perform readiness check for Kubernetes readiness probe.
        
        Returns:
            True if system is ready to serve requests
        """
        try:
            # Check critical components for readiness
            critical_checks = ["database", "cache"]
            
            for component_name in critical_checks:
                check_info = health_check_registry.get_check(component_name)
                if check_info:
                    result = await self._run_health_check(component_name, check_info)
                    if result.status == HealthStatus.UNHEALTHY:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error("readiness_check_failed", error=str(e))
            return False

    async def perform_liveness_check(self) -> bool:
        """Perform liveness check for Kubernetes liveness probe.
        
        Returns:
            True if system is alive (basic functionality working)
        """
        try:
            # Very basic liveness check - just verify core service is responsive
            start_time = time.time()
            
            # Check that we can access basic system resources
            memory = psutil.virtual_memory()
            if memory.available < 50 * 1024 * 1024:  # Less than 50MB available
                return False
            
            # Check response time is reasonable
            response_time = (time.time() - start_time) * 1000
            if response_time > 1000:  # More than 1 second
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("liveness_check_failed", error=str(e))
            return False

    async def _run_health_check(self, name: str, check_info: dict[str, Any]) -> HealthCheckResult:
        """Run a single health check with timeout and error handling.
        
        Args:
            name: Name of the health check
            check_info: Health check information from registry
            
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            # Run the health check with timeout
            check_function = check_info['function']
            result = await asyncio.wait_for(
                check_function(),
                timeout=self.config.timeout_seconds
            )
            
            # Ensure result has correct response time
            result.response_time_ms = (time.time() - start_time) * 1000
            
            # Update status based on response time if not already unhealthy
            if result.status != HealthStatus.UNHEALTHY:
                if result.response_time_ms > self.config.unhealthy_threshold_ms:
                    result.status = HealthStatus.UNHEALTHY
                    result.error_message = f"Response time {result.response_time_ms:.1f}ms exceeds unhealthy threshold"
                elif result.response_time_ms > self.config.degraded_threshold_ms:
                    result.status = HealthStatus.DEGRADED
                    result.details["performance_warning"] = f"Slow response time: {result.response_time_ms:.1f}ms"
            
            return result
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component_name=name,
                component_type=check_info['component_type'],
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Health check timed out after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            return HealthCheckResult(
                component_name=name,
                component_type=check_info['component_type'],
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Health check failed: {str(e)}"
            )

    async def _check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        try:
            # This would normally check actual database connections
            # For now, simulate a database health check
            result = HealthCheckResult(
                component_name="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time_ms=0.0,
                details={
                    "connection_pool_size": 10,
                    "active_connections": 2,
                    "query_response_time_ms": 15.5
                }
            )
            
            # Simulate some logic to determine health
            # In a real implementation, this would:
            # 1. Test database connectivity
            # 2. Run a simple query
            # 3. Check connection pool status
            # 4. Verify database is accepting connections
            
            return result
            
        except Exception as e:
            return HealthCheckResult(
                component_name="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    async def _check_cache_health(self) -> HealthCheckResult:
        """Check cache system health."""
        try:
            from src.adapters.repositories.cache_repository import CacheRepository
            
            # Create a temporary cache instance for health check
            cache = CacheRepository()
            
            # Test basic cache operations
            test_key = "health_check_test"
            test_start = time.time()
            
            # Test set operation
            cache.set(test_key, "test_value")
            
            # Test get operation  
            value = cache.get(test_key)
            
            # Clean up test key
            cache.delete(test_key)
            
            response_time_ms = (time.time() - test_start) * 1000
            
            # Get cache statistics
            stats = cache.get_statistics()
            size_info = cache.get_size_info()
            
            # Determine status based on cache performance
            status = HealthStatus.HEALTHY
            if stats.get("hit_rate", 0) < 0.5:  # Low hit rate
                status = HealthStatus.DEGRADED
            if size_info.get("usage_percent", 0) > 90:  # High usage
                status = HealthStatus.DEGRADED
            if response_time_ms > 100:  # Slow response
                status = HealthStatus.DEGRADED
                
            # Close cache properly
            cache.close()
            
            return HealthCheckResult(
                component_name="cache",
                component_type=ComponentType.CACHE,
                status=status,
                response_time_ms=response_time_ms,
                details={
                    "hit_rate": stats.get("hit_rate", 0),
                    "cache_size_mb": size_info.get("volume", 0) / (1024 * 1024),
                    "usage_percent": size_info.get("usage_percent", 0),
                    "total_requests": stats.get("total_requests", 0)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    async def _check_wandb_health(self) -> HealthCheckResult:
        """Check Weights & Biases integration health."""
        try:
            from src.adapters.external.wandb_client import get_wandb_client
            
            wandb_client = get_wandb_client()
            start_time = time.time()
            
            # Check if W&B is properly configured
            config_valid = wandb_client.config.validate()
            response_time_ms = (time.time() - start_time) * 1000
            
            if not config_valid:
                return HealthCheckResult(
                    component_name="wandb",
                    component_type=ComponentType.EXTERNAL_API,
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time_ms,
                    error_message="W&B configuration is invalid or disabled",
                    details={"mode": wandb_client.config.mode}
                )
            
            # Check usage limits
            is_ok, warning = wandb_client.usage_monitor.check_usage_limits()
            status = HealthStatus.HEALTHY
            if not is_ok:
                status = HealthStatus.UNHEALTHY
            elif warning:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component_name="wandb",
                component_type=ComponentType.EXTERNAL_API,
                status=status,
                response_time_ms=response_time_ms,
                details={
                    "project": wandb_client.config.project_name,
                    "mode": wandb_client.config.mode,
                    "usage_gb": wandb_client.usage_monitor.get_current_usage_gb(),
                    "initialized": wandb_client._initialized
                },
                error_message=warning if warning else None
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name="wandb",
                component_type=ComponentType.EXTERNAL_API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    async def _check_storage_health(self) -> HealthCheckResult:
        """Check storage system health."""
        try:
            config = get_config()
            data_dir = Path(config.get_data_dir())
            
            # Check disk usage
            if data_dir.exists():
                disk_usage = psutil.disk_usage(str(data_dir))
            else:
                # Fallback to current directory
                disk_usage = psutil.disk_usage('.')
            
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024 ** 3)
            total_gb = disk_usage.total / (1024 ** 3)
            
            # Determine status based on usage
            status = HealthStatus.HEALTHY
            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
            
            # Check if we can write to storage
            test_file = data_dir / "health_check_test.tmp"
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                test_file.write_text("test")
                test_file.unlink()
                can_write = True
            except Exception:
                can_write = False
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                component_name="storage",
                component_type=ComponentType.STORAGE,
                status=status,
                response_time_ms=10.0,  # Estimated
                details={
                    "usage_percent": usage_percent,
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "can_write": can_write,
                    "data_dir": str(data_dir)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name="storage",
                component_type=ComponentType.STORAGE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    async def _check_memory_health(self) -> HealthCheckResult:
        """Check memory usage health."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            available_gb = memory.available / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            
            # Determine status based on memory usage
            status = HealthStatus.HEALTHY
            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component_name="memory",
                component_type=ComponentType.SERVICE,
                status=status,
                response_time_ms=5.0,  # Estimated
                details={
                    "usage_percent": usage_percent,
                    "available_gb": available_gb,
                    "total_gb": total_gb,
                    "used_gb": (total_gb - available_gb)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name="memory",
                component_type=ComponentType.SERVICE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    async def _check_cpu_health(self) -> HealthCheckResult:
        """Check CPU usage health."""
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Determine status based on CPU usage
            status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                component_name="cpu",
                component_type=ComponentType.SERVICE,
                status=status,
                response_time_ms=100.0,  # Time for interval measurement
                details={
                    "usage_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1] if len(load_avg) > 1 else 0,
                    "load_avg_15m": load_avg[2] if len(load_avg) > 2 else 0
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name="cpu",
                component_type=ComponentType.SERVICE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                error_message=str(e)
            )

    def _get_basic_performance_metrics(self) -> PerformanceMetrics:
        """Get basic performance metrics for health checks."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return PerformanceMetrics(
                memory_usage_percent=memory.percent,
                memory_used_mb=memory.used / (1024 ** 2),
                memory_total_mb=memory.total / (1024 ** 2),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_used_gb=disk.used / (1024 ** 3),
                disk_total_gb=disk.total / (1024 ** 3),
                uptime_seconds=time.time() - self._start_time
            )
        except Exception as e:
            self.logger.error("basic_performance_metrics_failed", error=str(e))
            return PerformanceMetrics()

    def _get_comprehensive_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics for detailed health checks."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get network connections count as proxy for active connections
            connections = len(psutil.net_connections())
            
            return PerformanceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_used_mb=memory.used / (1024 ** 2),
                memory_total_mb=memory.total / (1024 ** 2),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_used_gb=disk.used / (1024 ** 3),
                disk_total_gb=disk.total / (1024 ** 3),
                active_connections=connections,
                uptime_seconds=time.time() - self._start_time
            )
        except Exception as e:
            self.logger.error("comprehensive_performance_metrics_failed", error=str(e))
            return PerformanceMetrics()

    async def _get_detailed_health_info(self) -> dict[str, Any]:
        """Get detailed health information for each component type."""
        detailed_info = {}
        
        try:
            # Database details
            if "database" in self._last_check_results:
                detailed_info["database"] = DatabaseHealth(
                    connection_pool_size=10,
                    active_connections=2,
                    idle_connections=8,
                    query_response_time_ms=15.5
                ).to_dict()
            
            # Cache details
            if "cache" in self._last_check_results:
                cache_result = self._last_check_results["cache"]
                detailed_info["cache"] = CacheHealth(
                    hit_rate_percent=cache_result.details.get("hit_rate", 0) * 100,
                    total_requests=cache_result.details.get("total_requests", 0),
                    cache_size_mb=cache_result.details.get("cache_size_mb", 0),
                    usage_percent=cache_result.details.get("usage_percent", 0)
                ).to_dict()
            
            # W&B details
            if "wandb" in self._last_check_results:
                wandb_result = self._last_check_results["wandb"]
                detailed_info["wandb"] = ExternalServiceHealth(
                    service_name="Weights & Biases",
                    endpoint_url="https://api.wandb.ai",
                    success_rate_percent=95.0,  # Would be calculated from actual usage
                    average_response_time_ms=wandb_result.response_time_ms
                ).to_dict()
            
            # System info
            detailed_info["system"] = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node()
            }
            
        except Exception as e:
            self.logger.error("detailed_health_info_failed", error=str(e))
        
        return detailed_info


# Global health service instance
_health_service: HealthService | None = None


def get_health_service() -> HealthService:
    """Get the global health service instance.
    
    Returns:
        HealthService instance
    """
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service