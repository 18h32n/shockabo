"""Advanced error recovery mechanisms for the ARC Prize evaluation framework.

This module provides sophisticated error recovery patterns including:
- Circuit breaker patterns for external service resilience
- Retry mechanisms with exponential backoff and jitter
- Fallback strategies for critical operations
- Health monitoring and auto-recovery
- Graceful degradation patterns
"""

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import structlog

from .error_handling import (
    ARCBaseException,
    ErrorCode,
    ErrorContext,
    ErrorLogger,
    ErrorSeverity,
)

logger = structlog.get_logger(__name__)
T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # successes needed to close from half-open
    timeout: float = 30.0  # request timeout in seconds


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: list[tuple] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker implementation for handling service failures."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        error_logger: ErrorLogger | None = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.error_logger = error_logger or ErrorLogger()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            current_state = self._get_current_state()

            if current_state == CircuitState.OPEN:
                logger.warning(
                    "circuit_breaker_open",
                    name=self.name,
                    failure_count=self.stats.failure_count,
                    last_failure=self.stats.last_failure_time
                )
                raise ARCBaseException(
                    message=f"Circuit breaker '{self.name}' is open",
                    error_code=ErrorCode.EXTERNAL_API_ERROR,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(additional_data={
                        "circuit_breaker": self.name,
                        "state": current_state.value,
                        "failure_count": self.stats.failure_count
                    }),
                    suggestions=[
                        "Wait for circuit breaker to recover",
                        f"Service may be down, try again in {self.config.recovery_timeout} seconds",
                        "Check service health status"
                    ],
                    retry_after=self.config.recovery_timeout
                )

        # Execute the function
        start_time = time.time()
        self.stats.total_requests += 1

        try:
            # Add timeout if it's an async function
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()

            execution_time = time.time() - start_time
            logger.debug(
                "circuit_breaker_success",
                name=self.name,
                execution_time_ms=execution_time * 1000,
                state=self.stats.state.value
            )

            return result

        except Exception as e:
            # Record failure
            await self._record_failure(e)

            execution_time = time.time() - start_time
            logger.error(
                "circuit_breaker_failure",
                name=self.name,
                error=str(e),
                execution_time_ms=execution_time * 1000,
                state=self.stats.state.value,
                failure_count=self.stats.failure_count
            )

            raise

    async def _record_success(self):
        """Record a successful operation."""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.now()

            # Reset failure count on success
            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    await self._change_state(CircuitState.CLOSED)
                    self.stats.failure_count = 0
                    self.stats.success_count = 0

    async def _record_failure(self, error: Exception):
        """Record a failed operation."""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.now()

            # Log error with circuit breaker context
            context = ErrorContext(additional_data={
                "circuit_breaker": self.name,
                "failure_count": self.stats.failure_count,
                "state": self.stats.state.value
            })

            if isinstance(error, ARCBaseException):
                error.context = context
                self.error_logger.log_error(error)
            else:
                arc_error = ARCBaseException(
                    message=f"Circuit breaker '{self.name}' recorded failure: {str(error)}",
                    error_code=ErrorCode.EXTERNAL_API_ERROR,
                    context=context,
                    cause=error
                )
                self.error_logger.log_error(arc_error)

            # Check if we should trip the circuit breaker
            if (self.stats.failure_count >= self.config.failure_threshold and
                self.stats.state == CircuitState.CLOSED):
                await self._change_state(CircuitState.OPEN)
            elif self.stats.state == CircuitState.HALF_OPEN:
                # Go back to open on any failure in half-open state
                await self._change_state(CircuitState.OPEN)
                self.stats.failure_count = 0
                self.stats.success_count = 0

    def _get_current_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        if self.stats.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (self.stats.last_failure_time and
                datetime.now() - self.stats.last_failure_time >=
                timedelta(seconds=self.config.recovery_timeout)):
                # Don't change state here, let the caller handle it
                return CircuitState.HALF_OPEN

        return self.stats.state

    async def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self.stats.state
        self.stats.state = new_state
        self.stats.state_changes.append((datetime.now(), old_state.value, new_state.value))

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
            failure_count=self.stats.failure_count,
            success_count=self.stats.success_count
        )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "failure_rate": self.stats.total_failures / max(self.stats.total_requests, 1),
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            "recent_state_changes": self.stats.state_changes[-10:]  # Last 10 changes
        }

    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self._lock:
            await self._change_state(CircuitState.CLOSED)
            self.stats.failure_count = 0
            self.stats.success_count = 0

            logger.info("circuit_breaker_reset", name=self.name)


class RetryStrategy:
    """Advanced retry strategy with exponential backoff, jitter, and conditions."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
        non_retryable_exceptions: tuple = (),
        retry_condition: Callable[[Exception], bool] | None = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions
        self.retry_condition = retry_condition

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False

        # Check non-retryable exceptions first
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Check retryable exceptions
        if not isinstance(exception, self.retryable_exceptions):
            return False

        # Apply custom retry condition if provided
        if self.retry_condition and not self.retry_condition(exception):
            return False

        return True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add jitter to avoid thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Minimum 100ms delay

        return delay

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 1:
                    logger.info(
                        "retry_success",
                        function=func.__name__,
                        attempt=attempt,
                        total_attempts=self.max_attempts
                    )

                return result

            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt):
                    logger.error(
                        "retry_abandoned",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        error=str(e),
                        reason="non_retryable" if not isinstance(e, self.retryable_exceptions) else "max_attempts_reached"
                    )
                    break

                delay = self.calculate_delay(attempt)

                logger.warning(
                    "retry_attempt",
                    function=func.__name__,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )

                await asyncio.sleep(delay)

        # All retries failed
        if isinstance(last_exception, ARCBaseException):
            raise last_exception
        else:
            raise ARCBaseException(
                message=f"Function {func.__name__} failed after {self.max_attempts} attempts",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(additional_data={
                    "function": func.__name__,
                    "attempts": self.max_attempts,
                    "final_error": str(last_exception)
                }),
                cause=last_exception,
                suggestions=[
                    "Check if the underlying service is available",
                    "Verify network connectivity",
                    "Try increasing retry attempts or delay"
                ]
            ) from last_exception


class FallbackStrategy:
    """Fallback mechanisms for when primary operations fail."""

    def __init__(self, name: str):
        self.name = name
        self.fallbacks: list[Callable] = []
        self.execution_stats = {
            "primary_success": 0,
            "primary_failure": 0,
            "fallback_success": 0,
            "fallback_failure": 0,
            "total_requests": 0
        }

    def add_fallback(self, fallback_func: Callable, priority: int = 0):
        """Add a fallback function with optional priority (higher = more preferred)."""
        self.fallbacks.append((priority, fallback_func))
        self.fallbacks.sort(key=lambda x: x[0], reverse=True)

    async def execute(
        self,
        primary_func: Callable[..., T],
        *args,
        fallback_args: dict | None = None,
        **kwargs
    ) -> T:
        """Execute with fallback logic."""
        self.execution_stats["total_requests"] += 1
        fallback_args = fallback_args or {}

        # Try primary function first
        try:
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)

            self.execution_stats["primary_success"] += 1
            return result

        except Exception as primary_error:
            self.execution_stats["primary_failure"] += 1

            logger.warning(
                "fallback_primary_failed",
                strategy=self.name,
                primary_function=primary_func.__name__,
                error=str(primary_error)
            )

            # Try fallbacks in order of priority
            last_fallback_error = None

            for priority, fallback_func in self.fallbacks:
                try:
                    logger.info(
                        "fallback_attempting",
                        strategy=self.name,
                        fallback_function=fallback_func.__name__,
                        priority=priority
                    )

                    # Use fallback-specific args if provided
                    fb_args = fallback_args.get(fallback_func.__name__, {})

                    if asyncio.iscoroutinefunction(fallback_func):
                        result = await fallback_func(*args, **fb_args, **kwargs)
                    else:
                        result = fallback_func(*args, **fb_args, **kwargs)

                    self.execution_stats["fallback_success"] += 1

                    logger.info(
                        "fallback_success",
                        strategy=self.name,
                        fallback_function=fallback_func.__name__,
                        priority=priority
                    )

                    return result

                except Exception as fallback_error:
                    last_fallback_error = fallback_error

                    logger.error(
                        "fallback_failed",
                        strategy=self.name,
                        fallback_function=fallback_func.__name__,
                        priority=priority,
                        error=str(fallback_error)
                    )

                    continue

            # All fallbacks failed
            self.execution_stats["fallback_failure"] += 1

            raise ARCBaseException(
                message=f"All fallback strategies failed for {self.name}",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(additional_data={
                    "strategy": self.name,
                    "primary_error": str(primary_error),
                    "last_fallback_error": str(last_fallback_error),
                    "fallback_count": len(self.fallbacks)
                }),
                cause=last_fallback_error,
                suggestions=[
                    "Check all service dependencies",
                    "Verify system resources",
                    "Consider adding more fallback options"
                ]
            ) from primary_error

    def get_stats(self) -> dict[str, Any]:
        """Get fallback execution statistics."""
        total = max(self.execution_stats["total_requests"], 1)
        return {
            "name": self.name,
            "fallback_count": len(self.fallbacks),
            "statistics": self.execution_stats.copy(),
            "primary_success_rate": self.execution_stats["primary_success"] / total,
            "fallback_usage_rate": (self.execution_stats["fallback_success"] + self.execution_stats["fallback_failure"]) / total,
            "overall_success_rate": (self.execution_stats["primary_success"] + self.execution_stats["fallback_success"]) / total
        }


class HealthMonitor:
    """Health monitoring and auto-recovery for system components."""

    def __init__(self, name: str, check_interval: int = 30):
        self.name = name
        self.check_interval = check_interval
        self.health_checks: dict[str, Callable[[], bool]] = {}
        self.health_status: dict[str, dict] = {}
        self.monitoring = False
        self._monitor_task = None

    def add_health_check(
        self,
        component: str,
        check_func: Callable[[], bool],
        recovery_func: Callable[[], None] | None = None,
        critical: bool = True
    ):
        """Add a health check for a component."""
        self.health_checks[component] = check_func
        self.health_status[component] = {
            "healthy": True,
            "last_check": None,
            "failure_count": 0,
            "recovery_func": recovery_func,
            "critical": critical,
            "last_recovery_attempt": None
        }

    async def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            "health_monitoring_started",
            monitor=self.name,
            components=list(self.health_checks.keys()),
            check_interval=self.check_interval
        )

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("health_monitoring_stopped", monitor=self.name)

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_monitor_error",
                    monitor=self.name,
                    error=str(e)
                )
                await asyncio.sleep(self.check_interval)

    async def _perform_health_checks(self):
        """Perform all health checks."""
        for component, check_func in self.health_checks.items():
            try:
                is_healthy = await self._check_component(check_func)
                await self._update_component_status(component, is_healthy)
            except Exception as e:
                logger.error(
                    "health_check_error",
                    monitor=self.name,
                    component=component,
                    error=str(e)
                )
                await self._update_component_status(component, False)

    async def _check_component(self, check_func: Callable) -> bool:
        """Execute a single health check."""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        else:
            return check_func()

    async def _update_component_status(self, component: str, is_healthy: bool):
        """Update component health status."""
        status = self.health_status[component]
        previous_health = status["healthy"]

        status["healthy"] = is_healthy
        status["last_check"] = datetime.now()

        if not is_healthy:
            status["failure_count"] += 1

            if previous_health:
                logger.warning(
                    "component_unhealthy",
                    monitor=self.name,
                    component=component,
                    failure_count=status["failure_count"],
                    critical=status["critical"]
                )

            # Attempt recovery if available
            if status["recovery_func"]:
                await self._attempt_recovery(component)

        elif not previous_health:
            # Component recovered
            status["failure_count"] = 0
            logger.info(
                "component_recovered",
                monitor=self.name,
                component=component
            )

    async def _attempt_recovery(self, component: str):
        """Attempt to recover a failed component."""
        status = self.health_status[component]
        recovery_func = status["recovery_func"]

        if not recovery_func:
            return

        # Don't attempt recovery too frequently
        last_attempt = status["last_recovery_attempt"]
        if last_attempt and (datetime.now() - last_attempt).total_seconds() < 60:
            return

        try:
            status["last_recovery_attempt"] = datetime.now()

            logger.info(
                "recovery_attempt",
                monitor=self.name,
                component=component,
                failure_count=status["failure_count"]
            )

            if asyncio.iscoroutinefunction(recovery_func):
                await recovery_func()
            else:
                recovery_func()

            logger.info(
                "recovery_completed",
                monitor=self.name,
                component=component
            )

        except Exception as e:
            logger.error(
                "recovery_failed",
                monitor=self.name,
                component=component,
                error=str(e)
            )

    def get_health_status(self) -> dict[str, Any]:
        """Get overall health status."""
        critical_components = [
            comp for comp, status in self.health_status.items()
            if status["critical"]
        ]

        unhealthy_critical = [
            comp for comp in critical_components
            if not self.health_status[comp]["healthy"]
        ]

        return {
            "monitor": self.name,
            "overall_healthy": len(unhealthy_critical) == 0,
            "monitoring": self.monitoring,
            "components": {
                comp: {
                    "healthy": status["healthy"],
                    "last_check": status["last_check"].isoformat() if status["last_check"] else None,
                    "failure_count": status["failure_count"],
                    "critical": status["critical"]
                }
                for comp, status in self.health_status.items()
            },
            "summary": {
                "total_components": len(self.health_status),
                "healthy_components": sum(1 for s in self.health_status.values() if s["healthy"]),
                "critical_components": len(critical_components),
                "unhealthy_critical": len(unhealthy_critical)
            }
        }


# Global instances for easy access
_circuit_breakers: dict[str, CircuitBreaker] = {}
_health_monitors: dict[str, HealthMonitor] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get or create a circuit breaker instance."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_health_monitor(
    name: str,
    check_interval: int = 30
) -> HealthMonitor:
    """Get or create a health monitor instance."""
    if name not in _health_monitors:
        _health_monitors[name] = HealthMonitor(name, check_interval)
    return _health_monitors[name]


def get_all_circuit_breaker_stats() -> dict[str, Any]:
    """Get statistics for all circuit breakers."""
    return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}


def get_all_health_statuses() -> dict[str, Any]:
    """Get health status for all monitors."""
    return {name: monitor.get_health_status() for name, monitor in _health_monitors.items()}
