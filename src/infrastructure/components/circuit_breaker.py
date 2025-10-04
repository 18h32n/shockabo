"""Circuit breaker pattern implementation for API failure handling."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: type[Exception] = Exception
    success_threshold: int = 2  # successes needed to close from half-open
    exclude_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list[tuple[CircuitState, datetime]] = field(default_factory=list)


class CircuitBreaker[T]:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable[..., T] | None = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_state_change = datetime.now()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats

    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        async with self._lock:
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self._last_state_change = datetime.now()
                self._stats.state_changes.append((new_state, self._last_state_change))
                logger.info(
                    f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}"
                )

    async def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from open state."""
        return (
            self._state == CircuitState.OPEN and
            datetime.now() - self._last_state_change > timedelta(seconds=self.config.recovery_timeout)
        )

    async def _handle_success(self):
        """Handle successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._stats.consecutive_successes = 0

    async def _handle_failure(self, exception: Exception):
        """Handle failed call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)

    def _should_exclude_exception(self, exception: Exception) -> bool:
        """Check if exception should be excluded from circuit breaker logic."""
        return isinstance(exception, self.config.exclude_exceptions)

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        # Check if we should attempt reset
        if await self._should_attempt_reset():
            await self._transition_to(CircuitState.HALF_OPEN)

        # Check circuit state
        if self._state == CircuitState.OPEN:
            if self.fallback:
                logger.warning(f"Circuit breaker '{self.name}' is OPEN, using fallback")
                return await self.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self.fallback) else self.fallback(*args, **kwargs)
            raise Exception(f"Circuit breaker '{self.name}' is OPEN")

        try:
            # Execute the function
            result = await func(*args, **kwargs)
            await self._handle_success()
            return result

        except Exception as e:
            if self._should_exclude_exception(e):
                raise

            if isinstance(e, self.config.expected_exception):
                await self._handle_failure(e)

            if self.fallback and self._state == CircuitState.OPEN:
                logger.warning(f"Circuit breaker '{self.name}' opened, using fallback")
                return await self.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self.fallback) else self.fallback(*args, **kwargs)
            raise

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute sync function with circuit breaker protection."""
        # Run async version in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def async_wrapper():
            return func(*args, **kwargs)

        return loop.run_until_complete(self.call_async(async_wrapper, *args, **kwargs))

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for applying circuit breaker to functions."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper

    def reset(self):
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0
        self._last_state_change = datetime.now()
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "failure_rate": self._stats.failed_calls / max(1, self._stats.total_calls),
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "last_failure_time": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            }
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def create_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable[..., Any] | None = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        breaker = CircuitBreaker(name, config, fallback)
        self.register(breaker)
        return breaker

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
