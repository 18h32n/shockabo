"""Timing coordination port for cross-strategy synchronization.

This module defines the interface for coordinating execution timing across
multiple strategies in the ensemble system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TerminationReason(Enum):
    """Reasons for strategy termination."""

    TIMEOUT = "timeout"
    SUCCESS_SIGNAL = "success_signal"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"


@dataclass
class ResourceBudget:
    """Resource budget for strategy execution."""

    max_cpu_seconds: float
    max_memory_mb: float
    max_gpu_memory_mb: float | None = None
    max_api_calls: dict[str, int] | None = None
    max_tokens: int | None = None


@dataclass
class ProgressReport:
    """Progress report from a running strategy."""

    strategy_id: str
    current_progress: float
    estimated_time_remaining_ms: int
    current_best_confidence: float
    timestamp: datetime


class TimingCoordinator(ABC):
    """Abstract interface for timing coordination across strategies.

    Enables parallel execution of multiple strategies with shared time budgets,
    early termination signals, and resource management.

    Performance:
        - Termination latency: <500ms from signal to stop
        - Progress reporting: <50ms per update
        - Coordination overhead: <100ms total

    Example:
        class StrategyExecutor:
            async def execute(self, task: ARCTask):
                coordinator = get_timing_coordinator()

                # Register with coordinator
                await coordinator.register_strategy("program_synthesis", timeout_ms=300000)

                # Check for early termination
                while not await coordinator.should_terminate("program_synthesis"):
                    # Do work
                    await coordinator.report_progress("program_synthesis", progress)

                # Signal completion
                await coordinator.signal_success("program_synthesis", confidence=0.95)
    """

    @abstractmethod
    async def register_strategy(
        self,
        strategy_id: str,
        timeout_ms: int,
        resource_budget: ResourceBudget | None = None,
    ) -> None:
        """Register a strategy for coordinated execution.

        Args:
            strategy_id: Unique identifier for this strategy instance
            timeout_ms: Maximum execution time in milliseconds
            resource_budget: Optional resource limits for this strategy

        Raises:
            ValueError: If strategy_id already registered
        """
        pass

    @abstractmethod
    async def should_terminate(self, strategy_id: str) -> bool:
        """Check if strategy should terminate early.

        Args:
            strategy_id: Strategy to check

        Returns:
            True if strategy should stop execution

        Performance:
            <10ms per check for minimal overhead

        Note:
            Strategies should check this periodically (e.g., every 100ms)
            during long-running operations.
        """
        pass

    @abstractmethod
    async def signal_success(
        self, strategy_id: str, confidence: float, metadata: dict[str, object] | None = None
    ) -> None:
        """Signal that strategy found a high-confidence solution.

        Args:
            strategy_id: Strategy reporting success
            confidence: Confidence score of the solution (0.0-1.0)
            metadata: Optional additional information

        Performance:
            <100ms to propagate signal to all other strategies

        Note:
            This triggers early termination of lower-priority strategies
            if the confidence exceeds configured thresholds.
        """
        pass

    @abstractmethod
    async def report_progress(
        self,
        strategy_id: str,
        progress: float,
        estimated_time_remaining_ms: int,
        current_best_confidence: float,
    ) -> None:
        """Report execution progress for monitoring.

        Args:
            strategy_id: Strategy reporting progress
            progress: Completion percentage (0.0-1.0)
            estimated_time_remaining_ms: Estimated milliseconds until completion
            current_best_confidence: Best confidence score so far

        Performance:
            <50ms per update for real-time monitoring

        Note:
            Strategies should report progress every 1-2 seconds for
            responsive monitoring and dashboard updates.
        """
        pass

    @abstractmethod
    async def request_resource_extension(
        self, strategy_id: str, resource_type: str, additional_amount: float
    ) -> bool:
        """Request additional resource allocation.

        Args:
            strategy_id: Strategy requesting extension
            resource_type: Type of resource (cpu_seconds, memory_mb, etc.)
            additional_amount: Amount of additional resource needed

        Returns:
            True if extension granted, False if denied

        Performance:
            <50ms for resource allocation decision

        Note:
            Extensions may be denied if total budget is exhausted or
            other strategies have higher priority.
        """
        pass

    @abstractmethod
    async def get_termination_reason(self, strategy_id: str) -> TerminationReason | None:
        """Get the reason why strategy should terminate.

        Args:
            strategy_id: Strategy to query

        Returns:
            TerminationReason if termination requested, None otherwise

        Performance:
            <5ms per check
        """
        pass

    @abstractmethod
    async def unregister_strategy(
        self, strategy_id: str, reason: TerminationReason
    ) -> None:
        """Unregister strategy after completion or termination.

        Args:
            strategy_id: Strategy to unregister
            reason: Why strategy is terminating

        Note:
            Strategies must call this to release coordination resources.
        """
        pass
