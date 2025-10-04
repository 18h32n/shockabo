"""Strategy port interface for hexagonal architecture.

This module defines the contract that all solving strategies must implement
to integrate with the ensemble and other system components.
"""

from abc import ABC, abstractmethod

from src.domain.models import ARCTask, ResourceUsage, StrategyOutput


class StrategyPort(ABC):
    """Abstract interface for ARC solving strategies.

    All solving strategies (TTT, Program Synthesis, Evolution, Imitation Learning)
    must implement this interface to ensure standardized integration with the
    ensemble and monitoring systems.

    Example:
        class ProgramSynthesisAdapter(StrategyPort):
            async def solve_task(self, task: ARCTask) -> StrategyOutput:
                programs = await self.evolution_engine.evolve(task)
                best = max(programs, key=lambda p: p.fitness)
                return StrategyOutput(
                    strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                    predicted_output=best.execute(task),
                    confidence_score=self._calculate_confidence(best)
                )
    """

    @abstractmethod
    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """Solve an ARC task and return standardized output.

        Args:
            task: The ARC task to solve

        Returns:
            StrategyOutput with prediction, confidence, and metadata

        Raises:
            TimeoutError: If task exceeds configured time limit
            ResourceExhaustedError: If resource budget exceeded
        """
        pass

    @abstractmethod
    def get_confidence_estimate(self, task: ARCTask) -> float:
        """Estimate confidence before full execution.

        Provides a quick confidence estimate without running the full strategy.
        Used by the router to decide which strategies to activate.

        Args:
            task: The ARC task to evaluate

        Returns:
            Estimated confidence score (0.0-1.0)

        Performance:
            Should complete in <100ms for routing decisions
        """
        pass

    @abstractmethod
    def get_resource_estimate(self, task: ARCTask) -> ResourceUsage:
        """Estimate resource requirements for this task.

        Provides estimates for CPU time, memory, GPU usage, and API calls
        to enable resource budgeting and scheduling decisions.

        Args:
            task: The ARC task to evaluate

        Returns:
            ResourceUsage with estimated requirements

        Performance:
            Should complete in <50ms for scheduling decisions
        """
        pass
