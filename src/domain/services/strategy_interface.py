"""
Unified strategy interface for ARC task solving strategies.

This module defines the common interface that all solving strategies must implement
to ensure compatibility with the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from src.domain.evaluation_models import StrategyType
from src.domain.models import ARCTask
from src.domain.services.evaluation_service import EvaluationResult


@runtime_checkable
class StrategyInterface(Protocol):
    """Protocol defining the interface for ARC solving strategies."""

    async def solve_task(
        self,
        task: ARCTask,
        max_attempts: int = 2,
        experiment_name: str | None = None
    ) -> EvaluationResult:
        """
        Solve an ARC task using the strategy.

        Args:
            task: The ARC task to solve
            max_attempts: Maximum number of solution attempts (default 2)
            experiment_name: Optional experiment name for tracking

        Returns:
            EvaluationResult with predictions and metadata
        """
        ...

    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get information about the strategy.

        Returns:
            Dictionary containing strategy name, type, version, and capabilities
        """
        ...


class BaseStrategy(ABC):
    """Abstract base class for ARC solving strategies."""

    def __init__(self, evaluation_service):
        """
        Initialize the strategy.

        Args:
            evaluation_service: The evaluation service for result formatting
        """
        self.evaluation_service = evaluation_service

    @abstractmethod
    async def solve_task(
        self,
        task: ARCTask,
        max_attempts: int = 2,
        experiment_name: str | None = None
    ) -> EvaluationResult:
        """
        Solve an ARC task using the strategy.

        Args:
            task: The ARC task to solve
            max_attempts: Maximum number of solution attempts (default 2)
            experiment_name: Optional experiment name for tracking

        Returns:
            EvaluationResult with predictions and metadata
        """
        pass

    @abstractmethod
    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get information about the strategy.

        Returns:
            Dictionary containing strategy name, type, version, and capabilities
        """
        pass

    @abstractmethod
    def get_strategy_type(self) -> StrategyType:
        """
        Get the strategy type enum.

        Returns:
            StrategyType enum value
        """
        pass

    def create_submission_format(
        self,
        task_id: str,
        predictions: list[list[list[int]]],
        confidence_scores: list[float],
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create unified submission format.

        Args:
            task_id: The task ID
            predictions: List of predicted outputs (up to 2)
            confidence_scores: Confidence scores for each prediction
            metadata: Optional metadata about the solution

        Returns:
            Dictionary in unified submission format
        """
        if len(predictions) != len(confidence_scores):
            raise ValueError("Number of predictions must match number of confidence scores")

        if len(predictions) > 2:
            raise ValueError("Maximum 2 predictions allowed per task")

        submission = {
            "task_id": task_id,
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "strategy": self.get_strategy_type().value,
            "metadata": metadata or {}
        }

        # Add strategy info to metadata
        submission["metadata"]["strategy_info"] = self.get_strategy_info()

        return submission
