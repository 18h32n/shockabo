from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class RewardRecord:
    """Individual reward record for a strategy execution."""
    timestamp: datetime
    strategy_id: str
    fitness_improvement: float
    convergence_speed: int | None  # Programs to solution
    api_cost: float
    normalized_reward: float


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy."""
    strategy_id: str
    total_executions: int = 0
    total_fitness: float = 0.0
    total_cost: float = 0.0
    fitness_history: list[float] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)
    convergence_history: list[int | None] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)


class RewardTracker:
    """
    Tracks rewards and metrics for generation strategies.

    Provides per-strategy metrics including fitness improvement, convergence speed,
    and API cost efficiency. Supports reward normalization across strategies.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize reward tracker.

        Args:
            max_history: Maximum number of records to keep per strategy (default: 10000)
        """
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.max_history = max_history
        self.metrics: dict[str, StrategyMetrics] = defaultdict(
            lambda: StrategyMetrics(strategy_id="")
        )
        self.records: list[RewardRecord] = []

        # Global normalization statistics
        self._fitness_min = float('inf')
        self._fitness_max = float('-inf')
        self._cost_min = float('inf')
        self._cost_max = float('-inf')

    def record_reward(
        self,
        strategy_id: str,
        fitness_improvement: float,
        convergence_speed: int | None = None,
        api_cost: float = 0.0
    ) -> float:
        """
        Record a reward for a strategy execution.

        Primary reward signal: fitness improvement (0.0-1.0 scale)
        Secondary signals: convergence speed, API cost

        Args:
            strategy_id: Strategy that was executed
            fitness_improvement: Fitness improvement achieved
                (0.0-1.0, where 1.0 = perfect solution)
            convergence_speed: Number of programs evaluated to reach solution (lower is better)
            api_cost: API cost incurred in USD

        Returns:
            Normalized reward value (0.0-1.0)

        Raises:
            ValueError: If fitness_improvement is not in [0.0, 1.0] or is NaN/Inf
        """
        # Validate inputs
        if not np.isfinite(fitness_improvement):
            raise ValueError(f"fitness_improvement must be finite, got {fitness_improvement}")
        if not (0.0 <= fitness_improvement <= 1.0):
            raise ValueError(
                f"fitness_improvement must be in [0.0, 1.0], got {fitness_improvement}"
            )
        if api_cost < 0:
            raise ValueError(f"api_cost must be non-negative, got {api_cost}")
        if convergence_speed is not None and convergence_speed < 0:
            raise ValueError(f"convergence_speed must be non-negative, got {convergence_speed}")

        # Update global min/max for normalization
        self._fitness_min = min(self._fitness_min, fitness_improvement)
        self._fitness_max = max(self._fitness_max, fitness_improvement)
        if api_cost > 0:
            self._cost_min = min(self._cost_min, api_cost)
            self._cost_max = max(self._cost_max, api_cost)

        # Calculate normalized reward
        normalized_reward = self._normalize_reward(fitness_improvement)

        # Update strategy metrics
        metrics = self.metrics[strategy_id]
        metrics.strategy_id = strategy_id
        metrics.total_executions += 1
        metrics.total_fitness += fitness_improvement
        metrics.total_cost += api_cost
        metrics.fitness_history.append(fitness_improvement)
        metrics.cost_history.append(api_cost)
        metrics.convergence_history.append(convergence_speed)
        metrics.reward_history.append(normalized_reward)

        # Trim history if needed
        if len(metrics.fitness_history) > self.max_history:
            metrics.fitness_history.pop(0)
            metrics.cost_history.pop(0)
            metrics.convergence_history.pop(0)
            metrics.reward_history.pop(0)

        # Create record
        record = RewardRecord(
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            fitness_improvement=fitness_improvement,
            convergence_speed=convergence_speed,
            api_cost=api_cost,
            normalized_reward=normalized_reward
        )
        self.records.append(record)

        # Trim records if needed
        if len(self.records) > self.max_history:
            self.records.pop(0)

        return normalized_reward

    def _normalize_reward(self, fitness_improvement: float) -> float:
        """
        Normalize reward to 0.0-1.0 scale.

        Uses min-max normalization based on observed fitness range.
        If only one unique value seen, returns the raw fitness.

        Args:
            fitness_improvement: Raw fitness value

        Returns:
            Normalized reward (0.0-1.0)
        """
        if self._fitness_max == self._fitness_min:
            # Only one unique value seen - return as-is
            return fitness_improvement

        # Min-max normalization
        normalized = (fitness_improvement - self._fitness_min) / (
            self._fitness_max - self._fitness_min
        )
        return np.clip(normalized, 0.0, 1.0)

    def get_strategy_metrics(self, strategy_id: str) -> StrategyMetrics | None:
        """
        Get metrics for a specific strategy.

        Args:
            strategy_id: Strategy to retrieve

        Returns:
            StrategyMetrics or None if strategy not found
        """
        return self.metrics.get(strategy_id)

    def get_all_metrics(self) -> dict[str, StrategyMetrics]:
        """
        Get metrics for all strategies.

        Returns:
            Dictionary mapping strategy_id to StrategyMetrics
        """
        return dict(self.metrics)

    def get_cost_efficiency(self, strategy_id: str) -> float:
        """
        Calculate cost efficiency for a strategy.

        Cost efficiency = total fitness / (1 + total cost)
        Higher values indicate better efficiency.

        Args:
            strategy_id: Strategy to analyze

        Returns:
            Cost efficiency score (higher is better)

        Raises:
            ValueError: If strategy not found
        """
        if strategy_id not in self.metrics:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        metrics = self.metrics[strategy_id]
        return metrics.total_fitness / (1.0 + metrics.total_cost)

    def get_average_convergence(self, strategy_id: str) -> float | None:
        """
        Calculate average convergence speed for a strategy.

        Args:
            strategy_id: Strategy to analyze

        Returns:
            Average programs to solution, or None if no convergence data

        Raises:
            ValueError: If strategy not found
        """
        if strategy_id not in self.metrics:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        metrics = self.metrics[strategy_id]
        convergence_values = [c for c in metrics.convergence_history if c is not None]

        if not convergence_values:
            return None

        return float(np.mean(convergence_values))

    def get_reward_statistics(self, strategy_id: str) -> dict[str, float]:
        """
        Calculate reward statistics for a strategy.

        Args:
            strategy_id: Strategy to analyze

        Returns:
            Dictionary with mean, std, min, max reward values

        Raises:
            ValueError: If strategy not found
        """
        if strategy_id not in self.metrics:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        metrics = self.metrics[strategy_id]

        if not metrics.reward_history:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }

        rewards = np.array(metrics.reward_history)
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        }

    def export_visualization_data(self) -> dict[str, dict[str, list]]:
        """
        Export data for visualization (charts, dashboards).

        Returns:
            Dictionary mapping strategy_id to time-series data:
                - fitness_history: List of fitness values
                - cost_history: List of cost values
                - reward_history: List of normalized rewards
                - convergence_history: List of convergence speeds
        """
        export_data = {}

        for strategy_id, metrics in self.metrics.items():
            export_data[strategy_id] = {
                'fitness_history': metrics.fitness_history.copy(),
                'cost_history': metrics.cost_history.copy(),
                'reward_history': metrics.reward_history.copy(),
                'convergence_history': metrics.convergence_history.copy(),
                'total_executions': metrics.total_executions,
                'avg_fitness': (
                    metrics.total_fitness / metrics.total_executions
                    if metrics.total_executions > 0 else 0.0
                ),
                'avg_cost': (
                    metrics.total_cost / metrics.total_executions
                    if metrics.total_executions > 0 else 0.0
                ),
                'cost_efficiency': self.get_cost_efficiency(strategy_id)
            }

        return export_data

    def reset_strategy(self, strategy_id: str) -> None:
        """
        Reset metrics for a strategy.

        Args:
            strategy_id: Strategy to reset
        """
        if strategy_id in self.metrics:
            del self.metrics[strategy_id]

    def reset_all(self) -> None:
        """Reset all metrics and records."""
        self.metrics.clear()
        self.records.clear()
        self._fitness_min = float('inf')
        self._fitness_max = float('-inf')
        self._cost_min = float('inf')
        self._cost_max = float('-inf')
