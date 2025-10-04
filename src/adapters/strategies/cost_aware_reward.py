
import numpy as np


class CostAwareRewardCalculator:
    """
    Calculates cost-adjusted rewards for strategy execution.

    Balances fitness improvement against API costs to optimize for
    both solution quality and resource efficiency.

    Formula: reward_final = fitness / (1 + cost_weight * cost_normalized)
    """

    def __init__(
        self,
        cost_weight: float = 0.3,
        track_costs: bool = True
    ):
        """
        Initialize cost-aware reward calculator.

        Args:
            cost_weight: Weight for cost penalty (0.0-1.0, default: 0.3)
            track_costs: Whether to track cost statistics (default: True)

        Raises:
            ValueError: If cost_weight not in [0.0, 1.0]
        """
        if not (0.0 <= cost_weight <= 1.0):
            raise ValueError(f"cost_weight must be in [0.0, 1.0], got {cost_weight}")

        self.cost_weight = cost_weight
        self.track_costs = track_costs

        self._cost_min = float('inf')
        self._cost_max = float('-inf')

        self.strategy_costs: dict[str, float] = {}
        self.strategy_api_calls: dict[str, int] = {}

    def calculate_reward(
        self,
        fitness: float,
        cost: float,
        strategy_id: str | None = None
    ) -> float:
        """
        Calculate cost-adjusted reward.

        Args:
            fitness: Fitness improvement (0.0-1.0)
            cost: API cost in USD
            strategy_id: Optional strategy ID for cost tracking

        Returns:
            Cost-adjusted reward value

        Raises:
            ValueError: If fitness not in [0.0, 1.0] or cost is negative
        """
        if not (0.0 <= fitness <= 1.0):
            raise ValueError(f"fitness must be in [0.0, 1.0], got {fitness}")
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")

        if self.track_costs and strategy_id:
            self._update_cost_stats(strategy_id, cost)

        if cost == 0:
            return fitness

        cost_normalized = self._normalize_cost(cost)

        reward = fitness / (1.0 + self.cost_weight * cost_normalized)

        return float(np.clip(reward, 0.0, 1.0))

    def _normalize_cost(self, cost: float) -> float:
        """
        Normalize cost to 0.0-1.0 scale using min-max normalization.

        Args:
            cost: Raw cost value

        Returns:
            Normalized cost (0.0-1.0)
        """
        if self._cost_max == float('-inf') or self._cost_min == float('inf'):
            return 0.0 if cost == 0 else 0.5

        if self._cost_max == self._cost_min:
            return 0.0 if cost == 0 else 0.5

        normalized = (cost - self._cost_min) / (self._cost_max - self._cost_min)
        return float(np.clip(normalized, 0.0, 1.0))

    def _update_cost_stats(self, strategy_id: str, cost: float) -> None:
        """
        Update cost statistics for min-max normalization.

        Args:
            strategy_id: Strategy that incurred the cost
            cost: Cost value
        """
        self._cost_min = min(self._cost_min, cost) if cost > 0 else self._cost_min
        self._cost_max = max(self._cost_max, cost)

        if strategy_id not in self.strategy_costs:
            self.strategy_costs[strategy_id] = 0.0
            self.strategy_api_calls[strategy_id] = 0

        self.strategy_costs[strategy_id] += cost
        self.strategy_api_calls[strategy_id] += 1

    def get_strategy_costs(self) -> dict[str, dict[str, float]]:
        """
        Get cost statistics for all strategies.

        Returns:
            Dictionary mapping strategy_id to cost stats:
                - total_cost: Cumulative cost
                - api_calls: Number of API calls
                - avg_cost_per_call: Average cost per API call
        """
        stats = {}
        for strategy_id in self.strategy_costs:
            total_cost = self.strategy_costs[strategy_id]
            api_calls = self.strategy_api_calls[strategy_id]
            stats[strategy_id] = {
                'total_cost': total_cost,
                'api_calls': api_calls,
                'avg_cost_per_call': total_cost / api_calls if api_calls > 0 else 0.0
            }
        return stats

    def get_cost_efficiency(self, strategy_id: str, total_fitness: float) -> float | None:
        """
        Calculate cost efficiency for a strategy.

        Cost efficiency = total_fitness / total_cost

        Args:
            strategy_id: Strategy to analyze
            total_fitness: Total fitness achieved by strategy

        Returns:
            Cost efficiency score (higher is better), or None if strategy not found
        """
        if strategy_id not in self.strategy_costs:
            return None

        total_cost = self.strategy_costs[strategy_id]
        if total_cost == 0:
            return float('inf')

        return total_fitness / total_cost

    def set_cost_weight(self, cost_weight: float) -> None:
        """
        Update cost weight parameter.

        Args:
            cost_weight: New cost weight (0.0-1.0)

        Raises:
            ValueError: If cost_weight not in [0.0, 1.0]
        """
        if not (0.0 <= cost_weight <= 1.0):
            raise ValueError(f"cost_weight must be in [0.0, 1.0], got {cost_weight}")
        self.cost_weight = cost_weight

    def reset_cost_stats(self) -> None:
        """Reset all cost statistics."""
        self._cost_min = float('inf')
        self._cost_max = float('-inf')
        self.strategy_costs.clear()
        self.strategy_api_calls.clear()


class ModelCostTracker:
    """
    Tracks API costs for different LLM models.

    Integrates with SmartModelRouter to track costs per model type.
    """

    MODEL_COSTS = {
        'gpt-5': {'input': 0.03, 'output': 0.06},  # per 1K tokens
        'gemini-pro': {'input': 0.0005, 'output': 0.0015},
        'glm-4.5': {'input': 0.001, 'output': 0.003},
        'qwen-2.5': {'input': 0.0008, 'output': 0.002},
        'local': {'input': 0.0, 'output': 0.0}
    }

    def __init__(self):
        """Initialize model cost tracker."""
        self.model_usage: dict[str, dict[str, int]] = {}

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a model API call.

        Args:
            model_name: Model identifier (e.g., 'gpt-5', 'gemini-pro')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD

        Raises:
            ValueError: If model_name not recognized
        """
        if model_name not in self.MODEL_COSTS:
            raise ValueError(f"Unknown model: {model_name}. Known models: {list(self.MODEL_COSTS.keys())}")

        costs = self.MODEL_COSTS[model_name]

        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']

        total_cost = input_cost + output_cost

        if model_name not in self.model_usage:
            self.model_usage[model_name] = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}

        self.model_usage[model_name]['input_tokens'] += input_tokens
        self.model_usage[model_name]['output_tokens'] += output_tokens
        self.model_usage[model_name]['calls'] += 1

        return total_cost

    def get_model_usage_stats(self) -> dict[str, dict[str, float]]:
        """
        Get usage statistics for all models.

        Returns:
            Dictionary mapping model_name to usage stats:
                - total_input_tokens: Total input tokens
                - total_output_tokens: Total output tokens
                - total_calls: Number of API calls
                - estimated_cost: Estimated total cost in USD
        """
        stats = {}
        for model_name, usage in self.model_usage.items():
            costs = self.MODEL_COSTS[model_name]
            input_cost = (usage['input_tokens'] / 1000) * costs['input']
            output_cost = (usage['output_tokens'] / 1000) * costs['output']

            stats[model_name] = {
                'total_input_tokens': usage['input_tokens'],
                'total_output_tokens': usage['output_tokens'],
                'total_calls': usage['calls'],
                'estimated_cost': input_cost + output_cost
            }

        return stats

    def get_total_cost(self) -> float:
        """
        Get total estimated cost across all models.

        Returns:
            Total cost in USD
        """
        total = 0.0
        for stats in self.get_model_usage_stats().values():
            total += stats['estimated_cost']
        return total

    def reset(self) -> None:
        """Reset all usage statistics."""
        self.model_usage.clear()
