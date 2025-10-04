from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenerationStrategy:
    """Represents a generation strategy for bandit selection."""
    strategy_id: str
    success_count: int = 0
    failure_count: int = 0
    total_reward: float = 0.0
    selection_count: int = 0
    consecutive_failures: int = 0
    is_circuit_broken: bool = False


@dataclass
class BanditState:
    """Thompson Sampling state for all strategies."""
    strategies: dict[str, GenerationStrategy] = field(default_factory=dict)
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    warmup_selections: int = 50
    success_threshold: float = 0.5


class BanditController:
    """
    Multi-Armed Bandit controller using Thompson Sampling for strategy selection.

    Implements Thompson Sampling algorithm with Beta distributions to balance
    exploration and exploitation across generation strategies.

    References:
        - Chapelle & Li (2011): "An Empirical Evaluation of Thompson Sampling"
    """

    def __init__(
        self,
        strategies: list[str],
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        warmup_selections: int = 50,
        success_threshold: float = 0.5,
        random_seed: int | None = None
    ):
        """
        Initialize Thompson Sampling controller.

        Args:
            strategies: List of strategy IDs to manage
            alpha_prior: Success prior for Beta distribution (default: 1.0)
            beta_prior: Failure prior for Beta distribution (default: 1.0)
            warmup_selections: Number of uniform selections per strategy before
                Thompson Sampling (default: 50)
            success_threshold: Reward threshold for success vs failure (default: 0.5)
            random_seed: Random seed for reproducibility (optional)
        """
        if not strategies:
            raise ValueError("Must provide at least one strategy")
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Alpha and beta priors must be positive")
        if warmup_selections < 0:
            raise ValueError("Warmup selections must be non-negative")

        self.state = BanditState(
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            warmup_selections=warmup_selections,
            success_threshold=success_threshold
        )

        # Initialize strategies
        for strategy_id in strategies:
            self.state.strategies[strategy_id] = GenerationStrategy(strategy_id=strategy_id)

        # Random number generator
        self.rng = np.random.default_rng(random_seed)

    def select_strategy(self, task_features: dict[str, float] | None = None) -> str:
        """
        Select best strategy using Thompson Sampling.

        During warmup period (first warmup_selections per strategy), uses uniform
        random selection. After warmup, samples from Beta posterior distributions
        and selects strategy with highest sample.

        Args:
            task_features: Optional task features for contextual bandit
                (not used in base implementation)

        Returns:
            Strategy ID to use

        Raises:
            RuntimeError: If all strategies are circuit-broken
        """
        # Filter out circuit-broken strategies
        available_strategies = [
            s for s in self.state.strategies.values()
            if not s.is_circuit_broken
        ]

        if not available_strategies:
            raise RuntimeError("All strategies are circuit-broken - cannot select")

        # Warmup period: uniform random selection
        min_selections = min(s.selection_count for s in available_strategies)
        if min_selections < self.state.warmup_selections:
            # Select strategy with fewest selections
            least_selected = min(available_strategies, key=lambda s: s.selection_count)
            least_selected.selection_count += 1
            return least_selected.strategy_id

        # Thompson Sampling: sample from Beta distributions
        samples = {}
        for strategy in available_strategies:
            alpha = self.state.alpha_prior + strategy.success_count
            beta = self.state.beta_prior + strategy.failure_count
            samples[strategy.strategy_id] = self.rng.beta(alpha, beta)

        # Select strategy with highest sample
        selected_id = max(samples, key=samples.get)
        self.state.strategies[selected_id].selection_count += 1

        return selected_id

    def update_reward(
        self,
        strategy_id: str,
        reward: float,
        cost: float = 0.0,
        is_failure: bool = False
    ) -> None:
        """
        Update strategy statistics with reward signal.

        Args:
            strategy_id: Strategy to update
            reward: Reward value (0.0-1.0 for success, -0.5 for failure)
            cost: API cost (not used in base implementation, available for cost-aware rewards)
            is_failure: Whether this was a failure (timeout, error, etc.)

        Raises:
            ValueError: If strategy_id not recognized or reward is NaN/Inf
        """
        if strategy_id not in self.state.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        if not np.isfinite(reward):
            raise ValueError(f"Reward must be finite, got {reward}")

        strategy = self.state.strategies[strategy_id]

        # Update reward total
        strategy.total_reward += reward

        # Update success/failure counts
        if is_failure or reward <= self.state.success_threshold:
            strategy.failure_count += 1
            strategy.consecutive_failures += 1
        else:
            strategy.success_count += 1
            strategy.consecutive_failures = 0

        # Circuit breaker: disable after 3 consecutive failures
        if strategy.consecutive_failures >= 3:
            strategy.is_circuit_broken = True

    def reset_circuit_breaker(self, strategy_id: str) -> None:
        """
        Reset circuit breaker for a strategy (for gradual re-enabling).

        Args:
            strategy_id: Strategy to reset

        Raises:
            ValueError: If strategy_id not recognized
        """
        if strategy_id not in self.state.strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        strategy = self.state.strategies[strategy_id]
        strategy.is_circuit_broken = False
        strategy.consecutive_failures = 0

    def get_strategy_stats(self) -> dict[str, dict[str, float]]:
        """
        Get current statistics for all strategies.

        Returns:
            Dictionary mapping strategy_id to stats dict containing:
                - success_count: Number of successes
                - failure_count: Number of failures
                - total_reward: Cumulative reward
                - selection_count: Number of times selected
                - avg_reward: Average reward per selection
                - win_rate: Success rate (successes / total)
                - is_circuit_broken: Whether circuit breaker is active
        """
        stats = {}
        for strategy_id, strategy in self.state.strategies.items():
            total_outcomes = strategy.success_count + strategy.failure_count
            stats[strategy_id] = {
                'success_count': strategy.success_count,
                'failure_count': strategy.failure_count,
                'total_reward': strategy.total_reward,
                'selection_count': strategy.selection_count,
                'avg_reward': (
                    strategy.total_reward / strategy.selection_count
                    if strategy.selection_count > 0 else 0.0
                ),
                'win_rate': (
                    strategy.success_count / total_outcomes
                    if total_outcomes > 0 else 0.0
                ),
                'is_circuit_broken': strategy.is_circuit_broken
            }
        return stats
