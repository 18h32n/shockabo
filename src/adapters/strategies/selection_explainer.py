import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SelectionDecision:
    """Record of a single strategy selection decision."""
    timestamp: datetime
    selected_strategy: str
    strategy_scores: dict[str, float]
    task_features: dict[str, float] | None
    exploration_vs_exploitation: str  # 'warmup', 'exploration', 'exploitation'
    selection_reason: str
    available_strategies: list[str]
    circuit_broken_strategies: list[str]


@dataclass
class StrategyStatistics:
    """Statistics for a strategy's selection history."""
    strategy_id: str
    selection_count: int = 0
    win_count: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    win_rate: float = 0.0


class SelectionExplainer:
    """
    Provides interpretable explanations for strategy selection decisions.

    Logs selection rationale, tracks statistics, and exports decision traces
    for debugging and analysis.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize selection explainer.

        Args:
            max_history: Maximum number of decision records to keep (default: 1000)
        """
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.max_history = max_history
        self.decisions: list[SelectionDecision] = []
        self.strategy_stats: dict[str, StrategyStatistics] = {}

    def log_selection(
        self,
        selected_strategy: str,
        strategy_scores: dict[str, float],
        task_features: dict[str, float] | None = None,
        exploration_type: str = 'exploitation',
        selection_reason: str = '',
        available_strategies: list[str] | None = None,
        circuit_broken_strategies: list[str] | None = None
    ) -> None:
        """
        Log a strategy selection decision.

        Args:
            selected_strategy: Strategy that was selected
            strategy_scores: Thompson Sampling scores for all strategies
            task_features: Task feature vector (if contextual bandit)
            exploration_type: 'warmup', 'exploration', or 'exploitation'
            selection_reason: Human-readable explanation of selection
            available_strategies: List of available strategies
            circuit_broken_strategies: List of circuit-broken strategies
        """
        decision = SelectionDecision(
            timestamp=datetime.now(),
            selected_strategy=selected_strategy,
            strategy_scores=strategy_scores.copy(),
            task_features=task_features.copy() if task_features else None,
            exploration_vs_exploitation=exploration_type,
            selection_reason=selection_reason,
            available_strategies=available_strategies or list(strategy_scores.keys()),
            circuit_broken_strategies=circuit_broken_strategies or []
        )

        self.decisions.append(decision)

        if len(self.decisions) > self.max_history:
            self.decisions.pop(0)

        if selected_strategy not in self.strategy_stats:
            self.strategy_stats[selected_strategy] = StrategyStatistics(
                strategy_id=selected_strategy
            )

        self.strategy_stats[selected_strategy].selection_count += 1

    def log_outcome(self, strategy_id: str, reward: float, is_win: bool) -> None:
        """
        Log the outcome of a strategy execution.

        Args:
            strategy_id: Strategy that was executed
            reward: Reward received
            is_win: Whether execution was successful
        """
        if strategy_id not in self.strategy_stats:
            self.strategy_stats[strategy_id] = StrategyStatistics(strategy_id=strategy_id)

        stats = self.strategy_stats[strategy_id]
        stats.total_reward += reward
        if is_win:
            stats.win_count += 1

        if stats.selection_count > 0:
            stats.avg_reward = stats.total_reward / stats.selection_count
            stats.win_rate = stats.win_count / stats.selection_count

    def get_selection_frequency(self) -> dict[str, float]:
        """
        Get selection frequency for all strategies.

        Returns:
            Dictionary mapping strategy_id to selection frequency (0.0-1.0)
        """
        total_selections = sum(
            stats.selection_count for stats in self.strategy_stats.values()
        )

        if total_selections == 0:
            return {}

        return {
            strategy_id: stats.selection_count / total_selections
            for strategy_id, stats in self.strategy_stats.items()
        }

    def get_win_rates(self) -> dict[str, float]:
        """
        Get win rate for all strategies.

        Returns:
            Dictionary mapping strategy_id to win rate (0.0-1.0)
        """
        return {
            strategy_id: stats.win_rate
            for strategy_id, stats in self.strategy_stats.items()
        }

    def get_average_rewards(self) -> dict[str, float]:
        """
        Get average reward for all strategies.

        Returns:
            Dictionary mapping strategy_id to average reward
        """
        return {
            strategy_id: stats.avg_reward
            for strategy_id, stats in self.strategy_stats.items()
        }

    def get_recent_decisions(self, n: int = 10) -> list[SelectionDecision]:
        """
        Get N most recent selection decisions.

        Args:
            n: Number of recent decisions to return

        Returns:
            List of SelectionDecision records
        """
        return self.decisions[-n:]

    def export_decision_trace(self, filepath: str) -> None:
        """
        Export decision trace to JSON file for debugging.

        Args:
            filepath: Path to output JSON file
        """
        trace_data = []

        for decision in self.decisions:
            trace_data.append({
                'timestamp': decision.timestamp.isoformat(),
                'selected_strategy': decision.selected_strategy,
                'strategy_scores': decision.strategy_scores,
                'task_features': decision.task_features,
                'exploration_vs_exploitation': decision.exploration_vs_exploitation,
                'selection_reason': decision.selection_reason,
                'available_strategies': decision.available_strategies,
                'circuit_broken_strategies': decision.circuit_broken_strategies
            })

        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)

    def get_strategy_performance_comparison(self) -> dict[str, dict[str, float]]:
        """
        Get comprehensive performance comparison for all strategies.

        Returns:
            Dictionary mapping strategy_id to performance metrics:
                - selection_count: Number of times selected
                - selection_frequency: Selection frequency (0.0-1.0)
                - win_count: Number of successful executions
                - win_rate: Success rate (0.0-1.0)
                - avg_reward: Average reward per selection
        """
        selection_freq = self.get_selection_frequency()

        comparison = {}
        for strategy_id, stats in self.strategy_stats.items():
            comparison[strategy_id] = {
                'selection_count': stats.selection_count,
                'selection_frequency': selection_freq.get(strategy_id, 0.0),
                'win_count': stats.win_count,
                'win_rate': stats.win_rate,
                'avg_reward': stats.avg_reward
            }

        return comparison

    def get_exploration_vs_exploitation_ratio(self) -> dict[str, float]:
        """
        Calculate ratio of exploration vs exploitation decisions.

        Returns:
            Dictionary with counts for 'warmup', 'exploration', 'exploitation'
        """
        counts = {'warmup': 0, 'exploration': 0, 'exploitation': 0}

        for decision in self.decisions:
            decision_type = decision.exploration_vs_exploitation
            if decision_type in counts:
                counts[decision_type] += 1

        total = len(self.decisions)
        if total == 0:
            return {'warmup': 0.0, 'exploration': 0.0, 'exploitation': 0.0}

        return {
            'warmup': counts['warmup'] / total,
            'exploration': counts['exploration'] / total,
            'exploitation': counts['exploitation'] / total
        }

    def get_strategy_probabilities_over_time(
        self,
        strategy_id: str,
        window_size: int = 50
    ) -> list[float]:
        """
        Get strategy selection probability over time (rolling window).

        Args:
            strategy_id: Strategy to track
            window_size: Size of rolling window

        Returns:
            List of selection probabilities over time
        """
        probabilities = []

        for i in range(len(self.decisions)):
            start = max(0, i - window_size + 1)
            window = self.decisions[start:i+1]

            selections = sum(
                1 for d in window if d.selected_strategy == strategy_id
            )
            probability = selections / len(window) if window else 0.0
            probabilities.append(probability)

        return probabilities

    def reset(self) -> None:
        """Reset all decision history and statistics."""
        self.decisions.clear()
        self.strategy_stats.clear()
