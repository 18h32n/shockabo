import pytest

from src.adapters.strategies.reward_tracker import RewardTracker, StrategyMetrics


class TestRewardTracker:
    """Unit tests for RewardTracker reward tracking implementation."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        tracker = RewardTracker(max_history=5000)
        assert tracker.max_history == 5000
        assert len(tracker.metrics) == 0
        assert len(tracker.records) == 0

    def test_initialization_invalid_history(self):
        """Test initialization fails with invalid max_history."""
        with pytest.raises(ValueError, match="max_history must be positive"):
            RewardTracker(max_history=0)

        with pytest.raises(ValueError, match="max_history must be positive"):
            RewardTracker(max_history=-5)

    def test_record_reward_basic(self):
        """Test basic reward recording."""
        tracker = RewardTracker()

        reward = tracker.record_reward(
            strategy_id='strategy_a',
            fitness_improvement=0.8,
            convergence_speed=100,
            api_cost=0.05
        )

        assert reward == 0.8  # First record, no normalization needed
        assert len(tracker.records) == 1

        metrics = tracker.get_strategy_metrics('strategy_a')
        assert metrics is not None
        assert metrics.total_executions == 1
        assert metrics.total_fitness == 0.8
        assert metrics.total_cost == 0.05
        assert metrics.fitness_history == [0.8]
        assert metrics.cost_history == [0.05]
        assert metrics.convergence_history == [100]

    def test_record_reward_invalid_fitness(self):
        """Test recording fails with invalid fitness."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="fitness_improvement must be in"):
            tracker.record_reward('a', fitness_improvement=-0.1)

        with pytest.raises(ValueError, match="fitness_improvement must be in"):
            tracker.record_reward('a', fitness_improvement=1.5)

        with pytest.raises(ValueError, match="fitness_improvement must be finite"):
            tracker.record_reward('a', fitness_improvement=float('nan'))

    def test_record_reward_invalid_cost(self):
        """Test recording fails with negative cost."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="api_cost must be non-negative"):
            tracker.record_reward('a', fitness_improvement=0.5, api_cost=-0.1)

    def test_record_reward_invalid_convergence(self):
        """Test recording fails with negative convergence."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="convergence_speed must be non-negative"):
            tracker.record_reward('a', fitness_improvement=0.5, convergence_speed=-10)

    def test_record_reward_normalization(self):
        """Test reward normalization across strategies."""
        tracker = RewardTracker()

        # Record multiple rewards
        tracker.record_reward('a', fitness_improvement=0.2)
        tracker.record_reward('b', fitness_improvement=0.8)

        # Third record should be normalized
        reward = tracker.record_reward('c', fitness_improvement=0.5)

        # Should be normalized: (0.5 - 0.2) / (0.8 - 0.2) = 0.5
        assert reward == pytest.approx(0.5)

    def test_record_reward_history_trimming(self):
        """Test history trimming at max_history limit."""
        tracker = RewardTracker(max_history=5)

        # Record 10 rewards
        for i in range(10):
            tracker.record_reward('a', fitness_improvement=i / 10.0)

        metrics = tracker.get_strategy_metrics('a')
        assert len(metrics.fitness_history) == 5
        assert len(metrics.reward_history) == 5
        assert len(tracker.records) == 5

    def test_get_strategy_metrics_missing(self):
        """Test getting metrics for missing strategy."""
        tracker = RewardTracker()

        metrics = tracker.get_strategy_metrics('nonexistent')
        assert metrics is None

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5)
        tracker.record_reward('b', fitness_improvement=0.7)
        tracker.record_reward('a', fitness_improvement=0.6)

        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) == 2
        assert 'a' in all_metrics
        assert 'b' in all_metrics
        assert all_metrics['a'].total_executions == 2
        assert all_metrics['b'].total_executions == 1

    def test_get_cost_efficiency(self):
        """Test cost efficiency calculation."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.8, api_cost=0.05)
        tracker.record_reward('a', fitness_improvement=0.6, api_cost=0.03)

        efficiency = tracker.get_cost_efficiency('a')
        # efficiency = 1.4 / (1 + 0.08) = 1.4 / 1.08
        assert efficiency == pytest.approx(1.4 / 1.08)

    def test_get_cost_efficiency_unknown_strategy(self):
        """Test cost efficiency fails for unknown strategy."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.get_cost_efficiency('unknown')

    def test_get_average_convergence(self):
        """Test average convergence calculation."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5, convergence_speed=100)
        tracker.record_reward('a', fitness_improvement=0.6, convergence_speed=80)
        tracker.record_reward('a', fitness_improvement=0.7, convergence_speed=120)

        avg_convergence = tracker.get_average_convergence('a')
        assert avg_convergence == pytest.approx(100.0)

    def test_get_average_convergence_with_none_values(self):
        """Test average convergence with None values."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5, convergence_speed=100)
        tracker.record_reward('a', fitness_improvement=0.6, convergence_speed=None)
        tracker.record_reward('a', fitness_improvement=0.7, convergence_speed=80)

        avg_convergence = tracker.get_average_convergence('a')
        # Should average only non-None values: (100 + 80) / 2 = 90
        assert avg_convergence == pytest.approx(90.0)

    def test_get_average_convergence_all_none(self):
        """Test average convergence with all None values."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5, convergence_speed=None)
        tracker.record_reward('a', fitness_improvement=0.6, convergence_speed=None)

        avg_convergence = tracker.get_average_convergence('a')
        assert avg_convergence is None

    def test_get_average_convergence_unknown_strategy(self):
        """Test average convergence fails for unknown strategy."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.get_average_convergence('unknown')

    def test_get_reward_statistics(self):
        """Test reward statistics calculation."""
        tracker = RewardTracker()

        # Record rewards to create distribution
        tracker.record_reward('a', fitness_improvement=0.2)
        tracker.record_reward('a', fitness_improvement=0.4)
        tracker.record_reward('a', fitness_improvement=0.6)
        tracker.record_reward('a', fitness_improvement=0.8)

        stats = tracker.get_reward_statistics('a')

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

        # Check reasonable values
        assert stats['mean'] > 0
        assert stats['std'] >= 0
        assert stats['min'] >= 0
        assert stats['max'] <= 1.0

    def test_get_reward_statistics_no_history(self):
        """Test reward statistics with no history."""
        tracker = RewardTracker()
        tracker.metrics['a'] = StrategyMetrics(strategy_id='a')

        stats = tracker.get_reward_statistics('a')
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0

    def test_get_reward_statistics_unknown_strategy(self):
        """Test reward statistics fails for unknown strategy."""
        tracker = RewardTracker()

        with pytest.raises(ValueError, match="Unknown strategy"):
            tracker.get_reward_statistics('unknown')

    def test_export_visualization_data(self):
        """Test visualization data export."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5, api_cost=0.02, convergence_speed=50)
        tracker.record_reward('a', fitness_improvement=0.7, api_cost=0.03, convergence_speed=30)
        tracker.record_reward('b', fitness_improvement=0.6, api_cost=0.01, convergence_speed=40)

        export_data = tracker.export_visualization_data()

        assert 'a' in export_data
        assert 'b' in export_data

        # Check strategy 'a' data
        assert export_data['a']['fitness_history'] == [0.5, 0.7]
        assert export_data['a']['cost_history'] == [0.02, 0.03]
        assert export_data['a']['convergence_history'] == [50, 30]
        assert export_data['a']['total_executions'] == 2
        assert 'avg_fitness' in export_data['a']
        assert 'avg_cost' in export_data['a']
        assert 'cost_efficiency' in export_data['a']

    def test_reset_strategy(self):
        """Test resetting a strategy."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5)
        tracker.record_reward('b', fitness_improvement=0.7)

        tracker.reset_strategy('a')

        assert tracker.get_strategy_metrics('a') is None
        assert tracker.get_strategy_metrics('b') is not None

    def test_reset_all(self):
        """Test resetting all data."""
        tracker = RewardTracker()

        tracker.record_reward('a', fitness_improvement=0.5)
        tracker.record_reward('b', fitness_improvement=0.7)

        tracker.reset_all()

        assert len(tracker.metrics) == 0
        assert len(tracker.records) == 0
        assert tracker._fitness_min == float('inf')
        assert tracker._fitness_max == float('-inf')

    def test_multiple_strategies_tracking(self):
        """Test tracking multiple strategies simultaneously."""
        tracker = RewardTracker()

        # Track multiple strategies
        for i in range(5):
            tracker.record_reward('strategy_1', fitness_improvement=0.5 + i * 0.1)
            tracker.record_reward('strategy_2', fitness_improvement=0.3 + i * 0.05)
            tracker.record_reward('strategy_3', fitness_improvement=0.7 + i * 0.02)

        # Verify all strategies tracked
        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) == 3
        assert all_metrics['strategy_1'].total_executions == 5
        assert all_metrics['strategy_2'].total_executions == 5
        assert all_metrics['strategy_3'].total_executions == 5

    def test_reward_normalization_edge_cases(self):
        """Test reward normalization handles edge cases."""
        tracker = RewardTracker()

        # Single value - should return as-is
        reward1 = tracker.record_reward('a', fitness_improvement=0.5)
        assert reward1 == 0.5

        # Same value again - should still be 0.5
        reward2 = tracker.record_reward('a', fitness_improvement=0.5)
        assert reward2 == 0.5

        # Different value triggers normalization
        reward3 = tracker.record_reward('a', fitness_improvement=0.8)
        # (0.8 - 0.5) / (0.8 - 0.5) = 1.0
        assert reward3 == pytest.approx(1.0)
