import pytest

from src.adapters.strategies.cost_aware_reward import CostAwareRewardCalculator, ModelCostTracker


class TestCostAwareRewardCalculator:
    """Unit tests for CostAwareRewardCalculator."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        calculator = CostAwareRewardCalculator(cost_weight=0.5, track_costs=True)
        assert calculator.cost_weight == 0.5
        assert calculator.track_costs is True

    def test_initialization_invalid_cost_weight(self):
        """Test initialization fails with invalid cost_weight."""
        with pytest.raises(ValueError, match="cost_weight must be in"):
            CostAwareRewardCalculator(cost_weight=-0.1)

        with pytest.raises(ValueError, match="cost_weight must be in"):
            CostAwareRewardCalculator(cost_weight=1.5)

    def test_calculate_reward_no_cost(self):
        """Test reward calculation with zero cost."""
        calculator = CostAwareRewardCalculator()

        reward = calculator.calculate_reward(fitness=0.8, cost=0.0)
        assert reward == 0.8

    def test_calculate_reward_with_cost(self):
        """Test reward calculation with API cost."""
        calculator = CostAwareRewardCalculator(cost_weight=0.3)

        calculator._cost_min = 0.01
        calculator._cost_max = 0.10

        reward = calculator.calculate_reward(fitness=0.9, cost=0.05)

        cost_normalized = (0.05 - 0.01) / (0.10 - 0.01)
        expected_reward = 0.9 / (1.0 + 0.3 * cost_normalized)

        assert reward == pytest.approx(expected_reward)

    def test_calculate_reward_invalid_fitness(self):
        """Test reward calculation fails with invalid fitness."""
        calculator = CostAwareRewardCalculator()

        with pytest.raises(ValueError, match="fitness must be in"):
            calculator.calculate_reward(fitness=-0.1, cost=0.0)

        with pytest.raises(ValueError, match="fitness must be in"):
            calculator.calculate_reward(fitness=1.5, cost=0.0)

    def test_calculate_reward_invalid_cost(self):
        """Test reward calculation fails with negative cost."""
        calculator = CostAwareRewardCalculator()

        with pytest.raises(ValueError, match="cost must be non-negative"):
            calculator.calculate_reward(fitness=0.8, cost=-0.1)

    def test_calculate_reward_tracks_costs(self):
        """Test reward calculation tracks costs when enabled."""
        calculator = CostAwareRewardCalculator(track_costs=True)

        calculator.calculate_reward(fitness=0.8, cost=0.05, strategy_id='strategy_a')
        calculator.calculate_reward(fitness=0.7, cost=0.03, strategy_id='strategy_a')

        stats = calculator.get_strategy_costs()
        assert 'strategy_a' in stats
        assert stats['strategy_a']['total_cost'] == pytest.approx(0.08)
        assert stats['strategy_a']['api_calls'] == 2
        assert stats['strategy_a']['avg_cost_per_call'] == pytest.approx(0.04)

    def test_normalize_cost_no_variation(self):
        """Test cost normalization with no variation."""
        calculator = CostAwareRewardCalculator()

        normalized = calculator._normalize_cost(0.05)
        assert normalized == 0.5

    def test_normalize_cost_with_range(self):
        """Test cost normalization with established range."""
        calculator = CostAwareRewardCalculator()
        calculator._cost_min = 0.01
        calculator._cost_max = 0.10

        normalized = calculator._normalize_cost(0.055)
        expected = (0.055 - 0.01) / (0.10 - 0.01)
        assert normalized == pytest.approx(expected)

    def test_normalize_cost_clamps_to_range(self):
        """Test cost normalization clamps to [0, 1]."""
        calculator = CostAwareRewardCalculator()
        calculator._cost_min = 0.01
        calculator._cost_max = 0.10

        normalized_low = calculator._normalize_cost(0.0)
        assert normalized_low == 0.0

        normalized_high = calculator._normalize_cost(1.0)
        assert normalized_high == 1.0

    def test_get_strategy_costs_empty(self):
        """Test getting costs with no tracked strategies."""
        calculator = CostAwareRewardCalculator()

        stats = calculator.get_strategy_costs()
        assert stats == {}

    def test_get_cost_efficiency(self):
        """Test cost efficiency calculation."""
        calculator = CostAwareRewardCalculator()

        calculator.calculate_reward(0.8, 0.05, strategy_id='strategy_a')
        calculator.calculate_reward(0.6, 0.03, strategy_id='strategy_a')

        efficiency = calculator.get_cost_efficiency('strategy_a', total_fitness=1.4)
        expected_efficiency = 1.4 / 0.08
        assert efficiency == pytest.approx(expected_efficiency)

    def test_get_cost_efficiency_zero_cost(self):
        """Test cost efficiency with zero cost (infinite efficiency)."""
        calculator = CostAwareRewardCalculator()
        calculator.strategy_costs['strategy_a'] = 0.0

        efficiency = calculator.get_cost_efficiency('strategy_a', total_fitness=1.0)
        assert efficiency == float('inf')

    def test_get_cost_efficiency_unknown_strategy(self):
        """Test cost efficiency for unknown strategy."""
        calculator = CostAwareRewardCalculator()

        efficiency = calculator.get_cost_efficiency('unknown', total_fitness=1.0)
        assert efficiency is None

    def test_set_cost_weight(self):
        """Test updating cost weight."""
        calculator = CostAwareRewardCalculator(cost_weight=0.3)

        calculator.set_cost_weight(0.5)
        assert calculator.cost_weight == 0.5

    def test_set_cost_weight_invalid(self):
        """Test setting invalid cost weight."""
        calculator = CostAwareRewardCalculator()

        with pytest.raises(ValueError, match="cost_weight must be in"):
            calculator.set_cost_weight(-0.1)

    def test_reset_cost_stats(self):
        """Test resetting cost statistics."""
        calculator = CostAwareRewardCalculator()

        calculator.calculate_reward(0.8, 0.05, strategy_id='strategy_a')
        calculator.reset_cost_stats()

        assert calculator.strategy_costs == {}
        assert calculator._cost_min == float('inf')
        assert calculator._cost_max == float('-inf')

    def test_cost_weight_zero_ignores_cost(self):
        """Test cost weight of 0 ignores cost completely."""
        calculator = CostAwareRewardCalculator(cost_weight=0.0)
        calculator._cost_min = 0.0
        calculator._cost_max = 1.0

        reward = calculator.calculate_reward(fitness=0.9, cost=0.5)
        assert reward == pytest.approx(0.9)

    def test_cost_weight_one_heavy_penalty(self):
        """Test cost weight of 1.0 applies heavy cost penalty."""
        calculator = CostAwareRewardCalculator(cost_weight=1.0)
        calculator._cost_min = 0.0
        calculator._cost_max = 0.1

        reward = calculator.calculate_reward(fitness=0.9, cost=0.1)
        expected = 0.9 / (1.0 + 1.0 * 1.0)
        assert reward == pytest.approx(expected)


class TestModelCostTracker:
    """Unit tests for ModelCostTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ModelCostTracker()
        assert len(tracker.model_usage) == 0

    def test_calculate_cost_gpt5(self):
        """Test cost calculation for GPT-5."""
        tracker = ModelCostTracker()

        cost = tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)

        expected_cost = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_gemini(self):
        """Test cost calculation for Gemini."""
        tracker = ModelCostTracker()

        cost = tracker.calculate_cost('gemini-pro', input_tokens=2000, output_tokens=1000)

        expected_cost = (2000 / 1000) * 0.0005 + (1000 / 1000) * 0.0015
        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_local_model(self):
        """Test cost calculation for local model (free)."""
        tracker = ModelCostTracker()

        cost = tracker.calculate_cost('local', input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation fails for unknown model."""
        tracker = ModelCostTracker()

        with pytest.raises(ValueError, match="Unknown model"):
            tracker.calculate_cost('unknown-model', input_tokens=100, output_tokens=50)

    def test_calculate_cost_tracks_usage(self):
        """Test cost calculation tracks usage statistics."""
        tracker = ModelCostTracker()

        tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        tracker.calculate_cost('gpt-5', input_tokens=500, output_tokens=250)

        stats = tracker.get_model_usage_stats()
        assert 'gpt-5' in stats
        assert stats['gpt-5']['total_input_tokens'] == 1500
        assert stats['gpt-5']['total_output_tokens'] == 750
        assert stats['gpt-5']['total_calls'] == 2

    def test_get_model_usage_stats(self):
        """Test getting model usage statistics."""
        tracker = ModelCostTracker()

        tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        tracker.calculate_cost('gemini-pro', input_tokens=2000, output_tokens=1000)

        stats = tracker.get_model_usage_stats()

        assert 'gpt-5' in stats
        assert 'gemini-pro' in stats
        assert stats['gpt-5']['total_calls'] == 1
        assert stats['gemini-pro']['total_calls'] == 1

    def test_get_total_cost(self):
        """Test total cost calculation across models."""
        tracker = ModelCostTracker()

        tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        tracker.calculate_cost('gemini-pro', input_tokens=2000, output_tokens=1000)

        total_cost = tracker.get_total_cost()

        gpt_cost = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        gemini_cost = (2000 / 1000) * 0.0005 + (1000 / 1000) * 0.0015
        expected_total = gpt_cost + gemini_cost

        assert total_cost == pytest.approx(expected_total)

    def test_reset(self):
        """Test reset functionality."""
        tracker = ModelCostTracker()

        tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        tracker.reset()

        assert len(tracker.model_usage) == 0
        assert tracker.get_total_cost() == 0.0

    def test_multiple_models_tracked_independently(self):
        """Test multiple models are tracked independently."""
        tracker = ModelCostTracker()

        tracker.calculate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        tracker.calculate_cost('gemini-pro', input_tokens=2000, output_tokens=1000)
        tracker.calculate_cost('local', input_tokens=5000, output_tokens=2500)

        stats = tracker.get_model_usage_stats()

        assert len(stats) == 3
        assert stats['gpt-5']['estimated_cost'] > 0
        assert stats['gemini-pro']['estimated_cost'] > 0
        assert stats['local']['estimated_cost'] == 0.0
