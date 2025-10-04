import numpy as np
import pytest

from src.adapters.strategies.bandit_controller import (
    BanditController,
)


class TestBanditController:
    """Unit tests for BanditController Thompson Sampling implementation."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            alpha_prior=2.0,
            beta_prior=3.0,
            warmup_selections=100,
            random_seed=42
        )

        assert len(controller.state.strategies) == 2
        assert 'strategy_a' in controller.state.strategies
        assert 'strategy_b' in controller.state.strategies
        assert controller.state.alpha_prior == 2.0
        assert controller.state.beta_prior == 3.0
        assert controller.state.warmup_selections == 100

    def test_initialization_empty_strategies(self):
        """Test initialization fails with empty strategy list."""
        with pytest.raises(ValueError, match="Must provide at least one strategy"):
            BanditController(strategies=[])

    def test_initialization_invalid_priors(self):
        """Test initialization fails with invalid priors."""
        with pytest.raises(ValueError, match="Alpha and beta priors must be positive"):
            BanditController(strategies=['a'], alpha_prior=0)

        with pytest.raises(ValueError, match="Alpha and beta priors must be positive"):
            BanditController(strategies=['a'], beta_prior=-1)

    def test_initialization_invalid_warmup(self):
        """Test initialization fails with negative warmup."""
        with pytest.raises(ValueError, match="Warmup selections must be non-negative"):
            BanditController(strategies=['a'], warmup_selections=-5)

    def test_select_strategy_warmup_uniform(self):
        """Test warmup period uses uniform selection."""
        controller = BanditController(
            strategies=['a', 'b', 'c'],
            warmup_selections=5,
            random_seed=42
        )

        # During warmup, should select each strategy at least once
        selections = [controller.select_strategy() for _ in range(15)]

        # Verify each strategy selected at least once during warmup
        assert 'a' in selections
        assert 'b' in selections
        assert 'c' in selections

        # Verify selection counts
        assert controller.state.strategies['a'].selection_count == 5
        assert controller.state.strategies['b'].selection_count == 5
        assert controller.state.strategies['c'].selection_count == 5

    def test_select_strategy_thompson_sampling(self):
        """Test Thompson Sampling selection after warmup."""
        controller = BanditController(
            strategies=['good', 'bad'],
            warmup_selections=0,  # Skip warmup
            random_seed=42
        )

        # Simulate good strategy having high success
        controller.state.strategies['good'].success_count = 10
        controller.state.strategies['good'].failure_count = 2

        # Simulate bad strategy having low success
        controller.state.strategies['bad'].success_count = 2
        controller.state.strategies['bad'].failure_count = 10

        # Select 100 times - good strategy should be selected more often
        selections = [controller.select_strategy() for _ in range(100)]
        good_count = sum(1 for s in selections if s == 'good')
        bad_count = sum(1 for s in selections if s == 'bad')

        assert good_count > bad_count
        assert good_count > 50  # Should be selected majority of time

    def test_select_strategy_with_circuit_breaker(self):
        """Test strategy selection skips circuit-broken strategies."""
        controller = BanditController(
            strategies=['a', 'b'],
            warmup_selections=0,
            random_seed=42
        )

        # Break strategy 'a'
        controller.state.strategies['a'].is_circuit_broken = True

        # Should always select 'b'
        for _ in range(10):
            assert controller.select_strategy() == 'b'

    def test_select_strategy_all_broken_raises(self):
        """Test selection fails when all strategies are broken."""
        controller = BanditController(
            strategies=['a', 'b'],
            warmup_selections=0
        )

        # Break all strategies
        controller.state.strategies['a'].is_circuit_broken = True
        controller.state.strategies['b'].is_circuit_broken = True

        with pytest.raises(RuntimeError, match="All strategies are circuit-broken"):
            controller.select_strategy()

    def test_update_reward_success(self):
        """Test reward update for successful strategy."""
        controller = BanditController(strategies=['a'])

        controller.update_reward('a', reward=0.8, is_failure=False)

        strategy = controller.state.strategies['a']
        assert strategy.success_count == 1
        assert strategy.failure_count == 0
        assert strategy.total_reward == 0.8
        assert strategy.consecutive_failures == 0

    def test_update_reward_failure(self):
        """Test reward update for failed strategy."""
        controller = BanditController(strategies=['a'])

        controller.update_reward('a', reward=-0.5, is_failure=True)

        strategy = controller.state.strategies['a']
        assert strategy.success_count == 0
        assert strategy.failure_count == 1
        assert strategy.total_reward == -0.5
        assert strategy.consecutive_failures == 1

    def test_update_reward_below_threshold(self):
        """Test reward below 0.5 counts as failure."""
        controller = BanditController(strategies=['a'])

        controller.update_reward('a', reward=0.3, is_failure=False)

        strategy = controller.state.strategies['a']
        assert strategy.success_count == 0
        assert strategy.failure_count == 1
        assert strategy.consecutive_failures == 1

    def test_update_reward_circuit_breaker_triggers(self):
        """Test circuit breaker triggers after 3 consecutive failures."""
        controller = BanditController(strategies=['a'])

        # First two failures - no circuit break
        controller.update_reward('a', reward=-0.5, is_failure=True)
        controller.update_reward('a', reward=-0.5, is_failure=True)
        assert not controller.state.strategies['a'].is_circuit_broken

        # Third failure - circuit breaks
        controller.update_reward('a', reward=-0.5, is_failure=True)
        assert controller.state.strategies['a'].is_circuit_broken

    def test_update_reward_consecutive_failures_reset_on_success(self):
        """Test consecutive failures reset on success."""
        controller = BanditController(strategies=['a'])

        # Two failures
        controller.update_reward('a', reward=-0.5, is_failure=True)
        controller.update_reward('a', reward=-0.5, is_failure=True)
        assert controller.state.strategies['a'].consecutive_failures == 2

        # Success resets consecutive failures
        controller.update_reward('a', reward=0.9, is_failure=False)
        assert controller.state.strategies['a'].consecutive_failures == 0
        assert not controller.state.strategies['a'].is_circuit_broken

    def test_update_reward_unknown_strategy(self):
        """Test update fails for unknown strategy."""
        controller = BanditController(strategies=['a'])

        with pytest.raises(ValueError, match="Unknown strategy: unknown"):
            controller.update_reward('unknown', reward=0.5)

    def test_update_reward_nan_raises(self):
        """Test update fails for NaN reward."""
        controller = BanditController(strategies=['a'])

        with pytest.raises(ValueError, match="Reward must be finite"):
            controller.update_reward('a', reward=float('nan'))

    def test_update_reward_inf_raises(self):
        """Test update fails for infinite reward."""
        controller = BanditController(strategies=['a'])

        with pytest.raises(ValueError, match="Reward must be finite"):
            controller.update_reward('a', reward=float('inf'))

    def test_reset_circuit_breaker(self):
        """Test circuit breaker reset."""
        controller = BanditController(strategies=['a'])

        # Trigger circuit breaker
        for _ in range(3):
            controller.update_reward('a', reward=-0.5, is_failure=True)
        assert controller.state.strategies['a'].is_circuit_broken

        # Reset
        controller.reset_circuit_breaker('a')
        assert not controller.state.strategies['a'].is_circuit_broken
        assert controller.state.strategies['a'].consecutive_failures == 0

    def test_reset_circuit_breaker_unknown_strategy(self):
        """Test reset fails for unknown strategy."""
        controller = BanditController(strategies=['a'])

        with pytest.raises(ValueError, match="Unknown strategy: unknown"):
            controller.reset_circuit_breaker('unknown')

    def test_get_strategy_stats(self):
        """Test strategy statistics retrieval."""
        controller = BanditController(strategies=['a', 'b'])

        # Update strategy 'a'
        controller.update_reward('a', reward=0.8)
        controller.update_reward('a', reward=0.6)
        controller.update_reward('a', reward=0.3)
        controller.state.strategies['a'].selection_count = 3

        stats = controller.get_strategy_stats()

        assert 'a' in stats
        assert stats['a']['success_count'] == 2
        assert stats['a']['failure_count'] == 1
        assert stats['a']['total_reward'] == pytest.approx(1.7)
        assert stats['a']['selection_count'] == 3
        assert stats['a']['avg_reward'] == pytest.approx(1.7 / 3)
        assert stats['a']['win_rate'] == pytest.approx(2.0 / 3.0)
        assert stats['a']['is_circuit_broken'] is False

    def test_get_strategy_stats_no_selections(self):
        """Test stats with no selections."""
        controller = BanditController(strategies=['a'])

        stats = controller.get_strategy_stats()

        assert stats['a']['avg_reward'] == 0.0
        assert stats['a']['win_rate'] == 0.0

    def test_beta_distribution_properties(self):
        """Test that Beta sampling respects distribution properties."""
        controller = BanditController(
            strategies=['a'],
            warmup_selections=0,
            alpha_prior=1.0,
            beta_prior=1.0,
            random_seed=42
        )

        # Set known success/failure counts
        controller.state.strategies['a'].success_count = 10
        controller.state.strategies['a'].failure_count = 5

        # Sample many times to check mean
        samples = []
        for _ in range(1000):
            # Manually sample to check distribution
            alpha = 1.0 + 10
            beta = 1.0 + 5
            sample = controller.rng.beta(alpha, beta)
            samples.append(sample)

        # Beta distribution mean: alpha / (alpha + beta)
        expected_mean = 11 / 16
        actual_mean = np.mean(samples)

        assert abs(actual_mean - expected_mean) < 0.05  # Within 5% tolerance
