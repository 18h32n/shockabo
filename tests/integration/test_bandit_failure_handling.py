import pytest

from src.adapters.strategies.bandit_controller import BanditController


class TestBanditFailureHandling:
    """Integration tests for bandit failure handling scenarios."""

    def test_circuit_breaker_activation(self):
        """Test circuit breaker activates after 3 consecutive failures."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0,
            random_seed=42
        )

        controller.state.strategies['strategy_a'].success_count = 10

        assert not controller.state.strategies['strategy_a'].is_circuit_broken

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
        assert not controller.state.strategies['strategy_a'].is_circuit_broken

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
        assert not controller.state.strategies['strategy_a'].is_circuit_broken

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
        assert controller.state.strategies['strategy_a'].is_circuit_broken

    def test_circuit_breaker_prevents_selection(self):
        """Test circuit-broken strategies are not selected."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0,
            random_seed=42
        )

        for _ in range(3):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        assert controller.state.strategies['strategy_a'].is_circuit_broken

        for _ in range(20):
            selected = controller.select_strategy()
            assert selected == 'strategy_b'

    def test_gradual_re_enabling_via_reset(self):
        """Test gradual re-enabling of failed strategies via reset."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0,
            random_seed=42
        )

        for _ in range(3):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        assert controller.state.strategies['strategy_a'].is_circuit_broken

        controller.reset_circuit_breaker('strategy_a')

        assert not controller.state.strategies['strategy_a'].is_circuit_broken
        assert controller.state.strategies['strategy_a'].consecutive_failures == 0

        selected = controller.select_strategy()
        assert selected in ['strategy_a', 'strategy_b']

    def test_fallback_when_all_strategies_broken(self):
        """Test error when all strategies are circuit-broken."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0
        )

        for _ in range(3):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
            controller.update_reward('strategy_b', reward=-0.5, is_failure=True)

        assert controller.state.strategies['strategy_a'].is_circuit_broken
        assert controller.state.strategies['strategy_b'].is_circuit_broken

        with pytest.raises(RuntimeError, match="All strategies are circuit-broken"):
            controller.select_strategy()

    def test_negative_reward_for_failures(self):
        """Test negative reward is properly recorded for failures."""
        controller = BanditController(strategies=['strategy_a'])

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        strategy = controller.state.strategies['strategy_a']
        assert strategy.total_reward == -0.5
        assert strategy.failure_count == 1
        assert strategy.consecutive_failures == 1

    def test_low_fitness_treated_as_failure(self):
        """Test rewards below 0.5 threshold are treated as failures."""
        controller = BanditController(strategies=['strategy_a'])

        controller.update_reward('strategy_a', reward=0.3, is_failure=False)

        strategy = controller.state.strategies['strategy_a']
        assert strategy.failure_count == 1
        assert strategy.success_count == 0

    def test_timeout_failure_handling(self):
        """Test timeout is handled as failure."""
        controller = BanditController(strategies=['strategy_a'])

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        strategy = controller.state.strategies['strategy_a']
        assert strategy.failure_count == 1
        assert strategy.consecutive_failures == 1

    def test_error_failure_handling(self):
        """Test errors are handled as failures."""
        controller = BanditController(strategies=['strategy_a'])

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        strategy = controller.state.strategies['strategy_a']
        assert strategy.failure_count == 1

    def test_consecutive_failures_reset_after_success(self):
        """Test consecutive failures reset after successful execution."""
        controller = BanditController(strategies=['strategy_a'])

        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
        controller.update_reward('strategy_a', reward=-0.5, is_failure=True)
        assert controller.state.strategies['strategy_a'].consecutive_failures == 2

        controller.update_reward('strategy_a', reward=0.9, is_failure=False)
        assert controller.state.strategies['strategy_a'].consecutive_failures == 0
        assert not controller.state.strategies['strategy_a'].is_circuit_broken

    def test_multiple_strategies_independent_failure_tracking(self):
        """Test failures are tracked independently for each strategy."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0
        )

        for _ in range(3):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        controller.update_reward('strategy_b', reward=0.8, is_failure=False)

        assert controller.state.strategies['strategy_a'].is_circuit_broken
        assert not controller.state.strategies['strategy_b'].is_circuit_broken

        selected = controller.select_strategy()
        assert selected == 'strategy_b'

    def test_failure_count_accumulates(self):
        """Test failure count accumulates across circuit breaker resets."""
        controller = BanditController(strategies=['strategy_a'])

        for _ in range(3):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        assert controller.state.strategies['strategy_a'].failure_count == 3

        controller.reset_circuit_breaker('strategy_a')

        assert controller.state.strategies['strategy_a'].failure_count == 3
        assert controller.state.strategies['strategy_a'].consecutive_failures == 0

    def test_partial_success_after_failures(self):
        """Test strategy can recover from failures."""
        controller = BanditController(
            strategies=['strategy_a', 'strategy_b'],
            warmup_selections=0,
            random_seed=42
        )

        for _ in range(2):
            controller.update_reward('strategy_a', reward=-0.5, is_failure=True)

        controller.update_reward('strategy_a', reward=0.9, is_failure=False)

        assert not controller.state.strategies['strategy_a'].is_circuit_broken
        assert controller.state.strategies['strategy_a'].consecutive_failures == 0
        assert controller.state.strategies['strategy_a'].success_count == 1
        assert controller.state.strategies['strategy_a'].failure_count == 2
