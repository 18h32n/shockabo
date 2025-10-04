import json

import pytest

from src.adapters.strategies.selection_explainer import (
    SelectionExplainer,
)


class TestSelectionExplainer:
    """Unit tests for SelectionExplainer."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        explainer = SelectionExplainer(max_history=500)
        assert explainer.max_history == 500
        assert len(explainer.decisions) == 0
        assert len(explainer.strategy_stats) == 0

    def test_initialization_invalid_history(self):
        """Test initialization fails with invalid max_history."""
        with pytest.raises(ValueError, match="max_history must be positive"):
            SelectionExplainer(max_history=0)

    def test_log_selection_basic(self):
        """Test basic selection logging."""
        explainer = SelectionExplainer()

        explainer.log_selection(
            selected_strategy='strategy_a',
            strategy_scores={'strategy_a': 0.8, 'strategy_b': 0.6},
            exploration_type='exploitation',
            selection_reason='Highest Thompson sample'
        )

        assert len(explainer.decisions) == 1
        assert explainer.decisions[0].selected_strategy == 'strategy_a'
        assert explainer.decisions[0].exploration_vs_exploitation == 'exploitation'
        assert 'strategy_a' in explainer.strategy_stats
        assert explainer.strategy_stats['strategy_a'].selection_count == 1

    def test_log_selection_with_task_features(self):
        """Test selection logging with task features."""
        explainer = SelectionExplainer()

        task_features = {'grid_height': 0.5, 'unique_colors': 0.3}

        explainer.log_selection(
            selected_strategy='strategy_a',
            strategy_scores={'strategy_a': 0.7},
            task_features=task_features,
            exploration_type='exploration'
        )

        assert explainer.decisions[0].task_features == task_features

    def test_log_selection_history_trimming(self):
        """Test decision history trimming at max_history."""
        explainer = SelectionExplainer(max_history=5)

        for i in range(10):
            explainer.log_selection(
                selected_strategy=f'strategy_{i}',
                strategy_scores={f'strategy_{i}': 0.5}
            )

        assert len(explainer.decisions) == 5
        assert explainer.decisions[0].selected_strategy == 'strategy_5'
        assert explainer.decisions[-1].selected_strategy == 'strategy_9'

    def test_log_outcome(self):
        """Test outcome logging."""
        explainer = SelectionExplainer()

        explainer.log_selection(
            selected_strategy='strategy_a',
            strategy_scores={'strategy_a': 0.7}
        )

        explainer.log_outcome('strategy_a', reward=0.8, is_win=True)

        stats = explainer.strategy_stats['strategy_a']
        assert stats.win_count == 1
        assert stats.total_reward == 0.8
        assert stats.avg_reward == pytest.approx(0.8)
        assert stats.win_rate == pytest.approx(1.0)

    def test_log_outcome_creates_stats_if_missing(self):
        """Test outcome logging creates stats if strategy not logged yet."""
        explainer = SelectionExplainer()

        explainer.log_outcome('strategy_a', reward=0.5, is_win=False)

        assert 'strategy_a' in explainer.strategy_stats
        assert explainer.strategy_stats['strategy_a'].total_reward == 0.5

    def test_get_selection_frequency(self):
        """Test selection frequency calculation."""
        explainer = SelectionExplainer()

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_selection('strategy_a', {'strategy_a': 0.8})
        explainer.log_selection('strategy_b', {'strategy_b': 0.6})

        freq = explainer.get_selection_frequency()

        assert freq['strategy_a'] == pytest.approx(2/3)
        assert freq['strategy_b'] == pytest.approx(1/3)

    def test_get_selection_frequency_empty(self):
        """Test selection frequency with no selections."""
        explainer = SelectionExplainer()

        freq = explainer.get_selection_frequency()
        assert freq == {}

    def test_get_win_rates(self):
        """Test win rate calculation."""
        explainer = SelectionExplainer()

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.8, is_win=True)

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.3, is_win=False)

        win_rates = explainer.get_win_rates()
        assert win_rates['strategy_a'] == pytest.approx(0.5)

    def test_get_average_rewards(self):
        """Test average reward calculation."""
        explainer = SelectionExplainer()

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.8, is_win=True)

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.6, is_win=True)

        avg_rewards = explainer.get_average_rewards()
        assert avg_rewards['strategy_a'] == pytest.approx(0.7)

    def test_get_recent_decisions(self):
        """Test recent decisions retrieval."""
        explainer = SelectionExplainer()

        for i in range(20):
            explainer.log_selection(f'strategy_{i}', {f'strategy_{i}': 0.5})

        recent = explainer.get_recent_decisions(n=5)

        assert len(recent) == 5
        assert recent[0].selected_strategy == 'strategy_15'
        assert recent[-1].selected_strategy == 'strategy_19'

    def test_export_decision_trace(self, tmp_path):
        """Test decision trace export to JSON."""
        explainer = SelectionExplainer()

        explainer.log_selection(
            selected_strategy='strategy_a',
            strategy_scores={'strategy_a': 0.8, 'strategy_b': 0.6},
            task_features={'grid_height': 0.5},
            exploration_type='exploitation',
            selection_reason='Best score',
            available_strategies=['strategy_a', 'strategy_b'],
            circuit_broken_strategies=[]
        )

        filepath = tmp_path / 'trace.json'
        explainer.export_decision_trace(str(filepath))

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]['selected_strategy'] == 'strategy_a'
        assert data[0]['strategy_scores'] == {'strategy_a': 0.8, 'strategy_b': 0.6}

    def test_get_strategy_performance_comparison(self):
        """Test strategy performance comparison."""
        explainer = SelectionExplainer()

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.8, is_win=True)

        explainer.log_selection('strategy_b', {'strategy_b': 0.6})
        explainer.log_outcome('strategy_b', 0.4, is_win=False)

        comparison = explainer.get_strategy_performance_comparison()

        assert 'strategy_a' in comparison
        assert comparison['strategy_a']['selection_count'] == 1
        assert comparison['strategy_a']['win_rate'] == pytest.approx(1.0)
        assert comparison['strategy_a']['avg_reward'] == pytest.approx(0.8)

        assert 'strategy_b' in comparison
        assert comparison['strategy_b']['win_rate'] == pytest.approx(0.0)

    def test_get_exploration_vs_exploitation_ratio(self):
        """Test exploration vs exploitation ratio calculation."""
        explainer = SelectionExplainer()

        explainer.log_selection('a', {'a': 0.5}, exploration_type='warmup')
        explainer.log_selection('b', {'b': 0.6}, exploration_type='warmup')
        explainer.log_selection('c', {'c': 0.7}, exploration_type='exploration')
        explainer.log_selection('d', {'d': 0.8}, exploration_type='exploitation')

        ratio = explainer.get_exploration_vs_exploitation_ratio()

        assert ratio['warmup'] == pytest.approx(0.5)
        assert ratio['exploration'] == pytest.approx(0.25)
        assert ratio['exploitation'] == pytest.approx(0.25)

    def test_get_exploration_vs_exploitation_ratio_empty(self):
        """Test ratio with no decisions."""
        explainer = SelectionExplainer()

        ratio = explainer.get_exploration_vs_exploitation_ratio()
        assert ratio == {'warmup': 0.0, 'exploration': 0.0, 'exploitation': 0.0}

    def test_get_strategy_probabilities_over_time(self):
        """Test strategy probability tracking over time."""
        explainer = SelectionExplainer()

        for i in range(10):
            strategy = 'strategy_a' if i < 5 else 'strategy_b'
            explainer.log_selection(strategy, {strategy: 0.5})

        probs = explainer.get_strategy_probabilities_over_time('strategy_a', window_size=5)

        assert len(probs) == 10
        assert probs[4] == pytest.approx(1.0)
        assert probs[9] == pytest.approx(0.0)

    def test_reset(self):
        """Test reset functionality."""
        explainer = SelectionExplainer()

        explainer.log_selection('strategy_a', {'strategy_a': 0.7})
        explainer.log_outcome('strategy_a', 0.8, is_win=True)

        explainer.reset()

        assert len(explainer.decisions) == 0
        assert len(explainer.strategy_stats) == 0

    def test_circuit_broken_strategies_logging(self):
        """Test circuit-broken strategies are logged in decisions."""
        explainer = SelectionExplainer()

        explainer.log_selection(
            selected_strategy='strategy_b',
            strategy_scores={'strategy_b': 0.8},
            circuit_broken_strategies=['strategy_a']
        )

        assert explainer.decisions[0].circuit_broken_strategies == ['strategy_a']
        assert explainer.decisions[0].selected_strategy == 'strategy_b'
