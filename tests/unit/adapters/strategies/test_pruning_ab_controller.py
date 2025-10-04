"""Unit tests for pruning A/B testing controller.

Tests the A/B testing framework for comparing different pruning strategies.
"""

import random
from datetime import datetime
from unittest.mock import patch

import pytest

from src.adapters.strategies.pruning_ab_controller import ABTestController
from src.domain.models import (
    PruningDecision,
    PruningMetrics,
    PruningResult,
    PruningStrategy,
)


class TestABTestController:
    """Test A/B testing controller functionality."""

    @pytest.fixture
    def strategies(self):
        """Create test pruning strategies."""
        return [
            PruningStrategy(
                strategy_id="conservative",
                name="Conservative",
                aggressiveness=0.3,
                syntax_checks=True,
                pattern_checks=False,
                partial_execution=False,
                confidence_threshold=0.8,
                max_partial_ops=2,
                timeout_ms=50,
            ),
            PruningStrategy(
                strategy_id="balanced",
                name="Balanced",
                aggressiveness=0.5,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.6,
                max_partial_ops=3,
                timeout_ms=100,
            ),
            PruningStrategy(
                strategy_id="aggressive",
                name="Aggressive",
                aggressiveness=0.8,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.4,
                max_partial_ops=5,
                timeout_ms=150,
            ),
        ]

    @pytest.fixture
    def ab_controller(self, strategies):
        """Create test A/B controller."""
        return ABTestController(
            strategies=strategies,
            exploration_rate=0.2,
            min_samples=100,
            confidence_level=0.95,
        )

    def test_initialization(self, ab_controller, strategies):
        """Test controller initialization."""
        assert len(ab_controller.strategies) == 3
        assert ab_controller.exploration_rate == 0.2
        assert ab_controller.min_samples == 100

        # Check initial allocations
        allocations = ab_controller.get_current_allocations()
        assert len(allocations) == 3
        assert all(abs(alloc - 1/3) < 0.01 for alloc in allocations.values())

    def test_exploration_selection(self, ab_controller):
        """Test exploration vs exploitation selection."""
        exploration_count = 0
        total_selections = 1000

        # Mock random to control exploration
        with patch('random.random') as mock_random:
            for i in range(total_selections):
                # First 20% will be exploration
                mock_random.return_value = 0.1 if i < 200 else 0.9

                strategy = ab_controller.select_strategy()
                assert strategy is not None

                if i < 200:
                    exploration_count += 1

        # Should have roughly 20% exploration
        exploration_rate = exploration_count / total_selections
        assert abs(exploration_rate - 0.2) < 0.05

    def test_weighted_strategy_selection(self, ab_controller):
        """Test weighted selection based on allocations."""
        # Set custom allocations
        ab_controller.allocations = {
            "conservative": 0.1,
            "balanced": 0.6,
            "aggressive": 0.3,
        }

        # Select many times and count
        selections = {"conservative": 0, "balanced": 0, "aggressive": 0}

        # Force exploitation mode
        ab_controller.exploration_rate = 0.0

        for _ in range(1000):
            strategy = ab_controller.select_strategy()
            selections[strategy.strategy_id] += 1

        # Check selections roughly match allocations
        assert selections["balanced"] > selections["aggressive"] > selections["conservative"]
        assert abs(selections["balanced"] / 1000 - 0.6) < 0.1

    def test_metrics_update(self, ab_controller):
        """Test updating strategy metrics."""
        # Create test metrics
        metrics = PruningMetrics(
            strategy_id="balanced",
            total_programs=100,
            programs_pruned=40,
            pruning_rate=0.4,
            false_negatives=2,
            false_negative_rate=0.02,
            avg_pruning_time_ms=5.0,
            time_saved_ms=1500.0,
            timestamp=datetime.now(),
        )

        # Create pruning results
        results = []
        for i in range(100):
            result = PruningResult(
                program_id=f"prog{i}",
                decision=PruningDecision.REJECT_PATTERN if i < 40 else PruningDecision.ACCEPT,
                confidence_score=0.7,
                pruning_time_ms=5.0,
                rejection_reason="Test" if i < 40 else None,
            )
            results.append(result)

        # Update metrics
        ab_controller.update_metrics("balanced", metrics, results)

        # Check stats were updated
        stats = ab_controller.strategy_stats["balanced"]
        assert stats["total_programs"] == 100
        assert stats["programs_pruned"] == 40
        assert stats["samples"] == 1
        assert len(stats["time_saved_ms"]) == 1
        assert stats["time_saved_ms"][0] == 1500.0

    def test_statistical_significance(self, ab_controller):
        """Test statistical significance detection."""
        # Not enough samples initially
        assert not ab_controller._has_significance()

        # Add samples for each strategy (need to reach min_samples threshold)
        for strategy_id in ab_controller.strategies:
            for _ in range(100):  # Exactly min_samples
                metrics = PruningMetrics(
                    strategy_id=strategy_id,
                    total_programs=10,
                    programs_pruned=random.randint(3, 7),
                    pruning_rate=random.uniform(0.3, 0.7),
                    false_negatives=random.randint(0, 2),
                    false_negative_rate=random.uniform(0, 0.1),
                    avg_pruning_time_ms=random.uniform(4, 6),
                    time_saved_ms=random.uniform(100, 200),
                    timestamp=datetime.now(),
                )

                ab_controller.update_metrics(strategy_id, metrics, [])

        # Now should have significance
        assert ab_controller._has_significance()

    def test_allocation_adjustment(self, ab_controller):
        """Test allocation adjustment based on performance."""
        # Create performance data with clear winner
        for i in range(150):
            for strategy_id in ["conservative", "balanced", "aggressive"]:
                # Make balanced perform best
                if strategy_id == "balanced":
                    time_saved = random.uniform(1800, 2000)
                    false_neg_rate = random.uniform(0, 0.02)
                    pruning_rate = random.uniform(0.4, 0.5)
                elif strategy_id == "aggressive":
                    time_saved = random.uniform(1500, 1700)
                    false_neg_rate = random.uniform(0.05, 0.1)  # Higher false negatives
                    pruning_rate = random.uniform(0.6, 0.7)
                else:  # conservative
                    time_saved = random.uniform(800, 1000)
                    false_neg_rate = random.uniform(0, 0.01)
                    pruning_rate = random.uniform(0.2, 0.3)

                metrics = PruningMetrics(
                    strategy_id=strategy_id,
                    total_programs=100,
                    programs_pruned=int(pruning_rate * 100),
                    pruning_rate=pruning_rate,
                    false_negatives=int(false_neg_rate * pruning_rate * 100),
                    false_negative_rate=false_neg_rate,
                    avg_pruning_time_ms=5.0,
                    time_saved_ms=time_saved,
                    timestamp=datetime.now(),
                )

                # Store detailed metrics
                ab_controller.strategy_stats[strategy_id]["time_saved_ms"].append(time_saved)
                ab_controller.strategy_stats[strategy_id]["total_programs"] += 100
                ab_controller.strategy_stats[strategy_id]["programs_pruned"] += metrics.programs_pruned
                ab_controller.strategy_stats[strategy_id]["false_negatives"] += metrics.false_negatives
                ab_controller.strategy_stats[strategy_id]["samples"] += 1

        # Force allocation adjustment
        ab_controller._adjust_allocations()

        # Check allocations were adjusted
        allocations = ab_controller.get_current_allocations()

        # Balanced should have highest allocation
        assert allocations["balanced"] > allocations["conservative"]
        assert allocations["balanced"] > allocations["aggressive"]

    def test_best_strategy_selection(self, ab_controller):
        """Test getting best performing strategy."""
        # Initially no best strategy (not enough data)
        best_id, metrics = ab_controller.get_best_strategy()
        assert best_id is None

        # Set performance data to make balanced the clear winner
        # Conservative: low time savings but low false negatives
        ab_controller.strategy_stats["conservative"].update({
            "total_programs": 500,
            "programs_pruned": 150,
            "false_negatives": 1,
            "samples": 50,
            "time_saved_ms": [800] * 50,
        })

        # Balanced: excellent time savings and very low false negatives
        ab_controller.strategy_stats["balanced"].update({
            "total_programs": 500,
            "programs_pruned": 300,
            "false_negatives": 2,  # Very low false negatives
            "samples": 50,
            "time_saved_ms": [1800] * 50,  # High time savings
        })

        # Aggressive: high time savings but excessive false negatives
        ab_controller.strategy_stats["aggressive"].update({
            "total_programs": 500,
            "programs_pruned": 350,
            "false_negatives": 35,  # High false negatives that hurt the score
            "samples": 50,
            "time_saved_ms": [1900] * 50,
        })

        # Get best strategy
        best_id, metrics = ab_controller.get_best_strategy()

        # Balanced should be best (good balance of performance and accuracy)
        assert best_id == "balanced"
        assert metrics["avg_time_saved_ms"] > 1500
        assert metrics["false_negative_rate"] < 0.01

    def test_performance_summary(self, ab_controller):
        """Test performance summary generation."""
        # Add some test data
        test_data = {
            "conservative": {
                "programs": 100,
                "pruned": 30,
                "false_negs": 1,
                "time_saved": [500],
                "pruning_times": [3.0],
                "confidence_scores": [0.9],
            },
            "balanced": {
                "programs": 100,
                "pruned": 50,
                "false_negs": 2,
                "time_saved": [800],
                "pruning_times": [4.0],
                "confidence_scores": [0.7],
            },
        }

        for strategy_id, data in test_data.items():
            stats = ab_controller.strategy_stats[strategy_id]
            stats["total_programs"] = data["programs"]
            stats["programs_pruned"] = data["pruned"]
            stats["false_negatives"] = data["false_negs"]
            stats["time_saved_ms"] = data["time_saved"]
            stats["pruning_times_ms"] = data["pruning_times"]
            stats["confidence_scores"] = data["confidence_scores"]
            stats["samples"] = 10

        # Get summary
        summary = ab_controller.get_performance_summary()

        assert "conservative" in summary
        assert "balanced" in summary

        # Check calculated metrics
        assert summary["conservative"]["pruning_rate"] == 0.3
        assert summary["balanced"]["pruning_rate"] == 0.5

    def test_export_results(self, ab_controller):
        """Test exporting A/B test results."""
        # Add some test data
        for _ in range(10):
            metrics = PruningMetrics(
                strategy_id="balanced",
                total_programs=10,
                programs_pruned=5,
                pruning_rate=0.5,
                false_negatives=0,
                false_negative_rate=0.0,
                avg_pruning_time_ms=5.0,
                time_saved_ms=150.0,
                timestamp=datetime.now(),
            )
            ab_controller.update_metrics("balanced", metrics, [])

        # Export results
        results = ab_controller.export_results()

        assert "test_config" in results
        assert "performance_summary" in results
        assert "best_strategy" in results
        assert "current_allocations" in results
        assert "decision_history" in results
        assert "timestamp" in results

        # Check config
        assert results["test_config"]["strategies"] == ["conservative", "balanced", "aggressive"]
        assert results["test_config"]["confidence_level"] == 0.95

    def test_context_aware_selection(self, ab_controller):
        """Test context-aware strategy selection."""
        # Test with different task features
        task_features = {
            "queue_length": 1000.0,  # High load
            "memory_pressure": 0.2,
            "platform_kaggle": 1.0,
            "platform_colab": 0.0,
            "platform_paperspace": 0.0,
        }

        # Force exploitation mode
        ab_controller.exploration_rate = 0.0

        # Select strategy with context
        strategy = ab_controller.select_strategy(task_features)

        assert strategy is not None
        # With high queue length, might prefer more aggressive pruning

    def test_decision_history(self, ab_controller):
        """Test decision history tracking."""
        # Update metrics multiple times
        for i in range(5):
            metrics = PruningMetrics(
                strategy_id="balanced",
                total_programs=10,
                programs_pruned=5,
                pruning_rate=0.5,
                false_negatives=i % 2,  # Alternate false negatives
                false_negative_rate=0.1 if i % 2 else 0.0,
                avg_pruning_time_ms=5.0,
                time_saved_ms=150.0,
                timestamp=datetime.now(),
            )

            ab_controller.update_metrics("balanced", metrics, [])

        # Check history
        assert len(ab_controller.decision_history) == 5

        # Check history entries
        for entry in ab_controller.decision_history:
            assert "timestamp" in entry
            assert "strategy_id" in entry
            assert "metrics" in entry
            assert "false_negative_rate" in entry
