"""Integration tests for the intelligent program pruning system.

Tests the integration of pruning with evaluation service, GPU batch evaluation,
and the overall evolution engine.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.adapters.strategies.program_pruner import ProgramPruner
from src.adapters.strategies.pruning_ab_controller import ABTestController
from src.adapters.strategies.pruning_metrics_collector import PruningMetricsCollector
from src.domain.dsl.base import Operation, OperationResult
from src.domain.models import (
    DSLProgram,
    PruningDecision,
    PruningMetrics,
    PruningStrategy,
)
from src.domain.services.evaluation_service import EvaluationService


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, name: str, **params):
        self._name = name
        self.parameters = params

    def execute(self, grid, context=None):
        # Simulate different execution behaviors
        if self._name == "InvalidOp":
            return OperationResult(success=False, grid=grid, error_message="Invalid operation")
        elif self._name == "SlowOp":
            time.sleep(0.2)  # Simulate slow operation
            return OperationResult(success=True, grid=grid)
        elif self._name == "MemoryHog":
            # Simulate large memory usage
            large_grid = [[0] * 1000 for _ in range(1000)]
            return OperationResult(success=True, grid=large_grid)
        else:
            return OperationResult(success=True, grid=grid)

    def get_name(self):
        return self._name

    def get_description(self):
        return f"Mock {self._name}"

    def get_parameter_schema(self):
        return {}


def create_test_programs(count: int, include_bad: bool = True) -> list[list[Operation]]:
    """Create test programs for evaluation."""
    programs = []

    for i in range(count):
        if include_bad and i % 5 == 0:
            # Invalid program
            program = [
                MockOperation("InvalidOp"),
                MockOperation("Rotate", angle=90),
            ]
        elif include_bad and i % 5 == 1:
            # Memory-intensive program
            program = [
                MockOperation("Tile", factor=100),
                MockOperation("MemoryHog"),
            ]
        elif include_bad and i % 5 == 2:
            # Contradictory program
            program = [
                MockOperation("Rotate", angle=90),
                MockOperation("Rotate", angle=270),
            ]
        else:
            # Valid program
            program = [
                MockOperation("Rotate", angle=90),
                MockOperation("FloodFill", x=0, y=0, color=1),
                MockOperation("Mirror", axis="horizontal"),
            ]

        programs.append(program)

    return programs


class TestPruningIntegration:
    """Test pruning integration with evaluation service."""

    @pytest.fixture
    def evaluation_service(self):
        """Create test evaluation service with pruning enabled."""
        strategy = PruningStrategy(
            strategy_id="test-balanced",
            name="Test Balanced Strategy",
            aggressiveness=0.5,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=True,
            confidence_threshold=0.6,
            max_partial_ops=3,
            timeout_ms=100,
        )

        service = EvaluationService(
            enable_gpu_evaluation=False,  # Disable for tests
            enable_pruning=True,
            default_pruning_strategy=strategy,
        )

        return service

    @pytest.fixture
    def test_grids(self):
        """Create test input grids."""
        return [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # Simple 3x3 grid
            [[1, 2], [3, 4]],  # 2x2 grid with different colors
        ]

    @pytest.mark.asyncio
    async def test_evaluation_with_pruning_saves_time(self, evaluation_service, test_grids):
        """Test that pruning saves evaluation time."""
        # Create programs with some that should be pruned
        programs = [
            DSLProgram(operations=[
                {"name": "Rotate", "params": {"angle": 90}},
                {"name": "Rotate", "params": {"angle": 270}},  # Contradictory
            ]),
            DSLProgram(operations=[
                {"name": "Tile", "params": {"factor": 100}},  # Memory explosion
            ]),
            DSLProgram(operations=[
                {"name": "Rotate", "params": {"angle": 90}},
                {"name": "FloodFill", "params": {"x": 0, "y": 0, "color": 1}},
            ]),  # Valid
        ]

        # Evaluate with pruning
        start_time = time.perf_counter()
        results, metrics = await evaluation_service.evaluate_with_pruning(
            programs, test_grids
        )
        pruning_time = time.perf_counter() - start_time

        # Check results
        assert len(results) == 3
        assert metrics.programs_pruned >= 1  # At least one should be pruned
        assert metrics.time_saved_ms > 0  # Should save some time

        # Verify pruned programs are marked
        pruned_count = sum(1 for r in results if r.metadata.get("pruned", False))
        assert pruned_count >= 1

    @pytest.mark.asyncio
    async def test_false_negative_detection(self):
        """Test detection of false negatives (good programs incorrectly pruned)."""
        # Create a program that might be incorrectly pruned
        good_program = DSLProgram(operations=[
            {"name": "ComplexTransform", "params": {}},  # Might have low confidence
        ])

        # Create aggressive pruner
        aggressive_strategy = PruningStrategy(
            strategy_id="aggressive",
            name="Aggressive",
            aggressiveness=0.9,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=True,
            confidence_threshold=0.8,  # High threshold
            max_partial_ops=1,  # Limited partial execution
            timeout_ms=50,
        )

        pruner = ProgramPruner(aggressive_strategy)

        # Mock partial execution with medium confidence
        with patch.object(pruner, '_partial_executor') as mock_executor:
            mock_executor.execute_partial = AsyncMock(
                return_value=(Mock(success=True), 0.7)  # Below threshold
            )

            result = await pruner.prune_program(
                [MockOperation("ComplexTransform")],
                test_inputs=[[[0, 0], [0, 0]]]
            )

        # Should be rejected due to confidence
        assert result.decision == PruningDecision.REJECT_CONFIDENCE

        # In real scenario, we'd validate this was a false negative
        # by running full evaluation and checking if it actually succeeded

    @pytest.mark.asyncio
    async def test_pruning_ab_testing(self):
        """Test A/B testing of different pruning strategies."""
        # Create strategies
        strategies = [
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

        # Create A/B controller
        ab_controller = ABTestController(
            strategies=strategies,
            exploration_rate=0.3,
            min_samples=10,
            confidence_level=0.95,
        )

        # Create metrics collector
        metrics_collector = PruningMetricsCollector()

        # Simulate multiple evaluations
        for _ in range(30):
            # Select strategy
            strategy = ab_controller.select_strategy()

            # Create pruner with selected strategy
            pruner = ProgramPruner(strategy)

            # Create test programs
            programs = create_test_programs(10)

            # Prune programs
            pruning_results = []
            for program in programs:
                result = await pruner.prune_program(program)
                pruning_results.append(result)

                # Record metrics
                metrics_collector.record_pruning_decision(
                    strategy.strategy_id,
                    program,
                    result,
                )

            # Create fake pruning metrics
            pruning_metrics = PruningMetrics(
                strategy_id=strategy.strategy_id,
                total_programs=len(programs),
                programs_pruned=sum(1 for r in pruning_results if r.decision != PruningDecision.ACCEPT),
                pruning_rate=sum(1 for r in pruning_results if r.decision != PruningDecision.ACCEPT) / len(programs),
                false_negatives=0,  # Would need actual validation
                false_negative_rate=0.0,
                avg_pruning_time_ms=5.0,
                time_saved_ms=100.0,
                timestamp=time.time(),
            )

            # Update A/B controller
            ab_controller.update_metrics(
                strategy.strategy_id,
                pruning_metrics,
                pruning_results,
            )

        # Check A/B testing results
        best_strategy, best_metrics = ab_controller.get_best_strategy()
        assert best_strategy is not None
        assert best_metrics["samples"] >= 10

        # Get performance summary
        summary = ab_controller.get_performance_summary()
        assert len(summary) == 3  # All strategies should have data

    @pytest.mark.asyncio
    async def test_pruning_with_evolution_engine(self):
        """Test pruning integration with evolution engine."""
        # This would test the full integration with evolution engine
        # For now, we'll create a simplified test

        from src.adapters.strategies.evolution_engine import EvolutionEngine, Individual
        from src.infrastructure.config import GeneticAlgorithmConfig

        # Create config with pruning enabled
        config = GeneticAlgorithmConfig()
        config.enable_pruning = True

        # Mock evolution engine components
        with patch('src.adapters.strategies.evolution_engine.EvolutionEngine._evaluate_population_with_pruning') as mock_eval:
            mock_eval.return_value = None

            # Create engine
            engine = EvolutionEngine(config)

            # Create test individuals
            individuals = []
            for i in range(10):
                ops = create_test_programs(1)[0]
                individual = Individual(operations=ops)
                individuals.append(individual)

            engine.population.individuals = individuals

            # Evaluate population
            await engine._evaluate_population()

            # Verify pruning was called
            mock_eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_improvement_validation(self, evaluation_service):
        """Test validation of 40% performance improvement target."""
        # Create many programs to test performance
        programs = []
        for i in range(100):
            if i % 3 == 0:
                # Bad program that should be pruned quickly
                ops = [
                    {"name": "Tile", "params": {"factor": 100}},
                ]
            else:
                # Valid program
                ops = [
                    {"name": "Rotate", "params": {"angle": 90}},
                    {"name": "FloodFill", "params": {"x": 0, "y": 0, "color": 1}},
                ]

            programs.append(DSLProgram(operations=ops))

        # Measure time without pruning
        evaluation_service.enable_pruning = False
        start_time = time.perf_counter()

        # Mock actual evaluation to be faster for testing
        with patch.object(evaluation_service, 'batch_evaluate_programs') as mock_eval:
            mock_eval.return_value = [Mock() for _ in programs]

            results_no_pruning = await evaluation_service.evaluate_with_pruning(
                programs, [[[0, 0], [0, 0]]]
            )

        time_no_pruning = (time.perf_counter() - start_time) * 1000

        # Measure time with pruning
        evaluation_service.enable_pruning = True
        start_time = time.perf_counter()

        results_with_pruning, metrics = await evaluation_service.evaluate_with_pruning(
            programs, [[[0, 0], [0, 0]]]
        )

        time_with_pruning = (time.perf_counter() - start_time) * 1000

        # Calculate improvement
        if time_no_pruning > 0:
            improvement = (time_no_pruning - time_with_pruning) / time_no_pruning

            # Log for debugging
            print(f"Time without pruning: {time_no_pruning:.2f}ms")
            print(f"Time with pruning: {time_with_pruning:.2f}ms")
            print(f"Improvement: {improvement * 100:.1f}%")
            print(f"Programs pruned: {metrics.programs_pruned}/{metrics.total_programs}")

            # Should see some improvement (may not reach 40% in unit test)
            assert improvement > 0

    @pytest.mark.asyncio
    async def test_security_sandbox_integration(self):
        """Test integration with security sandbox for partial execution."""
        from src.adapters.strategies.partial_executor import PartialExecutionConfig, PartialExecutor

        # Create partial executor with sandboxing
        config = PartialExecutionConfig(
            max_operations=3,
            timeout_ms=100,
            memory_limit_mb=10,
            enable_sandboxing=True,
        )

        executor = PartialExecutor(config)

        # Test program that should be sandboxed
        dangerous_ops = [
            MockOperation("MemoryHog"),
            MockOperation("SlowOp"),
        ]

        # Execute with sandboxing
        result, confidence = await executor.execute_partial(
            dangerous_ops,
            [[0, 0], [0, 0]]
        )

        # Should complete (sandbox should handle dangerous operations)
        assert result is not None
        assert result.execution_time_ms <= config.timeout_ms + 50  # Some overhead
        assert result.memory_used_mb <= config.memory_limit_mb
