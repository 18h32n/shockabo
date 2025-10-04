"""
Integration tests for evolution strategy with evaluation framework.

Tests Task 5 implementation: integration of evolution engine with
evaluation service, strategy interface, and unified submission format.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.evolution_strategy_adapter import EvolutionStrategyAdapter
from src.domain.models import ARCTask
from src.domain.services.dsl_engine import DSLEngine
from src.domain.services.evaluation_service import EvaluationService
from src.domain.services.submission_handler import SubmissionHandler
from src.infrastructure.config import GeneticAlgorithmConfig


@pytest.fixture
def evaluation_service():
    """Create evaluation service instance."""
    return EvaluationService()


@pytest.fixture
def submission_handler(tmp_path):
    """Create submission handler with temp directory."""
    return SubmissionHandler(output_dir=tmp_path / "submissions")


@pytest.fixture
def simple_arc_task():
    """Create simple ARC task for testing."""
    return ARCTask(
        task_id="test_evolution_001",
        task_source="test",
        train_examples=[
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]]
            },
            {
                "input": [[2, 3], [3, 2]],
                "output": [[3, 2], [2, 3]]
            }
        ],
        test_input=[[4, 5], [5, 4]],
        test_output=[[5, 4], [4, 5]]  # Ground truth for evaluation
    )


@pytest.fixture
def mock_dsl_engine():
    """Create mock DSL engine for testing."""
    mock = Mock(spec=DSLEngine)

    # Mock execute_program to return correct output sometimes
    call_count = 0
    def mock_execute(operations, grid):
        nonlocal call_count
        call_count += 1

        # Return correct answer 20% of the time to simulate evolution
        if call_count % 5 == 0:
            # Return inverted grid (correct answer for our test task)
            return {
                "success": True,
                "output": [[grid[1][0], grid[0][1]], [grid[1][1], grid[0][0]]]
            }
        else:
            # Return incorrect output
            return {
                "success": True,
                "output": [[0, 0], [0, 0]]
            }

    mock.execute_program.side_effect = mock_execute
    return mock


@pytest.fixture
def evolution_config():
    """Create minimal evolution config for fast testing."""
    config = GeneticAlgorithmConfig()
    config.population.size = 10
    config.convergence.max_generations = 3
    config.convergence.stagnation_patience = 2
    config.parallelization.workers = 1
    config.parallelization.batch_size = 5
    config.performance.generation_timeout = 5
    return config


class TestEvolutionEvaluationIntegration:
    """Test integration of evolution strategy with evaluation framework."""

    def test_evolution_adapter_implements_interface(self):
        """Test that EvolutionStrategyAdapter implements StrategyInterface."""
        # Check that adapter follows the protocol
        assert hasattr(EvolutionStrategyAdapter, 'solve_task')
        assert hasattr(EvolutionStrategyAdapter, 'get_strategy_info')
        assert hasattr(EvolutionStrategyAdapter, 'get_strategy_type')

        # Verify it's a proper strategy implementation
        from src.domain.services.strategy_interface import BaseStrategy
        assert issubclass(EvolutionStrategyAdapter, BaseStrategy)

    @pytest.mark.asyncio
    async def test_evolution_returns_evaluation_result(
        self,
        simple_arc_task,
        mock_dsl_engine,
        evaluation_service,
        evolution_config
    ):
        """Test that evolution adapter returns proper EvaluationResult."""
        # Create adapter
        adapter = EvolutionStrategyAdapter(
            config=evolution_config,
            dsl_engine=mock_dsl_engine,
            evaluation_service=evaluation_service
        )

        # Run task solving
        result = await adapter.solve_task(
            task=simple_arc_task,
            max_attempts=2,
            experiment_name="test_evolution"
        )

        # Verify result structure
        assert result.task_id == simple_arc_task.task_id
        assert result.strategy_used == "evolution"
        assert len(result.attempts) <= 2

        # Check metadata contains evolution-specific metrics
        metadata = result.metadata
        assert "total_programs_generated" in metadata
        assert "generations_run" in metadata
        assert "mutation_success_rate" in metadata
        assert "crossover_success_rate" in metadata
        assert "genealogy_depth" in metadata
        assert "evolution_time_seconds" in metadata
        assert "submission_format" in metadata

    @pytest.mark.asyncio
    async def test_evolution_accuracy_calculation(
        self,
        simple_arc_task,
        mock_dsl_engine,
        evaluation_service,
        evolution_config
    ):
        """Test that evolution results are properly evaluated for accuracy."""
        adapter = EvolutionStrategyAdapter(
            config=evolution_config,
            dsl_engine=mock_dsl_engine,
            evaluation_service=evaluation_service
        )

        # Force mock to return correct answer
        mock_dsl_engine.execute_program.return_value = {
            "success": True,
            "output": [[5, 4], [4, 5]]  # Correct answer
        }

        result = await adapter.solve_task(simple_arc_task)

        # Check accuracy calculation
        assert result.best_attempt is not None
        assert result.best_attempt.pixel_accuracy.perfect_match is True
        assert result.final_accuracy == 1.0

    def test_unified_submission_format(
        self,
        simple_arc_task,
        evaluation_service,
        submission_handler
    ):
        """Test unified submission format creation."""
        # Create mock evaluation result
        from src.domain.services.evaluation_service import (
            AttemptNumber,
            EvaluationResult,
            PixelAccuracy,
            TaskMetrics,
        )

        pixel_acc = PixelAccuracy(
            total_pixels=4,
            correct_pixels=4,
            accuracy=1.0,
            perfect_match=True
        )

        attempt = TaskMetrics(
            task_id=simple_arc_task.task_id,
            attempt_number=AttemptNumber.FIRST,
            pixel_accuracy=pixel_acc,
            processing_time_ms=100.0,
            confidence_score=0.95
        )

        eval_result = EvaluationResult(
            task_id=simple_arc_task.task_id,
            strategy_used="evolution",
            attempts=[attempt],
            metadata={"strategy_type": "evolution"}
        )

        # Create submission from evaluation result
        # Note: In practice, predicted_output would need to be passed along
        eval_result.attempts[0].error_details['predicted_output'] = [[5, 4], [4, 5]]

        submission = submission_handler.create_submission_from_evaluation(eval_result)

        # Verify submission format
        assert submission.task_id == simple_arc_task.task_id
        assert submission.strategy == "evolution"
        assert len(submission.predictions) == 1
        assert submission.confidence_scores[0] == 0.95

        # Test export to competition format
        comp_format = submission.to_competition_format()
        assert simple_arc_task.task_id in comp_format
        assert comp_format[simple_arc_task.task_id] == submission.predictions

    @pytest.mark.asyncio
    async def test_multiple_strategies_integration(
        self,
        simple_arc_task,
        mock_dsl_engine,
        evaluation_service,
        evolution_config,
        submission_handler
    ):
        """Test integration with multiple strategies using same interfaces."""
        # Create evolution adapter
        evolution_adapter = EvolutionStrategyAdapter(
            config=evolution_config,
            dsl_engine=mock_dsl_engine,
            evaluation_service=evaluation_service
        )

        # Mock a different strategy adapter
        class MockTTTAdapter:
            def __init__(self, evaluation_service):
                self.evaluation_service = evaluation_service

            async def solve_task(self, task, max_attempts=2, experiment_name=None):
                # Return a simple evaluation result
                predictions = [([[1, 1], [1, 1]], 0.8)]
                return self.evaluation_service.evaluate_task_with_attempts(
                    task=task,
                    predictions=predictions,
                    strategy_used="ttt",
                    metadata={"strategy_type": "ttt"}
                )

            def get_strategy_type(self):
                return "ttt"

            def get_strategy_info(self):
                return {"name": "TTT", "type": "ttt"}

        ttt_adapter = MockTTTAdapter(evaluation_service)

        # Run both strategies
        evolution_result = await evolution_adapter.solve_task(simple_arc_task)
        ttt_result = await ttt_adapter.solve_task(simple_arc_task)

        # Create submissions for both
        submission_handler.create_submission(
            task_id=simple_arc_task.task_id,
            predictions=[[[5, 4], [4, 5]]],
            confidence_scores=[0.9],
            strategy="evolution",
            metadata=evolution_result.metadata
        )

        submission_handler.create_submission(
            task_id=simple_arc_task.task_id + "_ttt",
            predictions=[[[1, 1], [1, 1]]],
            confidence_scores=[0.8],
            strategy="ttt",
            metadata=ttt_result.metadata
        )

        # Export all submissions
        export_path = submission_handler.export_detailed_format()

        # Verify export
        assert export_path.exists()
        summary = submission_handler.get_submission_summary()
        assert summary["total_tasks"] == 2
        assert set(summary["strategies_used"]) == {"evolution", "ttt"}

    @pytest.mark.asyncio
    async def test_evolution_specific_metrics_tracking(
        self,
        simple_arc_task,
        mock_dsl_engine,
        evaluation_service,
        evolution_config
    ):
        """Test that evolution-specific metrics are properly tracked."""
        adapter = EvolutionStrategyAdapter(
            config=evolution_config,
            dsl_engine=mock_dsl_engine,
            evaluation_service=evaluation_service
        )

        # Patch the evolution engine to track operator success
        with patch.object(adapter, 'evolution_engine') as mock_engine:
            # Set up mock evolution stats
            mock_engine.evolve.return_value = (
                Mock(fitness=0.9, operations=[], id="test_id"),
                {
                    "total_programs_generated": 150,
                    "generations": 5,
                    "best_fitness": 0.9,
                    "mutation_success_rate": 0.35,
                    "crossover_success_rate": 0.42,
                    "successful_mutations": [
                        {"generation": 2, "type": "operation_replacement", "improvement": 0.1},
                        {"generation": 4, "type": "operation_insertion", "improvement": 0.05}
                    ],
                    "max_genealogy_depth": 8,
                    "average_lineage_length": 4.5
                }
            )

            mock_engine.all_individuals_history = []

            result = await adapter.solve_task(simple_arc_task)

            # Verify evolution-specific metrics are included
            metadata = result.metadata
            assert metadata["mutation_success_rate"] == 0.35
            assert metadata["crossover_success_rate"] == 0.42
            assert len(metadata["successful_mutations"]) == 2
            assert metadata["genealogy_depth"] == 8

    def test_strategy_info_completeness(
        self,
        evaluation_service,
        evolution_config
    ):
        """Test that strategy info provides complete information."""
        adapter = EvolutionStrategyAdapter(
            config=evolution_config,
            dsl_engine=Mock(),
            evaluation_service=evaluation_service
        )

        info = adapter.get_strategy_info()

        # Check required fields
        assert "name" in info
        assert "type" in info
        assert "version" in info
        assert "capabilities" in info
        assert "configuration" in info

        # Check evolution-specific capabilities
        capabilities = info["capabilities"]
        assert capabilities["multi_attempt"] is True
        assert capabilities["parallel_evaluation"] is True
        assert capabilities["genealogy_tracking"] is True
        assert capabilities["diversity_preservation"] is True

        # Check configuration details
        config = info["configuration"]
        assert config["population_size"] == evolution_config.population.size
        assert config["target_programs"] == 500
        assert config["time_limit_seconds"] == 300


class TestEvolutionSubmissionExport:
    """Test submission export functionality."""

    @pytest.mark.asyncio
    async def test_batch_task_processing(
        self,
        evaluation_service,
        submission_handler,
        evolution_config,
        tmp_path
    ):
        """Test processing multiple tasks and exporting results."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            tasks.append(ARCTask(
                task_id=f"batch_test_{i}",
                task_source="test",
                train_examples=[
                    {"input": [[i]], "output": [[i+1]]}
                ],
                test_input=[[i+2]],
                test_output=[[i+3]]
            ))

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.solve_task = asyncio.coroutine(
            lambda task, **kwargs: evaluation_service.evaluate_task_with_attempts(
                task=task,
                predictions=[([[task.test_output[0][0]]], 0.95)],
                strategy_used="evolution",
                metadata={"task_id": task.task_id}
            )
        )

        # Process all tasks
        results = []
        for task in tasks:
            result = await mock_adapter.solve_task(task)
            results.append(result)

            # Create submission
            submission_handler.create_submission(
                task_id=task.task_id,
                predictions=[[[task.test_output[0][0]]]],
                confidence_scores=[0.95],
                strategy="evolution"
            )

        # Export in competition format
        comp_path = submission_handler.export_competition_format("competition.json")

        # Export detailed format
        detailed_path = submission_handler.export_detailed_format("detailed.json")

        # Verify exports
        assert comp_path.exists()
        assert detailed_path.exists()

        # Check competition format
        import json
        with open(comp_path) as f:
            comp_data = json.load(f)

        assert len(comp_data) == 3
        for i, task in enumerate(tasks):
            assert task.task_id in comp_data
            assert comp_data[task.task_id] == [[[i+3]]]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
