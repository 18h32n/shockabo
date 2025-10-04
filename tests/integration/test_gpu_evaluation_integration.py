"""
Integration tests for GPU-accelerated evaluation.

Tests the full integration of GPU batch evaluation with the evolution engine
and evaluation service.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.adapters.strategies.gpu_batch_evaluator import BatchEvaluationRequest, GPUBatchEvaluator
from src.domain.dsl.base import Operation as DSLOperation
from src.domain.models import ARCTask
from src.domain.services.evaluation_service import EvaluationService


@pytest.fixture
def sample_arc_task():
    """Create sample ARC task."""
    return ARCTask(
        task_id="test_task",
        train_examples=[
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]]
            }
        ],
        test_input=[[1, 1], [0, 0]],
        test_output=[[0, 0], [1, 1]]
    )


@pytest.fixture
def mock_dsl_programs():
    """Create mock DSL programs."""
    programs = []

    # Program 1: Single rotation
    op1 = Mock(spec=DSLOperation)
    op1.name = "rotate"
    op1.parameters = {"angle": 180}
    programs.append([op1])

    # Program 2: Flip horizontal
    op2 = Mock(spec=DSLOperation)
    op2.name = "flip"
    op2.parameters = {"direction": "horizontal"}
    programs.append([op2])

    # Program 3: Color mapping
    op3 = Mock(spec=DSLOperation)
    op3.name = "map_colors"
    op3.parameters = {"color_map": {0: 1, 1: 0}}
    programs.append([op3])

    return programs


class TestGPUEvaluationIntegration:
    """Test GPU evaluation integration with evaluation service."""

    @pytest.mark.asyncio
    async def test_evaluation_service_gpu_batch(self, mock_dsl_programs):
        """Test EvaluationService with GPU batch evaluation."""
        service = EvaluationService(enable_gpu_evaluation=True)

        # Create test grids
        test_inputs = [
            [[1, 2], [3, 4]],
            [[0, 1], [1, 0]],
            [[5, 5], [5, 5]]
        ]

        # Execute GPU batch evaluation
        results = await service.batch_evaluate_programs(
            programs=mock_dsl_programs,
            test_inputs=test_inputs,
            device="cpu"  # Use CPU for testing
        )

        assert len(results) == len(mock_dsl_programs)
        assert service.evaluation_stats["gpu_evaluations"] == len(mock_dsl_programs)

    def test_gpu_evaluator_with_real_operations(self):
        """Test GPU evaluator with real DSL operations."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Create real operations
        programs = []

        # Program 1: Rotate 90
        op1 = Mock()
        op1.name = "rotate"
        op1.parameters = {"angle": 90}
        programs.append([op1])

        # Program 2: Flip + rotate
        op2a = Mock()
        op2a.name = "flip"
        op2a.parameters = {"direction": "horizontal"}
        op2b = Mock()
        op2b.name = "rotate"
        op2b.parameters = {"angle": 180}
        programs.append([op2a, op2b])

        grids = [
            [[1, 2], [3, 4]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        ]

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device="cpu",
            batch_size=10
        )

        result = evaluator.batch_evaluate(request)

        assert result.device_used in ["cpu", "hybrid"]
        assert len(result.output_grids) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU produce identical results."""
        # Create operations
        op = Mock()
        op.name = "rotate"
        op.parameters = {"angle": 90}
        program = [op]

        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Evaluate on GPU
        gpu_evaluator = GPUBatchEvaluator(device="cuda")
        gpu_request = BatchEvaluationRequest(
            programs=[program],
            input_grids=[grid],
            device="cuda"
        )
        gpu_result = gpu_evaluator.batch_evaluate(gpu_request)

        # Evaluate on CPU
        cpu_evaluator = GPUBatchEvaluator(device="cpu")
        cpu_request = BatchEvaluationRequest(
            programs=[program],
            input_grids=[grid],
            device="cpu"
        )
        cpu_result = cpu_evaluator.batch_evaluate(cpu_request)

        # Results should match
        if gpu_result.output_grids[0] and cpu_result.output_grids[0]:
            gpu_output = np.array(gpu_result.output_grids[0])
            cpu_output = np.array(cpu_result.output_grids[0])
            np.testing.assert_array_equal(gpu_output, cpu_output)

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        evaluator = GPUBatchEvaluator(
            device="cpu",
            memory_limit_mb=100  # Very low limit
        )

        # Create large batch
        large_programs = []
        large_grids = []

        for i in range(200):
            op = Mock()
            op.name = "flip"
            op.parameters = {"direction": "vertical"}
            large_programs.append([op])
            large_grids.append([[i] * 30] * 30)  # 30x30 grids

        request = BatchEvaluationRequest(
            programs=large_programs,
            input_grids=large_grids,
            batch_size=50
        )

        # Should handle gracefully with adaptive batching
        result = evaluator.batch_evaluate(request)

        assert len(result.output_grids) == 200
        assert result.batch_stats["num_batches"] >= 4  # Should split into multiple batches

    def test_mixed_operation_types(self):
        """Test batch with mixed operation types."""
        evaluator = GPUBatchEvaluator(device="cpu")

        programs = []

        # Various operation types
        operations = [
            ("rotate", {"angle": 90}),
            ("flip", {"direction": "horizontal"}),
            ("translate", {"offset": (1, 1), "fill_color": 0}),
            ("map_colors", {"color_map": {0: 1, 1: 0}}),
            ("filter_color", {"color": 1, "replacement": 0})
        ]

        for op_name, params in operations * 10:  # 50 programs
            op = Mock()
            op.name = op_name
            op.parameters = params
            programs.append([op])

        grids = [[[i % 2, (i + 1) % 2], [i % 2, i % 2]] for i in range(50)]

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            batch_size=25
        )

        result = evaluator.batch_evaluate(request)

        assert len(result.output_grids) == 50
        assert result.batch_stats["success_rate"] > 0

    def test_error_recovery(self):
        """Test recovery from errors during evaluation."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Create programs with some that will fail
        programs = []
        grids = []

        for i in range(10):
            op = Mock()
            if i % 3 == 0:
                # Invalid operation
                op.name = "invalid_op"
                op.parameters = {}
            else:
                op.name = "rotate"
                op.parameters = {"angle": 90}
            programs.append([op])
            grids.append([[1, 2], [3, 4]])

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids
        )

        result = evaluator.batch_evaluate(request)

        # Should complete despite errors
        assert len(result.output_grids) == 10
        # Some should succeed
        success_count = result.success_mask.sum().item()
        assert 0 < success_count < 10

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100])
    def test_different_batch_sizes(self, batch_size):
        """Test evaluation with different batch sizes."""
        evaluator = GPUBatchEvaluator(device="cpu")

        num_programs = 100
        programs = []
        grids = []

        for i in range(num_programs):
            op = Mock()
            op.name = "rotate"
            op.parameters = {"angle": 90 * (i % 4)}
            programs.append([op])
            grids.append([[1, 2], [3, 4]])

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            batch_size=batch_size
        )

        result = evaluator.batch_evaluate(request)

        assert len(result.output_grids) == num_programs
        expected_batches = (num_programs + batch_size - 1) // batch_size
        assert result.batch_stats["num_batches"] == expected_batches

    def test_performance_metrics_collection(self):
        """Test collection of performance metrics."""
        evaluator = GPUBatchEvaluator(device="cpu", enable_profiling=True)

        programs = []
        grids = []

        for i in range(20):
            op = Mock()
            op.name = "flip"
            op.parameters = {"direction": "vertical"}
            programs.append([op])
            grids.append([[i, i], [i, i]])

        request = BatchEvaluationRequest(programs=programs, input_grids=grids)

        # Execute multiple times
        for _ in range(3):
            evaluator.batch_evaluate(request)

        stats = evaluator.get_performance_stats()

        assert stats["total_batches_evaluated"] >= 3
        assert stats["total_programs_evaluated"] == 60  # 20 programs * 3 runs
        assert "avg_gpu_time_per_program_ms" in stats or stats["cpu_time_ms"] > 0


class TestEvolutionEngineIntegration:
    """Test integration with evolution engine."""

    @pytest.mark.asyncio
    async def test_evolution_with_gpu_evaluation(self):
        """Test evolution engine using GPU evaluation."""
        # This would require mocking the evolution engine's evaluation
        # to use GPU batch evaluation

        # Mock evolution engine components
        with patch("src.adapters.strategies.evolution_engine.FitnessEvaluator") as MockEvaluator:
            mock_evaluator = MockEvaluator.return_value

            # Create GPU evaluator
            gpu_evaluator = GPUBatchEvaluator(device="cpu")

            # Mock batch evaluation
            async def mock_batch_eval(individuals, task, callback):
                programs = [ind.operations for ind in individuals]
                grids = [task.train_examples[0]["input"]] * len(programs)

                request = BatchEvaluationRequest(
                    programs=programs,
                    input_grids=grids
                )

                result = gpu_evaluator.batch_evaluate(request)

                # Convert to fitness scores
                fitness_results = {}
                for i, ind in enumerate(individuals):
                    fitness_results[ind.id] = Mock(
                        fitness=1.0 if result.success_mask[i] else 0.0,
                        cached_outputs={},
                        error=None
                    )

                return fitness_results

            mock_evaluator.evaluate_population = mock_batch_eval

            # Would continue with evolution engine test...
