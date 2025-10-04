"""
Unit tests for GPU batch evaluator.

Tests the GPU-accelerated batch evaluation system including vectorized operations,
memory management, and CPU fallback functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from src.adapters.strategies.gpu_batch_evaluator import (
    BatchEvaluationRequest,
    BatchEvaluationResult,
    GPUBatchEvaluator,
)
from src.domain.dsl.base import Operation as DSLOperation
from src.utils.gpu_ops import VectorizedOps


@pytest.fixture
def mock_dsl_operation():
    """Create mock DSL operation."""
    op = Mock(spec=DSLOperation)
    op.name = "rotate"
    op.parameters = {"angle": 90}
    return op


@pytest.fixture
def sample_grids():
    """Create sample grids for testing."""
    return [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[0, 1], [1, 0]],
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    ]


@pytest.fixture
def sample_programs(mock_dsl_operation):
    """Create sample DSL programs."""
    return [
        [mock_dsl_operation],
        [mock_dsl_operation, mock_dsl_operation],
        []
    ]


class TestGPUBatchEvaluator:
    """Test GPU batch evaluator functionality."""

    def test_initialization_auto_device(self):
        """Test automatic device selection."""
        evaluator = GPUBatchEvaluator(device="auto")
        assert evaluator.device.type in ["cuda", "cpu"]

    def test_initialization_cpu_fallback(self):
        """Test CPU fallback when CUDA not available."""
        with patch("torch.cuda.is_available", return_value=False):
            evaluator = GPUBatchEvaluator(device="cuda")
            assert evaluator.device.type == "cpu"

    def test_device_capabilities_detection(self):
        """Test GPU device capabilities detection."""
        if torch.cuda.is_available():
            evaluator = GPUBatchEvaluator(device="cuda")
            caps = evaluator.device_capabilities
            assert caps is not None
            assert caps.device_name is not None
            assert caps.total_memory_mb > 0
        else:
            evaluator = GPUBatchEvaluator(device="cpu")
            assert evaluator.device_capabilities is None

    def test_batch_creation(self, sample_programs, sample_grids):
        """Test batch creation from programs and grids."""
        evaluator = GPUBatchEvaluator()
        batches = evaluator._create_batches(
            sample_programs,
            sample_grids,
            batch_size=2
        )

        assert len(batches) == 2  # 3 programs with batch size 2
        assert len(batches[0][0]) == 2  # First batch has 2 programs
        assert len(batches[1][0]) == 1  # Second batch has 1 program

    def test_grids_to_tensors_conversion(self, sample_grids):
        """Test grid to tensor conversion."""
        evaluator = GPUBatchEvaluator(device="cpu")
        tensors = evaluator._grids_to_tensors(sample_grids)

        assert tensors.shape[0] == 3  # 3 grids
        assert tensors.shape[1] == 3  # Max height
        assert tensors.shape[2] == 3  # Max width
        assert tensors.device.type == "cpu"

    def test_tensors_to_grids_conversion(self):
        """Test tensor to grid conversion."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Create test tensor
        tensor = torch.tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
        ])
        success_mask = torch.tensor([True, True])

        grids = evaluator._tensors_to_grids(tensor, success_mask)

        assert len(grids) == 2
        assert grids[0] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert grids[1] == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]

    def test_memory_checking(self):
        """Test memory availability checking."""
        evaluator = GPUBatchEvaluator(device="cpu", memory_limit_mb=1000)

        # CPU always returns True
        assert evaluator._check_memory_available(100) is True

        # Mock GPU memory checking
        if torch.cuda.is_available():
            evaluator.device = torch.device("cuda")
            # Should work with reasonable batch size
            assert evaluator._check_memory_available(10) is True

    def test_adaptive_batch_sizing(self, sample_grids):
        """Test adaptive batch size calculation."""
        evaluator = GPUBatchEvaluator(
            device="cpu",
            max_batch_size=100,
            memory_limit_mb=1000
        )

        grid_sizes = [(len(g), len(g[0])) for g in sample_grids]
        batch_size = evaluator._calculate_adaptive_batch_size(
            num_programs=50,
            grid_sizes=grid_sizes
        )

        assert batch_size > 0
        assert batch_size <= 50  # Can't exceed number of programs
        assert batch_size <= 100  # Can't exceed max batch size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_evaluation(self, sample_programs, sample_grids):
        """Test GPU evaluation of programs."""
        evaluator = GPUBatchEvaluator(device="cuda")
        request = BatchEvaluationRequest(
            programs=sample_programs[:1],
            input_grids=sample_grids[:1],
            device="cuda",
            batch_size=1
        )

        result = evaluator.batch_evaluate(request)

        assert isinstance(result, BatchEvaluationResult)
        assert result.device_used in ["cuda", "hybrid"]
        assert len(result.output_grids) == 1

    def test_cpu_fallback_evaluation(self, sample_programs, sample_grids):
        """Test CPU fallback evaluation."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Mock the sandbox executor
        with patch("src.adapters.strategies.sandbox_executor.SandboxExecutor") as MockExecutor:
            mock_executor = MockExecutor.return_value
            mock_result = Mock()
            mock_result.success = True
            mock_result.output = [[1, 2], [3, 4]]
            mock_result.execution_time = 0.1
            mock_executor.execute_operations.return_value = mock_result

            request = BatchEvaluationRequest(
                programs=sample_programs[:1],
                input_grids=sample_grids[:1],
                device="cuda",  # Request CUDA but expect fallback to CPU
                batch_size=1
            )

            result = evaluator.batch_evaluate(request)

            assert result.device_used == "cpu"
            assert len(result.output_grids) == 1
            assert result.batch_stats["fallback"] is True

    def test_hybrid_execution(self, sample_programs, sample_grids):
        """Test hybrid GPU/CPU execution."""
        evaluator = GPUBatchEvaluator(device="auto")

        # Mock memory check to force some batches to CPU
        original_check = evaluator._check_memory_available
        evaluator._check_memory_available = lambda size: size < 2

        with patch("src.adapters.strategies.sandbox_executor.SandboxExecutor"):
            request = BatchEvaluationRequest(
                programs=sample_programs,
                input_grids=sample_grids,
                device="auto",
                batch_size=2
            )

            result = evaluator.batch_evaluate(request)

            # Should have used both devices
            assert result.batch_stats["device_usage"]["cpu"] > 0

        evaluator._check_memory_available = original_check

    def test_performance_stats_tracking(self, sample_programs, sample_grids):
        """Test performance statistics tracking."""
        evaluator = GPUBatchEvaluator(device="cpu")

        with patch("src.adapters.strategies.sandbox_executor.SandboxExecutor"):
            request = BatchEvaluationRequest(
                programs=sample_programs,
                input_grids=sample_grids,
                device="cpu"
            )

            evaluator.batch_evaluate(request)

            stats = evaluator.get_performance_stats()
            assert stats["total_batches_evaluated"] > 0
            assert stats["total_programs_evaluated"] == len(sample_programs)

    def test_operation_dispatch(self):
        """Test vectorized operation dispatch."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Test rotation operation
        tensor = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)

        # Create rotation operation
        rotate_op = Mock()
        rotate_op.name = "rotate"
        rotate_op.parameters = {"angle": 90}

        mask = torch.tensor([True])

        result = evaluator._apply_vectorized_operations(
            tensor,
            [rotate_op],
            mask
        )

        # Result should be rotated
        assert result.shape == tensor.shape

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        evaluator = GPUBatchEvaluator(device="cpu")

        request = BatchEvaluationRequest(
            programs=[],
            input_grids=[],
            device="cpu"
        )

        result = evaluator.batch_evaluate(request)

        assert len(result.output_grids) == 0
        assert result.batch_stats["total_programs"] == 0

    def test_error_handling_in_batch(self, sample_programs, sample_grids):
        """Test error handling during batch evaluation."""
        evaluator = GPUBatchEvaluator(device="cpu")

        # Force an error in evaluation
        evaluator._evaluate_batch_gpu = Mock(side_effect=Exception("Test error"))
        evaluator._evaluate_batch_cpu = Mock(side_effect=Exception("Test error"))

        request = BatchEvaluationRequest(
            programs=sample_programs,
            input_grids=sample_grids,
            device="auto"
        )

        result = evaluator.batch_evaluate(request)

        # Should handle errors gracefully
        assert len(result.output_grids) == len(sample_programs)
        assert all(grid is None for grid in result.output_grids)


class TestVectorizedOps:
    """Test vectorized GPU operations."""

    def test_rotation_operations(self):
        """Test rotation operations."""
        ops = VectorizedOps(device=torch.device("cpu"))

        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)

        # Test 90 degree rotation
        rotated_90 = ops.rotate_90(grid)
        expected_90 = torch.tensor([[[3, 1], [4, 2]]], dtype=torch.float32)
        assert torch.allclose(rotated_90, expected_90)

        # Test 180 degree rotation
        rotated_180 = ops.rotate_180(grid)
        expected_180 = torch.tensor([[[4, 3], [2, 1]]], dtype=torch.float32)
        assert torch.allclose(rotated_180, expected_180)

    def test_flip_operations(self):
        """Test flip operations."""
        ops = VectorizedOps(device=torch.device("cpu"))

        grid = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)

        # Test horizontal flip
        flipped_h = ops.flip_horizontal(grid)
        expected_h = torch.tensor([[[3, 2, 1], [6, 5, 4]]], dtype=torch.float32)
        assert torch.allclose(flipped_h, expected_h)

        # Test vertical flip
        flipped_v = ops.flip_vertical(grid)
        expected_v = torch.tensor([[[4, 5, 6], [1, 2, 3]]], dtype=torch.float32)
        assert torch.allclose(flipped_v, expected_v)

    def test_color_operations(self):
        """Test color mapping operations."""
        ops = VectorizedOps(device=torch.device("cpu"))

        grid = torch.tensor([[[0, 1, 2], [1, 2, 0]]], dtype=torch.float32)
        color_map = torch.tensor([5, 6, 7], dtype=torch.float32)

        mapped = ops.map_colors(grid, color_map)
        expected = torch.tensor([[[5, 6, 7], [6, 7, 5]]], dtype=torch.float32)
        assert torch.allclose(mapped, expected)

    def test_translation_operation(self):
        """Test grid translation."""
        ops = VectorizedOps(device=torch.device("cpu"))

        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)

        # Translate right by 1, down by 1
        translated = ops.translate(grid, shift_y=1, shift_x=1, fill_value=0)
        expected = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.float32)
        assert torch.allclose(translated, expected)

    def test_pattern_detection(self):
        """Test pattern detection operation."""
        ops = VectorizedOps(device=torch.device("cpu"))

        grid = torch.tensor([
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 0]]
        ], dtype=torch.float32)

        pattern = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        detections = ops.detect_patterns(grid, pattern, threshold=1.0)

        # Should detect pattern in multiple locations
        assert detections.sum() > 0
