"""
Unit tests for GPU vectorized operations.

Tests the correctness and performance of vectorized DSL operations
implemented using PyTorch for GPU acceleration.
"""

import pytest
import torch

from src.utils.gpu_ops import VectorizedOps


@pytest.fixture
def device():
    """Get test device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def vectorized_ops(device):
    """Create VectorizedOps instance."""
    return VectorizedOps(device=device)


@pytest.fixture
def sample_batch():
    """Create sample batch of grids."""
    return torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    ], dtype=torch.float32)


class TestVectorizedRotations:
    """Test vectorized rotation operations."""

    def test_rotate_90_single(self, vectorized_ops):
        """Test 90 degree rotation on single grid."""
        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
        result = vectorized_ops.rotate_90(grid)
        expected = torch.tensor([[[3, 1], [4, 2]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_rotate_90_batch(self, vectorized_ops, sample_batch):
        """Test 90 degree rotation on batch."""
        result = vectorized_ops.rotate_90(sample_batch)

        # Check dimensions changed correctly
        assert result.shape[0] == sample_batch.shape[0]  # Batch size
        assert result.shape[1] == sample_batch.shape[2]  # Width becomes height
        assert result.shape[2] == sample_batch.shape[1]  # Height becomes width

        # Verify first grid rotation
        expected_first = torch.tensor([[7, 4, 1], [8, 5, 2], [9, 6, 3]], dtype=torch.float32)
        assert torch.allclose(result[0], expected_first)

    def test_rotate_180(self, vectorized_ops):
        """Test 180 degree rotation."""
        grid = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
        result = vectorized_ops.rotate_180(grid)
        expected = torch.tensor([[[6, 5, 4], [3, 2, 1]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_rotate_270(self, vectorized_ops):
        """Test 270 degree rotation."""
        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
        result = vectorized_ops.rotate_270(grid)
        expected = torch.tensor([[[2, 4], [1, 3]]], dtype=torch.float32)
        assert torch.allclose(result, expected)


class TestVectorizedFlips:
    """Test vectorized flip operations."""

    def test_flip_horizontal_batch(self, vectorized_ops, sample_batch):
        """Test horizontal flip on batch."""
        result = vectorized_ops.flip_horizontal(sample_batch)

        # Check first grid
        expected_first = torch.tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=torch.float32)
        assert torch.allclose(result[0], expected_first)

    def test_flip_vertical_batch(self, vectorized_ops, sample_batch):
        """Test vertical flip on batch."""
        result = vectorized_ops.flip_vertical(sample_batch)

        # Check first grid
        expected_first = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float32)
        assert torch.allclose(result[0], expected_first)

    def test_flip_diagonal(self, vectorized_ops):
        """Test diagonal flips."""
        grid = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)

        # Main diagonal (transpose)
        result_main = vectorized_ops.flip_main_diagonal(grid)
        expected_main = torch.tensor([[[1, 4], [2, 5], [3, 6]]], dtype=torch.float32)
        assert torch.allclose(result_main, expected_main)

        # Anti-diagonal
        result_anti = vectorized_ops.flip_anti_diagonal(grid)
        assert result_anti.shape == (1, 3, 2)  # Dimensions swapped


class TestVectorizedTranslation:
    """Test vectorized translation operations."""

    def test_translate_positive(self, vectorized_ops):
        """Test positive translation (right and down)."""
        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
        result = vectorized_ops.translate(grid, shift_y=1, shift_x=1, fill_value=0)
        expected = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_translate_negative(self, vectorized_ops):
        """Test negative translation (left and up)."""
        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
        result = vectorized_ops.translate(grid, shift_y=-1, shift_x=-1, fill_value=9)
        expected = torch.tensor([[[4, 9], [9, 9]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_translate_batch(self, vectorized_ops, sample_batch):
        """Test translation on batch."""
        result = vectorized_ops.translate(sample_batch, shift_y=0, shift_x=2, fill_value=0)

        # First row should have 2 zeros at start
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 1  # Original first element


class TestColorOperations:
    """Test color mapping and filtering operations."""

    def test_map_colors_simple(self, vectorized_ops):
        """Test simple color mapping."""
        grid = torch.tensor([[[0, 1, 2], [1, 2, 0]]], dtype=torch.float32)
        color_map = torch.tensor([9, 8, 7], dtype=torch.float32)

        result = vectorized_ops.map_colors(grid.long(), color_map)
        expected = torch.tensor([[[9, 8, 7], [8, 7, 9]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_map_colors_batch(self, vectorized_ops):
        """Test color mapping on batch."""
        batch = torch.tensor([
            [[0, 1], [2, 3]],
            [[3, 2], [1, 0]]
        ], dtype=torch.float32)

        # Map: 0->5, 1->6, 2->7, 3->8
        color_map = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

        result = vectorized_ops.map_colors(batch.long(), color_map)
        assert result[0, 0, 0] == 5
        assert result[0, 0, 1] == 6
        assert result[1, 1, 1] == 5

    def test_filter_by_color(self, vectorized_ops):
        """Test color filtering."""
        grid = torch.tensor([[[1, 2, 1], [2, 1, 2]]], dtype=torch.float32)

        result = vectorized_ops.filter_by_color(grid, target_color=1, replacement=0)
        expected = torch.tensor([[[1, 0, 1], [0, 1, 0]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_count_colors(self, vectorized_ops):
        """Test color counting."""
        batch = torch.tensor([
            [[0, 1, 1], [1, 0, 2]],
            [[3, 3, 3], [3, 3, 3]]
        ], dtype=torch.float32)

        counts = vectorized_ops.count_colors(batch)

        # First grid: 2 zeros, 3 ones, 1 two
        assert counts[0, 0] == 2
        assert counts[0, 1] == 3
        assert counts[0, 2] == 1

        # Second grid: all threes
        assert counts[1, 3] == 6


class TestRegionOperations:
    """Test region extraction and padding operations."""

    def test_extract_region(self, vectorized_ops):
        """Test region extraction."""
        grid = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)

        # Extract center 2x2
        result = vectorized_ops.extract_region(grid, y_start=0, y_end=2, x_start=1, x_end=3)
        expected = torch.tensor([[[2, 3], [5, 6]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_pad_grids(self, vectorized_ops):
        """Test grid padding."""
        grid = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)

        # Pad to 4x4
        result = vectorized_ops.pad_grids(grid, target_height=4, target_width=4, pad_value=0)
        assert result.shape == (1, 4, 4)
        assert result[0, 0, 0] == 1  # Original content preserved
        assert result[0, 3, 3] == 0  # Padded area

    def test_pad_grids_center(self, vectorized_ops):
        """Test center-aligned padding."""
        grid = torch.tensor([[[5]]], dtype=torch.float32)

        result = vectorized_ops.pad_grids(grid, 3, 3, pad_value=0, align="center")
        expected = torch.tensor([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]], dtype=torch.float32)
        assert torch.allclose(result, expected)


class TestAdvancedOperations:
    """Test advanced operations like pattern detection and composition."""

    def test_get_bounding_box(self, vectorized_ops):
        """Test bounding box detection."""
        batch = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32)

        y_mins, y_maxs, x_mins, x_maxs = vectorized_ops.get_bounding_box(batch)

        # First grid: single pixel at (1, 1)
        assert y_mins[0] == 1 and y_maxs[0] == 1
        assert x_mins[0] == 1 and x_maxs[0] == 1

        # Second grid: 2x2 region
        assert y_mins[1] == 0 and y_maxs[1] == 1
        assert x_mins[1] == 0 and x_maxs[1] == 1

    def test_apply_mask(self, vectorized_ops):
        """Test mask application."""
        grid = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
        mask = torch.tensor([[[1, 0, 1], [0, 1, 0]]], dtype=torch.bool)

        result = vectorized_ops.apply_mask(grid, mask, fill_value=0)
        expected = torch.tensor([[[1, 0, 3], [0, 5, 0]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_compose_grids_overlay(self, vectorized_ops):
        """Test grid composition with overlay mode."""
        background = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.float32)
        foreground = torch.tensor([[[0, 2], [2, 0]]], dtype=torch.float32)

        result = vectorized_ops.compose_grids(background, foreground, mode="overlay")
        expected = torch.tensor([[[1, 2], [2, 1]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_pattern_detection_exact(self, vectorized_ops):
        """Test exact pattern matching."""
        grid = torch.tensor([
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 0],
             [0, 1, 0, 1]]
        ], dtype=torch.float32)

        pattern = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        detections = vectorized_ops.detect_patterns(grid, pattern, threshold=1.0)

        # Pattern appears at (0,0), (0,2), (2,0), (2,2)
        assert detections.sum() >= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """Test GPU-specific performance characteristics."""

    def test_large_batch_performance(self, device):
        """Test performance with large batches."""
        ops = VectorizedOps(device=device)

        # Create large batch
        large_batch = torch.randn(100, 30, 30, device=device)

        # Time rotation operation
        import time
        start = time.time()
        result = ops.rotate_90(large_batch)
        gpu_time = time.time() - start

        # Should complete quickly
        assert gpu_time < 0.1  # 100ms for 100 grids
        assert result.shape == (100, 30, 30)

    def test_memory_efficiency(self, device):
        """Test memory usage stays within bounds."""
        ops = VectorizedOps(device=device)

        initial_memory = torch.cuda.memory_allocated()

        # Process medium batch
        batch = torch.randn(50, 30, 30, device=device)
        _ = ops.rotate_90(batch)
        _ = ops.flip_horizontal(batch)
        _ = ops.translate(batch, 5, 5)

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)

        # Should use less than 100MB for these operations
        assert memory_used_mb < 100
