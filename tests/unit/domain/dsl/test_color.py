"""
Unit tests for color operations in the ARC DSL.

This module tests all color manipulation operations including mapping, filtering,
replacement, inversion, and thresholding operations.
"""


import pytest

from src.domain.dsl.color import (
    ColorFilterOperation,
    ColorInvertOperation,
    ColorMapOperation,
    ColorReplaceOperation,
    ColorThresholdOperation,
)


class TestColorMapOperation:
    """Test cases for ColorMapOperation."""

    def test_basic_color_mapping(self):
        """Test basic color mapping functionality."""
        # Arrange
        grid = [[0, 1, 2], [1, 2, 0]]
        mapping = {0: 9, 1: 8, 2: 7}
        operation = ColorMapOperation(mapping=mapping)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 8, 7], [8, 7, 9]]
        assert result.grid == expected

    def test_partial_mapping(self):
        """Test color mapping with only some colors mapped."""
        # Arrange
        grid = [[0, 1, 2, 3], [4, 5, 6, 7]]
        mapping = {0: 9, 2: 8}  # Only map 0->9 and 2->8
        operation = ColorMapOperation(mapping=mapping)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 1, 8, 3], [4, 5, 6, 7]]
        assert result.grid == expected

    def test_mapping_with_default_color(self):
        """Test color mapping with default color for unmapped values."""
        # Arrange
        grid = [[0, 1, 2, 3], [4, 5, 6, 7]]
        mapping = {0: 9, 2: 8}
        operation = ColorMapOperation(mapping=mapping, default_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 5, 8, 5], [5, 5, 5, 5]]
        assert result.grid == expected

    def test_empty_grid(self):
        """Test color mapping with empty grid."""
        # Arrange
        grid = []
        mapping = {0: 9}
        operation = ColorMapOperation(mapping=mapping)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_source_color(self):
        """Test validation of source color values."""
        with pytest.raises(ValueError, match="Invalid source color"):
            ColorMapOperation(mapping={-1: 5})

        with pytest.raises(ValueError, match="Invalid source color"):
            ColorMapOperation(mapping={10: 5})

    def test_invalid_target_color(self):
        """Test validation of target color values."""
        with pytest.raises(ValueError, match="Invalid target color"):
            ColorMapOperation(mapping={0: -1})

        with pytest.raises(ValueError, match="Invalid target color"):
            ColorMapOperation(mapping={0: 10})

    def test_invalid_default_color(self):
        """Test validation of default color values."""
        with pytest.raises(ValueError, match="Invalid default color"):
            ColorMapOperation(mapping={0: 1}, default_color=-1)

        with pytest.raises(ValueError, match="Invalid default color"):
            ColorMapOperation(mapping={0: 1}, default_color=10)

    def test_self_mapping(self):
        """Test mapping a color to itself."""
        # Arrange
        grid = [[0, 1, 2], [3, 4, 5]]
        mapping = {1: 1, 3: 3}  # Map to same colors
        operation = ColorMapOperation(mapping=mapping)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged


class TestColorFilterOperation:
    """Test cases for ColorFilterOperation."""

    def test_basic_color_filtering(self):
        """Test basic color filtering functionality."""
        # Arrange
        grid = [[0, 1, 2, 3], [4, 5, 6, 7]]
        keep_colors = [1, 3, 5]
        operation = ColorFilterOperation(keep_colors=keep_colors, fill_color=0)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[0, 1, 0, 3], [0, 5, 0, 0]]
        assert result.grid == expected

    def test_filter_with_different_fill_color(self):
        """Test filtering with non-zero fill color."""
        # Arrange
        grid = [[0, 1, 2], [3, 4, 5]]
        keep_colors = [1, 4]
        operation = ColorFilterOperation(keep_colors=keep_colors, fill_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 1, 9], [9, 4, 9]]
        assert result.grid == expected

    def test_filter_keep_all_colors(self):
        """Test filtering that keeps all present colors."""
        # Arrange
        grid = [[0, 1], [2, 3]]
        keep_colors = [0, 1, 2, 3, 4, 5]
        operation = ColorFilterOperation(keep_colors=keep_colors, fill_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged

    def test_filter_keep_no_colors(self):
        """Test filtering that keeps no present colors."""
        # Arrange
        grid = [[0, 1], [2, 3]]
        keep_colors = [7, 8, 9]
        operation = ColorFilterOperation(keep_colors=keep_colors, fill_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[5, 5], [5, 5]]
        assert result.grid == expected

    def test_empty_grid(self):
        """Test color filtering with empty grid."""
        # Arrange
        grid = []
        operation = ColorFilterOperation(keep_colors=[1, 2], fill_color=0)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_keep_color(self):
        """Test validation of keep color values."""
        with pytest.raises(ValueError, match="Invalid keep color"):
            ColorFilterOperation(keep_colors=[-1, 5], fill_color=0)

        with pytest.raises(ValueError, match="Invalid keep color"):
            ColorFilterOperation(keep_colors=[5, 10], fill_color=0)

    def test_invalid_fill_color(self):
        """Test validation of fill color values."""
        with pytest.raises(ValueError, match="Invalid fill color"):
            ColorFilterOperation(keep_colors=[1, 2], fill_color=-1)

        with pytest.raises(ValueError, match="Invalid fill color"):
            ColorFilterOperation(keep_colors=[1, 2], fill_color=10)


class TestColorReplaceOperation:
    """Test cases for ColorReplaceOperation."""

    def test_basic_color_replacement(self):
        """Test basic color replacement functionality."""
        # Arrange
        grid = [[0, 1, 2, 1], [1, 0, 1, 2]]
        operation = ColorReplaceOperation(source_color=1, target_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[0, 9, 2, 9], [9, 0, 9, 2]]
        assert result.grid == expected

    def test_replace_nonexistent_color(self):
        """Test replacing a color that doesn't exist in the grid."""
        # Arrange
        grid = [[0, 1, 2], [3, 4, 5]]
        operation = ColorReplaceOperation(source_color=8, target_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged

    def test_replace_with_same_color(self):
        """Test replacing a color with itself."""
        # Arrange
        grid = [[0, 1, 2], [1, 1, 0]]
        operation = ColorReplaceOperation(source_color=1, target_color=1)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged

    def test_replace_all_occurrences(self):
        """Test that all occurrences are replaced."""
        # Arrange
        grid = [[5, 5, 5], [5, 5, 5]]
        operation = ColorReplaceOperation(source_color=5, target_color=2)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[2, 2, 2], [2, 2, 2]]
        assert result.grid == expected

    def test_empty_grid(self):
        """Test color replacement with empty grid."""
        # Arrange
        grid = []
        operation = ColorReplaceOperation(source_color=1, target_color=2)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_source_color(self):
        """Test validation of source color values."""
        with pytest.raises(ValueError, match="Invalid source color"):
            ColorReplaceOperation(source_color=-1, target_color=5)

        with pytest.raises(ValueError, match="Invalid source color"):
            ColorReplaceOperation(source_color=10, target_color=5)

    def test_invalid_target_color(self):
        """Test validation of target color values."""
        with pytest.raises(ValueError, match="Invalid target color"):
            ColorReplaceOperation(source_color=0, target_color=-1)

        with pytest.raises(ValueError, match="Invalid target color"):
            ColorReplaceOperation(source_color=0, target_color=10)


class TestColorInvertOperation:
    """Test cases for ColorInvertOperation."""

    def test_basic_color_inversion(self):
        """Test basic color inversion functionality."""
        # Arrange
        grid = [[0, 1, 2], [7, 8, 9]]
        operation = ColorInvertOperation()

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 8, 7], [2, 1, 0]]
        assert result.grid == expected

    def test_invert_symmetric_colors(self):
        """Test inverting colors that map to themselves or pairs."""
        # Arrange
        grid = [[4, 5], [5, 4]]  # 4->5, 5->4
        operation = ColorInvertOperation()

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[5, 4], [4, 5]]
        assert result.grid == expected

    def test_invert_middle_values(self):
        """Test inverting with middle values."""
        # Arrange
        grid = [[0, 4, 9], [3, 6, 1]]
        operation = ColorInvertOperation()

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[9, 5, 0], [6, 3, 8]]
        assert result.grid == expected

    def test_double_inversion(self):
        """Test that double inversion returns to original."""
        # Arrange
        grid = [[0, 1, 2, 3], [4, 5, 6, 7]]
        operation = ColorInvertOperation()

        # Act
        result1 = operation.execute(grid)
        result2 = operation.execute(result1.grid)

        # Assert
        assert result1.success and result2.success
        assert result2.grid == grid

    def test_empty_grid(self):
        """Test color inversion with empty grid."""
        # Arrange
        grid = []
        operation = ColorInvertOperation()

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()


class TestColorThresholdOperation:
    """Test cases for ColorThresholdOperation."""

    def test_basic_thresholding(self):
        """Test basic color thresholding functionality."""
        # Arrange
        grid = [[0, 1, 2, 3], [4, 5, 6, 7]]
        operation = ColorThresholdOperation(threshold=4, low_color=0, high_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[0, 0, 0, 0], [9, 9, 9, 9]]
        assert result.grid == expected

    def test_threshold_at_boundary(self):
        """Test thresholding with values at the threshold boundary."""
        # Arrange
        grid = [[3, 4, 5], [4, 4, 6]]
        operation = ColorThresholdOperation(threshold=4, low_color=1, high_color=8)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[1, 8, 8], [8, 8, 8]]
        assert result.grid == expected

    def test_threshold_all_low(self):
        """Test thresholding where all values are below threshold."""
        # Arrange
        grid = [[0, 1, 2], [1, 2, 0]]
        operation = ColorThresholdOperation(threshold=5, low_color=7, high_color=3)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[7, 7, 7], [7, 7, 7]]
        assert result.grid == expected

    def test_threshold_all_high(self):
        """Test thresholding where all values are above threshold."""
        # Arrange
        grid = [[5, 6, 7], [8, 9, 6]]
        operation = ColorThresholdOperation(threshold=3, low_color=1, high_color=4)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[4, 4, 4], [4, 4, 4]]
        assert result.grid == expected

    def test_threshold_with_defaults(self):
        """Test thresholding with default low/high colors."""
        # Arrange
        grid = [[2, 3, 4], [5, 6, 7]]
        operation = ColorThresholdOperation(threshold=4)  # defaults: low=0, high=9

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [[0, 0, 9], [9, 9, 9]]
        assert result.grid == expected

    def test_empty_grid(self):
        """Test color thresholding with empty grid."""
        # Arrange
        grid = []
        operation = ColorThresholdOperation(threshold=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_threshold(self):
        """Test validation of threshold values."""
        with pytest.raises(ValueError, match="Invalid threshold"):
            ColorThresholdOperation(threshold=-1)

        with pytest.raises(ValueError, match="Invalid threshold"):
            ColorThresholdOperation(threshold=10)

    def test_invalid_low_color(self):
        """Test validation of low color values."""
        with pytest.raises(ValueError, match="Invalid low_color"):
            ColorThresholdOperation(threshold=5, low_color=-1)

        with pytest.raises(ValueError, match="Invalid low_color"):
            ColorThresholdOperation(threshold=5, low_color=10)

    def test_invalid_high_color(self):
        """Test validation of high color values."""
        with pytest.raises(ValueError, match="Invalid high_color"):
            ColorThresholdOperation(threshold=5, high_color=-1)

        with pytest.raises(ValueError, match="Invalid high_color"):
            ColorThresholdOperation(threshold=5, high_color=10)


class TestColorOperationEdgeCases:
    """Test edge cases and integration scenarios for color operations."""

    def test_single_cell_grid(self):
        """Test operations on single-cell grids."""
        grid = [[5]]

        # Test mapping
        map_op = ColorMapOperation(mapping={5: 2})
        result = map_op.execute(grid)
        assert result.success and result.grid == [[2]]

        # Test filter
        filter_op = ColorFilterOperation(keep_colors=[5], fill_color=0)
        result = filter_op.execute(grid)
        assert result.success and result.grid == [[5]]

        # Test replace
        replace_op = ColorReplaceOperation(source_color=5, target_color=8)
        result = replace_op.execute(grid)
        assert result.success and result.grid == [[8]]

        # Test invert
        invert_op = ColorInvertOperation()
        result = invert_op.execute(grid)
        assert result.success and result.grid == [[4]]

        # Test threshold
        thresh_op = ColorThresholdOperation(threshold=3, low_color=0, high_color=9)
        result = thresh_op.execute(grid)
        assert result.success and result.grid == [[9]]

    def test_large_grid_performance(self):
        """Test operations on larger grids for basic performance."""
        # Create a 10x10 grid
        grid = [[i % 10 for i in range(10)] for _ in range(10)]

        # Test that operations complete successfully
        operations = [
            ColorMapOperation(mapping={0: 9, 1: 8}),
            ColorFilterOperation(keep_colors=[0, 1, 2], fill_color=7),
            ColorReplaceOperation(source_color=5, target_color=3),
            ColorInvertOperation(),
            ColorThresholdOperation(threshold=5)
        ]

        for op in operations:
            result = op.execute(grid)
            assert result.success
            assert len(result.grid) == 10
            assert all(len(row) == 10 for row in result.grid)

    def test_operation_metadata(self):
        """Test that operations provide useful metadata."""
        grid = [[1, 2, 3], [4, 5, 6]]

        # Most operations don't provide metadata in current implementation
        # This test ensures the operations at least don't break when metadata is expected
        operations = [
            ColorMapOperation(mapping={1: 9}),
            ColorFilterOperation(keep_colors=[1, 2], fill_color=0),
            ColorReplaceOperation(source_color=3, target_color=7),
            ColorInvertOperation(),
            ColorThresholdOperation(threshold=3)
        ]

        for op in operations:
            result = op.execute(grid)
            assert result.success
            # Metadata may be None, which is fine
            if result.metadata is not None:
                assert isinstance(result.metadata, dict)
