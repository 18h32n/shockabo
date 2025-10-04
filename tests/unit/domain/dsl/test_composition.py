"""
Unit tests for composition operations.
"""

import pytest

from src.domain.dsl.composition import ConcatenateOperation, CropOperation, OverlayOperation, PadOperation


class TestCropOperation:
    """Test suite for CropOperation (composition version)."""

    def test_crop_with_dimensions(self):
        """Test cropping with specified dimensions."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6]
        ]
        op = CropOperation(top_left=(1, 1), dimensions=(2, 2))
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[6, 7], [0, 1]]

    def test_crop_with_bottom_right(self):
        """Test cropping with bottom-right corner."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = CropOperation(top_left=(0, 0), bottom_right=(1, 1))
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[1, 2], [4, 5]]

    def test_crop_full_grid(self):
        """Test cropping that returns the full grid."""
        grid = [[1, 2], [3, 4]]
        op = CropOperation(top_left=(0, 0), bottom_right=(1, 1))
        result = op.execute(grid)

        assert result.success
        assert result.grid == grid

    def test_crop_single_cell(self):
        """Test cropping to a single cell."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        op = CropOperation(top_left=(1, 1), dimensions=(1, 1))
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[5]]

    def test_crop_out_of_bounds(self):
        """Test cropping with out-of-bounds region."""
        grid = [[1, 2], [3, 4]]
        op = CropOperation(top_left=(0, 0), bottom_right=(5, 5))
        result = op.execute(grid)

        assert not result.success
        assert "out of bounds" in result.error_message

    def test_crop_invalid_params(self):
        """Test crop with invalid parameter combinations."""
        # Neither dimensions nor bottom_right specified
        with pytest.raises(ValueError):
            CropOperation(top_left=(0, 0))

        # Both dimensions and bottom_right specified
        with pytest.raises(ValueError):
            CropOperation(top_left=(0, 0), dimensions=(2, 2), bottom_right=(1, 1))

    def test_crop_empty_grid(self):
        """Test cropping an empty grid."""
        op = CropOperation(top_left=(0, 0), dimensions=(1, 1))
        result = op.execute([])

        assert not result.success
        assert "empty grid" in result.error_message


class TestPadOperation:
    """Test suite for PadOperation (composition version)."""

    def test_pad_uniform(self):
        """Test uniform padding on all sides."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(padding=1, fill_color=0)
        result = op.execute(grid)

        assert result.success
        expected = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ]
        assert result.grid == expected

    def test_pad_asymmetric(self):
        """Test asymmetric padding."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(padding=(1, 2, 3, 0), fill_color=9)  # top, right, bottom, left
        result = op.execute(grid)

        assert result.success
        expected = [
            [9, 9, 9, 9],  # 1 top row
            [1, 2, 9, 9],  # Original + 2 right
            [3, 4, 9, 9],  # Original + 2 right
            [9, 9, 9, 9],  # 3 bottom rows
            [9, 9, 9, 9],
            [9, 9, 9, 9]
        ]
        assert result.grid == expected

    def test_pad_zero_padding(self):
        """Test padding with zero values (no change)."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(padding=0, fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == grid

    def test_pad_large_padding(self):
        """Test with large padding values."""
        grid = [[5]]
        op = PadOperation(padding=3, fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert len(result.grid) == 7  # 1 + 3 + 3
        assert len(result.grid[0]) == 7
        assert result.grid[3][3] == 5  # Original value in center

    def test_pad_custom_fill_color(self):
        """Test padding with custom fill color."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(padding=1, fill_color=7)
        result = op.execute(grid)

        assert result.success
        # Check corners are filled with color 7
        assert result.grid[0][0] == 7
        assert result.grid[0][3] == 7
        assert result.grid[3][0] == 7
        assert result.grid[3][3] == 7

    def test_pad_invalid_padding(self):
        """Test padding with invalid values."""
        # Negative padding
        with pytest.raises(ValueError):
            PadOperation(padding=-1)

        # Invalid tuple length
        with pytest.raises(ValueError):
            PadOperation(padding=(1, 2, 3))  # Should be 4 values

        # Negative values in tuple
        with pytest.raises(ValueError):
            PadOperation(padding=(1, -1, 1, 1))

    def test_pad_invalid_fill_color(self):
        """Test padding with invalid fill color."""
        with pytest.raises(ValueError):
            PadOperation(padding=1, fill_color=10)

        with pytest.raises(ValueError):
            PadOperation(padding=1, fill_color=-1)

    def test_pad_empty_grid(self):
        """Test padding an empty grid."""
        op = PadOperation(padding=1, fill_color=0)
        result = op.execute([])

        assert not result.success
        assert "empty grid" in result.error_message


class TestOverlayOperation:
    """Test suite for OverlayOperation."""

    def test_overlay_simple(self):
        """Test simple overlay operation."""
        base_grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        overlay_grid = [
            [1, 2],
            [3, 4]
        ]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(1, 1))
        result = op.execute(base_grid)

        assert result.success
        expected = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ]
        assert result.grid == expected

    def test_overlay_with_transparency(self):
        """Test overlay with transparent color."""
        base_grid = [
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]
        ]
        overlay_grid = [
            [1, 0, 2],
            [0, 3, 0]
        ]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(0, 0), transparent_color=0)
        result = op.execute(base_grid)

        assert result.success
        expected = [
            [1, 5, 2],  # 0s are transparent
            [5, 3, 5],
            [5, 5, 5]
        ]
        assert result.grid == expected

    def test_overlay_partial_out_of_bounds(self):
        """Test overlay that extends beyond grid bounds."""
        base_grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        overlay_grid = [
            [2, 2],
            [2, 2]
        ]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(2, 2))
        result = op.execute(base_grid)

        assert result.success
        expected = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2]  # Only one cell overlaid
        ]
        assert result.grid == expected

    def test_overlay_at_origin(self):
        """Test overlay at origin position."""
        base_grid = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        overlay_grid = [
            [1, 2]
        ]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(0, 0))
        result = op.execute(base_grid)

        assert result.success
        expected = [
            [1, 2, 0],
            [0, 0, 0]
        ]
        assert result.grid == expected

    def test_overlay_on_empty_base(self):
        """Test overlay on empty base grid."""
        overlay_grid = [[1, 2], [3, 4]]

        # At origin should work
        op = OverlayOperation(overlay_grid=overlay_grid, position=(0, 0))
        result = op.execute([])
        assert result.success
        assert result.grid == overlay_grid

        # At non-origin should fail
        op2 = OverlayOperation(overlay_grid=overlay_grid, position=(1, 1))
        result2 = op2.execute([])
        assert not result2.success

    def test_overlay_completely_out_of_bounds(self):
        """Test overlay completely outside grid bounds."""
        base_grid = [[1, 1], [1, 1]]
        overlay_grid = [[2, 2], [2, 2]]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(5, 5))
        result = op.execute(base_grid)

        assert result.success
        assert result.grid == base_grid  # No changes

    def test_overlay_negative_position(self):
        """Test overlay with negative position."""
        base_grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        overlay_grid = [
            [2, 2],
            [2, 2]
        ]
        op = OverlayOperation(overlay_grid=overlay_grid, position=(-1, -1))
        result = op.execute(base_grid)

        assert result.success
        expected = [
            [2, 1, 1],  # Only bottom-right of overlay visible
            [1, 1, 1],
            [1, 1, 1]
        ]
        assert result.grid == expected

    def test_overlay_invalid_params(self):
        """Test overlay with invalid parameters."""
        # Empty overlay grid
        with pytest.raises(ValueError):
            OverlayOperation(overlay_grid=[])

        # Invalid transparent color
        with pytest.raises(ValueError):
            OverlayOperation(overlay_grid=[[1]], transparent_color=10)


class TestConcatenateOperation:
    """Test suite for ConcatenateOperation."""

    def test_concatenate_horizontal(self):
        """Test horizontal concatenation of two grids."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]
        op = ConcatenateOperation(other_grid=grid2, direction="horizontal")
        result = op.execute(grid1)

        assert result.success
        expected = [[1, 2, 5, 6], [3, 4, 7, 8]]
        assert result.grid == expected

    def test_concatenate_vertical(self):
        """Test vertical concatenation of two grids."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]
        op = ConcatenateOperation(other_grid=grid2, direction="vertical")
        result = op.execute(grid1)

        assert result.success
        expected = [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert result.grid == expected

    def test_concatenate_different_sizes(self):
        """Test concatenation with different sized grids."""
        grid1 = [[1, 2]]
        grid2 = [[3], [4]]
        op = ConcatenateOperation(other_grid=grid2, direction="horizontal", fill_color=0)
        result = op.execute(grid1)

        assert result.success
        expected = [[1, 2, 3], [0, 0, 4]]
        assert result.grid == expected

    def test_concatenate_invalid_direction(self):
        """Test concatenation with invalid direction."""
        grid2 = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            ConcatenateOperation(other_grid=grid2, direction="invalid")

    def test_concatenate_empty_other_grid(self):
        """Test concatenation with empty other grid."""
        with pytest.raises(ValueError):
            ConcatenateOperation(other_grid=[])
