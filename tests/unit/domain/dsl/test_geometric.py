"""
Unit tests for geometric transformation operations.
"""

import pytest

from src.domain.dsl.geometric import (
    CropOperation,
    FlipOperation,
    PadOperation,
    RotateOperation,
    TranslateOperation,
)


class TestRotateOperation:
    """Test suite for RotateOperation."""

    def test_rotate_90_clockwise(self):
        """Test 90-degree clockwise rotation."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = RotateOperation(angle=90)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ]

    def test_rotate_180(self):
        """Test 180-degree rotation."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = RotateOperation(angle=180)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]

    def test_rotate_270_clockwise(self):
        """Test 270-degree clockwise rotation."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = RotateOperation(angle=270)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [3, 6, 9],
            [2, 5, 8],
            [1, 4, 7]
        ]

    def test_rotate_rectangular_grid(self):
        """Test rotating a non-square grid."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        op = RotateOperation(angle=90)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [5, 1],
            [6, 2],
            [7, 3],
            [8, 4]
        ]

    def test_rotate_invalid_angle(self):
        """Test rotation with invalid angle."""
        grid = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            RotateOperation(angle=45)

    def test_rotate_empty_grid(self):
        """Test rotating an empty grid."""
        op = RotateOperation(angle=90)
        result = op.execute([])

        assert not result.success
        assert "empty grid" in result.error_message.lower()


class TestFlipOperation:
    """Test suite for FlipOperation."""

    def test_flip_horizontal(self):
        """Test horizontal flip (left-right mirror)."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = FlipOperation(direction="horizontal")
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ]

    def test_flip_vertical(self):
        """Test vertical flip (up-down mirror)."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = FlipOperation(direction="vertical")
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ]

    def test_flip_diagonal_main(self):
        """Test main diagonal flip (transpose)."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = FlipOperation(direction="diagonal_main")
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]

    def test_flip_diagonal_anti(self):
        """Test anti-diagonal flip."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = FlipOperation(direction="diagonal_anti")
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [9, 6, 3],
            [8, 5, 2],
            [7, 4, 1]
        ]

    def test_flip_invalid_direction(self):
        """Test flip with invalid direction."""
        with pytest.raises(ValueError):
            FlipOperation(direction="invalid")

    def test_flip_rectangular_grid(self):
        """Test flipping a non-square grid."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        op = FlipOperation(direction="horizontal")
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [4, 3, 2, 1],
            [8, 7, 6, 5]
        ]


class TestTranslateOperation:
    """Test suite for TranslateOperation."""

    def test_translate_right_down(self):
        """Test translating right and down."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = TranslateOperation(offset=(1, 1), fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [0, 0, 0],
            [0, 1, 2],
            [0, 4, 5]
        ]

    def test_translate_left_up(self):
        """Test translating left and up."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = TranslateOperation(offset=(-1, -1), fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [5, 6, 0],
            [8, 9, 0],
            [0, 0, 0]
        ]

    def test_translate_out_of_bounds(self):
        """Test translating completely out of bounds."""
        grid = [[1, 2], [3, 4]]
        op = TranslateOperation(offset=(5, 5), fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[0, 0], [0, 0]]

    def test_translate_custom_fill_color(self):
        """Test translate with custom fill color."""
        grid = [[1, 2], [3, 4]]
        op = TranslateOperation(offset=(1, 1), fill_color=9)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[9, 9], [9, 1]]


class TestCropOperation:
    """Test suite for CropOperation."""

    def test_crop_center(self):
        """Test cropping center region."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6]
        ]
        op = CropOperation(top=1, left=1, bottom=2, right=2)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [6, 7],
            [0, 1]
        ]
        assert result.metadata["original_size"] == (4, 4)
        assert result.metadata["new_size"] == (2, 2)

    def test_crop_full_grid(self):
        """Test cropping entire grid (no change)."""
        grid = [[1, 2], [3, 4]]
        op = CropOperation(top=0, left=0, bottom=1, right=1)
        result = op.execute(grid)

        assert result.success
        assert result.grid == grid

    def test_crop_single_cell(self):
        """Test cropping to single cell."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        op = CropOperation(top=1, left=1, bottom=1, right=1)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[5]]

    def test_crop_out_of_bounds(self):
        """Test cropping with out-of-bounds indices."""
        grid = [[1, 2], [3, 4]]
        op = CropOperation(top=0, left=0, bottom=5, right=5)
        result = op.execute(grid)

        assert not result.success
        assert "out of range" in result.error_message

    def test_crop_invalid_bounds(self):
        """Test cropping with invalid bounds (top > bottom)."""
        grid = [[1, 2], [3, 4]]
        op = CropOperation(top=1, left=0, bottom=0, right=1)
        result = op.execute(grid)

        assert not result.success
        assert "Invalid crop bounds" in result.error_message


class TestPadOperation:
    """Test suite for PadOperation."""

    def test_pad_all_sides(self):
        """Test padding on all sides."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ]
        assert result.metadata["original_size"] == (2, 2)
        assert result.metadata["new_size"] == (4, 4)

    def test_pad_asymmetric(self):
        """Test asymmetric padding."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(top=0, bottom=2, left=1, right=0, fill_color=9)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [9, 1, 2],
            [9, 3, 4],
            [9, 9, 9],
            [9, 9, 9]
        ]

    def test_pad_zero_padding(self):
        """Test padding with zero values (no change)."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(top=0, bottom=0, left=0, right=0)
        result = op.execute(grid)

        assert result.success
        assert result.grid == grid

    def test_pad_single_side(self):
        """Test padding only one side."""
        grid = [[1, 2], [3, 4]]
        op = PadOperation(top=2, bottom=0, left=0, right=0, fill_color=5)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [5, 5],
            [5, 5],
            [1, 2],
            [3, 4]
        ]

    def test_pad_with_default_fill_color(self):
        """Test padding with default fill color (0)."""
        grid = [[1]]
        op = PadOperation(top=1, bottom=1, left=1, right=1)
        result = op.execute(grid)

        assert result.success
        assert result.grid == [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]


class TestOperationChaining:
    """Test chaining multiple geometric operations."""

    def test_rotate_then_flip(self):
        """Test chaining rotation and flip operations."""
        grid = [[1, 2], [3, 4]]

        rotate = RotateOperation(angle=90)
        flip = FlipOperation(direction="horizontal")

        # Use the >> operator to chain
        composite = rotate >> flip
        result = composite.execute(grid)

        assert result.success
        # After 90° rotation: [[3, 1], [4, 2]]
        # After horizontal flip: [[1, 3], [2, 4]]
        assert result.grid == [[1, 3], [2, 4]]

    def test_translate_crop_pad_chain(self):
        """Test complex chain of translate, crop, and pad."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        translate = TranslateOperation(offset=(1, 1), fill_color=0)
        crop = CropOperation(top=1, left=1, bottom=2, right=2)
        pad = PadOperation(top=1, bottom=0, left=0, right=1, fill_color=0)

        composite = translate >> crop >> pad
        result = composite.execute(grid)

        assert result.success
        # After translate: [[0,0,0], [0,1,2], [0,4,5]]
        # After crop: [[1,2], [4,5]]
        # After pad: [[0,0,0], [1,2,0], [4,5,0]]
        assert result.grid == [[0, 0, 0], [1, 2, 0], [4, 5, 0]]
