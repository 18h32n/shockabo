"""
Unit tests for symmetry operations.
"""

import pytest

from src.domain.dsl.symmetry import CreateSymmetryOperation


class TestCreateSymmetryOperation:
    """Test suite for CreateSymmetryOperation."""

    def test_horizontal_symmetry_left_source(self):
        """Test creating horizontal symmetry from left half."""
        grid = [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [5, 6, 0, 0]
        ]
        op = CreateSymmetryOperation(axis="horizontal", source_half="left")
        result = op.execute(grid)

        assert result.success
        expected = [
            [1, 2, 2, 1],
            [3, 4, 4, 3],
            [5, 6, 6, 5]
        ]
        assert result.grid == expected

    def test_horizontal_symmetry_right_source(self):
        """Test creating horizontal symmetry from right half."""
        grid = [
            [0, 0, 1, 2],
            [0, 0, 3, 4],
            [0, 0, 5, 6]
        ]
        op = CreateSymmetryOperation(axis="horizontal", source_half="right")
        result = op.execute(grid)

        assert result.success
        expected = [
            [2, 1, 1, 2],
            [4, 3, 3, 4],
            [6, 5, 5, 6]
        ]
        assert result.grid == expected

    def test_vertical_symmetry_top_source(self):
        """Test creating vertical symmetry from top half."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        op = CreateSymmetryOperation(axis="vertical", source_half="top")
        result = op.execute(grid)

        assert result.success
        expected = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [1, 2, 3, 4]
        ]
        assert result.grid == expected

    def test_vertical_symmetry_bottom_source(self):
        """Test creating vertical symmetry from bottom half."""
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        op = CreateSymmetryOperation(axis="vertical", source_half="bottom")
        result = op.execute(grid)

        assert result.success
        expected = [
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        assert result.grid == expected

    def test_diagonal_main_symmetry(self):
        """Test creating main diagonal symmetry."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        op = CreateSymmetryOperation(axis="diagonal_main", source_half="left")
        result = op.execute(grid)

        assert result.success
        # Should transpose the grid
        expected = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]
        assert result.grid == expected

    def test_rotational_symmetry_4fold(self):
        """Test creating 4-fold rotational symmetry."""
        grid = [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        op = CreateSymmetryOperation(axis="rotational", order=4)
        result = op.execute(grid)

        assert result.success
        expected = [
            [1, 2, 2, 1],
            [3, 4, 4, 3],
            [3, 4, 4, 3],
            [1, 2, 2, 1]
        ]
        assert result.grid == expected

    def test_symmetry_odd_dimensions(self):
        """Test symmetry operations on odd-dimensioned grids."""
        grid = [
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 0]
        ]
        op = CreateSymmetryOperation(axis="horizontal", source_half="left")
        result = op.execute(grid)

        assert result.success
        # Middle column should remain unchanged
        expected = [
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 5]
        ]
        assert result.grid == expected

    def test_symmetry_empty_grid(self):
        """Test symmetry on empty grid."""
        op = CreateSymmetryOperation(axis="horizontal", source_half="left")
        result = op.execute([])

        assert not result.success
        assert "empty grid" in result.error_message

    def test_symmetry_invalid_axis(self):
        """Test symmetry with invalid axis."""
        with pytest.raises(ValueError):
            CreateSymmetryOperation(axis="invalid")

    def test_symmetry_invalid_source_half(self):
        """Test symmetry with invalid source half."""
        with pytest.raises(ValueError):
            CreateSymmetryOperation(axis="horizontal", source_half="invalid")

    def test_symmetry_invalid_order(self):
        """Test rotational symmetry with invalid order."""
        with pytest.raises(ValueError):
            CreateSymmetryOperation(axis="rotational", order=1)
