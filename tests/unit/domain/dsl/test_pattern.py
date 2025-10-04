"""
Unit tests for pattern operations in the ARC DSL.

This module tests all pattern-based operations including pattern filling,
matching, replacement, and flood filling operations.
"""


import pytest

from src.domain.dsl.pattern import (
    FloodFillOperation,
    PatternFillOperation,
    PatternMatchOperation,
    PatternReplaceOperation,
)


class TestPatternFillOperation:
    """Test cases for PatternFillOperation."""

    def test_flood_fill_from_position(self):
        """Test flood fill starting from a specific position."""
        # Arrange
        grid = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 3, 3]
        ]
        operation = PatternFillOperation(start_position=(0, 0), target_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [5, 5, 1, 1],
            [5, 5, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 3, 3]
        ]
        assert result.grid == expected

    def test_flood_fill_all_regions_by_color(self):
        """Test flood fill of all regions of a specific color."""
        # Arrange
        grid = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 2, 2],
            [1, 0, 2, 2]
        ]
        operation = PatternFillOperation(source_color=0, target_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [9, 1, 9, 1],
            [1, 9, 1, 9],
            [9, 1, 2, 2],
            [1, 9, 2, 2]
        ]
        assert result.grid == expected

    def test_fill_single_cell(self):
        """Test filling a single isolated cell."""
        # Arrange
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        operation = PatternFillOperation(start_position=(1, 1), target_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1]
        ]
        assert result.grid == expected

    def test_fill_with_same_color(self):
        """Test filling with the same color (should not change anything)."""
        # Arrange
        grid = [[0, 1, 2], [1, 2, 0]]
        operation = PatternFillOperation(start_position=(0, 0), target_color=0)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid

    def test_empty_grid(self):
        """Test pattern fill with empty grid."""
        # Arrange
        grid = []
        operation = PatternFillOperation(start_position=(0, 0), target_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_parameters(self):
        """Test validation of operation parameters."""
        with pytest.raises(ValueError, match="Invalid target color"):
            PatternFillOperation(start_position=(0, 0), target_color=-1)

        with pytest.raises(ValueError, match="Invalid target color"):
            PatternFillOperation(start_position=(0, 0), target_color=10)

        with pytest.raises(ValueError, match="Invalid source color"):
            PatternFillOperation(source_color=-1, target_color=5)

        with pytest.raises(ValueError, match="Invalid source color"):
            PatternFillOperation(source_color=10, target_color=5)

    def test_missing_parameters(self):
        """Test that either start_position or source_color must be provided."""
        # Arrange
        grid = [[0, 1], [1, 0]]
        operation = PatternFillOperation(target_color=5)  # Neither start_position nor source_color

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "start_position or source_color" in result.error_message


class TestPatternMatchOperation:
    """Test cases for PatternMatchOperation."""

    def test_find_single_pattern_match(self):
        """Test finding a single pattern match."""
        # Arrange
        grid = [
            [0, 1, 2, 3],
            [4, 1, 1, 6],
            [7, 1, 1, 9],
            [8, 5, 4, 2]
        ]
        pattern = [[1, 1], [1, 1]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.metadata["matches"] == [(1, 1)]
        assert result.metadata["match_count"] == 1

    def test_find_multiple_pattern_matches(self):
        """Test finding multiple pattern matches."""
        # Arrange
        grid = [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 2, 2],
            [2, 2, 0, 2, 2]
        ]
        pattern = [[1, 1], [1, 1]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        matches = result.metadata["matches"]
        assert set(matches) == {(0, 0), (0, 3)}
        assert result.metadata["match_count"] == 2

    def test_pattern_not_found(self):
        """Test when pattern is not found in the grid."""
        # Arrange
        grid = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
        pattern = [[9, 9], [9, 9]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.metadata["matches"] == []
        assert result.metadata["match_count"] == 0

    def test_pattern_with_mask(self):
        """Test pattern matching with a mask (wildcards)."""
        # Arrange
        grid = [
            [1, 2, 3, 4],
            [5, 1, 7, 1],
            [9, 6, 1, 5],
            [2, 3, 4, 5]
        ]
        pattern = [[1, 2], [5, 1]]
        mask = [[True, False], [True, True]]  # Second element in first row is wildcard
        operation = PatternMatchOperation(pattern=pattern, mask=mask)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        matches = result.metadata["matches"]
        # Only (0, 0) should match: pattern [1,?] (where ? is wildcard) matches [1,2]
        # and pattern [5,1] matches [5,1]
        assert (0, 0) in matches
        # Note: (1, 2) would be [7,1] and [1,5] which doesn't match pattern [1,?],[5,1]

    def test_single_cell_pattern(self):
        """Test matching a single-cell pattern."""
        # Arrange
        grid = [
            [0, 1, 2],
            [1, 1, 0],
            [2, 0, 1]
        ]
        pattern = [[1]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        matches = result.metadata["matches"]
        expected_positions = [(0, 1), (1, 0), (1, 1), (2, 2)]
        assert set(matches) == set(expected_positions)
        assert result.metadata["match_count"] == 4

    def test_pattern_larger_than_grid(self):
        """Test pattern that is larger than the grid."""
        # Arrange
        grid = [[0, 1], [2, 3]]
        pattern = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.metadata["matches"] == []
        assert result.metadata["match_count"] == 0

    def test_empty_grid(self):
        """Test pattern matching with empty grid."""
        # Arrange
        grid = []
        pattern = [[1, 2]]
        operation = PatternMatchOperation(pattern=pattern)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.metadata["matches"] == []

    def test_invalid_pattern(self):
        """Test validation of pattern values."""
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            PatternMatchOperation(pattern=[])

        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            PatternMatchOperation(pattern=[[]])

        with pytest.raises(ValueError, match="Invalid pattern color"):
            PatternMatchOperation(pattern=[[-1, 5]])

        with pytest.raises(ValueError, match="Invalid pattern color"):
            PatternMatchOperation(pattern=[[5, 10]])


class TestPatternReplaceOperation:
    """Test cases for PatternReplaceOperation."""

    def test_basic_pattern_replacement(self):
        """Test basic pattern replacement functionality."""
        # Arrange
        grid = [
            [0, 1, 1, 3],
            [0, 1, 1, 3],
            [2, 2, 2, 2],
            [4, 5, 6, 7]
        ]
        source_pattern = [[1, 1], [1, 1]]
        target_pattern = [[9, 9], [9, 9]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [0, 9, 9, 3],
            [0, 9, 9, 3],
            [2, 2, 2, 2],
            [4, 5, 6, 7]
        ]
        assert result.grid == expected
        assert result.metadata["replacements"] == 1

    def test_multiple_pattern_replacements(self):
        """Test replacing multiple occurrences of a pattern."""
        # Arrange
        grid = [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 3, 3],
            [2, 2, 0, 3, 3]
        ]
        source_pattern = [[1, 1], [1, 1]]
        target_pattern = [[8, 9], [9, 8]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [8, 9, 0, 8, 9],
            [9, 8, 0, 9, 8],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 3, 3],
            [2, 2, 0, 3, 3]
        ]
        assert result.grid == expected
        assert result.metadata["replacements"] == 2

    def test_no_pattern_found(self):
        """Test replacement when pattern is not found."""
        # Arrange
        grid = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
        source_pattern = [[9, 9], [9, 9]]
        target_pattern = [[0, 0], [0, 0]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged
        assert result.metadata["replacements"] == 0

    def test_overlapping_patterns(self):
        """Test replacement with potentially overlapping patterns."""
        # Arrange
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        source_pattern = [[1, 1], [1, 1]]
        target_pattern = [[2, 3], [4, 5]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        # Should replace non-overlapping matches
        # The exact result depends on the order of scanning (left-to-right, top-to-bottom)
        assert result.metadata["replacements"] >= 1
        # Verify that some 2,3,4,5 values are present
        flat_result = [cell for row in result.grid for cell in row]
        assert any(cell in [2, 3, 4, 5] for cell in flat_result)

    def test_single_cell_pattern(self):
        """Test replacing single-cell patterns."""
        # Arrange
        grid = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ]
        source_pattern = [[1]]
        target_pattern = [[9]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [0, 9, 2],
            [9, 0, 9],
            [2, 9, 0]
        ]
        assert result.grid == expected
        assert result.metadata["replacements"] == 4

    def test_pattern_with_mask(self):
        """Test pattern replacement with mask."""
        # Arrange
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 9, 3, 0],
            [5, 6, 7, 1]
        ]
        source_pattern = [[1, 2], [5, 6]]
        target_pattern = [[8, 9], [7, 6]]
        mask = [[True, False], [True, True]]  # Second element of first row is wildcard
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            source_mask=mask
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        # Should replace patterns where mask matches
        expected = [
            [8, 9, 3, 4],  # [1,2],[5,6] -> [8,9],[7,6] (2 is wildcard)
            [7, 6, 7, 8],
            [8, 9, 3, 0],  # [1,9],[5,6] -> [8,9],[7,6] (9 matches wildcard)
            [7, 6, 7, 1]
        ]
        assert result.grid == expected

    def test_empty_grid(self):
        """Test pattern replacement with empty grid."""
        # Arrange
        grid = []
        source_pattern = [[1, 2]]
        target_pattern = [[3, 4]]
        operation = PatternReplaceOperation(
            source_pattern=source_pattern,
            target_pattern=target_pattern
        )

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == []

    def test_invalid_patterns(self):
        """Test validation of pattern parameters."""
        with pytest.raises(ValueError, match="Patterns cannot be empty"):
            PatternReplaceOperation(source_pattern=[], target_pattern=[[1]])

        with pytest.raises(ValueError, match="Patterns cannot be empty"):
            PatternReplaceOperation(source_pattern=[[1]], target_pattern=[])

        with pytest.raises(ValueError, match="same dimensions"):
            PatternReplaceOperation(
                source_pattern=[[1, 2]],
                target_pattern=[[3], [4]]
            )


class TestFloodFillOperation:
    """Test cases for FloodFillOperation."""

    def test_basic_flood_fill(self):
        """Test basic flood fill functionality."""
        # Arrange
        grid = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 3, 3]
        ]
        operation = FloodFillOperation(start_position=(0, 0), fill_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [5, 5, 1, 1],
            [5, 5, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 3, 3]
        ]
        assert result.grid == expected
        assert result.metadata["filled_cells"] == 4
        assert result.metadata["source_color"] == 0
        assert result.metadata["fill_color"] == 5

    def test_flood_fill_different_regions(self):
        """Test flood fill from different starting positions."""
        # Arrange
        grid = [
            [0, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ]

        # Test filling from top-left (0)
        operation1 = FloodFillOperation(start_position=(0, 0), fill_color=9)
        result1 = operation1.execute(grid)

        # Test filling from center (1)
        operation2 = FloodFillOperation(start_position=(1, 1), fill_color=8)
        result2 = operation2.execute(grid)

        # Test filling from bottom-right region (2)
        operation3 = FloodFillOperation(start_position=(2, 2), fill_color=7)
        result3 = operation3.execute(grid)

        # Assert
        assert all(r.success for r in [result1, result2, result3])

        expected1 = [
            [9, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ]
        assert result1.grid == expected1
        assert result1.metadata["filled_cells"] == 1

        expected2 = [
            [0, 8, 2],
            [8, 8, 2],
            [2, 2, 2]
        ]
        assert result2.grid == expected2
        assert result2.metadata["filled_cells"] == 3

        expected3 = [
            [0, 1, 7],
            [1, 1, 7],
            [7, 7, 7]
        ]
        assert result3.grid == expected3
        assert result3.metadata["filled_cells"] == 5

    def test_flood_fill_single_cell(self):
        """Test flood fill on a single isolated cell."""
        # Arrange
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        operation = FloodFillOperation(start_position=(1, 1), fill_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1]
        ]
        assert result.grid == expected
        assert result.metadata["filled_cells"] == 1

    def test_flood_fill_entire_grid(self):
        """Test flood fill that covers the entire grid."""
        # Arrange
        grid = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ]
        operation = FloodFillOperation(start_position=(1, 1), fill_color=7)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [7, 7, 7],
            [7, 7, 7],
            [7, 7, 7]
        ]
        assert result.grid == expected
        assert result.metadata["filled_cells"] == 9

    def test_flood_fill_same_color(self):
        """Test flood fill with same color as target (no change expected)."""
        # Arrange
        grid = [[0, 1, 2], [1, 2, 0]]
        operation = FloodFillOperation(start_position=(0, 0), fill_color=0)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        assert result.grid == grid  # Should be unchanged
        assert result.metadata["filled_cells"] == 0  # No cells actually filled

    def test_flood_fill_out_of_bounds_position(self):
        """Test flood fill with start position out of bounds."""
        # Arrange
        grid = [[0, 1], [1, 0]]
        operation = FloodFillOperation(start_position=(5, 5), fill_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "out of bounds" in result.error_message.lower()

    def test_flood_fill_negative_position(self):
        """Test flood fill with negative start position."""
        # Arrange
        grid = [[0, 1], [1, 0]]
        operation = FloodFillOperation(start_position=(-1, 0), fill_color=9)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "out of bounds" in result.error_message.lower()

    def test_empty_grid(self):
        """Test flood fill with empty grid."""
        # Arrange
        grid = []
        operation = FloodFillOperation(start_position=(0, 0), fill_color=5)

        # Act
        result = operation.execute(grid)

        # Assert
        assert not result.success
        assert "empty grid" in result.error_message.lower()

    def test_invalid_fill_color(self):
        """Test validation of fill color values."""
        with pytest.raises(ValueError, match="Invalid fill color"):
            FloodFillOperation(start_position=(0, 0), fill_color=-1)

        with pytest.raises(ValueError, match="Invalid fill color"):
            FloodFillOperation(start_position=(0, 0), fill_color=10)

    def test_flood_fill_complex_shape(self):
        """Test flood fill on a complex connected shape."""
        # Arrange - L-shaped region of 0s
        grid = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0]
        ]
        operation = FloodFillOperation(start_position=(0, 0), fill_color=7)

        # Act
        result = operation.execute(grid)

        # Assert
        assert result.success
        expected = [
            [7, 7, 1, 1, 1],
            [7, 7, 1, 1, 1],
            [7, 7, 7, 7, 7],
            [1, 1, 1, 1, 7],
            [1, 1, 1, 1, 7]
        ]
        assert result.grid == expected
        assert result.metadata["filled_cells"] == 11  # Correct count: 4 top-left + 5 bottom row + 2 bottom-right


class TestPatternOperationEdgeCases:
    """Test edge cases and integration scenarios for pattern operations."""

    def test_single_cell_grid_all_operations(self):
        """Test all pattern operations on single-cell grids."""
        grid = [[3]]

        # Test pattern fill
        fill_op = PatternFillOperation(start_position=(0, 0), target_color=7)
        result = fill_op.execute(grid)
        assert result.success and result.grid == [[7]]

        # Test pattern match
        match_op = PatternMatchOperation(pattern=[[3]])
        result = match_op.execute(grid)
        assert result.success and result.metadata["match_count"] == 1

        # Test pattern replace
        replace_op = PatternReplaceOperation(source_pattern=[[3]], target_pattern=[[8]])
        result = replace_op.execute(grid)
        assert result.success and result.grid == [[8]]

        # Test flood fill
        flood_op = FloodFillOperation(start_position=(0, 0), fill_color=6)
        result = flood_op.execute(grid)
        assert result.success and result.grid == [[6]]

    def test_operation_chaining_compatibility(self):
        """Test that operations can be chained together logically."""
        # Start with a simple grid
        grid = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]

        # First, replace all 1s with 2s
        replace_op = PatternReplaceOperation(source_pattern=[[1]], target_pattern=[[2]])
        result1 = replace_op.execute(grid)
        assert result1.success

        # Then flood fill from center
        flood_op = FloodFillOperation(start_position=(1, 1), fill_color=9)
        result2 = flood_op.execute(result1.grid)
        assert result2.success

        # Verify final state
        # After replacing 1->2: [[0,2,0], [2,0,2], [0,2,0]]
        # After flood fill from (1,1) which is 0: only center 0 becomes 9
        # The corner 0s are not connected to center due to diagonal connectivity
        expected = [
            [0, 2, 0],
            [2, 9, 2],
            [0, 2, 0]
        ]
        assert result2.grid == expected

    def test_large_grid_performance(self):
        """Test pattern operations on larger grids for basic performance."""
        # Create a 10x10 grid with some patterns
        grid = []
        for i in range(10):
            row = []
            for j in range(10):
                if (i + j) % 3 == 0:
                    row.append(1)
                elif (i + j) % 3 == 1:
                    row.append(2)
                else:
                    row.append(0)
            grid.append(row)

        # Test operations complete successfully
        operations = [
            PatternFillOperation(start_position=(0, 0), target_color=9),
            PatternMatchOperation(pattern=[[1, 2], [0, 1]]),
            PatternReplaceOperation(source_pattern=[[2]], target_pattern=[[8]]),
            FloodFillOperation(start_position=(5, 5), fill_color=7)
        ]

        for op in operations:
            result = op.execute(grid)
            assert result.success
            assert len(result.grid) == 10
            assert all(len(row) == 10 for row in result.grid)

    def test_operation_metadata_consistency(self):
        """Test that operations provide consistent and useful metadata."""
        grid = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]

        # Test pattern match metadata
        match_op = PatternMatchOperation(pattern=[[0, 1], [1, 0]])
        result = match_op.execute(grid)
        assert result.success
        assert "matches" in result.metadata
        assert "match_count" in result.metadata
        assert result.metadata["match_count"] == len(result.metadata["matches"])

        # Test pattern replace metadata
        replace_op = PatternReplaceOperation(
            source_pattern=[[0, 1]],
            target_pattern=[[9, 8]]
        )
        result = replace_op.execute(grid)
        assert result.success
        assert "replacements" in result.metadata
        assert isinstance(result.metadata["replacements"], int)

        # Test flood fill metadata
        flood_op = FloodFillOperation(start_position=(0, 0), fill_color=5)
        result = flood_op.execute(grid)
        assert result.success
        assert "filled_cells" in result.metadata
        assert "source_color" in result.metadata
        assert "fill_color" in result.metadata
        assert "start_position" in result.metadata
