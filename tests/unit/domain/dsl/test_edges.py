"""
Unit tests for edge and boundary detection operations.
"""

from src.domain.dsl.edges import (
    BoundaryTracingOperation,
    ContourExtractionOperation,
    EdgeDetectionOperation,
)


class TestEdgeDetectionOperation:
    """Test suite for EdgeDetectionOperation."""

    def test_edge_detection_simple(self):
        """Test basic edge detection."""
        grid = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ]
        op = EdgeDetectionOperation(edge_color=9)
        result = op.execute(grid)

        assert result.success
        # Edges should be at color boundaries
        expected = [
            [0, 9, 9, 0],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [0, 9, 9, 0]
        ]
        assert result.grid == expected
        assert result.metadata["edge_count"] == 12

    def test_edge_detection_specific_transition(self):
        """Test edge detection for specific color transitions."""
        grid = [
            [1, 1, 2],
            [1, 0, 2],
            [3, 3, 3]
        ]
        op = EdgeDetectionOperation(from_color=1, to_color=0, edge_color=5)
        result = op.execute(grid)

        assert result.success
        # Only 1->0 transitions should be marked
        expected = [
            [0, 0, 0],
            [5, 0, 0],
            [0, 0, 0]
        ]
        assert result.grid == expected

    def test_edge_detection_8_connectivity(self):
        """Test edge detection with 8-connectivity."""
        grid = [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1]
        ]
        op = EdgeDetectionOperation(edge_color=3, connectivity=8)
        result = op.execute(grid)

        assert result.success
        # All cells should be edges with 8-connectivity
        assert all(result.grid[r][c] == 3 for r in range(3) for c in range(3))

    def test_edge_detection_uniform_grid(self):
        """Test edge detection on uniform grid (no edges)."""
        grid = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        op = EdgeDetectionOperation()
        result = op.execute(grid)

        assert result.success
        assert result.grid == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert result.metadata["edge_count"] == 0

    def test_edge_detection_empty_grid(self):
        """Test edge detection on empty grid."""
        op = EdgeDetectionOperation()
        result = op.execute([])

        assert result.success
        assert result.grid == []


class TestBoundaryTracingOperation:
    """Test suite for BoundaryTracingOperation."""

    def test_boundary_tracing_simple_object(self):
        """Test boundary tracing of a simple object."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        op = BoundaryTracingOperation(boundary_color=2)
        result = op.execute(grid)

        assert result.success
        # Only outer boundary should be marked
        expected = [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0]
        ]
        assert result.grid == expected
        assert result.metadata["boundary_count"] == 8

    def test_boundary_tracing_specific_color(self):
        """Test boundary tracing for specific color objects."""
        grid = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3]
        ]
        op = BoundaryTracingOperation(target_color=2, boundary_color=5)
        result = op.execute(grid)

        assert result.success
        # Only boundaries of color 2 objects
        expected = [
            [0, 0, 5, 5],
            [0, 0, 5, 5],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        assert result.grid == expected

    def test_boundary_tracing_with_hole(self):
        """Test boundary tracing with inner boundaries (holes)."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 2, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        op = BoundaryTracingOperation(target_color=1, boundary_color=3, include_inner=True)
        result = op.execute(grid)

        assert result.success
        # Both outer and inner boundaries should be marked
        assert result.metadata["boundary_count"] == 16  # All 1's are boundaries

    def test_boundary_tracing_edge_object(self):
        """Test boundary tracing for object at grid edge."""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        op = BoundaryTracingOperation(boundary_color=2)
        result = op.execute(grid)

        assert result.success
        # All non-zero cells are boundaries (edge of grid)
        expected = [
            [2, 2, 0],
            [2, 2, 0],
            [0, 0, 0]
        ]
        assert result.grid == expected

    def test_boundary_tracing_complex_shape(self):
        """Test boundary tracing on complex shape."""
        grid = [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ]
        op = BoundaryTracingOperation(boundary_color=2)
        result = op.execute(grid)

        assert result.success
        # Check that interior point is not marked
        assert result.grid[2][2] == 0  # Hole
        assert result.metadata["boundary_count"] > 0


class TestContourExtractionOperation:
    """Test suite for ContourExtractionOperation."""

    def test_contour_extraction_simple(self):
        """Test contour extraction on simple object."""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]
        op = ContourExtractionOperation()
        result = op.execute(grid)

        assert result.success
        assert result.metadata["contour_count"] == 1
        assert len(result.metadata["contours"][0]) == 8  # 8 boundary points

    def test_contour_extraction_multiple_objects(self):
        """Test contour extraction with multiple objects."""
        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4]
        ]
        op = ContourExtractionOperation()
        result = op.execute(grid)

        assert result.success
        assert result.metadata["contour_count"] == 4
        # Each 2x2 block has 8 boundary points
        assert all(len(c) == 8 for c in result.metadata["contours"])

    def test_contour_extraction_min_length(self):
        """Test contour extraction with minimum length filter."""
        grid = [
            [1, 0, 2],  # Single point
            [0, 0, 0],
            [3, 3, 3]   # Line
        ]
        op = ContourExtractionOperation(min_length=5)
        result = op.execute(grid)

        assert result.success
        # Only the line object should have contour length >= 5
        assert result.metadata["contour_count"] == 1
        assert result.metadata["contour_lengths"][0] >= 5

    def test_contour_extraction_specific_color(self):
        """Test contour extraction for specific color."""
        grid = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3]
        ]
        op = ContourExtractionOperation(target_color=1)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["contour_count"] == 1
        # Should only extract contour of color 1 object

    def test_contour_extraction_empty_grid(self):
        """Test contour extraction on empty grid."""
        op = ContourExtractionOperation()
        result = op.execute([])

        assert result.success
        assert result.metadata["contours"] == []
        assert result.metadata["contour_count"] == 0

    def test_contour_extraction_visualization(self):
        """Test that contour visualization uses different colors."""
        grid = [
            [1, 0, 2],
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
            [3, 0, 4]
        ]
        op = ContourExtractionOperation()
        result = op.execute(grid)

        assert result.success
        # Check that different contours have different colors
        contour_colors = set()
        for r in range(5):
            for c in range(3):
                if result.grid[r][c] != 0:
                    contour_colors.add(result.grid[r][c])

        # Should have 4 different colors for 4 objects
        assert len(contour_colors) == 4
