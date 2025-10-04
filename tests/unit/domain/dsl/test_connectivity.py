"""
Unit tests for connectivity operations.
"""

from src.domain.dsl.connectivity import ConnectedComponentsOperation, FilterComponentsOperation


class TestConnectedComponentsOperation:
    """Test suite for ConnectedComponentsOperation."""

    def test_connected_components_simple(self):
        """Test finding simple connected components."""
        grid = [
            [1, 1, 0, 2, 2],
            [1, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 0],
            [3, 3, 0, 4, 4]
        ]
        op = ConnectedComponentsOperation(background_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["component_count"] == 4
        assert result.metadata["component_sizes"] == [3, 4, 4, 3]

    def test_connected_components_specific_color(self):
        """Test finding components of specific color."""
        grid = [
            [1, 1, 2, 2],
            [1, 0, 2, 2],
            [0, 1, 0, 0],
            [1, 1, 1, 2]
        ]
        op = ConnectedComponentsOperation(target_color=1)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["component_count"] == 2
        # First component has 3 cells, second has 4 cells
        assert sorted(result.metadata["component_sizes"]) == [3, 4]

    def test_connected_components_8_connectivity(self):
        """Test 8-connectivity for connected components."""
        grid = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]

        # Test with 4-connectivity (default)
        op4 = ConnectedComponentsOperation(target_color=1, connectivity=4)
        result4 = op4.execute(grid)
        assert result4.metadata["component_count"] == 5  # Each 1 is separate

        # Test with 8-connectivity
        op8 = ConnectedComponentsOperation(target_color=1, connectivity=8)
        result8 = op8.execute(grid)
        assert result8.metadata["component_count"] == 1  # All connected diagonally

    def test_connected_components_all_colors(self):
        """Test finding all components regardless of color."""
        grid = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ]
        op = ConnectedComponentsOperation()
        result = op.execute(grid)

        assert result.success
        assert result.metadata["component_count"] == 4
        assert all(size == 4 for size in result.metadata["component_sizes"])

    def test_connected_components_empty_grid(self):
        """Test connected components on empty grid."""
        op = ConnectedComponentsOperation()
        result = op.execute([])

        assert result.success
        assert result.grid == []
        assert result.metadata["component_count"] == 0
        assert result.metadata["components"] == []

    def test_connected_components_single_color(self):
        """Test grid with single connected component."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        op = ConnectedComponentsOperation()
        result = op.execute(grid)

        assert result.success
        assert result.metadata["component_count"] == 1
        assert result.metadata["component_sizes"][0] == 9

    def test_connected_components_labeling(self):
        """Test that components are labeled correctly."""
        grid = [
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4]
        ]
        op = ConnectedComponentsOperation(background_color=0)
        result = op.execute(grid)

        # Each non-zero cell should get a unique label 1-4
        assert result.grid[0][0] in [1, 2, 3, 4]
        assert result.grid[0][2] in [1, 2, 3, 4]
        assert result.grid[2][0] in [1, 2, 3, 4]
        assert result.grid[2][2] in [1, 2, 3, 4]
        # All labels should be different
        labels = [result.grid[0][0], result.grid[0][2], result.grid[2][0], result.grid[2][2]]
        assert len(set(labels)) == 4


class TestFilterComponentsOperation:
    """Test suite for FilterComponentsOperation."""

    def test_filter_components_by_min_size(self):
        """Test filtering components by minimum size."""
        grid = [
            [1, 0, 2, 2],  # 1: size 1, 2: size 2
            [0, 0, 0, 0],
            [3, 3, 3, 0],  # 3: size 3
            [0, 0, 4, 4]   # 4: size 2
        ]
        op = FilterComponentsOperation(min_size=2)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["original_components"] == 4
        assert result.metadata["filtered_components"] == 3
        # Single cell (1) should be filtered out
        assert result.grid[0][0] == 0

    def test_filter_components_by_max_size(self):
        """Test filtering components by maximum size."""
        grid = [
            [1, 1, 1, 1],  # Size 4
            [2, 2, 0, 0],  # Size 2
            [3, 0, 0, 0],  # Size 1
            [0, 0, 0, 0]
        ]
        op = FilterComponentsOperation(max_size=2, background_color=0)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["filtered_components"] == 2
        # Large component should be filtered out
        assert all(result.grid[0][c] == 0 for c in range(4))

    def test_filter_keep_largest_component(self):
        """Test keeping only the largest component."""
        grid = [
            [1, 1, 0, 2],
            [1, 1, 0, 2],
            [1, 0, 0, 0],
            [0, 3, 3, 3]
        ]
        op = FilterComponentsOperation(keep_largest=True)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["filtered_components"] == 1
        # Only the 5-cell component should remain
        assert result.grid[0][0] == 1  # Part of largest
        assert result.grid[0][3] == 0  # Filtered out
        assert result.grid[3][1] == 0  # Filtered out

    def test_filter_keep_smallest_component(self):
        """Test keeping only the smallest component."""
        grid = [
            [1, 1, 1, 0],
            [0, 0, 2, 0],
            [3, 3, 0, 0],
            [3, 3, 0, 0]
        ]
        op = FilterComponentsOperation(keep_smallest=True)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["filtered_components"] == 1
        # Only the single-cell component should remain
        assert result.grid[1][2] == 2  # Smallest component
        assert result.grid[0][0] == 0  # Filtered out
        assert result.grid[2][0] == 0  # Filtered out

    def test_filter_components_size_range(self):
        """Test filtering with both min and max size."""
        grid = [
            [1, 0, 0, 0],  # Size 1
            [2, 2, 0, 0],  # Size 2
            [3, 3, 3, 0],  # Size 3
            [4, 4, 4, 4]   # Size 4
        ]
        op = FilterComponentsOperation(min_size=2, max_size=3)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["filtered_components"] == 2
        # Only size 2 and 3 components should remain
        assert result.grid[0][0] == 0  # Size 1 filtered
        assert result.grid[1][0] == 2  # Size 2 kept
        assert result.grid[2][0] == 3  # Size 3 kept
        assert result.grid[3][0] == 0  # Size 4 filtered

    def test_filter_components_custom_background(self):
        """Test filtering with custom background color."""
        grid = [
            [1, 1, 9, 9],
            [1, 1, 9, 9],
            [2, 2, 2, 2],
            [2, 2, 2, 2]
        ]
        op = FilterComponentsOperation(min_size=5, background_color=5)
        result = op.execute(grid)

        assert result.success
        # Only the 8-cell component should remain
        assert result.metadata["filtered_components"] == 1
        # Filtered areas should be filled with background color 5
        assert result.grid[0][0] == 5  # Filtered to background
        assert result.grid[2][0] == 2  # Kept

    def test_filter_components_empty_grid(self):
        """Test filtering on empty grid."""
        op = FilterComponentsOperation(min_size=1)
        result = op.execute([])

        assert result.success
        assert result.grid == []

    def test_filter_components_no_match(self):
        """Test filtering where no components match criteria."""
        grid = [[1, 1], [1, 1]]  # Single 4-cell component
        op = FilterComponentsOperation(min_size=5)
        result = op.execute(grid)

        assert result.success
        assert result.metadata["filtered_components"] == 0
        assert result.grid == [[0, 0], [0, 0]]
