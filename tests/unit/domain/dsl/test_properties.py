"""
Property-based tests for DSL operations using hypothesis.

This module tests operation invariants, composition properties, 
and boundary conditions with randomly generated grids.
"""

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.domain.dsl.color import ColorInvertOperation, ColorMapOperation, ColorReplaceOperation
from src.domain.dsl.geometric import FlipOperation, RotateOperation, TranslateOperation
from src.domain.dsl.pattern import FloodFillOperation
from src.domain.dsl.types import Grid


# Optimized grid generation strategies
def arc_color() -> st.SearchStrategy[int]:
    """Generate valid ARC colors (0-9)."""
    return st.integers(min_value=0, max_value=9)


def small_grid(min_size: int = 2, max_size: int = 4) -> st.SearchStrategy[Grid]:
    """Generate small grids for fast property testing."""
    return st.lists(
        st.lists(arc_color(), min_size=min_size, max_size=max_size),
        min_size=min_size,
        max_size=max_size
    )


def square_grid(min_size: int = 2, max_size: int = 4) -> st.SearchStrategy[Grid]:
    """Generate small square grids for fast property testing."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda size: st.lists(
            st.lists(arc_color(), min_size=size, max_size=size),
            min_size=size,
            max_size=size
        )
    )


# Common settings for all property tests
FAST_SETTINGS = settings(
    max_examples=10,
    deadline=5000,  # 5 second deadline
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)


class TestRotationProperties:
    """Property-based tests for rotation operations."""

    @given(grid=square_grid())
    @FAST_SETTINGS
    def test_rotate_four_times_is_identity(self, grid):
        """Test that rotating 90 degrees four times returns original grid."""
        rotate_op = RotateOperation(angle=90)

        current_grid = grid
        for _ in range(4):
            result = rotate_op.execute(current_grid)
            assert result.success
            current_grid = result.grid

        assert current_grid == grid

    @given(grid=square_grid())
    @FAST_SETTINGS
    def test_rotate_180_twice_is_identity(self, grid):
        """Test that rotating 180 degrees twice returns original grid."""
        rotate_op = RotateOperation(angle=180)

        result1 = rotate_op.execute(grid)
        assert result1.success

        result2 = rotate_op.execute(result1.grid)
        assert result2.success

        assert result2.grid == grid

    @given(grid=square_grid())
    @FAST_SETTINGS
    def test_rotation_preserves_colors(self, grid):
        """Test that rotation preserves all colors present in the grid."""
        original_colors = set()
        for row in grid:
            original_colors.update(row)

        rotate_op = RotateOperation(angle=90)
        result = rotate_op.execute(grid)
        assert result.success

        rotated_colors = set()
        for row in result.grid:
            rotated_colors.update(row)

        assert original_colors == rotated_colors


class TestFlipProperties:
    """Property-based tests for flip operations."""

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_flip_twice_is_identity(self, grid):
        """Test that flipping twice returns original grid."""
        directions = ["horizontal", "vertical"]

        for direction in directions:
            flip_op = FlipOperation(direction=direction)

            result1 = flip_op.execute(grid)
            assert result1.success

            result2 = flip_op.execute(result1.grid)
            assert result2.success

            assert result2.grid == grid

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_flip_preserves_colors(self, grid):
        """Test that flipping preserves all colors present in the grid."""
        original_colors = set()
        for row in grid:
            original_colors.update(row)

        flip_op = FlipOperation(direction="horizontal")
        result = flip_op.execute(grid)
        assert result.success

        flipped_colors = set()
        for row in result.grid:
            flipped_colors.update(row)

        assert original_colors == flipped_colors


class TestColorProperties:
    """Property-based tests for color operations."""

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_color_invert_twice_is_identity(self, grid):
        """Test that inverting colors twice returns original grid."""
        invert_op = ColorInvertOperation()

        result1 = invert_op.execute(grid)
        assert result1.success

        result2 = invert_op.execute(result1.grid)
        assert result2.success

        assert result2.grid == grid

    @given(grid=small_grid(), old_color=arc_color(), new_color=arc_color())
    @FAST_SETTINGS
    def test_color_replace_then_back_is_identity(self, grid, old_color, new_color):
        """Test that replacing a color and then replacing it back is identity."""
        assume(old_color != new_color)

        # Only test if the old color exists and new color does NOT exist in the grid
        has_old_color = any(old_color in row for row in grid)
        has_new_color = any(new_color in row for row in grid)
        assume(has_old_color and not has_new_color)

        replace_op1 = ColorReplaceOperation(source_color=old_color, target_color=new_color)
        replace_op2 = ColorReplaceOperation(source_color=new_color, target_color=old_color)

        result1 = replace_op1.execute(grid)
        assert result1.success

        result2 = replace_op2.execute(result1.grid)
        assert result2.success

        assert result2.grid == grid

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_color_map_preserves_grid_dimensions(self, grid):
        """Test that color mapping preserves grid dimensions."""
        color_mapping = {i: (i + 1) % 10 for i in range(10)}
        map_op = ColorMapOperation(mapping=color_mapping)

        result = map_op.execute(grid)
        assert result.success

        assert len(result.grid) == len(grid)
        assert all(len(result.grid[i]) == len(grid[i]) for i in range(len(grid)))


class TestCompositionProperties:
    """Property-based tests for operation composition."""

    @given(grid=square_grid())
    @FAST_SETTINGS
    def test_rotation_composition_commutativity(self, grid):
        """Test that certain rotation compositions are commutative."""
        rotate_90 = RotateOperation(angle=90)
        rotate_180 = RotateOperation(angle=180)

        # 180 then 90 should equal 270
        result1 = rotate_180.execute(grid)
        assert result1.success
        result1 = rotate_90.execute(result1.grid)
        assert result1.success

        # Direct 270 rotation
        rotate_270 = RotateOperation(angle=270)
        result2 = rotate_270.execute(grid)
        assert result2.success

        assert result1.grid == result2.grid

    @given(grid=small_grid(), color1=arc_color(), color2=arc_color())
    @FAST_SETTINGS
    def test_color_operation_composition(self, grid, color1, color2):
        """Test composition of color operations."""
        assume(color1 != color2)

        # Replace color1 with color2, then invert
        replace_op = ColorReplaceOperation(source_color=color1, target_color=color2)
        invert_op = ColorInvertOperation()

        # Apply operations sequentially
        result1 = replace_op.execute(grid)
        assert result1.success

        result2 = invert_op.execute(result1.grid)
        assert result2.success

        # Result should still have correct dimensions
        assert len(result2.grid) == len(grid)
        assert all(len(result2.grid[i]) == len(grid[i]) for i in range(len(grid)))

    @given(grid=square_grid())
    @FAST_SETTINGS
    def test_geometric_transformation_chain(self, grid):
        """Test chaining geometric transformations maintains validity."""
        rotate_op = RotateOperation(angle=90)
        flip_op = FlipOperation(direction="horizontal")

        # Chain operations
        result1 = rotate_op.execute(grid)
        assert result1.success

        result2 = flip_op.execute(result1.grid)
        assert result2.success

        # Final grid should be valid
        assert len(result2.grid) > 0
        assert all(len(row) > 0 for row in result2.grid)
        assert all(all(0 <= cell <= 9 for cell in row) for row in result2.grid)


class TestBoundaryConditions:
    """Property-based tests for boundary conditions."""

    @given(st.integers(min_value=2, max_value=4))
    def test_single_color_grid_operations(self, size):
        """Test operations on single-color grids."""
        color = 5
        grid = [[color] * size for _ in range(size)]

        # Test rotation
        rotate_op = RotateOperation(angle=90)
        result = rotate_op.execute(grid)
        assert result.success
        assert all(all(cell == color for cell in row) for row in result.grid)

        # Test flip
        flip_op = FlipOperation(direction="horizontal")
        result = flip_op.execute(grid)
        assert result.success
        assert all(all(cell == color for cell in row) for row in result.grid)

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_operations_preserve_valid_colors(self, grid):
        """Test that all operations preserve valid ARC colors (0-9)."""
        operations = [
            RotateOperation(angle=90),
            FlipOperation(direction="horizontal"),
            ColorInvertOperation()
        ]

        for op in operations:
            result = op.execute(grid)
            if result.success:
                # All colors should be valid (0-9)
                for row in result.grid:
                    for cell in row:
                        assert 0 <= cell <= 9

    @given(grid=small_grid(), dx=st.integers(-2, 2), dy=st.integers(-2, 2))
    @FAST_SETTINGS
    def test_translate_boundary_handling(self, grid, dx, dy):
        """Test that translation handles boundary conditions correctly."""
        translate_op = TranslateOperation(offset=(dx, dy), fill_color=0)
        result = translate_op.execute(grid)
        assert result.success

        # Result should have same dimensions
        assert len(result.grid) == len(grid)
        assert all(len(result.grid[i]) == len(grid[i]) for i in range(len(grid)))

        # All colors should be valid
        for row in result.grid:
            for cell in row:
                assert 0 <= cell <= 9


class TestOperationInvariants:
    """Test mathematical invariants for operations."""

    @given(grid=small_grid())
    @FAST_SETTINGS
    def test_flip_horizontal_vertical_commutes_with_180_rotation(self, grid):
        """Test that horizontal flip + vertical flip = 180 degree rotation."""
        # Apply horizontal then vertical flip
        h_flip = FlipOperation(direction="horizontal")
        v_flip = FlipOperation(direction="vertical")

        result1 = h_flip.execute(grid)
        assert result1.success
        result1 = v_flip.execute(result1.grid)
        assert result1.success

        # Apply 180 degree rotation
        rotate_180 = RotateOperation(angle=180)
        result2 = rotate_180.execute(grid)
        assert result2.success

        assert result1.grid == result2.grid

    @given(grid=small_grid(), fill_color=arc_color())
    @FAST_SETTINGS
    def test_flood_fill_connected_region_properties(self, grid, fill_color):
        """Test flood fill properties on connected regions."""
        # Pick a valid starting position
        start_row = len(grid) // 2
        start_col = len(grid[0]) // 2
        original_color = grid[start_row][start_col]

        # Skip if already the target color
        assume(original_color != fill_color)

        flood_op = FloodFillOperation(position=(start_row, start_col), color=fill_color)
        result = flood_op.execute(grid)
        assert result.success

        # The position we started from should now have the new color
        assert result.grid[start_row][start_col] == fill_color

        # Grid dimensions should be preserved
        assert len(result.grid) == len(grid)
        assert all(len(result.grid[i]) == len(grid[i]) for i in range(len(grid)))
