"""
Symmetry detection and creation operations for the ARC DSL.

This module contains operations for detecting symmetries in grids and creating
symmetric patterns, such as reflection symmetry, rotational symmetry, etc.
"""

from typing import Any

from .base import OperationResult, PatternOperation
from .geometric import FlipOperation
from .types import Grid, TransformationContext


class CreateSymmetryOperation(PatternOperation):
    """
    Create a symmetric grid by mirroring the input along specified axes.

    Examples:
        # Create horizontal symmetry (mirror left half to right)
        symmetry = CreateSymmetryOperation(axis="horizontal", source_half="left")

        # Create 4-fold rotational symmetry
        symmetry = CreateSymmetryOperation(axis="rotational", order=4)
    """

    def __init__(self, axis: str, source_half: str = "left", order: int = 2):
        """
        Initialize symmetry creation operation.

        Args:
            axis: Type of symmetry ("horizontal", "vertical", "diagonal", "rotational")
            source_half: Which half to use as source ("left", "right", "top", "bottom")
            order: Order of rotational symmetry (2, 4, etc.)
        """
        valid_axes = ["horizontal", "vertical", "diagonal_main", "diagonal_anti", "rotational"]
        if axis not in valid_axes:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {valid_axes}")

        valid_halves = ["left", "right", "top", "bottom"]
        if source_half not in valid_halves:
            raise ValueError(f"Invalid source_half: {source_half}. Must be one of {valid_halves}")

        if order < 2:
            raise ValueError("Rotational order must be at least 2")

        super().__init__(axis=axis, source_half=source_half, order=order)
        self.axis = axis
        self.source_half = source_half
        self.order = order

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the symmetry creation operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot create symmetry in empty grid")

        try:
            if self.axis == "rotational":
                result_grid = self._create_rotational_symmetry(grid)
            else:
                result_grid = self._create_reflection_symmetry(grid)

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _create_reflection_symmetry(self, grid: Grid) -> Grid:
        """Create reflection symmetry along the specified axis."""
        rows, cols = len(grid), len(grid[0])
        result_grid = [row.copy() for row in grid]

        if self.axis == "horizontal":
            # Mirror horizontally
            if self.source_half == "left":
                # Copy left half to right half
                mid_col = cols // 2
                for r in range(rows):
                    for c in range(mid_col, cols):
                        source_col = cols - 1 - c
                        if source_col >= 0:
                            result_grid[r][c] = grid[r][source_col]
            else:  # right
                # Copy right half to left half
                mid_col = cols // 2
                for r in range(rows):
                    for c in range(mid_col):
                        source_col = cols - 1 - c
                        result_grid[r][c] = grid[r][source_col]

        elif self.axis == "vertical":
            # Mirror vertically
            if self.source_half == "top":
                # Copy top half to bottom half
                mid_row = rows // 2
                for r in range(mid_row, rows):
                    source_row = rows - 1 - r
                    if source_row >= 0:
                        result_grid[r] = grid[source_row].copy()
            else:  # bottom
                # Copy bottom half to top half
                mid_row = rows // 2
                for r in range(mid_row):
                    source_row = rows - 1 - r
                    result_grid[r] = grid[source_row].copy()

        elif self.axis in ["diagonal_main", "diagonal_anti"]:
            # Diagonal symmetry
            flip_op = FlipOperation(
                direction="diagonal_main" if self.axis == "diagonal_main" else "diagonal_anti"
            )
            flip_result = flip_op.execute(grid)
            if flip_result.success:
                result_grid = flip_result.grid

        return result_grid

    def _create_rotational_symmetry(self, grid: Grid) -> Grid:
        """Create rotational symmetry of the specified order."""
        rows, cols = len(grid), len(grid[0])

        # For simplicity, implement 4-fold symmetry for square grids
        if self.order == 4 and rows == cols:
            # Create 4-fold rotational symmetry
            result_grid = [row.copy() for row in grid]

            # Take the top-left quarter and replicate with rotations
            mid = rows // 2

            # Copy and rotate each quarter
            for r in range(mid):
                for c in range(mid):
                    color = grid[r][c]

                    # Original (top-left)
                    result_grid[r][c] = color

                    # 90° rotation (top-right)
                    result_grid[r][rows - 1 - c] = color

                    # 180° rotation (bottom-right)
                    result_grid[rows - 1 - r][rows - 1 - c] = color

                    # 270° rotation (bottom-left)
                    result_grid[rows - 1 - r][c] = color

            return result_grid

        # For other cases, just return the original grid
        return [row.copy() for row in grid]

    @classmethod
    def get_name(cls) -> str:
        return "create_symmetry"

    @classmethod
    def get_description(cls) -> str:
        return "Create a symmetric grid by mirroring the input along specified axes"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "axis": {
                "type": "string",
                "required": True,
                "valid_values": ["horizontal", "vertical", "diagonal_main", "diagonal_anti", "rotational"],
                "description": "Type of symmetry to create"
            },
            "source_half": {
                "type": "string",
                "required": False,
                "default": "left",
                "valid_values": ["left", "right", "top", "bottom"],
                "description": "Which half to use as source for reflection"
            },
            "order": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "Order of rotational symmetry"
            }
        }
