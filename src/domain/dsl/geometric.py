"""
Geometric transformation operations for the ARC DSL.

This module contains operations that modify the spatial arrangement of grid elements
without changing the colors themselves, such as rotation, mirroring, translation, etc.
"""

from typing import Any

from .base import OperationResult, TransformOperation
from .types import FlipDirection, Grid, Position, RotationAngle, TransformationContext


class RotateOperation(TransformOperation):
    """
    Rotate a grid by 90, 180, or 270 degrees clockwise.

    Examples:
        # Rotate 90 degrees clockwise
        rotate_90 = RotateOperation(angle=90)

        # Rotate 180 degrees
        rotate_180 = RotateOperation(angle=180)
    """

    def __init__(self, angle: int = 90):
        """
        Initialize rotation operation.

        Args:
            angle: Rotation angle in degrees (90, 180, or 270)
        """
        super().__init__(angle=angle)
        self.angle = RotationAngle(angle)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the rotation operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot rotate empty grid")

        try:
            if self.angle == RotationAngle.ROTATE_90:
                result_grid = self._rotate_90_clockwise(grid)
            elif self.angle == RotationAngle.ROTATE_180:
                result_grid = self._rotate_180(grid)
            elif self.angle == RotationAngle.ROTATE_270:
                result_grid = self._rotate_270_clockwise(grid)
            else:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Invalid rotation angle: {self.angle}"
                )

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @staticmethod
    def _rotate_90_clockwise(grid: Grid) -> Grid:
        """Rotate grid 90 degrees clockwise."""
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows - 1 - c][r] for c in range(rows)] for r in range(cols)]

    @staticmethod
    def _rotate_180(grid: Grid) -> Grid:
        """Rotate grid 180 degrees."""
        return [[grid[len(grid) - 1 - r][len(grid[0]) - 1 - c]
                for c in range(len(grid[0]))]
               for r in range(len(grid))]

    @staticmethod
    def _rotate_270_clockwise(grid: Grid) -> Grid:
        """Rotate grid 270 degrees clockwise (90 degrees counterclockwise)."""
        rows, cols = len(grid), len(grid[0])
        return [[grid[c][cols - 1 - r] for c in range(rows)] for r in range(cols)]

    @classmethod
    def get_name(cls) -> str:
        return "rotate"

    @classmethod
    def get_description(cls) -> str:
        return "Rotate the grid by the specified angle (90, 180, or 270 degrees clockwise)"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "angle": {
                "type": "integer",
                "required": True,
                "valid_values": [90, 180, 270],
                "description": "Rotation angle in degrees clockwise"
            }
        }


class FlipOperation(TransformOperation):
    """
    Flip/mirror a grid along various axes.

    Examples:
        # Horizontal flip (left-right mirror)
        flip_h = FlipOperation(direction="horizontal")

        # Vertical flip (up-down mirror)
        flip_v = FlipOperation(direction="vertical")
    """

    def __init__(self, direction: str = "horizontal"):
        """
        Initialize flip operation.

        Args:
            direction: Flip direction ("horizontal", "vertical", "diagonal_main", "diagonal_anti")
        """
        direction_map = {
            "horizontal": FlipDirection.HORIZONTAL,
            "vertical": FlipDirection.VERTICAL,
            "diagonal_main": FlipDirection.DIAGONAL_MAIN,
            "diagonal_anti": FlipDirection.DIAGONAL_ANTI
        }

        if direction not in direction_map:
            raise ValueError(f"Invalid flip direction: {direction}")

        super().__init__(direction=direction)
        self.direction = direction_map[direction]

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the flip operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot flip empty grid")

        try:
            if self.direction == FlipDirection.HORIZONTAL:
                result_grid = [row[::-1] for row in grid]
            elif self.direction == FlipDirection.VERTICAL:
                result_grid = grid[::-1]
            elif self.direction == FlipDirection.DIAGONAL_MAIN:
                result_grid = self._flip_main_diagonal(grid)
            elif self.direction == FlipDirection.DIAGONAL_ANTI:
                result_grid = self._flip_anti_diagonal(grid)
            else:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Invalid flip direction: {self.direction}"
                )

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @staticmethod
    def _flip_main_diagonal(grid: Grid) -> Grid:
        """Flip along main diagonal (transpose)."""
        rows, cols = len(grid), len(grid[0])
        return [[grid[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def _flip_anti_diagonal(grid: Grid) -> Grid:
        """Flip along anti-diagonal."""
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows - 1 - c][cols - 1 - r] for c in range(cols)] for r in range(rows)]

    @classmethod
    def get_name(cls) -> str:
        return "flip"

    @classmethod
    def get_description(cls) -> str:
        return "Flip/mirror the grid along the specified axis"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "direction": {
                "type": "string",
                "required": True,
                "valid_values": ["horizontal", "vertical", "diagonal_main", "diagonal_anti"],
                "description": "Direction to flip the grid"
            }
        }


class TranslateOperation(TransformOperation):
    """
    Translate/shift a grid by the specified offset.

    Elements that move outside the grid bounds are lost, and new areas are
    filled with the specified fill color.

    Examples:
        # Shift right by 2, down by 1
        translate = TranslateOperation(offset=(1, 2), fill_color=0)
    """

    def __init__(self, offset: Position, fill_color: int = 0):
        """
        Initialize translate operation.

        Args:
            offset: (row_offset, col_offset) to shift by
            fill_color: Color to fill empty areas (default: 0)
        """
        super().__init__(offset=offset, fill_color=fill_color)
        self.row_offset, self.col_offset = offset
        self.fill_color = fill_color

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the translate operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot translate empty grid")

        try:
            rows, cols = len(grid), len(grid[0])
            result_grid = [[self.fill_color for _ in range(cols)] for _ in range(rows)]

            for r in range(rows):
                for c in range(cols):
                    new_r = r + self.row_offset
                    new_c = c + self.col_offset

                    # Only copy if the destination is within bounds
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        result_grid[new_r][new_c] = grid[r][c]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "translate"

    @classmethod
    def get_description(cls) -> str:
        return "Shift/translate the grid by the specified offset, filling empty areas with fill_color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "offset": {
                "type": "tuple",
                "required": True,
                "description": "Row and column offset (row_offset, col_offset)"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to fill empty areas"
            }
        }


class CropOperation(TransformOperation):
    """
    Crop a grid to a specified rectangular region.

    Examples:
        # Crop to top-left 3x3 region
        crop = CropOperation(top=0, left=0, bottom=2, right=2)

        # Crop center region
        crop_center = CropOperation(top=1, left=1, bottom=3, right=3)
    """

    def __init__(self, top: int, left: int, bottom: int, right: int):
        """
        Initialize crop operation.

        Args:
            top: Top row index (inclusive)
            left: Left column index (inclusive)
            bottom: Bottom row index (inclusive)
            right: Right column index (inclusive)
        """
        super().__init__(top=top, left=left, bottom=bottom, right=right)
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the crop operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot crop empty grid")

        try:
            rows, cols = len(grid), len(grid[0])

            # Validate bounds
            if self.top < 0 or self.left < 0 or self.bottom >= rows or self.right >= cols:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Crop bounds out of range: grid is {rows}x{cols}"
                )

            if self.top > self.bottom or self.left > self.right:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message="Invalid crop bounds: top/left must be <= bottom/right"
                )

            # Extract the cropped region
            result_grid = []
            for r in range(self.top, self.bottom + 1):
                row = []
                for c in range(self.left, self.right + 1):
                    row.append(grid[r][c])
                result_grid.append(row)

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "original_size": (rows, cols),
                    "crop_region": (self.top, self.left, self.bottom, self.right),
                    "new_size": (len(result_grid), len(result_grid[0]) if result_grid else 0)
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "crop"

    @classmethod
    def get_description(cls) -> str:
        return "Crop grid to specified rectangular region"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "top": {
                "type": "integer",
                "required": True,
                "description": "Top row index (inclusive)"
            },
            "left": {
                "type": "integer",
                "required": True,
                "description": "Left column index (inclusive)"
            },
            "bottom": {
                "type": "integer",
                "required": True,
                "description": "Bottom row index (inclusive)"
            },
            "right": {
                "type": "integer",
                "required": True,
                "description": "Right column index (inclusive)"
            }
        }


class PadOperation(TransformOperation):
    """
    Pad a grid by adding rows/columns around the edges.

    Examples:
        # Add 1-row/column border on all sides
        pad = PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0)

        # Pad only on right and bottom
        pad_rb = PadOperation(top=0, bottom=2, left=0, right=3, fill_color=5)
    """

    def __init__(self, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, fill_color: int = 0):
        """
        Initialize pad operation.

        Args:
            top: Number of rows to add at top
            bottom: Number of rows to add at bottom
            left: Number of columns to add at left
            right: Number of columns to add at right
            fill_color: Color to use for padding (default: 0)
        """
        super().__init__(top=top, bottom=bottom, left=left, right=right, fill_color=fill_color)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.fill_color = fill_color

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the pad operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot pad empty grid")

        try:
            rows, cols = len(grid), len(grid[0])
            new_rows = rows + self.top + self.bottom
            new_cols = cols + self.left + self.right

            # Create padded grid filled with fill_color
            result_grid = [[self.fill_color for _ in range(new_cols)] for _ in range(new_rows)]

            # Copy original grid into the center
            for r in range(rows):
                for c in range(cols):
                    result_grid[r + self.top][c + self.left] = grid[r][c]

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "original_size": (rows, cols),
                    "padding": (self.top, self.bottom, self.left, self.right),
                    "fill_color": self.fill_color,
                    "new_size": (new_rows, new_cols)
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "pad"

    @classmethod
    def get_description(cls) -> str:
        return "Pad grid by adding rows/columns around edges"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "top": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Rows to add at top"
            },
            "bottom": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Rows to add at bottom"
            },
            "left": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Columns to add at left"
            },
            "right": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Columns to add at right"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color for padding"
            }
        }
