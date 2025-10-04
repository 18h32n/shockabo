"""
Grid composition and decomposition operations for the ARC DSL.

This module contains operations that combine multiple grids or extract sub-grids,
such as overlay, concatenation, region extraction, cropping, padding, etc.
"""

from typing import Any

from .base import CompositionOperation, OperationResult
from .types import Color, Grid, GridRegion, Position, TransformationContext, create_empty_grid


class CropOperation(CompositionOperation):
    """
    Extract a rectangular region from the grid.

    Examples:
        # Extract 3x3 region starting from (1, 1)
        crop = CropOperation(top_left=(1, 1), dimensions=(3, 3))

        # Extract region defined by corners
        crop = CropOperation(top_left=(0, 0), bottom_right=(2, 2))
    """

    def __init__(self, top_left: Position, dimensions: tuple[int, int] | None = None,
                 bottom_right: Position | None = None):
        """
        Initialize crop operation.

        Args:
            top_left: Top-left corner of the region to extract
            dimensions: (height, width) of the region (if bottom_right not specified)
            bottom_right: Bottom-right corner of the region (if dimensions not specified)
        """
        if dimensions is None and bottom_right is None:
            raise ValueError("Must specify either dimensions or bottom_right")

        if dimensions is not None and bottom_right is not None:
            raise ValueError("Cannot specify both dimensions and bottom_right")

        super().__init__(top_left=top_left, dimensions=dimensions, bottom_right=bottom_right)
        self.top_left = top_left

        if dimensions is not None:
            height, width = dimensions
            self.bottom_right = (top_left[0] + height - 1, top_left[1] + width - 1)
        else:
            self.bottom_right = bottom_right

        self.region = GridRegion(top_left=self.top_left, bottom_right=self.bottom_right)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the crop operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot crop empty grid")

        try:
            rows, cols = len(grid), len(grid[0])

            # Validate region bounds
            if (self.top_left[0] < 0 or self.top_left[1] < 0 or
                self.bottom_right[0] >= rows or self.bottom_right[1] >= cols):
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Crop region {self.top_left} to {self.bottom_right} is out of bounds for {rows}x{cols} grid"
                )

            # Extract the region
            result_grid = []
            for r in range(self.top_left[0], self.bottom_right[0] + 1):
                row = []
                for c in range(self.top_left[1], self.bottom_right[1] + 1):
                    row.append(grid[r][c])
                result_grid.append(row)

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "crop"

    @classmethod
    def get_description(cls) -> str:
        return "Extract a rectangular region from the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "top_left": {
                "type": "tuple",
                "required": True,
                "description": "Top-left corner of the region (row, col)"
            },
            "dimensions": {
                "type": "tuple",
                "required": False,
                "description": "Dimensions of the region (height, width)"
            },
            "bottom_right": {
                "type": "tuple",
                "required": False,
                "description": "Bottom-right corner of the region (row, col)"
            }
        }


class PadOperation(CompositionOperation):
    """
    Add padding around the grid with the specified color.

    Examples:
        # Add 1-cell border of black (0) around the grid
        pad = PadOperation(padding=1, fill_color=0)

        # Add different padding on each side
        pad = PadOperation(padding=(2, 1, 2, 1), fill_color=0)  # top, right, bottom, left
    """

    def __init__(self, padding: int | tuple[int, int, int, int], fill_color: int = 0):
        """
        Initialize pad operation.

        Args:
            padding: Padding amount (int for uniform, tuple for (top, right, bottom, left))
            fill_color: Color to use for padding
        """
        if not (0 <= fill_color <= 9):
            raise ValueError(f"Invalid fill color: {fill_color}")

        if isinstance(padding, int):
            if padding < 0:
                raise ValueError("Padding cannot be negative")
            self.padding = (padding, padding, padding, padding)
        else:
            if len(padding) != 4 or any(p < 0 for p in padding):
                raise ValueError("Padding tuple must have 4 non-negative values")
            self.padding = padding

        super().__init__(padding=padding, fill_color=fill_color)
        self.fill_color = Color(fill_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the pad operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot pad empty grid")

        try:
            rows, cols = len(grid), len(grid[0])
            top_pad, right_pad, bottom_pad, left_pad = self.padding

            new_rows = rows + top_pad + bottom_pad
            new_cols = cols + left_pad + right_pad

            # Create padded grid
            result_grid = create_empty_grid(new_rows, new_cols, self.fill_color)

            # Copy original grid into the center
            for r in range(rows):
                for c in range(cols):
                    result_grid[r + top_pad][c + left_pad] = grid[r][c]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "pad"

    @classmethod
    def get_description(cls) -> str:
        return "Add padding around the grid with the specified color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "padding": {
                "type": "int_or_tuple",
                "required": True,
                "description": "Padding amount (int for uniform, tuple for (top, right, bottom, left))"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to use for padding"
            }
        }


class OverlayOperation(CompositionOperation):
    """
    Overlay one grid on top of another at a specified position.

    The overlay grid is placed on the base grid, with non-transparent colors
    replacing the base grid colors at those positions.

    Examples:
        # Overlay pattern at position (2, 3) with black as transparent
        overlay = OverlayOperation(overlay_grid=pattern, position=(2, 3), transparent_color=0)
    """

    def __init__(self, overlay_grid: Grid, position: Position = (0, 0),
                 transparent_color: int | None = None):
        """
        Initialize overlay operation.

        Args:
            overlay_grid: Grid to overlay on top
            position: Position to place the overlay (top-left corner)
            transparent_color: Color in overlay that should be transparent (None = no transparency)
        """
        if not overlay_grid:
            raise ValueError("Overlay grid cannot be empty")

        if transparent_color is not None and not (0 <= transparent_color <= 9):
            raise ValueError(f"Invalid transparent color: {transparent_color}")

        super().__init__(
            overlay_grid=overlay_grid,
            position=position,
            transparent_color=transparent_color
        )
        self.overlay_grid = overlay_grid
        self.position = position
        self.transparent_color = Color(transparent_color) if transparent_color is not None else None

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the overlay operation."""
        if not grid:
            # If base grid is empty, return the overlay grid if position is (0,0)
            if self.position == (0, 0):
                return OperationResult(success=True, grid=self.overlay_grid)
            else:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message="Cannot overlay on empty grid at non-zero position"
                )

        try:
            result_grid = [row.copy() for row in grid]
            base_rows, base_cols = len(grid), len(grid[0])
            overlay_rows, overlay_cols = len(self.overlay_grid), len(self.overlay_grid[0])
            start_row, start_col = self.position

            # Apply overlay
            for r in range(overlay_rows):
                for c in range(overlay_cols):
                    target_row = start_row + r
                    target_col = start_col + c

                    # Skip if overlay position is outside base grid
                    if not (0 <= target_row < base_rows and 0 <= target_col < base_cols):
                        continue

                    overlay_color = Color(self.overlay_grid[r][c])

                    # Skip transparent colors
                    if self.transparent_color is not None and overlay_color == self.transparent_color:
                        continue

                    result_grid[target_row][target_col] = overlay_color

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "overlay"

    @classmethod
    def get_description(cls) -> str:
        return "Overlay one grid on top of another at a specified position"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "overlay_grid": {
                "type": "list",
                "required": True,
                "description": "Grid to overlay on top"
            },
            "position": {
                "type": "tuple",
                "required": False,
                "default": (0, 0),
                "description": "Position to place the overlay (row, col)"
            },
            "transparent_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color in overlay that should be transparent"
            }
        }


class ConcatenateOperation(CompositionOperation):
    """
    Concatenate two grids either horizontally or vertically.

    Examples:
        # Concatenate horizontally (side by side)
        concat_h = ConcatenateOperation(other_grid=grid2, direction="horizontal")

        # Concatenate vertically (top and bottom)
        concat_v = ConcatenateOperation(other_grid=grid2, direction="vertical")
    """

    def __init__(self, other_grid: Grid, direction: str = "horizontal",
                 fill_color: int = 0):
        """
        Initialize concatenate operation.

        Args:
            other_grid: Grid to concatenate with the input grid
            direction: "horizontal" (side by side) or "vertical" (top and bottom)
            fill_color: Color to use for filling when grids have different sizes
        """
        if not other_grid:
            raise ValueError("Other grid cannot be empty")

        if direction not in ["horizontal", "vertical"]:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")

        if not (0 <= fill_color <= 9):
            raise ValueError(f"Invalid fill color: {fill_color}")

        super().__init__(other_grid=other_grid, direction=direction, fill_color=fill_color)
        self.other_grid = other_grid
        self.direction = direction
        self.fill_color = Color(fill_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the concatenate operation."""
        if not grid:
            return OperationResult(success=True, grid=self.other_grid)

        try:
            if self.direction == "horizontal":
                result_grid = self._concatenate_horizontal(grid, self.other_grid)
            else:  # vertical
                result_grid = self._concatenate_vertical(grid, self.other_grid)

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _concatenate_horizontal(self, grid1: Grid, grid2: Grid) -> Grid:
        """Concatenate grids horizontally (side by side)."""
        rows1, cols1 = len(grid1), len(grid1[0])
        rows2, cols2 = len(grid2), len(grid2[0])

        # Make sure both grids have the same height
        max_rows = max(rows1, rows2)
        result_grid = []

        for r in range(max_rows):
            row = []

            # Add cells from first grid
            for c in range(cols1):
                if r < rows1:
                    row.append(grid1[r][c])
                else:
                    row.append(self.fill_color)

            # Add cells from second grid
            for c in range(cols2):
                if r < rows2:
                    row.append(grid2[r][c])
                else:
                    row.append(self.fill_color)

            result_grid.append(row)

        return result_grid

    def _concatenate_vertical(self, grid1: Grid, grid2: Grid) -> Grid:
        """Concatenate grids vertically (top and bottom)."""
        rows1, cols1 = len(grid1), len(grid1[0])
        rows2, cols2 = len(grid2), len(grid2[0])

        # Make sure both grids have the same width
        max_cols = max(cols1, cols2)
        result_grid = []

        # Add rows from first grid
        for r in range(rows1):
            row = []
            for c in range(max_cols):
                if c < cols1:
                    row.append(grid1[r][c])
                else:
                    row.append(self.fill_color)
            result_grid.append(row)

        # Add rows from second grid
        for r in range(rows2):
            row = []
            for c in range(max_cols):
                if c < cols2:
                    row.append(grid2[r][c])
                else:
                    row.append(self.fill_color)
            result_grid.append(row)

        return result_grid

    @classmethod
    def get_name(cls) -> str:
        return "concatenate"

    @classmethod
    def get_description(cls) -> str:
        return "Concatenate two grids either horizontally or vertically"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "other_grid": {
                "type": "list",
                "required": True,
                "description": "Grid to concatenate with the input grid"
            },
            "direction": {
                "type": "string",
                "required": False,
                "default": "horizontal",
                "valid_values": ["horizontal", "vertical"],
                "description": "Direction to concatenate: 'horizontal' or 'vertical'"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to use for filling when grids have different sizes"
            }
        }


class ExtractRegionOperation(CompositionOperation):
    """
    Extract a specific region from the grid based on color patterns or conditions.

    Examples:
        # Extract all cells with a specific color
        extract_red = ExtractRegionOperation(target_color=2, extract_mode="color")

        # Extract a bounding box around non-zero colors
        extract_content = ExtractRegionOperation(extract_mode="bounding_box", ignore_color=0)
    """

    def __init__(self, extract_mode: str = "bounding_box", target_color: int | None = None,
                 ignore_color: int | None = None, padding: int = 0):
        """
        Initialize extract region operation.

        Args:
            extract_mode: "bounding_box", "color", or "non_zero"
            target_color: Specific color to extract (for "color" mode)
            ignore_color: Color to ignore when finding bounding box
            padding: Additional padding around the extracted region
        """
        valid_modes = ["bounding_box", "color", "non_zero"]
        if extract_mode not in valid_modes:
            raise ValueError(f"Extract mode must be one of: {valid_modes}")

        if extract_mode == "color" and target_color is None:
            raise ValueError("target_color is required for 'color' extract mode")

        if target_color is not None and not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        if ignore_color is not None and not (0 <= ignore_color <= 9):
            raise ValueError(f"Invalid ignore color: {ignore_color}")

        if padding < 0:
            raise ValueError("Padding cannot be negative")

        super().__init__(
            extract_mode=extract_mode,
            target_color=target_color,
            ignore_color=ignore_color,
            padding=padding
        )
        self.extract_mode = extract_mode
        self.target_color = Color(target_color) if target_color is not None else None
        self.ignore_color = Color(ignore_color) if ignore_color is not None else None
        self.padding = padding

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the extract region operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot extract from empty grid")

        try:
            if self.extract_mode == "bounding_box":
                result_grid = self._extract_bounding_box(grid)
            elif self.extract_mode == "color":
                result_grid = self._extract_by_color(grid)
            elif self.extract_mode == "non_zero":
                result_grid = self._extract_non_zero(grid)
            else:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Unknown extract mode: {self.extract_mode}"
                )

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _extract_bounding_box(self, grid: Grid) -> Grid:
        """Extract bounding box around non-ignored colors."""
        rows, cols = len(grid), len(grid[0])

        min_row, max_row = rows, -1
        min_col, max_col = cols, -1

        # Find bounding box
        for r in range(rows):
            for c in range(cols):
                color = Color(grid[r][c])
                if self.ignore_color is None or color != self.ignore_color:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        # If no content found, return original grid
        if max_row == -1:
            return grid

        # Apply padding
        min_row = max(0, min_row - self.padding)
        max_row = min(rows - 1, max_row + self.padding)
        min_col = max(0, min_col - self.padding)
        max_col = min(cols - 1, max_col + self.padding)

        # Extract the region
        result_grid = []
        for r in range(min_row, max_row + 1):
            row = []
            for c in range(min_col, max_col + 1):
                row.append(grid[r][c])
            result_grid.append(row)

        return result_grid

    def _extract_by_color(self, grid: Grid) -> Grid:
        """Extract a bounding box around cells of target color."""
        rows, cols = len(grid), len(grid[0])

        min_row, max_row = rows, -1
        min_col, max_col = cols, -1

        # Find bounding box around target color
        for r in range(rows):
            for c in range(cols):
                if Color(grid[r][c]) == self.target_color:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        # If target color not found, return empty grid
        if max_row == -1:
            return [[0]]

        # Apply padding
        min_row = max(0, min_row - self.padding)
        max_row = min(rows - 1, max_row + self.padding)
        min_col = max(0, min_col - self.padding)
        max_col = min(cols - 1, max_col + self.padding)

        # Extract the region
        result_grid = []
        for r in range(min_row, max_row + 1):
            row = []
            for c in range(min_col, max_col + 1):
                row.append(grid[r][c])
            result_grid.append(row)

        return result_grid

    def _extract_non_zero(self, grid: Grid) -> Grid:
        """Extract bounding box around all non-zero colors."""
        # Use bounding box logic with ignore_color=0
        old_ignore = self.ignore_color
        self.ignore_color = Color(0)
        result = self._extract_bounding_box(grid)
        self.ignore_color = old_ignore
        return result

    @classmethod
    def get_name(cls) -> str:
        return "extract_region"

    @classmethod
    def get_description(cls) -> str:
        return "Extract a specific region from the grid based on color patterns or conditions"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "extract_mode": {
                "type": "string",
                "required": False,
                "default": "bounding_box",
                "valid_values": ["bounding_box", "color", "non_zero"],
                "description": "Mode for extracting region"
            },
            "target_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Specific color to extract (required for 'color' mode)"
            },
            "ignore_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color to ignore when finding bounding box"
            },
            "padding": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Additional padding around the extracted region"
            }
        }


class ResizeOperation(CompositionOperation):
    """
    Resize a grid to new dimensions using various scaling strategies.

    Examples:
        # Resize to exact dimensions
        resize_exact = ResizeOperation(new_width=10, new_height=10, strategy="nearest")

        # Scale by a factor
        resize_scale = ResizeOperation(scale_factor=2.0, strategy="repeat")
    """

    def __init__(self, new_width: int | None = None, new_height: int | None = None,
                 scale_factor: float | None = None, strategy: str = "nearest",
                 fill_color: int = 0):
        """
        Initialize resize operation.

        Args:
            new_width: Target width (mutually exclusive with scale_factor)
            new_height: Target height (mutually exclusive with scale_factor)
            scale_factor: Scaling factor (mutually exclusive with new_width/new_height)
            strategy: Resizing strategy ("nearest", "repeat", "stretch")
            fill_color: Color to use when padding is needed
        """
        # Validate parameters
        if scale_factor is not None:
            if new_width is not None or new_height is not None:
                raise ValueError("Cannot specify both scale_factor and new dimensions")
            if scale_factor <= 0:
                raise ValueError("Scale factor must be positive")
        else:
            if new_width is None and new_height is None:
                raise ValueError("Must specify either scale_factor or new dimensions")
            if new_width is not None and new_width <= 0:
                raise ValueError("New width must be positive")
            if new_height is not None and new_height <= 0:
                raise ValueError("New height must be positive")

        valid_strategies = ["nearest", "repeat", "stretch"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")

        if not (0 <= fill_color <= 9):
            raise ValueError(f"Invalid fill color: {fill_color}")

        super().__init__(
            new_width=new_width,
            new_height=new_height,
            scale_factor=scale_factor,
            strategy=strategy,
            fill_color=fill_color
        )
        self.new_width = new_width
        self.new_height = new_height
        self.scale_factor = scale_factor
        self.strategy = strategy
        self.fill_color = Color(fill_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the resize operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot resize empty grid")

        try:
            rows, cols = len(grid), len(grid[0])

            # Calculate target dimensions
            if self.scale_factor is not None:
                target_width = int(cols * self.scale_factor)
                target_height = int(rows * self.scale_factor)
            else:
                target_width = self.new_width if self.new_width is not None else cols
                target_height = self.new_height if self.new_height is not None else rows

            # Apply resizing strategy
            if self.strategy == "nearest":
                result_grid = self._resize_nearest(grid, target_height, target_width)
            elif self.strategy == "repeat":
                result_grid = self._resize_repeat(grid, target_height, target_width)
            elif self.strategy == "stretch":
                result_grid = self._resize_stretch(grid, target_height, target_width)
            else:
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Unknown resize strategy: {self.strategy}"
                )

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _resize_nearest(self, grid: Grid, target_height: int, target_width: int) -> Grid:
        """Resize using nearest neighbor interpolation."""
        rows, cols = len(grid), len(grid[0])
        result_grid = []

        for r in range(target_height):
            row = []
            src_r = int(r * rows / target_height)
            src_r = min(src_r, rows - 1)

            for c in range(target_width):
                src_c = int(c * cols / target_width)
                src_c = min(src_c, cols - 1)
                row.append(grid[src_r][src_c])

            result_grid.append(row)

        return result_grid

    def _resize_repeat(self, grid: Grid, target_height: int, target_width: int) -> Grid:
        """Resize by repeating the pattern."""
        rows, cols = len(grid), len(grid[0])
        result_grid = []

        for r in range(target_height):
            row = []
            for c in range(target_width):
                row.append(grid[r % rows][c % cols])
            result_grid.append(row)

        return result_grid

    def _resize_stretch(self, grid: Grid, target_height: int, target_width: int) -> Grid:
        """Resize by stretching/compressing uniformly."""
        rows, cols = len(grid), len(grid[0])

        # If one dimension is larger and one is smaller, use fill_color for new areas
        result_grid = create_empty_grid(target_height, target_width, self.fill_color)

        copy_height = min(rows, target_height)
        copy_width = min(cols, target_width)

        for r in range(copy_height):
            for c in range(copy_width):
                result_grid[r][c] = grid[r][c]

        return result_grid

    @classmethod
    def get_name(cls) -> str:
        return "resize"

    @classmethod
    def get_description(cls) -> str:
        return "Resize a grid to new dimensions using various scaling strategies"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "new_width": {
                "type": "integer",
                "required": False,
                "description": "Target width (mutually exclusive with scale_factor)"
            },
            "new_height": {
                "type": "integer",
                "required": False,
                "description": "Target height (mutually exclusive with scale_factor)"
            },
            "scale_factor": {
                "type": "float",
                "required": False,
                "description": "Scaling factor (mutually exclusive with new_width/new_height)"
            },
            "strategy": {
                "type": "string",
                "required": False,
                "default": "nearest",
                "valid_values": ["nearest", "repeat", "stretch"],
                "description": "Resizing strategy"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to use when padding is needed"
            }
        }
