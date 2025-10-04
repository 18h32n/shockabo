"""
Color manipulation operations for the ARC DSL.

This module contains operations that modify the colors in the grid while typically
preserving spatial relationships, such as color mapping, filtering, replacement, etc.
"""

from typing import Any

from .base import ColorOperation, OperationResult
from .types import Color, ColorMapping, Grid, TransformationContext


class ColorMapOperation(ColorOperation):
    """
    Map colors in the grid according to a specified mapping.

    Examples:
        # Map red (2) to blue (1) and green (3) to yellow (4)
        color_map = ColorMapOperation(mapping={2: 1, 3: 4})

        # Map with default color for unmapped values
        color_map = ColorMapOperation(mapping={2: 1}, default_color=0)
    """

    def __init__(self, mapping: dict[int, int], default_color: int | None = None):
        """
        Initialize color mapping operation.

        Args:
            mapping: Dictionary mapping source colors to target colors
            default_color: Color to use for unmapped colors (None = keep original)
        """
        # Validate color values
        for source, target in mapping.items():
            if not (0 <= source <= 9):
                raise ValueError(f"Invalid source color: {source}")
            if not (0 <= target <= 9):
                raise ValueError(f"Invalid target color: {target}")

        if default_color is not None and not (0 <= default_color <= 9):
            raise ValueError(f"Invalid default color: {default_color}")

        super().__init__(mapping=mapping, default_color=default_color)
        self.color_mapping = ColorMapping(
            mapping={Color(k): Color(v) for k, v in mapping.items()},
            default_color=Color(default_color) if default_color is not None else None
        )

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the color mapping operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot map colors in empty grid")

        try:
            result_grid = [
                [self.color_mapping.apply(Color(cell)) for cell in row]
                for row in grid
            ]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "color_map"

    @classmethod
    def get_description(cls) -> str:
        return "Map colors in the grid according to the specified mapping"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "mapping": {
                "type": "dict",
                "required": True,
                "description": "Dictionary mapping source colors to target colors"
            },
            "default_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color to use for unmapped colors (None = keep original)"
            }
        }


class ColorFilterOperation(ColorOperation):
    """
    Filter the grid to keep only specified colors, replacing others with a fill color.

    Examples:
        # Keep only red (2) and blue (1), fill others with black (0)
        color_filter = ColorFilterOperation(keep_colors=[2, 1], fill_color=0)
    """

    def __init__(self, keep_colors: list[int], fill_color: int = 0):
        """
        Initialize color filter operation.

        Args:
            keep_colors: List of colors to preserve
            fill_color: Color to use for filtered out colors
        """
        # Validate colors
        for color in keep_colors:
            if not (0 <= color <= 9):
                raise ValueError(f"Invalid keep color: {color}")

        if not (0 <= fill_color <= 9):
            raise ValueError(f"Invalid fill color: {fill_color}")

        super().__init__(keep_colors=keep_colors, fill_color=fill_color)
        self.keep_colors: set[Color] = {Color(c) for c in keep_colors}
        self.fill_color = Color(fill_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the color filter operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot filter empty grid")

        try:
            result_grid = [
                [cell if Color(cell) in self.keep_colors else self.fill_color for cell in row]
                for row in grid
            ]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "color_filter"

    @classmethod
    def get_description(cls) -> str:
        return "Keep only specified colors, replace others with fill_color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "keep_colors": {
                "type": "list",
                "required": True,
                "description": "List of colors to preserve"
            },
            "fill_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to use for filtered out colors"
            }
        }


class ColorReplaceOperation(ColorOperation):
    """
    Replace all occurrences of a specific color with another color.

    Examples:
        # Replace all red (2) with blue (1)
        color_replace = ColorReplaceOperation(source_color=2, target_color=1)
    """

    def __init__(self, source_color: int, target_color: int):
        """
        Initialize color replace operation.

        Args:
            source_color: Color to replace
            target_color: Color to replace with
        """
        # Validate colors
        if not (0 <= source_color <= 9):
            raise ValueError(f"Invalid source color: {source_color}")
        if not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        super().__init__(source_color=source_color, target_color=target_color)
        self.source_color = Color(source_color)
        self.target_color = Color(target_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the color replace operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot replace colors in empty grid")

        try:
            result_grid = [
                [self.target_color if Color(cell) == self.source_color else cell for cell in row]
                for row in grid
            ]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "color_replace"

    @classmethod
    def get_description(cls) -> str:
        return "Replace all occurrences of source_color with target_color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "source_color": {
                "type": "integer",
                "required": True,
                "valid_range": [0, 9],
                "description": "Color to replace"
            },
            "target_color": {
                "type": "integer",
                "required": True,
                "valid_range": [0, 9],
                "description": "Color to replace with"
            }
        }


class ColorInvertOperation(ColorOperation):
    """
    Invert colors in the grid using 9-complement (9-color).

    This operation subtracts each color value from 9 to create an inverted color scheme.

    Examples:
        # Invert all colors: 0->9, 1->8, 2->7, etc.
        invert = ColorInvertOperation()
    """

    def __init__(self):
        """Initialize color invert operation."""
        super().__init__()

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the color invert operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot invert colors in empty grid")

        try:
            result_grid = [
                [Color(9 - cell) for cell in row]
                for row in grid
            ]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "color_invert"

    @classmethod
    def get_description(cls) -> str:
        return "Invert colors using 9-complement (0->9, 1->8, 2->7, etc.)"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {}


class ColorThresholdOperation(ColorOperation):
    """
    Apply binary threshold to colors based on a threshold value.

    Colors below the threshold are set to low_color, colors above or equal
    to the threshold are set to high_color.

    Examples:
        # Convert to binary: colors 0-4 -> black (0), colors 5-9 -> white (9)
        threshold = ColorThresholdOperation(threshold=5, low_color=0, high_color=9)
    """

    def __init__(self, threshold: int, low_color: int = 0, high_color: int = 9):
        """
        Initialize color threshold operation.

        Args:
            threshold: Threshold value (0-9)
            low_color: Color for values below threshold
            high_color: Color for values >= threshold
        """
        # Validate parameters
        if not (0 <= threshold <= 9):
            raise ValueError(f"Invalid threshold: {threshold}")
        if not (0 <= low_color <= 9):
            raise ValueError(f"Invalid low_color: {low_color}")
        if not (0 <= high_color <= 9):
            raise ValueError(f"Invalid high_color: {high_color}")

        super().__init__(threshold=threshold, low_color=low_color, high_color=high_color)
        self.threshold = threshold
        self.low_color = Color(low_color)
        self.high_color = Color(high_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the color threshold operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot threshold empty grid")

        try:
            result_grid = [
                [self.high_color if cell >= self.threshold else self.low_color for cell in row]
                for row in grid
            ]

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "color_threshold"

    @classmethod
    def get_description(cls) -> str:
        return "Apply binary threshold: values < threshold -> low_color, values >= threshold -> high_color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "threshold": {
                "type": "integer",
                "required": True,
                "valid_range": [0, 9],
                "description": "Threshold value"
            },
            "low_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color for values below threshold"
            },
            "high_color": {
                "type": "integer",
                "required": False,
                "default": 9,
                "valid_range": [0, 9],
                "description": "Color for values >= threshold"
            }
        }
