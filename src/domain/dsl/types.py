"""
Type definitions for the ARC DSL system.

This module defines all core types used throughout the DSL, including grids,
colors, positions, and other fundamental data structures for ARC transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, NewType

from src.infrastructure.config import TranspilerSandboxConfig

# Basic type definitions
Color = NewType('Color', int)  # Integer 0-9 representing ARC colors
Position = tuple[int, int]  # (row, column) coordinate
Grid = list[list[Color]]  # 2D grid of colors
Dimensions = tuple[int, int]  # (height, width) of a grid


class Direction(IntEnum):
    """Cardinal directions for transformations."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class RotationAngle(IntEnum):
    """Rotation angles in degrees."""
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


class FlipDirection(IntEnum):
    """Flip/mirror directions."""
    HORIZONTAL = 0  # Left-right flip
    VERTICAL = 1    # Up-down flip
    DIAGONAL_MAIN = 2  # Main diagonal (top-left to bottom-right)
    DIAGONAL_ANTI = 3  # Anti-diagonal (top-right to bottom-left)


@dataclass(frozen=True)
class GridRegion:
    """
    Represents a rectangular region within a grid.

    Attributes:
        top_left: Top-left corner position (row, col)
        bottom_right: Bottom-right corner position (row, col)
    """
    top_left: Position
    bottom_right: Position

    @property
    def width(self) -> int:
        """Get the width of the region."""
        return self.bottom_right[1] - self.top_left[1] + 1

    @property
    def height(self) -> int:
        """Get the height of the region."""
        return self.bottom_right[0] - self.top_left[0] + 1

    @property
    def area(self) -> int:
        """Get the area of the region."""
        return self.width * self.height

    def contains(self, position: Position) -> bool:
        """Check if a position is within this region."""
        row, col = position
        return (
            self.top_left[0] <= row <= self.bottom_right[0] and
            self.top_left[1] <= col <= self.bottom_right[1]
        )


@dataclass(frozen=True)
class ColorMapping:
    """
    Represents a mapping from one set of colors to another.

    Attributes:
        mapping: Dictionary mapping source colors to target colors
        default_color: Color to use for unmapped colors (None = keep original)
    """
    mapping: dict[Color, Color]
    default_color: Color | None = None

    def apply(self, color: Color) -> Color:
        """Apply the color mapping to a single color."""
        return self.mapping.get(color, self.default_color or color)


@dataclass(frozen=True)
class Pattern:
    """
    Represents a 2D pattern that can be searched for or applied to grids.

    Attributes:
        grid: The pattern as a 2D grid
        mask: Optional mask indicating which cells must match exactly
               (True = must match, False = wildcard, None = all must match)
    """
    grid: Grid
    mask: list[list[bool]] | None = None

    @property
    def dimensions(self) -> Dimensions:
        """Get the dimensions of the pattern."""
        return (len(self.grid), len(self.grid[0]) if self.grid else 0)

    def matches_at(self, target_grid: Grid, position: Position) -> bool:
        """
        Check if this pattern matches the target grid at the given position.

        Args:
            target_grid: The grid to check against
            position: The top-left position to check from

        Returns:
            True if the pattern matches at the position
        """
        row_offset, col_offset = position
        pattern_height, pattern_width = self.dimensions
        target_height, target_width = len(target_grid), len(target_grid[0]) if target_grid else 0

        # Check bounds
        if (row_offset + pattern_height > target_height or
            col_offset + pattern_width > target_width):
            return False

        # Check each cell
        for r in range(pattern_height):
            for c in range(pattern_width):
                # Skip if mask says this cell is a wildcard
                if self.mask and not self.mask[r][c]:
                    continue

                target_color = target_grid[row_offset + r][col_offset + c]
                pattern_color = self.grid[r][c]

                if target_color != pattern_color:
                    return False

        return True


@dataclass
class TransformationContext:
    """
    Context information passed to operations during execution.

    This allows operations to access metadata about the transformation
    being performed, such as the original input or intermediate results.
    """
    original_input: Grid
    current_grid: Grid
    step_number: int
    metadata: dict[str, Any]


# Validation functions for type safety

def is_valid_color(value: int) -> bool:
    """Check if a value is a valid ARC color (0-9)."""
    return isinstance(value, int) and 0 <= value <= 9


def is_valid_grid(grid: list[list[int]], config: TranspilerSandboxConfig | None = None) -> bool:
    """
    Check if a grid is valid for ARC.

    Args:
        grid: The grid to validate
        config: Optional configuration for grid size limits

    Returns:
        True if grid is valid (rectangular, colors 0-9, reasonable size)
    """
    if not grid or not isinstance(grid, list):
        return False

    # Check if it's a list of lists
    if not all(isinstance(row, list) for row in grid):
        return False

    # Check if rectangular
    if not grid:
        return True  # Empty grid is valid

    first_row_length = len(grid[0])
    if not all(len(row) == first_row_length for row in grid):
        return False

    # Check color values
    for row in grid:
        for cell in row:
            if not is_valid_color(cell):
                return False

    # Check reasonable size using configurable limits
    if config is None:
        config = TranspilerSandboxConfig()

    height, width = len(grid), first_row_length
    if height > config.max_grid_height or width > config.max_grid_width:
        return False

    return True


def is_valid_position(position: Position, grid: Grid) -> bool:
    """
    Check if a position is valid for the given grid.

    Args:
        position: The position to check (row, col)
        grid: The grid to check against

    Returns:
        True if position is within grid bounds
    """
    if not grid:
        return False

    row, col = position
    height, width = len(grid), len(grid[0])

    return 0 <= row < height and 0 <= col < width


# Utility functions for working with types

def create_empty_grid(height: int, width: int, fill_color: Color = Color(0)) -> Grid:
    """
    Create an empty grid filled with the specified color.

    Args:
        height: Grid height
        width: Grid width
        fill_color: Color to fill the grid with (default: 0/black)

    Returns:
        New grid filled with the specified color
    """
    return [[fill_color for _ in range(width)] for _ in range(height)]


def copy_grid(grid: Grid) -> Grid:
    """Create a deep copy of a grid."""
    return [row.copy() for row in grid]


def get_grid_dimensions(grid: Grid) -> Dimensions:
    """Get the dimensions of a grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid else 0)


def get_unique_colors(grid: Grid) -> list[Color]:
    """Get a sorted list of unique colors present in the grid."""
    colors = set()
    for row in grid:
        for color in row:
            colors.add(color)
    return sorted(colors)


def count_color_occurrences(grid: Grid) -> dict[Color, int]:
    """Count occurrences of each color in the grid."""
    counts: dict[Color, int] = {}
    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1
    return counts
