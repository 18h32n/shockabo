"""
Pattern-based operations for the ARC DSL.

This module contains operations that work with recurring structures or templates
in the grid, such as pattern matching, filling, replacement, etc.
"""

from typing import Any

from .base import OperationResult, PatternOperation
from .types import Color, Grid, Pattern, Position, TransformationContext


class PatternFillOperation(PatternOperation):
    """
    Fill connected regions of the same color with a new color.

    This is similar to a "flood fill" operation in image editing software.

    Examples:
        # Fill connected black (0) regions with red (2) starting from (0, 0)
        fill = PatternFillOperation(start_position=(0, 0), target_color=2)

        # Fill specific color regions
        fill = PatternFillOperation(source_color=0, target_color=2)
    """

    def __init__(self, target_color: int, start_position: Position | None = None,
                 source_color: int | None = None):
        """
        Initialize pattern fill operation.

        Args:
            target_color: Color to fill with
            start_position: Starting position for flood fill (if None, fill all matching regions)
            source_color: Color to replace (if None, use color at start_position)
        """
        if not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        if source_color is not None and not (0 <= source_color <= 9):
            raise ValueError(f"Invalid source color: {source_color}")

        super().__init__(
            target_color=target_color,
            start_position=start_position,
            source_color=source_color
        )
        self.target_color = Color(target_color)
        self.start_position = start_position
        self.source_color = Color(source_color) if source_color is not None else None

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the pattern fill operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot fill empty grid")

        try:
            result_grid = [row.copy() for row in grid]  # Deep copy

            if self.start_position is not None:
                # Single flood fill from start position
                self._flood_fill(result_grid, self.start_position)
            else:
                # Fill all regions of the source color
                if self.source_color is None:
                    return OperationResult(
                        success=False,
                        grid=grid,
                        error_message="Must specify either start_position or source_color"
                    )

                rows, cols = len(grid), len(grid[0])
                visited = [[False for _ in range(cols)] for _ in range(rows)]

                for r in range(rows):
                    for c in range(cols):
                        if not visited[r][c] and grid[r][c] == self.source_color:
                            self._flood_fill_with_visited(result_grid, (r, c), visited)

            return OperationResult(success=True, grid=result_grid)

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _flood_fill(self, grid: Grid, start: Position) -> None:
        """Perform flood fill from a starting position."""
        rows, cols = len(grid), len(grid[0])
        start_row, start_col = start

        if not (0 <= start_row < rows and 0 <= start_col < cols):
            return

        source_color = self.source_color or Color(grid[start_row][start_col])
        if source_color == self.target_color:
            return  # Nothing to fill

        stack = [start]
        visited = set()

        while stack:
            row, col = stack.pop()

            if (row, col) in visited:
                continue
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if grid[row][col] != source_color:
                continue

            visited.add((row, col))
            grid[row][col] = self.target_color

            # Add neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((row + dr, col + dc))

    def _flood_fill_with_visited(self, grid: Grid, start: Position, visited: list[list[bool]]) -> None:
        """Perform flood fill with shared visited array."""
        rows, cols = len(grid), len(grid[0])
        start_row, start_col = start

        source_color = Color(grid[start_row][start_col])
        if source_color == self.target_color:
            return

        stack = [start]

        while stack:
            row, col = stack.pop()

            if visited[row][col]:
                continue
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if grid[row][col] != source_color:
                continue

            visited[row][col] = True
            grid[row][col] = self.target_color

            # Add neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    stack.append((new_row, new_col))

    @classmethod
    def get_name(cls) -> str:
        return "pattern_fill"

    @classmethod
    def get_description(cls) -> str:
        return "Fill connected regions of the same color with a new color (flood fill)"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "target_color": {
                "type": "integer",
                "required": True,
                "valid_range": [0, 9],
                "description": "Color to fill with"
            },
            "start_position": {
                "type": "tuple",
                "required": False,
                "description": "Starting position for flood fill (row, col)"
            },
            "source_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color to replace (if None, use color at start_position)"
            }
        }


class PatternMatchOperation(PatternOperation):
    """
    Find all occurrences of a pattern in the grid.

    Examples:
        # Find 2x2 pattern of red (2) cells
        pattern_grid = [[2, 2], [2, 2]]
        matcher = PatternMatchOperation(pattern=pattern_grid)
    """

    def __init__(self, pattern: list[list[int]], mask: list[list[bool]] | None = None):
        """
        Initialize pattern match operation.

        Args:
            pattern: 2D pattern to search for
            mask: Optional mask indicating which cells must match (True = must match, False = wildcard)
        """
        if not pattern or not pattern[0]:
            raise ValueError("Pattern cannot be empty")

        # Validate pattern colors
        for row in pattern:
            for cell in row:
                if not (0 <= cell <= 9):
                    raise ValueError(f"Invalid pattern color: {cell}")

        super().__init__(pattern=pattern, mask=mask)
        self.pattern = Pattern(
            grid=[[Color(cell) for cell in row] for row in pattern],
            mask=mask
        )

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the pattern match operation."""
        if not grid:
            return OperationResult(
                success=True,
                grid=grid,
                metadata={"matches": []}
            )

        try:
            matches = []
            rows, cols = len(grid), len(grid[0])
            pattern_height, pattern_width = self.pattern.dimensions

            for r in range(rows - pattern_height + 1):
                for c in range(cols - pattern_width + 1):
                    if self.pattern.matches_at(grid, (r, c)):
                        matches.append((r, c))

            return OperationResult(
                success=True,
                grid=grid,
                metadata={"matches": matches, "match_count": len(matches)}
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "pattern_match"

    @classmethod
    def get_description(cls) -> str:
        return "Find all occurrences of a pattern in the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "pattern": {
                "type": "list",
                "required": True,
                "description": "2D pattern to search for"
            },
            "mask": {
                "type": "list",
                "required": False,
                "description": "Optional mask indicating which cells must match (True = must match, False = wildcard)"
            }
        }


class PatternReplaceOperation(PatternOperation):
    """
    Replace all occurrences of a pattern with another pattern.

    Examples:
        # Replace 2x2 red pattern with 2x2 blue pattern
        source = [[2, 2], [2, 2]]
        target = [[1, 1], [1, 1]]
        replace = PatternReplaceOperation(source_pattern=source, target_pattern=target)
    """

    def __init__(self, source_pattern: list[list[int]], target_pattern: list[list[int]],
                 source_mask: list[list[bool]] | None = None):
        """
        Initialize pattern replace operation.

        Args:
            source_pattern: Pattern to find and replace
            target_pattern: Pattern to replace with
            source_mask: Optional mask for source pattern matching
        """
        if not source_pattern or not target_pattern:
            raise ValueError("Patterns cannot be empty")

        # Patterns must have the same dimensions
        if (len(source_pattern) != len(target_pattern) or
            len(source_pattern[0]) != len(target_pattern[0])):
            raise ValueError("Source and target patterns must have the same dimensions")

        super().__init__(
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            source_mask=source_mask
        )

        self.source_pattern = Pattern(
            grid=[[Color(cell) for cell in row] for row in source_pattern],
            mask=source_mask
        )
        self.target_pattern = [[Color(cell) for cell in row] for row in target_pattern]

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the pattern replace operation."""
        if not grid:
            return OperationResult(success=True, grid=grid)

        try:
            result_grid = [row.copy() for row in grid]
            rows, cols = len(grid), len(grid[0])
            pattern_height, pattern_width = self.source_pattern.dimensions

            replacements = 0

            for r in range(rows - pattern_height + 1):
                for c in range(cols - pattern_width + 1):
                    if self.source_pattern.matches_at(result_grid, (r, c)):
                        # Replace the pattern
                        for pr in range(pattern_height):
                            for pc in range(pattern_width):
                                result_grid[r + pr][c + pc] = self.target_pattern[pr][pc]
                        replacements += 1

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={"replacements": replacements}
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "pattern_replace"

    @classmethod
    def get_description(cls) -> str:
        return "Replace all occurrences of source_pattern with target_pattern"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "source_pattern": {
                "type": "list",
                "required": True,
                "description": "Pattern to find and replace"
            },
            "target_pattern": {
                "type": "list",
                "required": True,
                "description": "Pattern to replace with"
            },
            "source_mask": {
                "type": "list",
                "required": False,
                "description": "Optional mask for source pattern matching"
            }
        }


class FloodFillOperation(PatternOperation):
    """
    Simple flood fill operation from a specific position.

    This operation fills a connected region of the same color with a new color,
    starting from a specified position. This is simpler than PatternFillOperation
    as it always starts from a single position.

    Examples:
        # Flood fill from position (2, 3) with red (2)
        flood = FloodFillOperation(start_position=(2, 3), fill_color=2)
    """

    def __init__(self, start_position: Position, fill_color: int):
        """
        Initialize flood fill operation.

        Args:
            start_position: Starting position for flood fill (row, col)
            fill_color: Color to fill with
        """
        if not (0 <= fill_color <= 9):
            raise ValueError(f"Invalid fill color: {fill_color}")

        super().__init__(start_position=start_position, fill_color=fill_color)
        self.start_position = start_position
        self.fill_color = Color(fill_color)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the flood fill operation."""
        if not grid:
            return OperationResult(success=False, grid=grid, error_message="Cannot flood fill empty grid")

        try:
            rows, cols = len(grid), len(grid[0])
            start_row, start_col = self.start_position

            # Validate start position
            if not (0 <= start_row < rows and 0 <= start_col < cols):
                return OperationResult(
                    success=False,
                    grid=grid,
                    error_message=f"Start position {self.start_position} is out of bounds for {rows}x{cols} grid"
                )

            result_grid = [row.copy() for row in grid]  # Deep copy
            source_color = Color(grid[start_row][start_col])

            # If source and target colors are the same, no work needed
            if source_color == self.fill_color:
                return OperationResult(
                    success=True,
                    grid=result_grid,
                    metadata={
                        "start_position": self.start_position,
                        "source_color": int(source_color),
                        "fill_color": int(self.fill_color),
                        "filled_cells": 0
                    }
                )

            # Perform flood fill using iterative approach
            stack = [self.start_position]
            visited = set()
            filled_cells = 0

            while stack:
                row, col = stack.pop()

                if (row, col) in visited:
                    continue
                if not (0 <= row < rows and 0 <= col < cols):
                    continue
                if result_grid[row][col] != source_color:
                    continue

                visited.add((row, col))
                result_grid[row][col] = self.fill_color
                filled_cells += 1

                # Add 4-connected neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((row + dr, col + dc))

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "start_position": self.start_position,
                    "source_color": int(source_color),
                    "fill_color": int(self.fill_color),
                    "filled_cells": filled_cells
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    @classmethod
    def get_name(cls) -> str:
        return "flood_fill"

    @classmethod
    def get_description(cls) -> str:
        return "Flood fill connected region from start position with specified color"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "start_position": {
                "type": "tuple",
                "required": True,
                "description": "Starting position for flood fill (row, col)"
            },
            "fill_color": {
                "type": "integer",
                "required": True,
                "valid_range": [0, 9],
                "description": "Color to fill with"
            }
        }
