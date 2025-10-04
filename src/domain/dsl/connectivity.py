"""
Connected component operations for the ARC DSL.

This module contains operations for analyzing and manipulating connected regions
in grids, such as component labeling, isolation, filtering, etc.
"""

from typing import Any

from .base import OperationResult, PatternOperation
from .types import Grid, Position, TransformationContext


class ConnectedComponentsOperation(PatternOperation):
    """
    Find and label connected components in the grid.

    Connected components are regions of the same color that are connected
    through adjacent cells (4-connectivity by default).

    Examples:
        # Find components of color 2 with 4-connectivity
        components = ConnectedComponentsOperation(target_color=2, connectivity=4)

        # Find all components (ignore background color 0)
        components = ConnectedComponentsOperation(background_color=0)
    """

    def __init__(self, target_color: int | None = None, background_color: int | None = None,
                 connectivity: int = 4):
        """
        Initialize connected components operation.

        Args:
            target_color: Specific color to find components for (None = all colors)
            background_color: Color to treat as background (None = no background)
            connectivity: 4 or 8 connectivity
        """
        if target_color is not None and not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        if background_color is not None and not (0 <= background_color <= 9):
            raise ValueError(f"Invalid background color: {background_color}")

        if connectivity not in [4, 8]:
            raise ValueError("Connectivity must be 4 or 8")

        super().__init__(
            target_color=target_color,
            background_color=background_color,
            connectivity=connectivity
        )
        self.target_color = target_color if target_color is not None else None
        self.background_color = background_color if background_color is not None else None
        self.connectivity = connectivity

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the connected components operation."""
        if not grid:
            return OperationResult(
                success=True,
                grid=grid,
                metadata={"components": [], "component_count": 0}
            )

        try:
            components = self._find_components(grid)

            # Create result grid with component labels
            rows, cols = len(grid), len(grid[0])
            result_grid = [[0 for _ in range(cols)] for _ in range(rows)]

            for i, component in enumerate(components, 1):
                for pos in component:
                    row, col = pos
                    if 0 <= row < rows and 0 <= col < cols:
                        result_grid[row][col] = i

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "components": [list(comp) for comp in components],
                    "component_count": len(components),
                    "component_sizes": [len(comp) for comp in components]
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _find_components(self, grid: Grid) -> list[set[Position]]:
        """Find all connected components in the grid."""
        rows, cols = len(grid), len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        components = []

        for r in range(rows):
            for c in range(cols):
                if visited[r][c]:
                    continue

                color = grid[r][c]

                # Skip background color
                if self.background_color is not None and color == self.background_color:
                    visited[r][c] = True
                    continue

                # Skip if not target color (when target color is specified)
                if self.target_color is not None and color != self.target_color:
                    visited[r][c] = True
                    continue

                # Start new component
                component = self._flood_fill_component(grid, (r, c), color, visited)
                if component:
                    components.append(component)

        return components

    def _flood_fill_component(self, grid: Grid, start: Position, color: int,
                             visited: list[list[bool]]) -> set[Position]:
        """Flood fill to find all cells in a connected component."""
        rows, cols = len(grid), len(grid[0])
        component = set()
        stack = [start]

        while stack:
            r, c = stack.pop()

            # Check bounds first
            if not (0 <= r < rows and 0 <= c < cols):
                continue

            if (r, c) in component or visited[r][c]:
                continue

            if grid[r][c] != color:
                continue

            component.add((r, c))
            visited[r][c] = True

            # Add neighbors based on connectivity
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]  # 4-connectivity

            if self.connectivity == 8:
                neighbors.extend([(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)])

            stack.extend(neighbors)

        return component

    @classmethod
    def get_name(cls) -> str:
        return "connected_components"

    @classmethod
    def get_description(cls) -> str:
        return "Find and label connected components in the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "target_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Specific color to find components for (None = all colors)"
            },
            "background_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color to treat as background (None = no background)"
            },
            "connectivity": {
                "type": "integer",
                "required": False,
                "default": 4,
                "valid_values": [4, 8],
                "description": "4 or 8 connectivity"
            }
        }


class FilterComponentsOperation(PatternOperation):
    """
    Filter connected components based on size or other criteria.

    Examples:
        # Keep only components with 5 or more cells
        filter_large = FilterComponentsOperation(min_size=5)

        # Keep only the largest component
        filter_largest = FilterComponentsOperation(keep_largest=True)
    """

    def __init__(self, min_size: int | None = None, max_size: int | None = None,
                 keep_largest: bool = False, keep_smallest: bool = False,
                 background_color: int = 0):
        """
        Initialize component filter operation.

        Args:
            min_size: Minimum component size to keep
            max_size: Maximum component size to keep
            keep_largest: Keep only the largest component
            keep_smallest: Keep only the smallest component
            background_color: Color to use for filtered out areas
        """
        if not (0 <= background_color <= 9):
            raise ValueError(f"Invalid background color: {background_color}")

        super().__init__(
            min_size=min_size,
            max_size=max_size,
            keep_largest=keep_largest,
            keep_smallest=keep_smallest,
            background_color=background_color
        )

        self.min_size = min_size
        self.max_size = max_size
        self.keep_largest = keep_largest
        self.keep_smallest = keep_smallest
        self.background_color = background_color

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the component filter operation."""
        if not grid:
            return OperationResult(success=True, grid=grid)

        try:
            # First find all components
            components_op = ConnectedComponentsOperation(background_color=self.background_color)
            components_result = components_op.execute(grid)

            if not components_result.success:
                return components_result

            components = components_result.metadata["components"]

            # Filter components based on criteria
            filtered_components = self._filter_components(components)

            # Create result grid
            result_grid = [[self.background_color for _ in range(len(grid[0]))]
                          for _ in range(len(grid))]

            # Fill in the kept components with original colors
            for component in filtered_components:
                for r, c in component:
                    result_grid[r][c] = grid[r][c]

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "original_components": len(components),
                    "filtered_components": len(filtered_components)
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _filter_components(self, components: list[list[Position]]) -> list[list[Position]]:
        """Filter components based on the specified criteria."""
        if not components:
            return []

        # Convert to sets for easier processing
        component_sets = [set(comp) for comp in components]
        filtered = []

        if self.keep_largest:
            # Keep only the largest component(s)
            max_size = max(len(comp) for comp in component_sets)
            filtered = [list(comp) for comp in component_sets if len(comp) == max_size]
        elif self.keep_smallest:
            # Keep only the smallest component(s)
            min_size = min(len(comp) for comp in component_sets)
            filtered = [list(comp) for comp in component_sets if len(comp) == min_size]
        else:
            # Filter by size range
            for comp in component_sets:
                size = len(comp)

                if self.min_size is not None and size < self.min_size:
                    continue

                if self.max_size is not None and size > self.max_size:
                    continue

                filtered.append(list(comp))

        return filtered

    @classmethod
    def get_name(cls) -> str:
        return "filter_components"

    @classmethod
    def get_description(cls) -> str:
        return "Filter connected components based on size or other criteria"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "min_size": {
                "type": "integer",
                "required": False,
                "description": "Minimum component size to keep"
            },
            "max_size": {
                "type": "integer",
                "required": False,
                "description": "Maximum component size to keep"
            },
            "keep_largest": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Keep only the largest component"
            },
            "keep_smallest": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Keep only the smallest component"
            },
            "background_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color to use for filtered out areas"
            }
        }
