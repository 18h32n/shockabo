"""
Edge and boundary detection operations for the ARC DSL.

This module contains operations for detecting edges, boundaries, and contours
in grids, useful for shape analysis and object detection.
"""

from typing import Any

from .base import OperationResult, PatternOperation
from .types import Grid, Position, TransformationContext


class EdgeDetectionOperation(PatternOperation):
    """
    Detect edges in a grid where colors change.

    Edges are locations where adjacent cells have different colors.

    Examples:
        # Detect all edges with 4-connectivity
        edges = EdgeDetectionOperation(edge_color=1)

        # Detect edges only for specific color transitions
        edges = EdgeDetectionOperation(from_color=2, to_color=0, edge_color=1)
    """

    def __init__(self, edge_color: int = 1, from_color: int | None = None,
                 to_color: int | None = None, connectivity: int = 4):
        """
        Initialize edge detection operation.

        Args:
            edge_color: Color to mark edges with
            from_color: Detect edges only from this color (None = any)
            to_color: Detect edges only to this color (None = any)
            connectivity: 4 or 8 connectivity for edge detection
        """
        if not (0 <= edge_color <= 9):
            raise ValueError(f"Invalid edge color: {edge_color}")

        if from_color is not None and not (0 <= from_color <= 9):
            raise ValueError(f"Invalid from color: {from_color}")

        if to_color is not None and not (0 <= to_color <= 9):
            raise ValueError(f"Invalid to color: {to_color}")

        if connectivity not in [4, 8]:
            raise ValueError("Connectivity must be 4 or 8")

        super().__init__(
            edge_color=edge_color,
            from_color=from_color,
            to_color=to_color,
            connectivity=connectivity
        )
        self.edge_color = edge_color
        self.from_color = from_color if from_color is not None else None
        self.to_color = to_color if to_color is not None else None
        self.connectivity = connectivity

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the edge detection operation."""
        if not grid:
            return OperationResult(success=True, grid=grid)

        try:
            rows, cols = len(grid), len(grid[0])
            result_grid = [[0 for _ in range(cols)] for _ in range(rows)]
            edge_count = 0

            for r in range(rows):
                for c in range(cols):
                    current_color = grid[r][c]

                    # Check if this is from_color (if specified)
                    if self.from_color is not None and current_color != self.from_color:
                        continue

                    # Check neighbors
                    is_edge = False
                    neighbors = self._get_neighbors(r, c, rows, cols)

                    for nr, nc in neighbors:
                        neighbor_color = grid[nr][nc]

                        # Check if colors differ
                        if current_color != neighbor_color:
                            # Check if transition matches to_color (if specified)
                            if self.to_color is None or neighbor_color == self.to_color:
                                is_edge = True
                                break

                    if is_edge:
                        result_grid[r][c] = self.edge_color
                        edge_count += 1

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={"edge_count": edge_count}
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _get_neighbors(self, r: int, c: int, rows: int, cols: int) -> list[Position]:
        """Get valid neighbor positions based on connectivity."""
        neighbors = []

        # 4-connectivity neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))

        # Additional neighbors for 8-connectivity
        if self.connectivity == 8:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))

        return neighbors

    @classmethod
    def get_name(cls) -> str:
        return "edge_detection"

    @classmethod
    def get_description(cls) -> str:
        return "Detect edges where colors change in the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "edge_color": {
                "type": "integer",
                "required": False,
                "default": 1,
                "valid_range": [0, 9],
                "description": "Color to mark edges with"
            },
            "from_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Detect edges only from this color"
            },
            "to_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Detect edges only to this color"
            },
            "connectivity": {
                "type": "integer",
                "required": False,
                "default": 4,
                "valid_values": [4, 8],
                "description": "Connectivity for edge detection"
            }
        }


class BoundaryTracingOperation(PatternOperation):
    """
    Trace the outer boundary of objects in the grid.

    Finds and marks the outermost cells of connected regions.

    Examples:
        # Trace boundaries of all non-zero objects
        boundaries = BoundaryTracingOperation(target_color=None, boundary_color=1)

        # Trace boundary of specific color objects
        boundaries = BoundaryTracingOperation(target_color=5, boundary_color=2)
    """

    def __init__(self, target_color: int | None = None, boundary_color: int = 1,
                 background_color: int = 0, include_inner: bool = False):
        """
        Initialize boundary tracing operation.

        Args:
            target_color: Color of objects to trace (None = all non-background)
            boundary_color: Color to mark boundaries with
            background_color: Color considered as background
            include_inner: Whether to include inner boundaries (holes)
        """
        if not (0 <= boundary_color <= 9):
            raise ValueError(f"Invalid boundary color: {boundary_color}")

        if not (0 <= background_color <= 9):
            raise ValueError(f"Invalid background color: {background_color}")

        if target_color is not None and not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        super().__init__(
            target_color=target_color,
            boundary_color=boundary_color,
            background_color=background_color,
            include_inner=include_inner
        )
        self.target_color = target_color if target_color is not None else None
        self.boundary_color = boundary_color
        self.background_color = background_color
        self.include_inner = include_inner

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the boundary tracing operation."""
        if not grid:
            return OperationResult(success=True, grid=grid)

        try:
            rows, cols = len(grid), len(grid[0])
            result_grid = [[self.background_color for _ in range(cols)] for _ in range(rows)]
            boundary_count = 0

            for r in range(rows):
                for c in range(cols):
                    color = grid[r][c]

                    # Skip background
                    if color == self.background_color:
                        continue

                    # Check if this is target color (if specified)
                    if self.target_color is not None and color != self.target_color:
                        continue

                    # Check if this is a boundary cell
                    is_boundary = self._is_boundary_cell(grid, r, c, rows, cols)

                    if is_boundary:
                        result_grid[r][c] = self.boundary_color
                        boundary_count += 1

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "boundary_count": boundary_count,
                    "include_inner": self.include_inner
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _is_boundary_cell(self, grid: Grid, r: int, c: int, rows: int, cols: int) -> bool:
        """Check if a cell is on the boundary of an object."""
        current_color = grid[r][c]

        # Check 4-neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Edge of grid counts as boundary
            if not (0 <= nr < rows and 0 <= nc < cols):
                return True

            neighbor_color = grid[nr][nc]

            # Outer boundary: neighbor is background
            if neighbor_color == self.background_color:
                return True

            # Inner boundary (hole): neighbor is different non-background
            if self.include_inner and neighbor_color != current_color:
                return True

        return False

    @classmethod
    def get_name(cls) -> str:
        return "boundary_tracing"

    @classmethod
    def get_description(cls) -> str:
        return "Trace the outer boundary of objects in the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "target_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color of objects to trace"
            },
            "boundary_color": {
                "type": "integer",
                "required": False,
                "default": 1,
                "valid_range": [0, 9],
                "description": "Color to mark boundaries with"
            },
            "background_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color considered as background"
            },
            "include_inner": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Whether to include inner boundaries"
            }
        }


class ContourExtractionOperation(PatternOperation):
    """
    Extract contours (connected boundaries) from the grid.

    Similar to boundary tracing but returns ordered lists of boundary points.

    Examples:
        # Extract all contours
        contours = ContourExtractionOperation()

        # Extract contours of specific objects
        contours = ContourExtractionOperation(target_color=3)
    """

    def __init__(self, target_color: int | None = None, background_color: int = 0,
                 min_length: int = 3):
        """
        Initialize contour extraction operation.

        Args:
            target_color: Color of objects to extract contours from
            background_color: Color considered as background
            min_length: Minimum contour length to include
        """
        if not (0 <= background_color <= 9):
            raise ValueError(f"Invalid background color: {background_color}")

        if target_color is not None and not (0 <= target_color <= 9):
            raise ValueError(f"Invalid target color: {target_color}")

        if min_length < 1:
            raise ValueError("Minimum contour length must be at least 1")

        super().__init__(
            target_color=target_color,
            background_color=background_color,
            min_length=min_length
        )
        self.target_color = target_color if target_color is not None else None
        self.background_color = background_color
        self.min_length = min_length

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """Execute the contour extraction operation."""
        if not grid:
            return OperationResult(success=True, grid=grid, metadata={"contours": []})

        try:
            # First, find all boundary points
            boundary_op = BoundaryTracingOperation(
                target_color=self.target_color,
                boundary_color=1,
                background_color=self.background_color
            )
            boundary_result = boundary_op.execute(grid)

            if not boundary_result.success:
                return boundary_result

            # Extract connected contours from boundary points
            contours = self._extract_contours(boundary_result.grid)

            # Filter by minimum length
            valid_contours = [c for c in contours if len(c) >= self.min_length]

            # Create visualization grid
            rows, cols = len(grid), len(grid[0])
            result_grid = [[self.background_color for _ in range(cols)] for _ in range(rows)]

            # Draw contours with different colors
            for i, contour in enumerate(valid_contours):
                contour_color = (i % 9) + 1  # Use colors 1-9
                for r, c in contour:
                    result_grid[r][c] = contour_color

            return OperationResult(
                success=True,
                grid=result_grid,
                metadata={
                    "contours": [list(c) for c in valid_contours],
                    "contour_count": len(valid_contours),
                    "contour_lengths": [len(c) for c in valid_contours]
                }
            )

        except Exception as e:
            return OperationResult(success=False, grid=grid, error_message=str(e))

    def _extract_contours(self, boundary_grid: Grid) -> list[list[Position]]:
        """Extract ordered contours from boundary grid."""
        rows, cols = len(boundary_grid), len(boundary_grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        contours = []

        for r in range(rows):
            for c in range(cols):
                if boundary_grid[r][c] != 0 and not visited[r][c]:
                    # Start new contour
                    contour = self._trace_contour(boundary_grid, (r, c), visited)
                    if contour:
                        contours.append(contour)

        return contours

    def _trace_contour(self, grid: Grid, start: Position, visited: list[list[bool]]) -> list[Position]:
        """Trace a single contour starting from a boundary point."""
        contour = []
        stack = [start]
        rows, cols = len(grid), len(grid[0])

        while stack:
            r, c = stack.pop()

            if visited[r][c]:
                continue

            visited[r][c] = True
            contour.append((r, c))

            # Check 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr][nc] != 0 and not visited[nr][nc]):
                        stack.append((nr, nc))

        return contour

    @classmethod
    def get_name(cls) -> str:
        return "contour_extraction"

    @classmethod
    def get_description(cls) -> str:
        return "Extract contours (connected boundaries) from the grid"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "target_color": {
                "type": "integer",
                "required": False,
                "valid_range": [0, 9],
                "description": "Color of objects to extract contours from"
            },
            "background_color": {
                "type": "integer",
                "required": False,
                "default": 0,
                "valid_range": [0, 9],
                "description": "Color considered as background"
            },
            "min_length": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "Minimum contour length to include"
            }
        }
