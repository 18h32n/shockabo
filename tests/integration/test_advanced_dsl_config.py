"""
Configuration and utilities for advanced DSL operation tests.

This module provides shared configuration, fixtures, and utility functions
for testing advanced DSL operations.
"""

from typing import Any

import pytest

from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxExecutor


class DSLTestConfig:
    """Configuration for DSL integration tests."""

    # Performance thresholds (in milliseconds)
    OPERATION_TIMEOUT_MS = 5000
    SIMPLE_OPERATION_MAX_MS = 50
    COMPLEX_OPERATION_MAX_MS = 200
    CHAIN_OPERATION_MAX_MS = 500

    # Grid size limits for testing
    MAX_TEST_GRID_SIZE = 30
    PERFORMANCE_TEST_GRID_SIZE = 20
    STRESS_TEST_GRID_SIZE = 25

    # Default colors for testing
    DEFAULT_BACKGROUND_COLOR = 0
    DEFAULT_FOREGROUND_COLOR = 1
    DEFAULT_BOUNDARY_COLOR = 9
    DEFAULT_EDGE_COLOR = 8


class DSLTestUtilities:
    """Utility functions for DSL testing."""

    @staticmethod
    def create_test_grid(rows: int, cols: int, pattern: str = "sequential") -> list[list[int]]:
        """Create test grids with different patterns."""
        if pattern == "sequential":
            return [[i * cols + j for j in range(cols)] for i in range(rows)]
        elif pattern == "checkerboard":
            return [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
        elif pattern == "random":
            import random
            return [[random.randint(0, 9) for j in range(cols)] for i in range(rows)]
        elif pattern == "uniform":
            return [[1 for j in range(cols)] for i in range(rows)]
        elif pattern == "border":
            grid = [[0 for j in range(cols)] for i in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if i == 0 or i == rows-1 or j == 0 or j == cols-1:
                        grid[i][j] = 1
            return grid
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    @staticmethod
    def create_square_object(size: int, color: int = 1, background: int = 0) -> list[list[int]]:
        """Create a square object for testing."""
        grid_size = size + 4  # Add padding
        grid = [[background for _ in range(grid_size)] for _ in range(grid_size)]

        start = 2
        for i in range(start, start + size):
            for j in range(start, start + size):
                grid[i][j] = color

        return grid

    @staticmethod
    def create_l_shape(color: int = 1, background: int = 0) -> list[list[int]]:
        """Create an L-shaped object for testing."""
        grid = [[background for _ in range(5)] for _ in range(5)]

        # Create L shape
        positions = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]
        for r, c in positions:
            grid[r][c] = color

        return grid

    @staticmethod
    def create_multiple_components(background: int = 0) -> list[list[int]]:
        """Create grid with multiple connected components of different sizes."""
        grid = [[background for _ in range(8)] for _ in range(8)]

        # Component 1: size 4 (2x2 square)
        for r in range(1, 3):
            for c in range(1, 3):
                grid[r][c] = 1

        # Component 2: size 2 (vertical line)
        grid[1][5] = 2
        grid[2][5] = 2

        # Component 3: size 6 (L shape)
        positions = [(4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (6, 4)]
        for r, c in positions:
            grid[r][c] = 3

        # Component 4: size 1 (single cell)
        grid[4][6] = 4

        return grid

    @staticmethod
    def assert_grid_dimensions(grid: list[list[int]], expected_rows: int, expected_cols: int):
        """Assert that grid has expected dimensions."""
        assert len(grid) == expected_rows, f"Expected {expected_rows} rows, got {len(grid)}"
        if expected_rows > 0:
            assert len(grid[0]) == expected_cols, f"Expected {expected_cols} cols, got {len(grid[0])}"

    @staticmethod
    def assert_grid_contains_color(grid: list[list[int]], color: int):
        """Assert that grid contains at least one cell of the specified color."""
        found = any(color in row for row in grid)
        assert found, f"Grid does not contain color {color}"

    @staticmethod
    def assert_grid_does_not_contain_color(grid: list[list[int]], color: int):
        """Assert that grid does not contain any cells of the specified color."""
        found = any(color in row for row in grid)
        assert not found, f"Grid unexpectedly contains color {color}"

    @staticmethod
    def count_color_cells(grid: list[list[int]], color: int) -> int:
        """Count the number of cells with the specified color."""
        return sum(row.count(color) for row in grid)

    @staticmethod
    def get_unique_colors(grid: list[list[int]]) -> set:
        """Get the set of unique colors in the grid."""
        return {cell for row in grid for cell in row}


@pytest.fixture
def dsl_transpiler():
    """Fixture providing a DSL transpiler instance."""
    return PythonTranspiler()


@pytest.fixture
def dsl_executor():
    """Fixture providing a DSL executor instance."""
    return SandboxExecutor()


@pytest.fixture
def dsl_test_config():
    """Fixture providing test configuration."""
    return DSLTestConfig()


@pytest.fixture
def dsl_utilities():
    """Fixture providing test utilities."""
    return DSLTestUtilities()


def execute_dsl_program_with_validation(
    transpiler: PythonTranspiler,
    executor: SandboxExecutor,
    program: dict[str, Any],
    grid: list[list[int]],
    config: DSLTestConfig | None = None
) -> Any:
    """
    Execute a DSL program with validation and error handling.
    
    Args:
        transpiler: DSL transpiler instance
        executor: DSL executor instance
        program: DSL program specification
        grid: Input grid
        config: Test configuration (optional)
    
    Returns:
        Execution result with validation
    """
    if config is None:
        config = DSLTestConfig()

    # Validate input grid size
    if len(grid) > config.MAX_TEST_GRID_SIZE or (grid and len(grid[0]) > config.MAX_TEST_GRID_SIZE):
        raise ValueError(f"Grid size exceeds maximum test size of {config.MAX_TEST_GRID_SIZE}x{config.MAX_TEST_GRID_SIZE}")

    # Transpile the program
    try:
        transpilation_result = transpiler.transpile(program)
    except Exception as e:
        raise RuntimeError(f"Transpilation failed: {e}")

    # Execute the transpiled code
    try:
        execution_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            grid
        )
    except Exception as e:
        raise RuntimeError(f"Execution failed: {e}")

    # Validate execution time
    if execution_result.success and hasattr(execution_result, 'metrics'):
        execution_time = execution_result.metrics.execution_time_ms
        if execution_time > config.OPERATION_TIMEOUT_MS:
            raise RuntimeError(f"Execution time {execution_time}ms exceeds timeout {config.OPERATION_TIMEOUT_MS}ms")

    return execution_result


# Export commonly used functions
__all__ = [
    'DSLTestConfig',
    'DSLTestUtilities',
    'dsl_transpiler',
    'dsl_executor',
    'dsl_test_config',
    'dsl_utilities',
    'execute_dsl_program_with_validation'
]
