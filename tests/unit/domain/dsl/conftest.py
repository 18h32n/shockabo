"""Pytest fixtures for DSL unit tests.

This module provides common test data, grid utilities, and performance testing
decorators for the Domain-Specific Language testing infrastructure.
"""

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


# Sample grids of various sizes
@pytest.fixture
def empty_grid_3x3() -> list[list[int]]:
    """Returns a 3x3 empty grid (all zeros)."""
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


@pytest.fixture
def empty_grid_5x5() -> list[list[int]]:
    """Returns a 5x5 empty grid (all zeros)."""
    return [[0] * 5 for _ in range(5)]


@pytest.fixture
def empty_grid_10x10() -> list[list[int]]:
    """Returns a 10x10 empty grid (all zeros)."""
    return [[0] * 10 for _ in range(10)]


@pytest.fixture
def single_color_grid_3x3() -> list[list[int]]:
    """Returns a 3x3 grid filled with color 1."""
    return [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


@pytest.fixture
def single_color_grid_5x5() -> list[list[int]]:
    """Returns a 5x5 grid filled with color 2."""
    return [[2] * 5 for _ in range(5)]


@pytest.fixture
def checkerboard_grid_4x4() -> list[list[int]]:
    """Returns a 4x4 checkerboard pattern grid."""
    return [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]


@pytest.fixture
def diagonal_pattern_grid_5x5() -> list[list[int]]:
    """Returns a 5x5 grid with diagonal pattern."""
    return [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]


@pytest.fixture
def horizontal_stripe_grid_4x6() -> list[list[int]]:
    """Returns a 4x6 grid with horizontal stripes."""
    return [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ]


@pytest.fixture
def vertical_stripe_grid_6x4() -> list[list[int]]:
    """Returns a 6x4 grid with vertical stripes."""
    return [
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0]
    ]


@pytest.fixture
def border_pattern_grid_5x5() -> list[list[int]]:
    """Returns a 5x5 grid with border pattern."""
    return [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 2, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]


@pytest.fixture
def corner_pattern_grid_3x3() -> list[list[int]]:
    """Returns a 3x3 grid with corners filled."""
    return [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]


@pytest.fixture
def sample_grids() -> dict[str, list[list[int]]]:
    """Returns a dictionary of various sample grids for testing."""
    return {
        "empty_3x3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "single_color_3x3": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "checkerboard_4x4": [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ],
        "diagonal_5x5": [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ],
        "border_5x5": [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 2, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
    }


# Common test cases
@pytest.fixture
def test_cases() -> list[dict[str, Any]]:
    """Returns common test cases for DSL operations."""
    return [
        {
            "name": "empty_to_single_color",
            "input": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "expected": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "description": "Fill empty grid with single color"
        },
        {
            "name": "flip_horizontal",
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "expected": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "description": "Flip grid horizontally"
        },
        {
            "name": "flip_vertical",
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "expected": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "description": "Flip grid vertically"
        },
        {
            "name": "rotate_90",
            "input": [[1, 0], [0, 1]],
            "expected": [[0, 1], [1, 0]],
            "description": "Rotate grid 90 degrees clockwise"
        },
        {
            "name": "color_replacement",
            "input": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
            "expected": [[2, 2, 0], [0, 2, 2], [2, 0, 2]],
            "description": "Replace color 1 with color 2"
        }
    ]


# Grid comparison utilities
@pytest.fixture
def grid_comparator():
    """Returns grid comparison utility functions."""

    def are_grids_equal(grid1: list[list[int]], grid2: list[list[int]]) -> bool:
        """Check if two grids are exactly equal."""
        if len(grid1) != len(grid2):
            return False
        for row1, row2 in zip(grid1, grid2, strict=False):
            if len(row1) != len(row2):
                return False
            if row1 != row2:
                return False
        return True

    def get_grid_differences(grid1: list[list[int]], grid2: list[list[int]]) -> list[tuple[int, int, int, int]]:
        """Get list of differences between two grids as (row, col, val1, val2)."""
        differences = []
        max_rows = max(len(grid1), len(grid2))

        for i in range(max_rows):
            row1 = grid1[i] if i < len(grid1) else []
            row2 = grid2[i] if i < len(grid2) else []
            max_cols = max(len(row1), len(row2))

            for j in range(max_cols):
                val1 = row1[j] if j < len(row1) else None
                val2 = row2[j] if j < len(row2) else None
                if val1 != val2:
                    differences.append((i, j, val1, val2))

        return differences

    def get_grid_similarity(grid1: list[list[int]], grid2: list[list[int]]) -> float:
        """Calculate similarity ratio between two grids (0.0 to 1.0)."""
        if not grid1 and not grid2:
            return 1.0
        if not grid1 or not grid2:
            return 0.0

        total_cells = 0
        matching_cells = 0

        max_rows = max(len(grid1), len(grid2))
        for i in range(max_rows):
            row1 = grid1[i] if i < len(grid1) else []
            row2 = grid2[i] if i < len(grid2) else []
            max_cols = max(len(row1), len(row2))

            for j in range(max_cols):
                total_cells += 1
                val1 = row1[j] if j < len(row1) else 0
                val2 = row2[j] if j < len(row2) else 0
                if val1 == val2:
                    matching_cells += 1

        return matching_cells / total_cells if total_cells > 0 else 0.0

    def assert_grids_equal(grid1: list[list[int]], grid2: list[list[int]], message: str = ""):
        """Assert that two grids are equal with detailed error message."""
        if not are_grids_equal(grid1, grid2):
            differences = get_grid_differences(grid1, grid2)
            error_msg = f"Grids are not equal. {message}\n"
            error_msg += f"Grid 1 shape: {len(grid1)}x{len(grid1[0]) if grid1 else 0}\n"
            error_msg += f"Grid 2 shape: {len(grid2)}x{len(grid2[0]) if grid2 else 0}\n"
            error_msg += f"Differences at positions: {differences[:10]}..."  # Show first 10 differences
            raise AssertionError(error_msg)

    def format_grid_for_display(grid: list[list[int]], title: str = "Grid") -> str:
        """Format grid for readable display in test output."""
        if not grid:
            return f"{title}: Empty grid"

        lines = [f"{title}:"]
        for i, row in enumerate(grid):
            line = f"  Row {i:2}: " + " ".join(f"{val:2}" for val in row)
            lines.append(line)
        return "\n".join(lines)

    return {
        "are_equal": are_grids_equal,
        "get_differences": get_grid_differences,
        "get_similarity": get_grid_similarity,
        "assert_equal": assert_grids_equal,
        "format_display": format_grid_for_display
    }


# Performance timing decorators
@pytest.fixture
def performance_timer():
    """Returns performance timing utilities for testing."""

    def time_function(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Time a function execution and return result and duration."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return result, duration

    def benchmark_function(func: Callable, iterations: int = 100, *args, **kwargs) -> dict[str, float]:
        """Benchmark a function over multiple iterations."""
        times = []
        results = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            results.append(result)

        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "iterations": iterations,
            "consistent_results": len(set(str(r) for r in results)) == 1
        }

    def performance_test(max_time: float = 1.0, iterations: int = 1):
        """Decorator for performance testing."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                total_time = 0
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    total_time += (end_time - start_time)

                avg_time = total_time / iterations
                if avg_time > max_time:
                    raise AssertionError(
                        f"Function {func.__name__} took {avg_time:.4f}s on average "
                        f"(max allowed: {max_time}s, iterations: {iterations})"
                    )

                return result
            return wrapper
        return decorator

    return {
        "time_function": time_function,
        "benchmark_function": benchmark_function,
        "performance_test": performance_test
    }


# Grid generation utilities
@pytest.fixture
def grid_generator():
    """Returns utilities for generating test grids."""

    def generate_random_grid(height: int, width: int, max_color: int = 9, seed: int = None) -> list[list[int]]:
        """Generate a random grid with specified dimensions and color range."""
        if seed is not None:
            np.random.seed(seed)

        return np.random.randint(0, max_color + 1, (height, width)).tolist()

    def generate_pattern_grid(height: int, width: int, pattern_func: Callable[[int, int], int]) -> list[list[int]]:
        """Generate a grid using a pattern function that takes (row, col) and returns color."""
        grid = []
        for i in range(height):
            row = []
            for j in range(width):
                color = pattern_func(i, j)
                row.append(color)
            grid.append(row)
        return grid

    def generate_grid_with_noise(base_grid: list[list[int]], noise_probability: float = 0.1,
                                 max_color: int = 9, seed: int = None) -> list[list[int]]:
        """Add random noise to a base grid."""
        if seed is not None:
            np.random.seed(seed)

        noisy_grid = []
        for row in base_grid:
            noisy_row = []
            for cell in row:
                if np.random.random() < noise_probability:
                    # Add noise
                    noisy_row.append(np.random.randint(0, max_color + 1))
                else:
                    noisy_row.append(cell)
            noisy_grid.append(noisy_row)

        return noisy_grid

    return {
        "random": generate_random_grid,
        "pattern": generate_pattern_grid,
        "with_noise": generate_grid_with_noise
    }


# Test data validation utilities
@pytest.fixture
def grid_validator():
    """Returns grid validation utilities."""

    def is_valid_grid(grid: Any) -> bool:
        """Check if input is a valid grid structure."""
        if not isinstance(grid, list):
            return False
        if not grid:  # Empty grid is valid
            return True

        # Check if all rows are lists of same length
        first_row_length = None
        for row in grid:
            if not isinstance(row, list):
                return False
            if first_row_length is None:
                first_row_length = len(row)
            elif len(row) != first_row_length:
                return False

            # Check if all cells are integers
            for cell in row:
                if not isinstance(cell, int):
                    return False

        return True

    def validate_grid_colors(grid: list[list[int]], min_color: int = 0, max_color: int = 9) -> bool:
        """Validate that all grid colors are within the specified range."""
        for row in grid:
            for cell in row:
                if not (min_color <= cell <= max_color):
                    return False
        return True

    def get_grid_stats(grid: list[list[int]]) -> dict[str, Any]:
        """Get statistics about a grid."""
        if not grid:
            return {"height": 0, "width": 0, "total_cells": 0, "unique_colors": set()}

        height = len(grid)
        width = len(grid[0]) if grid else 0
        total_cells = height * width
        all_colors = set()

        for row in grid:
            for cell in row:
                all_colors.add(cell)

        return {
            "height": height,
            "width": width,
            "total_cells": total_cells,
            "unique_colors": all_colors,
            "color_count": len(all_colors),
            "min_color": min(all_colors) if all_colors else None,
            "max_color": max(all_colors) if all_colors else None
        }

    return {
        "is_valid": is_valid_grid,
        "validate_colors": validate_grid_colors,
        "get_stats": get_grid_stats
    }
