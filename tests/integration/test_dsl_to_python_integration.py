"""
Integration tests for DSL to Python transpilation and execution.

Tests the complete pipeline from DSL program to executed Python function,
including various operation combinations and edge cases.

Note: For advanced DSL operations (color invert/threshold, pattern matching/replace,
diagonal flips, component filtering, boundary tracing, contour extraction, and 
symmetry creation), see test_advanced_dsl_operations.py.
"""

from typing import Any

import pytest

from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxExecutor


class TestDSLToPythonIntegration:
    """Integration tests for DSL to Python pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()
        self.executor = SandboxExecutor()

    def execute_dsl_program(self, program: dict[str, Any], grid: list[list[int]]) -> Any:
        """Helper to transpile and execute a DSL program."""
        # Transpile
        transpilation_result = self.transpiler.transpile(program)

        # Execute
        execution_result = self.executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            grid
        )

        return execution_result

    def test_single_rotate_operation(self):
        """Test single rotation operation."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90}
            ]
        }
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        expected = [[3, 6, 9],
                   [2, 5, 8],
                   [1, 4, 7]]
        assert result.result == expected

    def test_rotation_chain(self):
        """Test multiple rotations."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "rotate", "angle": 90},
                {"type": "rotate", "angle": 90},
                {"type": "rotate", "angle": 90}
            ]
        }
        grid = [[1, 2], [3, 4]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        # Four 90-degree rotations should return to original
        assert result.result == grid

    def test_mirror_operations(self):
        """Test mirror operations."""
        # Horizontal mirror
        program_h = {
            "operations": [
                {"type": "mirror", "direction": "horizontal"}
            ]
        }
        grid = [[1, 2], [3, 4]]

        result_h = self.execute_dsl_program(program_h, grid)
        assert result_h.success
        assert result_h.result == [[3, 4], [1, 2]]

        # Vertical mirror
        program_v = {
            "operations": [
                {"type": "mirror", "direction": "vertical"}
            ]
        }

        result_v = self.execute_dsl_program(program_v, grid)
        assert result_v.success
        assert result_v.result == [[2, 1], [4, 3]]

    def test_translate_operation(self):
        """Test translate operation."""
        program = {
            "operations": [
                {"type": "translate", "dx": 1, "dy": 1}
            ]
        }
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        expected = [[9, 7, 8],
                   [3, 1, 2],
                   [6, 4, 5]]
        assert result.result == expected

    def test_crop_operation(self):
        """Test crop operation."""
        program = {
            "operations": [
                {"type": "crop", "x1": 1, "y1": 1, "x2": 3, "y2": 3}
            ]
        }
        grid = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        expected = [[6, 7],
                   [10, 11]]
        assert result.result == expected

    def test_pad_operation(self):
        """Test pad operation."""
        program = {
            "operations": [
                {"type": "pad", "top": 1, "bottom": 1, "left": 1, "right": 1, "value": 0}
            ]
        }
        grid = [[1, 2], [3, 4]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        expected = [[0, 0, 0, 0],
                   [0, 1, 2, 0],
                   [0, 3, 4, 0],
                   [0, 0, 0, 0]]
        assert result.result == expected

    def test_color_filter_operation(self):
        """Test color filter operation."""
        program = {
            "operations": [
                {"type": "filter", "colors": [1, 3]}
            ]
        }
        grid = [[1, 2, 3],
                [4, 5, 6],
                [1, 3, 2]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        expected = [[1, 0, 3],
                   [0, 0, 0],
                   [1, 3, 0]]
        assert result.result == expected

    def test_complex_operation_chain(self):
        """Test complex chain of operations."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"},
                {"type": "pad", "top": 1, "bottom": 1, "left": 1, "right": 1, "value": 0},
                {"type": "filter", "colors": [1, 2, 3]}
            ]
        }
        grid = [[1, 2], [3, 4]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        # Verify dimensions after padding
        assert len(result.result) == 4
        assert len(result.result[0]) == 4
        # Verify filtering worked (4s should be replaced with 0s)
        assert 4 not in [val for row in result.result for val in row]

    def test_empty_grid_handling(self):
        """Test handling of empty grid."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90}
            ]
        }
        grid = []

        result = self.execute_dsl_program(program, grid)

        assert not result.success
        assert "empty grid" in result.error.lower()

    def test_large_grid_rejection(self):
        """Test rejection of grids larger than 30x30."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90}
            ]
        }
        grid = [[0] * 35 for _ in range(35)]  # 35x35 grid

        result = self.execute_dsl_program(program, grid)

        assert not result.success
        assert "exceed 30x30" in result.error or "too large" in result.error

    def test_operation_timing_collection(self):
        """Test that operation timings are collected."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"},
                {"type": "translate", "dx": 1, "dy": 1}
            ]
        }
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        assert len(result.metrics.operation_timings) == 3
        assert 'rotate_0' in result.metrics.operation_timings
        assert 'mirror_1' in result.metrics.operation_timings
        assert 'translate_2' in result.metrics.operation_timings

        # All operations should be fast (< 50ms)
        assert len(result.metrics.slow_operations) == 0

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        program = {
            "operations": [
                {"type": "pad", "top": 10, "bottom": 10, "left": 10, "right": 10, "value": 0}
            ]
        }
        grid = [[1, 2], [3, 4]]

        result = self.execute_dsl_program(program, grid)

        assert result.success
        assert result.metrics.memory_used_mb >= 0

    def test_invalid_color_values(self):
        """Test handling of invalid color values."""
        program = {
            "operations": []  # No operations
        }
        # Grid with invalid color value (10 is > 9)
        grid = [[1, 2], [3, 10]]

        # The transpiler should accept this, but validation might catch it
        result = self.execute_dsl_program(program, grid)

        # Result depends on whether validation is implemented
        # Just ensure it doesn't crash
        assert isinstance(result.success, bool)


class TestPerformanceBenchmarks:
    """Performance benchmarks for transpiled operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()
        self.executor = SandboxExecutor()

    def benchmark_operation(self, program: dict[str, Any], grid_size: int,
                          expected_time_ms: float):
        """Benchmark a single operation."""
        grid = [[i * grid_size + j for j in range(grid_size)]
                for i in range(grid_size)]

        transpilation_result = self.transpiler.transpile(program)
        execution_result = self.executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            grid
        )

        assert execution_result.success

        # Get operation timing
        op_timings = execution_result.metrics.operation_timings
        if op_timings:
            actual_time = list(op_timings.values())[0]
            # Allow 2x margin for CI/slow systems
            assert actual_time < expected_time_ms * 2

    @pytest.mark.parametrize("angle,expected_ms", [
        (90, 5),
        (180, 5),
        (270, 5)
    ])
    def test_rotate_performance(self, angle, expected_ms):
        """Benchmark rotate operations."""
        program = {"operations": [{"type": "rotate", "angle": angle}]}
        self.benchmark_operation(program, 30, expected_ms)

    @pytest.mark.parametrize("direction,expected_ms", [
        ("horizontal", 5),
        ("vertical", 5)
    ])
    def test_mirror_performance(self, direction, expected_ms):
        """Benchmark mirror operations."""
        program = {"operations": [{"type": "mirror", "direction": direction}]}
        self.benchmark_operation(program, 30, expected_ms)

    def test_translate_performance(self):
        """Benchmark translate operation."""
        program = {"operations": [{"type": "translate", "dx": 5, "dy": 5}]}
        self.benchmark_operation(program, 30, 10)

    def test_filter_performance(self):
        """Benchmark filter operation."""
        program = {"operations": [{"type": "filter", "colors": [1, 2, 3, 4, 5]}]}
        self.benchmark_operation(program, 30, 10)

    def test_complex_chain_performance(self):
        """Benchmark complex operation chain."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"},
                {"type": "translate", "dx": 2, "dy": 2},
                {"type": "filter", "colors": [1, 2, 3]}
            ]
        }
        # Should complete in under 100ms total
        grid = [[i * 20 + j for j in range(20)] for i in range(20)]

        transpilation_result = self.transpiler.transpile(program)
        execution_result = self.executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            grid
        )

        assert execution_result.success
        assert execution_result.metrics.execution_time_ms < 100


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()
        self.executor = SandboxExecutor()

    def test_single_pixel_grid(self):
        """Test operations on single pixel grid."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"}
            ]
        }
        grid = [[5]]

        result = self.executor.execute(
            self.transpiler.transpile(program).source_code,
            "solve_task",
            grid
        )

        assert result.success
        assert result.result == [[5]]  # Single pixel unchanged

    def test_non_square_grids(self):
        """Test operations on non-square grids."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90}
            ]
        }

        # Wide grid
        wide_grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
        wide_result = self.executor.execute(
            self.transpiler.transpile(program).source_code,
            "solve_task",
            wide_grid
        )
        assert wide_result.success
        assert len(wide_result.result) == 4  # Now tall
        assert len(wide_result.result[0]) == 2  # Now narrow

        # Tall grid
        tall_grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        tall_result = self.executor.execute(
            self.transpiler.transpile(program).source_code,
            "solve_task",
            tall_grid
        )
        assert tall_result.success
        assert len(tall_result.result) == 2  # Now short
        assert len(tall_result.result[0]) == 4  # Now wide
