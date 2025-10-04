"""
Unit tests for the Python transpiler module.

Tests transpilation of DSL programs to executable Python code,
including all operation types, error handling, and edge cases.
"""


import pytest

from src.adapters.strategies.python_transpiler import (
    PythonTranspiler,
    TranspilationError,
    TranspilationResult,
)


class TestPythonTranspiler:
    """Test cases for Python transpiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()

    def test_basic_transpilation(self):
        """Test basic transpilation of a simple program."""
        program = {
            "version": "1.0",
            "operations": [
                {
                    "type": "rotate",
                    "angle": 90
                }
            ]
        }

        result = self.transpiler.transpile(program)

        assert isinstance(result, TranspilationResult)
        assert result.function_name == "solve_task"
        assert "import numpy as np" in result.source_code
        assert "def solve_task" in result.source_code
        assert "np.rot90" in result.source_code

    def test_multiple_operations(self):
        """Test transpilation of multiple operations."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"},
                {"type": "translate", "dx": 2, "dy": 1}
            ]
        }

        result = self.transpiler.transpile(program)

        assert "Operation 0: rotate" in result.source_code
        assert "Operation 1: mirror" in result.source_code
        assert "Operation 2: translate" in result.source_code

    def test_custom_function_name(self):
        """Test transpilation with custom function name."""
        program = {"operations": [{"type": "rotate", "angle": 180}]}

        result = self.transpiler.transpile(program, function_name="custom_solver")

        assert result.function_name == "custom_solver"
        assert "def custom_solver" in result.source_code

    def test_empty_program(self):
        """Test error handling for empty program."""
        program = {"operations": []}

        result = self.transpiler.transpile(program)

        # Should still generate valid code that returns the input unchanged
        assert "def solve_task" in result.source_code
        assert "return grid.tolist()" in result.source_code

    def test_missing_operations_field(self):
        """Test error handling for missing operations field."""
        program = {"version": "1.0"}

        with pytest.raises(TranspilationError) as exc_info:
            self.transpiler.transpile(program)

        assert "missing 'operations' field" in str(exc_info.value).lower()

    def test_invalid_operation_format(self):
        """Test error handling for invalid operation format."""
        program = {
            "operations": [
                "invalid_operation"  # Should be dict, not string
            ]
        }

        with pytest.raises(TranspilationError) as exc_info:
            self.transpiler.transpile(program)

        assert "invalid operation format" in str(exc_info.value).lower()

    def test_program_hash_consistency(self):
        """Test that program hash is consistent for same program."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "vertical"}
            ]
        }

        result1 = self.transpiler.transpile(program)
        result2 = self.transpiler.transpile(program)

        assert result1.program_hash == result2.program_hash

    def test_program_hash_uniqueness(self):
        """Test that different programs have different hashes."""
        program1 = {"operations": [{"type": "rotate", "angle": 90}]}
        program2 = {"operations": [{"type": "rotate", "angle": 180}]}

        result1 = self.transpiler.transpile(program1)
        result2 = self.transpiler.transpile(program2)

        assert result1.program_hash != result2.program_hash

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        simple_program = {"operations": [{"type": "rotate", "angle": 90}]}
        complex_program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"},
                {"type": "translate", "dx": 1, "dy": 1},
                {"type": "crop", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"type": "pad", "top": 2, "bottom": 2, "left": 2, "right": 2, "value": 0}
            ]
        }

        simple_result = self.transpiler.transpile(simple_program)
        complex_result = self.transpiler.transpile(complex_program)

        assert simple_result.estimated_memory_mb > 0
        assert complex_result.estimated_memory_mb > simple_result.estimated_memory_mb

    def test_input_validation_code(self):
        """Test that input validation is included."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        result = self.transpiler.transpile(program)

        assert "if not grid or not grid[0]:" in result.source_code
        assert "if len(grid) > 30 or len(grid[0]) > 30:" in result.source_code
        assert "ValueError" in result.source_code

    def test_operation_timing_code(self):
        """Test that operation timing is included."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        result = self.transpiler.transpile(program)

        assert "_operation_timings = {}" in result.source_code
        assert "_op_start = time.time()" in result.source_code
        assert "_operation_timings[" in result.source_code

    def test_numpy_conversion(self):
        """Test numpy array conversion."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        result = self.transpiler.transpile(program)

        assert "grid = np.array(grid, dtype=np.int32)" in result.source_code
        assert "return grid.tolist()" in result.source_code

    def test_execution_error_class(self):
        """Test that ExecutionError class is included."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        result = self.transpiler.transpile(program)

        assert "class ExecutionError(Exception):" in result.source_code
        assert "self.operation_context = operation_context" in result.source_code


class TestOperationTemplates:
    """Test transpilation of specific operation types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()

    def test_geometric_operations(self):
        """Test geometric operation transpilation."""
        operations = [
            ({"type": "rotate", "angle": 90}, "np.rot90"),
            ({"type": "rotate", "angle": 180}, "np.rot90"),
            ({"type": "rotate", "angle": 270}, "np.rot90"),
            ({"type": "mirror", "direction": "horizontal"}, "[::-1, :]"),
            ({"type": "mirror", "direction": "vertical"}, "[:, ::-1]"),
            ({"type": "translate", "dx": 2, "dy": 1}, "np.roll")
        ]

        for op, expected_code in operations:
            program = {"operations": [op]}
            result = self.transpiler.transpile(program)
            assert expected_code in result.source_code

    def test_color_operations(self):
        """Test color operation transpilation."""
        operations = [
            ({"type": "filter", "colors": [1, 2, 3]}, "np.isin"),
            ({"type": "replace", "old_color": 1, "new_color": 2}, "np.where")
        ]

        for op, expected_code in operations:
            program = {"operations": [op]}
            result = self.transpiler.transpile(program)
            assert expected_code in result.source_code

    def test_unknown_operation(self):
        """Test handling of unknown operations."""
        program = {
            "operations": [
                {"type": "unknown_operation", "param": "value"}
            ]
        }

        result = self.transpiler.transpile(program)

        # Should include warning comment
        assert "WARNING: No template found for unknown_operation" in result.source_code
        assert "Using fallback implementation" in result.source_code


class TestHelperFunctions:
    """Test helper function generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()

    def test_validate_grid_helper(self):
        """Test validate_grid helper function."""
        helpers = self.transpiler._initialize_helper_functions()

        assert "validate_grid" in helpers
        validate_code = helpers["validate_grid"]

        assert "def validate_grid(grid):" in validate_code
        assert "if not grid or not grid[0]:" in validate_code
        assert "if h > 30 or w > 30:" in validate_code
        assert "val < 0 or val > 9" in validate_code

    def test_safe_bounds_helper(self):
        """Test safe_bounds helper function."""
        helpers = self.transpiler._initialize_helper_functions()

        assert "safe_bounds" in helpers
        bounds_code = helpers["safe_bounds"]

        assert "def safe_bounds(y, x, h, w):" in bounds_code
        assert "0 <= y < h and 0 <= x < w" in bounds_code


class TestTemplateApplication:
    """Test template application logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transpiler = PythonTranspiler()

    def test_simple_string_template(self):
        """Test applying simple string template."""
        template = "np.roll({input}, shift=({dy}, {dx}), axis=(0, 1))"
        params = {"dx": 2, "dy": 3}

        result = self.transpiler._apply_template(template, params)

        assert result == "np.roll(grid, shift=(3, 2), axis=(0, 1))"

    def test_dict_template(self):
        """Test applying dictionary template."""
        template = {
            "horizontal": "{input}[::-1, :]",
            "vertical": "{input}[:, ::-1]"
        }

        params_h = {"horizontal": "horizontal"}
        result_h = self.transpiler._apply_template(template, params_h)
        assert result_h == "grid[::-1, :]"

        params_v = {"vertical": "vertical"}
        result_v = self.transpiler._apply_template(template, params_v)
        assert result_v == "grid[:, ::-1]"

    def test_template_with_missing_params(self):
        """Test template application with missing parameters."""
        template = "crop({input}, {x1}, {y1}, {x2}, {y2})"
        params = {"x1": 0, "y1": 0}  # Missing x2, y2

        # Should not raise error, but may produce incomplete code
        result = self.transpiler._apply_template(template, params)
        assert "crop(grid, 0, 0," in result


class TestLRUCache:
    """Test LRU cache implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Small cache size for testing
        self.transpiler = PythonTranspiler(enable_cache=True, cache_size=3)

    def test_lru_cache_eviction(self):
        """Test that cache implements proper LRU eviction."""
        # Test programs with different operations
        programs = [
            {"operations": [{"type": "rotate", "angle": 90}]},
            {"operations": [{"type": "flip", "axis": "horizontal"}]},
            {"operations": [{"type": "translate", "dx": 1, "dy": 1}]},
            {"operations": [{"type": "pattern_fill", "pattern": [[1, 0]]}]}
        ]

        # Add first 3 programs to fill cache
        hashes = []
        for i, program in enumerate(programs[:3]):
            result = self.transpiler.transpile(program, f"test_func_{i}")
            program_hash = self.transpiler._compute_program_hash(program)
            hashes.append(program_hash)
            assert program_hash in self.transpiler._cache

        # Verify cache is full
        assert len(self.transpiler._cache) == 3

        # Access first program to make it recently used
        first_result = self.transpiler.transpile(programs[0], "test_func_recent")
        assert self.transpiler.get_cache_stats()['cache_hits'] == 1

        # Add 4th program - should evict second program (least recently used)
        fourth_result = self.transpiler.transpile(programs[3], "test_func_3")

        # Cache should still be size 3
        assert len(self.transpiler._cache) == 3

        # First program should still be cached (was accessed recently)
        assert hashes[0] in self.transpiler._cache

        # Second program should be evicted (least recently used)
        assert hashes[1] not in self.transpiler._cache

        # Third program should still be cached
        assert hashes[2] in self.transpiler._cache

        # Fourth program should be cached
        fourth_hash = self.transpiler._compute_program_hash(programs[3])
        assert fourth_hash in self.transpiler._cache

    def test_cache_hit_updates_lru_order(self):
        """Test that cache hits update the LRU order."""
        programs = [
            {"operations": [{"type": "rotate", "angle": 90}]},
            {"operations": [{"type": "flip", "axis": "horizontal"}]},
            {"operations": [{"type": "translate", "dx": 1, "dy": 1}]}
        ]

        # Fill cache with 3 programs
        for i, program in enumerate(programs):
            self.transpiler.transpile(program, f"test_func_{i}")

        # Access first program (should move to end of LRU order)
        self.transpiler.transpile(programs[0], "test_func_accessed")

        # Add new program - should evict second program (now least recently used)
        new_program = {"operations": [{"type": "pattern_fill", "pattern": [[1]]}]}
        self.transpiler.transpile(new_program, "test_func_new")

        # First program should still be cached
        first_hash = self.transpiler._compute_program_hash(programs[0])
        assert first_hash in self.transpiler._cache

        # Second program should be evicted
        second_hash = self.transpiler._compute_program_hash(programs[1])
        assert second_hash not in self.transpiler._cache

    def test_cache_size_limit_respected(self):
        """Test that cache size limit is always respected."""
        programs = [
            {"operations": [{"type": "rotate", "angle": i}]}
            for i in range(10)  # 10 different programs
        ]

        for program in programs:
            self.transpiler.transpile(program, "test_func")

        # Cache should never exceed configured size
        assert len(self.transpiler._cache) == self.transpiler.cache_size
        assert len(self.transpiler._cache) == 3
