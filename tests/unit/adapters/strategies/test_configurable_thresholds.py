"""
Tests for configurable thresholds in transpiler and sandbox execution.

This module tests that all hardcoded values have been replaced with
configurable thresholds and work correctly with custom configurations.
"""


from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxConfig, SandboxExecutor
from src.domain.dsl.types import is_valid_grid
from src.infrastructure.config import TranspilerSandboxConfig


class TestConfigurableThresholds:
    """Test configurable thresholds in transpiler and sandbox systems."""

    def test_transpiler_sandbox_config_defaults(self):
        """Test that TranspilerSandboxConfig has correct default values."""
        config = TranspilerSandboxConfig()

        assert config.slow_operation_threshold_ms == 50.0
        assert config.max_grid_width == 30
        assert config.max_grid_height == 30
        assert config.timeout_seconds == 1.0
        assert config.memory_limit_mb == 100
        assert config.max_operation_memory_overhead_factor == 0.5

    def test_custom_transpiler_config(self):
        """Test transpiler with custom configuration values."""
        # Create custom config with different thresholds
        custom_config = TranspilerSandboxConfig(
            slow_operation_threshold_ms=25.0,
            max_grid_width=50,
            max_grid_height=50,
            timeout_seconds=2.0,
            memory_limit_mb=200,
            max_operation_memory_overhead_factor=0.3
        )

        transpiler = PythonTranspiler(config=custom_config)

        # Test that the custom config is used
        assert transpiler.config.slow_operation_threshold_ms == 25.0
        assert transpiler.config.max_grid_width == 50
        assert transpiler.config.max_grid_height == 50
        assert transpiler.config.timeout_seconds == 2.0
        assert transpiler.config.memory_limit_mb == 200
        assert transpiler.config.max_operation_memory_overhead_factor == 0.3

    def test_transpiler_input_validation_custom_grid_size(self):
        """Test that input validation uses custom grid size limits."""
        custom_config = TranspilerSandboxConfig(
            max_grid_width=10,
            max_grid_height=10
        )

        transpiler = PythonTranspiler(config=custom_config)

        # Create a simple program
        program = {
            "operations": [
                {"type": "flip", "direction": "horizontal"}
            ]
        }

        result = transpiler.transpile(program)

        # Check that the generated validation code uses custom limits
        assert "len(grid) > 10 or len(grid[0]) > 10" in result.source_code
        assert "Grid dimensions exceed 10x10" in result.source_code

    def test_transpiler_helper_functions_custom_grid_size(self):
        """Test that helper functions use custom grid size limits."""
        custom_config = TranspilerSandboxConfig(
            max_grid_width=15,
            max_grid_height=20
        )

        transpiler = PythonTranspiler(config=custom_config)
        helpers = transpiler._initialize_helper_functions()

        validate_grid_code = helpers["validate_grid"]

        # Check that helper functions use custom limits
        assert "h > 20 or w > 15" in validate_grid_code
        assert "max: 20x15" in validate_grid_code

    def test_transpiler_bounds_check_custom_grid_size(self):
        """Test that bounds checking uses custom grid size limits."""
        custom_config = TranspilerSandboxConfig(
            max_grid_width=25,
            max_grid_height=25
        )

        transpiler = PythonTranspiler(config=custom_config)
        bounds_check = transpiler._generate_bounds_check("test_grid")

        # Check that bounds checking uses custom limits
        assert "test_grid.shape[0] > 25 or test_grid.shape[1] > 25" in bounds_check
        assert "larger than 25x25" in bounds_check

    def test_transpiler_memory_estimation_custom_overhead(self):
        """Test that memory estimation uses custom overhead factor."""
        custom_config = TranspilerSandboxConfig(
            max_operation_memory_overhead_factor=0.8,
            max_grid_width=30,
            max_grid_height=30
        )

        transpiler = PythonTranspiler(config=custom_config)

        program = {
            "operations": [
                {"type": "flip", "direction": "horizontal"},
                {"type": "rotate", "angle": 90}
            ]
        }

        estimated_memory = transpiler._estimate_memory_usage(program)

        # Calculate expected memory with custom overhead
        base_memory = (30 * 30 * 4) / (1024 * 1024)  # 30x30 int32 grid
        expected_overhead = 2 * base_memory * 0.8  # 2 operations with 80% overhead
        python_overhead = 5.0
        expected_total = base_memory + expected_overhead + python_overhead

        assert abs(estimated_memory - expected_total) < 0.01

    def test_sandbox_config_with_transpiler_config(self):
        """Test that SandboxConfig properly integrates TranspilerSandboxConfig."""
        transpiler_config = TranspilerSandboxConfig(
            timeout_seconds=3.0,
            memory_limit_mb=256
        )

        # Test that defaults are overridden by transpiler config
        sandbox_config = SandboxConfig(transpiler_config=transpiler_config)

        assert sandbox_config.timeout_seconds == 3.0
        assert sandbox_config.memory_limit_mb == 256
        assert sandbox_config.transpiler_config == transpiler_config

    def test_sandbox_config_explicit_values_override(self):
        """Test that explicit SandboxConfig values override transpiler config."""
        transpiler_config = TranspilerSandboxConfig(
            timeout_seconds=3.0,
            memory_limit_mb=256
        )

        # Explicit values should override transpiler config
        sandbox_config = SandboxConfig(
            timeout_seconds=5.0,
            memory_limit_mb=512,
            transpiler_config=transpiler_config
        )

        assert sandbox_config.timeout_seconds == 5.0
        assert sandbox_config.memory_limit_mb == 512

    def test_sandbox_executor_uses_config(self):
        """Test that SandboxExecutor properly uses configuration."""
        transpiler_config = TranspilerSandboxConfig(
            timeout_seconds=2.0,
            memory_limit_mb=150,
            slow_operation_threshold_ms=30.0
        )

        sandbox_config = SandboxConfig(transpiler_config=transpiler_config)
        executor = SandboxExecutor(sandbox_config)

        assert executor.config.timeout_seconds == 2.0
        assert executor.config.memory_limit_mb == 150
        assert executor.config.transpiler_config.slow_operation_threshold_ms == 30.0

    def test_is_valid_grid_with_custom_config(self):
        """Test that is_valid_grid uses custom configuration."""
        # Create grids at various sizes
        grid_20x20 = [[1] * 20 for _ in range(20)]
        grid_40x40 = [[1] * 40 for _ in range(40)]

        # Default config (30x30 limit)
        default_config = TranspilerSandboxConfig()
        assert is_valid_grid(grid_20x20, default_config) == True
        assert is_valid_grid(grid_40x40, default_config) == False

        # Custom config with larger limits (50x50)
        large_config = TranspilerSandboxConfig(
            max_grid_width=50,
            max_grid_height=50
        )
        assert is_valid_grid(grid_20x20, large_config) == True
        assert is_valid_grid(grid_40x40, large_config) == True

        # Custom config with smaller limits (15x15)
        small_config = TranspilerSandboxConfig(
            max_grid_width=15,
            max_grid_height=15
        )
        assert is_valid_grid(grid_20x20, small_config) == False

    def test_is_valid_grid_without_config_uses_defaults(self):
        """Test that is_valid_grid uses default config when none provided."""
        grid_20x20 = [[1] * 20 for _ in range(20)]
        grid_40x40 = [[1] * 40 for _ in range(40)]

        # Should use default limits (30x30)
        assert is_valid_grid(grid_20x20) == True
        assert is_valid_grid(grid_40x40) == False

    def test_slow_operation_threshold_configurable(self):
        """Test that slow operation threshold is configurable."""
        # This would need to be tested in an integration context
        # where we can actually time operations and check the slow_operations list

        transpiler_config = TranspilerSandboxConfig(
            slow_operation_threshold_ms=10.0  # Very low threshold
        )

        sandbox_config = SandboxConfig(transpiler_config=transpiler_config)

        # Verify the threshold is set correctly
        assert sandbox_config.transpiler_config.slow_operation_threshold_ms == 10.0

    def test_backward_compatibility(self):
        """Test that existing code without configuration still works."""
        # Test transpiler without config
        transpiler = PythonTranspiler()
        assert transpiler.config.max_grid_width == 30
        assert transpiler.config.max_grid_height == 30
        assert transpiler.config.timeout_seconds == 1.0
        assert transpiler.config.memory_limit_mb == 100

        # Test sandbox executor without config
        executor = SandboxExecutor()
        assert executor.config.timeout_seconds == 1.0
        assert executor.config.memory_limit_mb == 100

        # Test is_valid_grid without config
        grid_25x25 = [[1] * 25 for _ in range(25)]
        grid_35x35 = [[1] * 35 for _ in range(35)]
        assert is_valid_grid(grid_25x25) == True
        assert is_valid_grid(grid_35x35) == False

    def test_config_error_messages_use_configured_values(self):
        """Test that error messages reflect configured values."""
        custom_config = TranspilerSandboxConfig(
            max_grid_width=12,
            max_grid_height=8
        )

        transpiler = PythonTranspiler(config=custom_config)

        # Test input validation error message
        validation_code = transpiler._generate_input_validation()
        assert "Grid dimensions exceed 8x12" in validation_code

        # Test helper function error message
        helpers = transpiler._initialize_helper_functions()
        validate_grid_code = helpers["validate_grid"]
        assert "max: 8x12" in validate_grid_code

        # Test bounds check error message
        bounds_check = transpiler._generate_bounds_check("test_var")
        assert "larger than 8x12" in bounds_check
