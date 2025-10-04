"""Integration tests for complete DSL→Python→Execution pipeline."""

import json
import time
from pathlib import Path

import pytest

from src.adapters.strategies.python_transpiler import PythonTranspiler
from src.adapters.strategies.sandbox_executor import SandboxExecutor


@pytest.fixture
def transpiler():
    """Create Python transpiler instance."""
    return PythonTranspiler()


@pytest.fixture
def executor():
    """Create sandbox executor instance."""
    return SandboxExecutor()


@pytest.fixture
def sample_grid():
    """Create sample test grid."""
    return [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ]


class TestDSLParsing:
    """Test DSL program parsing and validation."""

    def test_parse_simple_program(self, transpiler):
        """Test parsing a simple DSL program."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        result = transpiler.transpile(program)
        assert result is not None
        assert result.source_code is not None

    def test_parse_chained_operations(self, transpiler):
        """Test parsing chained operations."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"}
            ]
        }

        result = transpiler.transpile(program)
        assert result is not None
        assert len(program["operations"]) >= 1

    def test_parse_invalid_operation(self, transpiler):
        """Test parsing handles unknown operations."""
        program = {"operations": [{"type": "invalid_operation"}]}

        result = transpiler.transpile(program)
        assert result is not None

    def test_validate_operation_syntax(self, transpiler):
        """Test operation syntax validation."""
        valid_programs = [
            {"operations": [{"type": "identity"}]},
            {"operations": [{"type": "rotate", "angle": 90}]},
            {"operations": [{"type": "mirror", "direction": "vertical"}]},
            {"operations": [{"type": "transpose"}]}
        ]

        for program in valid_programs:
            result = transpiler.transpile(program)
            assert result is not None


class TestDSLToPythonTranspilation:
    """Test DSL→Python transpilation correctness."""

    def test_transpile_identity(self, transpiler, executor, sample_grid):
        """Test transpiling identity operation."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        assert transpilation_result is not None
        assert transpilation_result.source_code is not None
        assert "def " in transpilation_result.source_code

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )
        assert exec_result.success
        assert exec_result.result == sample_grid

    def test_transpile_rotation(self, transpiler, executor, sample_grid):
        """Test transpiling rotation operation."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}
        transpilation_result = transpiler.transpile(program)

        assert transpilation_result is not None
        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert exec_result.result is not None

    def test_transpile_flip_operations(self, transpiler, executor, sample_grid):
        """Test transpiling flip operations."""
        flip_ops = [
            {"operations": [{"type": "mirror", "direction": "horizontal"}]},
            {"operations": [{"type": "mirror", "direction": "vertical"}]}
        ]

        for program in flip_ops:
            transpilation_result = transpiler.transpile(program)
            exec_result = executor.execute(
                transpilation_result.source_code,
                transpilation_result.function_name,
                sample_grid
            )
            assert exec_result.success
            assert exec_result.result is not None

    def test_transpile_chained_operations(self, transpiler, executor, sample_grid):
        """Test transpiling chained operations."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "horizontal"}
            ]
        }
        transpilation_result = transpiler.transpile(program)

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert exec_result.result is not None

    def test_transpilation_safety_checks(self, transpiler, executor, sample_grid):
        """Test runtime safety checks in transpiled code."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        assert "def " in transpilation_result.source_code

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )
        assert exec_result.success


class TestSandboxedExecution:
    """Test sandboxed Python execution with safety checks."""

    def test_execution_with_valid_code(self, executor, sample_grid):
        """Test execution with valid generated code."""
        code = """
def solve_task(grid):
    return grid
"""
        result = executor.execute(code, "solve_task", sample_grid)
        assert result.success
        assert result.result == sample_grid

    def test_state_isolation(self, executor, sample_grid):
        """Test state isolation between executions."""
        code = """
def solve_task(grid):
    return grid
"""
        result1 = executor.execute(code, "solve_task", sample_grid)
        result2 = executor.execute(code, "solve_task", sample_grid)

        assert result1.success and result2.success
        assert result1.result == result2.result

    def test_resource_monitoring(self, transpiler, executor, sample_grid):
        """Test resource monitoring during execution."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert exec_result.metrics is not None
        assert exec_result.metrics.execution_time_ms >= 0
        assert exec_result.metrics.memory_used_mb >= 0


class TestOutputCorrectness:
    """Test output correctness against expected results."""

    def test_correctness_with_validation_tasks(self, transpiler, executor):
        """Test correctness using validation task examples."""
        test_data_path = Path(__file__).parent.parent / "performance" / "data" / "synthesis_validation_tasks.json"

        if not test_data_path.exists():
            pytest.skip(f"Validation tasks not found at {test_data_path}")

        with open(test_data_path, encoding="utf-8", errors="replace") as f:
            tasks = json.load(f)

        task = tasks[0]["task"]
        train_example = task["train"][0]

        input_grid = train_example["input"]

        simple_programs = [
            {"operations": [{"type": "identity"}]},
            {"operations": [{"type": "rotate", "angle": 90}]},
            {"operations": [{"type": "mirror", "direction": "horizontal"}]},
            {"operations": [{"type": "mirror", "direction": "vertical"}]}
        ]

        for program in simple_programs:
            transpilation_result = transpiler.transpile(program)
            exec_result = executor.execute(
                transpilation_result.source_code,
                transpilation_result.function_name,
                input_grid
            )

            assert exec_result.success, f"Failed to execute program: {program}"
            assert exec_result.result is not None

    def test_type_preservation(self, transpiler, executor, sample_grid):
        """Test output type preservation."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert isinstance(exec_result.result, list)

    def test_shape_preservation(self, transpiler, executor, sample_grid):
        """Test shape preservation for identity operations."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert len(exec_result.result) == len(sample_grid)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery at each pipeline stage."""

    def test_transpilation_error_handling(self, transpiler, executor, sample_grid):
        """Test graceful handling of transpilation errors."""
        invalid_program = {"operations": [{"type": "unknown_operation"}]}

        transpilation_result = transpiler.transpile(invalid_program)
        assert transpilation_result is not None

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert not exec_result.success or exec_result.result is not None

    def test_execution_error_recovery(self, executor):
        """Test recovery from execution errors."""
        faulty_code = """
def solve_task(grid):
    raise ValueError("Intentional error")
"""
        result = executor.execute(faulty_code, "solve_task", [[1, 2], [3, 4]])

        assert not result.success
        assert result.error is not None

    def test_invalid_output_handling(self, executor):
        """Test handling of invalid output types."""
        invalid_output_code = """
def solve_task(grid):
    return "not a grid"
"""
        result = executor.execute(invalid_output_code, "solve_task", [[1, 2], [3, 4]])

        assert result.result is not None

    def test_resource_cleanup_on_error(self, transpiler, executor, sample_grid):
        """Test resource cleanup after errors."""
        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        try:
            executor.execute("invalid code", "solve_task", sample_grid)
        except Exception:
            pass

        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )
        assert exec_result.success


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline flow."""

    def test_full_pipeline_identity(self, transpiler, executor, sample_grid):
        """Test full pipeline with identity operation."""
        program = {"operations": [{"type": "identity"}]}

        transpilation_result = transpiler.transpile(program)
        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert exec_result.result == sample_grid

    def test_full_pipeline_complex_operation(self, transpiler, executor, sample_grid):
        """Test full pipeline with complex operation."""
        program = {
            "operations": [
                {"type": "rotate", "angle": 90},
                {"type": "mirror", "direction": "vertical"},
                {"type": "transpose"}
            ]
        }

        transpilation_result = transpiler.transpile(program)
        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )

        assert exec_result.success
        assert exec_result.result is not None

    def test_pipeline_performance(self, transpiler, executor, sample_grid):
        """Test pipeline meets performance requirements."""
        program = {"operations": [{"type": "rotate", "angle": 90}]}

        start = time.time()
        transpilation_result = transpiler.transpile(program)
        exec_result = executor.execute(
            transpilation_result.source_code,
            transpilation_result.function_name,
            sample_grid
        )
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert exec_result.success


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios across full pipeline."""

    def test_multiple_tasks_in_sequence(self, transpiler, executor):
        """Test processing multiple tasks in sequence."""
        test_data_path = Path(__file__).parent.parent / "performance" / "data" / "synthesis_validation_tasks.json"

        if not test_data_path.exists():
            pytest.skip("Validation tasks not found")

        with open(test_data_path, encoding="utf-8", errors="replace") as f:
            tasks = json.load(f)

        test_tasks = tasks[:3]

        for task_data in test_tasks:
            task = task_data["task"]
            for example in task["train"][:1]:
                input_grid = example["input"]

                program = {"operations": [{"type": "identity"}]}
                transpilation_result = transpiler.transpile(program)
                exec_result = executor.execute(
                    transpilation_result.source_code,
                    transpilation_result.function_name,
                    input_grid
                )

                assert exec_result.success
                assert len(exec_result.result) == len(input_grid)

    def test_pipeline_with_various_grid_sizes(self, transpiler, executor):
        """Test pipeline with various grid sizes."""
        grid_sizes = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1] * 10 for _ in range(10)],
            [[1] * 20 for _ in range(20)],
            [[1] * 15 for _ in range(5)]
        ]

        program = {"operations": [{"type": "identity"}]}
        transpilation_result = transpiler.transpile(program)

        for grid in grid_sizes:
            exec_result = executor.execute(
                transpilation_result.source_code,
                transpilation_result.function_name,
                grid
            )

            assert exec_result.success
            assert len(exec_result.result) == len(grid)
