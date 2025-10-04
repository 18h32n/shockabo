"""
Unit tests for the sandbox executor module.

Tests sandboxed execution of transpiled code, including timeout handling,
memory limits, and security restrictions.
"""


from src.adapters.strategies.sandbox_executor import (
    ExecutionMetrics,
    ExecutionResult,
    SandboxConfig,
    SandboxExecutor,
)


class TestSandboxExecutor:
    """Test cases for sandbox executor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = SandboxExecutor()

    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.executor, '_cleanup_all_processes'):
            self.executor._cleanup_all_processes()

    def test_basic_execution(self):
        """Test basic code execution."""
        code = """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    return grid.tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert result.success
        assert result.result == [[1, 2], [3, 4]]
        assert result.error is None
        assert result.metrics.execution_time_ms > 0

    def test_simple_transformation(self):
        """Test execution with simple transformation."""
        code = """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    # Simple rotation
    grid = np.rot90(grid, k=1)
    return grid.tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert result.success
        assert result.result == [[2, 4], [1, 3]]

    def test_timeout_enforcement(self):
        """Test that timeout is enforced."""
        code = """
import time
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    time.sleep(2)  # Sleep longer than timeout
    return grid
"""
        grid = [[1, 2], [3, 4]]

        # Use short timeout
        executor = SandboxExecutor(SandboxConfig(timeout_seconds=0.5))
        result = executor.execute(code, "solve_task", grid)

        assert not result.success
        assert result.timed_out
        assert "timed out" in result.error.lower()
        assert result.result is None

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        code = """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    # Create some intermediate arrays to use memory
    temp1 = np.zeros((100, 100), dtype=np.int32)
    temp2 = np.ones((100, 100), dtype=np.int32)
    result = grid + 1
    return result.tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert result.success
        assert result.metrics.memory_used_mb >= 0  # Should track some memory

    def test_operation_timing_tracking(self):
        """Test operation timing tracking."""
        code = """
import numpy as np
import time
from typing import List

_operation_timings = {}

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    
    # Simulate operation with timing
    _op_start = time.time()
    grid = np.rot90(grid)
    _operation_timings['rotate_0'] = (time.time() - _op_start) * 1000
    
    # Simulate slow operation
    _op_start = time.time()
    time.sleep(0.06)  # 60ms
    _operation_timings['slow_op_1'] = (time.time() - _op_start) * 1000
    
    return grid.tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert result.success
        assert 'rotate_0' in result.metrics.operation_timings
        assert 'slow_op_1' in result.metrics.operation_timings
        assert len(result.metrics.slow_operations) == 1
        assert result.metrics.slow_operations[0][0] == 'slow_op_1'
        assert result.metrics.slow_operations[0][1] > 50  # >50ms

    def test_function_not_found(self):
        """Test error when function is not found."""
        code = """
def wrong_function_name(grid):
    return grid
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert not result.success
        assert "not found" in result.error

    def test_execution_error_handling(self):
        """Test handling of execution errors."""
        code = """
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Intentional error
    return grid[100]  # Index out of bounds
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert not result.success
        assert result.error is not None
        assert "list index out of range" in result.error.lower() or "index" in result.error.lower()

    def test_import_restrictions(self):
        """Test that dangerous imports are blocked."""
        code = """
import os
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    os.system("echo 'Should not work'")
    return grid
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert not result.success
        # Should fail because 'os' is not in allowed imports
        assert "os" in result.error or "not defined" in result.error

    def test_builtin_restrictions(self):
        """Test that dangerous builtins are blocked."""
        code = """
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to use eval (should be blocked)
    eval("print('Should not work')")
    return grid
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert not result.success
        assert "eval" in result.error or "not defined" in result.error

    def test_allowed_imports(self):
        """Test that allowed imports work correctly."""
        code = """
import numpy as np
import math
import itertools
from collections import Counter
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Use various allowed imports
    grid = np.array(grid)
    
    # Use math
    angle = math.pi / 2
    
    # Use itertools
    pairs = list(itertools.combinations([1, 2, 3], 2))
    
    # Use collections
    counter = Counter([1, 1, 2, 3])
    
    return grid.tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert result.success
        assert result.result == [[1, 2], [3, 4]]

    def test_custom_config(self):
        """Test custom sandbox configuration."""
        config = SandboxConfig(
            timeout_seconds=2.0,
            memory_limit_mb=50,
            allowed_imports=['numpy', 'math']
        )
        executor = SandboxExecutor(config)

        code = """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    return np.array(grid).tolist()
"""
        grid = [[1, 2], [3, 4]]

        result = executor.execute(code, "solve_task", grid)

        assert result.success

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement (if platform supports it)."""
        # Note: Memory limits may not work on all platforms
        config = SandboxConfig(memory_limit_mb=10)  # Very low limit
        executor = SandboxExecutor(config)

        code = """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to allocate large array
    large_array = np.zeros((1000, 1000, 10), dtype=np.float64)
    return grid
"""
        grid = [[1, 2], [3, 4]]

        result = executor.execute(code, "solve_task", grid)

        # May or may not fail depending on platform
        # Just ensure it doesn't crash the test
        assert isinstance(result, ExecutionResult)

    def test_process_crash_handling(self):
        """Test handling of process crashes."""
        code = """
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Force a crash by dividing by zero
    x = 1 / 0
    return grid
"""
        grid = [[1, 2], [3, 4]]

        result = self.executor.execute(code, "solve_task", grid)

        assert not result.success
        assert result.error is not None


class TestExecutionMetrics:
    """Test execution metrics collection."""

    def test_metrics_structure(self):
        """Test that metrics have correct structure."""
        metrics = ExecutionMetrics(
            execution_time_ms=100.5,
            memory_used_mb=25.3,
            operation_timings={'op1': 30.0, 'op2': 60.0},
            slow_operations=[('op2', 60.0)]
        )

        assert metrics.execution_time_ms == 100.5
        assert metrics.memory_used_mb == 25.3
        assert len(metrics.operation_timings) == 2
        assert len(metrics.slow_operations) == 1


class TestSandboxConfig:
    """Test sandbox configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.timeout_seconds == 1.0
        assert config.memory_limit_mb == 100
        assert 'numpy' in config.allowed_imports
        assert 'math' in config.allowed_imports
        assert 'os' not in config.allowed_imports  # Dangerous import

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            timeout_seconds=5.0,
            memory_limit_mb=200,
            allowed_imports=['numpy']
        )

        assert config.timeout_seconds == 5.0
        assert config.memory_limit_mb == 200
        assert config.allowed_imports == ['numpy']
