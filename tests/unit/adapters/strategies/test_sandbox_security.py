"""
Adversarial security tests for the sandbox executor.

Tests various attack vectors and ensures proper containment.
"""

import sys
import time

import pytest

from src.adapters.strategies.sandbox_executor import SandboxConfig, SandboxExecutor


class TestSandboxSecurity:
    """Test suite for sandbox security and adversarial attacks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = SandboxExecutor(SandboxConfig())
        self.test_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.executor, '_cleanup_all_processes'):
            self.executor._cleanup_all_processes()

    def test_import_restriction_file_access(self):
        """Test that file system access is blocked."""
        malicious_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to read sensitive file
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    return grid
"""
        result = self.executor.execute(malicious_code, "solve_task", self.test_grid)

        assert not result.success
        assert "open" in result.error or "NameError" in result.error

    def test_import_restriction_os_access(self):
        """Test that OS module access is blocked."""
        malicious_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    import os
    # Try to execute system command
    os.system("echo 'pwned' > /tmp/hacked.txt")
    return grid
"""
        result = self.executor.execute(malicious_code, "solve_task", self.test_grid)

        assert not result.success
        assert ("import" in result.error or "__import__" in result.error or 
                "not allowed in sandbox" in result.error)

    def test_import_restriction_subprocess(self):
        """Test that subprocess module is blocked."""
        malicious_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    import subprocess
    # Try to run shell command
    subprocess.run(["ls", "-la", "/"])
    return grid
"""
        result = self.executor.execute(malicious_code, "solve_task", self.test_grid)

        assert not result.success
        assert ("import" in result.error or "__import__" in result.error or 
                "not allowed in sandbox" in result.error)

    def test_memory_bomb_protection(self):
        """Test protection against memory exhaustion attacks."""
        memory_bomb_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to allocate huge amount of memory
    huge_list = []
    for i in range(1000000):
        huge_list.append([0] * 10000)
    return grid
"""
        result = self.executor.execute(memory_bomb_code, "solve_task", self.test_grid)

        assert not result.success
        assert result.memory_exceeded or "memory" in result.error.lower() or result.timed_out

    def test_cpu_bomb_protection(self):
        """Test protection against infinite loops."""
        cpu_bomb_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Infinite loop
    while True:
        pass
    return grid
"""
        result = self.executor.execute(cpu_bomb_code, "solve_task", self.test_grid)

        assert not result.success
        assert result.timed_out
        assert result.metrics.execution_time_ms >= 900  # Should timeout at ~1000ms
        assert result.metrics.execution_time_ms < 1500  # But not too late

    def test_fork_bomb_protection(self):
        """Test protection against fork bomb attacks."""
        fork_bomb_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    import multiprocessing
    # Try to spawn many processes
    for i in range(100):
        p = multiprocessing.Process(target=lambda: None)
        p.start()
    return grid
"""
        result = self.executor.execute(fork_bomb_code, "solve_task", self.test_grid)

        assert not result.success
        assert "import" in result.error or "multiprocessing" in result.error

    def test_network_access_blocked(self):
        """Test that network access is blocked."""
        network_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    import urllib.request
    # Try to make network request
    response = urllib.request.urlopen('http://example.com')
    return grid
"""
        result = self.executor.execute(network_code, "solve_task", self.test_grid)

        assert not result.success
        assert "import" in result.error or "urllib" in result.error

    def test_eval_exec_blocked(self):
        """Test that eval and exec are blocked."""
        eval_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to use eval
    eval("__import__('os').system('ls')")
    return grid
"""
        result = self.executor.execute(eval_code, "solve_task", self.test_grid)

        assert not result.success
        assert "eval" in result.error or "NameError" in result.error

    def test_globals_locals_manipulation(self):
        """Test protection against globals/locals manipulation."""
        globals_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to access and modify globals
    globals()['__builtins__']['__import__'] = lambda x: None
    return grid
"""
        result = self.executor.execute(globals_code, "solve_task", self.test_grid)

        assert not result.success
        # Either globals() is blocked or the modification fails

    def test_recursion_limit(self):
        """Test protection against deep recursion."""
        recursion_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    def recurse(n):
        if n > 0:
            return recurse(n - 1)
        return n
    
    # Try deep recursion
    result = recurse(100000)
    return grid
"""
        result = self.executor.execute(recursion_code, "solve_task", self.test_grid)

        # Should either fail with recursion error or timeout
        assert not result.success

    def test_code_injection_via_grid(self):
        """Test protection against code injection via input data."""
        # Try to inject code through grid values
        malicious_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        injection_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to execute grid values as code
    for row in grid:
        for val in row:
            # This should fail - no eval available
            eval(str(val))
    return grid
"""
        result = self.executor.execute(injection_code, "solve_task", malicious_grid)

        assert not result.success
        assert "eval" in result.error or "NameError" in result.error

    def test_module_attribute_access(self):
        """Test that module attributes cannot be used for escape."""
        escape_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to access numpy internals to escape
    import numpy as np
    np.__loader__.__module__.__builtins__['__import__']('os')
    return grid
"""
        result = self.executor.execute(escape_code, "solve_task", self.test_grid)

        assert not result.success
        # Should fail at some point in the chain

    def test_class_escape_attempt(self):
        """Test protection against class-based escapes."""
        class_escape_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to escape via class internals
    class Evil:
        def __init__(self):
            self.__class__.__bases__[0].__subclasses__()
    
    e = Evil()
    return grid
"""
        result = self.executor.execute(class_escape_code, "solve_task", self.test_grid)

        # Should complete without escaping sandbox
        # May succeed but won't have access to dangerous functions
        if result.success:
            assert result.result == self.test_grid

    def test_concurrent_execution_isolation(self):
        """Test that concurrent executions are isolated."""
        code1 = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    global shared_data
    shared_data = 'code1'
    return [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
"""

        code2 = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    try:
        # Try to access data from other execution
        return [[shared_data]]
    except:
        return [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
"""

        # Execute concurrently
        result1 = self.executor.execute(code1, "solve_task", self.test_grid)
        result2 = self.executor.execute(code2, "solve_task", self.test_grid)

        # Second execution should not see first's data
        assert result2.success
        assert result2.result == [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_resource_limits(self):
        """Test Windows-specific resource limits.
        
        Note: Windows memory limits are not as strictly enforced as Unix systems
        due to virtual memory management. This test verifies basic functionality
        rather than strict enforcement.
        """
        memory_test_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Allocate reasonable amount
    size = 50 * 1024 * 1024  # 50MB
    data = bytearray(size)
    return grid
"""
        result = self.executor.execute(memory_test_code, "solve_task", self.test_grid)

        # Should succeed with reasonable allocation
        assert result.success

        # Test extreme memory allocation that should cause system issues
        extreme_memory_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Try to allocate extremely large amount that should cause issues
    try:
        size = 2 * 1024 * 1024 * 1024  # 2GB
        data = bytearray(size)
        return grid
    except MemoryError:
        # This is expected behavior on systems with limited memory
        raise RuntimeError("Memory allocation failed as expected")
"""
        result2 = self.executor.execute(extreme_memory_code, "solve_task", self.test_grid)

        # On Windows, we accept that memory limits might not be strictly enforced
        # The test passes if either:
        # 1. The allocation succeeds (Windows has virtual memory)
        # 2. It fails with memory-related error
        # 3. It times out due to system strain
        assert (result2.success or result2.memory_exceeded or result2.timed_out or
                (result2.error and ("memory" in result2.error.lower() or 
                                   "runtime" in result2.error.lower())))

    def test_process_cleanup_after_timeout(self):
        """Test that processes are cleaned up after timeout."""
        initial_count = len(self.executor._active_processes)

        timeout_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    import time
    time.sleep(10)  # Sleep longer than timeout
    return grid
"""
        result = self.executor.execute(timeout_code, "solve_task", self.test_grid)

        assert not result.success
        assert result.timed_out

        # Wait a bit for cleanup
        time.sleep(0.1)

        # Check that process was cleaned up
        assert len(self.executor._active_processes) == initial_count

    def test_scipy_imports_allowed(self):
        """Test that scipy.ndimage is properly allowed."""
        scipy_code = """
def solve_task(grid: List[List[int]]) -> List[List[int]]:
    from scipy import ndimage
    # Use scipy for connected components
    labeled, num = ndimage.label(np.array(grid) > 5)
    return labeled.tolist()
"""
        result = self.executor.execute(scipy_code, "solve_task", self.test_grid)

        # Should succeed since scipy.ndimage is allowed
        assert result.success
        assert isinstance(result.result, list)
