"""Secure sandbox for partial DSL program execution.

This module provides a secure, isolated environment for executing partial DSL
programs with strict resource limits and security controls.
"""

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass

import psutil

# Platform-specific imports for resource limits
if sys.platform != "win32":
    import resource

from src.domain.dsl.base import DSLProgram, Operation
from src.domain.dsl.types import Grid
from src.domain.models import PartialExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution environment."""
    max_memory_mb: int = 10  # 10MB max memory
    timeout_ms: int = 100     # 100ms max execution time
    max_grid_size: int = 30   # Max 30x30 grid
    max_operations: int = 5   # Max 5 operations for partial execution
    isolate_state: bool = True  # Isolate state between executions
    audit_logging: bool = True  # Enable security audit logging
    restrict_imports: bool = True  # Restrict dangerous imports


class SecurityValidator:
    """Validate programs for security risks before execution."""

    def __init__(self, config: SandboxConfig):
        """Initialize security validator.

        Args:
            config: Sandbox configuration
        """
        self.config = config

    def validate_operations(self, operations: list[Operation]) -> tuple[bool, str | None]:
        """Check for dangerous operation patterns.

        Args:
            operations: List of DSL operations to validate

        Returns:
            Tuple of (is_safe, rejection_reason)
        """
        # Check operation count
        if len(operations) > self.config.max_operations:
            return False, f"Too many operations: {len(operations)} > {self.config.max_operations}"

        # Check for dangerous patterns
        for i, op in enumerate(operations):
            # Check for infinite loops
            if self._could_cause_infinite_loop(op, operations[i+1:]):
                return False, f"Potential infinite loop detected at operation {i}"

            # Check for memory explosion
            if self._could_cause_memory_explosion(op):
                return False, f"Potential memory explosion at operation {i}"

        return True, None

    def validate_grid_operations(
        self,
        operation: Operation,
        input_grid: Grid
    ) -> tuple[bool, str | None]:
        """Validate that grid operations stay within bounds.

        Args:
            operation: Operation to validate
            input_grid: Input grid for the operation

        Returns:
            Tuple of (is_safe, rejection_reason)
        """
        # Check input grid size
        if len(input_grid) > self.config.max_grid_size:
            return False, f"Input grid too large: {len(input_grid)} rows"

        if any(len(row) > self.config.max_grid_size for row in input_grid):
            return False, f"Input grid too wide: max {self.config.max_grid_size} columns"

        # Estimate output size for certain operations
        op_name = operation.get_name().lower()

        if op_name in ["tile", "repeat", "expand"]:
            # These operations can multiply grid size
            if hasattr(operation, 'parameters'):
                factor = operation.parameters.get('factor', 1)
                if factor > 3:  # Conservative limit
                    return False, f"Operation {op_name} factor too large: {factor}"

        if op_name in ["zoom", "scale"]:
            # These can also expand the grid
            if hasattr(operation, 'parameters'):
                scale = operation.parameters.get('scale', 1)
                if scale > 2:
                    return False, f"Operation {op_name} scale too large: {scale}"

        return True, None

    def _could_cause_infinite_loop(
        self,
        operation: Operation,
        following_ops: list[Operation]
    ) -> bool:
        """Check if operation could cause infinite loop.

        Args:
            operation: Current operation
            following_ops: Operations that follow

        Returns:
            True if infinite loop is possible
        """
        op_name = operation.get_name().lower()

        # Check for recursive patterns
        if op_name in ["repeat_until", "while", "recurse"]:
            return True  # These aren't in our DSL but check anyway

        # Check for contradictory operations
        if following_ops:
            next_op = following_ops[0]
            if self._operations_contradict(operation, next_op):
                return False  # Not infinite, just inefficient

        return False

    def _could_cause_memory_explosion(self, operation: Operation) -> bool:
        """Check if operation could cause memory explosion.

        Args:
            operation: Operation to check

        Returns:
            True if memory explosion is possible
        """
        op_name = operation.get_name().lower()

        # Operations that can multiply data
        if op_name in ["tile", "repeat", "expand", "replicate"]:
            if hasattr(operation, 'parameters'):
                # Check multiplication factors
                factor = 1
                for param in ['factor', 'times', 'count', 'x', 'y']:
                    if param in operation.parameters:
                        factor *= operation.parameters[param]

                if factor > 10:
                    return True

        return False

    def _operations_contradict(self, op1: Operation, op2: Operation) -> bool:
        """Check if two operations contradict each other.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if operations contradict
        """
        name1 = op1.get_name().lower()
        name2 = op2.get_name().lower()

        # Rotation contradictions
        if name1 == "rotate" and name2 == "rotate":
            angle1 = op1.parameters.get('angle', 0)
            angle2 = op2.parameters.get('angle', 0)
            if (angle1 + angle2) % 360 == 0:
                return True

        # Color mapping contradictions
        if name1 == "map_colors" and name2 == "map_colors":
            # Could check if mappings cancel out
            pass

        return False


class PartialExecutionSandbox:
    """Secure sandbox for partial DSL program execution."""

    def __init__(self, config: SandboxConfig):
        """Initialize sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config
        self.validator = SecurityValidator(config)
        self._process_pool = None
        self._execution_count = 0

    def __enter__(self):
        """Enter sandbox context."""
        # Initialize process pool for isolation
        self._process_pool = ProcessPoolExecutor(
            max_workers=1,
            initializer=self._worker_init,
            initargs=(self.config,)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sandbox context and cleanup."""
        if self._process_pool:
            self._process_pool.shutdown(wait=True)

    @staticmethod
    def _worker_init(config: SandboxConfig):
        """Initialize worker process with resource limits.

        Args:
            config: Sandbox configuration
        """
        # Set resource limits only on non-Windows platforms
        if sys.platform != "win32":
            # Set memory limit (Linux only)
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (config.max_memory_mb * 1024 * 1024, config.max_memory_mb * 1024 * 1024)
                )

            # Set CPU time limit
            if hasattr(resource, 'RLIMIT_CPU'):
                cpu_limit = max(1, config.timeout_ms // 1000)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        else:
            # Windows: Use psutil or other methods for resource monitoring
            # Note: Windows doesn't have direct resource limits like Unix
            pass

        # Restrict imports if configured
        if config.restrict_imports:
            # Override __import__ to restrict modules
            import builtins
            original_import = builtins.__import__

            def restricted_import(name, *args, **kwargs):
                # Allow only safe modules
                safe_modules = {
                    'numpy', 'math', 'random', 'itertools',
                    'collections', 'functools', 'typing',
                }

                if name.split('.')[0] not in safe_modules:
                    raise ImportError(f"Import of '{name}' not allowed in sandbox")

                return original_import(name, *args, **kwargs)

            builtins.__import__ = restricted_import

    async def execute_partial(
        self,
        program: DSLProgram,
        input_grid: Grid,
        max_operations: int | None = None
    ) -> PartialExecutionResult:
        """Execute partial DSL program in sandbox.

        Args:
            program: DSL program to execute
            input_grid: Input grid
            max_operations: Maximum operations to execute

        Returns:
            Partial execution result
        """
        start_time = time.perf_counter()
        self._execution_count += 1
        execution_id = f"partial_exec_{self._execution_count}"

        # Validate program
        is_safe, rejection_reason = self.validator.validate_operations(program.operations)
        if not is_safe:
            return PartialExecutionResult(
                program_id=execution_id,
                operations_executed=0,
                intermediate_grid=input_grid,
                execution_time_ms=0,
                memory_used_mb=0,
                success=False,
                error=f"Security validation failed: {rejection_reason}"
            )

        # Determine operations to execute
        max_ops = max_operations or self.config.max_operations
        ops_to_execute = program.operations[:max_ops]

        # Execute in isolated process
        try:
            future = self._process_pool.submit(
                self._execute_operations_isolated,
                ops_to_execute,
                input_grid,
                execution_id,
                self.config
            )

            # Wait with timeout
            result = future.result(timeout=self.config.timeout_ms / 1000)

            # Calculate execution time
            execution_time = (time.perf_counter() - start_time) * 1000

            # Update result with timing
            result.execution_time_ms = execution_time

            # Audit log if enabled
            if self.config.audit_logging:
                self._audit_log(execution_id, result)

            return result

        except TimeoutError:
            return PartialExecutionResult(
                program_id=execution_id,
                operations_executed=0,
                intermediate_grid=input_grid,
                execution_time_ms=self.config.timeout_ms,
                memory_used_mb=0,
                success=False,
                error=f"Execution timeout ({self.config.timeout_ms}ms)"
            )
        except Exception as e:
            return PartialExecutionResult(
                program_id=execution_id,
                operations_executed=0,
                intermediate_grid=input_grid,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                memory_used_mb=0,
                success=False,
                error=f"Execution error: {str(e)}"
            )

    @staticmethod
    def _execute_operations_isolated(
        operations: list[Operation],
        input_grid: Grid,
        execution_id: str,
        config: SandboxConfig
    ) -> PartialExecutionResult:
        """Execute operations in isolated process.

        Args:
            operations: Operations to execute
            input_grid: Input grid
            execution_id: Execution ID for tracking
            config: Sandbox configuration

        Returns:
            Partial execution result
        """
        # Track resource usage
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        current_grid = input_grid
        operations_executed = 0

        try:
            for i, operation in enumerate(operations):
                # Check memory before each operation
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory - start_memory > config.max_memory_mb:
                    raise MemoryError(
                        f"Memory limit exceeded: {current_memory - start_memory:.1f}MB"
                    )

                # Execute operation
                result = operation.execute(current_grid)

                if not result.success:
                    return PartialExecutionResult(
                        program_id=execution_id,
                        operations_executed=i,
                        intermediate_grid=current_grid,
                        execution_time_ms=0,  # Set by caller
                        memory_used_mb=current_memory - start_memory,
                        success=False,
                        error=f"Operation {i} failed: {result.error_message}"
                    )

                current_grid = result.grid
                operations_executed += 1

            # Success
            final_memory = process.memory_info().rss / 1024 / 1024

            return PartialExecutionResult(
                program_id=execution_id,
                operations_executed=operations_executed,
                intermediate_grid=current_grid,
                execution_time_ms=0,  # Set by caller
                memory_used_mb=final_memory - start_memory,
                success=True,
                error=None
            )

        except Exception as e:
            current_memory = process.memory_info().rss / 1024 / 1024

            return PartialExecutionResult(
                program_id=execution_id,
                operations_executed=operations_executed,
                intermediate_grid=current_grid,
                execution_time_ms=0,  # Set by caller
                memory_used_mb=current_memory - start_memory,
                success=False,
                error=str(e)
            )

    def _audit_log(self, execution_id: str, result: PartialExecutionResult):
        """Log execution for security audit.

        Args:
            execution_id: Execution ID
            result: Execution result
        """
        audit_entry = {
            "execution_id": execution_id,
            "timestamp": time.time(),
            "operations_executed": result.operations_executed,
            "memory_used_mb": result.memory_used_mb,
            "execution_time_ms": result.execution_time_ms,
            "success": result.success,
            "error": result.error,
        }

        logger.info(f"Sandbox execution audit: {audit_entry}")
