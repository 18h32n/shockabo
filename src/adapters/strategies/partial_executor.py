"""Partial execution engine for DSL programs with confidence scoring.

This module implements partial execution of DSL programs to quickly evaluate
their likelihood of success without running the full program.
"""

import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.adapters.strategies.partial_execution_sandbox import (
    PartialExecutionSandbox,
    SandboxConfig,
)
from src.domain.dsl.base import Operation, OperationResult
from src.domain.dsl.types import Grid
from src.domain.models import PartialExecutionResult

logger = structlog.get_logger(__name__)


@dataclass
class PartialExecutionConfig:
    """Configuration for partial execution behavior."""
    max_operations: int = 3  # Max operations to execute
    timeout_ms: float = 100.0  # Timeout for partial execution
    memory_limit_mb: float = 10.0  # Memory limit for execution
    enable_sandboxing: bool = True  # Whether to use sandbox
    collect_intermediate_grids: bool = True  # Store intermediate results


@dataclass
class ConfidenceMetrics:
    """Metrics used for calculating confidence score."""
    grid_size_similarity: float  # How similar is the grid size to expected
    color_distribution_match: float  # Color palette similarity
    pattern_preservation: float  # Are patterns being preserved
    operation_success_rate: float  # How many operations succeeded
    execution_efficiency: float  # How fast operations execute


class PartialExecutor:
    """Executes partial DSL programs for early confidence evaluation."""

    def __init__(self, config: PartialExecutionConfig | None = None):
        """Initialize the partial executor.

        Args:
            config: Configuration for partial execution
        """
        self.config = config or PartialExecutionConfig()
        self.logger = structlog.get_logger(__name__).bind(
            service="partial_executor",
            max_operations=self.config.max_operations
        )

        # Initialize sandbox if enabled
        self._sandbox = None
        if self.config.enable_sandboxing:
            self._initialize_sandbox()

    def _initialize_sandbox(self) -> None:
        """Initialize the sandboxed execution environment."""
        try:
            # Use the new secure sandbox
            sandbox_config = SandboxConfig(
                max_memory_mb=int(self.config.memory_limit_mb),
                timeout_ms=int(self.config.timeout_ms),
                max_grid_size=30,
                max_operations=self.config.max_operations,
                isolate_state=True,
                audit_logging=True,
                restrict_imports=True
            )

            self._sandbox_config = sandbox_config
            self.logger.info("sandbox_config_initialized")
        except Exception as e:
            self.logger.warning("sandbox_config_initialization_failed", error=str(e))
            self._sandbox_config = None

    async def execute_partial(
        self,
        operations: list[Operation],
        input_grid: Grid,
        expected_output_hints: dict[str, Any] | None = None
    ) -> tuple[PartialExecutionResult, float]:
        """Execute a partial program and compute confidence score.

        Args:
            operations: List of DSL operations to partially execute
            input_grid: Input grid to transform
            expected_output_hints: Optional hints about expected output

        Returns:
            Tuple of (PartialExecutionResult, confidence_score)
        """
        start_time = time.perf_counter()
        program_id = str(id(operations))

        # Limit operations to execute
        ops_to_execute = operations[:self.config.max_operations]

        # Use sandbox if available
        if self.config.enable_sandboxing and hasattr(self, '_sandbox_config'):
            # Create a temporary DSL program for sandbox execution
            from src.domain.dsl.base import DSLProgram
            temp_program = DSLProgram(
                operations=[{"name": op.get_name(), "params": op.parameters} for op in ops_to_execute],
                metadata={"id": program_id}
            )

            # Execute in sandbox
            async with PartialExecutionSandbox(self._sandbox_config) as sandbox:
                result = await sandbox.execute_partial(
                    temp_program,
                    input_grid,
                    max_operations=self.config.max_operations
                )

            # Extract intermediate grids if available
            intermediate_grids = [result.intermediate_grid] if self.config.collect_intermediate_grids else []

        else:
            # Fallback to non-sandboxed execution
            current_grid = [row[:] for row in input_grid]  # Deep copy
            intermediate_grids = []
            operations_executed = 0
            memory_used_mb = 0.0
            success = True
            error = None

            # Execute operations one by one
            for i, operation in enumerate(ops_to_execute):
                try:
                    # Check timeout
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    if elapsed_ms > self.config.timeout_ms:
                        error = "Partial execution timeout"
                        success = False
                        break

                    # Execute operation
                    op_result = await self._execute_single_operation(
                        operation, current_grid
                    )

                    if not op_result.success:
                        error = op_result.error_message
                        success = False
                        break

                    # Update state
                    current_grid = op_result.grid
                    operations_executed += 1

                    if self.config.collect_intermediate_grids:
                        intermediate_grids.append([row[:] for row in current_grid])

                    # Estimate memory usage
                    grid_size = len(current_grid) * len(current_grid[0]) if current_grid else 0
                    memory_used_mb = max(memory_used_mb, grid_size * 4 / (1024 * 1024))

                except Exception as e:
                    error = f"Operation {i} failed: {str(e)}"
                    success = False
                    break

            # Create execution result
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            result = PartialExecutionResult(
                program_id=program_id,
                operations_executed=operations_executed,
                intermediate_grid=current_grid,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                success=success,
                error=error
            )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            result, input_grid, expected_output_hints, intermediate_grids
        )

        return result, confidence_score

    async def _execute_single_operation(
        self,
        operation: Operation,
        grid: Grid
    ) -> OperationResult:
        """Execute a single operation with sandboxing if enabled.

        Args:
            operation: DSL operation to execute
            grid: Current grid state

        Returns:
            OperationResult from the operation
        """
        if self._sandbox and self.config.enable_sandboxing:
            # Execute in sandbox (would need proper sandbox integration)
            # For now, execute directly
            return operation.execute(grid)
        else:
            # Direct execution
            return operation.execute(grid)

    def _calculate_confidence_score(
        self,
        execution_result: PartialExecutionResult,
        input_grid: Grid,
        expected_hints: dict[str, Any] | None,
        intermediate_grids: list[Grid]
    ) -> float:
        """Calculate confidence score based on partial execution.

        Args:
            execution_result: Result of partial execution
            input_grid: Original input grid
            expected_hints: Hints about expected output
            intermediate_grids: Grids after each operation

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not execution_result.success:
            return 0.0

        metrics = self._compute_confidence_metrics(
            execution_result, input_grid, expected_hints, intermediate_grids
        )

        # Weighted combination of metrics
        weights = {
            "grid_size": 0.2,
            "color_dist": 0.2,
            "pattern": 0.2,
            "success_rate": 0.3,
            "efficiency": 0.1
        }

        confidence = (
            weights["grid_size"] * metrics.grid_size_similarity +
            weights["color_dist"] * metrics.color_distribution_match +
            weights["pattern"] * metrics.pattern_preservation +
            weights["success_rate"] * metrics.operation_success_rate +
            weights["efficiency"] * metrics.execution_efficiency
        )

        return min(max(confidence, 0.0), 1.0)

    def _compute_confidence_metrics(
        self,
        execution_result: PartialExecutionResult,
        input_grid: Grid,
        expected_hints: dict[str, Any] | None,
        intermediate_grids: list[Grid]
    ) -> ConfidenceMetrics:
        """Compute individual confidence metrics.

        Args:
            execution_result: Result of partial execution
            input_grid: Original input grid
            expected_hints: Hints about expected output
            intermediate_grids: Grids after each operation

        Returns:
            ConfidenceMetrics object
        """
        current_grid = execution_result.intermediate_grid

        # Grid size similarity
        if expected_hints and "output_size" in expected_hints:
            expected_size = expected_hints["output_size"]
            current_size = (len(current_grid), len(current_grid[0]) if current_grid else 0)
            size_diff = abs(expected_size[0] - current_size[0]) + abs(expected_size[1] - current_size[1])
            grid_size_similarity = 1.0 / (1.0 + size_diff * 0.1)
        else:
            # Default: prefer grids that don't grow too large
            input_size = len(input_grid) * len(input_grid[0]) if input_grid else 1
            current_size = len(current_grid) * len(current_grid[0]) if current_grid else 1
            growth_ratio = current_size / input_size
            grid_size_similarity = 1.0 / (1.0 + abs(growth_ratio - 1.0))

        # Color distribution match
        if expected_hints and "expected_colors" in expected_hints:
            expected_colors = set(expected_hints["expected_colors"])
            current_colors = set()
            for row in current_grid:
                current_colors.update(row)

            # Check for invalid colors (outside 0-9)
            invalid_colors = {c for c in current_colors if c < 0 or c > 9}
            if invalid_colors:
                color_distribution_match = 0.0
            else:
                overlap = len(expected_colors & current_colors)
                color_distribution_match = overlap / len(expected_colors) if expected_colors else 0.5
        else:
            # Default: check for valid color range
            all_colors = set()
            for row in current_grid:
                all_colors.update(row)
            invalid_colors = {c for c in all_colors if c < 0 or c > 9}
            color_distribution_match = 0.0 if invalid_colors else 0.8

        # Pattern preservation (simple check - grid not becoming uniform)
        if current_grid:
            unique_values = len({val for row in current_grid for val in row})
            total_values = len(current_grid) * len(current_grid[0])
            pattern_preservation = min(unique_values / max(total_values * 0.1, 1), 1.0)
        else:
            pattern_preservation = 0.0

        # Operation success rate
        operation_success_rate = 1.0  # All operations succeeded if we got here

        # Execution efficiency
        time_per_op = execution_result.execution_time_ms / max(execution_result.operations_executed, 1)
        execution_efficiency = 1.0 / (1.0 + time_per_op / 10.0)  # 10ms is reasonable per op

        return ConfidenceMetrics(
            grid_size_similarity=grid_size_similarity,
            color_distribution_match=color_distribution_match,
            pattern_preservation=pattern_preservation,
            operation_success_rate=operation_success_rate,
            execution_efficiency=execution_efficiency
        )

    def get_expected_output_hints(
        self,
        task_examples: list[dict[str, Grid]]
    ) -> dict[str, Any]:
        """Extract hints about expected output from task examples.

        Args:
            task_examples: Training examples with input/output pairs

        Returns:
            Dictionary of hints about expected output characteristics
        """
        if not task_examples:
            return {}

        # Analyze output grids
        output_sizes = []
        all_colors = set()

        for example in task_examples:
            if "output" in example:
                output = example["output"]
                output_sizes.append((len(output), len(output[0]) if output else 0))

                for row in output:
                    all_colors.update(row)

        # Compute average output size
        if output_sizes:
            avg_height = sum(s[0] for s in output_sizes) / len(output_sizes)
            avg_width = sum(s[1] for s in output_sizes) / len(output_sizes)

            return {
                "output_size": (int(avg_height), int(avg_width)),
                "expected_colors": list(all_colors),
                "num_examples": len(task_examples)
            }

        return {}

    async def close(self) -> None:
        """Clean up resources."""
        if self._sandbox:
            try:
                await self._sandbox.close()
            except Exception as e:
                self.logger.warning("sandbox_cleanup_failed", error=str(e))
