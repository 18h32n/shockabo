"""
GPU-accelerated batch evaluation for DSL programs.

This module provides efficient batch evaluation of DSL programs using PyTorch
for GPU acceleration. It supports batching multiple programs, vectorized operations,
and automatic CPU fallback.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch

from src.domain.dsl.base import Operation as DSLOperation
from src.domain.dsl.types import Grid
from src.utils.gpu_ops import VectorizedOps

logger = structlog.get_logger(__name__)


@dataclass
class BatchProgram:
    """Represents a batched DSL program for GPU execution."""

    programs: list[list[DSLOperation]]
    max_length: int
    operation_masks: torch.Tensor  # Shape: (batch_size, max_length)
    padded_operations: list[list[DSLOperation]]  # Padded to max_length


@dataclass
class BatchEvaluationRequest:
    """Request for batch evaluation of programs."""

    programs: list[list[DSLOperation]]
    input_grids: list[Grid]
    device: str = "auto"
    batch_size: int = 100
    timeout_per_batch: float = 5.0


@dataclass
class BatchEvaluationResult:
    """Result of batch program evaluation."""

    output_grids: list[Grid | None]
    success_mask: torch.Tensor
    execution_times: list[float]
    device_used: str
    batch_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUMetrics:
    """Metrics for GPU execution."""

    kernel_time_ms: float
    memory_transfer_time_ms: float
    memory_allocated_mb: float
    memory_cached_mb: float
    device_utilization: float = 0.0


@dataclass
class DeviceCapabilities:
    """GPU device capabilities."""

    device_name: str
    compute_capability: tuple[int, int]
    total_memory_mb: float
    available_memory_mb: float
    max_threads_per_block: int
    max_grid_size: tuple[int, int, int]


class GPUBatchEvaluator:
    """
    Batch evaluator for DSL programs using GPU acceleration.

    Provides efficient batch evaluation of multiple DSL programs in parallel
    using PyTorch for GPU operations. Includes automatic memory management
    and CPU fallback.
    """

    def __init__(
        self,
        device: str = "auto",
        max_batch_size: int = 100,
        memory_limit_mb: float = 8000,
        enable_profiling: bool = False
    ):
        """
        Initialize GPU batch evaluator.

        Args:
            device: Device to use ("cuda", "cpu", or "auto")
            max_batch_size: Maximum programs to evaluate in parallel
            memory_limit_mb: Maximum GPU memory to use (MB)
            enable_profiling: Enable detailed profiling
        """
        self.device = self._initialize_device(device)
        self.max_batch_size = max_batch_size
        self.memory_limit_mb = memory_limit_mb
        self.enable_profiling = enable_profiling

        # Initialize vectorized operations
        self.vectorized_ops = VectorizedOps(device=self.device)

        # Performance tracking
        self.performance_stats = {
            "total_batches_evaluated": 0,
            "total_programs_evaluated": 0,
            "gpu_time_ms": 0.0,
            "cpu_time_ms": 0.0,
            "memory_peaks_mb": []
        }

        self.logger = structlog.get_logger(__name__).bind(
            evaluator="gpu_batch",
            device=self.device
        )

        # Check device capabilities
        self.device_capabilities = self._get_device_capabilities()

        # Memory management
        self.memory_tracker = {
            "current_allocated_mb": 0,
            "peak_allocated_mb": 0,
            "batch_size_history": []
        }

    def _initialize_device(self, device: str) -> torch.device:
        """Initialize and validate device selection."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("cuda_not_available_falling_back_to_cpu")
                return torch.device("cpu")
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _get_device_capabilities(self) -> DeviceCapabilities | None:
        """Get GPU device capabilities if available."""
        if self.device.type != "cuda":
            return None

        try:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024 * 1024)  # MB
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)

            return DeviceCapabilities(
                device_name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_mb=total_memory,
                available_memory_mb=total_memory - allocated,
                max_threads_per_block=props.max_threads_per_block,
                max_grid_size=props.max_grid_dim
            )
        except Exception as e:
            logger.error("failed_to_get_device_capabilities", error=str(e))
            return None

    def batch_evaluate(
        self,
        request: BatchEvaluationRequest
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple DSL programs in batch with automatic device selection.

        Args:
            request: Batch evaluation request

        Returns:
            BatchEvaluationResult with outputs and metrics
        """
        start_time = time.perf_counter()

        # Auto-select device if needed
        selected_device = self._select_device(request)

        # Calculate adaptive batch sizes based on grid sizes
        grid_sizes = [(len(g), len(g[0]) if g else 0) for g in request.input_grids]
        adaptive_batch_size = self._calculate_adaptive_batch_size(
            len(request.programs),
            grid_sizes
        )

        # Use smaller of requested and adaptive batch size
        effective_batch_size = min(request.batch_size, adaptive_batch_size)

        # Split into manageable batches
        batches = self._create_batches(
            request.programs,
            request.input_grids,
            effective_batch_size
        )

        all_outputs = []
        all_success = []
        all_times = []
        device_usage = {"gpu": 0, "cpu": 0}

        for batch_idx, (batch_programs, batch_grids) in enumerate(batches):
            try:
                # Determine device for this batch
                use_gpu = (
                    selected_device == "cuda" and
                    self.device.type == "cuda" and
                    self._check_memory_available(len(batch_programs))
                )

                if use_gpu:
                    try:
                        # Try GPU evaluation
                        batch_result = self._evaluate_batch_gpu(
                            batch_programs,
                            batch_grids,
                            request.timeout_per_batch
                        )
                        device_usage["gpu"] += len(batch_programs)
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            "gpu_oom_fallback_to_cpu",
                            batch_idx=batch_idx,
                            batch_size=len(batch_programs)
                        )
                        # Fallback to CPU
                        torch.cuda.empty_cache()
                        batch_result = self._evaluate_batch_cpu(
                            batch_programs,
                            batch_grids,
                            request.timeout_per_batch
                        )
                        device_usage["cpu"] += len(batch_programs)
                else:
                    # Use CPU
                    batch_result = self._evaluate_batch_cpu(
                        batch_programs,
                        batch_grids,
                        request.timeout_per_batch
                    )
                    device_usage["cpu"] += len(batch_programs)

                all_outputs.extend(batch_result.output_grids)
                all_success.append(batch_result.success_mask)
                all_times.extend(batch_result.execution_times)

            except Exception as e:
                logger.error(
                    "batch_evaluation_error",
                    batch_idx=batch_idx,
                    error=str(e),
                    exc_info=True
                )
                # Add failed results for this batch
                batch_size = len(batch_programs)
                all_outputs.extend([None] * batch_size)
                all_success.append(torch.zeros(batch_size, dtype=torch.bool))
                all_times.extend([0.0] * batch_size)

        # Combine results
        total_time = (time.perf_counter() - start_time) * 1000
        success_tensor = torch.cat(all_success) if all_success else torch.tensor([])

        # Update stats
        self.performance_stats["total_batches_evaluated"] += len(batches)
        self.performance_stats["total_programs_evaluated"] += len(request.programs)

        # Determine final device used
        final_device = "hybrid" if device_usage["gpu"] > 0 and device_usage["cpu"] > 0 else (
            "cuda" if device_usage["gpu"] > 0 else "cpu"
        )

        batch_stats = {
            "num_batches": len(batches),
            "total_programs": len(request.programs),
            "success_rate": float(success_tensor.float().mean()) if len(success_tensor) > 0 else 0.0,
            "total_time_ms": total_time,
            "avg_time_per_program_ms": total_time / len(request.programs) if request.programs else 0,
            "device_usage": device_usage,
            "effective_batch_size": effective_batch_size,
            "fallback": device_usage["cpu"] > 0 and selected_device == "cuda"
        }

        return BatchEvaluationResult(
            output_grids=all_outputs,
            success_mask=success_tensor,
            execution_times=all_times,
            device_used=final_device,
            batch_stats=batch_stats
        )

    def _select_device(self, request: BatchEvaluationRequest) -> str:
        """Select optimal device based on request and system state."""
        if request.device != "auto":
            return request.device

        # Check GPU availability
        if not torch.cuda.is_available():
            return "cpu"

        # Check if GPU has enough memory for minimal batch
        min_batch_memory = 10  # MB for small batch
        if self.device_capabilities:
            if self.device_capabilities.available_memory_mb < min_batch_memory:
                logger.info("gpu_memory_too_low_using_cpu")
                return "cpu"

        # Default to GPU if available
        return "cuda"

    def _create_batches(
        self,
        programs: list[list[DSLOperation]],
        input_grids: list[Grid],
        batch_size: int
    ) -> list[tuple[list[list[DSLOperation]], list[Grid]]]:
        """Split programs and grids into batches."""
        batches = []

        # Ensure we have matching programs and grids
        assert len(programs) == len(input_grids), "Programs and grids must match"

        # Handle empty programs or zero batch size
        if len(programs) == 0 or batch_size <= 0:
            return batches

        for i in range(0, len(programs), batch_size):
            batch_programs = programs[i:i + batch_size]
            batch_grids = input_grids[i:i + batch_size]
            batches.append((batch_programs, batch_grids))

        return batches

    def _check_memory_available(self, batch_size: int) -> bool:
        """Check if enough GPU memory is available for batch."""
        if self.device.type != "cuda":
            return True

        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

            # Estimate memory needed per program
            # Assume max 30x30 grids, float32, with operations overhead
            memory_per_program_mb = 0.5  # Conservative estimate
            estimated_mb = batch_size * memory_per_program_mb

            # Check against limit
            projected_usage = allocated + estimated_mb

            # Update tracker
            self.memory_tracker["current_allocated_mb"] = allocated

            return projected_usage < self.memory_limit_mb * 0.9  # 90% threshold

        except Exception:
            return True  # Optimistic if can't determine

    def _calculate_adaptive_batch_size(
        self,
        num_programs: int,
        grid_sizes: list[tuple[int, int]]
    ) -> int:
        """Calculate optimal batch size based on current memory usage."""
        if self.device.type != "cuda":
            return min(num_programs, self.max_batch_size)

        try:
            # Get current memory state
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            available = self.memory_limit_mb - allocated

            # Estimate memory per program based on grid sizes
            max_grid_size = max(h * w for h, w in grid_sizes) if grid_sizes else 900
            memory_per_program = (max_grid_size * 4 * 2) / (1024 * 1024)  # float32, input+output

            # Calculate safe batch size
            safe_batch_size = int(available * 0.8 / memory_per_program)  # 80% of available

            # Clamp to reasonable range
            batch_size = max(1, min(safe_batch_size, self.max_batch_size, num_programs))

            # Track batch size for analysis
            self.memory_tracker["batch_size_history"].append(batch_size)

            logger.debug(
                "adaptive_batch_size_calculated",
                num_programs=num_programs,
                allocated_mb=allocated,
                available_mb=available,
                batch_size=batch_size
            )

            return batch_size

        except Exception as e:
            logger.warning("adaptive_batch_size_error", error=str(e))
            return min(num_programs, self.max_batch_size // 2)  # Conservative fallback

    def _evaluate_batch_gpu(
        self,
        programs: list[list[DSLOperation]],
        input_grids: list[Grid],
        timeout: float
    ) -> BatchEvaluationResult:
        """Evaluate batch of programs on GPU with memory monitoring."""
        start_time = time.perf_counter()
        batch_size = len(programs)

        # Memory checkpoint before evaluation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        try:
            # Prepare batch data
            batch_program = self._prepare_batch_programs(programs)
            batch_tensors = self._grids_to_tensors(input_grids)

            # Initialize output tensors
            output_tensors = batch_tensors.clone()
            success_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            # Enable gradient checkpointing for memory efficiency
            with torch.no_grad():
                # Execute operations in sequence
                for op_idx in range(batch_program.max_length):
                    # Check memory periodically
                    if op_idx % 10 == 0 and self.device.type == "cuda":
                        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                        if current_memory > self.memory_limit_mb * 0.95:
                            logger.warning(
                                "gpu_memory_pressure",
                                op_idx=op_idx,
                                memory_mb=current_memory,
                                limit_mb=self.memory_limit_mb
                            )
                            # Force garbage collection
                            torch.cuda.empty_cache()

                    # Get operations for this step
                    current_ops = [
                        batch_program.padded_operations[i][op_idx]
                        for i in range(batch_size)
                    ]

                    # Apply operations where not masked
                    mask = batch_program.operation_masks[:, op_idx]
                    if mask.any():
                        try:
                            output_tensors = self._apply_vectorized_operations(
                                output_tensors,
                                current_ops,
                                mask
                            )
                        except torch.cuda.OutOfMemoryError:
                            logger.error(
                                "gpu_out_of_memory",
                                op_idx=op_idx,
                                batch_size=batch_size
                            )
                            # Try to recover by clearing cache
                            torch.cuda.empty_cache()
                            raise
                        except Exception as e:
                            logger.error(
                                "gpu_operation_error",
                                op_idx=op_idx,
                                error=str(e)
                            )
                            # Mark failed programs
                            success_mask[mask] = False

            # Convert back to grids
            output_grids = self._tensors_to_grids(output_tensors, success_mask)

            # Calculate execution times
            total_time = (time.perf_counter() - start_time) * 1000
            execution_times = [total_time / batch_size] * batch_size

            # Memory statistics
            if self.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.memory_tracker["peak_allocated_mb"] = max(
                    self.memory_tracker["peak_allocated_mb"],
                    peak_memory
                )
                self.performance_stats["memory_peaks_mb"].append(peak_memory)

                batch_stats = {
                    "gpu_time_ms": total_time,
                    "memory_used_mb": peak_memory - initial_memory,
                    "peak_memory_mb": peak_memory
                }
            else:
                batch_stats = {"gpu_time_ms": total_time}

            # Update performance stats
            self.performance_stats["gpu_time_ms"] += total_time

            return BatchEvaluationResult(
                output_grids=output_grids,
                success_mask=success_mask,
                execution_times=execution_times,
                device_used="cuda",
                batch_stats=batch_stats
            )

        finally:
            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _evaluate_batch_cpu(
        self,
        programs: list[list[DSLOperation]],
        input_grids: list[Grid],
        timeout: float
    ) -> BatchEvaluationResult:
        """Fallback CPU evaluation of batch using existing sandbox executor."""
        from src.adapters.strategies.sandbox_executor import SandboxExecutor

        start_time = time.perf_counter()
        batch_size = len(programs)

        # Initialize CPU executor if not available
        if not hasattr(self, '_cpu_executor'):
            self._cpu_executor = SandboxExecutor()

        output_grids = []
        success_mask = []
        execution_times = []

        logger.info(
            "cpu_fallback_initiated",
            batch_size=batch_size,
            reason="GPU unavailable or memory insufficient"
        )

        # Process programs sequentially on CPU
        for i, (program, grid) in enumerate(zip(programs, input_grids, strict=False)):
            try:
                # Execute program
                result = self._cpu_executor.execute_operations(
                    operations=program,
                    input_grid=grid,
                    timeout=timeout / batch_size,  # Distribute timeout
                    memory_limit_mb=100
                )

                if result.success and result.output is not None:
                    output_grids.append(result.output)
                    success_mask.append(True)
                    execution_times.append(result.execution_time)
                else:
                    output_grids.append(None)
                    success_mask.append(False)
                    execution_times.append(0.0)

            except Exception as e:
                logger.error(
                    "cpu_execution_error",
                    program_idx=i,
                    error=str(e)
                )
                output_grids.append(None)
                success_mask.append(False)
                execution_times.append(0.0)

        # Convert to tensors
        success_tensor = torch.tensor(success_mask, dtype=torch.bool)

        # Calculate total time
        total_time = (time.perf_counter() - start_time) * 1000

        # Update stats
        self.performance_stats["cpu_time_ms"] += total_time

        batch_stats = {
            "cpu_time_ms": total_time,
            "fallback": True,
            "success_rate": float(success_tensor.float().mean()),
            "avg_time_per_program": total_time / batch_size if batch_size > 0 else 0
        }

        return BatchEvaluationResult(
            output_grids=output_grids,
            success_mask=success_tensor,
            execution_times=execution_times,
            device_used="cpu",
            batch_stats=batch_stats
        )

    def _prepare_batch_programs(
        self,
        programs: list[list[DSLOperation]]
    ) -> BatchProgram:
        """Prepare programs for batch execution."""
        batch_size = len(programs)
        max_length = max(len(prog) for prog in programs)

        # Create operation masks
        operation_masks = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
            device=self.device
        )

        # Pad programs
        padded_operations = []
        for i, program in enumerate(programs):
            # Set mask for valid operations
            operation_masks[i, :len(program)] = True

            # Pad program with no-ops (create mock noop operations)
            noop = type('NoOp', (), {'name': 'noop', 'parameters': {}})()
            padded = program + [noop] * (max_length - len(program))
            padded_operations.append(padded)

        return BatchProgram(
            programs=programs,
            max_length=max_length,
            operation_masks=operation_masks,
            padded_operations=padded_operations
        )

    def _grids_to_tensors(self, grids: list[Grid]) -> torch.Tensor:
        """Convert grids to batch tensor."""
        # Find max dimensions
        max_h = max(len(grid) for grid in grids)
        max_w = max(max(len(row) for row in grid) if grid else 0 for grid in grids)

        # Create padded tensor
        batch_tensor = torch.zeros(
            (len(grids), max_h, max_w),
            dtype=torch.float32,
            device=self.device
        )

        # Fill tensor
        for i, grid in enumerate(grids):
            h = len(grid)
            for j in range(h):
                w = len(grid[j])
                batch_tensor[i, j, :w] = torch.tensor(grid[j], dtype=torch.float32)

        return batch_tensor

    def _tensors_to_grids(
        self,
        tensors: torch.Tensor,
        success_mask: torch.Tensor
    ) -> list[Grid | None]:
        """Convert batch tensor back to grids."""
        grids = []
        batch_size = tensors.shape[0]

        for i in range(batch_size):
            if not success_mask[i]:
                grids.append(None)
                continue

            # Convert tensor to numpy
            grid_np = tensors[i].cpu().numpy().astype(int)

            # Find actual grid size (remove padding)
            # Assume 0 is padding (might need adjustment)
            h, w = grid_np.shape

            # Convert to list format
            grid = []
            for row in grid_np:
                # Could add logic to detect actual width
                grid.append(row.tolist())

            grids.append(grid)

        return grids

    def _apply_vectorized_operations(
        self,
        tensors: torch.Tensor,
        operations: list[DSLOperation],
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply vectorized operations to batch of grids."""
        # Clone tensor to avoid in-place modifications
        result = tensors.clone()

        # Group operations by type for efficient dispatch
        op_groups = {}
        for i, (op, active) in enumerate(zip(operations, mask, strict=False)):
            if active and hasattr(op, 'name') and op.name != "noop":
                if op.name not in op_groups:
                    op_groups[op.name] = []
                op_groups[op.name].append((i, op))

        # Apply each operation type
        for op_name, op_list in op_groups.items():
            indices = [i for i, _ in op_list]

            # Dispatch to appropriate vectorized operation
            if op_name == "rotate":
                # Get rotation angles
                angles = [op.parameters.get("angle", 90) for _, op in op_list]
                # Apply rotation to subset
                for idx, angle in zip(indices, angles, strict=False):
                    if angle == 90:
                        result[idx] = self.vectorized_ops.rotate_90(result[idx].unsqueeze(0)).squeeze(0)
                    elif angle == 180:
                        result[idx] = self.vectorized_ops.rotate_180(result[idx].unsqueeze(0)).squeeze(0)
                    elif angle == 270:
                        result[idx] = self.vectorized_ops.rotate_270(result[idx].unsqueeze(0)).squeeze(0)

            elif op_name == "flip":
                # Get flip directions
                directions = [op.parameters.get("direction", "horizontal") for _, op in op_list]
                for idx, direction in zip(indices, directions, strict=False):
                    if direction == "horizontal":
                        result[idx] = self.vectorized_ops.flip_horizontal(result[idx].unsqueeze(0)).squeeze(0)
                    elif direction == "vertical":
                        result[idx] = self.vectorized_ops.flip_vertical(result[idx].unsqueeze(0)).squeeze(0)
                    elif direction == "diagonal_main":
                        result[idx] = self.vectorized_ops.flip_main_diagonal(result[idx].unsqueeze(0)).squeeze(0)
                    elif direction == "diagonal_anti":
                        result[idx] = self.vectorized_ops.flip_anti_diagonal(result[idx].unsqueeze(0)).squeeze(0)

            elif op_name == "translate":
                # Get translation parameters
                for idx, op in op_list:
                    offset = op.parameters.get("offset", (0, 0))
                    fill_color = op.parameters.get("fill_color", 0)
                    result[idx] = self.vectorized_ops.translate(
                        result[idx].unsqueeze(0),
                        shift_y=offset[0],
                        shift_x=offset[1],
                        fill_value=fill_color
                    ).squeeze(0)

            elif op_name == "map_colors":
                # Get color mapping
                for idx, op in op_list:
                    color_map = op.parameters.get("color_map", {})
                    if color_map:
                        # Convert dict to tensor mapping
                        max_color = max(max(color_map.keys()), int(result[idx].max().item())) + 1
                        map_tensor = torch.arange(max_color, device=self.device).float()
                        for old_color, new_color in color_map.items():
                            map_tensor[old_color] = new_color
                        result[idx] = self.vectorized_ops.map_colors(
                            result[idx].unsqueeze(0),
                            map_tensor
                        ).squeeze(0)

            elif op_name == "filter_color":
                # Get filter parameters
                for idx, op in op_list:
                    target_color = op.parameters.get("color", 0)
                    replacement = op.parameters.get("replacement", 0)
                    result[idx] = self.vectorized_ops.filter_by_color(
                        result[idx].unsqueeze(0),
                        target_color=target_color,
                        replacement=replacement
                    ).squeeze(0)

            # Add more operation dispatches as needed

        return result

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        # Calculate derived metrics
        if stats["total_programs_evaluated"] > 0:
            stats["avg_gpu_time_per_program_ms"] = (
                stats["gpu_time_ms"] / stats["total_programs_evaluated"]
            )

        # Add device info
        if self.device_capabilities:
            stats["device_info"] = {
                "name": self.device_capabilities.device_name,
                "total_memory_mb": self.device_capabilities.total_memory_mb,
                "compute_capability": self.device_capabilities.compute_capability
            }

        return stats

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "total_batches_evaluated": 0,
            "total_programs_evaluated": 0,
            "gpu_time_ms": 0.0,
            "cpu_time_ms": 0.0,
            "memory_peaks_mb": []
        }
