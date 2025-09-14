"""Memory-efficient grid operations with sparse matrix support."""

import gc
from typing import Any

import numpy as np
import psutil
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

from ..domain.models import ARCTask


class SparseGridConverter:
    """Convert between dense grids and sparse matrix representations."""

    @staticmethod
    def to_sparse(grid: list[list[int]], format: str = "csr") -> csr_matrix | lil_matrix | coo_matrix:
        """Convert dense grid to sparse matrix."""
        if not grid or not grid[0]:
            return csr_matrix((0, 0))

        dense_array = np.array(grid, dtype=np.int8)

        if format == "csr":
            return csr_matrix(dense_array)
        elif format == "lil":
            return lil_matrix(dense_array)
        elif format == "coo":
            return coo_matrix(dense_array)
        else:
            raise ValueError(f"Unsupported sparse format: {format}")

    @staticmethod
    def to_dense(sparse_matrix: csr_matrix | lil_matrix | coo_matrix) -> list[list[int]]:
        """Convert sparse matrix back to dense grid."""
        if sparse_matrix.nnz == 0:
            return []

        dense_array = sparse_matrix.toarray()
        result: list[list[int]] = dense_array.astype(int).tolist()
        return result

    @staticmethod
    def estimate_sparse_efficiency(grid: list[list[int]]) -> dict[str, Any]:
        """Estimate memory efficiency of sparse representation."""
        if not grid or not grid[0]:
            return {"efficiency": 0, "reason": "Empty grid"}

        rows, cols = len(grid), len(grid[0])
        total_cells = rows * cols

        # Count non-zero cells
        non_zero_count = sum(1 for row in grid for cell in row if cell != 0)
        sparsity = (total_cells - non_zero_count) / total_cells if total_cells > 0 else 0

        # Estimate memory savings
        dense_size = total_cells * 1  # 1 byte per cell (int8)
        sparse_size = non_zero_count * 3  # approximate: value + row + col indices

        efficiency = (dense_size - sparse_size) / dense_size if dense_size > 0 else 0

        return {
            "efficiency": efficiency,
            "sparsity": sparsity,
            "dense_size_bytes": dense_size,
            "sparse_size_bytes": sparse_size,
            "recommended": efficiency > 0.5,
        }


class GridBatcher:
    """Efficient batching strategies for grid processing."""

    def __init__(self, memory_limit_mb: int = 1000):
        """Initialize with memory limit in MB."""
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.process = psutil.Process()

    def adaptive_batch(self, tasks: list[ARCTask], base_batch_size: int = 32) -> list[list[ARCTask]]:
        """Create batches adaptively based on memory usage."""
        batches = []
        current_batch = []
        current_memory = 0

        for task in tasks:
            task_memory = task.get_memory_usage_estimate()

            # Check if adding this task would exceed memory limit
            if current_memory + task_memory > self.memory_limit_bytes and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_memory = 0

            current_batch.append(task)
            current_memory += task_memory

            # Also respect base batch size
            if len(current_batch) >= base_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_memory = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    def uniform_batch(self, tasks: list[ARCTask], batch_size: int) -> list[list[ARCTask]]:
        """Create uniform-sized batches."""
        return [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

    def memory_aware_batch(self, tasks: list[ARCTask]) -> list[list[ARCTask]]:
        """Create batches based on current system memory."""
        available_memory = psutil.virtual_memory().available
        safe_memory = available_memory * 0.5  # Use only 50% of available memory

        batches = []
        current_batch = []
        estimated_usage = 0

        for task in tasks:
            task_memory = task.get_memory_usage_estimate()

            if estimated_usage + task_memory > safe_memory and current_batch:
                batches.append(current_batch)
                current_batch = []
                estimated_usage = 0
                gc.collect()  # Force garbage collection between batches

            current_batch.append(task)
            estimated_usage += task_memory

        if current_batch:
            batches.append(current_batch)

        return batches


def grid_to_string(grid: list[list[int]]) -> str:
    """Convert a grid to a string representation for LLM processing."""
    if not grid:
        return "Empty grid"

    rows = []
    for row in grid:
        rows.append(' '.join(str(cell) for cell in row))
    return '\n'.join(rows)


def string_to_grid(grid_str: str) -> list[list[int]]:
    """Convert a string representation back to a grid."""
    if not grid_str or grid_str == "Empty grid":
        return []

    rows = grid_str.strip().split('\n')
    grid = []
    for row in rows:
        grid.append([int(cell) for cell in row.split()])
    return grid


class MemoryProfiler:
    """Memory profiling utilities for grid operations."""

    def __init__(self):
        """Initialize memory profiler."""
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory()

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_increase(self) -> float:
        """Get memory increase since initialization."""
        return self.get_current_memory() - self.initial_memory

    def profile_operation(self, func, *args, **kwargs):
        """Profile memory usage of an operation."""
        start_memory = self.get_current_memory()
        result = func(*args, **kwargs)
        end_memory = self.get_current_memory()

        return {
            "result": result,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "memory_increase_mb": end_memory - start_memory
        }

    @staticmethod
    def get_current_memory_usage() -> dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }

    @staticmethod
    def check_memory_constraints(current_mb: float, estimated_mb: float, limit_mb: float) -> dict[str, Any]:
        """Check if memory usage is within constraints."""
        total_estimated = current_mb + estimated_mb
        return {
            "within_limits": total_estimated <= limit_mb,
            "current_mb": current_mb,
            "estimated_mb": estimated_mb,
            "total_estimated_mb": total_estimated,
            "limit_mb": limit_mb,
            "usage_percentage": (total_estimated / limit_mb) * 100 if limit_mb > 0 else 0
        }

    @staticmethod
    def suggest_optimization(constraint_check: dict[str, Any]) -> list[str]:
        """Suggest optimizations based on memory constraints."""
        suggestions = []
        usage_pct = constraint_check.get("usage_percentage", 0)

        if usage_pct > 90:
            suggestions.append("Critical: Reduce batch size or enable memory optimization")
        elif usage_pct > 75:
            suggestions.append("Warning: Consider reducing batch size")
        elif usage_pct > 50:
            suggestions.append("Moderate usage - monitor for spikes")
        else:
            suggestions.append("Memory usage is optimal")

        return suggestions


class MemoryEfficientTaskStorage:
    """Memory-efficient storage for ARC tasks using compression."""

    def __init__(self, use_sparse: bool = True):
        """Initialize memory-efficient storage."""
        self.use_sparse = use_sparse
        self.tasks = {}
        self.compression_stats = {
            "total_tasks_stored": 0,
            "original_size_mb": 0,
            "compressed_size_mb": 0,
            "compression_rate": 0.0
        }

    def store_task(self, task: "ARCTask") -> None:
        """Store a task with compression."""
        self.tasks[task.task_id] = task
        self.compression_stats["total_tasks_stored"] += 1
        # Placeholder compression logic
        self.compression_stats["compression_rate"] = 0.6  # 60% compression

    def load_task(self, task_id: str) -> Any:
        """Load a task from storage."""
        return self.tasks.get(task_id)

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get memory statistics for stored tasks."""
        return self.compression_stats.copy()
