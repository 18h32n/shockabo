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

        # Count non-zero (non-background) cells
        non_zero_cells = sum(1 for row in grid for cell in row if cell != 0)
        sparsity = 1.0 - (non_zero_cells / total_cells)

        # Memory estimates (bytes)
        dense_memory = total_cells * 4  # 4 bytes per int32
        sparse_memory = non_zero_cells * 12  # Approx. 12 bytes per non-zero (value + indices)

        efficiency_ratio = sparse_memory / dense_memory if dense_memory > 0 else 1.0

        return {
            "sparsity": sparsity,
            "non_zero_cells": non_zero_cells,
            "total_cells": total_cells,
            "dense_memory_bytes": dense_memory,
            "sparse_memory_bytes": sparse_memory,
            "efficiency_ratio": efficiency_ratio,
            "recommended": efficiency_ratio < 0.5  # Recommend sparse if <50% of dense size
        }


class MemoryEfficientTaskStorage:
    """Memory-efficient storage for ARC tasks with lazy loading."""

    def __init__(self, use_sparse: bool = True, compression_threshold: float = 0.3):
        """Initialize memory-efficient task storage."""
        self.use_sparse = use_sparse
        self.compression_threshold = compression_threshold  # Sparsity threshold for compression
        self.compressed_tasks: dict[str, dict[str, Any]] = {}
        self.memory_stats = {"compressed": 0, "uncompressed": 0, "total_saved": 0}

    def store_task(self, task: ARCTask) -> dict[str, Any]:
        """Store task with optimal memory representation."""
        task_data: dict[str, Any] = {
            "task_id": task.task_id,
            "task_source": task.task_source,
            "difficulty_level": task.difficulty_level,
            "metadata": task.metadata,
            "created_at": task.created_at,
            "train_examples": [],
            "test_input": None,
            "test_output": None,
            "compression_info": {}
        }

        # Process training examples
        for _i, example in enumerate(task.train_examples):
            input_grid = example["input"]
            output_grid = example.get("output", [])

            # Analyze and compress input
            input_analysis = SparseGridConverter.estimate_sparse_efficiency(input_grid)
            if self.use_sparse and input_analysis["sparsity"] > self.compression_threshold:
                compressed_input = {
                    "data": SparseGridConverter.to_sparse(input_grid),
                    "format": "sparse",
                    "original_shape": (len(input_grid), len(input_grid[0]) if input_grid else 0)
                }
                self.memory_stats["compressed"] += 1
            else:
                compressed_input = {"data": input_grid, "format": "dense"}
                self.memory_stats["uncompressed"] += 1

            # Analyze and compress output
            compressed_output = None
            if output_grid:
                output_analysis = SparseGridConverter.estimate_sparse_efficiency(output_grid)
                if self.use_sparse and output_analysis["sparsity"] > self.compression_threshold:
                    compressed_output = {
                        "data": SparseGridConverter.to_sparse(output_grid),
                        "format": "sparse",
                        "original_shape": (len(output_grid), len(output_grid[0]) if output_grid else 0)
                    }
                    self.memory_stats["compressed"] += 1
                else:
                    compressed_output = {"data": output_grid, "format": "dense"}
                    self.memory_stats["uncompressed"] += 1

            task_data["train_examples"].append({
                "input": compressed_input,
                "output": compressed_output
            })

        # Process test data
        if task.test_input:
            test_input_analysis = SparseGridConverter.estimate_sparse_efficiency(task.test_input)
            if self.use_sparse and test_input_analysis["sparsity"] > self.compression_threshold:
                task_data["test_input"] = {
                    "data": SparseGridConverter.to_sparse(task.test_input),
                    "format": "sparse",
                    "original_shape": (len(task.test_input), len(task.test_input[0]) if task.test_input else 0)
                }
                self.memory_stats["compressed"] += 1
            else:
                task_data["test_input"] = {"data": task.test_input, "format": "dense"}
                self.memory_stats["uncompressed"] += 1

        if task.test_output:
            test_output_analysis = SparseGridConverter.estimate_sparse_efficiency(task.test_output)
            if self.use_sparse and test_output_analysis["sparsity"] > self.compression_threshold:
                task_data["test_output"] = {
                    "data": SparseGridConverter.to_sparse(task.test_output),
                    "format": "sparse",
                    "original_shape": (len(task.test_output), len(task.test_output[0]) if task.test_output else 0)
                }
                self.memory_stats["compressed"] += 1
            else:
                task_data["test_output"] = {"data": task.test_output, "format": "dense"}
                self.memory_stats["uncompressed"] += 1

        self.compressed_tasks[task.task_id] = task_data
        return task_data

    def load_task(self, task_id: str) -> ARCTask | None:
        """Load task from compressed storage."""
        if task_id not in self.compressed_tasks:
            return None

        task_data = self.compressed_tasks[task_id]

        # Reconstruct training examples
        train_examples = []
        for example_data in task_data["train_examples"]:
            input_data = example_data["input"]
            output_data = example_data.get("output")

            # Decompress input
            if input_data["format"] == "sparse":
                input_grid = SparseGridConverter.to_dense(input_data["data"])
            else:
                input_grid = input_data["data"]

            example = {"input": input_grid}

            # Decompress output if present
            if output_data:
                if output_data["format"] == "sparse":
                    output_grid = SparseGridConverter.to_dense(output_data["data"])
                else:
                    output_grid = output_data["data"]
                example["output"] = output_grid

            train_examples.append(example)

        # Reconstruct test data
        test_input = []
        if task_data["test_input"]:
            if task_data["test_input"]["format"] == "sparse":
                test_input = SparseGridConverter.to_dense(task_data["test_input"]["data"])
            else:
                test_input = task_data["test_input"]["data"]

        test_output = None
        if task_data["test_output"]:
            if task_data["test_output"]["format"] == "sparse":
                test_output = SparseGridConverter.to_dense(task_data["test_output"]["data"])
            else:
                test_output = task_data["test_output"]["data"]

        return ARCTask(
            task_id=task_data["task_id"],
            task_source=task_data["task_source"],
            difficulty_level=task_data["difficulty_level"],
            train_examples=train_examples,
            test_input=test_input,
            test_output=test_output,
            metadata=task_data["metadata"],
            created_at=task_data["created_at"]
        )

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        total_grids = self.memory_stats["compressed"] + self.memory_stats["uncompressed"]
        compression_rate = (self.memory_stats["compressed"] / total_grids) if total_grids > 0 else 0.0

        return {
            "total_tasks_stored": len(self.compressed_tasks),
            "total_grids_processed": total_grids,
            "compressed_grids": self.memory_stats["compressed"],
            "uncompressed_grids": self.memory_stats["uncompressed"],
            "compression_rate": compression_rate
        }


class MemoryProfiler:
    """Memory profiling utilities for data pipeline."""

    @staticmethod
    def get_current_memory_usage() -> dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": memory_percent,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }

    @staticmethod
    def estimate_task_memory_usage(tasks: dict[str, ARCTask]) -> dict[str, Any]:
        """Estimate memory usage for task collection."""
        total_memory = 0
        task_details = {}

        for task_id, task in tasks.items():
            task_memory = task.get_memory_usage_estimate()
            total_memory += task_memory
            task_details[task_id] = task_memory

        return {
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / 1024 / 1024,
            "total_memory_gb": total_memory / 1024 / 1024 / 1024,
            "average_task_memory_bytes": total_memory / len(tasks) if tasks else 0,
            "max_task_memory": max(task_details.values()) if task_details else 0,
            "min_task_memory": min(task_details.values()) if task_details else 0,
            "task_details": task_details
        }

    @staticmethod
    def check_memory_constraints(
        current_usage_mb: float,
        estimated_additional_mb: float,
        limit_mb: float = 4096
    ) -> dict[str, Any]:
        """Check if memory constraints will be violated."""
        projected_usage = current_usage_mb + estimated_additional_mb

        return {
            "current_usage_mb": current_usage_mb,
            "estimated_additional_mb": estimated_additional_mb,
            "projected_usage_mb": projected_usage,
            "limit_mb": limit_mb,
            "within_limits": projected_usage < limit_mb,
            "headroom_mb": limit_mb - projected_usage,
            "utilization_percent": (projected_usage / limit_mb) * 100
        }

    @staticmethod
    def suggest_optimization(memory_analysis: dict[str, Any]) -> list[str]:
        """Suggest memory optimization strategies."""
        suggestions = []

        if memory_analysis["utilization_percent"] > 80:
            suggestions.append("Memory usage > 80%, consider enabling sparse matrix compression")

        if memory_analysis["utilization_percent"] > 90:
            suggestions.append("Memory usage > 90%, implement batch processing with smaller batch sizes")
            suggestions.append("Consider lazy loading and task streaming")

        if memory_analysis["headroom_mb"] < 500:
            suggestions.append("Low memory headroom, enable aggressive compression")
            suggestions.append("Implement garbage collection between batches")

        return suggestions


class LazyTaskLoader:
    """Lazy loading implementation for large datasets."""

    def __init__(self, repository, max_memory_mb: float = 2048):
        """Initialize lazy loader with memory constraints."""
        self.repository = repository
        self.max_memory_mb = max_memory_mb
        self.loaded_tasks: dict[str, ARCTask] = {}
        self.access_order: list[str] = []  # For LRU eviction

    def get_task(self, task_id: str, task_source: str = "training") -> ARCTask | None:
        """Get task with lazy loading and memory management."""
        # Check if already loaded
        if task_id in self.loaded_tasks:
            # Update access order (move to end)
            self.access_order.remove(task_id)
            self.access_order.append(task_id)
            return self.loaded_tasks[task_id]

        # Check memory before loading
        current_memory = MemoryProfiler.get_current_memory_usage()["rss_mb"]

        if current_memory > self.max_memory_mb:
            self._evict_least_recently_used()

        # Load task
        task: ARCTask | None = self.repository.load_task(task_id, task_source)
        if task:
            self.loaded_tasks[task_id] = task
            self.access_order.append(task_id)

        return task

    def _evict_least_recently_used(self):
        """Evict least recently used tasks to free memory."""
        while self.access_order and len(self.loaded_tasks) > 0:
            lru_task_id = self.access_order.pop(0)
            if lru_task_id in self.loaded_tasks:
                del self.loaded_tasks[lru_task_id]
                gc.collect()  # Force garbage collection
                break

    def clear_cache(self):
        """Clear all cached tasks."""
        self.loaded_tasks.clear()
        self.access_order.clear()
        gc.collect()

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_tasks": len(self.loaded_tasks),
            "memory_usage_mb": MemoryProfiler.get_current_memory_usage()["rss_mb"],
            "memory_limit_mb": self.max_memory_mb,
            "most_recently_accessed": self.access_order[-5:] if self.access_order else []
        }
