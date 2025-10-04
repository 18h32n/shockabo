"""
Parallel evaluation system for genetic algorithm.

This module implements efficient parallel fitness evaluation using a hybrid
approach combining multiprocessing for CPU-bound tasks and asyncio for coordination.
"""

from __future__ import annotations

import asyncio
import builtins
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
    TorchTensor = torch.Tensor
    TorchDevice = torch.device
except ImportError:
    HAS_TORCH = False
    # Fallback types for when torch is not available
    TorchTensor = Any
    TorchDevice = Any

from src.adapters.strategies.evolution_engine import Individual
from src.domain.dsl.types import Grid
from src.domain.models import ARCTask


@dataclass
class EvaluationTask:
    """Represents a single evaluation task."""
    individual_id: str
    operations: list[dict[str, Any]]
    train_examples: list[dict[str, list[list[int]]]]
    task_id: str


@dataclass
class EvaluationResult:
    """Result of evaluating an individual."""
    individual_id: str
    fitness: float
    execution_time: float
    cached_outputs: dict[str, Grid] | None = None
    error: str | None = None


def evaluate_individual_worker(eval_task: EvaluationTask) -> EvaluationResult:
    """
    Worker function for evaluating a single individual.

    This runs in a separate process for true parallelism.
    """
    start_time = time.time()

    try:
        # Import here to avoid pickling issues
        from src.domain.services.dsl_engine import DSLEngine

        # Create DSL engine instance
        dsl_engine = DSLEngine()

        # Deserialize operations
        operations = []
        for op_dict in eval_task.operations:
            # This is simplified - real implementation would use proper deserialization
            operations.append(op_dict)

        # Evaluate on training examples
        total_similarity = 0.0
        cached_outputs = {}

        for i, example in enumerate(eval_task.train_examples):
            input_grid = example['input']
            expected_output = example['output']

            # Execute program
            result = dsl_engine.execute_program(operations, input_grid)

            if result['success']:
                output_grid = result['output']
                cached_outputs[f'train_{i}'] = output_grid

                # Calculate similarity
                similarity = calculate_grid_similarity(output_grid, expected_output)
                total_similarity += similarity
            else:
                # Execution failed
                continue

        # Calculate fitness
        avg_similarity = total_similarity / len(eval_task.train_examples) if eval_task.train_examples else 0.0

        # Add other fitness components
        program_length = len(eval_task.operations)
        execution_time = time.time() - start_time

        # Simple fitness calculation
        fitness = (
            0.7 * avg_similarity +
            0.2 * (1.0 / (1.0 + program_length / 10.0)) +
            0.1 * (1.0 / (1.0 + execution_time))
        )

        return EvaluationResult(
            individual_id=eval_task.individual_id,
            fitness=fitness,
            execution_time=execution_time,
            cached_outputs=cached_outputs
        )

    except Exception as e:
        return EvaluationResult(
            individual_id=eval_task.individual_id,
            fitness=0.0,
            execution_time=time.time() - start_time,
            error=str(e)
        )


def calculate_grid_similarity(output: Grid, expected: Grid) -> float:
    """Calculate similarity between two grids."""
    output_array = np.array(output)
    expected_array = np.array(expected)

    # Handle dimension mismatch
    if output_array.shape != expected_array.shape:
        return 0.0

    # Calculate exact match ratio
    matches = (output_array == expected_array).sum()
    total = output_array.size

    return matches / total if total > 0 else 0.0


class ParallelEvaluator:
    """
    Manages parallel evaluation of genetic algorithm populations.

    Uses a hybrid approach:
    - ProcessPoolExecutor for CPU-bound fitness evaluation
    - asyncio for coordinating tasks and handling results
    - Batching for efficient resource utilization
    """

    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 250,
        timeout_per_individual: float = 1.0,
        memory_limit_mb: int = 2048
    ):
        """
        Initialize parallel evaluator.

        Args:
            num_workers: Number of worker processes
            batch_size: Number of individuals per batch
            timeout_per_individual: Timeout in seconds per individual
            memory_limit_mb: Memory limit in MB
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.timeout_per_individual = timeout_per_individual
        self.memory_limit_mb = memory_limit_mb
        self.executor: ProcessPoolExecutor | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.executor:
            self.executor.shutdown(wait=True)

    async def evaluate_population(
        self,
        individuals: list[Individual],
        task: ARCTask,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate a population of individuals in parallel.

        Args:
            individuals: List of individuals to evaluate
            task: ARC task to evaluate against
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping individual IDs to evaluation results
        """
        if not self.executor:
            raise RuntimeError("ParallelEvaluator must be used as async context manager")

        results = {}

        # Create evaluation tasks
        eval_tasks = []
        for individual in individuals:
            # Serialize operations for pickling
            serialized_ops = []
            for op in individual.operations:
                serialized_ops.append({
                    "name": op.get_name(),
                    "parameters": op.parameters
                })

            eval_task = EvaluationTask(
                individual_id=individual.id,
                operations=serialized_ops,
                train_examples=task.train_examples,
                task_id=task.task_id
            )
            eval_tasks.append(eval_task)

        # Process in batches
        for batch_start in range(0, len(eval_tasks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(eval_tasks))
            batch = eval_tasks[batch_start:batch_end]

            # Submit batch to executor
            batch_results = await self._evaluate_batch(batch)

            # Update results
            for result in batch_results:
                results[result.individual_id] = result

            # Progress callback
            if progress_callback:
                progress_callback(batch_end, len(eval_tasks))

        return results

    async def _evaluate_batch(self, batch: list[EvaluationTask]) -> list[EvaluationResult]:
        """Evaluate a batch of individuals."""
        loop = asyncio.get_event_loop()

        # Submit all tasks
        futures = []
        for eval_task in batch:
            future = loop.run_in_executor(
                self.executor,
                evaluate_individual_worker,
                eval_task
            )
            futures.append(future)

        # Wait for completion with timeout
        timeout = len(batch) * self.timeout_per_individual

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout
            )

            # Process results
            batch_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle evaluation error
                    batch_results.append(
                        EvaluationResult(
                            individual_id=batch[i].individual_id,
                            fitness=0.0,
                            execution_time=self.timeout_per_individual,
                            error=str(result)
                        )
                    )
                else:
                    batch_results.append(result)

            return batch_results

        except builtins.TimeoutError:
            # Handle timeout - return partial results
            batch_results = []
            for eval_task in batch:
                batch_results.append(
                    EvaluationResult(
                        individual_id=eval_task.individual_id,
                        fitness=0.0,
                        execution_time=self.timeout_per_individual,
                        error="Evaluation timeout"
                    )
                )
            return batch_results


class GPUAcceleratedEvaluator(ParallelEvaluator):
    """
    GPU-accelerated evaluator for batch operations.

    Uses PyTorch for vectorized grid operations when available.
    """

    def __init__(self, *args, gpu_batch_size: int = 100, **kwargs):
        """
        Initialize GPU-accelerated evaluator.

        Args:
            gpu_batch_size: Batch size for GPU operations
            *args, **kwargs: Arguments for ParallelEvaluator
        """
        super().__init__(*args, **kwargs)
        self.gpu_batch_size = gpu_batch_size
        self.gpu_available = self._check_gpu_availability()

        # Platform-specific GPU optimizations (Task 8.2)
        self._configure_platform_gpu_settings()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _configure_platform_gpu_settings(self) -> None:
        """Configure GPU settings based on platform (Task 8.2)."""
        if not self.gpu_available:
            return

        try:
            import torch

            from src.infrastructure.components.platform_detector import (
                Platform,
                get_platform_detector,
            )

            # Detect platform
            detector = get_platform_detector()
            platform_info = detector.detect_platform()
            current_platform = platform_info.platform

            # Platform-specific GPU configurations
            if current_platform == Platform.COLAB:
                # Colab-specific optimizations
                torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
                self.gpu_batch_size = 200  # Larger batches for Colab GPU
                if platform_info.gpu_count > 0:
                    # Set memory fraction to avoid OOM on Colab
                    torch.cuda.set_per_process_memory_fraction(0.9)
                    print(f"Configured Colab GPU: batch_size={self.gpu_batch_size}")

            elif current_platform == Platform.KAGGLE:
                # Kaggle-specific optimizations
                torch.backends.cudnn.benchmark = True
                self.gpu_batch_size = 150  # Moderate batch size for Kaggle
                if platform_info.gpu_count > 0:
                    torch.cuda.set_per_process_memory_fraction(0.85)
                    print(f"Configured Kaggle GPU: batch_size={self.gpu_batch_size}")

            elif current_platform == Platform.PAPERSPACE:
                # Paperspace-specific optimizations
                self.gpu_batch_size = 50  # Conservative for free tier
                if platform_info.gpu_count > 0:
                    torch.cuda.set_per_process_memory_fraction(0.7)
                    print(f"Configured Paperspace GPU: batch_size={self.gpu_batch_size}")

            # Store platform info
            self.current_platform = current_platform
            self.platform_info = platform_info

        except Exception as e:
            print(f"Warning: Failed to configure platform GPU settings: {e}")
            # Continue with default settings

    async def evaluate_population(
        self,
        individuals: list[Individual],
        task: ARCTask,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate population with GPU acceleration where possible.

        Falls back to CPU evaluation for complex operations.
        """
        if not self.gpu_available:
            # Fall back to CPU evaluation
            return await super().evaluate_population(individuals, task, progress_callback)

        # Separate individuals by operation types
        simple_ops = []  # Can be GPU accelerated
        complex_ops = []  # Must use CPU

        for individual in individuals:
            if self._can_gpu_accelerate(individual):
                simple_ops.append(individual)
            else:
                complex_ops.append(individual)

        results = {}

        # GPU batch evaluation for simple operations
        if simple_ops:
            gpu_results = await self._gpu_evaluate_batch(simple_ops, task)
            results.update(gpu_results)

        # CPU evaluation for complex operations
        if complex_ops:
            cpu_results = await super().evaluate_population(complex_ops, task, progress_callback)
            results.update(cpu_results)

        return results

    def _can_gpu_accelerate(self, individual: Individual) -> bool:
        """Check if individual's operations can be GPU accelerated."""
        # Simple heuristic - only basic transformations
        gpu_accelerated_ops = {
            'rotate', 'flip', 'translate', 'scale',
            'replace_color', 'fill_background'
        }

        for op in individual.operations:
            if op.get_name() not in gpu_accelerated_ops:
                return False

        return True

    async def _gpu_evaluate_batch(
        self,
        individuals: list[Individual],
        task: ARCTask
    ) -> dict[str, EvaluationResult]:
        """Evaluate batch of individuals on GPU (Task 8.2 enhanced)."""
        try:
            import torch

            # Process in GPU-sized batches
            results = {}

            for batch_start in range(0, len(individuals), self.gpu_batch_size):
                batch_end = min(batch_start + self.gpu_batch_size, len(individuals))
                batch = individuals[batch_start:batch_end]

                # Prepare batch data for GPU
                batch_results = await self._evaluate_gpu_batch_torch(batch, task)
                results.update(batch_results)

                # Clear GPU cache periodically
                if batch_end % (self.gpu_batch_size * 5) == 0:
                    torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"GPU evaluation failed: {e}, falling back to CPU")
            # Fall back to CPU evaluation
            return await super().evaluate_population(individuals, task)

    async def _evaluate_gpu_batch_torch(
        self,
        batch: list[Individual],
        task: ARCTask
    ) -> dict[str, EvaluationResult]:
        """Evaluate a single GPU batch using PyTorch."""
        import torch

        results = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert grids to tensors for batch processing
        for individual in batch:
            start_time = time.time()
            try:
                # Initialize fitness
                total_fitness = 0.0
                cached_outputs = {}

                # Evaluate on each training example
                for i, example in enumerate(task.train):
                    input_grid = torch.tensor(example['input'], dtype=torch.int32, device=device)
                    expected_output = torch.tensor(example['output'], dtype=torch.int32, device=device)

                    # Apply operations on GPU
                    output_grid = self._apply_operations_gpu(
                        input_grid, individual.operations, device
                    )

                    # Calculate similarity on GPU
                    similarity = self._calculate_similarity_gpu(
                        output_grid, expected_output
                    )

                    total_fitness += similarity
                    cached_outputs[f"train_{i}"] = output_grid.cpu().numpy().tolist()

                # Average fitness
                fitness = total_fitness / len(task.train)

                results[individual.id] = EvaluationResult(
                    individual_id=individual.id,
                    fitness=float(fitness),
                    execution_time=time.time() - start_time,
                    cached_outputs=cached_outputs
                )

            except Exception as e:
                results[individual.id] = EvaluationResult(
                    individual_id=individual.id,
                    fitness=0.0,
                    execution_time=time.time() - start_time,
                    error=str(e)
                )

        return results

    def _apply_operations_gpu(
        self,
        grid: TorchTensor,
        operations: list,
        device: TorchDevice
    ) -> TorchTensor:
        """Apply operations to grid on GPU."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GPU operations")

        # This is a simplified implementation
        # Real implementation would translate DSL operations to GPU ops
        current_grid = grid.clone()

        for op in operations:
            op_name = op.get_name()

            if op_name == 'rotate':
                angle = op.parameters.get('angle', 90)
                rotations = angle // 90
                current_grid = torch.rot90(current_grid, k=rotations)

            elif op_name == 'flip':
                axis = op.parameters.get('axis', 'horizontal')
                if axis == 'horizontal':
                    current_grid = torch.flip(current_grid, dims=[0])
                else:
                    current_grid = torch.flip(current_grid, dims=[1])

            elif op_name == 'replace_color':
                source = op.parameters.get('source_color', 0)
                target = op.parameters.get('target_color', 1)
                current_grid = torch.where(
                    current_grid == source, target, current_grid
                )

            # Add more GPU-accelerated operations as needed

        return current_grid

    def _calculate_similarity_gpu(
        self,
        output: TorchTensor,
        expected: TorchTensor
    ) -> float:
        """Calculate grid similarity on GPU."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GPU operations")

        # Ensure same shape
        if output.shape != expected.shape:
            return 0.0

        # Calculate pixel-wise accuracy
        correct = (output == expected).sum().item()
        total = output.numel()

        return correct / total if total > 0 else 0.0
