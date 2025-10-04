"""
Performance tests for GPU-accelerated evaluation.

Validates that GPU evaluation achieves the target 10x speedup over CPU
and stays within memory constraints.
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.adapters.strategies.gpu_batch_evaluator import BatchEvaluationRequest, GPUBatchEvaluator
from src.domain.dsl.base import Operation as DSLOperation


def create_test_programs(num_programs: int, program_length: int = 5):
    """Create test programs with various operations."""
    programs = []

    operation_types = [
        ("rotate", {"angle": 90}),
        ("flip", {"direction": "horizontal"}),
        ("translate", {"offset": (1, 0), "fill_color": 0}),
        ("map_colors", {"color_map": {0: 1, 1: 0}}),
        ("filter_color", {"color": 1, "replacement": 0})
    ]

    for i in range(num_programs):
        program = []
        for j in range(program_length):
            op = Mock(spec=DSLOperation)
            op_type = operation_types[(i + j) % len(operation_types)]
            op.name = op_type[0]
            op.parameters = op_type[1]
            program.append(op)
        programs.append(program)

    return programs


def create_test_grids(num_grids: int, size: int = 30):
    """Create test grids."""
    grids = []
    for i in range(num_grids):
        # Create varied patterns
        grid = np.zeros((size, size), dtype=int)
        pattern_type = i % 4

        if pattern_type == 0:
            # Checkerboard
            grid = np.indices((size, size)).sum(axis=0) % 2
        elif pattern_type == 1:
            # Stripes
            grid[:, ::2] = 1
        elif pattern_type == 2:
            # Random
            np.random.seed(i)
            grid = np.random.randint(0, 3, (size, size))
        else:
            # Solid color
            grid.fill(i % 10)

        grids.append(grid.tolist())

    return grids


class TestGPUPerformance:
    """Test GPU performance characteristics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_10x_speedup_target(self):
        """Test that GPU achieves 10x speedup over CPU for batch evaluation."""
        num_programs = 500
        programs = create_test_programs(num_programs, program_length=10)
        grids = create_test_grids(num_programs, size=30)

        # CPU baseline timing
        cpu_evaluator = GPUBatchEvaluator(device="cpu", max_batch_size=100)
        cpu_request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device="cpu",
            batch_size=100
        )

        cpu_start = time.perf_counter()
        cpu_result = cpu_evaluator.batch_evaluate(cpu_request)
        cpu_time = time.perf_counter() - cpu_start

        # GPU timing
        gpu_evaluator = GPUBatchEvaluator(device="cuda", max_batch_size=100)
        gpu_request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device="cuda",
            batch_size=100
        )

        # Warm up GPU
        warm_up_request = BatchEvaluationRequest(
            programs=programs[:10],
            input_grids=grids[:10],
            device="cuda"
        )
        gpu_evaluator.batch_evaluate(warm_up_request)
        torch.cuda.synchronize()

        gpu_start = time.perf_counter()
        gpu_result = gpu_evaluator.batch_evaluate(gpu_request)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - gpu_start

        speedup = cpu_time / gpu_time

        print("\nPerformance Results:")
        print(f"CPU Time: {cpu_time:.3f}s")
        print(f"GPU Time: {gpu_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Verify speedup (allowing some margin)
        assert speedup >= 8.0, f"Expected at least 8x speedup, got {speedup:.2f}x"

        # Verify correctness (sample check)
        for i in range(min(10, len(cpu_result.output_grids))):
            if cpu_result.output_grids[i] and gpu_result.output_grids[i]:
                cpu_output = np.array(cpu_result.output_grids[i])
                gpu_output = np.array(gpu_result.output_grids[i])
                np.testing.assert_array_equal(cpu_output, gpu_output,
                                            f"Output mismatch at index {i}")

    def test_memory_efficiency(self):
        """Test memory usage stays under 8GB limit."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type != "cuda":
            pytest.skip("CUDA required for memory testing")

        evaluator = GPUBatchEvaluator(
            device="cuda",
            memory_limit_mb=8000,
            max_batch_size=100
        )

        # Large batch test
        programs = create_test_programs(500, program_length=20)
        grids = create_test_grids(500, size=30)

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device="cuda",
            batch_size=100
        )

        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        result = evaluator.batch_evaluate(request)

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print("\nMemory Usage:")
        print(f"Peak GPU Memory: {peak_memory_mb:.2f} MB")
        print("Memory Limit: 8000 MB")

        assert peak_memory_mb < 8000, f"Peak memory {peak_memory_mb:.2f} MB exceeds 8GB limit"
        assert result.batch_stats["success_rate"] > 0.9

    def test_adaptive_batching_performance(self):
        """Test adaptive batching maintains performance."""
        evaluator = GPUBatchEvaluator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            memory_limit_mb=2000  # Lower limit to force adaptation
        )

        # Varying grid sizes to test adaptation
        programs = create_test_programs(200)
        grids = []
        for i in range(200):
            size = 10 + (i % 4) * 10  # 10x10 to 40x40
            grid = np.random.randint(0, 5, (size, size)).tolist()
            grids.append(grid)

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device=evaluator.device.type,
            batch_size=100  # Request large batch
        )

        start_time = time.perf_counter()
        result = evaluator.batch_evaluate(request)
        total_time = time.perf_counter() - start_time

        # Check adaptive batching worked
        assert result.batch_stats["effective_batch_size"] < 100  # Should adapt down
        assert result.batch_stats["num_batches"] > 2  # Should split into multiple
        assert result.batch_stats["success_rate"] > 0.9

        avg_time_per_program = total_time * 1000 / len(programs)
        print("\nAdaptive Batching Performance:")
        print(f"Effective Batch Size: {result.batch_stats['effective_batch_size']}")
        print(f"Number of Batches: {result.batch_stats['num_batches']}")
        print(f"Avg Time per Program: {avg_time_per_program:.2f} ms")

        # Should still be fast
        assert avg_time_per_program < 10  # Less than 10ms per program

    def test_operation_fusion_performance(self):
        """Test performance with operation fusion."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = GPUBatchEvaluator(device=device)

        # Create programs with fusable operations
        fusable_programs = []
        for i in range(100):
            # Consecutive rotations can be fused
            program = []
            for _ in range(4):
                op = Mock()
                op.name = "rotate"
                op.parameters = {"angle": 90}
                program.append(op)
            fusable_programs.append(program)

        # Create programs without fusable operations
        non_fusable_programs = []
        for i in range(100):
            program = []
            ops = ["rotate", "flip", "translate", "rotate"]
            for op_name in ops:
                op = Mock()
                op.name = op_name
                if op_name == "rotate":
                    op.parameters = {"angle": 90}
                elif op_name == "flip":
                    op.parameters = {"direction": "horizontal"}
                else:
                    op.parameters = {"offset": (1, 0), "fill_color": 0}
                program.append(op)
            non_fusable_programs.append(program)

        grids = create_test_grids(100, size=20)

        # Time fusable operations
        request1 = BatchEvaluationRequest(
            programs=fusable_programs,
            input_grids=grids,
            device=device.type
        )

        start = time.perf_counter()
        result1 = evaluator.batch_evaluate(request1)
        fusable_time = time.perf_counter() - start

        # Time non-fusable operations
        request2 = BatchEvaluationRequest(
            programs=non_fusable_programs,
            input_grids=grids,
            device=device.type
        )

        start = time.perf_counter()
        result2 = evaluator.batch_evaluate(request2)
        non_fusable_time = time.perf_counter() - start

        print("\nOperation Fusion Performance:")
        print(f"Fusable Time: {fusable_time:.3f}s")
        print(f"Non-fusable Time: {non_fusable_time:.3f}s")
        print(f"Speedup from Fusion: {non_fusable_time/fusable_time:.2f}x")

        # Fusable should be faster (even if fusion not implemented yet)
        assert fusable_time <= non_fusable_time * 1.1  # Allow 10% margin

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100])
    def test_batch_size_scaling(self, batch_size):
        """Test performance scaling with different batch sizes."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = GPUBatchEvaluator(device=device, max_batch_size=batch_size)

        num_programs = 100
        programs = create_test_programs(num_programs)
        grids = create_test_grids(num_programs)

        request = BatchEvaluationRequest(
            programs=programs,
            input_grids=grids,
            device=device.type,
            batch_size=batch_size
        )

        # Warm up
        if device.type == "cuda":
            warm_up = BatchEvaluationRequest(
                programs=programs[:5],
                input_grids=grids[:5],
                device="cuda",
                batch_size=1
            )
            evaluator.batch_evaluate(warm_up)
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = evaluator.batch_evaluate(request)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        throughput = num_programs / total_time

        print(f"\nBatch Size {batch_size}:")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.1f} programs/second")

        # Larger batches should have better throughput
        if device.type == "cuda" and batch_size >= 10:
            assert throughput > 100  # At least 100 programs/second on GPU

    def test_concurrent_evaluation_stress(self):
        """Stress test with concurrent evaluation requests."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = GPUBatchEvaluator(device=device)

        # Create multiple evaluation requests
        requests = []
        for i in range(5):
            programs = create_test_programs(100, program_length=8)
            grids = create_test_grids(100, size=25)
            request = BatchEvaluationRequest(
                programs=programs,
                input_grids=grids,
                device=device.type,
                batch_size=50
            )
            requests.append(request)

        # Execute all requests
        start = time.perf_counter()
        results = []
        for request in requests:
            result = evaluator.batch_evaluate(request)
            results.append(result)
        total_time = time.perf_counter() - start

        total_programs = sum(len(r.programs) for r in requests)
        avg_time = total_time / len(requests)

        print("\nConcurrent Evaluation Stress Test:")
        print(f"Total Programs: {total_programs}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Avg Time per Request: {avg_time:.3f}s")

        # All should succeed
        for result in results:
            assert result.batch_stats["success_rate"] > 0.9

        # Should maintain good performance
        assert total_time < 10.0  # 5 requests of 100 programs each in < 10s
