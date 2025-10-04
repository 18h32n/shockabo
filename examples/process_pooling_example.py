"""
Example demonstrating process pooling in sandbox executor.

This example shows how to configure and use the process pool for improved
performance when executing multiple transpiled functions.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.strategies.sandbox_executor import (
    ProcessPoolConfig,
    SandboxConfig,
    SandboxExecutor,
)


def main():
    """Demonstrate process pooling functionality."""

    print("=== Process Pooling Example ===\n")

    # Configure process pool
    pool_config = ProcessPoolConfig(
        pool_size=3,  # 3 worker processes
        enabled=True,
        max_executions_per_worker=50,
        worker_timeout_seconds=5.0,
        idle_worker_timeout=60.0
    )

    # Configure sandbox with pooling
    config = SandboxConfig(
        timeout_seconds=2.0,
        memory_limit_mb=100,
        process_pool=pool_config
    )

    # Create executor with pooling
    executor = SandboxExecutor(config)

    try:
        # Wait for pool to initialize
        print("Initializing process pool...")
        time.sleep(1.0)

        # Show initial pool stats
        stats = executor.get_pool_stats()
        print(f"Pool initialized: {stats['total_workers']} workers ready")
        print(f"Pool enabled: {stats['pool_enabled']}")
        print()

        # Example code to execute
        sample_codes = [
            # Simple transformation
            """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    return np.array(grid).tolist()
""",

            # Rotation
            """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    return np.rot90(grid, k=1).tolist()
""",

            # Pattern detection
            """
import numpy as np
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    # Simple pattern: increment all values
    return (grid + 1).tolist()
""",

            # Mathematical operation
            """
import numpy as np
import math
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    # Apply mathematical transformation
    result = np.where(grid > 2, grid * 2, grid)
    return result.tolist()
""",

            # Collection operations
            """
import numpy as np
from collections import Counter
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid)
    flat = grid.flatten()
    counter = Counter(flat)
    most_common = counter.most_common(1)[0][0] if counter else 0
    return np.full_like(grid, most_common).tolist()
"""
        ]

        test_grids = [
            [[1, 2], [3, 4]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[5, 5], [5, 5]],
            [[1, 2, 3], [4, 5, 6]],
            [[2, 2, 1], [1, 2, 2], [2, 1, 1]]
        ]

        # Execute multiple tasks to demonstrate pooling
        print("Executing tasks with process pooling...")
        start_time = time.time()

        results = []
        for i, (code, grid) in enumerate(zip(sample_codes, test_grids, strict=False)):
            print(f"  Task {i+1}: Executing...")

            result = executor.execute(code, "solve_task", grid)
            results.append(result)

            if result.success:
                print(f"    Success! Result: {result.result}")
                print(f"    Used pooled worker: {result.used_pooled_worker}")
                print(f"    Execution time: {result.metrics.execution_time_ms:.1f}ms")
            else:
                print(f"    Failed: {result.error}")

            # Show pool stats periodically
            if i % 2 == 1:
                stats = executor.get_pool_stats()
                print(f"    Pool stats: {stats['idle_workers']} idle, "
                      f"{stats['busy_workers']} busy, "
                      f"{stats['total_workers']} total workers")
            print()

        total_time = time.time() - start_time
        print(f"All tasks completed in {total_time:.2f} seconds")

        # Show final statistics
        print("\n=== Final Statistics ===")
        stats = executor.get_pool_stats()
        print(f"Total workers: {stats['total_workers']}")
        print(f"Idle workers: {stats['idle_workers']}")
        print(f"Busy workers: {stats['busy_workers']}")
        print(f"Dead workers: {stats['dead_workers']}")
        print(f"Pending requests: {stats['pending_requests']}")

        successful_executions = sum(1 for r in results if r.success)
        pooled_executions = sum(1 for r in results if r.used_pooled_worker)

        print("\nExecution Summary:")
        print(f"  Successful: {successful_executions}/{len(results)}")
        print(f"  Used pooled workers: {pooled_executions}/{len(results)}")

        if stats['total_workers'] > 0:
            print("\nWorker Details:")
            for worker_id, details in stats['worker_details'].items():
                print(f"  Worker {worker_id[:8]}: "
                      f"state={details['state']}, "
                      f"executions={details['execution_count']}, "
                      f"pid={details['pid']}")

        # Demonstrate concurrent execution
        print("\n=== Concurrent Execution Test ===")

        def execute_concurrent_task(task_id: int) -> str:
            code = f"""
import numpy as np
import time
from typing import List

def solve_task(grid: List[List[int]]) -> List[List[int]]:
    # Brief sleep to simulate work and enable concurrency
    time.sleep(0.1)
    return [[{task_id}, {task_id}], [{task_id}, {task_id}]]
"""
            result = executor.execute(code, "solve_task", [[0, 0], [0, 0]])
            return f"Task {task_id}: {'Success' if result.success else 'Failed'} " \
                   f"(pooled: {result.used_pooled_worker})"

        # Execute tasks concurrently using threading
        import threading

        print("Executing 4 tasks concurrently...")
        concurrent_start = time.time()

        threads = []
        concurrent_results = {}

        for i in range(4):
            def run_task(task_id=i):
                concurrent_results[task_id] = execute_concurrent_task(task_id)

            thread = threading.Thread(target=run_task)
            threads.append(thread)
            thread.start()

        # Wait for all tasks to complete
        for thread in threads:
            thread.join()

        concurrent_time = time.time() - concurrent_start
        print(f"Concurrent tasks completed in {concurrent_time:.2f} seconds")

        for i in range(4):
            print(f"  {concurrent_results[i]}")

        # Show efficiency gain
        expected_sequential_time = 4 * 0.1  # 4 tasks × 0.1s each
        efficiency = expected_sequential_time / concurrent_time if concurrent_time > 0 else 1
        print(f"\nConcurrency efficiency: {efficiency:.1f}x speedup")

    except Exception as e:
        print(f"Error during execution: {e}")

    finally:
        # Clean up
        print("\nCleaning up process pool...")
        executor._cleanup_all_processes()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
