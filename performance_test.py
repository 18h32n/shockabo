#!/usr/bin/env python3
"""
Performance test for ARC data loading with real data
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adapters.repositories.arc_data_repository import ARCDataRepository
from adapters.repositories.cache_repository import CacheRepository


def test_performance():
    """Test current performance with real ARC data"""

    data_path = Path("data/tasks")  # Use the correct path that has training subdirectory
    cache_dir = Path("data/cache_test")
    cache_dir.mkdir(exist_ok=True)

    # Initialize repositories
    cache_repo = CacheRepository(
        cache_dir=str(cache_dir),
        size_limit=2 * 1024 * 1024 * 1024,  # 2GB
    )

    # Test different configurations
    configs = [
        {"max_workers": 1, "name": "Sequential (1 worker)"},
        {"max_workers": 2, "name": "Parallel (2 workers)"},
        {"max_workers": 4, "name": "Parallel (4 workers)"},
        {"max_workers": 8, "name": "Parallel (8 workers)"},
    ]

    results = {}

    for config in configs:
        print(f"\n=== Testing {config['name']} ===")

        # Clear cache to ensure fair testing
        cache_repo.clear()

        repository = ARCDataRepository(
            data_path=str(data_path),
            cache_repository=cache_repo,
            max_workers=config["max_workers"]
        )

        # Test different dataset sizes
        test_sizes = [50, 100, 150, 200]
        config_results = {}

        for size in test_sizes:
            print(f"\nLoading {size} tasks...")

            start_time = time.perf_counter()
            tasks = repository.load_all_tasks("training", limit=size)
            load_time = time.perf_counter() - start_time

            tasks_per_second = len(tasks) / load_time if load_time > 0 else 0
            estimated_1000 = (load_time / len(tasks)) * 1000 if len(tasks) > 0 else 0

            config_results[size] = {
                "load_time": load_time,
                "tasks_loaded": len(tasks),
                "tasks_per_second": tasks_per_second,
                "estimated_1000": estimated_1000
            }

            print(f"  Loaded {len(tasks)} tasks in {load_time:.2f}s")
            print(f"  Rate: {tasks_per_second:.1f} tasks/second")
            print(f"  Estimated time for 1000 tasks: {estimated_1000:.1f}s")

        results[config["name"]] = config_results

    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        for size, result in config_results.items():
            status = "PASS" if result["estimated_1000"] < 10 else "FAIL"
            print(f"  {size} tasks: {result['load_time']:.2f}s -> Est. 1000: {result['estimated_1000']:.1f}s {status}")

    # Find best configuration
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    best_config = None
    best_time = float('inf')

    for config_name, config_results in results.items():
        # Use 200 task result as most representative
        if 200 in config_results:
            est_time = config_results[200]["estimated_1000"]
            if est_time < best_time:
                best_time = est_time
                best_config = config_name

    if best_config and best_time < 10:
        print("PASS: PERFORMANCE TARGET MET")
        print(f"  Best configuration: {best_config}")
        print(f"  Estimated 1000-task load time: {best_time:.1f}s")
    else:
        print("FAIL: PERFORMANCE TARGET NOT MET")
        if best_config:
            print(f"  Best configuration: {best_config}")
            print(f"  Estimated 1000-task load time: {best_time:.1f}s")
        print("  TARGET: < 10.0s")
        print("  OPTIMIZATION NEEDED")

    # Cleanup
    try:
        cache_repo.close()
    except Exception:
        pass


if __name__ == "__main__":
    print("ARC Data Loading Performance Test")
    print("Testing with real ARC training data...")
    test_performance()
