"""
Example script demonstrating distributed evolution usage.

This script shows how to set up and run distributed evolution across
multiple platforms (Kaggle, Colab, Local).
"""

import asyncio
from pathlib import Path

from src.adapters.strategies.distributed_checkpoint_manager import AsyncCheckpointManager
from src.adapters.strategies.distributed_evolution import DistributedEvolutionCoordinator
from src.adapters.strategies.distributed_evolution_config import (
    DistributedEvolutionConfig,
)
from src.adapters.strategies.platform_health_monitor import PlatformHealthMonitor
from src.adapters.strategies.population_merger import PopulationMerger
from src.domain.models import ARCTask
from src.infrastructure.components.platform_detector import Platform


async def load_config() -> DistributedEvolutionConfig:
    """Load distributed evolution configuration from YAML."""
    import yaml

    config_path = Path("configs/strategies/evolution.yaml")
    with open(config_path, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    distributed_config = yaml_config.get("distributed_evolution", {})
    return DistributedEvolutionConfig.from_dict(distributed_config)


async def example_single_platform():
    """Example: Run evolution on single platform."""
    print("=== Single Platform Evolution Example ===\n")

    task = ARCTask(
        task_id="example_001",
        train=[
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]]
            }
        ],
        test=[
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]]
            }
        ]
    )

    coordinator = DistributedEvolutionCoordinator(
        checkpoint_dir=Path("example_checkpoints"),
        checkpoint_frequency=5
    )

    plan = coordinator.create_distribution_plan(
        task=task,
        available_platforms=[Platform.LOCAL],
        total_generations=50
    )

    print(f"Created plan for task {task.task_id}")
    print(f"  Total islands: {plan.total_islands}")
    print(f"  Total generations: {plan.total_generations}")
    print(f"  Platform tasks: {len(plan.platform_tasks)}\n")

    for platform_task in plan.platform_tasks:
        print(f"Platform: {platform_task.platform.value}")
        print(f"  Role: {platform_task.role.value}")
        print(f"  Islands: {platform_task.island_ids}")
        print(f"  Generations: {platform_task.generation_range}\n")


async def example_multi_platform():
    """Example: Run distributed evolution across multiple platforms."""
    print("=== Multi-Platform Distributed Evolution Example ===\n")

    config = await load_config()

    validation_errors = config.validate()
    if validation_errors:
        print("Configuration errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return

    print("Configuration loaded successfully\n")
    print(f"Enabled: {config.enabled}")
    print(f"Checkpoint frequency: every {config.checkpoint_frequency} generations")
    print(f"Heartbeat timeout: {config.heartbeat_timeout}s")
    print(f"Platforms configured: {len(config.platforms)}\n")

    health_monitor = PlatformHealthMonitor(
        heartbeat_timeout=config.heartbeat_timeout,
        heartbeat_interval=config.heartbeat_interval
    )

    for platform_config in config.platforms:
        health_monitor.register_platform(
            platform_id=platform_config.id,
            capabilities={
                "memory_mb": platform_config.memory_limit_mb,
                "workers": platform_config.worker_count,
                "batch_size": platform_config.batch_size
            }
        )
        print(f"Registered platform: {platform_config.id} ({platform_config.role})")

    print("\nSimulating heartbeats...")
    for platform_config in config.platforms:
        await health_monitor.process_heartbeat(
            platform_id=platform_config.id,
            status="healthy",
            metrics={"generation": 10, "fitness": 0.75}
        )

    print("\nPlatform health status:")
    all_status = health_monitor.get_all_platform_status()
    for platform_id, status in all_status.items():
        print(f"  {platform_id}: {status.status.value} (last seen: {status.last_seen})")


async def example_checkpoint_operations():
    """Example: Checkpoint save and load operations."""
    print("\n=== Checkpoint Operations Example ===\n")

    checkpoint_manager = AsyncCheckpointManager(
        platform_id="example-platform",
        checkpoint_dir=Path("example_checkpoints"),
        enable_compression=True,
        compression_level=6
    )

    population = [
        {
            "program": "map(identity) | filter(lambda x: x > 0)",
            "fitness": 0.85,
        },
        {
            "program": "fold(add, 0) | map(lambda x: x * 2)",
            "fitness": 0.75,
        },
        {
            "program": "zip | map(lambda pair: pair[0] + pair[1])",
            "fitness": 0.65,
        }
    ]

    print("Creating checkpoint...")
    checkpoint = checkpoint_manager.create_checkpoint(
        generation=10,
        population=population
    )

    print("Checkpoint created:")
    print(f"  Version: {checkpoint.version}")
    print(f"  Generation: {checkpoint.generation}")
    print(f"  Population size: {len(checkpoint.population)}")
    print(f"  Platform ID: {checkpoint.metadata.platform_id}\n")

    print("Serializing checkpoint...")
    serialized = checkpoint_manager.serialize_checkpoint(checkpoint)
    print(f"Serialized size: {len(serialized)} bytes")

    print("\nDeserializing checkpoint...")
    restored_checkpoint = checkpoint_manager.deserialize_checkpoint(serialized)
    print(f"Restored generation: {restored_checkpoint.generation}")
    print(f"Restored population size: {len(restored_checkpoint.population)}")

    print("\nValidating checkpoint...")
    is_valid = checkpoint_manager.validate_checkpoint(restored_checkpoint)
    print(f"Checkpoint valid: {is_valid}")


async def example_population_merge():
    """Example: Merge populations from multiple platforms."""
    print("\n=== Population Merge Example ===\n")

    population_kaggle = [
        {"program": "map(identity)", "fitness": 0.9, "hash": "hash1"},
        {"program": "filter(lambda x: x > 0)", "fitness": 0.8, "hash": "hash2"},
        {"program": "fold(add, 0)", "fitness": 0.7, "hash": "hash3"},
    ]

    population_colab = [
        {"program": "map(identity)", "fitness": 0.85, "hash": "hash1"},
        {"program": "zip | map(add)", "fitness": 0.75, "hash": "hash4"},
        {"program": "map(lambda x: x * 2)", "fitness": 0.65, "hash": "hash5"},
    ]

    population_local = [
        {"program": "fold(add, 0)", "fitness": 0.72, "hash": "hash3"},
        {"program": "map(square)", "fitness": 0.68, "hash": "hash6"},
    ]

    print("Populations before merge:")
    print(f"  Kaggle: {len(population_kaggle)} individuals")
    print(f"  Colab: {len(population_colab)} individuals")
    print(f"  Local: {len(population_local)} individuals")
    print(f"  Total: {len(population_kaggle) + len(population_colab) + len(population_local)}\n")

    merger = PopulationMerger()
    merged_population = merger.merge_populations([
        population_kaggle,
        population_colab,
        population_local
    ])

    print(f"Merged population: {len(merged_population)} unique individuals")
    print("\nMerged individuals (sorted by fitness):")
    for individual in sorted(merged_population, key=lambda x: x["fitness"], reverse=True):
        print(f"  Fitness: {individual['fitness']:.2f} - {individual['program']}")

    stats = merger.get_merge_stats()
    print("\nMerge statistics:")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Unique programs: {stats['unique_programs']}")
    print(f"  Total merged: {stats['total_merged']}")


async def example_end_to_end():
    """Example: Complete end-to-end distributed evolution workflow."""
    print("\n=== End-to-End Distributed Evolution Example ===\n")

    print("Step 1: Load configuration")
    config = await load_config()
    config.enabled = True
    config.gcs_bucket = "example-arc-checkpoints"
    config.gcs_credentials_path = Path("credentials/gcs-key.json")

    validation_errors = config.validate()
    if not validation_errors:
        print("  Configuration valid\n")
    else:
        print("  Skipping (would need valid GCS setup)\n")
        return

    print("Step 2: Initialize components")
    coordinator = DistributedEvolutionCoordinator(
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_frequency=config.checkpoint_frequency
    )
    health_monitor = PlatformHealthMonitor(
        heartbeat_timeout=config.heartbeat_timeout
    )
    print("  Components initialized\n")

    print("Step 3: Register platforms")
    for platform_config in config.platforms:
        health_monitor.register_platform(
            platform_id=platform_config.id,
            capabilities={
                "memory_mb": platform_config.memory_limit_mb,
                "workers": platform_config.worker_count,
                "batch_size": platform_config.batch_size
            }
        )
        print(f"  Registered {platform_config.id}")

    print("\nStep 4: Create distribution plan")
    task = ARCTask(
        task_id="example_end_to_end",
        train=[{"input": [[1, 0]], "output": [[0, 1]]}],
        test=[{"input": [[1, 0]], "output": [[0, 1]]}]
    )

    available_platforms = [
        p.platform_type for p in config.platforms if p.platform_type
    ]
    plan = coordinator.create_distribution_plan(
        task=task,
        available_platforms=available_platforms,
        total_generations=100
    )
    print(f"  Plan created for {len(plan.platform_tasks)} platforms\n")

    print("Step 5: Simulate evolution on each platform")
    print("  (In production, this would run actual evolution)")
    print("  Platforms would exchange populations every N generations")
    print("  Checkpoints would sync via GCS")
    print("  Health monitor would track platform status\n")

    print("Distributed evolution workflow complete!")


async def main():
    """Run all examples."""
    await example_single_platform()
    await example_multi_platform()
    await example_checkpoint_operations()
    await example_population_merge()
    await example_end_to_end()


if __name__ == "__main__":
    asyncio.run(main())
