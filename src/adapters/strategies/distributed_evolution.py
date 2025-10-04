"""
Distributed evolution support for running across multiple platforms.

Enables coordinated evolution across Kaggle, Colab, and Paperspace to
maximize available compute resources.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.domain.models import ARCTask
from src.infrastructure.components.platform_detector import Platform

logger = structlog.get_logger(__name__)


class PlatformRole(Enum):
    """Role of platform in distributed evolution."""
    COORDINATOR = "coordinator"  # Main platform coordinating others
    WORKER = "worker"  # Worker platform running sub-populations
    BACKUP = "backup"  # Backup platform for resilience


@dataclass
class PlatformTask:
    """Task assigned to a specific platform."""
    platform: Platform
    role: PlatformRole
    island_ids: list[int]  # Which islands to evolve
    generation_range: tuple[int, int]  # Start and end generation
    checkpoint_path: Path
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: dict[str, Any] | None = None


@dataclass
class DistributedEvolutionPlan:
    """Plan for distributed evolution across platforms."""
    task_id: str
    total_islands: int
    total_generations: int
    platform_tasks: list[PlatformTask] = field(default_factory=list)
    migration_schedule: list[int] = field(default_factory=list)  # Generations for migration
    checkpoint_dir: Path = Path("distributed_checkpoints")
    created_at: datetime = field(default_factory=datetime.now)


class DistributedEvolutionCoordinator:
    """Coordinates evolution across multiple platforms."""

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        checkpoint_frequency: int = 1
    ):
        self.checkpoint_dir = checkpoint_dir or Path("distributed_evolution")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.active_plans: dict[str, DistributedEvolutionPlan] = {}
        self.platform_metrics: dict[str, dict[str, Any]] = {}
        self.checkpoint_frequency = checkpoint_frequency

    def create_distribution_plan(
        self,
        task: ARCTask,
        available_platforms: list[Platform],
        total_generations: int = 200,
        islands_per_platform: dict[Platform, int] | None = None
    ) -> DistributedEvolutionPlan:
        """
        Create a plan for distributing evolution across platforms.

        Args:
            task: ARC task to solve
            available_platforms: Platforms available for computation
            total_generations: Total generations to run
            islands_per_platform: How many islands each platform should run

        Returns:
            Distribution plan
        """
        # Default island distribution
        if islands_per_platform is None:
            islands_per_platform = self._default_island_distribution(available_platforms)

        total_islands = sum(islands_per_platform.values())

        # Create plan
        plan = DistributedEvolutionPlan(
            task_id=task.task_id,
            total_islands=total_islands,
            total_generations=total_generations,
            checkpoint_dir=self.checkpoint_dir / task.task_id
        )

        # Create checkpoint directory
        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Assign platforms
        island_offset = 0
        coordinator_assigned = False

        for platform in available_platforms:
            num_islands = islands_per_platform.get(platform, 0)
            if num_islands == 0:
                continue

            # First platform with most resources is coordinator
            role = PlatformRole.COORDINATOR if not coordinator_assigned else PlatformRole.WORKER
            if role == PlatformRole.COORDINATOR:
                coordinator_assigned = True

            # Create platform task
            platform_task = PlatformTask(
                platform=platform,
                role=role,
                island_ids=list(range(island_offset, island_offset + num_islands)),
                generation_range=(0, total_generations),
                checkpoint_path=plan.checkpoint_dir / f"{platform.value}_checkpoint.json"
            )

            plan.platform_tasks.append(platform_task)
            island_offset += num_islands

        # Set migration schedule (every 10 generations)
        plan.migration_schedule = list(range(10, total_generations, 10))

        # Save plan
        self._save_plan(plan)
        self.active_plans[task.task_id] = plan

        return plan

    def _default_island_distribution(self, platforms: list[Platform]) -> dict[Platform, int]:
        """Get default island distribution based on platform capabilities."""
        distribution = {}

        # Platform capabilities (islands they can handle)
        platform_capacity = {
            Platform.KAGGLE: 4,  # Good resources
            Platform.COLAB: 3,   # Good GPU but session limits
            Platform.PAPERSPACE: 1,  # Limited free tier
            Platform.LOCAL: 2    # Depends on machine
        }

        for platform in platforms:
            distribution[platform] = platform_capacity.get(platform, 1)

        return distribution

    def calculate_platform_shard_sizes(
        self,
        platforms: list[Platform],
        total_population_size: int,
        platform_capabilities: dict[Platform, dict[str, float]] | None = None
    ) -> dict[Platform, int]:
        """
        Calculate dynamic shard sizes based on platform capabilities and current load.

        Args:
            platforms: Available platforms
            total_population_size: Total size of population to distribute
            platform_capabilities: Optional dict with platform metrics
                (memory_available, cpu_percent, current_load)

        Returns:
            Dict mapping platform to population shard size
        """
        if not platforms:
            raise ValueError("No platforms provided for sharding")

        # Default capabilities from platform configuration
        from src.adapters.strategies.platform_evolution_config import PlatformEvolutionConfigurator
        configurator = PlatformEvolutionConfigurator()

        # Calculate weights based on capabilities
        platform_weights = {}
        for platform in platforms:
            settings = configurator.PLATFORM_SETTINGS.get(platform)
            if not settings:
                platform_weights[platform] = 1.0
                continue

            # Base weight from configuration
            base_weight = (
                settings.workers *
                (settings.memory_limit_mb / 4096) *  # Normalize to 4GB
                (settings.batch_size / 500)  # Normalize to batch 500
            )

            # Adjust for current load if provided
            if platform_capabilities and platform in platform_capabilities:
                caps = platform_capabilities[platform]
                load_factor = 1.0 - caps.get('current_load', 0.0)
                memory_factor = caps.get('memory_available', 1.0)
                base_weight *= load_factor * memory_factor

            platform_weights[platform] = max(0.1, base_weight)  # Minimum weight

        # Calculate shard sizes proportionally
        total_weight = sum(platform_weights.values())
        shard_sizes = {}
        assigned = 0

        for platform in platforms[:-1]:
            shard_size = int(
                (platform_weights[platform] / total_weight) * total_population_size
            )
            shard_sizes[platform] = max(10, shard_size)  # Minimum 10 individuals
            assigned += shard_sizes[platform]

        # Last platform gets remainder to ensure exact total
        shard_sizes[platforms[-1]] = max(10, total_population_size - assigned)

        return shard_sizes

    def assign_islands_to_platforms(
        self,
        platforms: list[Platform],
        total_islands: int,
        shard_sizes: dict[Platform, int] | None = None
    ) -> dict[Platform, list[int]]:
        """
        Assign island IDs to platforms for distributed evolution.

        Args:
            platforms: Available platforms
            total_islands: Total number of islands
            shard_sizes: Optional population sizes per platform

        Returns:
            Dict mapping platform to list of island IDs
        """
        if not platforms:
            raise ValueError("No platforms provided for island assignment")

        if total_islands < len(platforms):
            raise ValueError(
                f"Total islands ({total_islands}) must be >= platforms ({len(platforms)})"
            )

        # Calculate islands per platform proportionally to shard sizes
        if shard_sizes:
            total_pop = sum(shard_sizes.values())
            islands_per_platform = {}
            assigned_islands = 0

            for platform in platforms[:-1]:
                num_islands = max(
                    1,
                    int((shard_sizes[platform] / total_pop) * total_islands)
                )
                islands_per_platform[platform] = num_islands
                assigned_islands += num_islands

            # Last platform gets remainder
            islands_per_platform[platforms[-1]] = max(1, total_islands - assigned_islands)
        else:
            # Even distribution
            base_islands = total_islands // len(platforms)
            remainder = total_islands % len(platforms)

            islands_per_platform = {}
            for i, platform in enumerate(platforms):
                islands_per_platform[platform] = base_islands + (1 if i < remainder else 0)

        # Assign actual island IDs
        island_assignments = {}
        current_island = 0

        for platform in platforms:
            num_islands = islands_per_platform[platform]
            island_assignments[platform] = list(range(current_island, current_island + num_islands))
            current_island += num_islands

        return island_assignments

    async def redistribute_tasks_on_failure(
        self,
        failed_platform: Platform,
        active_platforms: list[Platform],
        task_id: str
    ) -> bool:
        """
        Redistribute tasks from failed platform to active platforms.

        Args:
            failed_platform: Platform that has failed
            active_platforms: List of currently active platforms
            task_id: Task identifier

        Returns:
            True if redistribution successful
        """
        if task_id not in self.active_plans:
            logger.error("task_not_found", task_id=task_id)
            return False

        if not active_platforms:
            logger.error("no_active_platforms_for_redistribution")
            return False

        plan = self.active_plans[task_id]

        failed_task = None
        for task in plan.platform_tasks:
            if task.platform == failed_platform:
                failed_task = task
                break

        if not failed_task:
            logger.warning("failed_platform_not_in_plan", platform=failed_platform.value)
            return False

        logger.info(
            "redistributing_tasks",
            failed_platform=failed_platform.value,
            islands=failed_task.island_ids,
            active_platforms=[p.value for p in active_platforms]
        )

        failed_islands = failed_task.island_ids
        num_failed_islands = len(failed_islands)

        islands_per_platform = num_failed_islands // len(active_platforms)
        remainder = num_failed_islands % len(active_platforms)

        island_index = 0
        for i, platform in enumerate(active_platforms):
            num_islands = islands_per_platform + (1 if i < remainder else 0)

            redistributed_islands = failed_islands[island_index:island_index + num_islands]

            for task in plan.platform_tasks:
                if task.platform == platform and task.status in ["running", "pending"]:
                    task.island_ids.extend(redistributed_islands)
                    logger.info(
                        "islands_redistributed",
                        platform=platform.value,
                        added_islands=redistributed_islands,
                        total_islands=len(task.island_ids)
                    )
                    break

            island_index += num_islands

        failed_task.status = "failed"
        failed_task.completed_at = datetime.now()

        self._save_plan(plan)

        return True

    async def elect_backup_coordinator(
        self,
        task_id: str,
        failed_coordinator: Platform,
        active_platforms: list[Platform]
    ) -> Platform | None:
        """
        Elect backup coordinator when primary coordinator fails.

        Uses priority-based election: most capable platform becomes coordinator.

        Args:
            task_id: Task identifier
            failed_coordinator: Platform that failed
            active_platforms: List of currently active platforms

        Returns:
            Elected platform or None if election failed
        """
        if task_id not in self.active_plans:
            logger.error("task_not_found", task_id=task_id)
            return None

        if not active_platforms:
            logger.error("no_active_platforms_for_election")
            return None

        plan = self.active_plans[task_id]

        platform_priorities = {
            Platform.KAGGLE: 4,
            Platform.COLAB: 3,
            Platform.LOCAL: 2,
            Platform.PAPERSPACE: 1
        }

        eligible_platforms = [
            p for p in active_platforms
            if any(task.platform == p and task.status in ["running", "pending"]
                  for task in plan.platform_tasks)
        ]

        if not eligible_platforms:
            logger.error("no_eligible_platforms_for_coordinator")
            return None

        elected = max(eligible_platforms, key=lambda p: platform_priorities.get(p, 0))

        logger.info(
            "backup_coordinator_elected",
            failed_coordinator=failed_coordinator.value,
            elected_coordinator=elected.value,
            eligible_platforms=[p.value for p in eligible_platforms]
        )

        for task in plan.platform_tasks:
            if task.platform == failed_coordinator:
                task.role = PlatformRole.BACKUP
                task.status = "failed"
            elif task.platform == elected:
                old_role = task.role
                task.role = PlatformRole.COORDINATOR
                logger.info(
                    "coordinator_role_assigned",
                    platform=elected.value,
                    old_role=old_role.value,
                    new_role=PlatformRole.COORDINATOR.value
                )

        self._save_plan(plan)

        return elected

    async def recover_partial_results(
        self,
        task_id: str,
        disconnected_platform: Platform,
        checkpoint_manager: Any
    ) -> dict[str, Any] | None:
        """
        Recover partial results from disconnected platform.

        Attempts to load last checkpoint from GCS or local storage.

        Args:
            task_id: Task identifier
            disconnected_platform: Platform that disconnected
            checkpoint_manager: AsyncCheckpointManager instance

        Returns:
            Recovered checkpoint data or None if recovery failed
        """
        if task_id not in self.active_plans:
            logger.error("task_not_found", task_id=task_id)
            return None

        plan = self.active_plans[task_id]

        platform_task = None
        for task in plan.platform_tasks:
            if task.platform == disconnected_platform:
                platform_task = task
                break

        if not platform_task:
            logger.warning("platform_not_in_plan", platform=disconnected_platform.value)
            return None

        logger.info(
            "attempting_partial_recovery",
            platform=disconnected_platform.value,
            checkpoint_path=str(platform_task.checkpoint_path)
        )

        try:
            checkpoint = await checkpoint_manager.load_checkpoint_async(from_gcs=True)

            if checkpoint:
                logger.info(
                    "partial_recovery_successful",
                    platform=disconnected_platform.value,
                    generation=checkpoint.generation,
                    population_size=len(checkpoint.population)
                )

                return {
                    'generation': checkpoint.generation,
                    'population': [
                        {
                            'program': p.program,
                            'fitness': p.fitness,
                            'hash': p.hash
                        }
                        for p in checkpoint.population
                    ],
                    'metadata': {
                        'platform_id': checkpoint.metadata.platform_id,
                        'timestamp': checkpoint.metadata.timestamp,
                        'resource_usage': checkpoint.metadata.resource_usage
                    },
                    'recovered': True,
                    'recovery_timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(
                "partial_recovery_failed",
                platform=disconnected_platform.value,
                error=str(e)
            )

        try:
            if platform_task.checkpoint_path.exists():
                with open(platform_task.checkpoint_path) as f:
                    local_checkpoint = json.load(f)

                logger.info(
                    "local_recovery_successful",
                    platform=disconnected_platform.value,
                    generation=local_checkpoint.get('generation', 'unknown')
                )

                local_checkpoint['recovered'] = True
                local_checkpoint['recovery_timestamp'] = datetime.now().isoformat()
                return local_checkpoint
        except Exception as e:
            logger.error(
                "local_recovery_failed",
                platform=disconnected_platform.value,
                error=str(e)
            )

        logger.warning(
            "no_recovery_possible",
            platform=disconnected_platform.value
        )

        return None

    async def exchange_populations(
        self,
        source_platform: Platform,
        target_platform: Platform,
        source_population: list[dict[str, Any]],
        migration_size: int = 10
    ) -> list[dict[str, Any]]:
        """
        Exchange populations between platforms asynchronously.

        Args:
            source_platform: Platform sending individuals
            target_platform: Platform receiving individuals
            source_population: Source population to migrate from
            migration_size: Number of individuals to migrate

        Returns:
            List of migrated individuals
        """
        if not source_population:
            return []

        # Select top individuals for migration
        sorted_population = sorted(
            source_population,
            key=lambda x: x.get('fitness', 0.0),
            reverse=True
        )

        migrants = sorted_population[:min(migration_size, len(sorted_population))]

        # Log migration event
        print(f"Migrating {len(migrants)} individuals from {source_platform.value} to {target_platform.value}")

        # Record migration metrics
        migration_key = f"{source_platform.value}_to_{target_platform.value}"
        if migration_key not in self.platform_metrics:
            self.platform_metrics[migration_key] = {
                'total_migrations': 0,
                'total_individuals': 0,
                'avg_fitness': 0.0
            }

        metrics = self.platform_metrics[migration_key]
        metrics['total_migrations'] += 1
        metrics['total_individuals'] += len(migrants)

        if migrants:
            avg_fitness = sum(m.get('fitness', 0.0) for m in migrants) / len(migrants)
            metrics['avg_fitness'] = (
                (metrics['avg_fitness'] * (metrics['total_migrations'] - 1) + avg_fitness) /
                metrics['total_migrations']
            )

        return migrants

    def should_checkpoint(self, generation: int) -> bool:
        """
        Determine if checkpoint should be created for current generation.

        Args:
            generation: Current generation number

        Returns:
            True if checkpoint should be created
        """
        return generation % self.checkpoint_frequency == 0

    def get_platform_config(
        self,
        plan: DistributedEvolutionPlan,
        platform: Platform
    ) -> dict[str, Any]:
        """Get configuration for a specific platform in the plan."""
        # Find platform task
        platform_task = None
        for task in plan.platform_tasks:
            if task.platform == platform:
                platform_task = task
                break

        if not platform_task:
            raise ValueError(f"Platform {platform} not in distribution plan")

        # Create configuration
        config = {
            'task_id': plan.task_id,
            'role': platform_task.role.value,
            'island_ids': platform_task.island_ids,
            'total_islands': plan.total_islands,
            'generation_range': platform_task.generation_range,
            'migration_schedule': plan.migration_schedule,
            'checkpoint_path': str(platform_task.checkpoint_path),
            'coordinator_checkpoint': str(plan.checkpoint_dir / "coordinator_state.json")
        }

        # Add platform-specific settings
        if platform == Platform.KAGGLE:
            config['worker_count'] = 2
            config['batch_size'] = 500
        elif platform == Platform.COLAB:
            config['gpu_enabled'] = True
            config['gpu_batch_size'] = 200
        elif platform == Platform.PAPERSPACE:
            config['conservative_mode'] = True
            config['batch_size'] = 100

        return config

    async def coordinate_migration(
        self,
        plan: DistributedEvolutionPlan,
        generation: int
    ) -> bool:
        """
        Coordinate migration between platforms at specified generation.

        Returns:
            True if migration successful
        """
        if generation not in plan.migration_schedule:
            return True

        print(f"Coordinating migration at generation {generation}")

        # Collect checkpoints from all platforms
        all_populations = []

        for task in plan.platform_tasks:
            if task.status == "running":
                checkpoint_path = task.checkpoint_path
                if checkpoint_path.exists():
                    with open(checkpoint_path) as f:
                        checkpoint_data = json.load(f)
                    all_populations.extend(checkpoint_data.get('population', []))

        if not all_populations:
            print("Warning: No populations found for migration")
            return False

        # Select migrants (top 10% from each island)
        migrants = self._select_migrants(all_populations, plan.total_islands)

        # Distribute migrants to platforms
        migration_data = self._distribute_migrants(migrants, plan)

        # Save migration data for each platform
        for platform, data in migration_data.items():
            migration_path = plan.checkpoint_dir / f"{platform.value}_migration_{generation}.json"
            with open(migration_path, 'w') as f:
                json.dump(data, f, indent=2)

        return True

    def _select_migrants(
        self,
        all_populations: list[dict[str, Any]],
        num_islands: int
    ) -> list[dict[str, Any]]:
        """Select best individuals for migration."""
        # Sort by fitness
        sorted_pop = sorted(all_populations, key=lambda x: x.get('fitness', 0), reverse=True)

        # Select top individuals (migration_rate % from each island)
        migration_rate = 0.1
        migrants_per_island = max(1, int(len(sorted_pop) / num_islands * migration_rate))

        return sorted_pop[:migrants_per_island * num_islands]

    def _distribute_migrants(
        self,
        migrants: list[dict[str, Any]],
        plan: DistributedEvolutionPlan
    ) -> dict[Platform, dict[str, Any]]:
        """Distribute migrants across platforms."""
        distribution = {}

        # Round-robin distribution
        for i, migrant in enumerate(migrants):
            platform_idx = i % len(plan.platform_tasks)
            platform = plan.platform_tasks[platform_idx].platform

            if platform not in distribution:
                distribution[platform] = {'migrants': []}

            distribution[platform]['migrants'].append(migrant)

        return distribution

    def _save_plan(self, plan: DistributedEvolutionPlan) -> None:
        """Save distribution plan to disk."""
        plan_path = plan.checkpoint_dir / "distribution_plan.json"

        plan_data = {
            'task_id': plan.task_id,
            'total_islands': plan.total_islands,
            'total_generations': plan.total_generations,
            'migration_schedule': plan.migration_schedule,
            'created_at': plan.created_at.isoformat(),
            'platform_tasks': [
                {
                    'platform': task.platform.value,
                    'role': task.role.value,
                    'island_ids': task.island_ids,
                    'generation_range': task.generation_range,
                    'checkpoint_path': str(task.checkpoint_path),
                    'status': task.status
                }
                for task in plan.platform_tasks
            ]
        }

        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2)

    def update_platform_status(
        self,
        task_id: str,
        platform: Platform,
        status: str,
        results: dict[str, Any] | None = None
    ) -> None:
        """Update status of a platform task."""
        if task_id not in self.active_plans:
            return

        plan = self.active_plans[task_id]

        for task in plan.platform_tasks:
            if task.platform == platform:
                task.status = status
                if status == "running" and task.started_at is None:
                    task.started_at = datetime.now()
                elif status == "completed":
                    task.completed_at = datetime.now()
                    task.results = results
                break

        # Save updated plan
        self._save_plan(plan)

    def get_aggregated_results(self, task_id: str) -> dict[str, Any] | None:
        """Get aggregated results from all platforms."""
        if task_id not in self.active_plans:
            return None

        plan = self.active_plans[task_id]

        # Collect all results
        all_results = {
            'task_id': task_id,
            'best_fitness': 0.0,
            'best_individual': None,
            'total_evaluations': 0,
            'platform_results': {}
        }

        for task in plan.platform_tasks:
            if task.results:
                platform_name = task.platform.value
                all_results['platform_results'][platform_name] = task.results

                # Update aggregated metrics
                if task.results.get('best_fitness', 0) > all_results['best_fitness']:
                    all_results['best_fitness'] = task.results['best_fitness']
                    all_results['best_individual'] = task.results.get('best_individual')

                all_results['total_evaluations'] += task.results.get('total_evaluations', 0)

        return all_results


# Platform worker functions
async def run_platform_worker(
    platform: Platform,
    config: dict[str, Any],
    evolution_engine: Any
) -> dict[str, Any]:
    """
    Run evolution worker on a specific platform.

    This would be executed on each platform with platform-specific config.
    """
    print(f"Starting evolution worker on {platform.value}")
    print(f"Role: {config['role']}")
    print(f"Islands: {config['island_ids']}")

    # Configure evolution engine for this platform's islands
    evolution_engine.config.island_model['enabled'] = True
    evolution_engine.config.island_model['island_ids'] = config['island_ids']
    evolution_engine.config.island_model['total_islands'] = config['total_islands']

    # Run evolution with migration support
    results = await evolution_engine.evolve_distributed(
        task_id=config['task_id'],
        generation_range=config['generation_range'],
        migration_schedule=config['migration_schedule'],
        checkpoint_path=Path(config['checkpoint_path'])
    )

    return results
