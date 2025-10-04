"""
Island model implementation for parallel population evolution.

This module implements an island-based evolutionary algorithm where multiple
sub-populations evolve in parallel with periodic migration between islands.
This approach improves diversity and exploration while maintaining local optimization.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from src.adapters.strategies.evolution_engine import (
    EvolutionEngine,
    Individual,
    Population,
)
from src.domain.models import ARCTask
from src.domain.services.dsl_engine import DSLEngine
from src.infrastructure.config import GeneticAlgorithmConfig


@dataclass
class Island:
    """
    Represents an island (sub-population) in the island model.

    Attributes:
        id: Unique identifier for the island
        population: The island's population
        evolution_engine: Engine managing evolution for this island
        migration_history: History of migrations to/from this island
        fitness_trend: Trend of average fitness over generations
    """
    id: int
    population: Population
    evolution_engine: EvolutionEngine
    migration_history: list[dict[str, Any]] = field(default_factory=list)
    fitness_trend: list[float] = field(default_factory=list)

    def update_fitness_trend(self) -> None:
        """Update the fitness trend for this island."""
        avg_fitness = self.population.average_fitness()
        self.fitness_trend.append(avg_fitness)

    def is_stagnant(self, lookback: int = 5, threshold: float = 0.001) -> bool:
        """Check if the island population is stagnant."""
        if len(self.fitness_trend) < lookback:
            return False

        recent_trend = self.fitness_trend[-lookback:]
        improvement = recent_trend[-1] - recent_trend[0]
        return improvement < threshold


@dataclass
class MigrationPolicy:
    """
    Defines the migration policy between islands.

    Attributes:
        frequency: Number of generations between migrations
        migration_rate: Proportion of population to migrate
        selection_method: How to select migrants (best, random, tournament)
        topology: Migration topology (ring, fully_connected, random)
    """
    frequency: int = 10
    migration_rate: float = 0.1
    selection_method: str = "tournament"
    topology: str = "ring"
    adaptive: bool = True  # Adjust migration based on island performance


class IslandEvolutionEngine:
    """
    Implements island model for parallel population evolution.

    The island model divides the population into multiple sub-populations
    that evolve independently with periodic migration between islands.
    """

    def __init__(
        self,
        num_islands: int,
        config: GeneticAlgorithmConfig,
        dsl_engine: DSLEngine,
        migration_policy: MigrationPolicy | None = None
    ):
        """
        Initialize island evolution engine.

        Args:
            num_islands: Number of islands (sub-populations)
            config: Genetic algorithm configuration
            dsl_engine: DSL engine for program execution
            migration_policy: Policy for migration between islands
        """
        self.num_islands = num_islands
        self.config = config
        self.dsl_engine = dsl_engine
        self.migration_policy = migration_policy or MigrationPolicy()

        # Adjust per-island population size
        self.island_pop_size = config.population.size // num_islands

        # Create island-specific configurations
        self.island_configs = self._create_island_configs()

        # Initialize islands
        self.islands: list[Island] = []
        self.global_best_individual: Individual | None = None
        self.generation = 0
        self.migration_count = 0

        # Tracking metrics
        self.island_diversity_scores: dict[int, float] = {}
        self.migration_success_rate = 0.0
        self.island_specialization: dict[int, str] = {}

        # Initialize parallel executor
        self.executor = ProcessPoolExecutor(max_workers=min(num_islands, config.parallelization.workers))

    def _create_island_configs(self) -> list[GeneticAlgorithmConfig]:
        """
        Create variations of configuration for each island.

        This promotes diversity by having islands use different
        evolutionary parameters.
        """
        configs = []

        for i in range(self.num_islands):
            # Create a copy of base config
            island_config = self._copy_config(self.config)

            # Adjust population size
            island_config.population.size = self.island_pop_size

            # Vary evolutionary parameters per island
            if i % 3 == 0:
                # High mutation island
                island_config.genetic_operators.mutation.base_rate *= 1.5
                island_config.genetic_operators.crossover.rate *= 0.8
            elif i % 3 == 1:
                # High crossover island
                island_config.genetic_operators.crossover.rate *= 1.2
                island_config.genetic_operators.mutation.base_rate *= 0.8
            else:
                # Balanced island (no changes)
                pass

            # Vary diversity mechanisms
            diversity_methods = ["fitness_sharing", "speciation", "novelty", "crowding"]
            island_config.diversity.method = diversity_methods[i % len(diversity_methods)]

            configs.append(island_config)

        return configs

    def _copy_config(self, config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfig:
        """Create a deep copy of configuration."""
        import copy
        return copy.deepcopy(config)

    async def initialize_islands(self, task: ARCTask) -> None:
        """Initialize all islands with their populations."""
        for i in range(self.num_islands):
            # Create evolution engine for island
            engine = EvolutionEngine(
                config=self.island_configs[i],
                dsl_engine=self.dsl_engine
            )

            # Initialize population
            await engine._initialize_population()

            # Create island
            island = Island(
                id=i,
                population=engine.population,
                evolution_engine=engine
            )

            self.islands.append(island)

            # Track initial diversity
            self.island_diversity_scores[i] = self._calculate_island_diversity(island)

    async def evolve(
        self,
        task: ARCTask,
        max_generations: int = 200,
        callbacks: list[Callable] | None = None
    ) -> tuple[Individual, dict[str, Any]]:
        """
        Run island-based evolution for the given task.

        Args:
            task: ARC task to solve
            max_generations: Maximum number of generations
            callbacks: Optional callbacks for updates

        Returns:
            Tuple of (best individual, evolution statistics)
        """
        # Initialize islands
        await self.initialize_islands(task)

        # Evolution loop
        for generation in range(max_generations):
            self.generation = generation

            # Evolve each island in parallel
            island_futures = []
            for island in self.islands:
                future = self.executor.submit(
                    self._evolve_island_generation,
                    island,
                    task
                )
                island_futures.append((island.id, future))

            # Collect results
            for island_id, future in island_futures:
                try:
                    island_best = future.result()
                    self._update_global_best(island_best)
                except Exception as e:
                    print(f"Error evolving island {island_id}: {e}")

            # Perform migration if scheduled
            if generation > 0 and generation % self.migration_policy.frequency == 0:
                await self._perform_migration()

            # Update metrics
            self._update_island_metrics()

            # Check convergence
            if self._check_convergence():
                break

            # Adaptive island specialization
            if generation % 20 == 0:
                self._adapt_island_parameters()

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, generation)

        # Collect statistics
        stats = self._collect_evolution_stats()

        return self.global_best_individual, stats

    def _evolve_island_generation(self, island: Island, task: ARCTask) -> Individual | None:
        """Evolve one generation for a specific island."""
        # Set task for evaluation
        island.evolution_engine.current_task = task

        # Run one generation
        asyncio.run(island.evolution_engine._evolve_generation())

        # Update island metrics
        island.update_fitness_trend()

        # Return best individual from island
        return island.population.best_individual

    async def _perform_migration(self) -> None:
        """Perform migration between islands based on topology."""
        self.migration_count += 1
        migration_events = []

        if self.migration_policy.topology == "ring":
            # Ring topology: each island exchanges with neighbors
            for i in range(self.num_islands):
                source_island = self.islands[i]
                target_island = self.islands[(i + 1) % self.num_islands]

                migrants = self._select_migrants(source_island)
                self._integrate_migrants(target_island, migrants)

                migration_events.append({
                    'from': i,
                    'to': (i + 1) % self.num_islands,
                    'count': len(migrants),
                    'avg_fitness': sum(m.fitness for m in migrants) / len(migrants) if migrants else 0
                })

        elif self.migration_policy.topology == "fully_connected":
            # Fully connected: random exchanges between all islands
            num_migrations = self.num_islands * 2  # Average 2 migrations per island

            for _ in range(num_migrations):
                source_idx = random.randint(0, self.num_islands - 1)
                target_idx = random.randint(0, self.num_islands - 1)

                if source_idx != target_idx:
                    source_island = self.islands[source_idx]
                    target_island = self.islands[target_idx]

                    migrants = self._select_migrants(source_island)
                    self._integrate_migrants(target_island, migrants)

                    migration_events.append({
                        'from': source_idx,
                        'to': target_idx,
                        'count': len(migrants),
                        'avg_fitness': sum(m.fitness for m in migrants) / len(migrants) if migrants else 0
                    })

        elif self.migration_policy.topology == "adaptive":
            # Adaptive topology: migrate from high-performing to low-performing islands
            island_performances = [(i, island.population.average_fitness())
                                 for i, island in enumerate(self.islands)]
            island_performances.sort(key=lambda x: x[1], reverse=True)

            # Top half migrates to bottom half
            mid_point = self.num_islands // 2
            for i in range(mid_point):
                source_idx = island_performances[i][0]
                target_idx = island_performances[-(i+1)][0]

                source_island = self.islands[source_idx]
                target_island = self.islands[target_idx]

                migrants = self._select_migrants(source_island)
                self._integrate_migrants(target_island, migrants)

                migration_events.append({
                    'from': source_idx,
                    'to': target_idx,
                    'count': len(migrants),
                    'avg_fitness': sum(m.fitness for m in migrants) / len(migrants) if migrants else 0
                })

        # Track migration success
        self._track_migration_success(migration_events)

    def _select_migrants(self, island: Island) -> list[Individual]:
        """Select individuals to migrate from an island."""
        num_migrants = int(island.population.size() * self.migration_policy.migration_rate)

        if self.migration_policy.selection_method == "best":
            # Select best individuals
            migrants = island.population.get_elite(num_migrants)
        elif self.migration_policy.selection_method == "tournament":
            # Tournament selection
            migrants = []
            for _ in range(num_migrants):
                tournament_size = 3
                tournament = random.sample(island.population.individuals,
                                         min(tournament_size, island.population.size()))
                winner = max(tournament, key=lambda x: x.fitness)
                migrants.append(winner)
        else:
            # Random selection
            migrants = random.sample(island.population.individuals,
                                   min(num_migrants, island.population.size()))

        return migrants

    def _integrate_migrants(self, island: Island, migrants: list[Individual]) -> None:
        """Integrate migrants into target island population."""
        if not migrants:
            return

        # Replace worst individuals with migrants
        island.population.individuals.sort(key=lambda x: x.fitness, reverse=True)

        for i, migrant in enumerate(migrants):
            if i < len(island.population.individuals):
                # Clone migrant to avoid reference issues
                new_individual = self._clone_individual(migrant)

                # Add migration metadata
                new_individual.metadata['migration_generation'] = self.generation
                new_individual.metadata['source_island'] = migrant.metadata.get('island_id', -1)
                new_individual.metadata['island_id'] = island.id

                # Replace worst individual
                island.population.individuals[-(i+1)] = new_individual

        # Update island's best individual
        island.population._update_best_individual(max(migrants, key=lambda x: x.fitness))

    def _clone_individual(self, individual: Individual) -> Individual:
        """Create a deep copy of an individual."""
        import copy
        return copy.deepcopy(individual)

    def _update_global_best(self, individual: Individual | None) -> None:
        """Update global best individual if better found."""
        if individual is None:
            return

        if (self.global_best_individual is None or
            individual.fitness > self.global_best_individual.fitness):
            self.global_best_individual = self._clone_individual(individual)
            self.global_best_individual.metadata['found_generation'] = self.generation
            self.global_best_individual.metadata['found_island'] = individual.metadata.get('island_id', -1)

    def _calculate_island_diversity(self, island: Island) -> float:
        """Calculate diversity score for an island."""
        if island.population.size() == 0:
            return 0.0

        # Use unique program count as diversity metric
        unique_programs = len({ind.id for ind in island.population.individuals})
        return unique_programs / island.population.size()

    def _update_island_metrics(self) -> None:
        """Update metrics for all islands."""
        for island in self.islands:
            # Update diversity scores
            self.island_diversity_scores[island.id] = self._calculate_island_diversity(island)

            # Identify island specialization based on best individuals
            best_ind = island.population.best_individual
            if best_ind:
                # Analyze program characteristics
                program_length = best_ind.program_length()
                if program_length < 10:
                    specialization = "simple"
                elif program_length > 30:
                    specialization = "complex"
                else:
                    specialization = "balanced"
                self.island_specialization[island.id] = specialization

    def _check_convergence(self) -> bool:
        """Check if evolution should stop."""
        # Global convergence criteria
        if self.global_best_individual and self.global_best_individual.fitness >= 0.95:
            return True

        # Check if all islands are stagnant
        stagnant_islands = sum(1 for island in self.islands if island.is_stagnant())
        if stagnant_islands >= self.num_islands * 0.8:  # 80% islands stagnant
            return True

        return False

    def _adapt_island_parameters(self) -> None:
        """Adapt island parameters based on performance."""
        if not self.migration_policy.adaptive:
            return

        for island in self.islands:
            # Get island performance metrics
            diversity = self.island_diversity_scores.get(island.id, 0.0)

            # Adapt mutation rate based on diversity
            if diversity < 0.3:  # Low diversity
                island.evolution_engine.config.genetic_operators.mutation.base_rate *= 1.1
            elif diversity > 0.7:  # High diversity
                island.evolution_engine.config.genetic_operators.mutation.base_rate *= 0.9

            # Adapt based on fitness stagnation
            if island.is_stagnant():
                # Increase exploration
                island.evolution_engine.config.genetic_operators.mutation.base_rate *= 1.2
                island.evolution_engine.config.genetic_operators.crossover.rate *= 1.1

    def _track_migration_success(self, migration_events: list[dict[str, Any]]) -> None:
        """Track success of migrations."""
        # Calculate average fitness improvement from migrations
        # This would require tracking fitness changes post-migration
        # For now, track migration frequency and patterns
        for event in migration_events:
            source_island = self.islands[event['from']]
            target_island = self.islands[event['to']]

            # Add to migration history
            migration_record = {
                'generation': self.generation,
                'from_island': event['from'],
                'to_island': event['to'],
                'migrant_count': event['count'],
                'avg_migrant_fitness': event['avg_fitness'],
                'source_avg_fitness': source_island.population.average_fitness(),
                'target_avg_fitness': target_island.population.average_fitness()
            }

            source_island.migration_history.append(migration_record)
            target_island.migration_history.append(migration_record)

    def _collect_evolution_stats(self) -> dict[str, Any]:
        """Collect comprehensive evolution statistics."""
        stats = {
            'num_islands': self.num_islands,
            'total_generations': self.generation,
            'migration_count': self.migration_count,
            'best_fitness': self.global_best_individual.fitness if self.global_best_individual else 0.0,
            'island_stats': {},
            'migration_stats': {
                'total_migrations': self.migration_count * self.num_islands,
                'migration_success_rate': self.migration_success_rate
            }
        }

        # Collect per-island statistics
        for island in self.islands:
            island_stat = {
                'best_fitness': island.population.best_individual.fitness if island.population.best_individual else 0.0,
                'average_fitness': island.population.average_fitness(),
                'diversity_score': self.island_diversity_scores.get(island.id, 0.0),
                'population_size': island.population.size(),
                'specialization': self.island_specialization.get(island.id, 'unknown'),
                'fitness_trend': island.fitness_trend[-10:],  # Last 10 generations
                'stagnant': island.is_stagnant()
            }
            stats['island_stats'][island.id] = island_stat

        # Analyze migration patterns
        migration_matrix = [[0] * self.num_islands for _ in range(self.num_islands)]
        for island in self.islands:
            for migration in island.migration_history:
                if migration['from_island'] == island.id:
                    migration_matrix[island.id][migration['to_island']] += migration['migrant_count']

        stats['migration_stats']['migration_matrix'] = migration_matrix

        return stats
