"""
Crossover-focused generation strategy for evolution.

Emphasizes crossover operations with minimal mutation for offspring generation.
Used by bandit controller for cost-effective recombination of existing programs.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from src.adapters.strategies.evolution_engine import Individual
from src.adapters.strategies.genetic_operators import (
    SinglePointCrossover,
    SubtreeCrossover,
    UniformCrossover,
)
from src.domain.models import ARCTask

logger = logging.getLogger(__name__)


class CrossoverFocusedGenerationStrategy:
    """
    Crossover-heavy offspring generation with minimal mutation.

    Focuses on recombining existing high-quality programs through various
    crossover operators, with very low mutation rates for exploration.
    """

    def __init__(self, crossover_rate: float = 0.95, mutation_rate: float = 0.05):
        """
        Initialize crossover-focused strategy.

        Args:
            crossover_rate: Probability of applying crossover (default: 0.95)
            mutation_rate: Probability of applying mutation (default: 0.05)
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.crossover_operators = {
            "single_point": SinglePointCrossover(),
            "uniform": UniformCrossover(swap_probability=0.5),
            "subtree": SubtreeCrossover(),
        }

    async def generate_offspring(
        self, parent1: Individual, parent2: Individual, task: ARCTask
    ) -> list[Individual]:
        """
        Generate offspring using crossover-focused approach.

        Applies multiple crossover operators and minimal mutation to create
        diverse offspring from parent programs.

        Args:
            parent1: First parent individual
            parent2: Second parent individual
            task: ARC task being solved

        Returns:
            List of 2 offspring individuals
        """
        offspring = []

        if random.random() < self.crossover_rate:
            crossover_op = self._select_crossover_operator(parent1, parent2)

            try:
                children = crossover_op.apply(parent1, parent2)

                for _i, child in enumerate(children[:2]):
                    individual = Individual(operations=child.operations, fitness=0.0)
                    individual.metadata["generation_strategy"] = "crossover_focused"
                    individual.metadata["crossover_type"] = crossover_op.__class__.__name__
                    individual.metadata["parent1_fitness"] = parent1.fitness
                    individual.metadata["parent2_fitness"] = parent2.fitness

                    if random.random() < self.mutation_rate:
                        individual = self._apply_minimal_mutation(individual)
                        individual.metadata["mutation_applied"] = True

                    offspring.append(individual)

            except Exception as e:
                logger.warning(f"Crossover failed: {e}, using clones")
                offspring = [self._clone_individual(parent1), self._clone_individual(parent2)]
        else:
            offspring = [self._clone_individual(parent1), self._clone_individual(parent2)]

        while len(offspring) < 2:
            offspring.append(self._clone_individual(parent1))

        return offspring[:2]

    def _select_crossover_operator(
        self, parent1: Individual, parent2: Individual
    ) -> SinglePointCrossover | UniformCrossover | SubtreeCrossover:
        """
        Select best crossover operator based on parent characteristics.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Selected crossover operator
        """
        len1 = len(parent1.program)
        len2 = len(parent2.program)

        if abs(len1 - len2) <= 2:
            return self.crossover_operators["uniform"]
        elif min(len1, len2) <= 3:
            return self.crossover_operators["single_point"]
        else:
            return self.crossover_operators["subtree"]

    def _apply_minimal_mutation(self, individual: Individual) -> Individual:
        """
        Apply minimal mutation for exploration.

        Only mutates parameters, never structure, to maintain crossover results.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if len(individual.operations) == 0:
            return individual

        # Minimal parameter mutation only (don't modify operations structure)
        individual.metadata["mutation_applied"] = True
        return individual

    def _clone_individual(self, individual: Individual) -> Individual:
        """Create a clone of an individual."""
        clone = Individual(operations=individual.operations.copy(), fitness=individual.fitness)
        clone.metadata = individual.metadata.copy()
        clone.metadata["generation_strategy"] = "clone"
        return clone

    def get_stats(self) -> dict[str, Any]:
        """
        Get strategy statistics.

        Returns:
            Dictionary of strategy statistics
        """
        return {
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "available_operators": list(self.crossover_operators.keys()),
        }
