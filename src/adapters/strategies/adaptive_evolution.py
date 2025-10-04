"""
Advanced adaptive mutation strategies for evolution engine.

This module implements Task 7.2: Adaptive mutation rates based on fitness progress.
It provides sophisticated mutation rate adaptation at both individual and population levels.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.adapters.strategies.evolution_engine import Individual, Population
from src.adapters.strategies.genetic_operators import MutationOperator


@dataclass
class FitnessHistory:
    """Tracks fitness history for adaptation decisions."""
    max_history: int = 20
    history: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, fitness: float) -> None:
        """Add fitness value to history."""
        self.history.append(fitness)

    def get_trend(self, window: int = 5) -> float:
        """
        Calculate fitness trend over recent window.

        Returns:
            Trend value: positive for improvement, negative for decline
        """
        if len(self.history) < window:
            return 0.0

        recent = list(self.history)[-window:]
        # Linear regression slope
        x_mean = (window - 1) / 2
        y_mean = sum(recent) / window

        numerator = sum((i - x_mean) * (y - y_mean)
                       for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(window))

        return numerator / denominator if denominator > 0 else 0.0

    def get_variance(self) -> float:
        """Calculate fitness variance."""
        if len(self.history) < 2:
            return 0.0

        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        return variance

    def is_stagnant(self, threshold: float = 0.001, window: int = 10) -> bool:
        """Check if fitness is stagnant."""
        if len(self.history) < window:
            return False

        recent = list(self.history)[-window:]
        return max(recent) - min(recent) < threshold


@dataclass
class MutationRateSchedule:
    """Defines mutation rate adaptation schedule."""
    base_rate: float = 0.1
    min_rate: float = 0.01
    max_rate: float = 0.5
    adaptation_speed: float = 0.1

    # Different strategies
    strategy: str = "fitness_based"  # fitness_based, time_decay, cyclic, adaptive_landscape

    # Strategy-specific parameters
    decay_factor: float = 0.99
    cycle_period: int = 20
    landscape_threshold: float = 0.1


class AdaptiveMutationController:
    """
    Controls adaptive mutation rates based on various fitness indicators.

    This implements sophisticated mutation rate adaptation that considers:
    - Individual fitness history
    - Population diversity
    - Fitness landscape characteristics
    - Convergence indicators
    """

    def __init__(self, schedule: MutationRateSchedule | None = None):
        """Initialize adaptive mutation controller."""
        self.schedule = schedule or MutationRateSchedule()
        self.population_history = FitnessHistory()
        self.individual_histories: dict[str, FitnessHistory] = {}
        self.generation_count = 0
        self.diversity_history: deque[float] = deque(maxlen=50)

        # Track mutation effectiveness
        self.mutation_success_rates: dict[str, float] = {}
        self.operator_performance: dict[str, deque[float]] = {}

        # Landscape analysis
        self.fitness_landscape_ruggedness = 0.0
        self.local_optima_detected = 0

    def update_population_stats(self, population: Population) -> None:
        """Update controller with population statistics."""
        self.generation_count += 1

        # Update population fitness history
        avg_fitness = population.average_fitness()
        self.population_history.add(avg_fitness)

        # Update diversity
        unique_programs = len({ind.id for ind in population.individuals})
        diversity = unique_programs / population.size() if population.size() > 0 else 0
        self.diversity_history.append(diversity)

        # Analyze fitness landscape
        self._analyze_fitness_landscape(population)

    def get_mutation_rate(self, individual: Individual) -> float:
        """
        Get adaptive mutation rate for an individual.

        Args:
            individual: The individual to mutate

        Returns:
            Adapted mutation rate
        """
        ind_id = individual.id

        # Initialize history if needed
        if ind_id not in self.individual_histories:
            self.individual_histories[ind_id] = FitnessHistory()

        # Update individual history
        self.individual_histories[ind_id].add(individual.fitness)

        # Calculate rate based on strategy
        if self.schedule.strategy == "fitness_based":
            return self._fitness_based_rate(individual)
        elif self.schedule.strategy == "time_decay":
            return self._time_decay_rate()
        elif self.schedule.strategy == "cyclic":
            return self._cyclic_rate()
        elif self.schedule.strategy == "adaptive_landscape":
            return self._landscape_adaptive_rate(individual)
        else:
            return self.schedule.base_rate

    def _fitness_based_rate(self, individual: Individual) -> float:
        """Calculate fitness-based adaptive mutation rate."""
        # Get individual and population trends
        ind_history = self.individual_histories.get(individual.id)
        ind_trend = ind_history.get_trend() if ind_history else 0.0
        pop_trend = self.population_history.get_trend()

        # Calculate stagnation factors
        ind_stagnant = ind_history.is_stagnant() if ind_history else False
        pop_stagnant = self.population_history.is_stagnant()

        # Base rate adjustment
        rate = self.schedule.base_rate

        # Individual-level adaptation
        if ind_stagnant:
            # Increase rate for stagnant individuals
            rate *= 1.5
        elif ind_trend < 0:
            # Increase rate for declining fitness
            rate *= 1.2
        elif ind_trend > 0.01:
            # Decrease rate for improving individuals
            rate *= 0.8

        # Population-level adaptation
        if pop_stagnant:
            # Global stagnation - increase exploration
            rate *= 1.3
        elif pop_trend < -0.001:
            # Population declining - increase diversity
            rate *= 1.1

        # Diversity-based adjustment
        if self.diversity_history:
            current_diversity = self.diversity_history[-1]
            if current_diversity < 0.3:
                # Low diversity - increase mutation
                rate *= 1.2
            elif current_diversity > 0.7:
                # High diversity - can reduce mutation
                rate *= 0.9

        # Fitness-based adjustment
        if individual.fitness < 0.2:
            # Poor fitness - higher mutation
            rate *= 1.3
        elif individual.fitness > 0.8:
            # Good fitness - preserve with lower mutation
            rate *= 0.7

        # Apply bounds
        return max(self.schedule.min_rate, min(self.schedule.max_rate, rate))

    def _time_decay_rate(self) -> float:
        """Calculate time-decay mutation rate."""
        # Exponential decay over generations
        rate = self.schedule.base_rate * (
            self.schedule.decay_factor ** self.generation_count
        )
        return max(self.schedule.min_rate, rate)

    def _cyclic_rate(self) -> float:
        """Calculate cyclic mutation rate."""
        # Sinusoidal variation
        cycle_position = self.generation_count % self.schedule.cycle_period
        phase = 2 * math.pi * cycle_position / self.schedule.cycle_period

        # Oscillate between min and max
        amplitude = (self.schedule.max_rate - self.schedule.min_rate) / 2
        center = (self.schedule.max_rate + self.schedule.min_rate) / 2

        return center + amplitude * math.sin(phase)

    def _landscape_adaptive_rate(self, individual: Individual) -> float:
        """Calculate mutation rate based on fitness landscape analysis."""
        rate = self.schedule.base_rate

        # Adjust based on landscape ruggedness
        if self.fitness_landscape_ruggedness > 0.5:
            # Rugged landscape - need more exploration
            rate *= 1.4
        elif self.fitness_landscape_ruggedness < 0.2:
            # Smooth landscape - can use smaller steps
            rate *= 0.8

        # Adjust based on local optima
        if self.local_optima_detected > 3:
            # Many local optima - increase exploration
            rate *= 1.3

        # Distance from best fitness
        fitness_gap = 1.0 - individual.fitness

        if fitness_gap > 0.5:
            # Far from optimal - larger mutations
            rate *= (1 + fitness_gap * 0.5)

        return max(self.schedule.min_rate, min(self.schedule.max_rate, rate))

    def _analyze_fitness_landscape(self, population: Population) -> None:
        """Analyze fitness landscape characteristics."""
        if population.size() < 10:
            return

        # Calculate fitness variance
        fitness_values = [ind.fitness for ind in population.individuals]
        self._calculate_variance(fitness_values)

        # Estimate ruggedness by looking at fitness differences
        sorted_fitness = sorted(fitness_values)
        differences = [sorted_fitness[i+1] - sorted_fitness[i]
                      for i in range(len(sorted_fitness)-1)]

        if differences:
            # High variance in differences indicates rugged landscape
            diff_variance = self._calculate_variance(differences)
            self.fitness_landscape_ruggedness = min(1.0, diff_variance * 10)

        # Detect potential local optima
        if len(self.population_history.history) > 10:
            recent_best = max(list(self.population_history.history)[-10:])
            current_best = population.best_individual.fitness if population.best_individual else 0

            # If best fitness hasn't improved much, might be local optimum
            if abs(current_best - recent_best) < 0.01:
                self.local_optima_detected += 1

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def adapt_operator_rates(self, operator_name: str, success: bool) -> None:
        """
        Track and adapt specific operator success rates.

        Args:
            operator_name: Name of the mutation operator
            success: Whether the mutation improved fitness
        """
        if operator_name not in self.operator_performance:
            self.operator_performance[operator_name] = deque(maxlen=100)

        self.operator_performance[operator_name].append(1.0 if success else 0.0)

        # Update success rate
        if len(self.operator_performance[operator_name]) > 10:
            success_rate = sum(self.operator_performance[operator_name]) / len(
                self.operator_performance[operator_name]
            )
            self.mutation_success_rates[operator_name] = success_rate

    def get_operator_weights(self) -> dict[str, float]:
        """
        Get weights for different mutation operators based on performance.

        Returns:
            Dictionary mapping operator names to selection weights
        """
        if not self.mutation_success_rates:
            # No data yet, use uniform weights
            return {}

        # Calculate weights based on success rates
        weights = {}
        for operator, success_rate in self.mutation_success_rates.items():
            # Higher weight for more successful operators
            # But maintain minimum weight to avoid complete elimination
            weights[operator] = max(0.1, success_rate)

        return weights

    def get_adaptation_summary(self) -> dict[str, Any]:
        """Get summary of adaptation state."""
        return {
            'generation': self.generation_count,
            'population_trend': self.population_history.get_trend(),
            'population_stagnant': self.population_history.is_stagnant(),
            'average_diversity': (
                sum(self.diversity_history) / len(self.diversity_history)
                if self.diversity_history else 0
            ),
            'landscape_ruggedness': self.fitness_landscape_ruggedness,
            'local_optima_detected': self.local_optima_detected,
            'operator_success_rates': self.mutation_success_rates.copy(),
            'current_strategy': self.schedule.strategy
        }


class EnhancedAdaptiveMutation(MutationOperator):
    """
    Enhanced adaptive mutation operator using sophisticated adaptation.

    This replaces the basic AdaptiveMutation with more advanced features
    for Task 7.2.
    """

    def __init__(
        self,
        controller: AdaptiveMutationController | None = None,
        base_operators: list[MutationOperator] | None = None
    ):
        """
        Initialize enhanced adaptive mutation.

        Args:
            controller: Adaptive mutation controller
            base_operators: List of base mutation operators to use
        """
        self.controller = controller or AdaptiveMutationController()
        self.base_operators = base_operators or []
        self.last_fitness_by_individual: dict[str, float] = {}

    def get_name(self) -> str:
        return "enhanced_adaptive_mutation"

    def apply(self, individual: Individual) -> list[Individual]:
        """Apply adaptive mutation with sophisticated rate control."""
        # Get adaptive mutation rate
        mutation_rate = self.controller.get_mutation_rate(individual)

        # Decide whether to mutate based on rate
        if random.random() > mutation_rate:
            # No mutation - return copy
            return [self._copy_individual(individual)]

        # Select operator based on performance weights
        operator = self._select_operator()

        # Apply mutation
        mutated = operator.apply(individual)

        # Track mutation metadata
        if mutated:
            result = mutated[0]
            result.metadata['mutation_rate'] = mutation_rate
            result.metadata['mutation_operator'] = operator.get_name()
            result.metadata['mutation_strategy'] = self.controller.schedule.strategy

            # Track for success measurement (will be updated after evaluation)
            self.last_fitness_by_individual[individual.id] = individual.fitness

            return mutated

        return [self._copy_individual(individual)]

    def _select_operator(self) -> MutationOperator:
        """Select mutation operator based on performance."""
        weights = self.controller.get_operator_weights()

        if not weights or not self.base_operators:
            # Random selection
            return random.choice(self.base_operators) if self.base_operators else None

        # Weighted selection based on success rates
        operator_names = [op.get_name() for op in self.base_operators]
        operator_weights = [weights.get(name, 0.5) for name in operator_names]

        # Normalize weights
        total_weight = sum(operator_weights)
        if total_weight > 0:
            operator_weights = [w / total_weight for w in operator_weights]

            # Weighted random selection
            return random.choices(self.base_operators, weights=operator_weights)[0]

        return random.choice(self.base_operators)

    def _copy_individual(self, individual: Individual) -> Individual:
        """Create a copy of an individual."""
        from copy import deepcopy
        return deepcopy(individual)

    def update_success(self, individual: Individual) -> None:
        """
        Update mutation success tracking after fitness evaluation.

        Should be called after the mutated individual's fitness is evaluated.
        """
        # Check if this individual was mutated by us
        if individual.id in self.last_fitness_by_individual:
            parent_fitness = self.last_fitness_by_individual[individual.id]
            fitness_improvement = individual.fitness - parent_fitness
            success = fitness_improvement > 0

            # Update controller with operator success
            operator_name = individual.metadata.get('mutation_operator', 'unknown')
            self.controller.adapt_operator_rates(operator_name, success)

            # Clean up tracking
            del self.last_fitness_by_individual[individual.id]


def create_adaptive_mutation_schedule(
    strategy: str = "fitness_based",
    base_rate: float = 0.1,
    **kwargs
) -> MutationRateSchedule:
    """
    Factory function to create mutation rate schedules.

    Args:
        strategy: Adaptation strategy name
        base_rate: Base mutation rate
        **kwargs: Additional strategy-specific parameters

    Returns:
        Configured mutation rate schedule
    """
    schedule = MutationRateSchedule(
        base_rate=base_rate,
        strategy=strategy
    )

    # Apply strategy-specific defaults
    if strategy == "time_decay":
        schedule.decay_factor = kwargs.get('decay_factor', 0.99)
    elif strategy == "cyclic":
        schedule.cycle_period = kwargs.get('cycle_period', 20)
    elif strategy == "adaptive_landscape":
        schedule.landscape_threshold = kwargs.get('landscape_threshold', 0.1)

    # Apply common parameters
    schedule.min_rate = kwargs.get('min_rate', 0.01)
    schedule.max_rate = kwargs.get('max_rate', 0.5)
    schedule.adaptation_speed = kwargs.get('adaptation_speed', 0.1)

    return schedule
