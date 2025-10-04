"""
Co-evolution of programs and fitness functions.

This module implements Task 7.4: Add co-evolution of programs and fitness functions.
It evolves both solution programs and evaluation metrics simultaneously.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.adapters.strategies.evolution_engine import Individual, Population
from src.domain.models import ARCTask, Grid


@dataclass
class FitnessComponent:
    """Represents a component of a fitness function."""
    name: str
    weight: float
    evaluator: Callable[[Grid, Grid], float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, predicted: Grid, target: Grid) -> float:
        """Evaluate this component of fitness."""
        try:
            return self.evaluator(predicted, target) * self.weight
        except Exception:
            return 0.0


@dataclass
class FitnessFunction:
    """
    Evolvable fitness function composed of weighted components.

    This represents an individual in the fitness function population.
    """
    components: list[FitnessComponent]
    fitness_score: float = 0.0  # Meta-fitness: how good is this fitness function
    age: int = 0
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize ID after creation."""
        if not self.id:
            import hashlib
            comp_str = "".join([f"{c.name}{c.weight}" for c in self.components])
            self.id = hashlib.md5(comp_str.encode()).hexdigest()[:12]

    def evaluate(self, predicted: Grid, target: Grid) -> float:
        """Evaluate fitness of a prediction."""
        total_fitness = 0.0
        total_weight = sum(c.weight for c in self.components)

        if total_weight == 0:
            return 0.0

        for component in self.components:
            total_fitness += component.evaluate(predicted, target)

        # Normalize by total weight
        return total_fitness / total_weight

    def normalize_weights(self) -> None:
        """Normalize component weights to sum to 1.0."""
        total = sum(c.weight for c in self.components)
        if total > 0:
            for c in self.components:
                c.weight /= total


class FitnessComponentLibrary:
    """Library of available fitness components."""

    @staticmethod
    def get_available_components() -> list[FitnessComponent]:
        """Get all available fitness components."""
        return [
            FitnessComponent(
                name="exact_match",
                weight=1.0,
                evaluator=FitnessComponentLibrary._exact_match
            ),
            FitnessComponent(
                name="pixel_accuracy",
                weight=1.0,
                evaluator=FitnessComponentLibrary._pixel_accuracy
            ),
            FitnessComponent(
                name="shape_match",
                weight=0.5,
                evaluator=FitnessComponentLibrary._shape_match
            ),
            FitnessComponent(
                name="color_distribution",
                weight=0.3,
                evaluator=FitnessComponentLibrary._color_distribution
            ),
            FitnessComponent(
                name="pattern_similarity",
                weight=0.7,
                evaluator=FitnessComponentLibrary._pattern_similarity
            ),
            FitnessComponent(
                name="structural_similarity",
                weight=0.8,
                evaluator=FitnessComponentLibrary._structural_similarity
            ),
            FitnessComponent(
                name="edge_preservation",
                weight=0.4,
                evaluator=FitnessComponentLibrary._edge_preservation
            ),
            FitnessComponent(
                name="symmetry_score",
                weight=0.3,
                evaluator=FitnessComponentLibrary._symmetry_score
            ),
            FitnessComponent(
                name="complexity_penalty",
                weight=-0.1,  # Negative weight for penalty
                evaluator=FitnessComponentLibrary._complexity_penalty
            )
        ]

    @staticmethod
    def _exact_match(predicted: Grid, target: Grid) -> float:
        """1.0 if exact match, 0.0 otherwise."""
        if predicted.shape != target.shape:
            return 0.0

        for i in range(predicted.shape[0]):
            for j in range(predicted.shape[1]):
                if predicted.data[i][j] != target.data[i][j]:
                    return 0.0
        return 1.0

    @staticmethod
    def _pixel_accuracy(predicted: Grid, target: Grid) -> float:
        """Percentage of matching pixels."""
        if predicted.shape != target.shape:
            return 0.0

        total = predicted.shape[0] * predicted.shape[1]
        if total == 0:
            return 0.0

        correct = 0
        for i in range(predicted.shape[0]):
            for j in range(predicted.shape[1]):
                if predicted.data[i][j] == target.data[i][j]:
                    correct += 1

        return correct / total

    @staticmethod
    def _shape_match(predicted: Grid, target: Grid) -> float:
        """Score based on shape similarity."""
        pred_h, pred_w = predicted.shape
        targ_h, targ_w = target.shape

        # Exact shape match
        if pred_h == targ_h and pred_w == targ_w:
            return 1.0

        # Partial credit for aspect ratio
        pred_ratio = pred_w / pred_h if pred_h > 0 else 0
        targ_ratio = targ_w / targ_h if targ_h > 0 else 0

        if targ_ratio > 0:
            ratio_score = min(pred_ratio / targ_ratio, targ_ratio / pred_ratio)
        else:
            ratio_score = 0.0

        # Partial credit for size
        pred_size = pred_h * pred_w
        targ_size = targ_h * targ_w

        if targ_size > 0:
            size_score = min(pred_size / targ_size, targ_size / pred_size)
        else:
            size_score = 0.0

        return (ratio_score + size_score) / 2

    @staticmethod
    def _color_distribution(predicted: Grid, target: Grid) -> float:
        """Compare color histograms."""
        # Get color counts
        pred_colors = {}
        for row in predicted.data:
            for color in row:
                pred_colors[color] = pred_colors.get(color, 0) + 1

        targ_colors = {}
        for row in target.data:
            for color in row:
                targ_colors[color] = targ_colors.get(color, 0) + 1

        # Normalize to distributions
        pred_total = sum(pred_colors.values())
        targ_total = sum(targ_colors.values())

        if pred_total == 0 or targ_total == 0:
            return 0.0

        # Calculate overlap
        all_colors = set(pred_colors.keys()) | set(targ_colors.keys())
        overlap = 0.0

        for color in all_colors:
            pred_freq = pred_colors.get(color, 0) / pred_total
            targ_freq = targ_colors.get(color, 0) / targ_total
            overlap += min(pred_freq, targ_freq)

        return overlap

    @staticmethod
    def _pattern_similarity(predicted: Grid, target: Grid) -> float:
        """Measure pattern similarity using local features."""
        # Simple 2x2 pattern matching
        if predicted.shape != target.shape:
            return 0.0

        h, w = predicted.shape
        if h < 2 or w < 2:
            return FitnessComponentLibrary._pixel_accuracy(predicted, target)

        matches = 0
        total = 0

        for i in range(h - 1):
            for j in range(w - 1):
                # Extract 2x2 patterns
                pred_pattern = (
                    predicted.data[i][j],
                    predicted.data[i][j+1],
                    predicted.data[i+1][j],
                    predicted.data[i+1][j+1]
                )
                targ_pattern = (
                    target.data[i][j],
                    target.data[i][j+1],
                    target.data[i+1][j],
                    target.data[i+1][j+1]
                )

                if pred_pattern == targ_pattern:
                    matches += 1
                total += 1

        return matches / total if total > 0 else 0.0

    @staticmethod
    def _structural_similarity(predicted: Grid, target: Grid) -> float:
        """Measure structural similarity (connected components, etc)."""
        # Simplified: check if non-zero pixels form similar patterns
        if predicted.shape != target.shape:
            return 0.0

        # Count connected regions (simplified)
        pred_regions = FitnessComponentLibrary._count_regions(predicted)
        targ_regions = FitnessComponentLibrary._count_regions(target)

        if targ_regions == 0:
            return 1.0 if pred_regions == 0 else 0.0

        region_score = min(pred_regions / targ_regions, targ_regions / pred_regions)

        # Also consider density
        pred_density = sum(1 for row in predicted.data for val in row if val != 0) / (predicted.shape[0] * predicted.shape[1])
        targ_density = sum(1 for row in target.data for val in row if val != 0) / (target.shape[0] * target.shape[1])

        if targ_density > 0:
            density_score = min(pred_density / targ_density, targ_density / pred_density)
        else:
            density_score = 1.0 if pred_density == 0 else 0.0

        return (region_score + density_score) / 2

    @staticmethod
    def _count_regions(grid: Grid) -> int:
        """Count approximate number of regions (simplified)."""
        # Very simplified: count transitions between zero and non-zero
        transitions = 0
        h, w = grid.shape

        # Horizontal transitions
        for i in range(h):
            for j in range(w - 1):
                if (grid.data[i][j] == 0) != (grid.data[i][j+1] == 0):
                    transitions += 1

        # Vertical transitions
        for i in range(h - 1):
            for j in range(w):
                if (grid.data[i][j] == 0) != (grid.data[i+1][j] == 0):
                    transitions += 1

        # Approximate region count
        return max(1, transitions // 4)

    @staticmethod
    def _edge_preservation(predicted: Grid, target: Grid) -> float:
        """Measure how well edges are preserved."""
        if predicted.shape != target.shape:
            return 0.0

        h, w = predicted.shape
        edge_matches = 0
        total_edges = 0

        # Check horizontal edges
        for i in range(h):
            for j in range(w - 1):
                if target.data[i][j] != target.data[i][j+1]:
                    total_edges += 1
                    if predicted.data[i][j] != predicted.data[i][j+1]:
                        edge_matches += 1

        # Check vertical edges
        for i in range(h - 1):
            for j in range(w):
                if target.data[i][j] != target.data[i+1][j]:
                    total_edges += 1
                    if predicted.data[i][j] != predicted.data[i+1][j]:
                        edge_matches += 1

        return edge_matches / total_edges if total_edges > 0 else 1.0

    @staticmethod
    def _symmetry_score(predicted: Grid, target: Grid) -> float:
        """Measure symmetry preservation."""
        # Check if target has symmetry and if predicted preserves it
        targ_h_sym = FitnessComponentLibrary._has_horizontal_symmetry(target)
        targ_v_sym = FitnessComponentLibrary._has_vertical_symmetry(target)

        score = 0.0
        count = 0

        if targ_h_sym:
            count += 1
            if FitnessComponentLibrary._has_horizontal_symmetry(predicted):
                score += 1.0

        if targ_v_sym:
            count += 1
            if FitnessComponentLibrary._has_vertical_symmetry(predicted):
                score += 1.0

        # If target has no symmetry, check if predicted doesn't add false symmetry
        if count == 0:
            if not FitnessComponentLibrary._has_horizontal_symmetry(predicted) and \
               not FitnessComponentLibrary._has_vertical_symmetry(predicted):
                return 1.0
            else:
                return 0.5

        return score / count

    @staticmethod
    def _has_horizontal_symmetry(grid: Grid) -> bool:
        """Check for horizontal symmetry."""
        h, w = grid.shape
        for i in range(h // 2):
            for j in range(w):
                if grid.data[i][j] != grid.data[h-1-i][j]:
                    return False
        return True

    @staticmethod
    def _has_vertical_symmetry(grid: Grid) -> bool:
        """Check for vertical symmetry."""
        h, w = grid.shape
        for i in range(h):
            for j in range(w // 2):
                if grid.data[i][j] != grid.data[i][w-1-j]:
                    return False
        return True

    @staticmethod
    def _complexity_penalty(predicted: Grid, target: Grid) -> float:
        """Penalize overly complex predictions."""
        # Measure complexity as number of color transitions
        h, w = predicted.shape
        transitions = 0

        for i in range(h):
            for j in range(w - 1):
                if predicted.data[i][j] != predicted.data[i][j+1]:
                    transitions += 1

        for i in range(h - 1):
            for j in range(w):
                if predicted.data[i][j] != predicted.data[i+1][j]:
                    transitions += 1

        # Normalize by grid size
        max_transitions = 2 * h * w - h - w
        if max_transitions > 0:
            return transitions / max_transitions
        return 0.0


class CoevolutionEngine:
    """
    Co-evolves programs and fitness functions.

    This engine maintains two populations:
    1. Program population (solutions)
    2. Fitness function population (evaluators)

    They evolve together with mutual evaluation.
    """

    def __init__(
        self,
        program_pop_size: int = 100,
        fitness_pop_size: int = 20,
        elite_ratio: float = 0.1
    ):
        """
        Initialize co-evolution engine.

        Args:
            program_pop_size: Size of program population
            fitness_pop_size: Size of fitness function population
            elite_ratio: Proportion of elite individuals preserved
        """
        self.program_pop_size = program_pop_size
        self.fitness_pop_size = fitness_pop_size
        self.elite_ratio = elite_ratio

        # Populations
        self.program_population: list[Individual] = []
        self.fitness_population: list[FitnessFunction] = []

        # Component library
        self.component_library = FitnessComponentLibrary.get_available_components()

        # Tracking
        self.generation = 0
        self.best_program_fitness_history: list[float] = []
        self.fitness_diversity_history: list[float] = []

    def initialize_populations(self) -> None:
        """Initialize both populations."""
        # Initialize fitness functions
        self._initialize_fitness_population()

        # Program population will be initialized by evolution engine
        self.program_population = []

    def _initialize_fitness_population(self) -> None:
        """Initialize diverse fitness functions."""
        self.fitness_population = []

        # Create diverse fitness functions
        for i in range(self.fitness_pop_size):
            if i == 0:
                # Always include exact match as baseline
                components = [
                    FitnessComponent("exact_match", 1.0, FitnessComponentLibrary._exact_match)
                ]
            elif i == 1:
                # Include pixel accuracy
                components = [
                    FitnessComponent("pixel_accuracy", 1.0, FitnessComponentLibrary._pixel_accuracy)
                ]
            else:
                # Random combinations
                num_components = random.randint(2, 5)
                selected_indices = random.sample(
                    range(len(self.component_library)),
                    min(num_components, len(self.component_library))
                )

                components = []
                for idx in selected_indices:
                    comp = self.component_library[idx]
                    # Randomize weight
                    weight = random.uniform(0.1, 1.0)
                    if comp.name == "complexity_penalty":
                        weight *= -1  # Keep it negative

                    components.append(
                        FitnessComponent(
                            name=comp.name,
                            weight=weight,
                            evaluator=comp.evaluator
                        )
                    )

            fitness_func = FitnessFunction(components=components)
            fitness_func.normalize_weights()
            self.fitness_population.append(fitness_func)

    def coevolve_generation(
        self,
        program_population: Population,
        task: ARCTask
    ) -> tuple[Population, FitnessFunction]:
        """
        Perform one generation of co-evolution.

        Args:
            program_population: Current program population
            task: ARC task being solved

        Returns:
            Updated program population and best fitness function
        """
        self.generation += 1
        self.program_population = program_population.individuals

        # Step 1: Evaluate programs using multiple fitness functions
        self._evaluate_programs_with_ensemble(task)

        # Step 2: Evaluate fitness functions based on program performance
        self._evaluate_fitness_functions(task)

        # Step 3: Evolve fitness function population
        self._evolve_fitness_population()

        # Step 4: Update program fitness based on best fitness functions
        best_fitness_func = self._get_best_fitness_function()
        self._update_program_fitness(best_fitness_func, task)

        # Update tracking
        if program_population.best_individual:
            self.best_program_fitness_history.append(
                program_population.best_individual.fitness
            )

        diversity = self._calculate_fitness_diversity()
        self.fitness_diversity_history.append(diversity)

        return program_population, best_fitness_func

    def _evaluate_programs_with_ensemble(self, task: ARCTask) -> None:
        """Evaluate programs using ensemble of fitness functions."""
        # For each program, store evaluations from all fitness functions
        for program in self.program_population:
            program.metadata["fitness_evaluations"] = {}

            # Skip if no execution result
            if "execution_result" not in program.metadata:
                continue

            predicted = program.metadata["execution_result"]
            target = task.train_pairs[0].output  # Use first example

            # Evaluate with each fitness function
            for fitness_func in self.fitness_population:
                try:
                    score = fitness_func.evaluate(predicted, target)
                    program.metadata["fitness_evaluations"][fitness_func.id] = score
                except Exception:
                    program.metadata["fitness_evaluations"][fitness_func.id] = 0.0

    def _evaluate_fitness_functions(self, task: ARCTask) -> None:
        """Evaluate fitness functions based on their ability to identify good programs."""
        # For each fitness function, calculate meta-fitness
        for fitness_func in self.fitness_population:
            # Meta-fitness based on:
            # 1. Correlation with known good programs
            # 2. Diversity of rankings
            # 3. Consistency across examples

            scores = []
            for program in self.program_population:
                if fitness_func.id in program.metadata.get("fitness_evaluations", {}):
                    scores.append(program.metadata["fitness_evaluations"][fitness_func.id])

            if not scores:
                fitness_func.fitness_score = 0.0
                continue

            # Calculate meta-fitness components
            # 1. Range: Good fitness functions should differentiate programs
            score_range = max(scores) - min(scores) if len(scores) > 1 else 0

            # 2. Distribution: Should not give all programs the same score
            score_variance = self._calculate_variance(scores)

            # 3. Correlation with ground truth (if available)
            correlation_score = self._calculate_correlation_score(fitness_func, task)

            # Combine into meta-fitness
            fitness_func.fitness_score = (
                0.3 * min(1.0, score_range) +
                0.3 * min(1.0, score_variance * 10) +
                0.4 * correlation_score
            )

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _calculate_correlation_score(
        self,
        fitness_func: FitnessFunction,
        task: ARCTask
    ) -> float:
        """Calculate how well fitness function correlates with ground truth."""
        # Use exact match as ground truth
        exact_match_scores = []
        fitness_scores = []

        for program in self.program_population:
            if "execution_result" not in program.metadata:
                continue

            predicted = program.metadata["execution_result"]
            target = task.train_pairs[0].output

            # Ground truth score
            exact_match = FitnessComponentLibrary._exact_match(predicted, target)
            exact_match_scores.append(exact_match)

            # Fitness function score
            if fitness_func.id in program.metadata.get("fitness_evaluations", {}):
                score = program.metadata["fitness_evaluations"][fitness_func.id]
                fitness_scores.append(score)
            else:
                fitness_scores.append(0.0)

        # Calculate correlation
        if len(exact_match_scores) < 2:
            return 0.5

        # Simplified correlation: check if high scores align
        threshold = 0.5
        true_positives = sum(
            1 for em, fs in zip(exact_match_scores, fitness_scores, strict=False)
            if em > threshold and fs > threshold
        )
        total_positives = sum(1 for em in exact_match_scores if em > threshold)

        if total_positives > 0:
            return true_positives / total_positives

        # If no exact matches, check if fitness function also gives low scores
        all_low = all(fs < threshold for fs in fitness_scores)
        return 0.8 if all_low else 0.3

    def _evolve_fitness_population(self) -> None:
        """Evolve the fitness function population."""
        # Sort by meta-fitness
        self.fitness_population.sort(key=lambda f: f.fitness_score, reverse=True)

        # Keep elite
        elite_count = int(self.fitness_pop_size * self.elite_ratio)
        new_population = self.fitness_population[:elite_count]

        # Generate offspring
        while len(new_population) < self.fitness_pop_size:
            # Select parents
            parent1 = self._select_fitness_parent()
            parent2 = self._select_fitness_parent()

            # Create offspring
            if random.random() < 0.7:  # Crossover rate
                offspring = self._crossover_fitness_functions(parent1, parent2)
            else:
                offspring = self._copy_fitness_function(parent1)

            # Mutate
            if random.random() < 0.3:  # Mutation rate
                self._mutate_fitness_function(offspring)

            offspring.age = 0
            new_population.append(offspring)

        self.fitness_population = new_population[:self.fitness_pop_size]

        # Age increment
        for fitness_func in self.fitness_population:
            fitness_func.age += 1

    def _select_fitness_parent(self) -> FitnessFunction:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(
            self.fitness_population,
            min(tournament_size, len(self.fitness_population))
        )
        return max(tournament, key=lambda f: f.fitness_score)

    def _crossover_fitness_functions(
        self,
        parent1: FitnessFunction,
        parent2: FitnessFunction
    ) -> FitnessFunction:
        """Crossover two fitness functions."""
        # Combine components from both parents
        all_components = parent1.components + parent2.components

        # Remove duplicates
        seen = set()
        unique_components = []
        for comp in all_components:
            if comp.name not in seen:
                seen.add(comp.name)
                unique_components.append(comp)

        # Select subset
        max_components = min(5, len(unique_components))
        if max_components < 2:
            # If we don't have enough components, use what we have
            num_components = len(unique_components)
        else:
            num_components = random.randint(2, max_components)
        selected = random.sample(unique_components, num_components)

        # Create offspring
        import copy
        offspring_components = [copy.deepcopy(c) for c in selected]
        offspring = FitnessFunction(components=offspring_components)
        offspring.normalize_weights()

        return offspring

    def _copy_fitness_function(self, fitness_func: FitnessFunction) -> FitnessFunction:
        """Create a copy of fitness function."""
        import copy
        return copy.deepcopy(fitness_func)

    def _mutate_fitness_function(self, fitness_func: FitnessFunction) -> None:
        """Mutate a fitness function."""
        mutation_type = random.choice(["add", "remove", "weight", "replace"])

        if mutation_type == "add" and len(fitness_func.components) < 5:
            # Add new component
            new_comp = random.choice(self.component_library)
            import copy
            fitness_func.components.append(copy.deepcopy(new_comp))

        elif mutation_type == "remove" and len(fitness_func.components) > 1:
            # Remove random component
            idx = random.randint(0, len(fitness_func.components) - 1)
            fitness_func.components.pop(idx)

        elif mutation_type == "weight":
            # Mutate weights
            for comp in fitness_func.components:
                if random.random() < 0.5:
                    # Adjust weight
                    comp.weight *= random.uniform(0.5, 2.0)
                    if comp.name == "complexity_penalty":
                        comp.weight = -abs(comp.weight)  # Keep negative

        elif mutation_type == "replace" and fitness_func.components:
            # Replace a component
            idx = random.randint(0, len(fitness_func.components) - 1)
            new_comp = random.choice(self.component_library)
            import copy
            fitness_func.components[idx] = copy.deepcopy(new_comp)

        fitness_func.normalize_weights()

    def _get_best_fitness_function(self) -> FitnessFunction:
        """Get the best fitness function."""
        return max(self.fitness_population, key=lambda f: f.fitness_score)

    def _update_program_fitness(
        self,
        fitness_func: FitnessFunction,
        task: ARCTask
    ) -> None:
        """Update program fitness using best fitness function."""
        for program in self.program_population:
            if fitness_func.id in program.metadata.get("fitness_evaluations", {}):
                program.fitness = program.metadata["fitness_evaluations"][fitness_func.id]
            else:
                # Re-evaluate if needed
                if "execution_result" in program.metadata:
                    predicted = program.metadata["execution_result"]
                    target = task.train_pairs[0].output
                    program.fitness = fitness_func.evaluate(predicted, target)
                else:
                    program.fitness = 0.0

    def _calculate_fitness_diversity(self) -> float:
        """Calculate diversity of fitness functions."""
        # Measure diversity as variance in component usage
        component_counts = {}

        for fitness_func in self.fitness_population:
            for comp in fitness_func.components:
                component_counts[comp.name] = component_counts.get(comp.name, 0) + 1

        if not component_counts:
            return 0.0

        # Calculate entropy
        total = sum(component_counts.values())
        entropy = 0.0

        for count in component_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)

        # Normalize by maximum entropy
        max_entropy = math.log(len(self.component_library))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_coevolution_stats(self) -> dict[str, Any]:
        """Get statistics about co-evolution process."""
        return {
            "generation": self.generation,
            "best_program_fitness": max(
                (p.fitness for p in self.program_population),
                default=0.0
            ),
            "best_fitness_function_score": max(
                (f.fitness_score for f in self.fitness_population),
                default=0.0
            ),
            "fitness_diversity": self._calculate_fitness_diversity(),
            "avg_components_per_function": sum(
                len(f.components) for f in self.fitness_population
            ) / len(self.fitness_population) if self.fitness_population else 0,
            "component_usage": self._get_component_usage_stats(),
            "fitness_history": self.best_program_fitness_history[-10:],
            "diversity_history": self.fitness_diversity_history[-10:]
        }

    def _get_component_usage_stats(self) -> dict[str, int]:
        """Get usage statistics for each component type."""
        usage = {}
        for fitness_func in self.fitness_population:
            for comp in fitness_func.components:
                usage[comp.name] = usage.get(comp.name, 0) + 1
        return usage
