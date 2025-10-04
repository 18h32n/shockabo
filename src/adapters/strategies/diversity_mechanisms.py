"""
Diversity preservation mechanisms for genetic algorithm.

This module implements various methods to maintain population diversity,
prevent premature convergence, and encourage exploration of the search space.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.adapters.strategies.evolution_engine import DiversityMetric, Individual, Population
from src.domain.dsl.base import Operation


class FitnessSharing(DiversityMetric):
    """
    Fitness sharing mechanism for maintaining diversity.

    Reduces fitness of similar individuals to encourage
    exploration of different solution niches.
    """

    def __init__(self, niche_radius: float = 0.15, alpha: float = 1.0):
        """
        Initialize fitness sharing.

        Args:
            niche_radius: Distance threshold for niche membership
            alpha: Sharing function shape parameter
        """
        self.niche_radius = niche_radius
        self.alpha = alpha

    def calculate(self, population: Population) -> float:
        """Calculate overall diversity metric."""
        # Calculate average pairwise distance
        distances = []
        individuals = population.individuals

        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                dist = self._calculate_distance(individuals[i], individuals[j])
                distances.append(dist)

        if distances:
            return np.mean(distances)
        return 0.0

    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply fitness sharing to adjust fitness."""
        # Calculate niche count
        niche_count = 0.0

        for other in population.individuals:
            if other.id != individual.id:
                distance = self._calculate_distance(individual, other)
                if distance < self.niche_radius:
                    # Individual is in same niche
                    sharing_value = 1.0 - (distance / self.niche_radius) ** self.alpha
                    niche_count += sharing_value

        # Adjust fitness by niche count
        shared_fitness = individual.fitness / (1.0 + niche_count)

        return shared_fitness

    def _calculate_distance(self, ind1: Individual, ind2: Individual) -> float:
        """
        Calculate phenotypic distance between individuals.

        Uses a combination of program structure and behavior similarity.
        """
        # Structural distance (operation sequence similarity)
        struct_dist = self._structural_distance(ind1, ind2)

        # Behavioral distance (output similarity if cached)
        behav_dist = self._behavioral_distance(ind1, ind2)

        # Combine distances
        return 0.7 * struct_dist + 0.3 * behav_dist

    def _structural_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate structural distance based on operations."""
        ops1 = [op.get_name() for op in ind1.operations]
        ops2 = [op.get_name() for op in ind2.operations]

        # Normalized edit distance
        return self._normalized_edit_distance(ops1, ops2)

    def _behavioral_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate behavioral distance based on outputs."""
        if not ind1.cached_execution or not ind2.cached_execution:
            # No behavioral information available
            return 0.5  # Neutral distance

        # Compare outputs on training examples
        common_keys = set(ind1.cached_execution.keys()) & set(ind2.cached_execution.keys())
        if not common_keys:
            return 1.0

        total_distance = 0.0
        for key in common_keys:
            grid1 = np.array(ind1.cached_execution[key])
            grid2 = np.array(ind2.cached_execution[key])

            if grid1.shape == grid2.shape:
                # Calculate grid similarity
                similarity = np.mean(grid1 == grid2)
                total_distance += 1.0 - similarity
            else:
                # Different shapes, maximum distance
                total_distance += 1.0

        return total_distance / len(common_keys)

    def _normalized_edit_distance(self, seq1: list[str], seq2: list[str]) -> float:
        """Calculate normalized Levenshtein distance."""
        m, n = len(seq1), len(seq2)
        if m == 0 and n == 0:
            return 0.0
        if m == 0 or n == 0:
            return 1.0

        # Dynamic programming for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # Normalize by maximum possible distance
        return dp[m][n] / max(m, n)


class Speciation(DiversityMetric):
    """
    Speciation mechanism that groups similar individuals into species.

    Maintains separate subpopulations that compete primarily within
    their own species, preserving diverse evolutionary paths.
    """

    def __init__(self, compatibility_threshold: float = 0.3,
                 c1: float = 1.0, c2: float = 1.0, c3: float = 0.4):
        """
        Initialize speciation.

        Args:
            compatibility_threshold: Threshold for species membership
            c1: Weight for excess genes
            c2: Weight for disjoint genes
            c3: Weight for weight differences
        """
        self.compatibility_threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.species_representatives: dict[int, Individual] = {}
        self.next_species_id = 0

    def calculate(self, population: Population) -> float:
        """Calculate species diversity."""
        # Number of species relative to population size
        num_species = len(population.species)
        if population.size() == 0:
            return 0.0

        return num_species / math.sqrt(population.size())

    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply species-based fitness adjustment."""
        if individual.species_id is None:
            # Assign to species
            self._assign_species(individual, population)

        # Adjust fitness based on species size
        species_size = len(population.species.get(individual.species_id, []))
        if species_size > 0:
            # Larger species get fitness penalty
            return individual.fitness / species_size

        return individual.fitness

    def speciate_population(self, population: Population) -> None:
        """Assign all individuals to species."""
        # Clear existing species
        population.species.clear()

        # Reset species assignments
        for ind in population.individuals:
            ind.species_id = None

        # Assign each individual to species
        for ind in population.individuals:
            self._assign_species(ind, population)

    def _assign_species(self, individual: Individual, population: Population) -> None:
        """Assign individual to appropriate species."""
        # Find compatible species
        for species_id, representative in self.species_representatives.items():
            if self._is_compatible(individual, representative):
                individual.species_id = species_id
                if species_id not in population.species:
                    population.species[species_id] = []
                population.species[species_id].append(individual)
                return

        # No compatible species found, create new one
        species_id = self.next_species_id
        self.next_species_id += 1
        individual.species_id = species_id
        self.species_representatives[species_id] = individual
        population.species[species_id] = [individual]

    def _is_compatible(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if two individuals are compatible (same species)."""
        distance = self._compatibility_distance(ind1, ind2)
        return distance < self.compatibility_threshold

    def _compatibility_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate compatibility distance between individuals."""
        ops1 = ind1.operations
        ops2 = ind2.operations

        # Count matching, excess, and disjoint operations
        matching = 0
        i, j = 0, 0
        weight_diff_sum = 0.0

        while i < len(ops1) and j < len(ops2):
            if ops1[i].get_name() == ops2[j].get_name():
                matching += 1
                # Compare parameters
                weight_diff_sum += self._parameter_difference(ops1[i], ops2[j])
                i += 1
                j += 1
            else:
                # Operations don't match, advance one
                if ops1[i].get_name() < ops2[j].get_name():
                    i += 1
                else:
                    j += 1

        # Remaining operations are excess
        excess = len(ops1) - i + len(ops2) - j
        disjoint = len(ops1) + len(ops2) - 2 * matching - excess

        # Normalize by genome size
        n = max(len(ops1), len(ops2))
        if n < 20:
            n = 1  # Don't normalize for small genomes

        # Calculate distance
        avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0
        distance = (self.c1 * excess / n +
                   self.c2 * disjoint / n +
                   self.c3 * avg_weight_diff)

        return distance

    def _parameter_difference(self, op1: Operation, op2: Operation) -> float:
        """Calculate parameter difference between operations."""
        # Simple parameter comparison
        params1 = op1.parameters
        params2 = op2.parameters

        all_keys = set(params1.keys()) | set(params2.keys())
        if not all_keys:
            return 0.0

        diff_sum = 0.0
        for key in all_keys:
            val1 = params1.get(key, 0)
            val2 = params2.get(key, 0)

            # Normalize difference based on parameter type
            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Assume parameters are in [0, 10] range
                diff_sum += abs(val1 - val2) / 10.0
            else:
                # Binary difference for other types
                diff_sum += 0.0 if val1 == val2 else 1.0

        return diff_sum / len(all_keys)


class NoveltySearch(DiversityMetric):
    """
    Novelty search that rewards behavioral uniqueness.

    Instead of optimizing for fitness directly, rewards individuals
    that exhibit novel behaviors compared to archive and current population.
    """

    def __init__(self, archive_size: int = 100, k_nearest: int = 15,
                 novelty_threshold: float = 0.3):
        """
        Initialize novelty search.

        Args:
            archive_size: Maximum size of novelty archive
            k_nearest: Number of nearest neighbors to consider
            novelty_threshold: Threshold for adding to archive
        """
        self.archive_size = archive_size
        self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold
        self.archive: list[dict[str, np.ndarray]] = []

    def calculate(self, population: Population) -> float:
        """Calculate average novelty in population."""
        if not population.individuals:
            return 0.0

        total_novelty = sum(ind.novelty_score or 0.0 for ind in population.individuals)
        return total_novelty / len(population.individuals)

    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply novelty-based fitness adjustment."""
        # Calculate novelty score
        novelty = self._calculate_novelty(individual, population)
        individual.novelty_score = novelty

        # Consider adding to archive
        if novelty > self.novelty_threshold and individual.cached_execution:
            self._add_to_archive(individual.cached_execution)

        # Combine novelty with base fitness
        # Higher weight on novelty early in evolution
        novelty_weight = 0.7 if population.generation < 50 else 0.3
        adjusted_fitness = (novelty_weight * novelty +
                           (1 - novelty_weight) * individual.fitness)

        return adjusted_fitness

    def _calculate_novelty(self, individual: Individual, population: Population) -> float:
        """Calculate novelty score for individual."""
        if not individual.cached_execution:
            return 0.0

        # Get behavior characterization
        behavior = self._get_behavior_vector(individual.cached_execution)

        # Find k-nearest neighbors in population and archive
        distances = []

        # Population distances
        for other in population.individuals:
            if other.id != individual.id and other.cached_execution:
                other_behavior = self._get_behavior_vector(other.cached_execution)
                dist = np.linalg.norm(behavior - other_behavior)
                distances.append(dist)

        # Archive distances
        for archived_behavior in self.archive:
            archived_vec = self._get_behavior_vector(archived_behavior)
            dist = np.linalg.norm(behavior - archived_vec)
            distances.append(dist)

        if not distances:
            return 1.0  # Maximum novelty if no comparison points

        # Sort and take k-nearest
        distances.sort()
        k = min(self.k_nearest, len(distances))
        k_nearest_distances = distances[:k]

        # Average distance to k-nearest neighbors
        return np.mean(k_nearest_distances)

    def _get_behavior_vector(self, execution_results: dict[str, Any]) -> np.ndarray:
        """
        Extract behavior characterization vector from execution results.

        Uses features like output patterns, colors used, symmetries, etc.
        """
        features = []

        for _key, grid in execution_results.items():
            if isinstance(grid, list):
                grid_array = np.array(grid)

                # Grid statistics
                features.append(grid_array.mean())
                features.append(grid_array.std())

                # Color distribution
                color_counts = np.bincount(grid_array.flatten(), minlength=10)
                features.extend(color_counts / grid_array.size)

                # Spatial features
                features.append(self._calculate_symmetry(grid_array))
                features.append(self._calculate_connectivity(grid_array))

        return np.array(features)

    def _calculate_symmetry(self, grid: np.ndarray) -> float:
        """Calculate symmetry score of grid."""
        # Check horizontal symmetry
        h_sym = np.mean(grid == np.flip(grid, axis=1))

        # Check vertical symmetry
        v_sym = np.mean(grid == np.flip(grid, axis=0))

        return max(h_sym, v_sym)

    def _calculate_connectivity(self, grid: np.ndarray) -> float:
        """Calculate connectivity score of grid."""
        # Simple connected component ratio
        unique_colors = np.unique(grid)
        if len(unique_colors) <= 1:
            return 1.0

        # Approximate by checking adjacent cell similarity
        h_similar = np.mean(grid[:, :-1] == grid[:, 1:])
        v_similar = np.mean(grid[:-1, :] == grid[1:, :])

        return (h_similar + v_similar) / 2

    def _add_to_archive(self, execution_results: dict[str, Any]) -> None:
        """Add behavior to novelty archive."""
        if len(self.archive) >= self.archive_size:
            # Remove oldest entry
            self.archive.pop(0)

        self.archive.append(execution_results.copy())


class CrowdingDistance(DiversityMetric):
    """
    Crowding distance metric for maintaining diversity.

    Used in multi-objective optimization to preserve solutions
    that are far from others in objective space.
    """

    def __init__(self, objectives: list[str] = None):
        """
        Initialize crowding distance.

        Args:
            objectives: List of objective names to consider
        """
        self.objectives = objectives or ['fitness', 'program_length', 'novelty']

    def calculate(self, population: Population) -> float:
        """Calculate average crowding distance."""
        distances = self._calculate_crowding_distances(population)
        if distances:
            return np.mean(list(distances.values()))
        return 0.0

    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply crowding-based fitness adjustment."""
        distances = self._calculate_crowding_distances(population)

        # Higher crowding distance is better (more isolated)
        crowding_distance = distances.get(individual.id, 0.0)

        # Normalize and combine with fitness
        max_distance = max(distances.values()) if distances else 1.0
        normalized_distance = crowding_distance / max_distance if max_distance > 0 else 0.0

        # Weighted combination
        adjusted_fitness = 0.8 * individual.fitness + 0.2 * normalized_distance

        return adjusted_fitness

    def _calculate_crowding_distances(self, population: Population) -> dict[str, float]:
        """Calculate crowding distance for all individuals."""
        individuals = population.individuals
        n = len(individuals)

        if n <= 2:
            # All individuals have infinite crowding distance
            return {ind.id: float('inf') for ind in individuals}

        # Initialize distances
        distances = {ind.id: 0.0 for ind in individuals}

        # Calculate distance for each objective
        for obj in self.objectives:
            # Get objective values
            values = []
            for ind in individuals:
                if obj == 'fitness':
                    values.append((ind.id, ind.fitness))
                elif obj == 'program_length':
                    values.append((ind.id, ind.program_length()))
                elif obj == 'novelty':
                    values.append((ind.id, ind.novelty_score or 0.0))
                else:
                    values.append((ind.id, 0.0))

            # Sort by objective value
            values.sort(key=lambda x: x[1])

            # Boundary individuals get infinite distance
            distances[values[0][0]] = float('inf')
            distances[values[-1][0]] = float('inf')

            # Calculate distances for interior individuals
            obj_range = values[-1][1] - values[0][1]
            if obj_range > 0:
                for i in range(1, n - 1):
                    dist = (values[i + 1][1] - values[i - 1][1]) / obj_range
                    distances[values[i][0]] += dist

        return distances


# Hybrid diversity mechanism that combines multiple approaches
class HybridDiversity(DiversityMetric):
    """
    Combines multiple diversity mechanisms for robust diversity maintenance.

    Adaptively weights different diversity metrics based on
    population state and evolution progress.
    """

    def __init__(self, mechanisms: list[DiversityMetric] | None = None):
        """
        Initialize hybrid diversity.

        Args:
            mechanisms: List of diversity mechanisms to combine
        """
        if mechanisms is None:
            mechanisms = [
                FitnessSharing(niche_radius=0.15),
                Speciation(compatibility_threshold=0.3),
                NoveltySearch(archive_size=100),
                CrowdingDistance()
            ]
        self.mechanisms = mechanisms
        self.weights = [1.0 / len(mechanisms)] * len(mechanisms)

    def calculate(self, population: Population) -> float:
        """Calculate combined diversity metric."""
        total_diversity = 0.0

        for mechanism, weight in zip(self.mechanisms, self.weights, strict=False):
            diversity = mechanism.calculate(population)
            total_diversity += weight * diversity

        return total_diversity

    def apply_pressure(self, individual: Individual, population: Population) -> float:
        """Apply combined diversity pressure."""
        adjusted_fitnesses = []

        for mechanism in self.mechanisms:
            adj_fitness = mechanism.apply_pressure(individual, population)
            adjusted_fitnesses.append(adj_fitness)

        # Weighted combination
        combined_fitness = sum(f * w for f, w in zip(adjusted_fitnesses, self.weights, strict=False))

        return combined_fitness

    def adapt_weights(self, population: Population) -> None:
        """
        Adapt mechanism weights based on population state.

        Args:
            population: Current population
        """
        # Early evolution: emphasize novelty
        if population.generation < 20:
            self.weights = [0.2, 0.2, 0.4, 0.2]  # More novelty
        # Mid evolution: emphasize speciation
        elif population.generation < 50:
            self.weights = [0.25, 0.35, 0.2, 0.2]  # More speciation
        # Late evolution: emphasize fitness sharing
        else:
            self.weights = [0.35, 0.25, 0.2, 0.2]  # More fitness sharing

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
