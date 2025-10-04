"""
Novelty search implementation for evolutionary algorithms.

Task 7.5: Implement novelty search as alternative to fitness optimization.
Rewards behavioral diversity rather than objective fitness.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.adapters.strategies.evolution_engine import Individual, Population
from src.domain.models import ARCTask, Grid


@dataclass
class BehaviorCharacterization:
    """Represents the behavior of a program in the solution space."""

    features: np.ndarray  # Behavioral feature vector
    raw_output: Grid | None = None  # Raw program output
    metadata: dict[str, Any] = field(default_factory=dict)

    def distance(self, other: 'BehaviorCharacterization') -> float:
        """Calculate behavioral distance to another characterization."""
        # Euclidean distance in behavior space
        return float(np.linalg.norm(self.features - other.features))

    def __hash__(self) -> int:
        """Hash based on feature vector for novelty archive."""
        return hash(self.features.tobytes())


class BehaviorExtractor:
    """Extracts behavioral features from program execution results."""

    @staticmethod
    def extract_features(output_grid: Grid | None, input_grid: Grid | None) -> np.ndarray:
        """Extract behavioral features from program output.

        Features include:
        - Output dimensions (width, height)
        - Color distribution histogram
        - Spatial patterns (symmetry, connectivity)
        - Transformation characteristics
        - Statistical properties
        """
        features = []

        if output_grid is None:
            # Failed execution - return zero features
            return np.zeros(50)  # Fixed feature size

        # 1. Basic dimensions (2 features)
        features.extend([output_grid.shape[0], output_grid.shape[1]])

        # 2. Color histogram (10 features - for colors 0-9)
        color_hist = np.zeros(10)
        for row in output_grid.data:
            for val in row:
                if 0 <= val <= 9:
                    color_hist[val] += 1
        # Normalize by total pixels
        total_pixels = output_grid.shape[0] * output_grid.shape[1]
        if total_pixels > 0:
            color_hist = color_hist / total_pixels
        features.extend(color_hist.tolist())

        # 3. Spatial statistics (8 features)
        flat_data = np.array(output_grid.data).flatten()
        features.append(np.mean(flat_data))  # Mean value
        features.append(np.std(flat_data))   # Standard deviation
        features.append(np.min(flat_data))   # Min value
        features.append(np.max(flat_data))   # Max value

        # Spatial gradients
        if output_grid.shape[0] > 1 and output_grid.shape[1] > 1:
            h_gradient = np.diff(output_grid.data, axis=1).mean()
            v_gradient = np.diff(output_grid.data, axis=0).mean()
            features.extend([h_gradient, v_gradient])
        else:
            features.extend([0.0, 0.0])

        # Unique colors count
        unique_colors = len(set(flat_data))
        features.append(unique_colors / 10.0)  # Normalize by max colors

        # Color clustering (average distance between different colors)
        color_positions = {c: [] for c in set(flat_data)}
        for i, row in enumerate(output_grid.data):
            for j, val in enumerate(row):
                color_positions[val].append((i, j))

        avg_cluster_size = np.mean([len(positions) for positions in color_positions.values()])
        features.append(avg_cluster_size / total_pixels if total_pixels > 0 else 0)

        # 4. Symmetry detection (4 features)
        h_sym = BehaviorExtractor._check_horizontal_symmetry(output_grid)
        v_sym = BehaviorExtractor._check_vertical_symmetry(output_grid)
        d1_sym = BehaviorExtractor._check_diagonal_symmetry(output_grid)
        d2_sym = BehaviorExtractor._check_antidiagonal_symmetry(output_grid)
        features.extend([h_sym, v_sym, d1_sym, d2_sym])

        # 5. Pattern detection (8 features)
        # Repeating patterns
        h_repeat = BehaviorExtractor._check_horizontal_repetition(output_grid)
        v_repeat = BehaviorExtractor._check_vertical_repetition(output_grid)
        features.extend([h_repeat, v_repeat])

        # Connected components
        num_components = BehaviorExtractor._count_connected_components(output_grid)
        features.append(num_components / total_pixels if total_pixels > 0 else 0)

        # Edge density
        edge_pixels = BehaviorExtractor._count_edge_pixels(output_grid)
        features.append(edge_pixels / total_pixels if total_pixels > 0 else 0)

        # Fill ratio for each color (top 4 colors)
        color_ratios = sorted([(c, len(positions)/total_pixels)
                              for c, positions in color_positions.items()],
                             key=lambda x: x[1], reverse=True)[:4]
        for i in range(4):
            if i < len(color_ratios):
                features.append(color_ratios[i][1])
            else:
                features.append(0.0)

        # 6. Transformation features (8 features) - compare with input
        if input_grid is not None:
            # Size change
            size_ratio_h = output_grid.shape[0] / input_grid.shape[0]
            size_ratio_w = output_grid.shape[1] / input_grid.shape[1]
            features.extend([size_ratio_h, size_ratio_w])

            # Color mapping changes
            input_colors = set(np.array(input_grid.data).flatten())
            output_colors = set(flat_data)
            new_colors = len(output_colors - input_colors) / 10.0
            lost_colors = len(input_colors - output_colors) / 10.0
            features.extend([new_colors, lost_colors])

            # Structural similarity (simplified)
            if input_grid.shape == output_grid.shape:
                matching_pixels = sum(1 for i in range(input_grid.shape[0])
                                    for j in range(input_grid.shape[1])
                                    if input_grid.data[i][j] == output_grid.data[i][j])
                similarity = matching_pixels / total_pixels
            else:
                similarity = 0.0
            features.append(similarity)

            # Average color change
            if input_grid.shape == output_grid.shape:
                avg_change = np.mean(np.abs(np.array(output_grid.data) - np.array(input_grid.data)))
            else:
                avg_change = 5.0  # Max possible change
            features.append(avg_change / 9.0)  # Normalize

            # Entropy change
            input_entropy = BehaviorExtractor._calculate_entropy(input_grid)
            output_entropy = BehaviorExtractor._calculate_entropy(output_grid)
            features.extend([input_entropy, output_entropy])
        else:
            # No input comparison - pad with zeros
            features.extend([0.0] * 8)

        # 7. Statistical texture features (10 features to reach 50 total)
        # Local Binary Pattern-like features
        lbp_features = BehaviorExtractor._compute_texture_features(output_grid)
        features.extend(lbp_features[:10])  # Take first 10

        # Ensure exactly 50 features
        features = features[:50]  # Truncate if too many
        while len(features) < 50:  # Pad if too few
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _check_horizontal_symmetry(grid: Grid) -> float:
        """Check horizontal symmetry score."""
        score = 0.0
        h, w = grid.shape
        for i in range(h):
            for j in range(w // 2):
                if grid.data[i][j] == grid.data[i][w - 1 - j]:
                    score += 1
        return score / (h * (w // 2)) if h * (w // 2) > 0 else 0.0

    @staticmethod
    def _check_vertical_symmetry(grid: Grid) -> float:
        """Check vertical symmetry score."""
        score = 0.0
        h, w = grid.shape
        for i in range(h // 2):
            for j in range(w):
                if grid.data[i][j] == grid.data[h - 1 - i][j]:
                    score += 1
        return score / ((h // 2) * w) if (h // 2) * w > 0 else 0.0

    @staticmethod
    def _check_diagonal_symmetry(grid: Grid) -> float:
        """Check diagonal symmetry score."""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        score = 0.0
        n = grid.shape[0]
        for i in range(n):
            for j in range(i):
                if grid.data[i][j] == grid.data[j][i]:
                    score += 1
        return score / (n * (n - 1) / 2) if n > 1 else 0.0

    @staticmethod
    def _check_antidiagonal_symmetry(grid: Grid) -> float:
        """Check anti-diagonal symmetry score."""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        score = 0.0
        n = grid.shape[0]
        for i in range(n):
            for j in range(n - 1 - i):
                if grid.data[i][j] == grid.data[n - 1 - j][n - 1 - i]:
                    score += 1
        return score / (n * (n - 1) / 2) if n > 1 else 0.0

    @staticmethod
    def _check_horizontal_repetition(grid: Grid) -> float:
        """Check for horizontal repeating patterns."""
        h, w = grid.shape
        if w < 2:
            return 0.0

        pattern_scores = []
        for pattern_len in range(1, w // 2 + 1):
            score = 0
            for i in range(h):
                for j in range(w - pattern_len):
                    if j + pattern_len < w and grid.data[i][j] == grid.data[i][j + pattern_len]:
                        score += 1
            pattern_scores.append(score / (h * (w - pattern_len)))

        return max(pattern_scores) if pattern_scores else 0.0

    @staticmethod
    def _check_vertical_repetition(grid: Grid) -> float:
        """Check for vertical repeating patterns."""
        h, w = grid.shape
        if h < 2:
            return 0.0

        pattern_scores = []
        for pattern_len in range(1, h // 2 + 1):
            score = 0
            for i in range(h - pattern_len):
                for j in range(w):
                    if i + pattern_len < h and grid.data[i][j] == grid.data[i + pattern_len][j]:
                        score += 1
            pattern_scores.append(score / ((h - pattern_len) * w))

        return max(pattern_scores) if pattern_scores else 0.0

    @staticmethod
    def _count_connected_components(grid: Grid) -> int:
        """Count connected components using flood fill."""
        h, w = grid.shape
        visited = [[False] * w for _ in range(h)]
        components = 0

        def flood_fill(i, j, color):
            if i < 0 or i >= h or j < 0 or j >= w or visited[i][j]:
                return
            if grid.data[i][j] != color:
                return
            visited[i][j] = True
            # 4-connected
            flood_fill(i + 1, j, color)
            flood_fill(i - 1, j, color)
            flood_fill(i, j + 1, color)
            flood_fill(i, j - 1, color)

        for i in range(h):
            for j in range(w):
                if not visited[i][j]:
                    flood_fill(i, j, grid.data[i][j])
                    components += 1

        return components

    @staticmethod
    def _count_edge_pixels(grid: Grid) -> int:
        """Count pixels that are on edges (different from neighbors)."""
        h, w = grid.shape
        edge_count = 0

        for i in range(h):
            for j in range(w):
                is_edge = False
                current = grid.data[i][j]

                # Check 4-connected neighbors
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid.data[ni][nj] != current:
                            is_edge = True
                            break

                if is_edge:
                    edge_count += 1

        return edge_count

    @staticmethod
    def _calculate_entropy(grid: Grid) -> float:
        """Calculate Shannon entropy of grid colors."""
        flat_data = np.array(grid.data).flatten()
        value_counts = {}
        for val in flat_data:
            value_counts[val] = value_counts.get(val, 0) + 1

        total = len(flat_data)
        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy / np.log2(10)  # Normalize by max possible entropy

    @staticmethod
    def _compute_texture_features(grid: Grid) -> list[float]:
        """Compute texture-based features."""
        h, w = grid.shape
        features = []

        # Simple gradient features
        if h > 1 and w > 1:
            # Horizontal gradients
            h_grads = []
            for i in range(h):
                for j in range(w - 1):
                    h_grads.append(abs(grid.data[i][j + 1] - grid.data[i][j]))
            features.append(np.mean(h_grads) if h_grads else 0.0)
            features.append(np.std(h_grads) if h_grads else 0.0)

            # Vertical gradients
            v_grads = []
            for i in range(h - 1):
                for j in range(w):
                    v_grads.append(abs(grid.data[i + 1][j] - grid.data[i][j]))
            features.append(np.mean(v_grads) if v_grads else 0.0)
            features.append(np.std(v_grads) if v_grads else 0.0)

            # Corner features
            corner_sum = 0
            corner_count = 0
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    center = grid.data[i][j]
                    corners = [
                        grid.data[i - 1][j - 1], grid.data[i - 1][j + 1],
                        grid.data[i + 1][j - 1], grid.data[i + 1][j + 1]
                    ]
                    if all(c == center for c in corners):
                        corner_sum += 1
                    corner_count += 1

            features.append(corner_sum / corner_count if corner_count > 0 else 0.0)
        else:
            features.extend([0.0] * 5)

        # Fill ratio of most common color
        flat_data = np.array(grid.data).flatten()
        if len(flat_data) > 0:
            unique, counts = np.unique(flat_data, return_counts=True)
            max_ratio = np.max(counts) / len(flat_data)
            features.append(max_ratio)
        else:
            features.append(0.0)

        # Add more features to reach 10
        while len(features) < 10:
            features.append(0.0)

        return features


@dataclass
class NoveltyArchive:
    """Archive of novel behaviors discovered during search."""

    behaviors: list[BehaviorCharacterization] = field(default_factory=list)
    max_size: int = 2000  # Maximum archive size
    min_distance: float = 0.1  # Minimum distance to be considered novel

    def add_if_novel(self, behavior: BehaviorCharacterization) -> bool:
        """Add behavior to archive if it's novel enough."""
        if not self.behaviors:
            self.behaviors.append(behavior)
            return True

        # Check minimum distance to all archived behaviors
        min_dist = min(behavior.distance(b) for b in self.behaviors)

        if min_dist >= self.min_distance:
            self.behaviors.append(behavior)

            # Maintain size limit by removing least novel
            if len(self.behaviors) > self.max_size:
                self._prune_archive()

            return True

        return False

    def _prune_archive(self):
        """Remove least novel behaviors when archive is full."""
        # Calculate sparseness for each behavior
        sparseness_scores = []
        for i, behavior in enumerate(self.behaviors):
            # Average distance to k-nearest neighbors
            distances = sorted([behavior.distance(b) for j, b in enumerate(self.behaviors) if i != j])
            k = min(15, len(distances))  # k-nearest neighbors
            avg_distance = np.mean(distances[:k]) if k > 0 else 0
            sparseness_scores.append((i, avg_distance))

        # Keep behaviors with highest sparseness
        sparseness_scores.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep = {idx for idx, _ in sparseness_scores[:self.max_size - 100]}  # Remove 100 at a time

        self.behaviors = [b for i, b in enumerate(self.behaviors) if i in indices_to_keep]

    def get_novelty_score(self, behavior: BehaviorCharacterization, k: int = 15) -> float:
        """Calculate novelty score as average distance to k-nearest neighbors."""
        if not self.behaviors:
            return 1.0  # Maximum novelty if archive is empty

        # Calculate distances to all archived behaviors
        distances = sorted([behavior.distance(b) for b in self.behaviors])

        # Average distance to k-nearest neighbors
        k = min(k, len(distances))
        return np.mean(distances[:k]) if k > 0 else 0.0


class NoveltySearch:
    """Main novelty search engine."""

    def __init__(self,
                 archive_size: int = 2000,
                 min_novelty_distance: float = 0.1,
                 novelty_weight: float = 0.5,  # Balance between novelty and fitness
                 k_neighbors: int = 15):
        """Initialize novelty search.

        Args:
            archive_size: Maximum size of novelty archive
            min_novelty_distance: Minimum distance to be considered novel
            novelty_weight: Weight for novelty vs fitness (0=pure fitness, 1=pure novelty)
            k_neighbors: Number of neighbors for novelty calculation
        """
        self.archive = NoveltyArchive(max_size=archive_size, min_distance=min_novelty_distance)
        self.behavior_extractor = BehaviorExtractor()
        self.novelty_weight = novelty_weight
        self.k_neighbors = k_neighbors
        self.behavior_cache: dict[str, BehaviorCharacterization] = {}

    def characterize_behavior(self,
                            individual: Individual,
                            execution_result: Grid | None,
                            input_grid: Grid | None = None) -> BehaviorCharacterization:
        """Extract behavioral characterization from program execution."""
        # Check cache first
        ind_hash = hashlib.md5(str(individual.operations).encode()).hexdigest()
        if ind_hash in self.behavior_cache:
            return self.behavior_cache[ind_hash]

        # Extract features
        features = self.behavior_extractor.extract_features(execution_result, input_grid)

        behavior = BehaviorCharacterization(
            features=features,
            raw_output=execution_result,
            metadata={
                'individual_id': individual.id,
                'generation': individual.metadata.get('generation', 0),
                'operations': individual.operations
            }
        )

        # Cache behavior
        self.behavior_cache[ind_hash] = behavior

        return behavior

    def evaluate_novelty(self,
                        population: Population,
                        task: ARCTask,
                        execution_results: dict[str, Grid | None]) -> dict[str, float]:
        """Evaluate novelty scores for entire population.

        Returns:
            Dict mapping individual IDs to novelty scores
        """
        novelty_scores = {}
        behaviors = []

        # First, characterize all behaviors
        for individual in population.individuals:
            exec_result = execution_results.get(individual.id)
            input_grid = task.train_pairs[0].input if task.train_pairs else None

            behavior = self.characterize_behavior(individual, exec_result, input_grid)
            behaviors.append((individual, behavior))

        # Calculate novelty scores
        for individual, behavior in behaviors:
            # Get novelty score from archive
            novelty_score = self.archive.get_novelty_score(behavior, self.k_neighbors)

            # Also consider novelty within current population
            pop_distances = []
            for other_ind, other_behavior in behaviors:
                if individual.id != other_ind.id:
                    pop_distances.append(behavior.distance(other_behavior))

            if pop_distances:
                pop_distances.sort()
                k = min(self.k_neighbors, len(pop_distances))
                pop_novelty = np.mean(pop_distances[:k])

                # Combine archive and population novelty
                combined_novelty = 0.7 * novelty_score + 0.3 * pop_novelty
            else:
                combined_novelty = novelty_score

            novelty_scores[individual.id] = combined_novelty

            # Store in individual metadata
            individual.metadata['novelty_score'] = combined_novelty
            individual.metadata['behavior_features'] = behavior.features.tolist()

            # Try to add to archive
            if self.archive.add_if_novel(behavior):
                individual.metadata['added_to_archive'] = True

        return novelty_scores

    def combine_fitness_novelty(self,
                              individual: Individual,
                              objective_fitness: float,
                              novelty_score: float) -> float:
        """Combine objective fitness and novelty into final fitness score."""
        # Apply novelty weight
        combined = (1 - self.novelty_weight) * objective_fitness + self.novelty_weight * novelty_score

        # Store components for analysis
        individual.metadata['objective_fitness'] = objective_fitness
        individual.metadata['novelty_fitness'] = novelty_score
        individual.metadata['combined_fitness'] = combined

        return combined

    def update_population_fitness(self,
                                population: Population,
                                objective_fitness: dict[str, float],
                                novelty_scores: dict[str, float]):
        """Update population with combined fitness-novelty scores."""
        for individual in population.individuals:
            obj_fit = objective_fitness.get(individual.id, 0.0)
            nov_score = novelty_scores.get(individual.id, 0.0)

            # Update fitness with combined score
            individual.fitness = self.combine_fitness_novelty(individual, obj_fit, nov_score)

    def get_archive_diversity(self) -> dict[str, Any]:
        """Calculate diversity metrics for the novelty archive."""
        if not self.archive.behaviors:
            return {
                'archive_size': 0,
                'average_distance': 0.0,
                'coverage': 0.0,
                'clusters': 0
            }

        # Calculate pairwise distances
        distances = []
        for i, b1 in enumerate(self.archive.behaviors):
            for j, b2 in enumerate(self.archive.behaviors):
                if i < j:
                    distances.append(b1.distance(b2))

        # Estimate coverage using convex hull volume approximation
        if len(self.archive.behaviors) > 50:
            # Sample for efficiency
            sample_indices = np.random.choice(len(self.archive.behaviors), 50, replace=False)
            sample_behaviors = [self.archive.behaviors[i] for i in sample_indices]
        else:
            sample_behaviors = self.archive.behaviors

        # Get feature matrix
        features = np.array([b.features for b in sample_behaviors])

        # Calculate coverage metrics
        feature_ranges = np.ptp(features, axis=0)  # Range per feature
        coverage = np.mean(feature_ranges > 0.1)  # Fraction of features with significant range

        return {
            'archive_size': len(self.archive.behaviors),
            'average_distance': np.mean(distances) if distances else 0.0,
            'min_distance': np.min(distances) if distances else 0.0,
            'max_distance': np.max(distances) if distances else 0.0,
            'coverage': coverage,
            'cache_size': len(self.behavior_cache)
        }

    def clear_cache(self):
        """Clear behavior characterization cache."""
        self.behavior_cache.clear()

    def save_archive(self, path: str):
        """Save novelty archive to file."""
        archive_data = {
            'behaviors': [
                {
                    'features': b.features.tolist(),
                    'metadata': b.metadata
                }
                for b in self.archive.behaviors
            ],
            'archive_stats': self.get_archive_diversity()
        }

        with open(path, 'w') as f:
            json.dump(archive_data, f, indent=2)

    def load_archive(self, path: str):
        """Load novelty archive from file."""
        with open(path) as f:
            archive_data = json.load(f)

        self.archive.behaviors = []
        for b_data in archive_data['behaviors']:
            behavior = BehaviorCharacterization(
                features=np.array(b_data['features'], dtype=np.float32),
                metadata=b_data['metadata']
            )
            self.archive.behaviors.append(behavior)
