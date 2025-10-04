from collections import Counter

import numpy as np

from src.domain.models import ARCTask


class TaskFeatureExtractor:
    """
    Extracts features from ARC tasks for contextual bandit selection.

    Features include grid size, color diversity, and pattern complexity
    to enable context-aware strategy selection.
    """

    def __init__(self):
        """Initialize feature extractor."""
        pass

    def extract_features(self, task: ARCTask) -> dict[str, float]:
        """
        Extract normalized feature vector from ARC task.

        Args:
            task: ARCTask to extract features from

        Returns:
            Dictionary of normalized features (0.0-1.0 scale):
                Grid features:
                    - grid_height: Normalized height (0-1, max 30)
                    - grid_width: Normalized width (0-1, max 30)
                    - grid_area: Normalized area (0-1, max 900)
                    - aspect_ratio: width / height (0-1, normalized by max observed)
                Color features:
                    - unique_colors: Number of distinct colors (0-1, max 10)
                    - color_entropy: Entropy of color distribution (0-1)
                    - dominant_color_ratio: Ratio of most frequent color (0-1)
                Pattern features:
                    - edge_density: Color transitions / total cells (0-1)
                    - horizontal_symmetry: Whether grid has horizontal symmetry (0 or 1)
                    - vertical_symmetry: Whether grid has vertical symmetry (0 or 1)
                    - rotational_symmetry: Whether grid has 180° rotational symmetry (0 or 1)
        """
        # Use test_input as primary grid for feature extraction
        grid = np.array(task.test_input)

        if grid.size == 0:
            return self._get_default_features()

        features = {}

        # Extract grid features
        features.update(self._extract_grid_features(grid))

        # Extract color features
        features.update(self._extract_color_features(grid))

        # Extract pattern features
        features.update(self._extract_pattern_features(grid))

        return features

    def _extract_grid_features(self, grid: np.ndarray) -> dict[str, float]:
        """Extract grid size and aspect ratio features."""
        height, width = grid.shape
        area = height * width

        # Normalize by typical ARC grid limits
        max_dimension = 30
        max_area = 900

        return {
            'grid_height': min(height / max_dimension, 1.0),
            'grid_width': min(width / max_dimension, 1.0),
            'grid_area': min(area / max_area, 1.0),
            'aspect_ratio': width / height if height > 0 else 0.0
        }

    def _extract_color_features(self, grid: np.ndarray) -> dict[str, float]:
        """Extract color diversity and distribution features."""
        flat_grid = grid.flatten()
        color_counts = Counter(flat_grid)

        unique_colors = len(color_counts)
        total_cells = len(flat_grid)

        # Calculate color entropy
        probabilities = np.array([count / total_cells for count in color_counts.values()])
        # Avoid log(0) by adding epsilon only when needed
        entropy = -np.sum(probabilities * np.log2(np.where(probabilities > 0, probabilities, 1.0)))
        entropy = max(0.0, entropy)  # Clamp to non-negative

        # Max entropy for 10 colors
        max_entropy = np.log2(10)

        # Dominant color ratio
        most_common_count = max(color_counts.values()) if color_counts else 0
        dominant_ratio = most_common_count / total_cells if total_cells > 0 else 0.0

        return {
            'unique_colors': min(unique_colors / 10.0, 1.0),
            'color_entropy': min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0,
            'dominant_color_ratio': dominant_ratio
        }

    def _extract_pattern_features(self, grid: np.ndarray) -> dict[str, float]:
        """Extract pattern complexity features."""
        height, width = grid.shape

        # Edge density: count color transitions
        horizontal_edges = np.sum(grid[:, :-1] != grid[:, 1:]) if width > 1 else 0
        vertical_edges = np.sum(grid[:-1, :] != grid[1:, :]) if height > 1 else 0
        total_edges = horizontal_edges + vertical_edges
        max_edges = (width - 1) * height + (height - 1) * width
        edge_density = total_edges / max_edges if max_edges > 0 else 0.0

        # Symmetry detection
        horizontal_sym = self._check_horizontal_symmetry(grid)
        vertical_sym = self._check_vertical_symmetry(grid)
        rotational_sym = self._check_rotational_symmetry(grid)

        return {
            'edge_density': edge_density,
            'horizontal_symmetry': 1.0 if horizontal_sym else 0.0,
            'vertical_symmetry': 1.0 if vertical_sym else 0.0,
            'rotational_symmetry': 1.0 if rotational_sym else 0.0
        }

    def _check_horizontal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has horizontal (top-bottom) symmetry."""
        return np.array_equal(grid, np.flipud(grid))

    def _check_vertical_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has vertical (left-right) symmetry."""
        return np.array_equal(grid, np.fliplr(grid))

    def _check_rotational_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has 180-degree rotational symmetry."""
        return np.array_equal(grid, np.rot90(grid, 2))

    def _get_default_features(self) -> dict[str, float]:
        """Get default feature vector for empty grids."""
        return {
            'grid_height': 0.0,
            'grid_width': 0.0,
            'grid_area': 0.0,
            'aspect_ratio': 0.0,
            'unique_colors': 0.0,
            'color_entropy': 0.0,
            'dominant_color_ratio': 0.0,
            'edge_density': 0.0,
            'horizontal_symmetry': 0.0,
            'vertical_symmetry': 0.0,
            'rotational_symmetry': 0.0
        }

    def get_feature_names(self) -> list[str]:
        """
        Get list of all feature names.

        Returns:
            List of feature names in extraction order
        """
        return [
            'grid_height',
            'grid_width',
            'grid_area',
            'aspect_ratio',
            'unique_colors',
            'color_entropy',
            'dominant_color_ratio',
            'edge_density',
            'horizontal_symmetry',
            'vertical_symmetry',
            'rotational_symmetry'
        ]

    def features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to numpy vector.

        Args:
            features: Feature dictionary from extract_features

        Returns:
            Numpy array of feature values in consistent order
        """
        feature_names = self.get_feature_names()
        return np.array([features.get(name, 0.0) for name in feature_names])


class ContextualBanditController:
    """
    Extension of BanditController with contextual task features.

    Clusters tasks by feature similarity and tracks strategy performance
    per cluster for context-aware selection.
    """

    def __init__(
        self,
        base_controller,
        feature_extractor: TaskFeatureExtractor,
        num_clusters: int = 5,
        min_cluster_size: int = 10
    ):
        """
        Initialize contextual bandit controller.

        Args:
            base_controller: BanditController instance for strategy selection
            feature_extractor: TaskFeatureExtractor for extracting task features
            num_clusters: Number of feature clusters (default: 5)
            min_cluster_size: Minimum cluster size to use contextual selection (default: 10)
        """
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive")

        self.base_controller = base_controller
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size

        # Cluster-specific strategy tracking
        self.cluster_strategy_stats: dict[int, dict[str, dict[str, float]]] = {}
        self.task_cluster_assignments: list[tuple[np.ndarray, int]] = []

        # Cluster centroids (updated via k-means)
        self.cluster_centroids: list[np.ndarray] = []

    def select_strategy(self, task: ARCTask) -> str:
        """
        Select strategy based on task features.

        Args:
            task: ARCTask to select strategy for

        Returns:
            Strategy ID selected via contextual bandit
        """
        features = self.feature_extractor.extract_features(task)
        feature_vector = self.feature_extractor.features_to_vector(features)

        # Assign to nearest cluster
        cluster_id = self._assign_cluster(feature_vector)

        # Check if cluster has enough data for contextual selection
        cluster_stats = self.cluster_strategy_stats.get(cluster_id, {})
        if not cluster_stats or len(self.task_cluster_assignments) < self.min_cluster_size:
            # Fall back to global Thompson Sampling
            return self.base_controller.select_strategy(task_features=features)

        # TODO: Implement cluster-specific selection logic
        # For now, fall back to base controller
        return self.base_controller.select_strategy(task_features=features)

    def _assign_cluster(self, feature_vector: np.ndarray) -> int:
        """
        Assign feature vector to nearest cluster.

        Args:
            feature_vector: Normalized feature vector

        Returns:
            Cluster ID (0 to num_clusters-1)
        """
        if not self.cluster_centroids:
            # No clusters yet - assign to cluster 0
            return 0

        # Find nearest centroid
        distances = [
            np.linalg.norm(feature_vector - centroid)
            for centroid in self.cluster_centroids
        ]
        return int(np.argmin(distances))

    def update_reward(
        self,
        task: ARCTask,
        strategy_id: str,
        reward: float,
        cost: float = 0.0,
        is_failure: bool = False
    ) -> None:
        """
        Update strategy statistics with task context.

        Args:
            task: ARCTask that was processed
            strategy_id: Strategy that was used
            reward: Reward value
            cost: API cost
            is_failure: Whether execution failed
        """
        # Update base controller
        self.base_controller.update_reward(strategy_id, reward, cost, is_failure)

        # Extract features and assign cluster
        features = self.feature_extractor.extract_features(task)
        feature_vector = self.feature_extractor.features_to_vector(features)
        cluster_id = self._assign_cluster(feature_vector)

        # Track task-cluster assignment
        self.task_cluster_assignments.append((feature_vector, cluster_id))

        # Update cluster-specific statistics
        if cluster_id not in self.cluster_strategy_stats:
            self.cluster_strategy_stats[cluster_id] = {}
        if strategy_id not in self.cluster_strategy_stats[cluster_id]:
            self.cluster_strategy_stats[cluster_id][strategy_id] = {
                'success_count': 0,
                'failure_count': 0,
                'total_reward': 0.0
            }

        stats = self.cluster_strategy_stats[cluster_id][strategy_id]
        if is_failure or reward <= 0.5:
            stats['failure_count'] += 1
        else:
            stats['success_count'] += 1
        stats['total_reward'] += reward

        # Periodically update cluster centroids (every 50 tasks)
        if len(self.task_cluster_assignments) % 50 == 0:
            self._update_clusters()

    def _update_clusters(self) -> None:
        """Update cluster centroids using k-means."""
        if len(self.task_cluster_assignments) < self.num_clusters:
            return

        # Extract feature vectors
        feature_vectors = np.array([fv for fv, _ in self.task_cluster_assignments])

        # Simple k-means (1 iteration)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(feature_vectors)

        self.cluster_centroids = kmeans.cluster_centers_.tolist()
