import numpy as np
import pytest

from src.adapters.strategies.bandit_controller import BanditController
from src.adapters.strategies.task_feature_extractor import (
    ContextualBanditController,
    TaskFeatureExtractor,
)
from src.domain.models import ARCTask


class TestTaskFeatureExtractor:
    """Unit tests for TaskFeatureExtractor."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = TaskFeatureExtractor()
        assert extractor is not None

    def test_extract_features_basic_grid(self):
        """Test feature extraction from basic grid."""
        task = ARCTask(
            task_id='test_1',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert 'grid_height' in features
        assert 'grid_width' in features
        assert 'grid_area' in features
        assert 'unique_colors' in features
        assert 'edge_density' in features

        assert features['grid_height'] == 2 / 30
        assert features['grid_width'] == 2 / 30
        assert features['grid_area'] == 4 / 900

    def test_extract_features_empty_grid(self):
        """Test feature extraction from empty grid."""
        task = ARCTask(
            task_id='test_empty',
            task_source='test',
            test_input=[]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        default_features = extractor._get_default_features()
        assert features == default_features

    def test_grid_features_normalization(self):
        """Test grid features are normalized correctly."""
        task = ARCTask(
            task_id='test_large',
            task_source='test',
            test_input=[[0] * 30 for _ in range(30)]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['grid_height'] == 1.0
        assert features['grid_width'] == 1.0
        assert features['grid_area'] == 1.0

    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation."""
        task = ARCTask(
            task_id='test_aspect',
            task_source='test',
            test_input=[[0] * 10 for _ in range(5)]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['aspect_ratio'] == pytest.approx(10 / 5)

    def test_color_features_unique_colors(self):
        """Test unique color counting."""
        task = ARCTask(
            task_id='test_colors',
            task_source='test',
            test_input=[[1, 2, 3], [4, 5, 6]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['unique_colors'] == 6 / 10

    def test_color_features_entropy(self):
        """Test color entropy calculation."""
        task = ARCTask(
            task_id='test_entropy',
            task_source='test',
            test_input=[[1, 1], [1, 1]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['color_entropy'] == 0.0

    def test_color_features_dominant_ratio(self):
        """Test dominant color ratio."""
        task = ARCTask(
            task_id='test_dominant',
            task_source='test',
            test_input=[[1, 1, 1], [1, 2, 2]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['dominant_color_ratio'] == pytest.approx(4 / 6)

    def test_edge_density_uniform_grid(self):
        """Test edge density on uniform grid."""
        task = ARCTask(
            task_id='test_uniform',
            task_source='test',
            test_input=[[1, 1], [1, 1]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['edge_density'] == 0.0

    def test_edge_density_checkerboard(self):
        """Test edge density on checkerboard pattern."""
        task = ARCTask(
            task_id='test_checkerboard',
            task_source='test',
            test_input=[[0, 1], [1, 0]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['edge_density'] == 1.0

    def test_horizontal_symmetry_detection(self):
        """Test horizontal symmetry detection."""
        task = ARCTask(
            task_id='test_hsym',
            task_source='test',
            test_input=[[1, 2, 1], [3, 4, 3], [1, 2, 1]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['horizontal_symmetry'] == 1.0

    def test_horizontal_symmetry_not_present(self):
        """Test horizontal symmetry when not present."""
        task = ARCTask(
            task_id='test_no_hsym',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['horizontal_symmetry'] == 0.0

    def test_vertical_symmetry_detection(self):
        """Test vertical symmetry detection."""
        task = ARCTask(
            task_id='test_vsym',
            task_source='test',
            test_input=[[1, 2, 1], [3, 4, 3]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['vertical_symmetry'] == 1.0

    def test_vertical_symmetry_not_present(self):
        """Test vertical symmetry when not present."""
        task = ARCTask(
            task_id='test_no_vsym',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['vertical_symmetry'] == 0.0

    def test_rotational_symmetry_detection(self):
        """Test rotational symmetry detection."""
        task = ARCTask(
            task_id='test_rsym',
            task_source='test',
            test_input=[[1, 2], [2, 1]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['rotational_symmetry'] == 1.0

    def test_rotational_symmetry_not_present(self):
        """Test rotational symmetry when not present."""
        task = ARCTask(
            task_id='test_no_rsym',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        extractor = TaskFeatureExtractor()
        features = extractor.extract_features(task)

        assert features['rotational_symmetry'] == 0.0

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        extractor = TaskFeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 11
        assert 'grid_height' in names
        assert 'edge_density' in names
        assert 'vertical_symmetry' in names

    def test_features_to_vector(self):
        """Test feature dictionary to vector conversion."""
        extractor = TaskFeatureExtractor()
        features = {
            'grid_height': 0.5,
            'grid_width': 0.6,
            'unique_colors': 0.7
        }

        vector = extractor.features_to_vector(features)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 11
        assert vector[0] == 0.5  # grid_height
        assert vector[1] == 0.6  # grid_width

    def test_features_to_vector_missing_features(self):
        """Test vector conversion with missing features (defaults to 0)."""
        extractor = TaskFeatureExtractor()
        features = {'grid_height': 0.5}

        vector = extractor.features_to_vector(features)

        assert vector[0] == 0.5
        assert vector[1] == 0.0  # Missing features default to 0


class TestContextualBanditController:
    """Unit tests for ContextualBanditController."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        base_controller = BanditController(strategies=['a', 'b'])
        extractor = TaskFeatureExtractor()

        contextual = ContextualBanditController(
            base_controller=base_controller,
            feature_extractor=extractor,
            num_clusters=5,
            min_cluster_size=10
        )

        assert contextual.num_clusters == 5
        assert contextual.min_cluster_size == 10
        assert len(contextual.cluster_strategy_stats) == 0

    def test_initialization_invalid_clusters(self):
        """Test initialization fails with invalid num_clusters."""
        base_controller = BanditController(strategies=['a'])
        extractor = TaskFeatureExtractor()

        with pytest.raises(ValueError, match="num_clusters must be positive"):
            ContextualBanditController(
                base_controller=base_controller,
                feature_extractor=extractor,
                num_clusters=0
            )

    def test_initialization_invalid_min_cluster_size(self):
        """Test initialization fails with invalid min_cluster_size."""
        base_controller = BanditController(strategies=['a'])
        extractor = TaskFeatureExtractor()

        with pytest.raises(ValueError, match="min_cluster_size must be positive"):
            ContextualBanditController(
                base_controller=base_controller,
                feature_extractor=extractor,
                min_cluster_size=0
            )

    def test_select_strategy_fallback_to_base(self):
        """Test strategy selection falls back to base controller."""
        base_controller = BanditController(strategies=['a', 'b'], random_seed=42)
        extractor = TaskFeatureExtractor()
        contextual = ContextualBanditController(
            base_controller=base_controller,
            feature_extractor=extractor
        )

        task = ARCTask(
            task_id='test_1',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        strategy = contextual.select_strategy(task)
        assert strategy in ['a', 'b']

    def test_update_reward_tracks_clusters(self):
        """Test reward update tracks cluster statistics."""
        base_controller = BanditController(strategies=['a'])
        extractor = TaskFeatureExtractor()
        contextual = ContextualBanditController(
            base_controller=base_controller,
            feature_extractor=extractor
        )

        task = ARCTask(
            task_id='test_1',
            task_source='test',
            test_input=[[1, 2], [3, 4]]
        )

        contextual.update_reward(task, 'a', reward=0.8)

        assert len(contextual.task_cluster_assignments) == 1
        assert len(contextual.cluster_strategy_stats) == 1

    def test_assign_cluster_no_centroids(self):
        """Test cluster assignment with no centroids (defaults to 0)."""
        base_controller = BanditController(strategies=['a'])
        extractor = TaskFeatureExtractor()
        contextual = ContextualBanditController(
            base_controller=base_controller,
            feature_extractor=extractor
        )

        feature_vector = np.array([0.5] * 11)
        cluster_id = contextual._assign_cluster(feature_vector)

        assert cluster_id == 0

    def test_assign_cluster_with_centroids(self):
        """Test cluster assignment with existing centroids."""
        base_controller = BanditController(strategies=['a'])
        extractor = TaskFeatureExtractor()
        contextual = ContextualBanditController(
            base_controller=base_controller,
            feature_extractor=extractor
        )

        # Set up centroids
        contextual.cluster_centroids = [
            np.array([0.1] * 11),
            np.array([0.9] * 11)
        ]

        # Feature vector close to first centroid
        feature_vector = np.array([0.15] * 11)
        cluster_id = contextual._assign_cluster(feature_vector)
        assert cluster_id == 0

        # Feature vector close to second centroid
        feature_vector = np.array([0.85] * 11)
        cluster_id = contextual._assign_cluster(feature_vector)
        assert cluster_id == 1
