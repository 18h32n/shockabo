"""
Unit tests for novelty search implementation.

Tests Task 7.5: Implement novelty search as alternative to fitness optimization.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.adapters.strategies.evolution_engine import Individual, Population
from src.adapters.strategies.novelty_search import (
    BehaviorCharacterization,
    BehaviorExtractor,
    NoveltyArchive,
    NoveltySearch,
)
from src.domain.models import ARCTask, Grid


class TestBehaviorCharacterization:
    """Test behavior characterization functionality."""

    def test_behavior_distance(self):
        """Test distance calculation between behaviors."""
        features1 = np.array([1.0, 2.0, 3.0, 4.0])
        features2 = np.array([1.0, 2.0, 3.0, 7.0])  # Distance = 3.0

        behavior1 = BehaviorCharacterization(features=features1)
        behavior2 = BehaviorCharacterization(features=features2)

        distance = behavior1.distance(behavior2)
        assert pytest.approx(distance) == 3.0

    def test_behavior_hash(self):
        """Test behavior hashing for uniqueness."""
        features1 = np.array([1.0, 2.0, 3.0])
        features2 = np.array([1.0, 2.0, 3.0])
        features3 = np.array([1.0, 2.0, 4.0])

        b1 = BehaviorCharacterization(features=features1)
        b2 = BehaviorCharacterization(features=features2)
        b3 = BehaviorCharacterization(features=features3)

        # Same features should have same hash
        assert hash(b1) == hash(b2)
        # Different features should have different hash
        assert hash(b1) != hash(b3)


class TestBehaviorExtractor:
    """Test behavior feature extraction."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple test grid."""
        grid = MagicMock(spec=Grid)
        grid.shape = (3, 3)
        grid.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        return grid

    @pytest.fixture
    def symmetric_grid(self):
        """Create a symmetric test grid."""
        grid = MagicMock(spec=Grid)
        grid.shape = (3, 3)
        grid.data = [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
        return grid

    def test_extract_features_basic(self, simple_grid):
        """Test basic feature extraction."""
        features = BehaviorExtractor.extract_features(simple_grid, None)

        assert len(features) == 50  # Fixed feature vector size
        assert features[0] == 3  # Height
        assert features[1] == 3  # Width

        # Check color histogram (normalized)
        # Grid has one of each color 1-9
        for i in range(1, 10):
            assert pytest.approx(features[2 + i]) == 1/9

    def test_extract_features_none_grid(self):
        """Test feature extraction with failed execution."""
        features = BehaviorExtractor.extract_features(None, None)

        assert len(features) == 50
        assert all(f == 0.0 for f in features)

    def test_symmetry_detection(self, symmetric_grid):
        """Test symmetry feature extraction."""
        # Horizontal symmetry
        h_sym = BehaviorExtractor._check_horizontal_symmetry(symmetric_grid)
        assert h_sym == 1.0  # Perfect horizontal symmetry

        # Create vertically symmetric grid
        v_sym_grid = MagicMock(spec=Grid)
        v_sym_grid.shape = (3, 3)
        v_sym_grid.data = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]

        v_sym = BehaviorExtractor._check_vertical_symmetry(v_sym_grid)
        assert v_sym > 0  # Some vertical symmetry

    def test_repetition_detection(self):
        """Test pattern repetition detection."""
        # Repeating pattern grid
        repeat_grid = MagicMock(spec=Grid)
        repeat_grid.shape = (2, 4)
        repeat_grid.data = [[1, 2, 1, 2], [3, 4, 3, 4]]

        h_repeat = BehaviorExtractor._check_horizontal_repetition(repeat_grid)
        assert h_repeat > 0.5  # Strong horizontal repetition

    def test_connected_components(self):
        """Test connected component counting."""
        # Grid with 3 connected components
        grid = MagicMock(spec=Grid)
        grid.shape = (3, 3)
        grid.data = [[1, 1, 2], [1, 3, 2], [3, 3, 2]]

        components = BehaviorExtractor._count_connected_components(grid)
        assert components == 3

    def test_transformation_features(self):
        """Test transformation features when comparing input/output."""
        input_grid = MagicMock(spec=Grid)
        input_grid.shape = (2, 2)
        input_grid.data = [[1, 2], [3, 4]]

        output_grid = MagicMock(spec=Grid)
        output_grid.shape = (4, 4)  # Doubled size
        output_grid.data = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

        features = BehaviorExtractor.extract_features(output_grid, input_grid)

        # Size change features
        size_idx = 32  # Where transformation features start
        assert features[size_idx] == 2.0  # Height doubled
        assert features[size_idx + 1] == 2.0  # Width doubled

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        # Low entropy (uniform) grid
        uniform_grid = MagicMock(spec=Grid)
        uniform_grid.shape = (3, 3)
        uniform_grid.data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        entropy = BehaviorExtractor._calculate_entropy(uniform_grid)
        assert entropy == 0.0  # No entropy

        # High entropy grid
        diverse_grid = MagicMock(spec=Grid)
        diverse_grid.shape = (3, 3)
        diverse_grid.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        entropy = BehaviorExtractor._calculate_entropy(diverse_grid)
        assert entropy > 0.5  # High entropy


class TestNoveltyArchive:
    """Test novelty archive functionality."""

    def test_add_novel_behavior(self):
        """Test adding novel behaviors to archive."""
        archive = NoveltyArchive(min_distance=1.0)

        # First behavior always novel
        b1 = BehaviorCharacterization(features=np.array([0, 0, 0]))
        assert archive.add_if_novel(b1) is True
        assert len(archive.behaviors) == 1

        # Similar behavior - not novel
        b2 = BehaviorCharacterization(features=np.array([0.5, 0, 0]))
        assert archive.add_if_novel(b2) is False
        assert len(archive.behaviors) == 1

        # Different behavior - novel
        b3 = BehaviorCharacterization(features=np.array([2, 0, 0]))
        assert archive.add_if_novel(b3) is True
        assert len(archive.behaviors) == 2

    def test_archive_pruning(self):
        """Test archive pruning when size limit reached."""
        archive = NoveltyArchive(max_size=10, min_distance=0.1)

        # Add many behaviors
        for i in range(15):
            b = BehaviorCharacterization(features=np.array([i, 0, 0]))
            archive.add_if_novel(b)

        # Should have pruned to stay under max_size
        assert len(archive.behaviors) < archive.max_size

    def test_novelty_score_calculation(self):
        """Test novelty score calculation."""
        archive = NoveltyArchive()

        # Add some behaviors
        for i in range(5):
            b = BehaviorCharacterization(features=np.array([i * 2, 0, 0]))
            archive.behaviors.append(b)

        # Test novelty of new behavior
        test_behavior = BehaviorCharacterization(features=np.array([1, 0, 0]))
        novelty_score = archive.get_novelty_score(test_behavior, k=3)

        assert novelty_score > 0  # Should have some novelty

        # Very similar to existing should have low novelty
        similar_behavior = BehaviorCharacterization(features=np.array([0, 0, 0]))
        similar_score = archive.get_novelty_score(similar_behavior, k=3)

        # The similar behavior should actually have higher novelty in this case
        # because it's at the extreme and k=3 includes farther neighbors
        assert similar_score > novelty_score


class TestNoveltySearch:
    """Test main novelty search functionality."""

    @pytest.fixture
    def novelty_search(self):
        """Create novelty search instance."""
        return NoveltySearch(
            archive_size=100,
            min_novelty_distance=0.5,
            novelty_weight=0.5,
            k_neighbors=5
        )

    @pytest.fixture
    def mock_population(self):
        """Create mock population."""
        population = Population()
        for i in range(10):
            ind = Individual(operations=[])
            ind.id = f"ind_{i}"
            ind.fitness = i * 0.1
            population.add_individual(ind)
        return population

    @pytest.fixture
    def mock_task(self):
        """Create mock ARC task."""
        task = MagicMock(spec=ARCTask)
        input_grid = MagicMock(spec=Grid)
        input_grid.shape = (3, 3)
        input_grid.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        train_pair = MagicMock()
        train_pair.input = input_grid
        task.train_pairs = [train_pair]

        return task

    def test_behavior_characterization(self, novelty_search, mock_task):
        """Test characterizing individual behavior."""
        individual = Individual(operations=[{"type": "rotate", "params": {"k": 1}}])
        individual.id = "test_ind"

        # Mock execution result
        result_grid = MagicMock(spec=Grid)
        result_grid.shape = (3, 3)
        result_grid.data = [[3, 6, 9], [2, 5, 8], [1, 4, 7]]

        behavior = novelty_search.characterize_behavior(
            individual, result_grid, mock_task.train_pairs[0].input
        )

        assert isinstance(behavior, BehaviorCharacterization)
        assert len(behavior.features) == 50
        assert behavior.metadata['individual_id'] == "test_ind"

        # Check caching
        behavior2 = novelty_search.characterize_behavior(
            individual, result_grid, mock_task.train_pairs[0].input
        )
        assert behavior is behavior2  # Same object from cache

    def test_evaluate_novelty(self, novelty_search, mock_population, mock_task):
        """Test novelty evaluation for population."""
        # Create execution results with varying behaviors
        execution_results = {}
        for i, ind in enumerate(mock_population.individuals):
            grid = MagicMock(spec=Grid)
            grid.shape = (3, 3)
            # Different patterns for different individuals
            if i < 3:
                grid.data = [[i, i, i], [i, i, i], [i, i, i]]  # Uniform
            elif i < 6:
                grid.data = [[i, 0, i], [0, i, 0], [i, 0, i]]  # Pattern
            else:
                grid.data = [[j, j+1, j+2] for j in range(3)]  # Gradient

            execution_results[ind.id] = grid

        novelty_scores = novelty_search.evaluate_novelty(
            mock_population, mock_task, execution_results
        )

        assert len(novelty_scores) == len(mock_population.individuals)
        assert all(0 <= score <= 1 for score in novelty_scores.values())

        # Check that different patterns have different novelty
        uniform_novelty = [novelty_scores[f"ind_{i}"] for i in range(3)]
        pattern_novelty = [novelty_scores[f"ind_{i}"] for i in range(3, 6)]

        # Different groups should have different average novelty
        assert abs(np.mean(uniform_novelty) - np.mean(pattern_novelty)) > 0.01

    def test_combine_fitness_novelty(self, novelty_search):
        """Test combining objective fitness and novelty."""
        individual = Individual(operations=[])

        # Test with balanced weight (0.5)
        combined = novelty_search.combine_fitness_novelty(
            individual, objective_fitness=0.8, novelty_score=0.4
        )

        assert pytest.approx(combined) == 0.6  # (0.5 * 0.8 + 0.5 * 0.4)
        assert individual.metadata['objective_fitness'] == 0.8
        assert individual.metadata['novelty_fitness'] == 0.4

        # Test with novelty-only (weight = 1.0)
        novelty_search.novelty_weight = 1.0
        combined = novelty_search.combine_fitness_novelty(
            individual, objective_fitness=0.8, novelty_score=0.4
        )
        assert pytest.approx(combined) == 0.4

    def test_update_population_fitness(self, novelty_search, mock_population):
        """Test updating population with combined scores."""
        # Create fitness scores
        objective_fitness = {f"ind_{i}": i * 0.1 for i in range(10)}
        novelty_scores = {f"ind_{i}": (9 - i) * 0.1 for i in range(10)}  # Inverse

        novelty_search.update_population_fitness(
            mock_population, objective_fitness, novelty_scores
        )

        # Check updated fitness
        for i, ind in enumerate(mock_population.individuals):
            expected = 0.5 * (i * 0.1) + 0.5 * ((9 - i) * 0.1)
            assert pytest.approx(ind.fitness) == expected

    def test_archive_diversity_metrics(self, novelty_search):
        """Test archive diversity calculation."""
        # Empty archive
        metrics = novelty_search.get_archive_diversity()
        assert metrics['archive_size'] == 0
        assert metrics['average_distance'] == 0.0

        # Add some behaviors
        for i in range(5):
            behavior = BehaviorCharacterization(
                features=np.random.randn(50).astype(np.float32)
            )
            novelty_search.archive.behaviors.append(behavior)

        metrics = novelty_search.get_archive_diversity()
        assert metrics['archive_size'] == 5
        assert metrics['average_distance'] > 0
        assert 'coverage' in metrics

    def test_save_load_archive(self, novelty_search):
        """Test saving and loading novelty archive."""
        # Add some behaviors
        for i in range(3):
            behavior = BehaviorCharacterization(
                features=np.array([i, i*2, i*3], dtype=np.float32),
                metadata={'test': i}
            )
            novelty_search.archive.behaviors.append(behavior)

        # Save archive
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            novelty_search.save_archive(temp_path)

            # Create new instance and load
            new_search = NoveltySearch()
            new_search.load_archive(temp_path)

            # Verify loaded archive
            assert len(new_search.archive.behaviors) == 3
            assert np.array_equal(
                new_search.archive.behaviors[0].features,
                novelty_search.archive.behaviors[0].features
            )
        finally:
            os.unlink(temp_path)

    def test_cache_clearing(self, novelty_search):
        """Test behavior cache management."""
        # Add to cache
        individual = Individual(operations=[{"type": "test"}])
        grid = MagicMock(spec=Grid)
        grid.shape = (2, 2)
        grid.data = [[1, 2], [3, 4]]

        novelty_search.characterize_behavior(individual, grid)
        assert len(novelty_search.behavior_cache) == 1

        # Clear cache
        novelty_search.clear_cache()
        assert len(novelty_search.behavior_cache) == 0
