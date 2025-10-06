"""
Unit tests for Self-Consistency Validation
"""
import numpy as np
import pytest

from src.utils.ttt_self_consistency import (
    PermutationResult,
    SelfConsistencyConfig,
    SelfConsistencyValidator,
)


@pytest.fixture
def sample_grid():
    """Sample grid for testing."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def validator():
    """Create self-consistency validator."""
    return SelfConsistencyValidator()


class TestSelfConsistencyValidator:
    """Test suite for SelfConsistencyValidator."""

    def test_generate_permutations_default(self, validator, sample_grid):
        """Test default permutation generation."""
        permutations = validator.generate_permutations(sample_grid)

        assert len(permutations) <= validator.config.permute_n
        assert all(isinstance(p, PermutationResult) for p in permutations)
        assert all(p.permuted_input is not None for p in permutations)
        assert all(p.reverse_function is not None for p in permutations)

    def test_permutation_reversibility(self, validator, sample_grid):
        """Test that permutations are reversible."""
        permutations = validator.generate_permutations(sample_grid)

        for perm in permutations:
            permuted = perm.permuted_input
            reversed_grid = perm.reverse_function(permuted)

            if perm.permutation_type == "identity":
                assert np.array_equal(reversed_grid, sample_grid)

    def test_geometric_permutations(self, validator, sample_grid):
        """Test geometric permutation generation."""
        config = SelfConsistencyConfig(permute_n=8, enable_geometric=True)
        validator = SelfConsistencyValidator(config)

        permutations = validator._generate_geometric_permutations(sample_grid)

        assert len(permutations) == 8
        types = [p.permutation_type for p in permutations]
        assert "identity" in types
        assert "rotate_90" in types
        assert "flip_horizontal" in types

    def test_rotate_90_correctness(self, validator):
        """Test 90-degree rotation correctness."""
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[2, 4], [1, 3]])

        permutations = validator._generate_geometric_permutations(grid)
        rotate_90 = next(p for p in permutations if p.permutation_type == "rotate_90")

        assert np.array_equal(rotate_90.permuted_input, expected)

        reversed_grid = rotate_90.reverse_function(rotate_90.permuted_input)
        assert np.array_equal(reversed_grid, grid)

    def test_flip_horizontal_correctness(self):
        """Test horizontal flip correctness."""
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[2, 1], [4, 3]])

        config = SelfConsistencyConfig(permute_n=8)
        validator = SelfConsistencyValidator(config)
        permutations = validator._generate_geometric_permutations(grid)
        flip_h = next(p for p in permutations if p.permutation_type == "flip_horizontal")

        assert np.array_equal(flip_h.permuted_input, expected)

        reversed_grid = flip_h.reverse_function(flip_h.permuted_input)
        assert np.array_equal(reversed_grid, grid)

    def test_transpose_correctness(self):
        """Test transpose correctness."""
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 4], [2, 5], [3, 6]])

        config = SelfConsistencyConfig(permute_n=8)
        validator = SelfConsistencyValidator(config)
        permutations = validator._generate_geometric_permutations(grid)
        transpose = next(p for p in permutations if p.permutation_type == "transpose")

        assert np.array_equal(transpose.permuted_input, expected)

        reversed_grid = transpose.reverse_function(transpose.permuted_input)
        assert np.array_equal(reversed_grid, grid)

    def test_color_permutations(self, validator):
        """Test color remapping permutations."""
        grid = np.array([[0, 1, 2], [1, 2, 0]])
        config = SelfConsistencyConfig(enable_color_remap=True)
        validator = SelfConsistencyValidator(config)

        permutations = validator._generate_color_permutations(grid)

        assert len(permutations) >= 1
        color_perm = permutations[0]
        assert color_perm.permutation_type == "color_remap"

        reversed_grid = color_perm.reverse_function(color_perm.permuted_input)
        assert np.array_equal(reversed_grid, grid)

    def test_color_permutations_single_color(self, validator):
        """Test color permutations with single color grid."""
        grid = np.array([[1, 1], [1, 1]])
        config = SelfConsistencyConfig(enable_color_remap=True)
        validator = SelfConsistencyValidator(config)

        permutations = validator._generate_color_permutations(grid)

        assert len(permutations) == 0

    def test_aggregate_predictions_single(self, validator):
        """Test aggregation with single prediction."""
        pred = np.array([[1, 2], [3, 4]])
        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=pred,
                predicted_output=pred,
                reverse_function=lambda x: x,
            )
        ]

        best_pred, confidence, metadata = validator.aggregate_predictions(
            permutation_results
        )

        assert np.array_equal(best_pred, pred)
        assert confidence == 1.0
        assert metadata["num_predictions"] == 1

    def test_aggregate_predictions_consensus(self, validator):
        """Test aggregation with consensus."""
        pred1 = np.array([[1, 2], [3, 4]])
        pred2 = np.array([[1, 2], [3, 4]])
        pred3 = np.array([[5, 6], [7, 8]])

        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=pred1,
                predicted_output=pred1,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="rotate_90",
                permuted_input=pred2,
                predicted_output=pred2,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="flip_horizontal",
                permuted_input=pred3,
                predicted_output=pred3,
                reverse_function=lambda x: x,
            ),
        ]

        best_pred, confidence, metadata = validator.aggregate_predictions(
            permutation_results
        )

        assert np.array_equal(best_pred, pred1)
        assert confidence == pytest.approx(2.0 / 3.0, rel=1e-5)
        assert metadata["num_predictions"] == 3
        assert metadata["agreement_rate"] == pytest.approx(2.0 / 3.0, rel=1e-5)

    def test_aggregate_predictions_no_predictions(self, validator):
        """Test aggregation with no predictions."""
        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=np.array([[1]]),
                predicted_output=None,
                reverse_function=lambda x: x,
            )
        ]

        best_pred, confidence, metadata = validator.aggregate_predictions(
            permutation_results
        )

        assert confidence == 0.0
        assert "error" in metadata
        assert metadata["num_predictions"] == 0

    def test_consensus_threshold(self, validator):
        """Test consensus threshold checking."""
        pred1 = np.array([[1, 2], [3, 4]])
        pred2 = np.array([[1, 2], [3, 4]])
        pred3 = np.array([[5, 6], [7, 8]])

        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=pred1,
                predicted_output=pred1,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="rotate_90",
                permuted_input=pred2,
                predicted_output=pred2,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="flip_horizontal",
                permuted_input=pred3,
                predicted_output=pred3,
                reverse_function=lambda x: x,
            ),
        ]

        _, _, metadata = validator.aggregate_predictions(
            permutation_results, consensus_threshold=0.6
        )

        assert metadata["consensus_met"] is True
        assert metadata["consensus_threshold"] == 0.6

        _, _, metadata = validator.aggregate_predictions(
            permutation_results, consensus_threshold=0.7
        )

        assert metadata["consensus_met"] is False

    def test_per_pixel_confidence(self, validator):
        """Test per-pixel confidence calculation."""
        pred1 = np.array([[1, 2], [3, 4]])
        pred2 = np.array([[1, 2], [3, 5]])
        pred3 = np.array([[1, 2], [3, 4]])

        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=pred1,
                predicted_output=pred1,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="rotate_90",
                permuted_input=pred2,
                predicted_output=pred2,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="flip_horizontal",
                permuted_input=pred3,
                predicted_output=pred3,
                reverse_function=lambda x: x,
            ),
        ]

        confidence_map = validator.calculate_per_pixel_confidence(permutation_results)

        assert confidence_map is not None
        assert confidence_map.shape == pred1.shape
        assert np.all((confidence_map >= 0) & (confidence_map <= 1))

        assert confidence_map[0, 0] == 1.0
        assert confidence_map[0, 1] == 1.0
        assert confidence_map[1, 0] == 1.0
        assert confidence_map[1, 1] == pytest.approx(2.0 / 3.0, rel=1e-5)

    def test_per_pixel_confidence_different_shapes(self, validator):
        """Test per-pixel confidence with different shaped predictions."""
        pred1 = np.array([[1, 2], [3, 4]])
        pred2 = np.array([[1, 2, 3], [4, 5, 6]])

        permutation_results = [
            PermutationResult(
                permutation_type="identity",
                permuted_input=pred1,
                predicted_output=pred1,
                reverse_function=lambda x: x,
            ),
            PermutationResult(
                permutation_type="rotate_90",
                permuted_input=pred2,
                predicted_output=pred2,
                reverse_function=lambda x: x,
            ),
        ]

        confidence_map = validator.calculate_per_pixel_confidence(permutation_results)

        assert confidence_map is None

    def test_permutation_caching(self, validator, sample_grid):
        """Test permutation caching."""
        grid_id = "test_grid"

        permutations1 = validator.generate_permutations(sample_grid, grid_id=grid_id)
        permutations2 = validator.generate_permutations(sample_grid, grid_id=grid_id)

        assert permutations1 is permutations2

        validator.clear_cache()
        permutations3 = validator.generate_permutations(sample_grid, grid_id=grid_id)
        assert permutations3 is not permutations1

    def test_max_permutations_limit(self, validator, sample_grid):
        """Test maximum permutations limit."""
        config = SelfConsistencyConfig(max_permutations=3)
        validator = SelfConsistencyValidator(config)

        permutations = validator.generate_permutations(sample_grid)

        assert len(permutations) <= 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = SelfConsistencyConfig(
            permute_n=5, consensus_threshold=0.8, enable_color_remap=True
        )
        validator = SelfConsistencyValidator(config)

        assert validator.config.permute_n == 5
        assert validator.config.consensus_threshold == 0.8
        assert validator.config.enable_color_remap is True

    def test_vote_predictions_tie(self, validator):
        """Test voting with tied predictions."""
        pred1 = np.array([[1, 2]])
        pred2 = np.array([[3, 4]])

        predictions = [pred1, pred2]
        votes = validator._vote_predictions(predictions)

        assert votes["confidence"] == 0.5
        assert votes["best_prediction"] is not None
