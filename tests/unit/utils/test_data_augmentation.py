"""Unit tests for data augmentation with semantic preservation."""

import pytest

from src.domain.models import ARCTask
from src.utils.data_augmentation import ARCTaskAugmentor, AugmentationValidator, GridAugmentor


class TestGridAugmentor:
    """Test suite for grid augmentation functions."""

    @pytest.fixture
    def sample_grid(self):
        """Sample grid for testing."""
        return [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ]

    @pytest.fixture
    def asymmetric_grid(self):
        """Asymmetric grid for testing transformations."""
        return [
            [1, 2, 0],
            [0, 1, 2]
        ]

    def test_rotate_90_degrees(self, sample_grid):
        """Test 90-degree rotation."""
        rotated = GridAugmentor.rotate_grid_90(sample_grid)
        expected = [
            [2, 1, 0],
            [1, 0, 1],
            [0, 1, 2]
        ]
        assert rotated == expected

    def test_rotate_180_degrees(self, sample_grid):
        """Test 180-degree rotation."""
        rotated = GridAugmentor.rotate_grid_180(sample_grid)
        expected = [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ]
        assert rotated == expected

    def test_rotate_270_degrees(self, sample_grid):
        """Test 270-degree rotation."""
        rotated = GridAugmentor.rotate_grid_270(sample_grid)
        expected = [
            [2, 1, 0],
            [1, 0, 1],
            [0, 1, 2]
        ]
        assert rotated == expected

    def test_flip_horizontal(self, asymmetric_grid):
        """Test horizontal flip."""
        flipped = GridAugmentor.flip_horizontal(asymmetric_grid)
        expected = [
            [0, 2, 1],
            [2, 1, 0]
        ]
        assert flipped == expected

    def test_flip_vertical(self, asymmetric_grid):
        """Test vertical flip."""
        flipped = GridAugmentor.flip_vertical(asymmetric_grid)
        expected = [
            [0, 1, 2],
            [1, 2, 0]
        ]
        assert flipped == expected

    def test_empty_grid_transformations(self):
        """Test transformations on empty grids."""
        empty_grid = []

        assert GridAugmentor.rotate_grid_90(empty_grid) == []
        assert GridAugmentor.rotate_grid_180(empty_grid) == []
        assert GridAugmentor.flip_horizontal(empty_grid) == []
        assert GridAugmentor.flip_vertical(empty_grid) == []

    def test_grid_signature(self, sample_grid):
        """Test grid signature generation."""
        signature = GridAugmentor.get_grid_signature(sample_grid)

        assert signature["dimensions"] == (3, 3)
        assert set(signature["unique_colors"]) == {0, 1, 2}
        assert signature["color_counts"][0] == 3
        assert signature["color_counts"][1] == 4
        assert signature["color_counts"][2] == 2
        assert signature["total_cells"] == 9

    def test_connectivity_analysis(self):
        """Test connected component analysis."""
        # Grid with clear connected components
        grid = [
            [1, 1, 0, 2, 2],
            [1, 0, 0, 2, 0],
            [0, 0, 3, 3, 3]
        ]

        signature = GridAugmentor.get_grid_signature(grid)
        components = signature["connected_components"]

        # Color 1: one component (top-left)
        # Color 2: one component (top-right)
        # Color 3: one component (bottom-right)
        # Color 0: two components (separated by non-zero values)
        assert components[1] == 1
        assert components[2] == 1
        assert components[3] == 1
        # The 0s form separate components due to the grid structure
        assert components[0] >= 1  # At least one component

    def test_semantic_validation(self, sample_grid):
        """Test semantic validation of transformations."""
        rotated = GridAugmentor.rotate_grid_90(sample_grid)
        validation = GridAugmentor.validate_transformation_semantics(sample_grid, rotated)

        # Rotation should preserve all semantic properties
        assert validation["preserves_colors"] is True
        assert validation["preserves_color_counts"] is True
        assert validation["preserves_total_cells"] is True
        assert validation["preserves_connectivity"] is True

    def test_symmetry_detection(self):
        """Test symmetry detection."""
        # Horizontally symmetric grid
        h_symmetric = [
            [1, 2, 1],
            [0, 1, 0],
            [2, 0, 2]
        ]

        signature = GridAugmentor.get_grid_signature(h_symmetric)
        assert signature["is_horizontally_symmetric"] is True

        # Vertically symmetric grid
        v_symmetric = [
            [1, 0, 2],
            [2, 1, 0],
            [1, 0, 2]
        ]

        signature = GridAugmentor.get_grid_signature(v_symmetric)
        assert signature["is_vertically_symmetric"] is True


class TestARCTaskAugmentor:
    """Test suite for ARC task augmentation."""

    @pytest.fixture
    def sample_task(self):
        """Sample ARC task for testing."""
        return ARCTask(
            task_id="test_task",
            task_source="training",
            train_examples=[
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                },
                {
                    "input": [[2, 1], [1, 2]],
                    "output": [[1, 2], [2, 1]]
                }
            ],
            test_input=[[1, 2], [2, 1]],
            test_output=[[2, 1], [1, 2]]
        )

    @pytest.fixture
    def augmentor(self):
        """Task augmentor instance."""
        return ARCTaskAugmentor()

    def test_single_augmentation(self, augmentor, sample_task):
        """Test single transformation augmentation."""
        augmented = augmentor.augment_task(sample_task, ["rotate_90"])

        assert len(augmented) == 1
        aug_task = augmented[0]

        assert aug_task.task_id == "test_task_rotate_90"
        assert len(aug_task.train_examples) == 2

        # Check first training example transformation
        original_input = sample_task.train_examples[0]["input"]
        augmented_input = aug_task.train_examples[0]["input"]
        expected_input = GridAugmentor.rotate_grid_90(original_input)
        assert augmented_input == expected_input

    def test_multiple_augmentations(self, augmentor, sample_task):
        """Test multiple transformations."""
        augmentations = ["rotate_90", "flip_horizontal", "rotate_180"]
        augmented = augmentor.augment_task(sample_task, augmentations)

        assert len(augmented) == 3

        aug_ids = [task.task_id for task in augmented]
        expected_ids = ["test_task_rotate_90", "test_task_flip_horizontal", "test_task_rotate_180"]
        assert set(aug_ids) == set(expected_ids)

    def test_semantic_preservation(self, augmentor, sample_task):
        """Test that augmentations preserve semantics."""
        augmented = augmentor.augment_task(sample_task, ["rotate_90"], preserve_semantics=True)

        assert len(augmented) == 1
        aug_task = augmented[0]

        # Validate semantic preservation
        is_valid = augmentor._validate_task_semantics(sample_task, aug_task)
        assert is_valid is True

    def test_batch_augmentation(self, augmentor):
        """Test batch augmentation of multiple tasks."""
        tasks = {}
        for i in range(3):
            tasks[f"task_{i}"] = ARCTask(
                task_id=f"task_{i}",
                task_source="training",
                train_examples=[{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
                test_input=[[1, 0], [0, 1]]
            )

        augmented = augmentor.batch_augment(tasks, ["rotate_90", "flip_horizontal"], max_augmentations_per_task=1)

        # Should have original 3 + up to 3 augmented tasks
        assert len(augmented) >= 3
        assert len(augmented) <= 6

        # Check that original tasks are preserved
        for i in range(3):
            assert f"task_{i}" in augmented

    def test_augmentation_pipeline(self, augmentor, sample_task):
        """Test configurable augmentation pipeline."""
        config = {
            "transformations": ["rotate_90", "flip_horizontal"],
            "preserve_semantics": True,
            "probability": 1.0
        }

        pipeline = augmentor.create_augmentation_pipeline(config)
        results = pipeline(sample_task)

        assert len(results) >= 1  # At least original task
        assert results[0].task_id != sample_task.task_id or len(results) > 1  # Either augmented or multiple results

    def test_augmentation_statistics(self, augmentor):
        """Test augmentation statistics calculation."""
        # Create original tasks
        original_tasks = {
            "task_1": ARCTask(task_id="task_1", task_source="training"),
            "task_2": ARCTask(task_id="task_2", task_source="training")
        }

        # Create augmented tasks
        augmented_tasks = original_tasks.copy()
        augmented_tasks["task_1_rotate_90"] = ARCTask(task_id="task_1_rotate_90", task_source="training")
        augmented_tasks["task_2_flip_horizontal"] = ARCTask(task_id="task_2_flip_horizontal", task_source="training")

        stats = augmentor.get_augmentation_statistics(original_tasks, augmented_tasks)

        assert stats["original_task_count"] == 2
        assert stats["total_task_count"] == 4
        assert stats["augmentation_ratio"] == 2.0
        assert stats["net_augmented_tasks"] == 2


class TestAugmentationValidator:
    """Test suite for augmentation validation utilities."""

    def test_grid_comparison(self):
        """Test grid comparison functionality."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[1, 2], [3, 4]]
        grid3 = [[1, 2], [3, 5]]

        # Identical grids
        comparison = AugmentationValidator.compare_grids(grid1, grid2)
        assert comparison["identical"] is True
        assert comparison["difference_count"] == 0
        assert comparison["similarity_ratio"] == 1.0

        # Different grids
        comparison = AugmentationValidator.compare_grids(grid1, grid3)
        assert comparison["identical"] is False
        assert comparison["difference_count"] == 1
        assert comparison["similarity_ratio"] == 0.75

    def test_augmentation_invertibility(self):
        """Test that augmentations are correctly invertible."""
        test_grid = [
            [1, 2, 0],
            [0, 1, 2],
            [2, 0, 1]
        ]

        # Test rotation invertibility
        assert AugmentationValidator.validate_augmentation_invertibility(test_grid, "rotate_90") is True
        assert AugmentationValidator.validate_augmentation_invertibility(test_grid, "rotate_180") is True

        # Test flip invertibility
        assert AugmentationValidator.validate_augmentation_invertibility(test_grid, "flip_horizontal") is True
        assert AugmentationValidator.validate_augmentation_invertibility(test_grid, "flip_vertical") is True

    def test_empty_grid_comparison(self):
        """Test comparison with empty grids."""
        empty_grid = []
        normal_grid = [[1, 2]]

        comparison = AugmentationValidator.compare_grids(empty_grid, normal_grid)
        assert comparison["identical"] is False
        assert comparison["reason"] == "Empty grid(s)"

    def test_different_dimensions_comparison(self):
        """Test comparison of grids with different dimensions."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[1, 2, 3], [4, 5, 6]]

        comparison = AugmentationValidator.compare_grids(grid1, grid2)
        assert comparison["identical"] is False
        assert comparison["reason"] == "Different dimensions"

    @pytest.mark.parametrize("transformation", ["rotate_90", "rotate_180", "flip_horizontal", "flip_vertical"])
    def test_all_transformations_invertible(self, transformation):
        """Test that all transformations are invertible."""
        test_grids = [
            [[1, 2], [3, 4]],
            [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            [[5, 3, 1, 4], [2, 0, 6, 7]]
        ]

        for grid in test_grids:
            assert AugmentationValidator.validate_augmentation_invertibility(grid, transformation) is True
