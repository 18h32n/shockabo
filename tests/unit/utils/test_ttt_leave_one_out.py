"""
Unit tests for Leave-One-Out Task Generation
"""
import numpy as np
import pytest

from src.utils.ttt_leave_one_out import (
    LeaveOneOutConfig,
    LeaveOneOutGenerator,
    LeaveOneOutSplit,
)


@pytest.fixture
def sample_train_examples():
    """Sample training examples for testing."""
    return [
        {"input": [[0, 1], [2, 3]], "output": [[4, 5], [6, 7]]},
        {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
        {"input": [[2, 3], [4, 5]], "output": [[6, 7], [8, 9]]},
    ]


@pytest.fixture
def generator():
    """Create leave-one-out generator."""
    return LeaveOneOutGenerator()


class TestLeaveOneOutGenerator:
    """Test suite for LeaveOneOutGenerator."""

    def test_generate_splits_basic(self, generator, sample_train_examples):
        """Test basic split generation."""
        splits = generator.generate_splits(sample_train_examples)

        assert len(splits) == 3
        assert all(isinstance(s, LeaveOneOutSplit) for s in splits)

        for i, split in enumerate(splits):
            assert split.split_id == i
            assert split.val_index == i
            assert len(split.train_indices) == 2
            assert len(split.train_examples) == 2
            assert split.val_example == sample_train_examples[i]

    def test_split_correctness(self, generator, sample_train_examples):
        """Test that each split uses correct train/val examples."""
        splits = generator.generate_splits(sample_train_examples)

        for i, split in enumerate(splits):
            expected_train_indices = [j for j in range(3) if j != i]
            assert split.train_indices == expected_train_indices

            for train_idx in expected_train_indices:
                assert sample_train_examples[train_idx] in split.train_examples

            assert split.val_example not in split.train_examples

    def test_all_examples_used_as_validation(self, generator, sample_train_examples):
        """Test that all examples are used as validation exactly once."""
        splits = generator.generate_splits(sample_train_examples)

        val_indices = [split.val_index for split in splits]
        assert sorted(val_indices) == [0, 1, 2]

    def test_validate_input_empty_list(self, generator):
        """Test validation fails with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generator.generate_splits([])

    def test_validate_input_min_examples(self):
        """Test validation fails with too few examples."""
        config = LeaveOneOutConfig(min_examples=3)
        generator = LeaveOneOutGenerator(config)

        with pytest.raises(ValueError, match="Need at least 3 examples"):
            generator.generate_splits([{"input": [[0]], "output": [[1]]}])

    def test_validate_input_missing_keys(self, generator):
        """Test validation fails with missing input/output keys."""
        invalid_examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "wrong_key": [[1]]},
        ]

        with pytest.raises(ValueError, match="missing 'input' or 'output'"):
            generator.generate_splits(invalid_examples)

    def test_validate_grid_dimensions(self, generator):
        """Test grid dimension validation."""
        large_grid = [[0] * 31 for _ in range(31)]
        invalid_examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": large_grid, "output": [[1]]},
        ]

        with pytest.raises(ValueError, match="exceeds max dimensions"):
            generator.generate_splits(invalid_examples)

    def test_validate_grid_values(self, generator):
        """Test grid value validation (0-9 only)."""
        invalid_examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": [[10]], "output": [[1]]},
        ]

        with pytest.raises(ValueError, match="must be 0-9"):
            generator.generate_splits(invalid_examples)

        invalid_examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": [[-1]], "output": [[1]]},
        ]

        with pytest.raises(ValueError, match="must be 0-9"):
            generator.generate_splits(invalid_examples)

    def test_validate_grid_inconsistent_rows(self, generator):
        """Test validation fails with inconsistent row lengths."""
        invalid_examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0, 1], [2]], "output": [[1]]},
        ]

        with pytest.raises(ValueError, match="inconsistent row lengths"):
            generator.generate_splits(invalid_examples)

    def test_numpy_array_support(self, generator):
        """Test that numpy arrays are supported as grids."""
        numpy_examples = [
            {
                "input": np.array([[0, 1], [2, 3]]),
                "output": np.array([[4, 5], [6, 7]]),
            },
            {
                "input": np.array([[1, 2], [3, 4]]),
                "output": np.array([[5, 6], [7, 8]]),
            },
        ]

        splits = generator.generate_splits(numpy_examples)
        assert len(splits) == 2

    def test_track_validation_metrics(self, generator):
        """Test validation metrics tracking."""
        generator.track_validation_metrics(0, {"accuracy": 0.8, "loss": 0.2})
        generator.track_validation_metrics(1, {"accuracy": 0.9, "loss": 0.1})

        assert 0 in generator.validation_metrics
        assert 1 in generator.validation_metrics
        assert generator.validation_metrics[0]["accuracy"] == 0.8
        assert generator.validation_metrics[1]["accuracy"] == 0.9

    def test_get_aggregated_metrics(self, generator):
        """Test aggregated metrics calculation."""
        generator.track_validation_metrics(0, {"accuracy": 0.8, "loss": 0.2})
        generator.track_validation_metrics(1, {"accuracy": 0.9, "loss": 0.1})
        generator.track_validation_metrics(2, {"accuracy": 0.7, "loss": 0.3})

        aggregated = generator.get_aggregated_metrics()

        assert "accuracy_mean" in aggregated
        assert "accuracy_std" in aggregated
        assert "loss_mean" in aggregated
        assert "loss_std" in aggregated

        assert aggregated["accuracy_mean"] == pytest.approx(0.8, rel=1e-5)
        assert aggregated["loss_mean"] == pytest.approx(0.2, rel=1e-5)

    def test_get_aggregated_metrics_empty(self, generator):
        """Test aggregated metrics when no metrics tracked."""
        aggregated = generator.get_aggregated_metrics()
        assert aggregated == {}

    def test_get_best_split(self, generator):
        """Test finding best split by metric."""
        generator.track_validation_metrics(0, {"accuracy": 0.8})
        generator.track_validation_metrics(1, {"accuracy": 0.9})
        generator.track_validation_metrics(2, {"accuracy": 0.7})

        best_split = generator.get_best_split(metric="accuracy")
        assert best_split == 1

    def test_get_best_split_no_metrics(self, generator):
        """Test best split returns None when no metrics tracked."""
        best_split = generator.get_best_split()
        assert best_split is None

    def test_reset_metrics(self, generator):
        """Test metrics reset."""
        generator.track_validation_metrics(0, {"accuracy": 0.8})
        generator.track_validation_metrics(1, {"accuracy": 0.9})

        assert len(generator.validation_metrics) == 2

        generator.reset_metrics()

        assert len(generator.validation_metrics) == 0

    def test_max_examples_warning(self, generator, caplog):
        """Test warning when exceeding max examples."""
        config = LeaveOneOutConfig(max_examples=2)
        generator = LeaveOneOutGenerator(config)

        examples = [
            {"input": [[i]], "output": [[i + 1]]} for i in range(5)
        ]

        splits = generator.generate_splits(examples)
        assert len(splits) == 5

    def test_edge_case_single_pixel_grid(self, generator):
        """Test with single pixel grids."""
        examples = [
            {"input": [[0]], "output": [[1]]},
            {"input": [[2]], "output": [[3]]},
        ]
        splits = generator.generate_splits(examples)
        assert len(splits) == 2

    def test_edge_case_max_dimensions(self, generator):
        """Test with maximum allowed dimensions (30x30)."""
        grid_30x30_1 = [[i % 10 for i in range(30)] for _ in range(30)]
        grid_30x30_2 = [[(i + 1) % 10 for i in range(30)] for _ in range(30)]
        examples = [
            {"input": grid_30x30_1, "output": grid_30x30_1},
            {"input": grid_30x30_2, "output": grid_30x30_2},
        ]

        splits = generator.generate_splits(examples)
        assert len(splits) == 2
