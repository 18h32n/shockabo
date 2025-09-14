"""Integration tests for TTT with data pipeline."""

import pytest

from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask
from src.utils.grid_ops import grid_to_string, string_to_grid


class TestTTTDataIntegration:
    """Test TTT integration with data pipeline from Story 1.2."""

    @pytest.fixture
    def data_repository(self, tmp_path):
        """Create data repository for testing."""
        # Create test data structure
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        # Create a test task file
        test_task = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
            ],
            "test": [
                {"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]}
            ]
        }

        import json
        with open(training_dir / "test_task.json", "w") as f:
            json.dump(test_task, f)

        return ARCDataRepository(data_path=str(tmp_path))

    def test_data_loading_for_ttt(self, data_repository):
        """Test loading ARC data for TTT training."""
        # Load tasks - use load_all_tasks with limit
        tasks = data_repository.load_all_tasks(dataset="training", limit=1)

        assert len(tasks) == 1
        task = tasks[0]

        # Verify task structure matches TTT expectations
        assert isinstance(task, ARCTask)
        assert len(task.train_examples) == 2
        assert task.test_input is not None

        # Test grid conversion utilities
        for example in task.train_examples:
            input_str = grid_to_string(example["input"])
            output_str = grid_to_string(example["output"])

            # Verify string representation
            assert isinstance(input_str, str)
            assert isinstance(output_str, str)

            # Verify round-trip conversion
            recovered_input = string_to_grid(input_str)
            recovered_output = string_to_grid(output_str)

            assert recovered_input == example["input"]
            assert recovered_output == example["output"]

    def test_ttt_adapter_with_real_data(self, data_repository, tmp_path):
        """Test TTT adapter with data from repository."""
        # Load a task
        tasks = data_repository.load_all_tasks(dataset="training", limit=1)
        task = tasks[0]

        # Create TTT adapter
        config = TTTConfig(
            model_name="gpt2",  # Small model for testing
            device="cpu",
            quantization=False,
            max_examples=2,
            num_epochs=1,
            checkpoint_dir=tmp_path / "checkpoints",
            cache_dir=tmp_path / "cache"
        )

        adapter = TTTAdapter(config=config)

        # Prepare training examples
        examples = adapter._prepare_training_examples(task)

        # Verify example preparation
        assert len(examples) <= config.max_examples

        for example in examples:
            assert "prompt" in example
            assert "input_grid" in example
            assert "output_grid" in example
            assert "Task: Transform the input grid" in example["prompt"]

    def test_batch_processing_memory_efficiency(self, data_repository):
        """Test memory-efficient batch processing for TTT."""
        from src.utils.grid_ops import GridBatcher

        # Load multiple tasks - use load_all_tasks with limit
        tasks = data_repository.load_all_tasks(dataset="training", limit=10)

        # Create batcher with memory limit
        batcher = GridBatcher(memory_limit_mb=100)  # Small limit for testing

        # Test different batching strategies
        adaptive_batches = batcher.adaptive_batch(tasks, base_batch_size=3)
        uniform_batches = batcher.uniform_batch(tasks, batch_size=3)
        memory_batches = batcher.memory_aware_batch(tasks)

        # Verify batching
        assert len(adaptive_batches) > 0
        assert len(uniform_batches) > 0
        assert len(memory_batches) > 0

        # Verify all tasks are included
        total_adaptive = sum(len(batch) for batch in adaptive_batches)
        total_uniform = sum(len(batch) for batch in uniform_batches)
        total_memory = sum(len(batch) for batch in memory_batches)

        assert total_adaptive == len(tasks)
        assert total_uniform == len(tasks)
        assert total_memory == len(tasks)

    def test_sparse_grid_optimization(self):
        """Test sparse matrix optimization for large grids."""
        from src.utils.grid_ops import SparseGridConverter

        # Create a sparse grid (mostly zeros)
        sparse_grid = [[0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [3, 0, 0, 0, 0]]

        converter = SparseGridConverter()

        # Convert to sparse
        sparse_matrix = converter.to_sparse(sparse_grid, format="csr")

        # Check efficiency
        efficiency = converter.estimate_sparse_efficiency(sparse_grid)
        assert efficiency["sparsity"] > 0.8  # Grid is >80% zeros
        assert efficiency["recommended"] is True  # Should recommend sparse

        # Convert back to dense
        recovered_grid = converter.to_dense(sparse_matrix)
        assert recovered_grid == sparse_grid

    def test_data_augmentation_compatibility(self):
        """Test data augmentation for TTT training diversity."""
        from src.utils.data_augmentation import GridAugmentor

        augmenter = GridAugmentor()

        # Create test grid
        original_grid = [[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]

        # Test augmentations
        rotated = augmenter.rotate_grid_90(original_grid)
        flipped_h = augmenter.flip_horizontal(original_grid)
        flipped_v = augmenter.flip_vertical(original_grid)

        # Verify augmentations produce different grids
        assert rotated != original_grid
        assert flipped_h != original_grid
        assert flipped_v != original_grid

        # Verify dimensions preserved
        assert len(rotated) == len(original_grid)
        assert len(flipped_h) == len(original_grid)
        assert len(flipped_v) == len(original_grid)

    def test_cache_integration(self, tmp_path):
        """Test caching integration for TTT checkpoints."""
        from src.adapters.repositories.cache_repository import CacheRepository

        cache_repo = CacheRepository(cache_dir=tmp_path)

        # Create test data
        checkpoint_data = {
            "model_state": {"test": "data"},
            "training_metrics": {"accuracy": 0.45},
            "task_id": "test_task_001"
        }

        # Cache checkpoint
        key = "ttt_checkpoint_test_001"
        cache_repo.set(key, checkpoint_data, expire=3600)

        # Retrieve checkpoint
        cached_data = cache_repo.get(key)
        assert cached_data == checkpoint_data

        # Test cache statistics
        stats = cache_repo.get_statistics()
        assert stats is not None
        assert stats["sets"] > 0
        assert stats["hits"] >= 0
