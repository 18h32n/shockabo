"""Unit tests for TTT adapter."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask, StrategyType


@pytest.fixture
def sample_task():
    """Create a sample ARC task for testing."""
    return ARCTask(
        task_id="test_task_001",
        task_source="test",
        train_examples=[
            {
                "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            },
            {
                "input": [[2, 2, 0], [2, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 0, 2], [0, 2, 2]]
            }
        ],
        test_input=[[3, 3, 0], [3, 0, 0], [0, 0, 0]]
    )


@pytest.fixture
def ttt_config():
    """Create test TTT configuration."""
    return TTTConfig(
        model_name="test-model",
        max_examples=2,
        learning_rate=1e-4,
        num_epochs=1,
        batch_size=2,
        device="cpu",
        quantization=False,
        mixed_precision=False,
        lora_rank=32,
        lora_alpha=16,
        lora_dropout=0.1,
        checkpoint_dir=Path("test_data/models/ttt"),
        cache_dir=Path("test_data/cache/ttt"),
        config_path=None
    )


class TestTTTAdapter:
    """Test suite for TTT adapter."""

    def test_init(self, ttt_config):
        """Test TTT adapter initialization."""
        adapter = TTTAdapter(config=ttt_config)

        assert adapter.config == ttt_config
        assert adapter.model is None
        assert adapter.tokenizer is None
        assert adapter.device.type == "cpu"
        assert len(adapter.adaptations) == 0

    def test_setup_device_auto(self):
        """Test automatic device setup."""
        config = TTTConfig(device="auto")
        adapter = TTTAdapter(config=config)

        # Should select CPU if no GPU available
        assert adapter.device.type in ["cpu", "cuda", "mps"]

    @patch("src.adapters.strategies.ttt_adapter.MIT_TTTStrategy")
    def test_initialize_model(self, mock_strategy_class, ttt_config):
        """Test model initialization."""
        # Setup mock strategy
        mock_strategy = MagicMock()
        mock_strategy_class.return_value = mock_strategy

        # Initialize adapter (strategy is created during __init__)
        adapter = TTTAdapter(config=ttt_config)

        # Verify strategy was initialized during construction
        mock_strategy_class.assert_called_once()
        assert adapter.mit_ttt_strategy == mock_strategy

        # Test that initialize_model doesn't break
        adapter.initialize_model()

    def test_prepare_training_examples(self, ttt_config, sample_task):
        """Test preparation of training examples."""
        adapter = TTTAdapter(config=ttt_config)
        examples = adapter._prepare_training_examples(sample_task)

        assert len(examples) == 2  # Based on max_examples

        for i, example in enumerate(examples):
            assert "index" in example
            assert "prompt" in example
            assert "input_grid" in example
            assert "output_grid" in example
            assert example["index"] == i
            assert "Task: Transform the input grid" in example["prompt"]

    @patch("src.adapters.strategies.ttt_adapter.MIT_TTTStrategy")
    def test_adapt_to_task(self, mock_strategy_class, ttt_config, sample_task):
        """Test task adaptation."""
        # Setup mock strategy
        mock_strategy = MagicMock()
        mock_adaptation_result = MagicMock()
        mock_adaptation_result.adaptation_id = "test_adaptation_001"
        mock_adaptation_result.adapter_path = "/path/to/adapter"
        mock_adaptation_result.training_metrics = {
            "avg_loss": 0.5,
            "num_steps": 10
        }
        mock_adaptation_result.memory_usage = {"peak_mb": 1024}

        mock_metadata = {
            "success": True,
            "confidence": 0.8,
            "adaptation_result": mock_adaptation_result
        }

        mock_strategy.solve_task.return_value = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], mock_metadata)
        mock_strategy_class.return_value = mock_strategy

        adapter = TTTAdapter(config=ttt_config)
        # Ensure the mock strategy is used
        adapter.mit_ttt_strategy = mock_strategy

        adaptation = adapter.adapt_to_task(sample_task)

        assert adaptation.task_id == sample_task.task_id
        assert adaptation.adaptation_id.startswith(f"mit_ttt_{sample_task.task_id}_")
        assert len(adaptation.training_examples) == 2
        assert "mit_ttt_training_loss" in adaptation.adaptation_metrics
        assert sample_task.task_id in adapter.adaptations

    @patch("psutil.Process")
    def test_estimate_memory_usage(self, mock_process, ttt_config):
        """Test memory usage estimation."""
        # Setup mock
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info

        adapter = TTTAdapter(config=ttt_config)
        memory_mb = adapter._estimate_memory_usage()

        assert memory_mb == 1024.0  # 1GB in MB

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.is_available")
    def test_estimate_gpu_memory(self, mock_cuda_available, mock_cuda_memory, ttt_config):
        """Test GPU memory estimation."""
        mock_cuda_available.return_value = True
        mock_cuda_memory.return_value = 512 * 1024 * 1024  # 512MB in bytes

        ttt_config.device = "cuda"
        adapter = TTTAdapter(config=ttt_config)
        adapter.device = torch.device("cuda")

        gpu_memory_mb = adapter._estimate_gpu_memory()

        assert gpu_memory_mb == 512.0

    def test_estimate_gpu_memory_cpu(self, ttt_config):
        """Test GPU memory estimation on CPU returns 0."""
        adapter = TTTAdapter(config=ttt_config)
        gpu_memory_mb = adapter._estimate_gpu_memory()

        assert gpu_memory_mb == 0.0

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    def test_cleanup(self, mock_cuda_available, mock_empty_cache, ttt_config):
        """Test cleanup functionality."""
        mock_cuda_available.return_value = True

        adapter = TTTAdapter(config=ttt_config)

        # Setup some state
        adapter.mit_ttt_strategy = MagicMock()
        adapter.adaptations = {"task1": MagicMock()}

        # Cleanup
        adapter.cleanup()

        # Verify cleanup
        adapter.mit_ttt_strategy.cleanup.assert_called_once()
        assert len(adapter.adaptations) == 0
        mock_empty_cache.assert_called_once()

    def test_generate_prediction(self, ttt_config):
        """Test prediction generation (now fallback method)."""
        adapter = TTTAdapter(config=ttt_config)

        input_grid = [[1, 2], [3, 4]]
        adaptation = MagicMock()

        prediction = adapter._generate_prediction(input_grid, adaptation)

        # Check that fallback returns input grid
        assert prediction == input_grid

    @patch("src.adapters.strategies.ttt_adapter.MIT_TTTStrategy")
    def test_solve(self, mock_strategy_class, ttt_config, sample_task):
        """Test solving ARC task."""
        # Setup mock strategy
        mock_strategy = MagicMock()
        mock_prediction = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]
        mock_metadata = {
            "success": True,
            "confidence": 0.75,
            "total_tokens": 100,
            "total_time": 5.0,
            "permutations": 1,
            "augmentations": ["basic"]
        }

        mock_strategy.solve_task.return_value = (mock_prediction, mock_metadata)
        mock_strategy_class.return_value = mock_strategy

        adapter = TTTAdapter(config=ttt_config)
        # Ensure the mock strategy is used
        adapter.mit_ttt_strategy = mock_strategy

        solution = adapter.solve(sample_task)

        assert solution.task_id == sample_task.task_id
        assert solution.predictions == [mock_prediction]
        assert solution.strategy_used == StrategyType.TEST_TIME_TRAINING
        assert solution.confidence_score == 0.75
        assert solution.metadata["success"] is True
        assert solution.metadata["mit_ttt_strategy"] is True
        assert solution.resource_usage is not None
