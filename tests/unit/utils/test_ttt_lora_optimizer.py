"""
Unit tests for LoRA Optimizer
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.utils.ttt_lora_optimizer import (
    AdaptationMetrics,
    EarlyStoppingState,
    LoRAOptimizer,
    LoRAOptimizerConfig,
)


@pytest.fixture
def optimizer():
    """Create LoRA optimizer instance."""
    return LoRAOptimizer()


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return nn.Linear(10, 10)


@pytest.fixture
def torch_optimizer(simple_model):
    """Create PyTorch optimizer for testing."""
    return AdamW(simple_model.parameters(), lr=5e-5)


class TestLoRAOptimizer:
    """Test suite for LoRAOptimizer."""

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.config.rank == 64
        assert optimizer.config.alpha == 32
        assert optimizer.config.early_stopping_patience == 5
        assert isinstance(optimizer.early_stopping_state, EarlyStoppingState)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = LoRAOptimizerConfig(
            rank=128, alpha=64, early_stopping_patience=3, learning_rate=1e-4
        )
        optimizer = LoRAOptimizer(config)

        assert optimizer.config.rank == 128
        assert optimizer.config.alpha == 64
        assert optimizer.config.early_stopping_patience == 3
        assert optimizer.config.learning_rate == 1e-4

    def test_early_stopping_improvement(self, optimizer):
        """Test early stopping with improving loss."""
        assert not optimizer.should_stop_early(1.0)
        assert not optimizer.should_stop_early(0.8)
        assert not optimizer.should_stop_early(0.6)

        assert optimizer.early_stopping_state.best_loss == 0.6
        assert optimizer.early_stopping_state.patience_counter == 0
        assert not optimizer.early_stopping_state.should_stop

    def test_early_stopping_no_improvement(self, optimizer):
        """Test early stopping with no improvement."""
        optimizer.should_stop_early(1.0)

        for _ in range(optimizer.config.early_stopping_patience - 1):
            assert not optimizer.should_stop_early(1.1)

        assert optimizer.should_stop_early(1.1)
        assert optimizer.early_stopping_state.should_stop

    def test_early_stopping_with_validation_loss(self, optimizer):
        """Test early stopping prefers validation loss over train loss."""
        should_stop = optimizer.should_stop_early(train_loss=0.5, validation_loss=1.0)

        assert not should_stop
        assert optimizer.early_stopping_state.best_loss == 1.0

        should_stop = optimizer.should_stop_early(train_loss=0.3, validation_loss=0.8)
        assert not should_stop
        assert optimizer.early_stopping_state.best_loss == 0.8

    def test_early_stopping_min_delta(self, optimizer):
        """Test min_delta threshold in early stopping."""
        optimizer.config.min_delta = 0.1

        optimizer.should_stop_early(1.0)
        optimizer.should_stop_early(0.95)

        assert optimizer.early_stopping_state.patience_counter == 1

        optimizer.should_stop_early(0.85)
        assert optimizer.early_stopping_state.patience_counter == 0

    def test_gradient_clipping(self, optimizer, simple_model, torch_optimizer):
        """Test gradient norm clipping."""
        loss = simple_model(torch.randn(2, 10)).sum()
        loss.backward()

        grad_norm = optimizer.clip_gradients(simple_model, torch_optimizer)

        assert grad_norm > 0.0
        assert all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in simple_model.parameters()
        )

    def test_gradient_clipping_disabled(self, simple_model, torch_optimizer):
        """Test gradient clipping can be disabled."""
        config = LoRAOptimizerConfig(enable_gradient_clipping=False)
        optimizer = LoRAOptimizer(config)

        loss = simple_model(torch.randn(2, 10)).sum()
        loss.backward()

        grad_norm = optimizer.clip_gradients(simple_model, torch_optimizer)

        assert grad_norm == 0.0

    def test_warmup_cosine_scheduler(self, optimizer, torch_optimizer):
        """Test warmup + cosine learning rate scheduler."""
        num_training_steps = 100
        scheduler = optimizer.create_warmup_cosine_scheduler(
            torch_optimizer, num_training_steps
        )

        initial_lr = scheduler.get_last_lr()[0]

        for _ in range(10):
            torch_optimizer.step()
            scheduler.step()

        warmup_lr = scheduler.get_last_lr()[0]
        assert warmup_lr > initial_lr
        
        for _ in range(30):
            torch_optimizer.step()
            scheduler.step()

        mid_lr = scheduler.get_last_lr()[0]
        assert mid_lr >= 0.0

    def test_track_adaptation_step(self, optimizer):
        """Test adaptation metrics tracking."""
        optimizer.track_adaptation_step(
            epoch=0, train_loss=1.0, validation_loss=1.2, learning_rate=5e-5, gradient_norm=2.5
        )

        assert len(optimizer.early_stopping_state.metrics_history) == 1

        metrics = optimizer.early_stopping_state.metrics_history[0]
        assert metrics.epoch == 0
        assert metrics.train_loss == 1.0
        assert metrics.validation_loss == 1.2
        assert metrics.learning_rate == 5e-5
        assert metrics.gradient_norm == 2.5
        assert metrics.timestamp > 0

    def test_get_optimal_epoch_count_no_metrics(self, optimizer):
        """Test optimal epoch count with no metrics."""
        optimal_epochs = optimizer.get_optimal_epoch_count()
        assert optimal_epochs == optimizer.config.target_epochs

    def test_get_optimal_epoch_count_early_stopped(self, optimizer):
        """Test optimal epoch count after early stopping."""
        for i in range(5):
            optimizer.track_adaptation_step(
                epoch=i, train_loss=1.0 - (i * 0.1), validation_loss=None, learning_rate=5e-5, gradient_norm=1.0
            )

        optimizer.early_stopping_state.should_stop = True
        optimizer.early_stopping_state.best_epoch = 2

        optimal_epochs = optimizer.get_optimal_epoch_count()
        assert optimal_epochs == 3

    def test_tune_rank_alpha_disabled(self, optimizer):
        """Test rank/alpha tuning when disabled."""
        task_complexity = {
            "num_train_examples": 5,
            "grid_size_avg": 200,
            "unique_colors": 7,
        }

        rank, alpha = optimizer.tune_rank_alpha(task_complexity)

        assert rank == optimizer.config.rank
        assert alpha == optimizer.config.alpha

    def test_tune_rank_alpha_low_complexity(self):
        """Test rank/alpha tuning for low complexity tasks."""
        config = LoRAOptimizerConfig(enable_adaptive_rank=True)
        optimizer = LoRAOptimizer(config)

        task_complexity = {"num_train_examples": 2, "grid_size_avg": 50, "unique_colors": 3}

        rank, alpha = optimizer.tune_rank_alpha(task_complexity)

        assert rank == 32
        assert alpha == 16

    def test_tune_rank_alpha_medium_complexity(self):
        """Test rank/alpha tuning for medium complexity tasks."""
        config = LoRAOptimizerConfig(enable_adaptive_rank=True)
        optimizer = LoRAOptimizer(config)

        task_complexity = {"num_train_examples": 4, "grid_size_avg": 150, "unique_colors": 5}

        rank, alpha = optimizer.tune_rank_alpha(task_complexity)

        assert rank == 64
        assert alpha == 32

    def test_tune_rank_alpha_high_complexity(self):
        """Test rank/alpha tuning for high complexity tasks."""
        config = LoRAOptimizerConfig(enable_adaptive_rank=True)
        optimizer = LoRAOptimizer(config)

        task_complexity = {"num_train_examples": 8, "grid_size_avg": 500, "unique_colors": 9}

        rank, alpha = optimizer.tune_rank_alpha(task_complexity)

        assert rank == 128
        assert alpha == 64

    def test_validate_checkpoint_integrity_missing(self, optimizer):
        """Test checkpoint validation with missing file."""
        checkpoint_path = Path("nonexistent_checkpoint.pt")
        is_valid = optimizer.validate_checkpoint_integrity(checkpoint_path)
        assert not is_valid

    def test_validate_checkpoint_integrity_valid(self, optimizer):
        """Test checkpoint validation with valid checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = Path(tmp.name)

        checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "epoch": 5,
        }
        torch.save(checkpoint, checkpoint_path)

        is_valid = optimizer.validate_checkpoint_integrity(checkpoint_path)
        assert is_valid

        checkpoint_path.unlink()

    def test_validate_checkpoint_integrity_missing_keys(self, optimizer):
        """Test checkpoint validation with missing required keys."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = Path(tmp.name)

        checkpoint = {"model_state_dict": {}}
        torch.save(checkpoint, checkpoint_path)

        is_valid = optimizer.validate_checkpoint_integrity(checkpoint_path)
        assert not is_valid

        checkpoint_path.unlink()

    def test_validate_checkpoint_integrity_corrupted(self, optimizer):
        """Test checkpoint validation with corrupted file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = Path(tmp.name)
            tmp.write(b"corrupted data")

        is_valid = optimizer.validate_checkpoint_integrity(checkpoint_path)
        assert not is_valid

        checkpoint_path.unlink()

    def test_profile_lora_configuration(self, optimizer):
        """Test LoRA configuration profiling."""
        metrics = {"accuracy": 0.85, "training_time": 300.0, "memory_usage": 8192}

        optimizer.profile_lora_configuration("config_1", metrics)

        assert "config_1" in optimizer.profiling_results
        assert optimizer.profiling_results["config_1"]["metrics"] == metrics
        assert optimizer.profiling_results["config_1"]["config"]["rank"] == 64

    def test_get_profiling_summary_empty(self, optimizer):
        """Test profiling summary with no data."""
        summary = optimizer.get_profiling_summary()
        assert "error" in summary

    def test_get_profiling_summary(self, optimizer):
        """Test profiling summary with multiple configurations."""
        optimizer.profile_lora_configuration(
            "config_1", {"accuracy": 0.80, "training_time": 400.0}
        )
        optimizer.profile_lora_configuration(
            "config_2", {"accuracy": 0.85, "training_time": 350.0}
        )
        optimizer.profile_lora_configuration(
            "config_3", {"accuracy": 0.82, "training_time": 300.0}
        )

        summary = optimizer.get_profiling_summary()

        assert "best_configuration" in summary
        assert "all_results" in summary
        assert "recommendation" in summary
        assert summary["best_configuration"] in ["config_2", "config_3"]

    def test_reset(self, optimizer):
        """Test optimizer reset."""
        optimizer.track_adaptation_step(
            epoch=0, train_loss=1.0, validation_loss=None, learning_rate=5e-5, gradient_norm=1.0
        )
        optimizer.should_stop_early(1.0)

        assert len(optimizer.early_stopping_state.metrics_history) > 0
        assert optimizer.early_stopping_state.best_loss < float("inf")

        optimizer.reset()

        assert len(optimizer.early_stopping_state.metrics_history) == 0
        assert optimizer.early_stopping_state.best_loss == float("inf")
        assert optimizer.early_stopping_state.patience_counter == 0
        assert not optimizer.early_stopping_state.should_stop
