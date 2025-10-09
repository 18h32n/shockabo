"""
LoRA Optimization for TTT Strategy

This module implements LoRA adapter optimization with early stopping, learning rate scheduling,
gradient norm clipping, and adaptive rank/alpha tuning for efficient Test-Time Training.

Expected improvement: 20-30% faster training with same accuracy.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class LoRAOptimizerConfig:
    """Configuration for LoRA optimization."""

    rank: int = 64
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    target_epochs: int = 3
    max_epochs: int = 10
    enable_adaptive_rank: bool = False
    enable_gradient_clipping: bool = True


@dataclass
class AdaptationMetrics:
    """Metrics tracked during LoRA adaptation."""

    epoch: int
    train_loss: float
    validation_loss: float | None = None
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    timestamp: float = 0.0


@dataclass
class EarlyStoppingState:
    """State for early stopping logic."""

    best_loss: float = float("inf")
    patience_counter: int = 0
    best_epoch: int = 0
    should_stop: bool = False
    metrics_history: list[AdaptationMetrics] = field(default_factory=list)


class LoRAOptimizer:
    """
    Optimize LoRA adaptation for Test-Time Training.

    Features:
    - Early stopping based on validation loss convergence
    - Adaptive learning rate scheduling (warmup + cosine decay)
    - Gradient norm clipping for stability
    - Adaptive LoRA rank/alpha tuning based on task complexity
    - Performance profiling for different configurations
    - Checkpoint integrity validation
    """

    def __init__(self, config: LoRAOptimizerConfig | None = None):
        """
        Initialize LoRA optimizer.

        Args:
            config: LoRA optimization configuration
        """
        self.config = config or LoRAOptimizerConfig()
        self.early_stopping_state = EarlyStoppingState()
        self.profiling_results: dict[str, Any] = {}

    def should_stop_early(
        self, train_loss: float, validation_loss: float | None = None
    ) -> bool:
        """
        Check if training should stop early based on loss convergence.

        Args:
            train_loss: Current training loss
            validation_loss: Optional validation loss (preferred for early stopping)

        Returns:
            True if training should stop, False otherwise
        """
        loss_to_monitor = validation_loss if validation_loss is not None else train_loss

        if loss_to_monitor < self.early_stopping_state.best_loss - self.config.min_delta:
            self.early_stopping_state.best_loss = loss_to_monitor
            self.early_stopping_state.best_epoch = len(
                self.early_stopping_state.metrics_history
            )
            self.early_stopping_state.patience_counter = 0
            logger.debug(
                f"Loss improved to {loss_to_monitor:.6f}, resetting patience counter"
            )
            return False
        else:
            self.early_stopping_state.patience_counter += 1
            logger.debug(
                f"No improvement in loss, patience: "
                f"{self.early_stopping_state.patience_counter}/{self.config.early_stopping_patience}"
            )

            if self.early_stopping_state.patience_counter >= self.config.early_stopping_patience:
                self.early_stopping_state.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {len(self.early_stopping_state.metrics_history)} "
                    f"(best epoch: {self.early_stopping_state.best_epoch})"
                )
                return True

        return False

    def clip_gradients(
        self, model: torch.nn.Module, optimizer: Optimizer
    ) -> float:
        """
        Clip gradients by global norm for training stability.

        Args:
            model: PyTorch model with parameters to clip
            optimizer: Optimizer being used

        Returns:
            Gradient norm before clipping
        """
        if not self.config.enable_gradient_clipping:
            return 0.0

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.max_grad_norm
        )

        return float(grad_norm)

    def create_warmup_cosine_scheduler(
        self, optimizer: Optimizer, num_training_steps: int
    ) -> _LRScheduler:
        """
        Create learning rate scheduler with warmup and cosine decay.

        Args:
            optimizer: PyTorch optimizer
            num_training_steps: Total number of training steps

        Returns:
            Learning rate scheduler
        """
        from torch.optim.lr_scheduler import LambdaLR

        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress))))

        return LambdaLR(optimizer, lr_lambda)

    def track_adaptation_step(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float | None,
        learning_rate: float,
        gradient_norm: float,
    ) -> None:
        """
        Track metrics for a single adaptation step.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            validation_loss: Validation loss (optional)
            learning_rate: Current learning rate
            gradient_norm: Gradient norm before clipping
        """
        import time

        metrics = AdaptationMetrics(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            timestamp=time.time(),
        )

        self.early_stopping_state.metrics_history.append(metrics)

        val_loss_str = f"{validation_loss:.6f}" if validation_loss is not None else "N/A"
        logger.debug(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss_str}, "
            f"lr={learning_rate:.2e}, grad_norm={gradient_norm:.4f}"
        )

    def get_optimal_epoch_count(self) -> int:
        """
        Determine optimal epoch count based on tracked metrics.

        Returns:
            Recommended number of epochs (2-3 for speed optimization)
        """
        if not self.early_stopping_state.metrics_history:
            return self.config.target_epochs

        if self.early_stopping_state.should_stop:
            optimal_epochs = self.early_stopping_state.best_epoch + 1
            logger.info(
                f"Optimal epoch count determined: {optimal_epochs} "
                f"(early stopping at best epoch)"
            )
            return optimal_epochs

        return min(
            len(self.early_stopping_state.metrics_history), self.config.target_epochs
        )

    def tune_rank_alpha(self, task_complexity: dict[str, Any]) -> tuple[int, int]:
        """
        Tune LoRA rank and alpha based on task complexity.

        Args:
            task_complexity: Dictionary with task complexity metrics
                - num_train_examples: Number of training examples
                - grid_size_avg: Average grid size
                - unique_colors: Number of unique colors

        Returns:
            Tuple of (optimal_rank, optimal_alpha)
        """
        if not self.config.enable_adaptive_rank:
            return self.config.rank, self.config.alpha

        num_examples = task_complexity.get("num_train_examples", 3)
        grid_size = task_complexity.get("grid_size_avg", 100)
        unique_colors = task_complexity.get("unique_colors", 5)

        complexity_score = (num_examples * 10) + (grid_size / 10) + (unique_colors * 5)

        if complexity_score < 50:
            rank, alpha = 32, 16
        elif complexity_score < 100:
            rank, alpha = 64, 32
        else:
            rank, alpha = 128, 64

        logger.info(
            f"Adaptive rank/alpha tuning: complexity={complexity_score:.1f}, "
            f"rank={rank}, alpha={alpha}"
        )

        return rank, alpha

    def validate_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """
        Validate checkpoint file integrity with error handling for corrupted checkpoints.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"Missing required key in checkpoint: {key}")
                    return False

            logger.debug(f"Checkpoint validation successful: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Checkpoint corrupted or unreadable: {checkpoint_path}, error: {e}")
            return False

    def profile_lora_configuration(
        self, config_name: str, metrics: dict[str, float]
    ) -> None:
        """
        Profile performance for a specific LoRA configuration.

        Args:
            config_name: Name of the configuration
            metrics: Performance metrics (e.g., training_time, accuracy, memory_usage)
        """
        self.profiling_results[config_name] = {
            "metrics": metrics,
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "learning_rate": self.config.learning_rate,
                "warmup_ratio": self.config.warmup_ratio,
            },
        }

        logger.info(f"Profiled configuration '{config_name}': {metrics}")

    def get_profiling_summary(self) -> dict[str, Any]:
        """
        Get summary of all profiled configurations.

        Returns:
            Dictionary with profiling results and recommendations
        """
        if not self.profiling_results:
            return {"error": "No profiling data available"}

        best_config = None
        best_score = -float("inf")

        for config_name, data in self.profiling_results.items():
            metrics = data["metrics"]
            score = metrics.get("accuracy", 0.0) - (
                metrics.get("training_time", 1000.0) / 1000.0
            )

            if score > best_score:
                best_score = score
                best_config = config_name

        return {
            "best_configuration": best_config,
            "all_results": self.profiling_results,
            "recommendation": f"Use configuration '{best_config}' for optimal performance",
        }

    def reset(self) -> None:
        """Reset optimizer state for new adaptation run."""
        self.early_stopping_state = EarlyStoppingState()
        logger.debug("Reset LoRA optimizer state")
