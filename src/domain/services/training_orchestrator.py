"""
Training orchestration service for Test-Time Training pipeline.

Manages the complete training lifecycle including training loops, validation,
early stopping, gradient accumulation, and memory optimization.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.domain.models import ARCTask
from src.domain.services.ttt_service import TTTModelService
from src.utils.lora_adapter import apply_lora_to_model
from src.utils.memory_manager import AdaptiveBatchSizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    validation_accuracy: float = 0.0
    memory_mb: float = 0.0
    time_elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "validation_accuracy": self.validation_accuracy,
            "memory_mb": self.memory_mb,
            "time_elapsed": self.time_elapsed,
        }


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping mechanism."""
    patience: int = 5
    min_delta: float = 0.01
    monitor_metric: str = "validation_accuracy"
    mode: str = "max"  # "max" for accuracy, "min" for loss
    restore_best_weights: bool = True
    baseline: float | None = None
    verbose: bool = True
    auto_save_enabled: bool = True
    auto_save_interval_minutes: int = 10
    auto_save_on_improvement: bool = True
    auto_resume_enabled: bool = True
    resume_from_best: bool = True
    resume_threshold_hours: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training orchestration."""
    learning_rate: float = 1e-4
    num_epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    validation_frequency: int = 50  # Validate every N steps
    checkpoint_frequency: int = 100  # Save checkpoint every N steps
    max_training_time: int = 7200  # 2 hours in seconds
    target_accuracy: float = 0.95  # 95% target accuracy
    memory_limit_mb: float = 10240  # 10GB limit
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    early_stopping_config: EarlyStoppingConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None  # Alias for backward compatibility

    def __post_init__(self):
        """Handle backward compatibility for early_stopping parameter."""
        if self.early_stopping is not None and self.early_stopping_config is None:
            self.early_stopping_config = self.early_stopping


class ARCTaskDataset(Dataset):
    """Dataset for ARC task training examples."""

    def __init__(self, task: ARCTask, tokenizer: Any, max_length: int = 2048):
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples()

    def _prepare_examples(self) -> list[dict[str, Any]]:
        """Prepare training examples from task."""
        from src.utils.grid_ops import grid_to_string

        examples = []
        for train_example in self.task.train_examples:
            input_str = grid_to_string(train_example["input"])
            output_str = grid_to_string(train_example["output"])

            # Create prompt-completion pair
            prompt = f"Task: Transform the input grid to output grid.\n\nInput:\n{input_str}\n\nOutput:"
            completion = f" {output_str}"

            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

            # Combine tokens
            input_ids = prompt_tokens + completion_tokens

            # Create labels (mask prompt part)
            labels = [-100] * len(prompt_tokens) + completion_tokens

            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            # Pad
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

            examples.append({
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor([1] * (self.max_length - padding_length) + [0] * padding_length),
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class TrainingOrchestrator:
    """Orchestrates the complete TTT training pipeline."""

    def __init__(
        self,
        model_service: TTTModelService,
        config: TrainingConfig | None = None,
        checkpoint_repository: Any | None = None,  # For backward compatibility
    ):
        """Initialize training orchestrator."""
        self.model_service = model_service
        self.config = config or TrainingConfig()
        self.device = model_service.device
        self.checkpoint_repository = checkpoint_repository

        # Initialize adaptive batch sizer
        self.batch_sizer = AdaptiveBatchSizer(
            memory_manager=model_service.memory_manager,
            initial_batch_size=self.config.batch_size,
            min_batch_size=1,
            max_batch_size=self.config.batch_size * 4
        )

        # Training state
        self.model = None
        self.tokenizer = None
        self.lora_adapter = None
        self.optimizer = None
        self.scheduler = None

        # Metrics tracking
        self.training_history: list[TrainingMetrics] = []
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.start_time = None

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == "cuda" else None

        logger.info(f"Training orchestrator initialized with config: {self.config}")

    def setup_training(self, task: ARCTask) -> None:
        """Set up training components for a task."""
        logger.info(f"Setting up training for task: {task.task_id}")

        # Load model
        self.model, self.tokenizer = self.model_service.load_model()

        # Apply LoRA adaptation
        self.lora_adapter = apply_lora_to_model(
            self.model,
            rank=8,
            alpha=16,
            dropout=0.1,
        )

        # Prepare for training
        self.model_service.prepare_for_training()

        # Create dataset and dataloader
        dataset = ARCTaskDataset(task, self.tokenizer)
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Calculate total training steps
        total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps

        # Set up optimizer (only LoRA parameters)
        self.optimizer = AdamW(
            self.lora_adapter.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Set up learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Initialize tracking
        self.start_time = time.time()
        self.training_history.clear()
        self.best_accuracy = 0.0
        self.patience_counter = 0

        logger.info(f"Training setup complete. Total steps: {total_steps}")

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        step: int,
        epoch: int,
    ) -> TrainingMetrics:
        """Execute single training step with OOM protection."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Define OOM recovery function
        def handle_oom():
            logger.warning("OOM during training step - reducing batch size")
            self.batch_sizer.adjust_batch_size(success=False)
            # Clear gradients
            self.optimizer.zero_grad()

        # Execute with OOM protection
        with self.model_service.memory_manager.oom_protected(fallback_fn=handle_oom):
            # Mixed precision context
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient accumulation
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.lora_adapter.get_trainable_parameters(),
                self.config.max_grad_norm,
            )

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        # Create metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss.item() * self.config.gradient_accumulation_steps,
            learning_rate=self.scheduler.get_last_lr()[0],
            memory_mb=self.model_service._get_memory_usage() * 1024,
            time_elapsed=time.time() - self.start_time,
        )

        return metrics

    def validate(self, task: ARCTask) -> float:
        """Validate model performance on task."""
        self.model.eval()

        # Use test examples for validation
        correct = 0
        total = len(task.train_examples)  # Use train examples as validation

        with torch.no_grad():
            for example in task.train_examples:
                # Generate prediction
                from src.utils.grid_ops import grid_to_string, string_to_grid

                input_str = grid_to_string(example["input"])
                prompt = f"Task: Transform the input grid to output grid.\n\nInput:\n{input_str}\n\nOutput:"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate with constrained decoding
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode and parse
                generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                try:
                    predicted_grid = string_to_grid(generated.strip())
                    expected_grid = example["output"]

                    # Check if prediction matches
                    if predicted_grid == expected_grid:
                        correct += 1
                except Exception:
                    # Failed to parse prediction
                    pass

        accuracy = correct / total if total > 0 else 0.0
        self.model.train()

        return accuracy

    def should_stop_early(self, current_accuracy: float) -> bool:
        """Check if training should stop early."""
        # Check time limit
        if time.time() - self.start_time > self.config.max_training_time:
            logger.info("Stopping early: Time limit exceeded")
            return True

        # Check memory limit
        memory_mb = self.model_service._get_memory_usage() * 1024
        if memory_mb > self.config.memory_limit_mb:
            logger.warning(f"Stopping early: Memory limit exceeded ({memory_mb:.2f}MB > {self.config.memory_limit_mb}MB)")
            return True

        # Check target accuracy achieved
        if current_accuracy >= self.config.target_accuracy:
            logger.info(f"Target accuracy achieved: {current_accuracy:.2%} >= {self.config.target_accuracy:.2%}")
            return True

        # Check early stopping patience
        if current_accuracy > self.best_accuracy + self.config.early_stopping_threshold:
            self.best_accuracy = current_accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Stopping early: No improvement for {self.patience_counter} validations")
                return True

        return False

    def check_early_stopping(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        """
        Check if training should stop early based on provided metrics.

        Args:
            metrics: Dictionary containing training metrics

        Returns:
            Tuple of (should_stop, reason)
        """
        # Initialize start_time if not set (for testing)
        if self.start_time is None:
            self.start_time = time.time()

        # Use the validation accuracy from metrics
        current_accuracy = metrics.get("validation_accuracy", 0.0)

        # Check early stopping conditions individually for clearer logic

        # Check time limit
        if hasattr(self, 'start_time') and self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.config.max_training_time:
                return True, "Time limit exceeded"

        # Check memory limit
        if hasattr(self.model_service, '_get_memory_usage'):
            memory_mb = self.model_service._get_memory_usage() * 1024
            if memory_mb > self.config.memory_limit_mb:
                return True, "Memory limit exceeded"

        # Check target accuracy achieved
        if current_accuracy >= self.config.target_accuracy:
            return True, "Target accuracy achieved"

        # Check early stopping patience and handle checkpoint saving
        if current_accuracy > self.best_accuracy + self.config.early_stopping_threshold:
            self.best_accuracy = current_accuracy
            self.patience_counter = 0

            # Trigger auto-save on improvement if enabled
            if (self.config.early_stopping_config and
                self.config.early_stopping_config.auto_save_on_improvement and
                self.checkpoint_repository):
                self._save_checkpoint_on_improvement(metrics)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                return True, "Patience exceeded"

        return False, ""

    def _save_checkpoint_on_improvement(self, metrics: dict[str, Any]) -> None:
        """Save checkpoint when improvement is detected."""
        try:
            if hasattr(self, 'current_task_id') and self.current_task_id:
                # Prepare model state
                model_state = self.model.state_dict() if self.model else {}

                # Prepare training metrics
                training_metrics = {
                    "step": metrics.get("step", 0),
                    "epoch": metrics.get("epoch", 0),
                    "validation_accuracy": metrics.get("validation_accuracy", 0.0),
                    "loss": metrics.get("loss", 0.0),
                    "learning_rate": metrics.get("learning_rate", 0.0),
                    "memory_mb": metrics.get("memory_mb", 0.0),
                }

                # Prepare LoRA config
                lora_config = {}
                if hasattr(self.lora_adapter, 'get_adapter_state'):
                    lora_config = self.lora_adapter.get_adapter_state()
                elif self.lora_adapter:
                    lora_config = {"adapter": "present"}

                # Generate checkpoint ID
                import uuid
                checkpoint_id = f"improvement_{uuid.uuid4().hex[:8]}"

                # Save through checkpoint repository
                self.checkpoint_repository.save_checkpoint(
                    checkpoint_id=checkpoint_id,
                    task_id=self.current_task_id,
                    model_state=model_state,
                    training_metrics=training_metrics,
                    lora_config=lora_config,
                    tags=["auto_save", "improvement"]
                )
                logger.info(f"Saved checkpoint {checkpoint_id} on improvement for task {self.current_task_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint on improvement: {str(e)}")

    def train(self, task: ARCTask) -> dict[str, Any]:
        """
        Execute complete training pipeline.

        Args:
            task: ARC task to train on

        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for task {task.task_id}")

        # Setup training
        self.setup_training(task)

        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            # Check memory before epoch
            memory_status = self.model_service.memory_manager.get_memory_usage()
            logger.info(f"Epoch {epoch} - Memory usage: {memory_status['usage_percentage']:.1f}%")

            for _batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                metrics = self.train_step(batch, global_step, epoch)
                epoch_loss += metrics.loss
                epoch_steps += 1
                global_step += 1

                # Log progress
                if global_step % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs} - "
                        f"Step {global_step} - Loss: {metrics.loss:.4f} - "
                        f"LR: {metrics.learning_rate:.2e} - "
                        f"Memory: {metrics.memory_mb:.2f}MB"
                    )

                # Validation
                if global_step % self.config.validation_frequency == 0:
                    accuracy = self.validate(task)
                    metrics.validation_accuracy = accuracy
                    logger.info(f"Validation accuracy: {accuracy:.2%}")

                    # Check early stopping
                    if self.should_stop_early(accuracy):
                        logger.info("Early stopping triggered")
                        break

                # Track metrics
                self.training_history.append(metrics)

                # Memory optimization
                if global_step % 50 == 0:
                    self.model_service.optimize_memory()

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")

            # Check if we should stop
            if global_step % self.config.validation_frequency != 0:
                accuracy = self.validate(task)
                if self.should_stop_early(accuracy):
                    break

        # Final validation
        final_accuracy = self.validate(task)
        training_time = time.time() - self.start_time

        # Prepare results
        results = {
            "task_id": task.task_id,
            "final_accuracy": final_accuracy,
            "best_accuracy": self.best_accuracy,
            "training_time": training_time,
            "total_steps": global_step,
            "epochs_completed": epoch + 1,
            "final_memory_mb": self.model_service._get_memory_usage() * 1024,
            "target_achieved": final_accuracy >= self.config.target_accuracy,
            "training_history": [m.to_dict() for m in self.training_history],
        }

        logger.info(
            f"Training completed - Accuracy: {final_accuracy:.2%} - "
            f"Time: {training_time:.2f}s - Memory: {results['final_memory_mb']:.2f}MB"
        )

        return results

    def save_checkpoint(self, path: Path, metrics: dict[str, Any]) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "lora_adapter": self.lora_adapter.save_adapter(path / "lora_adapter.pt"),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(checkpoint, path / "checkpoint.pt")
        logger.info(f"Saved checkpoint to {path}")

    def cleanup(self) -> None:
        """Clean up training resources."""
        self.model = None
        self.tokenizer = None
        self.lora_adapter = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None

        self.model_service.cleanup()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
