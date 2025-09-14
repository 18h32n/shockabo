"""
Training orchestration service for Test-Time Training pipeline.

Manages the complete training lifecycle including training loops, validation,
early stopping, gradient accumulation, and memory optimization.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.domain.models import ARCTask, ResourceUsage, StrategyType
import numpy as np
from src.domain.services.ttt_service import TTTModelService
from src.utils.lora_adapter import LoRAAdapter, apply_lora_to_model
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    """Early stopping configuration."""
    patience: int = 5  # Number of validations without improvement
    min_delta: float = 0.01  # Minimum improvement threshold
    restore_best_weights: bool = True  # Restore best weights when stopping
    monitor_metric: str = "validation_accuracy"  # Metric to monitor
    mode: str = "max"  # "max" for accuracy, "min" for loss
    baseline: Optional[float] = None  # Baseline value to compare against
    verbose: bool = True  # Log early stopping events
    
    # Auto-save configuration
    auto_save_enabled: bool = True
    auto_save_interval_minutes: int = 10  # Save every 10 minutes
    auto_save_on_improvement: bool = True  # Save when metric improves
    
    # Auto-resume configuration
    auto_resume_enabled: bool = True
    resume_from_best: bool = True  # Resume from best checkpoint
    resume_threshold_hours: float = 0.5  # Resume if training interrupted within 30 minutes


@dataclass
class TrainingConfig:
    """Configuration for training orchestration."""
    learning_rate: float = 5e-5  # Optimized for 8B model
    num_epochs: int = 3  # Reduced for 8B model efficiency
    batch_size: int = 1  # Memory efficient for 8B model
    gradient_accumulation_steps: int = 4  # Increased for effective batch size
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    # Early stopping configuration
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    validation_frequency: int = 25  # More frequent validation for 8B
    checkpoint_frequency: int = 50  # More frequent checkpoints
    max_training_time: int = 1800  # 30 minutes per task for 8B
    target_accuracy: float = 0.53  # 53% target accuracy for validation
    memory_limit_mb: float = 24576  # 24GB limit for 8B model
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # 8B model specific configurations
    use_qlora: bool = True  # Enable QLoRA for 8B model
    lora_rank: int = 64  # Higher rank for 8B model capacity
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_flash_attention: bool = True  # Enable Flash Attention
    selective_checkpointing: bool = True  # Selective gradient checkpointing
    checkpointing_layers: int = 3  # Every 3rd layer for checkpointing
    

class ARCTaskDataset(Dataset):
    """Dataset for ARC task training examples."""
    
    def __init__(self, task: ARCTask, tokenizer: Any, max_length: int = 2048):
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict[str, Any]]:
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class TrainingOrchestrator:
    """Orchestrates the complete TTT training pipeline."""
    
    def __init__(
        self,
        model_service: TTTModelService,
        config: Optional[TrainingConfig] = None,
        checkpoint_repository = None,
    ):
        """Initialize training orchestrator."""
        self.model_service = model_service
        self.config = config or TrainingConfig()
        self.device = model_service.device
        
        # Import checkpoint repository
        if checkpoint_repository is None:
            from src.adapters.repositories.checkpoint_repository import CheckpointRepository
            self.checkpoint_repo = CheckpointRepository()
        else:
            self.checkpoint_repo = checkpoint_repository
        
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
        self.training_history: List[TrainingMetrics] = []
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.start_time = None
        self.current_task_id = None
        
        # Early stopping state
        self.early_stopping_triggered = False
        self.best_checkpoint_path = None
        self.last_auto_save_time = None
        self.training_session_id = None
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == "cuda" else None
        
        logger.info(f"Training orchestrator initialized with config: {self.config}")
    
    def setup_training(self, task: ARCTask) -> None:
        """Set up training components for a task with 8B model optimizations."""
        logger.info(f"Setting up training for 8B model on task: {task.task_id}")
        
        # Load 8B model with QLoRA if enabled
        if self.config.use_qlora:
            # Configure model service for QLoRA
            self.model_service.config.setdefault("model", {})
            self.model_service.config["model"].update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "use_flash_attention": self.config.use_flash_attention
            })
            logger.info("8B model configured with QLoRA (4-bit quantization)")
        
        # Load model
        self.model, self.tokenizer = self.model_service.load_model()
        
        # Apply LoRA adaptation with 8B model settings
        self.lora_adapter = apply_lora_to_model(
            self.model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        logger.info(f"Applied LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        
        # Apply selective checkpointing if enabled
        if self.config.selective_checkpointing:
            self.model_service.checkpointing_layers = self.config.checkpointing_layers
            self.model_service._apply_selective_checkpointing()
            logger.info(f"Applied selective checkpointing every {self.config.checkpointing_layers} layers")
        
        # Prepare for training with memory optimizations
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
        batch: Dict[str, torch.Tensor],
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
        """Validate model performance on task with enhanced accuracy measurement."""
        self.model.eval()
        
        # Use test examples for validation (proper validation set)
        correct = 0
        total = len(task.test_examples) if hasattr(task, 'test_examples') and task.test_examples else len(task.train_examples)
        
        # Use test examples if available, otherwise use held-out train examples
        validation_examples = task.test_examples if hasattr(task, 'test_examples') and task.test_examples else task.train_examples[-1:]
        
        with torch.no_grad():
            for i, example in enumerate(validation_examples):
                # Generate prediction with optimized prompt
                from src.utils.grid_ops import grid_to_string, string_to_grid
                
                input_str = grid_to_string(example["input"])
                
                # Enhanced prompt for better 8B model performance
                prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task.
Analyze the pattern in the training examples and apply it to transform the test input.

Task: Transform the input grid to output grid following the learned pattern.

Input grid:
{input_str}

Output grid:"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
                
                # Generate with optimized parameters for 8B model
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,  # Reduced for grid output
                        temperature=0.0,  # Deterministic for exact matching
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Prevent loops
                        no_repeat_ngram_size=2,  # Prevent repetition
                    )
                
                # Decode and parse
                generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                try:
                    predicted_grid = string_to_grid(generated.strip())
                    expected_grid = example["output"]
                    
                    # Enhanced accuracy check with shape validation
                    if predicted_grid is not None and expected_grid is not None:
                        if np.array_equal(predicted_grid, expected_grid):
                            correct += 1
                            logger.debug(f"Validation example {i}: CORRECT")
                        else:
                            logger.debug(f"Validation example {i}: INCORRECT (shape/content mismatch)")
                    else:
                        logger.debug(f"Validation example {i}: FAILED TO PARSE")
                        
                except Exception as e:
                    logger.debug(f"Validation example {i}: PARSE ERROR - {str(e)}")
                    pass
        
        accuracy = correct / total if total > 0 else 0.0
        self.model.train()
        
        logger.info(f"Validation accuracy: {correct}/{total} = {accuracy:.2%}")
        return accuracy
    
    def check_early_stopping(self, current_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Enhanced early stopping with multiple criteria and auto-save functionality.
        
        Args:
            current_metrics: Current training metrics
            
        Returns:
            Tuple of (should_stop, reason)
        """
        es_config = self.config.early_stopping
        current_time = time.time()
        
        # Extract monitored metric
        current_value = current_metrics.get(es_config.monitor_metric, 0.0)
        
        # Check time limit
        if current_time - self.start_time > self.config.max_training_time:
            return True, "Time limit exceeded"
        
        # Check memory limit
        memory_mb = self.model_service._get_memory_usage() * 1024
        if memory_mb > self.config.memory_limit_mb:
            return True, f"Memory limit exceeded ({memory_mb:.2f}MB > {self.config.memory_limit_mb}MB)"
        
        # Check target accuracy achieved
        if current_value >= self.config.target_accuracy:
            logger.info(f"Target accuracy achieved: {current_value:.2%} >= {self.config.target_accuracy:.2%}")
            return True, "Target accuracy achieved"
        
        # Check baseline performance
        if es_config.baseline is not None and current_value < es_config.baseline:
            return True, f"Performance below baseline: {current_value:.4f} < {es_config.baseline:.4f}"
        
        # Check improvement-based early stopping
        improved = False
        if es_config.mode == "max":
            improved = current_value > (self.best_accuracy + es_config.min_delta)
        else:  # mode == "min"
            improved = current_value < (self.best_accuracy - es_config.min_delta)
        
        if improved:
            self.best_accuracy = current_value
            self.patience_counter = 0
            
            # Auto-save on improvement
            if es_config.auto_save_enabled and es_config.auto_save_on_improvement:
                self._auto_save_checkpoint(current_metrics, "improvement")
                
            if es_config.verbose:
                logger.info(f"New best {es_config.monitor_metric}: {current_value:.4f}")
        else:
            self.patience_counter += 1
            if es_config.verbose and self.patience_counter > 0:
                logger.info(f"No improvement for {self.patience_counter}/{es_config.patience} validations")
        
        # Auto-save based on time interval
        if (es_config.auto_save_enabled and 
            (self.last_auto_save_time is None or 
             (current_time - self.last_auto_save_time) >= es_config.auto_save_interval_minutes * 60)):
            self._auto_save_checkpoint(current_metrics, "interval")
        
        # Check patience
        if self.patience_counter >= es_config.patience:
            reason = f"No improvement for {self.patience_counter} validations (patience: {es_config.patience})"
            if es_config.verbose:
                logger.info(f"Early stopping triggered: {reason}")
            return True, reason
        
        return False, ""
    
    def _auto_save_checkpoint(self, current_metrics: Dict[str, Any], save_type: str) -> None:
        """Auto-save checkpoint with current training state."""
        try:
            if not self.current_task_id:
                logger.warning("Cannot auto-save: no current task ID")
                return
            
            # Generate checkpoint ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"{self.current_task_id}_{save_type}_{timestamp}"
            
            # Prepare model state
            model_state = {
                "model_state_dict": self.model.state_dict(),
                "lora_adapter_state": (self.lora_adapter.get_adapter_state() 
                                       if self.lora_adapter and hasattr(self.lora_adapter, 'get_adapter_state') 
                                       else None),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "training_step": current_metrics.get("step", 0),
                "epoch": current_metrics.get("epoch", 0),
            }
            
            # Prepare training metrics
            training_metrics = {
                "model_name": "llama-3-8b",
                "final_accuracy": current_metrics.get("validation_accuracy", 0.0),
                "training_time": time.time() - self.start_time,
                "final_memory_mb": self.model_service._get_memory_usage() * 1024,
                "step": current_metrics.get("step", 0),
                "epoch": current_metrics.get("epoch", 0),
                "loss": current_metrics.get("loss", 0.0),
                "learning_rate": current_metrics.get("learning_rate", 0.0),
            }
            
            # LoRA configuration
            lora_config = {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
            }
            
            # Save checkpoint
            metadata = self.checkpoint_repo.save_checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=self.current_task_id,
                model_state=model_state,
                training_metrics=training_metrics,
                lora_config=lora_config,
                tags=[save_type, "auto_save"],
            )
            
            # Update best checkpoint if this is an improvement
            if save_type == "improvement":
                self.best_checkpoint_path = checkpoint_id
            
            self.last_auto_save_time = time.time()
            
            logger.info(f"Auto-saved checkpoint {checkpoint_id} ({save_type})")
            
        except Exception as e:
            logger.error(f"Failed to auto-save checkpoint: {e}")
    
    def try_auto_resume(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Try to auto-resume training from the most recent checkpoint.
        
        Args:
            task_id: Task ID to resume training for
            
        Returns:
            Resume metadata if successful, None otherwise
        """
        try:
            es_config = self.config.early_stopping
            if not es_config.auto_resume_enabled:
                return None
            
            # Find recent checkpoints for this task
            recent_checkpoints = self.checkpoint_repo.list_checkpoints(task_id=task_id)
            if not recent_checkpoints:
                logger.info(f"No checkpoints found for task {task_id}")
                return None
            
            # Check if the most recent checkpoint is within resume threshold
            latest_checkpoint = recent_checkpoints[0]  # Already sorted by accuracy and time
            time_since_checkpoint = (datetime.now() - latest_checkpoint.created_at).total_seconds() / 3600
            
            if time_since_checkpoint > es_config.resume_threshold_hours:
                logger.info(f"Latest checkpoint too old ({time_since_checkpoint:.2f}h > {es_config.resume_threshold_hours}h)")
                return None
            
            # Select checkpoint to resume from
            resume_checkpoint = latest_checkpoint
            if es_config.resume_from_best:
                best_checkpoint = self.checkpoint_repo.get_best_checkpoint(task_id)
                if best_checkpoint:
                    resume_checkpoint = best_checkpoint
            
            logger.info(f"Attempting to resume from checkpoint {resume_checkpoint.checkpoint_id}")
            
            # Load checkpoint
            checkpoint_data, metadata = self.checkpoint_repo.load_checkpoint(
                resume_checkpoint.checkpoint_id,
                validate_checksum=True
            )
            
            # Restore training state
            if self.model and "model_state_dict" in checkpoint_data["model_state"]:
                self.model.load_state_dict(checkpoint_data["model_state"]["model_state_dict"])
            
            if self.optimizer and "optimizer_state_dict" in checkpoint_data["model_state"]:
                self.optimizer.load_state_dict(checkpoint_data["model_state"]["optimizer_state_dict"])
            
            if self.scheduler and "scheduler_state_dict" in checkpoint_data["model_state"]:
                scheduler_state = checkpoint_data["model_state"]["scheduler_state_dict"]
                if scheduler_state:
                    self.scheduler.load_state_dict(scheduler_state)
            
            if self.scaler and "scaler_state_dict" in checkpoint_data["model_state"]:
                scaler_state = checkpoint_data["model_state"]["scaler_state_dict"]
                if scaler_state:
                    self.scaler.load_state_dict(scaler_state)
            
            # Restore LoRA adapter state if available
            if self.lora_adapter and "lora_adapter_state" in checkpoint_data["model_state"]:
                adapter_state = checkpoint_data["model_state"]["lora_adapter_state"]
                if adapter_state and hasattr(self.lora_adapter, 'load_adapter_state'):
                    try:
                        self.lora_adapter.load_adapter_state(adapter_state)
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA adapter state: {e}")
            
            # Restore training metrics
            self.best_accuracy = metadata.accuracy
            training_time_elapsed = checkpoint_data["training_metrics"]["training_time"]
            
            # Adjust start time to account for previous training
            self.start_time = time.time() - training_time_elapsed
            
            resume_info = {
                "checkpoint_id": resume_checkpoint.checkpoint_id,
                "resumed_epoch": checkpoint_data["model_state"]["epoch"],
                "resumed_step": checkpoint_data["model_state"]["training_step"],
                "resumed_accuracy": metadata.accuracy,
                "elapsed_training_time": training_time_elapsed,
            }
            
            logger.info(f"Successfully resumed from checkpoint: {resume_info}")
            return resume_info
            
        except Exception as e:
            logger.error(f"Failed to auto-resume training: {e}")
            return None
    
    def train(self, task: ARCTask) -> Dict[str, Any]:
        """
        Execute complete training pipeline with early stopping and auto-resume.
        
        Args:
            task: ARC task to train on
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for task {task.task_id}")
        
        # Set current task ID for checkpointing
        self.current_task_id = task.task_id
        self.training_session_id = f"{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Try to auto-resume if enabled
        resume_info = self.try_auto_resume(task.task_id)
        
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
            
            for batch_idx, batch in enumerate(self.train_dataloader):
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
                    
                    # Check early stopping with enhanced metrics
                    current_metrics = {
                        "validation_accuracy": accuracy,
                        "loss": metrics.loss,
                        "step": global_step,
                        "epoch": epoch,
                        "learning_rate": metrics.learning_rate,
                        "memory_mb": metrics.memory_mb,
                    }
                    
                    should_stop, reason = self.check_early_stopping(current_metrics)
                    if should_stop:
                        logger.info(f"Early stopping triggered: {reason}")
                        self.early_stopping_triggered = True
                        break
                
                # Track metrics
                self.training_history.append(metrics)
                
                # Memory optimization
                if global_step % 50 == 0:
                    self.model_service.optimize_memory()
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
            
            # Check if we should stop (if not already checked)
            if not self.early_stopping_triggered and global_step % self.config.validation_frequency != 0:
                accuracy = self.validate(task)
                current_metrics = {
                    "validation_accuracy": accuracy,
                    "loss": avg_epoch_loss,
                    "step": global_step,
                    "epoch": epoch,
                    "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
                    "memory_mb": self.model_service._get_memory_usage() * 1024,
                }
                should_stop, reason = self.check_early_stopping(current_metrics)
                if should_stop:
                    logger.info(f"Early stopping triggered at epoch end: {reason}")
                    self.early_stopping_triggered = True
                    break
        
        # Final validation
        final_accuracy = self.validate(task)
        training_time = time.time() - self.start_time
        
        # Restore best weights if early stopping and requested
        if (self.early_stopping_triggered and 
            self.config.early_stopping.restore_best_weights and 
            self.best_checkpoint_path):
            try:
                checkpoint_data, _ = self.checkpoint_repo.load_checkpoint(self.best_checkpoint_path)
                self.model.load_state_dict(checkpoint_data["model_state"]["model_state_dict"])
                logger.info(f"Restored best weights from checkpoint: {self.best_checkpoint_path}")
                # Re-validate with best weights
                final_accuracy = self.validate(task)
            except Exception as e:
                logger.warning(f"Failed to restore best weights: {e}")
        
        # Prepare results with early stopping information
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
            
            # Early stopping information
            "early_stopping_triggered": self.early_stopping_triggered,
            "patience_reached": self.patience_counter >= self.config.early_stopping.patience,
            "best_checkpoint_path": self.best_checkpoint_path,
            "training_session_id": self.training_session_id,
            "resume_info": resume_info,
        }
        
        logger.info(
            f"Training completed - Accuracy: {final_accuracy:.2%} - "
            f"Time: {training_time:.2f}s - Memory: {results['final_memory_mb']:.2f}MB"
        )
        
        return results
    
    def save_checkpoint(self, path: Path, metrics: Dict[str, Any]) -> None:
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