"""
Memory-Efficient TTT Training Pipeline

This module implements memory optimization strategies for MIT TTT training,
ensuring operations stay within the 10GB memory limit while maintaining performance.
"""
import gc
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_memory_mb: float
    current_memory_mb: float
    available_memory_mb: float
    memory_limit_mb: float
    memory_utilization: float
    timestamp: float


class MemoryMonitor:
    """Monitor and manage memory usage during TTT training."""

    def __init__(self, memory_limit_mb: float = 10240):
        """
        Initialize memory monitor.

        Args:
            memory_limit_mb: Memory limit in MB (default 10GB)
        """
        self.memory_limit_mb = memory_limit_mb
        self.peak_memory_mb = 0.0
        self.memory_history: list[MemoryStats] = []

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                return 0.0

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return self.peak_memory_mb

    def get_available_memory(self) -> float:
        """Get available memory in MB."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            used_memory = self.get_current_memory()
            return min(total_memory - used_memory, self.memory_limit_mb - used_memory)
        return self.memory_limit_mb - self.get_current_memory()

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        current_memory = self.get_current_memory()
        peak_memory = self.get_peak_memory()
        available_memory = self.get_available_memory()

        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)

        stats = MemoryStats(
            peak_memory_mb=peak_memory,
            current_memory_mb=current_memory,
            available_memory_mb=available_memory,
            memory_limit_mb=self.memory_limit_mb,
            memory_utilization=current_memory / self.memory_limit_mb,
            timestamp=time.time()
        )

        self.memory_history.append(stats)
        return stats

    def check_memory_threshold(self, threshold: float = 0.85) -> bool:
        """
        Check if memory usage exceeds threshold.

        Args:
            threshold: Memory threshold (0.0 to 1.0)

        Returns:
            True if memory usage exceeds threshold
        """
        stats = self.get_memory_stats()
        return stats.memory_utilization > threshold

    def clear_memory_cache(self) -> None:
        """Clear memory cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def memory_context(self, operation_name: str = "operation") -> Generator[None, None, None]:
        """
        Context manager for memory-aware operations.

        Args:
            operation_name: Name of the operation for logging
        """
        start_stats = self.get_memory_stats()
        logger.debug(f"Starting {operation_name} - Memory: {start_stats.current_memory_mb:.1f}MB")

        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            memory_delta = end_stats.current_memory_mb - start_stats.current_memory_mb

            logger.debug(
                f"Finished {operation_name} - Memory: {end_stats.current_memory_mb:.1f}MB "
                f"(Δ{memory_delta:+.1f}MB)"
            )

            # Clear cache if memory usage is high
            if end_stats.memory_utilization > 0.8:
                logger.info(f"High memory usage ({end_stats.memory_utilization:.1%}), clearing cache")
                self.clear_memory_cache()


class TTTDataset(Dataset):
    """Memory-efficient dataset for TTT training."""

    def __init__(
        self,
        prompts: list[str],
        tokenizer,
        max_length: int = 2048,
        cache_tokenized: bool = False
    ):
        """
        Initialize TTT dataset.

        Args:
            prompts: List of training prompts
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            cache_tokenized: Whether to cache tokenized data
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenized = cache_tokenized
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache_tokenized and idx in self._cache:
            return self._cache[idx]

        prompt = self.prompts[idx]

        # Tokenize on-demand to save memory
        encoded = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze batch dimension
        batch = {k: v.squeeze(0) for k, v in encoded.items()}
        batch["labels"] = batch["input_ids"].clone()

        if self.cache_tokenized:
            self._cache[idx] = batch

        return batch

    def clear_cache(self) -> None:
        """Clear tokenization cache."""
        self._cache.clear()


class MemoryEfficientTTTTrainer:
    """Memory-efficient trainer for MIT TTT methodology."""

    def __init__(
        self,
        model: nn.Module,
        adapter,  # LoRAAdapter
        memory_limit_mb: float = 10240,
        gradient_checkpointing: bool = True,
        mixed_precision: bool = True
    ):
        """
        Initialize memory-efficient TTT trainer.

        Args:
            model: Base model
            adapter: LoRA adapter
            memory_limit_mb: Memory limit in MB
            gradient_checkpointing: Enable gradient checkpointing
            mixed_precision: Enable mixed precision training
        """
        self.model = model
        self.adapter = adapter
        self.memory_monitor = MemoryMonitor(memory_limit_mb)
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        logger.info(f"Initialized memory-efficient TTT trainer with {memory_limit_mb}MB limit")

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1
    ) -> dict[str, float]:
        """
        Perform a single training step with memory optimization.

        Args:
            batch: Training batch
            optimizer: Optimizer instance
            accumulation_steps: Gradient accumulation steps

        Returns:
            Dictionary with training metrics
        """
        with self.memory_monitor.memory_context("training_step"):
            # Move batch to device efficiently
            device = next(self.model.parameters()).device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward pass with optional mixed precision
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Update weights if accumulation is complete
                if accumulation_steps == 1:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        max_norm=1.0
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                outputs = self.model(**batch)
                loss = outputs.loss / accumulation_steps

                loss.backward()

                if accumulation_steps == 1:
                    torch.nn.utils.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            # Clear intermediate tensors
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "loss": loss.item() * accumulation_steps,
                "memory_mb": self.memory_monitor.get_current_memory()
            }

    def train_epoch(
        self,
        dataset: TTTDataset,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 1,
        accumulation_steps: int = 1,
        max_steps: int | None = None
    ) -> dict[str, float]:
        """
        Train for one epoch with memory optimization.

        Args:
            dataset: Training dataset
            optimizer: Optimizer instance
            batch_size: Batch size (keep small for memory efficiency)
            accumulation_steps: Gradient accumulation steps
            max_steps: Maximum number of steps

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()

        # Use memory-efficient data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic for TTT
            pin_memory=torch.cuda.is_available(),
            num_workers=0  # Single process to avoid memory overhead
        )

        total_loss = 0.0
        num_steps = 0
        max_memory = 0.0

        with self.memory_monitor.memory_context("train_epoch"):
            for step, batch in enumerate(dataloader):
                # Check memory before each step
                if self.memory_monitor.check_memory_threshold(0.9):
                    logger.warning("Memory threshold exceeded, clearing cache")
                    self.memory_monitor.clear_memory_cache()

                # Perform training step
                step_metrics = self.train_step(batch, optimizer, accumulation_steps)

                total_loss += step_metrics["loss"]
                max_memory = max(max_memory, step_metrics["memory_mb"])
                num_steps += 1

                # Apply gradient updates if using accumulation
                if accumulation_steps > 1 and (step + 1) % accumulation_steps == 0:
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            max_norm=1.0
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            max_norm=1.0
                        )
                        optimizer.step()

                    optimizer.zero_grad()

                # Early stopping if max steps reached
                if max_steps and num_steps >= max_steps:
                    break

        # Clear dataset cache to free memory
        dataset.clear_cache()

        return {
            "avg_loss": total_loss / max(num_steps, 1),
            "total_loss": total_loss,
            "num_steps": num_steps,
            "max_memory_mb": max_memory,
            "memory_utilization": max_memory / self.memory_monitor.memory_limit_mb
        }

    def train_per_instance(
        self,
        prompts: list[str],
        tokenizer,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        batch_size: int = 1,
        max_length: int = 2048,
        max_time_seconds: float = 300.0
    ) -> dict[str, Any]:
        """
        Perform per-instance TTT training with memory optimization.

        Args:
            prompts: Training prompts for this instance
            tokenizer: Tokenizer instance
            optimizer: Optimizer instance
            num_epochs: Number of training epochs
            batch_size: Batch size
            max_length: Maximum sequence length
            max_time_seconds: Maximum training time

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        # Create memory-efficient dataset
        dataset = TTTDataset(
            prompts=prompts,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_tokenized=len(prompts) <= 10  # Cache only for small datasets
        )

        # Training metrics
        epoch_metrics = []
        total_loss = 0.0
        total_steps = 0

        with self.memory_monitor.memory_context("per_instance_training"):
            for epoch in range(num_epochs):
                # Check time constraint
                if time.time() - start_time > max_time_seconds:
                    logger.warning(f"Training timeout after {time.time() - start_time:.1f}s")
                    break

                # Train epoch
                metrics = self.train_epoch(
                    dataset=dataset,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    accumulation_steps=1,
                    max_steps=None
                )

                epoch_metrics.append(metrics)
                total_loss += metrics["total_loss"]
                total_steps += metrics["num_steps"]

                logger.debug(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Loss={metrics['avg_loss']:.4f}, "
                    f"Memory={metrics['max_memory_mb']:.1f}MB"
                )

        training_time = time.time() - start_time
        final_memory_stats = self.memory_monitor.get_memory_stats()

        return {
            "avg_loss": total_loss / max(total_steps, 1),
            "total_loss": total_loss,
            "num_steps": total_steps,
            "num_epochs": len(epoch_metrics),
            "training_time": training_time,
            "epoch_metrics": epoch_metrics,
            "final_memory_stats": final_memory_stats,
            "memory_efficiency": {
                "peak_memory_mb": final_memory_stats.peak_memory_mb,
                "memory_utilization": final_memory_stats.memory_utilization,
                "within_limit": final_memory_stats.peak_memory_mb <= self.memory_monitor.memory_limit_mb
            }
        }

    def cleanup(self) -> None:
        """Clean up trainer resources."""
        # Clear memory monitor history
        self.memory_monitor.memory_history.clear()

        # Clear GPU cache
        self.memory_monitor.clear_memory_cache()

        logger.info("TTT trainer cleanup complete")


@contextmanager
def memory_efficient_context(
    memory_limit_mb: float = 10240,
    cleanup_threshold: float = 0.8
) -> Generator[MemoryMonitor, None, None]:
    """
    Context manager for memory-efficient operations.

    Args:
        memory_limit_mb: Memory limit in MB
        cleanup_threshold: Threshold for automatic cleanup

    Yields:
        MemoryMonitor instance
    """
    monitor = MemoryMonitor(memory_limit_mb)

    try:
        yield monitor
    finally:
        # Cleanup if memory usage is high
        if monitor.get_memory_stats().memory_utilization > cleanup_threshold:
            monitor.clear_memory_cache()


def optimize_model_for_memory(model: nn.Module, enable_checkpointing: bool = True) -> None:
    """
    Apply memory optimizations to model.

    Args:
        model: Model to optimize
        enable_checkpointing: Enable gradient checkpointing
    """
    # Enable gradient checkpointing
    if enable_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    # Set model to use memory-efficient attention if available
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True
        logger.info("Enabled memory-efficient attention")

    # Disable unnecessary caching
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
        logger.info("Disabled model caching for memory efficiency")
