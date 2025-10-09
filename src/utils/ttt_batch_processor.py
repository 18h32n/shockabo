"""
Memory Efficient Batch Processing for TTT

Implements gradient accumulation, gradient checkpointing, dynamic batch sizing,
and memory monitoring for efficient training within memory constraints (24GB).
"""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Memory limits
    memory_limit_mb: float = 24576  # 24GB for 8B model
    memory_warning_threshold: float = 0.85  # Warn at 85% usage
    memory_critical_threshold: float = 0.95  # Critical at 95% usage

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    checkpointing_layers: int | None = None  # Checkpoint every N layers (None = all)

    # Dynamic batch sizing
    enable_dynamic_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 4
    batch_size_adjustment_step: int = 1

    # Memory monitoring
    enable_memory_monitoring: bool = True
    monitoring_interval_steps: int = 10


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    current_mb: float
    peak_mb: float
    allocated_mb: float
    reserved_mb: float
    utilization: float  # 0.0-1.0
    timestamp: float


class MemoryMonitor:
    """Monitor and track memory usage during training."""

    def __init__(self, config: MemoryConfig):
        """
        Initialize memory monitor.

        Args:
            config: Memory configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"

        self.metrics_history: list[MemoryMetrics] = []
        self.last_warning_time = 0.0

        logger.info(f"Initialized memory monitor (device: {self.device}, limit: {config.memory_limit_mb}MB)")

    def get_current_memory(self) -> MemoryMetrics:
        """
        Get current memory usage metrics.

        Returns:
            MemoryMetrics with current usage
        """
        if self.is_cuda:
            current_mb = torch.cuda.memory_allocated() / 1024**2
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
        else:
            # Use psutil for CPU memory
            process = psutil.Process()
            mem_info = process.memory_info()
            current_mb = mem_info.rss / 1024**2
            peak_mb = current_mb  # No peak tracking for CPU
            allocated_mb = current_mb
            reserved_mb = current_mb

        utilization = current_mb / self.config.memory_limit_mb if self.config.memory_limit_mb > 0 else 0.0

        return MemoryMetrics(
            current_mb=current_mb,
            peak_mb=peak_mb,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            utilization=utilization,
            timestamp=time.time()
        )

    def check_memory_status(self) -> tuple[bool, str]:
        """
        Check if memory usage is within safe limits.

        Returns:
            Tuple of (is_safe, status_message)
        """
        metrics = self.get_current_memory()

        if metrics.utilization >= self.config.memory_critical_threshold:
            return False, f"CRITICAL: Memory usage {metrics.current_mb:.0f}MB ({metrics.utilization:.1%})"

        if metrics.utilization >= self.config.memory_warning_threshold:
            # Rate-limit warnings to avoid spam
            current_time = time.time()
            if current_time - self.last_warning_time > 10.0:
                logger.warning(
                    f"HIGH memory usage: {metrics.current_mb:.0f}MB ({metrics.utilization:.1%})"
                )
                self.last_warning_time = current_time
            return True, f"WARNING: Memory usage {metrics.current_mb:.0f}MB ({metrics.utilization:.1%})"

        return True, f"OK: Memory usage {metrics.current_mb:.0f}MB ({metrics.utilization:.1%})"

    def record_metrics(self):
        """Record current metrics to history."""
        metrics = self.get_current_memory()
        self.metrics_history.append(metrics)

    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self) -> float:
        """
        Get peak memory usage in MB.

        Returns:
            Peak memory in MB
        """
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / 1024**2
        elif self.metrics_history:
            return max(m.current_mb for m in self.metrics_history)
        return 0.0

    def get_average_memory(self) -> float:
        """
        Get average memory usage in MB.

        Returns:
            Average memory in MB
        """
        if not self.metrics_history:
            return 0.0
        return sum(m.current_mb for m in self.metrics_history) / len(self.metrics_history)


class MemoryEfficientBatchProcessor:
    """
    Memory efficient batch processor with gradient accumulation and checkpointing.

    Implements:
    - Gradient accumulation to simulate larger batches
    - Selective gradient checkpointing to trade compute for memory
    - Dynamic batch sizing based on available memory
    - Memory monitoring with automatic scaling
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize batch processor.

        Args:
            config: Memory configuration
        """
        self.config = config
        self.monitor = MemoryMonitor(config)

        self.current_batch_size = config.min_batch_size
        self.accumulated_steps = 0

        logger.info(
            f"Initialized batch processor (gradient_accumulation: {config.gradient_accumulation_steps}, "
            f"checkpointing: {config.gradient_checkpointing})"
        )

    def enable_gradient_checkpointing(
        self,
        model: nn.Module,
        checkpoint_layers: int | None = None
    ):
        """
        Enable gradient checkpointing for model.

        Args:
            model: PyTorch model
            checkpoint_layers: Checkpoint every N layers (None = all layers)
        """
        if not self.config.gradient_checkpointing:
            return

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing via model method")
        else:
            # Manual checkpointing for custom models
            logger.warning("Model does not support gradient_checkpointing_enable()")

        if checkpoint_layers is not None:
            logger.info(f"Checkpointing every {checkpoint_layers} layers")

    def should_accumulate_gradients(self) -> bool:
        """
        Check if gradients should be accumulated (not stepped yet).

        Returns:
            True if should accumulate, False if should step optimizer
        """
        return (self.accumulated_steps + 1) % self.config.gradient_accumulation_steps != 0

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None
    ):
        """
        Step optimizer after gradient accumulation.

        Args:
            optimizer: PyTorch optimizer
            scaler: Optional gradient scaler for mixed precision
        """
        self.accumulated_steps += 1

        if not self.should_accumulate_gradients():
            # Time to step optimizer
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Monitor memory after step
            if self.config.enable_memory_monitoring:
                self.monitor.record_metrics()

    def adjust_batch_size(self, increase: bool) -> int:
        """
        Adjust batch size dynamically based on memory usage.

        Args:
            increase: True to increase batch size, False to decrease

        Returns:
            New batch size
        """
        if not self.config.enable_dynamic_batch_size:
            return self.current_batch_size

        if increase:
            new_size = min(
                self.current_batch_size + self.config.batch_size_adjustment_step,
                self.config.max_batch_size
            )
        else:
            new_size = max(
                self.current_batch_size - self.config.batch_size_adjustment_step,
                self.config.min_batch_size
            )

        if new_size != self.current_batch_size:
            logger.info(f"Adjusted batch size: {self.current_batch_size} -> {new_size}")
            self.current_batch_size = new_size

        return new_size

    def check_and_adjust_memory(self) -> bool:
        """
        Check memory status and adjust batch size if needed.

        Returns:
            True if memory is safe, False if critical
        """
        is_safe, status = self.monitor.check_memory_status()

        if not is_safe:
            # Critical memory - reduce batch size
            logger.error(f"Critical memory! {status}")
            self.adjust_batch_size(increase=False)
            return False

        metrics = self.monitor.get_current_memory()

        if metrics.utilization < 0.7:
            # Low memory usage - could increase batch size
            self.adjust_batch_size(increase=True)
        elif metrics.utilization > 0.85:
            # High memory usage - reduce batch size
            self.adjust_batch_size(increase=False)

        return True

    def process_batch(
        self,
        model: nn.Module,
        batch_data: Any,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None,
        use_mixed_precision: bool = False
    ) -> dict[str, float]:
        """
        Process a single batch with gradient accumulation and checkpointing.

        Args:
            model: PyTorch model
            batch_data: Input batch data
            loss_fn: Loss function callable
            optimizer: PyTorch optimizer
            scaler: Optional gradient scaler for mixed precision
            use_mixed_precision: Whether to use mixed precision training

        Returns:
            Dictionary with loss and memory metrics
        """
        # Check memory before processing
        if not self.check_and_adjust_memory():
            logger.error("Memory critical - skipping batch")
            return {"loss": 0.0, "memory_mb": 0.0, "skipped": True}

        # Forward pass with optional mixed precision
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch_data)
                loss = loss_fn(outputs)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
        else:
            outputs = model(batch_data)
            loss = loss_fn(outputs)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        # Step optimizer if accumulation complete
        self.step_optimizer(optimizer, scaler)

        # Record memory metrics
        metrics = self.monitor.get_current_memory()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "memory_mb": metrics.current_mb,
            "utilization": metrics.utilization,
            "batch_size": self.current_batch_size
        }

    def get_memory_report(self) -> dict[str, float]:
        """
        Get comprehensive memory usage report.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "peak_memory_mb": self.monitor.get_peak_memory(),
            "avg_memory_mb": self.monitor.get_average_memory(),
            "current_memory_mb": self.monitor.get_current_memory().current_mb,
            "memory_limit_mb": self.config.memory_limit_mb,
            "final_batch_size": self.current_batch_size
        }

    def clear_memory(self):
        """Clear cached memory and reset statistics."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("Cleared memory cache")
