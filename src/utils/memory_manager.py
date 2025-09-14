"""
GPU Memory Management utilities with OOM prevention.

Provides memory monitoring, automatic batch size adjustment, and OOM recovery.
"""
import gc
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager

import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages GPU/CPU memory with OOM prevention capabilities."""

    def __init__(
        self,
        device: torch.device,
        memory_limit_gb: float = 10.0,
        safety_margin: float = 0.9,  # Use only 90% of limit
        enable_monitoring: bool = True,
    ):
        """
        Initialize memory manager.
        
        Args:
            device: PyTorch device to manage
            memory_limit_gb: Maximum memory limit in GB
            safety_margin: Safety margin (0-1) to prevent OOM
            enable_monitoring: Whether to enable active monitoring
        """
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.safety_margin = safety_margin
        self.enable_monitoring = enable_monitoring
        self.safe_memory_gb = memory_limit_gb * safety_margin

        # Memory pressure thresholds
        self.warning_threshold = 0.7  # 70% usage
        self.critical_threshold = 0.85  # 85% usage

        # OOM prevention state
        self.oom_count = 0
        self.last_oom_time = None
        self.batch_size_history = []

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "usage_percentage": (allocated / self.memory_limit_gb) * 100,
                "available_gb": self.safe_memory_gb - allocated,
            }
        else:
            # CPU memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024**3

            return {
                "allocated_gb": process_memory,
                "reserved_gb": process_memory,
                "total_gb": memory.total / 1024**3,
                "usage_percentage": (process_memory / self.memory_limit_gb) * 100,
                "available_gb": self.safe_memory_gb - process_memory,
            }

    def check_memory_pressure(self) -> str:
        """Check current memory pressure level."""
        usage = self.get_memory_usage()
        usage_ratio = usage["allocated_gb"] / self.memory_limit_gb

        if usage_ratio >= self.critical_threshold:
            return "critical"
        elif usage_ratio >= self.warning_threshold:
            return "warning"
        else:
            return "normal"

    def suggest_batch_size(self, current_batch_size: int, memory_per_sample_mb: float) -> int:
        """Suggest optimal batch size based on available memory."""
        usage = self.get_memory_usage()
        available_mb = usage["available_gb"] * 1024

        # Conservative estimate: use only 80% of available memory
        safe_available_mb = available_mb * 0.8

        # Calculate maximum batch size
        max_batch_size = max(1, int(safe_available_mb / memory_per_sample_mb))

        # Apply reduction based on memory pressure
        pressure = self.check_memory_pressure()
        if pressure == "critical":
            suggested_size = max(1, min(max_batch_size, current_batch_size // 2))
        elif pressure == "warning":
            suggested_size = max(1, min(max_batch_size, int(current_batch_size * 0.75)))
        else:
            suggested_size = max_batch_size

        logger.info(f"Memory pressure: {pressure}, suggesting batch size: {suggested_size}")
        return suggested_size

    def clear_cache(self) -> None:
        """Clear GPU/CPU cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

        # Force garbage collection
        gc.collect()

        logger.info("Memory cache cleared")

    @contextmanager
    def oom_protected(self, fallback_fn: Callable | None = None):
        """
        Context manager for OOM-protected operations.
        
        Args:
            fallback_fn: Function to call if OOM occurs
        """
        try:
            yield
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                self.oom_count += 1
                self.last_oom_time = time.time()

                logger.error(f"OOM error detected (count: {self.oom_count}): {e}")

                # Clear cache immediately
                self.clear_cache()

                # Call fallback function if provided
                if fallback_fn:
                    logger.info("Executing OOM fallback function")
                    fallback_fn()

                # Re-raise if too many OOMs
                if self.oom_count > 3:
                    raise RuntimeError("Too many OOM errors, aborting") from e
            else:
                raise

    def monitor_memory_loop(self, callback: Callable | None = None, interval: float = 5.0):
        """
        Start memory monitoring loop.
        
        Args:
            callback: Function to call on memory pressure changes
            interval: Monitoring interval in seconds
        """
        if not self.enable_monitoring:
            return

        previous_pressure = "normal"

        while self.enable_monitoring:
            try:
                current_pressure = self.check_memory_pressure()

                if current_pressure != previous_pressure:
                    logger.warning(f"Memory pressure changed: {previous_pressure} -> {current_pressure}")

                    if callback:
                        callback(current_pressure, self.get_memory_usage())

                previous_pressure = current_pressure
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)

    def get_memory_summary(self) -> str:
        """Get formatted memory usage summary."""
        usage = self.get_memory_usage()
        pressure = self.check_memory_pressure()

        summary = f"""
Memory Usage Summary:
- Device: {self.device}
- Allocated: {usage['allocated_gb']:.2f} GB
- Available: {usage['available_gb']:.2f} GB
- Usage: {usage['usage_percentage']:.1f}%
- Pressure: {pressure}
- OOM Count: {self.oom_count}
"""
        return summary.strip()


class AdaptiveBatchSizer:
    """Automatically adjusts batch size based on memory constraints."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
    ):
        """Initialize adaptive batch sizer."""
        self.memory_manager = memory_manager
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Tracking successful runs
        self.successful_sizes = set()
        self.failed_sizes = set()

    def adjust_batch_size(self, success: bool = True) -> int:
        """
        Adjust batch size based on success/failure.
        
        Args:
            success: Whether the last batch was processed successfully
            
        Returns:
            New batch size
        """
        if success:
            self.successful_sizes.add(self.current_batch_size)

            # Try to increase if memory allows
            pressure = self.memory_manager.check_memory_pressure()
            if pressure == "normal" and self.current_batch_size < self.max_batch_size:
                # Conservative increase
                new_size = min(int(self.current_batch_size * 1.2), self.max_batch_size)
                if new_size not in self.failed_sizes:
                    self.current_batch_size = new_size
                    logger.info(f"Increasing batch size to {self.current_batch_size}")
        else:
            self.failed_sizes.add(self.current_batch_size)

            # Reduce batch size
            new_size = max(self.current_batch_size // 2, self.min_batch_size)
            self.current_batch_size = new_size
            logger.warning(f"Reducing batch size to {self.current_batch_size} due to failure")

        return self.current_batch_size

    def get_optimal_batch_size(self) -> int:
        """Get the largest known successful batch size."""
        if self.successful_sizes:
            return max(self.successful_sizes)
        return self.current_batch_size


def profile_memory_usage(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function."""
    def wrapper(*args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initial memory
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            start_memory = psutil.Process().memory_info().rss / 1024**2

        # Execute function
        result = func(*args, **kwargs)

        # Final memory
        if device.type == "cuda":
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            end_memory = psutil.Process().memory_info().rss / 1024**2

        memory_used = end_memory - start_memory
        logger.info(f"{func.__name__} memory usage: {memory_used:.2f} MB")

        return result

    return wrapper
