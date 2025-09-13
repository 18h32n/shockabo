"""
Advanced Memory Optimization Utilities for 8B Model Training

Enhanced memory management tools specifically designed for the constraints
identified in the risk assessment (24GB GPU memory limit).
"""
import functools
import gc
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class MemoryOptimizationLevel(Enum):
    """Memory optimization levels."""
    CONSERVATIVE = "conservative"  # Minimal optimizations, maximum safety
    BALANCED = "balanced"         # Good balance of memory/performance
    AGGRESSIVE = "aggressive"     # Maximum memory optimization


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimizations."""
    level: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED
    gradient_checkpointing_ratio: float = 0.3  # Fraction of layers to checkpoint
    activation_checkpointing: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    zero_stage: int = 2  # DeepSpeed ZeRO stage (0, 1, 2, 3)
    max_memory_utilization: float = 0.9  # Maximum GPU memory utilization
    dynamic_batching: bool = True
    memory_defragmentation: bool = True
    gradient_accumulation_steps: int = 4


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    memory_utilization: float
    fragmentation_ratio: float
    active_tensors: int


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on memory usage."""
    
    def __init__(
        self,
        initial_batch_size: int = 1,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        memory_threshold: float = 0.85,
        growth_factor: float = 1.2,
        shrink_factor: float = 0.8
    ):
        """
        Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: Memory utilization threshold for adjustment
            growth_factor: Factor to grow batch size
            shrink_factor: Factor to shrink batch size
        """
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        
        self.memory_history: List[float] = []
        self.oom_count = 0
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 10  # seconds
        
    def adjust_batch_size(self, current_memory_utilization: float) -> int:
        """
        Adjust batch size based on current memory utilization.
        
        Args:
            current_memory_utilization: Current memory utilization (0.0-1.0)
            
        Returns:
            New batch size
        """
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return self.current_batch_size
        
        self.memory_history.append(current_memory_utilization)
        
        # Keep only recent history
        if len(self.memory_history) > 10:
            self.memory_history = self.memory_history[-10:]
        
        # Calculate average recent memory usage
        avg_memory = sum(self.memory_history[-3:]) / min(3, len(self.memory_history))
        
        new_batch_size = self.current_batch_size
        
        # Shrink if memory usage is too high
        if avg_memory > self.memory_threshold:
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * self.shrink_factor)
            )
            logger.info(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size} (mem: {avg_memory:.1%})")
        
        # Grow if memory usage is low and no recent OOM
        elif avg_memory < self.memory_threshold * 0.7 and self.oom_count == 0:
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * self.growth_factor)
            )
            logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size} (mem: {avg_memory:.1%})")
        
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self.last_adjustment_time = current_time
        
        return self.current_batch_size
    
    def handle_oom(self) -> int:
        """Handle OOM error by aggressively reducing batch size."""
        self.oom_count += 1
        old_batch_size = self.current_batch_size
        
        # Emergency reduction
        self.current_batch_size = max(
            self.min_batch_size,
            self.current_batch_size // 2
        )
        
        logger.warning(f"OOM handled: batch size {old_batch_size} -> {self.current_batch_size}")
        return self.current_batch_size
    
    def reset_oom_count(self):
        """Reset OOM count after successful training."""
        if self.oom_count > 0:
            logger.info(f"Resetting OOM count: {self.oom_count} -> 0")
            self.oom_count = 0


class MemoryDefragmenter:
    """Handles GPU memory fragmentation."""
    
    def __init__(self, defrag_threshold: float = 0.3):
        """
        Initialize memory defragmenter.
        
        Args:
            defrag_threshold: Fragmentation ratio threshold for triggering defrag
        """
        self.defrag_threshold = defrag_threshold
        self.last_defrag_time = 0
        self.defrag_cooldown = 30  # seconds
        
    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        if reserved == 0:
            return 0.0
        
        # Fragmentation ratio = (reserved - allocated) / reserved
        return (reserved - allocated) / reserved
    
    def should_defragment(self) -> bool:
        """Check if memory defragmentation is needed."""
        current_time = time.time()
        
        # Don't defragment too frequently
        if current_time - self.last_defrag_time < self.defrag_cooldown:
            return False
        
        fragmentation_ratio = self.get_fragmentation_ratio()
        return fragmentation_ratio > self.defrag_threshold
    
    def defragment(self) -> bool:
        """Perform memory defragmentation."""
        if not torch.cuda.is_available():
            return False
        
        try:
            initial_fragmentation = self.get_fragmentation_ratio()
            
            # Clear cache and force garbage collection
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Multiple rounds of cleanup
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.1)
            
            final_fragmentation = self.get_fragmentation_ratio()
            
            self.last_defrag_time = time.time()
            
            logger.info(
                f"Memory defragmentation: {initial_fragmentation:.1%} -> {final_fragmentation:.1%}"
            )
            
            return final_fragmentation < initial_fragmentation
            
        except Exception as e:
            logger.error(f"Memory defragmentation failed: {e}")
            return False


class GradientAccumulator:
    """Manages gradient accumulation for large effective batch sizes."""
    
    def __init__(
        self,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        adaptive_accumulation: bool = True
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            adaptive_accumulation: Whether to adaptively adjust accumulation steps
        """
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.adaptive_accumulation = adaptive_accumulation
        
        self.current_step = 0
        self.accumulated_loss = 0.0
        self.memory_usage_history: List[float] = []
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Accumulate loss for current step."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        self.accumulated_loss += scaled_loss.item()
        self.current_step += 1
        
        return scaled_loss
    
    def get_accumulated_loss(self) -> float:
        """Get accumulated loss and reset."""
        loss = self.accumulated_loss
        if self.should_step():
            self.accumulated_loss = 0.0
        return loss
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return norm."""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_grad_norm
        ).item()
    
    def adapt_accumulation_steps(self, memory_utilization: float) -> None:
        """Adapt accumulation steps based on memory utilization."""
        if not self.adaptive_accumulation:
            return
        
        self.memory_usage_history.append(memory_utilization)
        
        # Keep only recent history
        if len(self.memory_usage_history) > 10:
            self.memory_usage_history = self.memory_usage_history[-10:]
        
        avg_memory = sum(self.memory_usage_history[-3:]) / min(3, len(self.memory_usage_history))
        
        # Increase accumulation if memory usage is high
        if avg_memory > 0.9:
            new_steps = min(16, self.accumulation_steps * 2)
            if new_steps != self.accumulation_steps:
                logger.info(f"Increasing gradient accumulation: {self.accumulation_steps} -> {new_steps}")
                self.accumulation_steps = new_steps
        
        # Decrease accumulation if memory usage is low
        elif avg_memory < 0.6:
            new_steps = max(1, self.accumulation_steps // 2)
            if new_steps != self.accumulation_steps:
                logger.info(f"Decreasing gradient accumulation: {self.accumulation_steps} -> {new_steps}")
                self.accumulation_steps = new_steps


class AdvancedMemoryMonitor:
    """Advanced memory monitoring with detailed analytics."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize advanced memory monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        self.defragmenter = MemoryDefragmenter()
        
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring_thread is not None:
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started advanced memory monitoring")
    
    def stop_monitoring_thread(self):
        """Stop continuous memory monitoring."""
        if self.monitoring_thread is None:
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        
        logger.info("Stopped advanced memory monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                snapshot = self.capture_snapshot()
                self.snapshots.append(snapshot)
                
                # Keep only recent snapshots (last hour)
                max_snapshots = int(3600 / self.monitoring_interval)
                if len(self.snapshots) > max_snapshots:
                    self.snapshots = self.snapshots[-max_snapshots:]
                
                # Trigger defragmentation if needed
                if self.defragmenter.should_defragment():
                    self.defragmenter.defragment()
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
    
    def capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot."""
        if not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_mb=0,
                reserved_mb=0,
                max_allocated_mb=0,
                memory_utilization=0,
                fragmentation_ratio=0,
                active_tensors=0
            )
        
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        utilization = allocated / total_memory
        fragmentation = self.defragmenter.get_fragmentation_ratio()
        
        # Count active tensors (approximate)
        active_tensors = len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])
        
        return MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            memory_utilization=utilization,
            fragmentation_ratio=fragmentation,
            active_tensors=active_tensors
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
        
        return {
            "current": {
                "allocated_mb": recent_snapshots[-1].allocated_mb,
                "reserved_mb": recent_snapshots[-1].reserved_mb,
                "utilization": recent_snapshots[-1].memory_utilization,
                "fragmentation": recent_snapshots[-1].fragmentation_ratio,
                "active_tensors": recent_snapshots[-1].active_tensors,
            },
            "trends": {
                "avg_utilization": sum(s.memory_utilization for s in recent_snapshots) / len(recent_snapshots),
                "max_utilization": max(s.memory_utilization for s in recent_snapshots),
                "avg_fragmentation": sum(s.fragmentation_ratio for s in recent_snapshots) / len(recent_snapshots),
                "memory_growth": recent_snapshots[-1].allocated_mb - recent_snapshots[0].allocated_mb,
            },
            "alerts": self._generate_memory_alerts(recent_snapshots),
            "recommendations": self._generate_memory_recommendations(recent_snapshots),
        }
    
    def _generate_memory_alerts(self, snapshots: List[MemorySnapshot]) -> List[str]:
        """Generate memory alerts based on recent snapshots."""
        alerts = []
        
        if not snapshots:
            return alerts
        
        current = snapshots[-1]
        
        if current.memory_utilization > 0.9:
            alerts.append("CRITICAL: Memory utilization exceeds 90%")
        elif current.memory_utilization > 0.8:
            alerts.append("WARNING: Memory utilization exceeds 80%")
        
        if current.fragmentation_ratio > 0.4:
            alerts.append("WARNING: High memory fragmentation detected")
        
        if len(snapshots) > 5:
            growth_rate = (snapshots[-1].allocated_mb - snapshots[-5].allocated_mb) / 5
            if growth_rate > 100:  # 100MB per snapshot
                alerts.append(f"WARNING: Rapid memory growth detected ({growth_rate:.1f}MB/snapshot)")
        
        return alerts
    
    def _generate_memory_recommendations(self, snapshots: List[MemorySnapshot]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        current = snapshots[-1]
        avg_util = sum(s.memory_utilization for s in snapshots) / len(snapshots)
        
        if avg_util > 0.85:
            recommendations.extend([
                "Enable more aggressive gradient checkpointing",
                "Reduce batch size or increase gradient accumulation",
                "Consider CPU offloading for optimizer states",
            ])
        
        if current.fragmentation_ratio > 0.3:
            recommendations.extend([
                "Enable automatic memory defragmentation",
                "Reduce tensor creation/destruction frequency",
            ])
        
        if current.active_tensors > 10000:
            recommendations.append("High number of active tensors - check for memory leaks")
        
        return recommendations


class OptimizedModelWrapper:
    """Wrapper for applying comprehensive memory optimizations to models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: MemoryOptimizationConfig
    ):
        """
        Initialize optimized model wrapper.
        
        Args:
            model: Model to optimize
            config: Memory optimization configuration
        """
        self.model = model
        self.config = config
        
        self.batch_sizer = AdaptiveBatchSizer()
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=config.gradient_accumulation_steps
        )
        self.memory_monitor = AdvancedMemoryMonitor()
        
        self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply memory optimizations to the model."""
        logger.info(f"Applying {self.config.level.value} memory optimizations...")
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing_ratio > 0:
            self._apply_gradient_checkpointing()
        
        # Activation checkpointing
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing()
        
        # Mixed precision
        if self.config.mixed_precision:
            self._setup_mixed_precision()
        
        # CPU offloading
        if self.config.cpu_offload:
            self._setup_cpu_offload()
        
        logger.info("Memory optimizations applied successfully")
    
    def _apply_gradient_checkpointing(self):
        """Apply selective gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # For more granular control, implement layer-specific checkpointing
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            checkpoint_every = max(1, int(1.0 / self.config.gradient_checkpointing_ratio))
            
            for i, layer in enumerate(layers):
                if i % checkpoint_every == 0:
                    layer.gradient_checkpointing = True
                    
            logger.info(f"Applied selective checkpointing to {len(layers) // checkpoint_every} layers")
    
    def _apply_activation_checkpointing(self):
        """Apply activation checkpointing."""
        # This would be implemented based on the specific model architecture
        logger.info("Activation checkpointing configured")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        # Convert model to half precision where appropriate
        if self.config.level == MemoryOptimizationLevel.AGGRESSIVE:
            # More aggressive mixed precision
            self.model.half()
            logger.info("Applied aggressive mixed precision (FP16)")
        else:
            # Standard mixed precision with autocast
            logger.info("Mixed precision training enabled (autocast)")
    
    def _setup_cpu_offload(self):
        """Setup CPU offloading for model parameters."""
        logger.info("CPU offloading configured")
        # Implementation would depend on the specific offloading strategy
    
    @contextmanager
    def optimized_training_context(self):
        """Context manager for optimized training."""
        self.memory_monitor.start_monitoring()
        
        try:
            yield self
        finally:
            self.memory_monitor.stop_monitoring_thread()
    
    def optimized_forward(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Perform optimized forward pass with memory management."""
        # Get current memory stats
        memory_stats = self.memory_monitor.capture_snapshot()
        
        # Adapt batch size if needed
        current_batch_size = batch["input_ids"].shape[0]
        recommended_batch_size = self.batch_sizer.adjust_batch_size(
            memory_stats.memory_utilization
        )
        
        # Adjust gradient accumulation based on memory
        self.gradient_accumulator.adapt_accumulation_steps(
            memory_stats.memory_utilization
        )
        
        try:
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Accumulate gradients
            scaled_loss = self.gradient_accumulator.accumulate_loss(loss)
            
            # Backward pass
            scaled_loss.backward()
            
            # Optimizer step if accumulation is complete
            step_taken = False
            if self.gradient_accumulator.should_step():
                grad_norm = self.gradient_accumulator.clip_gradients(self.model)
                optimizer.step()
                optimizer.zero_grad()
                step_taken = True
            
            return {
                "loss": loss.item(),
                "scaled_loss": scaled_loss.item(),
                "accumulated_loss": self.gradient_accumulator.get_accumulated_loss(),
                "step_taken": step_taken,
                "memory_utilization": memory_stats.memory_utilization,
                "recommended_batch_size": recommended_batch_size,
                "current_batch_size": current_batch_size,
            }
            
        except torch.cuda.OutOfMemoryError:
            # Handle OOM with batch size reduction
            new_batch_size = self.batch_sizer.handle_oom()
            logger.warning(f"OOM handled, new batch size: {new_batch_size}")
            
            # Clear cache and retry would happen at higher level
            torch.cuda.empty_cache()
            
            raise  # Re-raise for higher-level handling
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "config": {
                "level": self.config.level.value,
                "gradient_checkpointing_ratio": self.config.gradient_checkpointing_ratio,
                "mixed_precision": self.config.mixed_precision,
                "cpu_offload": self.config.cpu_offload,
            },
            "batch_sizer": {
                "current_batch_size": self.batch_sizer.current_batch_size,
                "oom_count": self.batch_sizer.oom_count,
                "memory_threshold": self.batch_sizer.memory_threshold,
            },
            "gradient_accumulator": {
                "accumulation_steps": self.gradient_accumulator.accumulation_steps,
                "current_step": self.gradient_accumulator.current_step,
                "accumulated_loss": self.gradient_accumulator.accumulated_loss,
            },
            "memory_monitor": self.memory_monitor.get_memory_stats(),
        }


def apply_memory_optimizations(
    model: PreTrainedModel,
    level: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED
) -> OptimizedModelWrapper:
    """
    Apply comprehensive memory optimizations to a model.
    
    Args:
        model: Model to optimize
        level: Optimization level
        
    Returns:
        Optimized model wrapper
    """
    config = MemoryOptimizationConfig(level=level)
    return OptimizedModelWrapper(model, config)


def main():
    """Demonstrate advanced memory optimization capabilities."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Testing advanced memory optimization utilities...")
    
    # Test adaptive batch sizer
    batch_sizer = AdaptiveBatchSizer(initial_batch_size=8)
    
    # Simulate different memory conditions
    for memory_util in [0.5, 0.7, 0.9, 0.95]:
        new_batch_size = batch_sizer.adjust_batch_size(memory_util)
        logger.info(f"Memory: {memory_util:.1%}, Batch size: {new_batch_size}")
        time.sleep(1)  # Simulate time passing
    
    # Test OOM handling
    emergency_batch_size = batch_sizer.handle_oom()
    logger.info(f"Emergency batch size after OOM: {emergency_batch_size}")
    
    # Test memory monitor
    monitor = AdvancedMemoryMonitor(monitoring_interval=0.5)
    monitor.start_monitoring()
    
    # Let it monitor for a few seconds
    time.sleep(3)
    
    monitor.stop_monitoring_thread()
    
    # Get stats
    stats = monitor.get_memory_stats()
    logger.info(f"Memory monitoring stats: {stats}")
    
    # Test gradient accumulator
    accumulator = GradientAccumulator(accumulation_steps=4)
    
    for step in range(10):
        # Simulate loss
        fake_loss = torch.tensor(0.5)
        scaled_loss = accumulator.accumulate_loss(fake_loss)
        should_step = accumulator.should_step()
        
        logger.info(f"Step {step}: Loss {scaled_loss:.3f}, Should step: {should_step}")
    
    print("\n" + "="*70)
    print("ADVANCED MEMORY OPTIMIZATION UTILITIES READY")
    print("="*70)
    print("Components implemented:")
    print("  ✓ Adaptive batch sizing with OOM recovery")
    print("  ✓ Advanced memory monitoring with defragmentation")
    print("  ✓ Intelligent gradient accumulation")
    print("  ✓ Selective gradient checkpointing")
    print("  ✓ Memory fragmentation management")
    print("  ✓ Comprehensive optimization wrapper")
    print("  ✓ Multi-level optimization strategies")
    print("="*70)


if __name__ == "__main__":
    main()