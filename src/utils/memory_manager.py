"""
GPU Memory Management utilities with OOM prevention.

Provides memory monitoring, automatic batch size adjustment, and OOM recovery.
"""
import gc
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, List

import psutil
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
        
        # Gradient checkpointing state
        self.checkpointed_modules = []
        self.checkpoint_ratio = 0.3  # Default: checkpoint 30% of layers
        
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
    def oom_protected(self, fallback_fn: Optional[Callable] = None):
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
    
    def monitor_memory_loop(self, callback: Optional[Callable] = None, interval: float = 5.0):
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
    
    def enable_selective_checkpointing(
        self, 
        model: nn.Module, 
        layers_to_checkpoint: int = 3,
        checkpoint_ratio: Optional[float] = None
    ) -> int:
        """
        Enable selective gradient checkpointing on model layers.
        
        Args:
            model: PyTorch model to apply checkpointing to
            layers_to_checkpoint: Checkpoint every N layers (default: 3)
            checkpoint_ratio: Alternative - checkpoint this fraction of layers
            
        Returns:
            Number of layers checkpointed
        """
        if checkpoint_ratio is not None:
            self.checkpoint_ratio = checkpoint_ratio
        
        checkpointed_count = 0
        
        # Handle different model architectures
        layers = self._get_model_layers(model)
        
        if not layers:
            logger.warning("No suitable layers found for checkpointing")
            return 0
        
        # Apply checkpointing based on strategy
        if checkpoint_ratio is not None:
            # Checkpoint based on ratio
            total_layers = len(layers)
            checkpoint_every = max(1, int(1.0 / checkpoint_ratio))
        else:
            # Checkpoint every N layers
            checkpoint_every = layers_to_checkpoint
        
        for i, layer in enumerate(layers):
            if i % checkpoint_every == 0:
                checkpointed_layer = self._wrap_layer_with_checkpoint(layer)
                if checkpointed_layer is not None:
                    self.checkpointed_modules.append((layer, checkpointed_layer))
                    checkpointed_count += 1
        
        logger.info(
            f"Enabled selective checkpointing on {checkpointed_count}/{len(layers)} layers "
            f"(every {checkpoint_every} layers)"
        )
        
        return checkpointed_count
    
    def _get_model_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract layers suitable for checkpointing from model."""
        layers = []
        
        # Try different common model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama-style models
            layers = list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style models
            layers = list(model.transformer.h)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style models
            layers = list(model.encoder.layer)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            layers = list(model.layers)
        else:
            # Fallback: look for nn.Module children that look like layers
            for name, module in model.named_children():
                if any(layer_name in name.lower() for layer_name in ['layer', 'block', 'stage']):
                    if hasattr(module, '__iter__'):
                        layers.extend(list(module))
                    else:
                        layers.append(module)
        
        return layers
    
    def _wrap_layer_with_checkpoint(self, layer: nn.Module) -> Optional[nn.Module]:
        """Wrap a layer with gradient checkpointing."""
        try:
            # Create a checkpointed version of the layer's forward method
            original_forward = layer.forward
            
            def checkpointed_forward(*args, **kwargs):
                # Use PyTorch's gradient checkpointing
                return checkpoint(original_forward, *args, **kwargs)
            
            layer.forward = checkpointed_forward
            layer._original_forward = original_forward
            layer._is_checkpointed = True
            
            return layer
            
        except Exception as e:
            logger.warning(f"Failed to apply checkpointing to layer: {e}")
            return None
    
    def disable_checkpointing(self) -> int:
        """Disable checkpointing on all previously checkpointed modules."""
        disabled_count = 0
        
        for original_layer, checkpointed_layer in self.checkpointed_modules:
            try:
                if hasattr(checkpointed_layer, '_original_forward'):
                    checkpointed_layer.forward = checkpointed_layer._original_forward
                    delattr(checkpointed_layer, '_original_forward')
                    delattr(checkpointed_layer, '_is_checkpointed')
                    disabled_count += 1
            except Exception as e:
                logger.warning(f"Failed to disable checkpointing: {e}")
        
        self.checkpointed_modules.clear()
        logger.info(f"Disabled checkpointing on {disabled_count} layers")
        
        return disabled_count
    
    def estimate_memory_savings(self, model: nn.Module, input_size: tuple) -> dict[str, float]:
        """
        Estimate memory savings from gradient checkpointing.
        
        Args:
            model: PyTorch model
            input_size: Size of model input (batch_size, ...)
            
        Returns:
            Dictionary with memory estimates
        """
        try:
            # Rough estimation based on model parameters and input size
            total_params = sum(p.numel() for p in model.parameters())
            param_memory_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
            
            # Estimate activation memory (very rough)
            batch_size = input_size[0] if input_size else 1
            activation_memory_mb = batch_size * total_params * 0.1 / 1024 / 1024  # Rough estimate
            
            # Checkpointing typically saves 30-50% of activation memory
            savings_ratio = self.checkpoint_ratio * 0.4  # 40% savings on checkpointed layers
            estimated_savings_mb = activation_memory_mb * savings_ratio
            
            return {
                "total_param_memory_mb": param_memory_mb,
                "estimated_activation_memory_mb": activation_memory_mb,
                "estimated_savings_mb": estimated_savings_mb,
                "savings_percentage": savings_ratio * 100,
                "checkpoint_ratio": self.checkpoint_ratio
            }
            
        except Exception as e:
            logger.warning(f"Failed to estimate memory savings: {e}")
            return {
                "error": str(e),
                "estimated_savings_mb": 0.0,
                "savings_percentage": 0.0
            }
    
    def get_checkpointing_stats(self) -> dict[str, Any]:
        """Get statistics about gradient checkpointing usage."""
        return {
            "checkpointed_modules_count": len(self.checkpointed_modules),
            "checkpoint_ratio": self.checkpoint_ratio,
            "checkpointing_enabled": len(self.checkpointed_modules) > 0,
            "oom_count": self.oom_count,
            "memory_pressure": self.check_memory_pressure()
        }


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


class GradientCheckpointingManager:
    """Specialized manager for gradient checkpointing operations."""
    
    def __init__(self, memory_manager: MemoryManager):
        """Initialize with a memory manager."""
        self.memory_manager = memory_manager
        self.checkpoint_policies = {}
        
    def create_adaptive_policy(
        self,
        model: nn.Module,
        memory_threshold: float = 0.8,
        base_checkpoint_ratio: float = 0.3
    ) -> callable:
        """
        Create an adaptive checkpointing policy based on memory pressure.
        
        Args:
            model: Model to apply policy to
            memory_threshold: Memory utilization threshold to increase checkpointing
            base_checkpoint_ratio: Base ratio of layers to checkpoint
            
        Returns:
            Policy function that can be called to update checkpointing
        """
        def policy():
            current_pressure = self.memory_manager.check_memory_pressure()
            current_usage = self.memory_manager.get_memory_usage()
            usage_ratio = current_usage["usage_percentage"] / 100.0
            
            if usage_ratio > memory_threshold or current_pressure == "critical":
                # Increase checkpointing under memory pressure
                new_ratio = min(0.8, base_checkpoint_ratio * 1.5)
                checkpoint_every = max(1, int(1.0 / new_ratio))
            elif current_pressure == "warning":
                # Moderate checkpointing
                new_ratio = base_checkpoint_ratio * 1.2
                checkpoint_every = max(1, int(1.0 / new_ratio))
            else:
                # Normal checkpointing
                new_ratio = base_checkpoint_ratio
                checkpoint_every = max(1, int(1.0 / new_ratio))
            
            # Apply the new policy
            if new_ratio != self.memory_manager.checkpoint_ratio:
                logger.info(f"Adjusting checkpointing: ratio {new_ratio:.2f}, every {checkpoint_every} layers")
                
                # Disable old checkpointing
                self.memory_manager.disable_checkpointing()
                
                # Enable new checkpointing
                self.memory_manager.enable_selective_checkpointing(
                    model, 
                    layers_to_checkpoint=checkpoint_every,
                    checkpoint_ratio=new_ratio
                )
            
            return {
                "checkpoint_ratio": new_ratio,
                "checkpoint_every": checkpoint_every,
                "memory_pressure": current_pressure,
                "usage_ratio": usage_ratio
            }
        
        policy_id = id(model)
        self.checkpoint_policies[policy_id] = policy
        return policy
    
    def apply_model_specific_checkpointing(
        self,
        model: nn.Module,
        model_type: str = "auto"
    ) -> dict[str, Any]:
        """
        Apply model-specific checkpointing strategies.
        
        Args:
            model: Model to optimize
            model_type: Type of model ("llama", "gpt", "bert", or "auto")
            
        Returns:
            Dictionary with checkpointing results
        """
        if model_type == "auto":
            model_type = self._detect_model_type(model)
        
        # Model-specific checkpointing strategies
        strategies = {
            "llama": {"layers_to_checkpoint": 3, "target_memory_reduction": 0.4},
            "gpt": {"layers_to_checkpoint": 4, "target_memory_reduction": 0.35},
            "bert": {"layers_to_checkpoint": 2, "target_memory_reduction": 0.3},
            "unknown": {"layers_to_checkpoint": 3, "target_memory_reduction": 0.35}
        }
        
        strategy = strategies.get(model_type, strategies["unknown"])
        
        # Apply checkpointing
        checkpointed_count = self.memory_manager.enable_selective_checkpointing(
            model,
            layers_to_checkpoint=strategy["layers_to_checkpoint"]
        )
        
        # Estimate memory savings
        input_size = (1, 512)  # Rough estimate for text models
        savings_estimate = self.memory_manager.estimate_memory_savings(model, input_size)
        
        return {
            "model_type": model_type,
            "strategy": strategy,
            "checkpointed_layers": checkpointed_count,
            "memory_savings_estimate": savings_estimate,
            "success": checkpointed_count > 0
        }
    
    def _detect_model_type(self, model: nn.Module) -> str:
        """Detect model type based on architecture."""
        model_str = str(type(model)).lower()
        
        if "llama" in model_str:
            return "llama"
        elif "gpt" in model_str:
            return "gpt"
        elif "bert" in model_str:
            return "bert"
        else:
            # Try to detect based on layer structure
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return "llama"
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return "gpt"
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                return "bert"
            else:
                return "unknown"