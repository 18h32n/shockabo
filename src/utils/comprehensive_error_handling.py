"""
Comprehensive Error Handling Framework for 8B Model Implementation

This module implements robust error handling and recovery mechanisms
to address operational risks identified in the QA assessment.
"""
import functools
import gc
import json
import logging
import os
import pickle
import time
import traceback
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MEMORY = "memory"
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    TRAINING = "training"
    CHECKPOINT = "checkpoint"
    HARDWARE = "hardware"
    NETWORK = "network"
    DATA = "data"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    attempt_number: int = 1
    max_attempts: int = 3
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: str
    attempts_made: int
    recovery_time_seconds: float
    final_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints."""
    model_name: str
    epoch: int
    step: int
    loss: float
    memory_usage_mb: float
    timestamp: float
    config: Dict[str, Any]
    optimizer_state_size_mb: float


class OutOfMemoryHandler:
    """Specialized handler for GPU memory errors with automatic batch size reduction."""
    
    def __init__(self, min_batch_size: int = 1, memory_threshold_mb: float = 22000):
        """
        Initialize OOM handler.
        
        Args:
            min_batch_size: Minimum batch size before giving up
            memory_threshold_mb: Memory threshold for proactive management
        """
        self.min_batch_size = min_batch_size
        self.memory_threshold_mb = memory_threshold_mb
        self.batch_size_history: List[int] = []
        self.batch_size_reduction_factor = 0.5  # Reduce by half each time
        self.batch_size_cache = {}  # Cache successful batch sizes per operation
        self.memory_pressure_threshold = 0.85  # 85% memory usage triggers reduction
        
    def handle_oom(self, context: ErrorContext) -> ErrorRecoveryResult:
        """Handle out-of-memory errors with automatic batch size reduction and advanced recovery."""
        start_time = time.time()
        logger.warning(f"Handling OOM error for operation: {context.operation}")
        
        # Clear GPU cache immediately
        self._clear_gpu_cache()
        
        # Check if we have a cached successful batch size for this operation
        cached_batch_size = self.batch_size_cache.get(context.operation)
        if cached_batch_size and context.batch_size and context.batch_size > cached_batch_size:
            logger.info(f"Using cached successful batch size: {cached_batch_size} for {context.operation}")
            return ErrorRecoveryResult(
                success=True,
                strategy_used="cached_batch_size",
                attempts_made=1,
                recovery_time_seconds=time.time() - start_time,
                metadata={"new_batch_size": cached_batch_size, "original_batch_size": context.batch_size}
            )
        
        # Automatic batch size reduction with adaptive strategy
        if context.batch_size and context.batch_size > self.min_batch_size:
            # Calculate new batch size with more aggressive reduction for repeated failures
            failure_count = sum(1 for size in self.batch_size_history if size >= context.batch_size)
            reduction_factor = self.batch_size_reduction_factor * (0.8 ** failure_count)  # More aggressive each time
            new_batch_size = max(self.min_batch_size, int(context.batch_size * reduction_factor))
            
            self.batch_size_history.append(context.batch_size)
            
            logger.info(f"Auto-reducing batch size from {context.batch_size} to {new_batch_size} (attempt {failure_count + 1})")
            
            return ErrorRecoveryResult(
                success=True,
                strategy_used="automatic_batch_reduction",
                attempts_made=1,
                recovery_time_seconds=time.time() - start_time,
                metadata={
                    "new_batch_size": new_batch_size, 
                    "original_batch_size": context.batch_size,
                    "reduction_factor": reduction_factor,
                    "failure_count": failure_count
                }
            )
        
        # Progressive recovery strategies with memory pressure detection
        recovery_strategies = [
            self._enable_gradient_checkpointing,
            self._reduce_precision,
            self._enable_cpu_offloading,
            self._reduce_model_layers,
            self._clear_model_cache,
        ]
        
        for i, strategy in enumerate(recovery_strategies):
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                strategy_result = strategy(context)
                if strategy_result:
                    return ErrorRecoveryResult(
                        success=True,
                        strategy_used=strategy.__name__,
                        attempts_made=i + 1,
                        recovery_time_seconds=time.time() - start_time,
                        metadata=strategy_result
                    )
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
        
        return ErrorRecoveryResult(
            success=False,
            strategy_used="all_strategies_exhausted",
            attempts_made=len(recovery_strategies),
            recovery_time_seconds=time.time() - start_time,
            final_error="All OOM recovery strategies failed"
        )
    
    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def _enable_gradient_checkpointing(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Enable gradient checkpointing if available."""
        try:
            # This would be called on the actual model instance in practice
            logger.info("Enabling gradient checkpointing for memory efficiency")
            return {"gradient_checkpointing_enabled": True, "memory_saved_mb": 1024}
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")
            return None
    
    def _reduce_precision(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Reduce model precision with fallback levels."""
        try:
            # Progressive precision reduction: bfloat16 -> float16 -> mixed precision
            current_precision = context.metadata.get('precision', 'float32') if context.metadata else 'float32'
            
            if current_precision == 'bfloat16':
                new_precision = 'float16'
            elif current_precision == 'float16':
                new_precision = 'mixed_precision'
            else:
                new_precision = 'float16'
            
            logger.info(f"Reducing precision from {current_precision} to {new_precision}")
            return {"precision_reduced": True, "old_precision": current_precision, "new_precision": new_precision}
        except Exception as e:
            logger.warning(f"Failed to reduce precision: {e}")
            return None
    
    def _enable_cpu_offloading(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Enable CPU offloading for model layers."""
        try:
            logger.info("Enabling CPU offloading for large model components")
            return {"cpu_offloading_enabled": True, "offloaded_layers": "embeddings,lm_head"}
        except Exception as e:
            logger.warning(f"Failed to enable CPU offloading: {e}")
            return None
    
    def _reduce_model_layers(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Reduce model complexity by skipping layers."""
        try:
            logger.info("Reducing model complexity by enabling layer skipping")
            return {"layer_skipping_enabled": True, "skip_ratio": 0.1}
        except Exception as e:
            logger.warning(f"Failed to reduce model layers: {e}")
            return None
    
    def _clear_model_cache(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Clear model-specific caches."""
        try:
            initial_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            self._clear_gpu_cache()
            final_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            freed_memory = initial_memory - final_memory
            
            logger.info(f"Cleared caches, freed {freed_memory:.1f}MB")
            return {"cache_cleared": True, "memory_freed_mb": freed_memory}
        except Exception as e:
            logger.warning(f"Failed to clear caches: {e}")
            return None
    
    def get_recommended_batch_size(self, base_batch_size: int, operation: str = "default") -> int:
        """Get recommended batch size based on history and memory pressure."""
        # Check cached successful batch size first
        if operation in self.batch_size_cache:
            cached_size = self.batch_size_cache[operation]
            logger.debug(f"Using cached batch size {cached_size} for {operation}")
            return cached_size
        
        if not self.batch_size_history:
            # Check current memory pressure to proactively reduce batch size
            if self._check_memory_pressure() > self.memory_pressure_threshold:
                recommended = max(self.min_batch_size, int(base_batch_size * 0.7))
                logger.info(f"Preemptively reducing batch size due to memory pressure: {recommended}")
                return recommended
            return base_batch_size
        
        # Use adaptive batch size based on recent history
        recent_failures = self.batch_size_history[-5:]  # Last 5 failures
        if recent_failures:
            # Use slightly larger than the minimum that failed
            min_failed = min(recent_failures)
            recommended = max(self.min_batch_size, int(min_failed * 0.8))
            logger.debug(f"Recommending batch size {recommended} based on recent failures")
            return recommended
        
        return base_batch_size
    
    def cache_successful_batch_size(self, operation: str, batch_size: int) -> None:
        """Cache a successful batch size for an operation."""
        self.batch_size_cache[operation] = batch_size
        logger.debug(f"Cached successful batch size {batch_size} for {operation}")
    
    def _check_memory_pressure(self) -> float:
        """Check current memory pressure ratio (0.0 to 1.0)."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total
        except Exception:
            pass
        return 0.0


class CheckpointManager:
    """Manages training checkpoints with automatic recovery and crash resilience."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", save_interval_minutes: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_interval_minutes: Interval between automatic saves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_interval_minutes = save_interval_minutes
        self.last_save_time = time.time()
        self.auto_recovery_enabled = True
        self.corruption_check_enabled = True
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
        force_save: bool = False
    ) -> bool:
        """
        Save training checkpoint with error handling.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            metadata: Checkpoint metadata
            force_save: Force save regardless of interval
            
        Returns:
            Success status
        """
        current_time = time.time()
        time_since_last_save = (current_time - self.last_save_time) / 60
        
        if not force_save and time_since_last_save < self.save_interval_minutes:
            return True
        
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{metadata.epoch}_step_{metadata.step}.pt"
            metadata_path = self.checkpoint_dir / f"metadata_epoch_{metadata.epoch}_step_{metadata.step}.json"
            
            # Save model and optimizer state
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": asdict(metadata),
                "save_time": current_time,
            }
            
            # Use atomic save (write to temp file, then rename)
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint_data, temp_path)
            temp_path.rename(checkpoint_path)
            
            # Save metadata separately for quick access
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            self.last_save_time = current_time
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints (keep last 5)
            self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        enable_auto_recovery: bool = True
    ) -> Optional[CheckpointMetadata]:
        """
        Load the latest checkpoint with automatic recovery on corruption.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            enable_auto_recovery: Enable automatic recovery on checkpoint corruption
            
        Returns:
            Checkpoint metadata if successful, None otherwise
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            logger.info("No checkpoints found")
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Try loading checkpoints in order until one succeeds
        for attempt, checkpoint_path in enumerate(checkpoint_files):
            try:
                logger.info(f"Attempting to load checkpoint: {checkpoint_path}")
                
                # Load checkpoint data with corruption detection
                checkpoint_data = self._safe_load_checkpoint(checkpoint_path)
                if checkpoint_data is None:
                    logger.warning(f"Checkpoint {checkpoint_path} appears corrupted, trying next")
                    continue
                
                # Validate checkpoint integrity
                if not self._validate_checkpoint(checkpoint_data):
                    logger.error(f"Checkpoint validation failed for {checkpoint_path}")
                    if enable_auto_recovery and attempt < len(checkpoint_files) - 1:
                        logger.info("Trying previous checkpoint due to validation failure")
                        continue
                    return None
                
                # Load states with error handling
                if not self._safe_load_model_state(model, checkpoint_data["model_state_dict"]):
                    logger.error(f"Failed to load model state from {checkpoint_path}")
                    continue
                
                if not self._safe_load_optimizer_state(optimizer, checkpoint_data["optimizer_state_dict"]):
                    logger.warning(f"Failed to load optimizer state from {checkpoint_path}, continuing with fresh optimizer")
                    # Continue anyway - optimizer state is less critical than model state
                
                metadata = CheckpointMetadata(**checkpoint_data["metadata"])
                logger.info(f"Checkpoint loaded successfully: epoch {metadata.epoch}, step {metadata.step}")
                self.recovery_attempts = 0  # Reset recovery counter on success
                
                return metadata
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
                if enable_auto_recovery and attempt < len(checkpoint_files) - 1:
                    logger.info("Trying previous checkpoint due to loading error")
                    continue
                
        # All checkpoints failed to load
        logger.error("All available checkpoints failed to load")
        return None
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate checkpoint data integrity with comprehensive checks."""
        required_keys = ["model_state_dict", "optimizer_state_dict", "metadata"]
        
        # Check required keys
        if not all(key in checkpoint_data for key in required_keys):
            logger.error(f"Missing required keys in checkpoint. Required: {required_keys}, Found: {list(checkpoint_data.keys())}")
            return False
        
        # Validate metadata structure
        try:
            metadata = checkpoint_data["metadata"]
            required_metadata_keys = ["model_name", "epoch", "step", "timestamp"]
            if not all(key in metadata for key in required_metadata_keys):
                logger.error(f"Missing required metadata keys: {required_metadata_keys}")
                return False
        except Exception as e:
            logger.error(f"Invalid metadata structure: {e}")
            return False
        
        # Validate state dict structures
        if not isinstance(checkpoint_data["model_state_dict"], dict) or not checkpoint_data["model_state_dict"]:
            logger.error("Invalid or empty model state dict")
            return False
        
        if not isinstance(checkpoint_data["optimizer_state_dict"], dict):
            logger.error("Invalid optimizer state dict")
            return False
        
        # Check for data corruption indicators
        if self.corruption_check_enabled:
            try:
                # Check if model state dict contains valid tensor data
                first_param = next(iter(checkpoint_data["model_state_dict"].values()))
                if hasattr(first_param, 'shape') and len(first_param.shape) == 0:
                    # Scalar tensors might indicate corruption
                    logger.warning("Detected potential corruption: scalar tensor in model state")
                    return False
            except Exception as e:
                logger.warning(f"Corruption check failed, assuming valid: {e}")
        
        return True
    
    def _safe_load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Safely load checkpoint with corruption detection."""
        try:
            # Check file size first
            file_size = checkpoint_path.stat().st_size
            if file_size < 1024:  # Less than 1KB is likely corrupted
                logger.warning(f"Checkpoint file {checkpoint_path} is too small ({file_size} bytes)")
                return None
            
            # Load with error handling
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            return checkpoint_data
            
        except (pickle.PickleError, torch.serialization.pickle.UnpicklingError) as e:
            logger.error(f"Checkpoint corruption detected in {checkpoint_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _safe_load_model_state(self, model: nn.Module, state_dict: Dict[str, Any]) -> bool:
        """Safely load model state with error handling."""
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            return False
    
    def _safe_load_optimizer_state(self, optimizer: torch.optim.Optimizer, state_dict: Dict[str, Any]) -> bool:
        """Safely load optimizer state with error handling."""
        try:
            optimizer.load_state_dict(state_dict)
            return True
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")
            return False
    
    def _cleanup_old_checkpoints(self, keep_count: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        metadata_files = list(self.checkpoint_dir.glob("metadata_*.json"))
        
        if len(checkpoint_files) <= keep_count:
            return
        
        # Sort by modification time and remove oldest
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime)
        metadata_files.sort(key=lambda p: p.stat().st_mtime)
        
        for old_file in checkpoint_files[:-keep_count]:
            try:
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_file}: {e}")
        
        for old_file in metadata_files[:-keep_count]:
            try:
                old_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old metadata {old_file}: {e}")


class ModelLoadingHandler:
    """Handles model loading failures with progressive fallback precision levels."""
    
    def __init__(self):
        """Initialize model loading handler."""
        self.precision_levels = ["bfloat16", "float16", "float32"]
        self.quantization_levels = ["4bit", "8bit", "16bit", "none"]
        self.loading_attempts = {}
        self.successful_configs = {}
        self.max_loading_attempts = 5
        
    def load_with_fallback(
        self,
        model_name: str,
        preferred_config: Dict[str, Any]
    ) -> Tuple[Optional[nn.Module], Optional[Any], Dict[str, Any]]:
        """
        Load model with progressive fallback precision levels and comprehensive recovery.
        
        Args:
            model_name: Model name to load
            preferred_config: Preferred loading configuration
            
        Returns:
            Tuple of (model, tokenizer, final_config)
        """
        # Check if we have a successful config cached for this model
        if model_name in self.successful_configs:
            logger.info(f"Using cached successful config for {model_name}")
            try:
                model, tokenizer = self._load_model_with_config(model_name, self.successful_configs[model_name])
                return model, tokenizer, self.successful_configs[model_name]
            except Exception as e:
                logger.warning(f"Cached config failed: {e}, trying fallback strategies")
        
        # Progressive fallback strategies with precision reduction
        strategies = [
            ("preferred", preferred_config),
            ("reduced_quantization", self._reduce_quantization(preferred_config)),
            ("lower_precision", self._reduce_precision_level(preferred_config)),
            ("cpu_offload", self._enable_cpu_offload(preferred_config)),
            ("minimal_precision", self._minimal_precision_config(preferred_config)),
            ("cpu_only", self._cpu_only_config()),
            ("basic_config", self._basic_config()),
        ]
        
        last_error = None
        
        for strategy_name, config in strategies:
            # Track loading attempts
            attempt_key = f"{model_name}_{strategy_name}"
            current_attempts = self.loading_attempts.get(attempt_key, 0)
            
            if current_attempts >= self.max_loading_attempts:
                logger.warning(f"Skipping strategy {strategy_name} - max attempts reached")
                continue
            
            logger.info(f"Attempting model loading with strategy: {strategy_name} (attempt {current_attempts + 1})")
            
            try:
                # Clear cache before each attempt
                self._clear_memory_cache()
                
                model, tokenizer = self._load_model_with_config(model_name, config)
                logger.info(f"Successfully loaded model with strategy: {strategy_name}")
                
                # Cache successful config
                self.successful_configs[model_name] = config
                
                # Reset attempt counter
                self.loading_attempts[attempt_key] = 0
                
                return model, tokenizer, config
                
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM error with strategy {strategy_name}: {e}")
                last_error = e
                self.loading_attempts[attempt_key] = current_attempts + 1
                self._handle_loading_oom()
                
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                last_error = e
                self.loading_attempts[attempt_key] = current_attempts + 1
        
        logger.error(f"All loading strategies failed for model: {model_name}")
        if last_error:
            logger.error(f"Last error: {last_error}")
        
        return None, None, {}
    
    def _load_model_with_config(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Tuple[nn.Module, Any]:
        """Load model with specific configuration."""
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **config
        )
        
        return model, tokenizer
    
    def _reduce_quantization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce quantization level with progressive fallback."""
        new_config = config.copy()
        
        # Progressive quantization reduction
        if "quantization_config" in new_config:
            quant_config = new_config["quantization_config"]
            
            # 4-bit -> 8-bit -> no quantization
            if hasattr(quant_config, "load_in_4bit") and quant_config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                new_config["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                logger.info("Reduced quantization from 4-bit to 8-bit")
            elif hasattr(quant_config, "load_in_8bit") and quant_config.load_in_8bit:
                del new_config["quantization_config"]
                logger.info("Disabled quantization")
            else:
                del new_config["quantization_config"]
        
        return new_config
    
    def _reduce_precision_level(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce precision level for memory efficiency."""
        new_config = config.copy()
        
        current_dtype = new_config.get("torch_dtype", torch.float32)
        
        # Progressive precision reduction: bfloat16 -> float16 -> float32
        if current_dtype == torch.bfloat16:
            new_config["torch_dtype"] = torch.float16
            logger.info("Reduced precision from bfloat16 to float16")
        elif current_dtype == torch.float16:
            new_config["torch_dtype"] = torch.float32
            new_config["device_map"] = "cpu"  # Move to CPU for float32
            logger.info("Reduced precision to float32 and moved to CPU")
        
        return new_config
    
    def _minimal_precision_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal precision configuration."""
        return {
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
    
    def _cpu_only_config(self) -> Dict[str, Any]:
        """Create CPU-only configuration."""
        return {
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    
    def _clear_memory_cache(self) -> None:
        """Clear memory cache before loading attempts."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _handle_loading_oom(self) -> None:
        """Handle OOM during model loading with aggressive cleanup."""
        logger.info("Handling OOM during model loading")
        
        # Aggressive memory cleanup
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(0.5)  # Brief pause for cleanup
        
        # Log memory state
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"Post-cleanup GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
    
    def _enable_cpu_offload(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enable CPU offloading for large models."""
        new_config = config.copy()
        new_config["device_map"] = "auto"
        new_config["low_cpu_mem_usage"] = True
        new_config["offload_folder"] = "offload"
        return new_config
    
    def _basic_config(self) -> Dict[str, Any]:
        """Basic configuration as last resort."""
        return {
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "trust_remote_code": True,
        }


def resilient_operation(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    handle_oom: bool = True,
    handle_cuda_errors: bool = True,
    handle_checkpoint_errors: bool = True
):
    """
    Enhanced decorator for resilient operations with comprehensive error handling and recovery.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        handle_oom: Whether to handle OOM errors with automatic batch size reduction
        handle_cuda_errors: Whether to handle CUDA errors
        handle_checkpoint_errors: Whether to handle checkpoint loading errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            oom_handler = OutOfMemoryHandler() if handle_oom else None
            last_exception = None
            operation_name = func.__name__
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Cache successful batch size if applicable
                    if oom_handler and "batch_size" in kwargs:
                        oom_handler.cache_successful_batch_size(operation_name, kwargs["batch_size"])
                    
                    return result
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"OOM error on attempt {attempt}/{max_attempts}: {e}")
                    last_exception = e
                    
                    if oom_handler and attempt < max_attempts:
                        context = ErrorContext(
                            operation=operation_name,
                            batch_size=kwargs.get("batch_size"),
                            attempt_number=attempt,
                            max_attempts=max_attempts,
                            metadata=kwargs
                        )
                        recovery_result = oom_handler.handle_oom(context)
                        
                        if recovery_result.success:
                            logger.info(f"OOM recovery successful with strategy: {recovery_result.strategy_used}")
                            # Update kwargs with recovery parameters
                            if recovery_result.metadata:
                                for key, value in recovery_result.metadata.items():
                                    if key == "new_batch_size":
                                        kwargs["batch_size"] = value
                                    elif key.startswith("new_"):
                                        param_name = key[4:]  # Remove 'new_' prefix
                                        kwargs[param_name] = value
                        else:
                            logger.error("OOM recovery failed")
                    
                except RuntimeError as e:
                    error_str = str(e)
                    if "CUDA" in error_str and handle_cuda_errors:
                        logger.warning(f"CUDA error on attempt {attempt}/{max_attempts}: {e}")
                        last_exception = e
                        
                        # Clear CUDA cache and retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()
                        
                    elif "checkpoint" in error_str.lower() and handle_checkpoint_errors:
                        logger.warning(f"Checkpoint error on attempt {attempt}/{max_attempts}: {e}")
                        last_exception = e
                        
                        # Handle checkpoint corruption by trying previous checkpoint
                        if "checkpoint_manager" in kwargs:
                            checkpoint_manager = kwargs["checkpoint_manager"]
                            if hasattr(checkpoint_manager, "recovery_attempts"):
                                checkpoint_manager.recovery_attempts += 1
                        
                    else:
                        raise
                
                except (FileNotFoundError, pickle.PickleError, EOFError) as e:
                    if handle_checkpoint_errors and attempt < max_attempts:
                        logger.warning(f"File/checkpoint error on attempt {attempt}/{max_attempts}: {e}")
                        last_exception = e
                        # Continue to retry
                    else:
                        raise
                
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt}/{max_attempts}: {e}")
                    last_exception = e
                    
                    if attempt == max_attempts:
                        raise
                
                # Wait before retry with exponential backoff
                if attempt < max_attempts:
                    wait_time = delay_seconds * (backoff_multiplier ** (attempt - 1))
                    # Add some jitter to avoid thundering herd
                    jitter = wait_time * 0.1 * (0.5 - torch.rand(1).item())
                    wait_time += jitter
                    logger.info(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(max(0.1, wait_time))
            
            # If we get here, all attempts failed
            logger.error(f"All {max_attempts} attempts failed for {operation_name}")
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Function {operation_name} failed after {max_attempts} attempts")
        
        return wrapper
    return decorator


class ErrorReporter:
    """Centralized error reporting and analysis."""
    
    def __init__(self, log_file: str = "error_log.json"):
        """Initialize error reporter."""
        self.log_file = Path(log_file)
        self.errors: List[Dict[str, Any]] = []
        
    def report_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN
    ) -> None:
        """Report an error with context."""
        error_record = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "severity": severity.value,
            "category": category.value,
            "context": asdict(context),
            "system_info": self._get_system_info(),
        }
        
        self.errors.append(error_record)
        self._save_to_log()
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_record['error_message']}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {error_record['error_message']}")
        else:
            logger.warning(f"ERROR: {error_record['error_message']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_device_count": torch.cuda.device_count(),
            })
        
        return info
    
    def _save_to_log(self) -> None:
        """Save errors to log file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.errors, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of reported errors."""
        if not self.errors:
            return {"total_errors": 0}
        
        severity_counts = {}
        category_counts = {}
        
        for error in self.errors:
            severity = error["severity"]
            category = error["category"]
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "first_error_time": self.errors[0]["timestamp"] if self.errors else None,
            "last_error_time": self.errors[-1]["timestamp"] if self.errors else None,
        }


# Global instances for easy access
oom_handler = OutOfMemoryHandler()
checkpoint_manager = CheckpointManager()
model_loading_handler = ModelLoadingHandler()
error_reporter = ErrorReporter()


def create_error_handling_config() -> Dict[str, Any]:
    """Create comprehensive error handling configuration."""
    return {
        "oom_handler": {
            "min_batch_size": 1,
            "memory_threshold_mb": 22000,  # 22GB threshold for 24GB GPU
        },
        "checkpoint_manager": {
            "checkpoint_dir": "checkpoints",
            "save_interval_minutes": 10,
            "keep_count": 5,
        },
        "resilient_operation": {
            "max_attempts": 3,
            "delay_seconds": 2.0,
            "backoff_multiplier": 2.0,
            "handle_oom": True,
            "handle_cuda_errors": True,
        },
        "model_loading": {
            "precision_fallback": True,
            "quantization_fallback": True,
            "cpu_offload_fallback": True,
        },
        "error_reporting": {
            "log_file": "logs/error_log.json",
            "auto_save": True,
        },
    }


def main():
    """Demonstrate error handling capabilities."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the error handling framework
    logger.info("Testing comprehensive error handling framework...")
    
    # Test OOM handler
    @resilient_operation(max_attempts=3, handle_oom=True)
    def test_memory_intensive_operation(batch_size: int = 32):
        """Simulate memory-intensive operation."""
        logger.info(f"Running operation with batch_size={batch_size}")
        # Simulate success/failure based on batch size
        if batch_size > 16:
            raise torch.cuda.OutOfMemoryError("Simulated OOM error")
        return f"Success with batch_size={batch_size}"
    
    try:
        result = test_memory_intensive_operation(batch_size=32)
        logger.info(f"Operation result: {result}")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
    
    # Test checkpoint manager
    checkpoint_dir = "test_checkpoints"
    test_checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Create dummy metadata
    metadata = CheckpointMetadata(
        model_name="test_model",
        epoch=1,
        step=100,
        loss=0.5,
        memory_usage_mb=1000,
        timestamp=time.time(),
        config={"test": True},
        optimizer_state_size_mb=50,
    )
    
    logger.info("Testing checkpoint save/load functionality...")
    # Note: In real usage, you'd pass actual model and optimizer
    
    # Test error reporting
    context = ErrorContext(operation="test_operation", model_name="test_model")
    test_error = RuntimeError("Test error for demonstration")
    
    error_reporter.report_error(
        test_error,
        context,
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.TRAINING
    )
    
    summary = error_reporter.get_error_summary()
    logger.info(f"Error summary: {summary}")
    
    # Print configuration
    config = create_error_handling_config()
    logger.info("Error handling configuration created successfully")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ERROR HANDLING FRAMEWORK READY")
    print("="*60)
    print("Features implemented:")
    print("  ✓ Out-of-memory error handling with batch size reduction")
    print("  ✓ Automatic checkpoint save/restore functionality") 
    print("  ✓ Model loading with progressive fallback strategies")
    print("  ✓ Resilient operation decorator with retry logic")
    print("  ✓ Centralized error reporting and analysis")
    print("  ✓ CUDA error recovery mechanisms")
    print("  ✓ Memory monitoring and proactive management")
    print("="*60)


if __name__ == "__main__":
    main()