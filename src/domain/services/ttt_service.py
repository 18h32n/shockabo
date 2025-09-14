"""
Test-Time Training service for managing TTT model lifecycle.

Handles model loading, memory optimization, and GPU management for 1B parameter models
within 16GB GPU constraints.
"""
import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import psutil
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.infrastructure.config import get_config
from src.utils.memory_manager import MemoryManager, AdaptiveBatchSizer
from src.utils.auth_config import setup_hf_auth, get_model_access_info, suggest_public_model
from src.utils.advanced_memory_optimization import (
    MemoryOptimizationLevel,
    MemoryOptimizationConfig,
    apply_memory_optimizations
)
from src.utils.comprehensive_error_handling import (
    OutOfMemoryHandler,
    CheckpointManager,
    ModelLoadingHandler,
    resilient_operation,
    ErrorContext,
    ErrorRecoveryResult,
    ErrorSeverity,
    ErrorCategory,
    ErrorReporter
)
from src.utils.error_recovery import (
    RetryStrategy,
    FallbackStrategy,
    CircuitBreaker,
    get_circuit_breaker
)

logger = logging.getLogger(__name__)


class TTTModelService:
    """Service for managing TTT models with memory optimization."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize TTT service with configuration."""
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        
        # Setup device and paths
        self.device = self._setup_device()
        self.model_path = Path(self.config.get("model", {}).get("cache_dir", "data/models"))
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Memory management settings
        self.max_memory_gb = self.config.get("resources", {}).get("max_memory_gb", 10)
        self.memory_monitor_enabled = True
        
        # Mixed precision training settings
        self.mixed_precision = self.config.get("training", {}).get("mixed_precision", True)
        self.gradient_checkpointing = self.config.get("training", {}).get("gradient_checkpointing", True)
        self.selective_checkpointing = self.config.get("training", {}).get("selective_checkpointing", True)
        self.checkpointing_layers = self.config.get("training", {}).get("checkpointing_layers", 3)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision and self.device.type == "cuda" else None
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            memory_limit_gb=self.max_memory_gb,
            safety_margin=0.9,  # Use only 90% of limit for safety
            enable_monitoring=True
        )
        
        # Initialize comprehensive error handling components
        self.oom_handler = OutOfMemoryHandler(
            min_batch_size=1,
            memory_threshold_mb=self.max_memory_gb * 1024 * 0.85  # 85% of max memory
        )
        self.model_loading_handler = ModelLoadingHandler()
        self.error_reporter = ErrorReporter(
            log_file=str(self.model_path / "error_log.json")
        )
        
        # Initialize retry and recovery strategies
        self.retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=2.0,
            oom_handler=self.oom_handler,
            checkpoint_recovery=True
        )
        
        # Circuit breakers for critical operations
        self.model_loading_breaker = get_circuit_breaker("model_loading")
        self.training_breaker = get_circuit_breaker("training")
        
        # Fallback strategies for different operations
        self.model_loading_fallback = FallbackStrategy("model_loading")
        self.training_fallback = FallbackStrategy("training")
        
        # Setup error-specific fallbacks
        self._setup_error_fallbacks()
        
        # Track loading attempts and recovery statistics
        self.loading_attempts = 0
        self.recovery_stats = {
            "oom_recoveries": 0,
            "loading_failures": 0,
            "checkpoint_recoveries": 0,
            "precision_fallbacks": 0
        }

        logger.info(f"TTT Service initialized with device: {self.device} and comprehensive error handling")

    def _setup_device(self) -> torch.device:
        """Set up computing device based on availability and configuration."""
        device_config = self.config.get("model", {}).get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU available with {gpu_memory:.2f}GB memory")
                if gpu_memory >= 14:  # Need at least 14GB for 1B model with overhead
                    return torch.device("cuda")
                else:
                    logger.warning(f"GPU memory ({gpu_memory:.2f}GB) insufficient, using CPU")
                    return torch.device("cpu")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        return torch.device(device_config)

@resilient_operation(
        max_attempts=3,
        delay_seconds=3.0,
        handle_oom=True,
        handle_cuda_errors=True,
        handle_checkpoint_errors=True
    )
    def load_model(self, model_name: str | None = None, preferred_config: dict[str, Any] | None = None) -> tuple[Any, Any]:
        """
        Load 8B parameter model with comprehensive error handling, automatic fallback precision levels,
        and recovery mechanisms for out-of-memory and loading failures.
        
        Args:
            model_name: Model identifier (defaults to 8B config)
            preferred_config: Preferred model loading configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is not None and self.tokenizer is not None:
            logger.info("Model already loaded, returning existing instance")
            return self.model, self.tokenizer

        model_name = model_name or self.config.get("model", {}).get("name", "meta-llama/Llama-3-8B")
        self.loading_attempts += 1
        
        error_context = ErrorContext(
            operation="model_loading",
            model_name=model_name,
            attempt_number=self.loading_attempts,
            memory_usage_mb=self._get_memory_usage() * 1024
        )
        
        # Set up HuggingFace authentication
        auth_setup = setup_hf_auth()
        if auth_setup:
            logger.info("HuggingFace authentication configured")
        
        # Check model access
        access_info = get_model_access_info(model_name)
        if not access_info["can_access"]:
            logger.warning(f"Cannot access {model_name} - authentication required but not available")
            suggested_model = suggest_public_model(model_name)
            logger.info(f"Using public model alternative: {suggested_model}")
            model_name = suggested_model
        
        logger.info(f"Loading model: {model_name} (attempt {self.loading_attempts})")

        # Monitor initial memory
        initial_memory = self._get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.2f}GB")
        
        # Check memory pressure before loading
        pressure = self.memory_manager.check_memory_pressure()
        if pressure == "critical":
            logger.warning("Critical memory pressure detected before model loading")
            self.memory_manager.clear_cache()
            
            # Report memory pressure
            self.error_reporter.report_error(
                MemoryError("Critical memory pressure before model loading"),
                error_context,
                ErrorSeverity.HIGH,
                ErrorCategory.MEMORY
            )
        
        # Use fallback loading strategy for comprehensive error recovery
        async def load_with_fallbacks():
            # Create preferred configuration
            if preferred_config is None:
                preferred_config_dict = self._create_default_loading_config()
            else:
                preferred_config_dict = preferred_config
            
            # Use model loading handler with progressive fallbacks
            model, tokenizer, final_config = self.model_loading_handler.load_with_fallback(
                model_name, preferred_config_dict
            )
            
            if model is None:
                raise RuntimeError(f"All fallback strategies failed to load model {model_name}")
            
            return model, tokenizer, final_config
        
        # Execute with circuit breaker protection
        try:
            import asyncio
            loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
            
            if loop.is_running():
                task = loop.create_task(self.model_loading_breaker.call(load_with_fallbacks))
                model, tokenizer, final_config = task.result() if hasattr(task, 'result') else (None, None, {})
            else:
                asyncio.set_event_loop(loop)
                model, tokenizer, final_config = loop.run_until_complete(
                    self.model_loading_breaker.call(load_with_fallbacks)
                )
        except Exception as circuit_error:
            logger.error(f"Circuit breaker execution failed: {circuit_error}")
            # Fallback to direct loading
            preferred_config_dict = preferred_config or self._create_default_loading_config()
            model, tokenizer, final_config = self.model_loading_handler.load_with_fallback(
                model_name, preferred_config_dict
            )
        
        if model is None or tokenizer is None:
            self.recovery_stats["loading_failures"] += 1
            error = RuntimeError(f"Failed to load model {model_name} with all fallback strategies")
            self.error_reporter.report_error(
                error,
                error_context,
                ErrorSeverity.CRITICAL,
                ErrorCategory.MODEL_LOADING
            )
            raise error

        try:
            # Store successful loading configuration
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info(f"Model loaded successfully with configuration: {final_config}")
            
            # Log precision fallback if used
            if "torch_dtype" in final_config:
                dtype = final_config["torch_dtype"]
                if dtype != torch.float16:  # Default expectation
                    self.recovery_stats["precision_fallbacks"] += 1
                    logger.info(f"Using fallback precision: {dtype}")

            # Move to device if not already placed
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                if "device_map" not in final_config:
                    self.model = self.model.to(self.device)
                    logger.info(f"Model moved to device: {self.device}")

            # Apply comprehensive memory optimizations with error handling
            try:
                self._apply_memory_optimizations()
            except Exception as opt_error:
                logger.warning(f"Memory optimizations failed: {opt_error}")
                self.error_reporter.report_error(
                    opt_error,
                    ErrorContext(operation="memory_optimization", model_name=model_name),
                    ErrorSeverity.MEDIUM,
                    ErrorCategory.MEMORY
                )
            
            # Enable gradient checkpointing for memory efficiency during training
            if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
                    
                    # Apply selective checkpointing if enabled
                    if self.selective_checkpointing:
                        self._apply_selective_checkpointing()
                        logger.info(f"Selective gradient checkpointing: every {self.checkpointing_layers} layers")
                except Exception as checkpoint_error:
                    logger.warning(f"Gradient checkpointing setup failed: {checkpoint_error}")
                    self.error_reporter.report_error(
                        checkpoint_error,
                        ErrorContext(operation="gradient_checkpointing", model_name=model_name),
                        ErrorSeverity.MEDIUM,
                        ErrorCategory.TRAINING
                    )

            # Enable Flash Attention if configured and available
            if self.config.get("model", {}).get("use_flash_attention", False):
                try:
                    if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                        self.model.enable_xformers_memory_efficient_attention()
                        logger.info("Flash Attention (xFormers) enabled for 8B model")
                    elif hasattr(self.model, "_use_flash_attention_2"):
                        self.model._use_flash_attention_2 = True
                        logger.info("Flash Attention 2 enabled for 8B model")
                except Exception as attention_error:
                    logger.warning(f"Could not enable Flash Attention: {attention_error}")
                    self.error_reporter.report_error(
                        attention_error,
                        ErrorContext(operation="flash_attention", model_name=model_name),
                        ErrorSeverity.LOW,
                        ErrorCategory.HARDWARE
                    )
            
            # Check final memory usage
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            logger.info(f"8B model loaded. Memory increase: {memory_increase:.2f}GB")
            logger.info(f"Total memory usage: {final_memory:.2f}GB / {self.max_memory_gb}GB")

            # Validate memory usage for 8B model
            if final_memory > self.max_memory_gb:
                memory_error = MemoryError(f"8B model loading exceeded memory limit of {self.max_memory_gb}GB")
                logger.error(f"8B model memory usage ({final_memory:.2f}GB) exceeds limit ({self.max_memory_gb}GB)")
                
                # Report memory error
                self.error_reporter.report_error(
                    memory_error,
                    ErrorContext(
                        operation="memory_validation",
                        model_name=model_name,
                        memory_usage_mb=final_memory * 1024
                    ),
                    ErrorSeverity.CRITICAL,
                    ErrorCategory.MEMORY
                )
                
                self.cleanup()
                raise memory_error

            # Log mixed precision status
            if self.mixed_precision:
                logger.info(f"Mixed precision training enabled with scaler: {self.scaler is not None}")
            
            # Log successful loading with recovery stats
            if any(count > 0 for count in self.recovery_stats.values()):
                logger.info(f"Model loaded with recovery statistics: {self.recovery_stats}")
            
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            
            # Report loading failure
            self.error_reporter.report_error(
                e,
                error_context,
                ErrorSeverity.CRITICAL,
                ErrorCategory.MODEL_LOADING
            )
            
            self.cleanup()
            raise

    def optimize_memory(self) -> dict[str, float]:
        """
        Optimize memory usage by clearing caches and garbage collection.
        
        Returns:
            Dictionary with memory statistics before and after optimization
        """
        stats = {"before_mb": self._get_memory_usage() * 1024}

        # Clear Python garbage
        gc.collect()

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        stats["after_mb"] = self._get_memory_usage() * 1024
        stats["freed_mb"] = stats["before_mb"] - stats["after_mb"]

        logger.info(f"Memory optimization freed {stats['freed_mb']:.2f}MB")
        return stats

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if self.device.type == "cuda":
            # GPU memory
            return torch.cuda.memory_allocated() / 1024**3
        else:
            # System memory
            process = psutil.Process()
            return process.memory_info().rss / 1024**3

    def get_memory_profile(self) -> dict[str, Any]:
        """Get detailed memory profile for monitoring."""
        profile = {
            "device": str(self.device),
            "current_usage_gb": self._get_memory_usage(),
            "max_limit_gb": self.max_memory_gb,
            "usage_percentage": (self._get_memory_usage() / self.max_memory_gb) * 100,
        }

        if self.device.type == "cuda":
            profile.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            })
        else:
            memory = psutil.virtual_memory()
            profile.update({
                "system_total_gb": memory.total / 1024**3,
                "system_available_gb": memory.available / 1024**3,
                "system_used_percentage": memory.percent,
            })

        return profile

    def validate_gpu_constraints(self) -> bool:
        """
        Validate that GPU meets 24GB requirement for 8B model with QLoRA.
        
        Returns:
            True if GPU meets requirements, False otherwise
        """
        if not torch.cuda.is_available():
            logger.warning("No GPU available for 8B model")
            return False

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # QLoRA 8B model needs at least 12GB, recommended 16GB+
        minimum_required = 12  # Minimum for QLoRA 8B
        recommended = 16  # Recommended for QLoRA 8B
        
        meets_minimum = gpu_memory_gb >= minimum_required
        meets_recommended = gpu_memory_gb >= recommended

        if meets_recommended:
            logger.info(f"GPU memory: {gpu_memory_gb:.2f}GB - Excellent for 8B model with QLoRA")
        elif meets_minimum:
            logger.info(f"GPU memory: {gpu_memory_gb:.2f}GB - Adequate for 8B model with QLoRA (tight fit)")
        else:
            logger.warning(f"GPU memory: {gpu_memory_gb:.2f}GB - Insufficient for 8B model (need {minimum_required}GB+)")
            
        return meets_minimum

    def prepare_for_training(self) -> None:
        """Prepare model for training with memory optimizations."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set to training mode
        self.model.train()

        # Enable memory efficient attention if available
        if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
            try:
                self.model.enable_xformers_memory_efficient_attention()
                logger.info("Memory efficient attention enabled")
            except Exception:
                logger.warning("Could not enable memory efficient attention")

        # Optimize memory before training
        self.optimize_memory()

    def prepare_for_inference(self) -> None:
        """Prepare model for inference with comprehensive optimizations."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set to evaluation mode
        self.model.eval()

        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        # Apply inference-specific optimizations
        self._apply_inference_optimizations()

        # Optimize memory
        self.optimize_memory()
        
        logger.info("Model prepared for optimized inference")

    def _handle_model_oom(self) -> None:
        """Handle OOM during model loading with comprehensive recovery."""
        logger.error("OOM during model loading - attempting comprehensive recovery")
        
        self.recovery_stats["oom_recoveries"] += 1
        
        # Clear any partial model data
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force comprehensive memory cleanup
        self.memory_manager.clear_cache()
        
        # Additional cleanup steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Brief pause to allow memory to be freed
        time.sleep(1.0)
        
        # Log memory state
        memory_summary = self.memory_manager.get_memory_summary()
        logger.info(f"Post-recovery memory state: {memory_summary}")
        
        # Report OOM recovery
        self.error_reporter.report_error(
            MemoryError("OOM during model loading - recovery attempted"),
            ErrorContext(operation="oom_recovery"),
            ErrorSeverity.HIGH,
            ErrorCategory.MEMORY
        )
    
    def cleanup(self) -> None:
        """Clean up resources and free memory with comprehensive error handling cleanup."""
        logger.info("Cleaning up TTT service resources with comprehensive error handling")
        
        try:
            # Log final statistics before cleanup
            if hasattr(self, 'error_reporter'):
                error_summary = self.error_reporter.get_error_summary()
                if error_summary.get('total_errors', 0) > 0:
                    logger.info(f"Session error summary: {error_summary}")
            
            logger.info(f"Recovery statistics: {self.recovery_stats}")
            
            # Log circuit breaker stats
            if hasattr(self, 'model_loading_breaker'):
                logger.info(f"Model loading breaker stats: {self.model_loading_breaker.get_stats()}")
            if hasattr(self, 'training_breaker'):
                logger.info(f"Training breaker stats: {self.training_breaker.get_stats()}")
            
            # Log fallback strategy stats
            if hasattr(self, 'model_loading_fallback'):
                logger.info(f"Model loading fallback stats: {self.model_loading_fallback.get_stats()}")
            if hasattr(self, 'training_fallback'):
                logger.info(f"Training fallback stats: {self.training_fallback.get_stats()}")

            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection multiple times for thorough cleanup
            for _ in range(3):
                gc.collect()

            # Clear GPU cache thoroughly
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # Second pass
                
            # Final memory report
            final_memory = self._get_memory_usage()
            logger.info(f"Post-cleanup memory usage: {final_memory:.2f}GB")
        
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        logger.info("Comprehensive cleanup complete")
    
    def _create_default_loading_config(self) -> dict[str, Any]:
        """Create default model loading configuration with QLoRA settings."""
        config = {
            "cache_dir": self.model_path,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization if enabled and on CUDA
        if self.config.get("model", {}).get("quantization", True) and self.device.type == "cuda":
            if self.config.get("model", {}).get("load_in_4bit", True):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.get("model", {}).get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_compute_dtype=torch.float16 if self.config.get("model", {}).get("bnb_4bit_compute_dtype") == "float16" else torch.bfloat16,
                    bnb_4bit_use_double_quant=self.config.get("model", {}).get("bnb_4bit_use_double_quant", True),
                )
                config["quantization_config"] = quantization_config
                config["device_map"] = "auto"
            else:
                # Fallback to 8-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
                config["quantization_config"] = quantization_config
                config["device_map"] = "auto"
        elif self.device.type == "cuda":
            config["device_map"] = "auto"
        
        return config
    
    def _setup_error_fallbacks(self) -> None:
        """Setup error-specific fallback strategies for model operations."""
        # Model loading fallbacks
        def reduce_precision_fallback(*args, **kwargs):
            """Fallback with reduced precision."""
            model_name = args[0] if args else kwargs.get('model_name')
            config = self._create_default_loading_config()
            config["torch_dtype"] = torch.float32
            config["device_map"] = "cpu"
            return self.model_loading_handler.load_with_fallback(model_name, config)
        
        def cpu_only_fallback(*args, **kwargs):
            """Fallback to CPU-only loading."""
            model_name = args[0] if args else kwargs.get('model_name')
            config = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            return self.model_loading_handler.load_with_fallback(model_name, config)
        
        # Add OOM-specific fallbacks
        self.model_loading_fallback.add_oom_fallback(reduce_precision_fallback, priority=20)
        self.model_loading_fallback.add_fallback(cpu_only_fallback, priority=10)
        
        # Training fallbacks
        def reduce_batch_size_fallback(*args, **kwargs):
            """Reduce batch size for training."""
            if "batch_size" in kwargs:
                kwargs["batch_size"] = max(1, kwargs["batch_size"] // 2)
            return args, kwargs
        
        def disable_mixed_precision_fallback(*args, **kwargs):
            """Disable mixed precision training."""
            self.mixed_precision = False
            if self.scaler is not None:
                self.scaler = None
            return args, kwargs
        
        self.training_fallback.add_oom_fallback(reduce_batch_size_fallback, priority=20)
        self.training_fallback.add_fallback(disable_mixed_precision_fallback, priority=15)
    
    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics including error handling metrics."""
        base_stats = {
            "device": str(self.device),
            "memory_profile": self.get_memory_profile(),
            "inference_profile": self.get_inference_profile(),
            "mixed_precision_stats": self.get_mixed_precision_stats(),
            "loading_attempts": self.loading_attempts,
            "recovery_stats": self.recovery_stats,
        }
        
        # Add error handling statistics
        if hasattr(self, 'error_reporter'):
            base_stats["error_summary"] = self.error_reporter.get_error_summary()
        
        if hasattr(self, 'retry_strategy'):
            base_stats["retry_stats"] = self.retry_strategy.get_recovery_stats()
        
        # Circuit breaker statistics
        circuit_stats = {}
        if hasattr(self, 'model_loading_breaker'):
            circuit_stats["model_loading"] = self.model_loading_breaker.get_stats()
        if hasattr(self, 'training_breaker'):
            circuit_stats["training"] = self.training_breaker.get_stats()
        
        if circuit_stats:
            base_stats["circuit_breaker_stats"] = circuit_stats
        
        # Fallback strategy statistics
        fallback_stats = {}
        if hasattr(self, 'model_loading_fallback'):
            fallback_stats["model_loading"] = self.model_loading_fallback.get_stats()
        if hasattr(self, 'training_fallback'):
            fallback_stats["training"] = self.training_fallback.get_stats()
        
        if fallback_stats:
            base_stats["fallback_stats"] = fallback_stats
        
        # OOM handler statistics
        if hasattr(self, 'oom_handler'):
            base_stats["oom_stats"] = {
                "batch_size_history": self.oom_handler.batch_size_history,
                "cached_batch_sizes": self.oom_handler.batch_size_cache,
                "recommended_batch_size": self.oom_handler.get_recommended_batch_size(8)  # Base batch size 8
            }
        
        return base_stats

    def _apply_memory_optimizations(self) -> None:
        """Apply comprehensive memory optimizations to the loaded model."""
        if self.model is None:
            logger.warning("No model loaded for memory optimization")
            return
        
        try:
            # Determine optimization level based on available memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            
            if gpu_memory_gb >= 24:
                level = MemoryOptimizationLevel.BALANCED
            elif gpu_memory_gb >= 16:
                level = MemoryOptimizationLevel.AGGRESSIVE
            else:
                level = MemoryOptimizationLevel.CONSERVATIVE
            
            # Apply memory optimizations
            self.optimized_wrapper = apply_memory_optimizations(self.model, level)
            logger.info(f"Applied {level.value} memory optimizations")
            
        except Exception as e:
            logger.warning(f"Failed to apply memory optimizations: {e}")
    
    def _apply_selective_checkpointing(self) -> None:
        """Apply selective gradient checkpointing to model layers."""
        if self.model is None:
            return
        
        try:
            # Apply to transformer layers if available
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers
                checkpointed_layers = 0
                
                for i, layer in enumerate(layers):
                    if i % self.checkpointing_layers == 0:
                        # Enable checkpointing for this layer
                        if hasattr(layer, 'gradient_checkpointing'):
                            layer.gradient_checkpointing = True
                        checkpointed_layers += 1
                
                logger.info(f"Applied selective checkpointing to {checkpointed_layers}/{len(layers)} layers")
            
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # For GPT-style models
                layers = self.model.transformer.h
                checkpointed_layers = 0
                
                for i, layer in enumerate(layers):
                    if i % self.checkpointing_layers == 0:
                        if hasattr(layer, 'gradient_checkpointing'):
                            layer.gradient_checkpointing = True
                        checkpointed_layers += 1
                
                logger.info(f"Applied selective checkpointing to {checkpointed_layers}/{len(layers)} transformer layers")
            
        except Exception as e:
            logger.warning(f"Failed to apply selective checkpointing: {e}")
    
    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision training."""
        if self.mixed_precision and self.device.type == "cuda":
            with autocast():
                yield self.scaler
        else:
            yield None
    
    def scale_loss_and_backward(self, loss: torch.Tensor, optimizer: Optional[torch.optim.Optimizer] = None) -> torch.Tensor:
        """
        Scale loss for mixed precision and perform backward pass.
        
        Args:
            loss: Training loss tensor
            optimizer: Optional optimizer for gradient scaling
            
        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None and self.mixed_precision:
            # Scale loss to prevent gradient underflow
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            if optimizer is not None:
                # Unscale gradients and update
                self.scaler.unscale_(optimizer)
                
                # Clip gradients if needed
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Step optimizer and scaler
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
            return scaled_loss
        else:
            # Standard training without mixed precision
            loss.backward()
            
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            return loss
    
    def get_mixed_precision_stats(self) -> dict[str, Any]:
        """Get mixed precision training statistics."""
        stats = {
            "mixed_precision_enabled": self.mixed_precision,
            "scaler_enabled": self.scaler is not None,
            "gradient_checkpointing": self.gradient_checkpointing,
            "selective_checkpointing": self.selective_checkpointing,
            "checkpointing_layers": self.checkpointing_layers,
        }
        
        if self.scaler is not None:
            stats.update({
                "scaler_scale": self.scaler.get_scale(),
                "scaler_growth_factor": self.scaler.get_growth_factor(),
                "scaler_backoff_factor": self.scaler.get_backoff_factor(),
                "scaler_growth_interval": self.scaler.get_growth_interval(),
            })
        
        return stats
    
    def validate_training_stability(self, loss_history: list[float], window_size: int = 10) -> dict[str, Any]:
        """
        Validate training stability with mixed precision.
        
        Args:
            loss_history: List of recent loss values
            window_size: Window size for stability analysis
            
        Returns:
            Dictionary with stability metrics
        """
        if len(loss_history) < window_size:
            return {
                "stable": True,
                "reason": "Insufficient data for stability analysis",
                "loss_trend": "unknown",
                "volatility": 0.0
            }
        
        recent_losses = loss_history[-window_size:]
        
        # Check for NaN or infinite values
        if any(not torch.isfinite(torch.tensor(loss)) for loss in recent_losses):
            return {
                "stable": False,
                "reason": "NaN or infinite loss detected",
                "loss_trend": "unstable",
                "volatility": float('inf')
            }
        
        # Calculate loss volatility
        import statistics
        mean_loss = statistics.mean(recent_losses)
        volatility = statistics.stdev(recent_losses) / mean_loss if mean_loss > 0 else float('inf')
        
        # Determine trend
        if len(recent_losses) >= 3:
            first_third = statistics.mean(recent_losses[:len(recent_losses)//3])
            last_third = statistics.mean(recent_losses[-len(recent_losses)//3:])
            
            if last_third < first_third * 0.95:
                trend = "decreasing"
            elif last_third > first_third * 1.05:
                trend = "increasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Stability criteria
        is_stable = (
            volatility < 0.5 and  # Low volatility
            all(loss < mean_loss * 3 for loss in recent_losses) and  # No extreme spikes
            trend != "increasing"  # Not consistently increasing
        )
        
        return {
            "stable": is_stable,
            "reason": "Training appears stable" if is_stable else "High volatility or unstable trend detected",
            "loss_trend": trend,
            "volatility": volatility,
            "mean_loss": mean_loss,
            "recent_losses": recent_losses[-3:]  # Last 3 losses for debugging
        }
    
    def _apply_inference_optimizations(self) -> None:
        """Apply comprehensive inference optimizations to the model."""
        if self.model is None:
            logger.warning("No model available for inference optimizations")
            return
        
        logger.info("Applying inference optimizations...")
        
        try:
            # Apply torch.compile for inference speedup
            if self.config.get("inference", {}).get("enable_torch_compile", True):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Applied torch.compile for inference optimization")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # Enable KV cache optimizations
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                logger.info("KV cache enabled for inference")
            
            # Enable static cache if supported
            if hasattr(self.model, 'enable_static_cache'):
                try:
                    self.model.enable_static_cache()
                    logger.info("Static KV cache enabled")
                except Exception as e:
                    logger.warning(f"Static cache enabling failed: {e}")
            
            # Enable memory efficient attention for inference
            if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.model.enable_xformers_memory_efficient_attention()
                    logger.info("Memory efficient attention enabled for inference")
                except Exception as e:
                    logger.warning(f"Memory efficient attention failed: {e}")
            
            # Configure attention optimizations if available
            if hasattr(self.model.config, 'attention_softmax_in_fp32'):
                self.model.config.attention_softmax_in_fp32 = False  # Use native precision for speed
            
            logger.info("Inference optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply inference optimizations: {e}")
    
    def optimized_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Optimized text generation with inference monitoring.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        start_time = time.time()
        
        # Prepare generation arguments with optimized defaults
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # Enable KV caching
            **generation_kwargs
        }
        
        # Add optimized parameters for speed
        if not do_sample:
            # Deterministic generation is faster
            gen_kwargs.pop("temperature", None)
            gen_kwargs["num_beams"] = 1
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **gen_kwargs)
            
            generation_time = time.time() - start_time
            input_length = input_ids.shape[1]
            generated_tokens = outputs.shape[1] - input_length
            tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
            
            logger.debug(f"Generated {generated_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            return outputs
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Generation failed after {generation_time:.2f}s: {e}")
            raise
    
    def get_inference_profile(self) -> dict[str, Any]:
        """Get comprehensive inference performance profile."""
        profile = self.get_memory_profile()
        
        # Add inference-specific metrics
        profile.update({
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "mixed_precision_enabled": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "selective_checkpointing": self.selective_checkpointing,
            "memory_monitoring": self.memory_monitor_enabled,
        })
        
        # Add model-specific information if available
        if self.model is not None:
            try:
                profile.update({
                    "model_dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
                    "device": str(next(self.model.parameters()).device) if next(self.model.parameters(), None) is not None else "unknown",
                    "num_parameters": sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, 'parameters') else 0,
                })
                
                # Add config information
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    profile.update({
                        "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                        "hidden_size": getattr(config, 'hidden_size', 'unknown'),
                        "num_attention_heads": getattr(config, 'num_attention_heads', 'unknown'),
                        "num_hidden_layers": getattr(config, 'num_hidden_layers', 'unknown'),
                        "use_cache": getattr(config, 'use_cache', 'unknown'),
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to get model information: {e}")
        
        return profile
