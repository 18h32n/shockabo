"""
Test-Time Training service for managing TTT model lifecycle.

Handles model loading, memory optimization, and GPU management for 1B parameter models
within 16GB GPU constraints.
"""
import gc
import logging
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.infrastructure.config import get_config
from src.utils.memory_manager import MemoryManager, AdaptiveBatchSizer
from src.utils.auth_config import setup_hf_auth, get_model_access_info, suggest_public_model

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
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            memory_limit_gb=self.max_memory_gb,
            safety_margin=0.9,  # Use only 90% of limit for safety
            enable_monitoring=True
        )

        logger.info(f"TTT Service initialized with device: {self.device}")

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

    def load_model(self, model_name: str | None = None) -> tuple[Any, Any]:
        """
        Load 1B parameter model with quantization for memory efficiency.
        
        Args:
            model_name: Model identifier (defaults to config)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer

        model_name = model_name or self.config.get("model", {}).get("name", "meta-llama/Llama-3.2-1B")
        
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
        
        logger.info(f"Loading model: {model_name}")

        # Monitor initial memory
        initial_memory = self._get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.2f}GB")
        
        # Check memory pressure before loading
        pressure = self.memory_manager.check_memory_pressure()
        if pressure == "critical":
            logger.warning("Critical memory pressure detected before model loading")
            self.memory_manager.clear_cache()

        try:
            # Load tokenizer first (lightweight)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_path,
                trust_remote_code=True,
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization for memory efficiency
            quantization_config = None
            if self.config.get("model", {}).get("quantization", True) and self.device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
                logger.info("8-bit quantization enabled for memory efficiency")

            # Model loading arguments
            model_kwargs = {
                "cache_dir": self.model_path,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif self.device.type == "cuda":
                model_kwargs["device_map"] = "auto"

            # Load model with memory monitoring and OOM protection
            with self.memory_manager.oom_protected(fallback_fn=self._handle_model_oom):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )

            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)

            # Enable gradient checkpointing for memory efficiency during training
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

            # Check final memory usage
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            logger.info(f"Model loaded. Memory increase: {memory_increase:.2f}GB")
            logger.info(f"Total memory usage: {final_memory:.2f}GB / {self.max_memory_gb}GB")

            # Validate memory usage
            if final_memory > self.max_memory_gb:
                logger.error(f"Memory usage ({final_memory:.2f}GB) exceeds limit ({self.max_memory_gb}GB)")
                self.cleanup()
                raise MemoryError(f"Model loading exceeded memory limit of {self.max_memory_gb}GB")

            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
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
        Validate that GPU meets 16GB requirement for 1B model.
        
        Returns:
            True if GPU meets requirements, False otherwise
        """
        if not torch.cuda.is_available():
            logger.warning("No GPU available")
            return False

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        meets_requirement = gpu_memory_gb >= 14  # 14GB minimum for 1B model with overhead

        logger.info(f"GPU memory: {gpu_memory_gb:.2f}GB - {'Meets' if meets_requirement else 'Does not meet'} 16GB requirement")
        return meets_requirement

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
        """Prepare model for inference with optimizations."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set to evaluation mode
        self.model.eval()

        # Disable gradient computation for inference
        torch.set_grad_enabled(False)

        # Optimize memory
        self.optimize_memory()

    def _handle_model_oom(self) -> None:
        """Handle OOM during model loading."""
        logger.error("OOM during model loading - attempting recovery")
        
        # Clear any partial model data
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        # Force memory cleanup
        self.memory_manager.clear_cache()
        
        # Log memory state
        logger.info(self.memory_manager.get_memory_summary())
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        logger.info("Cleaning up TTT service resources")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Cleanup complete")
