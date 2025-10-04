"""
Quantization utilities for memory-efficient model loading with QLoRA support.

This module provides utilities for configuring and validating 4-bit quantization
using BitsAndBytes, specifically optimized for Llama-3 8B model with QLoRA.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization with BitsAndBytes."""

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str | None = None

    # Additional optimization settings
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    llm_int8_skip_modules: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for BitsAndBytesConfig."""
        config = {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "llm_int8_threshold": self.llm_int8_threshold,
            "llm_int8_has_fp16_weight": self.llm_int8_has_fp16_weight,
        }

        # Handle compute dtype
        if self.bnb_4bit_compute_dtype:
            try:
                import torch
                dtype_mapping = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                config["bnb_4bit_compute_dtype"] = dtype_mapping.get(
                    self.bnb_4bit_compute_dtype, torch.float16
                )
            except ImportError:
                logger.warning("PyTorch not available, skipping dtype conversion")

        # Optional parameters
        if self.bnb_4bit_quant_storage:
            config["bnb_4bit_quant_storage"] = self.bnb_4bit_quant_storage

        if self.llm_int8_skip_modules:
            config["llm_int8_skip_modules"] = self.llm_int8_skip_modules

        return config


def create_bnb_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
) -> Any:
    """
    Create BitsAndBytesConfig for quantized model loading.

    Args:
        load_in_4bit: Whether to use 4-bit quantization
        quant_type: Quantization type ("nf4" or "fp4")
        compute_dtype: Compute dtype ("float16", "bfloat16", "float32")
        use_double_quant: Whether to use double quantization

    Returns:
        BitsAndBytesConfig instance or None if library not available
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig

        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        compute_dtype_torch = dtype_mapping.get(compute_dtype, torch.float16)

        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype_torch,
            bnb_4bit_use_double_quant=use_double_quant,
        )

    except ImportError as e:
        logger.error(f"BitsAndBytes not available: {e}")
        return None


def validate_quantization_support() -> dict[str, Any]:
    """
    Validate that quantization libraries are properly installed.

    Returns:
        dict: Validation results with library availability and versions
    """
    validation = {
        "torch_available": False,
        "transformers_available": False,
        "bitsandbytes_available": False,
        "accelerate_available": False,
        "cuda_available": False,
        "versions": {},
        "errors": [],
    }

    # Check PyTorch
    try:
        import torch
        validation["torch_available"] = True
        validation["versions"]["torch"] = torch.__version__
        validation["cuda_available"] = torch.cuda.is_available()
    except ImportError as e:
        validation["errors"].append(f"PyTorch not available: {e}")

    # Check Transformers
    try:
        import transformers
        validation["transformers_available"] = True
        validation["versions"]["transformers"] = transformers.__version__

        # Check if BitsAndBytesConfig is available
        from transformers import BitsAndBytesConfig  # noqa: F401
        validation["bnb_config_available"] = True
    except ImportError as e:
        validation["errors"].append(f"Transformers or BitsAndBytesConfig not available: {e}")
        validation["bnb_config_available"] = False

    # Check BitsAndBytes
    try:
        import bitsandbytes as bnb
        validation["bitsandbytes_available"] = True
        validation["versions"]["bitsandbytes"] = bnb.__version__
    except ImportError as e:
        validation["errors"].append(f"BitsAndBytes not available: {e}")

    # Check Accelerate
    try:
        import accelerate
        validation["accelerate_available"] = True
        validation["versions"]["accelerate"] = accelerate.__version__
    except ImportError as e:
        validation["errors"].append(f"Accelerate not available: {e}")

    return validation


def estimate_quantized_model_memory(
    model_params: int,
    quantization_bits: int = 4,
    overhead_factor: float = 1.2,
) -> dict[str, float]:
    """
    Estimate memory usage for quantized model.

    Args:
        model_params: Number of model parameters
        quantization_bits: Bits per parameter (4 for NF4, 8 for int8)
        overhead_factor: Memory overhead factor for activations, etc.

    Returns:
        dict: Memory usage estimates in MB and GB
    """
    # Base model memory (quantized)
    base_memory_bits = model_params * quantization_bits
    base_memory_mb = base_memory_bits / (8 * 1024 * 1024)

    # Apply overhead for activations, KV cache, etc.
    total_memory_mb = base_memory_mb * overhead_factor

    # Compare with FP16 baseline
    fp16_memory_mb = (model_params * 16) / (8 * 1024 * 1024)
    memory_saved_mb = fp16_memory_mb - base_memory_mb

    return {
        "base_model_mb": base_memory_mb,
        "total_estimated_mb": total_memory_mb,
        "total_estimated_gb": total_memory_mb / 1024,
        "fp16_baseline_mb": fp16_memory_mb,
        "memory_saved_mb": memory_saved_mb,
        "memory_saved_gb": memory_saved_mb / 1024,
        "compression_ratio": fp16_memory_mb / base_memory_mb if base_memory_mb > 0 else 1,
        "quantization_bits": quantization_bits,
        "overhead_factor": overhead_factor,
    }


def get_optimal_quantization_config(
    gpu_memory_mb: float,
    model_size: str = "8B",
    target_memory_usage: float = 0.8,  # Use 80% of available memory
) -> QuantizationConfig:
    """
    Get optimal quantization configuration based on available GPU memory.

    Args:
        gpu_memory_mb: Available GPU memory in MB
        model_size: Model size ("1B", "3B", "8B", "13B", etc.)
        target_memory_usage: Target memory usage as fraction of available

    Returns:
        QuantizationConfig: Optimized configuration
    """
    available_memory = gpu_memory_mb * target_memory_usage

    # Model parameter counts (approximate)
    model_params = {
        "1B": 1_000_000_000,
        "3B": 3_000_000_000,
        "8B": 8_000_000_000,
        "13B": 13_000_000_000,
        "30B": 30_000_000_000,
        "65B": 65_000_000_000,
    }

    params = model_params.get(model_size, 8_000_000_000)

    # Estimate memory for different configurations
    nf4_estimate = estimate_quantized_model_memory(params, 4)
    int8_estimate = estimate_quantized_model_memory(params, 8)

    # Choose configuration based on available memory
    if available_memory >= nf4_estimate["total_estimated_mb"]:
        # Optimal: 4-bit NF4 quantization
        config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using optimal NF4 quantization (estimated: {nf4_estimate['total_estimated_mb']:.1f} MB)")

    elif available_memory >= int8_estimate["total_estimated_mb"]:
        # Fallback: 8-bit quantization
        config = QuantizationConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        logger.info(f"Using 8-bit quantization (estimated: {int8_estimate['total_estimated_mb']:.1f} MB)")

    else:
        # Conservative: 4-bit with maximum compression
        config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",  # More aggressive compression
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        logger.warning(
            f"Memory constrained ({available_memory:.1f} MB available). "
            f"Using aggressive FP4 quantization."
        )

    return config


def validate_model_loading_requirements(
    model_name: str,
    quantization_config: QuantizationConfig,
    available_gpu_memory_mb: float,
) -> dict[str, Any]:
    """
    Validate that model can be loaded with given quantization configuration.

    Args:
        model_name: Model identifier
        quantization_config: Quantization configuration
        available_gpu_memory_mb: Available GPU memory in MB

    Returns:
        dict: Validation results and recommendations
    """
    # Estimate model size based on name
    model_size_mapping = {
        "1B": 1_000_000_000,
        "3B": 3_000_000_000,
        "8B": 8_000_000_000,
    }

    # Extract size from model name
    model_size = "8B"  # Default
    for size in model_size_mapping:
        if size in model_name:
            model_size = size
            break

    params = model_size_mapping.get(model_size, 8_000_000_000)

    # Estimate memory usage
    if quantization_config.load_in_4bit:
        bits = 4
    elif quantization_config.load_in_8bit:
        bits = 8
    else:
        bits = 16  # FP16

    memory_estimate = estimate_quantized_model_memory(params, bits)

    validation = {
        "model_name": model_name,
        "estimated_size": model_size,
        "estimated_params": params,
        "quantization_bits": bits,
        "estimated_memory_mb": memory_estimate["total_estimated_mb"],
        "available_memory_mb": available_gpu_memory_mb,
        "memory_sufficient": available_gpu_memory_mb >= memory_estimate["total_estimated_mb"],
        "memory_utilization": memory_estimate["total_estimated_mb"] / available_gpu_memory_mb if available_gpu_memory_mb > 0 else float('inf'),
        "recommendations": [],
    }

    # Generate recommendations
    if validation["memory_sufficient"]:
        if validation["memory_utilization"] > 0.9:
            validation["recommendations"].append("Memory usage is high (>90%). Consider reducing batch size.")
        elif validation["memory_utilization"] > 0.8:
            validation["recommendations"].append("Memory usage is moderate (>80%). Enable gradient checkpointing.")
        else:
            validation["recommendations"].append("Memory usage is optimal (<80%).")
    else:
        validation["recommendations"].extend([
            "Insufficient GPU memory for this configuration",
            "Try more aggressive quantization (FP4 instead of NF4)",
            "Use gradient checkpointing and smaller batch size",
            "Consider using a smaller model variant",
        ])

    return validation


# Predefined configurations for common models
LLAMA_8B_CONFIGS = {
    "optimal": QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    ),
    "memory_efficient": QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    ),
    "fallback_8bit": QuantizationConfig(
        load_in_4bit=False,
        load_in_8bit=True,
        bnb_4bit_compute_dtype="float16",
    ),
}
