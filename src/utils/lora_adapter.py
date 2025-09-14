"""
LoRA (Low-Rank Adaptation) implementation for efficient model fine-tuning.

Implements LoRA layers for parameter-efficient fine-tuning with configurable rank,
alpha, and dropout parameters.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.pytorch_utils import Conv1D

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation with QLoRA support."""

    rank: int = 64  # Increased for 8B model
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] = None
    bias: str = "none"  # "none", "all", "lora_only"
    
    # QLoRA quantization parameters
    use_quantization: bool = True
    quantization_config: dict = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    
    # Gradient checkpointing for LoRA layers
    gradient_checkpointing: bool = True
    selective_checkpointing: bool = True
    checkpointing_layers: int = 3  # Checkpoint every N LoRA layers

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Llama-3 8B model (optimized for QLoRA)
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention projections
                "gate_proj", "up_proj", "down_proj",     # MLP projections
                "c_attn", "c_proj",  # GPT-2 Conv1D modules (fallback)
            ]

        # Initialize quantization configuration
        if self.quantization_config is None and self.use_quantization:
            self.quantization_config = {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            }

        # Scaling factor
        self.scaling = self.alpha / self.rank


class CheckpointedLoRALayer(nn.Module):
    """Wrapper for LoRA layers with gradient checkpointing support."""
    
    def __init__(self, lora_layer: nn.Module, enable_checkpointing: bool = True):
        """
        Initialize checkpointed LoRA layer.
        
        Args:
            lora_layer: Base LoRA layer to wrap
            enable_checkpointing: Whether to enable gradient checkpointing
        """
        super().__init__()
        self.lora_layer = lora_layer
        self.enable_checkpointing = enable_checkpointing
        self._original_forward = lora_layer.forward
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.enable_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            return checkpoint(self._original_forward, x)
        else:
            # Standard forward pass
            return self._original_forward(x)
    
    def enable_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self.enable_checkpointing = True
        
    def disable_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.enable_checkpointing = False


class LoRALinear(nn.Module):
    """LoRA adapter layer for linear transformations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # Create LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=torch.nn.init.calculate_gain('leaky_relu'))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation (QLoRA compatible).
        
        Args:
            x: Input tensor
            base_output: Output from the base linear layer
            
        Returns:
            Adapted output tensor
        """
        if not self.merged:
            # Apply LoRA adaptation with proper dtype handling for quantization
            original_dtype = x.dtype
            
            # Ensure computation is done in appropriate precision
            if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
                x_compute = x.float()
            else:
                x_compute = x
            
            # LoRA forward pass: x @ A^T @ B^T
            lora_output = torch.matmul(x_compute, self.lora_A.T)
            lora_output = torch.matmul(lora_output, self.lora_B.T)
            lora_output = self.dropout(lora_output)
            
            # Scale and convert back to original dtype
            lora_output = (lora_output * self.scaling).to(original_dtype)
            
            return base_output + lora_output
        else:
            return base_output

    def merge(self) -> None:
        """Merge LoRA weights into the base layer (for inference)."""
        if not self.merged and self.merge_weights:
            self.merged = True
            logger.info("Merged LoRA weights")

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the base layer."""
        if self.merged and self.merge_weights:
            self.merged = False
            logger.info("Unmerged LoRA weights")


class LoRAConv1D(nn.Module):
    """LoRA adapter layer for Conv1D transformations (used in GPT-2 models)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # Create LoRA A and B matrices for Conv1D
        # Conv1D weight shape is (out_features, in_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=torch.nn.init.calculate_gain('leaky_relu'))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation for Conv1D (QLoRA compatible).
        
        Args:
            x: Input tensor
            base_output: Output from the base Conv1D layer
            
        Returns:
            Adapted output tensor
        """
        if not self.merged:
            # Apply LoRA adaptation with proper dtype handling
            original_dtype = x.dtype
            
            # Ensure computation is done in appropriate precision
            if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
                x_compute = x.float()
            else:
                x_compute = x
            
            # For Conv1D: input shape (..., in_features), output shape (..., out_features)
            lora_output = torch.matmul(x_compute, self.lora_A.T)  # (..., rank)
            lora_output = torch.matmul(lora_output, self.lora_B.T)  # (..., out_features)
            lora_output = self.dropout(lora_output)
            
            # Scale and convert back to original dtype
            lora_output = (lora_output * self.scaling).to(original_dtype)
            
            return base_output + lora_output
        else:
            return base_output

    def merge(self) -> None:
        """Merge LoRA weights into the base layer (for inference)."""
        if not self.merged and self.merge_weights:
            self.merged = True
            logger.info("Merged LoRA Conv1D weights")

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the base layer."""
        if self.merged and self.merge_weights:
            self.merged = False
            logger.info("Unmerged LoRA Conv1D weights")


class LoRAAdapter:
    """Main LoRA adapter for model adaptation."""

    def __init__(self, model: nn.Module, config: LoRAConfig):
        """
        Initialize LoRA adapter.
        
        Args:
            model: Base model to adapt
            config: LoRA configuration
        """
        self.model = model
        self.config = config
        self.lora_layers: dict[str, LoRALinear] = {}

        # Track original layers for restoration
        self.original_layers: dict[str, nn.Module] = {}
        
        # Track checkpointed LoRA layers
        self.checkpointed_layers: dict[str, CheckpointedLoRALayer] = {}

        # Apply LoRA to target modules
        self._apply_lora()

    def _apply_lora(self) -> None:
        """Apply LoRA adaptation to target modules."""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Store original module
                    self.original_layers[name] = module

                    # Create LoRA adapter for Linear layer
                    lora_layer = LoRALinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    )

                    self.lora_layers[name] = lora_layer

                    # Create wrapper module
                    self._replace_linear_module(name, module, lora_layer)

                    logger.info(f"Applied LoRA to Linear {name} (in={module.in_features}, out={module.out_features})")
                
                elif isinstance(module, Conv1D):
                    # Store original module
                    self.original_layers[name] = module

                    # Create LoRA adapter for Conv1D layer
                    # Conv1D has nf (out_features) and nx (in_features) attributes
                    lora_layer = LoRAConv1D(
                        in_features=module.nx,
                        out_features=module.nf,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    )

                    self.lora_layers[name] = lora_layer

                    # Create wrapper module
                    self._replace_conv1d_module(name, module, lora_layer)

                    logger.info(f"Applied LoRA to Conv1D {name} (in={module.nx}, out={module.nf})")

    def _replace_linear_module(self, module_name: str, original: nn.Linear, lora: LoRALinear) -> None:
        """Replace a Linear module with LoRA-adapted version."""
        # Create a wrapper that combines original and LoRA
        class LoRALinearWrapper(nn.Module):
            def __init__(self, base_layer: nn.Linear, lora_layer: LoRALinear):
                super().__init__()
                self.base_layer = base_layer
                self.lora_layer = lora_layer

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                base_output = self.base_layer(x)
                return self.lora_layer(x, base_output)

        # Navigate to parent module and replace
        self._replace_module_in_model(module_name, LoRALinearWrapper(original, lora))
    
    def _replace_conv1d_module(self, module_name: str, original: Conv1D, lora: LoRAConv1D) -> None:
        """Replace a Conv1D module with LoRA-adapted version."""
        # Create a wrapper that combines original and LoRA
        class LoRAConv1DWrapper(nn.Module):
            def __init__(self, base_layer: Conv1D, lora_layer: LoRAConv1D):
                super().__init__()
                self.base_layer = base_layer
                self.lora_layer = lora_layer

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                base_output = self.base_layer(x)
                return self.lora_layer(x, base_output)

        # Navigate to parent module and replace
        self._replace_module_in_model(module_name, LoRAConv1DWrapper(original, lora))
    
    def _replace_module_in_model(self, module_name: str, new_module: nn.Module) -> None:
        """Helper to replace a module in the model."""
        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]

        if parent_name:
            parent = self.model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            setattr(parent, child_name, new_module)
        else:
            setattr(self.model, child_name, new_module)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def enable_selective_checkpointing(self, layers_to_checkpoint: int = 3) -> int:
        """
        Enable selective gradient checkpointing on LoRA layers.
        
        Args:
            layers_to_checkpoint: Checkpoint every N LoRA layers
            
        Returns:
            Number of layers checkpointed
        """
        if not self.config.selective_checkpointing:
            logger.info("Selective checkpointing disabled in config")
            return 0
        
        checkpointed_count = 0
        lora_layer_names = list(self.lora_layers.keys())
        
        for i, layer_name in enumerate(lora_layer_names):
            if i % layers_to_checkpoint == 0:
                lora_layer = self.lora_layers[layer_name]
                
                # Wrap with checkpointed layer
                checkpointed_layer = CheckpointedLoRALayer(
                    lora_layer, 
                    enable_checkpointing=True
                )
                
                self.checkpointed_layers[layer_name] = checkpointed_layer
                
                # Replace the layer in the model
                parent_module, attr_name = self._get_parent_module_and_attr(layer_name)
                if parent_module is not None:
                    setattr(parent_module, attr_name, checkpointed_layer)
                    checkpointed_count += 1
        
        logger.info(f"Enabled selective checkpointing on {checkpointed_count}/{len(lora_layer_names)} LoRA layers")
        return checkpointed_count
    
    def disable_checkpointing(self) -> int:
        """
        Disable gradient checkpointing on all LoRA layers.
        
        Returns:
            Number of layers restored
        """
        restored_count = 0
        
        for layer_name, checkpointed_layer in self.checkpointed_layers.items():
            # Restore original LoRA layer
            original_lora_layer = checkpointed_layer.lora_layer
            
            parent_module, attr_name = self._get_parent_module_and_attr(layer_name)
            if parent_module is not None:
                setattr(parent_module, attr_name, original_lora_layer)
                restored_count += 1
        
        self.checkpointed_layers.clear()
        logger.info(f"Disabled checkpointing on {restored_count} LoRA layers")
        return restored_count
    
    def get_checkpointing_stats(self) -> Dict[str, Any]:
        """Get statistics about gradient checkpointing usage."""
        return {
            "total_lora_layers": len(self.lora_layers),
            "checkpointed_layers": len(self.checkpointed_layers),
            "checkpointing_enabled": len(self.checkpointed_layers) > 0,
            "checkpointing_ratio": len(self.checkpointed_layers) / max(1, len(self.lora_layers)),
            "selective_checkpointing_config": self.config.selective_checkpointing,
            "checkpointing_layers_config": self.config.checkpointing_layers,
        }
    
    def _get_parent_module_and_attr(self, layer_name: str) -> tuple[Optional[nn.Module], Optional[str]]:
        """Get parent module and attribute name for a layer path."""
        try:
            parts = layer_name.split('.')
            parent = self.model
            
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            attr_name = parts[-1]
            return parent, attr_name
            
        except (AttributeError, IndexError):
            logger.warning(f"Could not find parent for layer: {layer_name}")
            return None, None

    def get_parameter_count(self) -> dict[str, int]:
        """Get parameter counts for base model and LoRA."""
        base_params = sum(p.numel() for p in self.model.parameters())
        lora_params = sum(p.numel() for p in self.get_trainable_parameters())

        return {
            "base_parameters": base_params,
            "lora_parameters": lora_params,
            "trainable_parameters": lora_params,
            "reduction_ratio": 1 - (lora_params / base_params) if base_params > 0 else 0,
            "memory_saved_mb": (base_params - lora_params) * 4 / 1024 / 1024 if base_params > 0 else 0,  # Assuming float32
        }

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights."""
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = lora_layer.lora_A
            state_dict[f"{name}.lora_B"] = lora_layer.lora_B

        # Add configuration
        state_dict["config"] = {
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "target_modules": self.config.target_modules,
        }

        torch.save(state_dict, path)
        logger.info(f"Saved LoRA adapter to {path}")

    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights."""
        state_dict = torch.load(path, map_location="cpu")

        # Load configuration
        if "config" in state_dict:
            config_dict = state_dict.pop("config")
            # Validate configuration matches
            if config_dict["rank"] != self.config.rank or config_dict["alpha"] != self.config.alpha:
                logger.warning("Loaded LoRA config differs from current config")

        # Load weights
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A" in state_dict:
                lora_layer.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                lora_layer.lora_B.data = state_dict[f"{name}.lora_B"]

        logger.info(f"Loaded LoRA adapter from {path}")

    def merge_weights(self) -> None:
        """Merge LoRA weights with base model for inference."""
        for lora_layer in self.lora_layers.values():
            lora_layer.merge()

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base model."""
        for lora_layer in self.lora_layers.values():
            lora_layer.unmerge()

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable parameters."""
        stats = self.get_parameter_count()
        logger.info("=" * 50)
        logger.info("LoRA Adapter Statistics:")
        logger.info(f"Base model parameters: {stats['base_parameters']:,}")
        logger.info(f"LoRA parameters: {stats['lora_parameters']:,}")
        logger.info(f"Trainable parameters: {stats['trainable_parameters']:,}")
        logger.info(f"Parameter reduction: {stats['reduction_ratio']:.1%}")
        logger.info(f"Memory saved: {stats['memory_saved_mb']:.2f} MB")
        logger.info("=" * 50)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 64,  # Increased for 8B model
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
    use_quantization: bool = True,
) -> LoRAAdapter:
    """
    Apply LoRA adaptation to a model with QLoRA support.
    
    Args:
        model: Model to adapt
        rank: LoRA rank (default 64 for 8B model)
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout rate
        target_modules: List of module names to target
        use_quantization: Whether to use quantization-aware LoRA
        
    Returns:
        LoRAAdapter instance
    """
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        use_quantization=use_quantization,
    )

    adapter = LoRAAdapter(model, config)
    adapter.print_trainable_parameters()

    return adapter


def create_qlora_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True,
) -> dict:
    """
    Create QLoRA quantization configuration.
    
    Args:
        load_in_4bit: Whether to load model in 4-bit
        bnb_4bit_quant_type: Quantization type ("nf4", "fp4")
        bnb_4bit_compute_dtype: Compute dtype for quantized layers
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        dict: Quantization configuration for BitsAndBytes
    """
    try:
        import torch
        
        # Map string dtype to torch dtype
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        compute_dtype = dtype_mapping.get(bnb_4bit_compute_dtype, torch.float16)
        
        return {
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
        }
    except ImportError:
        logger.warning("PyTorch not available, returning basic config")
        return {
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
        }


def estimate_qlora_memory_usage(
    base_model_params: int,
    lora_rank: int = 64,
    num_lora_modules: int = 32,  # Typical for Llama-3 8B
    quantization_bits: int = 4,
) -> dict[str, float]:
    """
    Estimate memory usage for QLoRA configuration.
    
    Args:
        base_model_params: Number of parameters in base model
        lora_rank: LoRA rank
        num_lora_modules: Number of modules with LoRA adapters
        quantization_bits: Quantization bits (4 for NF4)
        
    Returns:
        dict: Memory usage estimates in MB
    """
    # Base model memory with quantization (bits per parameter)
    base_model_mb = (base_model_params * quantization_bits / 8) / (1024 * 1024)
    
    # LoRA parameters: each adapter has rank * (in_dim + out_dim) parameters
    # Approximating in_dim ≈ out_dim ≈ 4096 for Llama-3 8B
    avg_dim = 4096
    lora_params_per_module = lora_rank * avg_dim * 2  # A and B matrices
    total_lora_params = lora_params_per_module * num_lora_modules
    lora_mb = (total_lora_params * 32 / 8) / (1024 * 1024)  # Float32 for LoRA
    
    # Additional overhead (activations, gradients, optimizer states)
    overhead_mb = (base_model_mb + lora_mb) * 0.2  # 20% overhead
    
    total_mb = base_model_mb + lora_mb + overhead_mb
    
    return {
        "base_model_mb": base_model_mb,
        "lora_adapters_mb": lora_mb,
        "overhead_mb": overhead_mb,
        "total_estimated_mb": total_mb,
        "total_estimated_gb": total_mb / 1024,
        "quantization_savings_mb": (base_model_params * 16 / 8) / (1024 * 1024) - base_model_mb,
    }
