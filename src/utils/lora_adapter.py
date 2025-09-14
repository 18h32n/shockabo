"""
LoRA (Low-Rank Adaptation) implementation for efficient model fine-tuning.

Implements LoRA layers for parameter-efficient fine-tuning with configurable rank,
alpha, and dropout parameters.
"""
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] = None
    bias: str = "none"  # "none", "all", "lora_only"

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules supporting both GPT-2 (Conv1D) and Llama (Linear)
            self.target_modules = [
                "c_attn", "c_proj",  # GPT-2 Conv1D modules
                "q_proj", "v_proj", "k_proj", "o_proj"  # Llama Linear modules
            ]

        # Scaling factor
        self.scaling = self.alpha / self.rank


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
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            base_output: Output from the base linear layer
            
        Returns:
            Adapted output tensor
        """
        if not self.merged:
            # Apply LoRA adaptation
            lora_output = x @ self.lora_A.T @ self.lora_B.T
            lora_output = self.dropout(lora_output)
            return base_output + lora_output * self.scaling
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
        Forward pass with LoRA adaptation for Conv1D.
        
        Args:
            x: Input tensor
            base_output: Output from the base Conv1D layer
            
        Returns:
            Adapted output tensor
        """
        if not self.merged:
            # Apply LoRA adaptation
            # For Conv1D: input shape (..., in_features), output shape (..., out_features)
            lora_output = torch.matmul(x, self.lora_A.T)  # (..., rank)
            lora_output = torch.matmul(lora_output, self.lora_B.T)  # (..., out_features)
            lora_output = self.dropout(lora_output)
            return base_output + lora_output * self.scaling
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
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> LoRAAdapter:
    """
    Apply LoRA adaptation to a model.
    
    Args:
        model: Model to adapt
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout rate
        target_modules: List of module names to target
        
    Returns:
        LoRAAdapter instance
    """
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    adapter = LoRAAdapter(model, config)
    adapter.print_trainable_parameters()

    return adapter
