"""Unit tests for LoRA adapter."""
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.utils.lora_adapter import LoRAAdapter, LoRAConfig, LoRALinear, apply_lora_to_model


class TestLoRAConfig:
    """Test suite for LoRA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoRAConfig()

        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert config.bias == "none"
        assert config.scaling == 2.0  # alpha / rank = 16 / 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = LoRAConfig(
            rank=4,
            alpha=8,
            dropout=0.2,
            target_modules=["linear1", "linear2"],
            bias="all"
        )

        assert config.rank == 4
        assert config.alpha == 8
        assert config.dropout == 0.2
        assert config.target_modules == ["linear1", "linear2"]
        assert config.bias == "all"
        assert config.scaling == 2.0  # alpha / rank = 8 / 4


class TestLoRALinear:
    """Test suite for LoRA linear layer."""

    def test_init(self):
        """Test LoRA linear layer initialization."""
        lora = LoRALinear(
            in_features=768,
            out_features=768,
            rank=8,
            alpha=16,
            dropout=0.1
        )

        assert lora.in_features == 768
        assert lora.out_features == 768
        assert lora.rank == 8
        assert lora.alpha == 16
        assert lora.scaling == 2.0
        assert not lora.merged

        # Check parameter shapes
        assert lora.lora_A.shape == (8, 768)
        assert lora.lora_B.shape == (768, 8)

    def test_forward_unmerged(self):
        """Test forward pass without merged weights."""
        batch_size = 2
        in_features = 10
        out_features = 5
        rank = 4

        lora = LoRALinear(in_features, out_features, rank=rank)

        # Create test inputs
        x = torch.randn(batch_size, in_features)
        base_output = torch.randn(batch_size, out_features)

        # Forward pass
        output = lora(x, base_output)

        # Check output shape
        assert output.shape == (batch_size, out_features)

        # Output should be different from base (due to LoRA addition)
        assert not torch.allclose(output, base_output)

    def test_forward_merged(self):
        """Test forward pass with merged weights."""
        lora = LoRALinear(10, 5, rank=4, merge_weights=True)
        lora.merge()

        x = torch.randn(2, 10)
        base_output = torch.randn(2, 5)

        output = lora(x, base_output)

        # When merged, should return base output unchanged
        assert torch.equal(output, base_output)

    def test_merge_unmerge(self):
        """Test merging and unmerging functionality."""
        lora = LoRALinear(10, 5, rank=4, merge_weights=True)

        assert not lora.merged

        lora.merge()
        assert lora.merged

        lora.unmerge()
        assert not lora.merged


class TestLoRAAdapter:
    """Test suite for LoRA adapter."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.q_proj = nn.Linear(20, 20)
                self.v_proj = nn.Linear(20, 20)
                self.linear2 = nn.Linear(20, 10)

            def forward(self, x):
                x = self.linear1(x)
                q = self.q_proj(x)
                v = self.v_proj(x)
                x = q + v
                return self.linear2(x)

        return SimpleModel()

    def test_init(self, simple_model):
        """Test LoRA adapter initialization."""
        config = LoRAConfig(rank=4, alpha=8)
        adapter = LoRAAdapter(simple_model, config)

        assert adapter.model == simple_model
        assert adapter.config == config

        # Check that LoRA was applied to target modules
        assert "q_proj" in adapter.lora_layers
        assert "v_proj" in adapter.lora_layers
        assert "linear1" not in adapter.lora_layers  # Not in target modules

    def test_get_trainable_parameters(self, simple_model):
        """Test getting trainable parameters."""
        config = LoRAConfig(rank=4)
        adapter = LoRAAdapter(simple_model, config)

        params = adapter.get_trainable_parameters()

        # Should have 2 parameters (A and B) for each LoRA layer
        assert len(params) == 4  # 2 target modules × 2 parameters each

        # All should be parameters
        assert all(isinstance(p, nn.Parameter) for p in params)

    def test_get_parameter_count(self, simple_model):
        """Test parameter counting."""
        config = LoRAConfig(rank=4)
        adapter = LoRAAdapter(simple_model, config)

        counts = adapter.get_parameter_count()

        assert "base_parameters" in counts
        assert "lora_parameters" in counts
        assert "trainable_parameters" in counts
        assert "reduction_ratio" in counts
        assert "memory_saved_mb" in counts

        # LoRA parameters should be much fewer than base
        assert counts["lora_parameters"] < counts["base_parameters"]
        assert counts["reduction_ratio"] > 0.9  # Should reduce by >90%

    @patch("src.utils.lora_adapter.torch.save")
    def test_save_adapter(self, mock_save, simple_model, tmp_path):
        """Test saving LoRA adapter."""
        config = LoRAConfig(rank=4, alpha=8)
        adapter = LoRAAdapter(simple_model, config)

        save_path = tmp_path / "lora_adapter.pt"
        adapter.save_adapter(str(save_path))

        mock_save.assert_called_once()

        # Check saved state dict structure
        state_dict = mock_save.call_args[0][0]
        assert "config" in state_dict
        assert state_dict["config"]["rank"] == 4
        assert state_dict["config"]["alpha"] == 8

    @patch("src.utils.lora_adapter.torch.load")
    def test_load_adapter(self, mock_load, simple_model, tmp_path):
        """Test loading LoRA adapter."""
        config = LoRAConfig(rank=4)
        adapter = LoRAAdapter(simple_model, config)

        # Mock loaded state dict
        mock_state = {
            "config": {"rank": 4, "alpha": 16},
            "q_proj.lora_A": torch.randn(4, 20),
            "q_proj.lora_B": torch.randn(20, 4),
            "v_proj.lora_A": torch.randn(4, 20),
            "v_proj.lora_B": torch.randn(20, 4),
        }
        mock_load.return_value = mock_state

        load_path = tmp_path / "lora_adapter.pt"
        adapter.load_adapter(str(load_path))

        mock_load.assert_called_once_with(str(load_path), map_location="cpu")

    def test_merge_unmerge_weights(self, simple_model):
        """Test merging and unmerging weights."""
        config = LoRAConfig(rank=4)
        adapter = LoRAAdapter(simple_model, config)

        # Initially not merged
        for lora_layer in adapter.lora_layers.values():
            assert not lora_layer.merged

        # Merge
        adapter.merge_weights()
        for lora_layer in adapter.lora_layers.values():
            assert lora_layer.merged or not lora_layer.merge_weights

        # Unmerge
        adapter.unmerge_weights()
        for lora_layer in adapter.lora_layers.values():
            assert not lora_layer.merged


class TestApplyLoRAToModel:
    """Test the apply_lora_to_model function."""

    def test_apply_lora(self):
        """Test applying LoRA to a model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        # Apply LoRA targeting Linear layers
        adapter = apply_lora_to_model(
            model,
            rank=4,
            alpha=8,
            dropout=0.2,
            target_modules=["0", "2"]  # Target the linear layers
        )

        assert isinstance(adapter, LoRAAdapter)
        assert len(adapter.lora_layers) == 2  # Two linear layers

        # Check that LoRA parameters are much fewer
        counts = adapter.get_parameter_count()
        assert counts["reduction_ratio"] > 0.9
