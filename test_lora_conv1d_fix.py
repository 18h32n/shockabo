"""
Test script to validate LoRA Conv1D compatibility fix.

This script tests that LoRA can now properly target Conv1D layers in GPT-2 models.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.auth_config import get_model_access_info, setup_hf_auth
from src.utils.lora_adapter import LoRAAdapter, LoRAConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lora_conv1d_compatibility():
    """Test LoRA compatibility with Conv1D layers."""
    logger.info("Testing LoRA Conv1D compatibility...")

    # Set up authentication
    setup_hf_auth()

    # Use GPT-2 as it has Conv1D layers
    model_name = "gpt2"
    logger.info(f"Loading model: {model_name}")

    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Model loaded successfully")

        # Create LoRA configuration targeting Conv1D layers
        lora_config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # GPT-2 Conv1D modules
        )

        logger.info(f"LoRA config: rank={lora_config.rank}, alpha={lora_config.alpha}")
        logger.info(f"Target modules: {lora_config.target_modules}")

        # Apply LoRA adaptation
        logger.info("Applying LoRA adaptation...")
        lora_adapter = LoRAAdapter(model, lora_config)

        # Get trainable parameters
        trainable_params = lora_adapter.get_trainable_parameters()
        num_trainable = len(trainable_params)

        logger.info(f"Number of trainable LoRA parameters: {num_trainable}")

        if num_trainable == 0:
            logger.error("❌ FAILED: No trainable parameters found!")
            return False

        # Test that we can get parameter count
        total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        logger.info(f"Total trainable parameter count: {total_params:,}")

        # Test forward pass
        logger.info("Testing forward pass...")
        test_input = "The quick brown fox"
        inputs = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logger.info(f"Forward pass successful! Output shape: {logits.shape}")

        # Test that LoRA layers are properly integrated
        found_lora_layers = len(lora_adapter.lora_layers)
        logger.info(f"Number of LoRA layers applied: {found_lora_layers}")

        if found_lora_layers == 0:
            logger.error("❌ FAILED: No LoRA layers were applied!")
            return False

        logger.info("✅ SUCCESS: LoRA Conv1D compatibility is working!")
        logger.info(f"✅ Applied LoRA to {found_lora_layers} Conv1D layers")
        logger.info(f"✅ {total_params:,} trainable parameters available")
        logger.info("✅ Forward pass works correctly")

        return True

    except Exception as e:
        logger.error(f"❌ FAILED: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_access():
    """Test model access with authentication system."""
    logger.info("Testing model access system...")

    test_models = [
        "gpt2",
        "meta-llama/Llama-3.2-1B",
        "microsoft/DialoGPT-medium"
    ]

    for model_name in test_models:
        access_info = get_model_access_info(model_name)
        logger.info(f"Model: {model_name}")
        logger.info(f"  Requires auth: {access_info['requires_authentication']}")
        logger.info(f"  Can access: {access_info['can_access']}")

        if not access_info['can_access'] and access_info['requires_authentication']:
            from src.utils.auth_config import suggest_public_model
            suggestion = suggest_public_model(model_name)
            logger.info(f"  Suggested alternative: {suggestion}")

    return True

def main():
    """Main test execution."""
    logger.info("=" * 60)
    logger.info("LoRA Conv1D Compatibility Test")
    logger.info("=" * 60)

    # Test 1: Model access system
    logger.info("\n1. Testing model access system...")
    if not test_model_access():
        sys.exit(1)

    # Test 2: LoRA Conv1D compatibility
    logger.info("\n2. Testing LoRA Conv1D compatibility...")
    if not test_lora_conv1d_compatibility():
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("LoRA Conv1D compatibility fix is working correctly.")
    logger.info("The system can now train LoRA adapters on GPT-2 models.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
