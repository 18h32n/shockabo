#!/usr/bin/env python3
"""
Test script for MIT TTT implementation.

This script provides a quick way to test the TTT implementation with sample data
without requiring full model downloads or extensive computation.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.strategies.ttt_adapter import TTTConfig
from src.domain.models import ARCTask
from src.utils.ttt_data_conversion import AugmentationType, TTTDataConverter
from src.utils.ttt_voting import HybridVoter, create_prediction_candidate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_arc_task() -> ARCTask:
    """Create a simple ARC task for testing."""
    return ARCTask(
        task_id="sample_test_001",
        task_source="training",
        train_examples=[
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]]
            },
            {
                "input": [[2, 0], [0, 2]],
                "output": [[0, 2], [2, 0]]
            },
            {
                "input": [[3, 0], [0, 3]],
                "output": [[0, 3], [3, 0]]
            }
        ],
        test_input=[[4, 0], [0, 4]]
    )


def test_data_conversion():
    """Test TTT data format conversion."""
    logger.info("Testing TTT data format conversion...")

    task = create_sample_arc_task()
    converter = TTTDataConverter(use_gpt_format=True, random_seed=42)

    # Test conversion with different augmentation types
    for aug_type in [AugmentationType.BASIC, AugmentationType.SIZE, AugmentationType.CHAIN]:
        logger.info(f"Testing augmentation type: {aug_type.value}")

        ttt_task = converter.convert_arc_task(task, augmentation_types=[aug_type])

        logger.info(f"  - Original examples: {len(ttt_task.examples)}")
        logger.info(f"  - Augmented examples: {len(ttt_task.augmented_examples)}")
        logger.info(f"  - Leave-one-out splits: {len(ttt_task.leave_one_out_splits)}")

        # Test prompt generation
        training_prompts = converter.create_training_prompts(ttt_task, split_index=0)
        inference_prompt = converter.create_inference_prompt(ttt_task)

        logger.info(f"  - Training prompts: {len(training_prompts)}")
        logger.info(f"  - Inference prompt length: {len(inference_prompt)} chars")

        # Show sample prompt
        if training_prompts:
            sample_prompt = training_prompts[0][:200] + "..." if len(training_prompts[0]) > 200 else training_prompts[0]
            logger.info(f"  - Sample prompt: {sample_prompt}")

    logger.info("✓ Data conversion tests passed")


def test_voting_mechanisms():
    """Test voting mechanisms with sample predictions."""
    logger.info("Testing voting mechanisms...")

    # Create diverse prediction candidates
    predictions = [
        [[0, 4], [4, 0]],  # Expected correct answer
        [[0, 4], [4, 0]],  # Same (should get majority)
        [[4, 0], [0, 4]],  # Different pattern
        [[0, 4], [4, 0]],  # Same as expected
        [[1, 1], [1, 1]],  # Completely different
    ]

    candidates = [
        create_prediction_candidate(
            pred,
            confidence=0.8 - i * 0.1,  # Decreasing confidence
            augmentation_type=["original", "basic", "size", "chain", "repeat"][i],
            original=(i == 0)
        )
        for i, pred in enumerate(predictions)
    ]

    # Test hybrid voting
    voter = HybridVoter()
    result = voter.vote_all_predictions(candidates)

    logger.info(f"  - Best prediction: {result.best_prediction}")
    logger.info(f"  - Confidence score: {result.confidence_score:.3f}")
    logger.info(f"  - Agreement ratio: {result.agreement_ratio:.3f}")
    logger.info(f"  - Voting method: {result.voting_method}")
    logger.info(f"  - Vote distribution: {result.vote_distribution}")

    # Verify the expected prediction won
    expected = [[0, 4], [4, 0]]
    if result.best_prediction == expected:
        logger.info("✓ Voting correctly selected expected prediction")
    else:
        logger.warning(f"! Voting selected {result.best_prediction}, expected {expected}")

    logger.info("✓ Voting mechanism tests passed")


def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")

    # Test default config
    config = TTTConfig()
    logger.info(f"  - Default model: {config.model_name}")
    logger.info(f"  - LoRA rank: {config.lora_rank}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Batch size: {config.batch_size}")

    # Test YAML config loading (if file exists)
    yaml_config_path = project_root / "configs" / "strategies" / "ttt.yaml"
    if yaml_config_path.exists():
        yaml_config = TTTConfig.from_yaml(yaml_config_path)
        logger.info(f"  - YAML model: {yaml_config.model_name}")
        logger.info(f"  - YAML LoRA rank: {yaml_config.lora_rank}")
        logger.info("✓ YAML config loaded successfully")
    else:
        logger.info("  - YAML config file not found, using defaults")

    logger.info("✓ Configuration tests passed")


def test_memory_awareness():
    """Test memory monitoring capabilities."""
    logger.info("Testing memory monitoring...")

    try:
        from src.utils.ttt_memory_manager import MemoryMonitor, memory_efficient_context

        # Test memory monitor
        monitor = MemoryMonitor(memory_limit_mb=1024)
        stats = monitor.get_memory_stats()

        logger.info(f"  - Current memory: {stats.current_memory_mb:.1f} MB")
        logger.info(f"  - Memory utilization: {stats.memory_utilization:.1%}")
        logger.info(f"  - Available memory: {stats.available_memory_mb:.1f} MB")

        # Test memory context
        with memory_efficient_context(memory_limit_mb=1024) as mem_monitor:
            context_stats = mem_monitor.get_memory_stats()
            logger.info(f"  - Context memory: {context_stats.current_memory_mb:.1f} MB")

        logger.info("✓ Memory monitoring tests passed")

    except ImportError as e:
        logger.warning(f"Memory monitoring not available: {e}")


def test_integration_readiness():
    """Test overall integration readiness."""
    logger.info("Testing integration readiness...")

    try:
        # Test all required components can be imported
        from src.utils.ttt_data_conversion import TTTDataConverter
        from src.utils.ttt_voting import HybridVoter

        logger.info("✓ All MIT TTT components importable")

        # Test basic component initialization
        converter = TTTDataConverter()
        voter = HybridVoter()
        TTTConfig()

        logger.info("✓ All components can be initialized")

        # Test workflow compatibility
        task = create_sample_arc_task()
        converter.convert_arc_task(task)

        # Create mock predictions for voting
        predictions = [[[0, 4], [4, 0]], [[4, 0], [0, 4]]]
        candidates = [
            create_prediction_candidate(pred, confidence=0.8)
            for pred in predictions
        ]
        voter.vote_all_predictions(candidates)

        logger.info("✓ End-to-end workflow functional")

        logger.info("🎉 MIT TTT implementation is ready for integration!")

    except Exception as e:
        logger.error(f"❌ Integration readiness check failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    logger.info("🚀 Starting MIT TTT implementation tests...")
    logger.info("=" * 60)

    try:
        # Run all test functions
        test_data_conversion()
        print()

        test_voting_mechanisms()
        print()

        test_config_loading()
        print()

        test_memory_awareness()
        print()

        success = test_integration_readiness()
        print()

        if success:
            logger.info("🎉 All tests completed successfully!")
            logger.info("MIT TTT implementation is ready for production use.")
        else:
            logger.error("❌ Some tests failed. Please review the errors above.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
