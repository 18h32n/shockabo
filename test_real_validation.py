#!/usr/bin/env python3
"""
Quick test script to validate TTT pipeline with real ARC dataset.
Tests basic functionality before running full validation.
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from adapters.repositories.arc_data_repository import ARCDataRepository
from adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig


def test_real_dataset_loading():
    """Test real dataset loading functionality."""
    logger.info("Testing real ARC dataset loading...")

    try:
        # Initialize repository with real dataset
        repo = ARCDataRepository(use_real_dataset=True)

        # Test task ID retrieval
        task_ids = repo.get_task_ids("training")
        logger.info(f"Found {len(task_ids)} training tasks")

        # Test loading a few tasks
        test_tasks = repo.load_all_tasks("training", limit=3)
        logger.info(f"Successfully loaded {len(test_tasks)} test tasks")

        # Examine task structure
        for task_id, task in list(test_tasks.items())[:2]:
            logger.info(f"Task {task_id}:")
            logger.info(f"  - Train examples: {len(task.train_examples)}")
            logger.info(f"  - Test input: {len(task.test_input)}x{len(task.test_input[0]) if task.test_input else 0}")
            logger.info(f"  - Has test output: {task.test_output is not None}")

        return True, test_tasks

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return False, {}

def test_ttt_adapter_initialization():
    """Test TTT adapter initialization with CPU mode."""
    logger.info("Testing TTT adapter initialization (CPU mode)...")

    try:
        # Create TTT config for CPU testing
        config = TTTConfig(
            model_name="meta-llama/Llama-3.2-1B",
            device="cpu",  # Force CPU mode
            max_examples=2,
            num_epochs=1,  # Minimal epochs for testing
            learning_rate=1e-4,
            checkpoint_dir=Path("test_data/models/ttt"),
            cache_dir=Path("test_data/cache/ttt"),
        )

        # Initialize adapter
        adapter = TTTAdapter(config=config)
        logger.info("TTT adapter initialized successfully")

        return True, adapter

    except Exception as e:
        logger.error(f"TTT adapter initialization failed: {e}")
        return False, None

def test_single_task_processing(adapter, task):
    """Test processing a single task with TTT adapter."""
    logger.info(f"Testing single task processing: {task.task_id}")

    try:
        # Attempt to solve the task
        result = adapter.solve(task)

        logger.info("Task processing completed:")
        logger.info(f"  - Task ID: {task.task_id}")
        logger.info(f"  - Solution generated: {result is not None}")
        if result:
            logger.info(f"  - Predictions: {len(result.predictions)}")
            logger.info(f"  - Resource usage: CPU {result.resource_usage.cpu_seconds:.2f}s, Memory {result.resource_usage.memory_mb:.1f}MB")

        return True, result

    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        return False, None

def main():
    """Run basic validation tests."""
    logger.info("=== Starting Real ARC Dataset Validation Test ===")

    # Test 1: Dataset loading
    success, tasks = test_real_dataset_loading()
    if not success:
        logger.error("Dataset loading test failed - aborting")
        return False

    # Test 2: TTT adapter initialization
    success, adapter = test_ttt_adapter_initialization()
    if not success:
        logger.error("TTT adapter initialization test failed - aborting")
        return False

    # Test 3: Single task processing
    if tasks:
        sample_task = list(tasks.values())[0]
        success, result = test_single_task_processing(adapter, sample_task)
        if not success:
            logger.error("Task processing test failed")
            return False

    # Cleanup
    if adapter:
        adapter.cleanup()

    logger.info("=== All tests passed successfully! ===")
    logger.info("Ready to run full validation with real dataset")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
