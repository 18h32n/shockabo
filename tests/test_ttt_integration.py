"""
Integration tests for MIT TTT implementation.

This module tests the complete MIT TTT pipeline with sample data,
ensuring all components work together correctly.
"""
import tempfile
import unittest
from pathlib import Path

import pytest

from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask
from src.utils.ttt_data_conversion import (
    AugmentationEngine,
    AugmentationType,
    TextTaskRepresenter,
    TTTDataConverter,
)
from src.utils.ttt_voting import (
    AugmentationVoter,
    HybridVoter,
    SelfConsistencyVoter,
    create_prediction_candidate,
)


class TestTTTDataConversion(unittest.TestCase):
    """Test TTT data format conversion utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_task = self._create_sample_arc_task()
        self.converter = TTTDataConverter(use_gpt_format=True, random_seed=42)

    def _create_sample_arc_task(self) -> ARCTask:
        """Create a sample ARC task for testing."""
        train_examples = [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]]
            },
            {
                "input": [[2, 0], [0, 2]],
                "output": [[0, 2], [2, 0]]
            }
        ]

        test_input = [[3, 0], [0, 3]]

        return ARCTask(
            task_id="test_task_001",
            task_source="training",
            train_examples=train_examples,
            test_input=test_input
        )

    def test_text_representer(self):
        """Test text representation of grids."""
        representer = TextTaskRepresenter()

        grid = [[1, 0], [0, 1]]
        text = representer.grid_to_text(grid)
        self.assertEqual(text, "[[1, 0], [0, 1]]")

        example_text = representer.create_example_text(
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]]
        )
        self.assertIn("[[1, 0], [0, 1]] -> [[0, 1], [1, 0]]", example_text)

    def test_augmentation_engine(self):
        """Test augmentation generation."""
        engine = AugmentationEngine(random_seed=42)

        grid = [[1, 2], [3, 4]]

        # Test basic augmentations
        augmented = engine.basic_augmentations(grid, grid)
        self.assertGreater(len(augmented), 1)  # Should have original + augmentations

        # Test rotation
        rotated = engine.rotate_grid(grid, 1)
        expected = [[3, 1], [4, 2]]  # 90-degree clockwise rotation
        self.assertEqual(rotated, expected)

        # Test horizontal flip
        flipped = engine.flip_horizontal(grid)
        expected = [[2, 1], [4, 3]]
        self.assertEqual(flipped, expected)

    def test_ttt_conversion(self):
        """Test complete ARC to TTT conversion."""
        ttt_task = self.converter.convert_arc_task(
            self.sample_task,
            augmentation_types=[AugmentationType.BASIC]
        )

        # Check basic structure
        self.assertEqual(ttt_task.task_id, "test_task_001")
        self.assertEqual(len(ttt_task.examples), 2)  # 2 training examples
        self.assertGreater(len(ttt_task.augmented_examples), 0)  # Should have augmentations
        self.assertEqual(len(ttt_task.leave_one_out_splits), 2)  # One split per example

        # Check leave-one-out splits
        for _i, split in enumerate(ttt_task.leave_one_out_splits):
            # Each split should exclude one original example but include augmentations
            original_count = sum(1 for ex in split if ex.metadata.get("original", False))
            self.assertLess(original_count, len(ttt_task.examples))

    def test_training_prompt_generation(self):
        """Test training prompt generation."""
        ttt_task = self.converter.convert_arc_task(self.sample_task)
        prompts = self.converter.create_training_prompts(ttt_task, split_index=0)

        self.assertGreater(len(prompts), 0)
        for prompt in prompts:
            self.assertIsInstance(prompt, str)
            self.assertIn("Transform the input grid", prompt)

    def test_inference_prompt_generation(self):
        """Test inference prompt generation."""
        ttt_task = self.converter.convert_arc_task(self.sample_task)
        prompt = self.converter.create_inference_prompt(ttt_task)

        self.assertIsInstance(prompt, str)
        self.assertIn("Transform the input grid", prompt)
        self.assertIn("[[3, 0], [0, 3]]", prompt)  # Test input should be in prompt


class TestTTTVoting(unittest.TestCase):
    """Test TTT voting mechanisms."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_predictions = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],  # Same as first
            [[0, 1], [1, 0]],  # Different
            [[1, 0], [0, 1]],  # Same as first
        ]

    def test_self_consistency_voter(self):
        """Test self-consistency voting."""
        voter = SelfConsistencyVoter()

        candidates = [
            create_prediction_candidate(pred, confidence=0.8 + i * 0.1)
            for i, pred in enumerate(self.sample_predictions)
        ]

        result = voter.vote_predictions(candidates)

        # Should select the most common prediction
        self.assertEqual(result.best_prediction, [[1, 0], [0, 1]])
        self.assertGreater(result.confidence_score, 0.0)
        self.assertEqual(result.total_candidates, 4)
        self.assertGreater(result.agreement_ratio, 0.5)  # Majority agreement

    def test_augmentation_voter(self):
        """Test augmentation-aware voting."""
        voter = AugmentationVoter()

        candidates = [
            create_prediction_candidate(
                self.sample_predictions[0],
                confidence=0.8,
                augmentation_type="original"
            ),
            create_prediction_candidate(
                self.sample_predictions[1],
                confidence=0.7,
                augmentation_type="basic"
            ),
            create_prediction_candidate(
                self.sample_predictions[2],
                confidence=0.9,
                augmentation_type="size"
            ),
        ]

        result = voter.vote_augmented_predictions(candidates)

        self.assertIsNotNone(result.best_prediction)
        self.assertGreater(result.confidence_score, 0.0)
        self.assertEqual(result.total_candidates, 3)

    def test_hybrid_voter(self):
        """Test hybrid voting mechanism."""
        voter = HybridVoter()

        candidates = [
            create_prediction_candidate(
                pred,
                confidence=0.8,
                augmentation_type="original" if i == 0 else "basic"
            )
            for i, pred in enumerate(self.sample_predictions)
        ]

        result = voter.vote_all_predictions(candidates)

        self.assertIsNotNone(result.best_prediction)
        self.assertGreater(result.confidence_score, 0.0)
        self.assertIn("hybrid", result.voting_method)
        self.assertIn("metadata", result.__dict__)

    def test_voting_with_fallback(self):
        """Test voting with fallback prediction."""
        voter = HybridVoter()
        fallback = [[9, 9], [9, 9]]

        # Empty candidates should use fallback
        result = voter.vote_all_predictions([], fallback_prediction=fallback)
        self.assertEqual(result.best_prediction, fallback)
        self.assertTrue(result.metadata.get("used_fallback", False))


class TestTTTAdapter(unittest.TestCase):
    """Test TTT adapter integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = TTTConfig(
            model_name="meta-llama/Llama-3.2-1B",
            lora_rank=8,  # Small for testing
            num_epochs=1,
            per_instance_epochs=1,
            batch_size=1,
            memory_limit_mb=1024,  # 1GB for testing
            checkpoint_dir=self.temp_dir / "checkpoints",
            cache_dir=self.temp_dir / "cache"
        )

        self.sample_task = self._create_sample_arc_task()

    def _create_sample_arc_task(self) -> ARCTask:
        """Create a sample ARC task for testing."""
        return ARCTask(
            task_id="test_integration_001",
            task_source="training",
            train_examples=[
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            test_input=[[2, 0], [0, 2]]
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_adapter_initialization(self):
        """Test TTT adapter initialization."""
        adapter = TTTAdapter(self.config)

        self.assertIsNotNone(adapter.config)
        self.assertIsNotNone(adapter.ttt_config)
        self.assertIsNotNone(adapter.mit_ttt_strategy)

        # Check directories were created
        self.assertTrue(self.config.checkpoint_dir.exists())
        self.assertTrue(self.config.cache_dir.exists())

    @pytest.mark.slow
    def test_training_examples_preparation(self):
        """Test preparation of training examples."""
        adapter = TTTAdapter(self.config)

        examples = adapter._prepare_training_examples(self.sample_task)

        self.assertGreater(len(examples), 0)
        for example in examples:
            self.assertIn("prompt", example)
            self.assertIn("input_grid", example)
            self.assertIn("output_grid", example)
            self.assertIsInstance(example["prompt"], str)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        adapter = TTTAdapter(self.config)

        memory_mb = adapter._estimate_memory_usage()
        self.assertGreaterEqual(memory_mb, 0.0)

        gpu_memory_mb = adapter._estimate_gpu_memory()
        self.assertGreaterEqual(gpu_memory_mb, 0.0)

    def test_cleanup(self):
        """Test adapter cleanup."""
        adapter = TTTAdapter(self.config)

        # Should not raise exceptions
        adapter.cleanup()


class TestTTTConfigFromYAML(unittest.TestCase):
    """Test TTT configuration loading from YAML."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_ttt.yaml"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
model:
  name: "test-model"
  device: "cpu"
  quantization: false
  max_length: 1024

training:
  learning_rate: 0.00001
  num_epochs: 3
  batch_size: 4
  per_instance_lr: 0.0001
  per_instance_epochs: 1
  gradient_accumulation_steps: 1
  mixed_precision: false
  gradient_checkpointing: true
  memory_limit_mb: 1024
  max_training_time: 300
  max_examples: 5

lora:
  rank: 32
  alpha: 32
  dropout: 0.2

adaptation:
  use_basic_augmentation: true
  use_size_augmentation: false
  use_chain_augmentation: false
  permute_n: 1

inference:
  temperature: 0.1
"""

        with open(self.config_file, 'w') as f:
            f.write(yaml_content)

        config = TTTConfig.from_yaml(self.config_file)

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.quantization, False)
        self.assertEqual(config.max_length, 1024)
        self.assertEqual(config.learning_rate, 0.00001)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.per_instance_lr, 0.0001)
        self.assertEqual(config.per_instance_epochs, 1)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.mixed_precision, False)
        self.assertEqual(config.gradient_checkpointing, True)
        self.assertEqual(config.memory_limit_mb, 1024)
        self.assertEqual(config.max_training_time, 300)
        self.assertEqual(config.max_examples, 5)
        self.assertEqual(config.lora_rank, 32)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.2)
        self.assertEqual(config.use_basic_augmentation, True)
        self.assertEqual(config.use_size_augmentation, False)
        self.assertEqual(config.use_chain_augmentation, False)
        self.assertEqual(config.permute_n, 1)
        self.assertEqual(config.temperature, 0.1)

    def test_config_from_nonexistent_yaml(self):
        """Test loading configuration from non-existent YAML file."""
        nonexistent_file = self.temp_dir / "nonexistent.yaml"
        config = TTTConfig.from_yaml(nonexistent_file)

        # Should return default configuration
        self.assertEqual(config.model_name, "meta-llama/Llama-3.2-1B")
        self.assertEqual(config.lora_rank, 64)


class TestTTTEndToEnd(unittest.TestCase):
    """End-to-end integration tests for TTT pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_tasks = self._create_sample_tasks()

    def _create_sample_tasks(self) -> list[ARCTask]:
        """Create sample ARC tasks for testing."""
        return [
            ARCTask(
                task_id=f"e2e_test_{i:03d}",
                task_source="training",
                train_examples=[
                    {
                        "input": [[i % 3, 0], [0, (i + 1) % 3]],
                        "output": [[0, (i + 1) % 3], [i % 3, 0]]
                    }
                ],
                test_input=[[i % 3, 0], [0, (i + 2) % 3]]
            )
            for i in range(3)
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_ttt_pipeline(self):
        """Test complete TTT pipeline with sample data."""
        # Test data conversion
        converter = TTTDataConverter(random_seed=42)

        for task in self.sample_tasks:
            # Convert to TTT format
            ttt_task = converter.convert_arc_task(
                task,
                augmentation_types=[AugmentationType.BASIC]
            )

            # Verify conversion
            self.assertEqual(ttt_task.task_id, task.task_id)
            self.assertGreater(len(ttt_task.examples), 0)
            self.assertGreater(len(ttt_task.augmented_examples), 0)

            # Test prompt generation
            training_prompts = converter.create_training_prompts(ttt_task)
            self.assertGreater(len(training_prompts), 0)

            inference_prompt = converter.create_inference_prompt(ttt_task)
            self.assertIsInstance(inference_prompt, str)

    def test_voting_pipeline(self):
        """Test voting pipeline with diverse predictions."""
        # Create diverse prediction candidates
        predictions = [
            [[1, 0], [0, 1]],  # Prediction A
            [[1, 0], [0, 1]],  # Prediction A (duplicate)
            [[0, 1], [1, 0]],  # Prediction B
            [[1, 0], [0, 1]],  # Prediction A (duplicate)
            [[2, 0], [0, 2]],  # Prediction C
        ]

        candidates = [
            create_prediction_candidate(
                pred,
                confidence=0.7 + i * 0.05,
                augmentation_type=["original", "basic", "size", "chain", "repeat"][i % 5],
                original=(i == 0)
            )
            for i, pred in enumerate(predictions)
        ]

        # Test all voting methods
        sc_voter = SelfConsistencyVoter()
        sc_result = sc_voter.vote_predictions(candidates)
        self.assertIsNotNone(sc_result.best_prediction)

        aug_voter = AugmentationVoter()
        aug_result = aug_voter.vote_augmented_predictions(candidates)
        self.assertIsNotNone(aug_result.best_prediction)

        hybrid_voter = HybridVoter()
        hybrid_result = hybrid_voter.vote_all_predictions(candidates)
        self.assertIsNotNone(hybrid_result.best_prediction)

        # Verify metadata
        self.assertIn("metadata", hybrid_result.__dict__)
        self.assertGreater(hybrid_result.confidence_score, 0.0)


if __name__ == "__main__":
    # Run tests with different verbosity levels
    unittest.main(verbosity=2)
