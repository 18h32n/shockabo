"""Integration tests for checkpoint saving/loading and TTT adapter functionality."""
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask, StrategyType
from src.domain.services.training_orchestrator import TrainingConfig, TrainingOrchestrator
from src.domain.services.ttt_service import TTTModelService


@pytest.fixture
def checkpoint_test_dir():
    """Create temporary directory for checkpoint tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_checkpoint_data():
    """Create sample checkpoint data for testing."""
    return {
        "model_state_dict": {
            "embedding.weight": torch.randn(1000, 512),
            "transformer.h.0.attn.weight": torch.randn(512, 512),
            "transformer.h.0.mlp.weight": torch.randn(512, 2048),
            "lm_head.weight": torch.randn(50257, 512)
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }]
        },
        "scheduler_state_dict": {
            "last_epoch": 10,
            "last_lr": [1e-4]
        },
        "lora_adapter": {
            "lora_A": torch.randn(8, 512),
            "lora_B": torch.randn(512, 8),
            "scaling": torch.tensor(2.0)
        }
    }


@pytest.fixture
def sample_training_metrics():
    """Create sample training metrics for testing."""
    return {
        "model_name": "gpt2-medium",
        "final_accuracy": 0.72,
        "best_validation_accuracy": 0.75,
        "training_time": 3600.0,
        "total_steps": 500,
        "convergence_step": 350,
        "final_loss": 0.32,
        "best_loss": 0.28,
        "final_memory_mb": 7500.0,
        "peak_memory_mb": 8200.0,
        "gpu_memory_mb": 12000.0,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "epochs_completed": 3,
        "early_stopping_triggered": False,
        "checkpoint_size_mb": 1250.0
    }


@pytest.fixture
def arc_task_for_checkpoints():
    """Create ARC task for checkpoint testing."""
    return ARCTask(
        task_id="checkpoint_test_task_001",
        task_source="test",
        train_examples=[
            {
                "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            },
            {
                "input": [[2, 2, 0], [2, 0, 2], [0, 2, 2]],
                "output": [[0, 0, 2], [0, 2, 0], [2, 0, 0]]
            },
            {
                "input": [[3, 3, 3], [3, 0, 3], [3, 3, 3]],
                "output": [[0, 0, 0], [0, 3, 0], [0, 0, 0]]
            }
        ],
        test_input=[[4, 4, 0], [4, 0, 4], [0, 4, 4]]
    )


class TestCheckpointRepository:
    """Test checkpoint repository functionality."""

    def test_checkpoint_save_and_load_basic(self, checkpoint_test_dir, sample_checkpoint_data, sample_training_metrics):
        """Test basic checkpoint save and load operations."""
        repo = CheckpointRepository(base_path=checkpoint_test_dir)

        checkpoint_id = "basic_test_001"
        task_id = "basic_task_001"

        lora_config = {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }

        # Save checkpoint
        metadata = repo.save_checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_state=sample_checkpoint_data,
            training_metrics=sample_training_metrics,
            lora_config=lora_config,
            tags=["test", "basic", "integration"]
        )

        # Verify metadata
        assert metadata.checkpoint_id == checkpoint_id
        assert metadata.task_id == task_id
        assert metadata.accuracy == sample_training_metrics["final_accuracy"]
        assert metadata.training_time == sample_training_metrics["training_time"]
        assert len(metadata.tags) == 3

        # Load checkpoint
        loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)

        # Verify loaded data structure
        assert "model_state" in loaded_data
        assert "training_metrics" in loaded_data
        assert "lora_config" in loaded_data

        # Verify loaded metadata matches
        assert loaded_metadata.checkpoint_id == checkpoint_id
        assert loaded_metadata.task_id == task_id
        assert loaded_metadata.accuracy == sample_training_metrics["final_accuracy"]

        # Verify data integrity
        assert loaded_data["training_metrics"]["final_accuracy"] == 0.72
        assert loaded_data["lora_config"]["rank"] == 8

        # Check that tensors are properly loaded
        model_state = loaded_data["model_state"]["model_state_dict"]
        assert "embedding.weight" in model_state
        assert isinstance(model_state["embedding.weight"], torch.Tensor)
        assert model_state["embedding.weight"].shape == (1000, 512)

    def test_checkpoint_versioning_and_history(self, checkpoint_test_dir, sample_checkpoint_data, sample_training_metrics):
        """Test checkpoint versioning and historical tracking."""
        repo = CheckpointRepository(base_path=checkpoint_test_dir)

        task_id = "versioning_task_001"
        base_checkpoint_id = "version_test"

        # Create multiple versions with improving accuracy
        accuracies = [0.45, 0.62, 0.73, 0.81, 0.76]  # Last one is worse (overfitting)
        checkpoint_ids = []

        for i, accuracy in enumerate(accuracies):
            checkpoint_id = f"{base_checkpoint_id}_{i+1:03d}"
            checkpoint_ids.append(checkpoint_id)

            # Update metrics for this version
            metrics = sample_training_metrics.copy()
            metrics["final_accuracy"] = accuracy
            metrics["training_time"] = 1000 + i * 500  # Increasing training time
            metrics["total_steps"] = 100 + i * 50
            metrics["epochs_completed"] = i + 1

            # Save checkpoint
            metadata = repo.save_checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                model_state=sample_checkpoint_data,
                training_metrics=metrics,
                lora_config={"rank": 8, "alpha": 16},
                tags=[f"version_{i+1}", "accuracy_progression"]
            )

            assert metadata.accuracy == accuracy

        # List all checkpoints for task
        checkpoints = repo.list_checkpoints(task_id=task_id)
        assert len(checkpoints) == 5

        # Verify chronological order
        checkpoints_by_time = sorted(checkpoints, key=lambda x: x.created_at)
        for i, cp in enumerate(checkpoints_by_time):
            assert cp.checkpoint_id == checkpoint_ids[i]

        # Get best checkpoint (should be version 4 with 0.81 accuracy)
        best_checkpoint = repo.get_best_checkpoint(task_id)
        assert best_checkpoint.checkpoint_id == f"{base_checkpoint_id}_004"
        assert best_checkpoint.accuracy == 0.81

        # Test filtering by accuracy threshold
        high_accuracy_checkpoints = repo.list_checkpoints(
            task_id=task_id,
            min_accuracy=0.7
        )
        assert len(high_accuracy_checkpoints) == 3  # versions 3, 4, 5

        # Test filtering by tags
        version_checkpoints = repo.list_checkpoints(tags=["accuracy_progression"])
        assert len(version_checkpoints) == 5

    def test_checkpoint_cleanup_strategies(self, checkpoint_test_dir, sample_checkpoint_data, sample_training_metrics):
        """Test different checkpoint cleanup strategies."""
        repo = CheckpointRepository(base_path=checkpoint_test_dir)

        # Create checkpoints for multiple tasks
        tasks = ["cleanup_task_001", "cleanup_task_002", "cleanup_task_003"]
        checkpoint_data = []

        for task_id in tasks:
            for i in range(8):  # 8 checkpoints per task
                checkpoint_id = f"{task_id}_checkpoint_{i:03d}"
                accuracy = 0.3 + (i * 0.08)  # Improving accuracy: 0.3 to 0.86

                metrics = sample_training_metrics.copy()
                metrics["final_accuracy"] = accuracy
                metrics["training_time"] = 1000 + i * 200

                metadata = repo.save_checkpoint(
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    model_state=sample_checkpoint_data,
                    training_metrics=metrics,
                    lora_config={"rank": 8}
                )

                checkpoint_data.append(metadata)

        # Verify all checkpoints exist
        total_checkpoints = repo.list_checkpoints()
        assert len(total_checkpoints) == 24  # 3 tasks × 8 checkpoints

        # Test cleanup: keep best 3 per task
        cleanup_stats = repo.cleanup_storage(keep_best_per_task=3)

        assert cleanup_stats["deleted_count"] == 15  # 24 - (3 × 3)
        assert cleanup_stats["kept_count"] == 9  # 3 × 3

        # Verify only best 3 remain per task
        for task_id in tasks:
            remaining = repo.list_checkpoints(task_id=task_id)
            assert len(remaining) == 3

            # Verify these are the best ones (highest accuracy)
            accuracies = [cp.accuracy for cp in remaining]
            assert min(accuracies) >= 0.72  # Should keep checkpoints 5, 6, 7 (0.72, 0.8, 0.86)

        # Test cleanup by age (simulate old checkpoints)
        import time
        time.sleep(0.1)  # Small delay to differentiate timestamps

        # Add new checkpoint
        new_checkpoint_id = "new_checkpoint_001"
        repo.save_checkpoint(
            checkpoint_id=new_checkpoint_id,
            task_id="cleanup_task_001",
            model_state=sample_checkpoint_data,
            training_metrics=sample_training_metrics,
            lora_config={"rank": 8}
        )

        # Cleanup by age (keep only very recent ones)
        cleanup_stats = repo.cleanup_storage(
            keep_best_per_task=1,
            max_age_days=0.001  # Very short age limit
        )

        # Should keep only the newest checkpoints
        assert cleanup_stats["deleted_count"] > 0

    def test_checkpoint_integrity_validation(self, checkpoint_test_dir, sample_checkpoint_data):
        """Test checkpoint integrity validation and corruption detection."""
        repo = CheckpointRepository(base_path=checkpoint_test_dir)

        checkpoint_id = "integrity_test_001"
        task_id = "integrity_task_001"

        # Save valid checkpoint
        repo.save_checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_state=sample_checkpoint_data,
            training_metrics={"final_accuracy": 0.65, "training_time": 1500.0},
            lora_config={"rank": 8}
        )

        # Verify integrity of valid checkpoint
        assert repo.validate_checkpoint_integrity(checkpoint_id) is True

        # Test loading valid checkpoint
        loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)
        assert loaded_metadata.checkpoint_id == checkpoint_id

        # Corrupt the checkpoint file
        checkpoint_path = checkpoint_test_dir / "ttt" / task_id / f"{checkpoint_id}.pt"
        checkpoint_path.stat().st_size

        # Write corrupted data
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted_checkpoint_data_not_valid_pytorch")

        # Verify corruption is detected
        assert repo.validate_checkpoint_integrity(checkpoint_id) is False

        # Test that loading corrupted checkpoint raises appropriate error
        with pytest.raises(Exception):
            repo.load_checkpoint(checkpoint_id)

        # Test partial corruption (truncated file)
        with open(checkpoint_path, "rb") as f:
            original_data = f.read()

        # Write only first half of original data
        with open(checkpoint_path, "wb") as f:
            f.write(original_data[:len(original_data)//2])

        # Should also detect this corruption
        assert repo.validate_checkpoint_integrity(checkpoint_id) is False

    def test_concurrent_checkpoint_operations(self, checkpoint_test_dir, sample_checkpoint_data):
        """Test concurrent checkpoint save/load operations."""
        import concurrent.futures
        import threading

        repo = CheckpointRepository(base_path=checkpoint_test_dir)
        results = []
        errors = []
        lock = threading.Lock()

        def save_checkpoint_worker(worker_id):
            """Worker function for concurrent checkpoint saving."""
            try:
                checkpoint_id = f"concurrent_test_{worker_id:03d}"
                task_id = f"concurrent_task_{worker_id % 3}"  # Distribute across 3 tasks

                metrics = {
                    "final_accuracy": 0.5 + (worker_id * 0.01),
                    "training_time": 1000.0 + worker_id * 10,
                    "worker_id": worker_id
                }

                metadata = repo.save_checkpoint(
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    model_state=sample_checkpoint_data,
                    training_metrics=metrics,
                    lora_config={"rank": 8, "worker_id": worker_id}
                )

                with lock:
                    results.append({
                        "worker_id": worker_id,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "success": True,
                        "metadata": metadata
                    })

                return True

            except Exception as e:
                with lock:
                    errors.append({
                        "worker_id": worker_id,
                        "error": str(e)
                    })
                return False

        # Run concurrent saves
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(save_checkpoint_worker, i) for i in range(20)]
            [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all operations succeeded
        assert len(errors) == 0, f"Concurrent save errors: {errors}"
        assert len(results) == 20
        assert all(r["success"] for r in results)

        # Verify all checkpoints are accessible
        all_checkpoints = repo.list_checkpoints()
        assert len(all_checkpoints) == 20

        # Test concurrent loading
        def load_checkpoint_worker(checkpoint_info):
            """Worker function for concurrent checkpoint loading."""
            try:
                checkpoint_id = checkpoint_info["checkpoint_id"]
                loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)

                return {
                    "checkpoint_id": checkpoint_id,
                    "success": True,
                    "loaded_accuracy": loaded_data["training_metrics"]["final_accuracy"],
                    "worker_id": loaded_data["training_metrics"]["worker_id"]
                }

            except Exception as e:
                return {
                    "checkpoint_id": checkpoint_info["checkpoint_id"],
                    "success": False,
                    "error": str(e)
                }

        # Run concurrent loads
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            load_futures = [executor.submit(load_checkpoint_worker, r) for r in results]
            load_results = [future.result() for future in concurrent.futures.as_completed(load_futures)]

        # Verify all loads succeeded
        successful_loads = [r for r in load_results if r["success"]]
        assert len(successful_loads) == 20

        # Verify data integrity in concurrent operations
        for result, load_result in zip(results, successful_loads, strict=False):
            original_worker_id = result["metadata"].training_metrics.get("worker_id")
            loaded_worker_id = load_result["worker_id"]
            assert original_worker_id == loaded_worker_id


class TestTTTAdapterIntegration:
    """Test TTT Adapter integration with checkpoints."""

    @pytest.fixture
    def ttt_config(self, checkpoint_test_dir):
        """Create TTT configuration for testing."""
        return TTTConfig(
            model_name="gpt2",  # Small model for testing
            device="cpu",
            quantization=False,
            num_epochs=2,
            batch_size=1,
            learning_rate=1e-4,
            checkpoint_dir=checkpoint_test_dir / "ttt_checkpoints",
            cache_dir=checkpoint_test_dir / "ttt_cache",
            max_examples=3
        )

    def test_ttt_adapter_initialization(self, ttt_config):
        """Test TTT adapter initialization and model loading."""
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter
                adapter = TTTAdapter(config=ttt_config)

                # Test initialization
                assert adapter.config == ttt_config
                assert adapter.device.type == "cpu"
                assert adapter.model is None  # Not loaded yet
                assert adapter.tokenizer is None
                assert len(adapter.adaptations) == 0

                # Test model initialization
                adapter.initialize_model()

                # Verify model and tokenizer are loaded
                assert adapter.model is not None
                assert adapter.tokenizer is not None
                assert adapter.tokenizer.pad_token == "[EOS]"

                # Verify directories were created
                assert ttt_config.checkpoint_dir.exists()
                assert ttt_config.cache_dir.exists()

                # Cleanup
                adapter.cleanup()
                assert adapter.model is None
                assert adapter.tokenizer is None

    def test_ttt_adapter_task_adaptation(self, ttt_config, arc_task_for_checkpoints):
        """Test TTT adapter task adaptation process."""
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
                mock_tokenizer_inst.decode.return_value = "0 1 0\n1 0 1\n0 1 0"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(100, 100))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Mock training loss progression
                loss_values = [0.8, 0.6, 0.4, 0.3, 0.25]
                loss_iter = iter(loss_values)

                def mock_forward(*args, **kwargs):
                    try:
                        loss_val = next(loss_iter)
                    except StopIteration:
                        loss_val = 0.2
                    return Mock(loss=torch.tensor(loss_val, requires_grad=True))

                mock_model_inst.side_effect = mock_forward
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create and initialize adapter
                adapter = TTTAdapter(config=ttt_config)
                adapter.initialize_model()

                # Test task adaptation
                adaptation = adapter.adapt_to_task(arc_task_for_checkpoints)

                # Verify adaptation results
                assert adaptation.task_id == arc_task_for_checkpoints.task_id
                assert adaptation.adaptation_id.startswith("ttt_adaptation_")
                assert len(adaptation.training_examples) <= ttt_config.max_examples
                assert adaptation.base_model_checkpoint == ttt_config.model_name

                # Verify adaptation metrics
                assert "final_loss" in adaptation.adaptation_metrics
                assert "training_steps" in adaptation.adaptation_metrics
                assert "adaptation_time" in adaptation.adaptation_metrics

                # Verify adaptation is stored
                assert arc_task_for_checkpoints.task_id in adapter.adaptations

                # Test adaptation reuse (should not retrain)
                start_time = datetime.now()
                cached_adaptation = adapter.adapt_to_task(arc_task_for_checkpoints)
                adaptation_time = (datetime.now() - start_time).total_seconds()

                assert cached_adaptation.adaptation_id == adaptation.adaptation_id
                assert adaptation_time < 0.1  # Should be very fast (cached)

                # Cleanup
                adapter.cleanup()

    def test_ttt_adapter_solution_generation(self, ttt_config, arc_task_for_checkpoints):
        """Test TTT adapter solution generation."""
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
                mock_tokenizer_inst.decode.return_value = "0 1 0\n1 0 1\n0 1 0"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Mock model forward for training
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(
                    loss=torch.tensor(0.3, requires_grad=True)
                )

                # Mock generation
                mock_model_inst.generate = MagicMock(
                    return_value=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                )

                mock_model.from_pretrained.return_value = mock_model_inst

                # Create and setup adapter
                adapter = TTTAdapter(config=ttt_config)
                adapter.initialize_model()

                # Adapt to task first
                adaptation = adapter.adapt_to_task(arc_task_for_checkpoints)

                # Generate solution
                solution = adapter.solve(arc_task_for_checkpoints)

                # Verify solution structure
                assert solution.task_id == arc_task_for_checkpoints.task_id
                assert solution.strategy_used == StrategyType.TEST_TIME_TRAINING
                assert len(solution.predictions) == 1  # Single test case
                assert isinstance(solution.predictions[0], list)  # Grid format
                assert solution.confidence_score >= 0.0
                assert solution.confidence_score <= 1.0

                # Verify resource usage tracking
                assert solution.resource_usage is not None
                assert solution.resource_usage.task_id == arc_task_for_checkpoints.task_id
                assert solution.resource_usage.strategy_type == StrategyType.TEST_TIME_TRAINING
                assert solution.resource_usage.cpu_seconds > 0
                assert solution.resource_usage.memory_mb > 0

                # Verify metadata
                assert "adaptation_id" in solution.metadata
                assert solution.metadata["adaptation_id"] == adaptation.adaptation_id
                assert "model_name" in solution.metadata
                assert solution.metadata["model_name"] == ttt_config.model_name

                # Test multiple solutions (should be consistent)
                solution2 = adapter.solve(arc_task_for_checkpoints)
                assert solution2.task_id == solution.task_id
                assert solution2.strategy_used == solution.strategy_used

                # Cleanup
                adapter.cleanup()

    def test_ttt_adapter_resource_management(self, ttt_config, arc_task_for_checkpoints):
        """Test TTT adapter resource management and cleanup."""
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(
                    loss=torch.tensor(0.3, requires_grad=True)
                )
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter
                adapter = TTTAdapter(config=ttt_config)

                # Verify initial state
                assert adapter.model is None
                assert adapter.tokenizer is None
                assert len(adapter.adaptations) == 0
                assert len(adapter.lora_adapters) == 0

                # Initialize and use adapter
                adapter.initialize_model()
                assert adapter.model is not None
                assert adapter.tokenizer is not None

                # Create multiple adaptations
                tasks = []
                for i in range(3):
                    task = ARCTask(
                        task_id=f"resource_test_task_{i:03d}",
                        task_source="test",
                        train_examples=arc_task_for_checkpoints.train_examples[:2],
                        test_input=arc_task_for_checkpoints.test_input
                    )
                    tasks.append(task)
                    adapter.adapt_to_task(task)

                # Verify adaptations are stored
                assert len(adapter.adaptations) == 3
                for task in tasks:
                    assert task.task_id in adapter.adaptations

                # Test partial cleanup (clear adaptations but keep model)
                adapter.clear_adaptations()
                assert len(adapter.adaptations) == 0
                assert len(adapter.lora_adapters) == 0
                assert adapter.model is not None  # Model should remain

                # Test full cleanup
                adapter.cleanup()
                assert adapter.model is None
                assert adapter.tokenizer is None
                assert len(adapter.adaptations) == 0
                assert len(adapter.lora_adapters) == 0

    def test_ttt_adapter_error_handling(self, ttt_config, arc_task_for_checkpoints):
        """Test TTT adapter error handling and recovery."""
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup tokenizer mock
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                # Test model loading failure
                mock_model.from_pretrained.side_effect = RuntimeError("Model loading failed")

                adapter = TTTAdapter(config=ttt_config)

                # Should handle model loading failure gracefully
                with pytest.raises(RuntimeError, match="Model loading failed"):
                    adapter.initialize_model()

                # Reset mock for successful loading
                mock_model.from_pretrained.side_effect = None
                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model.from_pretrained.return_value = mock_model_inst

                # Test adaptation failure
                mock_model_inst.side_effect = RuntimeError("Training failed")

                adapter.initialize_model()  # This should work now

                # Should handle training failure gracefully
                with pytest.raises(RuntimeError, match="Training failed"):
                    adapter.adapt_to_task(arc_task_for_checkpoints)

                # Verify adapter is still in valid state after error
                assert adapter.model is not None
                assert adapter.tokenizer is not None

                # Reset for successful training
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(
                    loss=torch.tensor(0.3, requires_grad=True)
                )

                # Should recover and work normally
                adaptation = adapter.adapt_to_task(arc_task_for_checkpoints)
                assert adaptation.task_id == arc_task_for_checkpoints.task_id

                # Cleanup
                adapter.cleanup()


class TestEndToEndCheckpointIntegration:
    """Test end-to-end integration of checkpoints with TTT training."""

    def test_training_with_checkpoint_saving(self, checkpoint_test_dir, arc_task_for_checkpoints):
        """Test complete training workflow with checkpoint saving."""
        # Configuration for training with checkpointing
        training_config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=3,
            batch_size=1,
            checkpoint_frequency=5,  # Save every 5 steps
            max_training_time=300,
            memory_limit_mb=10240
        )

        TTTConfig(
            model_name="gpt2",
            device="cpu",
            quantization=False,
            checkpoint_dir=checkpoint_test_dir / "training_checkpoints"
        )

        with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
            with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                    with patch("torch.save") as mock_save:
                        # Setup comprehensive mocks
                        mock_tokenizer_inst = MagicMock()
                        mock_tokenizer_inst.pad_token_id = 0
                        mock_tokenizer_inst.eos_token_id = 1
                        mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                        mock_tokenizer_inst.decode.return_value = "0 1 0\n1 0 1\n0 1 0"
                        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                        mock_model_inst = MagicMock()
                        param1 = torch.nn.Parameter(torch.randn(50, 50))
                        param1.requires_grad = True
                        mock_model_inst.parameters.return_value = [param1]
                        mock_model_inst.train = MagicMock()
                        mock_model_inst.eval = MagicMock()
                        mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                        # Mock improving training
                        step_counter = {"step": 0}

                        def mock_forward(*args, **kwargs):
                            step_counter["step"] += 1
                            # Simulate improving loss
                            loss_value = max(0.1, 0.8 - (step_counter["step"] * 0.05))
                            return Mock(loss=torch.tensor(loss_value, requires_grad=True))

                        mock_model_inst.side_effect = mock_forward
                        mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                        mock_model.from_pretrained.return_value = mock_model_inst

                        mock_lora_adapter = MagicMock()
                        mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                        mock_lora.return_value = mock_lora_adapter

                        # Track checkpoint saves
                        checkpoint_saves = []

                        def track_save(checkpoint_dict, filepath):
                            checkpoint_saves.append({
                                "filepath": str(filepath),
                                "step": step_counter["step"],
                                "loss": checkpoint_dict.get("loss", 0.0)
                            })

                        mock_save.side_effect = track_save

                        # Create services
                        model_service = TTTModelService(config={
                            "model": {"name": "gpt2", "device": "cpu", "quantization": False}
                        })
                        orchestrator = TrainingOrchestrator(model_service, config=training_config)

                        # Run training
                        results = orchestrator.train(arc_task_for_checkpoints)

                        # Verify training completed
                        assert "task_id" in results
                        assert results["task_id"] == arc_task_for_checkpoints.task_id
                        assert "final_accuracy" in results
                        assert "training_time" in results
                        assert results["epochs_completed"] >= 1

                        # Verify checkpoints were saved
                        assert len(checkpoint_saves) > 0
                        assert mock_save.call_count >= len(checkpoint_saves)

                        # Verify checkpoint saving pattern (every 5 steps)
                        if len(checkpoint_saves) > 1:
                            step_differences = [
                                checkpoint_saves[i]["step"] - checkpoint_saves[i-1]["step"]
                                for i in range(1, len(checkpoint_saves))
                            ]
                            # Most step differences should be around the checkpoint frequency
                            assert any(diff <= training_config.checkpoint_frequency + 2 for diff in step_differences)

                        # Cleanup
                        orchestrator.cleanup()

    def test_checkpoint_loading_and_resume(self, checkpoint_test_dir, sample_checkpoint_data, sample_training_metrics):
        """Test loading checkpoints and resuming training."""
        repo = CheckpointRepository(base_path=checkpoint_test_dir)

        # Save initial checkpoint
        checkpoint_id = "resume_test_001"
        task_id = "resume_task_001"

        # Simulate partially trained model
        initial_metrics = sample_training_metrics.copy()
        initial_metrics.update({
            "final_accuracy": 0.45,
            "epochs_completed": 2,
            "total_steps": 150,
            "training_time": 1800.0
        })

        metadata = repo.save_checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_state=sample_checkpoint_data,
            training_metrics=initial_metrics,
            lora_config={"rank": 8, "alpha": 16},
            tags=["resume_test", "partial_training"]
        )

        # Load checkpoint
        loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)

        # Verify loaded data contains all necessary components for resuming
        assert "model_state" in loaded_data
        assert "training_metrics" in loaded_data
        assert "lora_config" in loaded_data

        loaded_metrics = loaded_data["training_metrics"]
        assert loaded_metrics["epochs_completed"] == 2
        assert loaded_metrics["total_steps"] == 150
        assert loaded_metrics["final_accuracy"] == 0.45

        # Simulate resuming training (creating updated checkpoint)
        resumed_checkpoint_id = "resume_test_002"
        resumed_metrics = loaded_metrics.copy()
        resumed_metrics.update({
            "final_accuracy": 0.72,
            "epochs_completed": 5,
            "total_steps": 350,
            "training_time": 3600.0,
            "resumed_from": checkpoint_id
        })

        # Save resumed checkpoint
        resumed_metadata = repo.save_checkpoint(
            checkpoint_id=resumed_checkpoint_id,
            task_id=task_id,
            model_state=sample_checkpoint_data,
            training_metrics=resumed_metrics,
            lora_config={"rank": 8, "alpha": 16},
            tags=["resume_test", "completed_training"]
        )

        # Verify training progression
        assert resumed_metadata.accuracy > metadata.accuracy
        assert resumed_metadata.training_time > metadata.training_time

        # List checkpoints to see progression
        checkpoints = repo.list_checkpoints(task_id=task_id)
        assert len(checkpoints) == 2

        # Sort by creation time
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x.created_at)
        assert checkpoints_sorted[0].checkpoint_id == checkpoint_id
        assert checkpoints_sorted[1].checkpoint_id == resumed_checkpoint_id
        assert checkpoints_sorted[1].accuracy > checkpoints_sorted[0].accuracy
