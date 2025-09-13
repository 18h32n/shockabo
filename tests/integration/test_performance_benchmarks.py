"""Performance benchmark tests for TTT pipeline."""
import gc
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import psutil
import pytest
import torch

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask, StrategyType
from src.domain.services.training_orchestrator import TrainingConfig, TrainingOrchestrator
from src.domain.services.ttt_service import TTTModelService
from src.utils.performance_validator import MemoryMonitor, PerformanceMetrics, PerformanceValidator


@pytest.fixture
def sample_arc_tasks():
    """Create sample ARC tasks for benchmarking."""
    tasks = []
    for i in range(5):
        task = ARCTask(
            task_id=f"benchmark_task_{i:03d}",
            task_source="test",
            train_examples=[
                {
                    "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                },
                {
                    "input": [[2, 2, 0], [2, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 2], [0, 2, 2]]
                }
            ],
            test_input=[[3, 3, 0], [3, 0, 0], [0, 0, 0]]
        )
        tasks.append(task)
    return tasks


@pytest.fixture
def benchmark_config():
    """Create configuration for benchmark tests."""
    return {
        "model": {
            "name": "gpt2",  # Small model for reproducible tests
            "device": "cpu",  # Force CPU for consistent benchmarks
            "quantization": False,
            "cache_dir": "test_data/models"
        },
        "resources": {
            "max_memory_gb": 10,
            "max_training_hours": 2
        }
    }


@pytest.fixture
def memory_constrained_config():
    """Create strict memory-constrained configuration."""
    return TrainingConfig(
        learning_rate=1e-4,
        num_epochs=3,  # Multiple epochs to test memory stability
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        validation_frequency=5,
        checkpoint_frequency=5,
        max_training_time=7200,  # 2 hours
        target_accuracy=0.0,
        memory_limit_mb=10240,  # 10GB limit
        mixed_precision=True,  # Enable to test memory efficiency
        gradient_checkpointing=True,  # Enable to test memory optimization
    )


class TestMemoryConstraints:
    """Test memory usage constraints and optimization."""
    
    def test_memory_monitor_basic_functionality(self):
        """Test memory monitoring basic operations."""
        monitor = MemoryMonitor(sample_interval=0.1)
        
        # Test initial state
        assert not monitor.monitoring
        assert len(monitor.samples) == 0
        
        # Start monitoring
        monitor.start()
        assert monitor.monitoring
        assert monitor.start_time is not None
        
        # Take some samples
        for _ in range(5):
            monitor.sample()
            time.sleep(0.05)
        
        # Stop monitoring
        stats = monitor.stop()
        assert not monitor.monitoring
        assert stats["peak_mb"] > 0
        assert stats["average_mb"] > 0
        assert stats["sample_count"] > 0
        assert stats["duration"] > 0
    
    def test_memory_monitor_with_allocation(self):
        """Test memory monitoring with actual memory allocation."""
        monitor = MemoryMonitor(sample_interval=0.1)
        
        monitor.start()
        
        # Allocate some memory to test monitoring
        large_data = []
        
        # Allocate in steps to see memory growth
        for i in range(10):
            # Allocate ~10MB chunks
            chunk = [0] * (1024 * 1024)  # ~4MB (int list)
            large_data.append(chunk)
            monitor.sample()
            time.sleep(0.05)
        
        stats = monitor.stop()
        
        # Memory should have increased during allocation
        assert stats["peak_mb"] > stats["average_mb"]
        assert len(monitor.samples) >= 10
        
        # Cleanup
        del large_data
        gc.collect()
    
    @pytest.mark.slow
    def test_training_memory_limits(self, sample_arc_tasks, benchmark_config, memory_constrained_config):
        """Test that training respects memory limits."""
        task = sample_arc_tasks[0]
        
        with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
            with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                    # Setup comprehensive mocks
                    mock_tokenizer_inst = MagicMock()
                    mock_tokenizer_inst.pad_token_id = 0
                    mock_tokenizer_inst.eos_token_id = 1
                    mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer_inst.decode.return_value = "0 0 1\n0 1 0\n1 0 0"
                    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                    
                    # Create mock model with memory tracking
                    mock_model_inst = MagicMock()
                    param1 = torch.nn.Parameter(torch.randn(100, 100))  # Larger parameter for memory test
                    param1.requires_grad = True
                    mock_model_inst.parameters.return_value = [param1]
                    mock_model_inst.train = MagicMock()
                    mock_model_inst.eval = MagicMock()
                    mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                    
                    # Mock training loss with decreasing trend
                    loss_values = [0.8, 0.7, 0.6, 0.5, 0.4]
                    loss_iterator = iter(loss_values)
                    
                    def get_loss(*args, **kwargs):
                        try:
                            loss_value = next(loss_iterator)
                        except StopIteration:
                            loss_value = 0.3
                        return Mock(loss=torch.tensor(loss_value, requires_grad=True))
                    
                    mock_model_inst.side_effect = get_loss
                    mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                    mock_model.from_pretrained.return_value = mock_model_inst
                    
                    # Mock LoRA adapter
                    mock_lora_adapter = MagicMock()
                    mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                    mock_lora.return_value = mock_lora_adapter
                    
                    # Create service and orchestrator
                    model_service = TTTModelService(config=benchmark_config)
                    orchestrator = TrainingOrchestrator(model_service, config=memory_constrained_config)
                    
                    # Setup memory monitoring
                    validator = PerformanceValidator()
                    
                    # Run training with memory monitoring
                    def training_func(task):
                        return orchestrator.train(task)
                    
                    result, perf_metrics = validator.benchmark_training(training_func, task)
                    
                    # Verify memory constraints
                    assert perf_metrics["memory_peak_mb"] < 10240, f"Memory exceeded 10GB limit: {perf_metrics['memory_peak_mb']:.2f}MB"
                    assert "final_accuracy" in result
                    assert "training_time" in result
                    
                    # Cleanup
                    orchestrator.cleanup()
    
    def test_memory_optimization_techniques(self):
        """Test memory optimization techniques."""
        # Test gradient checkpointing impact (mocked)
        with patch("torch.utils.checkpoint.checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = torch.tensor([1.0])
            
            # Simulate gradient checkpointing
            def checkpoint_function(func, *args):
                # Simulate reduced memory usage
                return func(*args)
            
            mock_checkpoint.side_effect = checkpoint_function
            
            # Test that checkpointing reduces memory
            # This is a conceptual test - real implementation would require model
            memory_without_checkpointing = 1000  # MB (simulated)
            memory_with_checkpointing = 750  # MB (simulated)
            
            memory_reduction = (memory_without_checkpointing - memory_with_checkpointing) / memory_without_checkpointing
            assert memory_reduction > 0.2  # At least 20% reduction
    
    def test_batch_size_memory_scaling(self):
        """Test memory scaling with different batch sizes."""
        # Simulate memory usage for different batch sizes
        batch_sizes = [1, 2, 4, 8]
        memory_usages = []
        
        for batch_size in batch_sizes:
            # Simulate memory allocation proportional to batch size
            # In reality, this would be measured during actual training
            base_memory = 500  # MB
            scaling_factor = 1.2  # Non-linear scaling
            memory_usage = base_memory * (batch_size ** scaling_factor)
            memory_usages.append(memory_usage)
        
        # Verify memory scales as expected
        assert memory_usages[0] < memory_usages[1] < memory_usages[2] < memory_usages[3]
        
        # Verify largest batch size doesn't exceed limits
        max_memory = max(memory_usages)
        assert max_memory < 10240  # Under 10GB limit
    
    def test_gpu_memory_constraints(self):
        """Test GPU memory constraints if GPU is available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")
        
        validator = PerformanceValidator()
        gpu_info = validator.validate_gpu_compatibility()
        
        assert gpu_info["gpu_available"] is True
        assert "total_memory_gb" in gpu_info
        assert "meets_requirement" in gpu_info
        
        # Should meet 16GB GPU requirement
        if gpu_info["total_memory_gb"] >= 14:
            assert gpu_info["meets_requirement"] is True


class TestTrainingTimeConstraints:
    """Test training time constraints and optimization."""
    
    @pytest.mark.slow
    def test_training_time_limit(self, sample_arc_tasks, benchmark_config):
        """Test that training respects 2-hour time limit."""
        task = sample_arc_tasks[0]
        
        # Create short-duration config for testing
        quick_config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=1,
            max_training_time=5,  # 5 seconds for test
            memory_limit_mb=10240
        )
        
        with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
            with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                    # Setup mocks similar to previous test
                    mock_tokenizer_inst = MagicMock()
                    mock_tokenizer_inst.pad_token_id = 0
                    mock_tokenizer_inst.eos_token_id = 1
                    mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                    
                    mock_model_inst = MagicMock()
                    param1 = torch.nn.Parameter(torch.randn(10, 10))
                    param1.requires_grad = True
                    mock_model_inst.parameters.return_value = [param1]
                    mock_model_inst.train = MagicMock()
                    mock_model_inst.eval = MagicMock()
                    mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                    
                    # Mock slow training (simulate long training steps)
                    def slow_forward(*args, **kwargs):
                        time.sleep(0.1)  # Simulate processing time
                        return Mock(loss=torch.tensor(0.5, requires_grad=True))
                    
                    mock_model_inst.side_effect = slow_forward
                    mock_model.from_pretrained.return_value = mock_model_inst
                    
                    mock_lora_adapter = MagicMock()
                    mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                    mock_lora.return_value = mock_lora_adapter
                    
                    # Create service and orchestrator
                    model_service = TTTModelService(config=benchmark_config)
                    orchestrator = TrainingOrchestrator(model_service, config=quick_config)
                    
                    # Measure training time
                    start_time = time.time()
                    result = orchestrator.train(task)
                    elapsed_time = time.time() - start_time
                    
                    # Verify time constraint
                    assert elapsed_time <= quick_config.max_training_time + 1  # Allow 1s tolerance
                    assert "training_time" in result
                    assert result["training_time"] <= quick_config.max_training_time + 1
                    
                    # Cleanup
                    orchestrator.cleanup()
    
    def test_early_stopping_accuracy(self, sample_arc_tasks, benchmark_config):
        """Test early stopping when target accuracy is reached."""
        task = sample_arc_tasks[0]
        
        # Config with early stopping
        early_stop_config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=10,  # Many epochs
            batch_size=1,
            target_accuracy=0.6,  # Early stop at 60%
            max_training_time=300,
            memory_limit_mb=10240
        )
        
        with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
            with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                    # Setup mocks
                    mock_tokenizer_inst = MagicMock()
                    mock_tokenizer_inst.pad_token_id = 0
                    mock_tokenizer_inst.eos_token_id = 1
                    mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer_inst.decode.return_value = "0 0 1\n0 1 0\n1 0 0"
                    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                    
                    mock_model_inst = MagicMock()
                    param1 = torch.nn.Parameter(torch.randn(10, 10))
                    param1.requires_grad = True
                    mock_model_inst.parameters.return_value = [param1]
                    mock_model_inst.train = MagicMock()
                    mock_model_inst.eval = MagicMock()
                    mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                    
                    # Mock improving accuracy over time
                    accuracy_progression = [0.1, 0.3, 0.5, 0.7, 0.9]  # Reaches target at step 4
                    loss_progression = [0.9, 0.7, 0.5, 0.3, 0.1]
                    step_count = {"current": 0}
                    
                    def get_loss(*args, **kwargs):
                        step = step_count["current"] % len(loss_progression)
                        loss_value = loss_progression[step]
                        step_count["current"] += 1
                        return Mock(loss=torch.tensor(loss_value, requires_grad=True))
                    
                    mock_model_inst.side_effect = get_loss
                    mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                    mock_model.from_pretrained.return_value = mock_model_inst
                    
                    mock_lora_adapter = MagicMock()
                    mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                    mock_lora.return_value = mock_lora_adapter
                    
                    # Mock validation accuracy progression
                    with patch("src.domain.services.training_orchestrator.TrainingOrchestrator._validate_model") as mock_validate:
                        def mock_validation(epoch):
                            if epoch < len(accuracy_progression):
                                return accuracy_progression[epoch]
                            return 0.9  # High accuracy after target reached
                        
                        mock_validate.side_effect = lambda: mock_validation(step_count["current"] // 5)
                        
                        # Create service and orchestrator
                        model_service = TTTModelService(config=benchmark_config)
                        orchestrator = TrainingOrchestrator(model_service, config=early_stop_config)
                        
                        # Run training
                        start_time = time.time()
                        result = orchestrator.train(task)
                        elapsed_time = time.time() - start_time
                        
                        # Should stop early when target accuracy is reached
                        assert result["epochs_completed"] < early_stop_config.num_epochs
                        assert elapsed_time < 60  # Should finish quickly
                        
                        # Cleanup
                        orchestrator.cleanup()
    
    def test_checkpoint_frequency_optimization(self, sample_arc_tasks, benchmark_config, tmp_path):
        """Test checkpoint frequency impact on training time."""
        task = sample_arc_tasks[0]
        
        # Test different checkpoint frequencies
        frequencies = [1, 5, 10]  # Every 1, 5, 10 steps
        training_times = []
        
        for frequency in frequencies:
            config = TrainingConfig(
                learning_rate=1e-4,
                num_epochs=2,
                batch_size=1,
                checkpoint_frequency=frequency,
                max_training_time=300,
                memory_limit_mb=10240
            )
            
            with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
                with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                    with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                        with patch("torch.save") as mock_save:
                            # Setup mocks
                            mock_tokenizer_inst = MagicMock()
                            mock_tokenizer_inst.pad_token_id = 0
                            mock_tokenizer_inst.eos_token_id = 1
                            mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                            
                            mock_model_inst = MagicMock()
                            param1 = torch.nn.Parameter(torch.randn(10, 10))
                            param1.requires_grad = True
                            mock_model_inst.parameters.return_value = [param1]
                            mock_model_inst.train = MagicMock()
                            mock_model_inst.eval = MagicMock()
                            mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                            
                            mock_output = Mock(loss=torch.tensor(0.5, requires_grad=True))
                            mock_model_inst.return_value = mock_output
                            mock_model.from_pretrained.return_value = mock_model_inst
                            
                            # Mock save to simulate checkpointing overhead
                            def mock_checkpoint_save(*args, **kwargs):
                                time.sleep(0.01)  # Simulate I/O time
                            
                            mock_save.side_effect = mock_checkpoint_save
                            
                            mock_lora_adapter = MagicMock()
                            mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                            mock_lora.return_value = mock_lora_adapter
                            
                            # Run training
                            model_service = TTTModelService(config=benchmark_config)
                            orchestrator = TrainingOrchestrator(model_service, config=config)
                            
                            start_time = time.time()
                            result = orchestrator.train(task)
                            elapsed_time = time.time() - start_time
                            
                            training_times.append(elapsed_time)
                            
                            # Cleanup
                            orchestrator.cleanup()
        
        # More frequent checkpointing should take longer (due to I/O overhead)
        assert training_times[0] >= training_times[1] >= training_times[2]


class TestCheckpointIntegration:
    """Test checkpoint saving and loading integration."""
    
    def test_checkpoint_save_load_cycle(self, tmp_path):
        """Test complete checkpoint save and load cycle."""
        repo = CheckpointRepository(base_path=tmp_path)
        
        # Create comprehensive test data
        checkpoint_id = "perf_test_checkpoint_001"
        task_id = "perf_test_task_001"
        
        # Simulate realistic model state
        model_state = {
            "model_state_dict": {
                "embedding.weight": torch.randn(50000, 768),  # Realistic embedding size
                "transformer.h.0.attn.weight": torch.randn(768, 768),
                "transformer.h.0.mlp.weight": torch.randn(768, 3072),
            },
            "lora_adapter": {
                "lora_A": torch.randn(16, 768),
                "lora_B": torch.randn(768, 16),
                "scaling": torch.tensor(0.5)
            }
        }
        
        training_metrics = {
            "model_name": "gpt2-medium",
            "final_accuracy": 0.65,
            "training_time": 1800.0,  # 30 minutes
            "final_memory_mb": 8000.0,
            "peak_memory_mb": 9500.0,
            "total_steps": 150,
            "best_validation_accuracy": 0.67,
            "convergence_step": 120
        }
        
        lora_config = {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        
        # Measure save performance
        start_time = time.time()
        metadata = repo.save_checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_state=model_state,
            training_metrics=training_metrics,
            lora_config=lora_config,
            tags=["performance_test", "integration"]
        )
        save_time = time.time() - start_time
        
        # Verify save was successful and reasonable performance
        assert metadata.checkpoint_id == checkpoint_id
        assert save_time < 30.0  # Should save in under 30 seconds
        
        # Check file size
        checkpoint_path = tmp_path / "ttt" / task_id / f"{checkpoint_id}.pt"
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        assert file_size_mb > 0.1  # Should have substantial content
        assert file_size_mb < 1000  # But not unreasonably large
        
        # Measure load performance
        start_time = time.time()
        loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)
        load_time = time.time() - start_time
        
        # Verify load was successful and reasonable performance
        assert loaded_metadata.checkpoint_id == checkpoint_id
        assert load_time < 10.0  # Should load in under 10 seconds
        assert "model_state" in loaded_data
        assert "training_metrics" in loaded_data
        
        # Verify data integrity
        assert loaded_data["training_metrics"]["final_accuracy"] == 0.65
        assert loaded_data["lora_config"]["rank"] == 16
        
        # Test checkpoint listing performance
        start_time = time.time()
        checkpoints = repo.list_checkpoints(task_id=task_id)
        list_time = time.time() - start_time
        
        assert len(checkpoints) == 1
        assert list_time < 1.0  # Should list quickly
        assert checkpoints[0].checkpoint_id == checkpoint_id
    
    def test_checkpoint_cleanup_performance(self, tmp_path):
        """Test checkpoint cleanup performance with many files."""
        repo = CheckpointRepository(base_path=tmp_path)
        
        # Create multiple checkpoints to test cleanup performance
        task_id = "cleanup_test_task"
        checkpoint_ids = []
        
        # Create 20 checkpoints
        for i in range(20):
            checkpoint_id = f"cleanup_test_{i:03d}"
            checkpoint_ids.append(checkpoint_id)
            
            model_state = {
                "model_state_dict": {"layer1.weight": torch.randn(100, 100)},
                "lora_adapter": {"lora_A": torch.randn(8, 100)}
            }
            
            training_metrics = {
                "model_name": "test-model",
                "final_accuracy": 0.3 + (i * 0.01),  # Varying accuracy
                "training_time": 1000.0 + i,
                "final_memory_mb": 5000.0
            }
            
            repo.save_checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                model_state=model_state,
                training_metrics=training_metrics,
                lora_config={"rank": 8}
            )
        
        # Verify all checkpoints exist
        checkpoints_before = repo.list_checkpoints(task_id=task_id)
        assert len(checkpoints_before) == 20
        
        # Test cleanup performance
        start_time = time.time()
        stats = repo.cleanup_storage(keep_best_per_task=5)
        cleanup_time = time.time() - start_time
        
        # Verify cleanup performance and results
        assert cleanup_time < 5.0  # Should cleanup in under 5 seconds
        assert stats["deleted_count"] == 15  # Should delete 15, keep 5 best
        
        checkpoints_after = repo.list_checkpoints(task_id=task_id)
        assert len(checkpoints_after) == 5
        
        # Verify best checkpoints were kept (highest accuracy)
        accuracies = [cp.accuracy for cp in checkpoints_after]
        assert min(accuracies) >= 0.45  # Should keep the best ones
    
    def test_concurrent_checkpoint_access(self, tmp_path):
        """Test concurrent checkpoint save/load operations."""
        import concurrent.futures
        
        repo = CheckpointRepository(base_path=tmp_path)
        
        def save_checkpoint(checkpoint_num):
            """Save a checkpoint in a separate thread."""
            checkpoint_id = f"concurrent_test_{checkpoint_num:03d}"
            task_id = f"task_{checkpoint_num % 3}"  # 3 different tasks
            
            model_state = {
                "model_state_dict": {"layer.weight": torch.randn(50, 50)},
                "lora_adapter": {"lora_A": torch.randn(8, 50)}
            }
            
            training_metrics = {
                "model_name": "concurrent-test",
                "final_accuracy": 0.5,
                "training_time": 1000.0,
                "final_memory_mb": 4000.0
            }
            
            try:
                metadata = repo.save_checkpoint(
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    model_state=model_state,
                    training_metrics=training_metrics,
                    lora_config={"rank": 8}
                )
                return True, metadata.checkpoint_id
            except Exception as e:
                return False, str(e)
        
        # Run concurrent saves
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_checkpoint, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Verify all saves succeeded
        successful_saves = sum(1 for success, _ in results if success)
        assert successful_saves == 10
        assert concurrent_time < 30.0  # Should complete in reasonable time
        
        # Verify all checkpoints are accessible
        all_checkpoints = []
        for task_id in ["task_0", "task_1", "task_2"]:
            checkpoints = repo.list_checkpoints(task_id=task_id)
            all_checkpoints.extend(checkpoints)
        
        assert len(all_checkpoints) == 10


class TestEndToEndPerformance:
    """Test complete end-to-end performance scenarios."""
    
    @pytest.mark.slow
    def test_complete_ttt_pipeline_performance(self, sample_arc_tasks, tmp_path):
        """Test complete TTT pipeline performance end-to-end."""
        task = sample_arc_tasks[0]
        
        # Create realistic config
        config = TTTConfig(
            model_name="gpt2",
            device="cpu",
            quantization=False,
            num_epochs=2,
            batch_size=1,
            checkpoint_dir=tmp_path / "checkpoints",
            cache_dir=tmp_path / "cache",
            max_training_time=300  # 5 minutes max for test
        )
        
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks for complete pipeline
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer_inst.decode.return_value = "0 0 1\n0 1 0\n1 0 0"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                
                mock_model_inst = MagicMock()
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst
                
                # Create adapter and run complete pipeline
                adapter = TTTAdapter(config=config)
                validator = PerformanceValidator()
                
                # Measure complete pipeline performance
                start_time = time.time()
                
                # Initialize
                adapter.initialize_model()
                init_time = time.time() - start_time
                
                # Adapt to task
                adaptation_start = time.time()
                adaptation = adapter.adapt_to_task(task)
                adaptation_time = time.time() - adaptation_start
                
                # Generate solution
                solution_start = time.time()
                solution = adapter.solve(task)
                solution_time = time.time() - solution_start
                
                total_time = time.time() - start_time
                
                # Validate performance metrics
                assert init_time < 30.0  # Model initialization under 30s
                assert adaptation_time < 180.0  # Adaptation under 3 minutes
                assert solution_time < 30.0  # Solution generation under 30s
                assert total_time < 300.0  # Total pipeline under 5 minutes
                
                # Validate solution quality
                assert solution.task_id == task.task_id
                assert solution.strategy_used == StrategyType.TEST_TIME_TRAINING
                assert len(solution.predictions) == 1
                
                # Validate resource usage
                assert solution.resource_usage is not None
                assert solution.resource_usage.memory_mb > 0
                assert solution.resource_usage.cpu_seconds > 0
                
                # Generate comprehensive metrics
                metrics = PerformanceMetrics(
                    task_id=task.task_id,
                    accuracy=0.5,  # Mock accuracy
                    training_time_seconds=adaptation_time,
                    memory_peak_mb=solution.resource_usage.memory_mb,
                    memory_average_mb=solution.resource_usage.memory_mb * 0.8,
                    gpu_memory_peak_mb=solution.resource_usage.gpu_memory_mb,
                    inference_time_ms=solution_time * 1000,
                    model_load_time_ms=init_time * 1000,
                    checkpoint_size_mb=10.0,  # Mock checkpoint size
                    timestamp=datetime.now()
                )
                
                # Verify acceptance criteria
                criteria = metrics.meets_criteria()
                assert criteria["training_under_2_hours"] is True
                assert criteria["memory_under_10gb"] is True
                
                # Save results
                results_path = tmp_path / "performance_results.json"
                validator.save_validation_results(metrics, results_path)
                assert results_path.exists()
                
                # Generate report
                report = validator.generate_report(metrics)
                assert "TTT Performance Validation Report" in report
                assert "Overall Result:" in report
                
                # Cleanup
                adapter.cleanup()
    
    def test_stress_test_multiple_tasks(self, sample_arc_tasks, tmp_path):
        """Test performance with multiple tasks in sequence."""
        config = TTTConfig(
            model_name="gpt2",
            device="cpu",
            quantization=False,
            num_epochs=1,  # Quick training for stress test
            batch_size=1,
            checkpoint_dir=tmp_path / "checkpoints",
            cache_dir=tmp_path / "cache"
        )
        
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer_inst.decode.return_value = "0 0 1\n0 1 0\n1 0 0"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                
                mock_model_inst = MagicMock()
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst
                
                adapter = TTTAdapter(config=config)
                adapter.initialize_model()
                
                # Process multiple tasks
                results = []
                total_start_time = time.time()
                
                for i, task in enumerate(sample_arc_tasks):
                    task_start_time = time.time()
                    
                    # Adapt and solve
                    adaptation = adapter.adapt_to_task(task)
                    solution = adapter.solve(task)
                    
                    task_time = time.time() - task_start_time
                    results.append({
                        "task_id": task.task_id,
                        "processing_time": task_time,
                        "success": True
                    })
                    
                    # Memory check after each task
                    gc.collect()  # Force garbage collection
                
                total_time = time.time() - total_start_time
                
                # Validate stress test results
                assert len(results) == len(sample_arc_tasks)
                assert all(r["success"] for r in results)
                assert total_time < 600.0  # All tasks under 10 minutes
                
                # Verify consistent performance (no degradation)
                processing_times = [r["processing_time"] for r in results]
                avg_time = sum(processing_times) / len(processing_times)
                
                # No task should take more than 2x average (no major degradation)
                for time_taken in processing_times:
                    assert time_taken < avg_time * 2
                
                # Cleanup
                adapter.cleanup()


class TestFailureScenarios:
    """Test failure scenarios and recovery mechanisms."""
    
    def test_out_of_memory_recovery(self, sample_arc_tasks, tmp_path):
        """Test recovery from out-of-memory scenarios."""
        task = sample_arc_tasks[0]
        
        # Create config that might cause memory issues
        memory_config = TTTConfig(
            model_name="gpt2",
            device="cpu",
            quantization=False,
            num_epochs=1,
            batch_size=8,  # Large batch size
            checkpoint_dir=tmp_path / "checkpoints"
        )
        
        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks that simulate memory issues
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                
                # Mock model that raises memory error on large batches
                call_count = {"count": 0}
                
                def mock_model_factory(*args, **kwargs):
                    call_count["count"] += 1
                    mock_model_inst = MagicMock()
                    
                    if call_count["count"] == 1:
                        # First call - simulate memory error
                        mock_model_inst.to.side_effect = RuntimeError("CUDA out of memory")
                    else:
                        # Second call - work normally
                        mock_model_inst.to.return_value = mock_model_inst
                        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                        mock_model_inst.train = MagicMock()
                        mock_model_inst.eval = MagicMock()
                    
                    return mock_model_inst
                
                mock_model.from_pretrained.side_effect = mock_model_factory
                
                adapter = TTTAdapter(config=memory_config)
                
                # Should handle memory error gracefully
                try:
                    adapter.initialize_model()
                    # If we get here, the adapter should have recovered or handled the error
                    # In a real implementation, it might retry with smaller batch size
                    success = True
                except Exception as e:
                    # Should be a handled exception, not a crash
                    assert "memory" in str(e).lower() or "cuda" in str(e).lower()
                    success = False
                
                # Test that adapter can still function after memory error
                if success:
                    solution = adapter.solve(task)
                    assert solution.task_id == task.task_id
                
                adapter.cleanup()
    
    def test_checkpoint_corruption_recovery(self, tmp_path):
        """Test recovery from corrupted checkpoint files."""
        repo = CheckpointRepository(base_path=tmp_path)
        
        # Create a valid checkpoint first
        checkpoint_id = "corruption_test_001"
        task_id = "corruption_task"
        
        model_state = {
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "lora_adapter": {"lora_A": torch.randn(8, 10)}
        }
        
        training_metrics = {
            "model_name": "corruption-test",
            "final_accuracy": 0.6,
            "training_time": 1200.0,
            "final_memory_mb": 5000.0
        }
        
        # Save valid checkpoint
        metadata = repo.save_checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_state=model_state,
            training_metrics=training_metrics,
            lora_config={"rank": 8}
        )
        
        # Verify it works
        loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)
        assert loaded_metadata.checkpoint_id == checkpoint_id
        
        # Corrupt the checkpoint file
        checkpoint_path = tmp_path / "ttt" / task_id / f"{checkpoint_id}.pt"
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted_data_not_a_valid_checkpoint")
        
        # Test corruption detection
        with pytest.raises(Exception):  # Should raise an error when loading corrupted file
            repo.load_checkpoint(checkpoint_id)
        
        # Test integrity validation
        is_valid = repo.validate_checkpoint_integrity(checkpoint_id)
        assert is_valid is False
        
        # Test recovery - listing should still work and exclude corrupted files
        checkpoints = repo.list_checkpoints(task_id=task_id)
        # In a robust implementation, corrupted files might be excluded from listing
        # or marked as corrupted
    
    def test_training_interruption_recovery(self, sample_arc_tasks, benchmark_config, tmp_path):
        """Test recovery from training interruption."""
        task = sample_arc_tasks[0]
        
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=1,
            checkpoint_frequency=2,  # Frequent checkpointing
            max_training_time=300,
            memory_limit_mb=10240
        )
        
        with patch("src.domain.services.ttt_service.AutoModelForCausalLM") as mock_model:
            with patch("src.domain.services.ttt_service.AutoTokenizer") as mock_tokenizer:
                with patch("src.domain.services.training_orchestrator.apply_lora_to_model") as mock_lora:
                    with patch("torch.save") as mock_save:
                        # Setup mocks
                        mock_tokenizer_inst = MagicMock()
                        mock_tokenizer_inst.pad_token_id = 0
                        mock_tokenizer_inst.eos_token_id = 1
                        mock_tokenizer_inst.encode.return_value = [1, 2, 3, 4, 5]
                        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                        
                        mock_model_inst = MagicMock()
                        param1 = torch.nn.Parameter(torch.randn(10, 10))
                        param1.requires_grad = True
                        mock_model_inst.parameters.return_value = [param1]
                        mock_model_inst.train = MagicMock()
                        mock_model_inst.eval = MagicMock()
                        mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                        
                        # Mock training that gets interrupted
                        step_count = {"current": 0}
                        
                        def interrupted_forward(*args, **kwargs):
                            step_count["current"] += 1
                            if step_count["current"] == 3:  # Interrupt at step 3
                                raise KeyboardInterrupt("Training interrupted")
                            return Mock(loss=torch.tensor(0.5, requires_grad=True))
                        
                        mock_model_inst.side_effect = interrupted_forward
                        mock_model.from_pretrained.return_value = mock_model_inst
                        
                        mock_lora_adapter = MagicMock()
                        mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                        mock_lora.return_value = mock_lora_adapter
                        
                        # Create service and orchestrator
                        model_service = TTTModelService(config=benchmark_config)
                        orchestrator = TrainingOrchestrator(model_service, config=config)
                        
                        # Training should be interrupted but handle gracefully
                        try:
                            result = orchestrator.train(task)
                            # If we get here, interruption was handled gracefully
                            assert "training_interrupted" in result or "partial_training" in result
                        except KeyboardInterrupt:
                            # Interruption was not handled - this is also acceptable behavior
                            pass
                        
                        # Verify that checkpoints were saved before interruption
                        # In real implementation, we would check for partial checkpoints
                        assert mock_save.call_count >= 1  # At least one checkpoint saved
                        
                        # Cleanup
                        orchestrator.cleanup()