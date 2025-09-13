"""Integration tests for TTT training pipeline."""
import gc
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.domain.models import ARCTask
from src.domain.services.ttt_service import TTTModelService
from src.domain.services.training_orchestrator import TrainingOrchestrator, TrainingConfig
from src.adapters.strategies.ttt_adapter import TTTAdapter
from src.adapters.repositories.checkpoint_repository import CheckpointRepository


@pytest.fixture
def sample_task():
    """Create a sample ARC task for testing."""
    return ARCTask(
        task_id="integration_test_001",
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


@pytest.fixture
def integration_test_dir():
    """Create temporary directory for integration tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(integration_test_dir):
    """Create test configuration with temporary directories."""
    return {
        "model": {
            "name": "gpt2",  # Use small model for testing
            "device": "cpu",
            "quantization": False,
            "cache_dir": str(integration_test_dir / "models")
        },
        "resources": {
            "max_memory_gb": 10
        }
    }


@pytest.fixture
def training_config():
    """Create training configuration for tests."""
    return TrainingConfig(
        learning_rate=1e-4,
        num_epochs=1,  # Quick test
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        validation_frequency=10,
        checkpoint_frequency=10,
        max_training_time=300,  # 5 minutes for test
        target_accuracy=0.0,  # No accuracy requirement for test
        memory_limit_mb=10240,
        mixed_precision=False,  # Disable for CPU
        gradient_checkpointing=False
    )


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Automatically cleanup resources after each test."""
    yield
    # Force garbage collection and clear GPU cache if available
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestTTTPipelineIntegration:
    """Integration tests for the complete TTT pipeline."""
    
    @pytest.mark.slow
    @patch("src.domain.services.ttt_service.AutoModelForCausalLM")
    @patch("src.domain.services.ttt_service.AutoTokenizer")
    def test_model_service_integration(self, mock_tokenizer_class, mock_model_class, test_config):
        """Test TTT model service integration."""
        service = None
        try:
            # Setup mocks
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "[EOS]"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_model = MagicMock(spec=torch.nn.Module)
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Test service
            service = TTTModelService(config=test_config)
            
            # Validate device setup
            assert service.device.type == "cpu"
            assert service.validate_gpu_constraints() is False  # No GPU in test
            
            # Load model
            model, tokenizer = service.load_model()
            assert model is not None
            assert tokenizer is not None
            
            # Check memory profile
            profile = service.get_memory_profile()
            assert "current_usage_gb" in profile
            assert "max_limit_gb" in profile
            assert profile["max_limit_gb"] == 10
            
        finally:
            # Cleanup
            if service is not None:
                service.cleanup()
                assert service.model is None
                assert service.tokenizer is None
    
    @pytest.mark.slow
    def test_ttt_adapter_integration(self, sample_task, integration_test_dir):
        """Test TTT adapter integration with real task."""
        from src.adapters.strategies.ttt_adapter import TTTConfig
        
        config = TTTConfig(
            model_name="gpt2",
            device="cpu",
            quantization=False,
            num_epochs=1,
            batch_size=1,
            checkpoint_dir=integration_test_dir / "checkpoints",
            cache_dir=integration_test_dir / "cache"
        )
        
        adapter = None
        try:
            with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
                with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                    # Setup mocks
                    mock_tokenizer_inst = MagicMock()
                    mock_tokenizer_inst.pad_token = None
                    mock_tokenizer_inst.eos_token = "[EOS]"
                    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst
                    
                    mock_model_inst = MagicMock(spec=torch.nn.Module)
                    mock_model.from_pretrained.return_value = mock_model_inst
                    
                    # Test adapter
                    adapter = TTTAdapter(config=config)
                    
                    # Initialize model
                    adapter.initialize_model()
                    assert adapter.model is not None
                    assert adapter.tokenizer is not None
                    
                    # Adapt to task
                    adaptation = adapter.adapt_to_task(sample_task)
                    assert adaptation.task_id == sample_task.task_id
                    assert len(adaptation.training_examples) == 2
                    
                    # Generate solution
                    solution = adapter.solve(sample_task)
                    assert solution.task_id == sample_task.task_id
                    assert len(solution.predictions) == 1
                    assert solution.strategy_used.value == "ttt"
                    
        finally:
            # Cleanup
            if adapter is not None:
                adapter.cleanup()
    
    @pytest.mark.slow
    @patch("src.domain.services.training_orchestrator.torch.save")
    def test_training_orchestrator_integration(self, mock_save, sample_task, test_config, training_config):
        """Test training orchestrator integration."""
        model_service = None
        orchestrator = None
        try:
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
                        
                        # Create mock model with required attributes
                        mock_model_inst = MagicMock()
                        # Create proper parameter mocks
                        param1 = torch.nn.Parameter(torch.randn(10, 10))
                        param1.requires_grad = True
                        mock_model_inst.parameters.return_value = [param1]
                        mock_model_inst.train = MagicMock()
                        mock_model_inst.eval = MagicMock()
                        mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                        
                        # Mock forward pass
                        mock_output = MagicMock()
                        mock_output.loss = torch.tensor(0.5, requires_grad=True)
                        mock_model_inst.return_value = mock_output
                        
                        # Mock generation
                        mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                        
                        mock_model.from_pretrained.return_value = mock_model_inst
                        
                        # Mock LoRA adapter
                        mock_lora_adapter = MagicMock()
                        mock_lora_adapter.get_trainable_parameters.return_value = [param1]
                        mock_lora.return_value = mock_lora_adapter
                        
                        # Create services
                        model_service = TTTModelService(config=test_config)
                        orchestrator = TrainingOrchestrator(model_service, config=training_config)
                        
                        # Run training
                        results = orchestrator.train(sample_task)
                        
                        # Verify results
                        assert "task_id" in results
                        assert results["task_id"] == sample_task.task_id
                        assert "final_accuracy" in results
                        assert "training_time" in results
                        assert "total_steps" in results
                        assert results["epochs_completed"] >= 1
                        
        finally:
            # Cleanup
            if orchestrator is not None:
                orchestrator.cleanup()
            if model_service is not None:
                model_service.cleanup()
    
    def test_checkpoint_repository_integration(self, integration_test_dir):
        """Test checkpoint repository integration."""
        repo = CheckpointRepository(base_path=integration_test_dir)
        
        try:
            # Create test checkpoint data
            checkpoint_id = "test_checkpoint_001"
            task_id = "test_task_001"
            
            model_state = {
                "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
                "lora_adapter": {"lora_A": torch.randn(8, 10), "lora_B": torch.randn(10, 8)}
            }
            
            training_metrics = {
                "model_name": "test-model",
                "final_accuracy": 0.45,
                "training_time": 3600.0,
                "final_memory_mb": 5000.0
            }
            
            lora_config = {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1
            }
            
            # Save checkpoint
            metadata = repo.save_checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                model_state=model_state,
                training_metrics=training_metrics,
                lora_config=lora_config,
                tags=["test", "integration"]
            )
            
            assert metadata.checkpoint_id == checkpoint_id
            assert metadata.task_id == task_id
            assert metadata.accuracy == 0.45
            
            # Load checkpoint
            loaded_data, loaded_metadata = repo.load_checkpoint(checkpoint_id)
            assert loaded_metadata.checkpoint_id == checkpoint_id
            assert "model_state" in loaded_data
            assert "training_metrics" in loaded_data
            
            # List checkpoints
            checkpoints = repo.list_checkpoints(task_id=task_id)
            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint_id == checkpoint_id
            
            # Get best checkpoint
            best = repo.get_best_checkpoint(task_id)
            assert best.checkpoint_id == checkpoint_id
            
            # Validate integrity
            assert repo.validate_checkpoint_integrity(checkpoint_id)
            
            # Cleanup
            stats = repo.cleanup_storage(keep_best_per_task=0)
            assert stats["deleted_count"] == 1
            
        except Exception:
            # Ensure cleanup even if test fails
            raise
    
    @pytest.mark.slow
    def test_end_to_end_pipeline(self, sample_task, integration_test_dir):
        """Test complete end-to-end TTT pipeline."""
        adapter = None
        try:
            # This test would require actual models, so we'll use extensive mocking
            with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM"):
                with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer"):
                    with patch("src.domain.services.ttt_service.AutoModelForCausalLM"):
                        with patch("src.domain.services.ttt_service.AutoTokenizer"):
                            # Create components
                            from src.adapters.strategies.ttt_adapter import TTTConfig
                            
                            ttt_config = TTTConfig(
                                model_name="gpt2",
                                device="cpu",
                                quantization=False,
                                checkpoint_dir=integration_test_dir / "checkpoints"
                            )
                            
                            adapter = TTTAdapter(config=ttt_config)
                            
                            # Mock the actual pipeline execution
                            adaptation = MagicMock()
                            adaptation.task_id = sample_task.task_id
                            adaptation.adaptation_metrics = {"accuracy": 0.5}
                            
                            adapter.adaptations[sample_task.task_id] = adaptation
                            
                            # Test solution generation
                            solution = adapter.solve(sample_task)
                            
                            assert solution.task_id == sample_task.task_id
                            assert solution.strategy_used.value == "ttt"
                            assert solution.resource_usage is not None
                            
                            # Verify resource tracking
                            assert solution.resource_usage.task_id == sample_task.task_id
                            assert solution.resource_usage.strategy_type.value == "ttt"
                            assert solution.resource_usage.memory_mb > 0
                            
        finally:
            # Cleanup
            if adapter is not None:
                adapter.cleanup()