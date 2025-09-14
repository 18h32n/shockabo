"""Unit tests for TTT model service."""
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.domain.services.ttt_service import TTTModelService


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "model": {
            "name": "test-model",
            "device": "auto",
            "quantization": True,
            "cache_dir": "test_cache"
        },
        "resources": {
            "max_memory_gb": 10
        }
    }


class TestTTTModelService:
    """Test suite for TTT model service."""

    def test_init(self, mock_config):
        """Test service initialization."""
        service = TTTModelService(config=mock_config)

        assert service.config == mock_config
        assert service.model is None
        assert service.tokenizer is None
        assert service.max_memory_gb == 10
        assert service.memory_monitor_enabled is True

    @patch("src.domain.services.ttt_service.torch.cuda.is_available")
    @patch("src.domain.services.ttt_service.torch.cuda.get_device_properties")
    def test_setup_device_cuda_available(self, mock_get_props, mock_cuda_available, mock_config):
        """Test device setup with CUDA available."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props

        service = TTTModelService(config=mock_config)
        device = service._setup_device()

        assert device.type == "cuda"

    @patch("src.domain.services.ttt_service.torch.cuda.is_available")
    @patch("src.domain.services.ttt_service.torch.cuda.get_device_properties")
    def test_setup_device_cuda_insufficient_memory(self, mock_get_props, mock_cuda_available, mock_config):
        """Test device setup with insufficient GPU memory."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB (insufficient)
        mock_get_props.return_value = mock_props

        service = TTTModelService(config=mock_config)
        device = service._setup_device()

        assert device.type == "cpu"

    @patch("src.domain.services.ttt_service.torch.cuda.is_available")
    def test_setup_device_no_cuda(self, mock_cuda_available, mock_config):
        """Test device setup without CUDA."""
        mock_cuda_available.return_value = False

        service = TTTModelService(config=mock_config)
        device = service._setup_device()

        assert device.type == "cpu"

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.domain.services.ttt_service.TTTModelService._get_memory_usage")
    def test_load_model_success(self, mock_memory, mock_model_class, mock_tokenizer_class, mock_config):
        """Test successful model loading."""
        # Setup mocks
        mock_memory.side_effect = [2.0, 5.0]  # Initial 2GB, final 5GB

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Test
        service = TTTModelService(config=mock_config)
        service.device = torch.device("cpu")

        model, tokenizer = service.load_model()

        # Verify
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert service.model == mock_model
        assert service.tokenizer == mock_tokenizer

        # Verify padding token was set
        assert mock_tokenizer.pad_token == "[EOS]"

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.domain.services.ttt_service.TTTModelService._get_memory_usage")
    def test_load_model_memory_exceeded(self, mock_memory, mock_model_class, mock_tokenizer_class, mock_config):
        """Test model loading with memory limit exceeded."""
        # Setup mocks
        mock_memory.side_effect = [2.0, 12.0]  # Initial 2GB, final 12GB (exceeds 10GB limit)

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Test
        service = TTTModelService(config=mock_config)
        service.device = torch.device("cpu")

        with pytest.raises(MemoryError):
            service.load_model()

        # Verify cleanup was called
        assert service.model is None
        assert service.tokenizer is None

    def test_load_model_already_loaded(self, mock_config):
        """Test loading model when already loaded."""
        service = TTTModelService(config=mock_config)

        # Set existing model
        service.model = MagicMock()
        service.tokenizer = MagicMock()

        model, tokenizer = service.load_model()

        assert model == service.model
        assert tokenizer == service.tokenizer

    @patch("src.domain.services.ttt_service.psutil.Process")
    def test_get_memory_usage_cpu(self, mock_process, mock_config):
        """Test memory usage calculation for CPU."""
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 4 * 1024**3  # 4GB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info

        service = TTTModelService(config=mock_config)
        service.device = torch.device("cpu")

        memory_gb = service._get_memory_usage()

        assert memory_gb == 4.0

    @patch("src.domain.services.ttt_service.torch.cuda.memory_allocated")
    def test_get_memory_usage_gpu(self, mock_cuda_memory, mock_config):
        """Test memory usage calculation for GPU."""
        mock_cuda_memory.return_value = 3 * 1024**3  # 3GB in bytes

        service = TTTModelService(config=mock_config)
        service.device = torch.device("cuda")

        memory_gb = service._get_memory_usage()

        assert memory_gb == 3.0

    @patch("gc.collect")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    @patch("src.domain.services.ttt_service.TTTModelService._get_memory_usage")
    def test_optimize_memory(self, mock_memory, mock_cuda_sync, mock_cuda_empty, mock_gc, mock_config):
        """Test memory optimization."""
        mock_memory.side_effect = [8.0, 6.0]  # Before and after optimization

        service = TTTModelService(config=mock_config)
        service.device = torch.device("cuda")

        stats = service.optimize_memory()

        assert stats["before_mb"] == 8192.0  # 8GB in MB
        assert stats["after_mb"] == 6144.0   # 6GB in MB
        assert stats["freed_mb"] == 2048.0   # 2GB freed

        mock_gc.assert_called_once()
        mock_cuda_empty.assert_called_once()
        mock_cuda_sync.assert_called_once()

    @patch("src.domain.services.ttt_service.torch.cuda.is_available")
    @patch("src.domain.services.ttt_service.torch.cuda.get_device_properties")
    def test_validate_gpu_constraints_meets_requirement(self, mock_get_props, mock_cuda_available, mock_config):
        """Test GPU validation with sufficient memory."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props

        service = TTTModelService(config=mock_config)
        result = service.validate_gpu_constraints()

        assert result is True

    @patch("src.domain.services.ttt_service.torch.cuda.is_available")
    def test_validate_gpu_constraints_no_gpu(self, mock_cuda_available, mock_config):
        """Test GPU validation without GPU."""
        mock_cuda_available.return_value = False

        service = TTTModelService(config=mock_config)
        result = service.validate_gpu_constraints()

        assert result is False

    def test_prepare_for_training(self, mock_config):
        """Test preparation for training."""
        service = TTTModelService(config=mock_config)

        # Set mock model
        mock_model = MagicMock()
        service.model = mock_model

        service.prepare_for_training()

        mock_model.train.assert_called_once()

    def test_prepare_for_training_no_model(self, mock_config):
        """Test preparation for training without model."""
        service = TTTModelService(config=mock_config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.prepare_for_training()

    @patch("torch.set_grad_enabled")
    def test_prepare_for_inference(self, mock_grad_enabled, mock_config):
        """Test preparation for inference."""
        service = TTTModelService(config=mock_config)

        # Set mock model
        mock_model = MagicMock()
        service.model = mock_model

        service.prepare_for_inference()

        mock_model.eval.assert_called_once()
        mock_grad_enabled.assert_called_once_with(False)

    @patch("gc.collect")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    def test_cleanup(self, mock_cuda_available, mock_cuda_empty, mock_gc, mock_config):
        """Test cleanup functionality."""
        mock_cuda_available.return_value = True

        service = TTTModelService(config=mock_config)

        # Set some state
        service.model = MagicMock()
        service.tokenizer = MagicMock()

        service.cleanup()

        assert service.model is None
        assert service.tokenizer is None
        mock_gc.assert_called_once()
        mock_cuda_empty.assert_called_once()
