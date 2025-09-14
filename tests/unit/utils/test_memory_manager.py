"""Unit tests for memory manager."""
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.memory_manager import AdaptiveBatchSizer, MemoryManager


class TestMemoryManager:
    """Test memory management functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def memory_manager(self, device):
        """Create memory manager for testing."""
        return MemoryManager(
            device=device,
            memory_limit_gb=10.0,
            safety_margin=0.9,
            enable_monitoring=False
        )

    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.memory_limit_gb == 10.0
        assert memory_manager.safety_margin == 0.9
        assert memory_manager.safe_memory_gb == 9.0  # 10 * 0.9
        assert memory_manager.warning_threshold == 0.7
        assert memory_manager.critical_threshold == 0.85
        assert memory_manager.oom_count == 0

    def test_get_memory_usage_cpu(self, memory_manager):
        """Test CPU memory usage retrieval."""
        usage = memory_manager.get_memory_usage()

        assert "allocated_gb" in usage
        assert "reserved_gb" in usage
        assert "total_gb" in usage
        assert "usage_percentage" in usage
        assert "available_gb" in usage

        assert usage["allocated_gb"] >= 0
        assert usage["available_gb"] <= memory_manager.safe_memory_gb

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_get_memory_usage_gpu(self, mock_props, mock_reserved, mock_allocated, mock_available):
        """Test GPU memory usage retrieval."""
        # Setup mocks
        mock_available.return_value = True
        mock_allocated.return_value = 2 * 1024**3  # 2GB
        mock_reserved.return_value = 3 * 1024**3   # 3GB

        mock_device_props = MagicMock()
        mock_device_props.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device_props

        # Create GPU memory manager
        device = torch.device("cuda:0")
        manager = MemoryManager(device, memory_limit_gb=10.0)

        usage = manager.get_memory_usage()

        assert usage["allocated_gb"] == 2.0
        assert usage["reserved_gb"] == 3.0
        assert usage["total_gb"] == 16.0
        assert usage["usage_percentage"] == 20.0  # 2/10 * 100
        assert usage["available_gb"] == 7.0  # 9 - 2

    def test_check_memory_pressure(self, memory_manager):
        """Test memory pressure detection."""
        # Mock different usage levels
        with patch.object(memory_manager, 'get_memory_usage') as mock_usage:
            # Normal pressure
            mock_usage.return_value = {"allocated_gb": 5.0}
            assert memory_manager.check_memory_pressure() == "normal"

            # Warning pressure
            mock_usage.return_value = {"allocated_gb": 7.5}
            assert memory_manager.check_memory_pressure() == "warning"

            # Critical pressure
            mock_usage.return_value = {"allocated_gb": 8.7}
            assert memory_manager.check_memory_pressure() == "critical"

    def test_suggest_batch_size(self, memory_manager):
        """Test batch size suggestions."""
        with patch.object(memory_manager, 'get_memory_usage') as mock_usage:
            # Normal pressure - should maintain or increase
            mock_usage.return_value = {"available_gb": 5.0}
            with patch.object(memory_manager, 'check_memory_pressure', return_value="normal"):
                suggested = memory_manager.suggest_batch_size(16, 100)  # 100MB per sample
                assert suggested > 0
                assert suggested <= 40  # 5GB * 1024 * 0.8 / 100

            # Critical pressure - should reduce
            with patch.object(memory_manager, 'check_memory_pressure', return_value="critical"):
                suggested = memory_manager.suggest_batch_size(16, 100)
                assert suggested <= 8  # Half of current

    def test_oom_protected_context(self, memory_manager):
        """Test OOM protection context manager."""
        # Test successful execution
        with memory_manager.oom_protected():
            result = 1 + 1
        assert result == 2
        assert memory_manager.oom_count == 0

        # Test OOM handling
        fallback_called = False
        def fallback():
            nonlocal fallback_called
            fallback_called = True

        # With fallback, error should be caught
        with memory_manager.oom_protected(fallback_fn=fallback):
            raise RuntimeError("CUDA out of memory")

        assert fallback_called
        assert memory_manager.oom_count == 1

        # Without fallback, error should propagate after too many OOMs
        memory_manager.oom_count = 4  # Exceed limit
        with pytest.raises(RuntimeError):
            with memory_manager.oom_protected():
                raise RuntimeError("CUDA out of memory")

    def test_clear_cache(self, memory_manager):
        """Test cache clearing."""
        with patch("gc.collect") as mock_gc:
            memory_manager.clear_cache()
            mock_gc.assert_called_once()


class TestAdaptiveBatchSizer:
    """Test adaptive batch sizing functionality."""

    @pytest.fixture
    def memory_manager(self):
        """Create mock memory manager."""
        manager = MagicMock()
        manager.check_memory_pressure.return_value = "normal"
        return manager

    @pytest.fixture
    def batch_sizer(self, memory_manager):
        """Create adaptive batch sizer."""
        return AdaptiveBatchSizer(
            memory_manager=memory_manager,
            initial_batch_size=16,
            min_batch_size=1,
            max_batch_size=64
        )

    def test_initialization(self, batch_sizer):
        """Test batch sizer initialization."""
        assert batch_sizer.current_batch_size == 16
        assert batch_sizer.min_batch_size == 1
        assert batch_sizer.max_batch_size == 64
        assert len(batch_sizer.successful_sizes) == 0
        assert len(batch_sizer.failed_sizes) == 0

    def test_adjust_batch_size_success(self, batch_sizer):
        """Test batch size adjustment on success."""
        # First success - should try to increase
        new_size = batch_sizer.adjust_batch_size(success=True)
        assert new_size == 19  # 16 * 1.2
        assert 16 in batch_sizer.successful_sizes

        # Further successes
        batch_sizer.current_batch_size = 32
        new_size = batch_sizer.adjust_batch_size(success=True)
        assert new_size == 38  # 32 * 1.2

    def test_adjust_batch_size_failure(self, batch_sizer):
        """Test batch size adjustment on failure."""
        # Failure - should reduce
        new_size = batch_sizer.adjust_batch_size(success=False)
        assert new_size == 8  # 16 / 2
        assert 16 in batch_sizer.failed_sizes

        # Further reduction
        new_size = batch_sizer.adjust_batch_size(success=False)
        assert new_size == 4  # 8 / 2

        # Should not go below minimum
        batch_sizer.current_batch_size = 2
        new_size = batch_sizer.adjust_batch_size(success=False)
        assert new_size == 1  # min_batch_size

    def test_get_optimal_batch_size(self, batch_sizer):
        """Test getting optimal batch size."""
        # No successful sizes yet
        assert batch_sizer.get_optimal_batch_size() == 16

        # With successful sizes
        batch_sizer.successful_sizes = {8, 16, 32}
        assert batch_sizer.get_optimal_batch_size() == 32

    def test_memory_pressure_limits(self, batch_sizer):
        """Test batch size limits based on memory pressure."""
        # Critical pressure should prevent increase
        batch_sizer.memory_manager.check_memory_pressure.return_value = "critical"
        batch_sizer.successful_sizes.add(16)

        new_size = batch_sizer.adjust_batch_size(success=True)
        assert new_size == 16  # Should not increase

        # Max limit enforcement
        batch_sizer.current_batch_size = 60
        batch_sizer.memory_manager.check_memory_pressure.return_value = "normal"
        new_size = batch_sizer.adjust_batch_size(success=True)
        assert new_size == 64  # Should cap at max
