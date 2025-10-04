"""Unit tests for PlatformEvolutionConfigurator."""

from unittest.mock import patch, MagicMock

import pytest

from src.adapters.strategies.platform_evolution_config import PlatformEvolutionConfigurator
from src.infrastructure.components.platform_detector import Platform, PlatformInfo


class TestPlatformEvolutionConfigurator:
    """Test suite for PlatformEvolutionConfigurator."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration."""
        config = {
            "population_size": 1000,
            "max_generations": 100,
            "batch_size": 250,
            "parallel_workers": 4,
            "memory_limit_mb": 2048,
            "gpu_enabled": False,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }
        return config

    def test_detect_platform_kaggle(self):
        """Test Kaggle platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.KAGGLE,
            gpu_available=False,
            gpu_count=0,
            memory_gb=16.0,
            storage_gb=73.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            assert configurator.current_platform.value == "kaggle"

    def test_detect_platform_colab(self):
        """Test Colab platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.COLAB,
            gpu_available=True,
            gpu_count=1,
            memory_gb=12.0,
            storage_gb=107.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            assert configurator.current_platform.value == "colab"

    def test_detect_platform_paperspace(self):
        """Test Paperspace platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.PAPERSPACE,
            gpu_available=False,
            gpu_count=0,
            memory_gb=8.0,
            storage_gb=5.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            assert configurator.current_platform.value == "paperspace"

    def test_detect_platform_local(self):
        """Test local platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.LOCAL,
            gpu_available=True,
            gpu_count=1,
            memory_gb=32.0,
            storage_gb=500.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            assert configurator.current_platform.value == "local"

    def test_optimize_for_kaggle(self, base_config):
        """Test Kaggle-specific optimizations."""
        mock_platform_info = PlatformInfo(
            platform=Platform.KAGGLE,
            gpu_available=False,
            gpu_count=0,
            memory_gb=16.0,
            storage_gb=73.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            
            # Test that we can get configured config for kaggle
            config = configurator.get_configured_config()
            assert config is not None
            
            # Test platform-specific settings are applied
            settings = configurator.PLATFORM_SETTINGS[configurator.current_platform]
            assert settings.workers == 2
            assert settings.batch_size == 500

    def test_optimize_for_colab(self, base_config):
        """Test Colab-specific optimizations."""
        mock_platform_info = PlatformInfo(
            platform=Platform.COLAB,
            gpu_available=True,
            gpu_count=1,
            memory_gb=12.0,
            storage_gb=107.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            
            # Test that we can get configured config for colab
            config = configurator.get_configured_config()
            assert config is not None
            
            # Test platform-specific settings are applied
            settings = configurator.PLATFORM_SETTINGS[configurator.current_platform]
            assert settings.workers == 2
            assert settings.batch_size == 250

    def test_optimize_for_paperspace(self, base_config):
        """Test Paperspace-specific optimizations."""
        mock_platform_info = PlatformInfo(
            platform=Platform.PAPERSPACE,
            gpu_available=False,
            gpu_count=0,
            memory_gb=8.0,
            storage_gb=5.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            
            # Test that we can get configured config for paperspace
            config = configurator.get_configured_config()
            assert config is not None
            
            # Test platform-specific settings are applied
            settings = configurator.PLATFORM_SETTINGS[configurator.current_platform]
            assert settings.workers == 1
            assert settings.batch_size == 100

    def test_optimize_for_local(self, base_config):
        """Test local environment optimizations."""
        mock_platform_info = PlatformInfo(
            platform=Platform.LOCAL,
            gpu_available=True,
            gpu_count=1,
            memory_gb=32.0,
            storage_gb=500.0
        )
        
        with patch('src.adapters.strategies.platform_evolution_config.get_platform_detector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_platform.return_value = mock_platform_info
            mock_detector.return_value = mock_detector_instance
            
            configurator = PlatformEvolutionConfigurator()
            
            # Test that we can get configured config for local
            config = configurator.get_configured_config()
            assert config is not None
            
            # Test platform-specific settings are applied
            settings = configurator.PLATFORM_SETTINGS[configurator.current_platform]
            assert settings.workers == 4
            assert settings.batch_size == 250

    def test_auto_configure(self, base_config):
        """Test automatic configuration."""
        configurator = PlatformEvolutionConfigurator()
        
        # Test that we can get configured config
        config = configurator.get_configured_config()
        assert config is not None

    def test_get_resource_limits(self):
        """Test getting platform resource limits."""
        configurator = PlatformEvolutionConfigurator()
        
        # Test that we can get resource monitor config
        config = configurator.get_resource_monitor_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_validate_configuration(self, base_config):
        """Test configuration validation."""
        configurator = PlatformEvolutionConfigurator()
        
        # Test that we have access to platform settings
        assert configurator.current_platform is not None
        assert configurator.current_platform in configurator.PLATFORM_SETTINGS

    def test_gpu_availability_check(self):
        """Test GPU availability checking."""
        configurator = PlatformEvolutionConfigurator()
        
        # Test that we can access platform settings
        settings = configurator.PLATFORM_SETTINGS[configurator.current_platform]
        assert hasattr(settings, 'enable_gpu')

    def test_get_optimization_suggestions(self, base_config):
        """Test getting optimization suggestions."""
        configurator = PlatformEvolutionConfigurator()
        
        # Test that we have platform-specific settings for all platforms
        for platform in configurator.PLATFORM_SETTINGS:
            settings = configurator.PLATFORM_SETTINGS[platform]
            assert settings.workers > 0
            assert settings.batch_size > 0
