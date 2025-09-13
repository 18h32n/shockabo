"""
Unit tests for platform configuration and detection.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from infrastructure.config import (
    ConfigManager,
    Platform,
    PlatformDetector,
    PlatformInfo,
    get_config,
    initialize_config,
)


class TestPlatform:
    """Test Platform enum."""

    def test_platform_values(self):
        """Test that Platform enum has correct values."""
        assert Platform.KAGGLE.value == "kaggle"
        assert Platform.COLAB.value == "colab"
        assert Platform.PAPERSPACE.value == "paperspace"
        assert Platform.LOCAL.value == "local"


class TestPlatformInfo:
    """Test PlatformInfo dataclass."""

    def test_platform_info_creation(self):
        """Test PlatformInfo object creation."""
        info = PlatformInfo(
            platform=Platform.LOCAL,
            gpu_hours_limit=100,
            reset_frequency="never",
            max_memory_gb=16,
            has_persistent_storage=True,
            setup_script="local_setup.py"
        )

        assert info.platform == Platform.LOCAL
        assert info.gpu_hours_limit == 100
        assert info.reset_frequency == "never"
        assert info.max_memory_gb == 16
        assert info.has_persistent_storage is True
        assert info.setup_script == "local_setup.py"


class TestPlatformDetector:
    """Test PlatformDetector class."""

    def test_detect_kaggle_environment(self):
        """Test detection of Kaggle environment."""
        with mock.patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}):
            platform = PlatformDetector.detect_platform()
            assert platform == Platform.KAGGLE

    def test_detect_kaggle_with_directory(self):
        """Test Kaggle detection with directory check."""
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: path == '/kaggle'
            platform = PlatformDetector.detect_platform()
            assert platform == Platform.KAGGLE

    def test_detect_colab_environment(self):
        """Test detection of Google Colab environment."""
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.dict(sys.modules, {'google.colab': mock.MagicMock()}):
                    platform = PlatformDetector.detect_platform()
                    assert platform == Platform.COLAB

    def test_detect_colab_with_env_var(self):
        """Test Colab detection with environment variable."""
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch.dict(os.environ, {'COLAB_GPU': '1'}, clear=True):
                platform = PlatformDetector.detect_platform()
                assert platform == Platform.COLAB

    def test_detect_paperspace_environment(self):
        """Test detection of Paperspace environment."""
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch.dict(os.environ, {'PS_API_KEY': 'test-key'}, clear=True):
                with mock.patch.dict(sys.modules, {}, clear=True):
                    platform = PlatformDetector.detect_platform()
                    assert platform == Platform.PAPERSPACE

    def test_detect_paperspace_with_directory(self):
        """Test Paperspace detection with directory check."""
        with mock.patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: path == '/storage'
            platform = PlatformDetector.detect_platform()
            assert platform == Platform.PAPERSPACE

    def test_detect_local_environment(self):
        """Test detection of local environment (default)."""
        # Clear any environment variables that might indicate other platforms
        env_vars_to_clear = ['KAGGLE_KERNEL_RUN_TYPE', 'COLAB_GPU', 'PS_API_KEY']

        with mock.patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            with mock.patch('os.path.exists', return_value=False):
                with mock.patch.dict(sys.modules, {}, clear=False):
                    if 'google.colab' in sys.modules:
                        del sys.modules['google.colab']

                    platform = PlatformDetector.detect_platform()
                    assert platform == Platform.LOCAL

    def test_get_platform_info_kaggle(self):
        """Test getting platform info for Kaggle."""
        info = PlatformDetector.get_platform_info(Platform.KAGGLE)

        assert info.platform == Platform.KAGGLE
        assert info.gpu_hours_limit == 30
        assert info.reset_frequency == "weekly"
        assert info.max_memory_gb == 32
        assert info.has_persistent_storage is True
        assert info.setup_script == "kaggle_setup.py"

    def test_get_platform_info_colab(self):
        """Test getting platform info for Colab."""
        info = PlatformDetector.get_platform_info(Platform.COLAB)

        assert info.platform == Platform.COLAB
        assert info.gpu_hours_limit == 12
        assert info.reset_frequency == "daily"
        assert info.max_memory_gb == 16
        assert info.has_persistent_storage is False
        assert info.setup_script == "colab_setup.py"

    def test_get_platform_info_paperspace(self):
        """Test getting platform info for Paperspace."""
        info = PlatformDetector.get_platform_info(Platform.PAPERSPACE)

        assert info.platform == Platform.PAPERSPACE
        assert info.gpu_hours_limit == 6
        assert info.reset_frequency == "daily"
        assert info.max_memory_gb == 8
        assert info.has_persistent_storage is True
        assert info.setup_script == "paperspace_setup.py"

    def test_get_platform_info_local(self):
        """Test getting platform info for local environment."""
        with mock.patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.total = 16 * (1024**3)  # 16GB

            info = PlatformDetector.get_platform_info(Platform.LOCAL)

            assert info.platform == Platform.LOCAL
            assert info.gpu_hours_limit == 9999
            assert info.reset_frequency == "never"
            assert info.max_memory_gb == 16
            assert info.has_persistent_storage is True
            assert info.setup_script == "local_setup.py"

    def test_get_platform_info_auto_detect(self):
        """Test getting platform info with auto-detection."""
        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
            mock_detect.return_value = Platform.LOCAL

            info = PlatformDetector.get_platform_info()

            mock_detect.assert_called_once()
            assert info.platform == Platform.LOCAL

    def test_is_gpu_available_with_torch(self):
        """Test GPU availability check with PyTorch."""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with mock.patch.dict(sys.modules, {'torch': mock_torch}):
            gpu_available = PlatformDetector.is_gpu_available()
            assert gpu_available is True

    def test_is_gpu_available_without_torch(self):
        """Test GPU availability check without PyTorch."""
        with mock.patch.dict(sys.modules, {}, clear=False):
            if 'torch' in sys.modules:
                del sys.modules['torch']

            with mock.patch('builtins.__import__', side_effect=ImportError):
                gpu_available = PlatformDetector.is_gpu_available()
                assert gpu_available is False

    def test_get_resource_limits(self):
        """Test getting resource limits."""
        with mock.patch('psutil.virtual_memory') as mock_memory, \
             mock.patch('psutil.cpu_count') as mock_cpu:

            mock_memory.return_value.total = 16 * (1024**3)
            mock_memory.return_value.available = 8 * (1024**3)
            mock_cpu.return_value = 8

            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect, \
                 mock.patch.object(PlatformDetector, 'is_gpu_available') as mock_gpu:

                mock_detect.return_value = Platform.LOCAL
                mock_gpu.return_value = True

                limits = PlatformDetector.get_resource_limits()

                assert limits['platform'] == 'local'
                assert limits['gpu_available'] is True
                assert limits['gpu_hours_limit'] == 9999
                assert limits['total_memory_gb'] == 16
                assert limits['available_memory_gb'] == 8
                assert limits['cpu_cores'] == 8


class TestConfigManager:
    """Test ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "configs"
        self.config_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        self.setUp()
        try:
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LOCAL

                config = ConfigManager(self.config_dir)

                assert config.platform == Platform.LOCAL
                assert config.config_dir == self.config_dir
        finally:
            self.tearDown()

    def test_load_base_config_development(self):
        """Test loading development configuration."""
        self.setUp()
        try:
            # Create a test development.yaml file
            dev_config = {
                'app': {'name': 'test-app'},
                'database': {'url': 'sqlite:///test.db'}
            }

            import yaml
            dev_config_path = self.config_dir / "development.yaml"
            with open(dev_config_path, 'w') as f:
                yaml.dump(dev_config, f)

            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LOCAL

                config = ConfigManager(self.config_dir)

                assert config.get('app.name') == 'test-app'
                assert config.get('database.url') == 'sqlite:///test.db'
        finally:
            self.tearDown()

    def test_load_platform_specific_config(self):
        """Test loading platform-specific configuration."""
        self.setUp()
        try:
            import yaml

            # Create development.yaml
            dev_config = {'app': {'name': 'test-app'}}
            dev_config_path = self.config_dir / "development.yaml"
            with open(dev_config_path, 'w') as f:
                yaml.dump(dev_config, f)

            # Create kaggle.yaml
            kaggle_config = {'platform': {'gpu_limit': 30}}
            kaggle_config_path = self.config_dir / "kaggle.yaml"
            with open(kaggle_config_path, 'w') as f:
                yaml.dump(kaggle_config, f)

            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.KAGGLE

                config = ConfigManager(self.config_dir)

                assert config.get('app.name') == 'test-app'
                assert config.get('platform.gpu_limit') == 30
        finally:
            self.tearDown()

    def test_get_with_dot_notation(self):
        """Test configuration access with dot notation."""
        self.setUp()
        try:
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LOCAL

                config = ConfigManager(self.config_dir)

                # Test default value
                assert config.get('nonexistent.key', 'default') == 'default'

                # Test with platform info being set
                assert config.get('platform.name') == 'local'
        finally:
            self.tearDown()

    def test_platform_specific_directories(self):
        """Test platform-specific directory configurations."""
        self.setUp()
        try:
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.KAGGLE

                config = ConfigManager(self.config_dir)

                data_dir = config.get_data_dir()
                output_dir = config.get_output_dir()
                cache_dir = config.get_cache_dir()

                assert str(data_dir) == '/kaggle/input'
                assert str(output_dir) == '/kaggle/working'
                assert str(cache_dir) == '/kaggle/working/cache'
        finally:
            self.tearDown()

    def test_is_development_mode(self):
        """Test development mode detection."""
        self.setUp()
        try:
            with mock.patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
                with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                    mock_detect.return_value = Platform.LOCAL

                    config = ConfigManager(self.config_dir)
                    assert config.is_development() is True

            with mock.patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
                with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                    mock_detect.return_value = Platform.LOCAL

                    config = ConfigManager(self.config_dir)
                    assert config.is_development() is False
        finally:
            self.tearDown()


class TestConfigGlobals:
    """Test global configuration functions."""

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        # Clear any existing global config
        import infrastructure.config
        infrastructure.config._config_manager = None

        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
            mock_detect.return_value = Platform.LOCAL

            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_initialize_config(self):
        """Test explicit configuration initialization."""
        # Clear any existing global config
        import infrastructure.config
        infrastructure.config._config_manager = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir()

            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LOCAL

                config = initialize_config(config_dir)

                assert config.config_dir == config_dir
                assert config.platform == Platform.LOCAL

                # Test that get_config now returns the same instance
                config2 = get_config()
                assert config is config2


# Pytest fixtures and marks
@pytest.fixture
def temp_config_dir():
    """Create a temporary configuration directory."""
    temp_dir = Path(tempfile.mkdtemp())
    config_dir = temp_dir / "configs"
    config_dir.mkdir()
    yield config_dir
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("platform,expected_data_dir", [
    (Platform.KAGGLE, "/kaggle/input"),
    (Platform.COLAB, "/content/data"),
    (Platform.PAPERSPACE, "/storage/data"),
    (Platform.LOCAL, "data"),
])
def test_platform_data_directories(platform, expected_data_dir, temp_config_dir):
    """Test data directories for different platforms."""
    with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
        mock_detect.return_value = platform

        config = ConfigManager(temp_config_dir)
        data_dir = config.get_data_dir()

        assert str(data_dir) == expected_data_dir


@pytest.mark.parametrize("env_vars,expected_platform", [
    ({'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'}, Platform.KAGGLE),
    ({'COLAB_GPU': '1'}, Platform.COLAB),
    ({'PS_API_KEY': 'test-key'}, Platform.PAPERSPACE),
    ({}, Platform.LOCAL),
])
def test_platform_detection_with_env_vars(env_vars, expected_platform):
    """Test platform detection with various environment variables."""
    # Clear all platform-indicating environment variables
    clear_vars = ['KAGGLE_KERNEL_RUN_TYPE', 'COLAB_GPU', 'PS_API_KEY']

    with mock.patch.dict(os.environ, env_vars, clear=False):
        for var in clear_vars:
            if var not in env_vars:
                os.environ.pop(var, None)

        with mock.patch('os.path.exists', return_value=False):
            with mock.patch.dict(sys.modules, {}, clear=False):
                if 'google.colab' in sys.modules and expected_platform != Platform.COLAB:
                    del sys.modules['google.colab']

                platform = PlatformDetector.detect_platform()
                assert platform == expected_platform
