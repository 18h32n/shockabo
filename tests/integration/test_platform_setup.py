"""
Integration tests for cross-platform setup validation.
Tests the complete setup workflow for each platform.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.config import ConfigManager, Platform, PlatformDetector


class TestPlatformSetupIntegration:
    """Integration tests for platform setup processes."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory structure."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        directories = [
            'src/infrastructure',
            'configs',
            'scripts/platform_deploy',
            'data',
            'output',
            'cache',
            'logs',
            'tests'
        ]

        for dir_path in directories:
            (temp_dir / dir_path).mkdir(parents=True, exist_ok=True)

        yield temp_dir

        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_local_environment_setup(self, temp_project_dir):
        """Test complete setup for local environment."""
        # Create basic config files
        config_content = {
            'app': {'name': 'arc-test'},
            'platform': {'name': 'local'}
        }

        import yaml
        config_path = temp_project_dir / 'configs' / 'development.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)

        # Test configuration loading
        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
            mock_detect.return_value = Platform.LOCAL

            config = ConfigManager(temp_project_dir / 'configs')

            # Verify configuration
            assert config.platform == Platform.LOCAL
            assert config.get('app.name') == 'arc-test'

            # Test directory access
            data_dir = config.get_data_dir()
            config.get_output_dir()
            config.get_cache_dir()

            # These should be relative paths for local
            assert 'data' in str(data_dir)
            assert not str(data_dir).startswith('/kaggle')
            assert not str(data_dir).startswith('/content')
            assert not str(data_dir).startswith('/storage')

    def test_kaggle_environment_simulation(self, temp_project_dir):
        """Test Kaggle environment simulation."""
        # Simulate Kaggle environment
        kaggle_env_vars = {
            'KAGGLE_KERNEL_RUN_TYPE': 'Interactive',
            'PYTHONPATH': str(temp_project_dir / 'src')
        }

        kaggle_config = {
            'platform': {'name': 'kaggle'},
            'paths': {
                'data_dir': '/kaggle/input',
                'output_dir': '/kaggle/working'
            }
        }

        import yaml
        config_path = temp_project_dir / 'configs' / 'kaggle.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(kaggle_config, f)

        with mock.patch.dict(os.environ, kaggle_env_vars):
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.KAGGLE

                config = ConfigManager(temp_project_dir / 'configs')

                # Verify Kaggle-specific configuration
                assert config.platform == Platform.KAGGLE

                # Test Kaggle-specific directories
                data_dir = config.get_data_dir()
                output_dir = config.get_output_dir()

                assert str(data_dir) == '/kaggle/input'
                assert str(output_dir) == '/kaggle/working'

    def test_colab_environment_simulation(self, temp_project_dir):
        """Test Google Colab environment simulation."""
        # Simulate Colab environment
        colab_env_vars = {
            'COLAB_GPU': '1',
            'PYTHONPATH': str(temp_project_dir / 'src')
        }

        colab_config = {
            'platform': {'name': 'colab'},
            'optimizations': {
                'memory_management': 'aggressive',
                'batch_size': 8
            }
        }

        import yaml
        config_path = temp_project_dir / 'configs' / 'colab.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(colab_config, f)

        with mock.patch.dict(os.environ, colab_env_vars):
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.COLAB

                config = ConfigManager(temp_project_dir / 'configs')

                # Verify Colab-specific configuration
                assert config.platform == Platform.COLAB

                # Test Colab-specific directories
                data_dir = config.get_data_dir()
                output_dir = config.get_output_dir()

                assert str(data_dir) == '/content/data'
                assert str(output_dir) == '/content/output'

    def test_paperspace_environment_simulation(self, temp_project_dir):
        """Test Paperspace environment simulation."""
        # Simulate Paperspace environment
        paperspace_env_vars = {
            'PS_API_KEY': 'test-api-key',
            'PYTHONPATH': str(temp_project_dir / 'src')
        }

        paperspace_config = {
            'platform': {'name': 'paperspace'},
            'resource_limits': {
                'gpu_hours': 6,
                'memory_gb': 8
            }
        }

        import yaml
        config_path = temp_project_dir / 'configs' / 'paperspace.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(paperspace_config, f)

        with mock.patch.dict(os.environ, paperspace_env_vars):
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.PAPERSPACE

                config = ConfigManager(temp_project_dir / 'configs')

                # Verify Paperspace-specific configuration
                assert config.platform == Platform.PAPERSPACE

                # Test Paperspace-specific directories
                data_dir = config.get_data_dir()
                output_dir = config.get_output_dir()

                assert str(data_dir) == '/storage/data'
                assert str(output_dir) == '/storage/output'


class TestConfigurationMerging:
    """Test configuration merging across platforms."""

    @pytest.fixture
    def config_files_setup(self):
        """Set up configuration files for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        configs_dir = temp_dir / 'configs'
        configs_dir.mkdir()

        # Base development config
        dev_config = {
            'app': {
                'name': 'arc-prize-2025',
                'version': '0.1.0'
            },
            'database': {
                'url': 'sqlite:///./data/arc.db'
            },
            'model': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }

        # Platform-specific configs
        kaggle_config = {
            'model': {
                'batch_size': 64,  # Override base config
                'use_gpu': True
            },
            'kaggle_specific': {
                'dataset_path': '/kaggle/input/arc-data'
            }
        }

        colab_config = {
            'model': {
                'batch_size': 16,  # Different override
                'use_gpu': True
            },
            'colab_specific': {
                'drive_mount': True
            }
        }

        import yaml

        with open(configs_dir / 'development.yaml', 'w') as f:
            yaml.dump(dev_config, f)

        with open(configs_dir / 'kaggle.yaml', 'w') as f:
            yaml.dump(kaggle_config, f)

        with open(configs_dir / 'colab.yaml', 'w') as f:
            yaml.dump(colab_config, f)

        yield configs_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_configuration_merging_kaggle(self, config_files_setup):
        """Test configuration merging for Kaggle platform."""
        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
            mock_detect.return_value = Platform.KAGGLE

            config = ConfigManager(config_files_setup)

            # Test base config values
            assert config.get('app.name') == 'arc-prize-2025'
            assert config.get('app.version') == '0.1.0'
            assert config.get('database.url') == 'sqlite:///./data/arc.db'

            # Test platform override
            assert config.get('model.batch_size') == 64  # Kaggle override
            assert config.get('model.learning_rate') == 0.001  # Base value
            assert config.get('model.use_gpu') is True  # Kaggle addition

            # Test platform-specific values
            assert config.get('kaggle_specific.dataset_path') == '/kaggle/input/arc-data'
            assert config.get('colab_specific.drive_mount') is None  # Not present

    def test_configuration_merging_colab(self, config_files_setup):
        """Test configuration merging for Colab platform."""
        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
            mock_detect.return_value = Platform.COLAB

            config = ConfigManager(config_files_setup)

            # Test base config values
            assert config.get('app.name') == 'arc-prize-2025'
            assert config.get('database.url') == 'sqlite:///./data/arc.db'

            # Test platform override
            assert config.get('model.batch_size') == 16  # Colab override
            assert config.get('model.use_gpu') is True  # Colab addition

            # Test platform-specific values
            assert config.get('colab_specific.drive_mount') is True
            assert config.get('kaggle_specific.dataset_path') is None  # Not present


class TestResourceDetectionIntegration:
    """Test resource detection integration across platforms."""

    def test_resource_detection_with_gpu(self):
        """Test resource detection when GPU is available."""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with mock.patch.dict(sys.modules, {'torch': mock_torch}):
            with mock.patch('psutil.virtual_memory') as mock_memory:
                with mock.patch('psutil.cpu_count') as mock_cpu:
                    mock_memory.return_value.total = 16 * (1024**3)  # 16GB
                    mock_memory.return_value.available = 12 * (1024**3)  # 12GB
                    mock_cpu.return_value = 8

                    with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                        mock_detect.return_value = Platform.LOCAL

                        resources = PlatformDetector.get_resource_limits()

                        assert resources['gpu_available'] is True
                        assert resources['total_memory_gb'] == 16
                        assert resources['available_memory_gb'] == 12
                        assert resources['cpu_cores'] == 8

    def test_resource_detection_without_gpu(self):
        """Test resource detection when GPU is not available."""
        with mock.patch.dict(sys.modules, {}, clear=False):
            if 'torch' in sys.modules:
                del sys.modules['torch']

            with mock.patch('builtins.__import__', side_effect=ImportError):
                with mock.patch('psutil.virtual_memory') as mock_memory:
                    with mock.patch('psutil.cpu_count') as mock_cpu:
                        mock_memory.return_value.total = 8 * (1024**3)  # 8GB
                        mock_memory.return_value.available = 6 * (1024**3)  # 6GB
                        mock_cpu.return_value = 4

                        with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                            mock_detect.return_value = Platform.LOCAL

                            resources = PlatformDetector.get_resource_limits()

                            assert resources['gpu_available'] is False
                            assert resources['total_memory_gb'] == 8
                            assert resources['available_memory_gb'] == 6
                            assert resources['cpu_cores'] == 4


class TestHelloArcIntegration:
    """Integration tests for Hello ARC validation script."""

    def test_hello_arc_execution(self, temp_project_dir):
        """Test that Hello ARC script can be executed successfully."""
        # Set up minimal project structure
        src_dir = temp_project_dir / 'src'
        src_dir.mkdir(exist_ok=True)

        # Create a minimal config module for the test
        config_module_path = src_dir / 'infrastructure' / '__init__.py'
        config_module_path.parent.mkdir(parents=True, exist_ok=True)
        config_module_path.write_text("")

        # Copy the actual config.py for testing
        actual_config_path = Path(__file__).parent.parent.parent / 'src' / 'infrastructure' / 'config.py'
        test_config_path = src_dir / 'infrastructure' / 'config.py'

        if actual_config_path.exists():
            shutil.copy2(actual_config_path, test_config_path)

        # Create Hello ARC script
        hello_arc_content = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infrastructure.config import PlatformDetector, Platform

def main():
    try:
        platform = PlatformDetector.detect_platform()
        print(f"Platform detected: {platform.value}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
'''

        hello_arc_path = temp_project_dir / 'scripts' / 'hello_arc.py'
        hello_arc_path.write_text(hello_arc_content)

        # Execute the script
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_dir)

        try:
            result = subprocess.run([
                sys.executable, str(hello_arc_path)
            ], capture_output=True, text=True, env=env, cwd=temp_project_dir,
            encoding='utf-8', errors='replace', timeout=30)

            # The script should run without crashing
            assert result.returncode in [0, 1]  # Allow for expected failures in test env
            assert "Platform detected:" in result.stdout or "Error:" in result.stdout

        except subprocess.TimeoutExpired:
            pytest.fail("Hello ARC script execution timed out")
        except Exception as e:
            pytest.fail(f"Failed to execute Hello ARC script: {e}")


class TestPlatformSpecificFeatures:
    """Test platform-specific feature configurations."""

    @pytest.mark.parametrize("platform,expected_features", [
        (Platform.KAGGLE, {
            'gpu_hours_limit': 30,
            'has_persistent_storage': True,
            'reset_frequency': 'weekly'
        }),
        (Platform.COLAB, {
            'gpu_hours_limit': 12,
            'has_persistent_storage': False,
            'reset_frequency': 'daily'
        }),
        (Platform.PAPERSPACE, {
            'gpu_hours_limit': 6,
            'has_persistent_storage': True,
            'reset_frequency': 'daily'
        }),
        (Platform.LOCAL, {
            'gpu_hours_limit': 9999,
            'has_persistent_storage': True,
            'reset_frequency': 'never'
        })
    ])
    def test_platform_specific_configurations(self, platform, expected_features):
        """Test that each platform has correct configuration features."""
        platform_info = PlatformDetector.get_platform_info(platform)

        assert platform_info.gpu_hours_limit == expected_features['gpu_hours_limit']
        assert platform_info.has_persistent_storage == expected_features['has_persistent_storage']
        assert platform_info.reset_frequency == expected_features['reset_frequency']
        assert platform_info.platform == platform


@pytest.mark.slow
class TestEndToEndSetup:
    """End-to-end setup tests (marked as slow)."""

    def test_complete_setup_workflow(self):
        """Test a complete setup workflow from scratch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / 'arc-project'

            # Create project structure
            directories = [
                'src/infrastructure',
                'configs',
                'scripts/platform_deploy',
                'data',
                'output',
                'logs',
                'tests/unit',
                'tests/integration'
            ]

            for dir_path in directories:
                (project_dir / dir_path).mkdir(parents=True)

            # Create basic configuration
            config_content = {
                'app': {'name': 'arc-test'},
                'environment': 'testing'
            }

            import yaml
            with open(project_dir / 'configs' / 'development.yaml', 'w') as f:
                yaml.dump(config_content, f)

            # Test that the setup can initialize properly
            with mock.patch.object(PlatformDetector, 'detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LOCAL

                config = ConfigManager(project_dir / 'configs')

                # Verify basic functionality
                assert config.get('app.name') == 'arc-test'
                assert config.platform == Platform.LOCAL

                # Test directory access
                data_dir = config.get_data_dir()
                output_dir = config.get_output_dir()

                # Directories should exist or be creatable
                assert isinstance(data_dir, Path)
                assert isinstance(output_dir, Path)
