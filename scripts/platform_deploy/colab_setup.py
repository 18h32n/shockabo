#!/usr/bin/env python3
"""
Google Colab platform setup script for ARC Prize 2025.
Configures the environment specifically for Google Colab notebooks.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.config import Platform, PlatformDetector


def setup_logging() -> logging.Logger:
    """Set up logging for Colab environment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/content/colab_setup.log')
        ]
    )
    return logging.getLogger(__name__)


def check_colab_environment() -> bool:
    """Verify we're running in Google Colab environment."""
    colab_indicators = [
        'google.colab' in sys.modules,
        os.path.exists('/content'),
        'COLAB_GPU' in os.environ,
        os.path.exists('/usr/local/lib/python3.10/dist-packages/google/colab')
    ]
    return any(colab_indicators)


def install_dependencies(logger: logging.Logger) -> None:
    """Install required packages in Colab environment."""
    logger.info("Installing dependencies...")

    # Colab has many packages pre-installed, but we need some specific ones
    packages = [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'python-socketio>=5.10.0',
        'diskcache>=5.6.0',
        'structlog>=23.0.0',
        'python-dotenv>=1.0.0',
        'psutil>=5.9.0',
        'pyyaml>=6.0.0',
        'ruff>=0.1.0',  # Colab might not have the latest
        'black>=23.0.0'
    ]

    for package in packages:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--quiet', package
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result.returncode == 0:
                logger.info(f"Successfully installed {package}")
            else:
                logger.warning(f"Failed to install {package}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error installing {package}: {e}")


def setup_directories(logger: logging.Logger) -> dict[str, str]:
    """Set up required directories for Colab environment."""
    logger.info("Setting up directories...")

    directories = {
        'data': '/content/data',
        'cache': '/content/cache',
        'logs': '/content/logs',
        'models': '/content/models',
        'output': '/content/output',
        'config': '/content/config',
        'src': '/content/src',  # We'll clone/copy source here
        'notebooks': '/content/notebooks'
    }

    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created {name} directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create {name} directory {path}: {e}")

    return directories


def setup_google_drive(logger: logging.Logger) -> str | None:
    """Optionally mount Google Drive for persistent storage."""
    logger.info("Checking Google Drive setup...")

    try:
        # Try to import and mount Google Drive
        from google.colab import drive

        drive_path = '/content/drive'
        if not os.path.exists(drive_path):
            logger.info("Mounting Google Drive...")
            drive.mount(drive_path)
            logger.info("✓ Google Drive mounted successfully")
        else:
            logger.info("✓ Google Drive already mounted")

        # Create persistent directories in Drive
        persistent_dirs = {
            'models': f'{drive_path}/MyDrive/ARC_Prize_2025/models',
            'data': f'{drive_path}/MyDrive/ARC_Prize_2025/data',
            'cache': f'{drive_path}/MyDrive/ARC_Prize_2025/cache'
        }

        for name, path in persistent_dirs.items():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created persistent {name} directory: {path}")

        return drive_path

    except Exception as e:
        logger.warning(f"Google Drive setup failed (optional): {e}")
        return None


def download_arc_data(logger: logging.Logger, data_dir: str) -> bool:
    """Download ARC dataset if not available."""
    logger.info("Setting up ARC dataset...")

    arc_urls = {
        'training': 'https://github.com/fchollet/ARC/raw/master/data/training',
        'evaluation': 'https://github.com/fchollet/ARC/raw/master/data/evaluation',
        'test': 'https://github.com/fchollet/ARC/raw/master/data/test'
    }

    success = True
    for dataset_name, base_url in arc_urls.items():
        try:
            # Try to download the dataset files
            target_file = Path(data_dir) / f'arc-{dataset_name}.json'

            if target_file.exists():
                logger.info(f"✓ ARC {dataset_name} dataset already exists")
                continue

            # Use wget or curl to download
            download_url = f'{base_url}'  # May need to adjust URL structure
            result = subprocess.run([
                'wget', '-q', '-O', str(target_file), download_url
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✓ Downloaded ARC {dataset_name} dataset")
            else:
                logger.warning(f"Could not download ARC {dataset_name} dataset")
                success = False

        except Exception as e:
            logger.warning(f"Error downloading ARC {dataset_name} dataset: {e}")
            success = False

    return success


def configure_environment_variables(logger: logging.Logger, directories: dict[str, str], drive_path: str | None) -> None:
    """Configure environment variables for Colab."""
    logger.info("Configuring environment variables...")

    env_vars = {
        'ENVIRONMENT': 'colab',
        'PLATFORM_OVERRIDE': 'colab',
        'PYTHONPATH': '/content/src',
        'DATA_DIR': directories['data'],
        'CACHE_DIR': directories['cache'],
        'LOGS_DIR': directories['logs'],
        'MODELS_DIR': directories['models'],
        'OUTPUT_DIR': directories['output'],
        'DATABASE_URL': f'sqlite:///{directories["data"]}/arc_prize.db',
        'LOG_LEVEL': 'INFO',
        'COLAB_ENVIRONMENT': 'true'
    }

    # Add Google Drive paths if available
    if drive_path:
        env_vars.update({
            'DRIVE_PATH': drive_path,
            'PERSISTENT_MODELS_DIR': f'{drive_path}/MyDrive/ARC_Prize_2025/models',
            'PERSISTENT_DATA_DIR': f'{drive_path}/MyDrive/ARC_Prize_2025/data',
        })

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def optimize_for_colab(logger: logging.Logger) -> None:
    """Apply Colab-specific optimizations."""
    logger.info("Applying Colab optimizations...")

    # Check available resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total // (1024**3)
        cpu_cores = psutil.cpu_count()

        logger.info(f"Available memory: {memory_gb}GB")
        logger.info(f"Available CPU cores: {cpu_cores}")

        # Check for GPU
        gpu_available = False
        gpu_type = "None"

        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_available = True
                # Try to get GPU type from nvidia-smi output
                if 'Tesla' in result.stdout:
                    gpu_type = "Tesla"
                elif 'T4' in result.stdout:
                    gpu_type = "T4"
                elif 'K80' in result.stdout:
                    gpu_type = "K80"
                else:
                    gpu_type = "Unknown"
                logger.info(f"GPU detected: {gpu_type}")
            else:
                logger.info("No GPU detected")
        except FileNotFoundError:
            logger.info("nvidia-smi not found - No GPU available")

        # Set conservative resource limits for Colab
        max_memory = min(memory_gb - 2, 12)  # Colab usually has ~13GB, leave headroom
        max_workers = min(cpu_cores, 2)      # Conservative for Colab

        # Set resource-based environment variables
        os.environ['GPU_AVAILABLE'] = str(gpu_available)
        os.environ['GPU_TYPE'] = gpu_type
        os.environ['MAX_MEMORY_GB'] = str(max_memory)
        os.environ['MAX_WORKERS'] = str(max_workers)

        # Colab-specific memory management
        os.environ['COLAB_GPU_HOURS_LIMIT'] = '12'
        os.environ['ENABLE_MEMORY_CLEANUP'] = 'true'

    except ImportError:
        logger.warning("psutil not available - cannot determine system resources")


def create_colab_config(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Create Colab-specific configuration file."""
    logger.info("Creating Colab configuration...")

    config = {
        'platform': 'colab',
        'environment': 'development',
        'directories': directories,
        'resource_limits': {
            'gpu_hours': 12,
            'memory_gb': int(os.environ.get('MAX_MEMORY_GB', 12)),
            'max_runtime_hours': 12,
            'auto_disconnect_hours': 12
        },
        'optimizations': {
            'use_gpu': os.environ.get('GPU_AVAILABLE', 'false').lower() == 'true',
            'gpu_type': os.environ.get('GPU_TYPE', 'None'),
            'batch_size': 16 if os.environ.get('GPU_AVAILABLE', 'false').lower() == 'true' else 4,
            'num_workers': int(os.environ.get('MAX_WORKERS', 1)),
            'enable_mixed_precision': True,
            'aggressive_memory_management': True,
            'enable_auto_cleanup': True
        },
        'colab_specific': {
            'drive_mounted': 'DRIVE_PATH' in os.environ,
            'notebook_environment': True,
            'auto_disconnect_warning': True,
            'save_checkpoints_frequently': True
        }
    }

    config_path = Path(directories['config']) / 'colab_runtime.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created Colab config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to create config file: {e}")


def setup_colab_notebook_helpers(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Create helper functions for Colab notebooks."""
    logger.info("Setting up Colab notebook helpers...")

    helpers_content = '''
"""
Colab notebook helper functions for ARC Prize 2025
Import this at the start of your notebooks: from helpers import *
"""

import os
import sys
import gc
import psutil
import time
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import numpy as np

# Add src to Python path
sys.path.insert(0, '/content/src')

def show_memory_usage():
    """Display current memory usage."""
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")

def cleanup_memory():
    """Force garbage collection and clear memory."""
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Memory cleanup completed")
    show_memory_usage()

def check_gpu_status():
    """Check GPU availability and memory."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"GPU: {device_name}")
            print(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        else:
            print("No GPU available")
    except ImportError:
        print("PyTorch not available - cannot check GPU status")

def setup_arc_environment():
    """Quick setup function for ARC development."""
    print("Setting up ARC Prize 2025 environment...")

    # Import key modules
    try:
        from infrastructure.config import get_config, PlatformDetector
        config = get_config()
        platform = PlatformDetector.detect_platform()

        print(f"✓ Platform detected: {platform.value}")
        print(f"✓ Configuration loaded")

        return config
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        return None

def save_to_drive(data, filename, subdir="output"):
    """Save data to Google Drive if mounted."""
    drive_path = os.environ.get('DRIVE_PATH')
    if drive_path:
        save_path = f"{drive_path}/MyDrive/ARC_Prize_2025/{subdir}/{filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if isinstance(data, (dict, list)):
            import json
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, np.ndarray):
            np.save(save_path, data)
        else:
            with open(save_path, 'w') as f:
                f.write(str(data))

        print(f"✓ Saved to Drive: {save_path}")
    else:
        print("Google Drive not mounted - cannot save")

def load_from_drive(filename, subdir="output"):
    """Load data from Google Drive if mounted."""
    drive_path = os.environ.get('DRIVE_PATH')
    if drive_path:
        load_path = f"{drive_path}/MyDrive/ARC_Prize_2025/{subdir}/{filename}"
        if os.path.exists(load_path):
            if filename.endswith('.json'):
                import json
                with open(load_path, 'r') as f:
                    return json.load(f)
            elif filename.endswith('.npy'):
                return np.load(load_path)
            else:
                with open(load_path, 'r') as f:
                    return f.read()
        else:
            print(f"File not found: {load_path}")
            return None
    else:
        print("Google Drive not mounted - cannot load")
        return None

# Auto-setup on import
print("ARC Prize 2025 Colab helpers loaded")
print("Functions available: show_memory_usage(), cleanup_memory(), check_gpu_status(), setup_arc_environment()")
print("Drive functions: save_to_drive(data, filename), load_from_drive(filename)")
'''

    helpers_path = Path(directories['config']) / 'colab_helpers.py'
    try:
        with open(helpers_path, 'w') as f:
            f.write(helpers_content)
        logger.info(f"Created Colab helpers: {helpers_path}")
    except Exception as e:
        logger.error(f"Failed to create helpers file: {e}")


def run_validation_tests(logger: logging.Logger) -> bool:
    """Run validation tests for Colab setup."""
    logger.info("Running validation tests...")

    tests_passed = True

    # Test 1: Platform detection
    try:
        platform = PlatformDetector.detect_platform()
        if platform == Platform.COLAB:
            logger.info("✓ Platform detection: PASSED")
        else:
            logger.warning(f"⚠ Platform detection: Expected COLAB, got {platform}")
    except Exception as e:
        logger.error(f"✗ Platform detection: ERROR - {e}")
        tests_passed = False

    # Test 2: Directory access
    required_dirs = ['/content']
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.access(dir_path, os.R_OK | os.W_OK):
            logger.info(f"✓ Directory access ({dir_path}): PASSED")
        else:
            logger.error(f"✗ Directory access ({dir_path}): FAILED")
            tests_passed = False

    # Test 3: Python imports
    try:
        sys.path.insert(0, '/content/src')
        from infrastructure.config import get_config
        get_config()
        logger.info("✓ Configuration import: PASSED")
    except Exception as e:
        logger.error(f"✗ Configuration import: FAILED - {e}")
        tests_passed = False

    # Test 4: Resource availability
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available > 1 * (1024**3):  # At least 1GB available
            logger.info("✓ Memory availability: PASSED")
        else:
            logger.warning("⚠ Memory availability: LOW")
    except Exception as e:
        logger.warning(f"⚠ Memory check: {e}")

    # Test 5: GPU availability (optional)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ GPU availability: PASSED")
        else:
            logger.info("ℹ GPU availability: Not available (CPU mode)")
    except Exception:
        logger.info("ℹ GPU availability: Not available (CPU mode)")

    return tests_passed


def main():
    """Main setup function for Google Colab platform."""
    print("=" * 60)
    print("ARC Prize 2025 - Google Colab Platform Setup")
    print("=" * 60)

    logger = setup_logging()

    # Verify we're in Colab environment (allow to proceed even if not detected)
    if not check_colab_environment():
        logger.warning("Colab environment not clearly detected, but proceeding...")
        logger.info("Detected platform: " + str(PlatformDetector.detect_platform()))

    logger.info("Starting Colab platform setup...")

    try:
        # Step 1: Install dependencies
        install_dependencies(logger)

        # Step 2: Set up directories
        directories = setup_directories(logger)

        # Step 3: Set up Google Drive (optional)
        drive_path = setup_google_drive(logger)

        # Step 4: Download ARC data (if needed)
        download_arc_data(logger, directories['data'])

        # Step 5: Configure environment
        configure_environment_variables(logger, directories, drive_path)

        # Step 6: Apply optimizations
        optimize_for_colab(logger)

        # Step 7: Create config
        create_colab_config(logger, directories)

        # Step 8: Set up notebook helpers
        setup_colab_notebook_helpers(logger, directories)

        # Step 9: Run validation
        if run_validation_tests(logger):
            logger.info("✓ Colab setup completed successfully!")
            print("\n" + "=" * 60)
            print("GOOGLE COLAB SETUP COMPLETE")
            print("=" * 60)
            print("You can now run: python scripts/hello_arc.py")
            print("Or import the helpers: from config/colab_helpers import *")
            print("=" * 60)
        else:
            logger.error("✗ Setup completed with errors")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
