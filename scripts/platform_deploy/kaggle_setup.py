#!/usr/bin/env python3
"""
Kaggle platform setup script for ARC Prize 2025.
Configures the environment specifically for Kaggle notebooks.
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
    """Set up logging for Kaggle environment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/kaggle/working/kaggle_setup.log')
        ]
    )
    return logging.getLogger(__name__)


def check_kaggle_environment() -> bool:
    """Verify we're running in Kaggle environment."""
    kaggle_indicators = [
        os.path.exists('/kaggle'),
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        os.path.exists('/kaggle/input'),
        os.path.exists('/kaggle/working')
    ]
    return any(kaggle_indicators)


def install_dependencies(logger: logging.Logger) -> None:
    """Install required packages in Kaggle environment."""
    logger.info("Installing dependencies...")

    # Kaggle notebooks often have limited pip capabilities
    # Try to install additional packages that might not be pre-installed
    packages = [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'python-socketio>=5.10.0',
        'diskcache>=5.6.0',
        'structlog>=23.0.0',
        'python-dotenv>=1.0.0',
        'psutil>=5.9.0',
        'pyyaml>=6.0.0'
    ]

    for package in packages:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--user', package
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result.returncode == 0:
                logger.info(f"Successfully installed {package}")
            else:
                logger.warning(f"Failed to install {package}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error installing {package}: {e}")


def setup_directories(logger: logging.Logger) -> dict[str, str]:
    """Set up required directories for Kaggle environment."""
    logger.info("Setting up directories...")

    directories = {
        'data': '/kaggle/working/data',
        'cache': '/kaggle/working/cache',
        'logs': '/kaggle/working/logs',
        'models': '/kaggle/working/models',
        'output': '/kaggle/working/output',
        'config': '/kaggle/working/config'
    }

    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created {name} directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create {name} directory {path}: {e}")

    return directories


def setup_kaggle_datasets(logger: logging.Logger) -> list[str]:
    """Set up access to Kaggle datasets."""
    logger.info("Setting up Kaggle datasets...")

    # Check available datasets in /kaggle/input
    input_dir = Path('/kaggle/input')
    available_datasets = []

    if input_dir.exists():
        for item in input_dir.iterdir():
            if item.is_dir():
                available_datasets.append(str(item))
                logger.info(f"Found dataset: {item.name}")

    # Look for ARC-specific datasets
    arc_datasets = [
        'arc-prize-2025',
        'arc-dataset',
        'abstraction-and-reasoning-corpus',
        'arc-agi'
    ]

    found_arc_data = False
    for dataset_name in arc_datasets:
        dataset_path = input_dir / dataset_name
        if dataset_path.exists():
            logger.info(f"Found ARC dataset: {dataset_name}")
            found_arc_data = True
            break

    if not found_arc_data:
        logger.warning("No ARC datasets found in /kaggle/input. You may need to add them as dataset sources.")

    return available_datasets


def configure_environment_variables(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Configure environment variables for Kaggle."""
    logger.info("Configuring environment variables...")

    env_vars = {
        'ENVIRONMENT': 'kaggle',
        'PLATFORM_OVERRIDE': 'kaggle',
        'PYTHONPATH': '/kaggle/working/src',
        'DATA_DIR': directories['data'],
        'CACHE_DIR': directories['cache'],
        'LOGS_DIR': directories['logs'],
        'MODELS_DIR': directories['models'],
        'OUTPUT_DIR': directories['output'],
        'DATABASE_URL': f'sqlite:///{directories["data"]}/arc_prize.db',
        'LOG_LEVEL': 'INFO',
        'KAGGLE_KERNEL_RUN_TYPE': os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')
    }

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def optimize_for_kaggle(logger: logging.Logger) -> None:
    """Apply Kaggle-specific optimizations."""
    logger.info("Applying Kaggle optimizations...")

    # Check available resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total // (1024**3)
        cpu_cores = psutil.cpu_count()

        logger.info(f"Available memory: {memory_gb}GB")
        logger.info(f"Available CPU cores: {cpu_cores}")

        # Check for GPU
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_available = True
                logger.info("GPU detected - CUDA available")
            else:
                logger.info("No GPU detected")
        except FileNotFoundError:
            logger.info("nvidia-smi not found - No GPU available")

        # Set resource-based environment variables
        os.environ['GPU_AVAILABLE'] = str(gpu_available)
        os.environ['MAX_MEMORY_GB'] = str(min(memory_gb - 2, 28))  # Leave headroom
        os.environ['MAX_WORKERS'] = str(min(cpu_cores, 4))

    except ImportError:
        logger.warning("psutil not available - cannot determine system resources")


def create_kaggle_config(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Create Kaggle-specific configuration file."""
    logger.info("Creating Kaggle configuration...")

    config = {
        'platform': 'kaggle',
        'environment': 'production',  # Kaggle is production-like
        'directories': directories,
        'resource_limits': {
            'gpu_hours': 30,
            'memory_gb': int(os.environ.get('MAX_MEMORY_GB', 28)),
            'max_runtime_hours': 12
        },
        'optimizations': {
            'use_gpu': os.environ.get('GPU_AVAILABLE', 'false').lower() == 'true',
            'batch_size': 32 if os.environ.get('GPU_AVAILABLE', 'false').lower() == 'true' else 8,
            'num_workers': int(os.environ.get('MAX_WORKERS', 2)),
            'enable_mixed_precision': True,
            'cache_models': True
        },
        'kaggle_specific': {
            'internet_enabled': os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive',
            'datasets_available': True,
            'time_limit_hours': 12
        }
    }

    config_path = Path(directories['config']) / 'kaggle_runtime.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created Kaggle config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to create config file: {e}")


def run_validation_tests(logger: logging.Logger) -> bool:
    """Run validation tests for Kaggle setup."""
    logger.info("Running validation tests...")

    tests_passed = True

    # Test 1: Platform detection
    try:
        platform = PlatformDetector.detect_platform()
        if platform == Platform.KAGGLE:
            logger.info("✓ Platform detection: PASSED")
        else:
            logger.error(f"✗ Platform detection: FAILED (detected {platform})")
            tests_passed = False
    except Exception as e:
        logger.error(f"✗ Platform detection: ERROR - {e}")
        tests_passed = False

    # Test 2: Directory access
    required_dirs = ['/kaggle/working', '/kaggle/input']
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.access(dir_path, os.R_OK):
            logger.info(f"✓ Directory access ({dir_path}): PASSED")
        else:
            logger.error(f"✗ Directory access ({dir_path}): FAILED")
            tests_passed = False

    # Test 3: Python imports
    try:
        sys.path.insert(0, '/kaggle/working/src')
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
        if memory.available > 2 * (1024**3):  # At least 2GB available
            logger.info("✓ Memory availability: PASSED")
        else:
            logger.warning("⚠ Memory availability: LOW")
    except Exception as e:
        logger.warning(f"⚠ Memory check: {e}")

    return tests_passed


def main():
    """Main setup function for Kaggle platform."""
    print("=" * 60)
    print("ARC Prize 2025 - Kaggle Platform Setup")
    print("=" * 60)

    logger = setup_logging()

    # Verify we're in Kaggle environment
    if not check_kaggle_environment():
        logger.error("This script should only be run in Kaggle environment")
        logger.error("Detected platform: " + str(PlatformDetector.detect_platform()))
        sys.exit(1)

    logger.info("Starting Kaggle platform setup...")

    try:
        # Step 1: Install dependencies
        install_dependencies(logger)

        # Step 2: Set up directories
        directories = setup_directories(logger)

        # Step 3: Set up datasets
        setup_kaggle_datasets(logger)

        # Step 4: Configure environment
        configure_environment_variables(logger, directories)

        # Step 5: Apply optimizations
        optimize_for_kaggle(logger)

        # Step 6: Create config
        create_kaggle_config(logger, directories)

        # Step 7: Run validation
        if run_validation_tests(logger):
            logger.info("✓ Kaggle setup completed successfully!")
            print("\n" + "=" * 60)
            print("KAGGLE SETUP COMPLETE")
            print("=" * 60)
            print("You can now run: python scripts/hello_arc.py")
            print("=" * 60)
        else:
            logger.error("✗ Setup completed with errors")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
