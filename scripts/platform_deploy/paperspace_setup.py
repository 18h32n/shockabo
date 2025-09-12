#!/usr/bin/env python3
"""
Paperspace platform setup script for ARC Prize 2025.
Configures the environment specifically for Paperspace Gradient notebooks.
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
    """Set up logging for Paperspace environment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/storage/paperspace_setup.log')
        ]
    )
    return logging.getLogger(__name__)


def check_paperspace_environment() -> bool:
    """Verify we're running in Paperspace environment."""
    paperspace_indicators = [
        'PS_API_KEY' in os.environ,
        os.path.exists('/storage'),
        'gradient' in os.getcwd().lower() if os.getcwd() else False,
        os.path.exists('/notebooks')  # Common in Paperspace
    ]
    return any(paperspace_indicators)


def install_dependencies(logger: logging.Logger) -> None:
    """Install required packages in Paperspace environment."""
    logger.info("Installing dependencies...")

    # Paperspace Gradient usually has basic packages, but we need some specific ones
    packages = [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'python-socketio>=5.10.0',
        'diskcache>=5.6.0',
        'structlog>=23.0.0',
        'python-dotenv>=1.0.0',
        'psutil>=5.9.0',
        'pyyaml>=6.0.0',
        'ruff>=0.1.0',
        'black>=23.0.0'
    ]

    for package in packages:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--quiet', '--user', package
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result.returncode == 0:
                logger.info(f"Successfully installed {package}")
            else:
                logger.warning(f"Failed to install {package}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error installing {package}: {e}")


def setup_directories(logger: logging.Logger) -> dict[str, str]:
    """Set up required directories for Paperspace environment."""
    logger.info("Setting up directories...")

    # Use /storage for persistent data in Paperspace
    directories = {
        'data': '/storage/data',
        'cache': '/storage/cache',
        'logs': '/storage/logs',
        'models': '/storage/models',
        'output': '/storage/output',
        'config': '/storage/config',
        'src': '/storage/src',  # We'll clone/copy source here
        'notebooks': '/notebooks'  # Standard Paperspace notebooks location
    }

    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created {name} directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create {name} directory {path}: {e}")

    return directories


def configure_paperspace_api(logger: logging.Logger) -> bool:
    """Configure Paperspace API if credentials are available."""
    logger.info("Configuring Paperspace API...")

    api_key = os.environ.get('PS_API_KEY')
    if not api_key:
        logger.warning("PS_API_KEY not found - some Paperspace features may be limited")
        return False

    try:
        # Try to install and configure Paperspace CLI
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet', '--user', 'gradient>=2.0.0'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✓ Paperspace Gradient CLI installed")

            # Set up API key for gradient CLI
            os.environ['GRADIENT_API_KEY'] = api_key
            logger.info("✓ Paperspace API configured")
            return True
        else:
            logger.warning("Could not install Paperspace CLI")
            return False

    except Exception as e:
        logger.error(f"Error configuring Paperspace API: {e}")
        return False


def setup_auto_shutdown(logger: logging.Logger) -> None:
    """Set up automatic shutdown to save GPU hours."""
    logger.info("Setting up auto-shutdown to save GPU hours...")

    shutdown_script = '''#!/bin/bash
# Auto-shutdown script for Paperspace to save GPU hours
# This script will shut down the machine after 1 hour of inactivity

IDLE_TIME=3600  # 1 hour in seconds
LOG_FILE="/storage/logs/auto_shutdown.log"

echo "$(date): Auto-shutdown monitor started (${IDLE_TIME}s idle timeout)" >> $LOG_FILE

while true; do
    # Check if any python processes are running (indicating active work)
    if pgrep -f python > /dev/null; then
        echo "$(date): Python processes detected - resetting idle timer" >> $LOG_FILE
        sleep 300  # Check again in 5 minutes
        continue
    fi

    # Check CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F% '{print $1}')
    if (( $(echo "$CPU_USAGE > 5" | bc -l) )); then
        echo "$(date): High CPU usage detected (${CPU_USAGE}%) - resetting idle timer" >> $LOG_FILE
        sleep 300  # Check again in 5 minutes
        continue
    fi

    # Check GPU usage if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        if (( GPU_USAGE > 5 )); then
            echo "$(date): GPU usage detected (${GPU_USAGE}%) - resetting idle timer" >> $LOG_FILE
            sleep 300  # Check again in 5 minutes
            continue
        fi
    fi

    # If we get here, system appears idle
    echo "$(date): System idle for check period - sleeping 5 more minutes" >> $LOG_FILE
    sleep 300

    # Double-check before shutdown
    if ! pgrep -f python > /dev/null && (( $(echo "$CPU_USAGE < 5" | bc -l) )); then
        echo "$(date): Initiating auto-shutdown due to inactivity" >> $LOG_FILE
        # Use gradient CLI to stop the machine if available, otherwise use standard shutdown
        if command -v gradient &> /dev/null; then
            gradient machines stop $(curl -s http://metadata.paperspace.com/meta-data/machine-id)
        else
            sudo shutdown -h +5 "Auto-shutdown: System idle for over 1 hour"
        fi
        break
    fi
done
'''

    try:
        script_path = Path('/storage/scripts/auto_shutdown.sh')
        script_path.parent.mkdir(exist_ok=True)

        with open(script_path, 'w') as f:
            f.write(shutdown_script)

        # Make executable
        os.chmod(script_path, 0o755)

        logger.info(f"✓ Auto-shutdown script created: {script_path}")
        logger.info("Note: Run 'nohup /storage/scripts/auto_shutdown.sh &' to enable auto-shutdown")

    except Exception as e:
        logger.error(f"Failed to create auto-shutdown script: {e}")


def configure_environment_variables(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Configure environment variables for Paperspace."""
    logger.info("Configuring environment variables...")

    env_vars = {
        'ENVIRONMENT': 'paperspace',
        'PLATFORM_OVERRIDE': 'paperspace',
        'PYTHONPATH': '/storage/src',
        'DATA_DIR': directories['data'],
        'CACHE_DIR': directories['cache'],
        'LOGS_DIR': directories['logs'],
        'MODELS_DIR': directories['models'],
        'OUTPUT_DIR': directories['output'],
        'DATABASE_URL': f'sqlite:///{directories["data"]}/arc_prize.db',
        'LOG_LEVEL': 'INFO',
        'PAPERSPACE_ENVIRONMENT': 'true'
    }

    # Add Paperspace-specific variables
    if 'PS_API_KEY' in os.environ:
        env_vars['GRADIENT_API_KEY'] = os.environ['PS_API_KEY']

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def optimize_for_paperspace(logger: logging.Logger) -> None:
    """Apply Paperspace-specific optimizations."""
    logger.info("Applying Paperspace optimizations...")

    # Check available resources (Paperspace can be quite limited)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total // (1024**3)
        cpu_cores = psutil.cpu_count()

        logger.info(f"Available memory: {memory_gb}GB")
        logger.info(f"Available CPU cores: {cpu_cores}")

        # Check for GPU
        gpu_available = False
        gpu_type = "None"
        gpu_memory_gb = 0

        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_available = True
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    gpu_type = parts[0] if parts else "Unknown"
                    gpu_memory_gb = int(parts[1]) // 1024 if len(parts) > 1 and parts[1].isdigit() else 0

                logger.info(f"GPU detected: {gpu_type} ({gpu_memory_gb}GB)")
            else:
                logger.info("No GPU detected")
        except FileNotFoundError:
            logger.info("nvidia-smi not found - No GPU available")

        # Very conservative resource limits for Paperspace
        max_memory = max(1, min(memory_gb - 1, 6))  # Very conservative
        max_workers = 1  # Single worker to minimize resource usage
        batch_size = 4 if gpu_available else 2  # Small batch sizes

        # Set resource-based environment variables
        os.environ['GPU_AVAILABLE'] = str(gpu_available)
        os.environ['GPU_TYPE'] = gpu_type
        os.environ['GPU_MEMORY_GB'] = str(gpu_memory_gb)
        os.environ['MAX_MEMORY_GB'] = str(max_memory)
        os.environ['MAX_WORKERS'] = str(max_workers)
        os.environ['DEFAULT_BATCH_SIZE'] = str(batch_size)

        # Paperspace-specific resource management
        os.environ['PAPERSPACE_GPU_HOURS_LIMIT'] = '6'
        os.environ['ENABLE_AGGRESSIVE_CLEANUP'] = 'true'
        os.environ['MINIMAL_MEMORY_MODE'] = 'true'

    except ImportError:
        logger.warning("psutil not available - cannot determine system resources")


def create_paperspace_config(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Create Paperspace-specific configuration file."""
    logger.info("Creating Paperspace configuration...")

    config = {
        'platform': 'paperspace',
        'environment': 'development',
        'directories': directories,
        'resource_limits': {
            'gpu_hours': 6,  # Very limited
            'memory_gb': int(os.environ.get('MAX_MEMORY_GB', 6)),
            'max_runtime_hours': 6,
            'auto_shutdown_idle_minutes': 60
        },
        'optimizations': {
            'use_gpu': os.environ.get('GPU_AVAILABLE', 'false').lower() == 'true',
            'gpu_type': os.environ.get('GPU_TYPE', 'None'),
            'gpu_memory_gb': int(os.environ.get('GPU_MEMORY_GB', 0)),
            'batch_size': int(os.environ.get('DEFAULT_BATCH_SIZE', 2)),
            'num_workers': int(os.environ.get('MAX_WORKERS', 1)),
            'enable_mixed_precision': True,
            'aggressive_memory_management': True,
            'minimal_logging': True,
            'disable_debug_features': True
        },
        'paperspace_specific': {
            'api_configured': 'GRADIENT_API_KEY' in os.environ,
            'persistent_storage': True,
            'auto_shutdown_enabled': False,  # User must enable manually
            'cost_optimization': True
        }
    }

    config_path = Path(directories['config']) / 'paperspace_runtime.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created Paperspace config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to create config file: {e}")


def create_paperspace_notebook_template(logger: logging.Logger, directories: dict[str, str]) -> None:
    """Create a template notebook optimized for Paperspace."""
    logger.info("Creating Paperspace notebook template...")

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ARC Prize 2025 - Paperspace Setup\n",
                    "\n",
                    "This notebook sets up the ARC Prize 2025 environment on Paperspace Gradient.\n",
                    "\n",
                    "## Important Notes:\n",
                    "- Paperspace has limited GPU hours (6/day)\n",
                    "- Use auto-shutdown to save resources\n",
                    "- Data is persistent in `/storage/`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Setup and imports\n",
                    "import sys\n",
                    "sys.path.append('/storage/src')\n",
                    "\n",
                    "# Run platform setup\n",
                    "!python /storage/scripts/platform_deploy/paperspace_setup.py\n",
                    "\n",
                    "print(\"Setup complete!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Check system resources\n",
                    "import psutil\n",
                    "import os\n",
                    "\n",
                    "print(f\"Memory: {psutil.virtual_memory().total // (1024**3)}GB\")\n",
                    "print(f\"CPU cores: {psutil.cpu_count()}\")\n",
                    "print(f\"GPU available: {os.environ.get('GPU_AVAILABLE', 'false')}\")\n",
                    "print(f\"GPU type: {os.environ.get('GPU_TYPE', 'None')}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load ARC configuration\n",
                    "from infrastructure.config import get_config, PlatformDetector\n",
                    "\n",
                    "config = get_config()\n",
                    "platform = PlatformDetector.detect_platform()\n",
                    "\n",
                    "print(f\"Platform: {platform.value}\")\n",
                    "print(f\"Data directory: {config.get_data_dir()}\")\n",
                    "print(f\"Output directory: {config.get_output_dir()}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Enable auto-shutdown (optional)\n",
                    "# Uncomment the following line to enable automatic shutdown after 1 hour of inactivity\n",
                    "# !nohup /storage/scripts/auto_shutdown.sh > /storage/logs/auto_shutdown.out 2>&1 &\n",
                    "\n",
                    "print(\"Auto-shutdown script available at /storage/scripts/auto_shutdown.sh\")\n",
                    "print(\"Enable with: !nohup /storage/scripts/auto_shutdown.sh > /storage/logs/auto_shutdown.out 2>&1 &\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    notebook_path = Path(directories['notebooks']) / 'arc_paperspace_setup.ipynb'
    try:
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        logger.info(f"Created Paperspace notebook template: {notebook_path}")
    except Exception as e:
        logger.error(f"Failed to create notebook template: {e}")


def run_validation_tests(logger: logging.Logger) -> bool:
    """Run validation tests for Paperspace setup."""
    logger.info("Running validation tests...")

    tests_passed = True

    # Test 1: Platform detection
    try:
        platform = PlatformDetector.detect_platform()
        if platform == Platform.PAPERSPACE:
            logger.info("✓ Platform detection: PASSED")
        else:
            logger.warning(f"⚠ Platform detection: Expected PAPERSPACE, got {platform}")
    except Exception as e:
        logger.error(f"✗ Platform detection: ERROR - {e}")
        tests_passed = False

    # Test 2: Directory access
    required_dirs = ['/storage']
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.access(dir_path, os.R_OK | os.W_OK):
            logger.info(f"✓ Directory access ({dir_path}): PASSED")
        else:
            logger.error(f"✗ Directory access ({dir_path}): FAILED")
            tests_passed = False

    # Test 3: Python imports
    try:
        sys.path.insert(0, '/storage/src')
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
        if memory.available > 512 * (1024**2):  # At least 512MB available
            logger.info("✓ Memory availability: PASSED")
        else:
            logger.error("✗ Memory availability: INSUFFICIENT")
            tests_passed = False
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
    """Main setup function for Paperspace platform."""
    print("=" * 60)
    print("ARC Prize 2025 - Paperspace Platform Setup")
    print("=" * 60)

    logger = setup_logging()

    # Verify we're in Paperspace environment (allow to proceed even if not detected)
    if not check_paperspace_environment():
        logger.warning("Paperspace environment not clearly detected, but proceeding...")
        logger.info("Detected platform: " + str(PlatformDetector.detect_platform()))

    logger.info("Starting Paperspace platform setup...")

    try:
        # Step 1: Install dependencies
        install_dependencies(logger)

        # Step 2: Set up directories
        directories = setup_directories(logger)

        # Step 3: Configure Paperspace API
        configure_paperspace_api(logger)

        # Step 4: Set up auto-shutdown
        setup_auto_shutdown(logger)

        # Step 5: Configure environment
        configure_environment_variables(logger, directories)

        # Step 6: Apply optimizations
        optimize_for_paperspace(logger)

        # Step 7: Create config
        create_paperspace_config(logger, directories)

        # Step 8: Create notebook template
        create_paperspace_notebook_template(logger, directories)

        # Step 9: Run validation
        if run_validation_tests(logger):
            logger.info("✓ Paperspace setup completed successfully!")
            print("\n" + "=" * 60)
            print("PAPERSPACE SETUP COMPLETE")
            print("=" * 60)
            print("You can now run: python scripts/hello_arc.py")
            print("Consider enabling auto-shutdown to save GPU hours:")
            print("  nohup /storage/scripts/auto_shutdown.sh > /storage/logs/auto_shutdown.out 2>&1 &")
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
