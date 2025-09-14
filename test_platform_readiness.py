"""
Platform readiness test for Kaggle/Colab deployment.

Tests that the TTT implementation will work correctly on competition platforms.
"""
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_platform():
    """Detect the current platform environment."""
    import os

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ:
        return 'paperspace'
    else:
        return 'local'

def test_basic_dependencies():
    """Test that basic dependencies are available."""
    logger.info("Testing basic dependencies...")

    try:
        import numpy as np
        import torch
        import transformers
        import yaml

        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ Transformers: {transformers.__version__}")
        logger.info(f"✅ NumPy: {np.__version__}")
        logger.info("✅ YAML: Available")

        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def test_device_compatibility():
    """Test device detection and compatibility."""
    logger.info("Testing device compatibility...")

    try:
        # Test CPU
        cpu_available = torch.cuda.is_available() or True  # CPU always available
        logger.info(f"✅ CPU available: {cpu_available}")

        # Test CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            logger.info(f"✅ CUDA available: {cuda_available}")
            logger.info(f"✅ GPU count: {gpu_count}")
            logger.info(f"✅ GPU name: {gpu_name}")
            logger.info(f"✅ GPU memory: {memory_gb:.1f} GB")
        else:
            logger.info("ℹ️ CUDA not available - will use CPU")

        # Test device auto-selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"✅ Auto-selected device: {device}")

        # Test basic tensor operations
        test_tensor = torch.randn(100, 100).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        logger.info(f"✅ Device operations working: {result.shape}")

        return True
    except Exception as e:
        logger.error(f"❌ Device compatibility test failed: {e}")
        return False

def test_memory_constraints():
    """Test memory usage and constraints."""
    logger.info("Testing memory constraints...")

    try:
        import psutil

        # System memory
        system_memory = psutil.virtual_memory()
        total_gb = system_memory.total / (1024**3)
        available_gb = system_memory.available / (1024**3)

        logger.info(f"✅ System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")

        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            torch.cuda.empty_cache()
            torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"✅ GPU memory: {gpu_memory_total:.1f} GB total")

        # Test if we have enough memory for TTT (minimum 8GB recommended)
        memory_sufficient = total_gb >= 8.0
        if memory_sufficient:
            logger.info("✅ Sufficient memory for TTT (≥8GB)")
        else:
            logger.warning(f"⚠️ Limited memory: {total_gb:.1f}GB (8GB+ recommended)")

        return True
    except ImportError:
        logger.warning("⚠️ psutil not available - cannot check system memory")
        return True
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading from YAML files."""
    logger.info("Testing configuration loading...")

    try:
        from src.infrastructure.config import get_config

        # Test loading development config
        get_config()
        logger.info("✅ Development config loaded")

        # Test TTT strategy config
        ttt_config_path = Path("configs/strategies/ttt.yaml")
        if ttt_config_path.exists():
            import yaml
            with open(ttt_config_path) as f:
                ttt_config = yaml.safe_load(f)
            logger.info("✅ TTT strategy config loaded")
            logger.info(f"✅ Model: {ttt_config.get('model', {}).get('name', 'Not specified')}")
        else:
            logger.warning("⚠️ TTT config file not found")

        return True
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False

def test_data_loading():
    """Test data loading capabilities."""
    logger.info("Testing data loading...")

    try:
        from src.adapters.repositories.arc_data_repository import ARCDataRepository

        # Test with real dataset if available
        data_repo = ARCDataRepository("arc-prize-2025", use_real_dataset=True)
        task_ids = data_repo.get_task_ids("training")

        if len(task_ids) > 0:
            logger.info(f"✅ Real dataset available: {len(task_ids)} training tasks")

            # Load a sample task
            sample_task = data_repo.load_task(task_ids[0], "training")
            logger.info(f"✅ Sample task loaded: {sample_task.task_id}")
        else:
            logger.warning("⚠️ No real dataset found - will use fallback data")

        return True
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False

def test_model_loading():
    """Test model loading with authentication."""
    logger.info("Testing model loading...")

    try:
        from src.utils.auth_config import get_model_access_info, suggest_public_model

        # Test model access
        test_models = ["gpt2", "meta-llama/Llama-3.2-1B"]
        accessible_model = None

        for model_name in test_models:
            access_info = get_model_access_info(model_name)
            logger.info(f"Model {model_name}: {'✅ Accessible' if access_info['can_access'] else '❌ Not accessible'}")

            if access_info['can_access']:
                accessible_model = model_name
                break
            else:
                suggested = suggest_public_model(model_name)
                logger.info(f"  → Suggested alternative: {suggested}")
                if suggested:
                    accessible_model = suggested

        if accessible_model:
            logger.info(f"✅ Will use model: {accessible_model}")
            return True
        else:
            logger.error("❌ No accessible models found")
            return False

    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def test_training_pipeline():
    """Test basic training pipeline setup."""
    logger.info("Testing training pipeline setup...")

    try:
        from src.adapters.strategies.ttt_adapter import TTTConfig

        # Create minimal config for testing
        config = TTTConfig(
            model_name="gpt2",
            max_examples=1,
            num_epochs=1,
            batch_size=1,
            device="cpu",
            quantization=False,
            mixed_precision=False,
            cache_dir=Path("test_data/cache/ttt"),
            checkpoint_dir=Path("test_data/models/ttt")
        )

        logger.info("✅ TTT config created")
        logger.info(f"✅ Model: {config.model_name}")
        logger.info(f"✅ Device: {config.device}")
        logger.info(f"✅ Cache dir: {config.cache_dir}")

        return True
    except Exception as e:
        logger.error(f"❌ Training pipeline test failed: {e}")
        return False

def generate_platform_report():
    """Generate a platform compatibility report."""
    platform = detect_platform()
    logger.info(f"Platform detected: {platform}")

    report = {
        "platform": platform,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    # Run all tests
    tests = [
        ("dependencies", test_basic_dependencies),
        ("device_compatibility", test_device_compatibility),
        ("memory_constraints", test_memory_constraints),
        ("configuration_loading", test_configuration_loading),
        ("data_loading", test_data_loading),
        ("model_loading", test_model_loading),
        ("training_pipeline", test_training_pipeline),
    ]

    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name.replace('_', ' ').title()} ---")
        try:
            result = test_func()
            report["tests"][test_name] = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            report["tests"][test_name] = "ERROR"
            all_passed = False

    report["overall_status"] = "READY" if all_passed else "ISSUES"

    return report

def main():
    """Main test execution."""
    logger.info("=" * 70)
    logger.info("TTT PLATFORM READINESS TEST")
    logger.info("=" * 70)

    # Generate report
    report = generate_platform_report()

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("PLATFORM READINESS REPORT")
    logger.info("=" * 70)
    logger.info(f"Platform: {report['platform']}")
    logger.info(f"Timestamp: {report['timestamp']}")
    logger.info(f"Overall Status: {report['overall_status']}")

    logger.info("\nTest Results:")
    for test_name, status in report["tests"].items():
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status_icon} {status}")

    # Recommendations
    logger.info("\nRecommendations:")
    if report["overall_status"] == "READY":
        logger.info("🎉 Platform is ready for TTT deployment!")
        logger.info("✅ All systems are compatible")
        logger.info("✅ Dependencies are available")
        logger.info("✅ Training pipeline can be set up")
    else:
        logger.info("⚠️ Platform has some compatibility issues")
        logger.info("🔧 Review failed tests above")
        logger.info("📋 Consider using fallback configurations")

    logger.info("=" * 70)

    # Return appropriate exit code
    return 0 if report["overall_status"] == "READY" else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
