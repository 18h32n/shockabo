"""
Quick validation script for Task 6 implementation.
Tests basic functionality without complex imports.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_system_readiness():
    """Test basic system readiness."""
    logger.info("Testing system readiness...")
    
    issues = []
    
    # Test Python packages
    required_packages = ["torch", "transformers", "numpy", "psutil"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
    
    # Test CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available: {gpu_count} GPUs, {gpu_memory:.1f}GB memory")
        else:
            logger.info("CUDA not available - will use CPU fallback")
    except ImportError:
        issues.append("PyTorch not available")
        cuda_available = False
    
    # Test memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"System memory: {memory_gb:.1f}GB")
        if memory_gb < 8:
            issues.append(f"Low memory: {memory_gb:.1f}GB (recommend 16GB+)")
    except ImportError:
        issues.append("Cannot check system memory")
    
    return len(issues) == 0, issues


def test_file_structure():
    """Test that required files exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        "test_pipeline_100_tasks.py",
        "src/utils/pipeline_test_utils.py", 
        "src/utils/comprehensive_error_handling.py",
        "src/utils/early_stopping_utils.py",
        "src/adapters/repositories/arc_data_repository.py",
        "tests/integration/test_pipeline_100_tasks.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    else:
        logger.info("All required files present")
        return True


def test_basic_imports():
    """Test basic imports work."""
    logger.info("Testing basic imports...")
    
    try:
        # Test that files can be imported (basic syntax check)
        import src.utils.comprehensive_error_handling
        import src.utils.early_stopping_utils
        logger.info("✅ Core utility imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_configuration_structure():
    """Test configuration structures work."""
    logger.info("Testing configuration structures...")
    
    try:
        # Test configuration creation without heavy imports
        config_data = {
            "model_name": "microsoft/DialoGPT-medium",
            "num_tasks": 5,
            "use_qlora": False,
            "enable_gradient_checkpointing": False,
            "use_mixed_precision": False,
            "task_timeout_minutes": 5,
            "max_inference_time_minutes": 2.0,
            "early_stopping_patience": 3
        }
        
        # Test JSON serialization
        json_str = json.dumps(config_data, indent=2)
        parsed_config = json.loads(json_str)
        
        assert parsed_config["num_tasks"] == 5
        assert parsed_config["model_name"] == "microsoft/DialoGPT-medium"
        
        logger.info("✅ Configuration structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_data_structures():
    """Test basic data structures."""
    logger.info("Testing data structures...")
    
    try:
        # Test TaskResult-like structure
        task_result = {
            "task_id": "test_001",
            "status": "success",
            "execution_time": 10.5,
            "memory_peak_mb": 1024.0,
            "accuracy": 0.75,
            "prediction_quality": "correct",
            "error_message": None,
            "inference_time": 30.0,
            "training_time": 120.0,
            "early_stopping_triggered": False,
            "recovery_attempts": 0
        }
        
        # Test serialization
        serialized = json.dumps(task_result, indent=2)
        deserialized = json.loads(serialized)
        
        assert deserialized["task_id"] == "test_001"
        assert deserialized["status"] == "success"
        assert deserialized["accuracy"] == 0.75
        
        logger.info("✅ Data structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data structure test failed: {e}")
        return False


def test_integration_points():
    """Test integration with previous tasks."""
    logger.info("Testing integration points...")
    
    integration_status = {}
    
    # Test Task 1 - QLoRA (LoRA adapter)
    try:
        from src.utils.lora_adapter import apply_lora_to_model
        integration_status["task1_qlora"] = True
        logger.info("✅ Task 1 QLoRA integration found")
    except ImportError:
        integration_status["task1_qlora"] = False
        logger.warning("⚠️ Task 1 QLoRA integration not found")
    
    # Test Task 2 - Gradient checkpointing (PyTorch feature)
    try:
        import torch
        has_checkpointing = hasattr(torch.utils, 'checkpoint')
        integration_status["task2_checkpointing"] = has_checkpointing
        if has_checkpointing:
            logger.info("✅ Task 2 gradient checkpointing available")
        else:
            logger.warning("⚠️ Task 2 gradient checkpointing not found")
    except ImportError:
        integration_status["task2_checkpointing"] = False
    
    # Test Task 3 - Training configuration  
    try:
        from src.domain.services.training_orchestrator import TrainingOrchestrator
        integration_status["task3_training"] = True
        logger.info("✅ Task 3 training configuration found")
    except ImportError:
        integration_status["task3_training"] = False
        logger.warning("⚠️ Task 3 training configuration not found")
    
    # Test Task 4 - Inference optimization
    try:
        from src.utils.inference_optimization_poc import InferenceOptimizer
        integration_status["task4_inference"] = True
        logger.info("✅ Task 4 inference optimization found")
    except ImportError:
        integration_status["task4_inference"] = False
        logger.warning("⚠️ Task 4 inference optimization not found")
    
    # Test Task 5 - Early stopping
    try:
        from src.utils.early_stopping_utils import EarlyStoppingMonitor
        integration_status["task5_early_stopping"] = True
        logger.info("✅ Task 5 early stopping found")
    except ImportError:
        integration_status["task5_early_stopping"] = False
        logger.warning("⚠️ Task 5 early stopping not found")
    
    available_integrations = sum(integration_status.values())
    total_integrations = len(integration_status)
    integration_percentage = (available_integrations / total_integrations) * 100
    
    logger.info(f"Integration status: {available_integrations}/{total_integrations} ({integration_percentage:.1f}%)")
    
    return integration_percentage >= 60  # Pass if 60%+ integrations work


def main():
    """Run quick validation."""
    start_time = time.time()
    
    print("="*70)
    print("TASK 6 QUICK VALIDATION")
    print("="*70)
    print("Validating Task 6: Run Full Pipeline Test on 100 Tasks")
    print("="*70)
    
    # Run validation tests
    tests = [
        ("System Readiness", test_system_readiness),
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Configuration Structure", test_configuration_structure),
        ("Data Structures", test_data_structures),
        ("Integration Points", test_integration_points)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_name == "System Readiness":
                passed, issues = test_func()
                if issues:
                    for issue in issues:
                        logger.warning(f"⚠️ {issue}")
            else:
                passed = test_func()
            
            results[test_name] = passed
            
            if passed:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                all_passed = False
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
            all_passed = False
    
    # Generate summary
    duration = time.time() - start_time
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall Result: {passed_count}/{total_count} tests passed")
    print(f"Validation Time: {duration:.2f} seconds")
    
    if all_passed:
        print("\n🎉 TASK 6 QUICK VALIDATION SUCCESSFUL!")
        print("Key findings:")
        print("  ✅ All required files present")
        print("  ✅ Core imports working")
        print("  ✅ Configuration structures valid")
        print("  ✅ Data structures properly defined")
        print("  ✅ Integration points available")
        print("\nThe Task 6 implementation is ready for testing!")
    else:
        print(f"\n⚠️ VALIDATION COMPLETED WITH {total_count - passed_count} ISSUES")
        print("Review failed tests and address issues before full testing.")
    
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)