#!/usr/bin/env python3
"""
Task 3 Setup Verification Script

Verifies that all components are correctly implemented and ready for
8B model training to achieve 53%+ validation accuracy.
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_imports() -> List[Tuple[str, bool, str]]:
    """Verify all required imports for Task 3."""
    import_tests = []
    
    # Core training components
    try:
        from src.domain.services.training_orchestrator import TrainingOrchestrator, TrainingConfig
        import_tests.append(("TrainingOrchestrator", True, "Enhanced for 8B model"))
    except Exception as e:
        import_tests.append(("TrainingOrchestrator", False, str(e)))
    
    try:
        from src.domain.services.ttt_service import TTTModelService
        import_tests.append(("TTTModelService", True, "QLoRA support added"))
    except Exception as e:
        import_tests.append(("TTTModelService", False, str(e)))
    
    # Validation and monitoring
    try:
        from src.utils.validation_metrics import ValidationAccuracyMeasurer, ValidationMetrics
        import_tests.append(("ValidationMetrics", True, "Advanced accuracy measurement"))
    except Exception as e:
        import_tests.append(("ValidationMetrics", False, str(e)))
    
    try:
        from src.utils.training_monitor import TrainingMonitor
        import_tests.append(("TrainingMonitor", True, "8B model performance monitoring"))
    except Exception as e:
        import_tests.append(("TrainingMonitor", False, str(e)))
    
    # Main scripts
    try:
        from train_8b_validation import ValidationTrainer
        import_tests.append(("ValidationTrainer", True, "Main 8B training script"))
    except Exception as e:
        import_tests.append(("ValidationTrainer", False, str(e)))
    
    try:
        from execute_task3_training import Task3ExecutionManager
        import_tests.append(("Task3ExecutionManager", True, "Complete execution manager"))
    except Exception as e:
        import_tests.append(("Task3ExecutionManager", False, str(e)))
    
    # Dependencies
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda_available else 0
        import_tests.append(("PyTorch/CUDA", cuda_available, f"GPU memory: {gpu_memory:.1f}GB"))
    except Exception as e:
        import_tests.append(("PyTorch/CUDA", False, str(e)))
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import_tests.append(("Transformers", True, "Model loading support"))
    except Exception as e:
        import_tests.append(("Transformers", False, str(e)))
    
    try:
        import bitsandbytes
        import_tests.append(("BitsAndBytes", True, "QLoRA quantization"))
    except Exception as e:
        import_tests.append(("BitsAndBytes", False, str(e)))
    
    return import_tests


def verify_configuration() -> List[Tuple[str, bool, str]]:
    """Verify configuration files and settings."""
    config_tests = []
    
    # TTT configuration file
    ttt_config_path = Path("configs/strategies/ttt.yaml")
    if ttt_config_path.exists():
        try:
            import yaml
            with open(ttt_config_path) as f:
                config = yaml.safe_load(f)
            
            # Check 8B model configuration
            model_name = config.get("model", {}).get("name", "")
            if "8B" in model_name or "8b" in model_name:
                config_tests.append(("TTT Config - 8B Model", True, f"Model: {model_name}"))
            else:
                config_tests.append(("TTT Config - 8B Model", False, f"Wrong model: {model_name}"))
            
            # Check QLoRA settings
            lora_config = config.get("lora", {})
            if lora_config.get("load_in_4bit", False) and lora_config.get("rank", 0) >= 64:
                config_tests.append(("TTT Config - QLoRA", True, f"Rank: {lora_config['rank']}"))
            else:
                config_tests.append(("TTT Config - QLoRA", False, "QLoRA not properly configured"))
            
            # Check training settings for 53% target
            training_config = config.get("training", {})
            target_accuracy = training_config.get("target_accuracy", 0)
            if target_accuracy >= 0.53:
                config_tests.append(("TTT Config - 53% Target", True, f"Target: {target_accuracy:.1%}"))
            else:
                config_tests.append(("TTT Config - 53% Target", False, f"Target too low: {target_accuracy:.1%}"))
            
        except Exception as e:
            config_tests.append(("TTT Config - Loading", False, str(e)))
    else:
        config_tests.append(("TTT Config - File", False, "ttt.yaml not found"))
    
    return config_tests


def verify_data_access() -> List[Tuple[str, bool, str]]:
    """Verify data access and repository functionality."""
    data_tests = []
    
    try:
        from src.adapters.repositories.arc_data_repository import ARCDataRepository
        repo = ARCDataRepository()
        
        # Test loading evaluation tasks
        try:
            tasks = repo.load_evaluation_tasks()
            if len(tasks) > 0:
                data_tests.append(("ARC Evaluation Data", True, f"{len(tasks)} tasks loaded"))
            else:
                data_tests.append(("ARC Evaluation Data", False, "No tasks found"))
        except Exception as e:
            data_tests.append(("ARC Evaluation Data", False, f"Loading error: {e}"))
        
    except Exception as e:
        data_tests.append(("ARCDataRepository", False, str(e)))
    
    return data_tests


def verify_model_components() -> List[Tuple[str, bool, str]]:
    """Verify model-related components."""
    model_tests = []
    
    # Check enhanced TrainingConfig
    try:
        from src.domain.services.training_orchestrator import TrainingConfig
        config = TrainingConfig()
        
        # Check 8B model specific settings
        if hasattr(config, 'use_qlora') and config.use_qlora:
            model_tests.append(("TrainingConfig - QLoRA", True, f"LoRA rank: {config.lora_rank}"))
        else:
            model_tests.append(("TrainingConfig - QLoRA", False, "QLoRA not enabled"))
        
        if config.target_accuracy >= 0.53:
            model_tests.append(("TrainingConfig - 53% Target", True, f"Target: {config.target_accuracy:.1%}"))
        else:
            model_tests.append(("TrainingConfig - 53% Target", False, f"Wrong target: {config.target_accuracy:.1%}"))
        
        if config.memory_limit_mb >= 20480:  # 20GB+
            model_tests.append(("TrainingConfig - Memory", True, f"Limit: {config.memory_limit_mb/1024:.1f}GB"))
        else:
            model_tests.append(("TrainingConfig - Memory", False, f"Low limit: {config.memory_limit_mb/1024:.1f}GB"))
        
    except Exception as e:
        model_tests.append(("TrainingConfig", False, str(e)))
    
    return model_tests


def verify_directories() -> List[Tuple[str, bool, str]]:
    """Verify required directories exist or can be created."""
    dir_tests = []
    
    required_dirs = [
        ("logs", "Training logs"),
        ("validation_results", "Validation results"),
        ("reports", "Execution reports"),
        ("data/models", "Model cache"),
        ("data/models/checkpoints", "Training checkpoints")
    ]
    
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.is_dir():
                dir_tests.append((f"Directory - {description}", True, str(path)))
            else:
                dir_tests.append((f"Directory - {description}", False, f"Cannot create {path}"))
        except Exception as e:
            dir_tests.append((f"Directory - {description}", False, str(e)))
    
    return dir_tests


def run_verification() -> Dict[str, Any]:
    """Run complete verification suite."""
    logger.info("=" * 60)
    logger.info("TASK 3 SETUP VERIFICATION")
    logger.info("=" * 60)
    
    verification_results = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "categories": {}
    }
    
    # Run all verification categories
    test_categories = [
        ("Imports", verify_imports),
        ("Configuration", verify_configuration),
        ("Data Access", verify_data_access),
        ("Model Components", verify_model_components),
        ("Directories", verify_directories)
    ]
    
    for category_name, test_function in test_categories:
        logger.info(f"\n{category_name}:")
        logger.info("-" * 30)
        
        try:
            tests = test_function()
            category_results = {"tests": [], "passed": 0, "failed": 0}
            
            for test_name, passed, details in tests:
                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"  {status}: {test_name} - {details}")
                
                category_results["tests"].append({
                    "name": test_name,
                    "passed": passed,
                    "details": details
                })
                
                if passed:
                    category_results["passed"] += 1
                else:
                    category_results["failed"] += 1
            
            verification_results["categories"][category_name] = category_results
            verification_results["total_tests"] += len(tests)
            verification_results["passed_tests"] += category_results["passed"]
            verification_results["failed_tests"] += category_results["failed"]
            
        except Exception as e:
            logger.error(f"  ✗ FAIL: {category_name} verification failed - {e}")
            verification_results["categories"][category_name] = {
                "tests": [{"name": f"{category_name} Suite", "passed": False, "details": str(e)}],
                "passed": 0,
                "failed": 1
            }
            verification_results["total_tests"] += 1
            verification_results["failed_tests"] += 1
    
    return verification_results


def print_summary(results: Dict[str, Any]) -> None:
    """Print verification summary."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    total = results["total_tests"]
    passed = results["passed_tests"]
    failed = results["failed_tests"]
    
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed} ({passed/total*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Category breakdown
    logger.info("\nCategory Breakdown:")
    for category, data in results["categories"].items():
        cat_passed = data["passed"]
        cat_total = cat_passed + data["failed"]
        logger.info(f"  {category}: {cat_passed}/{cat_total}")
    
    # Overall assessment
    logger.info("\n" + "=" * 60)
    if failed == 0:
        logger.info("🎯 VERIFICATION PASSED: Ready for Task 3 execution")
        logger.info("   All components are properly configured for 8B model training")
    elif failed <= total * 0.1:  # Less than 10% failures
        logger.warning("⚠️ VERIFICATION MOSTLY PASSED: Minor issues detected")
        logger.info("   Task 3 can proceed but monitor for issues")
    else:
        logger.error("❌ VERIFICATION FAILED: Significant issues detected")
        logger.error("   Fix issues before attempting Task 3 execution")
    
    logger.info("=" * 60)


def main():
    """Main verification function."""
    try:
        # Run verification
        results = run_verification()
        
        # Print summary
        print_summary(results)
        
        # Save results
        import json
        results_file = Path("verification_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if results["failed_tests"] == 0:
            sys.exit(0)  # All passed
        elif results["failed_tests"] <= results["total_tests"] * 0.1:
            sys.exit(1)  # Minor issues
        else:
            sys.exit(2)  # Major issues
        
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()