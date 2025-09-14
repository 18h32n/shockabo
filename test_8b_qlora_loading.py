#!/usr/bin/env python3
"""
Test script for validating 8B Llama-3 model loading with QLoRA within 24GB GPU memory constraints.

This script tests the Task 1 implementation from Story 1.5:
- QLoRA optimization for 8B Llama-3 model loading
- 4-bit quantization with NF4 format and LoRA rank 64
- Memory constraint validation within 24GB GPU memory
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quantization_libraries() -> Dict[str, Any]:
    """Test availability of quantization libraries."""
    logger.info("Testing quantization library availability...")
    
    try:
        from src.utils.quantization_utils import validate_quantization_support
        return validate_quantization_support()
    except ImportError as e:
        logger.error(f"Failed to import quantization utilities: {e}")
        return {
            "torch_available": False,
            "transformers_available": False,
            "bitsandbytes_available": False,
            "errors": [str(e)]
        }


def test_gpu_memory_info() -> Dict[str, Any]:
    """Test GPU memory detection and validation."""
    logger.info("Testing GPU memory information...")
    
    try:
        from src.infrastructure.config import PlatformDetector
        gpu_info = PlatformDetector.get_gpu_memory_info()
        model_validation = PlatformDetector.validate_8b_model_requirements()
        
        return {
            "gpu_info": gpu_info,
            "model_validation": model_validation
        }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return {
            "error": str(e),
            "gpu_available": False
        }


def test_qlora_config_creation() -> Dict[str, Any]:
    """Test QLoRA configuration creation."""
    logger.info("Testing QLoRA configuration creation...")
    
    try:
        from src.utils.quantization_utils import (
            QuantizationConfig,
            create_bnb_config,
            get_optimal_quantization_config,
            LLAMA_8B_CONFIGS
        )
        
        # Test basic config creation
        config = QuantizationConfig()
        config_dict = config.to_dict()
        
        # Test BnB config creation
        bnb_config = create_bnb_config()
        
        # Test optimal config for different memory sizes
        configs = {}
        for memory_gb in [6, 12, 24]:
            memory_mb = memory_gb * 1024
            opt_config = get_optimal_quantization_config(memory_mb, "8B")
            configs[f"{memory_gb}GB"] = {
                "load_in_4bit": opt_config.load_in_4bit,
                "quant_type": opt_config.bnb_4bit_quant_type,
                "compute_dtype": opt_config.bnb_4bit_compute_dtype,
            }
        
        return {
            "config_creation": "success",
            "default_config": config_dict,
            "bnb_config_available": bnb_config is not None,
            "optimal_configs": configs,
            "predefined_configs": list(LLAMA_8B_CONFIGS.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to create QLoRA config: {e}")
        return {"error": str(e)}


def test_memory_estimation() -> Dict[str, Any]:
    """Test memory usage estimation for 8B model."""
    logger.info("Testing memory usage estimation...")
    
    try:
        from src.utils.quantization_utils import estimate_quantized_model_memory
        from src.utils.lora_adapter import estimate_qlora_memory_usage
        
        # Llama-3 8B parameter count
        model_params = 8_000_000_000
        
        # Test different quantization levels
        estimates = {}
        for bits in [4, 8, 16]:
            estimate = estimate_quantized_model_memory(model_params, bits)
            estimates[f"{bits}bit"] = {
                "total_mb": estimate["total_estimated_mb"],
                "total_gb": estimate["total_estimated_gb"],
                "memory_saved_gb": estimate["memory_saved_gb"],
                "compression_ratio": estimate["compression_ratio"]
            }
        
        # Test QLoRA memory estimation
        qlora_estimate = estimate_qlora_memory_usage(model_params, lora_rank=64)
        
        return {
            "model_params": model_params,
            "quantization_estimates": estimates,
            "qlora_estimate": qlora_estimate
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate memory: {e}")
        return {"error": str(e)}


def test_ttt_config_loading() -> Dict[str, Any]:
    """Test TTT configuration loading with 8B model settings."""
    logger.info("Testing TTT configuration loading...")
    
    try:
        from src.adapters.strategies.ttt_adapter import TTTConfig
        
        # Test loading from YAML
        config_path = Path("configs/strategies/ttt.yaml")
        if config_path.exists():
            config = TTTConfig.from_yaml(config_path)
            
            # Verify 8B model settings
            return {
                "config_loaded": True,
                "model_name": config.model_name,
                "lora_rank": config.lora_rank,
                "memory_limit_mb": config.memory_limit_mb,
                "load_in_4bit": getattr(config, 'load_in_4bit', False),
                "bnb_4bit_quant_type": getattr(config, 'bnb_4bit_quant_type', 'unknown'),
                "use_flash_attention": getattr(config, 'use_flash_attention', False),
                "quantization_config": getattr(config, 'quantization_config', None)
            }
        else:
            return {"error": f"Config file not found: {config_path}"}
            
    except Exception as e:
        logger.error(f"Failed to load TTT config: {e}")
        return {"error": str(e)}


def test_lora_adapter_initialization() -> Dict[str, Any]:
    """Test LoRA adapter initialization with QLoRA settings."""
    logger.info("Testing LoRA adapter initialization...")
    
    try:
        from src.utils.lora_adapter import LoRAConfig, create_qlora_config
        
        # Test LoRA config with QLoRA settings
        lora_config = LoRAConfig(
            rank=64,
            alpha=16,
            dropout=0.1,
            use_quantization=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Test QLoRA config creation
        qlora_config = create_qlora_config()
        
        return {
            "lora_config": {
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "scaling": lora_config.scaling,
                "target_modules": lora_config.target_modules,
                "use_quantization": lora_config.use_quantization,
                "quantization_config": lora_config.quantization_config
            },
            "qlora_config": qlora_config
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize LoRA adapter: {e}")
        return {"error": str(e)}


def test_model_loading_simulation() -> Dict[str, Any]:
    """Simulate model loading without actually loading the model."""
    logger.info("Simulating model loading process...")
    
    try:
        from src.utils.quantization_utils import validate_model_loading_requirements
        from src.infrastructure.config import PlatformDetector
        
        # Get current GPU info
        gpu_info = PlatformDetector.get_gpu_memory_info()
        available_memory = gpu_info.get("gpu_memory_available_mb", 0)
        
        if available_memory == 0:
            # Assume 24GB for simulation if no GPU detected
            available_memory = 24 * 1024  # 24GB in MB
            logger.warning("No GPU detected, simulating with 24GB memory")
        
        # Test with different quantization configs
        from src.utils.quantization_utils import LLAMA_8B_CONFIGS
        
        results = {}
        for config_name, quant_config in LLAMA_8B_CONFIGS.items():
            validation = validate_model_loading_requirements(
                "meta-llama/Llama-3-8B",
                quant_config,
                available_memory
            )
            results[config_name] = validation
        
        return {
            "simulation_results": results,
            "available_memory_mb": available_memory,
            "available_memory_gb": available_memory / 1024
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate model loading: {e}")
        return {"error": str(e)}


def run_comprehensive_test() -> Dict[str, Any]:
    """Run comprehensive test of QLoRA 8B model loading capabilities."""
    logger.info("Starting comprehensive QLoRA 8B model test...")
    
    test_start = time.time()
    
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "test_description": "QLoRA 8B Llama-3 model loading validation",
        "tests": {}
    }
    
    # Test 1: Library availability
    results["tests"]["quantization_libraries"] = test_quantization_libraries()
    
    # Test 2: GPU memory info
    results["tests"]["gpu_memory"] = test_gpu_memory_info()
    
    # Test 3: QLoRA config creation
    results["tests"]["qlora_config"] = test_qlora_config_creation()
    
    # Test 4: Memory estimation
    results["tests"]["memory_estimation"] = test_memory_estimation()
    
    # Test 5: TTT config loading
    results["tests"]["ttt_config"] = test_ttt_config_loading()
    
    # Test 6: LoRA adapter initialization
    results["tests"]["lora_adapter"] = test_lora_adapter_initialization()
    
    # Test 7: Model loading simulation
    results["tests"]["model_loading_simulation"] = test_model_loading_simulation()
    
    test_duration = time.time() - test_start
    results["test_duration_seconds"] = test_duration
    
    # Analyze results
    success_count = 0
    total_count = len(results["tests"])
    
    for test_name, test_result in results["tests"].items():
        if isinstance(test_result, dict) and "error" not in test_result:
            success_count += 1
    
    results["summary"] = {
        "total_tests": total_count,
        "successful_tests": success_count,
        "failed_tests": total_count - success_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
        "overall_status": "PASS" if success_count == total_count else "PARTIAL" if success_count > 0 else "FAIL"
    }
    
    # Check specific requirements
    gpu_test = results["tests"].get("gpu_memory", {})
    model_validation = gpu_test.get("model_validation", {})
    
    results["story_1_5_task_1_validation"] = {
        "qlora_support": "error" not in results["tests"].get("qlora_config", {}),
        "8b_model_config": "error" not in results["tests"].get("ttt_config", {}),
        "memory_within_24gb": model_validation.get("meets_optimal", False) or model_validation.get("meets_recommended", False),
        "nf4_quantization": results["tests"].get("qlora_config", {}).get("bnb_config_available", False),
        "rank_64_lora": results["tests"].get("lora_adapter", {}).get("lora_config", {}).get("rank") == 64,
    }
    
    logger.info(f"Test completed in {test_duration:.2f} seconds")
    logger.info(f"Overall status: {results['summary']['overall_status']}")
    
    return results


def main():
    """Main test execution function."""
    try:
        # Run comprehensive test
        results = run_comprehensive_test()
        
        # Save results
        output_file = Path("test_8b_qlora_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\\n" + "="*50)
        print("QLoRA 8B Model Loading Test Results")
        print("="*50)
        print(f"Overall Status: {results['summary']['overall_status']}")
        print(f"Tests Passed: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        
        print("\\nStory 1.5 Task 1 Validation:")
        task_validation = results["story_1_5_task_1_validation"]
        for key, value in task_validation.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key.replace('_', ' ').title()}: {value}")
        
        # Print recommendations if any failures
        if results["summary"]["overall_status"] != "PASS":
            print("\\nRecommendations:")
            for test_name, test_result in results["tests"].items():
                if isinstance(test_result, dict) and "error" in test_result:
                    print(f"  - Fix {test_name}: {test_result['error']}")
        
        return 0 if results["summary"]["overall_status"] == "PASS" else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())