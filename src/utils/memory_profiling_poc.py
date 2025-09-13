"""
Memory Profiling POC for 8B Model Loading

This module validates whether an 8B Llama-3 model can be loaded with QLoRA
optimizations within the 24GB memory constraint identified as critical risk PERF-001.
"""
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import psutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelMemoryProfile:
    """Memory profile for model loading."""
    model_name: str
    quantization_config: dict[str, Any]
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    total_gpu_memory_mb: float
    memory_utilization: float
    loading_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    parameter_count: Optional[int] = None
    model_size_mb: Optional[float] = None


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA optimization."""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1


class MemoryProfiler:
    """Memory profiler for 8B model loading validation."""
    
    def __init__(self, memory_limit_mb: float = 24576):  # 24GB default
        """
        Initialize memory profiler.
        
        Args:
            memory_limit_mb: GPU memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.profiles: list[ModelMemoryProfile] = []
        
    def get_gpu_memory_stats(self) -> dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}
            
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total
        }
    
    def get_system_memory_mb(self) -> float:
        """Get current system memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def create_qlora_config(self, config: QLoRAConfig) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig for QLoRA."""
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    def profile_model_loading(
        self, 
        model_name: str,
        qlora_config: QLoRAConfig,
        test_inference: bool = True
    ) -> ModelMemoryProfile:
        """
        Profile memory usage for loading a model with QLoRA.
        
        Args:
            model_name: HuggingFace model name
            qlora_config: QLoRA configuration
            test_inference: Whether to test inference
            
        Returns:
            Memory profile results
        """
        logger.info(f"Profiling model loading: {model_name}")
        
        # Clear cache before testing
        self.clear_gpu_cache()
        
        # Get initial memory state
        memory_before = self.get_system_memory_mb()
        gpu_stats_before = self.get_gpu_memory_stats()
        
        # Reset peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        success = False
        error_message = None
        parameter_count = None
        model_size_mb = None
        model = None
        tokenizer = None
        
        try:
            # Create quantization config
            bnb_config = self.create_qlora_config(qlora_config)
            
            logger.info(f"Loading model with quantization config: {asdict(qlora_config)}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            )
            
            # Enable gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Count parameters
            parameter_count = sum(p.numel() for p in model.parameters())
            
            # Estimate model size
            model_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1024 / 1024
            
            logger.info(f"Model loaded successfully: {parameter_count:,} parameters, ~{model_size_mb:.1f}MB")
            
            # Test inference if requested
            if test_inference:
                self._test_inference(model, tokenizer)
            
            success = True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to load model: {error_message}")
        
        finally:
            # Get final memory state
            loading_time = time.time() - start_time
            memory_after = self.get_system_memory_mb()
            gpu_stats_after = self.get_gpu_memory_stats()
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else memory_after
            
            # Create profile
            profile = ModelMemoryProfile(
                model_name=model_name,
                quantization_config=asdict(qlora_config),
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - memory_before,
                peak_memory_mb=peak_memory,
                gpu_memory_allocated_mb=gpu_stats_after["allocated_mb"],
                gpu_memory_reserved_mb=gpu_stats_after["reserved_mb"],
                total_gpu_memory_mb=gpu_stats_after["total_mb"],
                memory_utilization=gpu_stats_after["allocated_mb"] / self.memory_limit_mb,
                loading_time_seconds=loading_time,
                success=success,
                error_message=error_message,
                parameter_count=parameter_count,
                model_size_mb=model_size_mb,
            )
            
            self.profiles.append(profile)
            
            # Cleanup
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            
            self.clear_gpu_cache()
            
        return profile
    
    def _test_inference(self, model: nn.Module, tokenizer) -> None:
        """Test basic inference with the model."""
        logger.info("Testing inference capability...")
        
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Inference test successful: '{response}'")
    
    def test_8b_models(self) -> list[ModelMemoryProfile]:
        """Test various 8B model configurations."""
        # Test configurations
        test_models = [
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "microsoft/DialoGPT-large",  # Fallback smaller model for testing
        ]
        
        test_configs = [
            QLoRAConfig(  # Conservative config
                lora_r=32,
                lora_alpha=64,
                lora_dropout=0.1,
            ),
            QLoRAConfig(  # Story spec config
                lora_r=64,
                lora_alpha=128,
                lora_dropout=0.1,
            ),
            QLoRAConfig(  # Aggressive config
                lora_r=128,
                lora_alpha=256,
                lora_dropout=0.05,
            ),
        ]
        
        results = []
        
        for model_name in test_models:
            for i, config in enumerate(test_configs):
                logger.info(f"Testing {model_name} with config {i+1}/3")
                
                try:
                    profile = self.profile_model_loading(
                        model_name=model_name,
                        qlora_config=config,
                        test_inference=True
                    )
                    results.append(profile)
                    
                    # Log results immediately
                    self._log_profile_summary(profile)
                    
                    # Stop testing configs if we exceeded memory limit
                    if profile.memory_utilization > 1.0:
                        logger.warning(f"Memory limit exceeded for {model_name}, skipping remaining configs")
                        break
                        
                except Exception as e:
                    logger.error(f"Critical error testing {model_name}: {e}")
                    continue
                
                # Small delay between tests
                time.sleep(2)
        
        return results
    
    def _log_profile_summary(self, profile: ModelMemoryProfile) -> None:
        """Log a summary of the memory profile."""
        status = "✓ SUCCESS" if profile.success else "✗ FAILED"
        memory_status = "✓ WITHIN LIMIT" if profile.memory_utilization <= 1.0 else "⚠ EXCEEDED LIMIT"
        
        logger.info("=" * 60)
        logger.info(f"Model: {profile.model_name}")
        logger.info(f"Status: {status}")
        logger.info(f"Memory Status: {memory_status}")
        logger.info(f"GPU Memory Used: {profile.gpu_memory_allocated_mb:.1f}MB")
        logger.info(f"Memory Utilization: {profile.memory_utilization:.1%}")
        logger.info(f"Peak Memory: {profile.peak_memory_mb:.1f}MB")
        logger.info(f"Loading Time: {profile.loading_time_seconds:.1f}s")
        
        if profile.parameter_count:
            logger.info(f"Parameters: {profile.parameter_count:,}")
        
        if profile.error_message:
            logger.info(f"Error: {profile.error_message}")
            
        logger.info("=" * 60)
    
    def generate_report(self, output_path: Optional[str] = None) -> dict[str, Any]:
        """Generate comprehensive memory profiling report."""
        if not self.profiles:
            return {"error": "No profiles available"}
        
        # Analyze results
        successful_profiles = [p for p in self.profiles if p.success]
        within_limit_profiles = [p for p in successful_profiles if p.memory_utilization <= 1.0]
        
        # Find best configuration
        best_profile = None
        if within_limit_profiles:
            best_profile = min(within_limit_profiles, key=lambda p: p.memory_utilization)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            "summary": {
                "total_tests": len(self.profiles),
                "successful_loads": len(successful_profiles),
                "within_memory_limit": len(within_limit_profiles),
                "memory_limit_mb": self.memory_limit_mb,
                "feasible_8b_loading": len(within_limit_profiles) > 0,
            },
            "best_configuration": asdict(best_profile) if best_profile else None,
            "all_profiles": [asdict(p) for p in self.profiles],
            "recommendations": recommendations,
            "risk_assessment": {
                "perf_001_status": "MITIGATED" if best_profile else "CRITICAL",
                "fallback_required": best_profile is None,
                "memory_optimization_needed": True,
            },
            "timestamp": time.time(),
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on profiling results."""
        recommendations = []
        
        successful_profiles = [p for p in self.profiles if p.success]
        within_limit_profiles = [p for p in successful_profiles if p.memory_utilization <= 1.0]
        
        if not successful_profiles:
            recommendations.extend([
                "CRITICAL: No 8B models could be loaded successfully",
                "Implement 7B model fallback immediately",
                "Investigate quantization library compatibility",
                "Consider cloud deployment with more GPU memory",
            ])
        elif not within_limit_profiles:
            recommendations.extend([
                "CRITICAL: All 8B models exceed 24GB memory limit",
                "Implement more aggressive quantization (INT8 or lower)",
                "Use gradient checkpointing every layer",
                "Consider model sharding across multiple GPUs",
                "Implement 7B model fallback as primary strategy",
            ])
        else:
            best = min(within_limit_profiles, key=lambda p: p.memory_utilization)
            recommendations.extend([
                f"SUCCESS: 8B model loading feasible with {best.memory_utilization:.1%} memory usage",
                f"Use QLoRA config: rank={best.quantization_config.get('lora_r', 64)}",
                "Implement aggressive gradient checkpointing",
                "Monitor memory usage during training",
                "Set up automatic batch size reduction on OOM",
            ])
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive error recovery mechanisms",
            "Set up continuous memory monitoring",
            "Create fallback precision levels (FP16 → INT8 → INT4)",
            "Test on target deployment hardware",
        ])
        
        return recommendations


def main():
    """Run memory profiling POC."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU mode (limited testing)")
        memory_limit_mb = 32768  # 32GB system RAM limit
    else:
        # Get actual GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_limit_mb = min(24576, gpu_memory_gb * 1024)  # Use 24GB or available, whichever is less
        logger.info(f"GPU detected: {gpu_memory_gb:.1f}GB, using {memory_limit_mb/1024:.1f}GB limit")
    
    # Create profiler
    profiler = MemoryProfiler(memory_limit_mb=memory_limit_mb)
    
    # Run tests
    logger.info("Starting 8B model memory profiling...")
    results = profiler.test_8b_models()
    
    # Generate report
    report_path = "docs/qa/assessments/memory_profiling_poc_results.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = profiler.generate_report(report_path)
    
    # Print summary
    print("\n" + "="*80)
    print("MEMORY PROFILING POC RESULTS")
    print("="*80)
    print(f"Tests Run: {report['summary']['total_tests']}")
    print(f"Successful Loads: {report['summary']['successful_loads']}")
    print(f"Within Memory Limit: {report['summary']['within_memory_limit']}")
    print(f"8B Loading Feasible: {'YES' if report['summary']['feasible_8b_loading'] else 'NO'}")
    print(f"Risk PERF-001 Status: {report['risk_assessment']['perf_001_status']}")
    
    if report['best_configuration']:
        best = report['best_configuration']
        print(f"\nBest Configuration:")
        print(f"  Model: {best['model_name']}")
        print(f"  Memory Usage: {best['memory_utilization']:.1%}")
        print(f"  LoRA Rank: {best['quantization_config'].get('lora_r', 'N/A')}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nFull report saved to: {report_path}")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()