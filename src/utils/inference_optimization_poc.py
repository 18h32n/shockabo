"""
Inference Optimization POC for 8B Model

This module benchmarks inference optimizations to ensure single task inference
stays under the 7.2 minute requirement (critical risk PERF-002).
"""
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceProfile:
    """Profile for inference optimization testing."""
    optimization_name: str
    model_name: str
    task_prompt: str
    inference_time_seconds: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    optimization_config: Optional[Dict[str, Any]] = None
    meets_time_requirement: bool = False


@dataclass
class ARCTaskSample:
    """Sample ARC task for benchmarking."""
    task_id: str
    prompt: str
    expected_output_length: int
    complexity_level: str  # simple, medium, complex


class InferenceOptimizer:
    """Inference optimization and benchmarking for 8B models."""
    
    def __init__(self, time_limit_seconds: float = 432):  # 7.2 minutes
        """
        Initialize inference optimizer.
        
        Args:
            time_limit_seconds: Maximum allowed inference time
        """
        self.time_limit_seconds = time_limit_seconds
        self.profiles: List[InferenceProfile] = []
        
    def get_arc_task_samples(self) -> List[ARCTaskSample]:
        """Get representative ARC task samples for benchmarking."""
        return [
            ARCTaskSample(
                task_id="simple_pattern",
                prompt="""Analyze this grid pattern and predict the output:

Input grid:
[[1, 0, 1],
 [0, 1, 0],
 [1, 0, 1]]

Pattern: The grid shows an alternating pattern of 1s and 0s.
Task: Generate the next transformation.

Think step by step:
1. Identify the pattern
2. Apply the transformation rule
3. Generate the output grid

Output:""",
                expected_output_length=200,
                complexity_level="simple"
            ),
            ARCTaskSample(
                task_id="medium_pattern",
                prompt="""Analyze this complex grid pattern:

Input grid:
[[2, 1, 2, 1],
 [1, 3, 1, 3],
 [2, 1, 2, 1],
 [1, 3, 1, 3]]

Training examples:
Example 1: [similar pattern] -> [transformation result]
Example 2: [similar pattern] -> [transformation result]

Pattern analysis: This involves color substitution and geometric transformation.
Task: Apply the learned pattern to generate the output.

Reasoning:
- Step 1: Identify color relationships
- Step 2: Detect geometric transformations
- Step 3: Apply rules to generate result
- Step 4: Verify pattern consistency

Output:""",
                expected_output_length=400,
                complexity_level="medium"
            ),
            ARCTaskSample(
                task_id="complex_pattern",
                prompt="""Solve this advanced ARC challenge:

Training examples:
Input 1: [[0,1,0,2,0],[1,0,1,0,1],[0,1,0,2,0],[2,0,2,0,2],[0,1,0,2,0]]
Output 1: [[0,1,0,2,0],[1,2,1,2,1],[0,1,0,2,0],[2,1,2,1,2],[0,1,0,2,0]]

Input 2: [[1,2,1,0,1],[2,1,2,1,2],[1,2,1,0,1],[0,1,0,1,0],[1,2,1,0,1]]
Output 2: [[1,2,1,0,1],[2,0,2,0,2],[1,2,1,0,1],[0,2,0,2,0],[1,2,1,0,1]]

Test input: [[0,2,0,1,0],[2,0,2,0,2],[0,2,0,1,0],[1,0,1,0,1],[0,2,0,1,0]]

Analysis required:
1. Pattern recognition across multiple examples
2. Rule abstraction and generalization
3. Systematic application to test case
4. Verification of logical consistency

Detailed reasoning:
- Analyze color mappings between input and output
- Identify positional transformation rules
- Consider symmetry and geometric properties
- Apply discovered patterns systematically

Generate the solution with full reasoning:""",
                expected_output_length=800,
                complexity_level="complex"
            )
        ]
    
    def create_optimized_model_configs(self) -> List[Dict[str, Any]]:
        """Create different optimization configurations to test."""
        return [
            {
                "name": "baseline",
                "description": "Standard loading without optimizations",
                "quantization_config": None,
                "torch_compile": False,
                "flash_attention": False,
                "gradient_checkpointing": False,
            },
            {
                "name": "qlora_only",
                "description": "QLoRA quantization only",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": False,
                "flash_attention": False,
                "gradient_checkpointing": False,
            },
            {
                "name": "qlora_flash_attention",
                "description": "QLoRA + Flash Attention 2",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": False,
                "flash_attention": True,
                "gradient_checkpointing": False,
            },
            {
                "name": "qlora_torch_compile",
                "description": "QLoRA + Torch Compile",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": True,
                "flash_attention": False,
                "gradient_checkpointing": False,
            },
            {
                "name": "full_optimization",
                "description": "All optimizations enabled",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": True,
                "flash_attention": True,
                "gradient_checkpointing": False,
            },
            {
                "name": "inference_optimized",
                "description": "Inference-specific optimizations with batching",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": True,
                "flash_attention": True,
                "gradient_checkpointing": False,
                "kv_cache_optimization": True,
                "static_cache": True,
            },
            {
                "name": "speed_focused",
                "description": "Maximum speed with minimal memory safety",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,  # Faster but less memory efficient
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "torch_compile": True,
                "flash_attention": True,
                "gradient_checkpointing": False,
                "kv_cache_optimization": True,
                "static_cache": True,
                "low_cpu_mem_usage": False,  # Faster loading
            },
        ]
    
    def load_optimized_model(self, model_name: str, config: Dict[str, Any]) -> tuple[nn.Module, Any]:
        """Load model with specified optimizations."""
        logger.info(f"Loading model with config: {config['name']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": config.get("low_cpu_mem_usage", True),
        }
        
        # Add quantization if specified
        if config["quantization_config"]:
            model_kwargs["quantization_config"] = config["quantization_config"]
        
        # Add flash attention if specified
        if config["flash_attention"] and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Apply KV cache optimizations if specified
        if config.get("kv_cache_optimization", False):
            try:
                if hasattr(model.config, 'use_cache'):
                    model.config.use_cache = True
                
                # Enable static cache if supported
                if config.get("static_cache", False) and hasattr(model, 'enable_static_cache'):
                    model.enable_static_cache()
                    logger.info("Static KV cache enabled")
                    
            except Exception as e:
                logger.warning(f"KV cache optimization failed: {e}")
        
        # Apply torch.compile if specified
        if config["torch_compile"]:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Enable gradient checkpointing if specified
        if config["gradient_checkpointing"] and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def benchmark_inference(
        self,
        model: nn.Module,
        tokenizer: Any,
        task: ARCTaskSample,
        config: Dict[str, Any],
        model_name: str
    ) -> InferenceProfile:
        """Benchmark inference for a single task."""
        logger.info(f"Benchmarking inference for task: {task.task_id}")
        
        # Clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        success = False
        error_message = None
        inference_time = 0
        tokens_generated = 0
        tokens_per_second = 0
        memory_usage_mb = 0
        
        try:
            # Tokenize input
            inputs = tokenizer(
                task.prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Benchmark inference
            start_time = time.time()
            
            with torch.no_grad():
                # Determine generation parameters based on optimization config
                generation_kwargs = {
                    "max_new_tokens": task.expected_output_length,
                    "min_new_tokens": min(50, task.expected_output_length // 2),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": config.get("kv_cache_optimization", True),
                }
                
                # Use different sampling strategies based on config
                if config["name"] in ["speed_focused", "inference_optimized"]:
                    # Faster, more deterministic generation
                    generation_kwargs.update({
                        "do_sample": False,
                        "num_beams": 1,  # No beam search for speed
                        "repetition_penalty": 1.1,
                    })
                else:
                    # Higher quality but slower generation
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                    })
                
                # Use optimized generation parameters
                outputs = model.generate(**inputs, **generation_kwargs)
            
            inference_time = time.time() - start_time
            
            # Calculate tokens generated
            input_length = inputs["input_ids"].shape[1]
            total_length = outputs.shape[1]
            tokens_generated = total_length - input_length
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            # Get memory usage
            if torch.cuda.is_available():
                memory_usage_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Decode output for logging
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            logger.info(f"Generated {tokens_generated} tokens in {inference_time:.2f}s ({tokens_per_second:.2f} tok/s)")
            logger.debug(f"Generated text preview: {generated_text[:200]}...")
            
            success = True
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Inference failed: {error_message}")
        
        # Create profile
        profile = InferenceProfile(
            optimization_name=config["name"],
            model_name=model_name,
            task_prompt=task.task_id,
            inference_time_seconds=inference_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage_mb,
            success=success,
            error_message=error_message,
            optimization_config=config,
            meets_time_requirement=inference_time <= self.time_limit_seconds
        )
        
        self.profiles.append(profile)
        return profile
    
    def run_comprehensive_benchmark(self, model_name: str = "meta-llama/Llama-3-8B") -> List[InferenceProfile]:
        """Run comprehensive inference benchmarking with 8B model support."""
        logger.info("Starting comprehensive inference benchmarking...")
        logger.info(f"Target model: {model_name}")
        logger.info(f"Time limit: {self.time_limit_seconds}s ({self.time_limit_seconds/60:.1f} minutes)")
        
        tasks = self.get_arc_task_samples()
        configs = self.create_optimized_model_configs()
        results = []
        
        # Track best configuration found so far
        best_config = None
        best_time = float('inf')
        
        for config in configs:
            logger.info(f"Testing optimization: {config['name']} - {config['description']}")
            
            try:
                # Load model with current optimization
                start_load = time.time()
                model, tokenizer = self.load_optimized_model(model_name, config)
                load_time = time.time() - start_load
                logger.info(f"Model loading time: {load_time:.2f}s")
                
                # Test on each task complexity level
                config_results = []
                for task in tasks:
                    profile = self.benchmark_inference(model, tokenizer, task, config, model_name)
                    results.append(profile)
                    config_results.append(profile)
                    
                    # Log immediate results
                    self._log_inference_result(profile)
                    
                    # Track best configuration
                    if profile.success and profile.meets_time_requirement:
                        if profile.inference_time_seconds < best_time:
                            best_time = profile.inference_time_seconds
                            best_config = config['name']
                    
                    # Early stop if inference is too slow for this configuration
                    if profile.inference_time_seconds > self.time_limit_seconds * 1.5:
                        logger.warning(f"Inference too slow ({profile.inference_time_seconds:.1f}s), skipping remaining tasks for this config")
                        break
                
                # Log configuration summary
                successful_tasks = [p for p in config_results if p.success]
                within_limit_tasks = [p for p in successful_tasks if p.meets_time_requirement]
                
                logger.info(f"Configuration '{config['name']}' summary:")
                logger.info(f"  Success rate: {len(successful_tasks)}/{len(config_results)} tasks")
                logger.info(f"  Within time limit: {len(within_limit_tasks)}/{len(successful_tasks)} successful tasks")
                
                if within_limit_tasks:
                    avg_time = sum(p.inference_time_seconds for p in within_limit_tasks) / len(within_limit_tasks)
                    avg_throughput = sum(p.tokens_per_second for p in within_limit_tasks) / len(within_limit_tasks)
                    logger.info(f"  Average time: {avg_time:.2f}s")
                    logger.info(f"  Average throughput: {avg_throughput:.1f} tok/s")
                
                # Cleanup
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to test optimization {config['name']}: {e}")
                # Create failure profile for tracking
                for task in tasks:
                    failure_profile = InferenceProfile(
                        optimization_name=config["name"],
                        model_name=model_name,
                        task_prompt=task.task_id,
                        inference_time_seconds=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        memory_usage_mb=0,
                        success=False,
                        error_message=str(e),
                        optimization_config=config,
                        meets_time_requirement=False
                    )
                    results.append(failure_profile)
                continue
        
        # Log final summary
        successful_profiles = [p for p in results if p.success]
        within_limit_profiles = [p for p in successful_profiles if p.meets_time_requirement]
        
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK COMPLETION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tests run: {len(results)}")
        logger.info(f"Successful tests: {len(successful_profiles)}")
        logger.info(f"Tests within time limit: {len(within_limit_profiles)}")
        logger.info(f"Best configuration: {best_config or 'None'}")
        if best_config:
            logger.info(f"Best time achieved: {best_time:.2f}s")
        logger.info(f"{'='*80}")
        
        return results
    
    def _log_inference_result(self, profile: InferenceProfile) -> None:
        """Log inference result summary."""
        status = "✓ SUCCESS" if profile.success else "✗ FAILED"
        time_status = "✓ WITHIN LIMIT" if profile.meets_time_requirement else "⚠ TOO SLOW"
        
        logger.info("=" * 60)
        logger.info(f"Optimization: {profile.optimization_name}")
        logger.info(f"Task: {profile.task_prompt}")
        logger.info(f"Status: {status}")
        logger.info(f"Time Status: {time_status}")
        logger.info(f"Inference Time: {profile.inference_time_seconds:.2f}s")
        logger.info(f"Tokens Generated: {profile.tokens_generated}")
        logger.info(f"Tokens/Second: {profile.tokens_per_second:.2f}")
        logger.info(f"Memory Usage: {profile.memory_usage_mb:.1f}MB")
        
        if profile.error_message:
            logger.info(f"Error: {profile.error_message}")
        
        logger.info("=" * 60)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmarking results and generate recommendations."""
        if not self.profiles:
            return {"error": "No benchmark results available"}
        
        successful_profiles = [p for p in self.profiles if p.success]
        within_time_profiles = [p for p in successful_profiles if p.meets_time_requirement]
        
        # Find best configuration
        best_profile = None
        if within_time_profiles:
            best_profile = min(within_time_profiles, key=lambda p: p.inference_time_seconds)
        
        # Analyze by optimization type
        optimization_analysis = {}
        for profile in successful_profiles:
            opt_name = profile.optimization_name
            if opt_name not in optimization_analysis:
                optimization_analysis[opt_name] = {
                    "profiles": [],
                    "avg_time": 0,
                    "avg_throughput": 0,
                    "success_rate": 0,
                    "within_time_rate": 0
                }
            optimization_analysis[opt_name]["profiles"].append(profile)
        
        # Calculate statistics for each optimization
        for opt_name, data in optimization_analysis.items():
            profiles = data["profiles"]
            data["avg_time"] = sum(p.inference_time_seconds for p in profiles) / len(profiles)
            data["avg_throughput"] = sum(p.tokens_per_second for p in profiles) / len(profiles)
            data["success_rate"] = len([p for p in profiles if p.success]) / len(profiles)
            data["within_time_rate"] = len([p for p in profiles if p.meets_time_requirement]) / len(profiles)
        
        return {
            "summary": {
                "total_tests": len(self.profiles),
                "successful_tests": len(successful_profiles),
                "within_time_limit": len(within_time_profiles),
                "time_limit_seconds": self.time_limit_seconds,
                "inference_feasible": len(within_time_profiles) > 0,
            },
            "best_configuration": asdict(best_profile) if best_profile else None,
            "optimization_analysis": optimization_analysis,
            "recommendations": self._generate_inference_recommendations(optimization_analysis, best_profile),
            "risk_assessment": {
                "perf_002_status": "MITIGATED" if best_profile else "CRITICAL",
                "optimization_required": True,
                "fallback_needed": best_profile is None,
            },
            "all_profiles": [asdict(p) for p in self.profiles],
        }
    
    def _generate_inference_recommendations(
        self, 
        optimization_analysis: Dict[str, Any], 
        best_profile: Optional[InferenceProfile]
    ) -> List[str]:
        """Generate recommendations based on inference benchmarking."""
        recommendations = []
        
        if not best_profile:
            recommendations.extend([
                "CRITICAL: No optimization configuration meets 7.2-minute requirement",
                "Implement model distillation to smaller size",
                "Consider switching to 7B model as primary strategy",
                "Investigate specialized inference frameworks (vLLM, TensorRT-LLM)",
                "Use progressive inference with early stopping",
            ])
        else:
            recommendations.extend([
                f"SUCCESS: Use {best_profile.optimization_name} optimization",
                f"Achieved {best_profile.inference_time_seconds:.1f}s inference time",
                f"Throughput: {best_profile.tokens_per_second:.1f} tokens/second",
                "Monitor inference time in production",
                "Implement timeout mechanisms for safety",
            ])
        
        # Analyze best optimization strategies
        if optimization_analysis:
            best_opt = min(optimization_analysis.items(), 
                          key=lambda x: x[1]["avg_time"] if x[1]["success_rate"] > 0 else float('inf'))
            
            recommendations.extend([
                f"Best optimization strategy: {best_opt[0]}",
                f"Average inference time: {best_opt[1]['avg_time']:.2f}s",
                f"Average throughput: {best_opt[1]['avg_throughput']:.1f} tok/s",
            ])
        
        # General recommendations
        recommendations.extend([
            "Implement KV cache optimization",
            "Use batched inference where possible",
            "Set up inference performance monitoring",
            "Create progressive timeout mechanisms",
            "Test on representative ARC tasks regularly",
        ])
        
        return recommendations
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive inference optimization report."""
        analysis = self.analyze_results()
        
        report = {
            **analysis,
            "timestamp": time.time(),
            "test_configuration": {
                "time_limit_seconds": self.time_limit_seconds,
                "time_limit_minutes": self.time_limit_seconds / 60,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        }
        
        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Inference optimization report saved to: {output_path}")
        
        return report


def main():
    """Run inference optimization POC."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize optimizer
    optimizer = InferenceOptimizer(time_limit_seconds=432)  # 7.2 minutes
    
    # Run benchmarks with 8B model (use smaller model for testing if needed)
    logger.info("Starting inference optimization benchmarking...")
    
    # Detect if we should use 8B model based on available GPU memory
    model_name = "meta-llama/Llama-3-8B"  # Default to 8B model
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Available GPU memory: {gpu_memory_gb:.2f}GB")
        
        if gpu_memory_gb < 12:
            # Use smaller model for testing on limited hardware
            model_name = "microsoft/DialoGPT-large"
            logger.warning(f"Using smaller test model due to limited GPU memory ({gpu_memory_gb:.2f}GB < 12GB required)")
        else:
            logger.info("Using Llama-3-8B model for inference optimization testing")
    else:
        # Use smaller model for CPU testing
        model_name = "microsoft/DialoGPT-large"
        logger.warning("Using smaller test model for CPU-only environment")
    
    results = optimizer.run_comprehensive_benchmark(model_name)
    
    # Generate report
    report_path = "docs/qa/assessments/inference_optimization_poc_results.json"
    report = optimizer.generate_report(report_path)
    
    # Print summary
    print("\n" + "="*80)
    print("INFERENCE OPTIMIZATION POC RESULTS")
    print("="*80)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful Tests: {report['summary']['successful_tests']}")
    print(f"Within Time Limit: {report['summary']['within_time_limit']}")
    print(f"Inference Feasible: {'YES' if report['summary']['inference_feasible'] else 'NO'}")
    print(f"Risk PERF-002 Status: {report['risk_assessment']['perf_002_status']}")
    
    if report['best_configuration']:
        best = report['best_configuration']
        print(f"\nBest Configuration:")
        print(f"  Optimization: {best['optimization_name']}")
        print(f"  Inference Time: {best['inference_time_seconds']:.2f}s")
        print(f"  Throughput: {best['tokens_per_second']:.1f} tok/s")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nFull report saved to: {report_path}")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()