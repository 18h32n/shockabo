#!/usr/bin/env python3
"""
Comprehensive inference benchmarking script for production validation.

This script validates that single task inference time stays under the 7.2-minute
requirement for the 8B model in production environments.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.inference_optimization_poc import InferenceOptimizer, InferenceProfile
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.services.ttt_service import TTTModelService
from src.domain.models import ARCTask


logger = logging.getLogger(__name__)


class ProductionInferenceBenchmark:
    """Production-ready inference benchmarking system."""
    
    def __init__(self, time_limit_minutes: float = 7.2):
        """
        Initialize production inference benchmark.
        
        Args:
            time_limit_minutes: Maximum allowed inference time in minutes
        """
        self.time_limit_seconds = time_limit_minutes * 60
        self.results: List[Dict[str, Any]] = []
        
    def create_sample_arc_tasks(self) -> List[ARCTask]:
        """Create sample ARC tasks for benchmarking."""
        return [
            ARCTask(
                task_id="benchmark_simple",
                train_examples=[
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[0, 1], [1, 0]]
                    }
                ],
                test_input=[[1, 1], [1, 1]],
                test_output=[[0, 0], [0, 0]],
                metadata={"complexity": "simple", "benchmark": True}
            ),
            ARCTask(
                task_id="benchmark_medium", 
                train_examples=[
                    {
                        "input": [[1, 2, 1], [2, 0, 2], [1, 2, 1]],
                        "output": [[2, 1, 2], [1, 0, 1], [2, 1, 2]]
                    },
                    {
                        "input": [[3, 4, 3], [4, 0, 4], [3, 4, 3]],
                        "output": [[4, 3, 4], [3, 0, 3], [4, 3, 4]]
                    }
                ],
                test_input=[[5, 6, 5], [6, 0, 6], [5, 6, 5]],
                test_output=[[6, 5, 6], [5, 0, 5], [6, 5, 6]],
                metadata={"complexity": "medium", "benchmark": True}
            ),
            ARCTask(
                task_id="benchmark_complex",
                train_examples=[
                    {
                        "input": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                        "output": [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]
                    },
                    {
                        "input": [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]],
                        "output": [[17, 16, 15, 14], [13, 12, 11, 10], [9, 8, 7, 6], [5, 4, 3, 2]]
                    }
                ],
                test_input=[[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]],
                test_output=[[18, 17, 16, 15], [14, 13, 12, 11], [10, 9, 8, 7], [6, 5, 4, 3]],
                metadata={"complexity": "complex", "benchmark": True}
            ),
        ]
    
    def benchmark_ttt_adapter(
        self,
        model_name: str = "meta-llama/Llama-3-8B",
        config_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Benchmark TTT adapter inference performance.
        
        Args:
            model_name: Model identifier to benchmark
            config_path: Path to TTT configuration file
            
        Returns:
            List of benchmark results
        """
        logger.info("Starting TTT Adapter inference benchmarking")
        
        results = []
        tasks = self.create_sample_arc_tasks()
        
        try:
            # Load TTT configuration
            if config_path and config_path.exists():
                config = TTTConfig.from_yaml(config_path)
            else:
                config = TTTConfig(
                    model_name=model_name,
                    max_inference_time=self.time_limit_seconds,
                    enable_torch_compile=True,
                    enable_kv_cache_optimization=True,
                    enable_static_cache=True,
                )
            
            # Initialize TTT adapter
            adapter = TTTAdapter(config)
            adapter.initialize_model()
            
            logger.info(f"TTT Adapter initialized with model: {model_name}")
            
            # Benchmark each task
            for task in tasks:
                logger.info(f"Benchmarking task: {task.task_id}")
                
                start_time = time.time()
                
                try:
                    solution = adapter.solve(task)
                    inference_time = time.time() - start_time
                    
                    result = {
                        "adapter_type": "TTT",
                        "model_name": model_name,
                        "task_id": task.task_id,
                        "task_complexity": task.metadata.get("complexity", "unknown"),
                        "inference_time_seconds": inference_time,
                        "inference_time_minutes": inference_time / 60,
                        "within_time_limit": inference_time <= self.time_limit_seconds,
                        "success": solution.metadata.get("success", False),
                        "confidence_score": solution.confidence_score,
                        "predictions_count": len(solution.predictions),
                        "resource_usage": {
                            "cpu_seconds": solution.resource_usage.cpu_seconds if solution.resource_usage else 0,
                            "memory_mb": solution.resource_usage.memory_mb if solution.resource_usage else 0,
                            "gpu_memory_mb": solution.resource_usage.gpu_memory_mb if solution.resource_usage else 0,
                        },
                        "metadata": solution.metadata,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    results.append(result)
                    self._log_benchmark_result(result)
                    
                except Exception as e:
                    inference_time = time.time() - start_time
                    
                    result = {
                        "adapter_type": "TTT",
                        "model_name": model_name,
                        "task_id": task.task_id,
                        "task_complexity": task.metadata.get("complexity", "unknown"),
                        "inference_time_seconds": inference_time,
                        "inference_time_minutes": inference_time / 60,
                        "within_time_limit": inference_time <= self.time_limit_seconds,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    results.append(result)
                    logger.error(f"Task {task.task_id} failed: {e}")
            
            # Cleanup
            adapter.cleanup()
            
        except Exception as e:
            logger.error(f"TTT Adapter benchmarking failed: {e}")
            
        return results
    
    def benchmark_ttt_service(
        self,
        model_name: str = "meta-llama/Llama-3-8B"
    ) -> List[Dict[str, Any]]:
        """
        Benchmark TTT service inference performance.
        
        Args:
            model_name: Model identifier to benchmark
            
        Returns:
            List of benchmark results
        """
        logger.info("Starting TTT Service inference benchmarking")
        
        results = []
        
        try:
            # Initialize TTT service
            service = TTTModelService()
            model, tokenizer = service.load_model(model_name)
            service.prepare_for_inference()
            
            logger.info(f"TTT Service initialized with model: {model_name}")
            
            # Create test prompts
            test_prompts = [
                "Simple ARC pattern: Transform [[1,0],[0,1]] to [[0,1],[1,0]]. Pattern: ",
                "Medium ARC pattern: Given examples, find transformation rule and apply to test case. Examples: ",
                "Complex ARC reasoning: Analyze multiple examples, identify abstract patterns, and generate solution: ",
            ]
            
            for i, prompt in enumerate(test_prompts):
                complexity = ["simple", "medium", "complex"][i]
                logger.info(f"Benchmarking {complexity} prompt")
                
                start_time = time.time()
                
                try:
                    # Tokenize prompt
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate response
                    outputs = service.optimized_generate(
                        inputs["input_ids"],
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                    )
                    
                    inference_time = time.time() - start_time
                    
                    # Calculate metrics
                    input_length = inputs["input_ids"].shape[1]
                    total_length = outputs.shape[1]
                    generated_tokens = total_length - input_length
                    tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
                    
                    result = {
                        "adapter_type": "TTT_Service",
                        "model_name": model_name,
                        "task_id": f"service_{complexity}",
                        "task_complexity": complexity,
                        "inference_time_seconds": inference_time,
                        "inference_time_minutes": inference_time / 60,
                        "within_time_limit": inference_time <= self.time_limit_seconds,
                        "success": True,
                        "tokens_generated": generated_tokens,
                        "tokens_per_second": tokens_per_second,
                        "inference_profile": service.get_inference_profile(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    results.append(result)
                    self._log_benchmark_result(result)
                    
                except Exception as e:
                    inference_time = time.time() - start_time
                    
                    result = {
                        "adapter_type": "TTT_Service",
                        "model_name": model_name,
                        "task_id": f"service_{complexity}",
                        "task_complexity": complexity,
                        "inference_time_seconds": inference_time,
                        "inference_time_minutes": inference_time / 60,
                        "within_time_limit": inference_time <= self.time_limit_seconds,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    results.append(result)
                    logger.error(f"Service benchmark failed: {e}")
            
            # Cleanup
            service.cleanup()
            
        except Exception as e:
            logger.error(f"TTT Service benchmarking failed: {e}")
        
        return results
    
    def benchmark_optimization_poc(
        self,
        model_name: str = "meta-llama/Llama-3-8B"
    ) -> List[InferenceProfile]:
        """
        Benchmark using inference optimization POC.
        
        Args:
            model_name: Model identifier to benchmark
            
        Returns:
            List of inference profiles
        """
        logger.info("Starting Inference Optimization POC benchmarking")
        
        optimizer = InferenceOptimizer(time_limit_seconds=self.time_limit_seconds)
        return optimizer.run_comprehensive_benchmark(model_name)
    
    def run_comprehensive_benchmark(
        self,
        model_name: str = "meta-llama/Llama-3-8B",
        config_path: Optional[Path] = None,
        include_poc: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive inference benchmarking across all components.
        
        Args:
            model_name: Model identifier to benchmark
            config_path: Path to TTT configuration file
            include_poc: Whether to include POC benchmarking
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive inference benchmarking")
        logger.info(f"Model: {model_name}")
        logger.info(f"Time limit: {self.time_limit_seconds}s ({self.time_limit_seconds/60:.1f} minutes)")
        
        benchmark_start = time.time()
        results = {
            "benchmark_config": {
                "model_name": model_name,
                "time_limit_seconds": self.time_limit_seconds,
                "time_limit_minutes": self.time_limit_seconds / 60,
                "timestamp": datetime.now().isoformat(),
                "gpu_available": torch.cuda.is_available(),
            },
            "results": {}
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            results["benchmark_config"]["gpu_info"] = {
                "name": gpu_props.name,
                "total_memory_gb": gpu_props.total_memory / 1024**3,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            }
        
        # Benchmark TTT Adapter
        logger.info("=" * 80)
        logger.info("BENCHMARKING TTT ADAPTER")
        logger.info("=" * 80)
        
        try:
            ttt_adapter_results = self.benchmark_ttt_adapter(model_name, config_path)
            results["results"]["ttt_adapter"] = ttt_adapter_results
            self._analyze_component_results("TTT Adapter", ttt_adapter_results)
        except Exception as e:
            logger.error(f"TTT Adapter benchmarking failed: {e}")
            results["results"]["ttt_adapter"] = {"error": str(e)}
        
        # Benchmark TTT Service
        logger.info("=" * 80)
        logger.info("BENCHMARKING TTT SERVICE")
        logger.info("=" * 80)
        
        try:
            ttt_service_results = self.benchmark_ttt_service(model_name)
            results["results"]["ttt_service"] = ttt_service_results
            self._analyze_component_results("TTT Service", ttt_service_results)
        except Exception as e:
            logger.error(f"TTT Service benchmarking failed: {e}")
            results["results"]["ttt_service"] = {"error": str(e)}
        
        # Benchmark Inference Optimization POC
        if include_poc:
            logger.info("=" * 80)
            logger.info("BENCHMARKING INFERENCE OPTIMIZATION POC")
            logger.info("=" * 80)
            
            try:
                poc_results = self.benchmark_optimization_poc(model_name)
                results["results"]["optimization_poc"] = [
                    {
                        "optimization_name": p.optimization_name,
                        "task_prompt": p.task_prompt,
                        "inference_time_seconds": p.inference_time_seconds,
                        "inference_time_minutes": p.inference_time_seconds / 60,
                        "within_time_limit": p.meets_time_requirement,
                        "success": p.success,
                        "tokens_generated": p.tokens_generated,
                        "tokens_per_second": p.tokens_per_second,
                        "memory_usage_mb": p.memory_usage_mb,
                        "error_message": p.error_message,
                    }
                    for p in poc_results
                ]
                
                self._analyze_poc_results(poc_results)
            except Exception as e:
                logger.error(f"Optimization POC benchmarking failed: {e}")
                results["results"]["optimization_poc"] = {"error": str(e)}
        
        # Generate comprehensive analysis
        benchmark_time = time.time() - benchmark_start
        results["benchmark_summary"] = self._generate_comprehensive_analysis(results["results"])
        results["benchmark_summary"]["total_benchmark_time"] = benchmark_time
        
        return results
    
    def _log_benchmark_result(self, result: Dict[str, Any]) -> None:
        """Log individual benchmark result."""
        status = "✓ SUCCESS" if result.get("success", False) else "✗ FAILED"
        time_status = "✓ WITHIN LIMIT" if result.get("within_time_limit", False) else "⚠ EXCEEDED LIMIT"
        
        logger.info(f"Task: {result.get('task_id', 'unknown')}")
        logger.info(f"Status: {status}")
        logger.info(f"Time Status: {time_status}")
        logger.info(f"Inference Time: {result.get('inference_time_seconds', 0):.2f}s")
        
        if "tokens_per_second" in result:
            logger.info(f"Throughput: {result['tokens_per_second']:.1f} tok/s")
        
        if "error" in result:
            logger.info(f"Error: {result['error']}")
        
        logger.info("-" * 60)
    
    def _analyze_component_results(self, component_name: str, results: List[Dict[str, Any]]) -> None:
        """Analyze results for a specific component."""
        if not results:
            logger.warning(f"No results for {component_name}")
            return
        
        successful = [r for r in results if r.get("success", False)]
        within_limit = [r for r in successful if r.get("within_time_limit", False)]
        
        logger.info(f"\n{component_name} Analysis:")
        logger.info(f"  Total tasks: {len(results)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Within time limit: {len(within_limit)}")
        
        if within_limit:
            avg_time = sum(r["inference_time_seconds"] for r in within_limit) / len(within_limit)
            logger.info(f"  Average inference time: {avg_time:.2f}s")
            
            if any("tokens_per_second" in r for r in within_limit):
                throughput_results = [r for r in within_limit if "tokens_per_second" in r]
                avg_throughput = sum(r["tokens_per_second"] for r in throughput_results) / len(throughput_results)
                logger.info(f"  Average throughput: {avg_throughput:.1f} tok/s")
    
    def _analyze_poc_results(self, profiles: List[InferenceProfile]) -> None:
        """Analyze POC benchmark results."""
        successful = [p for p in profiles if p.success]
        within_limit = [p for p in successful if p.meets_time_requirement]
        
        logger.info(f"\nInference Optimization POC Analysis:")
        logger.info(f"  Total tests: {len(profiles)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Within time limit: {len(within_limit)}")
        
        if within_limit:
            best = min(within_limit, key=lambda p: p.inference_time_seconds)
            logger.info(f"  Best configuration: {best.optimization_name}")
            logger.info(f"  Best time: {best.inference_time_seconds:.2f}s")
            logger.info(f"  Best throughput: {best.tokens_per_second:.1f} tok/s")
    
    def _generate_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis of all benchmark results."""
        analysis = {
            "overall_success": True,
            "components_tested": list(results.keys()),
            "time_limit_compliance": {
                "requirement_met": True,
                "failing_components": [],
                "recommendations": [],
            },
            "performance_summary": {},
        }
        
        # Analyze each component
        for component, component_results in results.items():
            if isinstance(component_results, dict) and "error" in component_results:
                analysis["overall_success"] = False
                analysis["time_limit_compliance"]["failing_components"].append(component)
                continue
            
            if not component_results:
                continue
            
            # Handle different result formats
            if component == "optimization_poc":
                successful = [r for r in component_results if r.get("success", False)]
                within_limit = [r for r in successful if r.get("within_time_limit", False)]
            else:
                successful = [r for r in component_results if r.get("success", False)]
                within_limit = [r for r in successful if r.get("within_time_limit", False)]
            
            component_analysis = {
                "total_tests": len(component_results),
                "successful_tests": len(successful),
                "within_time_limit": len(within_limit),
                "success_rate": len(successful) / len(component_results) if component_results else 0,
                "time_compliance_rate": len(within_limit) / len(successful) if successful else 0,
            }
            
            if within_limit:
                avg_time = sum(r.get("inference_time_seconds", 0) for r in within_limit) / len(within_limit)
                component_analysis["avg_inference_time"] = avg_time
                
                throughput_results = [r for r in within_limit if "tokens_per_second" in r]
                if throughput_results:
                    avg_throughput = sum(r["tokens_per_second"] for r in throughput_results) / len(throughput_results)
                    component_analysis["avg_throughput"] = avg_throughput
            
            analysis["performance_summary"][component] = component_analysis
            
            # Check if component meets requirements
            if len(within_limit) == 0 and len(successful) > 0:
                analysis["time_limit_compliance"]["requirement_met"] = False
                analysis["time_limit_compliance"]["failing_components"].append(component)
        
        # Generate recommendations
        if not analysis["time_limit_compliance"]["requirement_met"]:
            analysis["time_limit_compliance"]["recommendations"].extend([
                "Apply more aggressive optimization configurations",
                "Consider model distillation or smaller model variants",
                "Implement progressive timeout with early stopping",
                "Use specialized inference frameworks (vLLM, TensorRT-LLM)",
                "Implement batched inference where possible",
            ])
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save benchmark results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive inference benchmarking")
    parser.add_argument(
        "--model", 
        default="meta-llama/Llama-3-8B",
        help="Model name to benchmark"
    )
    parser.add_argument(
        "--time-limit", 
        type=float, 
        default=7.2,
        help="Time limit in minutes"
    )
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to TTT configuration file"
    )
    parser.add_argument(
        "--output", 
        default="validation_results/inference_benchmark_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--skip-poc", 
        action="store_true",
        help="Skip POC benchmarking"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmark
    benchmark = ProductionInferenceBenchmark(time_limit_minutes=args.time_limit)
    
    logger.info("Starting comprehensive inference benchmarking...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Time limit: {args.time_limit} minutes")
    
    results = benchmark.run_comprehensive_benchmark(
        model_name=args.model,
        config_path=args.config,
        include_poc=not args.skip_poc
    )
    
    # Save results
    benchmark.save_results(results, args.output)
    
    # Print final summary
    print("\n" + "="*100)
    print("COMPREHENSIVE INFERENCE BENCHMARK SUMMARY")
    print("="*100)
    
    summary = results["benchmark_summary"]
    print(f"Overall Success: {'✓ YES' if summary['overall_success'] else '✗ NO'}")
    print(f"Time Limit Compliance: {'✓ MET' if summary['time_limit_compliance']['requirement_met'] else '⚠ FAILED'}")
    print(f"Components Tested: {', '.join(summary['components_tested'])}")
    
    if summary["time_limit_compliance"]["failing_components"]:
        print(f"Failing Components: {', '.join(summary['time_limit_compliance']['failing_components'])}")
    
    print(f"\nPerformance Summary:")
    for component, perf in summary["performance_summary"].items():
        print(f"  {component}:")
        print(f"    Success Rate: {perf['success_rate']:.1%}")
        print(f"    Time Compliance: {perf['time_compliance_rate']:.1%}")
        if "avg_inference_time" in perf:
            print(f"    Avg Time: {perf['avg_inference_time']:.2f}s")
        if "avg_throughput" in perf:
            print(f"    Avg Throughput: {perf['avg_throughput']:.1f} tok/s")
    
    print(f"\nResults saved to: {args.output}")
    print("="*100)
    
    # Exit with error code if requirements not met
    if not summary["time_limit_compliance"]["requirement_met"]:
        sys.exit(1)


if __name__ == "__main__":
    main()