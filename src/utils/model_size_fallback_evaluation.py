"""
7B Model Fallback Evaluation

This module evaluates 7B models as fallback options if 8B models prove infeasible,
comparing performance, memory usage, and inference times to determine viability.
"""
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .comprehensive_error_handling import (
    resilient_operation,
)
from .inference_optimization_poc import ARCTaskSample
from .memory_profiling_poc import ModelMemoryProfile, QLoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Comparison between different model sizes."""
    model_name: str
    parameter_count: int
    memory_profile: ModelMemoryProfile
    inference_times: dict[str, float]  # task_id -> inference_time
    accuracy_estimate: float | None = None
    feasibility_score: float = 0.0
    recommended_config: dict[str, Any] | None = None


@dataclass
class FallbackEvaluation:
    """Complete fallback evaluation results."""
    target_8b_feasible: bool
    best_7b_option: ModelComparison | None
    all_comparisons: list[ModelComparison]
    recommendations: list[str]
    risk_mitigation: dict[str, str]
    timestamp: float


class ModelSizeFallbackEvaluator:
    """Evaluates model size fallback options."""

    def __init__(
        self,
        memory_limit_mb: float = 24576,  # 24GB
        time_limit_seconds: float = 432,  # 7.2 minutes
        target_accuracy: float = 0.53  # 53%
    ):
        """
        Initialize fallback evaluator.

        Args:
            memory_limit_mb: GPU memory limit
            time_limit_seconds: Inference time limit
            target_accuracy: Target accuracy threshold
        """
        self.memory_limit_mb = memory_limit_mb
        self.time_limit_seconds = time_limit_seconds
        self.target_accuracy = target_accuracy
        self.evaluations: list[ModelComparison] = []

    def get_candidate_models(self) -> list[dict[str, Any]]:
        """Get candidate models for evaluation."""
        return [
            # 8B Models (primary targets)
            {
                "name": "meta-llama/Meta-Llama-3-8B",
                "size_category": "8B",
                "expected_params": 8_000_000_000,
                "priority": 1,
                "description": "Primary 8B target model"
            },
            {
                "name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "size_category": "8B",
                "expected_params": 8_000_000_000,
                "priority": 2,
                "description": "Instruction-tuned 8B model"
            },
            # 7B Models (fallback options)
            {
                "name": "meta-llama/Llama-2-7b-hf",
                "size_category": "7B",
                "expected_params": 7_000_000_000,
                "priority": 3,
                "description": "Llama-2 7B fallback option"
            },
            {
                "name": "meta-llama/Llama-2-7b-chat-hf",
                "size_category": "7B",
                "expected_params": 7_000_000_000,
                "priority": 4,
                "description": "Llama-2 7B chat model"
            },
            {
                "name": "mistralai/Mistral-7B-v0.1",
                "size_category": "7B",
                "expected_params": 7_241_000_000,
                "priority": 5,
                "description": "Mistral 7B base model"
            },
            {
                "name": "mistralai/Mistral-7B-Instruct-v0.2",
                "size_category": "7B",
                "expected_params": 7_241_000_000,
                "priority": 6,
                "description": "Mistral 7B instruct model"
            },
            # Smaller models for testing/comparison
            {
                "name": "microsoft/DialoGPT-large",
                "size_category": "770M",
                "expected_params": 770_000_000,
                "priority": 10,
                "description": "Small model for testing framework"
            },
        ]

    def get_test_configurations(self) -> list[dict[str, Any]]:
        """Get different QLoRA configurations to test."""
        return [
            {
                "name": "conservative",
                "description": "Conservative memory usage",
                "qlora_config": QLoRAConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="bfloat16",
                    lora_r=32,
                    lora_alpha=64,
                    lora_dropout=0.1,
                ),
                "torch_compile": False,
                "flash_attention": True,
            },
            {
                "name": "balanced",
                "description": "Balanced performance/memory",
                "qlora_config": QLoRAConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="bfloat16",
                    lora_r=64,
                    lora_alpha=128,
                    lora_dropout=0.1,
                ),
                "torch_compile": True,
                "flash_attention": True,
            },
            {
                "name": "aggressive",
                "description": "Maximum performance",
                "qlora_config": QLoRAConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="bfloat16",
                    lora_r=128,
                    lora_alpha=256,
                    lora_dropout=0.05,
                ),
                "torch_compile": True,
                "flash_attention": True,
            },
        ]

    def get_arc_test_tasks(self) -> list[ARCTaskSample]:
        """Get ARC tasks for evaluation."""
        return [
            ARCTaskSample(
                task_id="pattern_completion",
                prompt="""Complete the pattern in this grid:
Input: [[1,0,1],[0,1,0],[1,0,?]]
What should replace the ? to complete the pattern?
Think step by step and provide your answer.""",
                expected_output_length=100,
                complexity_level="simple"
            ),
            ARCTaskSample(
                task_id="transformation_rule",
                prompt="""Learn the transformation rule from these examples:
Example 1: [1,2,3] -> [3,2,1]
Example 2: [4,5,6] -> [6,5,4]
Example 3: [7,8,9] -> [9,8,7]

Apply the rule to: [2,4,6] -> ?
Explain your reasoning and provide the answer.""",
                expected_output_length=150,
                complexity_level="medium"
            ),
            ARCTaskSample(
                task_id="complex_reasoning",
                prompt="""Analyze this complex ARC pattern:
Training:
Input: [[0,1,0],[1,2,1],[0,1,0]] -> Output: [[2,0,2],[0,1,0],[2,0,2]]
Input: [[1,0,1],[0,3,0],[1,0,1]] -> Output: [[3,1,3],[1,0,1],[3,1,3]]

Test: [[0,2,0],[2,1,2],[0,2,0]] -> ?

Identify the transformation rule and apply it to the test case.
Provide detailed reasoning and the final answer.""",
                expected_output_length=300,
                complexity_level="complex"
            ),
        ]

    @resilient_operation(max_attempts=2, handle_oom=True)
    def evaluate_model(
        self,
        model_info: dict[str, Any],
        config: dict[str, Any]
    ) -> ModelComparison | None:
        """Evaluate a single model with given configuration."""
        model_name = model_info["name"]
        logger.info(f"Evaluating model: {model_name} with config: {config['name']}")

        try:
            # Create memory profile
            memory_profile = self._profile_memory_usage(model_name, config)
            if not memory_profile.success:
                logger.warning(f"Memory profiling failed for {model_name}")
                return None

            # Check memory feasibility
            if memory_profile.memory_utilization > 1.0:
                logger.warning(f"Model {model_name} exceeds memory limit: {memory_profile.memory_utilization:.1%}")
                # Still continue to get baseline measurements

            # Benchmark inference times
            inference_times = self._benchmark_inference_times(model_name, config)

            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                memory_profile, inference_times, model_info["size_category"]
            )

            # Estimate accuracy potential
            accuracy_estimate = self._estimate_accuracy_potential(
                model_info["size_category"], model_name
            )

            comparison = ModelComparison(
                model_name=model_name,
                parameter_count=model_info["expected_params"],
                memory_profile=memory_profile,
                inference_times=inference_times,
                accuracy_estimate=accuracy_estimate,
                feasibility_score=feasibility_score,
                recommended_config=config if feasibility_score > 0.7 else None
            )

            self.evaluations.append(comparison)
            self._log_evaluation_result(comparison)

            return comparison

        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            return None

    def _profile_memory_usage(
        self,
        model_name: str,
        config: dict[str, Any]
    ) -> ModelMemoryProfile:
        """Profile memory usage for model loading."""
        from .memory_profiling_poc import MemoryProfiler

        profiler = MemoryProfiler(self.memory_limit_mb)

        # Create quantization config
        qlora_config = config["qlora_config"]

        try:
            profile = profiler.profile_model_loading(
                model_name=model_name,
                qlora_config=qlora_config,
                test_inference=False  # Skip inference test here
            )
            return profile

        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            # Return failed profile
            return ModelMemoryProfile(
                model_name=model_name,
                quantization_config=asdict(qlora_config),
                memory_before_mb=0,
                memory_after_mb=0,
                memory_delta_mb=0,
                peak_memory_mb=0,
                gpu_memory_allocated_mb=0,
                gpu_memory_reserved_mb=0,
                total_gpu_memory_mb=0,
                memory_utilization=0,
                loading_time_seconds=0,
                success=False,
                error_message=str(e)
            )

    def _benchmark_inference_times(
        self,
        model_name: str,
        config: dict[str, Any]
    ) -> dict[str, float]:
        """Benchmark inference times for different task complexities."""
        inference_times = {}
        tasks = self.get_arc_test_tasks()

        try:
            # Load model with config
            model, tokenizer = self._load_model_with_config(model_name, config)

            for task in tasks:
                try:
                    # Benchmark inference
                    start_time = time.time()

                    inputs = tokenizer(
                        task.prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )

                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        model.generate(
                            **inputs,
                            max_new_tokens=task.expected_output_length,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                        )

                    inference_time = time.time() - start_time
                    inference_times[task.task_id] = inference_time

                    logger.debug(f"Task {task.task_id}: {inference_time:.2f}s")

                except Exception as e:
                    logger.warning(f"Inference failed for task {task.task_id}: {e}")
                    inference_times[task.task_id] = float('inf')

            # Cleanup
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to benchmark inference for {model_name}: {e}")
            # Return failed times
            for task in tasks:
                inference_times[task.task_id] = float('inf')

        return inference_times

    def _load_model_with_config(
        self,
        model_name: str,
        config: dict[str, Any]
    ) -> tuple[nn.Module, Any]:
        """Load model with specific configuration."""
        # Create quantization config
        qlora_config = config["qlora_config"]
        compute_dtype = getattr(torch, qlora_config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

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
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        # Add flash attention if specified
        if config.get("flash_attention") and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Apply torch.compile if specified
        if config.get("torch_compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        return model, tokenizer

    def _calculate_feasibility_score(
        self,
        memory_profile: ModelMemoryProfile,
        inference_times: dict[str, float],
        size_category: str
    ) -> float:
        """Calculate overall feasibility score (0-1)."""
        score = 0.0

        # Memory score (40% weight)
        if memory_profile.success and memory_profile.memory_utilization <= 1.0:
            memory_score = max(0, 1.0 - memory_profile.memory_utilization)
            score += 0.4 * memory_score

        # Inference time score (40% weight)
        valid_times = [t for t in inference_times.values() if t != float('inf')]
        if valid_times:
            avg_inference_time = sum(valid_times) / len(valid_times)
            if avg_inference_time <= self.time_limit_seconds:
                time_score = max(0, 1.0 - (avg_inference_time / self.time_limit_seconds))
                score += 0.4 * time_score

        # Model size preference (20% weight) - prefer larger models if feasible
        size_scores = {
            "8B": 1.0,
            "7B": 0.8,
            "770M": 0.3,
        }
        score += 0.2 * size_scores.get(size_category, 0.5)

        return min(1.0, score)

    def _estimate_accuracy_potential(self, size_category: str, model_name: str) -> float:
        """Estimate potential accuracy based on model characteristics."""
        # Base accuracy estimates by model size (rough estimates)
        base_accuracy = {
            "8B": 0.55,   # Likely to meet 53% target
            "7B": 0.50,   # Might meet target with good training
            "770M": 0.35, # Unlikely to meet target
        }

        # Adjustments for specific models
        adjustments = {
            "meta-llama": 0.02,  # Llama models tend to perform well
            "mistral": 0.01,     # Mistral is efficient
            "instruct": 0.01,    # Instruction-tuned models
            "chat": 0.01,        # Chat models
        }

        estimated_accuracy = base_accuracy.get(size_category, 0.4)

        # Apply model-specific adjustments
        model_name_lower = model_name.lower()
        for keyword, adjustment in adjustments.items():
            if keyword in model_name_lower:
                estimated_accuracy += adjustment

        return min(1.0, estimated_accuracy)

    def _log_evaluation_result(self, comparison: ModelComparison) -> None:
        """Log evaluation result summary."""
        logger.info("=" * 60)
        logger.info(f"Model: {comparison.model_name}")
        logger.info(f"Parameters: {comparison.parameter_count:,}")
        logger.info(f"Memory Usage: {comparison.memory_profile.memory_utilization:.1%}")
        logger.info(f"Memory Success: {comparison.memory_profile.success}")
        logger.info(f"Feasibility Score: {comparison.feasibility_score:.2f}")
        logger.info(f"Estimated Accuracy: {comparison.accuracy_estimate:.1%}")

        avg_inference_time = sum(
            t for t in comparison.inference_times.values() if t != float('inf')
        ) / max(1, len([t for t in comparison.inference_times.values() if t != float('inf')]))

        logger.info(f"Avg Inference Time: {avg_inference_time:.2f}s")
        logger.info(f"Meets Time Limit: {'YES' if avg_inference_time <= self.time_limit_seconds else 'NO'}")
        logger.info("=" * 60)

    def run_comprehensive_evaluation(self) -> FallbackEvaluation:
        """Run comprehensive fallback evaluation."""
        logger.info("Starting comprehensive model size fallback evaluation...")

        candidates = self.get_candidate_models()
        configurations = self.get_test_configurations()

        all_comparisons = []

        # Test each model with each configuration
        for model_info in candidates:
            for config in configurations:
                logger.info(f"Testing {model_info['name']} with {config['name']} config")

                comparison = self.evaluate_model(model_info, config)
                if comparison:
                    all_comparisons.append(comparison)

                # Small delay between tests
                time.sleep(1)

        # Analyze results
        analysis = self._analyze_evaluation_results(all_comparisons)

        return FallbackEvaluation(
            target_8b_feasible=analysis["target_8b_feasible"],
            best_7b_option=analysis["best_7b_option"],
            all_comparisons=all_comparisons,
            recommendations=analysis["recommendations"],
            risk_mitigation=analysis["risk_mitigation"],
            timestamp=time.time()
        )

    def _analyze_evaluation_results(self, comparisons: list[ModelComparison]) -> dict[str, Any]:
        """Analyze evaluation results and generate recommendations."""
        if not comparisons:
            return {
                "target_8b_feasible": False,
                "best_7b_option": None,
                "recommendations": ["No models could be evaluated"],
                "risk_mitigation": {"critical": "All model loading failed"}
            }

        # Separate by model size
        eight_b_models = [c for c in comparisons if "8B" in c.model_name]
        seven_b_models = [c for c in comparisons if "7B" in c.model_name or "7b" in c.model_name]

        # Check 8B feasibility
        feasible_8b = [m for m in eight_b_models if m.feasibility_score > 0.7]
        target_8b_feasible = len(feasible_8b) > 0

        # Find best 7B option
        best_7b_option = None
        if seven_b_models:
            best_7b_option = max(seven_b_models, key=lambda m: m.feasibility_score)

        # Generate recommendations
        recommendations = self._generate_fallback_recommendations(
            target_8b_feasible, feasible_8b, best_7b_option, seven_b_models
        )

        # Risk mitigation strategies
        risk_mitigation = self._generate_risk_mitigation_strategies(
            target_8b_feasible, best_7b_option
        )

        return {
            "target_8b_feasible": target_8b_feasible,
            "best_7b_option": best_7b_option,
            "recommendations": recommendations,
            "risk_mitigation": risk_mitigation
        }

    def _generate_fallback_recommendations(
        self,
        target_8b_feasible: bool,
        feasible_8b: list[ModelComparison],
        best_7b_option: ModelComparison | None,
        seven_b_models: list[ModelComparison]
    ) -> list[str]:
        """Generate fallback recommendations."""
        recommendations = []

        if target_8b_feasible:
            best_8b = max(feasible_8b, key=lambda m: m.feasibility_score)
            recommendations.extend([
                f"SUCCESS: 8B model feasible - use {best_8b.model_name}",
                f"Best 8B feasibility score: {best_8b.feasibility_score:.2f}",
                f"Expected accuracy: {best_8b.accuracy_estimate:.1%}",
                "Proceed with 8B implementation as planned",
            ])
        else:
            recommendations.extend([
                "CRITICAL: No 8B models meet feasibility requirements",
                "Must implement 7B fallback strategy",
            ])

        if best_7b_option:
            recommendations.extend([
                f"Best 7B fallback: {best_7b_option.model_name}",
                f"7B feasibility score: {best_7b_option.feasibility_score:.2f}",
                f"7B expected accuracy: {best_7b_option.accuracy_estimate:.1%}",
            ])

            if best_7b_option.accuracy_estimate >= self.target_accuracy:
                recommendations.append("7B model likely to meet 53% accuracy target")
            else:
                recommendations.append("7B model may need enhanced training to reach 53% target")
        else:
            recommendations.extend([
                "WARNING: No viable 7B fallback options identified",
                "Consider smaller models or cloud deployment",
            ])

        # General recommendations
        recommendations.extend([
            "Implement adaptive model loading based on available resources",
            "Set up continuous monitoring of model performance",
            "Create automatic fallback chains for production deployment",
        ])

        return recommendations

    def _generate_risk_mitigation_strategies(
        self,
        target_8b_feasible: bool,
        best_7b_option: ModelComparison | None
    ) -> dict[str, str]:
        """Generate risk mitigation strategies."""
        strategies = {}

        if not target_8b_feasible:
            strategies["perf_001"] = "CRITICAL - Use 7B model as primary strategy"
            strategies["perf_002"] = "CRITICAL - Implement aggressive inference optimization"
        else:
            strategies["perf_001"] = "MITIGATED - 8B model loading feasible"
            strategies["perf_002"] = "MONITOR - Continue inference optimization"

        if best_7b_option and best_7b_option.feasibility_score > 0.7:
            strategies["fallback"] = "AVAILABLE - 7B fallback option viable"
        else:
            strategies["fallback"] = "CRITICAL - No viable fallback identified"

        strategies["implementation"] = "Use progressive model loading with automatic fallback"
        strategies["monitoring"] = "Implement real-time resource monitoring"

        return strategies

    def generate_report(self, output_path: str | None = None) -> dict[str, Any]:
        """Generate comprehensive fallback evaluation report."""
        if not self.evaluations:
            return {"error": "No evaluations completed"}

        # Run final analysis
        analysis = self._analyze_evaluation_results(self.evaluations)

        report = {
            "summary": {
                "total_models_tested": len(self.evaluations),
                "target_8b_feasible": analysis["target_8b_feasible"],
                "viable_7b_fallback": analysis["best_7b_option"] is not None,
                "memory_limit_mb": self.memory_limit_mb,
                "time_limit_seconds": self.time_limit_seconds,
                "target_accuracy": self.target_accuracy,
            },
            "best_7b_fallback": asdict(analysis["best_7b_option"]) if analysis["best_7b_option"] else None,
            "recommendations": analysis["recommendations"],
            "risk_mitigation": analysis["risk_mitigation"],
            "all_evaluations": [asdict(comp) for comp in self.evaluations],
            "timestamp": time.time(),
        }

        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Fallback evaluation report saved to: {output_path}")

        return report


def main():
    """Run model size fallback evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize evaluator
    evaluator = ModelSizeFallbackEvaluator()

    # Run evaluation (using test model for demonstration)
    logger.info("Starting model size fallback evaluation...")

    # Test with a single small model for demonstration
    test_model = {
        "name": "microsoft/DialoGPT-large",
        "size_category": "770M",
        "expected_params": 770_000_000,
        "priority": 1,
        "description": "Test model for framework validation"
    }

    test_config = {
        "name": "test",
        "description": "Test configuration",
        "qlora_config": QLoRAConfig(
            load_in_4bit=False,  # Use FP16 for smaller model
            lora_r=32,
            lora_alpha=64,
        ),
        "torch_compile": False,
        "flash_attention": False,
    }

    # Evaluate test model
    evaluator.evaluate_model(test_model, test_config)

    # Generate report
    report_path = "docs/qa/assessments/model_size_fallback_evaluation_results.json"
    report = evaluator.generate_report(report_path)

    # Print summary
    print("\n" + "="*80)
    print("MODEL SIZE FALLBACK EVALUATION RESULTS")
    print("="*80)
    print(f"Models Tested: {report['summary']['total_models_tested']}")
    print(f"8B Target Feasible: {'YES' if report['summary']['target_8b_feasible'] else 'NO'}")
    print(f"7B Fallback Available: {'YES' if report['summary']['viable_7b_fallback'] else 'NO'}")

    if report['best_7b_fallback']:
        best = report['best_7b_fallback']
        print("\nBest 7B Fallback:")
        print(f"  Model: {best['model_name']}")
        print(f"  Feasibility Score: {best['feasibility_score']:.2f}")
        print(f"  Expected Accuracy: {best['accuracy_estimate']:.1%}")

    print("\nKey Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")

    print(f"\nFull report saved to: {report_path}")
    print("="*80)

    return report


if __name__ == "__main__":
    main()
