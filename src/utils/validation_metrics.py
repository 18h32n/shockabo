"""
Advanced validation metrics and evaluation system for 8B model accuracy measurement.

This module provides comprehensive accuracy measurement tools specifically designed
for evaluating the 8B model's performance on ARC validation tasks.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.domain.models import ARCTask

logger = logging.getLogger(__name__)


@dataclass
class TaskValidationResult:
    """Detailed validation result for a single task."""
    task_id: str
    accuracy: float
    exact_matches: int
    total_examples: int
    prediction_success_rate: float  # How many predictions were parseable
    average_prediction_time: float
    memory_usage_mb: float
    individual_results: list[dict[str, Any]]
    reasoning_quality_score: float | None = None


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for model evaluation."""
    overall_accuracy: float
    tasks_above_threshold: dict[str, int]  # Different accuracy thresholds
    task_difficulty_analysis: dict[str, float]
    prediction_consistency: float
    parsing_success_rate: float
    average_inference_time: float
    memory_efficiency_score: float
    error_analysis: dict[str, int]


class ValidationAccuracyMeasurer:
    """
    Advanced accuracy measurement system for ARC validation tasks.

    Provides detailed analysis beyond simple accuracy including:
    - Grid-level exact matching
    - Shape and structure analysis
    - Reasoning quality assessment
    - Performance profiling
    """

    def __init__(self, tolerance_mode: str = "exact"):
        """
        Initialize the accuracy measurer.

        Args:
            tolerance_mode: "exact" for exact matching, "structural" for shape-aware matching
        """
        self.tolerance_mode = tolerance_mode
        self.validation_cache = {}

    def measure_task_accuracy(
        self,
        task: ARCTask,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        use_enhanced_prompting: bool = True
    ) -> TaskValidationResult:
        """
        Measure accuracy on a single ARC task with detailed analysis.

        Args:
            task: ARC task to evaluate
            model: Trained model
            tokenizer: Model tokenizer
            device: Compute device
            use_enhanced_prompting: Whether to use enhanced prompts

        Returns:
            Detailed validation results
        """
        logger.info(f"Measuring accuracy for task: {task.task_id}")

        time.time()
        initial_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        # Determine validation examples (prefer test examples)
        if hasattr(task, 'test_examples') and task.test_examples:
            validation_examples = task.test_examples
        else:
            # Use leave-one-out from training examples
            validation_examples = task.train_examples[-1:] if task.train_examples else []

        if not validation_examples:
            logger.warning(f"No validation examples for task {task.task_id}")
            return self._create_empty_result(task.task_id)

        individual_results = []
        correct_predictions = 0
        parseable_predictions = 0
        total_examples = len(validation_examples)

        model.eval()

        with torch.no_grad():
            for i, example in enumerate(validation_examples):
                example_start_time = time.time()

                try:
                    # Generate prediction
                    prediction_result = self._generate_prediction(
                        task, example, model, tokenizer, device, use_enhanced_prompting
                    )

                    # Evaluate prediction
                    evaluation_result = self._evaluate_prediction(
                        prediction_result["predicted_grid"],
                        example["output"],
                        prediction_result["raw_output"]
                    )

                    # Track results
                    if evaluation_result["parseable"]:
                        parseable_predictions += 1
                        if evaluation_result["exact_match"]:
                            correct_predictions += 1

                    # Detailed result tracking
                    individual_result = {
                        "example_index": i,
                        "prediction_time": time.time() - example_start_time,
                        "parseable": evaluation_result["parseable"],
                        "exact_match": evaluation_result["exact_match"],
                        "shape_match": evaluation_result["shape_match"],
                        "partial_match_score": evaluation_result["partial_match_score"],
                        "raw_prediction": prediction_result["raw_output"][:200],  # Truncated for storage
                        "error_type": evaluation_result.get("error_type", None)
                    }
                    individual_results.append(individual_result)

                except Exception as e:
                    logger.warning(f"Error processing example {i} of task {task.task_id}: {e}")
                    individual_results.append({
                        "example_index": i,
                        "prediction_time": time.time() - example_start_time,
                        "parseable": False,
                        "exact_match": False,
                        "shape_match": False,
                        "partial_match_score": 0.0,
                        "error_type": "processing_error",
                        "error_message": str(e)
                    })

        # Calculate metrics
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
        prediction_success_rate = parseable_predictions / total_examples if total_examples > 0 else 0.0
        average_prediction_time = np.mean([r.get("prediction_time", 0) for r in individual_results])
        final_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        result = TaskValidationResult(
            task_id=task.task_id,
            accuracy=accuracy,
            exact_matches=correct_predictions,
            total_examples=total_examples,
            prediction_success_rate=prediction_success_rate,
            average_prediction_time=average_prediction_time,
            memory_usage_mb=final_memory - initial_memory,
            individual_results=individual_results
        )

        logger.info(
            f"Task {task.task_id} accuracy: {accuracy:.2%} "
            f"({correct_predictions}/{total_examples}) "
            f"Parse rate: {prediction_success_rate:.2%}"
        )

        return result

    def _generate_prediction(
        self,
        task: ARCTask,
        example: dict[str, Any],
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        use_enhanced_prompting: bool
    ) -> dict[str, Any]:
        """Generate prediction for a single example."""
        from src.utils.grid_ops import grid_to_string, string_to_grid

        input_grid = example["input"]
        input_str = grid_to_string(input_grid)

        if use_enhanced_prompting:
            # Enhanced prompt with task context
            prompt = self._create_enhanced_prompt(task, input_str)
        else:
            # Basic prompt
            prompt = f"Task: Transform the input grid to output grid.\n\nInput:\n{input_str}\n\nOutput:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate with optimized parameters
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )

        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse grid
        try:
            predicted_grid = string_to_grid(generated_text.strip())
        except Exception as e:
            logger.debug(f"Failed to parse prediction: {e}")
            predicted_grid = None

        return {
            "predicted_grid": predicted_grid,
            "raw_output": generated_text,
            "prompt_length": inputs["input_ids"].shape[1]
        }

    def _create_enhanced_prompt(self, task: ARCTask, input_str: str) -> str:
        """Create enhanced prompt with task context."""
        # Include training examples for context
        context_examples = []
        for i, example in enumerate(task.train_examples[:2]):  # Use first 2 training examples
            from src.utils.grid_ops import grid_to_string

            example_input = grid_to_string(example["input"])
            example_output = grid_to_string(example["output"])

            context_examples.append(
                f"Training Example {i+1}:\n"
                f"Input:\n{example_input}\n"
                f"Output:\n{example_output}\n"
            )

        context_str = "\n".join(context_examples)

        prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task.
Analyze the pattern in the training examples and apply it to transform the test input.

{context_str}

Now apply the same transformation pattern to this test input:

Test Input:
{input_str}

Test Output:"""

        return prompt

    def _evaluate_prediction(
        self,
        predicted_grid: list[list[int]] | None,
        expected_grid: list[list[int]],
        raw_output: str
    ) -> dict[str, Any]:
        """Evaluate prediction quality."""
        result = {
            "parseable": predicted_grid is not None,
            "exact_match": False,
            "shape_match": False,
            "partial_match_score": 0.0
        }

        if predicted_grid is None:
            result["error_type"] = "unparseable"
            return result

        # Convert to numpy arrays for easier comparison
        try:
            pred_array = np.array(predicted_grid)
            expected_array = np.array(expected_grid)

            # Shape matching
            result["shape_match"] = pred_array.shape == expected_array.shape

            if result["shape_match"]:
                # Exact matching
                result["exact_match"] = np.array_equal(pred_array, expected_array)

                # Partial matching (percentage of correct cells)
                if pred_array.size > 0:
                    correct_cells = np.sum(pred_array == expected_array)
                    result["partial_match_score"] = correct_cells / pred_array.size
            else:
                result["error_type"] = "shape_mismatch"

        except Exception as e:
            result["error_type"] = "comparison_error"
            logger.debug(f"Error comparing grids: {e}")

        return result

    def _create_empty_result(self, task_id: str) -> TaskValidationResult:
        """Create empty result for tasks with no validation examples."""
        return TaskValidationResult(
            task_id=task_id,
            accuracy=0.0,
            exact_matches=0,
            total_examples=0,
            prediction_success_rate=0.0,
            average_prediction_time=0.0,
            memory_usage_mb=0.0,
            individual_results=[]
        )

    def compute_validation_metrics(
        self,
        task_results: list[TaskValidationResult],
        accuracy_thresholds: list[float] | None = None
    ) -> ValidationMetrics:
        """Compute comprehensive validation metrics from task results."""
        if not task_results:
            return self._create_empty_metrics()

        if accuracy_thresholds is None:
            accuracy_thresholds = [0.1, 0.25, 0.4, 0.5, 0.53, 0.6, 0.7, 0.8, 0.9]

        # Overall accuracy
        total_correct = sum(r.exact_matches for r in task_results)
        total_examples = sum(r.total_examples for r in task_results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        # Tasks above different thresholds
        tasks_above_threshold = {}
        for threshold in accuracy_thresholds:
            tasks_above_threshold[f"{threshold:.0%}"] = sum(
                1 for r in task_results if r.accuracy >= threshold
            )

        # Task difficulty analysis (based on training examples vs accuracy)
        task_difficulty = {}
        for result in task_results:
            # Simple heuristic: tasks with lower accuracy are considered harder
            if result.accuracy >= 0.8:
                difficulty = "easy"
            elif result.accuracy >= 0.5:
                difficulty = "medium"
            elif result.accuracy >= 0.2:
                difficulty = "hard"
            else:
                difficulty = "very_hard"

            task_difficulty[difficulty] = task_difficulty.get(difficulty, 0) + 1

        # Prediction consistency (std dev of accuracies)
        accuracies = [r.accuracy for r in task_results]
        prediction_consistency = 1.0 - (np.std(accuracies) if len(accuracies) > 1 else 0.0)

        # Parsing success rate
        total_parsing_attempts = sum(r.total_examples for r in task_results)
        successful_parses = sum(
            sum(1 for individual in r.individual_results if individual.get("parseable", False))
            for r in task_results
        )
        parsing_success_rate = successful_parses / total_parsing_attempts if total_parsing_attempts > 0 else 0.0

        # Performance metrics
        average_inference_time = np.mean([r.average_prediction_time for r in task_results if r.average_prediction_time > 0])
        memory_usage = np.mean([r.memory_usage_mb for r in task_results if r.memory_usage_mb > 0])
        memory_efficiency_score = max(0.0, 1.0 - (memory_usage / 1000))  # Normalized to 1GB baseline

        # Error analysis
        error_types = {}
        for result in task_results:
            for individual in result.individual_results:
                error_type = individual.get("error_type")
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1

        return ValidationMetrics(
            overall_accuracy=overall_accuracy,
            tasks_above_threshold=tasks_above_threshold,
            task_difficulty_analysis=task_difficulty,
            prediction_consistency=prediction_consistency,
            parsing_success_rate=parsing_success_rate,
            average_inference_time=average_inference_time,
            memory_efficiency_score=memory_efficiency_score,
            error_analysis=error_types
        )

    def _create_empty_metrics(self) -> ValidationMetrics:
        """Create empty validation metrics."""
        return ValidationMetrics(
            overall_accuracy=0.0,
            tasks_above_threshold={},
            task_difficulty_analysis={},
            prediction_consistency=0.0,
            parsing_success_rate=0.0,
            average_inference_time=0.0,
            memory_efficiency_score=0.0,
            error_analysis={}
        )

    def generate_validation_report(
        self,
        validation_metrics: ValidationMetrics,
        task_results: list[TaskValidationResult],
        save_path: Path | None = None
    ) -> str:
        """Generate comprehensive validation report."""
        report_lines = [
            "=" * 80,
            "8B MODEL VALIDATION ACCURACY REPORT",
            "=" * 80,
            "",
            f"Overall Accuracy: {validation_metrics.overall_accuracy:.2%}",
            f"Tasks Evaluated: {len(task_results)}",
            f"Parsing Success Rate: {validation_metrics.parsing_success_rate:.2%}",
            f"Prediction Consistency: {validation_metrics.prediction_consistency:.2%}",
            "",
            "ACCURACY THRESHOLDS:",
        ]

        for threshold, count in validation_metrics.tasks_above_threshold.items():
            percentage = count / len(task_results) * 100 if task_results else 0
            report_lines.append(f"  {threshold}: {count}/{len(task_results)} tasks ({percentage:.1f}%)")

        report_lines.extend([
            "",
            "TASK DIFFICULTY ANALYSIS:",
        ])

        for difficulty, count in validation_metrics.task_difficulty_analysis.items():
            percentage = count / len(task_results) * 100 if task_results else 0
            report_lines.append(f"  {difficulty.title()}: {count} tasks ({percentage:.1f}%)")

        report_lines.extend([
            "",
            "PERFORMANCE METRICS:",
            f"  Average Inference Time: {validation_metrics.average_inference_time:.2f}s",
            f"  Memory Efficiency Score: {validation_metrics.memory_efficiency_score:.2%}",
            "",
            "ERROR ANALYSIS:",
        ])

        for error_type, count in validation_metrics.error_analysis.items():
            report_lines.append(f"  {error_type}: {count} occurrences")

        report_lines.extend([
            "",
            "TOP PERFORMING TASKS:",
        ])

        # Sort tasks by accuracy
        sorted_tasks = sorted(task_results, key=lambda x: x.accuracy, reverse=True)
        for i, task in enumerate(sorted_tasks[:5]):
            report_lines.append(
                f"  {i+1}. {task.task_id}: {task.accuracy:.2%} "
                f"({task.exact_matches}/{task.total_examples})"
            )

        report_lines.extend([
            "",
            "VALIDATION TARGET ASSESSMENT:",
        ])

        validation_metrics.tasks_above_threshold.get("53%", 0)
        if validation_metrics.overall_accuracy >= 0.53:
            report_lines.append("✓ TARGET ACHIEVED: 53%+ overall accuracy reached")
        else:
            report_lines.append("✗ TARGET NOT ACHIEVED: Below 53% overall accuracy")
            report_lines.append(f"  Current: {validation_metrics.overall_accuracy:.2%}")
            report_lines.append(f"  Gap: {0.53 - validation_metrics.overall_accuracy:.2%}")

        report_lines.extend([
            "",
            "=" * 80
        ])

        report_text = "\n".join(report_lines)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to: {save_path}")

        return report_text
