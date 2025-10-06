"""
Integration tests for Enhanced TTT Strategy - Accuracy Validation

Tests the accuracy of the enhanced TTT strategy using leave-one-out generation,
self-consistency validation, and LoRA optimization. Target: 58%+ standalone accuracy
on 100 evaluation tasks.
"""
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.domain.models import ARCTask
from src.utils.ttt_leave_one_out import LeaveOneOutConfig, LeaveOneOutGenerator
from src.utils.ttt_lora_optimizer import LoRAOptimizerConfig
from src.utils.ttt_methodology import TTTTrainer, TTTTrainingConfig, MIT_TTTStrategy
from src.utils.ttt_self_consistency import SelfConsistencyConfig, SelfConsistencyValidator

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for a single task validation."""
    
    task_id: str
    difficulty: str
    correct: bool
    confidence: float
    inference_time: float
    adaptation_time: float
    error_message: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report for TTT accuracy testing."""
    
    total_tasks: int = 0
    correct_tasks: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_inference_time: float = 0.0
    avg_adaptation_time: float = 0.0
    
    # Difficulty breakdown
    difficulty_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Individual task results
    task_results: list[TaskMetrics] = field(default_factory=list)
    
    # Timing breakdown
    total_time: float = 0.0
    
    def add_task_result(self, metrics: TaskMetrics):
        """Add task result and update aggregates."""
        self.task_results.append(metrics)
        self.total_tasks += 1
        
        if metrics.correct:
            self.correct_tasks += 1
        
        # Update difficulty metrics
        if metrics.difficulty not in self.difficulty_metrics:
            self.difficulty_metrics[metrics.difficulty] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "avg_time": 0.0
            }
        
        diff_metrics = self.difficulty_metrics[metrics.difficulty]
        diff_metrics["total"] += 1
        if metrics.correct:
            diff_metrics["correct"] += 1
        diff_metrics["accuracy"] = diff_metrics["correct"] / diff_metrics["total"]
        
    def finalize(self):
        """Calculate final aggregate metrics."""
        if self.total_tasks == 0:
            return
        
        self.accuracy = self.correct_tasks / self.total_tasks
        self.avg_confidence = np.mean([m.confidence for m in self.task_results])
        self.avg_inference_time = np.mean([m.inference_time for m in self.task_results])
        self.avg_adaptation_time = np.mean([m.adaptation_time for m in self.task_results])
        self.total_time = sum(m.inference_time + m.adaptation_time for m in self.task_results)
        
        # Finalize difficulty metrics
        for diff, metrics in self.difficulty_metrics.items():
            diff_tasks = [m for m in self.task_results if m.difficulty == diff]
            metrics["avg_confidence"] = np.mean([m.confidence for m in diff_tasks])
            metrics["avg_time"] = np.mean([m.inference_time + m.adaptation_time for m in diff_tasks])
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "correct_tasks": self.correct_tasks,
                "accuracy": self.accuracy,
                "avg_confidence": self.avg_confidence,
                "avg_inference_time_sec": self.avg_inference_time,
                "avg_adaptation_time_sec": self.avg_adaptation_time,
                "total_time_sec": self.total_time
            },
            "difficulty_breakdown": self.difficulty_metrics,
            "task_results": [
                {
                    "task_id": m.task_id,
                    "difficulty": m.difficulty,
                    "correct": m.correct,
                    "confidence": m.confidence,
                    "inference_time": m.inference_time,
                    "adaptation_time": m.adaptation_time,
                    "error": m.error_message
                }
                for m in self.task_results
            ]
        }


class EnhancedTTTValidator:
    """Validator for enhanced TTT strategy accuracy testing."""
    
    def __init__(
        self,
        evaluation_data_path: Path,
        solutions_data_path: Path,
        config: TTTTrainingConfig | None = None
    ):
        """
        Initialize validator with evaluation dataset.
        
        Args:
            evaluation_data_path: Path to arc-agi_evaluation_challenges.json
            solutions_data_path: Path to arc-agi_evaluation_solutions.json
            config: Optional TTT configuration (uses defaults if None)
        """
        self.evaluation_data_path = evaluation_data_path
        self.solutions_data_path = solutions_data_path
        
        # Load evaluation data
        with open(evaluation_data_path) as f:
            self.evaluation_challenges = json.load(f)
        
        with open(solutions_data_path) as f:
            self.evaluation_solutions = json.load(f)
        
        # Initialize TTT strategy
        self.config = config or self._create_default_config()
        self.strategy = MIT_TTTStrategy(self.config)
        
        # Initialize model (this will take time on first run)
        logger.info("Initializing TTT strategy with model loading...")
        try:
            self.strategy.trainer.initialize_model()
            logger.info("TTT strategy initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTT strategy: {e}")
            raise
        
        logger.info(f"Loaded {len(self.evaluation_challenges)} evaluation tasks")
    
    def _create_default_config(self) -> TTTTrainingConfig:
        """Create default configuration for enhanced TTT."""
        return TTTTrainingConfig(
            model_name="meta-llama/Llama-3.2-1B",
            device="auto",
            quantization=True,
            mixed_precision=True,
            
            # LoRA config
            lora_rank=64,
            lora_alpha=16,
            lora_dropout=0.1,
            
            # Training config
            learning_rate=5e-5,
            num_epochs=2,
            batch_size=1,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            
            # Per-instance config
            per_instance_epochs=1,
            per_instance_lr=1e-4,
            max_training_time=300.0,
            
            # Self-consistency config
            permute_n=3,
            temperature=0.0,
            consensus_threshold=0.6,
            use_self_consistency=True,
            
            # Memory config
            max_sequence_length=2048,
            gradient_checkpointing=True,
            memory_limit_mb=24576
        )
    
    def _estimate_difficulty(self, task: ARCTask) -> str:
        """
        Estimate task difficulty based on heuristics.
        
        Args:
            task: ARCTask to estimate
            
        Returns:
            Difficulty level: 'easy', 'medium', 'hard'
        """
        # Heuristic: Based on grid size and number of training examples
        num_train = len(task.train_examples)
        grid_size = len(task.test_input) * len(task.test_input[0]) if task.test_input else 0
        
        # Simple heuristic (can be improved)
        if num_train >= 4 and grid_size <= 100:
            return "easy"
        elif num_train >= 3 and grid_size <= 300:
            return "medium"
        else:
            return "hard"
    
    def _grids_match(self, pred: list[list[int]], truth: list[list[int]]) -> bool:
        """
        Check if predicted grid matches ground truth.
        
        Args:
            pred: Predicted grid
            truth: Ground truth grid
            
        Returns:
            True if grids match exactly
        """
        if len(pred) != len(truth):
            return False
        
        for i, row in enumerate(pred):
            if i >= len(truth):
                return False
            if len(row) != len(truth[i]):
                return False
            for j, val in enumerate(row):
                if j >= len(truth[i]):
                    return False
                if val != truth[i][j]:
                    return False
        
        return True
    
    def validate_task(self, task_id: str, task_data: dict[str, Any]) -> TaskMetrics:
        """
        Validate a single task using enhanced TTT.
        
        Args:
            task_id: Task identifier
            task_data: Task challenge data
            
        Returns:
            TaskMetrics with validation results
        """
        start_time = time.time()
        
        try:
            # Create ARCTask
            task = ARCTask.from_dict(task_data, task_id, task_source="evaluation")
            difficulty = self._estimate_difficulty(task)
            
            # Get ground truth
            ground_truth = self.evaluation_solutions[task_id][0]  # First test output
            
            # Run enhanced TTT (with leave-one-out, self-consistency, LoRA optimization)
            logger.info(f"Validating task {task_id} (difficulty: {difficulty})")
            
            # Use actual MIT TTT strategy implementation
            try:
                start_inference = time.time()
                prediction, metadata = self.strategy.solve_task(task, use_self_consistency=True)
                inference_time = time.time() - start_inference
                
                # Extract metrics from metadata
                confidence = metadata.get("confidence", 0.0)
                adaptation_result = metadata.get("adaptation_result")
                adaptation_time = adaptation_result.adaptation_time if adaptation_result else 0.0
                
                # Handle case where strategy returns fallback
                if not metadata.get("success", True):
                    logger.warning(f"TTT strategy failed for task {task_id}: {metadata.get('error', 'Unknown error')}")
                    confidence = 0.0
                    
            except Exception as e:
                logger.error(f"Error running TTT strategy on task {task_id}: {e}")
                # Return fallback values if strategy fails
                prediction = [[0]]
                confidence = 0.0
                adaptation_time = 0.0
                inference_time = 0.0
            
            # Check correctness
            correct = self._grids_match(prediction, ground_truth)
            
            return TaskMetrics(
                task_id=task_id,
                difficulty=difficulty,
                correct=correct,
                confidence=confidence,
                inference_time=inference_time,
                adaptation_time=adaptation_time
            )
            
        except Exception as e:
            logger.error(f"Error validating task {task_id}: {e}")
            return TaskMetrics(
                task_id=task_id,
                difficulty="unknown",
                correct=False,
                confidence=0.0,
                inference_time=time.time() - start_time,
                adaptation_time=0.0,
                error_message=str(e)
            )
    
    def validate_dataset(
        self,
        max_tasks: int = 100,
        save_report: bool = True,
        report_path: Path | None = None
    ) -> ValidationReport:
        """
        Validate enhanced TTT on evaluation dataset.
        
        Args:
            max_tasks: Maximum number of tasks to validate (default: 100)
            save_report: Whether to save report to disk
            report_path: Path to save report (default: validation_results/enhanced_ttt_report.json)
            
        Returns:
            ValidationReport with complete results
        """
        report = ValidationReport()
        
        # Select first N tasks
        task_ids = list(self.evaluation_challenges.keys())[:max_tasks]
        
        logger.info(f"Starting validation on {len(task_ids)} tasks")
        
        for i, task_id in enumerate(task_ids, 1):
            logger.info(f"Validating task {i}/{len(task_ids)}: {task_id}")
            
            task_data = self.evaluation_challenges[task_id]
            metrics = self.validate_task(task_id, task_data)
            report.add_task_result(metrics)
            
            # Log progress
            if i % 10 == 0:
                interim_accuracy = report.correct_tasks / report.total_tasks
                logger.info(f"Progress: {i}/{len(task_ids)} - Accuracy: {interim_accuracy:.2%}")
        
        # Finalize report
        report.finalize()
        
        # Save report
        if save_report:
            if report_path is None:
                report_path = Path("validation_results/enhanced_ttt_report.json")
            
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.info(f"Report saved to {report_path}")
        
        return report
    
    def cleanup(self) -> None:
        """Clean up strategy resources."""
        if hasattr(self, 'strategy'):
            try:
                self.strategy.cleanup()
                logger.info("TTT strategy cleanup completed")
            except Exception as e:
                logger.warning(f"Error during strategy cleanup: {e}")


@pytest.fixture
def evaluation_data_paths():
    """Fixture providing paths to evaluation datasets."""
    base_path = Path("arc-prize-2025/data/downloaded")
    return {
        "challenges": base_path / "arc-agi_evaluation_challenges.json",
        "solutions": base_path / "arc-agi_evaluation_solutions.json"
    }


@pytest.fixture
def ttt_config():
    """Fixture providing TTT configuration for testing."""
    return TTTTrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        device="cpu",  # Use CPU for testing
        quantization=False,  # Disable for faster testing
        mixed_precision=False,
        
        # Minimal config for testing
        num_epochs=1,
        per_instance_epochs=1,
        batch_size=1,
        
        # Enable enhanced features
        permute_n=2,
        use_self_consistency=True,
        consensus_threshold=0.6
    )


def test_evaluation_dataset_exists(evaluation_data_paths):
    """Test that evaluation dataset files exist."""
    assert evaluation_data_paths["challenges"].exists(), \
        "arc-agi_evaluation_challenges.json not found"
    assert evaluation_data_paths["solutions"].exists(), \
        "arc-agi_evaluation_solutions.json not found"


def test_load_evaluation_dataset(evaluation_data_paths):
    """Test loading and parsing evaluation dataset."""
    with open(evaluation_data_paths["challenges"]) as f:
        challenges = json.load(f)
    
    with open(evaluation_data_paths["solutions"]) as f:
        solutions = json.load(f)
    
    assert len(challenges) > 0, "No challenges loaded"
    assert len(solutions) > 0, "No solutions loaded"
    assert len(challenges) == len(solutions), "Mismatch between challenges and solutions"


def test_validator_initialization(evaluation_data_paths, ttt_config):
    """Test EnhancedTTTValidator initialization."""
    validator = EnhancedTTTValidator(
        evaluation_data_paths["challenges"],
        evaluation_data_paths["solutions"],
        ttt_config
    )
    
    assert validator.evaluation_challenges is not None
    assert validator.evaluation_solutions is not None
    assert len(validator.evaluation_challenges) > 0


@pytest.mark.skip(reason="Requires model weights and GPU - run manually for full validation")
def test_single_task_validation(evaluation_data_paths, ttt_config):
    """Test validation on a single task."""
    validator = EnhancedTTTValidator(
        evaluation_data_paths["challenges"],
        evaluation_data_paths["solutions"],
        ttt_config
    )
    
    # Get first task
    task_id = list(validator.evaluation_challenges.keys())[0]
    task_data = validator.evaluation_challenges[task_id]
    
    # Validate task
    metrics = validator.validate_task(task_id, task_data)
    
    assert metrics.task_id == task_id
    assert metrics.difficulty in ["easy", "medium", "hard", "unknown"]
    assert 0.0 <= metrics.confidence <= 1.0
    assert metrics.inference_time >= 0.0
    assert metrics.adaptation_time >= 0.0


@pytest.mark.skip(reason="Requires model weights and GPU - run manually for full validation")
def test_baseline_ttt_accuracy(evaluation_data_paths):
    """
    Test baseline TTT accuracy (without enhancements).
    
    Expected: 53-55% accuracy on evaluation set.
    """
    # Create baseline config (no leave-one-out, no self-consistency)
    baseline_config = TTTTrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        use_self_consistency=False,
        permute_n=1
    )
    
    validator = EnhancedTTTValidator(
        evaluation_data_paths["challenges"],
        evaluation_data_paths["solutions"],
        baseline_config
    )
    
    # Validate on 100 tasks
    report = validator.validate_dataset(
        max_tasks=100,
        save_report=True,
        report_path=Path("validation_results/baseline_ttt_report.json")
    )
    
    # Check results
    logger.info(f"Baseline TTT Accuracy: {report.accuracy:.2%}")
    logger.info(f"Average Confidence: {report.avg_confidence:.3f}")
    logger.info(f"Average Time: {report.avg_inference_time + report.avg_adaptation_time:.1f}s")
    
    # Expected baseline: 53-55%
    assert 0.50 <= report.accuracy <= 0.60, \
        f"Baseline accuracy {report.accuracy:.2%} outside expected range (50-60%)"


def test_enhanced_ttt_accuracy(evaluation_data_paths):
    """
    Test enhanced TTT accuracy (with all optimizations).
    
    Target: 58%+ accuracy on evaluation set.
    """
    # Create enhanced config (leave-one-out, self-consistency, LoRA optimization)
    enhanced_config = TTTTrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        use_self_consistency=True,
        permute_n=5,
        consensus_threshold=0.6,
        
        # LoRA optimization
        lora_rank=64,
        lora_alpha=32,
        num_epochs=2,
        per_instance_epochs=1
    )
    
    validator = EnhancedTTTValidator(
        evaluation_data_paths["challenges"],
        evaluation_data_paths["solutions"],
        enhanced_config
    )
    
    # Validate on 100 tasks
    report = validator.validate_dataset(
        max_tasks=100,
        save_report=True,
        report_path=Path("validation_results/enhanced_ttt_report.json")
    )
    
    # Check results
    logger.info(f"Enhanced TTT Accuracy: {report.accuracy:.2%}")
    logger.info(f"Average Confidence: {report.avg_confidence:.3f}")
    logger.info(f"Average Time: {report.avg_inference_time + report.avg_adaptation_time:.1f}s")
    
    # Log difficulty breakdown
    for difficulty, metrics in report.difficulty_metrics.items():
        logger.info(f"{difficulty.capitalize()} tasks: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Target: 58%+ accuracy
    assert report.accuracy >= 0.58, \
        f"Enhanced TTT accuracy {report.accuracy:.2%} below target (58%)"
    
    # Check inference time < 5 minutes per task
    avg_total_time = report.avg_inference_time + report.avg_adaptation_time
    assert avg_total_time < 300.0, \
        f"Average time {avg_total_time:.1f}s exceeds 5 minute target"


@pytest.mark.skip(reason="Requires model weights and GPU - run manually for comparison")
def test_accuracy_improvement(evaluation_data_paths):
    """
    Test that enhanced TTT improves over baseline.
    
    Expected improvement: +3-5% accuracy.
    """
    # Load baseline report
    baseline_report_path = Path("validation_results/baseline_ttt_report.json")
    with open(baseline_report_path) as f:
        baseline_data = json.load(f)
    baseline_accuracy = baseline_data["summary"]["accuracy"]
    
    # Load enhanced report
    enhanced_report_path = Path("validation_results/enhanced_ttt_report.json")
    with open(enhanced_report_path) as f:
        enhanced_data = json.load(f)
    enhanced_accuracy = enhanced_data["summary"]["accuracy"]
    
    # Calculate improvement
    improvement = enhanced_accuracy - baseline_accuracy
    improvement_pct = improvement * 100
    
    logger.info(f"Baseline: {baseline_accuracy:.2%}")
    logger.info(f"Enhanced: {enhanced_accuracy:.2%}")
    logger.info(f"Improvement: +{improvement_pct:.1f}%")
    
    # Expected improvement: +3-5%
    assert improvement >= 0.03, \
        f"Accuracy improvement {improvement_pct:.1f}% below expected (+3%)"
