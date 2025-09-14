#!/usr/bin/env python3
"""
8B Model Training Script for 53%+ Validation Accuracy - Task 3 Story 1.5

This script implements the training pipeline for the 8B model to achieve 53%+ accuracy
on the validation set using QLoRA, gradient checkpointing, and optimized hyperparameters.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse

import torch
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.domain.models import ARCTask
from src.domain.services.ttt_service import TTTModelService
from src.domain.services.training_orchestrator import TrainingOrchestrator, TrainingConfig
from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.infrastructure.config import get_config
from src.utils.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/8b_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from validation training experiment."""
    total_tasks: int
    tasks_above_53_percent: int
    average_accuracy: float
    individual_accuracies: List[float]
    task_ids: List[str]
    training_times: List[float]
    memory_usage: List[float]
    target_achieved: bool
    training_config: Dict[str, Any]
    timestamp: str


class ValidationTrainer:
    """Trainer for achieving 53%+ accuracy on validation set with 8B model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the validation trainer."""
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize services
        self.model_service = TTTModelService(self.config)
        
        # Create training configuration optimized for 53%+ accuracy
        self.training_config = TrainingConfig(
            learning_rate=5e-5,  # Optimized for 8B model
            num_epochs=3,
            batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_grad_norm=1.0,
            early_stopping_patience=5,
            early_stopping_threshold=0.01,
            validation_frequency=25,
            checkpoint_frequency=50,
            max_training_time=1800,  # 30 minutes per task
            target_accuracy=0.53,  # 53% validation target
            memory_limit_mb=24576,  # 24GB for 8B model
            mixed_precision=True,
            gradient_checkpointing=True,
            use_qlora=True,
            lora_rank=64,
            lora_alpha=32,
            lora_dropout=0.05,
            use_flash_attention=True,
            selective_checkpointing=True,
            checkpointing_layers=3
        )
        
        # Initialize training orchestrator
        self.orchestrator = TrainingOrchestrator(
            model_service=self.model_service,
            config=self.training_config
        )
        
        # Data repository
        self.data_repo = ARCDataRepository()
        
        # Results tracking
        self.results: List[Dict[str, Any]] = []
        
        logger.info("8B Model Validation Trainer initialized")
        logger.info(f"Target accuracy: {self.training_config.target_accuracy:.1%}")
        logger.info(f"Memory limit: {self.training_config.memory_limit_mb / 1024:.1f}GB")
    
    def validate_system_requirements(self) -> bool:
        """Validate system can handle 8B model training."""
        logger.info("Validating system requirements for 8B model...")
        
        # Check GPU availability and memory
        if not torch.cuda.is_available():
            logger.error("CUDA not available - 8B model requires GPU")
            return False
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        required_memory = 12  # Minimum for QLoRA 8B
        
        if gpu_memory_gb < required_memory:
            logger.error(f"Insufficient GPU memory: {gpu_memory_gb:.1f}GB < {required_memory}GB required")
            return False
        
        logger.info(f"GPU validation passed: {gpu_memory_gb:.1f}GB available")
        
        # Check model access
        model_name = self.config.get("model", {}).get("name", "meta-llama/Llama-3-8B")
        logger.info(f"Target model: {model_name}")
        
        return True
    
    def load_validation_dataset(self, max_tasks: int = 100) -> List[ARCTask]:
        """Load validation dataset for training."""
        logger.info(f"Loading validation dataset (max {max_tasks} tasks)...")
        
        try:
            # Load evaluation tasks for validation
            tasks = self.data_repo.load_evaluation_tasks()
            
            if len(tasks) > max_tasks:
                # Randomly sample tasks for validation
                np.random.seed(42)  # Reproducible sampling
                sampled_indices = np.random.choice(len(tasks), max_tasks, replace=False)
                tasks = [tasks[i] for i in sampled_indices]
            
            logger.info(f"Loaded {len(tasks)} validation tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load validation dataset: {e}")
            
            # Fallback: create synthetic validation tasks
            logger.warning("Creating synthetic validation tasks for testing...")
            return self._create_synthetic_tasks(min(max_tasks, 5))
    
    def _create_synthetic_tasks(self, num_tasks: int) -> List[ARCTask]:
        """Create synthetic ARC tasks for validation testing."""
        synthetic_tasks = []
        
        for i in range(num_tasks):
            # Simple pattern tasks for validation
            task = ARCTask(
                task_id=f"synthetic_validation_{i}",
                train_examples=[
                    {
                        "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                        "output": [[1, 2, 1], [2, 3, 2], [1, 2, 1]]
                    },
                    {
                        "input": [[1, 0, 1], [0, 2, 0], [1, 0, 1]],
                        "output": [[2, 1, 2], [1, 3, 1], [2, 1, 2]]
                    }
                ],
                test_examples=[
                    {
                        "input": [[0, 2, 0], [2, 1, 2], [0, 2, 0]],
                        "output": [[1, 3, 1], [3, 2, 3], [1, 3, 1]]
                    }
                ]
            )
            synthetic_tasks.append(task)
        
        logger.info(f"Created {len(synthetic_tasks)} synthetic validation tasks")
        return synthetic_tasks
    
    def train_and_validate_task(self, task: ARCTask) -> Dict[str, Any]:
        """Train and validate on a single task."""
        logger.info(f"Training 8B model on validation task: {task.task_id}")
        
        start_time = time.time()
        initial_memory = self.model_service._get_memory_usage()
        
        try:
            # Execute training with TTT
            training_results = self.orchestrator.train(task)
            
            # Extract metrics
            final_accuracy = training_results["final_accuracy"]
            training_time = training_results["training_time"]
            target_achieved = training_results["target_achieved"]
            
            result = {
                "task_id": task.task_id,
                "accuracy": final_accuracy,
                "target_achieved": target_achieved,
                "training_time": training_time,
                "memory_usage_gb": self.model_service._get_memory_usage(),
                "total_steps": training_results["total_steps"],
                "epochs_completed": training_results["epochs_completed"],
                "success": True,
                "error": None
            }
            
            logger.info(
                f"Task {task.task_id} completed: "
                f"Accuracy={final_accuracy:.2%} "
                f"(Target: {self.training_config.target_accuracy:.1%}) "
                f"Time={training_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed for task {task.task_id}: {e}")
            
            return {
                "task_id": task.task_id,
                "accuracy": 0.0,
                "target_achieved": False,
                "training_time": time.time() - start_time,
                "memory_usage_gb": self.model_service._get_memory_usage(),
                "total_steps": 0,
                "epochs_completed": 0,
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Clean up resources after each task
            self.orchestrator.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run_validation_experiment(self, max_tasks: int = 20) -> ValidationResults:
        """Run the complete validation experiment."""
        logger.info("=" * 60)
        logger.info("STARTING 8B MODEL VALIDATION EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Target: 53%+ accuracy on validation set")
        logger.info(f"Max tasks: {max_tasks}")
        logger.info(f"Model configuration: QLoRA 8B with Flash Attention")
        
        # Validate system requirements
        if not self.validate_system_requirements():
            raise RuntimeError("System requirements not met for 8B model training")
        
        # Load validation dataset
        tasks = self.load_validation_dataset(max_tasks)
        if not tasks:
            raise RuntimeError("No validation tasks loaded")
        
        # Run training on each task
        results = []
        
        with tqdm(tasks, desc="Training on validation tasks") as pbar:
            for task in pbar:
                result = self.train_and_validate_task(task)
                results.append(result)
                
                # Update progress bar
                current_avg_accuracy = np.mean([r["accuracy"] for r in results])
                tasks_above_53 = sum(1 for r in results if r["accuracy"] >= 0.53)
                
                pbar.set_postfix({
                    "Avg_Acc": f"{current_avg_accuracy:.2%}",
                    "Above_53%": f"{tasks_above_53}/{len(results)}"
                })
        
        # Calculate final results
        accuracies = [r["accuracy"] for r in results]
        task_ids = [r["task_id"] for r in results]
        training_times = [r["training_time"] for r in results]
        memory_usage = [r["memory_usage_gb"] for r in results]
        
        average_accuracy = np.mean(accuracies)
        tasks_above_53_percent = sum(1 for acc in accuracies if acc >= 0.53)
        target_achieved = average_accuracy >= 0.53
        
        validation_results = ValidationResults(
            total_tasks=len(results),
            tasks_above_53_percent=tasks_above_53_percent,
            average_accuracy=average_accuracy,
            individual_accuracies=accuracies,
            task_ids=task_ids,
            training_times=training_times,
            memory_usage=memory_usage,
            target_achieved=target_achieved,
            training_config=self.training_config.__dict__,
            timestamp=datetime.now().isoformat()
        )
        
        # Log results
        self._log_final_results(validation_results, results)
        
        # Save detailed results
        self._save_results(validation_results, results)
        
        return validation_results
    
    def _log_final_results(self, validation_results: ValidationResults, detailed_results: List[Dict[str, Any]]) -> None:
        """Log final validation results."""
        logger.info("=" * 60)
        logger.info("VALIDATION EXPERIMENT RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"Total tasks: {validation_results.total_tasks}")
        logger.info(f"Average accuracy: {validation_results.average_accuracy:.2%}")
        logger.info(f"Tasks above 53%: {validation_results.tasks_above_53_percent}/{validation_results.total_tasks}")
        logger.info(f"Target achieved: {'YES' if validation_results.target_achieved else 'NO'}")
        logger.info(f"Average training time: {np.mean(validation_results.training_times):.1f}s per task")
        logger.info(f"Average memory usage: {np.mean(validation_results.memory_usage):.1f}GB")
        
        # Success analysis
        successful_tasks = [r for r in detailed_results if r["success"]]
        failed_tasks = [r for r in detailed_results if not r["success"]]
        
        logger.info(f"Successful tasks: {len(successful_tasks)}/{len(detailed_results)}")
        if failed_tasks:
            logger.warning(f"Failed tasks: {[t['task_id'] for t in failed_tasks]}")
        
        # Performance breakdown
        if successful_tasks:
            successful_accuracies = [t["accuracy"] for t in successful_tasks]
            logger.info(f"Success rate accuracy stats:")
            logger.info(f"  Min: {min(successful_accuracies):.2%}")
            logger.info(f"  Max: {max(successful_accuracies):.2%}")
            logger.info(f"  Mean: {np.mean(successful_accuracies):.2%}")
            logger.info(f"  Std: {np.std(successful_accuracies):.2%}")
        
        # Final verdict
        logger.info("=" * 60)
        if validation_results.target_achieved:
            logger.info("✓ SUCCESS: 8B model achieved 53%+ validation accuracy")
            logger.info("  Recommendation: Deploy for competition use")
        else:
            logger.warning("⚠ PARTIAL SUCCESS: 8B model shows promise but needs optimization")
            logger.info("  Recommendation: Fine-tune hyperparameters and continue training")
        logger.info("=" * 60)
    
    def _save_results(self, validation_results: ValidationResults, detailed_results: List[Dict[str, Any]]) -> None:
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"validation_results/8b_training_results_{timestamp}.json")
        results_file.parent.mkdir(exist_ok=True)
        
        # Prepare data for JSON serialization
        output_data = {
            "experiment_info": {
                "model": "meta-llama/Llama-3-8B",
                "training_approach": "QLoRA with Test-Time Training",
                "target_accuracy": 0.53,
                "timestamp": validation_results.timestamp
            },
            "summary": {
                "total_tasks": validation_results.total_tasks,
                "tasks_above_53_percent": validation_results.tasks_above_53_percent,
                "average_accuracy": validation_results.average_accuracy,
                "target_achieved": validation_results.target_achieved,
                "success_rate": len([r for r in detailed_results if r["success"]]) / len(detailed_results)
            },
            "training_config": validation_results.training_config,
            "detailed_results": detailed_results,
            "statistics": {
                "accuracy_stats": {
                    "mean": float(np.mean(validation_results.individual_accuracies)),
                    "std": float(np.std(validation_results.individual_accuracies)),
                    "min": float(min(validation_results.individual_accuracies)),
                    "max": float(max(validation_results.individual_accuracies))
                },
                "time_stats": {
                    "mean_seconds": float(np.mean(validation_results.training_times)),
                    "total_hours": float(sum(validation_results.training_times) / 3600)
                },
                "memory_stats": {
                    "mean_gb": float(np.mean(validation_results.memory_usage)),
                    "max_gb": float(max(validation_results.memory_usage))
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to: {results_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="8B Model Validation Training for 53%+ Accuracy")
    parser.add_argument("--max-tasks", type=int, default=20, help="Maximum number of tasks to validate")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create trainer
        trainer = ValidationTrainer(args.config)
        
        # Run validation experiment
        results = trainer.run_validation_experiment(args.max_tasks)
        
        # Exit with appropriate code
        exit_code = 0 if results.target_achieved else 1
        logger.info(f"Experiment completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Validation experiment failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()