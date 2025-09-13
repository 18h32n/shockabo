"""
Performance validation runner for TTT baseline implementation.

Validates:
- 40% accuracy on validation set (negotiating to 25%)
- Training completes in under 2 hours
- Memory usage stays under 10GB
"""
import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask
from src.domain.services.ttt_service import TTTModelService
from src.domain.services.training_orchestrator import TrainingOrchestrator, TrainingConfig
from src.utils.performance_validator import PerformanceValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationRunner:
    """Runs comprehensive validation of TTT baseline implementation."""
    
    def __init__(
        self,
        data_path: str = "arc-prize-2025",
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: str = "auto",
        output_dir: str = "validation_results",
        use_real_dataset: bool = True,
    ):
        """Initialize validation runner."""
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_real_dataset = use_real_dataset
        
        # Initialize components
        self.data_repository = ARCDataRepository(str(self.data_path), use_real_dataset=use_real_dataset)
        self.performance_validator = PerformanceValidator()
        
        # Validation criteria
        self.accuracy_target = 0.25  # Negotiated down from 40% to 25%
        self.time_limit_hours = 2
        self.memory_limit_gb = 10
        
        # Results tracking
        self.results = {
            "validation_start": datetime.now().isoformat(),
            "model": model_name,
            "device": device,
            "accuracy_target": self.accuracy_target,
            "time_limit_hours": self.time_limit_hours,
            "memory_limit_gb": self.memory_limit_gb,
            "tasks_validated": 0,
            "tasks_correct": 0,
            "accuracy": 0.0,
            "total_time_seconds": 0,
            "max_memory_gb": 0,
            "individual_results": [],
            "validation_passed": False
        }
    
    def validate_task(self, task: ARCTask, adapter: TTTAdapter) -> Dict[str, Any]:
        """Validate a single task."""
        task_start = time.time()
        
        # Monitor memory before
        memory_before = adapter._estimate_memory_usage()
        
        try:
            # Solve task using TTT
            solution = adapter.solve(task)
            
            # Check if prediction matches expected output
            is_correct = False
            if task.test_output is not None:
                is_correct = self.performance_validator.validate_prediction(
                    solution.predictions[0],
                    task.test_output
                )
            
            # Monitor memory after
            memory_after = adapter._estimate_memory_usage()
            max_memory = max(memory_before, memory_after) / 1024  # Convert to GB
            
            # Calculate time
            task_time = time.time() - task_start
            
            result = {
                "task_id": task.task_id,
                "correct": is_correct,
                "time_seconds": task_time,
                "max_memory_gb": max_memory,
                "adaptation_metrics": solution.metadata.get("adaptation_metrics", {}),
                "resource_usage": {
                    "cpu_seconds": solution.resource_usage.cpu_seconds,
                    "memory_mb": solution.resource_usage.memory_mb,
                    "gpu_memory_mb": solution.resource_usage.gpu_memory_mb,
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating task {task.task_id}: {e}")
            return {
                "task_id": task.task_id,
                "correct": False,
                "time_seconds": time.time() - task_start,
                "max_memory_gb": memory_before / 1024,
                "error": str(e)
            }
    
    def run_validation(
        self,
        num_tasks: int = 100,
        subset: str = "validation"
    ) -> Dict[str, Any]:
        """
        Run full validation suite.
        
        Args:
            num_tasks: Number of tasks to validate
            subset: Data subset to use (validation/test)
        """
        logger.info(f"Starting validation on {num_tasks} tasks from {subset} set")
        
        # Load validation tasks
        logger.info("Loading validation tasks...")
        
        if self.use_real_dataset:
            # Use real ARC dataset
            if subset == "validation":
                # Use evaluation set for validation, or subset of training for testing
                available_sources = ["evaluation", "training"]
                selected_source = None
                for source in available_sources:
                    task_ids = self.data_repository.get_task_ids(source)
                    if len(task_ids) > 0:
                        selected_source = source
                        break
                
                if selected_source:
                    all_tasks = self.data_repository.load_all_tasks(selected_source, limit=num_tasks)
                    validation_tasks = list(all_tasks.values())
                else:
                    raise RuntimeError("No validation data available in real dataset")
            else:
                all_tasks = self.data_repository.load_all_tasks(subset, limit=num_tasks)
                validation_tasks = list(all_tasks.values())
        else:
            # Use legacy logic
            if subset == "validation":
                # Use official ARC validation set (first 100 tasks from training)
                all_tasks = self.data_repository.load_all_tasks("training", limit=400)
                validation_task_dict = dict(list(all_tasks.items())[300:400])  # Tasks 300-400 as validation
                validation_tasks = list(validation_task_dict.values())[:num_tasks]
            else:
                all_tasks = self.data_repository.load_all_tasks(subset, limit=num_tasks)
                validation_tasks = list(all_tasks.values())
        
        logger.info(f"Loaded {len(validation_tasks)} validation tasks")
        
        # Initialize TTT adapter
        logger.info("Initializing TTT adapter...")
        ttt_config = TTTConfig(
            model_name=self.model_name,
            device=self.device,
            max_examples=3,
            num_epochs=5,
            learning_rate=1e-4,
            checkpoint_dir=Path("data/models/ttt/validation"),
            cache_dir=Path("data/cache/ttt/validation"),
        )
        adapter = TTTAdapter(config=ttt_config)
        
        # Validation loop
        validation_start = time.time()
        correct_count = 0
        max_memory_used = 0
        
        with tqdm(validation_tasks, desc="Validating tasks") as pbar:
            for task in pbar:
                # Check time limit
                elapsed_hours = (time.time() - validation_start) / 3600
                if elapsed_hours > self.time_limit_hours:
                    logger.warning(f"Time limit exceeded ({elapsed_hours:.2f} hours)")
                    break
                
                # Validate task
                result = self.validate_task(task, adapter)
                
                # Update metrics
                if result["correct"]:
                    correct_count += 1
                
                max_memory_used = max(max_memory_used, result["max_memory_gb"])
                
                # Check memory limit
                if max_memory_used > self.memory_limit_gb:
                    logger.error(f"Memory limit exceeded ({max_memory_used:.2f} GB)")
                    result["memory_limit_exceeded"] = True
                
                # Store result
                self.results["individual_results"].append(result)
                
                # Update progress bar
                current_accuracy = correct_count / len(self.results["individual_results"])
                pbar.set_postfix({
                    "accuracy": f"{current_accuracy:.2%}",
                    "memory": f"{max_memory_used:.1f}GB"
                })
        
        # Cleanup
        adapter.cleanup()
        
        # Calculate final metrics
        total_time = time.time() - validation_start
        self.results["tasks_validated"] = len(self.results["individual_results"])
        self.results["tasks_correct"] = correct_count
        self.results["accuracy"] = correct_count / self.results["tasks_validated"]
        self.results["total_time_seconds"] = total_time
        self.results["total_time_hours"] = total_time / 3600
        self.results["max_memory_gb"] = max_memory_used
        self.results["validation_end"] = datetime.now().isoformat()
        
        # Check if validation passed
        self.results["validation_passed"] = (
            self.results["accuracy"] >= self.accuracy_target and
            self.results["total_time_hours"] <= self.time_limit_hours and
            self.results["max_memory_gb"] <= self.memory_limit_gb
        )
        
        # Log summary
        self._log_summary()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _log_summary(self):
        """Log validation summary."""
        logger.info("\n" + "="*50)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Tasks validated: {self.results['tasks_validated']}")
        logger.info(f"Tasks correct: {self.results['tasks_correct']}")
        logger.info(f"Accuracy: {self.results['accuracy']:.2%} (target: {self.accuracy_target:.0%})")
        logger.info(f"Total time: {self.results['total_time_hours']:.2f} hours (limit: {self.time_limit_hours} hours)")
        logger.info(f"Max memory: {self.results['max_memory_gb']:.2f} GB (limit: {self.memory_limit_gb} GB)")
        logger.info(f"Validation passed: {self.results['validation_passed']}")
        logger.info("="*50)
        
        # Detailed breakdown
        if not self.results["validation_passed"]:
            logger.warning("\nVALIDATION FAILED:")
            if self.results["accuracy"] < self.accuracy_target:
                logger.warning(f"- Accuracy {self.results['accuracy']:.2%} below target {self.accuracy_target:.0%}")
            if self.results["total_time_hours"] > self.time_limit_hours:
                logger.warning(f"- Time {self.results['total_time_hours']:.2f}h exceeds limit {self.time_limit_hours}h")
            if self.results["max_memory_gb"] > self.memory_limit_gb:
                logger.warning(f"- Memory {self.results['max_memory_gb']:.2f}GB exceeds limit {self.memory_limit_gb}GB")
    
    def _save_results(self):
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Also save a summary report
        report_file = self.output_dir / f"validation_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write("TTT BASELINE VALIDATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Validation Date: {self.results['validation_start']}\n\n")
            f.write(f"RESULTS:\n")
            f.write(f"- Accuracy: {self.results['accuracy']:.2%} (target: {self.accuracy_target:.0%})\n")
            f.write(f"- Time: {self.results['total_time_hours']:.2f} hours (limit: {self.time_limit_hours} hours)\n")
            f.write(f"- Memory: {self.results['max_memory_gb']:.2f} GB (limit: {self.memory_limit_gb} GB)\n")
            f.write(f"- Tasks: {self.results['tasks_correct']}/{self.results['tasks_validated']} correct\n\n")
            f.write(f"VALIDATION {'PASSED' if self.results['validation_passed'] else 'FAILED'}\n")
        
        logger.info(f"Report saved to: {report_file}")


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Run TTT baseline validation")
    parser.add_argument("--data-path", default="arc-prize-2025", help="Path to ARC data")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="Model to use")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to validate")
    parser.add_argument("--subset", default="validation", help="Data subset (validation/test/training)")
    parser.add_argument("--output-dir", default="validation_results", help="Output directory")
    parser.add_argument("--use-legacy-data", action="store_true", help="Use legacy data format instead of real dataset")
    
    args = parser.parse_args()
    
    # Create validation runner
    runner = ValidationRunner(
        data_path=args.data_path,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_real_dataset=not args.use_legacy_data
    )
    
    # Run validation
    results = runner.run_validation(
        num_tasks=args.num_tasks,
        subset=args.subset
    )
    
    # Exit with appropriate code
    exit(0 if results["validation_passed"] else 1)


if __name__ == "__main__":
    main()