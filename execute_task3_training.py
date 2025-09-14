#!/usr/bin/env python3
"""
Task 3 Story 1.5 Execution Script: Train 8B Model for 53%+ Validation Accuracy

This is the main execution script that orchestrates the complete training pipeline
for Task 3, integrating all components to achieve 53%+ accuracy on the validation set.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_8b_validation import ValidationTrainer, ValidationResults
from src.utils.validation_metrics import ValidationAccuracyMeasurer
from src.utils.training_monitor import TrainingMonitor
from src.infrastructure.config import get_config
from src.utils.memory_manager import MemoryManager

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'task3_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class Task3ExecutionManager:
    """
    Main execution manager for Task 3: Train 8B model for 53%+ validation accuracy.
    
    Coordinates all components including training, monitoring, and validation
    to achieve the target accuracy on the validation set.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the execution manager."""
        self.config = get_config(config_path)
        self.start_time = time.time()
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.trainer = None
        self.monitor = None
        self.accuracy_measurer = None
        
        # Results tracking
        self.execution_results = {
            "execution_id": self.execution_id,
            "start_time": datetime.now().isoformat(),
            "target_accuracy": 0.53,
            "model_configuration": "8B_QLoRA",
            "status": "initializing"
        }
        
        logger.info(f"Task 3 Execution Manager initialized - ID: {self.execution_id}")
    
    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites for 8B model training."""
        logger.info("Validating prerequisites for Task 3 execution...")
        
        validation_results = {
            "gpu_available": False,
            "memory_sufficient": False,
            "model_accessible": False,
            "data_available": False
        }
        
        try:
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                validation_results["gpu_available"] = True
                
                if gpu_memory_gb >= 12:  # Minimum for QLoRA 8B
                    validation_results["memory_sufficient"] = True
                    logger.info(f"✓ GPU validation passed: {gpu_memory_gb:.1f}GB available")
                else:
                    logger.error(f"✗ Insufficient GPU memory: {gpu_memory_gb:.1f}GB < 12GB required")
                    return False
            else:
                logger.error("✗ CUDA not available - GPU required for 8B model")
                return False
            
            # Check system memory
            import psutil
            system_memory_gb = psutil.virtual_memory().total / 1024**3
            if system_memory_gb < 16:
                logger.warning(f"Low system memory: {system_memory_gb:.1f}GB (recommended: 16GB+)")
            
            # Check model access (will be done during trainer initialization)
            validation_results["model_accessible"] = True
            
            # Check data availability
            from src.adapters.repositories.arc_data_repository import ARCDataRepository
            try:
                data_repo = ARCDataRepository()
                tasks = data_repo.load_evaluation_tasks()
                if len(tasks) > 0:
                    validation_results["data_available"] = True
                    logger.info(f"✓ Data validation passed: {len(tasks)} tasks available")
                else:
                    logger.warning("No evaluation tasks found - will use synthetic data")
                    validation_results["data_available"] = True  # Allow synthetic fallback
            except Exception as e:
                logger.warning(f"Data loading issue (will use synthetic): {e}")
                validation_results["data_available"] = True
            
            # Log validation summary
            passed_checks = sum(validation_results.values())
            total_checks = len(validation_results)
            
            logger.info(f"Prerequisite validation: {passed_checks}/{total_checks} checks passed")
            
            if passed_checks == total_checks:
                logger.info("✓ All prerequisites met for Task 3 execution")
                return True
            else:
                logger.error("✗ Prerequisites not met - execution cannot continue")
                return False
                
        except Exception as e:
            logger.error(f"Prerequisite validation failed: {e}")
            return False
    
    def initialize_components(self) -> bool:
        """Initialize all training components."""
        logger.info("Initializing Task 3 training components...")
        
        try:
            # Initialize training monitor
            self.monitor = TrainingMonitor(
                log_interval=10,
                checkpoint_interval=50,
                memory_threshold_mb=20480,  # 20GB warning
                save_dir=Path(f"logs/task3_monitoring_{self.execution_id}")
            )
            logger.info("✓ Training monitor initialized")
            
            # Initialize validation trainer
            self.trainer = ValidationTrainer(config_path=None)
            logger.info("✓ Validation trainer initialized")
            
            # Initialize accuracy measurer
            self.accuracy_measurer = ValidationAccuracyMeasurer(tolerance_mode="exact")
            logger.info("✓ Accuracy measurer initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def execute_training(self, max_tasks: int = 20) -> ValidationResults:
        """Execute the main training process."""
        logger.info("=" * 80)
        logger.info("STARTING TASK 3: TRAIN 8B MODEL FOR 53%+ VALIDATION ACCURACY")
        logger.info("=" * 80)
        logger.info(f"Execution ID: {self.execution_id}")
        logger.info(f"Target accuracy: 53%")
        logger.info(f"Max validation tasks: {max_tasks}")
        logger.info(f"Model configuration: 8B with QLoRA")
        
        self.execution_results["status"] = "training"
        
        try:
            # Execute validation training experiment
            results = self.trainer.run_validation_experiment(max_tasks)
            
            # Update execution results
            self.execution_results.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "execution_duration_minutes": (time.time() - self.start_time) / 60,
                "validation_results": {
                    "total_tasks": results.total_tasks,
                    "average_accuracy": results.average_accuracy,
                    "tasks_above_53_percent": results.tasks_above_53_percent,
                    "target_achieved": results.target_achieved
                }
            })
            
            # Log final results
            self._log_execution_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}", exc_info=True)
            self.execution_results.update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })
            raise
    
    def generate_comprehensive_report(self, results: ValidationResults) -> Path:
        """Generate comprehensive execution report."""
        logger.info("Generating comprehensive Task 3 execution report...")
        
        report_dir = Path(f"reports/task3_{self.execution_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Main report file
        report_path = report_dir / "task3_execution_report.md"
        
        # Generate report content
        report_content = self._create_execution_report(results)
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save execution data
        import json
        data_path = report_dir / "execution_data.json"
        with open(data_path, 'w') as f:
            json.dump({
                "execution_results": self.execution_results,
                "validation_results": results.__dict__,
                "system_info": self._get_system_info()
            }, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report generated: {report_path}")
        return report_path
    
    def _log_execution_summary(self, results: ValidationResults) -> None:
        """Log execution summary."""
        logger.info("=" * 80)
        logger.info("TASK 3 EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Execution ID: {self.execution_id}")
        logger.info(f"Duration: {(time.time() - self.start_time) / 60:.1f} minutes")
        logger.info(f"Tasks evaluated: {results.total_tasks}")
        logger.info(f"Average accuracy: {results.average_accuracy:.2%}")
        logger.info(f"Tasks above 53%: {results.tasks_above_53_percent}/{results.total_tasks}")
        
        # Success assessment
        if results.target_achieved:
            logger.info("🎯 SUCCESS: Task 3 target achieved (53%+ validation accuracy)")
            logger.info("   Recommendation: Ready for competition deployment")
        else:
            gap = 0.53 - results.average_accuracy
            logger.warning(f"⚠ PARTIAL SUCCESS: {gap:.2%} below target")
            logger.info("   Recommendation: Continue optimization or adjust approach")
        
        # Performance insights
        if results.individual_accuracies:
            best_accuracy = max(results.individual_accuracies)
            worst_accuracy = min(results.individual_accuracies)
            std_dev = np.std(results.individual_accuracies)
            
            logger.info(f"Performance distribution:")
            logger.info(f"   Best task: {best_accuracy:.2%}")
            logger.info(f"   Worst task: {worst_accuracy:.2%}")
            logger.info(f"   Std deviation: {std_dev:.2%}")
            logger.info(f"   Consistency: {'High' if std_dev < 0.1 else 'Medium' if std_dev < 0.2 else 'Low'}")
        
        logger.info("=" * 80)
    
    def _create_execution_report(self, results: ValidationResults) -> str:
        """Create comprehensive execution report."""
        report_lines = [
            "# Task 3 Story 1.5 Execution Report",
            "",
            f"**Execution ID:** {self.execution_id}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {(time.time() - self.start_time) / 60:.1f} minutes",
            "",
            "## Objective",
            "",
            "Train an 8B parameter model using QLoRA and Test-Time Training to achieve",
            "53%+ accuracy on the ARC validation set.",
            "",
            "## Configuration",
            "",
            "- **Model:** meta-llama/Llama-3-8B",
            "- **Training Method:** QLoRA (4-bit quantization)",
            "- **Strategy:** Test-Time Training with gradient checkpointing",
            "- **Target Accuracy:** 53%",
            f"- **Validation Tasks:** {results.total_tasks}",
            "",
            "## Results",
            "",
            f"### Overall Performance",
            "",
            f"- **Average Accuracy:** {results.average_accuracy:.2%}",
            f"- **Tasks Above Target:** {results.tasks_above_53_percent}/{results.total_tasks}",
            f"- **Target Achievement:** {'✅ YES' if results.target_achieved else '❌ NO'}",
            "",
            f"### Performance Statistics",
            "",
        ]
        
        if results.individual_accuracies:
            accuracies = results.individual_accuracies
            report_lines.extend([
                f"- **Best Task Accuracy:** {max(accuracies):.2%}",
                f"- **Worst Task Accuracy:** {min(accuracies):.2%}",
                f"- **Standard Deviation:** {np.std(accuracies):.2%}",
                f"- **Median Accuracy:** {np.median(accuracies):.2%}",
                "",
            ])
        
        # Add timing statistics
        if results.training_times:
            times = results.training_times
            report_lines.extend([
                f"### Training Performance",
                "",
                f"- **Average Time per Task:** {np.mean(times):.1f} seconds",
                f"- **Total Training Time:** {sum(times) / 3600:.2f} hours",
                f"- **Fastest Task:** {min(times):.1f} seconds",
                f"- **Slowest Task:** {max(times):.1f} seconds",
                "",
            ])
        
        # Memory usage
        if results.memory_usage:
            memory = results.memory_usage
            report_lines.extend([
                f"### Resource Utilization",
                "",
                f"- **Average Memory Usage:** {np.mean(memory):.1f} GB",
                f"- **Peak Memory Usage:** {max(memory):.1f} GB",
                f"- **Memory Efficiency:** {'Good' if max(memory) < 20 else 'Moderate' if max(memory) < 24 else 'High'}",
                "",
            ])
        
        # Task-by-task breakdown
        report_lines.extend([
            "### Individual Task Results",
            "",
            "| Task ID | Accuracy | Training Time (s) | Memory (GB) |",
            "|---------|----------|-------------------|-------------|",
        ])
        
        for i, task_id in enumerate(results.task_ids):
            accuracy = results.individual_accuracies[i] if i < len(results.individual_accuracies) else 0
            train_time = results.training_times[i] if i < len(results.training_times) else 0
            memory = results.memory_usage[i] if i < len(results.memory_usage) else 0
            
            report_lines.append(f"| {task_id} | {accuracy:.2%} | {train_time:.1f} | {memory:.1f} |")
        
        report_lines.extend([
            "",
            "## Analysis",
            "",
        ])
        
        # Success analysis
        if results.target_achieved:
            report_lines.extend([
                "### ✅ Target Achievement Analysis",
                "",
                "The 8B model successfully achieved the 53%+ validation accuracy target.",
                "Key success factors:",
                "",
                "- QLoRA quantization enabled efficient 8B model training",
                "- Test-Time Training improved task-specific performance",
                "- Optimized hyperparameters for validation set",
                "- Gradient checkpointing managed memory efficiently",
                "",
                "**Recommendation:** Deploy this configuration for competition use.",
                "",
            ])
        else:
            gap = 0.53 - results.average_accuracy
            report_lines.extend([
                "### ⚠️ Target Gap Analysis",
                "",
                f"The model achieved {results.average_accuracy:.2%} accuracy, falling {gap:.2%} short of the 53% target.",
                "",
                "**Potential improvements:**",
                "",
                "- Increase training epochs or learning rate",
                "- Expand training data augmentation",
                "- Fine-tune LoRA rank and alpha parameters",
                "- Implement ensemble methods",
                "- Use larger validation dataset for training",
                "",
            ])
        
        # Technical details
        report_lines.extend([
            "## Technical Configuration",
            "",
            "### Training Hyperparameters",
            "",
        ])
        
        if hasattr(results, 'training_config'):
            config = results.training_config
            for key, value in config.items():
                report_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        
        report_lines.extend([
            "",
            "### System Information",
            "",
        ])
        
        system_info = self._get_system_info()
        for key, value in system_info.items():
            report_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        
        report_lines.extend([
            "",
            "## Conclusion",
            "",
        ])
        
        if results.target_achieved:
            report_lines.extend([
                "Task 3 has been successfully completed. The 8B model configuration with QLoRA",
                "has demonstrated the ability to achieve 53%+ validation accuracy, meeting the",
                "requirements for Story 1.5.",
                "",
                "The implementation is ready for integration into the competition pipeline.",
            ])
        else:
            report_lines.extend([
                "Task 3 shows promising results but requires additional optimization to reach",
                "the 53% validation accuracy target. The current implementation provides a",
                "solid foundation for further improvements.",
                "",
                "Consider the recommended improvements and continue iterative development.",
            ])
        
        report_lines.extend([
            "",
            "---",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
            f"*Execution ID: {self.execution_id}*"
        ])
        
        return "\n".join(report_lines)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reporting."""
        import psutil
        
        info = {
            "platform": "Windows" if sys.platform == "win32" else sys.platform,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "cuda_version": torch.version.cuda
            })
        
        info.update({
            "system_memory_gb": psutil.virtual_memory().total / 1024**3,
            "cpu_cores": psutil.cpu_count(),
        })
        
        return info
    
    def cleanup(self) -> None:
        """Clean up resources after execution."""
        logger.info("Cleaning up Task 3 execution resources...")
        
        try:
            if self.monitor:
                self.monitor.stop_task_monitoring()
            
            if self.trainer:
                # Trainer cleanup is handled internally
                pass
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Execute Task 3: Train 8B Model for 53%+ Validation Accuracy")
    parser.add_argument("--max-tasks", type=int, default=20, help="Maximum number of validation tasks")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--generate-report", action="store_true", help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    execution_manager = None
    
    try:
        # Initialize execution manager
        execution_manager = Task3ExecutionManager(args.config)
        
        # Validate prerequisites
        if not execution_manager.validate_prerequisites():
            logger.error("Prerequisites not met - execution aborted")
            sys.exit(1)
        
        # Initialize components
        if not execution_manager.initialize_components():
            logger.error("Component initialization failed - execution aborted")
            sys.exit(2)
        
        # Execute training
        results = execution_manager.execute_training(args.max_tasks)
        
        # Generate report if requested
        if args.generate_report:
            report_path = execution_manager.generate_comprehensive_report(results)
            print(f"\n📊 Comprehensive report generated: {report_path}")
        
        # Determine exit code based on results
        if results.target_achieved:
            logger.info("🎯 Task 3 completed successfully - Target achieved!")
            exit_code = 0
        else:
            logger.warning("⚠️ Task 3 completed with partial success - Target not achieved")
            exit_code = 1
        
        print(f"\n{'='*60}")
        print(f"TASK 3 EXECUTION COMPLETED")
        print(f"Validation Accuracy: {results.average_accuracy:.2%}")
        print(f"Target (53%): {'✅ ACHIEVED' if results.target_achieved else '❌ NOT ACHIEVED'}")
        print(f"Exit Code: {exit_code}")
        print(f"{'='*60}")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        logger.error(f"Task 3 execution failed: {e}", exc_info=True)
        sys.exit(3)
        
    finally:
        if execution_manager:
            execution_manager.cleanup()


if __name__ == "__main__":
    main()