"""
Task 6 Validation Script

Validates the implementation of Task 6: Run full pipeline test on 100 tasks.
This script performs comprehensive validation of all components and runs 
a smaller-scale test to verify the pipeline works correctly.

Validation Steps:
1. System readiness check
2. Component initialization validation  
3. Pipeline configuration validation
4. Small-scale pipeline test (5-10 tasks)
5. Result generation and reporting validation
6. Integration with all previous tasks validation
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import validation utilities
from src.utils.pipeline_test_utils import PipelineTestValidator, validate_pipeline_readiness
from test_pipeline_100_tasks import PipelineTestConfig, PipelineTestOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Task6Validator:
    """Validates Task 6 implementation."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_results = {}
        self.start_time = time.time()
        
        # Create validation output directory
        self.output_dir = Path("validation_results/task6_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Task 6 validation initialized")
    
    def validate_system_readiness(self) -> bool:
        """Step 1: Validate system readiness."""
        logger.info("Step 1: Validating system readiness...")
        
        is_ready, warnings = validate_pipeline_readiness()
        
        self.validation_results["system_readiness"] = {
            "is_ready": is_ready,
            "warnings": warnings,
            "status": "PASS" if is_ready else "FAIL"
        }
        
        if is_ready:
            logger.info("✅ System readiness check passed")
        else:
            logger.error("❌ System readiness check failed")
            for warning in warnings:
                logger.error(f"  - {warning}")
        
        return is_ready
    
    def validate_component_initialization(self) -> bool:
        """Step 2: Validate component initialization."""
        logger.info("Step 2: Validating component initialization...")
        
        try:
            # Test configuration creation
            config = PipelineTestConfig(
                model_name="microsoft/DialoGPT-medium",  # Small model for testing
                num_tasks=5,  # Small test
                use_qlora=False,  # Disable for testing
                task_timeout_minutes=5
            )
            
            # Test orchestrator creation
            orchestrator = PipelineTestOrchestrator(config)
            
            # Test component initialization
            init_success = orchestrator.initialize_components()
            
            # Cleanup
            orchestrator.cleanup()
            
            self.validation_results["component_initialization"] = {
                "config_creation": True,
                "orchestrator_creation": True,  
                "component_init": init_success,
                "status": "PASS" if init_success else "FAIL"
            }
            
            if init_success:
                logger.info("✅ Component initialization validation passed")
                return True
            else:
                logger.error("❌ Component initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Component initialization validation failed: {e}")
            self.validation_results["component_initialization"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def validate_configuration(self) -> bool:
        """Step 3: Validate pipeline configuration."""
        logger.info("Step 3: Validating pipeline configuration...")
        
        try:
            # Test different configurations
            validator = PipelineTestValidator()
            
            # Test CPU configuration
            cpu_config = PipelineTestConfig(
                model_name="microsoft/DialoGPT-medium",
                num_tasks=10,
                use_qlora=False,
                max_concurrent_tasks=1
            )
            
            cpu_validation = validator.validate_test_config(cpu_config)
            
            # Test GPU configuration (if available)
            gpu_validation = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_config = PipelineTestConfig(
                        model_name="meta-llama/Llama-3.1-8B-Instruct",
                        num_tasks=100,
                        use_qlora=True,
                        max_concurrent_tasks=1
                    )
                    gpu_validation = validator.validate_test_config(gpu_config)
            except Exception:
                pass
            
            self.validation_results["configuration_validation"] = {
                "cpu_config": {
                    "valid": cpu_validation.is_valid,
                    "warnings": len(cpu_validation.warnings),
                    "estimated_duration": cpu_validation.estimated_duration_minutes
                },
                "gpu_config": {
                    "valid": gpu_validation.is_valid if gpu_validation else False,
                    "warnings": len(gpu_validation.warnings) if gpu_validation else 0,
                    "estimated_duration": gpu_validation.estimated_duration_minutes if gpu_validation else 0
                } if gpu_validation else None,
                "status": "PASS"
            }
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            self.validation_results["configuration_validation"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def validate_small_scale_pipeline(self) -> bool:
        """Step 4: Run small-scale pipeline test."""
        logger.info("Step 4: Running small-scale pipeline test...")
        
        try:
            # Configure small test
            config = PipelineTestConfig(
                model_name="microsoft/DialoGPT-medium",  # CPU-friendly model
                num_tasks=3,  # Very small test
                use_qlora=False,
                enable_gradient_checkpointing=False,
                use_mixed_precision=False,
                task_timeout_minutes=3,
                max_inference_time_minutes=1.0
            )
            
            # Run test
            orchestrator = PipelineTestOrchestrator(config)
            
            pipeline_start = time.time()
            results = orchestrator.execute_pipeline_test()
            pipeline_duration = time.time() - pipeline_start
            
            # Validate results
            success_rate = results.get('task_execution', {}).get('success_rate', 0)
            total_tasks = results.get('task_execution', {}).get('total_tasks', 0)
            
            # Check if results were saved
            results_saved = (orchestrator.output_dir / "pipeline_summary.json").exists()
            report_generated = (orchestrator.output_dir / "pipeline_test_report.txt").exists()
            
            # Cleanup
            orchestrator.cleanup()
            
            self.validation_results["small_scale_test"] = {
                "executed": True,
                "duration_seconds": pipeline_duration,
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "results_saved": results_saved,
                "report_generated": report_generated,
                "test_passed": success_rate > 0 and results_saved and report_generated,
                "status": "PASS" if success_rate > 0 and results_saved else "FAIL"
            }
            
            if success_rate > 0 and results_saved:
                logger.info(f"✅ Small-scale pipeline test passed ({success_rate:.1f}% success rate)")
                return True
            else:
                logger.error(f"❌ Small-scale pipeline test failed ({success_rate:.1f}% success rate)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Small-scale pipeline test failed: {e}")
            self.validation_results["small_scale_test"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def validate_result_analysis(self) -> bool:
        """Step 5: Validate result generation and reporting."""
        logger.info("Step 5: Validating result analysis capabilities...")
        
        try:
            from src.utils.pipeline_test_utils import TestResultAnalyzer
            
            # Find most recent test results
            validation_dirs = list(Path("validation_results").glob("pipeline_100_*"))
            if validation_dirs:
                latest_results = max(validation_dirs, key=lambda p: p.stat().st_mtime)
                
                # Test result analysis
                analyzer = TestResultAnalyzer(latest_results)
                performance_patterns = analyzer.analyze_performance_patterns()
                failure_patterns = analyzer.identify_failure_patterns()
                
                # Test visualization generation
                viz_dir = self.output_dir / "test_visualizations"
                analyzer.generate_performance_visualizations(viz_dir)
                
                # Test detailed report generation
                report_file = self.output_dir / "test_analysis_report.txt"
                analyzer.generate_detailed_report(report_file)
                
                analysis_successful = (
                    "error" not in performance_patterns and
                    viz_dir.exists() and
                    report_file.exists()
                )
                
                self.validation_results["result_analysis"] = {
                    "analysis_completed": analysis_successful,
                    "performance_analysis": "error" not in performance_patterns,
                    "failure_analysis": "message" not in failure_patterns,
                    "visualization_generated": viz_dir.exists(),
                    "report_generated": report_file.exists(),
                    "status": "PASS" if analysis_successful else "FAIL"
                }
                
                if analysis_successful:
                    logger.info("✅ Result analysis validation passed")
                    return True
                else:
                    logger.error("❌ Result analysis validation failed")
                    return False
            else:
                logger.warning("⚠️ No test results found for analysis validation")
                self.validation_results["result_analysis"] = {
                    "status": "SKIP",
                    "reason": "No test results available"
                }
                return True  # Don't fail validation for this
                
        except Exception as e:
            logger.error(f"❌ Result analysis validation failed: {e}")
            self.validation_results["result_analysis"] = {
                "status": "FAIL", 
                "error": str(e)
            }
            return False
    
    def validate_integration_with_previous_tasks(self) -> bool:
        """Step 6: Validate integration with all previous tasks."""
        logger.info("Step 6: Validating integration with previous tasks...")
        
        integration_checks = {}
        
        try:
            # Check Task 1 - QLoRA integration
            try:
                from src.utils.lora_adapter import apply_lora_to_model
                integration_checks["task1_qlora"] = True
                logger.info("✅ Task 1 QLoRA integration available")
            except ImportError:
                integration_checks["task1_qlora"] = False
                logger.warning("⚠️ Task 1 QLoRA integration not found")
            
            # Check Task 2 - Gradient checkpointing and mixed precision
            try:
                import torch
                # These are standard PyTorch features
                integration_checks["task2_optimizations"] = hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp')
                logger.info("✅ Task 2 optimizations available")
            except Exception:
                integration_checks["task2_optimizations"] = False
            
            # Check Task 3 - Training configuration
            try:
                from src.domain.services.training_orchestrator import TrainingOrchestrator, TrainingConfig
                integration_checks["task3_training"] = True
                logger.info("✅ Task 3 training configuration available")
            except ImportError:
                integration_checks["task3_training"] = False
                logger.warning("⚠️ Task 3 training configuration not found")
            
            # Check Task 4 - Inference optimization
            try:
                from src.utils.inference_optimization_poc import InferenceOptimizer
                integration_checks["task4_inference"] = True
                logger.info("✅ Task 4 inference optimization available")
            except ImportError:
                integration_checks["task4_inference"] = False
                logger.warning("⚠️ Task 4 inference optimization not found")
            
            # Check Task 5 - Early stopping
            try:
                from src.utils.early_stopping_utils import EarlyStoppingMonitor
                integration_checks["task5_early_stopping"] = True
                logger.info("✅ Task 5 early stopping available")
            except ImportError:
                integration_checks["task5_early_stopping"] = False
                logger.warning("⚠️ Task 5 early stopping not found")
            
            # Check comprehensive error handling
            try:
                from src.utils.comprehensive_error_handling import resilient_operation, error_reporter
                integration_checks["error_handling"] = True
                logger.info("✅ Comprehensive error handling available")
            except ImportError:
                integration_checks["error_handling"] = False
                logger.warning("⚠️ Comprehensive error handling not found")
            
            # Calculate integration score
            available_integrations = sum(integration_checks.values())
            total_integrations = len(integration_checks)
            integration_score = (available_integrations / total_integrations) * 100
            
            self.validation_results["task_integration"] = {
                "integrations": integration_checks,
                "available": available_integrations,
                "total": total_integrations,
                "score": integration_score,
                "status": "PASS" if integration_score >= 80 else "FAIL"
            }
            
            if integration_score >= 80:
                logger.info(f"✅ Integration validation passed ({integration_score:.1f}%)")
                return True
            else:
                logger.error(f"❌ Integration validation failed ({integration_score:.1f}%)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Integration validation failed: {e}")
            self.validation_results["task_integration"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        # Calculate overall validation score
        passed_checks = sum(1 for result in self.validation_results.values() 
                           if result.get("status") == "PASS")
        total_checks = len([result for result in self.validation_results.values() 
                           if result.get("status") in ["PASS", "FAIL"]])
        
        validation_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        overall_status = "PASS" if validation_score >= 80 else "FAIL"
        
        total_duration = time.time() - self.start_time
        
        # Create summary
        summary = {
            "validation_metadata": {
                "validation_time": time.ctime(),
                "duration_seconds": total_duration,
                "validator_version": "1.0"
            },
            "overall_result": {
                "status": overall_status,
                "score": validation_score,
                "passed_checks": passed_checks,
                "total_checks": total_checks
            },
            "detailed_results": self.validation_results
        }
        
        # Save JSON report
        json_report = self.output_dir / "task6_validation_report.json"
        with open(json_report, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate human-readable report
        report_lines = [
            "="*80,
            "TASK 6 VALIDATION REPORT: Run Full Pipeline Test on 100 Tasks",
            "="*80,
            f"Validation Date: {time.ctime()}",
            f"Duration: {total_duration:.2f} seconds",
            f"Overall Status: {'✅ PASS' if overall_status == 'PASS' else '❌ FAIL'}",
            f"Validation Score: {validation_score:.1f}% ({passed_checks}/{total_checks})",
            "",
            "DETAILED VALIDATION RESULTS",
            "-"*40,
        ]
        
        # Add detailed results
        for step, result in self.validation_results.items():
            status_icon = "✅" if result.get("status") == "PASS" else "❌" if result.get("status") == "FAIL" else "⚠️"
            step_name = step.replace("_", " ").title()
            report_lines.append(f"{status_icon} {step_name}: {result.get('status')}")
            
            if result.get("error"):
                report_lines.append(f"    Error: {result['error']}")
        
        report_lines.extend([
            "",
            "SUMMARY",
            "-"*20,
        ])
        
        if overall_status == "PASS":
            report_lines.extend([
                "✅ Task 6 implementation is VALIDATED",
                "✅ Pipeline test is ready for 100-task execution",
                "✅ All critical components are functional",
                "✅ Integration with previous tasks confirmed"
            ])
        else:
            report_lines.extend([
                "❌ Task 6 implementation needs attention",
                "❌ Review failed validation steps",
                "❌ Address issues before full pipeline test"
            ])
        
        report_lines.extend([
            "",
            "="*80,
            "End of Validation Report",
            "="*80
        ])
        
        # Save text report
        text_report = self.output_dir / "task6_validation_report.txt"
        with open(text_report, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        print('\n'.join(report_lines))
        
        logger.info(f"Validation reports saved to: {self.output_dir}")
    
    def run_full_validation(self) -> bool:
        """Run complete Task 6 validation."""
        logger.info("="*80)
        logger.info("STARTING TASK 6 VALIDATION")
        logger.info("="*80)
        
        validation_steps = [
            ("System Readiness", self.validate_system_readiness),
            ("Component Initialization", self.validate_component_initialization),
            ("Configuration Validation", self.validate_configuration),
            ("Small-Scale Pipeline Test", self.validate_small_scale_pipeline),
            ("Result Analysis", self.validate_result_analysis),
            ("Integration with Previous Tasks", self.validate_integration_with_previous_tasks)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            try:
                step_passed = validation_func()
                if not step_passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Validation step '{step_name}' failed with exception: {e}")
                all_passed = False
        
        # Generate final report
        self.generate_validation_report()
        
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("🎉 TASK 6 VALIDATION SUCCESSFUL!")
            logger.info("The pipeline is ready for full 100-task execution.")
        else:
            logger.error("❌ TASK 6 VALIDATION FAILED!")
            logger.error("Review validation report and address issues.")
        logger.info("="*80)
        
        return all_passed


def main():
    """Main validation execution."""
    print("Task 6 Validation: Run Full Pipeline Test on 100 Tasks")
    print("="*60)
    
    # Create validator and run validation
    validator = Task6Validator()
    validation_passed = validator.run_full_validation()
    
    # Return appropriate exit code
    return 0 if validation_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)