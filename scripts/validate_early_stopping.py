#!/usr/bin/env python3
"""
Early Stopping Validation Script for Story 1.5 Task 5

Comprehensive validation script to test early stopping functionality,
checkpoint recovery, and integration with the training system.
"""
import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.domain.services.training_orchestrator import (
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingOrchestrator,
)
from src.domain.services.ttt_service import TTTModelService
from src.utils.comprehensive_error_handling import ErrorContext, ErrorReporter, ErrorSeverity
from src.utils.early_stopping_utils import (
    EarlyStoppingConfigManager,
    EarlyStoppingMonitor,
    validate_early_stopping_config,
)

logger = logging.getLogger(__name__)


class EarlyStoppingValidator:
    """Validator for early stopping functionality."""

    def __init__(self, output_dir: Path):
        """Initialize validator."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[str, Any] = {
            "validation_timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_results": [],
            "errors": [],
        }

        self.error_reporter = ErrorReporter()

    def run_validation(self, quick_mode: bool = False) -> dict[str, Any]:
        """
        Run complete early stopping validation.
        
        Args:
            quick_mode: If True, run abbreviated tests
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting early stopping validation")
        print("🔍 Early Stopping System Validation")
        print("=" * 50)

        try:
            # Configuration validation
            self._test_configuration_validation()

            # Core functionality tests
            self._test_early_stopping_mechanisms()

            # Checkpoint integration tests
            self._test_checkpoint_integration()

            # Auto-save and resume tests
            if not quick_mode:
                self._test_auto_save_resume()
                self._test_error_recovery_integration()

            # Performance and efficiency tests
            self._test_training_efficiency()

            # Generate final report
            self._generate_validation_report()

            success_rate = (self.results["tests_passed"] /
                          (self.results["tests_passed"] + self.results["tests_failed"]) * 100)

            print(f"\n✅ Validation completed: {self.results['tests_passed']} passed, "
                  f"{self.results['tests_failed']} failed ({success_rate:.1f}% success rate)")

            return self.results

        except Exception as e:
            self._record_error("Validation framework error", e)
            print(f"❌ Validation failed with error: {e}")
            return self.results

    def _test_configuration_validation(self) -> None:
        """Test configuration validation."""
        print("\n1. Configuration Validation Tests")
        print("-" * 30)

        config_manager = EarlyStoppingConfigManager()

        # Test 1.1: Default configuration validation
        try:
            default_config = EarlyStoppingConfig()
            warnings = validate_early_stopping_config(default_config)

            self._record_test(
                "Default Configuration",
                len(warnings) == 0,
                f"Default config has {len(warnings)} warnings: {warnings}"
            )
        except Exception as e:
            self._record_test("Default Configuration", False, str(e))

        # Test 1.2: Predefined configurations
        try:
            predefined_configs = ["conservative", "aggressive", "balanced"]
            all_valid = True
            config_details = []

            for config_name in predefined_configs:
                config = config_manager.get_config(config_name)
                if config:
                    warnings = validate_early_stopping_config(config)
                    config_details.append(f"{config_name}: {len(warnings)} warnings")
                    if len(warnings) > 2:  # Allow minor warnings
                        all_valid = False
                else:
                    all_valid = False
                    config_details.append(f"{config_name}: NOT FOUND")

            self._record_test(
                "Predefined Configurations",
                all_valid,
                f"Config validation: {', '.join(config_details)}"
            )
        except Exception as e:
            self._record_test("Predefined Configurations", False, str(e))

        # Test 1.3: Adaptive configuration creation
        try:
            adaptive_config = config_manager.create_adaptive_config("8B", 45, 24)
            warnings = validate_early_stopping_config(adaptive_config)

            self._record_test(
                "Adaptive Configuration",
                adaptive_config.patience >= 2 and len(warnings) <= 2,
                f"8B adaptive config: patience={adaptive_config.patience}, warnings={len(warnings)}"
            )
        except Exception as e:
            self._record_test("Adaptive Configuration", False, str(e))

    def _test_early_stopping_mechanisms(self) -> None:
        """Test core early stopping mechanisms."""
        print("\n2. Early Stopping Mechanism Tests")
        print("-" * 35)

        # Test 2.1: Patience-based early stopping
        try:
            config = EarlyStoppingConfig(patience=3, min_delta=0.01)
            monitor = EarlyStoppingMonitor(config)

            session_id = "patience_test"

            # Simulate no improvement
            should_stop = False
            for i in range(5):
                should_stop, reason = monitor.track_training_session(
                    session_id, epoch=0, step=i*10, metric_value=0.5, all_metrics={}
                )
                if should_stop:
                    break

            self._record_test(
                "Patience-based Early Stopping",
                should_stop and "Patience exceeded" in reason,
                f"Stopped after patience exceeded: {reason}"
            )
        except Exception as e:
            self._record_test("Patience-based Early Stopping", False, str(e))

        # Test 2.2: Improvement detection
        try:
            config = EarlyStoppingConfig(patience=3, min_delta=0.01)
            monitor = EarlyStoppingMonitor(config)

            session_id = "improvement_test"

            # Simulate improvements
            values = [0.3, 0.4, 0.42, 0.45, 0.47]  # Consistent improvements
            should_stop = False

            for i, value in enumerate(values):
                should_stop, reason = monitor.track_training_session(
                    session_id, epoch=0, step=i*10, metric_value=value, all_metrics={}
                )
                if should_stop:
                    break

            summary = monitor.get_session_summary(session_id)
            improvements = summary["improvements_count"] if summary else 0

            self._record_test(
                "Improvement Detection",
                not should_stop and improvements >= 4,
                f"Detected {improvements} improvements, no early stopping"
            )
        except Exception as e:
            self._record_test("Improvement Detection", False, str(e))

        # Test 2.3: Memory and time limits
        try:
            # Mock orchestrator for limit testing
            mock_model_service = Mock(spec=TTTModelService)
            mock_model_service.device = Mock()
            mock_model_service.memory_manager = Mock()
            mock_model_service._get_memory_usage.return_value = 25.0  # Over 24GB limit

            config = TrainingConfig(
                memory_limit_mb=24576,  # 24GB
                max_training_time=300,  # 5 minutes
                early_stopping=EarlyStoppingConfig()
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_repo = CheckpointRepository(Path(temp_dir) / "checkpoints")
                orchestrator = TrainingOrchestrator(
                    model_service=mock_model_service,
                    config=config,
                    checkpoint_repository=checkpoint_repo
                )

                orchestrator.start_time = time.time() - 400  # Started 400s ago (over limit)

                should_stop, reason = orchestrator.check_early_stopping({"validation_accuracy": 0.5})

                memory_limit_triggered = "Memory limit exceeded" in reason or "Time limit exceeded" in reason

            self._record_test(
                "Memory/Time Limits",
                should_stop and memory_limit_triggered,
                f"Limit-based stopping: {reason}"
            )
        except Exception as e:
            self._record_test("Memory/Time Limits", False, str(e))

    def _test_checkpoint_integration(self) -> None:
        """Test checkpoint integration."""
        print("\n3. Checkpoint Integration Tests")
        print("-" * 32)

        # Test 3.1: Checkpoint repository functionality
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_repo = CheckpointRepository(Path(temp_dir) / "checkpoints")

                # Create test checkpoint
                model_state = {"model_param": "test_value"}
                training_metrics = {
                    "final_accuracy": 0.65,
                    "training_time": 100.0,
                    "final_memory_mb": 8192.0,
                }
                lora_config = {"rank": 8, "alpha": 16}

                metadata = checkpoint_repo.save_checkpoint(
                    checkpoint_id="test_checkpoint",
                    task_id="test_task",
                    model_state=model_state,
                    training_metrics=training_metrics,
                    lora_config=lora_config
                )

                # Verify checkpoint exists and is valid
                checkpoints = checkpoint_repo.list_checkpoints(task_id="test_task")
                is_valid = checkpoint_repo.validate_checkpoint_integrity("test_checkpoint")

                self._record_test(
                    "Checkpoint Repository",
                    len(checkpoints) == 1 and is_valid,
                    f"Checkpoint saved and validated: {metadata.checkpoint_id}"
                )
        except Exception as e:
            self._record_test("Checkpoint Repository", False, str(e))

        # Test 3.2: Auto-resume candidates
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_repo = CheckpointRepository(Path(temp_dir) / "checkpoints")

                # Create multiple checkpoints
                for i, accuracy in enumerate([0.4, 0.6, 0.55]):
                    checkpoint_repo.save_checkpoint(
                        checkpoint_id=f"checkpoint_{i}",
                        task_id="resume_test",
                        model_state={"param": f"value_{i}"},
                        training_metrics={"final_accuracy": accuracy, "training_time": 50.0, "final_memory_mb": 4096.0},
                        lora_config={"rank": 8, "alpha": 16}
                    )

                # Test auto-resume candidate selection
                candidates = checkpoint_repo.get_auto_resume_candidates("resume_test")
                best_checkpoint = checkpoint_repo.get_best_checkpoint("resume_test")

                self._record_test(
                    "Auto-resume Candidates",
                    len(candidates) >= 2 and best_checkpoint.accuracy == 0.6,
                    f"Found {len(candidates)} candidates, best accuracy: {best_checkpoint.accuracy}"
                )
        except Exception as e:
            self._record_test("Auto-resume Candidates", False, str(e))

    def _test_auto_save_resume(self) -> None:
        """Test auto-save and resume functionality."""
        print("\n4. Auto-save and Resume Tests")
        print("-" * 29)

        # Test 4.1: Auto-save triggering
        try:
            mock_model_service = Mock(spec=TTTModelService)
            mock_model_service.device = Mock()
            mock_model_service.memory_manager = Mock()
            mock_model_service._get_memory_usage.return_value = 8.0

            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_repo = CheckpointRepository(Path(temp_dir) / "checkpoints")

                config = TrainingConfig(
                    early_stopping=EarlyStoppingConfig(
                        auto_save_enabled=True,
                        auto_save_interval_minutes=1,  # Very short for testing
                        auto_save_on_improvement=True
                    )
                )

                orchestrator = TrainingOrchestrator(
                    model_service=mock_model_service,
                    config=config,
                    checkpoint_repository=checkpoint_repo
                )

                # Mock required components
                orchestrator.model = Mock()
                orchestrator.optimizer = Mock()
                orchestrator.scheduler = Mock()
                orchestrator.lora_adapter = Mock()
                orchestrator.current_task_id = "autosave_test"
                orchestrator.start_time = time.time() - 120  # 2 minutes ago

                # Mock state dicts
                orchestrator.model.state_dict.return_value = {"param": "value"}
                orchestrator.optimizer.state_dict.return_value = {"opt": "value"}
                orchestrator.scheduler.state_dict.return_value = {"sched": "value"}
                orchestrator.lora_adapter.get_adapter_state.return_value = {"lora": "value"}

                # Trigger auto-save through improvement
                metrics = {
                    "validation_accuracy": 0.65,  # Improvement over default 0
                    "loss": 0.35,
                    "step": 25,
                    "epoch": 1,
                    "learning_rate": 1e-5,
                    "memory_mb": 8192
                }

                should_stop, _ = orchestrator.check_early_stopping(metrics)

                # Check if checkpoint was created
                checkpoints = checkpoint_repo.list_checkpoints(task_id="autosave_test")

                self._record_test(
                    "Auto-save Triggering",
                    len(checkpoints) > 0,
                    f"Auto-save created {len(checkpoints)} checkpoint(s)"
                )
        except Exception as e:
            self._record_test("Auto-save Triggering", False, str(e))

    def _test_error_recovery_integration(self) -> None:
        """Test error recovery integration."""
        print("\n5. Error Recovery Integration")
        print("-" * 30)

        # Test 5.1: Error reporting integration
        try:
            context = ErrorContext(
                operation="test_early_stopping",
                model_name="test_model",
                attempt_number=1
            )

            test_error = RuntimeError("Test early stopping error")

            self.error_reporter.report_error(
                error=test_error,
                context=context,
                severity=ErrorSeverity.MEDIUM
            )

            error_summary = self.error_reporter.get_error_summary()

            self._record_test(
                "Error Reporting Integration",
                error_summary["total_errors"] >= 1,
                f"Error reported and tracked: {error_summary['total_errors']} total errors"
            )
        except Exception as e:
            self._record_test("Error Reporting Integration", False, str(e))

    def _test_training_efficiency(self) -> None:
        """Test training efficiency analysis."""
        print("\n6. Training Efficiency Analysis")
        print("-" * 32)

        # Test 6.1: Efficiency metrics calculation
        try:
            config = EarlyStoppingConfig(patience=4, min_delta=0.01)
            monitor = EarlyStoppingMonitor(config)

            session_id = "efficiency_test"

            # Simulate realistic training progression
            training_progression = [
                (0, 0.25),
                (1, 0.35),  # Improvement
                (2, 0.42),  # Improvement
                (3, 0.44),  # Small improvement
                (4, 0.43),  # Slight decrease
                (5, 0.42),  # Decrease
                (6, 0.41),  # Continued decrease - should trigger early stopping
            ]

            early_stopped = False
            final_step = 0

            for step, accuracy in training_progression:
                should_stop, reason = monitor.track_training_session(
                    session_id, epoch=step//2, step=step*10, metric_value=accuracy, all_metrics={}
                )
                final_step = step
                if should_stop:
                    early_stopped = True
                    break

            # Analyze efficiency
            efficiency = monitor.analyze_training_efficiency(session_id)
            summary = monitor.get_session_summary(session_id)

            efficiency_gain = efficiency.get("efficiency_gain", 0)
            improvement_ratio = efficiency.get("improvement_ratio", 0)

            self._record_test(
                "Training Efficiency Analysis",
                early_stopped and efficiency_gain > 0 and improvement_ratio > 0,
                f"Early stopped at step {final_step}, efficiency gain: {efficiency_gain:.2%}, "
                f"improvement ratio: {improvement_ratio:.2%}"
            )
        except Exception as e:
            self._record_test("Training Efficiency Analysis", False, str(e))

    def _record_test(self, test_name: str, passed: bool, details: str) -> None:
        """Record test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed or details:
            print(f"    {details}")

        self.results["test_results"].append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

        if passed:
            self.results["tests_passed"] += 1
        else:
            self.results["tests_failed"] += 1

    def _record_error(self, context: str, error: Exception) -> None:
        """Record validation error."""
        error_info = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }

        self.results["errors"].append(error_info)
        logger.error(f"Validation error in {context}: {error}")

    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        report_path = self.output_dir / "early_stopping_validation_report.json"

        # Add summary statistics
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        success_rate = (self.results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0

        self.results["summary"] = {
            "total_tests": total_tests,
            "success_rate": success_rate,
            "validation_successful": self.results["tests_failed"] == 0,
            "critical_failures": len([t for t in self.results["test_results"]
                                    if not t["passed"] and "Configuration" in t["test_name"]]),
        }

        # Error summary
        if self.results["errors"]:
            self.results["error_summary"] = self.error_reporter.get_error_summary()

        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Validation report saved: {report_path}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate Early Stopping Implementation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip comprehensive tests)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save validation report to file"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Early Stopping Validation for Story 1.5 Task 5")
    print("=" * 50)

    try:
        # Run validation
        validator = EarlyStoppingValidator(args.output_dir)
        results = validator.run_validation(quick_mode=args.quick)

        # Print summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 20)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['tests_passed']}")
        print(f"Failed: {results['tests_failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")

        if results["summary"]["validation_successful"]:
            print("\n🎉 VALIDATION SUCCESSFUL - Early stopping system is ready!")
        else:
            print(f"\n⚠️  VALIDATION ISSUES FOUND - {results['tests_failed']} tests failed")

            # Show failed tests
            failed_tests = [t for t in results["test_results"] if not t["passed"]]
            if failed_tests:
                print("\nFailed Tests:")
                for test in failed_tests:
                    print(f"  ❌ {test['test_name']}: {test['details']}")

        # Exit code based on validation result
        sys.exit(0 if results["summary"]["validation_successful"] else 1)

    except KeyboardInterrupt:
        print("\n❌ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
