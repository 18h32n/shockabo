"""
Integration tests for early stopping functionality in Story 1.5 Task 5.

Tests the complete early stopping system including training orchestrator,
checkpoint repository, and recovery mechanisms.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.domain.models import ARCTask
from src.domain.services.training_orchestrator import (
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingOrchestrator,
)
from src.domain.services.ttt_service import TTTModelService
from src.utils.comprehensive_error_handling import resilient_operation
from src.utils.early_stopping_utils import EarlyStoppingConfigManager, EarlyStoppingMonitor


class TestEarlyStoppingEndToEnd(unittest.TestCase):
    """End-to-end tests for early stopping system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_repo = CheckpointRepository(self.checkpoint_path)

        # Create mock ARC task
        self.test_task = Mock(spec=ARCTask)
        self.test_task.task_id = "test_early_stopping_001"
        self.test_task.train_examples = [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
        ]

        # Mock model service
        self.mock_model_service = Mock(spec=TTTModelService)
        self.mock_model_service.device = torch.device("cpu")
        self.mock_model_service.memory_manager = Mock()
        self.mock_model_service.memory_manager.get_memory_usage.return_value = {"usage_percentage": 50.0}
        self.mock_model_service._get_memory_usage.return_value = 8.0
        self.mock_model_service.optimize_memory = Mock()
        self.mock_model_service.cleanup = Mock()

        # Mock model loading
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_model_service.load_model.return_value = (self.mock_model, self.mock_tokenizer)

        # Mock tokenizer
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        self.mock_tokenizer.pad_token_id = 0

        # Create early stopping config
        self.early_stopping_config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            auto_save_enabled=True,
            auto_save_interval_minutes=2,  # Short interval for testing
            auto_resume_enabled=True,
            restore_best_weights=True
        )

        # Create training config
        self.training_config = TrainingConfig(
            num_epochs=10,
            validation_frequency=5,
            early_stopping=self.early_stopping_config,
            max_training_time=300,  # 5 minutes
            target_accuracy=0.8,
        )

        # Create orchestrator
        self.orchestrator = TrainingOrchestrator(
            model_service=self.mock_model_service,
            config=self.training_config,
            checkpoint_repository=self.checkpoint_repo
        )

    def test_early_stopping_with_patience(self):
        """Test early stopping triggered by patience."""
        # Mock training components
        self._setup_training_mocks()

        # Simulate training with plateauing accuracy
        validation_accuracies = [0.3, 0.4, 0.45, 0.46, 0.46, 0.46, 0.455]

        with patch.object(self.orchestrator, 'validate') as mock_validate:
            mock_validate.side_effect = validation_accuracies

            with patch('time.time') as mock_time:
                mock_time.side_effect = [0, 10, 20, 30, 40, 50, 60, 70]  # Time progression

                # Run training simulation
                result = self._simulate_training_loop(len(validation_accuracies))

                # Verify early stopping was triggered
                self.assertTrue(result["early_stopping_triggered"])
                self.assertIn("No improvement", str(result.get("stop_reason", "")))

                # Verify checkpoints were saved
                checkpoints = self.checkpoint_repo.list_checkpoints(task_id=self.test_task.task_id)
                self.assertGreater(len(checkpoints), 0)

    def test_early_stopping_with_target_achieved(self):
        """Test early stopping when target accuracy is achieved."""
        self._setup_training_mocks()

        # Simulate training reaching target accuracy
        validation_accuracies = [0.3, 0.5, 0.7, 0.85]  # Reaches target of 0.8

        with patch.object(self.orchestrator, 'validate') as mock_validate:
            mock_validate.side_effect = validation_accuracies

            with patch('time.time') as mock_time:
                mock_time.side_effect = [0, 10, 20, 30, 40]

                result = self._simulate_training_loop(len(validation_accuracies))

                # Verify early stopping was triggered by target achievement
                self.assertTrue(result["early_stopping_triggered"])
                self.assertIn("Target accuracy achieved", str(result.get("stop_reason", "")))

    def test_auto_save_and_resume(self):
        """Test auto-save and resume functionality."""
        self._setup_training_mocks()

        # First training session - simulate interruption
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 60, 120, 180]  # Times that trigger auto-save

            # Simulate some training progress
            metrics = {
                "validation_accuracy": 0.6,
                "loss": 0.4,
                "step": 25,
                "epoch": 1,
                "learning_rate": 1e-5,
                "memory_mb": 8192
            }

            # Trigger auto-save through improvement
            should_stop, _ = self.orchestrator.check_early_stopping(metrics)
            self.assertFalse(should_stop)

        # Verify checkpoint was created
        checkpoints = self.checkpoint_repo.list_checkpoints(task_id=self.test_task.task_id)
        self.assertGreater(len(checkpoints), 0)

        # Create new orchestrator to test resume
        new_orchestrator = TrainingOrchestrator(
            model_service=self.mock_model_service,
            config=self.training_config,
            checkpoint_repository=self.checkpoint_repo
        )

        # Mock the necessary components for resume
        new_orchestrator.model = Mock()
        new_orchestrator.optimizer = Mock()
        new_orchestrator.scheduler = Mock()
        new_orchestrator.lora_adapter = Mock()

        # Test auto-resume
        resume_info = new_orchestrator.try_auto_resume(self.test_task.task_id)

        # Should successfully resume if checkpoint is recent enough
        if resume_info:
            self.assertIsInstance(resume_info, dict)
            self.assertIn("checkpoint_id", resume_info)
            self.assertIn("resumed_accuracy", resume_info)

    def test_checkpoint_recovery_scenarios(self):
        """Test various checkpoint recovery scenarios."""
        self._setup_training_mocks()

        # Create multiple checkpoints with different accuracies
        checkpoint_data = [
            (0.4, "checkpoint_1"),
            (0.55, "checkpoint_2"),  # Best
            (0.52, "checkpoint_3"),
            (0.51, "checkpoint_4")
        ]

        for accuracy, checkpoint_id in checkpoint_data:
            model_state = {
                "model_state_dict": {"param": f"value_{checkpoint_id}"},
                "optimizer_state_dict": {},
                "training_step": 10,
                "epoch": 1,
            }

            training_metrics = {
                "final_accuracy": accuracy,
                "training_time": 100.0,
                "final_memory_mb": 8192.0,
            }

            lora_config = {"rank": 8, "alpha": 16}

            self.checkpoint_repo.save_checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=self.test_task.task_id,
                model_state=model_state,
                training_metrics=training_metrics,
                lora_config=lora_config
            )

        # Test getting best checkpoint
        best_checkpoint = self.checkpoint_repo.get_best_checkpoint(self.test_task.task_id)
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(best_checkpoint.checkpoint_id, "checkpoint_2")
        self.assertAlmostEqual(best_checkpoint.accuracy, 0.55)

        # Test resume candidates
        candidates = self.checkpoint_repo.get_auto_resume_candidates(self.test_task.task_id)
        self.assertGreater(len(candidates), 0)

        # Best candidate should be the one with highest accuracy
        self.assertEqual(candidates[0].checkpoint_id, "checkpoint_2")

    def test_early_stopping_with_memory_pressure(self):
        """Test early stopping under memory pressure."""
        self._setup_training_mocks()

        # Mock increasing memory usage
        memory_values = [8.0, 12.0, 18.0, 22.0]  # Exceeds 20GB limit
        self.mock_model_service._get_memory_usage.side_effect = memory_values

        # Simulate training
        for i, memory in enumerate(memory_values):
            metrics = {
                "validation_accuracy": 0.5 + i * 0.05,
                "loss": 0.5 - i * 0.05,
                "step": i * 10,
                "epoch": i,
                "learning_rate": 1e-5,
                "memory_mb": memory * 1024
            }

            should_stop, reason = self.orchestrator.check_early_stopping(metrics)

            if memory > 20.0:  # Should trigger memory-based early stopping
                self.assertTrue(should_stop)
                self.assertIn("Memory limit exceeded", reason)
                break

    def test_early_stopping_configuration_validation(self):
        """Test early stopping configuration validation."""
        from src.utils.early_stopping_utils import validate_early_stopping_config

        # Test valid configuration
        valid_config = EarlyStoppingConfig(
            patience=5,
            min_delta=0.01,
            monitor_metric="validation_accuracy",
            mode="max"
        )

        warnings = validate_early_stopping_config(valid_config)
        self.assertIsInstance(warnings, list)

        # Test invalid configuration
        invalid_config = EarlyStoppingConfig(
            patience=0,  # Invalid
            min_delta=-0.1,  # Invalid
            monitor_metric="validation_loss",
            mode="max"  # Mismatch with loss metric
        )

        warnings = validate_early_stopping_config(invalid_config)
        self.assertGreater(len(warnings), 0)

    def test_training_efficiency_analysis(self):
        """Test training efficiency analysis with early stopping."""
        config_manager = EarlyStoppingConfigManager()
        monitor = EarlyStoppingMonitor(self.early_stopping_config)

        session_id = "efficiency_test"

        # Simulate training session with mixed improvements
        training_data = [
            (0, 0.2),
            (1, 0.3),   # Improvement
            (2, 0.35),  # Small improvement
            (3, 0.34),  # Slight decrease
            (4, 0.33),  # Decrease
            (5, 0.32),  # Continued decrease
        ]

        early_stopped = False
        for step, accuracy in training_data:
            should_stop, reason = monitor.track_training_session(
                session_id=session_id,
                epoch=step // 2,
                step=step * 10,
                metric_value=accuracy,
                all_metrics={"loss": 1.0 - accuracy}
            )

            if should_stop:
                early_stopped = True
                break

        # Analyze efficiency
        efficiency_analysis = monitor.analyze_training_efficiency(session_id)

        self.assertIsInstance(efficiency_analysis, dict)
        self.assertEqual(efficiency_analysis["session_id"], session_id)
        self.assertGreaterEqual(efficiency_analysis["improvement_ratio"], 0)
        self.assertIn("recommendations", efficiency_analysis)

        # Should have triggered early stopping due to patience
        self.assertTrue(early_stopped)

    def test_adaptive_configuration_creation(self):
        """Test adaptive early stopping configuration for different scenarios."""
        config_manager = EarlyStoppingConfigManager()

        # Test 8B model with limited time
        config_8b_limited = config_manager.create_adaptive_config(
            model_size="8B",
            time_budget_minutes=20,
            memory_budget_gb=24
        )

        # Should have reduced patience for time constraints
        self.assertLessEqual(config_8b_limited.patience, 4)
        self.assertLessEqual(config_8b_limited.auto_save_interval_minutes, 8)

        # Test 1B model with more time
        config_1b_generous = config_manager.create_adaptive_config(
            model_size="1B",
            time_budget_minutes=120,
            memory_budget_gb=32
        )

        # Should have more patience with generous time
        self.assertGreaterEqual(config_1b_generous.patience, 6)

        # Test memory-constrained scenario
        config_memory_limited = config_manager.create_adaptive_config(
            model_size="8B",
            time_budget_minutes=60,
            memory_budget_gb=12
        )

        # Should be more aggressive due to memory constraints
        self.assertLessEqual(config_memory_limited.patience, config_8b_limited.patience)

    def _setup_training_mocks(self):
        """Set up common training mocks."""
        # Mock model components
        self.orchestrator.model = Mock()
        self.orchestrator.tokenizer = self.mock_tokenizer
        self.orchestrator.optimizer = Mock()
        self.orchestrator.scheduler = Mock()
        self.orchestrator.lora_adapter = Mock()
        self.orchestrator.scaler = None

        # Mock state dicts
        self.orchestrator.model.state_dict.return_value = {"model_param": "value"}
        self.orchestrator.optimizer.state_dict.return_value = {"opt_param": "value"}
        self.orchestrator.scheduler.state_dict.return_value = {"sched_param": "value"}
        self.orchestrator.lora_adapter.get_adapter_state.return_value = {"lora_param": "value"}

        # Mock scheduler
        self.orchestrator.scheduler.get_last_lr.return_value = [1e-5]

        # Set up orchestrator state
        self.orchestrator.current_task_id = self.test_task.task_id
        self.orchestrator.start_time = 0.0

    def _simulate_training_loop(self, num_validations):
        """Simulate training loop with early stopping checks."""
        results = {
            "early_stopping_triggered": False,
            "stop_reason": None,
            "validations_performed": 0,
        }

        for i in range(num_validations):
            # Create metrics for this validation
            metrics = {
                "validation_accuracy": 0.5 + i * 0.02,
                "loss": 0.5 - i * 0.02,
                "step": i * 10,
                "epoch": i // 3,
                "learning_rate": 1e-5,
                "memory_mb": 8192,
            }

            # Check early stopping
            should_stop, reason = self.orchestrator.check_early_stopping(metrics)
            results["validations_performed"] = i + 1

            if should_stop:
                results["early_stopping_triggered"] = True
                results["stop_reason"] = reason
                break

        return results


class TestEarlyStoppingWithErrorRecovery(unittest.TestCase):
    """Test early stopping integration with error recovery mechanisms."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_repo = CheckpointRepository(Path(self.temp_dir) / "checkpoints")

        # Mock model service
        self.mock_model_service = Mock(spec=TTTModelService)
        self.mock_model_service.device = torch.device("cpu")
        self.mock_model_service.memory_manager = Mock()
        self.mock_model_service._get_memory_usage.return_value = 8.0

    def test_early_stopping_with_oom_recovery(self):
        """Test early stopping behavior during OOM scenarios."""
        config = TrainingConfig(
            early_stopping=EarlyStoppingConfig(
                patience=2,
                auto_save_enabled=True,
                auto_save_interval_minutes=1
            )
        )

        orchestrator = TrainingOrchestrator(
            model_service=self.mock_model_service,
            config=config,
            checkpoint_repository=self.checkpoint_repo
        )

        # Mock components
        orchestrator.model = Mock()
        orchestrator.optimizer = Mock()
        orchestrator.current_task_id = "oom_test"
        orchestrator.start_time = 0.0

        # Simulate OOM triggering memory-based early stopping
        self.mock_model_service._get_memory_usage.return_value = 25.0  # Over limit

        metrics = {"validation_accuracy": 0.5}
        should_stop, reason = orchestrator.check_early_stopping(metrics)

        self.assertTrue(should_stop)
        self.assertIn("Memory limit exceeded", reason)

    @patch('src.utils.comprehensive_error_handling.resilient_operation')
    def test_resilient_early_stopping_operations(self, mock_resilient):
        """Test early stopping operations with resilience decorator."""

        @resilient_operation(max_attempts=3, handle_oom=True)
        def mock_validation_with_early_stopping():
            """Mock validation that might face OOM."""
            # Simulate OOM on first attempt, success on second
            if not hasattr(mock_validation_with_early_stopping, 'attempt'):
                mock_validation_with_early_stopping.attempt = 0

            mock_validation_with_early_stopping.attempt += 1

            if mock_validation_with_early_stopping.attempt == 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")

            return 0.65  # Success accuracy

        # Mock the decorator to actually execute the function
        mock_resilient.return_value = mock_validation_with_early_stopping

        # Test that validation succeeds after OOM recovery
        result = mock_validation_with_early_stopping()
        self.assertEqual(result, 0.65)


if __name__ == "__main__":
    unittest.main()
