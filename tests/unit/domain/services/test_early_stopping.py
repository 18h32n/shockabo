"""
Unit tests for early stopping functionality in Story 1.5 Task 5.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.domain.services.training_orchestrator import (
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingOrchestrator,
)
from src.domain.services.ttt_service import TTTModelService
from src.utils.early_stopping_utils import EarlyStoppingConfigManager, EarlyStoppingMonitor


class TestEarlyStoppingConfig(unittest.TestCase):
    """Test early stopping configuration."""

    def test_default_config(self):
        """Test default early stopping configuration."""
        config = EarlyStoppingConfig()

        self.assertEqual(config.patience, 5)
        self.assertEqual(config.min_delta, 0.01)
        self.assertTrue(config.restore_best_weights)
        self.assertEqual(config.monitor_metric, "validation_accuracy")
        self.assertEqual(config.mode, "max")
        self.assertIsNone(config.baseline)
        self.assertTrue(config.verbose)
        self.assertTrue(config.auto_save_enabled)
        self.assertEqual(config.auto_save_interval_minutes, 10)
        self.assertTrue(config.auto_save_on_improvement)
        self.assertTrue(config.auto_resume_enabled)
        self.assertTrue(config.resume_from_best)
        self.assertEqual(config.resume_threshold_hours, 0.5)

    def test_custom_config(self):
        """Test custom early stopping configuration."""
        config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.02,
            monitor_metric="validation_loss",
            mode="min",
            baseline=0.5,
            auto_save_interval_minutes=15
        )

        self.assertEqual(config.patience, 3)
        self.assertEqual(config.min_delta, 0.02)
        self.assertEqual(config.monitor_metric, "validation_loss")
        self.assertEqual(config.mode, "min")
        self.assertEqual(config.baseline, 0.5)
        self.assertEqual(config.auto_save_interval_minutes, 15)


class TestTrainingOrchestratorEarlyStopping(unittest.TestCase):
    """Test early stopping in training orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock model service
        self.mock_model_service = Mock(spec=TTTModelService)
        self.mock_model_service.device = torch.device("cpu")
        self.mock_model_service.memory_manager = Mock()
        self.mock_model_service.memory_manager.get_memory_usage.return_value = {"usage_percentage": 50.0}
        self.mock_model_service._get_memory_usage.return_value = 8.0  # 8GB

        # Mock checkpoint repository
        self.mock_checkpoint_repo = Mock(spec=CheckpointRepository)

        # Create test config
        early_stopping_config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            auto_save_enabled=True,
            auto_save_interval_minutes=5
        )

        config = TrainingConfig(
            early_stopping=early_stopping_config,
            max_training_time=300,  # 5 minutes
            target_accuracy=0.8,
            memory_limit_mb=16384  # 16GB
        )

        # Create orchestrator
        self.orchestrator = TrainingOrchestrator(
            model_service=self.mock_model_service,
            config=config,
            checkpoint_repository=self.mock_checkpoint_repo
        )

        # Mock time
        self.orchestrator.start_time = 0.0
        self.orchestrator.current_task_id = "test_task_001"

    def test_check_early_stopping_patience(self):
        """Test early stopping based on patience."""
        # Simulate no improvement for multiple validations
        metrics = {
            "validation_accuracy": 0.5,
            "loss": 0.5,
            "step": 10,
            "epoch": 1,
            "learning_rate": 1e-5,
            "memory_mb": 8192
        }

        # First few checks should not trigger early stopping
        for i in range(3):
            should_stop, reason = self.orchestrator.check_early_stopping(metrics)
            self.assertFalse(should_stop, f"Should not stop at validation {i+1}")

        # After patience is exceeded, should trigger early stopping
        should_stop, reason = self.orchestrator.check_early_stopping(metrics)
        self.assertTrue(should_stop)
        self.assertIn("No improvement for 3 validations", reason)

    def test_check_early_stopping_improvement(self):
        """Test early stopping with improvements."""
        # First validation
        metrics = {
            "validation_accuracy": 0.5,
            "loss": 0.5,
            "step": 10,
            "epoch": 1,
            "learning_rate": 1e-5,
            "memory_mb": 8192
        }
        should_stop, _ = self.orchestrator.check_early_stopping(metrics)
        self.assertFalse(should_stop)

        # Improvement - should reset patience counter
        metrics["validation_accuracy"] = 0.52
        metrics["step"] = 20
        should_stop, _ = self.orchestrator.check_early_stopping(metrics)
        self.assertFalse(should_stop)

        # Another improvement
        metrics["validation_accuracy"] = 0.54
        metrics["step"] = 30
        should_stop, _ = self.orchestrator.check_early_stopping(metrics)
        self.assertFalse(should_stop)

        # No improvement for patience period
        metrics["validation_accuracy"] = 0.535  # Small decrease
        for i in range(4):  # Exceed patience
            metrics["step"] = 40 + i * 10
            should_stop, reason = self.orchestrator.check_early_stopping(metrics)

        self.assertTrue(should_stop)
        self.assertIn("No improvement", reason)

    def test_check_early_stopping_time_limit(self):
        """Test early stopping due to time limit."""
        # Mock time to exceed limit
        with patch('time.time', return_value=400.0):  # 400 seconds from start
            metrics = {"validation_accuracy": 0.5}
            should_stop, reason = self.orchestrator.check_early_stopping(metrics)

            self.assertTrue(should_stop)
            self.assertEqual(reason, "Time limit exceeded")

    def test_check_early_stopping_memory_limit(self):
        """Test early stopping due to memory limit."""
        # Mock memory usage exceeding limit
        self.mock_model_service._get_memory_usage.return_value = 20.0  # 20GB > 16GB limit

        metrics = {"validation_accuracy": 0.5}
        should_stop, reason = self.orchestrator.check_early_stopping(metrics)

        self.assertTrue(should_stop)
        self.assertIn("Memory limit exceeded", reason)

    def test_check_early_stopping_target_achieved(self):
        """Test early stopping when target accuracy is achieved."""
        metrics = {"validation_accuracy": 0.85}  # Exceeds target of 0.8
        should_stop, reason = self.orchestrator.check_early_stopping(metrics)

        self.assertTrue(should_stop)
        self.assertEqual(reason, "Target accuracy achieved")

    def test_check_early_stopping_baseline(self):
        """Test early stopping when performance falls below baseline."""
        # Set baseline in config
        self.orchestrator.config.early_stopping.baseline = 0.4

        metrics = {"validation_accuracy": 0.35}  # Below baseline
        should_stop, reason = self.orchestrator.check_early_stopping(metrics)

        self.assertTrue(should_stop)
        self.assertIn("Performance below baseline", reason)

    @patch('time.time')
    def test_auto_save_on_improvement(self, mock_time):
        """Test auto-save functionality on improvement."""
        mock_time.return_value = 100.0

        # Mock orchestrator components
        self.orchestrator.model = Mock()
        self.orchestrator.optimizer = Mock()
        self.orchestrator.scheduler = Mock()
        self.orchestrator.lora_adapter = Mock()
        self.orchestrator.scaler = None

        # Mock methods
        self.orchestrator.model.state_dict.return_value = {"param": "value"}
        self.orchestrator.optimizer.state_dict.return_value = {"opt_param": "value"}
        self.orchestrator.scheduler.state_dict.return_value = {"sched_param": "value"}
        self.orchestrator.lora_adapter.get_adapter_state.return_value = {"lora_param": "value"}

        # Configure checkpoint repo mock
        mock_metadata = Mock()
        self.mock_checkpoint_repo.save_checkpoint.return_value = mock_metadata

        # Test improvement triggering auto-save
        metrics = {
            "validation_accuracy": 0.6,  # Improvement
            "loss": 0.4,
            "step": 10,
            "epoch": 1,
            "learning_rate": 1e-5,
            "memory_mb": 8192
        }

        should_stop, _ = self.orchestrator.check_early_stopping(metrics)
        self.assertFalse(should_stop)

        # Verify save_checkpoint was called
        self.mock_checkpoint_repo.save_checkpoint.assert_called_once()

        # Check call arguments
        call_args = self.mock_checkpoint_repo.save_checkpoint.call_args
        self.assertEqual(call_args[1]["task_id"], "test_task_001")
        self.assertIn("improvement", call_args[1]["tags"])

    @patch('time.time')
    def test_auto_save_interval(self, mock_time):
        """Test auto-save based on time interval."""
        # Set up time progression
        mock_time.side_effect = [0, 300, 400]  # Start, first call, second call (400s = 6.67min > 5min interval)

        # Mock orchestrator components
        self.orchestrator.model = Mock()
        self.orchestrator.optimizer = Mock()
        self.orchestrator.scheduler = Mock()
        self.orchestrator.lora_adapter = Mock()
        self.orchestrator.last_auto_save_time = None

        # Mock methods
        self.orchestrator.model.state_dict.return_value = {"param": "value"}
        self.orchestrator.optimizer.state_dict.return_value = {"opt_param": "value"}
        self.orchestrator.scheduler.state_dict.return_value = {"sched_param": "value"}
        self.orchestrator.lora_adapter.get_adapter_state.return_value = {"lora_param": "value"}

        metrics = {
            "validation_accuracy": 0.5,  # No improvement
            "loss": 0.5,
            "step": 10,
            "epoch": 1,
            "learning_rate": 1e-5,
            "memory_mb": 8192
        }

        # First call - should trigger interval-based save
        should_stop, _ = self.orchestrator.check_early_stopping(metrics)
        self.assertFalse(should_stop)

        # Verify interval-based save was called
        self.mock_checkpoint_repo.save_checkpoint.assert_called_once()
        call_args = self.mock_checkpoint_repo.save_checkpoint.call_args
        self.assertIn("interval", call_args[1]["tags"])


class TestEarlyStoppingMonitor(unittest.TestCase):
    """Test early stopping monitor utility."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            monitor_metric="validation_accuracy",
            mode="max"
        )
        self.monitor = EarlyStoppingMonitor(self.config)

    def test_track_training_session_improvement(self):
        """Test tracking training session with improvements."""
        session_id = "test_session"

        # First validation - establishes baseline
        should_stop, reason = self.monitor.track_training_session(
            session_id, epoch=0, step=10, metric_value=0.5, all_metrics={}
        )
        self.assertFalse(should_stop)

        # Improvement
        should_stop, reason = self.monitor.track_training_session(
            session_id, epoch=0, step=20, metric_value=0.52, all_metrics={}
        )
        self.assertFalse(should_stop)

        # Another improvement
        should_stop, reason = self.monitor.track_training_session(
            session_id, epoch=1, step=30, metric_value=0.55, all_metrics={}
        )
        self.assertFalse(should_stop)

        # Get session summary
        summary = self.monitor.get_session_summary(session_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["best_value"], 0.55)
        self.assertEqual(summary["improvements_count"], 3)  # Including initial value
        self.assertFalse(summary["early_stopped"])

    def test_track_training_session_early_stopping(self):
        """Test early stopping trigger in monitor."""
        session_id = "test_session"

        # Establish baseline
        self.monitor.track_training_session(
            session_id, epoch=0, step=10, metric_value=0.5, all_metrics={}
        )

        # No improvements for patience period
        for i in range(4):  # Exceed patience of 3
            should_stop, reason = self.monitor.track_training_session(
                session_id, epoch=0, step=20+i*10, metric_value=0.49, all_metrics={}
            )

        self.assertTrue(should_stop)
        self.assertIn("Patience exceeded", reason)

        # Check session state
        summary = self.monitor.get_session_summary(session_id)
        self.assertTrue(summary["early_stopped"])
        self.assertEqual(summary["trigger_reason"], "Patience exceeded (3)")

    def test_analyze_training_efficiency(self):
        """Test training efficiency analysis."""
        session_id = "test_session"

        # Simulate training with some improvements
        values = [0.3, 0.4, 0.45, 0.46, 0.46, 0.45]  # Early improvements, then plateau

        for i, value in enumerate(values):
            should_stop, _ = self.monitor.track_training_session(
                session_id, epoch=i//2, step=i*10, metric_value=value, all_metrics={}
            )
            if should_stop:
                break

        # Analyze efficiency
        analysis = self.monitor.analyze_training_efficiency(session_id)

        self.assertIsInstance(analysis, dict)
        self.assertEqual(analysis["session_id"], session_id)
        self.assertGreater(analysis["total_validation_points"], 0)
        self.assertGreaterEqual(analysis["improvement_ratio"], 0)
        self.assertIn("recommendations", analysis)
        self.assertIsInstance(analysis["recommendations"], list)


class TestEarlyStoppingConfigManager(unittest.TestCase):
    """Test early stopping configuration manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = EarlyStoppingConfigManager()

    def test_get_predefined_config(self):
        """Test getting predefined configurations."""
        # Test conservative config
        conservative = self.config_manager.get_config("conservative")
        self.assertIsNotNone(conservative)
        self.assertEqual(conservative.patience, 10)

        # Test aggressive config
        aggressive = self.config_manager.get_config("aggressive")
        self.assertIsNotNone(aggressive)
        self.assertEqual(aggressive.patience, 3)

        # Test non-existent config
        invalid = self.config_manager.get_config("non_existent")
        self.assertIsNone(invalid)

    def test_create_adaptive_config(self):
        """Test adaptive configuration creation."""
        # Test 8B model config
        config_8b = self.config_manager.create_adaptive_config("8B", 45, 24)
        self.assertIsInstance(config_8b, EarlyStoppingConfig)
        self.assertLessEqual(config_8b.patience, 6)  # Should be reduced for 8B

        # Test 1B model config
        config_1b = self.config_manager.create_adaptive_config("1B", 60, 16)
        self.assertIsInstance(config_1b, EarlyStoppingConfig)

        # Test memory-constrained config
        config_limited = self.config_manager.create_adaptive_config("8B", 20, 12)
        self.assertIsInstance(config_limited, EarlyStoppingConfig)
        self.assertLess(config_limited.patience, config_8b.patience)  # More aggressive

    def test_save_and_load_config(self):
        """Test saving and loading configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Create and save config
            original_config = EarlyStoppingConfig(patience=7, min_delta=0.05)
            self.config_manager.save_config("test", original_config, config_path)

            # Load config
            loaded_config = self.config_manager.load_config(config_path)

            self.assertIsNotNone(loaded_config)
            self.assertEqual(loaded_config.patience, 7)
            self.assertEqual(loaded_config.min_delta, 0.05)


class TestEarlyStoppingIntegration(unittest.TestCase):
    """Integration tests for early stopping functionality."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_repo = CheckpointRepository(Path(self.temp_dir) / "checkpoints")

        # Mock model service
        self.mock_model_service = Mock(spec=TTTModelService)
        self.mock_model_service.device = torch.device("cpu")
        self.mock_model_service.memory_manager = Mock()
        self.mock_model_service._get_memory_usage.return_value = 8.0

        # Create test config
        early_stopping_config = EarlyStoppingConfig(
            patience=2,
            min_delta=0.01,
            auto_save_enabled=True,
            auto_save_interval_minutes=1  # Very short for testing
        )

        config = TrainingConfig(early_stopping=early_stopping_config)

        # Create orchestrator with real checkpoint repo
        self.orchestrator = TrainingOrchestrator(
            model_service=self.mock_model_service,
            config=config,
            checkpoint_repository=self.checkpoint_repo
        )

    def test_integration_with_checkpoint_saving(self):
        """Test integration between early stopping and checkpoint saving."""
        # Set up orchestrator state
        self.orchestrator.current_task_id = "integration_test"
        self.orchestrator.start_time = 0.0

        # Mock model components
        self.orchestrator.model = Mock()
        self.orchestrator.optimizer = Mock()
        self.orchestrator.scheduler = Mock()
        self.orchestrator.lora_adapter = Mock()

        self.orchestrator.model.state_dict.return_value = {"param": "value"}
        self.orchestrator.optimizer.state_dict.return_value = {"opt": "value"}
        self.orchestrator.scheduler.state_dict.return_value = {"sched": "value"}
        self.orchestrator.lora_adapter.get_adapter_state.return_value = {"lora": "value"}

        # Simulate training with improvement (should trigger auto-save)
        with patch('time.time', return_value=100.0):
            metrics = {
                "validation_accuracy": 0.6,
                "loss": 0.4,
                "step": 10,
                "epoch": 1,
                "learning_rate": 1e-5,
                "memory_mb": 8192
            }

            should_stop, _ = self.orchestrator.check_early_stopping(metrics)
            self.assertFalse(should_stop)

        # Verify checkpoint was saved
        checkpoints = self.checkpoint_repo.list_checkpoints(task_id="integration_test")
        self.assertGreater(len(checkpoints), 0)

        # Verify checkpoint contains correct data
        checkpoint = checkpoints[0]
        self.assertEqual(checkpoint.task_id, "integration_test")
        self.assertIn("improvement", checkpoint.tags)


if __name__ == "__main__":
    unittest.main()
