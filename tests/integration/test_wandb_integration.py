"""Integration tests for Weights & Biases integration.

These tests verify the W&B client functionality including secure credential
handling, experiment tracking, and usage monitoring.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapters.external.wandb_client import (
    BatchedWandBClient, 
    BatchOperationType, 
    WandBClient, 
    WandBConfig, 
    UsageMonitor
)
from src.domain.evaluation_models import (
    ExperimentMetrics,
    ExperimentRun,
    ResourceUsage,
    StrategyType,
    TaskStatus,
)
from src.domain.services.evaluation_service import TaskMetrics, EvaluationResult, PixelAccuracy
from src.utils.secure_credentials import SecureCredentialManager


class TestWandBConfig:
    """Test W&B configuration handling."""
    
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, {
            "WANDB_API_KEY": "test_key",
            "WANDB_PROJECT": "test_project",
            "WANDB_ENTITY": "test_entity",
            "WANDB_MODE": "offline"
        }):
            config = WandBConfig()
            
            # Should get API key from secure storage or env
            assert config.project_name == "test_project"
            assert config.entity == "test_entity"
            assert config.mode == "offline"
    
    def test_config_validation_without_api_key(self):
        """Test configuration validation when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = WandBConfig()
            config.api_key = None  # Simulate missing key
            
            # Should fail validation in online mode
            assert not config.validate()
            
            # Should pass validation in offline mode
            config.mode = "offline"
            assert config.validate()
    
    def test_secure_credential_integration(self):
        """Test integration with secure credential manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create credential manager with temp directory
            cred_manager = SecureCredentialManager(credential_dir=temp_dir)
            
            # Store test API key
            test_key = "test_wandb_api_key_12345"
            assert cred_manager.store_credential("WANDB_API_KEY", test_key)
            
            # Config should retrieve it
            with patch("src.adapters.external.wandb_client.get_credential_manager", return_value=cred_manager):
                config = WandBConfig()
                assert config.api_key == test_key


class TestUsageMonitor:
    """Test W&B usage monitoring functionality."""
    
    def test_usage_calculation(self):
        """Test storage usage calculation."""
        config = WandBConfig()
        monitor = UsageMonitor(config)
        
        # Mock W&B API responses
        with patch("wandb.Api") as mock_api:
            mock_run = MagicMock()
            mock_run.summary = {"_runtime": 3600}  # 1 hour
            
            mock_artifact = MagicMock()
            mock_artifact.size = 1024 * 1024 * 1024  # 1GB
            mock_run.logged_artifacts.return_value = [mock_artifact]
            
            mock_api.return_value.runs.return_value = [mock_run]
            
            with patch("wandb.run", MagicMock()):
                usage_gb = monitor.get_current_usage_gb()
                
                # Should be ~1GB (artifact) + ~3.6MB (logs estimate)
                assert 1.0 <= usage_gb <= 1.1
    
    def test_usage_limits_warning(self):
        """Test usage limit warnings."""
        config = WandBConfig()
        config.storage_warning_threshold = 0.8
        monitor = UsageMonitor(config)
        
        # Mock 85% usage
        with patch.object(monitor, "get_current_usage_gb", return_value=85.0):
            is_ok, warning = monitor.check_usage_limits()
            
            assert is_ok  # Still OK but with warning
            assert warning is not None
            assert "WARNING" in warning
            assert "85.0%" in warning
    
    def test_usage_limits_critical(self):
        """Test critical usage limit."""
        config = WandBConfig()
        config.storage_critical_threshold = 0.95
        monitor = UsageMonitor(config)
        
        # Mock 96% usage
        with patch.object(monitor, "get_current_usage_gb", return_value=96.0):
            is_ok, warning = monitor.check_usage_limits()
            
            assert not is_ok  # Not OK
            assert warning is not None
            assert "CRITICAL" in warning
            assert "96.0%" in warning


class TestWandBClient:
    """Test W&B client functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            client = WandBClient()
            return client
    
    def test_initialization_blocked_by_usage(self, client):
        """Test initialization blocked when usage limit exceeded."""
        # Mock usage check to return critical
        with patch.object(client.usage_monitor, "check_usage_limits", return_value=(False, "Critical")):
            assert not client.initialize()
    
    def test_experiment_tracking(self, client):
        """Test experiment run tracking."""
        # Mock W&B availability and initialization
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("wandb.login"):
                with patch("wandb.init") as mock_init:
                    mock_run = MagicMock()
                    mock_run.id = "test_run_id"
                    mock_init.return_value = mock_run
                    
                    # Initialize client
                    assert client.initialize()
                    
                    # Create experiment
                    experiment = ExperimentRun(
                        run_id="exp_123",
                        experiment_name="Test Experiment",
                        task_ids=["task_1", "task_2"],
                        strategy_config={"strategy": "test"},
                        metrics={},
                        status=TaskStatus.IN_PROGRESS,
                        started_at=datetime.now()
                    )
                    
                    # Start experiment
                    run_id = client.start_experiment(experiment, {"test_param": 123})
                    assert run_id == "test_run_id"
                    
                    # Verify W&B init was called correctly
                    mock_init.assert_called_once()
                    call_args = mock_init.call_args
                    assert call_args.kwargs["project"] == "arc-prize-2025"
                    assert call_args.kwargs["name"] == "Test Experiment"
                    assert "test" in call_args.kwargs["tags"]
    
    def test_evaluation_result_logging(self, client):
        """Test logging evaluation results."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    # Mock current run
                    client._current_run = MagicMock()
                    
                    # Create evaluation result
                    result = EvaluationResult(
                        task_id="task_123",
                        strategy_used="test_strategy",
                        attempts=[
                            TaskMetrics(
                                task_id="task_123",
                                attempt_number=AttemptNumber.FIRST,
                                pixel_accuracy=PixelAccuracy(
                                    accuracy=0.85,
                                    total_pixels=100,
                                    correct_pixels=85,
                                    perfect_match=False
                                ),
                                confidence_score=0.9,
                                processing_time_ms=150.5,
                                error_category=None,
                                error_details={}
                            )
                        ],
                        total_processing_time_ms=150.5
                    )
                    
                    # Log result
                    assert client.log_evaluation_result(result)
                    
                    # Verify metrics were logged
                    mock_log.assert_called()
                    logged_metrics = mock_log.call_args[0][0]
                    assert logged_metrics["task_id"] == "task_123"
                    assert logged_metrics["accuracy"] == 0.85
                    assert logged_metrics["attempt_1/accuracy"] == 0.85
                    assert logged_metrics["attempt_1/confidence"] == 0.9
    
    def test_resource_usage_logging(self, client):
        """Test logging resource usage metrics."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    # Mock current run
                    client._current_run = MagicMock()
                    
                    # Create resource usage
                usage = ResourceUsage(
                    task_id="task_123",
                    strategy_type=StrategyType.DIRECT_SOLVE,
                    cpu_seconds=12.5,
                    memory_mb=256.0,
                    gpu_memory_mb=1024.0,
                    api_calls={"openai": 3, "anthropic": 2},
                    total_tokens=1500,
                    estimated_cost=0.05,
                    timestamp=datetime.now()
                )
                
                # Log usage
                assert client.log_resource_usage(usage)
                
                # Verify metrics were logged
                mock_log.assert_called()
                logged_metrics = mock_log.call_args[0][0]
                assert logged_metrics["resource/cpu_seconds"] == 12.5
                assert logged_metrics["resource/memory_mb"] == 256.0
                assert logged_metrics["resource/gpu_memory_mb"] == 1024.0
                assert logged_metrics["resource/api_calls/openai"] == 3
    
    def test_experiment_summary_logging(self, client):
        """Test logging experiment summary metrics."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("wandb.run") as mock_run:
                # Mock current run
                client._current_run = MagicMock()
                mock_run.summary = MagicMock()
                
                # Create experiment metrics
                metrics = ExperimentMetrics(
                    experiment_id="exp_123",
                    total_tasks=100,
                    successful_tasks=85,
                    failed_tasks=15,
                    average_accuracy=0.82,
                    perfect_matches=12,
                    total_processing_time_ms=45000,
                    total_resource_cost=2.5,
                    strategy_performance={
                        "direct_solve": {
                            "tasks_evaluated": 50,
                            "average_accuracy": 0.78,
                            "perfect_matches": 5
                        },
                        "pattern_match": {
                            "tasks_evaluated": 50,
                            "average_accuracy": 0.86,
                            "perfect_matches": 7
                        }
                    },
                    error_distribution={
                        "shape_mismatch": 8,
                        "color_error": 5,
                        "pattern_error": 2
                    }
                )
                
                # Log summary
                assert client.log_experiment_summary(metrics)
                
                # Verify summary was updated
                summary_update = mock_run.summary.update.call_args[0][0]
                assert summary_update["total_tasks"] == 100
                assert summary_update["average_accuracy"] == 0.82
                assert summary_update["success_rate"] == 0.85
    
    def test_artifact_saving_with_usage_check(self, client):
        """Test artifact saving with usage limit check."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("wandb.Artifact") as mock_artifact_class:
                with patch("wandb.log_artifact") as mock_log_artifact:
                    # Mock current run
                    client._current_run = MagicMock()
                    
                    # Mock usage check - at limit
                    with patch.object(
                        client.usage_monitor, 
                        "check_usage_limits", 
                        return_value=(False, "Storage limit exceeded")
                    ):
                        # Should fail to save
                        assert not client.save_model_artifact(
                            "/path/to/model",
                            "test_model",
                            "model"
                        )
                        mock_log_artifact.assert_not_called()
                    
                    # Mock usage check - under limit
                    with patch.object(
                        client.usage_monitor,
                        "check_usage_limits",
                        return_value=(True, None)
                    ):
                        # Should succeed
                        mock_artifact = MagicMock()
                        mock_artifact_class.return_value = mock_artifact
                        
                        assert client.save_model_artifact(
                            "/path/to/model",
                            "test_model",
                            "model"
                        )
                        mock_log_artifact.assert_called_once_with(mock_artifact)
    
    def test_end_experiment_with_alerts(self, client):
        """Test ending experiment with usage alerts."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("wandb.finish") as mock_finish:
                with patch("wandb.alert") as mock_alert:
                    # Mock current run
                    client._current_run = MagicMock()
                    
                    # Mock usage warning
                    with patch.object(
                        client.usage_monitor,
                        "check_usage_limits",
                        return_value=(True, "WARNING: At 85% capacity")
                    ):
                        assert client.end_experiment()
                        
                        # Should send alert
                        mock_alert.assert_called_once()
                        alert_args = mock_alert.call_args
                        assert "Storage Usage Warning" in alert_args.kwargs["title"]
                        
                        # Should finish run
                        mock_finish.assert_called_once()
                        assert client._current_run is None


class TestBatchedWandBClient:
    """Test BatchedWandBClient functionality."""
    
    @pytest.fixture
    def batched_client(self):
        """Create a test batched client."""
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            client = BatchedWandBClient(
                batch_size=3,
                flush_interval_seconds=0.1,
                enable_batching=True,
            )
            return client
    
    @pytest.mark.asyncio
    async def test_batch_context_manager(self, batched_client):
        """Test async context manager functionality."""
        assert not batched_client._running
        
        async with batched_client as client:
            assert client._running
            assert client._flush_task is not None
        
        assert not client._running
        assert client._flush_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_results(self, batched_client):
        """Test batching of evaluation results."""
        from src.domain.services.evaluation_service import TaskMetrics, EvaluationResult, PixelAccuracy, AttemptNumber
        
        # Mock W&B
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    # Mock current run
                    batched_client._current_run = MagicMock()
                    
                    # Start batch processing
                    await batched_client.start_batch_processing()
                    
                    # Create multiple evaluation results
                    results = []
                    for i in range(5):
                        result = EvaluationResult(
                            task_id=f"task_{i}",
                            strategy_used="test_strategy",
                            attempts=[
                                TaskMetrics(
                                    task_id=f"task_{i}",
                                    attempt_number=AttemptNumber.FIRST,
                                    pixel_accuracy=PixelAccuracy(
                                        accuracy=0.8 + (i * 0.02),
                                        total_pixels=100,
                                        correct_pixels=80 + (i * 2),
                                        perfect_match=i == 4
                                    ),
                                    confidence_score=0.9,
                                    processing_time_ms=100 + i * 10,
                                    error_category=None,
                                    error_details=None
                                )
                            ],
                            total_processing_time_ms=100 + i * 10
                        )
                        results.append(result)
                    
                    # Log results async
                    for result in results:
                        success = await batched_client.log_evaluation_result_async(result)
                        assert success
                    
                    # Wait for batch processing
                    await asyncio.sleep(0.2)  # Allow batch to flush
                    
                    # Verify batched logging occurred
                    assert mock_log.call_count >= 1  # At least one batch was logged
                    
                    # Stop batch processing
                    await batched_client.stop_batch_processing()
    
    @pytest.mark.asyncio
    async def test_batch_resource_usage(self, batched_client):
        """Test batching of resource usage metrics."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    batched_client._current_run = MagicMock()
                    
                    await batched_client.start_batch_processing()
                    
                    # Create multiple resource usage records
                    for i in range(4):
                        usage = ResourceUsage(
                            task_id=f"task_{i}",
                            strategy_type=StrategyType.DIRECT_SOLVE,
                            cpu_seconds=10.0 + i,
                            memory_mb=256.0 + i * 64,
                            gpu_memory_mb=1024.0,
                            api_calls={"openai": i + 1, "anthropic": i},
                            total_tokens=1000 + i * 100,
                            estimated_cost=0.01 + i * 0.005,
                            timestamp=datetime.now()
                        )
                        
                        success = await batched_client.log_resource_usage_async(usage)
                        assert success
                    
                    # Force flush
                    await batched_client._flush_batches(force=True)
                    
                    # Verify batch was logged
                    assert mock_log.call_count >= 1
                    
                    await batched_client.stop_batch_processing()
    
    @pytest.mark.asyncio
    async def test_batch_custom_metrics(self, batched_client):
        """Test batching of custom metrics."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    batched_client._current_run = MagicMock()
                    
                    await batched_client.start_batch_processing()
                    
                    # Log custom metrics
                    for i in range(3):
                        metrics = {
                            f"custom_metric_{i}": i * 10,
                            f"accuracy_{i}": 0.8 + i * 0.1,
                            "timestamp": datetime.now().isoformat(),
                        }
                        
                        success = await batched_client.log_custom_metrics_async(metrics)
                        assert success
                    
                    # Force flush
                    await batched_client._flush_batches(force=True)
                    
                    # Verify batch was logged with prefixes
                    assert mock_log.called
                    logged_data = mock_log.call_args[0][0]
                    
                    # Check that batch prefixes were applied
                    batch_keys = [k for k in logged_data.keys() if k.startswith("batch_")]
                    assert len(batch_keys) > 0
                    
                    await batched_client.stop_batch_processing()
    
    @pytest.mark.asyncio
    async def test_batch_retry_logic(self, batched_client):
        """Test retry logic for failed batch operations."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            batched_client._current_run = MagicMock()
            
            # Mock wandb.log to fail then succeed
            call_count = 0
            def mock_log_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Simulated failure")
                return None
            
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    mock_log.side_effect = mock_log_side_effect
                    await batched_client.start_batch_processing()
                    
                    # Add a custom metric that will initially fail
                    success = await batched_client.log_custom_metrics_async({"test": 123})
                    assert success  # Queued successfully
                    
                    # Force flush - this will fail and add to retry queue
                    await batched_client._flush_batches(force=True)
                    
                    # Check retry queue has items
                    queue_sizes = batched_client.get_queue_sizes()
                    assert queue_sizes["retry_queue"] > 0
                    
                    # Process retry queue - this should succeed
                    await batched_client._process_retry_queue()
                    
                    # Check metrics
                    metrics = batched_client.get_batch_metrics()
                    assert metrics["retry_operations"] > 0
                    
                    await batched_client.stop_batch_processing()
    
    @pytest.mark.asyncio
    async def test_batch_metrics_tracking(self, batched_client):
        """Test batch processing metrics tracking."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    batched_client._current_run = MagicMock()
                    
                    await batched_client.start_batch_processing()
                    
                    # Process several operations
                    for i in range(5):
                        await batched_client.log_custom_metrics_async({"metric": i})
                    
                    # Force flush
                    await batched_client._flush_batches(force=True)
                    
                    # Check metrics
                    metrics = batched_client.get_batch_metrics()
                    assert metrics["total_operations"] >= 5
                    assert metrics["total_batches"] >= 1
                    assert metrics["success_rate"] > 0
                    assert metrics["average_batch_size"] > 0
                    assert metrics["average_flush_time_ms"] >= 0
                    
                    # Check queue sizes
                    queue_sizes = batched_client.get_queue_sizes()
                    assert "batch_queue" in queue_sizes
                    assert "retry_queue" in queue_sizes
                    
                    await batched_client.stop_batch_processing()
    
    @pytest.mark.asyncio
    async def test_batch_disabled_fallback(self):
        """Test fallback to sync methods when batching is disabled."""
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
            client = BatchedWandBClient(enable_batching=False)
            
            with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
                with patch.object(client, "log_evaluation_result") as mock_sync_log:
                    mock_sync_log.return_value = True
                    
                    # Mock evaluation result
                    from src.domain.services.evaluation_service import TaskMetrics, EvaluationResult, PixelAccuracy, AttemptNumber
                    
                    result = EvaluationResult(
                        task_id="test_task",
                        strategy_used="test_strategy",
                        attempts=[
                            TaskMetrics(
                                task_id="test_task",
                                attempt_number=AttemptNumber.FIRST,
                                pixel_accuracy=PixelAccuracy(
                                    accuracy=0.85,
                                    total_pixels=100,
                                    correct_pixels=85,
                                    perfect_match=False
                                ),
                                confidence_score=0.9,
                                processing_time_ms=150,
                                error_category=None,
                                error_details=None
                            )
                        ],
                        total_processing_time_ms=150
                    )
                    
                    # Should use sync method when batching disabled
                    success = await client.log_evaluation_result_async(result)
                    assert success
                    mock_sync_log.assert_called_once_with(result)
    
    def test_batch_operation_types(self):
        """Test batch operation type enum."""
        assert BatchOperationType.EVALUATION_RESULT.value == "evaluation_result"
        assert BatchOperationType.RESOURCE_USAGE.value == "resource_usage"
        assert BatchOperationType.EXPERIMENT_SUMMARY.value == "experiment_summary"
        assert BatchOperationType.CUSTOM_METRICS.value == "custom_metrics"


@pytest.mark.integration
class TestWandBIntegrationE2E:
    """End-to-end integration tests with W&B (requires W&B to be installed)."""
    
    @pytest.mark.skipif(
        not os.environ.get("WANDB_API_KEY") or os.environ.get("CI"),
        reason="Requires W&B API key and not suitable for CI"
    )
    def test_full_experiment_workflow(self):
        """Test complete experiment workflow with real W&B."""
        # This test would actually connect to W&B
        # Skipped in CI to avoid external dependencies
        pass
    
    @pytest.mark.skipif(
        not os.environ.get("WANDB_API_KEY") or os.environ.get("CI"),
        reason="Requires W&B API key and not suitable for CI"
    )
    @pytest.mark.asyncio
    async def test_batched_experiment_workflow(self):
        """Test complete batched experiment workflow with real W&B."""
        # This test would actually connect to W&B with batch processing
        # Skipped in CI to avoid external dependencies
        pass