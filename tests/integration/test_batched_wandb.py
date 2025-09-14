"""Tests for BatchedWandBClient async batch processing functionality."""

import asyncio
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.external.wandb_client import (
    BatchedWandBClient,
    BatchOperationType,
    get_batched_wandb_client,
)
from src.domain.evaluation_models import (
    ResourceUsage,
    StrategyType,
)
from src.domain.services.evaluation_service import (
    AttemptNumber,
    EvaluationResult,
    PixelAccuracy,
    TaskMetrics,
)


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

    def test_batch_operation_types(self):
        """Test batch operation type enum."""
        assert BatchOperationType.EVALUATION_RESULT.value == "evaluation_result"
        assert BatchOperationType.RESOURCE_USAGE.value == "resource_usage"
        assert BatchOperationType.EXPERIMENT_SUMMARY.value == "experiment_summary"
        assert BatchOperationType.CUSTOM_METRICS.value == "custom_metrics"

    @pytest.mark.asyncio
    async def test_batch_evaluation_results(self, batched_client):
        """Test batching of evaluation results."""
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
                                    error_details={}
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
                                error_details={}
                            )
                        ],
                        total_processing_time_ms=150
                    )

                    # Should use sync method when batching disabled
                    success = await client.log_evaluation_result_async(result)
                    assert success
                    mock_sync_log.assert_called_once_with(result)

    def test_get_batched_client_singleton(self):
        """Test that get_batched_wandb_client returns the same instance."""
        client1 = get_batched_wandb_client(batch_size=10)
        client2 = get_batched_wandb_client(batch_size=20)  # Different params

        # Should return the same instance (singleton pattern)
        assert client1 is client2

        # Should use original batch_size from first call
        assert client1.batch_size == 10

    def test_batch_configuration(self):
        """Test batch configuration parameters."""
        client = BatchedWandBClient(
            batch_size=100,
            flush_interval_seconds=10.0,
            max_retry_attempts=5,
            retry_delay_seconds=2.0,
            enable_batching=False,
        )

        assert client.batch_size == 100
        assert client.flush_interval_seconds == 10.0
        assert client.max_retry_attempts == 5
        assert client.retry_delay_seconds == 2.0
        assert client.enable_batching == False

        # Verify initial state
        assert not client._running
        assert len(client._batch_queue) == 0
        assert len(client._retry_queue) == 0

    @pytest.mark.asyncio
    async def test_performance_improvement_simulation(self, batched_client):
        """Test that demonstrates potential performance improvements."""
        with patch("src.adapters.external.wandb_client.WANDB_AVAILABLE", True):
            with patch("sys.modules", {"wandb": MagicMock()}):
                import wandb
                with patch.object(wandb, "log") as mock_log:
                    batched_client._current_run = MagicMock()

                    # Simulate logging many metrics individually (sync approach)
                    sync_calls = []
                    for i in range(100):
                        sync_calls.append({"metric": i, "value": i * 0.1})

                    # Now use batched approach
                    await batched_client.start_batch_processing()

                    start_time = asyncio.get_event_loop().time()
                    for metrics in sync_calls:
                        await batched_client.log_custom_metrics_async(metrics)

                    # Force final flush
                    await batched_client._flush_batches(force=True)
                    end_time = asyncio.get_event_loop().time()

                    # Get performance metrics
                    perf_metrics = batched_client.get_batch_metrics()

                    # Verify batching effectiveness
                    assert perf_metrics["total_operations"] == 100
                    assert perf_metrics["total_batches"] < 100  # Should be batched
                    assert perf_metrics["success_rate"] == 1.0

                    # Should have made fewer wandb.log calls than individual operations
                    assert mock_log.call_count < 100

                    processing_time = end_time - start_time
                    print(f"Batched processing time: {processing_time:.3f}s")
                    print(f"Average batch size: {perf_metrics['average_batch_size']}")
                    print(f"Total API calls made: {mock_log.call_count}")

                    await batched_client.stop_batch_processing()
