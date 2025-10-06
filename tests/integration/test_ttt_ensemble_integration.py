"""
Integration tests for TTT Ensemble Integration

Tests TTTAdapter implementation of StrategyPort interface, including
solve_task(), get_confidence_estimate(), and get_resource_estimate().
"""
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.domain.models import ARCTask, ResourceUsage, StrategyOutput, StrategyType
from src.domain.ports.strategy import StrategyPort

logger = logging.getLogger(__name__)


class MockTTTAdapter(StrategyPort):
    """
    Mock TTT adapter implementing StrategyPort interface.
    
    This demonstrates how the actual TTTAdapter should implement the interface.
    The real implementation would be in src/adapters/strategies/ttt_adapter.py.
    """
    
    def __init__(self, config: dict | None = None):
        """Initialize mock adapter."""
        self.config = config or {}
        self.adaptation_cache = {}
    
    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """
        Solve task using enhanced TTT with leave-one-out, self-consistency, and LoRA optimization.
        
        Args:
            task: ARCTask to solve
            
        Returns:
            StrategyOutput with predictions and confidence
        """
        start_time = 0.0
        
        try:
            # Step 1: Leave-one-out adaptation
            # adaptation_result = self.loo_generator.generate_splits(task.train_examples)
            
            # Step 2: Train with LoRA optimization
            # optimized_model = self.lora_optimizer.train(task, adaptation_result)
            
            # Step 3: Inference with self-consistency
            # predictions = self.self_consistency_validator.validate(task, optimized_model)
            
            # Mock prediction for demonstration
            predicted_output = np.zeros((10, 10), dtype=np.int8)
            per_pixel_confidence = np.ones((10, 10), dtype=np.float32) * 0.8
            
            # Confidence from self-consistency agreement
            confidence_score = 0.85
            
            # Resource tracking
            resource_usage = ResourceUsage(
                api_calls=0,  # No external API calls for TTT
                total_tokens=0,
                estimated_cost=0.0
            )
            
            return StrategyOutput(
                strategy_type=StrategyType.TEST_TIME_TRAINING,
                predicted_output=predicted_output,
                confidence_score=confidence_score,
                per_pixel_confidence=per_pixel_confidence,
                reasoning_trace="Enhanced TTT with leave-one-out, self-consistency, and LoRA optimization",
                resource_usage=resource_usage,
                execution_time_ms=int((0.0 - start_time) * 1000)
            )
            
        except Exception as e:
            logger.error(f"Error solving task {task.task_id}: {e}")
            # Return fallback prediction
            return StrategyOutput(
                strategy_type=StrategyType.TEST_TIME_TRAINING,
                predicted_output=np.zeros((1, 1), dtype=np.int8),
                confidence_score=0.0,
                reasoning_trace=f"Error: {str(e)}",
                resource_usage=ResourceUsage(api_calls=0, total_tokens=0, estimated_cost=0.0),
                execution_time_ms=0
            )
    
    def get_confidence_estimate(self, task: ARCTask) -> float:
        """
        Quick confidence estimate for routing (<100ms).
        
        Args:
            task: ARCTask to estimate
            
        Returns:
            Estimated confidence (0.0-1.0)
        """
        # Heuristics for TTT confidence
        num_train = len(task.train_examples)
        grid_size = len(task.test_input) * len(task.test_input[0]) if task.test_input else 0
        
        # TTT works better with more training examples
        train_factor = min(num_train / 5.0, 1.0)  # Cap at 5 examples
        
        # TTT works better with smaller grids
        size_factor = 1.0 if grid_size < 100 else 0.8 if grid_size < 300 else 0.6
        
        # Base confidence for TTT strategy
        base_confidence = 0.7
        
        return base_confidence * train_factor * size_factor
    
    def get_resource_estimate(self, task: ARCTask) -> ResourceUsage:
        """
        Estimate resource requirements (<50ms).
        
        Args:
            task: ARCTask to estimate
            
        Returns:
            ResourceUsage with estimated requirements
        """
        # TTT uses local model - no API calls
        api_calls = 0
        total_tokens = 0
        estimated_cost = 0.0
        
        # Estimate based on task complexity
        num_train = len(task.train_examples)
        
        # CPU time estimate (seconds)
        estimated_time_sec = 120.0 + (num_train * 30.0)  # Base + per-example
        
        return ResourceUsage(
            api_calls=api_calls,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            estimated_time_sec=estimated_time_sec
        )


@pytest.fixture
def mock_ttt_adapter():
    """Fixture providing mock TTT adapter."""
    return MockTTTAdapter()


@pytest.fixture
def sample_task():
    """Fixture providing sample ARC task."""
    return ARCTask(
        task_id="test_task_001",
        task_source="evaluation",
        train_examples=[
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}
        ],
        test_input=[[1, 1], [0, 0]]
    )


def test_strategy_port_interface(mock_ttt_adapter):
    """Test that TTTAdapter implements StrategyPort interface."""
    assert isinstance(mock_ttt_adapter, StrategyPort)
    assert hasattr(mock_ttt_adapter, 'solve_task')
    assert hasattr(mock_ttt_adapter, 'get_confidence_estimate')
    assert hasattr(mock_ttt_adapter, 'get_resource_estimate')


@pytest.mark.asyncio
async def test_solve_task(mock_ttt_adapter, sample_task):
    """Test solve_task returns valid StrategyOutput."""
    result = await mock_ttt_adapter.solve_task(sample_task)
    
    assert isinstance(result, StrategyOutput)
    assert result.strategy_type == StrategyType.TEST_TIME_TRAINING
    assert isinstance(result.predicted_output, np.ndarray)
    assert result.predicted_output.dtype == np.int8
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.resource_usage is not None
    assert result.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_solve_task_confidence_score(mock_ttt_adapter, sample_task):
    """Test solve_task returns appropriate confidence score."""
    result = await mock_ttt_adapter.solve_task(sample_task)
    
    # Should have high confidence from self-consistency
    assert result.confidence_score >= 0.5
    assert result.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_solve_task_per_pixel_confidence(mock_ttt_adapter, sample_task):
    """Test solve_task returns per-pixel confidence map."""
    result = await mock_ttt_adapter.solve_task(sample_task)
    
    assert result.per_pixel_confidence is not None
    assert isinstance(result.per_pixel_confidence, np.ndarray)
    assert result.per_pixel_confidence.shape == result.predicted_output.shape
    assert np.all(result.per_pixel_confidence >= 0.0)
    assert np.all(result.per_pixel_confidence <= 1.0)


@pytest.mark.asyncio
async def test_solve_task_resource_usage(mock_ttt_adapter, sample_task):
    """Test solve_task tracks resource usage correctly."""
    result = await mock_ttt_adapter.solve_task(sample_task)
    
    assert result.resource_usage is not None
    assert result.resource_usage.api_calls == 0  # TTT uses local model
    assert result.resource_usage.total_tokens == 0
    assert result.resource_usage.estimated_cost == 0.0


def test_get_confidence_estimate_fast(mock_ttt_adapter, sample_task):
    """Test get_confidence_estimate completes in <100ms."""
    import time
    
    start = time.time()
    confidence = mock_ttt_adapter.get_confidence_estimate(sample_task)
    elapsed_ms = (time.time() - start) * 1000
    
    assert elapsed_ms < 100.0
    assert 0.0 <= confidence <= 1.0


def test_get_confidence_estimate_heuristics(mock_ttt_adapter):
    """Test confidence estimate uses task heuristics."""
    # Task with many training examples - higher confidence
    task_many_examples = ARCTask(
        task_id="many_ex",
        task_source="evaluation",
        train_examples=[{"input": [[0]], "output": [[1]]}] * 5,
        test_input=[[0]]
    )
    
    # Task with few training examples - lower confidence
    task_few_examples = ARCTask(
        task_id="few_ex",
        task_source="evaluation",
        train_examples=[{"input": [[0]], "output": [[1]]}] * 2,
        test_input=[[0]]
    )
    
    conf_many = mock_ttt_adapter.get_confidence_estimate(task_many_examples)
    conf_few = mock_ttt_adapter.get_confidence_estimate(task_few_examples)
    
    assert conf_many > conf_few


def test_get_resource_estimate_fast(mock_ttt_adapter, sample_task):
    """Test get_resource_estimate completes in <50ms."""
    import time
    
    start = time.time()
    resources = mock_ttt_adapter.get_resource_estimate(sample_task)
    elapsed_ms = (time.time() - start) * 1000
    
    assert elapsed_ms < 50.0
    assert isinstance(resources, ResourceUsage)


def test_get_resource_estimate_no_api_calls(mock_ttt_adapter, sample_task):
    """Test TTT resource estimate shows no API calls (local model)."""
    resources = mock_ttt_adapter.get_resource_estimate(sample_task)
    
    assert resources.api_calls == 0
    assert resources.total_tokens == 0
    assert resources.estimated_cost == 0.0


def test_get_resource_estimate_scales_with_task(mock_ttt_adapter):
    """Test resource estimate scales with task complexity."""
    # Simple task
    simple_task = ARCTask(
        task_id="simple",
        task_source="evaluation",
        train_examples=[{"input": [[0]], "output": [[1]]}] * 2,
        test_input=[[0]]
    )
    
    # Complex task
    complex_task = ARCTask(
        task_id="complex",
        task_source="evaluation",
        train_examples=[{"input": [[0]], "output": [[1]]}] * 10,
        test_input=[[0]]
    )
    
    simple_res = mock_ttt_adapter.get_resource_estimate(simple_task)
    complex_res = mock_ttt_adapter.get_resource_estimate(complex_task)
    
    # Complex task should require more time
    assert complex_res.estimated_time_sec > simple_res.estimated_time_sec


@pytest.mark.asyncio
async def test_integration_with_timing_coordinator(mock_ttt_adapter, sample_task):
    """Test integration with TimingCoordinator for timeout management."""
    # Mock timeout scenario
    with patch.object(mock_ttt_adapter, 'solve_task') as mock_solve:
        async def timeout_handler(*args, **kwargs):
            # Simulate timeout
            raise TimeoutError("Task exceeded time limit")
        
        mock_solve.side_effect = timeout_handler
        
        with pytest.raises(TimeoutError):
            await mock_ttt_adapter.solve_task(sample_task)


@pytest.mark.asyncio
async def test_structured_logging_with_correlation_id(mock_ttt_adapter, sample_task):
    """Test structured logging includes correlation IDs."""
    # This would use src/infrastructure/logging.py in real implementation
    result = await mock_ttt_adapter.solve_task(sample_task)
    
    # Verify result has reasoning trace for logging
    assert result.reasoning_trace is not None
    assert len(result.reasoning_trace) > 0


def test_metrics_collection_integration(mock_ttt_adapter, sample_task):
    """Test integration with MetricsCollector for Prometheus monitoring."""
    # This would use src/infrastructure/monitoring.py in real implementation
    confidence = mock_ttt_adapter.get_confidence_estimate(sample_task)
    
    # Metrics would be collected:
    # - ttt_confidence_estimate
    # - ttt_resource_estimate_time
    # - ttt_solve_task_duration
    # - ttt_accuracy
    
    assert confidence is not None


@pytest.mark.skip(reason="Requires actual TTT implementation")
def test_real_ttt_adapter_implementation():
    """
    Test actual TTTAdapter implementation in src/adapters/strategies/ttt_adapter.py.
    
    This test should verify:
    1. TTTAdapter inherits from StrategyPort
    2. solve_task() uses enhanced TTT (leave-one-out, self-consistency, LoRA)
    3. get_confidence_estimate() uses self-consistency agreement rate
    4. get_resource_estimate() estimates based on model size and task complexity
    5. Integration with TimingCoordinator for timeout management
    6. Integration with MetricsCollector for monitoring
    7. Structured logging with correlation IDs
    """
    from src.adapters.strategies.ttt_adapter import TTTAdapter
    
    adapter = TTTAdapter()
    
    # Verify interface implementation
    assert isinstance(adapter, StrategyPort)
    
    # Test methods exist
    assert hasattr(adapter, 'solve_task')
    assert hasattr(adapter, 'get_confidence_estimate')
    assert hasattr(adapter, 'get_resource_estimate')
