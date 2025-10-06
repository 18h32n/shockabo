"""
Integration tests for real TTTAdapter with StrategyPort interface.

Tests the actual TTTAdapter implementation (not mock) with enhanced components:
- Leave-one-out generation
- Self-consistency validation
- LoRA optimization
- Memory efficient batch processing
- TimingCoordinator integration
- MetricsCollector integration
"""
import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask, ResourceUsage, StrategyOutput, StrategyType
from src.domain.ports.strategy import StrategyPort
from src.domain.ports.timing import ResourceBudget, TerminationReason, TimingCoordinator
from src.infrastructure.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class MockTimingCoordinator(TimingCoordinator):
    """Mock timing coordinator for testing."""
    
    def __init__(self):
        self.registered_strategies = {}
        self.termination_signals = {}
        self.success_signals = {}
        self.progress_reports = {}
    
    async def register_strategy(
        self,
        strategy_id: str,
        timeout_ms: int,
        resource_budget: ResourceBudget | None = None,
    ) -> None:
        self.registered_strategies[strategy_id] = {
            "timeout_ms": timeout_ms,
            "resource_budget": resource_budget
        }
    
    async def should_terminate(self, strategy_id: str) -> bool:
        return self.termination_signals.get(strategy_id, False)
    
    async def signal_success(
        self, strategy_id: str, confidence: float, metadata: dict | None = None
    ) -> None:
        self.success_signals[strategy_id] = {
            "confidence": confidence,
            "metadata": metadata or {}
        }
    
    async def report_progress(
        self,
        strategy_id: str,
        progress: float,
        estimated_time_remaining_ms: int,
        current_best_confidence: float,
    ) -> None:
        self.progress_reports[strategy_id] = {
            "progress": progress,
            "estimated_time_remaining_ms": estimated_time_remaining_ms,
            "current_best_confidence": current_best_confidence
        }
    
    async def request_resource_extension(
        self, strategy_id: str, resource_type: str, additional_amount: float
    ) -> bool:
        return True
    
    async def get_termination_reason(self, strategy_id: str) -> TerminationReason | None:
        return None
    
    async def unregister_strategy(
        self, strategy_id: str, reason: TerminationReason
    ) -> None:
        if strategy_id in self.registered_strategies:
            del self.registered_strategies[strategy_id]


@pytest.fixture
def ttt_config():
    """Fixture providing minimal TTT configuration for testing."""
    return TTTConfig(
        model_name="meta-llama/Llama-3.2-1B",
        device="cpu",  # Use CPU for testing
        quantization=False,  # Disable for faster testing
        mixed_precision=False,
        
        # Minimal training config
        num_epochs=1,
        per_instance_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        
        # Enhanced features enabled
        permute_n=2,
        max_examples=3,
        
        # Fast inference
        max_inference_time=60.0,  # 1 minute for testing
        enable_progressive_inference=False
    )


@pytest.fixture
def sample_task():
    """Fixture providing sample ARC task."""
    return ARCTask(
        task_id="test_real_adapter_001",
        task_source="evaluation",
        train_examples=[
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}
        ],
        test_input=[[1, 1], [0, 0]]
    )


@pytest.fixture
def mock_timing_coordinator():
    """Fixture providing mock timing coordinator."""
    return MockTimingCoordinator()


@pytest.fixture
def metrics_collector():
    """Fixture providing clean metrics collector."""
    collector = MetricsCollector()
    collector.reset()
    return collector


def test_ttt_adapter_implements_strategy_port():
    """Test that TTTAdapter implements StrategyPort interface."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    assert isinstance(adapter, StrategyPort)
    assert hasattr(adapter, 'solve_task')
    assert hasattr(adapter, 'get_confidence_estimate')
    assert hasattr(adapter, 'get_resource_estimate')


def test_ttt_adapter_has_enhanced_components():
    """Test that TTTAdapter initializes enhanced components."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    # Check enhanced components are initialized
    assert hasattr(adapter, 'leave_one_out_generator')
    assert hasattr(adapter, 'self_consistency_validator')
    assert hasattr(adapter, 'lora_optimizer')
    assert hasattr(adapter, 'batch_processor')
    assert hasattr(adapter, 'metrics_collector')
    
    assert adapter.leave_one_out_generator is not None
    assert adapter.self_consistency_validator is not None
    assert adapter.lora_optimizer is not None
    assert adapter.batch_processor is not None
    assert adapter.metrics_collector is not None


def test_get_confidence_estimate_fast(sample_task):
    """Test get_confidence_estimate completes in <100ms."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    start = time.time()
    confidence = adapter.get_confidence_estimate(sample_task)
    elapsed_ms = (time.time() - start) * 1000
    
    assert elapsed_ms < 100.0
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.4  # Enhanced TTT should have reasonable confidence


def test_get_confidence_estimate_uses_task_heuristics():
    """Test confidence estimate uses task characteristics."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
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
    
    conf_many = adapter.get_confidence_estimate(task_many_examples)
    conf_few = adapter.get_confidence_estimate(task_few_examples)
    
    assert conf_many > conf_few


def test_get_resource_estimate_fast(sample_task):
    """Test get_resource_estimate completes in <50ms."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    start = time.time()
    resources = adapter.get_resource_estimate(sample_task)
    elapsed_ms = (time.time() - start) * 1000
    
    assert elapsed_ms < 50.0
    assert isinstance(resources, ResourceUsage)


def test_get_resource_estimate_no_api_calls(sample_task):
    """Test TTT resource estimate shows no API calls (local model)."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    resources = adapter.get_resource_estimate(sample_task)
    
    assert resources.api_calls == {}
    assert resources.total_tokens == 0
    assert resources.estimated_cost == 0.0
    assert resources.cpu_seconds > 0  # Uses cpu_seconds not estimated_time_sec
    assert resources.memory_mb > 0


def test_get_resource_estimate_scales_with_complexity(sample_task):
    """Test resource estimate scales with task complexity."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
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
    
    simple_res = adapter.get_resource_estimate(simple_task)
    complex_res = adapter.get_resource_estimate(complex_task)
    
    # Complex task should require more time
    assert complex_res.cpu_seconds > simple_res.cpu_seconds


@pytest.mark.skip(reason="Requires model weights - run manually with GPU")
@pytest.mark.asyncio
async def test_solve_task_returns_strategy_output(sample_task):
    """Test solve_task returns valid StrategyOutput."""
    config = TTTConfig(device="auto", enable_progressive_inference=False)
    adapter = TTTAdapter(config)
    
    # Initialize model
    adapter.initialize_model()
    
    result = await adapter.solve_task(sample_task)
    
    assert isinstance(result, StrategyOutput)
    assert result.strategy_type == StrategyType.TEST_TIME_TRAINING
    assert isinstance(result.predicted_output, np.ndarray)
    assert result.predicted_output.dtype == np.int8
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.resource_usage is not None
    assert result.execution_time_ms >= 0


@pytest.mark.skip(reason="Requires model weights - run manually with GPU")
@pytest.mark.asyncio
async def test_solve_task_with_timing_coordinator(sample_task, mock_timing_coordinator):
    """Test solve_task integrates with TimingCoordinator."""
    config = TTTConfig(device="auto", max_inference_time=60.0)
    adapter = TTTAdapter(config)
    adapter.set_timing_coordinator(mock_timing_coordinator)
    
    # Initialize model
    adapter.initialize_model()
    
    result = await adapter.solve_task(sample_task)
    
    # Verify timing coordinator interactions
    strategy_id = f"ttt_{sample_task.task_id}"
    assert strategy_id in mock_timing_coordinator.success_signals
    assert mock_timing_coordinator.success_signals[strategy_id]["confidence"] == result.confidence_score


@pytest.mark.skip(reason="Requires model weights - run manually with GPU")
@pytest.mark.asyncio
async def test_solve_task_records_metrics(sample_task, metrics_collector):
    """Test solve_task records metrics in MetricsCollector."""
    config = TTTConfig(device="auto")
    adapter = TTTAdapter(config)
    
    # Initialize model
    adapter.initialize_model()
    
    result = await adapter.solve_task(sample_task)
    
    # Verify metrics were recorded
    duration = metrics_collector.get_average_duration("test_time_training")
    assert duration > 0.0
    
    # Confidence should be recorded
    conf_dist = metrics_collector.get_confidence_distribution("test_time_training")
    assert conf_dist["mean"] >= 0.0


def test_timing_coordinator_setter():
    """Test set_timing_coordinator method."""
    config = TTTConfig(device="cpu")
    adapter = TTTAdapter(config)
    
    coordinator = MockTimingCoordinator()
    adapter.set_timing_coordinator(coordinator)
    
    assert adapter.timing_coordinator is coordinator


def test_enhanced_components_configuration():
    """Test enhanced components use correct configuration."""
    config = TTTConfig(
        device="cpu",
        permute_n=5,
        max_examples=8,
        lora_rank=64,
        lora_alpha=32,
        per_instance_lr=1e-4,
        gradient_accumulation_steps=4,
        memory_limit_mb=24576,
        checkpointing_layers=6
    )
    adapter = TTTAdapter(config)
    
    # Check self-consistency config
    assert adapter.self_consistency_validator.config.permute_n == 5
    assert adapter.self_consistency_validator.config.consensus_threshold == 0.6
    
    # Check leave-one-out config
    assert adapter.leave_one_out_generator.config.max_examples == 8
    
    # Check LoRA optimizer config
    assert adapter.lora_optimizer.config.rank == 64
    assert adapter.lora_optimizer.config.alpha == 32
    assert adapter.lora_optimizer.config.learning_rate == 1e-4
    
    # Check batch processor config
    assert adapter.batch_processor.config.gradient_accumulation_steps == 4
    assert adapter.batch_processor.config.memory_limit_mb == 24576
    assert adapter.batch_processor.config.checkpointing_layers == 6


@pytest.mark.skip(reason="Requires model weights - run manually for end-to-end test")
@pytest.mark.asyncio
async def test_end_to_end_enhanced_ttt_pipeline(sample_task):
    """
    End-to-end test of enhanced TTT pipeline.
    
    Tests full integration:
    1. Leave-one-out splits generated
    2. LoRA optimization with early stopping
    3. Self-consistency validation with permutations
    4. Memory efficient batch processing
    5. TimingCoordinator registration
    6. MetricsCollector recording
    7. StrategyOutput returned
    """
    config = TTTConfig(
        device="auto",
        permute_n=3,
        num_epochs=2,
        per_instance_epochs=1,
        max_inference_time=300.0
    )
    
    adapter = TTTAdapter(config)
    coordinator = MockTimingCoordinator()
    adapter.set_timing_coordinator(coordinator)
    
    # Initialize model
    adapter.initialize_model()
    
    # Execute solve_task
    result = await adapter.solve_task(sample_task)
    
    # Verify result
    assert isinstance(result, StrategyOutput)
    assert result.strategy_type == StrategyType.TEST_TIME_TRAINING
    assert isinstance(result.predicted_output, np.ndarray)
    assert 0.0 <= result.confidence_score <= 1.0
    
    # Verify per-pixel confidence if available
    if result.per_pixel_confidence is not None:
        assert isinstance(result.per_pixel_confidence, np.ndarray)
        assert result.per_pixel_confidence.shape == result.predicted_output.shape
    
    # Verify timing coordinator integration
    strategy_id = f"ttt_{sample_task.task_id}"
    assert strategy_id in coordinator.success_signals
    
    # Verify metrics collection
    duration = adapter.metrics_collector.get_average_duration("test_time_training")
    assert duration > 0.0
    
    # Verify resource usage
    assert result.resource_usage.api_calls == 0
    assert result.resource_usage.estimated_cost == 0.0
    
    # Verify execution time
    assert result.execution_time_ms > 0
    assert result.execution_time_ms < config.max_inference_time * 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
