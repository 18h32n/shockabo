"""
Unit tests for Memory Efficient Batch Processor

Tests gradient accumulation, gradient checkpointing, dynamic batch sizing,
and memory monitoring functionality.
"""
import time
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.utils.ttt_batch_processor import (
    MemoryConfig,
    MemoryEfficientBatchProcessor,
    MemoryMetrics,
    MemoryMonitor,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def memory_config():
    """Fixture providing memory configuration."""
    return MemoryConfig(
        memory_limit_mb=1024,
        memory_warning_threshold=0.85,
        memory_critical_threshold=0.95,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        enable_dynamic_batch_size=True,
        min_batch_size=1,
        max_batch_size=4,
        enable_memory_monitoring=True
    )


@pytest.fixture
def memory_monitor(memory_config):
    """Fixture providing memory monitor."""
    return MemoryMonitor(memory_config)


@pytest.fixture
def batch_processor(memory_config):
    """Fixture providing batch processor."""
    return MemoryEfficientBatchProcessor(memory_config)


def test_memory_config_defaults():
    """Test MemoryConfig default values."""
    config = MemoryConfig()
    
    assert config.memory_limit_mb == 24576
    assert config.memory_warning_threshold == 0.85
    assert config.memory_critical_threshold == 0.95
    assert config.gradient_accumulation_steps == 4
    assert config.gradient_checkpointing is True
    assert config.enable_dynamic_batch_size is True


def test_memory_monitor_initialization(memory_monitor):
    """Test MemoryMonitor initialization."""
    assert memory_monitor.config is not None
    assert memory_monitor.device is not None
    assert memory_monitor.metrics_history == []


def test_get_current_memory(memory_monitor):
    """Test getting current memory metrics."""
    metrics = memory_monitor.get_current_memory()
    
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.current_mb >= 0.0
    assert 0.0 <= metrics.utilization <= 2.0


def test_batch_processor_initialization(batch_processor):
    """Test MemoryEfficientBatchProcessor initialization."""
    assert batch_processor.config is not None
    assert batch_processor.monitor is not None
    assert batch_processor.current_batch_size == batch_processor.config.min_batch_size
