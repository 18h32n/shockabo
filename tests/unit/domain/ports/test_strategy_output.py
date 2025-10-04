"""Unit tests for StrategyOutput model and serialization."""

import time
from datetime import datetime

import numpy as np
import pytest

from src.domain.models import ARCTaskSolution, ResourceUsage, StrategyOutput, StrategyType


class TestStrategyOutputValidation:
    """Test StrategyOutput validation logic."""

    def test_valid_strategy_output(self):
        output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.array([[1, 2], [3, 4]], dtype=np.int8),
            confidence_score=0.85,
        )
        assert output.confidence_score == 0.85
        assert output.predicted_output.shape == (2, 2)

    def test_grid_size_validation(self):
        with pytest.raises(ValueError, match="exceeds max 30x30"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((31, 31), dtype=np.int8),
                confidence_score=0.5,
            )

    def test_confidence_score_validation_too_high(self):
        with pytest.raises(ValueError, match="not in range"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((5, 5), dtype=np.int8),
                confidence_score=1.5,
            )

    def test_confidence_score_validation_too_low(self):
        with pytest.raises(ValueError, match="not in range"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((5, 5), dtype=np.int8),
                confidence_score=-0.1,
            )

    def test_per_pixel_confidence_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape must match"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((5, 5), dtype=np.int8),
                confidence_score=0.8,
                per_pixel_confidence=np.zeros((4, 4), dtype=np.float32),
            )

    def test_per_pixel_confidence_value_validation(self):
        with pytest.raises(ValueError, match="must be in range"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((5, 5), dtype=np.int8),
                confidence_score=0.8,
                per_pixel_confidence=np.full((5, 5), 1.5, dtype=np.float32),
            )

    def test_reasoning_trace_size_limit(self):
        with pytest.raises(ValueError, match="limited to 100 entries"):
            StrategyOutput(
                strategy_type=StrategyType.PROGRAM_SYNTHESIS,
                predicted_output=np.zeros((5, 5), dtype=np.int8),
                confidence_score=0.8,
                reasoning_trace=[f"step_{i}" for i in range(101)],
            )

    def test_valid_per_pixel_confidence(self):
        output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.array([[1, 2], [3, 4]], dtype=np.int8),
            confidence_score=0.85,
            per_pixel_confidence=np.array([[0.9, 0.8], [0.7, 0.95]], dtype=np.float32),
        )
        assert output.per_pixel_confidence.shape == (2, 2)
        assert np.all(output.per_pixel_confidence >= 0.0)
        assert np.all(output.per_pixel_confidence <= 1.0)


class TestStrategyOutputSerialization:
    """Test StrategyOutput msgpack serialization/deserialization."""

    def test_basic_serialization_roundtrip(self):
        original = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8),
            confidence_score=0.75,
            processing_time_ms=1500,
        )

        serialized = original.to_msgpack()
        deserialized = StrategyOutput.from_msgpack(serialized)

        assert deserialized.strategy_type == original.strategy_type
        assert np.array_equal(deserialized.predicted_output, original.predicted_output)
        assert deserialized.confidence_score == original.confidence_score
        assert deserialized.processing_time_ms == original.processing_time_ms

    def test_serialization_with_per_pixel_confidence(self):
        original = StrategyOutput(
            strategy_type=StrategyType.EVOLUTION,
            predicted_output=np.array([[1, 2], [3, 4]], dtype=np.int8),
            confidence_score=0.85,
            per_pixel_confidence=np.array([[0.9, 0.8], [0.7, 0.95]], dtype=np.float32),
        )

        serialized = original.to_msgpack()
        deserialized = StrategyOutput.from_msgpack(serialized)

        assert np.array_equal(deserialized.per_pixel_confidence, original.per_pixel_confidence)

    def test_serialization_with_reasoning_trace(self):
        original = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.zeros((5, 5), dtype=np.int8),
            confidence_score=0.6,
            reasoning_trace=["Generated program", "Evaluated fitness", "Selected best"],
        )

        serialized = original.to_msgpack()
        deserialized = StrategyOutput.from_msgpack(serialized)

        assert deserialized.reasoning_trace == original.reasoning_trace

    def test_serialization_with_resource_usage(self):
        resource_usage = ResourceUsage(
            task_id="test_123",
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            cpu_seconds=45.2,
            memory_mb=512.0,
            gpu_memory_mb=2048.0,
            api_calls={"tier1": 10, "tier2": 5},
            total_tokens=5000,
            estimated_cost=0.15,
            timestamp=datetime.now(),
        )

        original = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.zeros((10, 10), dtype=np.int8),
            confidence_score=0.9,
            resource_usage=resource_usage,
        )

        serialized = original.to_msgpack()
        deserialized = StrategyOutput.from_msgpack(serialized)

        assert deserialized.resource_usage.task_id == resource_usage.task_id
        assert deserialized.resource_usage.cpu_seconds == resource_usage.cpu_seconds
        assert deserialized.resource_usage.memory_mb == resource_usage.memory_mb
        assert deserialized.resource_usage.api_calls == resource_usage.api_calls

    def test_serialization_with_strategy_metadata(self):
        original = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.zeros((5, 5), dtype=np.int8),
            confidence_score=0.7,
            strategy_metadata={
                "programs_evaluated": 500,
                "best_fitness": 0.92,
                "generation": 25,
            },
        )

        serialized = original.to_msgpack()
        deserialized = StrategyOutput.from_msgpack(serialized)

        assert deserialized.strategy_metadata == original.strategy_metadata


class TestStrategyOutputPerformance:
    """Test StrategyOutput performance requirements."""

    def test_serialization_performance_10x10_grid(self):
        output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.random.randint(0, 10, (10, 10), dtype=np.int8),
            confidence_score=0.85,
            per_pixel_confidence=np.random.rand(10, 10).astype(np.float32),
        )

        start = time.perf_counter()
        serialized = output.to_msgpack()
        serialize_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        StrategyOutput.from_msgpack(serialized)
        deserialize_time = (time.perf_counter() - start) * 1000

        assert serialize_time < 10, f"Serialization took {serialize_time:.2f}ms (target: <10ms)"
        assert deserialize_time < 10, f"Deserialization took {deserialize_time:.2f}ms (target: <10ms)"

    def test_serialization_performance_30x30_grid(self):
        output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.random.randint(0, 10, (30, 30), dtype=np.int8),
            confidence_score=0.85,
            per_pixel_confidence=np.random.rand(30, 30).astype(np.float32),
            strategy_metadata={"programs": 500, "fitness": 0.9},
            reasoning_trace=[f"step_{i}" for i in range(50)],
        )

        start = time.perf_counter()
        serialized = output.to_msgpack()
        serialize_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        StrategyOutput.from_msgpack(serialized)
        deserialize_time = (time.perf_counter() - start) * 1000

        assert serialize_time < 10, f"Serialization took {serialize_time:.2f}ms (target: <10ms)"
        assert deserialize_time < 10, f"Deserialization took {deserialize_time:.2f}ms (target: <10ms)"

    def test_memory_usage_estimate(self):
        output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.random.randint(0, 10, (30, 30), dtype=np.int8),
            confidence_score=0.85,
            per_pixel_confidence=np.random.rand(30, 30).astype(np.float32),
        )

        serialized = output.to_msgpack()
        size_kb = len(serialized) / 1024

        assert size_kb < 1024, f"Serialized size {size_kb:.2f}KB exceeds 1MB limit"


class TestStrategyOutputVsARCTaskSolution:
    """Compare StrategyOutput performance vs ARCTaskSolution."""

    def test_serialization_speedup_comparison(self):
        """Benchmark StrategyOutput vs ARCTaskSolution full round-trip.
        
        Tests full serialize + deserialize cycle to measure real-world performance.
        StrategyOutput msgpack is optimized for network transfer and storage.
        """
        import json

        grid_30x30 = [[int(x) for x in row] for row in np.random.randint(0, 10, (30, 30))]

        arc_solution = ARCTaskSolution(
            task_id="test_123",
            predictions=[grid_30x30],
            strategy_used=StrategyType.PROGRAM_SYNTHESIS,
            confidence_score=0.85,
            metadata={"programs": 500, "fitness": 0.9},
        )

        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            json_str = json.dumps({
                "task_id": arc_solution.task_id,
                "predictions": arc_solution.predictions,
                "strategy_used": arc_solution.strategy_used.value,
                "confidence_score": arc_solution.confidence_score,
                "metadata": arc_solution.metadata,
            })
            json.loads(json_str)
        arc_roundtrip_time = ((time.perf_counter() - start) / iterations) * 1000

        strategy_output = StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=np.array(grid_30x30, dtype=np.int8),
            confidence_score=0.85,
            strategy_metadata={"programs": 500, "fitness": 0.9},
        )

        start = time.perf_counter()
        for _ in range(iterations):
            packed = strategy_output.to_msgpack()
            StrategyOutput.from_msgpack(packed)
        strategy_roundtrip_time = ((time.perf_counter() - start) / iterations) * 1000

        assert strategy_roundtrip_time < 10, f"StrategyOutput round-trip {strategy_roundtrip_time:.2f}ms exceeds 10ms target"
