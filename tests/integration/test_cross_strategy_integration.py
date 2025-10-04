"""Integration tests for cross-strategy integration points."""

import asyncio
import time

import numpy as np
import pytest

from src.domain.models import ARCTask, StrategyOutput, StrategyType
from src.infrastructure.logging import LogContext, configure_logging, get_logger
from src.infrastructure.monitoring import MetricsCollector
from tests.mocks.mock_evolution_strategy import MockEvolutionStrategy
from tests.mocks.mock_imitation_strategy import MockImitationStrategy
from tests.mocks.mock_ttt_strategy import MockTTTStrategy


@pytest.fixture(scope="module")
def configure_test_logging():
    """Configure logging for tests."""
    configure_logging(log_level="DEBUG", json_output=False)


@pytest.fixture
def sample_task():
    """Create a sample ARC task for testing."""
    return ARCTask(
        task_id="test_integration_001",
        task_source="training",
        train_examples=[
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}
        ],
        test_input=[[5, 6], [7, 8]],
        test_output=[[8, 7], [6, 5]],
    )


class TestStrategyOutputFormat:
    """Test standardized output format compatibility across strategies."""

    @pytest.mark.asyncio
    async def test_ttt_output_format(self, sample_task):
        """Test TTT strategy produces valid StrategyOutput."""
        strategy = MockTTTStrategy(min_delay_ms=10, max_delay_ms=20)
        output = await strategy.solve_task(sample_task)

        assert isinstance(output, StrategyOutput)
        assert output.strategy_type == StrategyType.TEST_TIME_TRAINING
        assert isinstance(output.predicted_output, np.ndarray)
        assert 0.0 <= output.confidence_score <= 1.0
        assert output.processing_time_ms > 0
        assert output.resource_usage is not None

    @pytest.mark.asyncio
    async def test_evolution_output_format(self, sample_task):
        """Test evolution strategy produces valid StrategyOutput."""
        strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)
        output = await strategy.solve_task(sample_task)

        assert isinstance(output, StrategyOutput)
        assert output.strategy_type == StrategyType.PROGRAM_SYNTHESIS
        assert isinstance(output.predicted_output, np.ndarray)
        assert 0.0 <= output.confidence_score <= 1.0
        assert output.reasoning_trace is not None
        assert len(output.reasoning_trace) > 0

    @pytest.mark.asyncio
    async def test_imitation_output_format(self, sample_task):
        """Test imitation strategy produces valid StrategyOutput."""
        strategy = MockImitationStrategy(min_delay_ms=10, max_delay_ms=20)
        output = await strategy.solve_task(sample_task)

        assert isinstance(output, StrategyOutput)
        assert output.strategy_type == StrategyType.IMITATION_LEARNING
        assert output.per_pixel_confidence is not None
        assert output.per_pixel_confidence.shape == output.predicted_output.shape

    @pytest.mark.asyncio
    async def test_output_serialization_compatibility(self, sample_task):
        """Test all strategies produce serializable outputs."""
        strategies = [
            MockTTTStrategy(min_delay_ms=10, max_delay_ms=20),
            MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20),
            MockImitationStrategy(min_delay_ms=10, max_delay_ms=20),
        ]

        for strategy in strategies:
            output = await strategy.solve_task(sample_task)

            serialized = output.to_msgpack()
            assert isinstance(serialized, bytes)

            deserialized = StrategyOutput.from_msgpack(serialized)
            assert deserialized.strategy_type == output.strategy_type
            assert np.array_equal(deserialized.predicted_output, output.predicted_output)
            assert deserialized.confidence_score == output.confidence_score


class TestConfidenceScoring:
    """Test confidence scoring across different strategies."""

    @pytest.mark.asyncio
    async def test_confidence_range_validation(self, sample_task):
        """Test all strategies produce confidence in valid range."""
        strategies = [
            MockTTTStrategy(min_delay_ms=10, max_delay_ms=20),
            MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20),
            MockImitationStrategy(min_delay_ms=10, max_delay_ms=20),
        ]

        for strategy in strategies:
            for _ in range(5):
                output = await strategy.solve_task(sample_task)
                assert 0.0 <= output.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_estimate_accuracy(self, sample_task):
        """Test confidence estimates are reasonable."""
        ttt = MockTTTStrategy(success_rate=0.6, min_delay_ms=10, max_delay_ms=20)
        evolution = MockEvolutionStrategy(success_rate=0.4, min_delay_ms=10, max_delay_ms=20)
        imitation = MockImitationStrategy(success_rate=0.3, min_delay_ms=10, max_delay_ms=20)

        assert ttt.get_confidence_estimate(sample_task) > evolution.get_confidence_estimate(sample_task)
        assert evolution.get_confidence_estimate(sample_task) > imitation.get_confidence_estimate(sample_task)


class TestMetricsCollection:
    """Test metrics collection across strategies."""

    @pytest.mark.asyncio
    async def test_solve_duration_metrics(self, sample_task):
        """Test solve duration is tracked correctly."""
        metrics = MetricsCollector()
        strategy = MockTTTStrategy(min_delay_ms=50, max_delay_ms=100)

        start = time.time()
        output = await strategy.solve_task(sample_task)
        duration = time.time() - start

        metrics.record_solve_duration(
            strategy="test_time_training",
            duration_seconds=duration,
            task_type="transformation",
        )

        avg_duration = metrics.get_average_duration("test_time_training", "transformation")
        assert avg_duration > 0
        assert abs(avg_duration - duration) < 0.1

    @pytest.mark.asyncio
    async def test_accuracy_tracking(self, sample_task):
        """Test accuracy tracking across multiple tasks."""
        metrics = MetricsCollector()

        metrics.record_accuracy("program_synthesis", "transformation", True)
        metrics.record_accuracy("program_synthesis", "transformation", False)
        metrics.record_accuracy("program_synthesis", "transformation", True)

        accuracy = metrics.get_accuracy("program_synthesis", "transformation")
        assert accuracy == 2 / 3

    @pytest.mark.asyncio
    async def test_confidence_distribution(self, sample_task):
        """Test confidence score distribution tracking."""
        metrics = MetricsCollector()
        strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)

        for _ in range(10):
            output = await strategy.solve_task(sample_task)
            metrics.record_confidence_score(
                strategy="program_synthesis",
                confidence=output.confidence_score,
            )

        dist = metrics.get_confidence_distribution("program_synthesis")
        assert 0.0 <= dist["min"] <= dist["max"] <= 1.0
        assert dist["min"] <= dist["mean"] <= dist["max"]

    @pytest.mark.asyncio
    async def test_api_call_tracking(self, sample_task):
        """Test API call counting."""
        metrics = MetricsCollector()
        strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)

        output = await strategy.solve_task(sample_task)

        if output.resource_usage and output.resource_usage.api_calls:
            for tier, count in output.resource_usage.api_calls.items():
                metrics.record_api_call("program_synthesis", tier, count)

        totals = metrics.get_api_call_totals("program_synthesis")
        assert len(totals) > 0

    @pytest.mark.asyncio
    async def test_prometheus_export(self, sample_task):
        """Test Prometheus metrics export format."""
        metrics = MetricsCollector()
        strategy = MockTTTStrategy(min_delay_ms=10, max_delay_ms=20)

        output = await strategy.solve_task(sample_task)
        metrics.record_solve_duration("test_time_training", 2.5, "transformation")
        metrics.record_accuracy("test_time_training", "transformation", True)
        metrics.record_confidence_score("test_time_training", output.confidence_score)

        prom_metrics = metrics.export_prometheus_metrics()
        assert isinstance(prom_metrics, str)
        assert "arc_strategy_solve_duration_seconds" in prom_metrics
        assert "arc_strategy_accuracy" in prom_metrics
        assert "arc_strategy_confidence_score" in prom_metrics


class TestUnifiedLogging:
    """Test unified logging with correlation IDs."""

    @pytest.mark.asyncio
    async def test_correlation_id_tracking(self, sample_task, configure_test_logging):
        """Test correlation IDs are propagated through logs."""
        log = get_logger(__name__)

        with LogContext(correlation_id="test_corr_123", strategy="program_synthesis"):
            strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)
            output = await strategy.solve_task(sample_task)

            log.info(
                "task_completed",
                task_id=sample_task.task_id,
                confidence=output.confidence_score,
            )

    @pytest.mark.asyncio
    async def test_multi_strategy_logging(self, sample_task, configure_test_logging):
        """Test logging with multiple strategies in parallel."""
        log = get_logger(__name__)

        async def run_strategy(strategy_name, strategy):
            with LogContext(strategy=strategy_name, task_id=sample_task.task_id):
                output = await strategy.solve_task(sample_task)
                log.info(
                    "strategy_complete",
                    confidence=output.confidence_score,
                    processing_time_ms=output.processing_time_ms,
                )
                return output

        results = await asyncio.gather(
            run_strategy("ttt", MockTTTStrategy(min_delay_ms=10, max_delay_ms=20)),
            run_strategy("evolution", MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)),
            run_strategy("imitation", MockImitationStrategy(min_delay_ms=10, max_delay_ms=20)),
        )

        assert len(results) == 3
        assert all(isinstance(r, StrategyOutput) for r in results)


class TestErrorHandling:
    """Test error handling across strategies."""

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, sample_task):
        """Test timeout error handling."""
        strategy = MockTTTStrategy(min_delay_ms=10, max_delay_ms=20)
        strategy.inject_error("timeout")

        with pytest.raises(TimeoutError):
            await strategy.solve_task(sample_task)

    @pytest.mark.asyncio
    async def test_resource_exhausted_error_handling(self, sample_task):
        """Test resource exhaustion error handling."""
        strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)
        strategy.inject_error("resource_exhausted")

        with pytest.raises(RuntimeError, match="resource exhausted"):
            await strategy.solve_task(sample_task)


class TestPerformanceRequirements:
    """Test performance requirements are met."""

    @pytest.mark.asyncio
    async def test_serialization_performance(self, sample_task):
        """Test serialization meets <10ms requirement."""
        strategy = MockEvolutionStrategy(min_delay_ms=10, max_delay_ms=20)
        output = await strategy.solve_task(sample_task)

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            serialized = output.to_msgpack()
            StrategyOutput.from_msgpack(serialized)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 10, f"Serialization round-trip took {elapsed:.2f}ms (target: <10ms)"

    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, sample_task):
        """Test metrics collection overhead is minimal."""
        metrics = MetricsCollector()

        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            metrics.record_confidence_score("test_strategy", 0.5 + i * 0.0001)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 1, f"Metrics recording took {elapsed:.3f}ms per call (target: <1ms)"
