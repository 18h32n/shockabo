"""
Performance tests for distributed evolution.

Tests throughput scaling and validates 2.5x improvement target.
"""

from pathlib import Path

import pytest

from scripts.benchmark_distributed_evolution import (
    BenchmarkConfig,
    DistributedEvolutionBenchmark,
)
from src.infrastructure.components.platform_detector import Platform


class TestDistributedEvolutionPerformance:
    """Performance tests for distributed evolution system."""

    @pytest.fixture
    def test_tasks_path(self) -> Path:
        """Path to test tasks."""
        return Path("tests/performance/data/distributed_test_tasks.json")

    @pytest.fixture
    def benchmark_config(self, test_tasks_path: Path, tmp_path: Path) -> BenchmarkConfig:
        """Create benchmark configuration."""
        return BenchmarkConfig(
            test_tasks_path=test_tasks_path,
            output_path=tmp_path / "benchmark_results.json",
            generations_per_task=10,
            single_platform=Platform.KAGGLE,
            multi_platforms=[Platform.KAGGLE, Platform.COLAB, Platform.LOCAL],
        )

    @pytest.mark.asyncio
    async def test_single_platform_baseline(self, benchmark_config: BenchmarkConfig):
        """Test single platform baseline measurement."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        result = await benchmark.run_single_platform_baseline()

        assert result.configuration == "single_platform"
        assert result.platform_count == 1
        assert result.total_tasks == len(benchmark.test_tasks)
        assert result.duration_seconds > 0
        assert result.tasks_per_hour > 0
        assert result.throughput_multiplier == 1.0
        assert result.checkpoint_overhead_seconds >= 0
        assert result.merge_overhead_seconds == 0

    @pytest.mark.asyncio
    async def test_multi_platform_benchmark(self, benchmark_config: BenchmarkConfig):
        """Test multi-platform benchmark execution."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        multi_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        assert multi_result.configuration == "multi_platform"
        assert multi_result.platform_count == 3
        assert multi_result.total_tasks == len(benchmark.test_tasks)
        assert multi_result.duration_seconds > 0
        assert multi_result.tasks_per_hour > 0
        assert multi_result.throughput_multiplier > 0
        assert multi_result.checkpoint_overhead_seconds > 0
        assert multi_result.merge_overhead_seconds >= 0

    @pytest.mark.asyncio
    async def test_throughput_improvement(self, benchmark_config: BenchmarkConfig):
        """Test that multi-platform achieves 2.5x throughput improvement."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        multi_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        assert multi_result.throughput_multiplier >= 2.5, (
            f"Expected 2.5x throughput improvement, got {multi_result.throughput_multiplier:.2f}x"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_overhead(self, benchmark_config: BenchmarkConfig):
        """Test checkpoint overhead is reasonable."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        multi_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        overhead_ratio = (
            multi_result.checkpoint_overhead_seconds / multi_result.duration_seconds
        )
        assert overhead_ratio < 0.2, (
            f"Checkpoint overhead too high: {overhead_ratio:.2%} of total time"
        )

    @pytest.mark.asyncio
    async def test_merge_overhead(self, benchmark_config: BenchmarkConfig):
        """Test merge overhead is reasonable."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        multi_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        overhead_ratio = multi_result.merge_overhead_seconds / multi_result.duration_seconds
        assert overhead_ratio < 0.1, (
            f"Merge overhead too high: {overhead_ratio:.2%} of total time"
        )

    @pytest.mark.asyncio
    async def test_platform_metrics_collected(self, benchmark_config: BenchmarkConfig):
        """Test that platform metrics are collected."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        multi_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        assert len(multi_result.platform_metrics) == 3
        for _platform_id, metrics in multi_result.platform_metrics.items():
            assert "completed_tasks" in metrics
            assert "checkpoints_created" in metrics
            assert "merge_operations" in metrics
            assert metrics["completed_tasks"] >= 0

    @pytest.mark.asyncio
    async def test_results_saved(self, benchmark_config: BenchmarkConfig):
        """Test benchmark results are saved correctly."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()
        await benchmark.run_multi_platform_benchmark(baseline_result.tasks_per_hour)

        benchmark.save_results()

        assert benchmark_config.output_path.exists()
        assert benchmark_config.output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_scaling_efficiency(self, benchmark_config: BenchmarkConfig):
        """Test scaling efficiency with different platform counts."""
        benchmark = DistributedEvolutionBenchmark(benchmark_config)
        benchmark.load_test_tasks()

        baseline_result = await benchmark.run_single_platform_baseline()

        benchmark_config.multi_platforms = [Platform.KAGGLE, Platform.COLAB]
        two_platform_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        benchmark_config.multi_platforms = [
            Platform.KAGGLE,
            Platform.COLAB,
            Platform.LOCAL,
        ]
        three_platform_result = await benchmark.run_multi_platform_benchmark(
            baseline_result.tasks_per_hour
        )

        assert three_platform_result.throughput_multiplier > two_platform_result.throughput_multiplier, (
            "Adding more platforms should improve throughput"
        )
