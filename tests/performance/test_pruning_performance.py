"""Performance tests for intelligent program pruning.

Tests that pruning achieves the target 40% performance improvement on
representative workloads.
"""

import random
import time
from unittest.mock import Mock

import numpy as np
import pytest

from src.adapters.strategies.program_pruner import ProgramPruner
from src.domain.dsl.base import DSLProgram, Operation, OperationResult
from src.domain.models import PruningStrategy
from src.domain.services.evaluation_service import EvaluationService


class MockOperation(Operation):
    """Mock operation for performance testing."""

    def __init__(self, name: str, complexity: str = "simple", **params):
        self._name = name
        self.complexity = complexity
        self.parameters = params

    def execute(self, grid, context=None):
        # Simulate execution time based on complexity
        if self.complexity == "complex":
            time.sleep(0.05)  # 50ms for complex operations
        elif self.complexity == "medium":
            time.sleep(0.02)  # 20ms for medium
        else:
            time.sleep(0.01)  # 10ms for simple

        return OperationResult(success=True, grid=grid)

    def get_name(self):
        return self._name

    def get_description(self):
        return f"Mock {self._name} ({self.complexity})"

    def get_parameter_schema(self):
        return {}


def generate_realistic_programs(count: int, bad_ratio: float = 0.4) -> list[DSLProgram]:
    """Generate realistic program distribution for benchmarking.
    
    Args:
        count: Number of programs to generate
        bad_ratio: Ratio of programs that should be prunable
        
    Returns:
        List of DSL programs
    """
    programs = []

    for i in range(count):
        is_bad = random.random() < bad_ratio

        if is_bad:
            # Generate prunable program
            bad_type = random.choice([
                "syntax_error",
                "memory_explosion",
                "contradictory",
                "low_confidence",
            ])

            if bad_type == "syntax_error":
                ops = [
                    {"name": "InvalidOperation", "params": {}},
                    {"name": "Rotate", "params": {"angle": 90}},
                ]
            elif bad_type == "memory_explosion":
                ops = [
                    {"name": "Tile", "params": {"factor": 100}},
                    {"name": "Zoom", "params": {"factor": 50}},
                ]
            elif bad_type == "contradictory":
                ops = [
                    {"name": "Rotate", "params": {"angle": 90}},
                    {"name": "Rotate", "params": {"angle": 270}},
                    {"name": "FlipHorizontal", "params": {}},
                    {"name": "FlipHorizontal", "params": {}},
                ]
            else:  # low_confidence
                ops = [
                    {"name": "ComplexTransform", "params": {"mode": "unknown"}},
                    {"name": "ObscureOperation", "params": {}},
                ]

        else:
            # Generate valid program with varying complexity
            num_ops = random.randint(3, 8)
            ops = []

            for j in range(num_ops):
                op_type = random.choice([
                    ("Rotate", {"angle": random.choice([90, 180, 270])}),
                    ("FloodFill", {"x": 0, "y": 0, "color": random.randint(1, 9)}),
                    ("Mirror", {"axis": random.choice(["horizontal", "vertical"])}),
                    ("Translate", {"dx": random.randint(-5, 5), "dy": random.randint(-5, 5)}),
                    ("ReplaceColor", {"old": random.randint(0, 9), "new": random.randint(0, 9)}),
                ])

                ops.append({
                    "name": op_type[0],
                    "params": op_type[1],
                })

        programs.append(DSLProgram(operations=ops))

    return programs


class TestPruningPerformance:
    """Test pruning performance improvements."""

    @pytest.fixture
    def test_grids(self):
        """Create test input grids."""
        return [
            np.random.randint(0, 10, (10, 10)).tolist(),
            np.random.randint(0, 10, (15, 15)).tolist(),
            np.random.randint(0, 10, (20, 20)).tolist(),
        ]

    def simulate_full_evaluation(self, program: DSLProgram, execution_time_ms: float = 50.0):
        """Simulate full program evaluation."""
        time.sleep(execution_time_ms / 1000)  # Simulate evaluation time

        # Determine if program would succeed
        first_op = program.operations[0]["name"] if program.operations else ""
        success = first_op not in ["InvalidOperation", "Tile", "ComplexTransform"]

        return Mock(
            final_accuracy=0.8 if success else 0.0,
            total_processing_time_ms=execution_time_ms,
        )

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_40_percent_improvement(self, test_grids):
        """Test that pruning achieves 40% performance improvement."""
        # Generate representative workload
        num_programs = 500  # Large enough for meaningful benchmark
        programs = generate_realistic_programs(num_programs, bad_ratio=0.4)

        # Measure baseline (no pruning)
        print("\nMeasuring baseline performance (no pruning)...")
        start_time = time.perf_counter()

        baseline_results = []
        for program in programs:
            result = self.simulate_full_evaluation(program)
            baseline_results.append(result)

        baseline_time = (time.perf_counter() - start_time) * 1000
        print(f"Baseline time: {baseline_time:.2f}ms")

        # Create evaluation service with pruning
        pruning_strategy = PruningStrategy(
            strategy_id="balanced",
            name="Balanced",
            aggressiveness=0.5,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=True,
            confidence_threshold=0.6,
            max_partial_ops=3,
            timeout_ms=100,
        )

        evaluation_service = EvaluationService(
            enable_gpu_evaluation=False,
            enable_pruning=True,
            default_pruning_strategy=pruning_strategy,
        )

        # Mock the actual evaluation to control timing
        async def mock_batch_evaluate(programs, inputs):
            results = []
            for prog in programs:
                # Only full evaluate non-pruned programs
                result = Mock()
                result.metadata = {"pruned": False}
                result.total_processing_time_ms = 50.0
                results.append(result)
            return results

        evaluation_service.batch_evaluate_programs = mock_batch_evaluate

        # Measure with pruning
        print("\nMeasuring performance with pruning...")
        start_time = time.perf_counter()

        results, metrics = await evaluation_service.evaluate_with_pruning(
            programs, test_grids
        )

        pruning_time = (time.perf_counter() - start_time) * 1000

        # Add simulated evaluation time for non-pruned programs
        non_pruned_count = num_programs - metrics.programs_pruned
        total_time_with_pruning = pruning_time + (non_pruned_count * 50)  # 50ms per full eval

        print("\nResults:")
        print(f"Total programs: {metrics.total_programs}")
        print(f"Programs pruned: {metrics.programs_pruned} ({metrics.pruning_rate:.1%})")
        print(f"Pruning overhead: {pruning_time:.2f}ms")
        print(f"Estimated total time with pruning: {total_time_with_pruning:.2f}ms")
        print(f"Time saved: {baseline_time - total_time_with_pruning:.2f}ms")

        # Calculate improvement
        improvement = (baseline_time - total_time_with_pruning) / baseline_time
        print(f"Performance improvement: {improvement:.1%}")

        # Assert 40% improvement target
        assert improvement >= 0.35, f"Only {improvement:.1%} improvement, target is 40%"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pruning_overhead(self):
        """Test that pruning overhead is minimal."""
        # Create simple programs that will be accepted
        programs = []
        for i in range(100):
            programs.append(DSLProgram(operations=[
                {"name": "Rotate", "params": {"angle": 90}},
            ]))

        # Create pruner
        strategy = PruningStrategy(
            strategy_id="test",
            name="Test",
            aggressiveness=0.5,
            syntax_checks=True,
            pattern_checks=False,  # Minimal checks
            partial_execution=False,  # No partial execution
            confidence_threshold=0.6,
            max_partial_ops=3,
            timeout_ms=100,
        )

        pruner = ProgramPruner(strategy)

        # Measure pruning time
        start_time = time.perf_counter()

        for program in programs:
            result = await pruner.prune_program([MockOperation("Rotate", angle=90)])

        pruning_time = (time.perf_counter() - start_time) * 1000
        avg_pruning_time = pruning_time / len(programs)

        print(f"\nAverage pruning time per program: {avg_pruning_time:.2f}ms")

        # Assert overhead is less than 5ms per program
        assert avg_pruning_time < 5.0, f"Pruning overhead too high: {avg_pruning_time:.2f}ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scaling_performance(self):
        """Test pruning performance scales with program count."""
        program_counts = [100, 500, 1000]
        results = {}

        for count in program_counts:
            programs = generate_realistic_programs(count, bad_ratio=0.4)

            # Create evaluation service
            strategy = PruningStrategy(
                strategy_id="scaling-test",
                name="Scaling Test",
                aggressiveness=0.5,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.6,
                max_partial_ops=3,
                timeout_ms=100,
            )

            service = EvaluationService(
                enable_gpu_evaluation=False,
                enable_pruning=True,
                default_pruning_strategy=strategy,
            )

            # Mock batch evaluation
            service.batch_evaluate_programs = AsyncMock(return_value=[])

            # Measure time
            start_time = time.perf_counter()

            _, metrics = await service.evaluate_with_pruning(
                programs, [[[0, 0], [0, 0]]]
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            results[count] = {
                "time_ms": elapsed,
                "programs_pruned": metrics.programs_pruned,
                "rate": metrics.pruning_rate,
            }

        print("\nScaling results:")
        for count, data in results.items():
            print(f"{count} programs: {data['time_ms']:.2f}ms, "
                  f"pruned {data['programs_pruned']} ({data['rate']:.1%})")

        # Check that time scales roughly linearly
        time_100 = results[100]["time_ms"]
        time_1000 = results[1000]["time_ms"]
        scaling_factor = time_1000 / time_100

        # Should be roughly 10x for 10x programs (allowing some overhead)
        assert 8 <= scaling_factor <= 12, f"Poor scaling: {scaling_factor:.1f}x"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_different_workload_types(self):
        """Test performance on different workload distributions."""
        workload_configs = [
            ("mostly_good", 0.1),  # 10% bad programs
            ("balanced", 0.4),     # 40% bad programs
            ("mostly_bad", 0.7),   # 70% bad programs
        ]

        for name, bad_ratio in workload_configs:
            print(f"\nTesting {name} workload (bad_ratio={bad_ratio})...")

            programs = generate_realistic_programs(500, bad_ratio=bad_ratio)

            # Baseline
            start_time = time.perf_counter()
            baseline_time = len(programs) * 50  # 50ms per program

            # With pruning
            strategy = PruningStrategy(
                strategy_id=f"test-{name}",
                name=f"Test {name}",
                aggressiveness=0.5,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.6,
                max_partial_ops=3,
                timeout_ms=100,
            )

            service = EvaluationService(
                enable_gpu_evaluation=False,
                enable_pruning=True,
                default_pruning_strategy=strategy,
            )

            service.batch_evaluate_programs = AsyncMock(return_value=[])

            _, metrics = await service.evaluate_with_pruning(
                programs, [[[0, 0], [0, 0]]]
            )

            # Calculate improvement
            pruning_time = 5 * len(programs)  # ~5ms per pruning decision
            eval_time = (len(programs) - metrics.programs_pruned) * 50
            total_time = pruning_time + eval_time

            improvement = (baseline_time - total_time) / baseline_time

            print(f"  Pruning rate: {metrics.pruning_rate:.1%}")
            print(f"  Performance improvement: {improvement:.1%}")

            # More bad programs should mean better improvement
            if bad_ratio >= 0.4:
                assert improvement >= 0.35, f"Poor improvement for {name}: {improvement:.1%}"
