"""Mock Test-Time Training strategy for integration testing."""

import asyncio
import random
from datetime import datetime

import numpy as np

from src.domain.models import ARCTask, ResourceUsage, StrategyOutput, StrategyType
from src.domain.ports.strategy import StrategyPort


class MockTTTStrategy(StrategyPort):
    """Mock TTT strategy with configurable behavior.

    Simulates Test-Time Training strategy characteristics:
    - Execution time: 3-5 minutes
    - Success rate: ~58%
    - Deterministic outputs based on seed

    Example:
        strategy = MockTTTStrategy(success_rate=0.58, min_delay_ms=180000)
        output = await strategy.solve_task(task)
    """

    def __init__(
        self,
        success_rate: float = 0.58,
        min_delay_ms: int = 180000,
        max_delay_ms: int = 300000,
        deterministic_seed: int = 42,
    ):
        """Initialize mock TTT strategy.

        Args:
            success_rate: Probability of generating high-confidence output
            min_delay_ms: Minimum execution delay (default: 3 min)
            max_delay_ms: Maximum execution delay (default: 5 min)
            deterministic_seed: Seed for reproducible outputs
        """
        self.success_rate = success_rate
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.deterministic_seed = deterministic_seed
        self._injected_error: str | None = None
        self._rng = random.Random(deterministic_seed)

    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """Simulate TTT solving with configurable delay and success rate."""
        if self._injected_error == "timeout":
            await asyncio.sleep(self.max_delay_ms / 1000)
            raise TimeoutError("Mock TTT timeout")

        if self._injected_error == "resource_exhausted":
            raise RuntimeError("Mock TTT resource exhausted")

        delay_ms = self._rng.uniform(self.min_delay_ms, self.max_delay_ms)
        await asyncio.sleep(delay_ms / 1000)

        is_successful = self._rng.random() < self.success_rate

        output_grid = self._generate_deterministic_output(task, is_successful)

        confidence = self._rng.uniform(0.75, 0.95) if is_successful else self._rng.uniform(0.2, 0.5)

        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=delay_ms / 1000,
            memory_mb=2048.0,
            gpu_memory_mb=4096.0,
            api_calls={},
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

        return StrategyOutput(
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            predicted_output=output_grid,
            confidence_score=confidence,
            strategy_metadata={
                "training_epochs": 50,
                "adaptation_loss": 0.05 if is_successful else 0.2,
            },
            processing_time_ms=int(delay_ms),
            resource_usage=resource_usage,
        )

    def get_confidence_estimate(self, task: ARCTask) -> float:
        """Return estimated confidence before execution."""
        return self.success_rate

    def get_resource_estimate(self, task: ARCTask) -> ResourceUsage:
        """Return estimated resource requirements."""
        return ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=240.0,
            memory_mb=2048.0,
            gpu_memory_mb=4096.0,
            api_calls={},
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

    def inject_error(self, error_type: str) -> None:
        """Inject error for robustness testing."""
        self._injected_error = error_type

    def _generate_deterministic_output(
        self, task: ARCTask, is_successful: bool
    ) -> np.ndarray:
        """Generate deterministic output based on task and seed."""
        if not task.test_input:
            return np.zeros((5, 5), dtype=np.int8)

        height = len(task.test_input)
        width = len(task.test_input[0]) if height > 0 else 5

        np_rng = np.random.RandomState(self.deterministic_seed + hash(task.task_id) % 1000000)

        if is_successful and task.test_output:
            return np.array(task.test_output, dtype=np.int8)
        else:
            return np_rng.randint(0, 10, size=(height, width), dtype=np.int8)
