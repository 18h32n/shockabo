"""Mock Imitation Learning strategy for integration testing."""

import asyncio
import random
from datetime import datetime

import numpy as np

from src.domain.models import ARCTask, ResourceUsage, StrategyOutput, StrategyType
from src.domain.ports.strategy import StrategyPort


class MockImitationStrategy(StrategyPort):
    """Mock imitation learning strategy with configurable behavior.

    Simulates imitation learning characteristics:
    - Execution time: 20-40 seconds (fast inference)
    - Success rate: ~35%
    - Low resource usage

    Example:
        strategy = MockImitationStrategy(success_rate=0.35, min_delay_ms=20000)
        output = await strategy.solve_task(task)
    """

    def __init__(
        self,
        success_rate: float = 0.35,
        min_delay_ms: int = 20000,
        max_delay_ms: int = 40000,
        deterministic_seed: int = 42,
    ):
        """Initialize mock imitation learning strategy.

        Args:
            success_rate: Probability of generating high-confidence output
            min_delay_ms: Minimum execution delay (default: 20 sec)
            max_delay_ms: Maximum execution delay (default: 40 sec)
            deterministic_seed: Seed for reproducible outputs
        """
        self.success_rate = success_rate
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.deterministic_seed = deterministic_seed
        self._injected_error: str | None = None
        self._rng = random.Random(deterministic_seed)

    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """Simulate fast imitation learning inference."""
        if self._injected_error == "timeout":
            await asyncio.sleep(self.max_delay_ms / 1000)
            raise TimeoutError("Mock imitation timeout")

        if self._injected_error == "resource_exhausted":
            raise RuntimeError("Mock imitation resource exhausted")

        delay_ms = self._rng.uniform(self.min_delay_ms, self.max_delay_ms)
        await asyncio.sleep(delay_ms / 1000)

        is_successful = self._rng.random() < self.success_rate

        output_grid = self._generate_deterministic_output(task, is_successful)

        confidence = self._rng.uniform(0.6, 0.8) if is_successful else self._rng.uniform(0.1, 0.4)

        per_pixel_confidence = np.full_like(
            output_grid, confidence, dtype=np.float32
        )

        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.IMITATION_LEARNING,
            cpu_seconds=delay_ms / 1000,
            memory_mb=512.0,
            gpu_memory_mb=1024.0,
            api_calls={},
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now(),
        )

        return StrategyOutput(
            strategy_type=StrategyType.IMITATION_LEARNING,
            predicted_output=output_grid,
            confidence_score=confidence,
            per_pixel_confidence=per_pixel_confidence,
            strategy_metadata={
                "model_name": "imitation_v1",
                "inference_mode": "beam_search",
                "beam_width": 5,
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
            strategy_type=StrategyType.IMITATION_LEARNING,
            cpu_seconds=30.0,
            memory_mb=512.0,
            gpu_memory_mb=1024.0,
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
