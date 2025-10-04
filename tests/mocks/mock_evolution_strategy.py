"""Mock Evolution strategy for integration testing."""

import asyncio
import random
from datetime import datetime

import numpy as np

from src.domain.models import ARCTask, ResourceUsage, StrategyOutput, StrategyType
from src.domain.ports.strategy import StrategyPort


class MockEvolutionStrategy(StrategyPort):
    """Mock evolution/program synthesis strategy with configurable behavior.

    Simulates evolutionary search characteristics:
    - Execution time: 4-6 minutes
    - Success rate: ~45%
    - Variable diversity in outputs

    Example:
        strategy = MockEvolutionStrategy(success_rate=0.45, diversity_factor=0.8)
        output = await strategy.solve_task(task)
    """

    def __init__(
        self,
        success_rate: float = 0.45,
        min_delay_ms: int = 240000,
        max_delay_ms: int = 360000,
        deterministic_seed: int = 42,
        diversity_factor: float = 0.5,
    ):
        """Initialize mock evolution strategy.

        Args:
            success_rate: Probability of generating high-confidence output
            min_delay_ms: Minimum execution delay (default: 4 min)
            max_delay_ms: Maximum execution delay (default: 6 min)
            deterministic_seed: Seed for reproducible outputs
            diversity_factor: Amount of diversity in evolved programs (0.0-1.0)
        """
        self.success_rate = success_rate
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.deterministic_seed = deterministic_seed
        self.diversity_factor = diversity_factor
        self._injected_error: str | None = None
        self._rng = random.Random(deterministic_seed)

    async def solve_task(self, task: ARCTask) -> StrategyOutput:
        """Simulate evolutionary search with configurable delay and success rate."""
        if self._injected_error == "timeout":
            await asyncio.sleep(self.max_delay_ms / 1000)
            raise TimeoutError("Mock evolution timeout")

        if self._injected_error == "resource_exhausted":
            raise RuntimeError("Mock evolution resource exhausted")

        delay_ms = self._rng.uniform(self.min_delay_ms, self.max_delay_ms)
        await asyncio.sleep(delay_ms / 1000)

        is_successful = self._rng.random() < self.success_rate

        output_grid = self._generate_deterministic_output(task, is_successful)

        confidence = self._rng.uniform(0.7, 0.9) if is_successful else self._rng.uniform(0.15, 0.45)

        programs_evaluated = self._rng.randint(300, 600)
        generations = self._rng.randint(15, 30)

        api_calls = {
            "tier1": self._rng.randint(50, 100),
            "tier2": self._rng.randint(10, 30),
        }

        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            cpu_seconds=delay_ms / 1000,
            memory_mb=1024.0,
            gpu_memory_mb=None,
            api_calls=api_calls,
            total_tokens=sum(api_calls.values()) * 1000,
            estimated_cost=sum(api_calls.values()) * 0.002,
            timestamp=datetime.now(),
        )

        reasoning_trace = [
            f"Generation {i}: best_fitness={0.5 + i*0.02:.2f}"
            for i in range(min(5, generations))
        ]

        return StrategyOutput(
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            predicted_output=output_grid,
            confidence_score=confidence,
            strategy_metadata={
                "programs_evaluated": programs_evaluated,
                "generations": generations,
                "best_fitness": 0.9 if is_successful else 0.6,
                "diversity": self.diversity_factor,
            },
            reasoning_trace=reasoning_trace,
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
            strategy_type=StrategyType.PROGRAM_SYNTHESIS,
            cpu_seconds=300.0,
            memory_mb=1024.0,
            gpu_memory_mb=None,
            api_calls={"tier1": 75, "tier2": 20},
            total_tokens=95000,
            estimated_cost=0.19,
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
            base_grid = np_rng.randint(0, 10, size=(height, width), dtype=np.int8)
            if self.diversity_factor > 0.5:
                noise_mask = np_rng.random((height, width)) < (self.diversity_factor - 0.5)
                base_grid = np.where(noise_mask, np_rng.randint(0, 10, size=(height, width)), base_grid)
            return base_grid
