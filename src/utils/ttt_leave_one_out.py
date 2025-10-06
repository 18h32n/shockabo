"""
Leave-One-Out Task Generation for TTT Strategy

Implements cyclic leave-one-out training splits for better per-instance adaptation.
Based on MIT TTT research for improved generalization during test-time training.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeaveOneOutSplit:
    """Single leave-one-out training split."""

    split_id: int
    train_indices: list[int]
    val_index: int
    train_examples: list[dict[str, Any]]
    val_example: dict[str, Any]


@dataclass
class LeaveOneOutConfig:
    """Configuration for leave-one-out splitting."""

    min_examples: int = 2
    max_examples: int = 10
    enable_validation: bool = True


class LeaveOneOutGenerator:
    """
    Generate leave-one-out training splits for TTT adaptation.

    This implements the N-fold leave-one-out strategy where:
    - For N training examples, create N adaptation runs
    - Each run uses N-1 examples for training, 1 for validation
    - All examples are used as validation exactly once
    - Validation metrics are tracked per split for better adaptation

    Expected improvement: 3-5% accuracy gain from better generalization.
    """

    def __init__(self, config: LeaveOneOutConfig | None = None):
        """
        Initialize leave-one-out generator.

        Args:
            config: Configuration for leave-one-out splitting
        """
        self.config = config or LeaveOneOutConfig()
        self.validation_metrics: dict[int, dict[str, float]] = {}

    def generate_splits(
        self, train_examples: list[dict[str, Any]]
    ) -> list[LeaveOneOutSplit]:
        """
        Generate all leave-one-out splits for training examples.

        Args:
            train_examples: List of training examples with 'input' and 'output' keys

        Returns:
            List of leave-one-out splits

        Raises:
            ValueError: If input validation fails
        """
        self._validate_input(train_examples)

        n_examples = len(train_examples)
        splits = []

        for val_idx in range(n_examples):
            train_indices = [i for i in range(n_examples) if i != val_idx]
            train_split = [train_examples[i] for i in train_indices]
            val_split = train_examples[val_idx]

            split = LeaveOneOutSplit(
                split_id=val_idx,
                train_indices=train_indices,
                val_index=val_idx,
                train_examples=train_split,
                val_example=val_split,
            )
            splits.append(split)

        logger.info(f"Generated {len(splits)} leave-one-out splits from {n_examples} examples")
        return splits

    def _validate_input(self, train_examples: list[dict[str, Any]]) -> None:
        """
        Validate input training examples.

        Args:
            train_examples: List of training examples

        Raises:
            ValueError: If validation fails
        """
        if not train_examples:
            raise ValueError("train_examples cannot be empty")

        if len(train_examples) < self.config.min_examples:
            raise ValueError(
                f"Need at least {self.config.min_examples} examples, got {len(train_examples)}"
            )

        if len(train_examples) > self.config.max_examples:
            logger.warning(
                f"Using only first {self.config.max_examples} of {len(train_examples)} examples"
            )

        for idx, example in enumerate(train_examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {idx} must be a dict, got {type(example)}")

            if "input" not in example or "output" not in example:
                raise ValueError(f"Example {idx} missing 'input' or 'output' key")

            self._validate_grid(example["input"], f"Example {idx} input")
            self._validate_grid(example["output"], f"Example {idx} output")

    def _validate_grid(self, grid: Any, name: str) -> None:
        """
        Validate grid dimensions and values.

        Args:
            grid: Grid to validate (list of lists)
            name: Name for error messages

        Raises:
            ValueError: If grid validation fails
        """
        if not isinstance(grid, (list, np.ndarray)):
            raise ValueError(f"{name} must be a list or numpy array, got {type(grid)}")

        if isinstance(grid, np.ndarray):
            grid = grid.tolist()

        if not grid:
            raise ValueError(f"{name} cannot be empty")

        if not all(isinstance(row, list) for row in grid):
            raise ValueError(f"{name} must be a 2D list")

        height = len(grid)
        width = len(grid[0]) if grid else 0

        if height > 30 or width > 30:
            raise ValueError(f"{name} exceeds max dimensions (30x30), got ({height}x{width})")

        for row_idx, row in enumerate(grid):
            if len(row) != width:
                raise ValueError(f"{name} has inconsistent row lengths")

            for col_idx, value in enumerate(row):
                if not isinstance(value, (int, np.integer)):
                    raise ValueError(
                        f"{name}[{row_idx}][{col_idx}] must be an integer, got {type(value)}"
                    )
                if not (0 <= value <= 9):
                    raise ValueError(
                        f"{name}[{row_idx}][{col_idx}] must be 0-9, got {value}"
                    )

    def track_validation_metrics(
        self, split_id: int, metrics: dict[str, float]
    ) -> None:
        """
        Track validation metrics for a split.

        Args:
            split_id: Split identifier (0 to N-1)
            metrics: Validation metrics (e.g., accuracy, loss)
        """
        self.validation_metrics[split_id] = metrics
        logger.debug(f"Tracked metrics for split {split_id}: {metrics}")

    def get_aggregated_metrics(self) -> dict[str, float]:
        """
        Get aggregated metrics across all splits.

        Returns:
            Dictionary with mean and std of metrics across splits
        """
        if not self.validation_metrics:
            return {}

        all_metric_names = set()
        for metrics in self.validation_metrics.values():
            all_metric_names.update(metrics.keys())

        aggregated = {}
        for metric_name in all_metric_names:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in self.validation_metrics.values()
            ]
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))

        return aggregated

    def get_best_split(self, metric: str = "accuracy") -> int | None:
        """
        Get split ID with best validation metric.

        Args:
            metric: Metric name to optimize (default: accuracy)

        Returns:
            Split ID with highest metric value, or None if no metrics tracked
        """
        if not self.validation_metrics:
            return None

        best_split = None
        best_value = -float("inf")

        for split_id, metrics in self.validation_metrics.items():
            value = metrics.get(metric, -float("inf"))
            if value > best_value:
                best_value = value
                best_split = split_id

        return best_split

    def reset_metrics(self) -> None:
        """Reset all tracked validation metrics."""
        self.validation_metrics.clear()
        logger.debug("Reset all validation metrics")
