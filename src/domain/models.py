"""Domain models for ARC Prize 2025 competition."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ARCTask:
    """Core ARC task representation."""

    task_id: str
    task_source: str  # 'training', 'evaluation', 'test'
    difficulty_level: str = "unknown"  # 'easy', 'medium', 'hard', 'unknown'
    train_examples: list[dict[str, list[list[int]]]] = field(default_factory=list)
    test_input: list[list[int]] = field(default_factory=list)
    test_output: list[list[int]] | None = None
    family_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: dict[str, Any], task_id: str, task_source: str = "training") -> "ARCTask":
        """Create ARCTask from dictionary format (compatible with task_loader.py)."""
        train_examples = data.get("train", [])
        test_examples = data.get("test", [])

        test_input = test_examples[0]["input"] if test_examples else []
        test_output = test_examples[0].get("output") if test_examples else None

        return cls(
            task_id=task_id,
            task_source=task_source,
            train_examples=train_examples,
            test_input=test_input,
            test_output=test_output
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format (compatible with task_loader.py)."""
        test_example = {"input": self.test_input}
        if self.test_output is not None:
            test_example["output"] = self.test_output

        return {
            "train": self.train_examples,
            "test": [test_example]
        }

    def get_grid_dimensions(self) -> dict[str, list[tuple]]:
        """Get dimensions of all grids in the task."""
        dimensions = {"train_input": [], "train_output": [], "test_input": [], "test_output": []}

        for example in self.train_examples:
            input_grid = example["input"]
            dimensions["train_input"].append((len(input_grid), len(input_grid[0])))

            if "output" in example:
                output_grid = example["output"]
                dimensions["train_output"].append((len(output_grid), len(output_grid[0])))

        if self.test_input:
            dimensions["test_input"].append((len(self.test_input), len(self.test_input[0])))

        if self.test_output:
            dimensions["test_output"].append((len(self.test_output), len(self.test_output[0])))

        return dimensions

    def get_memory_usage_estimate(self) -> int:
        """Estimate memory usage in bytes."""
        total_cells = 0

        # Count training example cells
        for example in self.train_examples:
            input_grid = example["input"]
            total_cells += len(input_grid) * len(input_grid[0])

            if "output" in example:
                output_grid = example["output"]
                total_cells += len(output_grid) * len(output_grid[0])

        # Count test cells
        if self.test_input:
            total_cells += len(self.test_input) * len(self.test_input[0])

        if self.test_output:
            total_cells += len(self.test_output) * len(self.test_output[0])

        # Assume 4 bytes per integer + overhead
        return total_cells * 4 + 1000  # 1KB overhead per task
