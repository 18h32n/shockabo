"""Data augmentation utilities for ARC tasks with semantic preservation."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from ..domain.models import ARCTask


class GridAugmentor:
    """Grid transformation utilities maintaining data integrity and semantics."""

    @staticmethod
    def rotate_grid_90(grid: list[list[int]]) -> list[list[int]]:
        """Rotate grid 90 degrees clockwise."""
        if not grid or not grid[0]:
            return grid

        rows, cols = len(grid), len(grid[0])
        rotated = [[0 for _ in range(rows)] for _ in range(cols)]

        for i in range(rows):
            for j in range(cols):
                rotated[j][rows - 1 - i] = grid[i][j]

        return rotated

    @staticmethod
    def rotate_grid_180(grid: list[list[int]]) -> list[list[int]]:
        """Rotate grid 180 degrees."""
        if not grid:
            return grid

        return [row[::-1] for row in grid[::-1]]

    @staticmethod
    def rotate_grid_270(grid: list[list[int]]) -> list[list[int]]:
        """Rotate grid 270 degrees clockwise (90 degrees counter-clockwise)."""
        if not grid or not grid[0]:
            return grid

        rows, cols = len(grid), len(grid[0])
        rotated = [[0 for _ in range(rows)] for _ in range(cols)]

        for i in range(rows):
            for j in range(cols):
                rotated[cols - 1 - j][i] = grid[i][j]

        return rotated

    @staticmethod
    def flip_horizontal(grid: list[list[int]]) -> list[list[int]]:
        """Flip grid horizontally (left-right mirror)."""
        return [row[::-1] for row in grid]

    @staticmethod
    def flip_vertical(grid: list[list[int]]) -> list[list[int]]:
        """Flip grid vertically (top-bottom mirror)."""
        return grid[::-1]

    @staticmethod
    def get_grid_signature(grid: list[list[int]]) -> dict[str, Any]:
        """Get grid signature for semantic validation."""
        if not grid or not grid[0]:
            return {}

        flat_grid = [cell for row in grid for cell in row]
        unique_colors = set(flat_grid)
        color_counts = {color: flat_grid.count(color) for color in unique_colors}

        # Spatial patterns
        rows, cols = len(grid), len(grid[0])

        # Check for symmetries
        is_h_symmetric = grid == GridAugmentor.flip_horizontal(grid)
        is_v_symmetric = grid == GridAugmentor.flip_vertical(grid)

        # Color connectivity patterns
        connected_components = GridAugmentor._analyze_connectivity(grid)

        return {
            "dimensions": (rows, cols),
            "unique_colors": sorted(unique_colors),
            "color_counts": color_counts,
            "total_cells": rows * cols,
            "is_horizontally_symmetric": is_h_symmetric,
            "is_vertically_symmetric": is_v_symmetric,
            "connected_components": connected_components
        }

    @staticmethod
    def _analyze_connectivity(grid: list[list[int]]) -> dict[int, int]:
        """Analyze connected components for each color."""
        if not grid or not grid[0]:
            return {}

        rows, cols = len(grid), len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        color_components: dict[int, list[int]] = {}

        def dfs(r: int, c: int, color: int) -> int:
            if (r < 0 or r >= rows or c < 0 or c >= cols or
                visited[r][c] or grid[r][c] != color):
                return 0

            visited[r][c] = True
            size = 1

            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                size += dfs(r + dr, c + dc, color)

            return size

        for i in range(rows):
            for j in range(cols):
                if not visited[i][j]:
                    color = grid[i][j]
                    if color not in color_components:
                        color_components[color] = []

                    component_size = dfs(i, j, color)
                    color_components[color].append(component_size)

        # Return component count for each color
        return {color: len(components) for color, components in color_components.items()}

    @staticmethod
    def validate_transformation_semantics(
        original_grid: list[list[int]],
        transformed_grid: list[list[int]]
    ) -> dict[str, bool]:
        """Validate that transformation preserves semantic properties."""
        orig_sig = GridAugmentor.get_grid_signature(original_grid)
        trans_sig = GridAugmentor.get_grid_signature(transformed_grid)

        return {
            "preserves_colors": orig_sig.get("unique_colors") == trans_sig.get("unique_colors"),
            "preserves_color_counts": orig_sig.get("color_counts") == trans_sig.get("color_counts"),
            "preserves_total_cells": orig_sig.get("total_cells") == trans_sig.get("total_cells"),
            "preserves_connectivity": orig_sig.get("connected_components") == trans_sig.get("connected_components")
        }


class ARCTaskAugmentor:
    """ARC task augmentation with configurable pipelines and semantic preservation."""

    def __init__(self):
        """Initialize task augmentor."""
        self.transformations = {
            "rotate_90": lambda grid: GridAugmentor.rotate_grid_90(grid),
            "rotate_180": lambda grid: GridAugmentor.rotate_grid_180(grid),
            "rotate_270": lambda grid: GridAugmentor.rotate_grid_270(grid),
            "flip_horizontal": lambda grid: GridAugmentor.flip_horizontal(grid),
            "flip_vertical": lambda grid: GridAugmentor.flip_vertical(grid)
        }

    def augment_task(
        self,
        task: ARCTask,
        augmentations: list[str],
        preserve_semantics: bool = True
    ) -> list[ARCTask]:
        """Augment ARC task with specified transformations."""
        augmented_tasks = []

        for aug_name in augmentations:
            if aug_name not in self.transformations:
                continue

            try:
                augmented_task = self._apply_transformation(task, aug_name)

                if preserve_semantics and not self._validate_task_semantics(task, augmented_task):
                    continue

                augmented_tasks.append(augmented_task)

            except Exception as e:
                print(f"Error applying {aug_name} to task {task.task_id}: {e}")

        return augmented_tasks

    def _apply_transformation(self, task: ARCTask, transformation_name: str) -> ARCTask:
        """Apply transformation to entire task."""
        transform_func = self.transformations[transformation_name]

        # Create new task with transformed data
        augmented_task = deepcopy(task)
        augmented_task.task_id = f"{task.task_id}_{transformation_name}"

        # Transform training examples
        for example in augmented_task.train_examples:
            example["input"] = transform_func(example["input"])
            if "output" in example:
                example["output"] = transform_func(example["output"])

        # Transform test data
        if augmented_task.test_input:
            augmented_task.test_input = transform_func(augmented_task.test_input)

        if augmented_task.test_output:
            augmented_task.test_output = transform_func(augmented_task.test_output)

        return augmented_task

    def _validate_task_semantics(self, original: ARCTask, augmented: ARCTask) -> bool:
        """Validate that task augmentation preserves semantic properties."""
        # Check training examples
        for orig_ex, aug_ex in zip(original.train_examples, augmented.train_examples, strict=False):
            # Validate input semantics
            input_validation = GridAugmentor.validate_transformation_semantics(
                orig_ex["input"], aug_ex["input"]
            )
            if not all(input_validation.values()):
                return False

            # Validate output semantics if present
            if "output" in orig_ex and "output" in aug_ex:
                output_validation = GridAugmentor.validate_transformation_semantics(
                    orig_ex["output"], aug_ex["output"]
                )
                if not all(output_validation.values()):
                    return False

        # Check test input
        if original.test_input and augmented.test_input:
            test_validation = GridAugmentor.validate_transformation_semantics(
                original.test_input, augmented.test_input
            )
            if not all(test_validation.values()):
                return False

        return True

    def create_augmentation_pipeline(self, pipeline_config: dict[str, Any]) -> Callable:
        """Create configurable augmentation pipeline."""
        augmentations = pipeline_config.get("transformations", [])
        preserve_semantics = pipeline_config.get("preserve_semantics", True)
        probability = pipeline_config.get("probability", 1.0)

        def pipeline(task: ARCTask) -> list[ARCTask]:
            if np.random.random() > probability:
                return [task]

            return self.augment_task(task, augmentations, preserve_semantics)

        return pipeline

    def batch_augment(
        self,
        tasks: dict[str, ARCTask],
        augmentations: list[str],
        max_augmentations_per_task: int = 2
    ) -> dict[str, ARCTask]:
        """Batch augmentation of multiple tasks."""
        augmented_tasks = {}

        for task_id, task in tasks.items():
            # Keep original
            augmented_tasks[task_id] = task

            # Add augmentations
            aug_tasks = self.augment_task(task, augmentations)

            for i, aug_task in enumerate(aug_tasks[:max_augmentations_per_task]):
                aug_id = f"{task_id}_aug_{i}"
                aug_task.task_id = aug_id
                augmented_tasks[aug_id] = aug_task

        return augmented_tasks

    def get_augmentation_statistics(
        self,
        original_tasks: dict[str, ARCTask],
        augmented_tasks: dict[str, ARCTask]
    ) -> dict[str, Any]:
        """Get statistics about augmentation results."""
        original_count = len(original_tasks)
        augmented_count = len(augmented_tasks)

        # Analyze augmentation types
        aug_type_counts: dict[str, int] = {}
        for task_id in augmented_tasks:
            if "_" in task_id:
                parts = task_id.split("_")
                if len(parts) >= 2:
                    aug_type = parts[-2] if parts[-1].isdigit() or parts[-1] == "aug" else parts[-1]
                    aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

        return {
            "original_task_count": original_count,
            "total_task_count": augmented_count,
            "augmentation_ratio": augmented_count / original_count if original_count > 0 else 0,
            "augmentation_type_counts": aug_type_counts,
            "net_augmented_tasks": augmented_count - original_count
        }


class AugmentationValidator:
    """Validation utilities for augmented data."""

    @staticmethod
    def compare_grids(
        grid1: list[list[int]],
        grid2: list[list[int]],
        tolerance: float = 0.0
    ) -> dict[str, Any]:
        """Compare two grids and return detailed comparison."""
        if not grid1 or not grid2:
            return {"identical": False, "reason": "Empty grid(s)"}

        # Dimension check
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return {"identical": False, "reason": "Different dimensions"}

        # Cell-by-cell comparison
        differences = []
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    differences.append((i, j, grid1[i][j], grid2[i][j]))

        return {
            "identical": len(differences) == 0,
            "differences": differences,
            "difference_count": len(differences),
            "similarity_ratio": 1.0 - (len(differences) / (len(grid1) * len(grid1[0])))
        }

    @staticmethod
    def validate_augmentation_invertibility(
        original_grid: list[list[int]],
        transformation_name: str
    ) -> bool:
        """Validate that transformations are correctly invertible."""
        augmentor = GridAugmentor()

        # Apply transformation
        if transformation_name == "rotate_90":
            transformed = augmentor.rotate_grid_90(original_grid)
            # Apply inverse (3 more 90-degree rotations)
            restored = augmentor.rotate_grid_90(
                augmentor.rotate_grid_90(augmentor.rotate_grid_90(transformed))
            )
        elif transformation_name == "rotate_180":
            transformed = augmentor.rotate_grid_180(original_grid)
            restored = augmentor.rotate_grid_180(transformed)
        elif transformation_name == "flip_horizontal":
            transformed = augmentor.flip_horizontal(original_grid)
            restored = augmentor.flip_horizontal(transformed)
        elif transformation_name == "flip_vertical":
            transformed = augmentor.flip_vertical(original_grid)
            restored = augmentor.flip_vertical(transformed)
        else:
            return False

        comparison = AugmentationValidator.compare_grids(original_grid, restored)
        return bool(comparison["identical"])
