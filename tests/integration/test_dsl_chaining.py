"""Integration tests for DSL operation chaining.

This module tests complex operation sequences, error propagation through chains,
performance of long chains, and different operation category combinations.
"""

import time

from src.domain.dsl.color import (
    ColorFilterOperation,
    ColorInvertOperation,
    ColorMapOperation,
    ColorReplaceOperation,
)
from src.domain.dsl.composition import OverlayOperation
from src.domain.dsl.geometric import (
    CropOperation,
    FlipOperation,
    PadOperation,
    RotateOperation,
    TranslateOperation,
)
from src.domain.dsl.pattern import FloodFillOperation, PatternFillOperation


class TestDSLChaining:
    """Test cases for DSL operation chaining."""

    def test_basic_geometric_chain(self):
        """Test basic geometric operation chaining."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # Create chain: rotate 90° → flip horizontal → rotate 90° again
        rotate_90 = RotateOperation(angle=90)
        flip_h = FlipOperation(direction="horizontal")

        composite = rotate_90 >> flip_h >> rotate_90
        result = composite.execute(grid)

        assert result.success
        assert result.execution_time is not None
        assert result.metadata["operations_count"] == 3

        # Verify result has correct dimensions
        assert len(result.grid) == len(grid)
        assert len(result.grid[0]) == len(grid[0])

    def test_color_transformation_chain(self):
        """Test chaining color transformation operations."""
        grid = [
            [1, 2, 1],
            [3, 1, 4],
            [2, 4, 3]
        ]

        # Chain: replace 1→5 → invert colors → filter for specific colors
        replace_op = ColorReplaceOperation(source_color=1, target_color=5)
        invert_op = ColorInvertOperation()
        filter_op = ColorFilterOperation(keep_colors=[0, 1, 2, 3, 4])

        composite = replace_op >> invert_op >> filter_op
        result = composite.execute(grid)

        assert result.success
        assert len(result.grid) == len(grid)
        assert len(result.grid[0]) == len(grid[0])

    def test_mixed_category_chain(self):
        """Test chaining operations from different categories."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]

        # Mixed chain: geometric → color → pattern → geometric
        rotate = RotateOperation(angle=90)
        color_replace = ColorReplaceOperation(source_color=1, target_color=9)
        flood_fill = FloodFillOperation(position=(0, 0), color=0)
        translate = TranslateOperation(offset=(1, 1), fill_color=0)

        composite = rotate >> color_replace >> flood_fill >> translate
        result = composite.execute(grid)

        assert result.success
        assert result.metadata["operations_count"] == 4

    def test_crop_pad_chain(self):
        """Test chaining crop and pad operations."""
        grid = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5]
        ]

        # Chain: crop center → pad edges → crop again
        crop1 = CropOperation(top=1, left=1, bottom=3, right=3)
        pad = PadOperation(top=1, bottom=1, left=1, right=1, fill_color=0)
        crop2 = CropOperation(top=0, left=0, bottom=2, right=2)

        composite = crop1 >> pad >> crop2
        result = composite.execute(grid)

        assert result.success
        assert len(result.grid) == 3
        assert len(result.grid[0]) == 3

    def test_long_operation_chain(self):
        """Test performance of long operation chains."""
        grid = [[i % 10 for i in range(j, j + 5)] for j in range(5)]

        # Create a long chain of operations
        operations = []
        for i in range(10):
            if i % 3 == 0:
                operations.append(RotateOperation(angle=90))
            elif i % 3 == 1:
                operations.append(FlipOperation(direction="horizontal"))
            else:
                operations.append(ColorReplaceOperation(source_color=i % 10, target_color=(i + 1) % 10))

        # Build composite operation
        composite = operations[0]
        for op in operations[1:]:
            composite = composite >> op

        start_time = time.time()
        result = composite.execute(grid)
        execution_time = time.time() - start_time

        assert result.success
        assert execution_time < 1.0  # Should complete within 1 second
        assert result.metadata["operations_count"] == 10

    def test_pattern_operation_chain(self):
        """Test chaining pattern-based operations."""
        grid = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]

        # Pattern chain: flood fill → pattern detection → pattern fill
        flood_op = FloodFillOperation(position=(0, 0), color=2)

        # Create a simple pattern to fill
        pattern = [[2, 3], [3, 2]]
        pattern_fill = PatternFillOperation(pattern=pattern, fill_color=5)

        composite = flood_op >> pattern_fill
        result = composite.execute(grid)

        assert result.success

    def test_chain_with_translate_and_overlay(self):
        """Test complex chain with translation and overlay operations."""
        base_grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

        overlay_grid = [
            [2, 0],
            [0, 2]
        ]

        # Chain: translate → overlay with another grid
        translate = TranslateOperation(offset=(1, 1), fill_color=0)
        overlay = OverlayOperation(overlay_grid=overlay_grid, position=(0, 0), blend_mode="replace")

        composite = translate >> overlay
        result = composite.execute(base_grid)

        assert result.success
        assert len(result.grid) == len(base_grid)

    def test_identity_operation_chains(self):
        """Test chains that should result in identity transformations."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # Identity chain 1: rotate 360 degrees
        rotate_90_1 = RotateOperation(angle=90)
        rotate_90_2 = RotateOperation(angle=90)
        rotate_90_3 = RotateOperation(angle=90)
        rotate_90_4 = RotateOperation(angle=90)

        composite1 = rotate_90_1 >> rotate_90_2 >> rotate_90_3 >> rotate_90_4
        result1 = composite1.execute(grid)

        assert result1.success
        assert result1.grid == grid

        # Identity chain 2: flip horizontal twice
        flip_h1 = FlipOperation(direction="horizontal")
        flip_h2 = FlipOperation(direction="horizontal")

        composite2 = flip_h1 >> flip_h2
        result2 = composite2.execute(grid)

        assert result2.success
        assert result2.grid == grid

    def test_complex_color_mapping_chain(self):
        """Test complex color mapping chain."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # Complex color chain: map colors → replace specific → invert → map again
        color_map1 = {i: (i + 2) % 10 for i in range(10)}
        color_map2 = {i: (i * 2) % 10 for i in range(10)}

        map_op1 = ColorMapOperation(mapping=color_map1)
        replace_op = ColorReplaceOperation(source_color=3, target_color=0)
        invert_op = ColorInvertOperation()
        map_op2 = ColorMapOperation(mapping=color_map2)

        composite = map_op1 >> replace_op >> invert_op >> map_op2
        result = composite.execute(grid)

        assert result.success
        assert len(result.grid) == len(grid)
        assert len(result.grid[0]) == len(grid[0])


class TestDSLErrorPropagation:
    """Test error handling and propagation in operation chains."""

    def test_error_stops_chain_execution(self):
        """Test that an error in the chain stops further execution."""
        grid = [
            [1, 2],
            [3, 4]
        ]

        # Create a chain where the second operation will fail
        rotate = RotateOperation(angle=90)
        # This crop will fail because bounds are invalid
        invalid_crop = CropOperation(top=0, left=0, bottom=10, right=10)
        flip = FlipOperation(direction="horizontal")

        composite = rotate >> invalid_crop >> flip
        result = composite.execute(grid)

        assert not result.success
        assert "out of range" in result.error_message.lower()
        assert "operation 1" in result.error_message.lower()  # Should indicate which operation failed

    def test_empty_grid_error_propagation(self):
        """Test error propagation with empty grids."""
        empty_grid = []

        rotate = RotateOperation(angle=90)
        flip = FlipOperation(direction="horizontal")

        composite = rotate >> flip
        result = composite.execute(empty_grid)

        assert not result.success
        assert "empty" in result.error_message.lower()

    def test_chain_partial_execution_tracking(self):
        """Test that we can track how far in a chain execution got."""
        grid = [[1, 2], [3, 4]]

        # Create chain with failing operation in the middle
        op1 = RotateOperation(angle=90)  # Will succeed
        op2 = CropOperation(top=0, left=0, bottom=5, right=5)  # Will fail
        op3 = FlipOperation(direction="horizontal")  # Won't execute

        composite = op1 >> op2 >> op3
        result = composite.execute(grid)

        assert not result.success
        assert "operation 1" in result.error_message  # Failed at operation index 1
        assert result.execution_time is not None


class TestDSLPerformanceIntegration:
    """Test performance characteristics of operation chains."""

    def test_chain_performance_scales_linearly(self):
        """Test that chain execution time scales approximately linearly."""
        grid = [[i % 10 for i in range(j, j + 10)] for j in range(10)]

        # Test chains of different lengths
        base_op = RotateOperation(angle=90)

        # Short chain (3 operations)
        short_chain = base_op >> base_op >> base_op
        start_time = time.time()
        result_short = short_chain.execute(grid)
        short_time = time.time() - start_time

        assert result_short.success

        # Long chain (9 operations)
        long_chain = short_chain >> short_chain >> short_chain
        start_time = time.time()
        result_long = long_chain.execute(grid)
        long_time = time.time() - start_time

        assert result_long.success
        # Long chain should take roughly 3x as long (with some tolerance)
        assert long_time < short_time * 5  # Allow for overhead

    def test_memory_usage_in_long_chains(self):
        """Test memory usage doesn't explode in long chains."""
        import os

        import psutil

        grid = [[1, 2, 3] for _ in range(3)]

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create very long chain
        ops = [RotateOperation(angle=90) for _ in range(20)]
        composite = ops[0]
        for op in ops[1:]:
            composite = composite >> op

        result = composite.execute(grid)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        assert result.success
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024

    def test_chain_execution_timeout_handling(self):
        """Test that chain execution respects timeout limits."""
        grid = [[i % 10 for i in range(j, j + 20)] for j in range(20)]

        # Create a very long chain that should complete quickly
        ops = [FlipOperation(direction="horizontal") for _ in range(100)]
        composite = ops[0]
        for op in ops[1:]:
            composite = composite >> op

        start_time = time.time()
        result = composite.execute(grid)
        execution_time = time.time() - start_time

        assert result.success
        # Even with 100 operations, should complete well under 1 second
        assert execution_time < 1.0


class TestDSLCategoryInteractions:
    """Test interactions between different operation categories."""

    def test_geometric_color_interaction(self):
        """Test interaction between geometric and color operations."""
        grid = [
            [1, 2, 1],
            [3, 1, 3],
            [1, 2, 1]
        ]

        # Geometric transformation followed by color transformation
        rotate = RotateOperation(angle=90)
        color_replace = ColorReplaceOperation(source_color=1, target_color=5)

        composite = rotate >> color_replace
        result = composite.execute(grid)

        assert result.success

        # Manually verify: rotate then replace
        rotated = rotate.execute(grid)
        expected = color_replace.execute(rotated.grid)

        assert result.grid == expected.grid

    def test_pattern_geometric_interaction(self):
        """Test interaction between pattern and geometric operations."""
        grid = [
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0]
        ]

        # Pattern operation followed by geometric transformation
        flood_fill = FloodFillOperation(position=(1, 1), color=9)
        flip = FlipOperation(direction="horizontal")

        composite = flood_fill >> flip
        result = composite.execute(grid)

        assert result.success

    def test_all_categories_chain(self):
        """Test chain involving all operation categories."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]

        # Create a chain with operations from all categories
        geometric_op = RotateOperation(angle=90)
        color_op = ColorReplaceOperation(source_color=1, target_color=9)
        pattern_op = FloodFillOperation(position=(0, 0), color=0)
        composition_op = CropOperation(top=1, left=1, bottom=2, right=2)

        composite = geometric_op >> color_op >> pattern_op >> composition_op
        result = composite.execute(grid)

        assert result.success
        assert result.metadata["operations_count"] == 4

    def test_bidirectional_transformation_chain(self):
        """Test chains that transform and then reverse the transformation."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # Transform and reverse: color replace → geometric transform → reverse color replace → reverse geometric
        color_forward = ColorReplaceOperation(source_color=5, target_color=0)
        geometric_forward = RotateOperation(angle=90)
        color_reverse = ColorReplaceOperation(source_color=0, target_color=5)
        geometric_reverse = RotateOperation(angle=270)  # Reverse of 90° is 270°

        composite = color_forward >> geometric_forward >> color_reverse >> geometric_reverse
        result = composite.execute(grid)

        assert result.success
        assert result.grid == grid  # Should return to original
