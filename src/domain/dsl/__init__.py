"""
ARC Domain-Specific Language package.

This package provides a comprehensive DSL for expressing ARC grid transformations
through composable operations organized by category.
"""

from .base import (
    ColorOperation,
    CompositeOperation,
    CompositionOperation,
    DSLProgram,
    Operation,
    OperationRegistry,
    OperationResult,
    PatternOperation,
    TransformOperation,
)
from .types import (
    Color,
    ColorMapping,
    Dimensions,
    Direction,
    FlipDirection,
    Grid,
    GridRegion,
    Pattern,
    Position,
    RotationAngle,
    TransformationContext,
)

__all__ = [
    # Types
    'Grid', 'Color', 'Position', 'Dimensions',
    'Direction', 'RotationAngle', 'FlipDirection',
    'GridRegion', 'ColorMapping', 'Pattern', 'TransformationContext',

    # Base classes
    'Operation', 'OperationResult', 'DSLProgram',
    'TransformOperation', 'ColorOperation', 'PatternOperation', 'CompositionOperation',
    'CompositeOperation', 'OperationRegistry'
]
