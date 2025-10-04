"""
Base classes and interfaces for the ARC DSL system.

This module defines the core abstractions for DSL operations, including the base
Operation interface, result types, and program representation structures.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from src.domain.dsl.types import Grid, TransformationContext


@dataclass
class OperationResult:
    """
    Result of executing a DSL operation.

    Attributes:
        success: Whether the operation completed successfully
        grid: The resulting grid after the operation
        error_message: Error description if operation failed
        execution_time: Time taken to execute the operation in seconds
        metadata: Additional information about the operation execution
    """
    success: bool
    grid: Grid
    error_message: str | None = None
    execution_time: float | None = None
    metadata: dict[str, Any] | None = None


T = TypeVar('T', bound='Operation')


class Operation(ABC):
    """
    Abstract base class for all DSL operations.

    Operations are the fundamental building blocks of DSL programs. Each operation
    transforms an input grid into an output grid according to specific rules.
    Operations must be composable and type-safe.
    """

    def __init__(self, **parameters: Any):
        """
        Initialize the operation with parameters.

        Args:
            **parameters: Operation-specific parameters
        """
        self.parameters = parameters
        self._validate_parameters()

    @abstractmethod
    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """
        Execute the operation on the input grid.

        Args:
            grid: The input grid to transform
            context: Optional context information for the transformation

        Returns:
            OperationResult containing the transformed grid and metadata
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the name of this operation for registry purposes.

        Returns:
            Unique name identifying this operation type
        """
        pass

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """
        Get a human-readable description of what this operation does.

        Returns:
            Description of the operation's purpose and behavior
        """
        pass

    @classmethod
    @abstractmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """
        Get the schema defining valid parameters for this operation.

        Returns:
            Dictionary describing parameter names, types, and constraints
        """
        pass

    def _validate_parameters(self) -> None:
        """
        Validate that the provided parameters are valid for this operation.

        Raises:
            ValueError: If parameters are invalid
        """
        schema = self.get_parameter_schema()

        # Check for required parameters
        required_params = {k for k, v in schema.items() if v.get('required', False)}
        missing_params = required_params - set(self.parameters.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Check for unknown parameters
        unknown_params = set(self.parameters.keys()) - set(schema.keys())
        if unknown_params:
            raise ValueError(f"Unknown parameters: {unknown_params}")

    def _execute_with_timing(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """
        Execute the operation with timing measurement.

        Args:
            grid: The input grid
            context: Optional transformation context

        Returns:
            OperationResult with execution time recorded
        """
        start_time = time.time()
        try:
            result = self.execute(grid, context)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return OperationResult(
                success=False,
                grid=grid,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def compose_with(self, other: Operation) -> CompositeOperation:
        """
        Create a composite operation by chaining this operation with another.

        Args:
            other: The operation to chain after this one

        Returns:
            CompositeOperation representing the chained operations
        """
        return CompositeOperation([self, other])

    def __rshift__(self, other: Operation) -> CompositeOperation:
        """
        Operator overload for >> to enable chaining syntax: op1 >> op2.

        Args:
            other: The operation to chain after this one

        Returns:
            CompositeOperation representing the chained operations
        """
        return self.compose_with(other)


class TransformOperation(Operation):
    """
    Base class for geometric transformation operations.

    Transform operations modify the spatial arrangement of grid elements
    without changing the colors themselves (rotation, mirroring, etc.).
    """

    @classmethod
    def get_category(cls) -> str:
        """Get the operation category."""
        return "transform"


class ColorOperation(Operation):
    """
    Base class for color manipulation operations.

    Color operations modify the colors in the grid while typically
    preserving spatial relationships (color mapping, filtering, etc.).
    """

    @classmethod
    def get_category(cls) -> str:
        """Get the operation category."""
        return "color"


class PatternOperation(Operation):
    """
    Base class for pattern-based operations.

    Pattern operations work with recurring structures or templates
    in the grid (pattern matching, filling, replacement, etc.).
    """

    @classmethod
    def get_category(cls) -> str:
        """Get the operation category."""
        return "pattern"


class CompositionOperation(Operation):
    """
    Base class for grid composition and decomposition operations.

    Composition operations combine multiple grids or extract sub-grids
    (overlay, concatenation, region extraction, etc.).
    """

    @classmethod
    def get_category(cls) -> str:
        """Get the operation category."""
        return "composition"


class CompositeOperation(Operation):
    """
    Represents a sequence of operations chained together.

    Composite operations enable building complex transformations
    from simpler building blocks while maintaining type safety.
    """

    def __init__(self, operations: list[Operation]):
        """
        Initialize the composite operation.

        Args:
            operations: List of operations to execute in sequence
        """
        if not operations:
            raise ValueError("Composite operation requires at least one operation")

        self.operations = operations
        super().__init__(operations=operations)

    def execute(self, grid: Grid, context: TransformationContext | None = None) -> OperationResult:
        """
        Execute all operations in sequence.

        Args:
            grid: The input grid
            context: Optional transformation context

        Returns:
            OperationResult from the final operation in the sequence
        """
        current_grid = grid
        total_time = 0.0

        for i, operation in enumerate(self.operations):
            # Update context for current step
            if context:
                context.current_grid = current_grid
                context.step_number = i

            result = operation._execute_with_timing(current_grid, context)

            if not result.success:
                return OperationResult(
                    success=False,
                    grid=current_grid,
                    error_message=f"Operation {i} ({operation.get_name()}) failed: {result.error_message}",
                    execution_time=total_time + (result.execution_time or 0)
                )

            current_grid = result.grid
            total_time += result.execution_time or 0

        return OperationResult(
            success=True,
            grid=current_grid,
            execution_time=total_time,
            metadata={"operations_count": len(self.operations)}
        )

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this operation."""
        return "composite"

    @classmethod
    def get_description(cls) -> str:
        """Get the description of this operation."""
        return "Executes a sequence of operations in order"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        """Get the parameter schema for this operation."""
        return {
            "operations": {
                "type": "list",
                "required": True,
                "description": "List of operations to execute in sequence"
            }
        }

    def __rshift__(self, other: Operation) -> CompositeOperation:
        """
        Extend the composite operation with another operation.

        Args:
            other: The operation to add to the sequence

        Returns:
            New CompositeOperation with the additional operation
        """
        return CompositeOperation(self.operations + [other])


@dataclass
class DSLProgram:
    """
    Represents a complete DSL program consisting of multiple operations.

    Programs are serializable and can be cached for performance. They contain
    metadata about their expected behavior and execution characteristics.
    """
    operations: list[dict[str, Any]]
    version: str = "1.0"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the program to a dictionary for serialization.

        Returns:
            Dictionary representation of the program
        """
        return {
            "version": self.version,
            "operations": self.operations,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DSLProgram:
        """
        Create a program from a dictionary representation.

        Args:
            data: Dictionary containing program data

        Returns:
            DSLProgram instance

        Raises:
            ValueError: If the dictionary format is invalid
        """
        if "operations" not in data:
            raise ValueError("Program dictionary must contain 'operations' field")

        return cls(
            operations=data["operations"],
            version=data.get("version", "1.0"),
            metadata=data.get("metadata")
        )

    def get_operation_count(self) -> int:
        """Get the number of operations in the program."""
        return len(self.operations)

    def get_operation_names(self) -> list[str]:
        """Get a list of operation names in execution order."""
        return [op.get("name", "unknown") for op in self.operations]


class OperationRegistry:
    """
    Registry for managing available DSL operations.

    The registry maintains a catalog of all available operations and provides
    methods for discovering and instantiating them.
    """

    def __init__(self):
        """Initialize the registry."""
        self._operations: dict[str, type[Operation]] = {}

    def register(self, operation_class: type[Operation]) -> None:
        """
        Register an operation class.

        Args:
            operation_class: The operation class to register

        Raises:
            ValueError: If operation name conflicts with existing registration
        """
        name = operation_class.get_name()
        if name in self._operations:
            existing_class = self._operations[name]
            if existing_class != operation_class:
                raise ValueError(f"Operation name '{name}' already registered with different class")

        self._operations[name] = operation_class

    def get_operation(self, name: str) -> type[Operation] | None:
        """
        Get an operation class by name.

        Args:
            name: The operation name

        Returns:
            Operation class or None if not found
        """
        return self._operations.get(name)

    def get_all_operations(self) -> dict[str, type[Operation]]:
        """
        Get all registered operations.

        Returns:
            Dictionary mapping operation names to classes
        """
        return self._operations.copy()

    def get_operations_by_category(self, category: str) -> dict[str, type[Operation]]:
        """
        Get all operations in a specific category.

        Args:
            category: The category to filter by

        Returns:
            Dictionary of operations in the specified category
        """
        result = {}
        for name, op_class in self._operations.items():
            if hasattr(op_class, 'get_category') and op_class.get_category() == category:
                result[name] = op_class
        return result
