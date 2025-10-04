"""
Serialization and deserialization for DSL programs.

This module provides functionality to serialize DSL programs to JSON/YAML format
and deserialize them back to executable program objects. It also includes
program validation and fingerprinting for caching purposes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .base import Operation, OperationRegistry


class SerializationError(Exception):
    """Exception raised during serialization/deserialization operations."""
    pass


class ValidationError(SerializationError):
    """Exception raised when program validation fails."""
    pass


class DSLProgramSerializer:
    """
    Handles serialization and deserialization of DSL programs.

    Supports JSON format with version compatibility and program validation.
    """

    SCHEMA_VERSION = "1.0"
    SUPPORTED_VERSIONS = ["1.0"]

    def __init__(self, operation_registry: OperationRegistry | None = None):
        """
        Initialize the serializer.

        Args:
            operation_registry: Registry for validating operation names during deserialization
        """
        self.operation_registry = operation_registry or OperationRegistry()

    def serialize_program(self, operations: list[Operation], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Serialize a list of operations to a dictionary.

        Args:
            operations: List of Operation instances to serialize
            metadata: Optional metadata to include in the program

        Returns:
            Dictionary containing serialized program

        Raises:
            SerializationError: If serialization fails
        """
        try:
            # Serialize operations
            serialized_ops = []
            for op in operations:
                op_dict = {
                    "name": op.get_name(),
                    "parameters": op.parameters
                }
                serialized_ops.append(op_dict)

            # Create program dictionary
            program_dict = {
                "version": self.SCHEMA_VERSION,
                "operations": serialized_ops,
                "metadata": metadata or {},
                "created_at": datetime.now(UTC).isoformat().replace('+00:00', 'Z')
            }

            # Add program hash
            program_dict["hash"] = self.compute_program_hash(program_dict)

            return program_dict

        except Exception as e:
            raise SerializationError(f"Failed to serialize program: {e}") from e

    def deserialize_program(self, program_dict: dict[str, Any], validate: bool = True) -> list[Operation]:
        """
        Deserialize a program dictionary to a list of operations.

        Args:
            program_dict: Dictionary containing serialized program
            validate: Whether to validate operations against registry

        Returns:
            List of Operation instances

        Raises:
            ValidationError: If program validation fails
            SerializationError: If deserialization fails
        """
        try:
            # Validate schema version
            version = program_dict.get("version", "1.0")
            if version not in self.SUPPORTED_VERSIONS:
                raise ValidationError(f"Unsupported schema version: {version}")

            # Extract operations
            operations_data = program_dict.get("operations", [])
            if not isinstance(operations_data, list):
                raise ValidationError("'operations' field must be a list")

            # Validate hash if present
            if "hash" in program_dict:
                expected_hash = self.compute_program_hash(program_dict, exclude_hash=True)
                if program_dict["hash"] != expected_hash:
                    raise ValidationError("Program hash mismatch - program may be corrupted")

            # Deserialize operations
            operations = []
            for i, op_data in enumerate(operations_data):
                try:
                    operation = self._deserialize_operation(op_data, validate)
                    operations.append(operation)
                except Exception as e:
                    raise ValidationError(f"Failed to deserialize operation {i}: {e}") from e

            return operations

        except ValidationError:
            raise
        except Exception as e:
            raise SerializationError(f"Failed to deserialize program: {e}") from e

    def _deserialize_operation(self, op_data: dict[str, Any], validate: bool) -> Operation:
        """
        Deserialize a single operation.

        Args:
            op_data: Dictionary containing operation data
            validate: Whether to validate against registry

        Returns:
            Operation instance

        Raises:
            ValidationError: If operation validation fails
        """
        if not isinstance(op_data, dict):
            raise ValidationError("Operation data must be a dictionary")

        # Extract operation name and parameters
        op_name = op_data.get("name")
        if not op_name:
            raise ValidationError("Operation missing 'name' field")

        parameters = op_data.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValidationError("Operation parameters must be a dictionary")

        # Validate against registry if requested
        if validate:
            op_class = self.operation_registry.get_operation(op_name)
            if not op_class:
                raise ValidationError(f"Unknown operation: {op_name}")

            # Validate parameters against schema
            self._validate_operation_parameters(op_class, parameters)

            # Create operation instance
            return op_class(**parameters)
        else:
            # Create a minimal operation placeholder (for testing)
            # This is not recommended for production use
            raise ValidationError("Operation validation is required")

    def _validate_operation_parameters(self, op_class: type[Operation], parameters: dict[str, Any]) -> None:
        """
        Validate operation parameters against the operation's schema.

        Args:
            op_class: Operation class to validate against
            parameters: Parameters to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            schema = op_class.get_parameter_schema()

            # Check required parameters
            for param_name, param_info in schema.items():
                if param_info.get("required", False) and param_name not in parameters:
                    raise ValidationError(f"Missing required parameter: {param_name}")

            # Check for unknown parameters
            schema_params = set(schema.keys())
            provided_params = set(parameters.keys())
            unknown_params = provided_params - schema_params
            if unknown_params:
                raise ValidationError(f"Unknown parameters: {unknown_params}")

            # Basic type validation
            for param_name, param_value in parameters.items():
                if param_name in schema:
                    param_info = schema[param_name]
                    self._validate_parameter_value(param_name, param_value, param_info)

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Parameter validation failed: {e}") from e

    def _validate_parameter_value(self, param_name: str, value: Any, param_info: dict[str, Any]) -> None:
        """
        Validate a single parameter value.

        Args:
            param_name: Name of the parameter
            value: Value to validate
            param_info: Parameter schema information

        Raises:
            ValidationError: If value is invalid
        """
        param_type = param_info.get("type")

        # Type validation
        if param_type == "integer" and not isinstance(value, int):
            raise ValidationError(f"Parameter {param_name} must be an integer")
        elif param_type == "string" and not isinstance(value, str):
            raise ValidationError(f"Parameter {param_name} must be a string")
        elif param_type == "boolean" and not isinstance(value, bool):
            raise ValidationError(f"Parameter {param_name} must be a boolean")
        elif param_type == "list" and not isinstance(value, list):
            raise ValidationError(f"Parameter {param_name} must be a list")
        elif param_type == "tuple" and not isinstance(value, list | tuple):
            raise ValidationError(f"Parameter {param_name} must be a tuple/list")

        # Range validation
        if "valid_range" in param_info and isinstance(value, int | float):
            min_val, max_val = param_info["valid_range"]
            if not (min_val <= value <= max_val):
                raise ValidationError(f"Parameter {param_name} must be between {min_val} and {max_val}")

        # Valid values validation
        if "valid_values" in param_info and value not in param_info["valid_values"]:
            raise ValidationError(f"Parameter {param_name} must be one of: {param_info['valid_values']}")

    def compute_program_hash(self, program_dict: dict[str, Any], exclude_hash: bool = False) -> str:
        """
        Compute a deterministic hash of the program for caching.

        Args:
            program_dict: Program dictionary to hash
            exclude_hash: Whether to exclude the hash field from computation

        Returns:
            SHA-256 hex digest of the program
        """
        try:
            # Create a copy to avoid modifying original
            hash_dict = program_dict.copy()

            # Remove fields that shouldn't affect the hash
            hash_dict.pop("hash", None)
            hash_dict.pop("created_at", None)
            metadata = hash_dict.get("metadata", {})
            if isinstance(metadata, dict):
                # Remove timestamp metadata but keep functional metadata
                metadata = {k: v for k, v in metadata.items()
                          if not k.endswith("_at") and not k.startswith("timestamp")}
                hash_dict["metadata"] = metadata

            # Sort keys for deterministic hashing
            hash_json = json.dumps(hash_dict, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(hash_json.encode('utf-8')).hexdigest()

        except Exception as e:
            raise SerializationError(f"Failed to compute program hash: {e}") from e

    def create_program_fingerprint(self, operations: list[Operation]) -> str:
        """
        Create a fingerprint of operations without full serialization.

        Args:
            operations: List of operations to fingerprint

        Returns:
            Fingerprint string suitable for caching keys
        """
        try:
            # Create minimal representation for hashing
            fingerprint_data = []
            for op in operations:
                op_data = {
                    "name": op.get_name(),
                    "params": op.parameters
                }
                fingerprint_data.append(op_data)

            fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
            hash_hex = hashlib.sha256(fingerprint_json.encode('utf-8')).hexdigest()
            return f"dsl_program_{hash_hex[:16]}"

        except Exception as e:
            raise SerializationError(f"Failed to create program fingerprint: {e}") from e

    def to_json(self, program_dict: dict[str, Any], indent: int | None = None) -> str:
        """
        Convert program dictionary to JSON string.

        Args:
            program_dict: Program dictionary to serialize
            indent: Optional indentation for pretty printing

        Returns:
            JSON string representation
        """
        try:
            return json.dumps(program_dict, indent=indent, separators=(',', ': ') if indent else (',', ':'))
        except Exception as e:
            raise SerializationError(f"Failed to convert to JSON: {e}") from e

    def from_json(self, json_str: str) -> dict[str, Any]:
        """
        Parse JSON string to program dictionary.

        Args:
            json_str: JSON string to parse

        Returns:
            Program dictionary

        Raises:
            SerializationError: If JSON parsing fails
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}") from e
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON: {e}") from e


class CachedProgram:
    """
    Represents a cached DSL program with metadata.
    """

    def __init__(self, operations: list[Operation], hash_value: str,
                 metadata: dict[str, Any] | None = None):
        """
        Initialize cached program.

        Args:
            operations: List of operations
            hash_value: Program hash for cache lookup
            metadata: Optional program metadata
        """
        self.operations = operations
        self.hash = hash_value
        self.metadata = metadata or {}
        self.created_at = datetime.now(UTC)
        self.access_count = 0
        self.last_accessed = self.created_at

    def mark_accessed(self) -> None:
        """Mark the program as accessed for cache statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert cached program to dictionary."""
        return {
            "hash": self.hash,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }


class DSLProgramCache:
    """
    Simple in-memory cache for DSL programs.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of programs to cache
        """
        self.max_size = max_size
        self._cache: dict[str, CachedProgram] = {}

    def put(self, program_hash: str, operations: list[Operation],
            metadata: dict[str, Any] | None = None) -> None:
        """
        Store a program in the cache.

        Args:
            program_hash: Unique hash identifying the program
            operations: List of operations
            metadata: Optional program metadata
        """
        if len(self._cache) >= self.max_size:
            # Simple LRU eviction - remove least recently accessed
            oldest_key = min(self._cache.keys(),
                            key=lambda k: self._cache[k].last_accessed)
            del self._cache[oldest_key]

        cached_program = CachedProgram(operations, program_hash, metadata)
        self._cache[program_hash] = cached_program

    def get(self, program_hash: str) -> list[Operation] | None:
        """
        Retrieve a program from the cache.

        Args:
            program_hash: Hash of the program to retrieve

        Returns:
            List of operations if found, None otherwise
        """
        cached_program = self._cache.get(program_hash)
        if cached_program:
            cached_program.mark_accessed()
            return cached_program.operations
        return None

    def contains(self, program_hash: str) -> bool:
        """Check if a program is cached."""
        return program_hash in self._cache

    def clear(self) -> None:
        """Clear all cached programs."""
        self._cache.clear()

    def size(self) -> int:
        """Get the number of cached programs."""
        return len(self._cache)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(p.access_count for p in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "average_accesses": total_accesses / len(self._cache) if self._cache else 0
        }


# Example usage and JSON schema documentation
EXAMPLE_PROGRAM_JSON = {
    "version": "1.0",
    "hash": "sha256:abc123...",
    "operations": [
        {
            "name": "rotate",
            "parameters": {"angle": 90}
        },
        {
            "name": "color_map",
            "parameters": {
                "mapping": {"1": 2, "3": 4}
            }
        },
        {
            "name": "flip",
            "parameters": {"direction": "horizontal"}
        }
    ],
    "metadata": {
        "created_at": "2025-09-25T10:00:00Z",
        "description": "Rotate, recolor, and flip",
        "complexity": 3,
        "expected_execution_time_ms": 15
    },
    "created_at": "2025-09-25T10:00:00Z"
}

# JSON Schema definition
PROGRAM_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": "DSL Program Schema",
    "description": "Schema for ARC DSL program serialization",
    "properties": {
        "version": {
            "type": "string",
            "description": "Schema version for backward compatibility",
            "enum": ["1.0"]
        },
        "hash": {
            "type": "string",
            "description": "SHA-256 hash of program for caching and integrity",
            "pattern": "^[a-f0-9]{64}$"
        },
        "operations": {
            "type": "array",
            "description": "Ordered list of operations to execute",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the operation"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Operation-specific parameters"
                    }
                },
                "required": ["name", "parameters"]
            }
        },
        "metadata": {
            "type": "object",
            "description": "Optional metadata about the program",
            "properties": {
                "description": {"type": "string"},
                "complexity": {"type": "integer"},
                "expected_execution_time_ms": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "ISO timestamp when program was created"
        }
    },
    "required": ["version", "operations"],
    "additionalProperties": False
}
