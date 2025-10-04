"""
Unit tests for DSL program serialization and deserialization.

Tests cover serialization, deserialization, validation, hashing, and caching
functionality for DSL programs.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.domain.dsl.base import Operation, OperationRegistry, OperationResult
from src.domain.dsl.geometric import FlipOperation, RotateOperation
from src.domain.dsl.serialization import (
    CachedProgram,
    DSLProgramCache,
    DSLProgramSerializer,
    SerializationError,
    ValidationError,
)
from src.domain.dsl.types import Grid


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def execute(self, grid: Grid, context=None) -> OperationResult:
        return OperationResult(success=True, grid=grid)

    @classmethod
    def get_name(cls) -> str:
        return "mock_op"

    @classmethod
    def get_description(cls) -> str:
        return "Mock operation for testing"

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Any]:
        return {
            "test_param": {
                "type": "string",
                "required": True,
                "description": "Test parameter"
            },
            "optional_param": {
                "type": "integer",
                "required": False,
                "default": 42,
                "description": "Optional parameter"
            }
        }


class TestDSLProgramSerializer:
    """Test cases for DSLProgramSerializer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = OperationRegistry()
        self.registry.register(MockOperation)
        self.registry.register(RotateOperation)
        self.registry.register(FlipOperation)
        self.serializer = DSLProgramSerializer(self.registry)

    def test_serializer_initialization(self):
        """Test serializer initialization."""
        serializer = DSLProgramSerializer()
        assert serializer.SCHEMA_VERSION == "1.0"
        assert serializer.operation_registry is not None

        serializer_with_registry = DSLProgramSerializer(self.registry)
        assert serializer_with_registry.operation_registry is self.registry

    def test_serialize_simple_program(self):
        """Test serialization of a simple program."""
        operations = [
            MockOperation(test_param="hello"),
            RotateOperation(angle=90)
        ]

        result = self.serializer.serialize_program(operations)

        assert result["version"] == "1.0"
        assert len(result["operations"]) == 2
        assert result["operations"][0]["name"] == "mock_op"
        assert result["operations"][0]["parameters"] == {"test_param": "hello"}
        assert result["operations"][1]["name"] == "rotate"
        assert result["operations"][1]["parameters"] == {"angle": 90}
        assert "hash" in result
        assert "created_at" in result
        assert result["metadata"] == {}

    def test_serialize_with_metadata(self):
        """Test serialization with custom metadata."""
        operations = [MockOperation(test_param="test")]
        metadata = {
            "description": "Test program",
            "complexity": 2,
            "tags": ["test", "simple"]
        }

        result = self.serializer.serialize_program(operations, metadata)

        assert result["metadata"] == metadata

    def test_serialize_empty_program(self):
        """Test serialization of empty program."""
        result = self.serializer.serialize_program([])

        assert result["version"] == "1.0"
        assert result["operations"] == []
        assert "hash" in result

    def test_serialization_error_handling(self):
        """Test error handling during serialization."""
        # Mock operation that fails to get name
        mock_op = Mock()
        mock_op.get_name.side_effect = Exception("Test error")
        mock_op.parameters = {}

        with pytest.raises(SerializationError, match="Failed to serialize program"):
            self.serializer.serialize_program([mock_op])

    def test_deserialize_simple_program(self):
        """Test deserialization of a simple program."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "mock_op",
                    "parameters": {"test_param": "hello"}
                },
                {
                    "name": "rotate",
                    "parameters": {"angle": 180}
                }
            ]
        }

        operations = self.serializer.deserialize_program(program_dict)

        assert len(operations) == 2
        assert operations[0].get_name() == "mock_op"
        assert operations[0].parameters["test_param"] == "hello"
        assert operations[1].get_name() == "rotate"
        assert operations[1].parameters["angle"] == 180

    def test_deserialize_with_hash_validation(self):
        """Test deserialization with hash validation."""
        operations = [MockOperation(test_param="test")]
        program_dict = self.serializer.serialize_program(operations)

        # Should deserialize successfully with correct hash
        result_ops = self.serializer.deserialize_program(program_dict)
        assert len(result_ops) == 1
        assert result_ops[0].get_name() == "mock_op"

    def test_deserialize_corrupted_hash(self):
        """Test deserialization with corrupted hash."""
        program_dict = {
            "version": "1.0",
            "hash": "invalid_hash",
            "operations": [
                {"name": "mock_op", "parameters": {"test_param": "test"}}
            ]
        }

        with pytest.raises(ValidationError, match="Program hash mismatch"):
            self.serializer.deserialize_program(program_dict)

    def test_deserialize_unsupported_version(self):
        """Test deserialization with unsupported version."""
        program_dict = {
            "version": "2.0",
            "operations": []
        }

        with pytest.raises(ValidationError, match="Unsupported schema version"):
            self.serializer.deserialize_program(program_dict)

    def test_deserialize_invalid_operations_field(self):
        """Test deserialization with invalid operations field."""
        program_dict = {
            "version": "1.0",
            "operations": "not a list"
        }

        with pytest.raises(ValidationError, match="'operations' field must be a list"):
            self.serializer.deserialize_program(program_dict)

    def test_deserialize_missing_operation_name(self):
        """Test deserialization with missing operation name."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {"parameters": {"test_param": "test"}}
            ]
        }

        with pytest.raises(ValidationError, match="Operation missing 'name' field"):
            self.serializer.deserialize_program(program_dict)

    def test_deserialize_unknown_operation(self):
        """Test deserialization with unknown operation."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {"name": "unknown_op", "parameters": {}}
            ]
        }

        with pytest.raises(ValidationError, match="Unknown operation"):
            self.serializer.deserialize_program(program_dict)

    def test_deserialize_invalid_parameters(self):
        """Test deserialization with invalid parameters."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "mock_op",
                    "parameters": "not a dict"
                }
            ]
        }

        with pytest.raises(ValidationError, match="Operation parameters must be a dictionary"):
            self.serializer.deserialize_program(program_dict)

    def test_parameter_validation_missing_required(self):
        """Test parameter validation with missing required parameter."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "mock_op",
                    "parameters": {"optional_param": 123}  # Missing test_param
                }
            ]
        }

        with pytest.raises(ValidationError, match="Missing required parameter"):
            self.serializer.deserialize_program(program_dict)

    def test_parameter_validation_unknown_parameter(self):
        """Test parameter validation with unknown parameter."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "mock_op",
                    "parameters": {
                        "test_param": "test",
                        "unknown_param": "value"
                    }
                }
            ]
        }

        with pytest.raises(ValidationError, match="Unknown parameters"):
            self.serializer.deserialize_program(program_dict)

    def test_parameter_type_validation(self):
        """Test parameter type validation."""
        # Test integer type validation
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "rotate",
                    "parameters": {"angle": "not_an_integer"}
                }
            ]
        }

        with pytest.raises(ValidationError, match="must be an integer"):
            self.serializer.deserialize_program(program_dict)

    def test_parameter_valid_values_validation(self):
        """Test parameter valid values validation."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {
                    "name": "rotate",
                    "parameters": {"angle": 45}  # Invalid angle
                }
            ]
        }

        with pytest.raises(ValidationError, match="must be one of"):
            self.serializer.deserialize_program(program_dict)

    def test_compute_program_hash(self):
        """Test program hash computation."""
        program_dict = {
            "version": "1.0",
            "operations": [
                {"name": "mock_op", "parameters": {"test_param": "test"}}
            ],
            "metadata": {"description": "test"}
        }

        hash1 = self.serializer.compute_program_hash(program_dict)
        hash2 = self.serializer.compute_program_hash(program_dict)

        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA-256 hex digest length
        assert all(c in '0123456789abcdef' for c in hash1)  # Valid hex

    def test_compute_program_hash_excludes_metadata(self):
        """Test that hash computation excludes timestamp metadata."""
        program_dict1 = {
            "version": "1.0",
            "operations": [{"name": "mock_op", "parameters": {"test_param": "test"}}],
            "metadata": {"created_at": "2025-01-01T00:00:00Z", "description": "test"}
        }

        program_dict2 = {
            "version": "1.0",
            "operations": [{"name": "mock_op", "parameters": {"test_param": "test"}}],
            "metadata": {"created_at": "2025-01-02T00:00:00Z", "description": "test"}
        }

        hash1 = self.serializer.compute_program_hash(program_dict1)
        hash2 = self.serializer.compute_program_hash(program_dict2)

        assert hash1 == hash2  # Should be same despite different timestamps

    def test_create_program_fingerprint(self):
        """Test program fingerprint creation."""
        operations = [
            MockOperation(test_param="test"),
            RotateOperation(angle=90)
        ]

        fingerprint = self.serializer.create_program_fingerprint(operations)

        assert fingerprint.startswith("dsl_program_")
        assert len(fingerprint) == 28  # "dsl_program_" + 16 hex chars

        # Should be deterministic
        fingerprint2 = self.serializer.create_program_fingerprint(operations)
        assert fingerprint == fingerprint2

    def test_to_json(self):
        """Test JSON string conversion."""
        program_dict = {
            "version": "1.0",
            "operations": [{"name": "test", "parameters": {}}]
        }

        # Test compact JSON
        json_str = self.serializer.to_json(program_dict)
        assert isinstance(json_str, str)
        assert json.loads(json_str) == program_dict

        # Test pretty JSON
        pretty_json = self.serializer.to_json(program_dict, indent=2)
        assert "\n" in pretty_json
        assert json.loads(pretty_json) == program_dict

    def test_from_json(self):
        """Test JSON string parsing."""
        program_dict = {
            "version": "1.0",
            "operations": [{"name": "test", "parameters": {}}]
        }
        json_str = json.dumps(program_dict)

        parsed_dict = self.serializer.from_json(json_str)
        assert parsed_dict == program_dict

    def test_from_json_invalid(self):
        """Test JSON parsing with invalid JSON."""
        with pytest.raises(SerializationError, match="Invalid JSON"):
            self.serializer.from_json("invalid json {")

    def test_json_roundtrip(self):
        """Test complete JSON roundtrip serialization."""
        operations = [
            MockOperation(test_param="hello", optional_param=123),
            RotateOperation(angle=180),
            FlipOperation(direction="horizontal")
        ]
        metadata = {"description": "Test program", "complexity": 3}

        # Serialize to dict
        program_dict = self.serializer.serialize_program(operations, metadata)

        # Convert to JSON and back
        json_str = self.serializer.to_json(program_dict, indent=2)
        parsed_dict = self.serializer.from_json(json_str)

        # Deserialize back to operations
        result_ops = self.serializer.deserialize_program(parsed_dict)

        # Verify operations are equivalent
        assert len(result_ops) == 3
        assert result_ops[0].get_name() == "mock_op"
        assert result_ops[0].parameters == {"test_param": "hello", "optional_param": 123}
        assert result_ops[1].get_name() == "rotate"
        assert result_ops[1].parameters == {"angle": 180}
        assert result_ops[2].get_name() == "flip"
        assert result_ops[2].parameters == {"direction": "horizontal"}


class TestCachedProgram:
    """Test cases for CachedProgram."""

    def test_cached_program_initialization(self):
        """Test cached program initialization."""
        operations = [MockOperation(test_param="test")]
        hash_value = "abc123"
        metadata = {"description": "test"}

        cached = CachedProgram(operations, hash_value, metadata)

        assert cached.operations == operations
        assert cached.hash == hash_value
        assert cached.metadata == metadata
        assert cached.access_count == 0
        assert isinstance(cached.created_at, datetime)
        assert cached.last_accessed == cached.created_at

    def test_mark_accessed(self):
        """Test marking program as accessed."""
        cached = CachedProgram([], "hash", {})
        initial_access_time = cached.last_accessed

        cached.mark_accessed()

        assert cached.access_count == 1
        assert cached.last_accessed >= initial_access_time

        cached.mark_accessed()
        assert cached.access_count == 2

    def test_to_dict(self):
        """Test converting cached program to dictionary."""
        operations = [MockOperation(test_param="test")]
        cached = CachedProgram(operations, "hash123", {"desc": "test"})
        cached.mark_accessed()

        result = cached.to_dict()

        assert result["hash"] == "hash123"
        assert result["metadata"] == {"desc": "test"}
        assert result["access_count"] == 1
        assert "created_at" in result
        assert "last_accessed" in result


class TestDSLProgramCache:
    """Test cases for DSLProgramCache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = DSLProgramCache()
        assert cache.max_size == 1000
        assert cache.size() == 0

        cache_custom = DSLProgramCache(max_size=100)
        assert cache_custom.max_size == 100

    def test_cache_put_get(self):
        """Test putting and getting programs from cache."""
        cache = DSLProgramCache()
        operations = [MockOperation(test_param="test")]
        metadata = {"description": "test"}

        cache.put("hash123", operations, metadata)

        assert cache.size() == 1
        assert cache.contains("hash123")

        retrieved = cache.get("hash123")
        assert retrieved == operations

        # Should mark as accessed
        cached_program = cache._cache["hash123"]
        assert cached_program.access_count == 1

    def test_cache_get_nonexistent(self):
        """Test getting nonexistent program from cache."""
        cache = DSLProgramCache()

        result = cache.get("nonexistent")
        assert result is None
        assert not cache.contains("nonexistent")

    def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        cache = DSLProgramCache(max_size=2)

        # Add first program
        cache.put("hash1", [MockOperation(test_param="1")])
        assert cache.size() == 1

        # Add second program
        cache.put("hash2", [MockOperation(test_param="2")])
        assert cache.size() == 2

        # Access first program to make it more recently used
        cache.get("hash1")

        # Add third program - should evict hash2 (least recently used)
        cache.put("hash3", [MockOperation(test_param="3")])
        assert cache.size() == 2
        assert cache.contains("hash1")
        assert not cache.contains("hash2")
        assert cache.contains("hash3")

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = DSLProgramCache()
        cache.put("hash1", [MockOperation(test_param="1")])
        cache.put("hash2", [MockOperation(test_param="2")])

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert not cache.contains("hash1")
        assert not cache.contains("hash2")

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = DSLProgramCache()

        # Empty cache stats
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["total_accesses"] == 0
        assert stats["average_accesses"] == 0

        # Add programs and access them
        cache.put("hash1", [MockOperation(test_param="1")])
        cache.put("hash2", [MockOperation(test_param="2")])

        cache.get("hash1")  # 1 access
        cache.get("hash1")  # 2 accesses
        cache.get("hash2")  # 1 access

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 1000
        assert stats["total_accesses"] == 3
        assert stats["average_accesses"] == 1.5


class TestIntegrationScenarios:
    """Integration test scenarios for complete workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = OperationRegistry()
        self.registry.register(RotateOperation)
        self.registry.register(FlipOperation)
        self.serializer = DSLProgramSerializer(self.registry)
        self.cache = DSLProgramCache()

    def test_complete_serialize_deserialize_workflow(self):
        """Test complete workflow from operations to JSON and back."""
        # Create complex program
        operations = [
            RotateOperation(angle=90),
            FlipOperation(direction="horizontal"),
            RotateOperation(angle=180),
            FlipOperation(direction="vertical")
        ]

        metadata = {
            "description": "Complex transformation sequence",
            "complexity": 4,
            "expected_execution_time_ms": 25,
            "tags": ["geometric", "rotation", "mirroring"]
        }

        # 1. Serialize to dictionary
        program_dict = self.serializer.serialize_program(operations, metadata)

        # 2. Convert to JSON
        json_str = self.serializer.to_json(program_dict, indent=2)

        # 3. Parse JSON back to dictionary
        parsed_dict = self.serializer.from_json(json_str)

        # 4. Deserialize to operations
        result_operations = self.serializer.deserialize_program(parsed_dict)

        # 5. Verify integrity
        assert len(result_operations) == 4
        assert result_operations[0].get_name() == "rotate"
        assert result_operations[0].parameters["angle"] == 90
        assert result_operations[1].get_name() == "flip"
        assert result_operations[1].parameters["direction"] == "horizontal"
        assert result_operations[2].get_name() == "rotate"
        assert result_operations[2].parameters["angle"] == 180
        assert result_operations[3].get_name() == "flip"
        assert result_operations[3].parameters["direction"] == "vertical"

        # Verify metadata preservation
        assert parsed_dict["metadata"]["description"] == metadata["description"]
        assert parsed_dict["metadata"]["complexity"] == metadata["complexity"]

    def test_caching_workflow(self):
        """Test caching workflow with serialization."""
        operations = [
            RotateOperation(angle=270),
            FlipOperation(direction="diagonal_main")
        ]

        # Generate fingerprint
        fingerprint = self.serializer.create_program_fingerprint(operations)

        # Check cache miss
        assert not self.cache.contains(fingerprint)
        cached_ops = self.cache.get(fingerprint)
        assert cached_ops is None

        # Store in cache
        self.cache.put(fingerprint, operations, {"cached": True})

        # Check cache hit
        assert self.cache.contains(fingerprint)
        cached_ops = self.cache.get(fingerprint)
        assert cached_ops == operations

        # Verify fingerprint is deterministic
        fingerprint2 = self.serializer.create_program_fingerprint(operations)
        assert fingerprint == fingerprint2

    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        # Test with corrupted JSON
        invalid_json = '{"version": "1.0", "operations": [invalid]}'

        with pytest.raises(SerializationError):
            self.serializer.from_json(invalid_json)

        # Test with valid JSON but invalid program structure
        invalid_program = {
            "version": "999.0",
            "operations": "not_a_list"
        }

        with pytest.raises(ValidationError):
            self.serializer.deserialize_program(invalid_program)

    @patch('src.domain.dsl.serialization.datetime')
    def test_timestamp_handling(self, mock_datetime):
        """Test timestamp handling in serialization."""
        # Mock datetime.now()
        mock_time = datetime(2025, 9, 25, 10, 0, 0, tzinfo=UTC)
        mock_now = Mock()
        mock_now.isoformat.return_value = "2025-09-25T10:00:00+00:00"
        mock_datetime.now.return_value = mock_now

        operations = [RotateOperation(angle=90)]
        program_dict = self.serializer.serialize_program(operations)

        assert program_dict["created_at"] == "2025-09-25T10:00:00Z"

    def test_schema_compliance(self):
        """Test that serialized programs comply with the expected schema."""
        operations = [
            RotateOperation(angle=180),
            FlipOperation(direction="vertical")
        ]

        metadata = {
            "description": "Schema compliance test",
            "complexity": 2,
            "expected_execution_time_ms": 10
        }

        program_dict = self.serializer.serialize_program(operations, metadata)

        # Verify required fields
        assert "version" in program_dict
        assert "operations" in program_dict
        assert "hash" in program_dict
        assert "created_at" in program_dict
        assert "metadata" in program_dict

        # Verify field types
        assert isinstance(program_dict["version"], str)
        assert isinstance(program_dict["operations"], list)
        assert isinstance(program_dict["hash"], str)
        assert isinstance(program_dict["created_at"], str)
        assert isinstance(program_dict["metadata"], dict)

        # Verify operations structure
        for op in program_dict["operations"]:
            assert "name" in op
            assert "parameters" in op
            assert isinstance(op["name"], str)
            assert isinstance(op["parameters"], dict)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = OperationRegistry()
        self.registry.register(MockOperation)
        self.serializer = DSLProgramSerializer(self.registry)

    def test_empty_operations_list(self):
        """Test handling of empty operations list."""
        program_dict = self.serializer.serialize_program([])
        operations = self.serializer.deserialize_program(program_dict)
        assert operations == []

    def test_operations_with_no_parameters(self):
        """Test operations with no parameters."""
        # Mock operation with no required parameters
        class NoParamOperation(Operation):
            def execute(self, grid, context=None):
                return OperationResult(success=True, grid=grid)

            @classmethod
            def get_name(cls):
                return "no_param_op"

            @classmethod
            def get_description(cls):
                return "Operation with no parameters"

            @classmethod
            def get_parameter_schema(cls):
                return {}

        self.registry.register(NoParamOperation)

        operations = [NoParamOperation()]
        program_dict = self.serializer.serialize_program(operations)
        result_ops = self.serializer.deserialize_program(program_dict)

        assert len(result_ops) == 1
        assert result_ops[0].get_name() == "no_param_op"
        assert result_ops[0].parameters == {}

    def test_large_metadata(self):
        """Test handling of large metadata objects."""
        operations = [MockOperation(test_param="test")]
        large_metadata = {
            "description": "x" * 1000,
            "tags": [f"tag_{i}" for i in range(100)],
            "complex_data": {
                "nested": {
                    "deep": {
                        "values": list(range(50))
                    }
                }
            }
        }

        program_dict = self.serializer.serialize_program(operations, large_metadata)
        result_ops = self.serializer.deserialize_program(program_dict)

        assert len(result_ops) == 1
        assert program_dict["metadata"] == large_metadata

    def test_unicode_in_parameters(self):
        """Test handling of unicode characters in parameters."""
        operations = [MockOperation(test_param="Hello 世界 🌍")]
        program_dict = self.serializer.serialize_program(operations)

        json_str = self.serializer.to_json(program_dict)
        parsed_dict = self.serializer.from_json(json_str)
        result_ops = self.serializer.deserialize_program(parsed_dict)

        assert result_ops[0].parameters["test_param"] == "Hello 世界 🌍"

    def test_special_numeric_values(self):
        """Test handling of special numeric values."""
        # Note: JSON doesn't support NaN or Infinity, so these should be rejected
        # or handled gracefully during serialization

        class NumericOperation(Operation):
            def execute(self, grid, context=None):
                return OperationResult(success=True, grid=grid)

            @classmethod
            def get_name(cls):
                return "numeric_op"

            @classmethod
            def get_description(cls):
                return "Operation with numeric parameter"

            @classmethod
            def get_parameter_schema(cls):
                return {
                    "value": {
                        "type": "number",
                        "required": True,
                        "description": "Numeric value"
                    }
                }

        self.registry.register(NumericOperation)

        # Test with normal numbers
        operations = [NumericOperation(value=3.14159)]
        program_dict = self.serializer.serialize_program(operations)
        json_str = self.serializer.to_json(program_dict)

        # Should successfully roundtrip
        parsed_dict = self.serializer.from_json(json_str)
        result_ops = self.serializer.deserialize_program(parsed_dict)
        assert result_ops[0].parameters["value"] == 3.14159
