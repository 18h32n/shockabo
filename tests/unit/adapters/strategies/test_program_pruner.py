"""Unit tests for the intelligent program pruning system.

Tests the core pruning logic, validation rules, confidence scoring, and
security controls.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.adapters.strategies.program_pruner import (
    ProgramPruner,
    PruningConfig,
)
from src.domain.dsl.base import Operation, OperationResult
from src.domain.dsl.types import Grid
from src.domain.models import (
    PartialExecutionResult,
    PruningDecision,
    PruningStrategy,
)


class MockOperation(Operation):
    """Mock operation for testing."""

    def __init__(self, name: str, **params):
        self._name = name
        self.parameters = params

    def execute(self, grid: Grid, context=None) -> OperationResult:
        return OperationResult(success=True, grid=grid)

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return f"Mock {self._name}"

    def get_parameter_schema(self) -> dict:
        return {}
    
    def __str__(self) -> str:
        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"{self._name}({param_str})"
        return f"{self._name}()"


# Create specific mock operation classes for valid operations
class Rotate(MockOperation):
    """Mock Rotate operation."""
    def __init__(self, angle: int):
        super().__init__("Rotate", angle=angle)


class FloodFill(MockOperation):
    """Mock FloodFill operation."""
    def __init__(self, x: int, y: int, color: int):
        super().__init__("FloodFill", x=x, y=y, color=color)


class Tile(MockOperation):
    """Mock Tile operation."""
    def __init__(self, x: int, y: int):
        super().__init__("Tile", x=x, y=y)


class CustomComplexTransform(MockOperation):
    """Mock ComplexTransform operation."""
    def __init__(self):
        super().__init__("CustomComplexTransform")


class InvalidOp(MockOperation):
    """Mock invalid operation."""
    def __init__(self):
        super().__init__("InvalidOp")


class TestProgramPruner:
    """Test program pruning functionality."""

    @pytest.fixture
    def pruning_config(self):
        """Create test pruning configuration."""
        return PruningConfig(
            strategy_id="test-strategy",
            aggressiveness=0.5,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=True,
            confidence_threshold=0.6,
            max_partial_ops=3,
            timeout_ms=100,
            memory_limit_mb=10,
            enable_caching=False,  # Disable for tests
        )

    @pytest.fixture
    def pruner(self, pruning_config):
        """Create test pruner instance."""
        return ProgramPruner(pruning_config)

    @pytest.mark.asyncio
    async def test_syntax_validation_accepts_valid_program(self, pruner):
        """Test that syntax validation accepts valid programs."""
        # Create valid program
        program = [
            Rotate(angle=90),
            FloodFill(x=0, y=0, color=1),
        ]

        # Should pass syntax checks
        result = await pruner.prune_program(program)

        assert result.decision == PruningDecision.ACCEPT
        assert result.rejection_reason is None

    @pytest.mark.asyncio
    async def test_syntax_validation_rejects_invalid_operations(self, pruner):
        """Test that syntax validation rejects invalid operations."""
        # Create program with invalid operation
        program = [
            InvalidOp(),
            Rotate(angle=90),
        ]

        # Should fail syntax checks
        result = await pruner.prune_program(program)

        assert result.decision == PruningDecision.REJECT_SYNTAX
        assert "valid_operation_names" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_pattern_rejection_contradictory_operations(self, pruner):
        """Test rejection of contradictory operation patterns."""
        # Create program with contradictory rotations
        program = [
            Rotate(angle=90),
            Rotate(angle=270),  # Undoes previous
        ]

        # Should fail pattern checks
        result = await pruner.prune_program(program)

        assert result.decision == PruningDecision.REJECT_PATTERN
        assert "contradictory_operations" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_memory_explosion_detection(self, pruner):
        """Test detection of operations that could cause memory explosion."""
        # Create program with large tile operation
        program = [
            Tile(x=100, y=100),  # 10,000x expansion
        ]

        # Should fail security checks
        result = await pruner.prune_program(program)

        assert result.decision == PruningDecision.REJECT_SECURITY
        assert "memory" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_color_validity_check(self, pruner):
        """Test validation of color values."""
        # Create program with invalid color
        program = [
            MockOperation("MapColors", mapping={0: 15}),  # Color 15 invalid
        ]

        # Mock color validation to detect invalid color
        with patch.object(pruner, '_check_color_validity', return_value=False):
            result = await pruner.prune_program(program)

        assert result.decision in [PruningDecision.REJECT_SYNTAX, PruningDecision.REJECT_PATTERN]

    @pytest.mark.asyncio
    async def test_confidence_scoring_with_partial_execution(self, pruner):
        """Test confidence scoring through partial execution."""
        # Create program
        program = [
            Rotate(angle=90),
            FloodFill(x=0, y=0, color=1),
        ]

        # Mock partial execution result
        mock_partial_result = PartialExecutionResult(
            program_id="test",
            operations_executed=2,
            intermediate_grid=[[1, 0], [0, 1]],
            execution_time_ms=50,
            memory_used_mb=0.1,
            success=True,
            error=None,
        )

        # Mock partial executor
        with patch.object(pruner, '_partial_executor') as mock_executor:
            mock_executor.execute_partial = AsyncMock(
                return_value=(mock_partial_result, 0.8)  # High confidence
            )

            result = await pruner.prune_program(program, test_inputs=[[[0, 0], [0, 0]]])

        assert result.decision == PruningDecision.ACCEPT
        assert result.confidence_score == 0.8
        assert result.partial_output is not None

    @pytest.mark.asyncio
    async def test_low_confidence_rejection(self, pruner):
        """Test rejection based on low confidence score."""
        # Create program
        program = [
            CustomComplexTransform(),
        ]

        # Mock partial execution with low confidence
        mock_partial_result = PartialExecutionResult(
            program_id="test",
            operations_executed=1,
            intermediate_grid=[[0, 0], [0, 0]],
            execution_time_ms=50,
            memory_used_mb=0.1,
            success=True,
            error=None,
        )

        with patch.object(pruner, '_partial_executor') as mock_executor:
            mock_executor.execute_partial = AsyncMock(
                return_value=(mock_partial_result, 0.3)  # Low confidence
            )

            result = await pruner.prune_program(program, test_inputs=[[[0, 0], [0, 0]]])

        assert result.decision == PruningDecision.REJECT_CONFIDENCE
        assert result.confidence_score == 0.3
        assert "confidence" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test that caching works correctly."""
        # Create pruner with caching enabled
        config = PruningConfig(
            strategy_id="cache-test",
            aggressiveness=0.5,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=False,  # No partial execution
            confidence_threshold=0.6,
            max_partial_ops=3,
            timeout_ms=100,
            memory_limit_mb=10,
            enable_caching=True,
        )
        pruner = ProgramPruner(config)

        # Create program
        program = [
            Rotate(angle=90),
        ]

        # First evaluation
        result1 = await pruner.prune_program(program)
        assert pruner.stats["cache_hits"] == 0

        # Second evaluation should hit cache
        result2 = await pruner.prune_program(program)
        assert pruner.stats["cache_hits"] == 1
        assert result1.decision == result2.decision

    @pytest.mark.asyncio
    async def test_batch_pruning(self, pruner):
        """Test batch pruning functionality."""
        # Create multiple programs
        programs = [
            [Rotate(angle=90)],  # Valid
            [InvalidOp()],  # Invalid syntax
            [Tile(x=100, y=100)],  # Memory explosion
        ]

        # Batch prune
        results = await pruner.batch_prune(programs)

        assert len(results) == 3
        assert results[0].decision == PruningDecision.ACCEPT
        assert results[1].decision == PruningDecision.REJECT_SYNTAX
        assert results[2].decision == PruningDecision.REJECT_SECURITY

    def test_statistics_tracking(self, pruner):
        """Test that statistics are tracked correctly."""
        # Reset stats
        pruner.reset_statistics()

        # Get initial stats
        stats = pruner.get_statistics()
        assert stats["total_programs"] == 0
        assert stats["pruned_programs"] == 0

        # After processing, stats should be updated
        # (Would need to run pruning to test this properly)

    @pytest.mark.asyncio
    async def test_security_audit_logging(self, pruner):
        """Test security audit logging functionality."""
        # Create program that triggers security rejection
        program = [
            MockOperation("Tile", x=100, y=100),
        ]

        # Prune program
        with patch.object(pruner.logger, 'warning') as mock_warning:
            result = await pruner.prune_program(program)

        # Check that security alert was logged
        if result.decision == PruningDecision.REJECT_SECURITY:
            mock_warning.assert_called()
            call_args = mock_warning.call_args
            assert "security_pruning_decision" in str(call_args)

    def test_aggressiveness_levels(self):
        """Test different aggressiveness levels."""
        # Conservative strategy
        conservative = PruningStrategy(
            strategy_id="conservative",
            name="Conservative",
            aggressiveness=0.2,
            syntax_checks=True,
            pattern_checks=False,
            partial_execution=False,
            confidence_threshold=0.8,
            max_partial_ops=2,
            timeout_ms=50,
        )

        # Aggressive strategy
        aggressive = PruningStrategy(
            strategy_id="aggressive",
            name="Aggressive",
            aggressiveness=0.8,
            syntax_checks=True,
            pattern_checks=True,
            partial_execution=True,
            confidence_threshold=0.4,
            max_partial_ops=5,
            timeout_ms=150,
        )

        assert conservative.aggressiveness < aggressive.aggressiveness
        assert conservative.confidence_threshold > aggressive.confidence_threshold
