"""Program pruning module for early termination of obviously wrong programs.

This module implements intelligent pruning strategies to identify and reject
programs that are likely to fail before full evaluation, saving compute time.
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from src.adapters.strategies.partial_executor import PartialExecutionConfig, PartialExecutor
from src.domain.dsl.base import Operation
from src.domain.dsl.types import Grid
from src.domain.models import PruningDecision, PruningResult

logger = structlog.get_logger(__name__)


class PruningLevel(Enum):
    """Pruning aggressiveness levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PruningConfig:
    """Configuration for program pruning behavior."""
    strategy_id: str = "default-pruning-v1"
    aggressiveness: float = 0.5  # 0.0 (conservative) to 1.0 (aggressive)
    syntax_checks: bool = True
    pattern_checks: bool = True
    partial_execution: bool = True
    confidence_threshold: float = 0.6
    max_partial_ops: int = 3
    timeout_ms: float = 100.0
    memory_limit_mb: float = 10.0
    enable_caching: bool = True
    cache_size: int = 10000


@dataclass
class ValidationRule:
    """Definition of a validation rule for program pruning."""
    rule_id: str
    name: str
    category: str
    check_function: Any  # Callable[[List[Operation]], bool]
    severity: str = "warning"  # "warning" or "error"
    enabled: bool = True


class ProgramPruner:
    """Intelligent program pruner for early rejection of invalid programs."""

    def __init__(self, config: PruningConfig | None = None):
        """Initialize the program pruner.

        Args:
            config: Pruning configuration (uses defaults if None)
        """
        self.config = config or PruningConfig()
        self.logger = structlog.get_logger(__name__).bind(
            service="program_pruner",
            strategy_id=self.config.strategy_id
        )

        # Initialize pattern database
        self._pattern_database = self._initialize_pattern_database()

        # Initialize validation rules
        self._validation_rules = self._initialize_validation_rules()

        # Caching for repeated programs
        self._cache: dict[str, PruningResult] = {}

        # Initialize partial executor if enabled
        self._partial_executor = None
        if self.config.partial_execution:
            partial_config = PartialExecutionConfig(
                max_operations=self.config.max_partial_ops,
                timeout_ms=self.config.timeout_ms,
                memory_limit_mb=self.config.memory_limit_mb
            )
            self._partial_executor = PartialExecutor(partial_config)

        # Statistics tracking
        self.stats = {
            "total_programs": 0,
            "pruned_programs": 0,
            "syntax_failures": 0,
            "pattern_failures": 0,
            "confidence_failures": 0,
            "cache_hits": 0,
            "total_pruning_time_ms": 0.0
        }

    def _initialize_pattern_database(self) -> dict[str, list[tuple[str, float]]]:
        """Initialize database of known bad program patterns.

        Returns:
            Dictionary mapping pattern categories to list of (pattern, severity) tuples
        """
        return {
            "contradictory_operations": [
                # Immediate undoing operations
                (r"Rotate\(angle=90\).*Rotate\(angle=270\)", 0.9),
                (r"Rotate\(angle=180\).*Rotate\(angle=180\)", 0.9),
                (r"FlipHorizontal.*FlipHorizontal", 0.9),
                (r"FlipVertical.*FlipVertical", 0.9),
                (r"FloodFill\((\d+),\s*(\d+),\s*(\d+)\).*FloodFill\(\1,\s*\2,\s*\d+\)", 0.8),
            ],
            "invalid_color_operations": [
                # Colors outside valid ARC range (0-9)
                (r"MapColors.*[1-9]\d+", 1.0),  # Double digit colors
                (r"DrawLine.*,\s*[1-9]\d+\)", 1.0),
                (r"FloodFill.*,\s*[1-9]\d+\)", 1.0),
                (r"ReplaceColor\([1-9]\d+", 1.0),
                (r"FilterByColor\([1-9]\d+\)", 0.9),
            ],
            "memory_explosion": [
                # Operations that could cause excessive memory usage
                (r"Tile\([5-9]\d+,\s*[5-9]\d+\)", 1.0),  # Tiling > 50x50
                (r"Tile\(\d{3,},\s*\d{3,}\)", 1.0),  # Tiling > 100x100
                (r"Zoom\([5-9]\d+\)", 1.0),  # Zoom > 50x
                (r"RepeatPattern\(\d{3,}\)", 1.0),  # Repeat > 100 times
            ],
            "empty_grid_operations": [
                # Operations that can't work on empty grids
                (r"FilterByColor\(99\).*RotateShapes", 0.9),
                (r"RemoveAllColors.*ApplyPattern", 0.9),
                (r"ClearGrid.*ExtractShapes", 0.9),
            ],
            "infinite_loops": [
                # Patterns that suggest infinite loops
                (r"(RepeatUntil.*){3,}", 1.0),  # Too many nested repeats
                (r"WhileCondition.*WhileCondition.*WhileCondition", 0.9),
            ]
        }

    def _initialize_validation_rules(self) -> list[ValidationRule]:
        """Initialize validation rules for program checking.

        Returns:
            List of validation rules
        """
        rules = [
            ValidationRule(
                rule_id="syntax_001",
                name="valid_operation_names",
                category="syntax",
                check_function=self._check_valid_operations,
                severity="error"
            ),
            ValidationRule(
                rule_id="syntax_002",
                name="balanced_parentheses",
                category="syntax",
                check_function=self._check_balanced_parentheses,
                severity="error"
            ),
            ValidationRule(
                rule_id="bounds_001",
                name="grid_bounds_check",
                category="bounds",
                check_function=self._check_grid_bounds,
                severity="warning"
            ),
            ValidationRule(
                rule_id="color_001",
                name="valid_color_range",
                category="color",
                check_function=self._check_color_validity,
                severity="error"
            ),
            ValidationRule(
                rule_id="sequence_001",
                name="operation_sequence_validity",
                category="sequence",
                check_function=self._check_sequence_validity,
                severity="warning"
            ),
            ValidationRule(
                rule_id="memory_001",
                name="memory_usage_estimate",
                category="memory",
                check_function=self._check_memory_usage,
                severity="error"
            )
        ]

        # Filter by enabled rules
        return [r for r in rules if r.enabled]

    def _check_valid_operations(self, operations: list[Operation]) -> bool:
        """Check if all operations have valid names."""
        valid_ops = {
            "Rotate", "FlipHorizontal", "FlipVertical", "Tile", "Zoom",
            "FloodFill", "DrawLine", "DrawRectangle", "MapColors",
            "FilterByColor", "ReplaceColor", "ExtractShapes", "ApplyPattern",
            "RepeatPattern", "Mirror", "Translate", "Scale", "Compose"
        }

        for op in operations:
            op_name = type(op).__name__
            if op_name not in valid_ops and not op_name.startswith("Custom"):
                return False
        return True

    def _check_balanced_parentheses(self, operations: list[Operation]) -> bool:
        """Check if operations have balanced parentheses."""
        # This is a simplified check - real implementation would parse AST
        op_str = str(operations)
        return op_str.count('(') == op_str.count(')')

    def _check_grid_bounds(self, operations: list[Operation]) -> bool:
        """Check if operations stay within reasonable grid bounds."""
        for op in operations:
            op_str = str(op)
            # Check for coordinates outside typical bounds
            coords = re.findall(r'\b(\d+)\b', op_str)
            for coord in coords:
                if int(coord) > 100:  # Assuming max grid size of 100x100
                    return False
        return True

    def _check_color_validity(self, operations: list[Operation]) -> bool:
        """Check if all color values are within valid ARC range (0-9)."""
        for op in operations:
            op_str = str(op)
            # Extract color values from common operations
            color_matches = re.findall(r'(?:Color|color|Fill|fill).*?(\d+)', op_str)
            for color in color_matches:
                if int(color) > 9:
                    return False
        return True

    def _check_sequence_validity(self, operations: list[Operation]) -> bool:
        """Check for obviously invalid operation sequences."""
        if len(operations) < 2:
            return True

        for i in range(len(operations) - 1):
            curr_op = str(operations[i])
            next_op = str(operations[i + 1])

            # Check for immediate contradictions
            if "Rotate(90)" in curr_op and "Rotate(270)" in next_op:
                return False
            if "FlipHorizontal" in curr_op and "FlipHorizontal" in next_op:
                return False

        return True

    def _check_memory_usage(self, operations: list[Operation]) -> bool:
        """Estimate memory usage and reject if excessive."""
        estimated_size = 1  # Start with 1x multiplier

        for op in operations:
            op_str = str(op)

            # Check for operations that expand grid size
            if "Tile" in op_str:
                # Handle both formats: Tile(x=100, y=100) and Tile(100, 100)
                matches = re.findall(r'Tile\((?:x=)?(\d+),?\s*(?:y=)?(\d+)\)', op_str)
                if matches:
                    x, y = int(matches[0][0]), int(matches[0][1])
                    estimated_size *= x * y

            if "Zoom" in op_str:
                matches = re.findall(r'Zoom\((\d+)\)', op_str)
                if matches:
                    factor = int(matches[0])
                    estimated_size *= factor * factor

        # Assume starting grid of 30x30, 4 bytes per pixel
        estimated_bytes = 30 * 30 * 4 * estimated_size
        estimated_mb = estimated_bytes / (1024 * 1024)

        return estimated_mb <= self.config.memory_limit_mb

    def _apply_pattern_checks(self, program: list[Operation]) -> tuple[bool, str]:
        """Apply pattern-based rejection rules.

        Args:
            program: List of DSL operations

        Returns:
            Tuple of (passes_check, rejection_reason)
        """
        program_str = " ".join(str(op) for op in program)

        for category, patterns in self._pattern_database.items():
            for pattern_regex, severity in patterns:
                if re.search(pattern_regex, program_str):
                    # Severity threshold based on aggressiveness
                    if severity >= (1.0 - self.config.aggressiveness):
                        return False, f"Failed {category}: {pattern_regex}"

        return True, ""

    def _calculate_program_hash(self, program: list[Operation]) -> str:
        """Calculate a hash for program caching."""
        # Simple string representation for now
        return " ".join(str(op) for op in program)

    async def prune_program(
        self,
        program: list[Operation],
        test_inputs: list[Grid] | None = None
    ) -> PruningResult:
        """Evaluate a single program for pruning.

        Args:
            program: DSL program to evaluate
            test_inputs: Optional input grids for partial execution

        Returns:
            PruningResult with decision and metrics
        """
        start_time = time.perf_counter()
        self.stats["total_programs"] += 1

        # Calculate program hash for caching
        prog_hash = self._calculate_program_hash(program) if self.config.enable_caching else None

        # Check cache
        if self.config.enable_caching and prog_hash:
            if prog_hash in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[prog_hash]

        # Syntax validation
        if self.config.syntax_checks:
            for rule in self._validation_rules:
                if rule.category == "syntax" and not rule.check_function(program):
                    self.stats["syntax_failures"] += 1
                    result = self._create_pruning_result(
                        program, PruningDecision.REJECT_SYNTAX,
                        f"Failed {rule.name}", 0.0, start_time
                    )
                    self._update_cache(prog_hash, result)
                    return result

        # Pattern-based checks
        if self.config.pattern_checks:
            passes, reason = self._apply_pattern_checks(program)
            if not passes:
                self.stats["pattern_failures"] += 1
                result = self._create_pruning_result(
                    program, PruningDecision.REJECT_PATTERN,
                    reason, 0.0, start_time
                )
                self._update_cache(prog_hash, result)
                return result

        # Memory and bounds checks
        for rule in self._validation_rules:
            if rule.category in ["bounds", "color", "memory"]:
                if not rule.check_function(program):
                    result = self._create_pruning_result(
                        program, PruningDecision.REJECT_SECURITY,
                        f"Failed {rule.name}", 0.0, start_time
                    )
                    self._update_cache(prog_hash, result)
                    return result

        # If no partial execution, accept the program
        if not self.config.partial_execution or test_inputs is None or not self._partial_executor:
            result = self._create_pruning_result(
                program, PruningDecision.ACCEPT,
                "Passed all static checks", 1.0, start_time
            )
            self._update_cache(prog_hash, result)
            return result

        # Perform partial execution with confidence scoring
        try:
            # Get hints about expected output from training examples
            expected_hints = None
            if test_inputs and hasattr(test_inputs[0], '__len__'):
                # Basic hints - could be enhanced with more task context
                expected_hints = {
                    "expected_colors": list(range(10)),  # Valid ARC colors
                }

            # Execute partial program on first test input
            partial_result, confidence_score = await self._partial_executor.execute_partial(
                program,
                test_inputs[0] if test_inputs else [[0]],  # Use first test input
                expected_hints
            )

            # Update result with partial execution output
            partial_output = partial_result.intermediate_grid if partial_result.success else None

        except Exception as e:
            # If partial execution fails, reject with low confidence
            self.logger.warning("partial_execution_failed", error=str(e))
            confidence_score = 0.0
            partial_output = None

        if confidence_score >= self.config.confidence_threshold:
            decision = PruningDecision.ACCEPT
            reason = "Passed confidence threshold"
        else:
            decision = PruningDecision.REJECT_CONFIDENCE
            reason = f"Low confidence: {confidence_score:.2f}"
            self.stats["confidence_failures"] += 1

        result = self._create_pruning_result(
            program, decision, reason, confidence_score, start_time
        )
        result.partial_output = partial_output

        self._update_cache(prog_hash, result)
        return result

    async def batch_prune(
        self,
        programs: list[list[Operation]],
        test_inputs: list[Grid] | None = None
    ) -> list[PruningResult]:
        """Batch prune multiple programs.

        Args:
            programs: List of DSL programs
            test_inputs: Optional input grids for partial execution

        Returns:
            List of PruningResult objects
        """
        results = []

        for program in programs:
            result = await self.prune_program(program, test_inputs)
            results.append(result)

            if result.decision != PruningDecision.ACCEPT:
                self.stats["pruned_programs"] += 1

        self.logger.info(
            "batch_pruning_complete",
            total_programs=len(programs),
            pruned_count=self.stats["pruned_programs"],
            pruning_rate=self.stats["pruned_programs"] / len(programs) if programs else 0
        )

        return results

    def _create_pruning_result(
        self,
        program: list[Operation],
        decision: PruningDecision,
        reason: str,
        confidence: float,
        start_time: float
    ) -> PruningResult:
        """Create a pruning result object with security audit logging."""
        pruning_time_ms = (time.perf_counter() - start_time) * 1000
        self.stats["total_pruning_time_ms"] += pruning_time_ms

        result = PruningResult(
            program_id=str(id(program)),  # Simple ID generation
            decision=decision,
            confidence_score=confidence,
            pruning_time_ms=pruning_time_ms,
            rejection_reason=reason if decision != PruningDecision.ACCEPT else None,
            partial_output=None  # Would be populated by partial execution
        )

        # Security audit logging
        self._audit_log_pruning_decision(result, program)

        return result

    def _update_cache(self, prog_hash: str, result: PruningResult) -> None:
        """Update the pruning cache."""
        if self.config.enable_caching and prog_hash:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.config.cache_size:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[prog_hash] = result

    def get_statistics(self) -> dict[str, Any]:
        """Get pruning statistics."""
        stats = self.stats.copy()

        # Calculate derived metrics
        if stats["total_programs"] > 0:
            stats["pruning_rate"] = stats["pruned_programs"] / stats["total_programs"]
            stats["syntax_failure_rate"] = stats["syntax_failures"] / stats["total_programs"]
            stats["pattern_failure_rate"] = stats["pattern_failures"] / stats["total_programs"]
            stats["confidence_failure_rate"] = stats["confidence_failures"] / stats["total_programs"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_programs"]
            stats["avg_pruning_time_ms"] = stats["total_pruning_time_ms"] / stats["total_programs"]

        return stats

    def _audit_log_pruning_decision(
        self,
        result: PruningResult,
        program: list[Operation]
    ) -> None:
        """Log pruning decision for security audit.

        Args:
            result: Pruning result containing decision details
            program: The program that was evaluated
        """
        # Create audit entry
        audit_entry = {
            "timestamp": time.time(),
            "program_id": result.program_id,
            "decision": result.decision.value,
            "confidence_score": result.confidence_score,
            "pruning_time_ms": result.pruning_time_ms,
            "rejection_reason": result.rejection_reason,
            "strategy_id": self.config.strategy_id,
            "aggressiveness": self.config.aggressiveness,
            "program_length": len(program),
            "program_hash": self._calculate_program_hash(program),
        }

        # Log security-relevant decisions with higher priority
        if result.decision == PruningDecision.REJECT_SECURITY:
            self.logger.warning(
                "security_pruning_decision",
                **audit_entry,
                security_alert=True
            )
        else:
            self.logger.info(
                "pruning_decision",
                **audit_entry
            )

        # Track security-related rejections
        if result.decision == PruningDecision.REJECT_SECURITY:
            if not hasattr(self, "_security_rejections"):
                self._security_rejections = []

            self._security_rejections.append({
                "timestamp": audit_entry["timestamp"],
                "program_id": result.program_id,
                "reason": result.rejection_reason,
                "operations": [op.get_name() for op in program[:5]]  # First 5 ops
            })

            # Alert if too many security rejections
            if len(self._security_rejections) > 100:
                recent_rejections = [
                    r for r in self._security_rejections
                    if time.time() - r["timestamp"] < 300  # Last 5 minutes
                ]

                if len(recent_rejections) > 50:
                    self.logger.critical(
                        "high_security_rejection_rate",
                        recent_count=len(recent_rejections),
                        total_count=len(self._security_rejections),
                        alert_type="security_anomaly"
                    )

    def get_security_audit_summary(self) -> dict[str, Any]:
        """Get summary of security-related pruning decisions.

        Returns:
            Dictionary containing security audit metrics
        """
        if not hasattr(self, "_security_rejections"):
            return {"security_rejections": 0}

        recent_rejections = [
            r for r in self._security_rejections
            if time.time() - r["timestamp"] < 3600  # Last hour
        ]

        return {
            "total_security_rejections": len(self._security_rejections),
            "recent_security_rejections": len(recent_rejections),
            "rejection_reasons": {
                r["reason"]: sum(1 for rej in self._security_rejections if rej["reason"] == r["reason"])
                for r in self._security_rejections
            },
            "most_common_rejected_ops": self._get_most_common_rejected_ops(),
        }

    def _get_most_common_rejected_ops(self) -> dict[str, int]:
        """Get the most commonly rejected operations.

        Returns:
            Dictionary mapping operation names to rejection counts
        """
        if not hasattr(self, "_security_rejections"):
            return {}

        op_counts = {}
        for rejection in self._security_rejections:
            for op_name in rejection.get("operations", []):
                op_counts[op_name] = op_counts.get(op_name, 0) + 1

        # Sort by count and return top 10
        return dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    def reset_statistics(self) -> None:
        """Reset pruning statistics."""
        self.stats = {
            "total_programs": 0,
            "pruned_programs": 0,
            "syntax_failures": 0,
            "pattern_failures": 0,
            "confidence_failures": 0,
            "cache_hits": 0,
            "total_pruning_time_ms": 0.0
        }
        self._cache.clear()

    async def close(self) -> None:
        """Clean up resources."""
        if self._partial_executor:
            try:
                await self._partial_executor.close()
            except Exception as e:
                self.logger.warning("partial_executor_cleanup_failed", error=str(e))
