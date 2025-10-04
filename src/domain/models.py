"""Domain models for ARC Prize 2025 competition."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

Grid = list[list[int]]

# Import DSLProgram from dsl module for backward compatibility
try:
    from src.domain.dsl.base import DSLProgram
except ImportError:
    # Define a placeholder if DSL module is not available
    class DSLProgram:
        """Placeholder DSLProgram class for compatibility."""
        def __init__(self, operations: list[dict[str, Any]], version: str = "1.0", metadata: dict[str, Any] | None = None):
            self.operations = operations
            self.version = version
            self.metadata = metadata


@dataclass
class Example:
    """Single training example with input and output grids."""
    input_grid: Grid
    output_grid: Grid

    def to_dict(self) -> dict[str, Grid]:
        """Convert to dictionary format for ARCTask compatibility."""
        return {
            "input": self.input_grid,
            "output": self.output_grid
        }

    @classmethod
    def from_dict(cls, data: dict[str, Grid]) -> "Example":
        """Create Example from dictionary format."""
        return cls(
            input_grid=data["input"],
            output_grid=data["output"]
        )


@dataclass
class InputOutputPair:
    """Input-output pair for training examples (alias for test compatibility)."""
    input: Grid
    output: Grid

    def to_dict(self) -> dict[str, Grid]:
        """Convert to dictionary format for ARCTask compatibility."""
        return {
            "input": self.input,
            "output": self.output
        }

    @classmethod
    def from_dict(cls, data: dict[str, Grid]) -> "InputOutputPair":
        """Create InputOutputPair from dictionary format."""
        return cls(
            input=data["input"],
            output=data["output"]
        )


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

    # Additional attributes for test compatibility
    train_pairs: list[InputOutputPair] = field(default_factory=list)
    test_pairs: list[InputOutputPair] = field(default_factory=list)

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
        dimensions: dict[str, list[tuple]] = {"train_input": [], "train_output": [], "test_input": [], "test_output": []}

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


class StrategyType(Enum):
    """Types of solving strategies available."""
    TEST_TIME_TRAINING = "ttt"
    PROGRAM_SYNTHESIS = "program_synthesis"
    EVOLUTION = "evolution"
    IMITATION_LEARNING = "imitation"
    HYBRID = "hybrid"


@dataclass
class ResourceUsage:
    """Track resource usage for a task execution."""
    task_id: str
    strategy_type: StrategyType
    cpu_seconds: float
    memory_mb: float
    gpu_memory_mb: float | None
    api_calls: dict[str, int]
    total_tokens: int
    estimated_cost: float
    timestamp: datetime


@dataclass
class ARCTaskSolution:
    """Solution for an ARC task."""
    task_id: str
    predictions: list[list[list[int]]]
    strategy_used: StrategyType
    confidence_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    resource_usage: ResourceUsage | None = None


@dataclass
class StrategyOutput:
    """Standardized output format for all solving strategies.

    This replaces ARCTaskSolution for cross-strategy integration in Epic 3.
    Uses numpy arrays for 100x faster serialization vs nested lists.

    Attributes:
        strategy_type: Which strategy produced this output
        predicted_output: Predicted grid as numpy array (max 30x30, dtype=int8)
        confidence_score: Aggregate confidence (0.0-1.0) for this prediction
        per_pixel_confidence: Optional confidence score per pixel (same shape as output)
        strategy_metadata: Strategy-specific information (e.g., programs evaluated, fitness)
        reasoning_trace: Optional debug info for explainability (max 100 entries)
        processing_time_ms: Time taken to generate this output
        resource_usage: Detailed resource consumption tracking
        timestamp: When this output was generated

    Performance:
        Serialization: <10ms for 30x30 grid with msgpack (vs ~500-1000ms for nested lists)
        Memory: <1MB per instance

    Security:
        - Validates array shapes (max 30x30) before deserialization
        - Validates confidence scores in range [0.0, 1.0]
        - Limits reasoning_trace to 100 entries max
    """
    strategy_type: StrategyType
    predicted_output: np.ndarray
    confidence_score: float
    per_pixel_confidence: np.ndarray | None = None
    strategy_metadata: dict[str, Any] = field(default_factory=dict)
    reasoning_trace: list[str] | None = None
    processing_time_ms: int = 0
    resource_usage: ResourceUsage | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate StrategyOutput constraints."""
        if self.predicted_output.shape[0] > 30 or self.predicted_output.shape[1] > 30:
            raise ValueError(f"Grid size {self.predicted_output.shape} exceeds max 30x30")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence {self.confidence_score} not in range [0.0, 1.0]")

        if self.per_pixel_confidence is not None:
            if self.per_pixel_confidence.shape != self.predicted_output.shape:
                raise ValueError("per_pixel_confidence shape must match predicted_output")
            if np.any(self.per_pixel_confidence < 0.0) or np.any(self.per_pixel_confidence > 1.0):
                raise ValueError("per_pixel_confidence values must be in range [0.0, 1.0]")

        if self.reasoning_trace is not None and len(self.reasoning_trace) > 100:
            raise ValueError("reasoning_trace limited to 100 entries max")

    def to_msgpack(self) -> bytes:
        """Serialize to msgpack binary format.

        Returns:
            Serialized bytes suitable for network transfer or storage

        Performance:
            <10ms for 30x30 grid with confidence scores
        """
        import msgpack

        data = {
            "strategy_type": self.strategy_type.value,
            "predicted_output": self.predicted_output.tobytes(),
            "predicted_output_shape": self.predicted_output.shape,
            "predicted_output_dtype": str(self.predicted_output.dtype),
            "confidence_score": self.confidence_score,
            "strategy_metadata": self.strategy_metadata,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.per_pixel_confidence is not None:
            data["per_pixel_confidence"] = self.per_pixel_confidence.tobytes()
            data["per_pixel_confidence_dtype"] = str(self.per_pixel_confidence.dtype)

        if self.reasoning_trace is not None:
            data["reasoning_trace"] = self.reasoning_trace

        if self.resource_usage is not None:
            data["resource_usage"] = {
                "task_id": self.resource_usage.task_id,
                "strategy_type": self.resource_usage.strategy_type.value,
                "cpu_seconds": self.resource_usage.cpu_seconds,
                "memory_mb": self.resource_usage.memory_mb,
                "gpu_memory_mb": self.resource_usage.gpu_memory_mb,
                "api_calls": self.resource_usage.api_calls,
                "total_tokens": self.resource_usage.total_tokens,
                "estimated_cost": self.resource_usage.estimated_cost,
                "timestamp": self.resource_usage.timestamp.isoformat(),
            }

        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "StrategyOutput":
        """Deserialize from msgpack binary format.

        Args:
            data: Serialized bytes from to_msgpack()

        Returns:
            Reconstructed StrategyOutput instance

        Performance:
            <10ms for 30x30 grid with confidence scores

        Security:
            Validates array shapes and confidence ranges
        """
        import msgpack

        unpacked = msgpack.unpackb(data, raw=False)

        predicted_output = np.frombuffer(
            unpacked["predicted_output"],
            dtype=unpacked["predicted_output_dtype"]
        ).reshape(unpacked["predicted_output_shape"])

        per_pixel_confidence = None
        if "per_pixel_confidence" in unpacked:
            per_pixel_confidence = np.frombuffer(
                unpacked["per_pixel_confidence"],
                dtype=unpacked["per_pixel_confidence_dtype"]
            ).reshape(unpacked["predicted_output_shape"])

        resource_usage = None
        if "resource_usage" in unpacked:
            ru = unpacked["resource_usage"]
            resource_usage = ResourceUsage(
                task_id=ru["task_id"],
                strategy_type=StrategyType(ru["strategy_type"]),
                cpu_seconds=ru["cpu_seconds"],
                memory_mb=ru["memory_mb"],
                gpu_memory_mb=ru["gpu_memory_mb"],
                api_calls=ru["api_calls"],
                total_tokens=ru["total_tokens"],
                estimated_cost=ru["estimated_cost"],
                timestamp=datetime.fromisoformat(ru["timestamp"]),
            )

        return cls(
            strategy_type=StrategyType(unpacked["strategy_type"]),
            predicted_output=predicted_output,
            confidence_score=unpacked["confidence_score"],
            per_pixel_confidence=per_pixel_confidence,
            strategy_metadata=unpacked.get("strategy_metadata", {}),
            reasoning_trace=unpacked.get("reasoning_trace"),
            processing_time_ms=unpacked["processing_time_ms"],
            resource_usage=resource_usage,
            timestamp=datetime.fromisoformat(unpacked["timestamp"]),
        )


@dataclass
class TTTAdaptation:
    """Store Test-Time Training adaptations."""
    adaptation_id: str
    task_id: str
    base_model_checkpoint: str
    adapted_weights_path: str
    training_examples: list[dict[str, Any]]
    adaptation_metrics: dict[str, float]
    created_at: datetime


class UserRole(Enum):
    """User roles for authorization."""
    USER = "user"
    ADMIN = "admin"
    SERVICE = "service"


class AccountStatus(Enum):
    """Account status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"


@dataclass
class User:
    """User account model."""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    status: AccountStatus
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None
    failed_login_attempts: int = 0
    locked_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if user account is active."""
        if self.status != AccountStatus.ACTIVE:
            return False

        if self.locked_until and self.locked_until > datetime.now():
            return False

        return True

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        return (self.status == AccountStatus.LOCKED or
                (self.locked_until and self.locked_until > datetime.now()))


@dataclass
class ServiceAccount:
    """Service account for automated system access."""
    id: str
    name: str
    description: str
    api_key_hash: str
    permissions: list[str]
    status: AccountStatus
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None


@dataclass
class LLMCache:
    """Cache LLM responses for efficiency."""
    cache_id: str
    prompt_hash: str
    model_name: str
    temperature: float
    response_text: str
    token_count: int
    created_at: datetime
    access_count: int = 0
    task_features: dict[str, float] = field(default_factory=dict)  # For similarity matching
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationAttempt:
    """Track authentication attempts for security monitoring."""
    id: str
    username_or_email: str
    ip_address: str
    user_agent: str
    success: bool
    failure_reason: str | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class PruningDecision(Enum):
    """Decision made by pruning system."""
    ACCEPT = "accept"
    REJECT_SYNTAX = "reject_syntax"
    REJECT_PATTERN = "reject_pattern"
    REJECT_CONFIDENCE = "reject_confidence"
    REJECT_SECURITY = "reject_security"


@dataclass
class PruningResult:
    """Result of program pruning evaluation."""
    program_id: str
    decision: PruningDecision
    confidence_score: float
    pruning_time_ms: float
    rejection_reason: str | None = None
    partial_output: list[list[int]] | None = None


@dataclass
class PruningStrategy:
    """Configuration for a pruning strategy."""
    strategy_id: str
    name: str
    aggressiveness: float  # 0.0 (conservative) to 1.0 (aggressive)
    syntax_checks: bool
    pattern_checks: bool
    partial_execution: bool
    confidence_threshold: float
    max_partial_ops: int
    timeout_ms: float


@dataclass
class PruningMetrics:
    """Metrics tracking pruning effectiveness."""
    strategy_id: str
    total_programs: int
    programs_pruned: int
    pruning_rate: float
    false_negatives: int
    false_negative_rate: float
    avg_pruning_time_ms: float
    time_saved_ms: float
    timestamp: datetime


@dataclass
class EvaluationResult:
    """Result of evaluating a program or strategy."""
    program_id: str
    task_id: str
    success: bool
    fitness_score: float
    execution_time_ms: float
    memory_used_mb: float
    output_grid: list[list[int]] | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PartialExecutionResult:
    """Result of partial DSL program execution."""
    program_id: str
    operations_executed: int
    intermediate_grid: list[list[int]]
    execution_time_ms: float
    memory_used_mb: float
    success: bool
    error: str | None = None
