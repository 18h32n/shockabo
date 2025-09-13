"""Extended evaluation models for comprehensive experiment tracking and resource management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StrategyType(Enum):
    """Types of strategies available for solving ARC tasks."""

    DIRECT_SOLVE = "direct_solve"
    PATTERN_MATCH = "pattern_match"
    TRANSFORMATION_SEARCH = "transformation_search"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Status of a task or experiment run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskSubmission:
    """Represents a submission for an ARC task with comprehensive metadata."""

    submission_id: str
    task_id: str
    user_id: str
    predicted_output: list[list[int]]
    strategy_used: StrategyType
    confidence_score: float
    processing_time_ms: int
    resource_usage: dict[str, float]
    metadata: dict[str, Any]
    submitted_at: datetime

    def __post_init__(self):
        """Validate submission data."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")
        if self.processing_time_ms < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time_ms}")
        if not self.predicted_output:
            raise ValueError("Predicted output cannot be empty")


@dataclass
class ExperimentRun:
    """Tracks a complete experiment run across multiple tasks."""

    run_id: str
    experiment_name: str
    task_ids: list[str]
    strategy_config: dict[str, Any]
    metrics: dict[str, float]
    status: TaskStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_log: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate run duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def is_finished(self) -> bool:
        """Check if the experiment has finished."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]

    def update_metrics(self, new_metrics: dict[str, float]) -> None:
        """Update experiment metrics."""
        self.metrics.update(new_metrics)

    def mark_completed(self, final_metrics: dict[str, float] | None = None) -> None:
        """Mark the experiment as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if final_metrics:
            self.update_metrics(final_metrics)

    def mark_failed(self, error_message: str) -> None:
        """Mark the experiment as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_log = error_message


@dataclass
class ResourceUsage:
    """Tracks resource usage for a task execution."""

    task_id: str
    strategy_type: StrategyType
    cpu_seconds: float
    memory_mb: float
    gpu_memory_mb: float | None
    api_calls: dict[str, int]
    total_tokens: int
    estimated_cost: float
    timestamp: datetime

    @property
    def total_api_calls(self) -> int:
        """Get total number of API calls."""
        return sum(self.api_calls.values())

    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token."""
        if self.total_tokens == 0:
            return 0.0
        return self.estimated_cost / self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "strategy_type": self.strategy_type.value,
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment run."""

    experiment_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    average_accuracy: float
    perfect_matches: int
    total_processing_time_ms: float
    total_resource_cost: float
    strategy_performance: dict[str, dict[str, float]]
    error_distribution: dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def perfect_match_rate(self) -> float:
        """Calculate perfect match rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.perfect_matches / self.total_tasks

    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_tasks


@dataclass
class RegressionAlert:
    """Alert for performance regression detection."""

    alert_id: str
    experiment_id: str
    baseline_experiment_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percentage: float
    severity: str  # "low", "medium", "high", "critical"
    affected_tasks: list[str]
    details: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    @property
    def is_improvement(self) -> bool:
        """Check if this is actually an improvement rather than regression."""
        # For accuracy metrics, higher is better
        if "accuracy" in self.metric_name.lower():
            return self.current_value > self.baseline_value
        # For time/cost metrics, lower is better
        elif any(x in self.metric_name.lower() for x in ["time", "cost", "memory"]):
            return self.current_value < self.baseline_value
        # Default: assume higher is better
        return self.current_value > self.baseline_value

    def acknowledge(self, user_id: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()


@dataclass
class DashboardMetrics:
    """Real-time metrics for dashboard display."""

    timestamp: datetime
    active_experiments: int
    tasks_processed_last_hour: int
    average_accuracy_last_hour: float
    resource_utilization: dict[str, float]  # cpu, memory, gpu percentages
    processing_queue_size: int
    error_rate_last_hour: float
    top_performing_strategies: list[tuple[str, float]]  # (strategy, accuracy)
    recent_alerts: list[RegressionAlert]
    system_health: dict[str, str]  # component -> status

    def to_websocket_message(self) -> dict[str, Any]:
        """Convert to WebSocket message format."""
        return {
            "type": "dashboard_update",
            "timestamp": self.timestamp.isoformat(),
            "data": {
                "active_experiments": self.active_experiments,
                "tasks_processed": self.tasks_processed_last_hour,
                "average_accuracy": round(self.average_accuracy_last_hour, 4),
                "resource_utilization": self.resource_utilization,
                "queue_size": self.processing_queue_size,
                "error_rate": round(self.error_rate_last_hour, 4),
                "top_strategies": [
                    {"name": name, "accuracy": round(acc, 4)} for name, acc in self.top_performing_strategies[:5]
                ],
                "alerts": [
                    {
                        "id": alert.alert_id,
                        "metric": alert.metric_name,
                        "severity": alert.severity,
                        "regression": round(alert.regression_percentage, 2),
                    }
                    for alert in self.recent_alerts[:10]
                ],
                "system_health": self.system_health,
            },
        }
