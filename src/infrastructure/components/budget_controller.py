"""Budget controller for tracking and limiting LLM API costs."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Budget status levels."""
    HEALTHY = "healthy"  # Under 50% of budget
    WARNING = "warning"  # 50-80% of budget
    CRITICAL = "critical"  # 80-95% of budget
    EXCEEDED = "exceeded"  # Over budget limit


@dataclass
class ModelCost:
    """Cost configuration for a model."""
    name: str
    cost_per_input_token: float  # Cost per token for input
    cost_per_output_token: float  # Cost per token for output
    cost_per_million_tokens: float | None = None  # Alternative pricing model


@dataclass
class UsageRecord:
    """Record of API usage."""
    timestamp: datetime
    model_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Budget alert configuration."""
    threshold_percent: float
    callback: Callable[[float, float], None] | None = None
    message: str = ""
    triggered: bool = False


class BudgetController:
    """Controller for managing API usage budgets."""

    def __init__(
        self,
        budget_limit: float,
        persist_path: Path | None = None,
        alert_thresholds: list[float] | None = None
    ):
        self.budget_limit = budget_limit
        self.persist_path = persist_path or Path("data/budget_usage.json")
        self.alert_thresholds = alert_thresholds or [0.5, 0.8, 0.95]

        self._usage_records: list[UsageRecord] = []
        self._total_cost = 0.0
        self._model_costs: dict[str, ModelCost] = {}
        self._alerts: list[BudgetAlert] = []
        self._lock = asyncio.Lock()

        # Initialize alerts
        for threshold in self.alert_thresholds:
            self._alerts.append(BudgetAlert(
                threshold_percent=threshold,
                message=f"Budget usage has reached {threshold*100:.0f}%"
            ))

        # Load persisted data
        self._load_usage_data()

    def register_model(self, model_cost: ModelCost):
        """Register a model with its cost configuration."""
        self._model_costs[model_cost.name] = model_cost
        logger.info(f"Registered model '{model_cost.name}' with costs")

    def register_alert(self, alert: BudgetAlert):
        """Register a custom budget alert."""
        self._alerts.append(alert)
        self._alerts.sort(key=lambda x: x.threshold_percent)

    async def track_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> UsageRecord:
        """Track API usage and calculate cost."""
        async with self._lock:
            # Calculate cost
            cost = self._calculate_cost(model_name, input_tokens, output_tokens)

            # Check if budget would be exceeded
            if self._total_cost + cost > self.budget_limit:
                raise BudgetExceededException(
                    f"Operation would exceed budget limit of ${self.budget_limit:.2f}. "
                    f"Current: ${self._total_cost:.2f}, Requested: ${cost:.2f}"
                )

            # Create usage record
            record = UsageRecord(
                timestamp=datetime.now(),
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                task_id=task_id,
                metadata=metadata or {}
            )

            # Update tracking
            self._usage_records.append(record)
            self._total_cost += cost

            # Check alerts
            await self._check_alerts()

            # Persist data
            self._save_usage_data()

            return record

    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        if model_name not in self._model_costs:
            logger.warning(f"Model '{model_name}' not registered, using zero cost")
            return 0.0

        model_cost = self._model_costs[model_name]

        # Use per-million pricing if available and no separate input/output costs
        if (model_cost.cost_per_million_tokens is not None and
            model_cost.cost_per_input_token == 0 and
            model_cost.cost_per_output_token == 0):
            total_tokens = input_tokens + output_tokens
            return (total_tokens / 1_000_000) * model_cost.cost_per_million_tokens

        # Otherwise use input/output pricing
        input_cost = (input_tokens / 1_000_000) * model_cost.cost_per_input_token
        output_cost = (output_tokens / 1_000_000) * model_cost.cost_per_output_token
        return input_cost + output_cost

    async def _check_alerts(self):
        """Check and trigger budget alerts."""
        usage_percent = self._total_cost / self.budget_limit

        for alert in self._alerts:
            if not alert.triggered and usage_percent >= alert.threshold_percent:
                alert.triggered = True
                logger.warning(f"Budget alert: {alert.message} (${self._total_cost:.2f}/${self.budget_limit:.2f})")

                if alert.callback:
                    try:
                        await alert.callback(self._total_cost, self.budget_limit)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")

    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        usage_percent = self._total_cost / self.budget_limit

        if usage_percent >= 1.0:
            return BudgetStatus.EXCEEDED
        elif usage_percent >= 0.8:
            return BudgetStatus.CRITICAL
        elif usage_percent >= 0.5:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.HEALTHY

    def get_remaining_budget(self) -> float:
        """Get remaining budget amount."""
        return max(0, self.budget_limit - self._total_cost)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get summary of budget usage."""
        model_usage = {}
        for record in self._usage_records:
            if record.model_name not in model_usage:
                model_usage[record.model_name] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }

            model_usage[record.model_name]["count"] += 1
            model_usage[record.model_name]["input_tokens"] += record.input_tokens
            model_usage[record.model_name]["output_tokens"] += record.output_tokens
            model_usage[record.model_name]["cost"] += record.cost

        return {
            "total_cost": self._total_cost,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.get_remaining_budget(),
            "usage_percent": (self._total_cost / self.budget_limit) * 100,
            "status": self.get_status().value,
            "model_usage": model_usage,
            "total_requests": len(self._usage_records),
            "first_usage": self._usage_records[0].timestamp.isoformat() if self._usage_records else None,
            "last_usage": self._usage_records[-1].timestamp.isoformat() if self._usage_records else None
        }

    def get_usage_by_timeframe(self, hours: int = 24) -> dict[str, Any]:
        """Get usage statistics for a specific timeframe."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_records = [r for r in self._usage_records if r.timestamp >= cutoff_time]

        recent_cost = sum(r.cost for r in recent_records)

        return {
            "timeframe_hours": hours,
            "cost": recent_cost,
            "requests": len(recent_records),
            "average_cost_per_request": recent_cost / max(1, len(recent_records))
        }

    def can_afford_request(self, model_name: str, estimated_tokens: int) -> bool:
        """Check if a request can be afforded within budget."""
        # Assume 50/50 split between input and output for estimation
        estimated_cost = self._calculate_cost(
            model_name,
            estimated_tokens // 2,
            estimated_tokens // 2
        )
        return self._total_cost + estimated_cost <= self.budget_limit

    def _save_usage_data(self):
        """Save usage data to disk."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "budget_limit": self.budget_limit,
                "total_cost": self._total_cost,
                "usage_records": [
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "model_name": record.model_name,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                        "cost": record.cost,
                        "task_id": record.task_id,
                        "metadata": record.metadata
                    }
                    for record in self._usage_records[-1000:]  # Keep last 1000 records
                ]
            }

            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save budget data: {e}")

    def _load_usage_data(self):
        """Load usage data from disk."""
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    data = json.load(f)

                # Restore usage records
                for record_data in data.get("usage_records", []):
                    self._usage_records.append(UsageRecord(
                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                        model_name=record_data["model_name"],
                        input_tokens=record_data["input_tokens"],
                        output_tokens=record_data["output_tokens"],
                        cost=record_data["cost"],
                        task_id=record_data.get("task_id"),
                        metadata=record_data.get("metadata", {})
                    ))

                # Recalculate total cost
                self._total_cost = sum(r.cost for r in self._usage_records)

                logger.info(f"Loaded budget data: ${self._total_cost:.2f} used of ${self.budget_limit:.2f}")

        except Exception as e:
            logger.error(f"Failed to load budget data: {e}")

    def reset(self):
        """Reset budget tracking (careful!)."""
        self._usage_records.clear()
        self._total_cost = 0.0
        for alert in self._alerts:
            alert.triggered = False
        self._save_usage_data()
        logger.info("Budget controller reset")


class BudgetExceededException(Exception):
    """Exception raised when budget would be exceeded."""
    pass


# Convenience function to create a budget controller with default model costs
def create_default_budget_controller(budget_limit: float = 100.0) -> BudgetController:
    """Create a budget controller with default model configurations."""
    controller = BudgetController(budget_limit)

    # Register default models from the story
    controller.register_model(ModelCost(
        name="Qwen2.5-Coder",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        cost_per_million_tokens=0.15
    ))

    controller.register_model(ModelCost(
        name="Gemini 2.5 Flash",
        cost_per_input_token=0.31,
        cost_per_output_token=2.62,
    ))

    controller.register_model(ModelCost(
        name="GLM-4.5",
        cost_per_input_token=0.59,
        cost_per_output_token=2.19,
    ))

    controller.register_model(ModelCost(
        name="GPT-5",
        cost_per_input_token=1.25,
        cost_per_output_token=10.00,
    ))

    controller.register_model(ModelCost(
        name="Falcon Mamba 7B",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        cost_per_million_tokens=0.0  # Local model, free
    ))

    return controller
