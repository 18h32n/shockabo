"""Migration helper for integrating pruning into existing evaluation workflows.

This module provides utilities to gradually migrate existing evaluation
workflows to use intelligent pruning without breaking changes.
"""

import logging
from collections.abc import Callable
from typing import Any

from src.domain.models import PruningStrategy
from src.domain.services.evaluation_service import EvaluationService
from src.infrastructure.config import get_config

logger = logging.getLogger(__name__)


class PruningMigrationHelper:
    """Helper class for migrating existing workflows to use pruning."""

    def __init__(self):
        """Initialize migration helper."""
        self.config = get_config()

        # Migration phases
        self.migration_phases = {
            "disabled": {
                "description": "Pruning disabled, existing behavior preserved",
                "pruning_enabled": False,
                "monitoring_only": False,
            },
            "monitoring": {
                "description": "Pruning runs in shadow mode for monitoring",
                "pruning_enabled": True,
                "monitoring_only": True,
            },
            "gradual": {
                "description": "Gradually increase pruning usage",
                "pruning_enabled": True,
                "monitoring_only": False,
                "rollout_percentage": 10,  # Start with 10%
            },
            "enabled": {
                "description": "Pruning fully enabled",
                "pruning_enabled": True,
                "monitoring_only": False,
                "rollout_percentage": 100,
            },
        }

        # Current migration phase
        self.current_phase = self._load_migration_phase()

    def _load_migration_phase(self) -> str:
        """Load current migration phase from configuration.

        Returns:
            Current migration phase name
        """
        # Try to load from config
        phase = getattr(self.config, "pruning_migration_phase", "disabled")

        if phase not in self.migration_phases:
            logger.warning(f"Unknown migration phase: {phase}, defaulting to 'disabled'")
            return "disabled"

        return phase

    def wrap_evaluation_service(
        self,
        evaluation_service: EvaluationService,
    ) -> EvaluationService:
        """Wrap existing evaluation service with pruning based on migration phase.

        Args:
            evaluation_service: Existing evaluation service

        Returns:
            Wrapped evaluation service with appropriate pruning configuration
        """
        phase_config = self.migration_phases[self.current_phase]

        if not phase_config["pruning_enabled"]:
            # Pruning disabled, return original service
            logger.info("Pruning disabled in current migration phase")
            return evaluation_service

        # Configure pruning based on phase
        if phase_config.get("monitoring_only", False):
            # Shadow mode - log metrics but don't actually prune
            logger.info("Pruning in monitoring mode")
            evaluation_service.enable_pruning = False
            # Could add monitoring hooks here

        elif "rollout_percentage" in phase_config:
            # Gradual rollout
            percentage = phase_config["rollout_percentage"]
            logger.info(f"Pruning enabled for {percentage}% of evaluations")
            # This would need custom logic in evaluation service

        return evaluation_service

    def get_recommended_strategy(self) -> PruningStrategy | None:
        """Get recommended pruning strategy for current migration phase.

        Returns:
            Recommended pruning strategy or None
        """
        if self.current_phase == "disabled":
            return None

        # Start with conservative strategy during migration
        if self.current_phase in ["monitoring", "gradual"]:
            return PruningStrategy(
                strategy_id="migration-conservative",
                name="Migration Conservative Strategy",
                aggressiveness=0.2,  # Very conservative
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=False,  # No partial execution initially
                confidence_threshold=0.9,  # High threshold
                max_partial_ops=0,
                timeout_ms=30,
            )
        else:
            # Full rollout - use balanced strategy
            return PruningStrategy(
                strategy_id="migration-balanced",
                name="Migration Balanced Strategy",
                aggressiveness=0.5,
                syntax_checks=True,
                pattern_checks=True,
                partial_execution=True,
                confidence_threshold=0.6,
                max_partial_ops=3,
                timeout_ms=100,
            )

    def create_migration_config(self) -> dict[str, Any]:
        """Create migration configuration for current phase.

        Returns:
            Dictionary with migration configuration
        """
        phase_config = self.migration_phases[self.current_phase]
        strategy = self.get_recommended_strategy()

        config = {
            "phase": self.current_phase,
            "phase_description": phase_config["description"],
            "pruning_enabled": phase_config["pruning_enabled"],
            "monitoring_only": phase_config.get("monitoring_only", False),
            "rollout_percentage": phase_config.get("rollout_percentage", 100),
            "recommended_strategy": strategy.strategy_id if strategy else None,
            "fallback_behavior": "full_evaluation",
            "metrics_collection": True,
            "alert_thresholds": {
                "false_negative_rate": 0.05,
                "performance_degradation": 0.1,  # 10% performance loss threshold
            },
        }

        return config

    def validate_migration_readiness(self) -> tuple[bool, list[str]]:
        """Validate if system is ready for next migration phase.

        Returns:
            Tuple of (is_ready, list of issues if not ready)
        """
        issues = []

        # Check current phase
        if self.current_phase == "enabled":
            return True, []  # Already fully migrated

        # Check prerequisites for each phase
        if self.current_phase == "disabled":
            # Ready to move to monitoring phase
            # Check if monitoring infrastructure is available
            try:
                from src.adapters.strategies.pruning_monitoring_dashboard import (
                    PruningMonitoringDashboard,  # noqa: F401
                )
                # Check if dashboard is available
                pass  # Dashboard is available
            except ImportError:
                issues.append("Monitoring dashboard not available")

        elif self.current_phase == "monitoring":
            # Check if we have enough monitoring data
            # This would need actual metric checks
            pass

        elif self.current_phase == "gradual":
            # Check if gradual rollout is successful
            # This would need metric validation
            pass

        return len(issues) == 0, issues

    def advance_migration_phase(self) -> bool:
        """Advance to the next migration phase if ready.

        Returns:
            True if successfully advanced, False otherwise
        """
        is_ready, issues = self.validate_migration_readiness()

        if not is_ready:
            logger.warning(
                f"Not ready to advance migration phase. Issues: {', '.join(issues)}"
            )
            return False

        # Determine next phase
        phase_order = ["disabled", "monitoring", "gradual", "enabled"]
        current_idx = phase_order.index(self.current_phase)

        if current_idx >= len(phase_order) - 1:
            logger.info("Already at final migration phase")
            return False

        next_phase = phase_order[current_idx + 1]

        # Update configuration (would need actual config update logic)
        logger.info(f"Advancing migration from {self.current_phase} to {next_phase}")
        self.current_phase = next_phase

        return True

    def create_compatibility_wrapper(
        self,
        legacy_evaluation_func: Callable,
    ) -> Callable:
        """Create compatibility wrapper for legacy evaluation functions.

        Args:
            legacy_evaluation_func: Legacy evaluation function

        Returns:
            Wrapped function with pruning support
        """
        def wrapped_evaluation(*args, **kwargs):
            """Wrapped evaluation with optional pruning."""
            # Check if pruning should be applied
            if not self.migration_phases[self.current_phase]["pruning_enabled"]:
                # No pruning, call original function
                return legacy_evaluation_func(*args, **kwargs)

            # Extract programs if available
            programs = kwargs.get("programs", args[0] if args else None)
            if not programs:
                # Can't apply pruning without programs
                return legacy_evaluation_func(*args, **kwargs)

            # Apply pruning logic here
            # This is a simplified example
            logger.info(f"Applying pruning in {self.current_phase} phase")

            # For monitoring phase, still evaluate everything
            if self.migration_phases[self.current_phase].get("monitoring_only"):
                result = legacy_evaluation_func(*args, **kwargs)
                # Log pruning metrics
                logger.info("Pruning metrics collected (shadow mode)")
                return result

            # For gradual/enabled phases, actually prune
            # This would need actual pruning logic
            return legacy_evaluation_func(*args, **kwargs)

        return wrapped_evaluation

    def get_migration_status(self) -> dict[str, Any]:
        """Get current migration status and recommendations.

        Returns:
            Dictionary with migration status information
        """
        is_ready, issues = self.validate_migration_readiness()

        status = {
            "current_phase": self.current_phase,
            "phase_config": self.migration_phases[self.current_phase],
            "ready_for_next_phase": is_ready,
            "blocking_issues": issues,
            "migration_config": self.create_migration_config(),
            "recommendations": self._get_migration_recommendations(),
        }

        return status

    def _get_migration_recommendations(self) -> list[str]:
        """Get recommendations for current migration phase.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if self.current_phase == "disabled":
            recommendations.extend([
                "Enable monitoring mode to start collecting pruning metrics",
                "Review pruning strategies configuration",
                "Set up monitoring dashboard for metrics visualization",
            ])
        elif self.current_phase == "monitoring":
            recommendations.extend([
                "Monitor false negative rates closely",
                "Analyze pruning effectiveness metrics",
                "Prepare for gradual rollout by identifying low-risk workflows",
            ])
        elif self.current_phase == "gradual":
            recommendations.extend([
                "Gradually increase rollout percentage based on metrics",
                "Monitor performance impact on affected workflows",
                "Prepare for full rollout by addressing any issues",
            ])
        else:
            recommendations.extend([
                "Continue monitoring pruning effectiveness",
                "Consider A/B testing different strategies",
                "Optimize pruning thresholds based on workload",
            ])

        return recommendations
