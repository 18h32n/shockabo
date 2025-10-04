"""Validation manager for coordinating pruning validation and refinement.

This module manages the validation process, tracks metrics, and triggers
automatic refinements when performance degrades.
"""

import time
from dataclasses import dataclass
from datetime import datetime

import structlog

from src.adapters.strategies.false_negative_detector import FalseNegativeDetector, ValidationConfig
from src.adapters.strategies.program_pruner import ProgramPruner, PruningConfig
from src.domain.dsl.base import Operation
from src.domain.dsl.types import Grid
from src.domain.models import PruningDecision, PruningMetrics, PruningResult

logger = structlog.get_logger(__name__)


@dataclass
class ValidationMode:
    """Configuration for validation mode operation."""
    enabled: bool = False
    full_validation_rate: float = 1.0  # Validate 100% in validation mode
    comparison_mode: bool = True  # Compare against non-pruned baseline
    collect_detailed_metrics: bool = True
    max_validation_duration_s: float = 3600  # 1 hour max


class PruningValidationManager:
    """Manages pruning validation and continuous improvement."""

    def __init__(
        self,
        evaluation_service,
        validation_config: ValidationConfig | None = None
    ):
        """Initialize the validation manager.

        Args:
            evaluation_service: Service for full evaluation
            validation_config: Configuration for validation
        """
        self.evaluation_service = evaluation_service
        self.validation_config = validation_config or ValidationConfig()
        self.logger = structlog.get_logger(__name__).bind(
            service="pruning_validation_manager"
        )

        # Initialize false negative detector
        self.fn_detector = FalseNegativeDetector(
            evaluation_service,
            self.validation_config
        )

        # Validation mode state
        self.validation_mode = ValidationMode()
        self._validation_start_time: float | None = None

        # Metrics tracking
        self.validation_metrics: list[PruningMetrics] = []
        self.refinement_history: list[dict] = []

        # Pruner instances for A/B comparison
        self._current_pruner: ProgramPruner | None = None
        self._baseline_pruner: ProgramPruner | None = None

    async def validate_pruning_batch(
        self,
        pruner: ProgramPruner,
        programs: list[list[Operation]],
        test_inputs: list[Grid]
    ) -> tuple[list[PruningResult], PruningMetrics]:
        """Validate a batch of programs with pruning and track metrics.

        Args:
            pruner: Pruner instance to use
            programs: List of DSL programs
            test_inputs: Test input grids

        Returns:
            Tuple of (pruning results, metrics)
        """
        start_time = time.perf_counter()

        # Get pruning decisions
        pruning_results = await pruner.batch_prune(programs, test_inputs)

        # Separate accepted and rejected programs
        validation_candidates = []
        for _i, (program, result) in enumerate(zip(programs, pruning_results, strict=False)):
            if result.decision != PruningDecision.ACCEPT:
                # Candidate for validation
                test_input = test_inputs[0] if test_inputs else None
                if test_input and (
                    self.validation_mode.enabled or
                    self.fn_detector.should_validate(result)
                ):
                    validation_candidates.append((result, program, test_input))

        # Validate samples
        if validation_candidates:
            fn_events = await self.fn_detector.batch_validate(validation_candidates)

            # Count false negatives
            false_negatives = sum(1 for event in fn_events if event is not None)
        else:
            false_negatives = 0

        # Calculate metrics
        pruning_time = (time.perf_counter() - start_time) * 1000
        num_pruned = sum(1 for r in pruning_results if r.decision != PruningDecision.ACCEPT)

        metrics = PruningMetrics(
            strategy_id=pruner.config.strategy_id,
            total_programs=len(programs),
            programs_pruned=num_pruned,
            pruning_rate=num_pruned / len(programs) if programs else 0,
            false_negatives=false_negatives,
            false_negative_rate=false_negatives / num_pruned if num_pruned > 0 else 0,
            avg_pruning_time_ms=pruning_time / len(programs) if programs else 0,
            time_saved_ms=0,  # Would be calculated with full eval comparison
            timestamp=datetime.now()
        )

        self.validation_metrics.append(metrics)

        # Check if refinement needed
        if self.fn_detector.stats.false_negative_rate > self.validation_config.alert_threshold:
            await self._trigger_refinement(pruner)

        return pruning_results, metrics

    async def run_validation_mode(
        self,
        pruner: ProgramPruner,
        test_programs: list[list[Operation]],
        test_inputs: list[Grid],
        baseline_config: PruningConfig | None = None
    ) -> dict:
        """Run comprehensive validation mode to assess pruning effectiveness.

        Args:
            pruner: Pruner to validate
            test_programs: Test programs for validation
            test_inputs: Test input grids
            baseline_config: Optional baseline config for comparison

        Returns:
            Dictionary with validation results
        """
        self.logger.info(
            "entering_validation_mode",
            num_programs=len(test_programs),
            strategy_id=pruner.config.strategy_id
        )

        # Enable validation mode
        self.validation_mode.enabled = True
        self._validation_start_time = time.time()

        # Reset statistics
        self.fn_detector.reset_statistics()

        # Create baseline pruner if comparison enabled
        if self.validation_mode.comparison_mode and baseline_config:
            self._baseline_pruner = ProgramPruner(baseline_config)

        results = {
            "test_results": [],
            "baseline_results": [] if self._baseline_pruner else None,
            "false_negative_analysis": None,
            "performance_comparison": None
        }

        try:
            # Process in batches
            batch_size = 100
            for i in range(0, len(test_programs), batch_size):
                batch_programs = test_programs[i:i + batch_size]
                batch_inputs = test_inputs[i:i + batch_size] if test_inputs else []

                # Test pruner
                test_results, test_metrics = await self.validate_pruning_batch(
                    pruner, batch_programs, batch_inputs
                )
                results["test_results"].extend(test_results)

                # Baseline comparison
                if self._baseline_pruner:
                    baseline_results, baseline_metrics = await self.validate_pruning_batch(
                        self._baseline_pruner, batch_programs, batch_inputs
                    )
                    results["baseline_results"].extend(baseline_results)

                # Check time limit
                if (time.time() - self._validation_start_time >
                    self.validation_mode.max_validation_duration_s):
                    self.logger.warning("validation_mode_timeout")
                    break

            # Analyze results
            results["false_negative_analysis"] = self.fn_detector.get_validation_report()
            results["performance_comparison"] = self._compare_performance()

            return results

        finally:
            # Exit validation mode
            self.validation_mode.enabled = False
            self._validation_start_time = None

    async def _trigger_refinement(self, pruner: ProgramPruner) -> None:
        """Trigger automatic refinement of pruning rules.

        Args:
            pruner: Pruner instance to refine
        """
        self.logger.info("triggering_pruning_refinement")

        # Get suggested refinements
        suggestions = self.fn_detector.suggest_rule_refinements()

        # Apply refinements
        for suggestion in suggestions:
            if suggestion["type"] == "global_adjustment":
                # Reduce aggressiveness
                new_aggressiveness = max(
                    0.1,
                    pruner.config.aggressiveness - 0.1
                )

                self.logger.info(
                    "adjusting_aggressiveness",
                    old=pruner.config.aggressiveness,
                    new=new_aggressiveness
                )

                pruner.config.aggressiveness = new_aggressiveness

            elif suggestion["type"] == "adjust_threshold":
                # Adjust confidence threshold
                pruner.config.confidence_threshold = suggestion["suggested"]

        # Record refinement
        self.refinement_history.append({
            "timestamp": time.time(),
            "trigger": "high_false_negative_rate",
            "suggestions": suggestions,
            "applied": True
        })

    def _compare_performance(self) -> dict:
        """Compare test pruner performance against baseline.

        Returns:
            Performance comparison metrics
        """
        if not self.validation_metrics:
            return {}

        # Get recent metrics for test pruner
        test_metrics = [
            m for m in self.validation_metrics[-100:]
            if m.strategy_id == self._current_pruner.config.strategy_id
        ]

        if not test_metrics:
            return {}

        comparison = {
            "test_pruner": {
                "strategy_id": self._current_pruner.config.strategy_id,
                "avg_pruning_rate": sum(m.pruning_rate for m in test_metrics) / len(test_metrics),
                "avg_false_negative_rate": sum(m.false_negative_rate for m in test_metrics) / len(test_metrics),
                "avg_pruning_time_ms": sum(m.avg_pruning_time_ms for m in test_metrics) / len(test_metrics)
            }
        }

        # Add baseline comparison if available
        if self._baseline_pruner:
            baseline_metrics = [
                m for m in self.validation_metrics[-100:]
                if m.strategy_id == self._baseline_pruner.config.strategy_id
            ]

            if baseline_metrics:
                comparison["baseline_pruner"] = {
                    "strategy_id": self._baseline_pruner.config.strategy_id,
                    "avg_pruning_rate": sum(m.pruning_rate for m in baseline_metrics) / len(baseline_metrics),
                    "avg_false_negative_rate": sum(m.false_negative_rate for m in baseline_metrics) / len(baseline_metrics),
                    "avg_pruning_time_ms": sum(m.avg_pruning_time_ms for m in baseline_metrics) / len(baseline_metrics)
                }

                # Calculate improvements
                comparison["improvements"] = {
                    "pruning_rate_change": (
                        comparison["test_pruner"]["avg_pruning_rate"] -
                        comparison["baseline_pruner"]["avg_pruning_rate"]
                    ),
                    "false_negative_rate_change": (
                        comparison["test_pruner"]["avg_false_negative_rate"] -
                        comparison["baseline_pruner"]["avg_false_negative_rate"]
                    ),
                    "speed_improvement": (
                        comparison["baseline_pruner"]["avg_pruning_time_ms"] -
                        comparison["test_pruner"]["avg_pruning_time_ms"]
                    )
                }

        return comparison

    def get_validation_summary(self) -> dict:
        """Get summary of validation activities and findings.

        Returns:
            Dictionary with validation summary
        """
        return {
            "total_validations": self.fn_detector.stats.total_validated,
            "false_negative_rate": self.fn_detector.stats.false_negative_rate,
            "precision": self.fn_detector.stats.precision,
            "problematic_patterns": self.fn_detector.get_problematic_patterns(),
            "refinements_applied": len(self.refinement_history),
            "validation_mode_active": self.validation_mode.enabled,
            "metrics_collected": len(self.validation_metrics)
        }
