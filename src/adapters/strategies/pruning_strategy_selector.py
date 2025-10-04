"""Automatic pruning strategy selection based on A/B test results.

This module implements intelligent strategy selection that adapts based on
performance metrics and system conditions.
"""

import logging

import yaml

from src.adapters.strategies.pruning_ab_controller import ABTestController
from src.adapters.strategies.pruning_metrics_collector import PruningMetricsCollector
from src.domain.models import PruningStrategy
from src.infrastructure.config import get_platform_config

logger = logging.getLogger(__name__)


class PruningStrategySelector:
    """Automatically select optimal pruning strategies based on A/B results."""

    def __init__(
        self,
        config_path: str,
        ab_controller: ABTestController | None = None,
        metrics_collector: PruningMetricsCollector | None = None,
    ):
        """Initialize strategy selector.

        Args:
            config_path: Path to pruning strategies configuration file
            ab_controller: Optional A/B test controller instance
            metrics_collector: Optional metrics collector instance
        """
        self.config_path = config_path
        self.ab_controller = ab_controller
        self.metrics_collector = metrics_collector

        # Load configuration
        self._load_config()

        # Track current system state
        self.current_state = {
            "queue_length": 0,
            "memory_pressure": 0.0,
            "platform": get_platform_config().platform_type,
        }

    def _load_config(self):
        """Load pruning strategies configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Parse strategies
        self.strategies = {}
        for strategy_config in self.config["pruning_strategies"].values():
            strategy = PruningStrategy(
                strategy_id=strategy_config["strategy_id"],
                name=strategy_config["name"],
                aggressiveness=strategy_config["aggressiveness"],
                syntax_checks=strategy_config["syntax_checks"],
                pattern_checks=strategy_config["pattern_checks"],
                partial_execution=strategy_config["partial_execution"],
                confidence_threshold=strategy_config["confidence_threshold"],
                max_partial_ops=strategy_config["max_partial_ops"],
                timeout_ms=strategy_config["timeout_ms"],
            )
            self.strategies[strategy.strategy_id] = strategy

    def select_strategy(
        self,
        queue_length: int | None = None,
        memory_pressure: float | None = None,
        force_strategy: str | None = None,
    ) -> PruningStrategy:
        """Select optimal pruning strategy based on current conditions.

        Args:
            queue_length: Current evaluation queue length
            memory_pressure: Current memory pressure (0-1)
            force_strategy: Optional strategy ID to force selection

        Returns:
            Selected pruning strategy
        """
        # Update system state
        if queue_length is not None:
            self.current_state["queue_length"] = queue_length
        if memory_pressure is not None:
            self.current_state["memory_pressure"] = memory_pressure

        # Check if forced strategy is requested
        if force_strategy and force_strategy in self.strategies:
            logger.info(f"Using forced strategy: {force_strategy}")
            return self.strategies[force_strategy]

        # Check critical bypass conditions
        if self._should_bypass_pruning():
            # Return a minimal pruning strategy
            return self._get_bypass_strategy()

        # Use A/B controller if available and enabled
        if self.ab_controller and self.config["ab_test_config"]["enabled"]:
            strategy = self.ab_controller.select_strategy(
                task_features=self._extract_task_features()
            )
            logger.debug(f"A/B controller selected: {strategy.strategy_id}")
            return strategy

        # Otherwise, use rule-based selection
        return self._rule_based_selection()

    def _should_bypass_pruning(self) -> bool:
        """Check if pruning should be bypassed entirely.

        Returns:
            True if pruning should be bypassed
        """
        # Check memory pressure
        platform_settings = self.config["platform_settings"].get(
            self.current_state["platform"], {}
        )
        memory_threshold = platform_settings.get("memory_pressure_threshold", 0.9)

        if self.current_state["memory_pressure"] > memory_threshold:
            logger.warning(
                f"Memory pressure too high ({self.current_state['memory_pressure']:.2f}), "
                f"bypassing pruning"
            )
            return True

        return False

    def _get_bypass_strategy(self) -> PruningStrategy:
        """Get minimal strategy for bypass scenarios.

        Returns:
            Minimal pruning strategy that only does basic checks
        """
        return PruningStrategy(
            strategy_id="bypass-minimal",
            name="Minimal Bypass Strategy",
            aggressiveness=0.1,
            syntax_checks=True,  # Only syntax checks
            pattern_checks=False,
            partial_execution=False,
            confidence_threshold=0.9,
            max_partial_ops=0,
            timeout_ms=10,
        )

    def _extract_task_features(self) -> dict[str, float]:
        """Extract features for strategy selection.

        Returns:
            Dictionary of task features
        """
        return {
            "queue_length": float(self.current_state["queue_length"]),
            "memory_pressure": self.current_state["memory_pressure"],
            "platform_kaggle": 1.0 if self.current_state["platform"] == "kaggle" else 0.0,
            "platform_colab": 1.0 if self.current_state["platform"] == "colab" else 0.0,
            "platform_paperspace": 1.0 if self.current_state["platform"] == "paperspace" else 0.0,
        }

    def _rule_based_selection(self) -> PruningStrategy:
        """Select strategy using rule-based logic.

        Returns:
            Selected pruning strategy
        """
        platform = self.current_state["platform"]
        queue_length = self.current_state["queue_length"]
        memory_pressure = self.current_state["memory_pressure"]

        # Get platform default
        platform_settings = self.config["platform_settings"].get(platform, {})
        default_strategy_id = platform_settings.get(
            "default_strategy", "balanced-v1"
        )

        # Adjust based on queue length
        if queue_length > 1000:
            # High load - use aggressive pruning
            strategy_id = "aggressive-v1"
        elif queue_length > 500:
            # Medium load - use balanced
            strategy_id = "balanced-v1"
        elif queue_length < 100:
            # Low load - can afford conservative
            strategy_id = "conservative-v1"
        else:
            # Default case
            strategy_id = default_strategy_id

        # Further adjust based on memory pressure
        if memory_pressure > 0.7:
            # High memory pressure - shift toward aggressive
            if strategy_id == "conservative-v1":
                strategy_id = "balanced-v1"
            elif strategy_id == "balanced-v1":
                strategy_id = "aggressive-v1"

        logger.info(
            f"Rule-based selection: {strategy_id} "
            f"(queue={queue_length}, memory={memory_pressure:.2f})"
        )

        return self.strategies.get(strategy_id, self.strategies["balanced-v1"])

    def get_adaptive_strategy(self) -> PruningStrategy:
        """Get adaptive strategy that adjusts based on conditions.

        Returns:
            Adaptive pruning strategy
        """
        # Start with experimental adaptive strategy
        base_strategy = self.strategies.get("experimental-v1")
        if not base_strategy:
            return self._rule_based_selection()

        # Get adaptive settings
        adaptive_config = self.config["pruning_strategies"]["experimental"].get(
            "adaptive_settings", {}
        )

        # Calculate adjusted aggressiveness
        queue_length = self.current_state["queue_length"]
        min_aggressive = adaptive_config.get("min_aggressiveness", 0.3)
        max_aggressive = adaptive_config.get("max_aggressiveness", 0.9)
        queue_low = adaptive_config.get("queue_threshold_low", 100)
        queue_high = adaptive_config.get("queue_threshold_high", 1000)

        if queue_length <= queue_low:
            aggressiveness = min_aggressive
        elif queue_length >= queue_high:
            aggressiveness = max_aggressive
        else:
            # Linear interpolation
            ratio = (queue_length - queue_low) / (queue_high - queue_low)
            aggressiveness = min_aggressive + ratio * (max_aggressive - min_aggressive)

        # Create adjusted strategy
        return PruningStrategy(
            strategy_id=f"{base_strategy.strategy_id}-adapted",
            name=f"{base_strategy.name} (Adapted)",
            aggressiveness=aggressiveness,
            syntax_checks=base_strategy.syntax_checks,
            pattern_checks=base_strategy.pattern_checks,
            partial_execution=base_strategy.partial_execution,
            confidence_threshold=base_strategy.confidence_threshold,
            max_partial_ops=base_strategy.max_partial_ops,
            timeout_ms=base_strategy.timeout_ms,
        )

    def update_selection_metrics(
        self,
        strategy_id: str,
        success: bool,
        time_saved_ms: float,
    ):
        """Update metrics for strategy selection feedback.

        Args:
            strategy_id: ID of the strategy used
            success: Whether the selection was successful
            time_saved_ms: Time saved by pruning
        """
        # This can be used for online learning/adaptation
        logger.debug(
            f"Strategy {strategy_id} feedback: "
            f"success={success}, time_saved={time_saved_ms}ms"
        )

    def get_recommended_strategy(self) -> tuple[str, dict[str, float]]:
        """Get recommended strategy based on all available data.

        Returns:
            Tuple of (strategy_id, confidence_scores)
        """
        if not self.ab_controller:
            # No A/B data available
            strategy = self._rule_based_selection()
            return strategy.strategy_id, {"rule_based": 1.0}

        # Get best strategy from A/B testing
        best_strategy_id, metrics = self.ab_controller.get_best_strategy()

        if not best_strategy_id:
            # Not enough data yet
            strategy = self._rule_based_selection()
            return strategy.strategy_id, {"rule_based": 1.0}

        # Calculate confidence based on sample size
        samples = metrics.get("samples", 0)
        confidence = min(samples / self.config["ab_test_config"]["min_samples_per_strategy"], 1.0)

        return best_strategy_id, {
            "ab_testing": confidence,
            "rule_based": 1.0 - confidence,
        }

    def export_selection_report(self) -> dict:
        """Export strategy selection report.

        Returns:
            Dictionary containing selection statistics and recommendations
        """
        report = {
            "current_state": self.current_state.copy(),
            "available_strategies": list(self.strategies.keys()),
        }

        if self.ab_controller:
            report["ab_test_results"] = self.ab_controller.export_results()
            report["performance_summary"] = self.ab_controller.get_performance_summary()

        if self.metrics_collector:
            report["strategy_comparison"] = self.metrics_collector.get_strategy_comparison()
            report["efficiency_metrics"] = self.metrics_collector.get_efficiency_metrics()

        # Add recommendation
        recommended_id, confidence = self.get_recommended_strategy()
        report["recommendation"] = {
            "strategy_id": recommended_id,
            "confidence_scores": confidence,
            "reasoning": self._explain_recommendation(recommended_id, confidence),
        }

        return report

    def _explain_recommendation(
        self,
        strategy_id: str,
        confidence_scores: dict[str, float]
    ) -> str:
        """Explain why a strategy was recommended.

        Args:
            strategy_id: Recommended strategy ID
            confidence_scores: Confidence scores by method

        Returns:
            Human-readable explanation
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return f"Unknown strategy: {strategy_id}"

        explanation = f"Recommended '{strategy.name}' because: "

        if confidence_scores.get("ab_testing", 0) > 0.5:
            explanation += "A/B testing shows it performs best. "
        else:
            explanation += "Rule-based selection for current conditions. "

        explanation += f"Queue length: {self.current_state['queue_length']}, "
        explanation += f"Memory pressure: {self.current_state['memory_pressure']:.2f}, "
        explanation += f"Platform: {self.current_state['platform']}"

        return explanation
