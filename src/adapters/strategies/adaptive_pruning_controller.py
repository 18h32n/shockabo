"""Adaptive pruning controller that adjusts aggressiveness based on system load.

This module implements dynamic pruning control that adapts to evaluation queue
length and system resource usage.
"""

import time
from dataclasses import dataclass, field

import structlog
import yaml

from src.adapters.strategies.program_pruner import ProgramPruner, PruningConfig
from src.domain.models import PruningMetrics

logger = structlog.get_logger(__name__)


@dataclass
class AdaptivePruningConfig:
    """Configuration for adaptive pruning behavior."""
    base_aggressiveness: float = 0.5
    min_aggressiveness: float = 0.3
    max_aggressiveness: float = 0.9
    queue_threshold_low: int = 100
    queue_threshold_high: int = 1000
    adaptation_rate: float = 0.1
    memory_pressure_threshold: float = 0.8
    cpu_pressure_threshold: float = 0.9


@dataclass
class SystemMetrics:
    """Current system resource metrics."""
    queue_length: int
    memory_usage_percent: float
    cpu_usage_percent: float
    active_evaluations: int
    timestamp: float = field(default_factory=time.time)


class AdaptivePruningController:
    """Controls pruning aggressiveness based on system load and performance."""

    def __init__(self, config_path: str | None = None):
        """Initialize the adaptive pruning controller.

        Args:
            config_path: Path to pruning strategies YAML config
        """
        self.logger = structlog.get_logger(__name__).bind(
            service="adaptive_pruning_controller"
        )

        # Load strategies configuration
        self.strategies = {}
        self.adaptive_config = AdaptivePruningConfig()

        if config_path:
            self._load_strategies(config_path)

        # Current pruning instance
        self._current_pruner = None
        self._current_strategy_id = None
        self._current_aggressiveness = self.adaptive_config.base_aggressiveness

        # Performance tracking
        self.performance_history: list[PruningMetrics] = []
        self.adaptation_history: list[dict] = []

    def _load_strategies(self, config_path: str) -> None:
        """Load pruning strategies from YAML configuration.

        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load strategies
            for strategy_id, strategy_config in config.get('pruning_strategies', {}).items():
                self.strategies[strategy_id] = PruningConfig(
                    strategy_id=strategy_config['strategy_id'],
                    aggressiveness=strategy_config['aggressiveness'],
                    syntax_checks=strategy_config['syntax_checks'],
                    pattern_checks=strategy_config['pattern_checks'],
                    partial_execution=strategy_config['partial_execution'],
                    confidence_threshold=strategy_config['confidence_threshold'],
                    max_partial_ops=strategy_config['max_partial_ops'],
                    timeout_ms=strategy_config['timeout_ms'],
                    memory_limit_mb=strategy_config.get('memory_limit_mb', 10.0),
                    enable_caching=strategy_config.get('enable_caching', True),
                    cache_size=strategy_config.get('cache_size', 10000)
                )

            # Load adaptive settings if available
            if 'experimental' in config['pruning_strategies']:
                adaptive_settings = config['pruning_strategies']['experimental'].get('adaptive_settings', {})
                self.adaptive_config = AdaptivePruningConfig(
                    base_aggressiveness=config['pruning_strategies']['experimental']['aggressiveness'],
                    min_aggressiveness=adaptive_settings.get('min_aggressiveness', 0.3),
                    max_aggressiveness=adaptive_settings.get('max_aggressiveness', 0.9),
                    queue_threshold_low=adaptive_settings.get('queue_threshold_low', 100),
                    queue_threshold_high=adaptive_settings.get('queue_threshold_high', 1000),
                    adaptation_rate=adaptive_settings.get('adaptation_rate', 0.1)
                )

            self.logger.info(
                "strategies_loaded",
                num_strategies=len(self.strategies),
                strategies=list(self.strategies.keys())
            )

        except Exception as e:
            self.logger.error("failed_to_load_strategies", error=str(e))

    def get_pruner(self, system_metrics: SystemMetrics) -> ProgramPruner:
        """Get a pruner instance with adapted aggressiveness.

        Args:
            system_metrics: Current system resource metrics

        Returns:
            ProgramPruner configured for current conditions
        """
        # Calculate target aggressiveness
        target_aggressiveness = self._calculate_target_aggressiveness(system_metrics)

        # Smoothly adapt current aggressiveness
        self._current_aggressiveness = self._smooth_adaptation(
            self._current_aggressiveness,
            target_aggressiveness
        )

        # Select strategy based on aggressiveness
        strategy_id = self._select_strategy(self._current_aggressiveness)

        # Create or update pruner if strategy changed
        if strategy_id != self._current_strategy_id:
            config = self._create_adapted_config(strategy_id, self._current_aggressiveness)
            self._current_pruner = ProgramPruner(config)
            self._current_strategy_id = strategy_id

            self.logger.info(
                "pruner_adapted",
                strategy_id=strategy_id,
                aggressiveness=self._current_aggressiveness,
                queue_length=system_metrics.queue_length
            )

        # Record adaptation
        self.adaptation_history.append({
            "timestamp": time.time(),
            "strategy_id": strategy_id,
            "aggressiveness": self._current_aggressiveness,
            "queue_length": system_metrics.queue_length,
            "memory_usage": system_metrics.memory_usage_percent,
            "cpu_usage": system_metrics.cpu_usage_percent
        })

        return self._current_pruner

    def _calculate_target_aggressiveness(self, metrics: SystemMetrics) -> float:
        """Calculate target aggressiveness based on system metrics.

        Args:
            metrics: Current system metrics

        Returns:
            Target aggressiveness value between min and max
        """
        # Base calculation on queue length
        if metrics.queue_length <= self.adaptive_config.queue_threshold_low:
            queue_factor = 0.0
        elif metrics.queue_length >= self.adaptive_config.queue_threshold_high:
            queue_factor = 1.0
        else:
            # Linear interpolation
            range_size = self.adaptive_config.queue_threshold_high - self.adaptive_config.queue_threshold_low
            queue_factor = (metrics.queue_length - self.adaptive_config.queue_threshold_low) / range_size

        # Adjust for resource pressure
        resource_factor = 0.0

        if metrics.memory_usage_percent > self.adaptive_config.memory_pressure_threshold:
            resource_factor += 0.3

        if metrics.cpu_usage_percent > self.adaptive_config.cpu_pressure_threshold:
            resource_factor += 0.2

        # Combine factors
        combined_factor = min(queue_factor + resource_factor, 1.0)

        # Map to aggressiveness range
        aggressiveness_range = (
            self.adaptive_config.max_aggressiveness -
            self.adaptive_config.min_aggressiveness
        )

        target = (
            self.adaptive_config.min_aggressiveness +
            combined_factor * aggressiveness_range
        )

        return max(self.adaptive_config.min_aggressiveness,
                  min(self.adaptive_config.max_aggressiveness, target))

    def _smooth_adaptation(self, current: float, target: float) -> float:
        """Smoothly adapt from current to target aggressiveness.

        Args:
            current: Current aggressiveness
            target: Target aggressiveness

        Returns:
            New aggressiveness value
        """
        diff = target - current
        adjustment = diff * self.adaptive_config.adaptation_rate

        return current + adjustment

    def _select_strategy(self, aggressiveness: float) -> str:
        """Select best strategy for given aggressiveness.

        Args:
            aggressiveness: Target aggressiveness value

        Returns:
            Strategy ID
        """
        # Find closest matching strategy
        best_strategy = None
        min_diff = float('inf')

        for strategy_id, config in self.strategies.items():
            diff = abs(config.aggressiveness - aggressiveness)
            if diff < min_diff:
                min_diff = diff
                best_strategy = strategy_id

        return best_strategy or "balanced-v1"

    def _create_adapted_config(self, strategy_id: str, aggressiveness: float) -> PruningConfig:
        """Create pruning config with adapted aggressiveness.

        Args:
            strategy_id: Base strategy ID
            aggressiveness: Target aggressiveness

        Returns:
            Adapted pruning configuration
        """
        base_config = self.strategies.get(strategy_id)
        if not base_config:
            # Fallback to default
            base_config = PruningConfig()

        # Create new config with adapted aggressiveness
        adapted_config = PruningConfig(
            strategy_id=f"{strategy_id}-adapted",
            aggressiveness=aggressiveness,
            syntax_checks=base_config.syntax_checks,
            pattern_checks=base_config.pattern_checks,
            partial_execution=base_config.partial_execution,
            confidence_threshold=self._adapt_threshold(
                base_config.confidence_threshold, aggressiveness
            ),
            max_partial_ops=base_config.max_partial_ops,
            timeout_ms=base_config.timeout_ms,
            memory_limit_mb=base_config.memory_limit_mb,
            enable_caching=base_config.enable_caching,
            cache_size=base_config.cache_size
        )

        return adapted_config

    def _adapt_threshold(self, base_threshold: float, aggressiveness: float) -> float:
        """Adapt confidence threshold based on aggressiveness.

        Higher aggressiveness = lower confidence threshold

        Args:
            base_threshold: Base confidence threshold
            aggressiveness: Current aggressiveness

        Returns:
            Adapted threshold
        """
        # Linear mapping: aggressiveness 0.3 -> threshold 0.8
        #                 aggressiveness 0.9 -> threshold 0.4
        adjusted = 1.0 - (0.4 + 0.4 * aggressiveness)
        return max(0.3, min(0.9, adjusted))

    def update_performance_metrics(self, metrics: PruningMetrics) -> None:
        """Update controller with latest performance metrics.

        Args:
            metrics: Latest pruning performance metrics
        """
        self.performance_history.append(metrics)

        # Keep only recent history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]

        # Log if false negative rate is too high
        if metrics.false_negative_rate > 0.05:
            self.logger.warning(
                "high_false_negative_rate",
                rate=metrics.false_negative_rate,
                strategy_id=metrics.strategy_id
            )

    def get_performance_summary(self) -> dict:
        """Get summary of pruning performance.

        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_history:
            return {
                "num_evaluations": 0,
                "avg_pruning_rate": 0.0,
                "avg_false_negative_rate": 0.0,
                "avg_time_saved_ms": 0.0,
                "adaptations": len(self.adaptation_history)
            }

        recent_metrics = self.performance_history[-100:]  # Last 100 evaluations

        return {
            "num_evaluations": len(self.performance_history),
            "avg_pruning_rate": sum(m.pruning_rate for m in recent_metrics) / len(recent_metrics),
            "avg_false_negative_rate": sum(m.false_negative_rate for m in recent_metrics) / len(recent_metrics),
            "avg_time_saved_ms": sum(m.time_saved_ms for m in recent_metrics) / len(recent_metrics),
            "current_aggressiveness": self._current_aggressiveness,
            "current_strategy": self._current_strategy_id,
            "adaptations": len(self.adaptation_history)
        }

    def should_bypass_pruning(self, task_metadata: dict) -> bool:
        """Check if pruning should be bypassed for a specific task.

        Args:
            task_metadata: Task metadata including priority, size, etc.

        Returns:
            True if pruning should be bypassed
        """
        # Check bypass conditions
        if task_metadata.get('priority') == 'HIGH':
            return True

        if task_metadata.get('validation_mode', False):
            return True

        if task_metadata.get('program_size', float('inf')) < 3:
            return True

        return False
