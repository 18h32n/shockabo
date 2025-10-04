"""
Early Stopping Utilities for Story 1.5 Task 5

Comprehensive early stopping utilities with monitoring, analysis, and configuration management.
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.domain.services.training_orchestrator import EarlyStoppingConfig

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingState:
    """State tracking for early stopping mechanism."""
    best_value: float
    best_epoch: int
    best_step: int
    patience_counter: int
    total_epochs: int
    total_steps: int
    improvement_history: list[tuple[int, float]]
    triggered: bool = False
    trigger_reason: str = ""
    trigger_epoch: int | None = None
    trigger_step: int | None = None


class EarlyStoppingMonitor:
    """Monitor and analyze early stopping behavior."""

    def __init__(self, config: EarlyStoppingConfig):
        """Initialize early stopping monitor."""
        self.config = config
        self.states: dict[str, EarlyStoppingState] = {}
        self.monitoring_data: list[dict[str, Any]] = []

    def track_training_session(
        self,
        session_id: str,
        epoch: int,
        step: int,
        metric_value: float,
        all_metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Track training session and determine early stopping.

        Args:
            session_id: Unique training session ID
            epoch: Current epoch
            step: Current step
            metric_value: Value of monitored metric
            all_metrics: All available metrics

        Returns:
            Tuple of (should_stop, reason)
        """
        # Initialize state if new session
        if session_id not in self.states:
            self.states[session_id] = EarlyStoppingState(
                best_value=metric_value if self.config.mode == "max" else float('inf'),
                best_epoch=epoch,
                best_step=step,
                patience_counter=0,
                total_epochs=0,
                total_steps=0,
                improvement_history=[(step, metric_value)]
            )

        state = self.states[session_id]
        state.total_epochs = max(state.total_epochs, epoch)
        state.total_steps = max(state.total_steps, step)

        # Check for improvement
        improved = False
        if self.config.mode == "max":
            improved = metric_value > (state.best_value + self.config.min_delta)
        else:  # mode == "min"
            improved = metric_value < (state.best_value - self.config.min_delta)

        if improved:
            state.best_value = metric_value
            state.best_epoch = epoch
            state.best_step = step
            state.patience_counter = 0
            state.improvement_history.append((step, metric_value))

            if self.config.verbose:
                logger.info(f"Session {session_id}: New best {self.config.monitor_metric}: {metric_value:.4f}")
        else:
            state.patience_counter += 1

        # Record monitoring data
        self.monitoring_data.append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "metric_value": metric_value,
            "best_value": state.best_value,
            "patience_counter": state.patience_counter,
            "improved": improved,
            "all_metrics": all_metrics.copy()
        })

        # Check stopping criteria
        if state.patience_counter >= self.config.patience:
            state.triggered = True
            state.trigger_reason = f"Patience exceeded ({self.config.patience})"
            state.trigger_epoch = epoch
            state.trigger_step = step
            return True, state.trigger_reason

        return False, ""

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of training session."""
        if session_id not in self.states:
            return None

        state = self.states[session_id]

        return {
            "session_id": session_id,
            "best_value": state.best_value,
            "best_epoch": state.best_epoch,
            "best_step": state.best_step,
            "total_epochs": state.total_epochs,
            "total_steps": state.total_steps,
            "improvements_count": len(state.improvement_history),
            "final_patience_counter": state.patience_counter,
            "early_stopped": state.triggered,
            "trigger_reason": state.trigger_reason,
            "trigger_epoch": state.trigger_epoch,
            "trigger_step": state.trigger_step,
        }

    def analyze_training_efficiency(self, session_id: str) -> dict[str, Any]:
        """Analyze training efficiency and early stopping effectiveness."""
        if session_id not in self.states:
            return {"error": "Session not found"}

        state = self.states[session_id]
        session_data = [d for d in self.monitoring_data if d["session_id"] == session_id]

        if not session_data:
            return {"error": "No monitoring data found"}

        # Calculate efficiency metrics
        total_time = len(session_data)  # Number of validation points
        improvement_ratio = len(state.improvement_history) / total_time if total_time > 0 else 0

        # Calculate wasted training time if early stopping wasn't used
        if state.triggered:
            wasted_steps = state.total_steps - state.trigger_step
            efficiency_gain = wasted_steps / state.total_steps if state.total_steps > 0 else 0
        else:
            efficiency_gain = 0

        # Analyze improvement pattern
        if len(state.improvement_history) >= 2:
            improvement_steps = [step for step, _ in state.improvement_history]
            improvement_intervals = np.diff(improvement_steps)
            avg_improvement_interval = np.mean(improvement_intervals) if len(improvement_intervals) > 0 else 0
        else:
            avg_improvement_interval = 0

        return {
            "session_id": session_id,
            "total_validation_points": total_time,
            "improvement_ratio": improvement_ratio,
            "efficiency_gain": efficiency_gain,
            "avg_improvement_interval": avg_improvement_interval,
            "early_stopping_effective": state.triggered and efficiency_gain > 0.1,
            "recommendations": self._generate_recommendations(state, improvement_ratio, efficiency_gain)
        }

    def _generate_recommendations(
        self,
        state: EarlyStoppingState,
        improvement_ratio: float,
        efficiency_gain: float
    ) -> list[str]:
        """Generate recommendations for early stopping configuration."""
        recommendations = []

        if state.patience_counter < self.config.patience // 2:
            recommendations.append("Consider reducing patience for faster early stopping")
        elif not state.triggered:
            recommendations.append("Consider increasing patience to allow more training")

        if improvement_ratio < 0.1:
            recommendations.append("Very few improvements - check learning rate or model capacity")
        elif improvement_ratio > 0.5:
            recommendations.append("Frequent improvements - early stopping working well")

        if efficiency_gain > 0.2:
            recommendations.append("Early stopping saved significant training time")
        elif efficiency_gain < 0.05 and state.triggered:
            recommendations.append("Early stopping triggered too late - consider tighter criteria")

        if not recommendations:
            recommendations.append("Early stopping configuration appears optimal")

        return recommendations

    def save_analysis_report(self, output_path: Path) -> None:
        """Save comprehensive analysis report."""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "sessions": {},
            "monitoring_data_points": len(self.monitoring_data),
        }

        # Analyze each session
        for session_id in self.states:
            summary = self.get_session_summary(session_id)
            efficiency = self.analyze_training_efficiency(session_id)

            report["sessions"][session_id] = {
                "summary": summary,
                "efficiency_analysis": efficiency,
            }

        # Overall statistics
        total_sessions = len(self.states)
        early_stopped_sessions = sum(1 for state in self.states.values() if state.triggered)

        report["overall_statistics"] = {
            "total_sessions": total_sessions,
            "early_stopped_sessions": early_stopped_sessions,
            "early_stopping_rate": early_stopped_sessions / total_sessions if total_sessions > 0 else 0,
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Early stopping analysis report saved to: {output_path}")


class EarlyStoppingConfigManager:
    """Manage early stopping configurations for different scenarios."""

    def __init__(self):
        """Initialize configuration manager."""
        self.configs: dict[str, EarlyStoppingConfig] = {}
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default configurations for common scenarios."""

        # Conservative early stopping (high patience)
        self.configs["conservative"] = EarlyStoppingConfig(
            patience=10,
            min_delta=0.005,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=15,
            auto_resume_enabled=True,
        )

        # Aggressive early stopping (low patience)
        self.configs["aggressive"] = EarlyStoppingConfig(
            patience=3,
            min_delta=0.02,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=5,
            auto_resume_enabled=True,
        )

        # Balanced early stopping (Story 1.5 default)
        self.configs["balanced"] = EarlyStoppingConfig(
            patience=5,
            min_delta=0.01,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=10,
            auto_resume_enabled=True,
        )

        # Memory-aware early stopping
        self.configs["memory_aware"] = EarlyStoppingConfig(
            patience=4,
            min_delta=0.01,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=8,
            auto_resume_enabled=True,
            baseline=0.4,  # Stop if below 40% accuracy
        )

        # Time-critical early stopping
        self.configs["time_critical"] = EarlyStoppingConfig(
            patience=2,
            min_delta=0.015,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=5,
            auto_resume_enabled=True,
        )

    def get_config(self, config_name: str) -> EarlyStoppingConfig | None:
        """Get configuration by name."""
        return self.configs.get(config_name)

    def create_adaptive_config(
        self,
        model_size: str,
        time_budget_minutes: int,
        memory_budget_gb: int
    ) -> EarlyStoppingConfig:
        """
        Create adaptive configuration based on constraints.

        Args:
            model_size: Size of model ("1B", "8B", etc.)
            time_budget_minutes: Available training time
            memory_budget_gb: Available memory in GB

        Returns:
            Optimized early stopping configuration
        """
        # Base configuration
        base_config = self.configs["balanced"]

        # Adjust based on model size
        if "8B" in model_size:
            patience = 4  # Reduced for larger models
            min_delta = 0.01
            save_interval = 8
        elif "1B" in model_size:
            patience = 6
            min_delta = 0.005
            save_interval = 12
        else:
            patience = base_config.patience
            min_delta = base_config.min_delta
            save_interval = base_config.auto_save_interval_minutes

        # Adjust based on time budget
        if time_budget_minutes < 30:
            patience = max(2, patience // 2)
            save_interval = max(3, save_interval // 2)
        elif time_budget_minutes > 120:
            patience = min(10, patience + 2)
            save_interval = min(20, save_interval + 5)

        # Adjust based on memory budget
        if memory_budget_gb < 16:
            # More aggressive for memory-constrained environments
            patience = max(2, patience - 1)
            save_interval = max(5, save_interval - 2)

        return EarlyStoppingConfig(
            patience=patience,
            min_delta=min_delta,
            monitor_metric="validation_accuracy",
            mode="max",
            verbose=True,
            auto_save_enabled=True,
            auto_save_interval_minutes=save_interval,
            auto_resume_enabled=True,
        )

    def save_config(self, name: str, config: EarlyStoppingConfig, path: Path) -> None:
        """Save configuration to file."""
        config_data = asdict(config)
        config_data["name"] = name
        config_data["created_at"] = datetime.now().isoformat()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved early stopping config '{name}' to {path}")

    def load_config(self, path: Path) -> EarlyStoppingConfig | None:
        """Load configuration from file."""
        try:
            with open(path) as f:
                config_data = json.load(f)

            # Remove metadata fields
            config_data.pop("name", None)
            config_data.pop("created_at", None)

            return EarlyStoppingConfig(**config_data)

        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return None


def create_early_stopping_visualization(
    training_history: list[dict[str, Any]],
    early_stopping_info: dict[str, Any],
    output_path: Path
) -> None:
    """
    Create visualization of training progress and early stopping.

    Args:
        training_history: List of training metrics over time
        early_stopping_info: Early stopping information
        output_path: Path to save visualization
    """
    if not training_history:
        logger.warning("No training history provided for visualization")
        return

    # Extract data
    steps = [entry.get("step", 0) for entry in training_history]
    validation_accuracy = [entry.get("validation_accuracy", 0) for entry in training_history]
    loss = [entry.get("loss", 0) for entry in training_history]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot validation accuracy
    ax1.plot(steps, validation_accuracy, 'b-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Training Progress with Early Stopping')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Mark early stopping point if triggered
    if early_stopping_info.get("early_stopping_triggered"):
        best_step = None
        best_accuracy = early_stopping_info.get("best_accuracy", 0)

        # Find step with best accuracy
        for i, acc in enumerate(validation_accuracy):
            if abs(acc - best_accuracy) < 0.001:
                best_step = steps[i]
                break

        if best_step is not None:
            ax1.axvline(x=best_step, color='g', linestyle='--',
                       label=f'Best Performance (Step {best_step})', alpha=0.7)
            ax1.legend()

    # Plot loss
    ax2.plot(steps, loss, 'r-', label='Training Loss', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add early stopping information as text
    info_text = f"""Early Stopping Info:
Triggered: {early_stopping_info.get('early_stopping_triggered', False)}
Best Accuracy: {early_stopping_info.get('best_accuracy', 0):.4f}
Total Steps: {early_stopping_info.get('total_steps', 0)}
Training Time: {early_stopping_info.get('training_time', 0):.2f}s"""

    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8})

    plt.tight_layout()

    # Save visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Early stopping visualization saved to: {output_path}")


def validate_early_stopping_config(config: EarlyStoppingConfig) -> list[str]:
    """
    Validate early stopping configuration and return warnings.

    Args:
        config: Early stopping configuration to validate

    Returns:
        List of validation warnings
    """
    warnings = []

    # Check patience
    if config.patience < 1:
        warnings.append("Patience must be at least 1")
    elif config.patience > 20:
        warnings.append("Very high patience may waste training time")

    # Check min_delta
    if config.min_delta < 0:
        warnings.append("min_delta should be non-negative")
    elif config.min_delta > 0.1:
        warnings.append("Very high min_delta may prevent convergence")

    # Check mode and metric compatibility
    accuracy_metrics = ["accuracy", "validation_accuracy", "val_acc"]
    loss_metrics = ["loss", "validation_loss", "val_loss"]

    if config.mode == "max" and any(loss_metric in config.monitor_metric.lower()
                                   for loss_metric in loss_metrics):
        warnings.append("Using 'max' mode with loss metric - consider 'min' mode")
    elif config.mode == "min" and any(acc_metric in config.monitor_metric.lower()
                                     for acc_metric in accuracy_metrics):
        warnings.append("Using 'min' mode with accuracy metric - consider 'max' mode")

    # Check auto-save configuration
    if config.auto_save_enabled and config.auto_save_interval_minutes < 1:
        warnings.append("Auto-save interval too short - may impact performance")
    elif config.auto_save_enabled and config.auto_save_interval_minutes > 60:
        warnings.append("Auto-save interval very long - may lose progress on crashes")

    # Check resume threshold
    if config.auto_resume_enabled and config.resume_threshold_hours > 24:
        warnings.append("Resume threshold very high - may resume very old training")

    return warnings


# Example usage and testing functions
def main():
    """Demonstrate early stopping utilities."""
    logging.basicConfig(level=logging.INFO)

    # Create configuration manager
    config_manager = EarlyStoppingConfigManager()

    # Test adaptive configuration
    adaptive_config = config_manager.create_adaptive_config(
        model_size="8B",
        time_budget_minutes=45,
        memory_budget_gb=24
    )

    logger.info(f"Created adaptive config: patience={adaptive_config.patience}, "
                f"min_delta={adaptive_config.min_delta}, "
                f"save_interval={adaptive_config.auto_save_interval_minutes}min")

    # Validate configuration
    warnings = validate_early_stopping_config(adaptive_config)
    if warnings:
        logger.warning(f"Configuration warnings: {warnings}")
    else:
        logger.info("Configuration validation passed")

    # Create monitor and simulate training
    monitor = EarlyStoppingMonitor(adaptive_config)

    # Simulate training session
    session_id = "test_session_001"

    # Simulate training progress with decreasing improvements
    accuracies = [0.20, 0.35, 0.42, 0.48, 0.51, 0.52, 0.521, 0.519, 0.518, 0.517]

    for step, accuracy in enumerate(accuracies):
        should_stop, reason = monitor.track_training_session(
            session_id=session_id,
            epoch=step // 3,
            step=step,
            metric_value=accuracy,
            all_metrics={"loss": 1.0 - accuracy, "lr": 1e-5}
        )

        if should_stop:
            logger.info(f"Early stopping at step {step}: {reason}")
            break

    # Generate analysis
    summary = monitor.get_session_summary(session_id)
    efficiency = monitor.analyze_training_efficiency(session_id)

    logger.info(f"Training summary: {summary}")
    logger.info(f"Efficiency analysis: {efficiency}")

    # Save analysis report
    report_path = Path("logs/early_stopping_analysis.json")
    monitor.save_analysis_report(report_path)

    print("\n" + "="*60)
    print("EARLY STOPPING UTILITIES DEMONSTRATION COMPLETE")
    print("="*60)
    print("✓ Adaptive configuration creation")
    print("✓ Configuration validation")
    print("✓ Training session monitoring")
    print("✓ Efficiency analysis")
    print("✓ Analysis report generation")
    print("="*60)


if __name__ == "__main__":
    main()
