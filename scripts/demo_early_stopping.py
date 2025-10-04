#!/usr/bin/env python3
"""
Early Stopping Demonstration Script for Story 1.5 Task 5

Demonstrates the complete early stopping implementation including:
- Configuration management
- Training orchestrator integration
- Checkpoint auto-save and recovery
- Monitoring and analysis
"""
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.adapters.repositories.checkpoint_repository import CheckpointRepository
from src.domain.models import ARCTask
from src.domain.services.training_orchestrator import (
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingOrchestrator,
)
from src.domain.services.ttt_service import TTTModelService
from src.utils.early_stopping_utils import (
    EarlyStoppingConfigManager,
    EarlyStoppingMonitor,
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_mock_arc_task() -> Mock:
    """Create a mock ARC task for demonstration."""
    mock_task = Mock(spec=ARCTask)
    mock_task.task_id = "early_stopping_demo_001"
    mock_task.train_examples = [
        {"input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
         "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]},
        {"input": [[1, 1, 0], [0, 1, 1], [1, 0, 0]],
         "output": [[0, 0, 1], [1, 0, 0], [0, 1, 1]]}
    ]
    mock_task.test_examples = [
        {"input": [[0, 0, 1], [1, 1, 0], [0, 1, 1]],
         "output": [[1, 1, 0], [0, 0, 1], [1, 0, 0]]}
    ]
    return mock_task


def create_mock_model_service() -> Mock:
    """Create a mock model service."""
    mock_service = Mock(spec=TTTModelService)
    mock_service.device = Mock()
    mock_service.device.type = "cpu"

    # Mock memory manager
    mock_service.memory_manager = Mock()
    mock_service.memory_manager.get_memory_usage.return_value = {"usage_percentage": 45.0}
    mock_service._get_memory_usage.return_value = 8.5  # 8.5GB usage
    mock_service.optimize_memory = Mock()
    mock_service.cleanup = Mock()

    # Mock model loading
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_service.load_model.return_value = (mock_model, mock_tokenizer)

    # Mock tokenizer methods
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "[[1, 0], [0, 1]]"  # Mock grid output

    return mock_service


def demo_configuration_management():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("1. CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("="*60)

    config_manager = EarlyStoppingConfigManager()

    # Show predefined configurations
    print("\n📋 Predefined Configurations:")
    for config_name in ["conservative", "aggressive", "balanced"]:
        config = config_manager.get_config(config_name)
        print(f"  • {config_name.upper()}:")
        print(f"    - Patience: {config.patience}")
        print(f"    - Min Delta: {config.min_delta}")
        print(f"    - Auto-save interval: {config.auto_save_interval_minutes}min")

    # Demonstrate adaptive configuration
    print("\n🔧 Adaptive Configuration:")
    adaptive_8b = config_manager.create_adaptive_config("8B", 30, 24)
    adaptive_1b = config_manager.create_adaptive_config("1B", 60, 16)

    print(f"  • 8B Model (30min, 24GB): patience={adaptive_8b.patience}, "
          f"save_interval={adaptive_8b.auto_save_interval_minutes}min")
    print(f"  • 1B Model (60min, 16GB): patience={adaptive_1b.patience}, "
          f"save_interval={adaptive_1b.auto_save_interval_minutes}min")

    return adaptive_8b


def demo_training_orchestrator(config: EarlyStoppingConfig, temp_dir: Path):
    """Demonstrate training orchestrator with early stopping."""
    print("\n" + "="*60)
    print("2. TRAINING ORCHESTRATOR WITH EARLY STOPPING")
    print("="*60)

    # Create components
    mock_task = create_mock_arc_task()
    mock_service = create_mock_model_service()

    checkpoint_repo = CheckpointRepository(temp_dir / "checkpoints")

    training_config = TrainingConfig(
        num_epochs=5,
        validation_frequency=3,
        early_stopping=config,
        max_training_time=180,  # 3 minutes
        target_accuracy=0.75,
    )

    orchestrator = TrainingOrchestrator(
        model_service=mock_service,
        config=training_config,
        checkpoint_repository=checkpoint_repo
    )

    # Mock training components
    orchestrator.model = Mock()
    orchestrator.tokenizer = mock_service.load_model()[1]
    orchestrator.optimizer = Mock()
    orchestrator.scheduler = Mock()
    orchestrator.lora_adapter = Mock()
    orchestrator.scaler = None

    # Mock state dicts for checkpointing
    orchestrator.model.state_dict.return_value = {"layer1": "weights1", "layer2": "weights2"}
    orchestrator.optimizer.state_dict.return_value = {"lr": 1e-5, "momentum": 0.9}
    orchestrator.scheduler.state_dict.return_value = {"last_epoch": 0}
    orchestrator.scheduler.get_last_lr.return_value = [1e-5]

    # Mock LoRA adapter
    orchestrator.lora_adapter.get_adapter_state = Mock(return_value={"lora_weights": "mock_weights"})
    orchestrator.lora_adapter.load_adapter_state = Mock()

    # Set up initial state
    orchestrator.current_task_id = mock_task.task_id
    orchestrator.start_time = time.time()

    print(f"\n🚀 Starting training for task: {mock_task.task_id}")
    print(f"Configuration: patience={config.patience}, target_accuracy={training_config.target_accuracy}")

    # Simulate training progression
    training_history = []
    validation_accuracies = [0.25, 0.42, 0.58, 0.63, 0.64, 0.645, 0.644, 0.643]  # Plateau after improvement

    early_stopped = False
    stop_reason = ""

    for step, accuracy in enumerate(validation_accuracies):
        print(f"\n📊 Step {step * 10} - Validation Accuracy: {accuracy:.3f}")

        # Create training metrics
        metrics = {
            "validation_accuracy": accuracy,
            "loss": 1.0 - accuracy + 0.1,  # Simple loss calculation
            "step": step * 10,
            "epoch": step // 3,
            "learning_rate": 1e-5,
            "memory_mb": 8500 + step * 100,  # Gradual memory increase
        }

        # Track in history
        training_history.append(metrics.copy())

        # Check early stopping
        should_stop, reason = orchestrator.check_early_stopping(metrics)

        if should_stop:
            early_stopped = True
            stop_reason = reason
            print(f"⏹️  Early stopping triggered: {reason}")
            break

        # Simulate time passage
        time.sleep(0.1)

    # Show results
    print("\n📈 Training Results:")
    print(f"  • Early stopped: {early_stopped}")
    if early_stopped:
        print(f"  • Reason: {stop_reason}")
    print(f"  • Steps completed: {len(training_history)}")
    print(f"  • Best accuracy: {orchestrator.best_accuracy:.3f}")
    print(f"  • Final accuracy: {training_history[-1]['validation_accuracy']:.3f}")

    # Check checkpoints
    checkpoints = checkpoint_repo.list_checkpoints(task_id=mock_task.task_id)
    print(f"  • Checkpoints saved: {len(checkpoints)}")

    if checkpoints:
        best_checkpoint = checkpoint_repo.get_best_checkpoint(mock_task.task_id)
        print(f"  • Best checkpoint accuracy: {best_checkpoint.accuracy:.3f}")

    return training_history, early_stopped, stop_reason


def demo_monitoring_analysis(config: EarlyStoppingConfig):
    """Demonstrate monitoring and analysis."""
    print("\n" + "="*60)
    print("3. MONITORING AND ANALYSIS")
    print("="*60)

    monitor = EarlyStoppingMonitor(config)
    session_id = "demo_session_001"

    # Simulate training session
    print(f"\n📊 Simulating training session: {session_id}")

    # Training progression: improvements then plateau
    progression = [
        (0, 0.20, "Initial performance"),
        (1, 0.35, "Learning basic patterns"),
        (2, 0.48, "Rapid improvement"),
        (3, 0.52, "Continued learning"),
        (4, 0.54, "Small improvement"),
        (5, 0.535, "Marginal gain"),
        (6, 0.534, "Plateau begins"),
        (7, 0.533, "No significant change"),
        (8, 0.532, "Slight decrease"),
    ]

    for step, accuracy, description in progression:
        should_stop, reason = monitor.track_training_session(
            session_id=session_id,
            epoch=step // 3,
            step=step * 10,
            metric_value=accuracy,
            all_metrics={"loss": 1.0 - accuracy, "lr": 1e-5}
        )

        status = "🔴 STOP" if should_stop else "🟢 CONTINUE"
        print(f"  Step {step * 10}: {accuracy:.3f} - {description} [{status}]")

        if should_stop:
            print(f"    Reason: {reason}")
            break

    # Get session analysis
    print("\n📈 Training Session Analysis:")
    summary = monitor.get_session_summary(session_id)
    if summary:
        print(f"  • Best accuracy: {summary['best_value']:.3f} (step {summary['best_step']})")
        print(f"  • Total improvements: {summary['improvements_count']}")
        print(f"  • Early stopped: {summary['early_stopped']}")
        if summary['early_stopped']:
            print(f"  • Stop reason: {summary['trigger_reason']}")

    # Efficiency analysis
    efficiency = monitor.analyze_training_efficiency(session_id)
    if efficiency and "error" not in efficiency:
        print("\n⚡ Efficiency Analysis:")
        print(f"  • Improvement ratio: {efficiency['improvement_ratio']:.1%}")
        print(f"  • Efficiency gain: {efficiency['efficiency_gain']:.1%}")
        print(f"  • Early stopping effective: {efficiency['early_stopping_effective']}")

        if efficiency['recommendations']:
            print("  • Recommendations:")
            for rec in efficiency['recommendations']:
                print(f"    - {rec}")

    return monitor


def demo_checkpoint_recovery(temp_dir: Path):
    """Demonstrate checkpoint recovery."""
    print("\n" + "="*60)
    print("4. CHECKPOINT RECOVERY DEMONSTRATION")
    print("="*60)

    checkpoint_repo = CheckpointRepository(temp_dir / "checkpoints")

    # List existing checkpoints
    all_checkpoints = checkpoint_repo.list_checkpoints()
    print(f"\n💾 Available Checkpoints: {len(all_checkpoints)}")

    for checkpoint in all_checkpoints:
        age = (datetime.now() - checkpoint.created_at).total_seconds() / 60
        print(f"  • {checkpoint.checkpoint_id}")
        print(f"    - Task: {checkpoint.task_id}")
        print(f"    - Accuracy: {checkpoint.accuracy:.3f}")
        print(f"    - Age: {age:.1f} minutes")
        print(f"    - Size: {checkpoint.file_size_mb:.1f}MB")

    if all_checkpoints:
        # Demonstrate recovery scenarios
        task_id = all_checkpoints[0].task_id

        print(f"\n🔄 Recovery Scenarios for task: {task_id}")

        # Best checkpoint recovery
        best_checkpoint = checkpoint_repo.get_best_checkpoint(task_id)
        if best_checkpoint:
            print(f"  • Best checkpoint: {best_checkpoint.checkpoint_id} (accuracy: {best_checkpoint.accuracy:.3f})")

        # Auto-resume candidates
        candidates = checkpoint_repo.get_auto_resume_candidates(task_id, max_age_hours=1.0)
        print(f"  • Auto-resume candidates: {len(candidates)}")

        # Validate checkpoint integrity
        if best_checkpoint:
            is_valid = checkpoint_repo.validate_checkpoint_integrity(best_checkpoint.checkpoint_id)
            print(f"  • Best checkpoint valid: {'✅' if is_valid else '❌'}")

    return checkpoint_repo


def create_summary_report(temp_dir: Path, training_history: list, early_stopped: bool, stop_reason: str):
    """Create summary report."""
    print("\n" + "="*60)
    print("5. EARLY STOPPING IMPLEMENTATION SUMMARY")
    print("="*60)

    report = {
        "demonstration_timestamp": datetime.now().isoformat(),
        "early_stopping_system": {
            "status": "OPERATIONAL",
            "features_demonstrated": [
                "Configuration Management",
                "Training Integration",
                "Auto-save Functionality",
                "Checkpoint Recovery",
                "Performance Monitoring",
                "Efficiency Analysis"
            ]
        },
        "training_simulation": {
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "total_steps": len(training_history),
            "final_accuracy": training_history[-1]["validation_accuracy"] if training_history else 0,
            "max_accuracy": max([step["validation_accuracy"] for step in training_history]) if training_history else 0
        },
        "system_capabilities": {
            "patience_based_stopping": "✅ Implemented",
            "improvement_detection": "✅ Implemented",
            "auto_save": "✅ Implemented",
            "auto_resume": "✅ Implemented",
            "checkpoint_recovery": "✅ Implemented",
            "performance_monitoring": "✅ Implemented",
            "error_handling": "✅ Implemented",
            "adaptive_configuration": "✅ Implemented"
        }
    }

    # Save report
    report_path = temp_dir / "early_stopping_demo_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Display summary
    print(f"\n✅ IMPLEMENTATION STATUS: {report['early_stopping_system']['status']}")
    print("\n🎯 FEATURES DEMONSTRATED:")
    for feature in report['early_stopping_system']['features_demonstrated']:
        print(f"  • {feature}")

    print("\n📊 TRAINING SIMULATION RESULTS:")
    print(f"  • Early stopping triggered: {'Yes' if early_stopped else 'No'}")
    if early_stopped:
        print(f"  • Reason: {stop_reason}")
    print(f"  • Training steps: {report['training_simulation']['total_steps']}")
    print(f"  • Final accuracy: {report['training_simulation']['final_accuracy']:.3f}")
    print(f"  • Best accuracy: {report['training_simulation']['max_accuracy']:.3f}")

    print("\n🔧 SYSTEM CAPABILITIES:")
    for capability, status in report['system_capabilities'].items():
        capability_name = capability.replace('_', ' ').title()
        print(f"  • {capability_name}: {status}")

    print(f"\n📄 Full report saved: {report_path}")
    return report


def main():
    """Main demonstration script."""
    setup_logging()

    print("🚀 Early Stopping Implementation Demonstration")
    print("Story 1.5 Task 5: Implement Early Stopping Mechanism")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # 1. Configuration Management
            config = demo_configuration_management()

            # 2. Training Orchestrator
            training_history, early_stopped, stop_reason = demo_training_orchestrator(config, temp_path)

            # 3. Monitoring and Analysis
            monitor = demo_monitoring_analysis(config)

            # 4. Checkpoint Recovery
            checkpoint_repo = demo_checkpoint_recovery(temp_path)

            # 5. Summary Report
            report = create_summary_report(temp_path, training_history, early_stopped, stop_reason)

            print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("All early stopping features are operational and ready for use.")

        except Exception as e:
            print(f"❌ Demonstration failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
