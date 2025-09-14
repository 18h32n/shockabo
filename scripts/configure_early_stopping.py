#!/usr/bin/env python3
"""
Early Stopping Configuration Script for Story 1.5

Interactive script to configure early stopping parameters for different training scenarios.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.domain.services.training_orchestrator import EarlyStoppingConfig, TrainingConfig
from src.utils.early_stopping_utils import EarlyStoppingConfigManager, validate_early_stopping_config


logger = logging.getLogger(__name__)


def create_interactive_config() -> EarlyStoppingConfig:
    """Create early stopping configuration interactively."""
    print("\n" + "="*60)
    print("INTERACTIVE EARLY STOPPING CONFIGURATION")
    print("="*60)
    
    # Basic parameters
    print("\n1. Basic Early Stopping Parameters:")
    patience = int(input("Enter patience (number of validations without improvement) [5]: ") or "5")
    min_delta = float(input("Enter minimum improvement threshold [0.01]: ") or "0.01")
    
    # Monitor metric
    print("\n2. Monitoring Configuration:")
    print("Available metrics: validation_accuracy, validation_loss, training_loss")
    monitor_metric = input("Enter metric to monitor [validation_accuracy]: ") or "validation_accuracy"
    
    if "loss" in monitor_metric.lower():
        mode = "min"
        print("Using 'min' mode for loss metric")
    else:
        mode = "max"
        print("Using 'max' mode for accuracy metric")
    
    # Advanced options
    print("\n3. Advanced Options:")
    restore_best = input("Restore best weights on early stopping? [y/N]: ").lower().startswith('y')
    baseline_str = input("Enter baseline value (optional, press Enter to skip): ")
    baseline = float(baseline_str) if baseline_str else None
    
    # Auto-save configuration
    print("\n4. Auto-Save Configuration:")
    auto_save = input("Enable auto-save? [Y/n]: ").lower() != 'n'
    if auto_save:
        save_interval = int(input("Auto-save interval in minutes [10]: ") or "10")
        save_on_improvement = input("Save on improvement? [Y/n]: ").lower() != 'n'
    else:
        save_interval = 10
        save_on_improvement = False
    
    # Auto-resume configuration
    print("\n5. Auto-Resume Configuration:")
    auto_resume = input("Enable auto-resume? [Y/n]: ").lower() != 'n'
    if auto_resume:
        resume_from_best = input("Resume from best checkpoint? [Y/n]: ").lower() != 'n'
        resume_threshold = float(input("Resume threshold in hours [0.5]: ") or "0.5")
    else:
        resume_from_best = True
        resume_threshold = 0.5
    
    # Create configuration
    config = EarlyStoppingConfig(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best,
        monitor_metric=monitor_metric,
        mode=mode,
        baseline=baseline,
        verbose=True,
        auto_save_enabled=auto_save,
        auto_save_interval_minutes=save_interval,
        auto_save_on_improvement=save_on_improvement,
        auto_resume_enabled=auto_resume,
        resume_from_best=resume_from_best,
        resume_threshold_hours=resume_threshold,
    )
    
    return config


def create_scenario_config(scenario: str) -> EarlyStoppingConfig:
    """Create configuration for predefined scenarios."""
    config_manager = EarlyStoppingConfigManager()
    
    if scenario in ["conservative", "aggressive", "balanced", "memory_aware", "time_critical"]:
        config = config_manager.get_config(scenario)
        if config:
            return config
    
    # Scenario-specific configurations
    if scenario == "8b_model":
        return config_manager.create_adaptive_config("8B", 45, 24)
    elif scenario == "1b_model":
        return config_manager.create_adaptive_config("1B", 30, 12)
    elif scenario == "quick_test":
        return EarlyStoppingConfig(
            patience=2,
            min_delta=0.02,
            auto_save_interval_minutes=3,
            resume_threshold_hours=0.1
        )
    elif scenario == "long_training":
        return EarlyStoppingConfig(
            patience=15,
            min_delta=0.005,
            auto_save_interval_minutes=20,
            resume_threshold_hours=2.0
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def display_config_summary(config: EarlyStoppingConfig, name: str = "Configuration") -> None:
    """Display configuration summary."""
    print(f"\n{name} Summary:")
    print("-" * (len(name) + 9))
    print(f"Patience: {config.patience} validations")
    print(f"Min Delta: {config.min_delta}")
    print(f"Monitor: {config.monitor_metric} ({'maximize' if config.mode == 'max' else 'minimize'})")
    print(f"Restore Best Weights: {config.restore_best_weights}")
    
    if config.baseline is not None:
        print(f"Baseline Threshold: {config.baseline}")
    
    print(f"Auto-Save: {config.auto_save_enabled}")
    if config.auto_save_enabled:
        print(f"  - Interval: {config.auto_save_interval_minutes} minutes")
        print(f"  - On Improvement: {config.auto_save_on_improvement}")
    
    print(f"Auto-Resume: {config.auto_resume_enabled}")
    if config.auto_resume_enabled:
        print(f"  - From Best: {config.resume_from_best}")
        print(f"  - Threshold: {config.resume_threshold_hours} hours")


def save_training_config(
    early_stopping_config: EarlyStoppingConfig,
    output_path: Path,
    additional_params: Dict[str, Any] = None
) -> None:
    """Save complete training configuration with early stopping."""
    # Create complete training config
    training_config = TrainingConfig(
        early_stopping=early_stopping_config,
        **(additional_params or {})
    )
    
    # Convert to dictionary
    config_dict = {
        "training_config": {
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "batch_size": training_config.batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "warmup_steps": training_config.warmup_steps,
            "max_grad_norm": training_config.max_grad_norm,
            "validation_frequency": training_config.validation_frequency,
            "checkpoint_frequency": training_config.checkpoint_frequency,
            "max_training_time": training_config.max_training_time,
            "target_accuracy": training_config.target_accuracy,
            "memory_limit_mb": training_config.memory_limit_mb,
            "mixed_precision": training_config.mixed_precision,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "use_qlora": training_config.use_qlora,
            "lora_rank": training_config.lora_rank,
            "lora_alpha": training_config.lora_alpha,
            "lora_dropout": training_config.lora_dropout,
            "use_flash_attention": training_config.use_flash_attention,
            "selective_checkpointing": training_config.selective_checkpointing,
            "checkpointing_layers": training_config.checkpointing_layers,
        },
        "early_stopping": {
            "patience": early_stopping_config.patience,
            "min_delta": early_stopping_config.min_delta,
            "restore_best_weights": early_stopping_config.restore_best_weights,
            "monitor_metric": early_stopping_config.monitor_metric,
            "mode": early_stopping_config.mode,
            "baseline": early_stopping_config.baseline,
            "verbose": early_stopping_config.verbose,
            "auto_save_enabled": early_stopping_config.auto_save_enabled,
            "auto_save_interval_minutes": early_stopping_config.auto_save_interval_minutes,
            "auto_save_on_improvement": early_stopping_config.auto_save_on_improvement,
            "auto_resume_enabled": early_stopping_config.auto_resume_enabled,
            "resume_from_best": early_stopping_config.resume_from_best,
            "resume_threshold_hours": early_stopping_config.resume_threshold_hours,
        },
        "metadata": {
            "created_at": "2025-09-13T00:00:00",
            "created_by": "configure_early_stopping.py",
            "version": "1.0",
        }
    }
    
    # Save configuration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nConfiguration saved to: {output_path}")


def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate existing configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Extract early stopping config
    es_config_data = config_data.get("early_stopping", {})
    early_stopping_config = EarlyStoppingConfig(**es_config_data)
    
    # Validate configuration
    warnings = validate_early_stopping_config(early_stopping_config)
    if warnings:
        print(f"\nConfiguration warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("✅ Configuration validation passed")
    
    return config_data


def main():
    """Main configuration script."""
    parser = argparse.ArgumentParser(description="Configure early stopping for Story 1.5")
    parser.add_argument(
        "--mode",
        choices=["interactive", "scenario", "validate"],
        default="interactive",
        help="Configuration mode"
    )
    parser.add_argument(
        "--scenario",
        choices=["conservative", "aggressive", "balanced", "memory_aware", "time_critical", 
                "8b_model", "1b_model", "quick_test", "long_training"],
        help="Predefined scenario for scenario mode"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/early_stopping_config.json"),
        help="Output path for configuration file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input configuration file to validate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Early Stopping Configuration Tool for Story 1.5")
    print("=" * 50)
    
    try:
        if args.mode == "validate":
            if not args.input:
                print("❌ Input file required for validation mode")
                sys.exit(1)
            
            print(f"Validating configuration: {args.input}")
            config_data = load_and_validate_config(args.input)
            
            # Display current config
            es_config = EarlyStoppingConfig(**config_data["early_stopping"])
            display_config_summary(es_config, "Current Configuration")
            
        elif args.mode == "scenario":
            if not args.scenario:
                print("❌ Scenario required for scenario mode")
                print("Available scenarios:", 
                      "conservative, aggressive, balanced, memory_aware, time_critical,")
                print("                     8b_model, 1b_model, quick_test, long_training")
                sys.exit(1)
            
            print(f"Creating configuration for scenario: {args.scenario}")
            config = create_scenario_config(args.scenario)
            display_config_summary(config, f"Scenario: {args.scenario}")
            
            # Validate and save
            warnings = validate_early_stopping_config(config)
            if warnings:
                print(f"\nWarnings for scenario '{args.scenario}':")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
            
            save_training_config(config, args.output)
            
        else:  # interactive mode
            config = create_interactive_config()
            display_config_summary(config, "Your Configuration")
            
            # Validate configuration
            warnings = validate_early_stopping_config(config)
            if warnings:
                print("\nConfiguration Warnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
                
                proceed = input("\nProceed with warnings? [y/N]: ")
                if not proceed.lower().startswith('y'):
                    print("Configuration cancelled.")
                    sys.exit(0)
            
            # Save configuration
            save_training_config(config, args.output)
        
        print("\n✅ Configuration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()