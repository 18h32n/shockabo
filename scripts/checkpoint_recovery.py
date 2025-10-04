#!/usr/bin/env python3
"""
Checkpoint Recovery Script for Story 1.5 Task 5

Script for managing, analyzing, and recovering from training checkpoints.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.adapters.repositories.checkpoint_repository import CheckpointMetadata, CheckpointRepository
from src.utils.comprehensive_error_handling import CheckpointManager

logger = logging.getLogger(__name__)


class CheckpointRecoveryTool:
    """Tool for checkpoint recovery and management."""

    def __init__(self, checkpoint_path: Path | None = None):
        """Initialize recovery tool."""
        self.checkpoint_repo = CheckpointRepository(checkpoint_path)
        self.backup_manager = CheckpointManager()

    def list_checkpoints(
        self,
        task_id: str | None = None,
        show_details: bool = False
    ) -> list[CheckpointMetadata]:
        """List available checkpoints."""
        checkpoints = self.checkpoint_repo.list_checkpoints(task_id=task_id)

        if not checkpoints:
            print("No checkpoints found.")
            return []

        print(f"\nFound {len(checkpoints)} checkpoint(s):")
        print("-" * 80)

        if show_details:
            for checkpoint in checkpoints:
                self._display_checkpoint_details(checkpoint)
                print("-" * 80)
        else:
            # Table header
            print(f"{'ID':<20} {'Task':<15} {'Accuracy':<10} {'Size(MB)':<10} {'Created':<20}")
            print("-" * 80)

            for checkpoint in checkpoints:
                created_str = checkpoint.created_at.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{checkpoint.checkpoint_id:<20} {checkpoint.task_id:<15} "
                      f"{checkpoint.accuracy:<10.4f} {checkpoint.file_size_mb:<10.2f} {created_str:<20}")

        return checkpoints

    def _display_checkpoint_details(self, checkpoint: CheckpointMetadata) -> None:
        """Display detailed checkpoint information."""
        print(f"Checkpoint ID: {checkpoint.checkpoint_id}")
        print(f"Task ID: {checkpoint.task_id}")
        print(f"Model: {checkpoint.model_name}")
        print(f"Accuracy: {checkpoint.accuracy:.4f}")
        print(f"Training Time: {checkpoint.training_time:.2f}s")
        print(f"Memory Usage: {checkpoint.memory_usage_mb:.2f}MB")
        print(f"File Size: {checkpoint.file_size_mb:.2f}MB")
        print(f"LoRA Config: rank={checkpoint.lora_rank}, alpha={checkpoint.lora_alpha}")
        print(f"Created: {checkpoint.created_at}")
        print(f"Checksum: {checkpoint.checksum[:16]}...")
        if checkpoint.tags:
            print(f"Tags: {', '.join(checkpoint.tags)}")

    def validate_checkpoint_integrity(self, checkpoint_id: str) -> bool:
        """Validate checkpoint integrity."""
        print(f"Validating checkpoint: {checkpoint_id}")

        try:
            # Load and validate
            is_valid = self.checkpoint_repo.validate_checkpoint_integrity(checkpoint_id)

            if is_valid:
                print("✅ Checkpoint validation successful")

                # Additional checks
                checkpoint_data, metadata = self.checkpoint_repo.load_checkpoint(checkpoint_id)

                # Check required components
                model_state = checkpoint_data.get("model_state", {})
                required_components = [
                    "model_state_dict",
                    "training_metrics",
                    "lora_config"
                ]

                missing_components = [comp for comp in required_components
                                    if comp not in checkpoint_data]

                if missing_components:
                    print(f"⚠️  Missing components: {missing_components}")
                else:
                    print("✅ All required components present")

                # Check model state components
                model_components = [
                    "model_state_dict",
                    "optimizer_state_dict"
                ]

                missing_model_components = [comp for comp in model_components
                                          if comp not in model_state]

                if missing_model_components:
                    print(f"⚠️  Missing model components: {missing_model_components}")
                else:
                    print("✅ All model components present")

                return True
            else:
                print("❌ Checkpoint validation failed")
                return False

        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

    def analyze_training_session(self, task_id: str) -> dict[str, Any]:
        """Analyze training session progress from checkpoints."""
        checkpoints = self.checkpoint_repo.list_checkpoints(task_id=task_id)

        if not checkpoints:
            return {"error": "No checkpoints found for task"}

        print(f"\nAnalyzing training session for task: {task_id}")
        print("-" * 50)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x.created_at)

        # Calculate progress metrics
        accuracies = [cp.accuracy for cp in checkpoints]
        times = [cp.created_at for cp in checkpoints]

        total_training_time = sum(cp.training_time for cp in checkpoints)
        best_accuracy = max(accuracies)
        improvement_count = 0

        for i in range(1, len(accuracies)):
            if accuracies[i] > accuracies[i-1]:
                improvement_count += 1

        analysis = {
            "task_id": task_id,
            "total_checkpoints": len(checkpoints),
            "training_duration": (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0,
            "total_training_time": total_training_time,
            "best_accuracy": best_accuracy,
            "final_accuracy": accuracies[-1],
            "improvement_count": improvement_count,
            "improvement_rate": improvement_count / len(checkpoints) if checkpoints else 0,
            "average_checkpoint_interval": total_training_time / len(checkpoints) if checkpoints else 0,
        }

        # Display analysis
        print(f"Total Checkpoints: {analysis['total_checkpoints']}")
        print(f"Training Duration: {analysis['training_duration']:.2f}s")
        print(f"Total Training Time: {analysis['total_training_time']:.2f}s")
        print(f"Best Accuracy: {analysis['best_accuracy']:.4f}")
        print(f"Final Accuracy: {analysis['final_accuracy']:.4f}")
        print(f"Improvements: {analysis['improvement_count']}/{analysis['total_checkpoints']}")
        print(f"Improvement Rate: {analysis['improvement_rate']:.2%}")

        # Progress visualization
        print("\nAccuracy Progress:")
        for i, (checkpoint, accuracy) in enumerate(zip(checkpoints, accuracies, strict=False)):
            marker = "📈" if i > 0 and accuracy > accuracies[i-1] else "📉" if i > 0 and accuracy < accuracies[i-1] else "➡️"
            print(f"  {marker} {checkpoint.created_at.strftime('%H:%M:%S')}: {accuracy:.4f}")

        return analysis

    def recover_training_session(
        self,
        task_id: str,
        recovery_strategy: str = "best",
        dry_run: bool = False
    ) -> str | None:
        """
        Recover training session from checkpoints.
        
        Args:
            task_id: Task ID to recover
            recovery_strategy: "best", "latest", or specific checkpoint_id
            dry_run: If True, only show what would be recovered
            
        Returns:
            Checkpoint ID used for recovery, or None if failed
        """
        print(f"\nRecovery strategy: {recovery_strategy}")

        if recovery_strategy == "best":
            checkpoint = self.checkpoint_repo.get_best_checkpoint(task_id)
            if not checkpoint:
                print("❌ No best checkpoint found")
                return None
        elif recovery_strategy == "latest":
            checkpoints = self.checkpoint_repo.list_checkpoints(task_id=task_id)
            if not checkpoints:
                print("❌ No checkpoints found")
                return None
            checkpoint = checkpoints[0]  # Already sorted by accuracy and time
        else:
            # Specific checkpoint ID
            try:
                _, checkpoint = self.checkpoint_repo.load_checkpoint(recovery_strategy)
            except Exception as e:
                print(f"❌ Failed to load checkpoint {recovery_strategy}: {e}")
                return None

        print(f"Selected checkpoint: {checkpoint.checkpoint_id}")
        self._display_checkpoint_details(checkpoint)

        if dry_run:
            print("\n🔍 DRY RUN - No actual recovery performed")
            return checkpoint.checkpoint_id

        # Validate checkpoint before recovery
        if not self.validate_checkpoint_integrity(checkpoint.checkpoint_id):
            print("❌ Checkpoint validation failed - cannot recover")
            return None

        print(f"\n✅ Checkpoint {checkpoint.checkpoint_id} ready for recovery")
        print("Use this checkpoint ID to resume training in your training script")

        return checkpoint.checkpoint_id

    def cleanup_old_checkpoints(
        self,
        max_age_days: int = 7,
        keep_best_per_task: int = 3,
        dry_run: bool = False
    ) -> dict[str, Any]:
        """Clean up old checkpoints."""
        print(f"\nCleaning up checkpoints older than {max_age_days} days")
        print(f"Keeping {keep_best_per_task} best checkpoints per task")

        if dry_run:
            print("🔍 DRY RUN - No actual deletion performed")

        # Get cleanup statistics
        cleanup_stats = self.checkpoint_repo.cleanup_storage(keep_best_per_task)

        if not dry_run:
            print("✅ Cleanup completed:")
            print(f"  - Deleted: {cleanup_stats['deleted_count']} checkpoints")
            print(f"  - Freed: {cleanup_stats['freed_size_mb']:.2f}MB")
            print(f"  - Remaining: {cleanup_stats['final_count']} checkpoints")
        else:
            print(f"Would delete: {cleanup_stats['deleted_count']} checkpoints")
            print(f"Would free: {cleanup_stats['freed_size_mb']:.2f}MB")

        return cleanup_stats

    def export_checkpoint(self, checkpoint_id: str, export_path: Path) -> None:
        """Export checkpoint for backup or sharing."""
        print(f"Exporting checkpoint {checkpoint_id} to {export_path}")

        try:
            self.checkpoint_repo.export_checkpoint(checkpoint_id, export_path)
            print("✅ Export completed successfully")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def generate_recovery_report(self, output_path: Path) -> None:
        """Generate comprehensive recovery report."""
        print(f"Generating recovery report: {output_path}")

        # Collect all checkpoint data
        all_checkpoints = self.checkpoint_repo.list_checkpoints()

        # Group by task
        tasks = {}
        for checkpoint in all_checkpoints:
            if checkpoint.task_id not in tasks:
                tasks[checkpoint.task_id] = []
            tasks[checkpoint.task_id].append(checkpoint)

        # Generate report
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_checkpoints": len(all_checkpoints),
            "total_tasks": len(tasks),
            "total_size_mb": sum(cp.file_size_mb for cp in all_checkpoints),
            "tasks": {}
        }

        for task_id, checkpoints in tasks.items():
            analysis = self.analyze_training_session(task_id)
            report["tasks"][task_id] = {
                "analysis": analysis,
                "checkpoints": [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "accuracy": cp.accuracy,
                        "created_at": cp.created_at.isoformat(),
                        "file_size_mb": cp.file_size_mb,
                        "tags": cp.tags or []
                    }
                    for cp in checkpoints
                ]
            }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Recovery report saved: {output_path}")


def main():
    """Main recovery script."""
    parser = argparse.ArgumentParser(description="Checkpoint Recovery Tool for Story 1.5")
    parser.add_argument(
        "--action",
        choices=["list", "validate", "analyze", "recover", "cleanup", "export", "report"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument("--task-id", help="Task ID for task-specific operations")
    parser.add_argument("--checkpoint-id", help="Specific checkpoint ID")
    parser.add_argument("--checkpoint-path", type=Path, help="Custom checkpoint directory")
    parser.add_argument("--details", action="store_true", help="Show detailed information")
    parser.add_argument("--strategy", choices=["best", "latest"], default="best",
                       help="Recovery strategy")
    parser.add_argument("--export-path", type=Path, help="Export destination path")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--max-age-days", type=int, default=7, help="Maximum age for cleanup")
    parser.add_argument("--keep-best", type=int, default=3, help="Number of best checkpoints to keep per task")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Checkpoint Recovery Tool for Story 1.5")
    print("=" * 40)

    try:
        # Initialize recovery tool
        recovery_tool = CheckpointRecoveryTool(args.checkpoint_path)

        if args.action == "list":
            recovery_tool.list_checkpoints(args.task_id, args.details)

        elif args.action == "validate":
            if not args.checkpoint_id:
                print("❌ Checkpoint ID required for validation")
                sys.exit(1)
            recovery_tool.validate_checkpoint_integrity(args.checkpoint_id)

        elif args.action == "analyze":
            if not args.task_id:
                print("❌ Task ID required for analysis")
                sys.exit(1)
            recovery_tool.analyze_training_session(args.task_id)

        elif args.action == "recover":
            if not args.task_id:
                print("❌ Task ID required for recovery")
                sys.exit(1)
            checkpoint_id = recovery_tool.recover_training_session(
                args.task_id, args.strategy, args.dry_run
            )
            if checkpoint_id:
                print(f"\n✅ Recovery checkpoint identified: {checkpoint_id}")

        elif args.action == "cleanup":
            recovery_tool.cleanup_old_checkpoints(
                args.max_age_days, args.keep_best, args.dry_run
            )

        elif args.action == "export":
            if not args.checkpoint_id or not args.export_path:
                print("❌ Checkpoint ID and export path required for export")
                sys.exit(1)
            recovery_tool.export_checkpoint(args.checkpoint_id, args.export_path)

        elif args.action == "report":
            output_path = args.output or Path("logs/checkpoint_recovery_report.json")
            recovery_tool.generate_recovery_report(output_path)

        print("\n✅ Operation completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
