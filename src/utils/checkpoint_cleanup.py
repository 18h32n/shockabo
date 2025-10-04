"""Checkpoint cleanup strategy for managing 5GB GCS free tier limit."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .checkpoint_manager import CheckpointManager, CheckpointVersion


class CleanupStrategy(Enum):
    """Cleanup strategies for managing storage limits."""
    OLDEST_FIRST = "oldest_first"
    LARGEST_FIRST = "largest_first"
    LOWEST_PERFORMANCE = "lowest_performance"
    EXPERIMENT_BASED = "experiment_based"
    SMART_RETENTION = "smart_retention"


@dataclass
class CleanupPolicy:
    """Cleanup policy configuration."""
    max_storage_gb: float = 4.5  # Leave 0.5GB buffer from 5GB limit
    max_checkpoints_per_experiment: int = 5
    min_retention_days: int = 1
    keep_best_checkpoints: int = 2  # Always keep N best performing checkpoints
    strategy: CleanupStrategy = CleanupStrategy.SMART_RETENTION
    emergency_cleanup_threshold: float = 4.8  # Emergency cleanup at 4.8GB
    performance_metric: str = "loss"  # Metric to use for performance-based cleanup
    performance_ascending: bool = True  # True if lower values are better (e.g., loss)


@dataclass
class CleanupAction:
    """Represents a cleanup action to be taken."""
    checkpoint_name: str
    size_bytes: int
    reason: str
    priority: int  # Lower number = higher priority for deletion
    metadata: dict = None


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    deleted_checkpoints: list[str]
    space_freed_gb: float
    total_space_gb: float
    remaining_checkpoints: int
    errors: list[str]
    actions_taken: list[CleanupAction]


class CheckpointCleanupManager:
    """Manages checkpoint cleanup to stay within storage limits."""

    def __init__(self, checkpoint_manager: CheckpointManager,
                 policy: CleanupPolicy | None = None):
        self.checkpoint_manager = checkpoint_manager
        self.policy = policy or CleanupPolicy()
        self.logger = self._setup_logging()
        self._cleanup_history: list[CleanupResult] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cleanup manager."""
        logger = logging.getLogger('checkpoint_cleanup')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def check_cleanup_needed(self) -> tuple[bool, float, str]:
        """
        Check if cleanup is needed based on current storage usage.

        Returns:
            Tuple of (cleanup_needed, current_usage_gb, reason)
        """
        if not self.checkpoint_manager.gcs_manager:
            return False, 0.0, "No GCS manager available"

        try:
            usage_info = self.checkpoint_manager.gcs_manager.get_bucket_usage()
            current_usage_gb = usage_info.get('total_size_gb', 0.0)

            if current_usage_gb >= self.policy.emergency_cleanup_threshold:
                return True, current_usage_gb, f"Emergency cleanup needed ({current_usage_gb:.2f}GB)"
            elif current_usage_gb >= self.policy.max_storage_gb:
                return True, current_usage_gb, f"Storage limit exceeded ({current_usage_gb:.2f}GB)"
            else:
                return False, current_usage_gb, f"Storage within limits ({current_usage_gb:.2f}GB)"

        except Exception as e:
            self.logger.error(f"Failed to check storage usage: {e}")
            return False, 0.0, f"Error checking usage: {e}"

    async def auto_cleanup(self, target_usage_gb: float | None = None) -> CleanupResult:
        """
        Perform automatic cleanup based on policy.

        Args:
            target_usage_gb: Target usage after cleanup (defaults to policy max)

        Returns:
            CleanupResult with details of cleanup performed
        """
        target_gb = target_usage_gb or self.policy.max_storage_gb

        self.logger.info(f"Starting auto cleanup with target: {target_gb:.2f}GB")

        try:
            # Get current state
            cloud_checkpoints = await self.checkpoint_manager._list_cloud_checkpoints()
            current_usage_gb = sum(cp.size_bytes for cp in cloud_checkpoints) / (1024**3)

            if current_usage_gb <= target_gb:
                self.logger.info(f"No cleanup needed. Current usage: {current_usage_gb:.2f}GB")
                return CleanupResult(
                    deleted_checkpoints=[],
                    space_freed_gb=0.0,
                    total_space_gb=current_usage_gb,
                    remaining_checkpoints=len(cloud_checkpoints),
                    errors=[],
                    actions_taken=[]
                )

            # Calculate how much space we need to free
            space_to_free_gb = current_usage_gb - target_gb
            self.logger.info(f"Need to free {space_to_free_gb:.2f}GB from {current_usage_gb:.2f}GB")

            # Generate cleanup actions based on strategy
            cleanup_actions = await self._generate_cleanup_actions(
                cloud_checkpoints, space_to_free_gb
            )

            # Execute cleanup actions
            result = await self._execute_cleanup_actions(cleanup_actions, cloud_checkpoints)

            # Log results
            self.logger.info(
                f"Cleanup completed: freed {result.space_freed_gb:.2f}GB, "
                f"deleted {len(result.deleted_checkpoints)} checkpoints"
            )

            # Store in history
            self._cleanup_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Auto cleanup failed: {e}")
            return CleanupResult(
                deleted_checkpoints=[],
                space_freed_gb=0.0,
                total_space_gb=0.0,
                remaining_checkpoints=0,
                errors=[str(e)],
                actions_taken=[]
            )

    async def _generate_cleanup_actions(self, checkpoints: list[CheckpointVersion],
                                       space_to_free_gb: float) -> list[CleanupAction]:
        """Generate cleanup actions based on strategy."""
        actions = []

        if self.policy.strategy == CleanupStrategy.OLDEST_FIRST:
            actions = self._strategy_oldest_first(checkpoints)
        elif self.policy.strategy == CleanupStrategy.LARGEST_FIRST:
            actions = self._strategy_largest_first(checkpoints)
        elif self.policy.strategy == CleanupStrategy.LOWEST_PERFORMANCE:
            actions = self._strategy_lowest_performance(checkpoints)
        elif self.policy.strategy == CleanupStrategy.EXPERIMENT_BASED:
            actions = self._strategy_experiment_based(checkpoints)
        elif self.policy.strategy == CleanupStrategy.SMART_RETENTION:
            actions = self._strategy_smart_retention(checkpoints)
        else:
            # Default to oldest first
            actions = self._strategy_oldest_first(checkpoints)

        # Filter actions to meet space requirements
        space_to_free_bytes = space_to_free_gb * (1024**3)
        selected_actions = []
        total_freed = 0

        for action in actions:
            if total_freed >= space_to_free_bytes:
                break
            selected_actions.append(action)
            total_freed += action.size_bytes

        self.logger.info(f"Generated {len(selected_actions)} cleanup actions")
        return selected_actions

    def _strategy_oldest_first(self, checkpoints: list[CheckpointVersion]) -> list[CleanupAction]:
        """Strategy: Delete oldest checkpoints first."""
        # Sort by timestamp (oldest first)
        sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp.timestamp)

        actions = []
        for i, cp in enumerate(sorted_checkpoints):
            # Skip if within minimum retention period
            age_days = (datetime.now() - cp.timestamp).days
            if age_days < self.policy.min_retention_days:
                continue

            actions.append(CleanupAction(
                checkpoint_name=cp.name if hasattr(cp, 'name') else cp.local_path.split('/')[-1],
                size_bytes=cp.size_bytes,
                reason=f"Oldest checkpoint (age: {age_days} days)",
                priority=i,
                metadata={'age_days': age_days, 'timestamp': cp.timestamp}
            ))

        return actions

    def _strategy_largest_first(self, checkpoints: list[CheckpointVersion]) -> list[CleanupAction]:
        """Strategy: Delete largest checkpoints first."""
        # Sort by size (largest first)
        sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp.size_bytes, reverse=True)

        actions = []
        for i, cp in enumerate(sorted_checkpoints):
            # Skip if within minimum retention period
            age_days = (datetime.now() - cp.timestamp).days
            if age_days < self.policy.min_retention_days:
                continue

            size_mb = cp.size_bytes / (1024**2)
            actions.append(CleanupAction(
                checkpoint_name=cp.name if hasattr(cp, 'name') else cp.local_path.split('/')[-1],
                size_bytes=cp.size_bytes,
                reason=f"Largest checkpoint ({size_mb:.1f}MB)",
                priority=i,
                metadata={'size_mb': size_mb}
            ))

        return actions

    def _strategy_lowest_performance(self, checkpoints: list[CheckpointVersion]) -> list[CleanupAction]:
        """Strategy: Delete worst performing checkpoints first."""
        # Get performance metric for each checkpoint
        checkpoint_performance = []
        for cp in checkpoints:
            if not cp.metadata or self.policy.performance_metric not in cp.metadata:
                # No performance data, treat as worst
                performance = float('inf') if self.policy.performance_ascending else float('-inf')
            else:
                performance = float(cp.metadata[self.policy.performance_metric])

            checkpoint_performance.append((cp, performance))

        # Sort by performance (worst first)
        checkpoint_performance.sort(
            key=lambda x: x[1],
            reverse=not self.policy.performance_ascending
        )

        actions = []
        for i, (cp, performance) in enumerate(checkpoint_performance):
            # Skip if within minimum retention period
            age_days = (datetime.now() - cp.timestamp).days
            if age_days < self.policy.min_retention_days:
                continue

            # Skip best performing checkpoints
            if i < self.policy.keep_best_checkpoints:
                continue

            actions.append(CleanupAction(
                checkpoint_name=cp.name if hasattr(cp, 'name') else cp.local_path.split('/')[-1],
                size_bytes=cp.size_bytes,
                reason=f"Low performance ({self.policy.performance_metric}: {performance})",
                priority=i,
                metadata={'performance': performance, 'metric': self.policy.performance_metric}
            ))

        return actions

    def _strategy_experiment_based(self, checkpoints: list[CheckpointVersion]) -> list[CleanupAction]:
        """Strategy: Keep limited checkpoints per experiment."""
        # Group by experiment
        by_experiment = {}
        for cp in checkpoints:
            exp_id = cp.experiment_id
            if exp_id not in by_experiment:
                by_experiment[exp_id] = []
            by_experiment[exp_id].append(cp)

        actions = []
        priority = 0

        for exp_id, exp_checkpoints in by_experiment.items():
            # Sort by timestamp (newest first) for this experiment
            exp_checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)

            # Keep only the newest N checkpoints per experiment
            for cp in exp_checkpoints[self.policy.max_checkpoints_per_experiment:]:
                age_days = (datetime.now() - cp.timestamp).days
                if age_days < self.policy.min_retention_days:
                    continue

                actions.append(CleanupAction(
                    checkpoint_name=cp.name if hasattr(cp, 'name') else cp.local_path.split('/')[-1],
                    size_bytes=cp.size_bytes,
                    reason=f"Excess checkpoint for experiment {exp_id}",
                    priority=priority,
                    metadata={'experiment_id': exp_id, 'age_days': age_days}
                ))
                priority += 1

        # Sort actions by age (oldest first)
        actions.sort(key=lambda a: a.metadata.get('age_days', 0), reverse=True)

        return actions

    def _strategy_smart_retention(self, checkpoints: list[CheckpointVersion]) -> list[CleanupAction]:
        """Strategy: Smart retention combining multiple factors."""
        actions = []

        # Group by experiment
        by_experiment = {}
        for cp in checkpoints:
            exp_id = cp.experiment_id
            if exp_id not in by_experiment:
                by_experiment[exp_id] = []
            by_experiment[exp_id].append(cp)

        for exp_id, exp_checkpoints in by_experiment.items():
            # Sort by timestamp (newest first)
            exp_checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)

            # Keep the newest checkpoint always
            keep_indices = {0} if exp_checkpoints else set()

            # Keep best performing checkpoints
            if len(exp_checkpoints) > 1:
                # Sort by performance
                with_performance = []
                for i, cp in enumerate(exp_checkpoints):
                    if cp.metadata and self.policy.performance_metric in cp.metadata:
                        perf = float(cp.metadata[self.policy.performance_metric])
                        with_performance.append((i, perf))

                if with_performance:
                    with_performance.sort(
                        key=lambda x: x[1],
                        reverse=not self.policy.performance_ascending
                    )
                    # Keep best N performing
                    for i, _ in with_performance[:self.policy.keep_best_checkpoints]:
                        keep_indices.add(i)

            # Keep milestone checkpoints (every 10th epoch)
            for i, cp in enumerate(exp_checkpoints):
                if cp.metadata and cp.metadata.get('epoch', 0) % 10 == 0:
                    keep_indices.add(i)

            # Mark others for deletion (with age consideration)
            for i, cp in enumerate(exp_checkpoints):
                if i in keep_indices:
                    continue

                age_days = (datetime.now() - cp.timestamp).days
                if age_days < self.policy.min_retention_days:
                    continue

                # Calculate priority based on multiple factors
                priority = self._calculate_smart_priority(cp, i, len(exp_checkpoints))

                actions.append(CleanupAction(
                    checkpoint_name=cp.name if hasattr(cp, 'name') else cp.local_path.split('/')[-1],
                    size_bytes=cp.size_bytes,
                    reason=f"Smart retention for {exp_id}",
                    priority=priority,
                    metadata={
                        'experiment_id': exp_id,
                        'age_days': age_days,
                        'experiment_position': i,
                        'experiment_total': len(exp_checkpoints)
                    }
                ))

        # Sort by priority (higher priority = delete first)
        actions.sort(key=lambda a: a.priority, reverse=True)

        return actions

    def _calculate_smart_priority(self, checkpoint: CheckpointVersion,
                                position: int, total_count: int) -> int:
        """Calculate smart priority for deletion (higher = delete first)."""
        priority = 0

        # Age factor (older = higher priority for deletion)
        age_days = (datetime.now() - checkpoint.timestamp).days
        priority += age_days * 10

        # Position in experiment (middle checkpoints less important)
        position_factor = min(position, total_count - position)
        priority += (total_count - position_factor) * 5

        # Size factor (larger = slightly higher priority)
        size_gb = checkpoint.size_bytes / (1024**3)
        priority += int(size_gb * 2)

        # Performance factor (if available)
        if (checkpoint.metadata and
            self.policy.performance_metric in checkpoint.metadata):
            try:
                performance = float(checkpoint.metadata[self.policy.performance_metric])
                # Normalize performance to 0-100 range for priority calculation
                # This is a simplified approach - in practice, you'd want to normalize
                # based on the distribution of values across all checkpoints
                if self.policy.performance_ascending:
                    # Higher loss = higher priority for deletion
                    priority += int(min(performance * 10, 100))
                else:
                    # Lower accuracy = higher priority for deletion
                    priority += int(max(100 - performance * 10, 0))
            except (ValueError, TypeError):
                pass

        return priority

    async def _execute_cleanup_actions(self, actions: list[CleanupAction],
                                     all_checkpoints: list[CheckpointVersion]) -> CleanupResult:
        """Execute the cleanup actions."""
        deleted_checkpoints = []
        space_freed_bytes = 0
        errors = []

        for action in actions:
            try:
                success = await self.checkpoint_manager.delete_cloud_checkpoint(
                    action.checkpoint_name
                )

                if success:
                    deleted_checkpoints.append(action.checkpoint_name)
                    space_freed_bytes += action.size_bytes
                    self.logger.info(
                        f"Deleted checkpoint: {action.checkpoint_name} "
                        f"({action.size_bytes / (1024**2):.1f}MB) - {action.reason}"
                    )
                else:
                    error_msg = f"Failed to delete {action.checkpoint_name}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                error_msg = f"Error deleting {action.checkpoint_name}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        # Calculate final state
        remaining_checkpoints = len(all_checkpoints) - len(deleted_checkpoints)
        total_remaining_bytes = sum(
            cp.size_bytes for cp in all_checkpoints
            if (hasattr(cp, 'name') and cp.name not in deleted_checkpoints) or
               (hasattr(cp, 'local_path') and cp.local_path.split('/')[-1] not in deleted_checkpoints)
        )

        return CleanupResult(
            deleted_checkpoints=deleted_checkpoints,
            space_freed_gb=space_freed_bytes / (1024**3),
            total_space_gb=total_remaining_bytes / (1024**3),
            remaining_checkpoints=remaining_checkpoints,
            errors=errors,
            actions_taken=actions[:len(deleted_checkpoints)]
        )

    async def emergency_cleanup(self, target_percentage: float = 0.8) -> CleanupResult:
        """
        Perform emergency cleanup to reduce storage to target percentage of limit.

        Args:
            target_percentage: Target storage as percentage of limit (0.8 = 80% of 5GB)

        Returns:
            CleanupResult with cleanup details
        """
        target_gb = self.policy.max_storage_gb * target_percentage
        self.logger.warning(f"Performing emergency cleanup to {target_gb:.2f}GB")

        # Use aggressive cleanup strategy
        original_strategy = self.policy.strategy
        original_retention = self.policy.min_retention_days

        # Temporarily use more aggressive settings
        self.policy.strategy = CleanupStrategy.OLDEST_FIRST
        self.policy.min_retention_days = 0  # Remove retention requirement for emergency

        try:
            result = await self.auto_cleanup(target_gb)
            self.logger.warning(
                f"Emergency cleanup completed: freed {result.space_freed_gb:.2f}GB"
            )
            return result
        finally:
            # Restore original settings
            self.policy.strategy = original_strategy
            self.policy.min_retention_days = original_retention

    def get_cleanup_history(self) -> list[CleanupResult]:
        """Get history of cleanup operations."""
        return self._cleanup_history.copy()

    def update_policy(self, **kwargs):
        """Update cleanup policy parameters."""
        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                self.logger.info(f"Updated cleanup policy: {key} = {value}")
            else:
                self.logger.warning(f"Unknown policy parameter: {key}")


# Convenience function for automated cleanup
async def run_automated_cleanup(checkpoint_manager: CheckpointManager,
                              policy: CleanupPolicy | None = None) -> CleanupResult:
    """Run automated cleanup with default or provided policy."""
    cleanup_manager = CheckpointCleanupManager(checkpoint_manager, policy)

    # Check if cleanup is needed
    cleanup_needed, current_usage, reason = await cleanup_manager.check_cleanup_needed()

    if cleanup_needed:
        if current_usage >= cleanup_manager.policy.emergency_cleanup_threshold:
            return await cleanup_manager.emergency_cleanup()
        else:
            return await cleanup_manager.auto_cleanup()
    else:
        logging.getLogger('checkpoint_cleanup').info(f"No cleanup needed: {reason}")
        return CleanupResult(
            deleted_checkpoints=[],
            space_freed_gb=0.0,
            total_space_gb=current_usage,
            remaining_checkpoints=0,
            errors=[],
            actions_taken=[]
        )
