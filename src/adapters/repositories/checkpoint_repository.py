"""
Checkpoint repository for managing TTT model checkpoints.

Handles checkpoint saving, loading, versioning, validation, and storage optimization.
"""
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    task_id: str
    model_name: str
    accuracy: float
    training_time: float
    memory_usage_mb: float
    lora_rank: int
    lora_alpha: int
    created_at: datetime
    file_size_mb: float
    checksum: str
    tags: list[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class CheckpointRepository:
    """Repository for managing model checkpoints."""

    def __init__(self, base_path: Path | None = None):
        """Initialize checkpoint repository."""
        self.base_path = base_path or Path("data/models/checkpoints")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_file = self.base_path / "checkpoint_metadata.json"
        self.metadata_cache: dict[str, CheckpointMetadata] = {}

        # Storage limits
        self.max_checkpoints_per_task = 5
        self.max_total_size_gb = 50

        # Load existing metadata
        self._load_metadata()

        logger.info(f"Checkpoint repository initialized at {self.base_path}")

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    for checkpoint_id, metadata_dict in data.items():
                        self.metadata_cache[checkpoint_id] = CheckpointMetadata.from_dict(metadata_dict)
                logger.info(f"Loaded metadata for {len(self.metadata_cache)} checkpoints")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata_cache = {}

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        try:
            data = {
                checkpoint_id: metadata.to_dict()
                for checkpoint_id, metadata in self.metadata_cache.items()
            }
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint."""
        return self.base_path / f"{checkpoint_id}.pt"

    def save_checkpoint(
        self,
        checkpoint_id: str,
        task_id: str,
        model_state: dict[str, Any],
        training_metrics: dict[str, Any],
        lora_config: dict[str, Any],
        tags: list[str] | None = None,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint with metadata.
        
        Args:
            checkpoint_id: Unique identifier for checkpoint
            task_id: Task this checkpoint was trained on
            model_state: Model state dictionary including LoRA weights
            training_metrics: Training metrics and results
            lora_config: LoRA configuration used
            tags: Optional tags for categorization
            
        Returns:
            CheckpointMetadata object
        """
        logger.info(f"Saving checkpoint {checkpoint_id}")

        # Create checkpoint data
        checkpoint_data = {
            "model_state": model_state,
            "training_metrics": training_metrics,
            "lora_config": lora_config,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
        }

        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        torch.save(checkpoint_data, checkpoint_path)

        # Calculate file size and checksum
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        checksum = self._calculate_checksum(checkpoint_path)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            model_name=training_metrics.get("model_name", "unknown"),
            accuracy=training_metrics.get("final_accuracy", 0.0),
            training_time=training_metrics.get("training_time", 0.0),
            memory_usage_mb=training_metrics.get("final_memory_mb", 0.0),
            lora_rank=lora_config.get("rank", 8),
            lora_alpha=lora_config.get("alpha", 16),
            created_at=datetime.now(),
            file_size_mb=file_size_mb,
            checksum=checksum,
            tags=tags or [],
        )

        # Update metadata cache
        self.metadata_cache[checkpoint_id] = metadata
        self._save_metadata()

        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints(task_id)

        logger.info(f"Saved checkpoint {checkpoint_id} ({file_size_mb:.2f}MB)")

        return metadata

    def load_checkpoint(
        self,
        checkpoint_id: str,
        validate_checksum: bool = True,
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """
        Load a checkpoint with validation.
        
        Args:
            checkpoint_id: Checkpoint to load
            validate_checksum: Whether to validate checksum
            
        Returns:
            Tuple of (checkpoint data, metadata)
        """
        logger.info(f"Loading checkpoint {checkpoint_id}")

        # Check if checkpoint exists
        if checkpoint_id not in self.metadata_cache:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        metadata = self.metadata_cache[checkpoint_id]
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Validate checksum if requested
        if validate_checksum:
            actual_checksum = self._calculate_checksum(checkpoint_path)
            if actual_checksum != metadata.checksum:
                raise ValueError(f"Checksum mismatch for checkpoint {checkpoint_id}")

        # Load checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Loaded checkpoint {checkpoint_id}")
            return checkpoint_data, metadata
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise

    def list_checkpoints(
        self,
        task_id: str | None = None,
        tags: list[str] | None = None,
        min_accuracy: float | None = None,
    ) -> list[CheckpointMetadata]:
        """
        List checkpoints with optional filtering.
        
        Args:
            task_id: Filter by task ID
            tags: Filter by tags (any match)
            min_accuracy: Filter by minimum accuracy
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = list(self.metadata_cache.values())

        # Apply filters
        if task_id:
            checkpoints = [cp for cp in checkpoints if cp.task_id == task_id]

        if tags:
            checkpoints = [
                cp for cp in checkpoints
                if cp.tags and any(tag in cp.tags for tag in tags)
            ]

        if min_accuracy is not None:
            checkpoints = [cp for cp in checkpoints if cp.accuracy >= min_accuracy]

        # Sort by accuracy (descending) and creation time (descending)
        checkpoints.sort(key=lambda x: (x.accuracy, x.created_at), reverse=True)

        return checkpoints

    def get_best_checkpoint(self, task_id: str) -> CheckpointMetadata | None:
        """Get the best checkpoint for a task based on accuracy."""
        checkpoints = self.list_checkpoints(task_id=task_id)
        return checkpoints[0] if checkpoints else None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if deleted, False if not found
        """
        if checkpoint_id not in self.metadata_cache:
            return False

        # Delete file
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Remove from metadata
        del self.metadata_cache[checkpoint_id]
        self._save_metadata()

        logger.info(f"Deleted checkpoint {checkpoint_id}")
        return True

    def _cleanup_old_checkpoints(self, task_id: str) -> None:
        """Clean up old checkpoints for a task, keeping only the best ones."""
        task_checkpoints = self.list_checkpoints(task_id=task_id)

        if len(task_checkpoints) > self.max_checkpoints_per_task:
            # Keep the best checkpoints
            to_delete = task_checkpoints[self.max_checkpoints_per_task:]

            for checkpoint in to_delete:
                logger.info(f"Deleting old checkpoint {checkpoint.checkpoint_id}")
                self.delete_checkpoint(checkpoint.checkpoint_id)

    def cleanup_storage(self, keep_best_per_task: int = 3) -> dict[str, Any]:
        """
        Clean up storage to stay within limits.
        
        Args:
            keep_best_per_task: Number of best checkpoints to keep per task
            
        Returns:
            Cleanup statistics
        """
        initial_count = len(self.metadata_cache)
        initial_size = sum(cp.file_size_mb for cp in self.metadata_cache.values())

        # Group checkpoints by task
        task_checkpoints: dict[str, list[CheckpointMetadata]] = {}
        for checkpoint in self.metadata_cache.values():
            if checkpoint.task_id not in task_checkpoints:
                task_checkpoints[checkpoint.task_id] = []
            task_checkpoints[checkpoint.task_id].append(checkpoint)

        # Keep only best per task
        checkpoints_to_keep = []
        for task_id, checkpoints in task_checkpoints.items():
            # Sort by accuracy
            checkpoints.sort(key=lambda x: x.accuracy, reverse=True)
            checkpoints_to_keep.extend(checkpoints[:keep_best_per_task])

        # Delete others
        checkpoints_to_delete = [
            cp for cp in self.metadata_cache.values()
            if cp not in checkpoints_to_keep
        ]

        for checkpoint in checkpoints_to_delete:
            self.delete_checkpoint(checkpoint.checkpoint_id)

        # Calculate statistics
        final_count = len(self.metadata_cache)
        final_size = sum(cp.file_size_mb for cp in self.metadata_cache.values())

        stats = {
            "initial_count": initial_count,
            "final_count": final_count,
            "deleted_count": initial_count - final_count,
            "initial_size_mb": initial_size,
            "final_size_mb": final_size,
            "freed_size_mb": initial_size - final_size,
        }

        logger.info(f"Storage cleanup: Deleted {stats['deleted_count']} checkpoints, freed {stats['freed_size_mb']:.2f}MB")

        return stats

    def validate_checkpoint_integrity(self, checkpoint_id: str) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_id: Checkpoint to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load checkpoint with checksum validation
            _, metadata = self.load_checkpoint(checkpoint_id, validate_checksum=True)

            # Additional validation could be added here
            # e.g., checking model architecture compatibility

            return True
        except Exception as e:
            logger.error(f"Checkpoint {checkpoint_id} validation failed: {e}")
            return False

    def export_checkpoint(self, checkpoint_id: str, export_path: Path) -> None:
        """Export a checkpoint to a different location."""
        if checkpoint_id not in self.metadata_cache:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        source_path = self._get_checkpoint_path(checkpoint_id)

        # Copy checkpoint file
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, export_path)

        # Export metadata
        metadata_path = export_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata_cache[checkpoint_id].to_dict(), f, indent=2)

        logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
