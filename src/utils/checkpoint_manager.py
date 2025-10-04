"""Checkpoint management utilities with versioning and cloud storage integration."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .gcs_integration import (
    CheckpointMetadata,
    GCSManager,
    calculate_file_hash,
    create_gcs_manager_from_env,
    generate_checkpoint_name,
)


@dataclass
class CheckpointVersion:
    """Checkpoint version information."""
    version: str
    timestamp: datetime
    size_bytes: int
    hash: str
    platform: str
    experiment_id: str
    local_path: str | None = None
    cloud_path: str | None = None
    metadata: dict[str, Any] = None


@dataclass
class CheckpointSyncStatus:
    """Status of checkpoint synchronization."""
    local_checkpoints: list[CheckpointVersion]
    cloud_checkpoints: list[CheckpointVersion]
    sync_needed: list[str]  # Checkpoint names that need sync
    conflicts: list[str]    # Checkpoints with conflicts
    total_local_size: int
    total_cloud_size: int


class CheckpointManager:
    """Manages checkpoint versioning, upload, download, and synchronization."""

    def __init__(self, local_checkpoint_dir: str,
                 gcs_manager: GCSManager | None = None):
        self.local_dir = Path(local_checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.gcs_manager = gcs_manager or create_gcs_manager_from_env()
        self.logger = self._setup_logging()

        # Version tracking
        self.version_file = self.local_dir / "checkpoint_versions.json"
        self._local_versions: dict[str, CheckpointVersion] = {}
        self._load_local_versions()

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=3)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for checkpoint manager."""
        logger = logging.getLogger('checkpoint_manager')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_local_versions(self):
        """Load local checkpoint version information."""
        if self.version_file.exists():
            try:
                with open(self.version_file) as f:
                    versions_data = json.load(f)

                for name, data in versions_data.items():
                    # Convert timestamp string back to datetime
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    self._local_versions[name] = CheckpointVersion(**data)

            except Exception as e:
                self.logger.error(f"Failed to load version file: {e}")
                self._local_versions = {}

    def _save_local_versions(self):
        """Save local checkpoint version information."""
        try:
            versions_data = {}
            for name, version in self._local_versions.items():
                data = asdict(version)
                data['timestamp'] = version.timestamp.isoformat()
                versions_data[name] = data

            with open(self.version_file, 'w') as f:
                json.dump(versions_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save version file: {e}")

    def save_checkpoint(self, checkpoint_data: Any, experiment_id: str,
                       epoch: int, loss: float, metrics: dict[str, float] | None = None,
                       model_type: str = "pytorch", platform: str = "unknown",
                       tags: list[str] | None = None) -> str:
        """
        Save checkpoint locally with versioning.

        Args:
            checkpoint_data: Checkpoint data (model state, etc.)
            experiment_id: Unique experiment identifier
            epoch: Training epoch
            loss: Current loss value
            metrics: Additional metrics
            model_type: Type of model (pytorch, tensorflow, etc.)
            platform: Platform where checkpoint was created
            tags: Optional tags for the checkpoint

        Returns:
            Checkpoint name/identifier
        """
        timestamp = datetime.now()
        checkpoint_name = generate_checkpoint_name(experiment_id, epoch, platform, timestamp)
        local_path = self.local_dir / checkpoint_name

        try:
            # Save checkpoint data based on type
            if model_type.lower() == "pytorch":
                import torch
                torch.save(checkpoint_data, local_path)
            elif model_type.lower() == "tensorflow":
                # Handle TensorFlow checkpoints
                checkpoint_data.save(str(local_path))
            else:
                # Generic save (assumes checkpoint_data has save method or is serializable)
                if hasattr(checkpoint_data, 'save'):
                    checkpoint_data.save(str(local_path))
                else:
                    # Try pickle
                    import pickle
                    with open(local_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)

            # Calculate file hash for integrity
            file_hash = calculate_file_hash(local_path)
            size_bytes = local_path.stat().st_size

            # Create version info
            version = CheckpointVersion(
                version="1.0",
                timestamp=timestamp,
                size_bytes=size_bytes,
                hash=file_hash,
                platform=platform,
                experiment_id=experiment_id,
                local_path=str(local_path),
                metadata={
                    'epoch': epoch,
                    'loss': loss,
                    'metrics': metrics or {},
                    'model_type': model_type,
                    'tags': tags or []
                }
            )

            # Store version info
            self._local_versions[checkpoint_name] = version
            self._save_local_versions()

            self.logger.info(f"Saved checkpoint: {checkpoint_name} ({size_bytes} bytes)")
            return checkpoint_name

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            # Clean up partial file
            if local_path.exists():
                local_path.unlink()
            raise

    def load_checkpoint(self, checkpoint_name: str,
                       model_type: str = "pytorch") -> Any:
        """
        Load checkpoint from local storage.

        Args:
            checkpoint_name: Name of checkpoint to load
            model_type: Type of model to load

        Returns:
            Loaded checkpoint data
        """
        if checkpoint_name not in self._local_versions:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")

        version = self._local_versions[checkpoint_name]
        local_path = Path(version.local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {local_path}")

        try:
            # Verify integrity
            current_hash = calculate_file_hash(local_path)
            if current_hash != version.hash:
                self.logger.warning(f"Hash mismatch for {checkpoint_name}, file may be corrupted")

            # Load checkpoint based on type
            if model_type.lower() == "pytorch":
                import torch
                return torch.load(local_path)
            elif model_type.lower() == "tensorflow":
                # Handle TensorFlow checkpoints
                import tensorflow as tf
                return tf.train.load_checkpoint(str(local_path))
            else:
                # Generic load
                import pickle
                with open(local_path, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    async def upload_checkpoint(self, checkpoint_name: str,
                              force: bool = False) -> bool:
        """
        Upload checkpoint to cloud storage.

        Args:
            checkpoint_name: Name of checkpoint to upload
            force: Force upload even if already exists in cloud

        Returns:
            True if upload successful, False otherwise
        """
        if not self.gcs_manager:
            self.logger.error("GCS manager not available for upload")
            return False

        if checkpoint_name not in self._local_versions:
            self.logger.error(f"Checkpoint not found locally: {checkpoint_name}")
            return False

        version = self._local_versions[checkpoint_name]
        local_path = Path(version.local_path)

        if not local_path.exists():
            self.logger.error(f"Checkpoint file not found: {local_path}")
            return False

        try:
            # Check if already uploaded
            if not force:
                cloud_checkpoints = await self._list_cloud_checkpoints()
                if any(cp.hash == version.hash for cp in cloud_checkpoints
                      if cp.experiment_id == version.experiment_id):
                    self.logger.info(f"Checkpoint {checkpoint_name} already exists in cloud")
                    return True

            # Create metadata for upload
            metadata = CheckpointMetadata(
                name=checkpoint_name,
                version=version.version,
                created_at=version.timestamp,
                size_bytes=version.size_bytes,
                platform=version.platform,
                experiment_id=version.experiment_id,
                model_type=version.metadata.get('model_type', 'unknown'),
                epoch=version.metadata.get('epoch', 0),
                loss=version.metadata.get('loss', 0.0),
                metrics=version.metadata.get('metrics'),
                tags=version.metadata.get('tags')
            )

            # Upload using thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self.gcs_manager.upload_checkpoint,
                local_path, checkpoint_name, metadata
            )

            if success:
                # Update version with cloud path
                version.cloud_path = f"gs://{self.gcs_manager.config.bucket_name}/checkpoints/{checkpoint_name}"
                self._local_versions[checkpoint_name] = version
                self._save_local_versions()

                self.logger.info(f"Successfully uploaded checkpoint: {checkpoint_name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint: {e}")
            return False

    async def download_checkpoint(self, checkpoint_name: str,
                                experiment_id: str | None = None) -> bool:
        """
        Download checkpoint from cloud storage.

        Args:
            checkpoint_name: Name of checkpoint to download
            experiment_id: Filter by experiment ID

        Returns:
            True if download successful, False otherwise
        """
        if not self.gcs_manager:
            self.logger.error("GCS manager not available for download")
            return False

        try:
            # Check if already exists locally
            if checkpoint_name in self._local_versions:
                local_path = Path(self._local_versions[checkpoint_name].local_path)
                if local_path.exists():
                    self.logger.info(f"Checkpoint {checkpoint_name} already exists locally")
                    return True

            # Download to local directory
            local_path = self.local_dir / checkpoint_name

            # Download using thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self.gcs_manager.download_checkpoint,
                checkpoint_name, local_path
            )

            if success:
                # Get metadata from cloud
                cloud_checkpoints = await self._list_cloud_checkpoints()
                cloud_checkpoint = next(
                    (cp for cp in cloud_checkpoints if cp.name == checkpoint_name),
                    None
                )

                if cloud_checkpoint:
                    # Create local version entry
                    file_hash = calculate_file_hash(local_path)

                    version = CheckpointVersion(
                        version=cloud_checkpoint.version,
                        timestamp=cloud_checkpoint.created_at,
                        size_bytes=cloud_checkpoint.size_bytes,
                        hash=file_hash,
                        platform=cloud_checkpoint.platform,
                        experiment_id=cloud_checkpoint.experiment_id,
                        local_path=str(local_path),
                        cloud_path=f"gs://{self.gcs_manager.config.bucket_name}/checkpoints/{checkpoint_name}",
                        metadata={
                            'epoch': cloud_checkpoint.epoch,
                            'loss': cloud_checkpoint.loss,
                            'metrics': cloud_checkpoint.metrics or {},
                            'model_type': cloud_checkpoint.model_type,
                            'tags': cloud_checkpoint.tags or []
                        }
                    )

                    self._local_versions[checkpoint_name] = version
                    self._save_local_versions()

                self.logger.info(f"Successfully downloaded checkpoint: {checkpoint_name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to download checkpoint: {e}")
            return False

    async def sync_checkpoints(self, experiment_id: str | None = None,
                             upload: bool = True, download: bool = True) -> CheckpointSyncStatus:
        """
        Synchronize local and cloud checkpoints.

        Args:
            experiment_id: Sync only checkpoints for specific experiment
            upload: Upload local checkpoints not in cloud
            download: Download cloud checkpoints not local

        Returns:
            Synchronization status
        """
        try:
            # Get local and cloud checkpoint lists
            local_checkpoints = self._get_local_checkpoint_versions(experiment_id)
            cloud_checkpoints = await self._list_cloud_checkpoints(experiment_id)

            # Find checkpoints that need syncing
            local_names = {cp.experiment_id + "_" + cp.hash for cp in local_checkpoints}
            cloud_names = {cp.experiment_id + "_" + cp.hash for cp in cloud_checkpoints}

            upload_needed = []
            download_needed = []
            conflicts = []

            # Find checkpoints to upload
            if upload:
                for local_cp in local_checkpoints:
                    local_key = local_cp.experiment_id + "_" + local_cp.hash
                    if local_key not in cloud_names:
                        upload_needed.append(local_cp.local_path.split('/')[-1])

            # Find checkpoints to download
            if download:
                for cloud_cp in cloud_checkpoints:
                    cloud_key = cloud_cp.experiment_id + "_" + cloud_cp.hash
                    if cloud_key not in local_names:
                        download_needed.append(cloud_cp.name)

            # Perform uploads
            upload_tasks = []
            for checkpoint_name in upload_needed:
                task = self.upload_checkpoint(checkpoint_name)
                upload_tasks.append(task)

            if upload_tasks:
                upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
                failed_uploads = [name for name, result in zip(upload_needed, upload_results, strict=False)
                                if isinstance(result, Exception) or not result]
                if failed_uploads:
                    self.logger.warning(f"Failed to upload: {failed_uploads}")

            # Perform downloads
            download_tasks = []
            for checkpoint_name in download_needed:
                task = self.download_checkpoint(checkpoint_name, experiment_id)
                download_tasks.append(task)

            if download_tasks:
                download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
                failed_downloads = [name for name, result in zip(download_needed, download_results, strict=False)
                                  if isinstance(result, Exception) or not result]
                if failed_downloads:
                    self.logger.warning(f"Failed to download: {failed_downloads}")

            # Calculate sizes
            total_local_size = sum(cp.size_bytes for cp in local_checkpoints)
            total_cloud_size = sum(cp.size_bytes for cp in cloud_checkpoints)

            sync_status = CheckpointSyncStatus(
                local_checkpoints=local_checkpoints,
                cloud_checkpoints=cloud_checkpoints,
                sync_needed=upload_needed + download_needed,
                conflicts=conflicts,
                total_local_size=total_local_size,
                total_cloud_size=total_cloud_size
            )

            self.logger.info(f"Sync completed: {len(upload_needed)} uploaded, "
                           f"{len(download_needed)} downloaded")

            return sync_status

        except Exception as e:
            self.logger.error(f"Failed to sync checkpoints: {e}")
            raise

    def _get_local_checkpoint_versions(self, experiment_id: str | None = None) -> list[CheckpointVersion]:
        """Get list of local checkpoint versions."""
        checkpoints = []
        for _name, version in self._local_versions.items():
            if experiment_id is None or version.experiment_id == experiment_id:
                # Verify file still exists
                if version.local_path and Path(version.local_path).exists():
                    checkpoints.append(version)
        return checkpoints

    async def _list_cloud_checkpoints(self, experiment_id: str | None = None) -> list[CheckpointVersion]:
        """Get list of cloud checkpoint versions."""
        if not self.gcs_manager:
            return []

        try:
            loop = asyncio.get_event_loop()
            cloud_metadata_list = await loop.run_in_executor(
                self.executor,
                self.gcs_manager.list_checkpoints,
                "", experiment_id
            )

            checkpoints = []
            for metadata in cloud_metadata_list:
                version = CheckpointVersion(
                    version=metadata.version,
                    timestamp=metadata.created_at,
                    size_bytes=metadata.size_bytes,
                    hash="",  # Hash not stored in cloud metadata
                    platform=metadata.platform,
                    experiment_id=metadata.experiment_id,
                    cloud_path=f"gs://{self.gcs_manager.config.bucket_name}/checkpoints/{metadata.name}",
                    metadata={
                        'epoch': metadata.epoch,
                        'loss': metadata.loss,
                        'metrics': metadata.metrics or {},
                        'model_type': metadata.model_type,
                        'tags': metadata.tags or []
                    }
                )
                checkpoints.append(version)

            return checkpoints

        except Exception as e:
            self.logger.error(f"Failed to list cloud checkpoints: {e}")
            return []

    def list_local_checkpoints(self, experiment_id: str | None = None) -> list[CheckpointVersion]:
        """List local checkpoints."""
        return self._get_local_checkpoint_versions(experiment_id)

    def delete_local_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete local checkpoint."""
        if checkpoint_name not in self._local_versions:
            return False

        try:
            version = self._local_versions[checkpoint_name]
            local_path = Path(version.local_path)

            if local_path.exists():
                local_path.unlink()

            del self._local_versions[checkpoint_name]
            self._save_local_versions()

            self.logger.info(f"Deleted local checkpoint: {checkpoint_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete local checkpoint: {e}")
            return False

    async def delete_cloud_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete cloud checkpoint."""
        if not self.gcs_manager:
            return False

        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self.gcs_manager.delete_checkpoint,
                checkpoint_name
            )

            # Update local version info if exists
            if checkpoint_name in self._local_versions:
                version = self._local_versions[checkpoint_name]
                version.cloud_path = None
                self._save_local_versions()

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete cloud checkpoint: {e}")
            return False

    def get_checkpoint_info(self, checkpoint_name: str) -> CheckpointVersion | None:
        """Get information about a specific checkpoint."""
        return self._local_versions.get(checkpoint_name)

    def get_storage_usage(self) -> dict[str, Any]:
        """Get storage usage information."""
        local_size = sum(
            Path(version.local_path).stat().st_size
            for version in self._local_versions.values()
            if version.local_path and Path(version.local_path).exists()
        )

        usage_info = {
            'local_storage': {
                'checkpoint_count': len(self._local_versions),
                'total_size_bytes': local_size,
                'total_size_gb': local_size / (1024**3),
                'directory': str(self.local_dir)
            }
        }

        if self.gcs_manager:
            try:
                cloud_usage = self.gcs_manager.get_bucket_usage()
                usage_info['cloud_storage'] = cloud_usage
            except Exception as e:
                self.logger.error(f"Failed to get cloud usage: {e}")

        return usage_info

    def cleanup_old_checkpoints(self, keep_count: int = 10,
                              experiment_id: str | None = None) -> list[str]:
        """
        Clean up old local checkpoints, keeping only the most recent ones.

        Args:
            keep_count: Number of checkpoints to keep per experiment
            experiment_id: Clean only checkpoints for specific experiment

        Returns:
            List of deleted checkpoint names
        """
        deleted = []

        # Group checkpoints by experiment
        by_experiment = {}
        for name, version in self._local_versions.items():
            exp_id = version.experiment_id
            if experiment_id is None or exp_id == experiment_id:
                if exp_id not in by_experiment:
                    by_experiment[exp_id] = []
                by_experiment[exp_id].append((name, version))

        # Clean up each experiment's checkpoints
        for _exp_id, checkpoints in by_experiment.items():
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)

            # Delete old checkpoints
            for name, _version in checkpoints[keep_count:]:
                if self.delete_local_checkpoint(name):
                    deleted.append(name)

        if deleted:
            self.logger.info(f"Cleaned up {len(deleted)} old checkpoints")

        return deleted
