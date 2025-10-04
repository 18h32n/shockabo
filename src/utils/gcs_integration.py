"""Google Cloud Storage integration for checkpoint management."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage
    from google.cloud.exceptions import Forbidden, NotFound
    GCS_AVAILABLE = True
except ImportError:
    storage = None
    NotFound = Exception
    Forbidden = Exception
    DefaultCredentialsError = Exception
    GCS_AVAILABLE = False


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    name: str
    version: str
    created_at: datetime
    size_bytes: int
    platform: str
    experiment_id: str
    model_type: str = "unknown"
    epoch: int = 0
    loss: float = 0.0
    metrics: dict[str, float] = None
    tags: list[str] = None


@dataclass
class GCSConfig:
    """Google Cloud Storage configuration."""
    project_id: str
    bucket_name: str
    credentials_path: str | None = None
    credentials_json: str | None = None
    region: str = "us-central1"
    storage_class: str = "STANDARD"


class GCSManager:
    """Manages Google Cloud Storage operations for checkpoint management."""

    def __init__(self, config: GCSConfig):
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage package is required for GCS integration")

        self.config = config
        self.logger = self._setup_logging()
        self._client = None
        self._bucket = None
        self._authenticated = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for GCS manager."""
        logger = logging.getLogger('gcs_manager')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def authenticate(self) -> bool:
        """
        Authenticate with Google Cloud Storage.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Try different authentication methods
            if self.config.credentials_json:
                # Use JSON credentials from environment or config
                credentials_info = json.loads(self.config.credentials_json)
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                self._client = storage.Client(
                    project=self.config.project_id,
                    credentials=credentials
                )
            elif self.config.credentials_path and os.path.exists(self.config.credentials_path):
                # Use credentials file
                self._client = storage.Client.from_service_account_json(
                    self.config.credentials_path,
                    project=self.config.project_id
                )
            else:
                # Use default credentials (e.g., from GOOGLE_APPLICATION_CREDENTIALS)
                self._client = storage.Client(project=self.config.project_id)

            # Test authentication by listing buckets
            list(self._client.list_buckets(max_results=1))
            self._authenticated = True
            self.logger.info("Successfully authenticated with Google Cloud Storage")
            return True

        except DefaultCredentialsError as e:
            self.logger.error(f"Authentication failed - no valid credentials: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def create_bucket_if_not_exists(self) -> bool:
        """
        Create bucket if it doesn't exist.

        Returns:
            True if bucket exists or was created, False otherwise
        """
        if not self._authenticated:
            if not self.authenticate():
                return False

        try:
            # Check if bucket exists
            self._bucket = self._client.bucket(self.config.bucket_name)
            self._bucket.reload()  # Will raise NotFound if bucket doesn't exist
            self.logger.info(f"Bucket {self.config.bucket_name} already exists")
            return True

        except NotFound:
            # Bucket doesn't exist, create it
            try:
                self._bucket = self._client.create_bucket(
                    self.config.bucket_name,
                    location=self.config.region
                )
                self.logger.info(f"Created bucket {self.config.bucket_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to create bucket: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error checking bucket: {e}")
            return False

    def get_bucket(self):
        """Get the configured bucket."""
        if not self._bucket:
            if not self.create_bucket_if_not_exists():
                return None
        return self._bucket

    def upload_checkpoint(self, local_path: str | Path,
                         checkpoint_name: str,
                         metadata: CheckpointMetadata) -> bool:
        """
        Upload checkpoint file to GCS.

        Args:
            local_path: Local path to checkpoint file
            checkpoint_name: Name for checkpoint in GCS
            metadata: Checkpoint metadata

        Returns:
            True if upload successful, False otherwise
        """
        bucket = self.get_bucket()
        if not bucket:
            return False

        local_path = Path(local_path)
        if not local_path.exists():
            self.logger.error(f"Local checkpoint file not found: {local_path}")
            return False

        try:
            # Create blob with versioned name
            blob_name = f"checkpoints/{checkpoint_name}"
            blob = bucket.blob(blob_name)

            # Set metadata
            blob.metadata = {
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat(),
                'platform': metadata.platform,
                'experiment_id': metadata.experiment_id,
                'model_type': metadata.model_type,
                'epoch': str(metadata.epoch),
                'loss': str(metadata.loss),
                'size_bytes': str(metadata.size_bytes)
            }

            if metadata.metrics:
                blob.metadata.update({
                    f'metric_{k}': str(v) for k, v in metadata.metrics.items()
                })

            if metadata.tags:
                blob.metadata['tags'] = ','.join(metadata.tags)

            # Upload file
            self.logger.info(f"Uploading checkpoint {checkpoint_name} ({local_path.stat().st_size} bytes)")
            blob.upload_from_filename(str(local_path))

            self.logger.info(f"Successfully uploaded checkpoint: {blob_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint: {e}")
            return False

    def download_checkpoint(self, checkpoint_name: str,
                          local_path: str | Path) -> bool:
        """
        Download checkpoint file from GCS.

        Args:
            checkpoint_name: Name of checkpoint in GCS
            local_path: Local path to save downloaded file

        Returns:
            True if download successful, False otherwise
        """
        bucket = self.get_bucket()
        if not bucket:
            return False

        try:
            blob_name = f"checkpoints/{checkpoint_name}"
            blob = bucket.blob(blob_name)

            # Check if blob exists
            if not blob.exists():
                self.logger.error(f"Checkpoint not found in GCS: {blob_name}")
                return False

            # Create local directory if needed
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.logger.info(f"Downloading checkpoint {checkpoint_name} to {local_path}")
            blob.download_to_filename(str(local_path))

            self.logger.info(f"Successfully downloaded checkpoint: {local_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download checkpoint: {e}")
            return False

    def list_checkpoints(self, prefix: str = "",
                        experiment_id: str | None = None) -> list[CheckpointMetadata]:
        """
        List available checkpoints.

        Args:
            prefix: Filter checkpoints by name prefix
            experiment_id: Filter by experiment ID

        Returns:
            List of checkpoint metadata
        """
        bucket = self.get_bucket()
        if not bucket:
            return []

        try:
            checkpoints = []
            blobs = bucket.list_blobs(prefix=f"checkpoints/{prefix}")

            for blob in blobs:
                # Skip directories
                if blob.name.endswith('/'):
                    continue

                metadata = self._parse_checkpoint_metadata(blob)

                # Filter by experiment ID if specified
                if experiment_id and metadata.experiment_id != experiment_id:
                    continue

                checkpoints.append(metadata)

            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            return checkpoints

        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []

    def _parse_checkpoint_metadata(self, blob) -> CheckpointMetadata:
        """Parse checkpoint metadata from GCS blob."""
        metadata_dict = blob.metadata or {}

        # Extract metrics
        metrics = {}
        tags = []

        for key, value in metadata_dict.items():
            if key.startswith('metric_'):
                metric_name = key[7:]  # Remove 'metric_' prefix
                try:
                    metrics[metric_name] = float(value)
                except ValueError:
                    pass
            elif key == 'tags' and value:
                tags = value.split(',')

        return CheckpointMetadata(
            name=blob.name.replace('checkpoints/', ''),
            version=metadata_dict.get('version', '1.0'),
            created_at=datetime.fromisoformat(
                metadata_dict.get('created_at', blob.time_created.isoformat())
            ),
            size_bytes=blob.size,
            platform=metadata_dict.get('platform', 'unknown'),
            experiment_id=metadata_dict.get('experiment_id', 'unknown'),
            model_type=metadata_dict.get('model_type', 'unknown'),
            epoch=int(metadata_dict.get('epoch', 0)),
            loss=float(metadata_dict.get('loss', 0.0)),
            metrics=metrics if metrics else None,
            tags=tags if tags else None
        )

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete checkpoint from GCS.

        Args:
            checkpoint_name: Name of checkpoint to delete

        Returns:
            True if deletion successful, False otherwise
        """
        bucket = self.get_bucket()
        if not bucket:
            return False

        try:
            blob_name = f"checkpoints/{checkpoint_name}"
            blob = bucket.blob(blob_name)

            if blob.exists():
                blob.delete()
                self.logger.info(f"Deleted checkpoint: {blob_name}")
                return True
            else:
                self.logger.warning(f"Checkpoint not found for deletion: {blob_name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {e}")
            return False

    def get_bucket_usage(self) -> dict[str, any]:
        """Get bucket usage information."""
        bucket = self.get_bucket()
        if not bucket:
            return {}

        try:
            total_size = 0
            checkpoint_count = 0

            blobs = bucket.list_blobs(prefix="checkpoints/")
            for blob in blobs:
                if not blob.name.endswith('/'):
                    total_size += blob.size
                    checkpoint_count += 1

            # Convert to GB
            total_size_gb = total_size / (1024**3)

            return {
                'total_size_bytes': total_size,
                'total_size_gb': total_size_gb,
                'checkpoint_count': checkpoint_count,
                'free_tier_limit_gb': 5.0,
                'usage_percentage': (total_size_gb / 5.0) * 100,
                'space_remaining_gb': max(0, 5.0 - total_size_gb)
            }

        except Exception as e:
            self.logger.error(f"Failed to get bucket usage: {e}")
            return {}


class GCSCredentialsManager:
    """Manages GCS credentials securely."""

    @staticmethod
    def load_credentials_from_env() -> GCSConfig | None:
        """Load GCS credentials from environment variables."""
        project_id = os.getenv('GCS_PROJECT_ID')
        bucket_name = os.getenv('GCS_BUCKET_NAME')

        if not project_id or not bucket_name:
            return None

        return GCSConfig(
            project_id=project_id,
            bucket_name=bucket_name,
            credentials_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            credentials_json=os.getenv('GCS_CREDENTIALS_JSON'),
            region=os.getenv('GCS_REGION', 'us-central1'),
            storage_class=os.getenv('GCS_STORAGE_CLASS', 'STANDARD')
        )

    @staticmethod
    def setup_service_account(credentials_json: str,
                            local_credentials_path: str) -> bool:
        """
        Setup service account credentials locally.

        Args:
            credentials_json: JSON credentials string
            local_credentials_path: Path to save credentials file

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Validate JSON format
            credentials_dict = json.loads(credentials_json)
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']

            for field in required_fields:
                if field not in credentials_dict:
                    raise ValueError(f"Missing required field: {field}")

            # Save credentials to file
            credentials_path = Path(local_credentials_path)
            credentials_path.parent.mkdir(parents=True, exist_ok=True)

            with open(credentials_path, 'w') as f:
                json.dump(credentials_dict, f, indent=2)

            # Set permissions (Unix only)
            if os.name != 'nt':  # Not Windows
                os.chmod(credentials_path, 0o600)

            # Set environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

            return True

        except Exception as e:
            logging.error(f"Failed to setup service account: {e}")
            return False

    @staticmethod
    def validate_credentials(config: GCSConfig) -> bool:
        """Validate GCS credentials."""
        try:
            manager = GCSManager(config)
            return manager.authenticate()
        except Exception as e:
            logging.error(f"Credential validation failed: {e}")
            return False


def create_gcs_manager_from_env() -> GCSManager | None:
    """Create GCS manager from environment variables."""
    config = GCSCredentialsManager.load_credentials_from_env()
    if not config:
        return None

    return GCSManager(config)


def generate_checkpoint_name(experiment_id: str, epoch: int,
                           platform: str, timestamp: datetime | None = None) -> str:
    """Generate standardized checkpoint name."""
    if timestamp is None:
        timestamp = datetime.now()

    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"{experiment_id}_e{epoch:04d}_{platform}_{timestamp_str}.pt"


def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate SHA256 hash of file for integrity checking."""
    hash_sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()
