"""
Asynchronous checkpoint management for distributed evolution.

Provides non-blocking checkpoint operations with GCS storage integration
and comprehensive validation.
"""

import asyncio
import hashlib
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack
import structlog

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = structlog.get_logger(__name__)


@dataclass
class ProgramCheckpoint:
    """Individual program in checkpoint."""
    program: str
    fitness: float
    hash: str


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata."""
    platform_id: str
    timestamp: str
    resource_usage: dict[str, float] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Complete checkpoint structure."""
    version: str
    generation: int
    population: list[ProgramCheckpoint]
    metadata: CheckpointMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "generation": self.generation,
            "population": [
                {
                    "program": p.program,
                    "fitness": p.fitness,
                    "hash": p.hash
                }
                for p in self.population
            ],
            "metadata": {
                "platform_id": self.metadata.platform_id,
                "timestamp": self.metadata.timestamp,
                "resource_usage": self.metadata.resource_usage
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        population = [
            ProgramCheckpoint(
                program=p["program"],
                fitness=p["fitness"],
                hash=p["hash"]
            )
            for p in data["population"]
        ]

        metadata = CheckpointMetadata(
            platform_id=data["metadata"]["platform_id"],
            timestamp=data["metadata"]["timestamp"],
            resource_usage=data["metadata"].get("resource_usage", {})
        )

        return cls(
            version=data["version"],
            generation=data["generation"],
            population=population,
            metadata=metadata
        )


class AsyncCheckpointManager:
    """Manages asynchronous checkpoint operations for distributed evolution."""

    CHECKPOINT_VERSION = "1.0"

    def __init__(
        self,
        platform_id: str,
        checkpoint_dir: Path | None = None,
        gcs_bucket: str | None = None,
        gcs_credentials: str | None = None,
        enable_compression: bool = True,
        compression_level: int = 6
    ):
        """
        Initialize async checkpoint manager.

        Args:
            platform_id: Unique identifier for this platform
            checkpoint_dir: Local checkpoint directory
            gcs_bucket: GCS bucket name for cloud storage
            gcs_credentials: Path to GCS service account credentials
            enable_compression: Enable zlib compression for checkpoints
            compression_level: Compression level 0-9 (6 is default balance)
        """
        self.platform_id = platform_id
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.gcs_bucket = gcs_bucket
        self.gcs_credentials = gcs_credentials
        self.gcs_client = None
        self.enable_compression = enable_compression
        self.compression_level = compression_level

        if gcs_bucket and gcs_credentials:
            self._initialize_gcs_client()

        self._checkpoint_queue: asyncio.Queue = asyncio.Queue()
        self._checkpoint_task: asyncio.Task | None = None
        self._batch_buffer: list[tuple[int, Checkpoint]] = []
        self._batch_size = 5
        self._batch_timeout = 5.0

    def _initialize_gcs_client(self) -> None:
        """Initialize GCS client for cloud storage."""
        try:
            import os

            from google.cloud import storage

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.gcs_credentials
            self.gcs_client = storage.Client()
            logger.info(
                "gcs_client_initialized",
                bucket=self.gcs_bucket,
                platform_id=self.platform_id
            )
        except ImportError:
            logger.warning(
                "gcs_library_not_available",
                platform_id=self.platform_id
            )
        except Exception as e:
            logger.error(
                "gcs_client_initialization_failed",
                platform_id=self.platform_id,
                error=str(e)
            )

    def _calculate_program_hash(self, program: str) -> str:
        """Calculate SHA-256 hash of program for deduplication."""
        return hashlib.sha256(program.encode('utf-8')).hexdigest()

    def _get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage metrics."""
        if not HAS_PSUTIL:
            return {"cpu_percent": 0.0, "memory_mb": 0.0}

        try:
            process = psutil.Process()
            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / (1024 * 1024)
            }
        except Exception as e:
            logger.warning("resource_usage_failed", error=str(e))
            return {"cpu_percent": 0.0, "memory_mb": 0.0}

    def create_checkpoint(
        self,
        generation: int,
        population: list[dict[str, Any]]
    ) -> Checkpoint:
        """
        Create checkpoint from population.

        Args:
            generation: Current generation number
            population: List of individuals with program and fitness

        Returns:
            Checkpoint object
        """
        checkpoint_population = []
        for individual in population:
            program = individual.get("program", "")
            fitness = individual.get("fitness", 0.0)

            program_hash = individual.get("hash")
            if not program_hash:
                program_hash = self._calculate_program_hash(program)

            checkpoint_population.append(
                ProgramCheckpoint(
                    program=program,
                    fitness=fitness,
                    hash=program_hash
                )
            )

        metadata = CheckpointMetadata(
            platform_id=self.platform_id,
            timestamp=datetime.now().isoformat(),
            resource_usage=self._get_resource_usage()
        )

        return Checkpoint(
            version=self.CHECKPOINT_VERSION,
            generation=generation,
            population=checkpoint_population,
            metadata=metadata
        )

    def serialize_checkpoint(self, checkpoint: Checkpoint) -> bytes:
        """
        Serialize checkpoint to msgpack format with optional compression.

        Args:
            checkpoint: Checkpoint to serialize

        Returns:
            Serialized checkpoint bytes (compressed if enabled)
        """
        checkpoint_dict = checkpoint.to_dict()
        serialized = msgpack.packb(checkpoint_dict, use_bin_type=True)

        if self.enable_compression:
            return zlib.compress(serialized, level=self.compression_level)

        return serialized

    def deserialize_checkpoint(self, data: bytes) -> Checkpoint:
        """
        Deserialize checkpoint from msgpack format with decompression.

        Args:
            data: Serialized checkpoint bytes (possibly compressed)

        Returns:
            Checkpoint object
        """
        if self.enable_compression:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                pass

        checkpoint_dict = msgpack.unpackb(data, raw=False)
        return Checkpoint.from_dict(checkpoint_dict)

    async def save_checkpoint_async(
        self,
        checkpoint: Checkpoint,
        upload_to_gcs: bool = True
    ) -> Path:
        """
        Save checkpoint asynchronously.

        Args:
            checkpoint: Checkpoint to save
            upload_to_gcs: Whether to upload to GCS

        Returns:
            Path to saved checkpoint file
        """
        generation = checkpoint.generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{self.platform_id}_{generation}_{timestamp}.msgpack"
        local_path = self.checkpoint_dir / filename

        # Serialize checkpoint
        serialized_data = await asyncio.to_thread(
            self.serialize_checkpoint,
            checkpoint
        )

        # Write to local file
        await asyncio.to_thread(
            local_path.write_bytes,
            serialized_data
        )

        logger.info(
            "checkpoint_saved_locally",
            platform_id=self.platform_id,
            generation=generation,
            path=str(local_path),
            size_bytes=len(serialized_data)
        )

        # Upload to GCS if enabled
        if upload_to_gcs and self.gcs_client and self.gcs_bucket:
            await self._upload_to_gcs_async(local_path, filename)

        return local_path

    async def _upload_to_gcs_async(self, local_path: Path, filename: str) -> None:
        """Upload checkpoint to GCS asynchronously."""
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"generation_{local_path.stem.split('_')[2]}/{filename}")

            await asyncio.to_thread(
                blob.upload_from_filename,
                str(local_path)
            )

            logger.info(
                "checkpoint_uploaded_to_gcs",
                platform_id=self.platform_id,
                filename=filename,
                gcs_path=blob.name
            )
        except Exception as e:
            logger.error(
                "gcs_upload_failed",
                platform_id=self.platform_id,
                filename=filename,
                error=str(e)
            )

    async def load_checkpoint_async(
        self,
        generation: int | None = None,
        from_gcs: bool = False
    ) -> Checkpoint | None:
        """
        Load checkpoint asynchronously.

        Args:
            generation: Specific generation to load (None for latest)
            from_gcs: Whether to download from GCS

        Returns:
            Checkpoint object or None if not found
        """
        if from_gcs and self.gcs_client and self.gcs_bucket:
            checkpoint_data = await self._download_from_gcs_async(generation)
            if checkpoint_data:
                return self.deserialize_checkpoint(checkpoint_data)

        # Load from local storage
        checkpoint_files = sorted(self.checkpoint_dir.glob(f"checkpoint_{self.platform_id}_*.msgpack"))

        if not checkpoint_files:
            logger.warning("no_checkpoints_found", platform_id=self.platform_id)
            return None

        if generation is not None:
            # Find specific generation
            for checkpoint_file in checkpoint_files:
                if f"_{generation}_" in checkpoint_file.name:
                    data = await asyncio.to_thread(checkpoint_file.read_bytes)
                    return self.deserialize_checkpoint(data)
            return None
        else:
            # Load latest
            latest_file = checkpoint_files[-1]
            data = await asyncio.to_thread(latest_file.read_bytes)
            return self.deserialize_checkpoint(data)

    async def _download_from_gcs_async(
        self,
        generation: int | None = None
    ) -> bytes | None:
        """Download checkpoint from GCS asynchronously."""
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket)

            if generation is not None:
                prefix = f"generation_{generation}/"
            else:
                prefix = ""

            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                return None

            # Get latest blob
            latest_blob = sorted(blobs, key=lambda b: b.time_created, reverse=True)[0]

            data = await asyncio.to_thread(latest_blob.download_as_bytes)

            logger.info(
                "checkpoint_downloaded_from_gcs",
                platform_id=self.platform_id,
                gcs_path=latest_blob.name
            )

            return data
        except Exception as e:
            logger.error(
                "gcs_download_failed",
                platform_id=self.platform_id,
                generation=generation,
                error=str(e)
            )
            return None

    def validate_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint: Checkpoint to validate

        Returns:
            True if checkpoint is valid
        """
        try:
            # Check version
            if checkpoint.version != self.CHECKPOINT_VERSION:
                logger.warning(
                    "checkpoint_version_mismatch",
                    expected=self.CHECKPOINT_VERSION,
                    actual=checkpoint.version
                )
                return False

            # Check population
            if not checkpoint.population:
                logger.warning("checkpoint_empty_population")
                return False

            # Validate each program
            for program_checkpoint in checkpoint.population:
                if not program_checkpoint.program:
                    logger.warning("checkpoint_empty_program")
                    return False

                # Verify hash
                calculated_hash = self._calculate_program_hash(program_checkpoint.program)
                if calculated_hash != program_checkpoint.hash:
                    logger.warning(
                        "checkpoint_hash_mismatch",
                        expected=program_checkpoint.hash,
                        calculated=calculated_hash
                    )
                    return False

            # Check metadata
            if not checkpoint.metadata.platform_id:
                logger.warning("checkpoint_missing_platform_id")
                return False

            if not checkpoint.metadata.timestamp:
                logger.warning("checkpoint_missing_timestamp")
                return False

            return True

        except Exception as e:
            logger.error("checkpoint_validation_error", error=str(e))
            return False

    async def start_background_checkpointing(self) -> None:
        """Start background task for processing checkpoint queue."""
        if self._checkpoint_task is None or self._checkpoint_task.done():
            self._checkpoint_task = asyncio.create_task(self._checkpoint_worker())
            logger.info("background_checkpointing_started", platform_id=self.platform_id)

    async def _checkpoint_worker(self) -> None:
        """Worker task for processing checkpoint queue."""
        while True:
            try:
                checkpoint_data = await self._checkpoint_queue.get()

                if checkpoint_data is None:
                    break

                checkpoint, upload_to_gcs = checkpoint_data
                await self.save_checkpoint_async(checkpoint, upload_to_gcs)

                self._checkpoint_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("checkpoint_worker_error", error=str(e))

    async def queue_checkpoint(
        self,
        checkpoint: Checkpoint,
        upload_to_gcs: bool = True
    ) -> None:
        """
        Queue checkpoint for background saving.

        Args:
            checkpoint: Checkpoint to save
            upload_to_gcs: Whether to upload to GCS
        """
        await self._checkpoint_queue.put((checkpoint, upload_to_gcs))

    async def stop_background_checkpointing(self) -> None:
        """Stop background checkpointing and wait for queue to empty."""
        await self._checkpoint_queue.join()
        await self._checkpoint_queue.put(None)

        if self._checkpoint_task:
            await self._checkpoint_task

        logger.info("background_checkpointing_stopped", platform_id=self.platform_id)
