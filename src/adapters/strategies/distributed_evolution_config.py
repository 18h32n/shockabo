"""
Configuration classes for distributed evolution.

Provides structured configuration for distributed evolution across platforms.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.infrastructure.components.platform_detector import Platform


@dataclass
class PlatformConfig:
    """Configuration for a single platform in distributed evolution."""

    id: str
    role: str
    memory_limit_mb: int
    worker_count: int
    batch_size: int
    platform_type: Platform | None = None


@dataclass
class CoordinatorConfig:
    """Configuration for the coordinator service."""

    api_host: str = "localhost"
    api_port: int = 8000
    coordinator_id: str = "coordinator-1"


@dataclass
class DistributedEvolutionConfig:
    """
    Complete configuration for distributed evolution.

    This configuration enables evolution across multiple platforms
    with checkpoint synchronization and fault tolerance.
    """

    enabled: bool = False
    checkpoint_frequency: int = 1
    heartbeat_timeout: int = 30
    heartbeat_interval: int = 10
    recovery_timeout: int = 60
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    platforms: list[PlatformConfig] = field(default_factory=list)
    gcs_bucket: str | None = None
    gcs_credentials_path: Path | None = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("distributed_checkpoints"))
    enable_compression: bool = True
    compression_level: int = 6
    batch_checkpoint_size: int = 5
    batch_checkpoint_timeout: float = 5.0

    def get_coordinator_url(self) -> str:
        """Get full coordinator URL."""
        return f"http://{self.coordinator.api_host}:{self.coordinator.api_port}"

    def get_platform_config(self, platform_id: str) -> PlatformConfig | None:
        """Get configuration for specific platform."""
        for platform_config in self.platforms:
            if platform_config.id == platform_id:
                return platform_config
        return None

    def get_coordinator_platform(self) -> PlatformConfig | None:
        """Get the coordinator platform configuration."""
        for platform_config in self.platforms:
            if platform_config.role == "coordinator":
                return platform_config
        return None

    def get_worker_platforms(self) -> list[PlatformConfig]:
        """Get all worker platform configurations."""
        return [
            platform_config
            for platform_config in self.platforms
            if platform_config.role == "worker"
        ]

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.enabled:
            return errors

        if not self.platforms:
            errors.append("No platforms configured for distributed evolution")

        coordinator_count = sum(
            1 for p in self.platforms if p.role == "coordinator"
        )
        if coordinator_count == 0:
            errors.append("No coordinator platform configured")
        elif coordinator_count > 1:
            errors.append(f"Multiple coordinators configured ({coordinator_count})")

        if self.checkpoint_frequency < 1:
            errors.append("Checkpoint frequency must be >= 1")

        if self.heartbeat_timeout < 1:
            errors.append("Heartbeat timeout must be >= 1")

        if self.heartbeat_interval < 1:
            errors.append("Heartbeat interval must be >= 1")

        if self.heartbeat_interval >= self.heartbeat_timeout:
            errors.append("Heartbeat interval must be less than timeout")

        if self.recovery_timeout < 1:
            errors.append("Recovery timeout must be >= 1")

        if self.compression_level < 0 or self.compression_level > 9:
            errors.append("Compression level must be 0-9")

        for platform_config in self.platforms:
            if platform_config.memory_limit_mb < 1024:
                errors.append(
                    f"Platform {platform_config.id} memory limit too low "
                    f"({platform_config.memory_limit_mb}MB < 1024MB)"
                )

            if platform_config.worker_count < 1:
                errors.append(
                    f"Platform {platform_config.id} must have at least 1 worker"
                )

            if platform_config.batch_size < 1:
                errors.append(
                    f"Platform {platform_config.id} batch size must be >= 1"
                )

            if platform_config.role not in ["coordinator", "worker", "backup"]:
                errors.append(
                    f"Platform {platform_config.id} has invalid role: {platform_config.role}"
                )

        if self.enabled and self.gcs_bucket is None:
            errors.append("GCS bucket required when distributed evolution is enabled")

        if self.gcs_bucket and not self.gcs_credentials_path:
            errors.append("GCS credentials path required when GCS bucket is specified")

        if self.gcs_credentials_path:
            creds_path = Path(self.gcs_credentials_path)
            if not creds_path.exists():
                errors.append(f"GCS credentials file not found: {creds_path}")

        return errors

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistributedEvolutionConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            DistributedEvolutionConfig instance
        """
        coordinator_data = config_dict.get("coordinator", {})
        coordinator = CoordinatorConfig(
            api_host=coordinator_data.get("api_host", "localhost"),
            api_port=coordinator_data.get("api_port", 8000),
            coordinator_id=coordinator_data.get("coordinator_id", "coordinator-1"),
        )

        platforms = []
        for platform_data in config_dict.get("platforms", []):
            platform_type_str = platform_data.get("platform_type")
            platform_type = None
            if platform_type_str:
                try:
                    platform_type = Platform(platform_type_str)
                except ValueError:
                    pass

            platform = PlatformConfig(
                id=platform_data["id"],
                role=platform_data["role"],
                memory_limit_mb=platform_data["memory_limit_mb"],
                worker_count=platform_data["worker_count"],
                batch_size=platform_data["batch_size"],
                platform_type=platform_type,
            )
            platforms.append(platform)

        gcs_credentials_path = config_dict.get("gcs_credentials_path")
        if gcs_credentials_path:
            gcs_credentials_path = Path(gcs_credentials_path)

        checkpoint_dir = Path(config_dict.get("checkpoint_dir", "distributed_checkpoints"))

        return cls(
            enabled=config_dict.get("enabled", False),
            checkpoint_frequency=config_dict.get("checkpoint_frequency", 1),
            heartbeat_timeout=config_dict.get("heartbeat_timeout", 30),
            heartbeat_interval=config_dict.get("heartbeat_interval", 10),
            recovery_timeout=config_dict.get("recovery_timeout", 60),
            coordinator=coordinator,
            platforms=platforms,
            gcs_bucket=config_dict.get("gcs_bucket"),
            gcs_credentials_path=gcs_credentials_path,
            checkpoint_dir=checkpoint_dir,
            enable_compression=config_dict.get("enable_compression", True),
            compression_level=config_dict.get("compression_level", 6),
            batch_checkpoint_size=config_dict.get("batch_checkpoint_size", 5),
            batch_checkpoint_timeout=config_dict.get("batch_checkpoint_timeout", 5.0),
        )
