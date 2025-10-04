"""
Platform-specific configuration manager for evolution engine.

Provides automatic configuration based on detected platform to optimize
performance and resource utilization.
"""

from dataclasses import dataclass
from pathlib import Path

from src.adapters.strategies.evolution_engine import load_evolution_config
from src.infrastructure.components.platform_detector import Platform, get_platform_detector
from src.infrastructure.config import GeneticAlgorithmConfig


@dataclass
class PlatformEvolutionSettings:
    """Platform-specific evolution settings."""
    workers: int
    batch_size: int
    gpu_batch_size: int
    memory_limit_mb: int
    max_generations: int
    generation_timeout: int
    enable_gpu: bool
    enable_checkpointing: bool
    checkpoint_frequency: int
    cpu_limit_percent: float = 90.0
    disk_limit_mb: int = 10240
    network_bandwidth_mbps: float = 100.0
    max_concurrent_tasks: int = 4


class PlatformEvolutionConfigurator:
    """Configures evolution engine based on detected platform."""

    # Platform-specific settings
    PLATFORM_SETTINGS = {
        Platform.KAGGLE: PlatformEvolutionSettings(
            workers=2,
            batch_size=500,
            gpu_batch_size=150,
            memory_limit_mb=4096,
            max_generations=200,
            generation_timeout=30,
            enable_gpu=True,
            enable_checkpointing=True,
            checkpoint_frequency=10,
            cpu_limit_percent=85.0,
            disk_limit_mb=5120,
            network_bandwidth_mbps=50.0,
            max_concurrent_tasks=2
        ),
        Platform.COLAB: PlatformEvolutionSettings(
            workers=2,
            batch_size=250,
            gpu_batch_size=200,
            memory_limit_mb=12288,
            max_generations=100,
            generation_timeout=25,
            enable_gpu=True,
            enable_checkpointing=True,
            checkpoint_frequency=5,
            cpu_limit_percent=90.0,
            disk_limit_mb=15360,
            network_bandwidth_mbps=100.0,
            max_concurrent_tasks=3
        ),
        Platform.PAPERSPACE: PlatformEvolutionSettings(
            workers=1,
            batch_size=100,
            gpu_batch_size=50,
            memory_limit_mb=1024,
            max_generations=50,
            generation_timeout=20,
            enable_gpu=False,
            enable_checkpointing=True,
            checkpoint_frequency=5,
            cpu_limit_percent=80.0,
            disk_limit_mb=2048,
            network_bandwidth_mbps=25.0,
            max_concurrent_tasks=1
        ),
        Platform.LOCAL: PlatformEvolutionSettings(
            workers=4,
            batch_size=250,
            gpu_batch_size=100,
            memory_limit_mb=8192,
            max_generations=200,
            generation_timeout=30,
            enable_gpu=True,
            enable_checkpointing=False,
            checkpoint_frequency=20,
            cpu_limit_percent=95.0,
            disk_limit_mb=20480,
            network_bandwidth_mbps=1000.0,
            max_concurrent_tasks=8
        )
    }

    def __init__(self):
        self.detector = get_platform_detector()
        self.platform_info = self.detector.detect_platform()
        self.current_platform = self.platform_info.platform

    def get_configured_config(
        self,
        base_config: GeneticAlgorithmConfig | None = None,
        config_path: Path | None = None
    ) -> GeneticAlgorithmConfig:
        """
        Get evolution config with platform-specific optimizations applied.

        Args:
            base_config: Base configuration to modify (if None, loads from YAML)
            config_path: Path to YAML config file

        Returns:
            Configured GeneticAlgorithmConfig
        """
        # Load base config
        if base_config is None:
            if config_path:
                config = load_evolution_config(config_path)
            else:
                config = load_evolution_config()  # Default path
        else:
            config = base_config

        # Get platform settings
        settings = self.PLATFORM_SETTINGS.get(
            self.current_platform,
            self.PLATFORM_SETTINGS[Platform.LOCAL]  # Fallback
        )

        # Apply platform-specific settings
        self._apply_platform_settings(config, settings)

        # Apply additional platform-specific tweaks
        self._apply_platform_tweaks(config)

        print(f"Configured evolution for {self.current_platform.value} platform:")
        print(f"  Workers: {config.parallelization.workers}")
        print(f"  Batch size: {config.parallelization.batch_size}")
        print(f"  Memory limit: {config.performance.memory_limit}MB")
        print(f"  GPU enabled: {config.parallelization.gpu_acceleration}")
        print(f"  Max generations: {config.convergence.max_generations}")

        return config

    def _apply_platform_settings(
        self,
        config: GeneticAlgorithmConfig,
        settings: PlatformEvolutionSettings
    ) -> None:
        """Apply platform settings to config."""
        # Parallelization settings
        config.parallelization.workers = settings.workers
        config.parallelization.batch_size = settings.batch_size
        config.parallelization.gpu_batch_size = settings.gpu_batch_size
        config.parallelization.gpu_acceleration = (
            settings.enable_gpu and self.platform_info.gpu_available
        )

        # Performance settings
        config.performance.memory_limit = settings.memory_limit_mb
        config.performance.generation_timeout = settings.generation_timeout

        # Convergence settings
        config.convergence.max_generations = settings.max_generations

        # Reproducibility settings
        config.reproducibility.checkpoint_enabled = settings.enable_checkpointing
        config.reproducibility.checkpoint_frequency = settings.checkpoint_frequency

    def _apply_platform_tweaks(self, config: GeneticAlgorithmConfig) -> None:
        """Apply additional platform-specific tweaks."""
        if self.current_platform == Platform.KAGGLE:
            # Kaggle-specific tweaks
            config.population.size = 1000  # Can handle larger populations
            config.diversity.method = "speciation"  # More efficient

        elif self.current_platform == Platform.COLAB:
            # Colab-specific tweaks
            config.population.size = 500  # Balance GPU/memory
            # Enable island model for better GPU utilization
            if config.island_model:
                config.island_model['enabled'] = True
                config.island_model['num_islands'] = 2  # Match CPU cores

        elif self.current_platform == Platform.PAPERSPACE:
            # Paperspace-specific tweaks
            config.population.size = 200  # Small population
            config.fitness.early_termination['enabled'] = True
            config.fitness.early_termination['threshold'] = 0.9  # Stop early

        # Adjust cache settings based on available storage
        if self.platform_info.storage_gb < 10:
            config.fitness.cache_enabled = False  # Disable cache on low storage

    def get_resource_monitor_config(self) -> dict:
        """Get resource monitoring configuration for platform."""
        monitor_config = {
            'enabled': True,
            'check_interval': 60,  # seconds
            'memory_threshold': 0.9,  # 90% memory usage
            'time_limit': None,
            'gpu_memory_threshold': 0.95
        }

        # Platform-specific monitoring
        if self.current_platform == Platform.KAGGLE:
            monitor_config['time_limit'] = 12 * 3600  # 12 hour session
            monitor_config['quota_check'] = True

        elif self.current_platform == Platform.COLAB:
            monitor_config['time_limit'] = 12 * 3600  # 12 hour session
            monitor_config['check_interval'] = 30  # More frequent checks

        elif self.current_platform == Platform.PAPERSPACE:
            monitor_config['time_limit'] = 6 * 3600  # 6 hour session
            monitor_config['memory_threshold'] = 0.8  # More conservative

        return monitor_config


def create_platform_optimized_config(
    config_path: Path | None = None
) -> GeneticAlgorithmConfig:
    """
    Create a platform-optimized evolution configuration.

    This is a convenience function that detects the platform and
    returns an optimized configuration.
    """
    configurator = PlatformEvolutionConfigurator()
    return configurator.get_configured_config(config_path=config_path)
