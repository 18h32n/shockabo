"""Configuration management with platform detection and environment setup."""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

import psutil
import yaml
from dotenv import load_dotenv


class Platform(Enum):
    """Supported platforms for ARC development."""
    KAGGLE = "kaggle"
    COLAB = "colab"
    PAPERSPACE = "paperspace"
    LOCAL = "local"


@dataclass
class PlatformInfo:
    """Platform-specific information and constraints."""
    platform: Platform
    gpu_hours_limit: int
    reset_frequency: str
    max_memory_gb: int
    has_persistent_storage: bool
    setup_script: str


class PlatformDetector:
    """Detects the current execution platform and provides configuration."""

    PLATFORM_CONFIGS = {
        Platform.KAGGLE: PlatformInfo(
            platform=Platform.KAGGLE,
            gpu_hours_limit=30,
            reset_frequency="weekly",
            max_memory_gb=32,  # 24GB GPU + 8GB system RAM
            has_persistent_storage=True,
            setup_script="kaggle_setup.py"
        ),
        Platform.COLAB: PlatformInfo(
            platform=Platform.COLAB,
            gpu_hours_limit=12,
            reset_frequency="daily",
            max_memory_gb=16,
            has_persistent_storage=False,
            setup_script="colab_setup.py"
        ),
        Platform.PAPERSPACE: PlatformInfo(
            platform=Platform.PAPERSPACE,
            gpu_hours_limit=6,
            reset_frequency="daily",
            max_memory_gb=8,
            has_persistent_storage=True,
            setup_script="paperspace_setup.py"
        ),
        Platform.LOCAL: PlatformInfo(
            platform=Platform.LOCAL,
            gpu_hours_limit=9999,  # No limit for local
            reset_frequency="never",
            max_memory_gb=32,  # Increased for 8B model support
            has_persistent_storage=True,
            setup_script="local_setup.py"
        )
    }

    @staticmethod
    def detect_platform() -> Platform:
        """
        Detect the current execution platform.

        Returns:
            Platform: The detected platform
        """
        # Check for Kaggle environment
        if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return Platform.KAGGLE

        # Check for Google Colab environment
        if 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
            return Platform.COLAB

        # Check for Paperspace environment
        if 'PS_API_KEY' in os.environ or os.path.exists('/storage'):
            return Platform.PAPERSPACE

        # Default to local environment
        return Platform.LOCAL

    @staticmethod
    def get_platform_info(platform: Platform | None = None) -> PlatformInfo:
        """
        Get platform-specific information.

        Args:
            platform: Platform to get info for. If None, detects current platform.

        Returns:
            PlatformInfo: Platform configuration and constraints
        """
        if platform is None:
            platform = PlatformDetector.detect_platform()

        return PlatformDetector.PLATFORM_CONFIGS[platform]

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available on the current platform."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_gpu_memory_info() -> dict[str, Any]:
        """Get GPU memory information."""
        gpu_info = {
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_memory_total_mb": 0,
            "gpu_memory_available_mb": 0,
            "gpu_names": [],
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["gpu_available"] = True
                gpu_info["gpu_count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info["gpu_names"].append(props.name)
                    
                    # Get memory info for first GPU
                    if i == 0:
                        total_memory = props.total_memory / (1024 * 1024)  # Convert to MB
                        reserved_memory = torch.cuda.memory_reserved(i) / (1024 * 1024)
                        allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
                        
                        gpu_info["gpu_memory_total_mb"] = total_memory
                        gpu_info["gpu_memory_available_mb"] = total_memory - reserved_memory
                        gpu_info["gpu_memory_allocated_mb"] = allocated_memory
                        
        except ImportError:
            pass
            
        return gpu_info
    
    @staticmethod
    def validate_8b_model_requirements() -> dict[str, Any]:
        """Validate system requirements for 8B model with QLoRA."""
        gpu_info = PlatformDetector.get_gpu_memory_info()
        platform_info = PlatformDetector.get_platform_info()
        
        # Requirements for Llama-3 8B with 4-bit quantization
        min_gpu_memory_mb = 6000   # 6GB minimum with QLoRA
        recommended_gpu_memory_mb = 8000  # 8GB recommended
        optimal_gpu_memory_mb = 24000     # 24GB optimal
        
        validation = {
            "platform": platform_info.platform.value,
            "gpu_available": gpu_info["gpu_available"],
            "gpu_memory_total_mb": gpu_info["gpu_memory_total_mb"],
            "gpu_memory_available_mb": gpu_info["gpu_memory_available_mb"],
            "meets_minimum": False,
            "meets_recommended": False,
            "meets_optimal": False,
            "memory_level": "insufficient",
            "recommendations": [],
        }
        
        if gpu_info["gpu_available"] and gpu_info["gpu_memory_total_mb"] > 0:
            available_memory = gpu_info["gpu_memory_available_mb"]
            
            validation["meets_minimum"] = available_memory >= min_gpu_memory_mb
            validation["meets_recommended"] = available_memory >= recommended_gpu_memory_mb
            validation["meets_optimal"] = available_memory >= optimal_gpu_memory_mb
            
            if validation["meets_optimal"]:
                validation["memory_level"] = "optimal"
                validation["recommendations"].append("System has optimal memory for 8B model")
            elif validation["meets_recommended"]:
                validation["memory_level"] = "recommended"
                validation["recommendations"].append("System has recommended memory for 8B model")
            elif validation["meets_minimum"]:
                validation["memory_level"] = "minimum"
                validation["recommendations"].extend([
                    "System meets minimum requirements",
                    "Consider using gradient checkpointing",
                    "Use batch size 1 for training"
                ])
            else:
                validation["recommendations"].extend([
                    "Insufficient GPU memory for 8B model",
                    "Consider using smaller model (1B or 3B)",
                    "Upgrade to GPU with more memory",
                    "Use CPU-only mode (very slow)"
                ])
        else:
            validation["recommendations"].append("No GPU available, CPU-only mode will be very slow")
            
        return validation
    
    @staticmethod
    def get_resource_limits() -> dict[str, Any]:
        """Get current platform resource limits with 8B model validation."""
        platform_info = PlatformDetector.get_platform_info()
        memory = psutil.virtual_memory()
        gpu_info = PlatformDetector.get_gpu_memory_info()
        model_validation = PlatformDetector.validate_8b_model_requirements()

        return {
            "platform": platform_info.platform.value,
            "gpu_available": gpu_info["gpu_available"],
            "gpu_hours_limit": platform_info.gpu_hours_limit,
            "max_memory_gb": platform_info.max_memory_gb,
            "available_memory_gb": memory.available // (1024**3),
            "total_memory_gb": memory.total // (1024**3),
            "cpu_cores": psutil.cpu_count(),
            "has_persistent_storage": platform_info.has_persistent_storage,
            "gpu_info": gpu_info,
            "model_8b_validation": model_validation,
        }


class ConfigManager:
    """Manages configuration loading and platform-specific adjustments."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("configs")
        self.platform = PlatformDetector.detect_platform()
        self.platform_info = PlatformDetector.get_platform_info(self.platform)

        # Load environment variables
        load_dotenv()

        # Load base configuration
        self._config = self._load_base_config()

        # Apply platform-specific overrides
        self._apply_platform_overrides()

    def _load_base_config(self) -> dict[str, Any]:
        """Load base configuration from YAML files."""
        config: dict[str, Any] = {}

        # Load development.yaml as base
        dev_config_path = self.config_dir / "development.yaml"
        if dev_config_path.exists():
            with open(dev_config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

        # Load platform-specific config if it exists
        platform_config_path = self.config_dir / f"{self.platform.value}.yaml"
        if platform_config_path.exists():
            with open(platform_config_path, encoding='utf-8') as f:
                platform_config = yaml.safe_load(f) or {}
                # Perform a deep merge of platform-specific config
                def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
                    result = base.copy()
                    for key, value in override.items():
                        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                            result[key] = deep_merge(result[key], value)
                        else:
                            result[key] = value
                    return result
                config = deep_merge(config, platform_config)

        return config

    def _apply_platform_overrides(self) -> None:
        """Apply platform-specific configuration overrides."""
        overrides = {
            "platform": {
                "name": self.platform.value,
                "gpu_hours_limit": self.platform_info.gpu_hours_limit,
                "max_memory_gb": self.platform_info.max_memory_gb,
                "has_persistent_storage": self.platform_info.has_persistent_storage,
            },
            "resources": PlatformDetector.get_resource_limits(),
        }

        # Apply memory-based adjustments for 8B model
        available_memory = psutil.virtual_memory().available // (1024**3)
        model_validation = PlatformDetector.validate_8b_model_requirements()
        
        overrides["model"] = overrides.get("model", {})
        
        # Configure based on GPU memory availability
        if model_validation["meets_optimal"]:
            # Optimal configuration for 24GB+ GPU
            overrides["model"].update({
                "name": "meta-llama/Llama-3-8B",
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "use_flash_attention": True,
            })
        elif model_validation["meets_recommended"]:
            # Recommended configuration for 8GB+ GPU
            overrides["model"].update({
                "name": "meta-llama/Llama-3-8B",
                "batch_size": 1,
                "gradient_accumulation_steps": 2,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "gradient_checkpointing": True,
                "use_flash_attention": True,
            })
        elif model_validation["meets_minimum"]:
            # Minimum configuration for 6GB+ GPU
            overrides["model"].update({
                "name": "meta-llama/Llama-3-8B",
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "gradient_checkpointing": True,
                "use_flash_attention": True,
                "max_sequence_length": 1024,  # Reduce context length
            })
        else:
            # Fallback to smaller model
            overrides["model"].update({
                "name": "meta-llama/Llama-3.2-1B",
                "batch_size": min(self._config.get("model", {}).get("batch_size", 32), 8),
                "max_sequence_length": min(self._config.get("model", {}).get("max_sequence_length", 2048), 1024),
            })
        
        # Additional memory constraints for system RAM
        if available_memory < 4:
            overrides["model"]["batch_size"] = 1
            overrides["model"]["gradient_accumulation_steps"] = max(
                overrides["model"].get("gradient_accumulation_steps", 1), 2
            )

        # Apply GPU-based adjustments
        if not PlatformDetector.is_gpu_available():
            overrides["model"] = overrides.get("model", {})
            overrides["model"]["device"] = "cpu"
            overrides["model"]["use_fp16"] = False

        # Deep merge overrides
        self._config = self._deep_merge(self._config, overrides)

    def _deep_merge(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()

    def get_platform_info(self) -> PlatformInfo:
        """Get current platform information."""
        return self.platform_info

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'development'

    def get_data_dir(self) -> Path | PurePath:
        """Get platform-appropriate data directory."""
        if self.platform == Platform.KAGGLE:
            return PurePosixPath('/kaggle/input')
        elif self.platform == Platform.COLAB:
            return PurePosixPath('/content/data')
        elif self.platform == Platform.PAPERSPACE:
            return PurePosixPath('/storage/data')
        else:
            return Path('data')

    def get_output_dir(self) -> Path | PurePath:
        """Get platform-appropriate output directory."""
        if self.platform == Platform.KAGGLE:
            return PurePosixPath('/kaggle/working')
        elif self.platform == Platform.COLAB:
            return PurePosixPath('/content/output')
        elif self.platform == Platform.PAPERSPACE:
            return PurePosixPath('/storage/output')
        else:
            return Path('output')

    def get_cache_dir(self) -> Path | PurePath:
        """Get platform-appropriate cache directory."""
        if self.platform == Platform.KAGGLE:
            return PurePosixPath('/kaggle/working/cache')
        elif self.platform == Platform.COLAB:
            return PurePosixPath('/content/cache')
        elif self.platform == Platform.PAPERSPACE:
            return PurePosixPath('/storage/cache')
        else:
            return Path('data/cache')


# Global configuration instance
_config_manager: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config(config_dir: Path | None = None) -> ConfigManager:
    """
    Initialize the global configuration manager.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        ConfigManager: Initialized configuration manager
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return _config_manager