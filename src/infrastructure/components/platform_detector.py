"""Platform detection service for identifying current execution environment."""

import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from enum import Enum


class Platform(Enum):
    """Supported platform types."""
    KAGGLE = "kaggle"
    COLAB = "colab"
    PAPERSPACE = "paperspace"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Platform information container."""
    platform: Platform
    gpu_available: bool
    gpu_count: int
    memory_gb: float
    storage_gb: float
    runtime_limits: dict[str, any] = None
    metadata: dict[str, any] = None


class PlatformDetector:
    """Service for detecting current platform and its capabilities."""

    def __init__(self):
        self._cached_info: PlatformInfo | None = None

    def detect_platform(self, force_refresh: bool = False) -> PlatformInfo:
        """
        Detect current platform and return platform information.

        Args:
            force_refresh: Force re-detection even if cached

        Returns:
            PlatformInfo object with detected platform details
        """
        if self._cached_info is not None and not force_refresh:
            return self._cached_info

        platform_type = self._identify_platform()
        gpu_info = self._detect_gpu_capabilities()
        system_info = self._get_system_info()
        limits = self._detect_runtime_limits(platform_type)
        metadata = self._gather_platform_metadata(platform_type)

        self._cached_info = PlatformInfo(
            platform=platform_type,
            gpu_available=gpu_info[0],
            gpu_count=gpu_info[1],
            memory_gb=system_info[0],
            storage_gb=system_info[1],
            runtime_limits=limits,
            metadata=metadata
        )

        return self._cached_info

    def _identify_platform(self) -> Platform:
        """Identify the current platform based on environment indicators."""
        # Check for Kaggle environment
        if self._is_kaggle():
            return Platform.KAGGLE

        # Check for Google Colab
        if self._is_colab():
            return Platform.COLAB

        # Check for Paperspace
        if self._is_paperspace():
            return Platform.PAPERSPACE

        # Check for local development
        if self._is_local():
            return Platform.LOCAL

        return Platform.UNKNOWN

    def _is_kaggle(self) -> bool:
        """Check if running on Kaggle."""
        return (
            os.path.exists('/kaggle') or
            'KAGGLE_KERNEL_RUN_TYPE' in os.environ or
            'KAGGLE_URL_BASE' in os.environ or
            'KAGGLE_USER_SECRETS_TOKEN' in os.environ
        )

    def _is_colab(self) -> bool:
        """Check if running on Google Colab."""
        colab_env_check = (
            'COLAB_GPU' in os.environ or
            'COLAB_TPU_ADDR' in os.environ or
            os.path.exists('/content')
        )

        # Check if running in Jupyter notebook (Colab indicator)
        try:
            from IPython import get_ipython
            ipython_check = 'google.colab' in str(get_ipython()) if get_ipython() else False
        except ImportError:
            ipython_check = False

        return colab_env_check or ipython_check

    def _is_paperspace(self) -> bool:
        """Check if running on Paperspace."""
        return (
            'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ or
            'PAPERSPACE_API_KEY' in os.environ or
            'PS_API_KEY' in os.environ or
            os.path.exists('/notebooks')
        )

    def _is_local(self) -> bool:
        """Check if running on local machine."""
        # This is the fallback, but we can add specific checks
        socket.gethostname()
        return not any([
            self._is_kaggle(),
            self._is_colab(),
            self._is_paperspace()
        ])

    def _detect_gpu_capabilities(self) -> tuple[bool, int]:
        """
        Detect GPU availability and count.

        Returns:
            Tuple of (gpu_available, gpu_count)
        """
        try:
            import torch
            if torch.cuda.is_available():
                return True, torch.cuda.device_count()
        except ImportError:
            pass

        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                return True, gpu_count
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False, 0

    def _get_system_info(self) -> tuple[float, float]:
        """
        Get system memory and storage information.

        Returns:
            Tuple of (memory_gb, storage_gb)
        """
        try:
            # Memory info
            import psutil
            memory_bytes = psutil.virtual_memory().total
            memory_gb = memory_bytes / (1024**3)

            # Storage info
            disk_usage = psutil.disk_usage('/')
            storage_gb = disk_usage.total / (1024**3)

            return memory_gb, storage_gb
        except ImportError:
            # Fallback methods
            try:
                # Memory from /proc/meminfo
                with open('/proc/meminfo') as f:
                    mem_total = int([line for line in f if 'MemTotal' in line][0].split()[1])
                    memory_gb = mem_total / (1024**2)  # kB to GB
            except (FileNotFoundError, IndexError):
                memory_gb = 0.0

            try:
                # Storage from df
                result = subprocess.run(['df', '/'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        total_kb = int(lines[1].split()[1])
                        storage_gb = total_kb / (1024**2)  # kB to GB
                    else:
                        storage_gb = 0.0
                else:
                    storage_gb = 0.0
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                storage_gb = 0.0

            return memory_gb, storage_gb

    def _detect_runtime_limits(self, platform_type: Platform) -> dict[str, any]:
        """Detect platform-specific runtime limits."""
        limits = {}

        if platform_type == Platform.KAGGLE:
            limits.update({
                'gpu_hours_weekly': 30,
                'session_timeout_hours': 12,
                'internet_access': True,
                'storage_gb_limit': 73,  # Standard Kaggle notebook
                'weekly_reset': True
            })
        elif platform_type == Platform.COLAB:
            limits.update({
                'gpu_hours_daily': 12,
                'session_timeout_hours': 12,
                'internet_access': True,
                'storage_gb_limit': 107,  # Standard Colab
                'daily_reset': True
            })
        elif platform_type == Platform.PAPERSPACE:
            limits.update({
                'gpu_hours_daily': 6,
                'session_timeout_hours': 6,
                'internet_access': True,
                'storage_gb_limit': 5,  # Free tier
                'daily_reset': True
            })
        elif platform_type == Platform.LOCAL:
            limits.update({
                'gpu_hours_daily': -1,  # Unlimited
                'session_timeout_hours': -1,  # No timeout
                'internet_access': True,
                'storage_gb_limit': -1,  # Based on actual disk
                'daily_reset': False
            })

        return limits

    def _gather_platform_metadata(self, platform_type: Platform) -> dict[str, any]:
        """Gather platform-specific metadata."""
        metadata = {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'architecture': platform.machine(),
            'hostname': socket.gethostname()
        }

        # Platform-specific metadata
        if platform_type == Platform.KAGGLE:
            metadata.update({
                'kernel_type': os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
                'username': os.environ.get('KAGGLE_USERNAME'),
                'competition': os.environ.get('KAGGLE_COMPETITION_NAME')
            })
        elif platform_type == Platform.COLAB:
            try:
                # Try to get Colab-specific info
                from google.colab import drive  # noqa: F401
                metadata['colab_pro'] = False  # TODO: Detect Pro tier
            except ImportError:
                pass
        elif platform_type == Platform.PAPERSPACE:
            metadata.update({
                'notebook_id': os.environ.get('PAPERSPACE_NOTEBOOK_REPO_ID'),
                'api_key_present': 'PAPERSPACE_API_KEY' in os.environ
            })

        return metadata

    def is_platform(self, platform_type: Platform) -> bool:
        """Check if current platform matches the specified type."""
        current = self.detect_platform()
        return current.platform == platform_type

    def get_quota_info(self) -> dict[str, any]:
        """Get platform quota and usage information."""
        info = self.detect_platform()
        quota_info = {}

        if info.platform == Platform.KAGGLE:
            # TODO: Implement Kaggle-specific quota checking
            quota_info['gpu_hours_used'] = 0  # Placeholder
            quota_info['gpu_hours_remaining'] = info.runtime_limits.get('gpu_hours_weekly', 30)
        elif info.platform == Platform.COLAB:
            # TODO: Implement Colab-specific quota checking
            quota_info['gpu_hours_used'] = 0  # Placeholder
            quota_info['gpu_hours_remaining'] = info.runtime_limits.get('gpu_hours_daily', 12)
        elif info.platform == Platform.PAPERSPACE:
            # TODO: Implement Paperspace-specific quota checking
            quota_info['gpu_hours_used'] = 0  # Placeholder
            quota_info['gpu_hours_remaining'] = info.runtime_limits.get('gpu_hours_daily', 6)

        return quota_info


# Singleton instance
_detector_instance = None


def get_platform_detector() -> PlatformDetector:
    """Get singleton platform detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlatformDetector()
    return _detector_instance
