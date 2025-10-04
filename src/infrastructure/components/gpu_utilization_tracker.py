"""GPU utilization tracking across all platforms.

This module provides comprehensive GPU monitoring and utilization tracking
for Kaggle, Colab, Paperspace, and local environments.
"""

import asyncio
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from .platform_detector import Platform, get_platform_detector

logger = structlog.get_logger(__name__)


class GPUMetricType(Enum):
    """Types of GPU metrics."""
    UTILIZATION = "utilization"           # GPU core utilization %
    MEMORY_USAGE = "memory_usage"         # GPU memory usage %
    MEMORY_USED_MB = "memory_used_mb"     # GPU memory used in MB
    MEMORY_TOTAL_MB = "memory_total_mb"   # Total GPU memory in MB
    TEMPERATURE = "temperature"           # GPU temperature in Celsius
    POWER_USAGE = "power_usage"          # Power consumption in Watts
    CLOCK_SPEED = "clock_speed"          # GPU clock speed in MHz
    MEMORY_CLOCK = "memory_clock"        # Memory clock speed in MHz


@dataclass
class GPUInfo:
    """GPU hardware information."""
    gpu_id: int
    name: str
    driver_version: str
    cuda_version: str
    total_memory_mb: int
    platform: Platform
    architecture: str | None = None
    compute_capability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gpu_id': self.gpu_id,
            'name': self.name,
            'driver_version': self.driver_version,
            'cuda_version': self.cuda_version,
            'total_memory_mb': self.total_memory_mb,
            'platform': self.platform.value,
            'architecture': self.architecture,
            'compute_capability': self.compute_capability
        }


@dataclass
class GPUMetrics:
    """GPU metrics at a point in time."""
    timestamp: datetime
    gpu_id: int
    platform: Platform
    utilization_percent: float
    memory_usage_percent: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_celsius: float | None = None
    power_usage_watts: float | None = None
    clock_speed_mhz: int | None = None
    memory_clock_mhz: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'gpu_id': self.gpu_id,
            'platform': self.platform.value,
            'utilization_percent': self.utilization_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'temperature_celsius': self.temperature_celsius,
            'power_usage_watts': self.power_usage_watts,
            'clock_speed_mhz': self.clock_speed_mhz,
            'memory_clock_mhz': self.memory_clock_mhz
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'GPUMetrics':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            gpu_id=data['gpu_id'],
            platform=Platform(data['platform']),
            utilization_percent=data['utilization_percent'],
            memory_usage_percent=data['memory_usage_percent'],
            memory_used_mb=data['memory_used_mb'],
            memory_total_mb=data['memory_total_mb'],
            temperature_celsius=data.get('temperature_celsius'),
            power_usage_watts=data.get('power_usage_watts'),
            clock_speed_mhz=data.get('clock_speed_mhz'),
            memory_clock_mhz=data.get('memory_clock_mhz')
        )


@dataclass
class UtilizationSummary:
    """Summary of GPU utilization over time period."""
    start_time: datetime
    end_time: datetime
    platform: Platform
    avg_utilization_percent: float
    max_utilization_percent: float
    min_utilization_percent: float
    avg_memory_usage_percent: float
    max_memory_usage_percent: float
    total_gpu_hours: float
    active_gpu_hours: float  # Hours with >10% utilization
    efficiency_score: float  # active_hours / total_hours
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PlatformGPUMonitor:
    """Base class for platform-specific GPU monitoring."""

    def __init__(self, platform: Platform):
        self.platform = platform
        self.logger = structlog.get_logger(f'gpu_monitor_{platform.value}')

    async def detect_gpus(self) -> list[GPUInfo]:
        """Detect available GPUs."""
        raise NotImplementedError

    async def get_gpu_metrics(self, gpu_id: int) -> GPUMetrics | None:
        """Get current GPU metrics."""
        raise NotImplementedError

    async def get_all_gpu_metrics(self) -> list[GPUMetrics]:
        """Get metrics for all GPUs."""
        gpus = await self.detect_gpus()
        metrics = []

        for gpu in gpus:
            gpu_metrics = await self.get_gpu_metrics(gpu.gpu_id)
            if gpu_metrics:
                metrics.append(gpu_metrics)

        return metrics


class NvidiaGPUMonitor(PlatformGPUMonitor):
    """NVIDIA GPU monitoring using nvidia-ml-py or nvidia-smi."""

    def __init__(self, platform: Platform):
        super().__init__(platform)
        self.nvml_available = self._check_nvml_availability()
        self.nvidia_smi_available = self._check_nvidia_smi_availability()

    def _check_nvml_availability(self) -> bool:
        """Check if nvidia-ml-py is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            return False
        except Exception as e:
            self.logger.warning("nvml_init_failed", error=str(e))
            return False

    def _check_nvidia_smi_availability(self) -> bool:
        """Check if nvidia-smi command is available."""
        try:
            result = subprocess.run(['nvidia-smi', '--version'],
                                  capture_output=True, text=True, timeout=10,
                                  encoding='utf-8', errors='replace')
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def detect_gpus(self) -> list[GPUInfo]:
        """Detect NVIDIA GPUs."""
        if self.nvml_available:
            return await self._detect_gpus_nvml()
        elif self.nvidia_smi_available:
            return await self._detect_gpus_nvidia_smi()
        else:
            return []

    async def _detect_gpus_nvml(self) -> list[GPUInfo]:
        """Detect GPUs using NVML."""
        try:
            import pynvml

            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = mem_info.total // (1024 * 1024)

                # Get driver version
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')

                # Get CUDA version
                cuda_version = f"{pynvml.nvmlSystemGetCudaDriverVersion() // 1000}.{(pynvml.nvmlSystemGetCudaDriverVersion() % 1000) // 10}"

                gpu_info = GPUInfo(
                    gpu_id=i,
                    name=name,
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    total_memory_mb=total_memory_mb,
                    platform=self.platform
                )

                gpus.append(gpu_info)

            return gpus

        except Exception as e:
            self.logger.error("nvml_gpu_detection_failed", error=str(e))
            return []

    async def _detect_gpus_nvidia_smi(self) -> list[GPUInfo]:
        """Detect GPUs using nvidia-smi."""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,driver_version,memory.total',
                '--format=csv,noheader,nounits'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                  encoding='utf-8', errors='replace')

            if result.returncode != 0:
                self.logger.error("nvidia_smi_failed", error=result.stderr)
                return []

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_info = GPUInfo(
                            gpu_id=int(parts[0]),
                            name=parts[1],
                            driver_version=parts[2],
                            cuda_version="unknown",  # Not available from this query
                            total_memory_mb=int(parts[3]),
                            platform=self.platform
                        )
                        gpus.append(gpu_info)

            return gpus

        except Exception as e:
            self.logger.error("nvidia_smi_gpu_detection_failed", error=str(e))
            return []

    async def get_gpu_metrics(self, gpu_id: int) -> GPUMetrics | None:
        """Get GPU metrics."""
        if self.nvml_available:
            return await self._get_gpu_metrics_nvml(gpu_id)
        elif self.nvidia_smi_available:
            return await self._get_gpu_metrics_nvidia_smi(gpu_id)
        else:
            return None

    async def _get_gpu_metrics_nvml(self, gpu_id: int) -> GPUMetrics | None:
        """Get GPU metrics using NVML."""
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # Get utilization
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util_rates.gpu

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_mb = mem_info.total // (1024 * 1024)
            memory_used_mb = mem_info.used // (1024 * 1024)
            memory_usage_percent = (mem_info.used / mem_info.total) * 100

            # Get temperature (optional)
            temperature = None
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

            # Get power usage (optional)
            power_usage = None
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except Exception:
                pass

            # Get clock speeds (optional)
            clock_speed = None
            memory_clock = None
            try:
                clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass

            return GPUMetrics(
                timestamp=datetime.now(),
                gpu_id=gpu_id,
                platform=self.platform,
                utilization_percent=float(gpu_utilization),
                memory_usage_percent=memory_usage_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                temperature_celsius=temperature,
                power_usage_watts=power_usage,
                clock_speed_mhz=clock_speed,
                memory_clock_mhz=memory_clock
            )

        except Exception as e:
            self.logger.error("nvml_metrics_failed", gpu_id=gpu_id, error=str(e))
            return None

    async def _get_gpu_metrics_nvidia_smi(self, gpu_id: int) -> GPUMetrics | None:
        """Get GPU metrics using nvidia-smi."""
        try:
            cmd = [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory',
                '--format=csv,noheader,nounits'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                  encoding='utf-8', errors='replace')

            if result.returncode != 0:
                self.logger.error("nvidia_smi_metrics_failed", gpu_id=gpu_id, error=result.stderr)
                return None

            line = result.stdout.strip()
            parts = [p.strip() for p in line.split(',')]

            if len(parts) >= 3:
                # Parse required fields
                gpu_utilization = float(parts[0]) if parts[0].replace('.', '').isdigit() else 0.0
                memory_used_mb = int(parts[1]) if parts[1].isdigit() else 0
                memory_total_mb = int(parts[2]) if parts[2].isdigit() else 0

                memory_usage_percent = (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0.0

                # Parse optional fields
                temperature = None
                if len(parts) > 3 and parts[3].replace('.', '').isdigit():
                    temperature = float(parts[3])

                power_usage = None
                if len(parts) > 4 and parts[4].replace('.', '').isdigit():
                    power_usage = float(parts[4])

                clock_speed = None
                if len(parts) > 5 and parts[5].isdigit():
                    clock_speed = int(parts[5])

                memory_clock = None
                if len(parts) > 6 and parts[6].isdigit():
                    memory_clock = int(parts[6])

                return GPUMetrics(
                    timestamp=datetime.now(),
                    gpu_id=gpu_id,
                    platform=self.platform,
                    utilization_percent=gpu_utilization,
                    memory_usage_percent=memory_usage_percent,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    temperature_celsius=temperature,
                    power_usage_watts=power_usage,
                    clock_speed_mhz=clock_speed,
                    memory_clock_mhz=memory_clock
                )

            return None

        except Exception as e:
            self.logger.error("nvidia_smi_metrics_parsing_failed", gpu_id=gpu_id, error=str(e))
            return None


class ColabGPUMonitor(NvidiaGPUMonitor):
    """Google Colab GPU monitoring."""

    def __init__(self):
        super().__init__(Platform.COLAB)

    async def detect_gpus(self) -> list[GPUInfo]:
        """Detect GPUs in Colab environment."""
        try:
            # Check if we're in Colab
            try:
                import google.colab  # noqa: F401
            except ImportError:
                return []

            # Use parent NVIDIA detection
            gpus = await super().detect_gpus()

            # Colab-specific adjustments
            for gpu in gpus:
                if "Tesla" in gpu.name or "T4" in gpu.name or "P100" in gpu.name:
                    gpu.architecture = "Colab GPU"

            return gpus

        except Exception as e:
            self.logger.error("colab_gpu_detection_failed", error=str(e))
            return []


class KaggleGPUMonitor(NvidiaGPUMonitor):
    """Kaggle GPU monitoring."""

    def __init__(self):
        super().__init__(Platform.KAGGLE)

    async def detect_gpus(self) -> list[GPUInfo]:
        """Detect GPUs in Kaggle environment."""
        try:
            # Check if we're in Kaggle
            import os
            if 'KAGGLE_KERNEL_RUN_TYPE' not in os.environ:
                return []

            # Use parent NVIDIA detection
            gpus = await super().detect_gpus()

            # Kaggle-specific adjustments
            for gpu in gpus:
                gpu.architecture = "Kaggle GPU"

            return gpus

        except Exception as e:
            self.logger.error("kaggle_gpu_detection_failed", error=str(e))
            return []


class PaperspaceGPUMonitor(NvidiaGPUMonitor):
    """Paperspace GPU monitoring."""

    def __init__(self):
        super().__init__(Platform.PAPERSPACE)

    async def detect_gpus(self) -> list[GPUInfo]:
        """Detect GPUs in Paperspace environment."""
        try:
            # Use parent NVIDIA detection
            gpus = await super().detect_gpus()

            # Paperspace-specific adjustments
            for gpu in gpus:
                gpu.architecture = "Paperspace GPU"

            return gpus

        except Exception as e:
            self.logger.error("paperspace_gpu_detection_failed", error=str(e))
            return []


class LocalGPUMonitor(NvidiaGPUMonitor):
    """Local environment GPU monitoring."""

    def __init__(self):
        super().__init__(Platform.LOCAL)


class GPUUtilizationTracker:
    """Main GPU utilization tracking system."""

    def __init__(self,
                 collection_interval_seconds: int = 60,
                 retention_hours: int = 168,  # 1 week
                 storage_dir: Path | None = None):
        """Initialize GPU utilization tracker.

        Args:
            collection_interval_seconds: How often to collect metrics
            retention_hours: How long to retain metrics
            storage_dir: Directory to store metrics
        """
        self.collection_interval_seconds = collection_interval_seconds
        self.retention_hours = retention_hours
        self.storage_dir = storage_dir or Path.home() / ".arc-gpu-metrics"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Platform detection and monitoring
        self.platform_detector = get_platform_detector()
        self.monitors = self._initialize_monitors()
        self.current_monitor: PlatformGPUMonitor | None = None

        # State
        self.gpu_info: list[GPUInfo] = []
        self.metrics_history: list[GPUMetrics] = []
        self.tracking = False
        self.tracking_task: asyncio.Task | None = None

        # Callbacks
        self.callbacks: dict[str, list[Callable]] = {
            'metrics_collected': [],
            'gpu_detected': [],
            'utilization_threshold': []
        }

        self.logger = structlog.get_logger('gpu_utilization_tracker')

        # Load historical data
        self._load_metrics_history()

    def _initialize_monitors(self) -> dict[Platform, PlatformGPUMonitor]:
        """Initialize platform-specific GPU monitors."""
        monitors = {
            Platform.KAGGLE: KaggleGPUMonitor(),
            Platform.COLAB: ColabGPUMonitor(),
            Platform.PAPERSPACE: PaperspaceGPUMonitor(),
            Platform.LOCAL: LocalGPUMonitor()
        }
        return monitors

    async def start_tracking(self) -> bool:
        """Start GPU utilization tracking.

        Returns:
            True if started successfully
        """
        if self.tracking:
            self.logger.warning("gpu_tracking_already_running")
            return False

        try:
            # Detect current platform
            platform_info = self.platform_detector.detect_platform()
            current_platform = platform_info.platform

            # Get appropriate monitor
            self.current_monitor = self.monitors.get(current_platform)
            if not self.current_monitor:
                self.logger.error("no_gpu_monitor_for_platform", platform=current_platform.value)
                return False

            # Detect GPUs
            self.gpu_info = await self.current_monitor.detect_gpus()

            if not self.gpu_info:
                self.logger.warning("no_gpus_detected", platform=current_platform.value)
                # Continue tracking anyway in case GPUs become available
            else:
                self.logger.info("gpus_detected",
                               platform=current_platform.value,
                               gpu_count=len(self.gpu_info),
                               gpus=[gpu.name for gpu in self.gpu_info])

                # Trigger callbacks
                for gpu in self.gpu_info:
                    self._trigger_callbacks('gpu_detected', gpu)

            # Start tracking loop
            self.tracking = True
            self.tracking_task = asyncio.create_task(self._tracking_loop())

            self.logger.info("gpu_tracking_started",
                           platform=current_platform.value,
                           interval=self.collection_interval_seconds)

            return True

        except Exception as e:
            self.logger.error("gpu_tracking_start_failed", error=str(e))
            return False

    async def stop_tracking(self) -> bool:
        """Stop GPU utilization tracking.

        Returns:
            True if stopped successfully
        """
        if not self.tracking:
            return True

        try:
            self.tracking = False

            if self.tracking_task:
                self.tracking_task.cancel()
                try:
                    await self.tracking_task
                except asyncio.CancelledError:
                    pass

            # Save final metrics
            self._save_metrics_history()

            self.logger.info("gpu_tracking_stopped")
            return True

        except Exception as e:
            self.logger.error("gpu_tracking_stop_failed", error=str(e))
            return False

    async def _tracking_loop(self):
        """Main tracking loop."""
        while self.tracking:
            try:
                # Collect metrics
                await self._collect_metrics()

                # Clean up old metrics
                self._cleanup_old_metrics()

                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 collections
                    self._save_metrics_history()

                # Wait for next collection
                await asyncio.sleep(self.collection_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("tracking_loop_error", error=str(e))
                await asyncio.sleep(self.collection_interval_seconds)

    async def _collect_metrics(self):
        """Collect GPU metrics."""
        if not self.current_monitor:
            return

        try:
            # Get metrics for all GPUs
            gpu_metrics_list = await self.current_monitor.get_all_gpu_metrics()

            for metrics in gpu_metrics_list:
                # Add to history
                self.metrics_history.append(metrics)

                # Trigger callbacks
                self._trigger_callbacks('metrics_collected', metrics)

                # Check utilization thresholds
                self._check_utilization_thresholds(metrics)

            if gpu_metrics_list:
                self.logger.debug("gpu_metrics_collected",
                                gpu_count=len(gpu_metrics_list),
                                avg_utilization=sum(m.utilization_percent for m in gpu_metrics_list) / len(gpu_metrics_list))

        except Exception as e:
            self.logger.error("metrics_collection_failed", error=str(e))

    def _check_utilization_thresholds(self, metrics: GPUMetrics):
        """Check for utilization threshold events."""
        # High utilization threshold (>90%)
        if metrics.utilization_percent > 90:
            self._trigger_callbacks('utilization_threshold', {
                'type': 'high_utilization',
                'metrics': metrics,
                'threshold': 90
            })

        # Low utilization threshold (<5%)
        elif metrics.utilization_percent < 5:
            self._trigger_callbacks('utilization_threshold', {
                'type': 'low_utilization',
                'metrics': metrics,
                'threshold': 5
            })

        # High memory usage (>95%)
        if metrics.memory_usage_percent > 95:
            self._trigger_callbacks('utilization_threshold', {
                'type': 'high_memory',
                'metrics': metrics,
                'threshold': 95
            })

    def _cleanup_old_metrics(self):
        """Clean up old metrics."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_current_utilization(self) -> dict[str, Any]:
        """Get current GPU utilization summary.

        Returns:
            Current utilization information
        """
        if not self.metrics_history:
            return {'status': 'no_data'}

        # Get most recent metrics for each GPU
        latest_metrics = {}
        for metrics in reversed(self.metrics_history):
            if metrics.gpu_id not in latest_metrics:
                latest_metrics[metrics.gpu_id] = metrics

        if not latest_metrics:
            return {'status': 'no_recent_data'}

        # Calculate summary
        total_gpus = len(latest_metrics)
        avg_utilization = sum(m.utilization_percent for m in latest_metrics.values()) / total_gpus
        avg_memory_usage = sum(m.memory_usage_percent for m in latest_metrics.values()) / total_gpus

        return {
            'timestamp': datetime.now().isoformat(),
            'platform': self.current_monitor.platform.value if self.current_monitor else 'unknown',
            'total_gpus': total_gpus,
            'avg_utilization_percent': avg_utilization,
            'avg_memory_usage_percent': avg_memory_usage,
            'max_utilization_percent': max(m.utilization_percent for m in latest_metrics.values()),
            'min_utilization_percent': min(m.utilization_percent for m in latest_metrics.values()),
            'gpus': [
                {
                    'gpu_id': metrics.gpu_id,
                    'utilization_percent': metrics.utilization_percent,
                    'memory_usage_percent': metrics.memory_usage_percent,
                    'memory_used_mb': metrics.memory_used_mb,
                    'memory_total_mb': metrics.memory_total_mb,
                    'temperature_celsius': metrics.temperature_celsius
                }
                for metrics in latest_metrics.values()
            ]
        }

    def get_utilization_summary(self, hours: int = 1) -> UtilizationSummary | None:
        """Get utilization summary for time period.

        Args:
            hours: Number of hours to summarize

        Returns:
            Utilization summary or None if no data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Filter metrics for time period
        period_metrics = [
            m for m in self.metrics_history
            if start_time <= m.timestamp <= end_time
        ]

        if not period_metrics:
            return None

        platform = period_metrics[0].platform

        # Calculate statistics
        utilizations = [m.utilization_percent for m in period_metrics]
        memory_usages = [m.memory_usage_percent for m in period_metrics]

        avg_utilization = sum(utilizations) / len(utilizations)
        max_utilization = max(utilizations)
        min_utilization = min(utilizations)

        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        max_memory_usage = max(memory_usages)

        # Calculate active hours (>10% utilization)
        active_samples = len([u for u in utilizations if u > 10])
        total_hours = (end_time - start_time).total_seconds() / 3600
        active_hours = (active_samples / len(utilizations)) * total_hours
        efficiency_score = active_hours / total_hours if total_hours > 0 else 0

        return UtilizationSummary(
            start_time=start_time,
            end_time=end_time,
            platform=platform,
            avg_utilization_percent=avg_utilization,
            max_utilization_percent=max_utilization,
            min_utilization_percent=min_utilization,
            avg_memory_usage_percent=avg_memory_usage,
            max_memory_usage_percent=max_memory_usage,
            total_gpu_hours=total_hours * len({m.gpu_id for m in period_metrics}),
            active_gpu_hours=active_hours * len({m.gpu_id for m in period_metrics}),
            efficiency_score=efficiency_score,
            sample_count=len(period_metrics)
        )

    def get_metrics_history(self, hours: int = 24) -> list[GPUMetrics]:
        """Get metrics history for specified time period.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of metrics within the time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_gpu_info(self) -> list[GPUInfo]:
        """Get detected GPU information.

        Returns:
            List of detected GPUs
        """
        return self.gpu_info.copy()

    def calculate_efficiency_score(self, utilization_summary: UtilizationSummary) -> float:
        """Calculate GPU efficiency score (0-100).

        Args:
            utilization_summary: Utilization summary to score

        Returns:
            Efficiency score between 0 and 100
        """
        base_score = utilization_summary.avg_utilization_percent

        # Bonus for consistent utilization
        utilization_range = utilization_summary.max_utilization_percent - utilization_summary.min_utilization_percent
        consistency_bonus = max(0, 10 - (utilization_range / 10))  # Up to 10 points

        # Bonus for high active time
        active_time_bonus = utilization_summary.efficiency_score * 20  # Up to 20 points

        # Memory efficiency factor
        memory_efficiency = min(utilization_summary.avg_memory_usage_percent / 80, 1.0)  # Optimal around 80%

        total_score = (base_score + consistency_bonus + active_time_bonus) * memory_efficiency

        return min(100, max(0, total_score))

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for tracking events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for an event.

        Args:
            event_type: Event type
            data: Event data
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error("callback_failed", event_type=event_type, error=str(e))

    def _save_metrics_history(self):
        """Save metrics history to disk."""
        try:
            metrics_file = self.storage_dir / "gpu_metrics_history.json"

            # Convert metrics to serializable format
            serializable_metrics = [metrics.to_dict() for metrics in self.metrics_history[-1000:]]  # Keep last 1000

            # Also save GPU info
            gpu_info_data = [gpu.to_dict() for gpu in self.gpu_info]

            data = {
                'gpu_info': gpu_info_data,
                'metrics': serializable_metrics,
                'last_saved': datetime.now().isoformat()
            }

            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.debug("gpu_metrics_saved", count=len(serializable_metrics))

        except Exception as e:
            self.logger.error("gpu_metrics_save_failed", error=str(e))

    def _load_metrics_history(self):
        """Load metrics history from disk."""
        try:
            metrics_file = self.storage_dir / "gpu_metrics_history.json"

            if not metrics_file.exists():
                return

            with open(metrics_file) as f:
                data = json.load(f)

            # Load GPU info
            gpu_info_data = data.get('gpu_info', [])
            self.gpu_info = []
            for gpu_data in gpu_info_data:
                try:
                    gpu_info = GPUInfo(
                        gpu_id=gpu_data['gpu_id'],
                        name=gpu_data['name'],
                        driver_version=gpu_data['driver_version'],
                        cuda_version=gpu_data['cuda_version'],
                        total_memory_mb=gpu_data['total_memory_mb'],
                        platform=Platform(gpu_data['platform']),
                        architecture=gpu_data.get('architecture'),
                        compute_capability=gpu_data.get('compute_capability')
                    )
                    self.gpu_info.append(gpu_info)
                except Exception as e:
                    self.logger.error("gpu_info_load_failed", error=str(e))

            # Load metrics
            metrics_data = data.get('metrics', [])
            self.metrics_history = []
            for metric_data in metrics_data:
                try:
                    metrics = GPUMetrics.from_dict(metric_data)
                    self.metrics_history.append(metrics)
                except Exception as e:
                    self.logger.error("gpu_metrics_load_failed", error=str(e))

            # Clean up old data
            self._cleanup_old_metrics()

            self.logger.info("gpu_metrics_loaded",
                           gpu_count=len(self.gpu_info),
                           metrics_count=len(self.metrics_history))

        except Exception as e:
            self.logger.error("gpu_metrics_load_failed", error=str(e))

    def get_tracking_statistics(self) -> dict[str, Any]:
        """Get tracking system statistics.

        Returns:
            Dictionary with tracking statistics
        """
        current_utilization = self.get_current_utilization()

        stats = {
            'tracking_active': self.tracking,
            'platform': self.current_monitor.platform.value if self.current_monitor else 'none',
            'detected_gpus': len(self.gpu_info),
            'metrics_collected': len(self.metrics_history),
            'collection_interval_seconds': self.collection_interval_seconds,
            'retention_hours': self.retention_hours,
            'current_utilization': current_utilization
        }

        # Add recent summaries
        for hours in [1, 6, 24]:
            summary = self.get_utilization_summary(hours)
            if summary:
                stats[f'utilization_summary_{hours}h'] = summary.to_dict()

        return stats


# Singleton instance
_gpu_utilization_tracker = None


def get_gpu_utilization_tracker() -> GPUUtilizationTracker:
    """Get singleton GPU utilization tracker instance."""
    global _gpu_utilization_tracker
    if _gpu_utilization_tracker is None:
        _gpu_utilization_tracker = GPUUtilizationTracker()
    return _gpu_utilization_tracker
