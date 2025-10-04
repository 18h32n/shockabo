"""Advanced GPU monitoring and utilization optimization system.

This module provides comprehensive GPU monitoring, analytics, and intelligent
scheduling recommendations to achieve 95%+ utilization across all platforms.
Integrates with existing platform detection and monitoring systems.
"""

import asyncio
import json
import statistics
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import structlog

from .gpu_utilization_tracker import get_gpu_utilization_tracker
from .platform_detector import Platform, get_platform_detector

logger = structlog.get_logger(__name__)


class UtilizationPattern(Enum):
    """GPU utilization patterns."""
    IDLE = "idle"                    # <5% utilization
    LOW = "low"                      # 5-30% utilization
    MODERATE = "moderate"            # 30-70% utilization
    HIGH = "high"                    # 70-90% utilization
    OPTIMAL = "optimal"              # 90-98% utilization
    SATURATED = "saturated"          # >98% utilization


class OptimizationAction(Enum):
    """Recommended optimization actions."""
    INCREASE_BATCH_SIZE = "increase_batch_size"
    DECREASE_BATCH_SIZE = "decrease_batch_size"
    ENABLE_MIXED_PRECISION = "enable_mixed_precision"
    OPTIMIZE_DATA_LOADING = "optimize_data_loading"
    REDUCE_MODEL_COMPLEXITY = "reduce_model_complexity"
    INCREASE_PARALLELIZATION = "increase_parallelization"
    SCHEDULE_ADDITIONAL_TASKS = "schedule_additional_tasks"
    CONSOLIDATE_WORKLOADS = "consolidate_workloads"
    MIGRATE_TO_DIFFERENT_GPU = "migrate_to_different_gpu"
    ADD_GRADIENT_ACCUMULATION = "add_gradient_accumulation"


@dataclass
class UtilizationAlert:
    """Utilization alert information."""
    timestamp: datetime
    gpu_id: int
    alert_type: str
    severity: str  # "info", "warning", "critical"
    message: str
    current_utilization: float
    threshold: float
    suggested_actions: list[OptimizationAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for workload optimization."""
    workload_id: str
    gpu_utilization_target: float
    memory_utilization_target: float
    batch_size_optimal: int | None = None
    mixed_precision_effective: bool | None = None
    data_loading_optimal: bool | None = None
    parallelization_level: int | None = None
    historical_performance: dict[str, float] = field(default_factory=dict)


@dataclass
class SchedulingRecommendation:
    """Intelligent scheduling recommendation."""
    timestamp: datetime
    priority: str  # "high", "medium", "low"
    action: OptimizationAction
    description: str
    expected_improvement: float  # Expected % improvement in utilization
    implementation_difficulty: str  # "easy", "medium", "hard"
    estimated_time_to_implement: int  # minutes
    prerequisites: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkloadCharacteristics(NamedTuple):
    """Workload characteristics for optimization."""
    compute_intensity: float  # 0-1, higher = more compute intensive
    memory_intensity: float   # 0-1, higher = more memory intensive
    io_intensity: float      # 0-1, higher = more I/O intensive
    parallelizability: float # 0-1, higher = more parallelizable
    batch_sensitivity: float # 0-1, higher = more sensitive to batch size


@dataclass
class GPUAnalytics:
    """Advanced GPU analytics and insights."""
    gpu_id: int
    platform: Platform
    analysis_period_hours: float

    # Utilization statistics
    avg_utilization: float
    median_utilization: float
    p95_utilization: float
    p99_utilization: float
    utilization_variance: float

    # Memory statistics
    avg_memory_usage: float
    peak_memory_usage: float
    memory_fragmentation_score: float

    # Performance metrics
    efficiency_score: float  # Overall efficiency (0-100)
    idle_time_percentage: float
    underutilization_time_percentage: float  # <50% utilization
    optimal_time_percentage: float  # 90-98% utilization

    # Patterns and trends
    dominant_pattern: UtilizationPattern
    pattern_distribution: dict[UtilizationPattern, float]
    trend_direction: str  # "improving", "stable", "declining"
    seasonal_patterns: dict[str, float] = field(default_factory=dict)

    # Recommendations
    top_recommendations: list[SchedulingRecommendation] = field(default_factory=list)
    estimated_improvement_potential: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GPUMonitor:
    """Advanced GPU monitoring and optimization system."""

    def __init__(self,
                 monitoring_interval: int = 30,
                 analytics_window_hours: int = 24,
                 optimization_check_interval: int = 300,  # 5 minutes
                 storage_dir: Path | None = None):
        """Initialize GPU monitor.

        Args:
            monitoring_interval: Seconds between monitoring checks
            analytics_window_hours: Hours of data for analytics
            optimization_check_interval: Seconds between optimization checks
            storage_dir: Directory for persistent storage
        """
        self.monitoring_interval = monitoring_interval
        self.analytics_window_hours = analytics_window_hours
        self.optimization_check_interval = optimization_check_interval
        self.storage_dir = storage_dir or Path.home() / ".arc-gpu-monitor"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.platform_detector = get_platform_detector()
        self.gpu_tracker = get_gpu_utilization_tracker()

        # State
        self.monitoring = False
        self.monitoring_task: asyncio.Task | None = None
        self.optimization_task: asyncio.Task | None = None

        # Data structures
        self.utilization_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=2880))  # 24h at 30s intervals
        self.alerts: deque = deque(maxlen=1000)
        self.performance_profiles: dict[str, PerformanceProfile] = {}
        self.workload_characteristics: dict[str, WorkloadCharacteristics] = {}
        self.analytics_cache: dict[int, GPUAnalytics] = {}

        # Thresholds and configurations
        self.utilization_thresholds = {
            'idle': 5.0,
            'low': 30.0,
            'moderate': 70.0,
            'high': 90.0,
            'optimal_min': 90.0,
            'optimal_max': 98.0,
            'saturated': 98.0
        }

        self.alert_thresholds = {
            'idle_duration_minutes': 10,
            'low_utilization_duration_minutes': 30,
            'memory_pressure_threshold': 95.0,
            'temperature_warning': 85.0,
            'temperature_critical': 90.0
        }

        # Callbacks
        self.callbacks: dict[str, list[Callable]] = {
            'alert_generated': [],
            'recommendation_created': [],
            'pattern_detected': [],
            'optimization_applied': []
        }

        self.logger = structlog.get_logger('gpu_monitor')

        # Load historical data
        self._load_historical_data()

    async def start_monitoring(self) -> bool:
        """Start GPU monitoring and optimization.

        Returns:
            True if started successfully
        """
        if self.monitoring:
            self.logger.warning("gpu_monitoring_already_active")
            return False

        try:
            # Start GPU tracker if not already running
            if not self.gpu_tracker.tracking:
                success = await self.gpu_tracker.start_tracking()
                if not success:
                    self.logger.error("failed_to_start_gpu_tracker")
                    return False

            self.monitoring = True

            # Start monitoring and optimization tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())

            self.logger.info("gpu_monitoring_started",
                           interval=self.monitoring_interval,
                           optimization_interval=self.optimization_check_interval)

            return True

        except Exception as e:
            self.logger.error("gpu_monitoring_start_failed", error=str(e))
            return False

    async def stop_monitoring(self) -> bool:
        """Stop GPU monitoring.

        Returns:
            True if stopped successfully
        """
        if not self.monitoring:
            return True

        try:
            self.monitoring = False

            # Cancel tasks
            for task in [self.monitoring_task, self.optimization_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Save data
            self._save_historical_data()

            self.logger.info("gpu_monitoring_stopped")
            return True

        except Exception as e:
            self.logger.error("gpu_monitoring_stop_failed", error=str(e))
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_and_analyze_metrics()
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(self.monitoring_interval)

    async def _optimization_loop(self):
        """Optimization and recommendation loop."""
        while self.monitoring:
            try:
                await self._generate_optimizations()
                await asyncio.sleep(self.optimization_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("optimization_loop_error", error=str(e))
                await asyncio.sleep(self.optimization_check_interval)

    async def _collect_and_analyze_metrics(self):
        """Collect GPU metrics and perform real-time analysis."""
        try:
            # Get current utilization
            current_utilization = self.gpu_tracker.get_current_utilization()

            if current_utilization.get('status') not in ['no_data', 'no_recent_data']:
                # Process each GPU
                for gpu_data in current_utilization.get('gpus', []):
                    gpu_id = gpu_data['gpu_id']
                    utilization = gpu_data['utilization_percent']
                    memory_usage = gpu_data['memory_usage_percent']
                    temperature = gpu_data.get('temperature_celsius')

                    # Store in history
                    self.utilization_history[gpu_id].append({
                        'timestamp': datetime.now(),
                        'utilization': utilization,
                        'memory_usage': memory_usage,
                        'temperature': temperature
                    })

                    # Check for alerts
                    await self._check_utilization_alerts(gpu_id, gpu_data)

                    # Update analytics cache if needed
                    if gpu_id not in self.analytics_cache or \
                       (datetime.now() - self.analytics_cache[gpu_id].timestamp).total_seconds() > 3600:
                        self.analytics_cache[gpu_id] = await self._generate_analytics(gpu_id)

        except Exception as e:
            self.logger.error("metrics_collection_failed", error=str(e))

    async def _check_utilization_alerts(self, gpu_id: int, gpu_data: dict[str, Any]):
        """Check for utilization alerts and patterns."""
        utilization = gpu_data['utilization_percent']
        memory_usage = gpu_data['memory_usage_percent']
        temperature = gpu_data.get('temperature_celsius')

        alerts = []

        # Check utilization patterns
        if utilization < self.utilization_thresholds['idle']:
            if self._is_sustained_pattern(gpu_id, 'idle', self.alert_thresholds['idle_duration_minutes']):
                alerts.append(UtilizationAlert(
                    timestamp=datetime.now(),
                    gpu_id=gpu_id,
                    alert_type="sustained_idle",
                    severity="warning",
                    message=f"GPU {gpu_id} has been idle (<{self.utilization_thresholds['idle']}%) for {self.alert_thresholds['idle_duration_minutes']} minutes",
                    current_utilization=utilization,
                    threshold=self.utilization_thresholds['idle'],
                    suggested_actions=[
                        OptimizationAction.SCHEDULE_ADDITIONAL_TASKS,
                        OptimizationAction.CONSOLIDATE_WORKLOADS
                    ]
                ))

        elif utilization < self.utilization_thresholds['moderate']:
            if self._is_sustained_pattern(gpu_id, 'low', self.alert_thresholds['low_utilization_duration_minutes']):
                alerts.append(UtilizationAlert(
                    timestamp=datetime.now(),
                    gpu_id=gpu_id,
                    alert_type="sustained_low_utilization",
                    severity="info",
                    message=f"GPU {gpu_id} has low utilization (<{self.utilization_thresholds['moderate']}%) for {self.alert_thresholds['low_utilization_duration_minutes']} minutes",
                    current_utilization=utilization,
                    threshold=self.utilization_thresholds['moderate'],
                    suggested_actions=[
                        OptimizationAction.INCREASE_BATCH_SIZE,
                        OptimizationAction.INCREASE_PARALLELIZATION,
                        OptimizationAction.OPTIMIZE_DATA_LOADING
                    ]
                ))

        # Check memory pressure
        if memory_usage > self.alert_thresholds['memory_pressure_threshold']:
            alerts.append(UtilizationAlert(
                timestamp=datetime.now(),
                gpu_id=gpu_id,
                alert_type="high_memory_pressure",
                severity="warning",
                message=f"GPU {gpu_id} memory usage is {memory_usage:.1f}%",
                current_utilization=utilization,
                threshold=self.alert_thresholds['memory_pressure_threshold'],
                suggested_actions=[
                    OptimizationAction.DECREASE_BATCH_SIZE,
                    OptimizationAction.ENABLE_MIXED_PRECISION,
                    OptimizationAction.REDUCE_MODEL_COMPLEXITY
                ],
                metadata={'memory_usage': memory_usage}
            ))

        # Check temperature
        if temperature:
            if temperature > self.alert_thresholds['temperature_critical']:
                alerts.append(UtilizationAlert(
                    timestamp=datetime.now(),
                    gpu_id=gpu_id,
                    alert_type="critical_temperature",
                    severity="critical",
                    message=f"GPU {gpu_id} temperature is critically high: {temperature}°C",
                    current_utilization=utilization,
                    threshold=self.alert_thresholds['temperature_critical'],
                    suggested_actions=[
                        OptimizationAction.REDUCE_MODEL_COMPLEXITY,
                        OptimizationAction.DECREASE_BATCH_SIZE
                    ],
                    metadata={'temperature': temperature}
                ))
            elif temperature > self.alert_thresholds['temperature_warning']:
                alerts.append(UtilizationAlert(
                    timestamp=datetime.now(),
                    gpu_id=gpu_id,
                    alert_type="high_temperature",
                    severity="warning",
                    message=f"GPU {gpu_id} temperature is high: {temperature}°C",
                    current_utilization=utilization,
                    threshold=self.alert_thresholds['temperature_warning'],
                    metadata={'temperature': temperature}
                ))

        # Store alerts and trigger callbacks
        for alert in alerts:
            self.alerts.append(alert)
            self._trigger_callbacks('alert_generated', alert)

    def _is_sustained_pattern(self, gpu_id: int, pattern: str, duration_minutes: int) -> bool:
        """Check if a utilization pattern has been sustained."""
        if gpu_id not in self.utilization_history:
            return False

        history = list(self.utilization_history[gpu_id])
        if len(history) < 2:
            return False

        # Check last N minutes
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_history = [h for h in history if h['timestamp'] >= cutoff_time]

        if len(recent_history) < max(1, duration_minutes // (self.monitoring_interval // 60)):
            return False

        # Check if pattern persists
        pattern_count = 0
        for entry in recent_history:
            utilization = entry['utilization']
            if pattern == 'idle' and utilization < self.utilization_thresholds['idle']:
                pattern_count += 1
            elif pattern == 'low' and utilization < self.utilization_thresholds['moderate']:
                pattern_count += 1

        return pattern_count / len(recent_history) > 0.8  # 80% of samples match pattern

    async def _generate_analytics(self, gpu_id: int) -> GPUAnalytics:
        """Generate comprehensive analytics for a GPU."""
        if gpu_id not in self.utilization_history:
            return self._create_empty_analytics(gpu_id)

        history = list(self.utilization_history[gpu_id])
        if not history:
            return self._create_empty_analytics(gpu_id)

        # Filter to analysis window
        cutoff_time = datetime.now() - timedelta(hours=self.analytics_window_hours)
        recent_history = [h for h in history if h['timestamp'] >= cutoff_time]

        if not recent_history:
            return self._create_empty_analytics(gpu_id)

        # Extract metrics
        utilizations = [h['utilization'] for h in recent_history]
        memory_usages = [h['memory_usage'] for h in recent_history if h['memory_usage'] is not None]

        # Calculate statistics
        avg_utilization = statistics.mean(utilizations)
        median_utilization = statistics.median(utilizations)

        # Calculate percentiles
        utilizations_sorted = sorted(utilizations)
        p95_utilization = utilizations_sorted[int(0.95 * len(utilizations_sorted))]
        p99_utilization = utilizations_sorted[int(0.99 * len(utilizations_sorted))]

        utilization_variance = statistics.variance(utilizations) if len(utilizations) > 1 else 0

        # Memory statistics
        avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0
        peak_memory_usage = max(memory_usages) if memory_usages else 0

        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(utilizations, memory_usages)

        # Calculate time percentages
        total_samples = len(recent_history)
        idle_samples = len([u for u in utilizations if u < self.utilization_thresholds['idle']])
        underutil_samples = len([u for u in utilizations if u < 50])
        optimal_samples = len([u for u in utilizations if self.utilization_thresholds['optimal_min'] <= u <= self.utilization_thresholds['optimal_max']])

        idle_time_percentage = (idle_samples / total_samples) * 100
        underutilization_time_percentage = (underutil_samples / total_samples) * 100
        optimal_time_percentage = (optimal_samples / total_samples) * 100

        # Determine dominant pattern
        pattern_counts = {
            UtilizationPattern.IDLE: len([u for u in utilizations if u < self.utilization_thresholds['idle']]),
            UtilizationPattern.LOW: len([u for u in utilizations if self.utilization_thresholds['idle'] <= u < self.utilization_thresholds['low']]),
            UtilizationPattern.MODERATE: len([u for u in utilizations if self.utilization_thresholds['low'] <= u < self.utilization_thresholds['moderate']]),
            UtilizationPattern.HIGH: len([u for u in utilizations if self.utilization_thresholds['moderate'] <= u < self.utilization_thresholds['high']]),
            UtilizationPattern.OPTIMAL: len([u for u in utilizations if self.utilization_thresholds['optimal_min'] <= u <= self.utilization_thresholds['optimal_max']]),
            UtilizationPattern.SATURATED: len([u for u in utilizations if u > self.utilization_thresholds['saturated']])
        }

        dominant_pattern = max(pattern_counts, key=pattern_counts.get)
        pattern_distribution = {pattern: count / total_samples for pattern, count in pattern_counts.items()}

        # Determine trend
        if len(utilizations) >= 10:
            first_half = utilizations[:len(utilizations)//2]
            second_half = utilizations[len(utilizations)//2:]
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg + 5:
                trend_direction = "improving"
            elif second_avg < first_avg - 5:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

        # Generate recommendations
        recommendations = await self._generate_recommendations_for_gpu(gpu_id, utilizations, memory_usages)

        # Calculate improvement potential
        current_avg = avg_utilization
        target_utilization = 95.0
        improvement_potential = max(0, target_utilization - current_avg)

        platform_info = self.platform_detector.detect_platform()

        return GPUAnalytics(
            gpu_id=gpu_id,
            platform=platform_info.platform,
            analysis_period_hours=self.analytics_window_hours,
            avg_utilization=avg_utilization,
            median_utilization=median_utilization,
            p95_utilization=p95_utilization,
            p99_utilization=p99_utilization,
            utilization_variance=utilization_variance,
            avg_memory_usage=avg_memory_usage,
            peak_memory_usage=peak_memory_usage,
            memory_fragmentation_score=self._calculate_memory_fragmentation_score(memory_usages),
            efficiency_score=efficiency_score,
            idle_time_percentage=idle_time_percentage,
            underutilization_time_percentage=underutilization_time_percentage,
            optimal_time_percentage=optimal_time_percentage,
            dominant_pattern=dominant_pattern,
            pattern_distribution=pattern_distribution,
            trend_direction=trend_direction,
            top_recommendations=recommendations,
            estimated_improvement_potential=improvement_potential
        )

    def _create_empty_analytics(self, gpu_id: int) -> GPUAnalytics:
        """Create empty analytics for GPU with no data."""
        platform_info = self.platform_detector.detect_platform()

        return GPUAnalytics(
            gpu_id=gpu_id,
            platform=platform_info.platform,
            analysis_period_hours=0,
            avg_utilization=0,
            median_utilization=0,
            p95_utilization=0,
            p99_utilization=0,
            utilization_variance=0,
            avg_memory_usage=0,
            peak_memory_usage=0,
            memory_fragmentation_score=0,
            efficiency_score=0,
            idle_time_percentage=100,
            underutilization_time_percentage=100,
            optimal_time_percentage=0,
            dominant_pattern=UtilizationPattern.IDLE,
            pattern_distribution=dict.fromkeys(UtilizationPattern, 0),
            trend_direction="stable"
        )

    def _calculate_efficiency_score(self, utilizations: list[float], memory_usages: list[float]) -> float:
        """Calculate GPU efficiency score (0-100)."""
        if not utilizations:
            return 0.0

        avg_util = statistics.mean(utilizations)

        # Base score from utilization
        if avg_util >= 90:
            base_score = 100
        elif avg_util >= 70:
            base_score = 70 + (avg_util - 70) * 1.5  # Scale 70-90 to 70-100
        else:
            base_score = avg_util

        # Consistency bonus (lower variance is better)
        if len(utilizations) > 1:
            variance = statistics.variance(utilizations)
            consistency_bonus = max(0, 10 - (variance / 100))  # Up to 10 points
        else:
            consistency_bonus = 0

        # Memory efficiency factor
        if memory_usages:
            avg_memory = statistics.mean(memory_usages)
            # Optimal memory usage is around 70-85%
            if 70 <= avg_memory <= 85:
                memory_factor = 1.1
            elif 50 <= avg_memory < 70:
                memory_factor = 1.0
            elif avg_memory < 50:
                memory_factor = 0.9  # Underutilizing memory
            else:
                memory_factor = 0.8  # Memory pressure
        else:
            memory_factor = 1.0

        total_score = (base_score + consistency_bonus) * memory_factor
        return min(100, max(0, total_score))

    def _calculate_memory_fragmentation_score(self, memory_usages: list[float]) -> float:
        """Calculate memory fragmentation score (0-100, higher is worse)."""
        if not memory_usages or len(memory_usages) < 2:
            return 0.0

        # Calculate variation in memory usage
        variance = statistics.variance(memory_usages)
        mean_usage = statistics.mean(memory_usages)

        # Normalize by mean usage
        if mean_usage > 0:
            fragmentation_score = min(100, (variance / mean_usage) * 10)
        else:
            fragmentation_score = 0

        return fragmentation_score

    async def _generate_recommendations_for_gpu(self,
                                              gpu_id: int,
                                              utilizations: list[float],
                                              memory_usages: list[float]) -> list[SchedulingRecommendation]:
        """Generate optimization recommendations for a specific GPU."""
        recommendations = []

        if not utilizations:
            return recommendations

        avg_util = statistics.mean(utilizations)
        avg_memory = statistics.mean(memory_usages) if memory_usages else 0

        current_time = datetime.now()

        # Low utilization recommendations
        if avg_util < 50:
            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="high",
                action=OptimizationAction.INCREASE_BATCH_SIZE,
                description=f"GPU {gpu_id} utilization is {avg_util:.1f}%. Consider increasing batch size to improve GPU utilization.",
                expected_improvement=min(30, 90 - avg_util),
                implementation_difficulty="easy",
                estimated_time_to_implement=5,
                prerequisites=["Sufficient GPU memory available"],
                metadata={"current_utilization": avg_util, "target_utilization": 90}
            ))

            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="medium",
                action=OptimizationAction.OPTIMIZE_DATA_LOADING,
                description=f"GPU {gpu_id} may be waiting for data. Optimize data loading pipeline with prefetching and multi-threading.",
                expected_improvement=15,
                implementation_difficulty="medium",
                estimated_time_to_implement=30,
                prerequisites=["Access to training pipeline code"]
            ))

        # High memory usage recommendations
        if avg_memory > 90:
            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="high",
                action=OptimizationAction.ENABLE_MIXED_PRECISION,
                description=f"GPU {gpu_id} memory usage is {avg_memory:.1f}%. Enable mixed precision (FP16) to reduce memory usage.",
                expected_improvement=20,
                implementation_difficulty="easy",
                estimated_time_to_implement=10,
                prerequisites=["Modern GPU with Tensor Cores"],
                metadata={"current_memory": avg_memory}
            ))

        # Moderate utilization with room for improvement
        if 50 <= avg_util < 85:
            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="medium",
                action=OptimizationAction.ADD_GRADIENT_ACCUMULATION,
                description=f"GPU {gpu_id} utilization is {avg_util:.1f}%. Use gradient accumulation to effectively increase batch size without memory overhead.",
                expected_improvement=10,
                implementation_difficulty="medium",
                estimated_time_to_implement=20,
                prerequisites=["Training framework supporting gradient accumulation"]
            ))

        # Optimal utilization - maintain and fine-tune
        if 85 <= avg_util < 95:
            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="low",
                action=OptimizationAction.INCREASE_PARALLELIZATION,
                description=f"GPU {gpu_id} utilization is good ({avg_util:.1f}%). Consider minor increases in parallelization for optimal performance.",
                expected_improvement=5,
                implementation_difficulty="medium",
                estimated_time_to_implement=15
            ))

        # Very low utilization - scheduling recommendations
        if avg_util < 20:
            recommendations.append(SchedulingRecommendation(
                timestamp=current_time,
                priority="high",
                action=OptimizationAction.SCHEDULE_ADDITIONAL_TASKS,
                description=f"GPU {gpu_id} is significantly underutilized ({avg_util:.1f}%). Schedule additional tasks or workloads.",
                expected_improvement=70,
                implementation_difficulty="hard",
                estimated_time_to_implement=60,
                prerequisites=["Available tasks in queue", "Resource orchestration system"]
            ))

        # Sort by priority and expected improvement
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: (priority_order[x.priority], x.expected_improvement), reverse=True)

        return recommendations[:5]  # Return top 5 recommendations

    async def _generate_optimizations(self):
        """Generate system-wide optimizations and recommendations."""
        try:
            # Generate analytics for all GPUs
            for gpu_id in self.utilization_history.keys():
                if gpu_id not in self.analytics_cache or \
                   (datetime.now() - getattr(self.analytics_cache[gpu_id], 'timestamp', datetime.min)).total_seconds() > 3600:

                    analytics = await self._generate_analytics(gpu_id)
                    self.analytics_cache[gpu_id] = analytics

                    # Trigger recommendations
                    for rec in analytics.top_recommendations:
                        self._trigger_callbacks('recommendation_created', rec)

            # Generate system-wide recommendations
            await self._generate_system_wide_recommendations()

        except Exception as e:
            self.logger.error("optimization_generation_failed", error=str(e))

    async def _generate_system_wide_recommendations(self):
        """Generate system-wide optimization recommendations."""
        if not self.analytics_cache:
            return

        analytics_list = list(self.analytics_cache.values())

        # Calculate system averages
        system_avg_util = statistics.mean([a.avg_utilization for a in analytics_list])
        system_avg_efficiency = statistics.mean([a.efficiency_score for a in analytics_list])

        # System-wide recommendations
        if system_avg_util < 60:
            recommendation = SchedulingRecommendation(
                timestamp=datetime.now(),
                priority="high",
                action=OptimizationAction.CONSOLIDATE_WORKLOADS,
                description=f"System-wide GPU utilization is {system_avg_util:.1f}%. Consider consolidating workloads across fewer GPUs to improve efficiency.",
                expected_improvement=30,
                implementation_difficulty="hard",
                estimated_time_to_implement=120,
                prerequisites=["Workload migration capabilities", "Load balancing system"],
                metadata={"system_utilization": system_avg_util, "gpu_count": len(analytics_list)}
            )

            self._trigger_callbacks('recommendation_created', recommendation)

        if system_avg_efficiency < 70:
            recommendation = SchedulingRecommendation(
                timestamp=datetime.now(),
                priority="medium",
                action=OptimizationAction.OPTIMIZE_DATA_LOADING,
                description=f"System efficiency score is {system_avg_efficiency:.1f}. Implement system-wide data loading optimizations.",
                expected_improvement=20,
                implementation_difficulty="medium",
                estimated_time_to_implement=90,
                prerequisites=["Data pipeline access", "Profiling tools"],
                metadata={"system_efficiency": system_avg_efficiency}
            )

            self._trigger_callbacks('recommendation_created', recommendation)

    def get_current_status(self) -> dict[str, Any]:
        """Get current monitoring status and metrics."""
        status = {
            'monitoring_active': self.monitoring,
            'platform': self.platform_detector.detect_platform().platform.value,
            'monitored_gpus': list(self.utilization_history.keys()),
            'analytics_available': list(self.analytics_cache.keys()),
            'recent_alerts': len([a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]),
            'total_alerts': len(self.alerts),
            'configuration': {
                'monitoring_interval': self.monitoring_interval,
                'analytics_window_hours': self.analytics_window_hours,
                'optimization_check_interval': self.optimization_check_interval
            }
        }

        # Add current GPU status
        current_utilization = self.gpu_tracker.get_current_utilization()
        if current_utilization.get('status') not in ['no_data', 'no_recent_data']:
            status['current_utilization'] = current_utilization

        return status

    def get_analytics(self, gpu_id: int | None = None) -> dict[str, Any]:
        """Get analytics for specific GPU or all GPUs.

        Args:
            gpu_id: Specific GPU ID, or None for all GPUs

        Returns:
            Analytics data
        """
        if gpu_id is not None:
            if gpu_id in self.analytics_cache:
                return self.analytics_cache[gpu_id].to_dict()
            else:
                return {"error": f"No analytics available for GPU {gpu_id}"}
        else:
            return {
                gpu_id: analytics.to_dict()
                for gpu_id, analytics in self.analytics_cache.items()
            }

    def get_recommendations(self,
                          priority_filter: str | None = None,
                          action_filter: OptimizationAction | None = None,
                          hours: int = 24) -> list[dict[str, Any]]:
        """Get optimization recommendations.

        Args:
            priority_filter: Filter by priority ("high", "medium", "low")
            action_filter: Filter by action type
            hours: Hours to look back for recommendations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Collect recommendations from analytics
        for analytics in self.analytics_cache.values():
            recommendations.extend(analytics.top_recommendations)

        # Apply time filter
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recommendations = [r for r in recommendations if r.timestamp >= cutoff_time]

        # Apply filters
        if priority_filter:
            recommendations = [r for r in recommendations if r.priority == priority_filter]

        if action_filter:
            recommendations = [r for r in recommendations if r.action == action_filter]

        # Convert to dictionaries and sort
        rec_dicts = [asdict(r) for r in recommendations]
        priority_order = {"high": 3, "medium": 2, "low": 1}
        rec_dicts.sort(key=lambda x: (priority_order[x['priority']], x['expected_improvement']), reverse=True)

        return rec_dicts

    def get_alerts(self,
                   severity_filter: str | None = None,
                   hours: int = 24) -> list[dict[str, Any]]:
        """Get utilization alerts.

        Args:
            severity_filter: Filter by severity ("info", "warning", "critical")
            hours: Hours to look back for alerts

        Returns:
            List of alerts
        """
        # Apply time filter
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        # Apply severity filter
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity_filter]

        # Convert to dictionaries and sort by timestamp (newest first)
        alert_dicts = [asdict(a) for a in filtered_alerts]
        alert_dicts.sort(key=lambda x: x['timestamp'], reverse=True)

        return alert_dicts

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for monitoring events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.logger.warning("unknown_callback_event_type", event_type=event_type)

    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error("callback_failed", event_type=event_type, error=str(e))

    def _save_historical_data(self):
        """Save historical data to disk."""
        try:
            # Save utilization history
            history_file = self.storage_dir / "utilization_history.json"
            history_data = {}

            for gpu_id, history in self.utilization_history.items():
                history_data[str(gpu_id)] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'utilization': entry['utilization'],
                        'memory_usage': entry['memory_usage'],
                        'temperature': entry['temperature']
                    }
                    for entry in list(history)[-1000:]  # Keep last 1000 entries per GPU
                ]

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

            # Save analytics cache
            analytics_file = self.storage_dir / "analytics_cache.json"
            analytics_data = {
                str(gpu_id): analytics.to_dict()
                for gpu_id, analytics in self.analytics_cache.items()
            }

            with open(analytics_file, 'w') as f:
                json.dump(analytics_data, f, indent=2)

            # Save alerts
            alerts_file = self.storage_dir / "alerts_history.json"
            alerts_data = [asdict(alert) for alert in list(self.alerts)[-500:]]  # Keep last 500 alerts

            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)

            self.logger.debug("gpu_monitor_data_saved")

        except Exception as e:
            self.logger.error("gpu_monitor_data_save_failed", error=str(e))

    def _load_historical_data(self):
        """Load historical data from disk."""
        try:
            # Load utilization history
            history_file = self.storage_dir / "utilization_history.json"
            if history_file.exists():
                with open(history_file) as f:
                    history_data = json.load(f)

                for gpu_id_str, history in history_data.items():
                    gpu_id = int(gpu_id_str)
                    self.utilization_history[gpu_id] = deque(maxlen=2880)

                    for entry in history:
                        self.utilization_history[gpu_id].append({
                            'timestamp': datetime.fromisoformat(entry['timestamp']),
                            'utilization': entry['utilization'],
                            'memory_usage': entry['memory_usage'],
                            'temperature': entry['temperature']
                        })

            # Load alerts
            alerts_file = self.storage_dir / "alerts_history.json"
            if alerts_file.exists():
                with open(alerts_file) as f:
                    alerts_data = json.load(f)

                for alert_dict in alerts_data:
                    try:
                        alert = UtilizationAlert(
                            timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                            gpu_id=alert_dict['gpu_id'],
                            alert_type=alert_dict['alert_type'],
                            severity=alert_dict['severity'],
                            message=alert_dict['message'],
                            current_utilization=alert_dict['current_utilization'],
                            threshold=alert_dict['threshold'],
                            suggested_actions=[OptimizationAction(action) for action in alert_dict.get('suggested_actions', [])],
                            metadata=alert_dict.get('metadata', {})
                        )
                        self.alerts.append(alert)
                    except Exception as e:
                        self.logger.error("alert_load_failed", error=str(e))

            self.logger.info("gpu_monitor_historical_data_loaded")

        except Exception as e:
            self.logger.error("gpu_monitor_data_load_failed", error=str(e))

    def create_performance_profile(self,
                                 workload_id: str,
                                 target_utilization: float = 95.0,
                                 target_memory: float = 80.0) -> PerformanceProfile:
        """Create a performance profile for workload optimization.

        Args:
            workload_id: Unique workload identifier
            target_utilization: Target GPU utilization percentage
            target_memory: Target memory utilization percentage

        Returns:
            Performance profile
        """
        profile = PerformanceProfile(
            workload_id=workload_id,
            gpu_utilization_target=target_utilization,
            memory_utilization_target=target_memory
        )

        self.performance_profiles[workload_id] = profile
        return profile

    def update_performance_profile(self,
                                 workload_id: str,
                                 performance_data: dict[str, Any]) -> bool:
        """Update performance profile with new data.

        Args:
            workload_id: Workload identifier
            performance_data: Performance metrics

        Returns:
            True if updated successfully
        """
        if workload_id not in self.performance_profiles:
            return False

        profile = self.performance_profiles[workload_id]

        # Update profile fields based on performance data
        if 'optimal_batch_size' in performance_data:
            profile.batch_size_optimal = performance_data['optimal_batch_size']

        if 'mixed_precision_effective' in performance_data:
            profile.mixed_precision_effective = performance_data['mixed_precision_effective']

        if 'data_loading_optimal' in performance_data:
            profile.data_loading_optimal = performance_data['data_loading_optimal']

        if 'parallelization_level' in performance_data:
            profile.parallelization_level = performance_data['parallelization_level']

        # Update historical performance
        timestamp = datetime.now().isoformat()
        profile.historical_performance[timestamp] = performance_data

        # Keep only recent history (last 100 entries)
        if len(profile.historical_performance) > 100:
            oldest_keys = sorted(profile.historical_performance.keys())[:50]
            for key in oldest_keys:
                del profile.historical_performance[key]

        return True

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get comprehensive optimization summary.

        Returns:
            Optimization summary with recommendations and metrics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': {
                'active': self.monitoring,
                'monitored_gpus': len(self.utilization_history),
                'analytics_available': len(self.analytics_cache)
            },
            'system_metrics': {},
            'recommendations_summary': {
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'total': 0
            },
            'alerts_summary': {
                'critical': 0,
                'warning': 0,
                'info': 0,
                'total': len(self.alerts)
            },
            'optimization_potential': {
                'average_improvement': 0.0,
                'total_wasted_capacity': 0.0
            }
        }

        if self.analytics_cache:
            # System metrics
            analytics_list = list(self.analytics_cache.values())
            summary['system_metrics'] = {
                'average_utilization': statistics.mean([a.avg_utilization for a in analytics_list]),
                'average_efficiency': statistics.mean([a.efficiency_score for a in analytics_list]),
                'total_idle_time': sum([a.idle_time_percentage for a in analytics_list]) / len(analytics_list),
                'optimal_time_percentage': sum([a.optimal_time_percentage for a in analytics_list]) / len(analytics_list)
            }

            # Recommendations summary
            all_recommendations = []
            for analytics in analytics_list:
                all_recommendations.extend(analytics.top_recommendations)

            for rec in all_recommendations:
                summary['recommendations_summary'][f"{rec.priority}_priority"] += 1
                summary['recommendations_summary']['total'] += 1

            # Optimization potential
            if all_recommendations:
                summary['optimization_potential']['average_improvement'] = statistics.mean([r.expected_improvement for r in all_recommendations])

            # Calculate wasted capacity
            target_utilization = 95.0
            current_avg = summary['system_metrics']['average_utilization']
            summary['optimization_potential']['total_wasted_capacity'] = max(0, target_utilization - current_avg)

        # Alerts summary
        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 86400]  # Last 24h
        for alert in recent_alerts:
            summary['alerts_summary'][alert.severity] += 1

        return summary


# Singleton instance
_gpu_monitor = None


def get_gpu_monitor() -> GPUMonitor:
    """Get singleton GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


# Convenience functions for quick access
async def start_gpu_monitoring() -> bool:
    """Start GPU monitoring system."""
    monitor = get_gpu_monitor()
    return await monitor.start_monitoring()


async def stop_gpu_monitoring() -> bool:
    """Stop GPU monitoring system."""
    monitor = get_gpu_monitor()
    return await monitor.stop_monitoring()


def get_gpu_analytics(gpu_id: int | None = None) -> dict[str, Any]:
    """Get GPU analytics."""
    monitor = get_gpu_monitor()
    return monitor.get_analytics(gpu_id)


def get_gpu_recommendations(priority_filter: str | None = None) -> list[dict[str, Any]]:
    """Get GPU optimization recommendations."""
    monitor = get_gpu_monitor()
    return monitor.get_recommendations(priority_filter=priority_filter)


def get_gpu_alerts(severity_filter: str | None = None) -> list[dict[str, Any]]:
    """Get GPU utilization alerts."""
    monitor = get_gpu_monitor()
    return monitor.get_alerts(severity_filter=severity_filter)
