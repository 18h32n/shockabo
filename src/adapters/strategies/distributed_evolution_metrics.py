"""
Prometheus metrics for distributed evolution monitoring.

Provides instrumentation for tracking distributed evolution performance.
"""


try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class DistributedEvolutionMetrics:
    """Prometheus metrics for distributed evolution system."""

    def __init__(self):
        """Initialize metrics collectors."""
        if not HAS_PROMETHEUS:
            return

        self.active_platforms = Gauge(
            'distributed_evolution_active_platforms',
            'Number of currently active platforms',
            ['coordinator_id']
        )

        self.throughput = Gauge(
            'distributed_evolution_throughput',
            'Tasks completed per hour',
            ['platform_id', 'configuration']
        )

        self.generation_counter = Counter(
            'distributed_evolution_generation_total',
            'Total number of generations completed',
            ['platform_id', 'task_id']
        )

        self.checkpoint_size_bytes = Histogram(
            'distributed_evolution_checkpoint_size_bytes',
            'Size of checkpoints in bytes',
            ['platform_id'],
            buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000]
        )

        self.heartbeat_latency = Histogram(
            'distributed_evolution_heartbeat_latency_seconds',
            'Heartbeat round-trip latency in seconds',
            ['platform_id'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        self.checkpoint_upload_duration = Histogram(
            'distributed_evolution_checkpoint_upload_seconds',
            'Time taken to upload checkpoint',
            ['platform_id'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        )

        self.checkpoint_download_duration = Histogram(
            'distributed_evolution_checkpoint_download_seconds',
            'Time taken to download checkpoint',
            ['platform_id'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        )

        self.merge_duration = Histogram(
            'distributed_evolution_merge_seconds',
            'Time taken to merge populations',
            ['platform_count'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )

        self.population_diversity = Gauge(
            'distributed_evolution_population_diversity',
            'Unique programs in population',
            ['platform_id', 'generation']
        )

        self.platform_health = Gauge(
            'distributed_evolution_platform_health',
            'Platform health status (1=healthy, 0=unhealthy)',
            ['platform_id']
        )

        self.platform_disconnections = Counter(
            'distributed_evolution_platform_disconnections_total',
            'Total number of platform disconnections',
            ['platform_id', 'reason']
        )

        self.checkpoint_failures = Counter(
            'distributed_evolution_checkpoint_failures_total',
            'Total number of checkpoint operation failures',
            ['platform_id', 'operation', 'error_type']
        )

        self.merge_conflicts = Counter(
            'distributed_evolution_merge_conflicts_total',
            'Total number of merge conflicts',
            ['conflict_type']
        )

        self.fitness_distribution = Histogram(
            'distributed_evolution_fitness',
            'Distribution of program fitness scores',
            ['platform_id'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.task_redistribution_count = Counter(
            'distributed_evolution_task_redistributions_total',
            'Number of task redistributions due to failures',
            ['from_platform', 'to_platform']
        )

        self.coordinator_elections = Counter(
            'distributed_evolution_coordinator_elections_total',
            'Number of coordinator elections',
            ['reason']
        )

    def record_platform_registration(self, platform_id: str, coordinator_id: str):
        """Record platform coming online."""
        if not HAS_PROMETHEUS:
            return
        self.active_platforms.labels(coordinator_id=coordinator_id).inc()
        self.platform_health.labels(platform_id=platform_id).set(1)

    def record_platform_disconnection(self, platform_id: str, coordinator_id: str, reason: str):
        """Record platform going offline."""
        if not HAS_PROMETHEUS:
            return
        self.active_platforms.labels(coordinator_id=coordinator_id).dec()
        self.platform_health.labels(platform_id=platform_id).set(0)
        self.platform_disconnections.labels(platform_id=platform_id, reason=reason).inc()

    def record_throughput(self, platform_id: str, configuration: str, tasks_per_hour: float):
        """Record throughput measurement."""
        if not HAS_PROMETHEUS:
            return
        self.throughput.labels(
            platform_id=platform_id,
            configuration=configuration
        ).set(tasks_per_hour)

    def record_generation(self, platform_id: str, task_id: str):
        """Record generation completion."""
        if not HAS_PROMETHEUS:
            return
        self.generation_counter.labels(
            platform_id=platform_id,
            task_id=task_id
        ).inc()

    def record_checkpoint_size(self, platform_id: str, size_bytes: int):
        """Record checkpoint size."""
        if not HAS_PROMETHEUS:
            return
        self.checkpoint_size_bytes.labels(platform_id=platform_id).observe(size_bytes)

    def record_heartbeat_latency(self, platform_id: str, latency_seconds: float):
        """Record heartbeat latency."""
        if not HAS_PROMETHEUS:
            return
        self.heartbeat_latency.labels(platform_id=platform_id).observe(latency_seconds)

    def record_checkpoint_upload(self, platform_id: str, duration_seconds: float):
        """Record checkpoint upload duration."""
        if not HAS_PROMETHEUS:
            return
        self.checkpoint_upload_duration.labels(platform_id=platform_id).observe(duration_seconds)

    def record_checkpoint_download(self, platform_id: str, duration_seconds: float):
        """Record checkpoint download duration."""
        if not HAS_PROMETHEUS:
            return
        self.checkpoint_download_duration.labels(platform_id=platform_id).observe(duration_seconds)

    def record_merge_duration(self, platform_count: int, duration_seconds: float):
        """Record population merge duration."""
        if not HAS_PROMETHEUS:
            return
        self.merge_duration.labels(platform_count=str(platform_count)).observe(duration_seconds)

    def record_population_diversity(self, platform_id: str, generation: int, unique_count: int):
        """Record population diversity."""
        if not HAS_PROMETHEUS:
            return
        self.population_diversity.labels(
            platform_id=platform_id,
            generation=str(generation)
        ).set(unique_count)

    def record_checkpoint_failure(self, platform_id: str, operation: str, error_type: str):
        """Record checkpoint operation failure."""
        if not HAS_PROMETHEUS:
            return
        self.checkpoint_failures.labels(
            platform_id=platform_id,
            operation=operation,
            error_type=error_type
        ).inc()

    def record_merge_conflict(self, conflict_type: str):
        """Record merge conflict."""
        if not HAS_PROMETHEUS:
            return
        self.merge_conflicts.labels(conflict_type=conflict_type).inc()

    def record_fitness_score(self, platform_id: str, fitness: float):
        """Record individual fitness score."""
        if not HAS_PROMETHEUS:
            return
        self.fitness_distribution.labels(platform_id=platform_id).observe(fitness)

    def record_task_redistribution(self, from_platform: str, to_platform: str):
        """Record task redistribution."""
        if not HAS_PROMETHEUS:
            return
        self.task_redistribution_count.labels(
            from_platform=from_platform,
            to_platform=to_platform
        ).inc()

    def record_coordinator_election(self, reason: str):
        """Record coordinator election event."""
        if not HAS_PROMETHEUS:
            return
        self.coordinator_elections.labels(reason=reason).inc()


_global_metrics: DistributedEvolutionMetrics | None = None


def get_metrics() -> DistributedEvolutionMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = DistributedEvolutionMetrics()
    return _global_metrics
