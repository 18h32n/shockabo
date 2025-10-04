# Distributed Evolution Performance Tuning Guide

## Overview

This guide provides recommendations for optimizing distributed evolution performance across multiple platforms.

## Key Performance Metrics

### Target Metrics

- **Throughput Multiplier**: ≥ 2.5x single platform baseline
- **Checkpoint Overhead**: < 20% of total execution time
- **Merge Overhead**: < 10% of total execution time
- **Platform Recovery Time**: < 60 seconds
- **Deduplication Rate**: 100% (no duplicate programs)

### Monitoring

Use Prometheus metrics to track performance:

```python
from src.adapters.strategies.distributed_evolution_metrics import get_metrics

metrics = get_metrics()
metrics.record_throughput("kaggle-1", "multi_platform", 150.0)
```

## Optimization Strategies

### 1. Checkpoint Compression

Enable compression to reduce network transfer time:

```python
checkpoint_manager = AsyncCheckpointManager(
    platform_id="kaggle-1",
    enable_compression=True,
    compression_level=6  # Balance between speed and size
)
```

**Compression Levels**:
- Level 1-3: Fast compression, lower ratio (use for frequent checkpoints)
- Level 6: Default balanced setting
- Level 9: Maximum compression, slower (use for large populations)

**Expected Savings**: 60-80% size reduction for typical DSL programs

### 2. Batch Checkpoint Operations

Reduce API overhead by batching checkpoint saves:

```python
checkpoint_manager._batch_size = 5  # Checkpoint every 5 generations
checkpoint_manager._batch_timeout = 5.0  # Or after 5 seconds
```

**Trade-offs**:
- Larger batch size = lower overhead but higher recovery cost
- Recommended: 3-5 generations for most workloads

### 3. Platform-Specific Tuning

Adjust resource allocation based on platform capabilities:

```yaml
platforms:
  - id: kaggle-1
    memory_limit_mb: 4096
    worker_count: 2
    batch_size: 500  # Higher for Kaggle's stable environment
  - id: colab-1
    memory_limit_mb: 12288
    worker_count: 2
    batch_size: 250  # Lower due to session time limits
```

### 4. Heartbeat Optimization

Balance responsiveness with network overhead:

```yaml
distributed_evolution:
  heartbeat_timeout: 30  # Seconds before platform considered dead
  heartbeat_interval: 10  # Seconds between heartbeat pings
```

**Guidelines**:
- Stable networks: 30s timeout, 10s interval
- Unstable networks: 60s timeout, 20s interval
- Low-latency requirements: 15s timeout, 5s interval

### 5. Population Sharding Strategy

Optimize island distribution across platforms:

```python
coordinator.create_distribution_plan(
    task=task,
    available_platforms=platforms,
    islands_per_platform={
        Platform.KAGGLE: 4,  # Most stable platform
        Platform.COLAB: 3,   # Good resources, session limits
        Platform.LOCAL: 2    # Variable performance
    }
)
```

**Best Practices**:
- Assign more islands to stable platforms
- Consider memory constraints (4GB = max 3 islands typically)
- Monitor diversity metrics to ensure good exploration

### 6. Merge Optimization

Reduce merge overhead with smart deduplication:

```python
merger = PopulationMerger()
merged_population = merger.merge_populations(
    populations,
    preserve_top_n=20  # Keep only top individuals
)
```

**Trade-offs**:
- Full merge: Maximum diversity, higher overhead
- Top-N merge: Lower overhead, risk of losing good candidates
- Recommended: Keep top 20% from each platform

### 7. GCS Storage Optimization

Configure GCS client for better performance:

```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket(bucket_name)
bucket.blob(blob_name).upload_from_file(
    file_obj,
    timeout=30,  # Shorter timeout for faster failures
    retry=None   # Disable automatic retries for control
)
```

### 8. Async Operation Tuning

Optimize asyncio concurrency:

```python
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent uploads

async with semaphore:
    await checkpoint_manager.save_checkpoint_async(checkpoint)
```

## Common Performance Issues

### Issue 1: High Checkpoint Overhead (>20%)

**Symptoms**: Checkpoint operations taking significant time
**Solutions**:
1. Enable compression: `enable_compression=True`
2. Increase batch size: `_batch_size = 5`
3. Use faster compression level: `compression_level=3`

### Issue 2: Platform Disconnections

**Symptoms**: Frequent platform health changes
**Solutions**:
1. Increase heartbeat timeout: `heartbeat_timeout: 60`
2. Check network stability
3. Reduce batch size to minimize recovery time

### Issue 3: Low Throughput Multiplier (<2.5x)

**Symptoms**: Multi-platform not faster than single platform
**Solutions**:
1. Verify platform sharding is balanced
2. Check for merge bottlenecks (overhead >10%)
3. Ensure platforms are not waiting for each other (async operations)
4. Profile with `cProfile` to identify bottlenecks

### Issue 4: Memory Issues

**Symptoms**: Out of memory errors on platforms
**Solutions**:
1. Reduce population size per island
2. Lower `batch_size` in platform configuration
3. Increase checkpoint frequency to free memory
4. Monitor `resource_usage` in checkpoint metadata

## Performance Benchmarking

### Running Benchmarks

```bash
python scripts/benchmark_distributed_evolution.py \
  tests/performance/data/distributed_test_tasks.json \
  results/benchmark_output.json
```

### Interpreting Results

```json
{
  "configuration": "multi_platform",
  "platform_count": 3,
  "throughput_multiplier": 2.8,
  "checkpoint_overhead_seconds": 5.2,
  "merge_overhead_seconds": 1.1
}
```

**Good Results**:
- Throughput multiplier ≥ 2.5x
- Checkpoint overhead < 20% of duration
- Merge overhead < 10% of duration

**Poor Results**:
- Throughput multiplier < 2.0x → Check platform utilization
- Checkpoint overhead > 30% → Enable compression
- Merge overhead > 15% → Optimize merge strategy

## Platform-Specific Recommendations

### Kaggle

- Best for: Coordinator role, stable long-running evolution
- Settings: `worker_count: 2`, `batch_size: 500`, `memory_limit: 4096`
- Checkpointing: Every 1-2 generations (stable environment)

### Google Colab

- Best for: Worker role, GPU-accelerated evaluation
- Settings: `worker_count: 2`, `batch_size: 250`, `memory_limit: 12288`
- Checkpointing: Every generation (session time limits)
- Note: Enable checkpoint recovery for session disconnects

### Paperspace

- Best for: Supplementary worker role
- Settings: `worker_count: 1`, `batch_size: 100`, `memory_limit: 2048`
- Checkpointing: Every generation (limited free tier)

### Local Machine

- Best for: Development, backup coordinator
- Settings: Variable based on hardware
- Checkpointing: Less frequent (stable environment)

## Monitoring Setup

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'distributed_evolution'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard

Import the distributed evolution dashboard template:

1. Active Platforms Panel: Track online platforms
2. Throughput Panel: Monitor tasks/hour
3. Checkpoint Stats: Upload/download times
4. Platform Health: Heartbeat status

### Alert Rules

```yaml
groups:
  - name: distributed_evolution
    rules:
      - alert: LowThroughput
        expr: distributed_evolution_throughput < 50
        for: 5m
        annotations:
          summary: "Throughput below 50 tasks/hour"
      
      - alert: PlatformDisconnected
        expr: distributed_evolution_platform_health == 0
        for: 1m
        annotations:
          summary: "Platform {{ $labels.platform_id }} disconnected"
```

## Troubleshooting Checklist

- [ ] Verify all platforms are registered and healthy
- [ ] Check checkpoint compression is enabled
- [ ] Confirm batch size is appropriate for platform
- [ ] Monitor checkpoint overhead (<20% target)
- [ ] Monitor merge overhead (<10% target)
- [ ] Verify GCS credentials are configured
- [ ] Check network latency between platforms
- [ ] Review Prometheus metrics for bottlenecks
- [ ] Profile with cProfile if needed
- [ ] Verify population sharding is balanced

## Advanced Tuning

### Custom Sharding Strategy

Implement custom shard size calculation:

```python
def calculate_platform_shard_sizes(
    platforms: list[dict],
    total_population: int
) -> dict[str, int]:
    """Calculate shard sizes weighted by platform capabilities."""
    total_capacity = sum(
        p['workers'] * p['memory_mb'] * p['batch_size'] 
        for p in platforms
    )
    
    shard_sizes = {}
    for platform in platforms:
        capacity = platform['workers'] * platform['memory_mb'] * platform['batch_size']
        shard_size = int((capacity / total_capacity) * total_population)
        shard_sizes[platform['id']] = shard_size
    
    return shard_sizes
```

### Dynamic Load Balancing

Adjust sharding based on runtime performance:

```python
if platform_metrics[platform_id]['avg_generation_time'] > threshold:
    reduce_island_count(platform_id)
    redistribute_islands()
```

### Predictive Checkpointing

Checkpoint before expected disruptions:

```python
if time_since_last_checkpoint > (heartbeat_timeout * 0.8):
    await checkpoint_manager.save_checkpoint_async(checkpoint)
```

## References

- [Story 2.9: Distributed Evolution](../stories/2.9.distributed-evolution-across-platforms.story.md)
- [Architecture: Infrastructure](../architecture/infrastructure.md)
- [GCS Python Client Documentation](https://googleapis.dev/python/storage/latest/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)