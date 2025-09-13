# Health Monitoring and System Status

This document describes the comprehensive health monitoring system implemented for the ARC Prize 2025 competition platform. The system provides robust health checks, performance monitoring, and system status reporting for reliable operation in production environments.

## Overview

The health monitoring system provides:

- **Real-time health status** for all system components
- **Kubernetes-compatible probes** for container orchestration
- **Performance metrics** for monitoring and alerting
- **Component-specific monitoring** with detailed diagnostics
- **Configurable thresholds** and extensible architecture
- **HTTP status codes** appropriate for different health states

## Architecture

### Core Components

1. **Health Models** (`src/domain/health_models.py`)
   - Data structures for health status representation
   - Performance metrics definitions
   - Component-specific health information
   - Configurable health check parameters

2. **Health Service** (`src/domain/services/health_service.py`)
   - Core health check logic and component monitoring
   - Asynchronous health checks with timeout handling
   - Performance metrics collection
   - Extensible health check registry

3. **Health API Routes** (`src/adapters/api/routes/health.py`)
   - RESTful health check endpoints
   - Kubernetes probe endpoints
   - Component-specific health checks
   - Performance metrics API

4. **Configuration** (`configs/health_check.yaml`)
   - Health check settings and thresholds
   - Component-specific configuration
   - Kubernetes probe settings
   - Performance monitoring parameters

## Health Check Endpoints

### Basic Health Checks

#### `GET /health/`
Basic health check for critical system components.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-12T10:30:00Z",
  "version": "1.3.0",
  "environment": "production",
  "uptime_seconds": 3600.5,
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15.2
    },
    "cache": {
      "status": "healthy", 
      "response_time_ms": 8.1
    },
    "memory": {
      "status": "degraded",
      "response_time_ms": 2.3
    }
  }
}
```

**HTTP Status Codes:**
- `200`: System is healthy
- `503`: System is unhealthy or degraded

#### `GET /health/detailed`
Comprehensive health check with detailed metrics.

**Response includes:**
- All component health status
- Performance metrics (CPU, memory, disk usage)
- Component-specific details
- Resource utilization statistics
- Error information and diagnostics

#### `GET /health/status`
Simple status summary for load balancers.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-12T10:30:00Z",
  "uptime": 3600.5
}
```

### Kubernetes Probes

#### `GET /health/live`
**Liveness Probe** - Indicates if the service is alive and running.

- Returns `200 OK` if service is responsive
- Returns `503 Service Unavailable` if service is dead
- **Purpose**: Kubernetes uses this to restart failed containers
- **Response**: Plain text ("OK" or "Not Alive")

#### `GET /health/ready`
**Readiness Probe** - Indicates if the service is ready to accept traffic.

- Returns `200 OK` if service can handle requests  
- Returns `503 Service Unavailable` if service is not ready
- **Purpose**: Kubernetes uses this to route traffic
- **Response**: Plain text ("OK" or "Not Ready")

#### `GET /health/startup`
**Startup Probe** - Indicates if the service has finished starting up.

- Returns `200 OK` if service has started successfully
- Returns `503 Service Unavailable` if service is still starting
- **Purpose**: Kubernetes uses this for slow-starting containers
- **Response**: Plain text ("OK" or "Starting")

### Component-Specific Health

#### `GET /health/component/{component_name}`
Individual component health check.

**Available components:**
- `database` - Database connectivity and performance
- `cache` - Cache system status and performance  
- `wandb` - Weights & Biases integration health
- `storage` - Disk space and file system access
- `memory` - Memory usage and availability
- `cpu` - CPU utilization and performance

**Example Response:**
```json
{
  "component_name": "cache",
  "component_type": "cache",
  "status": "healthy",
  "response_time_ms": 8.1,
  "details": {
    "hit_rate": 0.85,
    "cache_size_mb": 256.4,
    "usage_percent": 45.2,
    "total_requests": 15420
  },
  "last_checked": "2025-01-12T10:30:00Z",
  "detailed_info": {
    "hit_rate_percent": 85.0,
    "total_hits": 13107,
    "total_misses": 2313,
    "evictions": 42
  }
}
```

### Performance Monitoring

#### `GET /health/metrics`
Performance metrics for monitoring systems.

**Response includes:**
- CPU and memory utilization
- Disk usage statistics  
- Network connection counts
- Request rates and response times
- Error rates and component status
- Uptime and availability metrics

#### `GET /health/config`
Health check configuration and registered components.

**Response:**
```json
{
  "timestamp": "2025-01-12T10:30:00Z",
  "health_check_config": {
    "enabled": true,
    "check_interval_seconds": 30,
    "timeout_seconds": 5,
    "degraded_threshold_ms": 1000.0,
    "unhealthy_threshold_ms": 5000.0,
    "components_to_check": ["database", "cache", "wandb", "storage", "memory", "cpu"]
  },
  "registered_checks": ["database", "cache", "wandb", "storage", "memory", "cpu"]
}
```

## Health Status Levels

The system uses four health status levels:

### `HEALTHY`
- All components functioning normally
- Response times within acceptable thresholds
- No errors or warnings
- **HTTP Status**: 200

### `DEGRADED`  
- System operational but with performance issues
- Some components responding slowly
- Non-critical errors present
- **HTTP Status**: 200 (still operational)

### `UNHEALTHY`
- Critical components failing
- System cannot serve requests reliably
- Major errors or timeouts
- **HTTP Status**: 503

### `UNKNOWN`
- Health status cannot be determined
- Health checks failing or timing out
- **HTTP Status**: 503

## Component Monitoring

### Database Health
- **Connection pool status**: Active and idle connections
- **Query performance**: Response times and throughput  
- **Connection availability**: Ability to establish new connections
- **Database size**: Storage utilization

### Cache Health  
- **Hit rate**: Cache effectiveness percentage
- **Storage utilization**: Cache size and usage limits
- **Performance**: Get/set operation response times
- **Eviction rate**: Memory pressure indicators

### External Service Health (W&B)
- **API connectivity**: Endpoint reachability
- **Authentication status**: Valid credentials
- **Usage limits**: Storage quota and rate limits
- **Success rates**: API call reliability

### Storage Health
- **Disk usage**: Available space and utilization
- **Write access**: File system permissions  
- **I/O performance**: Read/write operation times
- **Mount status**: Volume availability

### System Resource Health
- **Memory usage**: RAM utilization and availability
- **CPU usage**: Processor utilization
- **Network**: Connection counts and bandwidth
- **Process health**: Application resource usage

## Configuration

Health checks are configured via `configs/health_check.yaml`:

```yaml
health_check:
  enabled: true
  check_interval_seconds: 30
  timeout_seconds: 5
  max_failures: 3
  
  # Response time thresholds
  degraded_threshold_ms: 1000.0
  unhealthy_threshold_ms: 5000.0
  
  # Components to monitor
  components_to_check:
    - database
    - cache
    - wandb
    - storage
    - memory
    - cpu
    
  # Component-specific settings
  component_settings:
    database:
      timeout_seconds: 10
      max_connection_time_ms: 5000
      
    cache:
      min_hit_rate_percent: 50.0
      max_usage_percent: 90.0
      
    # ... additional component settings
```

### Kubernetes Configuration

The health check endpoints integrate seamlessly with Kubernetes:

```yaml
# Deployment snippet
containers:
- name: arc-system
  image: arc-prize-2025:v1.3.0
  
  livenessProbe:
    httpGet:
      path: /health/live
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  
  readinessProbe:
    httpGet:
      path: /health/ready  
      port: 8000
    initialDelaySeconds: 5
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
    
  startupProbe:
    httpGet:
      path: /health/startup
      port: 8000
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 30  # 5 minutes for startup
```

## Docker Integration

The Docker container includes built-in health checks:

```dockerfile
# Health check using dedicated endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1
```

## Monitoring Integration

### Prometheus Metrics

The health check system can be integrated with Prometheus for metrics collection:

- Health status metrics (healthy/unhealthy counts)
- Response time histograms
- Component availability percentages
- Resource utilization metrics

### Alerting

Configure alerts based on health check results:

- **Critical**: System unhealthy for > 2 minutes
- **Warning**: System degraded for > 5 minutes  
- **Info**: Component response time increased

### Log Monitoring

Health check events are logged with structured data:

```json
{
  "timestamp": "2025-01-12T10:30:00Z",
  "level": "INFO",
  "message": "health_check_completed",
  "status": "healthy",
  "components_checked": 6,
  "duration_ms": 45.2
}
```

## Testing

### Manual Testing

Use the provided test script:

```bash
# Test all health endpoints
python scripts/test_health_endpoints.py

# Test specific endpoint
curl http://localhost:8000/health/

# Test with detailed output
curl -s http://localhost:8000/health/detailed | jq
```

### Automated Testing

Integration tests are provided in `tests/integration/test_health_checks.py`:

- Endpoint functionality tests
- Response format validation
- Performance benchmarks
- Concurrent request handling
- Error condition testing

### Load Testing

Benchmark health check performance:

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health/

# Using curl for single test
time curl -s http://localhost:8000/health/ > /dev/null
```

## Extensibility

### Adding New Components

1. **Register health check function**:
```python
from src.domain.health_models import health_check_registry, ComponentType

async def check_my_component() -> HealthCheckResult:
    # Implement health check logic
    return HealthCheckResult(
        component_name="my_component",
        component_type=ComponentType.SERVICE,
        status=HealthStatus.HEALTHY,
        response_time_ms=50.0
    )

# Register the check
health_check_registry.register(
    "my_component", 
    check_my_component,
    ComponentType.SERVICE
)
```

2. **Add to configuration**:
```yaml
health_check:
  components_to_check:
    - database
    - cache
    - my_component  # Add new component
```

### Custom Health Checks

Implement custom health check logic by extending the `HealthService` class:

```python
class CustomHealthService(HealthService):
    async def _check_custom_component(self) -> HealthCheckResult:
        # Custom health check implementation
        pass
```

## Best Practices

### Health Check Design
- **Keep checks lightweight**: Health checks should complete quickly
- **Use appropriate timeouts**: Balance responsiveness with reliability
- **Implement proper error handling**: Gracefully handle exceptions
- **Monitor check performance**: Track health check response times

### Thresholds
- **Conservative degraded thresholds**: Catch issues early
- **Aggressive unhealthy thresholds**: Prevent cascading failures
- **Environment-specific tuning**: Adjust for different deployments

### Monitoring
- **Monitor the monitors**: Ensure health checks themselves are reliable
- **Historical trending**: Track health metrics over time
- **Automated alerting**: Set up proactive notifications

### Operations
- **Document component dependencies**: Understand health check relationships
- **Test failure scenarios**: Verify health checks detect actual problems
- **Regular review**: Update thresholds based on operational experience

## Troubleshooting

### Common Issues

1. **Health checks timing out**
   - Increase timeout values in configuration
   - Check for resource constraints
   - Review component-specific implementations

2. **False positive failures**
   - Adjust degraded/unhealthy thresholds
   - Review error handling in health checks
   - Check for environmental factors

3. **Performance impact**
   - Reduce health check frequency
   - Optimize check implementations
   - Use asynchronous processing

### Debug Information

Enable detailed logging for troubleshooting:

```yaml
logging:
  log_health_checks: true
  log_level: "DEBUG"
  log_component_details: true
  log_response_times: true
```

### Health Check Status

Monitor health check system itself:

```bash
# Check configuration
curl http://localhost:8000/health/config

# Monitor performance
curl http://localhost:8000/health/metrics

# Individual component status
curl http://localhost:8000/health/component/cache
```

## Security Considerations

- **No sensitive data exposure**: Health checks don't reveal system internals
- **Rate limiting**: Prevent health check endpoint abuse
- **Authentication**: Consider securing detailed health endpoints
- **Information disclosure**: Limit error details in production

## Conclusion

The health monitoring system provides comprehensive visibility into system status with:

- **Multiple endpoint types** for different use cases
- **Kubernetes-native integration** for container orchestration
- **Configurable and extensible** architecture
- **Performance monitoring** capabilities
- **Production-ready** reliability and security

This system enables confident deployment and operation of the ARC Prize 2025 platform with full observability and automated health management.