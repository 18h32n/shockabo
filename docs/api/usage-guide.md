# ARC Evaluation Framework API Usage Guide

## Overview

The ARC Evaluation Framework API provides comprehensive endpoints for submitting, evaluating, and monitoring ARC (Abstraction and Reasoning Corpus) task solutions. This guide covers authentication, usage patterns, best practices, and troubleshooting.

## Base URL and Versioning

```
Base URL: http://localhost:8000/api/v1/evaluation
API Version: v1
Content-Type: application/json
```

## Authentication

### JWT Token Authentication

All API endpoints require JWT authentication via the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Obtaining Tokens (Development)

For development and testing, use the token creation endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "your_user_id"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user_id": "your_user_id"
}
```

### Production Authentication

In production, implement proper authentication:
- OAuth 2.0 / OpenID Connect
- User credential validation
- Multi-factor authentication (MFA)
- Rate limiting and brute force protection

## Core API Usage Patterns

### 1. Single Task Submission

Submit individual ARC task solutions for evaluation:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/submit" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "arc_2024_001",
       "predicted_output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
       "strategy": "PATTERN_MATCH",
       "confidence_score": 0.92,
       "attempt_number": 1,
       "metadata": {
         "processing_time_ms": 1450,
         "pattern_type": "symmetry"
       }
     }'
```

### 2. Batch Evaluation

Process multiple tasks efficiently:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/batch" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "evaluations": [
         {
           "task_id": "arc_2024_001",
           "predicted_output": [[1, 0], [0, 1]],
           "confidence": 0.92,
           "attempt_number": 1
         },
         {
           "task_id": "arc_2024_002",
           "predicted_output": [[2, 3], [4, 5]],
           "confidence": 0.78,
           "attempt_number": 1
         }
       ],
       "strategy": "ENSEMBLE",
       "experiment_id": "exp_validation_001",
       "parallel_processing": true
     }'
```

### 3. Real-time Monitoring

Connect to WebSocket for live updates:

```javascript
// JavaScript WebSocket client
const token = 'your_jwt_token';
const ws = new WebSocket(`ws://localhost:8000/api/v1/evaluation/ws?token=${token}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'connection_established':
      console.log('Connected to evaluation dashboard');
      break;
      
    case 'task_submitted':
      console.log(`Task ${data.task_id} evaluated: ${data.accuracy}`);
      break;
      
    case 'experiment_progress':
      console.log(`Experiment ${data.experiment_id}: ${data.progress * 100}% complete`);
      break;
      
    case 'dashboard_update':
      updateDashboard(data.data);
      break;
  }
};

// Subscribe to specific experiment
ws.send(JSON.stringify({
  type: 'subscribe_experiment',
  experiment_id: 'exp_validation_001'
}));
```

## API Endpoints Reference

### Task Submission

#### POST /submit
Submit single ARC task solution for evaluation.

**Request Body:**
- `task_id` (string): ARC task identifier
- `predicted_output` (array): 2D grid solution
- `strategy` (enum): Solving strategy used
- `confidence_score` (float): Prediction confidence (0.0-1.0)
- `attempt_number` (int): Attempt number (1-2)
- `metadata` (object): Additional context

**Response:**
- `submission_id` (string): Unique submission identifier
- `accuracy` (float): Pixel-level accuracy score
- `perfect_match` (bool): Exact solution match
- `processing_time_ms` (float): Evaluation time
- `evaluation_details` (object): Detailed metrics

### Batch Processing

#### POST /evaluate/batch
Process multiple task evaluations efficiently.

**Key Features:**
- Parallel processing for improved performance
- Real-time progress updates via WebSocket
- Experiment grouping and tracking
- Error isolation per task

### Experiment Tracking

#### GET /experiments/{experiment_id}/status
Monitor experiment progress and performance.

**Response Includes:**
- Current progress percentage
- Completed vs total tasks
- Average accuracy metrics
- Estimated completion time
- Performance statistics
- Error summaries

### Dashboard Metrics

#### GET /dashboard/metrics
Retrieve system performance metrics.

**Metrics Included:**
- Active experiments count
- Processing statistics
- Resource utilization
- Strategy performance rankings
- System health status

#### GET /strategies/performance
Analyze strategy effectiveness over time.

**Parameters:**
- `time_window`: Analysis period (1h, 6h, 24h, 7d, 30d)

**Analysis Includes:**
- Accuracy comparisons
- Processing efficiency
- Cost analysis
- Performance trends

## WebSocket Protocol

### Connection Authentication

WebSocket connections require JWT authentication:

```javascript
// Option 1: Query parameter
const ws = new WebSocket(`ws://localhost:8000/api/v1/evaluation/ws?token=${token}`);

// Option 2: Authorization header (if supported by client)
const ws = new WebSocket('ws://localhost:8000/api/v1/evaluation/ws', {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### Message Types

#### Server → Client Messages

**Dashboard Updates** (every 500ms):
```json
{
  "type": "dashboard_update",
  "timestamp": "2024-01-01T14:30:22Z",
  "data": {
    "active_experiments": 3,
    "tasks_processed": 1247,
    "average_accuracy": 0.834,
    "resource_utilization": {"cpu": 45.2, "memory": 67.8},
    "system_health": {"evaluation_service": "healthy"}
  }
}
```

**Task Submissions**:
```json
{
  "type": "task_submitted",
  "submission_id": "sub_arc_2024_001_123",
  "task_id": "arc_2024_001",
  "accuracy": 0.94,
  "perfect_match": false,
  "timestamp": "2024-01-01T14:30:22Z"
}
```

**Experiment Progress**:
```json
{
  "type": "experiment_progress",
  "experiment_id": "exp_batch_001",
  "completed_tasks": 45,
  "total_tasks": 100,
  "progress": 0.45,
  "current_accuracy": 0.823
}
```

#### Client → Server Messages

**Subscribe to Experiment**:
```json
{
  "type": "subscribe_experiment",
  "experiment_id": "exp_batch_001"
}
```

**Heartbeat/Ping**:
```json
{"type": "ping"}
```

## Error Handling

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid input data, malformed JSON |
| 401 | Unauthorized | Missing or invalid JWT token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Task or experiment not found |
| 429 | Rate Limited | Too many requests |
| 500 | Internal Error | Server-side processing error |

### Error Response Format

```json
{
  "detail": "Descriptive error message",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-01T14:30:22Z",
  "request_id": "req_12345",
  "additional_info": {
    "field_errors": ["Invalid grid values"],
    "suggestions": ["Check grid format documentation"]
  }
}
```

### Common Error Scenarios

#### Authentication Errors

```json
{
  "detail": "JWT token has expired",
  "error_code": "TOKEN_EXPIRED",
  "expires_at": "2024-01-01T14:00:00Z"
}
```

**Solution**: Refresh token or obtain new authentication

#### Validation Errors

```json
{
  "detail": "Predicted output grid contains invalid values",
  "error_code": "INVALID_GRID_VALUES",
  "invalid_values": [10, -1],
  "valid_range": "0-9"
}
```

**Solution**: Ensure grid values are integers 0-9

#### Rate Limiting

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60,
  "current_limit": "60 requests per minute"
}
```

**Solution**: Implement exponential backoff or use batch endpoints

## Best Practices

### 1. Authentication Management

```python
import requests
from datetime import datetime, timedelta

class APIClient:
    def __init__(self, base_url, user_id):
        self.base_url = base_url
        self.user_id = user_id
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
    
    def authenticate(self):
        """Get new authentication tokens"""
        response = requests.post(f"{self.base_url}/auth/token", 
                               json={"user_id": self.user_id})
        data = response.json()
        
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires = datetime.now() + timedelta(seconds=data["expires_in"])
    
    def get_headers(self):
        """Get authenticated request headers"""
        if not self.access_token or datetime.now() >= self.token_expires:
            self.authenticate()
        
        return {"Authorization": f"Bearer {self.access_token}"}
```

### 2. Batch Processing Optimization

```python
def submit_tasks_efficiently(tasks, batch_size=50):
    """Submit tasks in optimized batches"""
    
    # Group tasks into batches
    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
    
    experiment_ids = []
    
    for i, batch in enumerate(batches):
        evaluations = [
            {
                "task_id": task["id"],
                "predicted_output": task["prediction"],
                "confidence": task["confidence"],
                "attempt_number": 1
            }
            for task in batch
        ]
        
        response = requests.post(
            f"{base_url}/evaluate/batch",
            headers=client.get_headers(),
            json={
                "evaluations": evaluations,
                "strategy": "ENSEMBLE",
                "experiment_id": f"batch_{i}_{int(time.time())}",
                "parallel_processing": True,
                "timeout_seconds": 300
            }
        )
        
        if response.status_code == 200:
            experiment_ids.append(response.json()["experiment_id"])
    
    return experiment_ids
```

### 3. Real-time Monitoring

```python
import asyncio
import websockets
import json

async def monitor_experiments(token, experiment_ids):
    """Monitor multiple experiments via WebSocket"""
    
    uri = f"ws://localhost:8000/api/v1/evaluation/ws?token={token}"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to all experiments
        for exp_id in experiment_ids:
            await websocket.send(json.dumps({
                "type": "subscribe_experiment",
                "experiment_id": exp_id
            }))
        
        # Monitor progress
        experiment_status = {exp_id: {"completed": False, "progress": 0.0} 
                           for exp_id in experiment_ids}
        
        while not all(exp["completed"] for exp in experiment_status.values()):
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "experiment_progress":
                exp_id = data["experiment_id"]
                if exp_id in experiment_status:
                    experiment_status[exp_id]["progress"] = data["progress"]
                    print(f"Experiment {exp_id}: {data['progress']*100:.1f}% complete")
            
            elif data["type"] == "experiment_completed":
                exp_id = data["experiment_id"]
                if exp_id in experiment_status:
                    experiment_status[exp_id]["completed"] = True
                    print(f"Experiment {exp_id} completed with accuracy: {data['final_accuracy']}")
```

### 4. Error Recovery and Retry Logic

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """Decorator for API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    response = func(*args, **kwargs)
                    
                    if response.status_code == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        time.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    return response
                    
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    print(f"Request failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def submit_task_with_retry(task_data):
    return requests.post(f"{base_url}/submit", 
                        headers=client.get_headers(), 
                        json=task_data)
```

## Performance Optimization

### 1. Rate Limiting Awareness

- **Standard Users**: 60 requests/minute for individual submissions
- **Batch Operations**: More efficient for multiple tasks
- **WebSocket**: Real-time updates without polling overhead

### 2. Connection Management

```python
# Use connection pooling for HTTP requests
session = requests.Session()
session.headers.update(client.get_headers())

# Keep WebSocket connections alive
async def keep_alive(websocket):
    while True:
        await websocket.send(json.dumps({"type": "ping"}))
        await asyncio.sleep(30)  # Ping every 30 seconds
```

### 3. Data Optimization

- Use appropriate confidence scores (avoid unnecessary precision)
- Minimize metadata payload size
- Compress large grid predictions when possible

## Security Considerations

### 1. Token Security

- Store tokens securely (avoid localStorage for sensitive applications)
- Use HttpOnly cookies for web applications
- Implement proper token refresh logic
- Never log tokens in application logs

### 2. Input Validation

- Validate grid dimensions and values client-side
- Implement request size limits
- Sanitize metadata fields
- Use HTTPS in production

### 3. Rate Limiting Compliance

- Implement client-side rate limiting
- Use exponential backoff for retries
- Monitor API usage patterns
- Consider caching for repeated requests

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. WebSocket Connection Failures

**Problem**: WebSocket connections are rejected or disconnected

**Solutions**:
- Verify JWT token validity
- Check connection pool capacity
- Ensure proper authentication method
- Monitor network connectivity

#### 2. Batch Processing Timeouts

**Problem**: Large batches timeout before completion

**Solutions**:
- Reduce batch size (recommended: 25-50 tasks)
- Increase timeout_seconds parameter
- Use parallel_processing: true
- Monitor system resource usage

#### 3. High Error Rates

**Problem**: Many tasks fail evaluation

**Solutions**:
- Validate grid format before submission
- Check task_id format (arc_YYYY_NNN)
- Ensure grid values are integers 0-9
- Verify grid dimensions match expected output

#### 4. Performance Issues

**Problem**: Slow API response times

**Solutions**:
- Use batch endpoints for multiple tasks
- Implement connection pooling
- Monitor system metrics via /dashboard/metrics
- Consider load balancing for high traffic

### Debug Information

Enable debug logging to troubleshoot issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add request/response logging
def log_request_response(response):
    print(f"Request: {response.request.method} {response.request.url}")
    print(f"Response: {response.status_code} {response.text}")
    return response
```

## Migration and Updates

### API Versioning

The API uses semantic versioning:
- **Major versions** (v1 → v2): Breaking changes requiring code updates
- **Minor versions** (v1.1 → v1.2): New features, backward compatible
- **Patch versions** (v1.1.0 → v1.1.1): Bug fixes, fully compatible

### Deprecation Policy

- Deprecated endpoints remain available for 6 months
- New endpoints are marked as "beta" for 1 month
- Breaking changes are announced 3 months in advance
- Migration guides provided for major version updates

## Support and Resources

### Documentation Links

- [OpenAPI/Swagger Documentation](http://localhost:8000/docs)
- [ReDoc Documentation](http://localhost:8000/redoc)
- [Architecture Guide](../architecture/index.md)
- [Test Strategy](../architecture/test-strategy.md)

### Getting Help

1. Check this usage guide first
2. Review the interactive API documentation
3. Search existing GitHub issues
4. Create new issue with reproduction steps
5. Contact development team for critical issues

### Example Applications

- [Python Client Library](../examples/python-client.py)
- [JavaScript Dashboard](../examples/dashboard.html)
- [Batch Processing Script](../examples/batch-processor.py)
- [WebSocket Monitor](../examples/websocket-monitor.js)