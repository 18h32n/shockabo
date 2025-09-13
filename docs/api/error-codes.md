# API Error Codes Reference

## Overview

This document provides comprehensive documentation of all error codes returned by the ARC Evaluation Framework API, including descriptions, common causes, and troubleshooting steps.

## Error Response Format

All API errors follow a consistent JSON structure:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2024-01-01T14:30:22Z",
  "request_id": "req_12345_abcdef",
  "additional_info": {
    "field_errors": ["Specific field validation errors"],
    "suggestions": ["Recommended solutions"],
    "documentation_url": "https://docs.example.com/errors/CODE"
  }
}
```

## Authentication Errors (AUTH_*)

### AUTH_REQUIRED
**HTTP Status**: 401 Unauthorized  
**Description**: No authentication credentials provided  

**Example**:
```json
{
  "detail": "Authentication credentials required",
  "error_code": "AUTH_REQUIRED",
  "additional_info": {
    "suggestions": [
      "Include Authorization header with Bearer token",
      "Use query parameter ?token=<jwt_token> for WebSocket"
    ]
  }
}
```

**Solutions**:
- Include `Authorization: Bearer <token>` header
- Obtain valid JWT token via `/auth/token` endpoint
- For WebSocket: Use `?token=<jwt_token>` query parameter

### AUTH_INVALID_TOKEN
**HTTP Status**: 401 Unauthorized  
**Description**: Provided JWT token is malformed or invalid  

**Common Causes**:
- Corrupted token string
- Invalid JWT signature
- Token from different system/environment

**Solutions**:
- Verify token string integrity
- Obtain fresh token from authentication endpoint
- Check token issuer and audience claims

### TOKEN_EXPIRED
**HTTP Status**: 401 Unauthorized  
**Description**: JWT token has passed its expiration time  

**Example**:
```json
{
  "detail": "JWT token expired at 2024-01-01T14:00:00Z",
  "error_code": "TOKEN_EXPIRED",
  "additional_info": {
    "expired_at": "2024-01-01T14:00:00Z",
    "current_time": "2024-01-01T14:30:22Z",
    "suggestions": ["Obtain new access token", "Use refresh token if available"]
  }
}
```

**Solutions**:
- Request new access token
- Use refresh token if available
- Implement automatic token refresh logic

### INSUFFICIENT_PERMISSIONS
**HTTP Status**: 403 Forbidden  
**Description**: Valid authentication but insufficient permissions for requested operation  

**Solutions**:
- Verify user account permissions
- Contact administrator for access upgrade
- Check if endpoint requires specific user roles

## Validation Errors (VALIDATION_*)

### VALIDATION_ERROR
**HTTP Status**: 400 Bad Request  
**Description**: Generic validation error for request data  

**Example**:
```json
{
  "detail": "Request validation failed",
  "error_code": "VALIDATION_ERROR",
  "additional_info": {
    "field_errors": [
      "confidence_score must be between 0.0 and 1.0",
      "task_id format must match arc_YYYY_NNN"
    ]
  }
}
```

### INVALID_GRID_VALUES
**HTTP Status**: 400 Bad Request  
**Description**: Grid contains values outside the valid range (0-9)  

**Example**:
```json
{
  "detail": "Grid contains invalid values. Only integers 0-9 are allowed.",
  "error_code": "INVALID_GRID_VALUES",
  "additional_info": {
    "invalid_values": [10, -1, 15],
    "valid_range": "0-9",
    "positions": [[0, 1], [2, 3], [1, 0]]
  }
}
```

**Solutions**:
- Ensure all grid values are integers 0-9
- Validate grid data before submission
- Check for data conversion errors

### GRID_SIZE_MISMATCH
**HTTP Status**: 400 Bad Request  
**Description**: Predicted output grid dimensions don't match expected size  

**Example**:
```json
{
  "detail": "Grid size mismatch. Expected 3x3, got 3x4",
  "error_code": "GRID_SIZE_MISMATCH",
  "additional_info": {
    "expected_size": [3, 3],
    "actual_size": [3, 4],
    "suggestions": ["Verify task requirements", "Check grid construction logic"]
  }
}
```

**Solutions**:
- Verify expected output grid dimensions for the task
- Check grid construction and manipulation logic
- Ensure consistent row lengths in 2D arrays

### INVALID_TASK_ID
**HTTP Status**: 400 Bad Request  
**Description**: Task ID format is invalid  

**Expected Format**: `arc_YYYY_NNN` (e.g., `arc_2024_001`)

**Solutions**:
- Use correct task ID format
- Verify task ID exists in database
- Check for typos in task identifier

### INVALID_STRATEGY
**HTTP Status**: 400 Bad Request  
**Description**: Specified strategy type is not recognized  

**Valid Strategies**:
- `DIRECT_SOLVE`
- `PATTERN_MATCH`
- `TRANSFORMATION_SEARCH`
- `NEURAL_NETWORK`
- `ENSEMBLE`
- `CUSTOM`

### CONFIDENCE_OUT_OF_RANGE
**HTTP Status**: 400 Bad Request  
**Description**: Confidence score must be between 0.0 and 1.0  

**Solutions**:
- Normalize confidence values to 0.0-1.0 range
- Check confidence calculation logic
- Use default value if confidence unavailable

## Resource Errors (RESOURCE_*)

### TASK_NOT_FOUND
**HTTP Status**: 404 Not Found  
**Description**: Specified ARC task does not exist in the database  

**Example**:
```json
{
  "detail": "Task arc_2024_999 not found",
  "error_code": "TASK_NOT_FOUND",
  "additional_info": {
    "task_id": "arc_2024_999",
    "suggestions": [
      "Verify task ID format and spelling",
      "Check available tasks via task listing endpoint",
      "Ensure task is loaded in current environment"
    ]
  }
}
```

**Solutions**:
- Verify task ID spelling and format
- Check if task exists in current dataset
- Ensure database is properly populated

### EXPERIMENT_NOT_FOUND
**HTTP Status**: 404 Not Found  
**Description**: Specified experiment does not exist or is not accessible  

**Solutions**:
- Verify experiment ID is correct
- Check if experiment belongs to authenticated user
- Ensure experiment hasn't been deleted

### SUBMISSION_NOT_FOUND
**HTTP Status**: 404 Not Found  
**Description**: Submission record not found  

**Solutions**:
- Verify submission ID format
- Check if submission was successful
- Ensure proper submission tracking

## Rate Limiting Errors (RATE_*)

### RATE_LIMIT_EXCEEDED
**HTTP Status**: 429 Too Many Requests  
**Description**: API rate limit has been exceeded  

**Example**:
```json
{
  "detail": "Rate limit exceeded. 60 requests per minute allowed.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "additional_info": {
    "current_limit": "60 requests per minute",
    "retry_after": 45,
    "reset_time": "2024-01-01T14:31:00Z",
    "suggestions": [
      "Implement exponential backoff",
      "Use batch endpoints for multiple tasks",
      "Consider upgrading to higher rate limits"
    ]
  }
}
```

**Rate Limits by Endpoint**:
- `/submit`: 60 requests/minute (standard), 300 requests/minute (premium)
- `/evaluate/batch`: 10 requests/minute
- `/dashboard/metrics`: 60 requests/minute
- WebSocket connections: 100 connections/minute

**Solutions**:
- Implement exponential backoff retry logic
- Use batch endpoints for multiple operations
- Cache responses when appropriate
- Upgrade to higher rate limit tier

### CONCURRENT_LIMIT_EXCEEDED
**HTTP Status**: 429 Too Many Requests  
**Description**: Too many concurrent operations  

**Limits**:
- Concurrent batch operations: 3 per user
- WebSocket connections: 1000 total, 10 per user
- Concurrent experiments: 5 per user

**Solutions**:
- Wait for existing operations to complete
- Implement queue management
- Use sequential processing for large workloads

## Processing Errors (PROCESSING_*)

### PROCESSING_TIMEOUT
**HTTP Status**: 408 Request Timeout  
**Description**: Request processing exceeded maximum allowed time  

**Timeouts**:
- Individual task evaluation: 30 seconds
- Batch processing: 1 hour (configurable)
- WebSocket message processing: 5 seconds

**Solutions**:
- Reduce batch size
- Optimize input data complexity
- Increase timeout parameter if available
- Break large operations into smaller chunks

### EVALUATION_FAILED
**HTTP Status**: 500 Internal Server Error  
**Description**: Task evaluation process encountered an error  

**Example**:
```json
{
  "detail": "Evaluation failed due to grid processing error",
  "error_code": "EVALUATION_FAILED",
  "additional_info": {
    "task_id": "arc_2024_001",
    "stage": "grid_comparison",
    "internal_error": "Matrix dimension mismatch",
    "suggestions": ["Retry request", "Check grid format", "Contact support if persistent"]
  }
}
```

**Solutions**:
- Retry the request (may be transient)
- Verify input data format
- Check system status via health endpoints
- Contact support for persistent errors

### RESOURCE_EXHAUSTED
**HTTP Status**: 503 Service Unavailable  
**Description**: System resources temporarily unavailable  

**Solutions**:
- Retry after delay
- Reduce request complexity
- Try during off-peak hours
- Use smaller batch sizes

## Batch Processing Errors (BATCH_*)

### EMPTY_BATCH
**HTTP Status**: 400 Bad Request  
**Description**: Batch evaluation request contains no tasks  

**Solutions**:
- Include at least one task in evaluations array
- Verify batch construction logic
- Check for filtering that removes all tasks

### BATCH_SIZE_EXCEEDED
**HTTP Status**: 400 Bad Request  
**Description**: Batch contains more tasks than maximum allowed  

**Example**:
```json
{
  "detail": "Batch size (150) exceeds maximum (100)",
  "error_code": "BATCH_SIZE_EXCEEDED",
  "additional_info": {
    "batch_size": 150,
    "max_batch_size": 100,
    "suggestions": ["Split into smaller batches", "Use pagination"]
  }
}
```

**Limits**:
- Maximum batch size: 100 tasks
- Recommended batch size: 25-50 tasks for optimal performance

**Solutions**:
- Split large batches into smaller chunks
- Process batches sequentially or with controlled concurrency

### BATCH_PROCESSING_FAILED
**HTTP Status**: 500 Internal Server Error  
**Description**: Batch processing encountered critical error  

**Possible Causes**:
- System resource exhaustion
- Database connectivity issues
- Individual task processing failures

**Solutions**:
- Check individual task validity
- Retry with smaller batch size
- Monitor system resource usage
- Review error logs for specific failures

## WebSocket Errors (WS_*)

### WS_CONNECTION_REJECTED
**WebSocket Close Code**: 1013 Try Again Later  
**Description**: WebSocket connection rejected due to capacity limits  

**Solutions**:
- Retry connection after delay
- Check connection pool statistics
- Close unused connections
- Contact administrator about capacity

### WS_AUTH_REQUIRED
**WebSocket Close Code**: 1008 Policy Violation  
**Description**: WebSocket authentication failed or missing  

**Solutions**:
- Provide valid JWT token in query parameter or header
- Ensure token hasn't expired
- Verify token format and encoding

### WS_PROTOCOL_ERROR
**WebSocket Close Code**: 1002 Protocol Error  
**Description**: Invalid message format or protocol violation  

**Solutions**:
- Ensure messages are valid JSON
- Follow documented message format
- Check message type fields
- Verify subscription message format

## System Errors (SYSTEM_*)

### SYSTEM_MAINTENANCE
**HTTP Status**: 503 Service Unavailable  
**Description**: System is temporarily unavailable due to maintenance  

**Solutions**:
- Wait for maintenance window to complete
- Check system status page
- Subscribe to maintenance notifications

### DATABASE_ERROR
**HTTP Status**: 503 Service Unavailable  
**Description**: Database connectivity or query error  

**Solutions**:
- Retry request after brief delay
- Check system health endpoints
- Contact support for persistent issues

### CACHE_ERROR
**HTTP Status**: 503 Service Unavailable  
**Description**: Cache system unavailable  

**Impact**: Performance may be degraded but functionality maintained

**Solutions**:
- Requests will be slower but functional
- No action required from client
- Monitor system status for resolution

## Error Handling Best Practices

### 1. Implement Retry Logic

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    response = func(*args, **kwargs)
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', base_delay))
                        time.sleep(retry_after)
                        continue
                    
                    # Handle server errors with exponential backoff
                    if response.status_code >= 500:
                        if attempt < max_retries - 1:
                            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                            time.sleep(delay)
                            continue
                    
                    return response
                    
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator
```

### 2. Parse Error Responses

```python
def handle_api_error(response):
    """Parse API error response and extract actionable information"""
    try:
        error_data = response.json()
        error_code = error_data.get('error_code', 'UNKNOWN_ERROR')
        detail = error_data.get('detail', 'Unknown error occurred')
        suggestions = error_data.get('additional_info', {}).get('suggestions', [])
        
        print(f"API Error [{error_code}]: {detail}")
        if suggestions:
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
                
        return {
            'code': error_code,
            'message': detail,
            'suggestions': suggestions,
            'retriable': is_retriable_error(error_code)
        }
    except:
        return {'code': 'PARSE_ERROR', 'message': response.text, 'retriable': False}

def is_retriable_error(error_code):
    """Determine if error is worth retrying"""
    retriable_codes = {
        'RATE_LIMIT_EXCEEDED',
        'PROCESSING_TIMEOUT',
        'RESOURCE_EXHAUSTED',
        'DATABASE_ERROR',
        'SYSTEM_MAINTENANCE'
    }
    return error_code in retriable_codes
```

### 3. Monitor Error Patterns

```python
from collections import defaultdict
import logging

class ErrorTracker:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = []
    
    def record_error(self, error_code, endpoint, timestamp=None):
        """Record error occurrence for pattern analysis"""
        timestamp = timestamp or time.time()
        self.error_counts[error_code] += 1
        self.error_history.append({
            'code': error_code,
            'endpoint': endpoint,
            'timestamp': timestamp
        })
        
        # Alert on high error rates
        if self.error_counts[error_code] > 10:
            logging.warning(f"High error rate for {error_code}: {self.error_counts[error_code]} occurrences")
    
    def get_error_summary(self):
        """Get summary of error patterns"""
        return dict(self.error_counts)
```

## Support and Escalation

### When to Contact Support

1. **Persistent System Errors** (5xx codes that continue after retries)
2. **Authentication Issues** not resolved by token refresh
3. **Data Corruption** or unexpected evaluation results
4. **Performance Degradation** beyond normal operational parameters
5. **Rate Limit Increases** for legitimate high-volume usage

### Error Reporting Template

When reporting errors, include:

```
Error Code: [ERROR_CODE]
HTTP Status: [STATUS_CODE]
Request ID: [REQUEST_ID]
Timestamp: [ISO_TIMESTAMP]
Endpoint: [API_ENDPOINT]
Request Payload: [SANITIZED_JSON]
Response Body: [ERROR_RESPONSE]
Reproduction Steps: [DETAILED_STEPS]
Environment: [DEV/STAGING/PROD]
```

### Emergency Contact

For critical production issues:
- Email: api-support@example.com
- Slack: #api-alerts
- On-call: +1-555-0199 (24/7 for severity 1 issues)

## Monitoring and Alerting

### Recommended Monitoring

1. **Error Rate Thresholds**:
   - 4xx errors > 5% of requests
   - 5xx errors > 1% of requests
   - Authentication failures > 2% of requests

2. **Response Time Alerts**:
   - Individual submissions > 10 seconds
   - Batch processing > 30 minutes
   - Dashboard metrics > 2 seconds

3. **Rate Limiting Alerts**:
   - Approaching rate limits (80% of quota)
   - Frequent rate limit violations
   - Unusual traffic patterns

### Health Check Integration

```python
async def health_check():
    """Comprehensive API health verification"""
    health_status = {
        'status': 'healthy',
        'checks': {},
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Test authentication
        auth_response = requests.post(f"{base_url}/auth/token", json={"user_id": "health_check"})
        health_status['checks']['authentication'] = 'healthy' if auth_response.status_code == 200 else 'unhealthy'
        
        # Test task submission
        token = auth_response.json().get('access_token')
        headers = {'Authorization': f'Bearer {token}'}
        
        test_submission = {
            'task_id': 'arc_test_001',
            'predicted_output': [[1, 0], [0, 1]],
            'strategy': 'DIRECT_SOLVE',
            'confidence_score': 0.5,
            'attempt_number': 1
        }
        
        submit_response = requests.post(f"{base_url}/submit", headers=headers, json=test_submission)
        health_status['checks']['task_submission'] = 'healthy' if submit_response.status_code in [200, 404] else 'unhealthy'
        
        # Test metrics endpoint
        metrics_response = requests.get(f"{base_url}/dashboard/metrics", headers=headers)
        health_status['checks']['dashboard'] = 'healthy' if metrics_response.status_code == 200 else 'unhealthy'
        
        # Determine overall status
        if any(status == 'unhealthy' for status in health_status['checks'].values()):
            health_status['status'] = 'degraded'
            
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['error'] = str(e)
    
    return health_status
```