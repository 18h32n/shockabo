# 11. Error Handling

## Error Response Format

```json
{
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "Task with ID 'arc_999' not found",
    "details": {
      "task_id": "arc_999",
      "search_scope": "training"
    },
    "timestamp": "2024-11-28T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| TASK_NOT_FOUND | 404 | Requested task does not exist |
| INVALID_SUBMISSION | 400 | Submission format is invalid |
| STRATEGY_UNAVAILABLE | 503 | Strategy temporarily unavailable |
| BUDGET_EXCEEDED | 402 | Task would exceed budget limit |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Unexpected server error |

## Error Handling Patterns

```python
class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
        error_code = "INTERNAL_ERROR"
        status_code = 500
        details = {}
        
        if isinstance(exc, TaskNotFoundError):
            error_code = "TASK_NOT_FOUND"
            status_code = 404
            details = {"task_id": exc.task_id}
        elif isinstance(exc, ValidationError):
            error_code = "INVALID_SUBMISSION"
            status_code = 400
            details = exc.errors()
        elif isinstance(exc, BudgetExceededError):
            error_code = "BUDGET_EXCEEDED"
            status_code = 402
            details = {"estimated_cost": exc.cost}
            
        # Log error with context
        logger.error(
            f"API error: {error_code}",
            extra={
                "request_id": request.state.request_id,
                "path": request.url.path,
                "method": request.method,
                "error_details": details
            }
        )
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": error_code,
                    "message": str(exc),
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request.state.request_id
                }
            }
        )
```

## Recovery Strategies

```python
class RecoveryManager:
    """Automated error recovery"""
    
    async def recover_from_failure(
        self, 
        task_id: str, 
        error: Exception
    ) -> Optional[TaskSubmission]:
        recovery_strategies = [
            self._retry_with_backoff,
            self._fallback_to_simple_strategy,
            self._use_cached_similar_result,
            self._return_partial_result
        ]
        
        for strategy in recovery_strategies:
            try:
                result = await strategy(task_id, error)
                if result:
                    logger.info(f"Recovered using {strategy.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                
        return None
```
