"""Rate limiting middleware for FastAPI to prevent DoS attacks.

This module provides configurable rate limiting with different limits
for different endpoint types and proper HTTP response codes.
"""

import time
from collections import defaultdict, deque

import structlog
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, identifier: str) -> tuple[bool, dict]:
        """Check if request is allowed and return rate limit info.
        
        Args:
            identifier: Client identifier (IP address, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, rate_limit_headers)
        """
        current_time = time.time()
        client_requests = self.requests[identifier]

        # Remove old requests outside the window
        while client_requests and client_requests[0] <= current_time - self.window_seconds:
            client_requests.popleft()

        # Check if client is within limits
        current_count = len(client_requests)
        is_allowed = current_count < self.max_requests

        if is_allowed:
            client_requests.append(current_time)

        # Calculate reset time (next window)
        reset_time = int(current_time + self.window_seconds)
        remaining = max(0, self.max_requests - current_count - (1 if is_allowed else 0))

        headers = {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Window": str(self.window_seconds)
        }

        return is_allowed, headers


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limits: dict[str, RateLimiter] | None = None):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            rate_limits: Dictionary mapping path patterns to rate limiters
        """
        super().__init__(app)

        # Default rate limits if none provided
        self.rate_limits = rate_limits or {
            "/api/v1/evaluation/submit": RateLimiter(max_requests=100, window_seconds=60),  # 100/min for submissions
            "/api/v1/evaluation/evaluate/batch": RateLimiter(max_requests=10, window_seconds=60),  # 10/min for batch
            "/api/v1/evaluation/dashboard": RateLimiter(max_requests=1000, window_seconds=60),  # 1000/min for dashboard
            "/api/v1/evaluation/ws": RateLimiter(max_requests=50, window_seconds=60),  # 50/min for WebSocket
            "default": RateLimiter(max_requests=200, window_seconds=60)  # 200/min default
        }

        # Track violations for monitoring
        self.violations = defaultdict(int)
        self.total_requests = 0
        self.blocked_requests = 0

    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client identifier string
        """
        # Try to get user ID from JWT token first
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from src.utils.jwt_auth import get_jwt_manager
                jwt_manager = get_jwt_manager()
                token = auth_header.split(" ")[1]
                payload = jwt_manager.verify_token(token)
                user_id = payload.get("sub")
                if user_id:
                    return f"user:{user_id}"
            except Exception:
                pass  # Fall back to IP-based limiting

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"

        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip

        return f"ip:{client_ip}"

    def get_rate_limiter(self, path: str) -> RateLimiter:
        """Get appropriate rate limiter for the given path.
        
        Args:
            path: Request path
            
        Returns:
            RateLimiter instance
        """
        # Find matching rate limiter by checking path prefixes
        for pattern, limiter in self.rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limiter

        # Return default limiter
        return self.rate_limits["default"]

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in chain
            
        Returns:
            Response with rate limit headers
        """
        self.total_requests += 1

        # Skip rate limiting for health checks and internal endpoints
        if request.url.path.startswith("/health") or request.url.path.startswith("/docs"):
            return await call_next(request)

        # Get client identifier and rate limiter
        client_id = self.get_client_identifier(request)
        rate_limiter = self.get_rate_limiter(request.url.path)

        # Check rate limit
        is_allowed, headers = rate_limiter.is_allowed(client_id)

        if not is_allowed:
            self.blocked_requests += 1
            self.violations[client_id] += 1

            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                path=request.url.path,
                method=request.method,
                violations=self.violations[client_id]
            )

            # Return 429 Too Many Requests with headers
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_limiter.max_requests} per {rate_limiter.window_seconds}s",
                    "retry_after": headers["X-RateLimit-Reset"]
                },
                headers=headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        logger.debug(
            "request_processed",
            client_id=client_id,
            path=request.url.path,
            remaining=headers["X-RateLimit-Remaining"]
        )

        return response

    def get_statistics(self) -> dict:
        """Get rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting stats
        """
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / self.total_requests if self.total_requests > 0 else 0,
            "active_clients": len(set().union(*(rl.requests.keys() for rl in self.rate_limits.values()))),
            "violations_by_client": dict(self.violations),
            "rate_limits": {
                pattern: {
                    "max_requests": rl.max_requests,
                    "window_seconds": rl.window_seconds,
                    "active_clients": len(rl.requests)
                }
                for pattern, rl in self.rate_limits.items()
            }
        }


# Global rate limiting instance
_rate_limiting_middleware = None


def get_rate_limiting_middleware(custom_limits: dict[str, RateLimiter] | None = None) -> RateLimitingMiddleware:
    """Get rate limiting middleware instance.
    
    Args:
        custom_limits: Optional custom rate limits
        
    Returns:
        RateLimitingMiddleware instance
    """
    global _rate_limiting_middleware

    if _rate_limiting_middleware is None:
        import os

        # Load rate limits from environment variables
        evaluation_limit = int(os.environ.get("RATE_LIMIT_EVALUATION", "100"))
        batch_limit = int(os.environ.get("RATE_LIMIT_BATCH", "10"))
        dashboard_limit = int(os.environ.get("RATE_LIMIT_DASHBOARD", "1000"))
        websocket_limit = int(os.environ.get("RATE_LIMIT_WEBSOCKET", "50"))
        default_limit = int(os.environ.get("RATE_LIMIT_DEFAULT", "200"))
        window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

        rate_limits = custom_limits or {
            "/api/v1/evaluation/submit": RateLimiter(evaluation_limit, window_seconds),
            "/api/v1/evaluation/evaluate/batch": RateLimiter(batch_limit, window_seconds),
            "/api/v1/evaluation/dashboard": RateLimiter(dashboard_limit, window_seconds),
            "/api/v1/evaluation/ws": RateLimiter(websocket_limit, window_seconds),
            "default": RateLimiter(default_limit, window_seconds)
        }

        _rate_limiting_middleware = RateLimitingMiddleware(None, rate_limits)

        logger.info(
            "rate_limiting_initialized",
            evaluation_limit=evaluation_limit,
            batch_limit=batch_limit,
            dashboard_limit=dashboard_limit,
            default_limit=default_limit,
            window_seconds=window_seconds
        )

    return _rate_limiting_middleware
