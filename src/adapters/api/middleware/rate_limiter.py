"""Advanced rate limiting middleware for FastAPI with DoS protection.

This module provides comprehensive rate limiting with different limits for different
endpoint types, burst protection, and configurable backends (memory/Redis).
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting middleware."""

    # Evaluation endpoints (more restrictive)
    evaluation_requests_per_minute: int = 100
    evaluation_requests_per_hour: int = 1000
    evaluation_burst_size: int = 10

    # Dashboard endpoints (more permissive)
    dashboard_requests_per_minute: int = 1000
    dashboard_requests_per_hour: int = 10000
    dashboard_burst_size: int = 50

    # WebSocket connections
    websocket_connections_per_ip: int = 5

    # Authentication endpoints
    auth_requests_per_minute: int = 20
    auth_requests_per_hour: int = 200

    # Global settings
    enable_rate_limiting: bool = True
    redis_url: str | None = None
    use_redis_backend: bool = False

    # Advanced protection
    enable_adaptive_limiting: bool = True
    suspicious_threshold_multiplier: float = 2.0  # Reduce limits by this factor for suspicious IPs
    whitelist_ips: list[str] = field(default_factory=list)
    blacklist_ips: list[str] = field(default_factory=list)


@dataclass
class RateLimit:
    """Rate limit configuration for a specific endpoint type."""
    requests_per_minute: int
    requests_per_hour: int
    burst_size: int
    window_size_minutes: int = 1


class TokenBucket:
    """Token bucket implementation for rate limiting with burst support."""

    def __init__(self, capacity: int, refill_rate: float, initial_tokens: int | None = None):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.tokens = initial_tokens or capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            bool: True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time until enough tokens are available."""
        if self.tokens >= tokens:
            return 0.0

        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate


class MemoryRateLimitBackend:
    """In-memory rate limiting backend with sliding windows."""

    def __init__(self):
        self.windows: dict[str, deque] = defaultdict(deque)
        self.token_buckets: dict[str, TokenBucket] = {}
        self.suspicious_ips: set[str] = set()
        self.connection_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def is_rate_limited(
        self,
        key: str,
        limit: RateLimit,
        is_suspicious: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if a request should be rate limited.
        
        Args:
            key: Unique identifier for the rate limit (usually IP + endpoint)
            limit: Rate limit configuration
            is_suspicious: Whether the IP is considered suspicious
            
        Returns:
            Tuple of (is_limited, headers_dict)
        """
        async with self._lock:
            now = time.time()

            # Adjust limits for suspicious IPs
            effective_rpm = limit.requests_per_minute
            effective_rph = limit.requests_per_hour
            effective_burst = limit.burst_size

            if is_suspicious:
                effective_rpm = int(effective_rpm / 2)
                effective_rph = int(effective_rph / 2)
                effective_burst = max(1, int(effective_burst / 2))

            # Initialize or get token bucket
            bucket_key = f"{key}_bucket"
            if bucket_key not in self.token_buckets:
                # Tokens per second = requests per minute / 60
                refill_rate = effective_rpm / 60.0
                self.token_buckets[bucket_key] = TokenBucket(
                    capacity=effective_burst,
                    refill_rate=refill_rate
                )

            bucket = self.token_buckets[bucket_key]

            # Check token bucket (for burst protection)
            can_consume = await bucket.consume(1)
            if not can_consume:
                retry_after = bucket.get_wait_time(1)
                return True, {
                    "X-RateLimit-Limit": str(effective_rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                    "Retry-After": str(int(retry_after)),
                }

            # Sliding window for minute-based limiting
            minute_key = f"{key}_minute"
            minute_window = self.windows[minute_key]

            # Remove old entries (older than 1 minute)
            minute_cutoff = now - 60
            while minute_window and minute_window[0] < minute_cutoff:
                minute_window.popleft()

            # Check minute limit
            if len(minute_window) >= effective_rpm:
                retry_after = minute_window[0] + 60 - now
                return True, {
                    "X-RateLimit-Limit": str(effective_rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(minute_window[0] + 60)),
                    "Retry-After": str(max(1, int(retry_after))),
                }

            # Sliding window for hour-based limiting
            hour_key = f"{key}_hour"
            hour_window = self.windows[hour_key]

            # Remove old entries (older than 1 hour)
            hour_cutoff = now - 3600
            while hour_window and hour_window[0] < hour_cutoff:
                hour_window.popleft()

            # Check hour limit
            if len(hour_window) >= effective_rph:
                retry_after = hour_window[0] + 3600 - now
                return True, {
                    "X-RateLimit-Limit": str(effective_rph),
                    "X-RateLimit-Remaining": str(effective_rph - len(hour_window)),
                    "X-RateLimit-Reset": str(int(hour_window[0] + 3600)),
                    "Retry-After": str(max(1, int(retry_after))),
                }

            # Record the request
            minute_window.append(now)
            hour_window.append(now)

            # Calculate remaining requests
            minute_remaining = max(0, effective_rpm - len(minute_window))
            hour_remaining = max(0, effective_rph - len(hour_window))

            return False, {
                "X-RateLimit-Limit": str(effective_rpm),
                "X-RateLimit-Remaining": str(min(minute_remaining, hour_remaining)),
                "X-RateLimit-Reset": str(int(now + 60)),
            }

    async def track_websocket_connection(self, ip: str, max_connections: int = 5) -> bool:
        """Track WebSocket connection and check if limit exceeded."""
        async with self._lock:
            current_count = self.connection_counts.get(ip, 0)
            if current_count >= max_connections:
                return False
            self.connection_counts[ip] = current_count + 1
            return True

    async def release_websocket_connection(self, ip: str):
        """Release a WebSocket connection count."""
        async with self._lock:
            if ip in self.connection_counts:
                self.connection_counts[ip] = max(0, self.connection_counts[ip] - 1)
                if self.connection_counts[ip] == 0:
                    del self.connection_counts[ip]

    async def mark_suspicious(self, ip: str):
        """Mark an IP as suspicious for adaptive rate limiting."""
        async with self._lock:
            self.suspicious_ips.add(ip)
            logger.warning("ip_marked_suspicious", ip=ip)

    async def is_suspicious(self, ip: str) -> bool:
        """Check if an IP is marked as suspicious."""
        return ip in self.suspicious_ips


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for comprehensive rate limiting."""

    def __init__(self, app: Callable, config: RateLimitConfig):
        super().__init__(app)
        self.config = config
        self.backend = MemoryRateLimitBackend()

        # Define rate limits for different endpoint patterns
        self.rate_limits = {
            "evaluation": RateLimit(
                requests_per_minute=config.evaluation_requests_per_minute,
                requests_per_hour=config.evaluation_requests_per_hour,
                burst_size=config.evaluation_burst_size,
            ),
            "dashboard": RateLimit(
                requests_per_minute=config.dashboard_requests_per_minute,
                requests_per_hour=config.dashboard_requests_per_hour,
                burst_size=config.dashboard_burst_size,
            ),
            "auth": RateLimit(
                requests_per_minute=config.auth_requests_per_minute,
                requests_per_hour=config.auth_requests_per_hour,
                burst_size=max(5, config.auth_requests_per_minute // 4),
            ),
            "default": RateLimit(
                requests_per_minute=config.dashboard_requests_per_minute,
                requests_per_hour=config.dashboard_requests_per_hour,
                burst_size=config.dashboard_burst_size,
            ),
        }

        logger.info(
            "rate_limit_middleware_initialized",
            evaluation_rpm=config.evaluation_requests_per_minute,
            dashboard_rpm=config.dashboard_requests_per_minute,
            auth_rpm=config.auth_requests_per_minute,
            use_redis=config.use_redis_backend,
        )

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    def get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type based on URL path."""
        path_lower = path.lower()

        if "/api/v1/evaluation" in path_lower:
            if "/submit" in path_lower or "/evaluate" in path_lower:
                return "evaluation"
            else:
                return "dashboard"

        if "/auth" in path_lower or "/token" in path_lower:
            return "auth"

        if "/dashboard" in path_lower or "/metrics" in path_lower or "/ws" in path_lower:
            return "dashboard"

        return "default"

    def create_rate_limit_key(self, ip: str, endpoint_type: str, path: str) -> str:
        """Create a unique key for rate limiting."""
        # Use hashed combination for privacy and efficiency
        combined = f"{ip}:{endpoint_type}:{path}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is in whitelist."""
        return ip in self.config.whitelist_ips

    async def is_ip_blacklisted(self, ip: str) -> bool:
        """Check if IP is in blacklist."""
        return ip in self.config.blacklist_ips

    async def detect_suspicious_behavior(self, request: Request, ip: str) -> bool:
        """Detect suspicious behavior patterns."""
        # Check if already marked suspicious
        if await self.backend.is_suspicious(ip):
            return True

        # Check for common attack patterns in user agent
        user_agent = request.headers.get("User-Agent", "").lower()
        suspicious_patterns = [
            "bot", "crawler", "spider", "scraper", "curl", "wget",
            "python-requests", "go-http-client", "java", "php"
        ]

        # Check for missing or suspicious user agents
        if not user_agent or any(pattern in user_agent for pattern in suspicious_patterns):
            return True

        # Check for missing Accept header (common in automated requests)
        if not request.headers.get("Accept"):
            return True

        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        if not self.config.enable_rate_limiting:
            return await call_next(request)

        # Extract client information
        client_ip = self.get_client_ip(request)
        endpoint_type = self.get_endpoint_type(request.url.path)

        # Check blacklist
        if await self.is_ip_blacklisted(client_ip):
            logger.warning("blacklisted_ip_blocked", ip=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "access_denied",
                    "message": "Access denied",
                },
            )

        # Skip rate limiting for whitelisted IPs
        if await self.is_ip_whitelisted(client_ip):
            return await call_next(request)

        # Handle WebSocket connections separately
        if request.url.path.startswith("/api/v1/evaluation/ws"):
            if not await self.backend.track_websocket_connection(client_ip):
                logger.warning(
                    "websocket_connection_limit_exceeded",
                    ip=client_ip,
                    current_connections=self.backend.connection_counts.get(client_ip, 0)
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "too_many_websocket_connections",
                        "message": f"Maximum {self.config.websocket_connections_per_ip} WebSocket connections per IP allowed",
                    },
                )

            # Process WebSocket request
            try:
                response = await call_next(request)
                return response
            finally:
                await self.backend.release_websocket_connection(client_ip)

        # Get appropriate rate limit
        rate_limit = self.rate_limits.get(endpoint_type, self.rate_limits["default"])

        # Create rate limiting key
        rate_key = self.create_rate_limit_key(client_ip, endpoint_type, request.url.path)

        # Check for suspicious behavior
        is_suspicious = await self.detect_suspicious_behavior(request, client_ip)
        if is_suspicious and self.config.enable_adaptive_limiting:
            await self.backend.mark_suspicious(client_ip)

        # Check rate limits
        is_limited, headers = await self.backend.is_rate_limited(
            rate_key,
            rate_limit,
            is_suspicious
        )

        if is_limited:
            logger.warning(
                "rate_limit_exceeded",
                ip=client_ip,
                endpoint_type=endpoint_type,
                path=request.url.path,
                user_agent=request.headers.get("User-Agent", "unknown"),
                is_suspicious=is_suspicious,
            )

            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "endpoint_type": endpoint_type,
                },
                headers=headers,
            )
            return response

        # Process the request
        start_time = time.time()
        try:
            response = await call_next(request)

            # Add rate limit headers to successful responses
            for key, value in headers.items():
                response.headers[key] = value

            # Log successful request
            processing_time = time.time() - start_time
            logger.info(
                "request_processed",
                ip=client_ip,
                method=request.method,
                path=request.url.path,
                endpoint_type=endpoint_type,
                status_code=response.status_code,
                processing_time_ms=int(processing_time * 1000),
                remaining_requests=headers.get("X-RateLimit-Remaining", "unknown"),
            )

            return response

        except Exception as e:
            # Log failed request
            processing_time = time.time() - start_time
            logger.error(
                "request_failed",
                ip=client_ip,
                method=request.method,
                path=request.url.path,
                endpoint_type=endpoint_type,
                processing_time_ms=int(processing_time * 1000),
                error=str(e),
                exc_info=True,
            )
            raise
