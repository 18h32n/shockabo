"""Tests for rate limiting middleware functionality."""

import asyncio
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adapters.api.middleware.rate_limiter import (
    MemoryRateLimitBackend,
    RateLimit,
    RateLimitConfig,
    RateLimitMiddleware,
    TokenBucket,
)


class TestTokenBucket:
    """Test token bucket implementation."""

    @pytest.mark.asyncio
    async def test_token_bucket_basic_consumption(self):
        """Test basic token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)  # 1 token per second

        # Should be able to consume all initial tokens
        for _ in range(10):
            assert await bucket.consume(1) is True

        # Should fail to consume more
        assert await bucket.consume(1) is False

    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test token bucket refill over time."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)  # 10 tokens per second

        # Consume all tokens
        for _ in range(5):
            await bucket.consume(1)

        # Should fail immediately
        assert await bucket.consume(1) is False

        # Wait a bit and should be able to consume again
        await asyncio.sleep(0.2)  # 0.2 seconds = 2 tokens at 10/sec
        assert await bucket.consume(1) is True
        assert await bucket.consume(1) is True
        assert await bucket.consume(1) is False  # Only 2 tokens refilled

    def test_token_bucket_wait_time(self):
        """Test wait time calculation."""
        bucket = TokenBucket(capacity=5, refill_rate=2.0)  # 2 tokens per second

        # Consume all tokens
        bucket.tokens = 0

        # Need 3 tokens, should take 1.5 seconds
        wait_time = bucket.get_wait_time(3)
        assert wait_time == 1.5


class TestMemoryRateLimitBackend:
    """Test memory-based rate limiting backend."""

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        backend = MemoryRateLimitBackend()
        limit = RateLimit(
            requests_per_minute=5,
            requests_per_hour=10,
            burst_size=2
        )

        # Should allow initial requests within burst
        is_limited, headers = await backend.is_rate_limited("test_key", limit)
        assert is_limited is False
        assert "X-RateLimit-Limit" in headers

        is_limited, headers = await backend.is_rate_limited("test_key", limit)
        assert is_limited is False

        # Third request should be rate limited (burst_size=2)
        is_limited, headers = await backend.is_rate_limited("test_key", limit)
        assert is_limited is True
        assert "Retry-After" in headers

    @pytest.mark.asyncio
    async def test_suspicious_ip_limiting(self):
        """Test adaptive limiting for suspicious IPs."""
        backend = MemoryRateLimitBackend()
        limit = RateLimit(
            requests_per_minute=10,
            requests_per_hour=20,
            burst_size=5
        )

        # Mark IP as suspicious
        await backend.mark_suspicious("suspicious_ip")

        # Should have reduced limits
        key = "suspicious_ip_test"

        # With suspicious marking, burst should be halved (5 -> 2)
        # Allow 2 requests
        for _ in range(2):
            is_limited, _ = await backend.is_rate_limited(key, limit, is_suspicious=True)
            assert is_limited is False

        # Third should be limited
        is_limited, _ = await backend.is_rate_limited(key, limit, is_suspicious=True)
        assert is_limited is True

    @pytest.mark.asyncio
    async def test_websocket_connection_tracking(self):
        """Test WebSocket connection tracking."""
        backend = MemoryRateLimitBackend()

        # Should allow up to 5 connections per IP
        for i in range(5):
            allowed = await backend.track_websocket_connection("test_ip")
            assert allowed is True

        # 6th connection should be rejected
        allowed = await backend.track_websocket_connection("test_ip")
        assert allowed is False

        # Release a connection
        await backend.release_websocket_connection("test_ip")

        # Should allow connection again
        allowed = await backend.track_websocket_connection("test_ip")
        assert allowed is True


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    def create_test_app(self, config: RateLimitConfig) -> FastAPI:
        """Create test FastAPI app with rate limiting."""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.get("/api/v1/evaluation/submit")
        async def evaluation_endpoint():
            return {"status": "ok"}

        @app.get("/api/v1/evaluation/dashboard/metrics")
        async def dashboard_endpoint():
            return {"metrics": "ok"}

        @app.post("/auth/token")
        async def auth_endpoint():
            return {"token": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        return app

    def test_evaluation_endpoint_rate_limiting(self):
        """Test rate limiting on evaluation endpoints."""
        config = RateLimitConfig(
            evaluation_requests_per_minute=2,
            evaluation_requests_per_hour=10,
            evaluation_burst_size=1,
            enable_rate_limiting=True,
        )

        app = self.create_test_app(config)
        client = TestClient(app)

        # First request should succeed
        response = client.get("/api/v1/evaluation/submit")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers

        # Second request should be rate limited (burst_size=1)
        response = client.get("/api/v1/evaluation/submit")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

        error_data = response.json()
        assert error_data["error"] == "rate_limit_exceeded"
        assert error_data["endpoint_type"] == "evaluation"

    def test_dashboard_endpoint_more_permissive(self):
        """Test that dashboard endpoints have higher limits."""
        config = RateLimitConfig(
            evaluation_requests_per_minute=1,
            evaluation_burst_size=1,
            dashboard_requests_per_minute=10,
            dashboard_burst_size=5,
            enable_rate_limiting=True,
        )

        app = self.create_test_app(config)
        client = TestClient(app)

        # Dashboard endpoint should allow more requests
        for _ in range(3):  # More than evaluation burst_size
            response = client.get("/api/v1/evaluation/dashboard/metrics")
            assert response.status_code == 200

    def test_different_ips_separate_limits(self):
        """Test that different IPs have separate rate limits."""
        config = RateLimitConfig(
            evaluation_requests_per_minute=1,
            evaluation_burst_size=1,
            enable_rate_limiting=True,
        )

        app = self.create_test_app(config)

        # Create clients with different IPs
        with TestClient(app) as client1:
            with TestClient(app) as client2:
                # Both should be able to make one request
                response1 = client1.get(
                    "/api/v1/evaluation/submit",
                    headers={"X-Forwarded-For": "192.168.1.1"}
                )
                assert response1.status_code == 200

                response2 = client2.get(
                    "/api/v1/evaluation/submit",
                    headers={"X-Forwarded-For": "192.168.1.2"}
                )
                assert response2.status_code == 200

    def test_rate_limiting_disabled(self):
        """Test that rate limiting can be disabled."""
        config = RateLimitConfig(
            evaluation_requests_per_minute=1,
            evaluation_burst_size=1,
            enable_rate_limiting=False,  # Disabled
        )

        app = self.create_test_app(config)
        client = TestClient(app)

        # Should allow unlimited requests when disabled
        for _ in range(10):
            response = client.get("/api/v1/evaluation/submit")
            assert response.status_code == 200

    def test_endpoint_type_detection(self):
        """Test proper endpoint type detection."""
        middleware = RateLimitMiddleware(
            app=Mock(),
            config=RateLimitConfig(enable_rate_limiting=True)
        )

        assert middleware.get_endpoint_type("/api/v1/evaluation/submit") == "evaluation"
        assert middleware.get_endpoint_type("/api/v1/evaluation/evaluate/batch") == "evaluation"
        assert middleware.get_endpoint_type("/api/v1/evaluation/dashboard/metrics") == "dashboard"
        assert middleware.get_endpoint_type("/api/v1/evaluation/ws") == "dashboard"
        assert middleware.get_endpoint_type("/auth/token") == "auth"
        assert middleware.get_endpoint_type("/health") == "default"

    def test_client_ip_extraction(self):
        """Test client IP extraction from headers."""
        middleware = RateLimitMiddleware(
            app=Mock(),
            config=RateLimitConfig(enable_rate_limiting=True)
        )

        # Test X-Forwarded-For header
        request = Mock()
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client = Mock()
        request.client.host = "127.0.0.1"

        ip = middleware.get_client_ip(request)
        assert ip == "192.168.1.1"  # Should take first IP

        # Test X-Real-IP header
        request.headers = {"X-Real-IP": "203.0.113.1"}
        ip = middleware.get_client_ip(request)
        assert ip == "203.0.113.1"

        # Test fallback to client.host
        request.headers = {}
        ip = middleware.get_client_ip(request)
        assert ip == "127.0.0.1"

    def test_suspicious_behavior_detection(self):
        """Test suspicious behavior detection."""
        middleware = RateLimitMiddleware(
            app=Mock(),
            config=RateLimitConfig(enable_rate_limiting=True, enable_adaptive_limiting=True)
        )

        # Test suspicious user agents
        request = Mock()
        request.headers = {"User-Agent": "python-requests/2.28.1", "Accept": "application/json"}

        # Should detect as suspicious
        result = asyncio.run(middleware.detect_suspicious_behavior(request, "192.168.1.1"))
        assert result is True

        # Test normal user agent with Accept header
        request.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml"
        }
        result = asyncio.run(middleware.detect_suspicious_behavior(request, "192.168.1.2"))
        assert result is False

        # Test missing Accept header
        request.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        result = asyncio.run(middleware.detect_suspicious_behavior(request, "192.168.1.3"))
        assert result is True  # No Accept header


@pytest.mark.asyncio
async def test_rate_limit_headers():
    """Test that proper rate limit headers are returned."""
    backend = MemoryRateLimitBackend()
    limit = RateLimit(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_size=10
    )

    is_limited, headers = await backend.is_rate_limited("test_key", limit)

    assert not is_limited
    assert "X-RateLimit-Limit" in headers
    assert "X-RateLimit-Remaining" in headers
    assert "X-RateLimit-Reset" in headers
    assert int(headers["X-RateLimit-Limit"]) == 60
    assert int(headers["X-RateLimit-Remaining"]) <= 60


if __name__ == "__main__":
    pytest.main([__file__])
