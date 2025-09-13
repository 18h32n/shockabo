"""Integration tests for health check endpoints."""

import pytest
import requests
from fastapi.testclient import TestClient

from src.main import create_application


class TestHealthCheckEndpoints:
    """Test suite for health check endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        app = create_application()
        return TestClient(app)

    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        
        assert response.status_code in [200, 503]  # May be degraded in test environment
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert "components" in data
        
        # Verify status is valid
        assert data["status"] in ["healthy", "degraded", "unhealthy", "unknown"]
        
        # Verify components structure
        assert isinstance(data["components"], dict)
        for component_name, component_data in data["components"].items():
            assert "status" in component_data
            assert "response_time_ms" in component_data

    def test_detailed_health_check(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "overall_status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "performance_metrics" in data
        assert "detailed_health" in data
        
        # Verify performance metrics structure
        performance = data["performance_metrics"]
        assert "memory_usage_percent" in performance
        assert "disk_usage_percent" in performance
        assert "uptime_seconds" in performance
        
        # Verify components list structure
        components = data["components"]
        assert isinstance(components, list)
        for component in components:
            assert "component_name" in component
            assert "component_type" in component
            assert "status" in component
            assert "response_time_ms" in component
            assert "last_checked" in component

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code in [200, 503]
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        if response.status_code == 200:
            assert response.text == "OK"
        else:
            assert response.text == "Not Ready"

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe endpoint.""" 
        response = client.get("/health/live")
        
        assert response.status_code in [200, 503]
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        if response.status_code == 200:
            assert response.text == "OK"
        else:
            assert response.text == "Not Alive"

    def test_startup_probe(self, client):
        """Test Kubernetes startup probe endpoint."""
        response = client.get("/health/startup")
        
        assert response.status_code in [200, 503]
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        if response.status_code == 200:
            assert response.text == "OK"
        else:
            assert response.text == "Starting"

    def test_performance_metrics_endpoint(self, client):
        """Test performance metrics endpoint."""
        response = client.get("/health/metrics")
        
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "timestamp" in data
        assert "performance_metrics" in data
        assert "component_metrics" in data
        
        # Verify performance metrics structure
        perf_metrics = data["performance_metrics"]
        expected_metrics = [
            "memory_usage_percent", "disk_usage_percent", "uptime_seconds",
            "memory_used_mb", "memory_total_mb", "disk_used_gb", "disk_total_gb"
        ]
        for metric in expected_metrics:
            assert metric in perf_metrics

    def test_component_health_check(self, client):
        """Test individual component health check."""
        # Test cache component
        response = client.get("/health/component/cache")
        
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert data["component_name"] == "cache"
        assert "component_type" in data
        assert "status" in data
        assert "response_time_ms" in data
        assert "last_checked" in data
        
        # Test memory component
        response = client.get("/health/component/memory")
        
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert data["component_name"] == "memory"

    def test_component_health_check_not_found(self, client):
        """Test component health check for non-existent component."""
        response = client.get("/health/component/nonexistent")
        
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "nonexistent" in data["detail"]

    def test_health_config_endpoint(self, client):
        """Test health check configuration endpoint."""
        response = client.get("/health/config")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "health_check_config" in data
        assert "registered_checks" in data
        
        # Verify config structure
        config = data["health_check_config"]
        assert "enabled" in config
        assert "check_interval_seconds" in config
        assert "timeout_seconds" in config
        assert "components_to_check" in config

    def test_health_status_summary(self, client):
        """Test simple health status summary endpoint."""
        response = client.get("/health/status")
        
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime" in data
        
        assert data["status"] in ["healthy", "degraded", "unhealthy", "unknown"]

    def test_root_health_endpoint(self, client):
        """Test root health endpoint for load balancers."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "arc-prize-2025"
        assert data["version"] == "1.3.0"

    def test_system_info_endpoint(self, client):
        """Test system information endpoint."""
        response = client.get("/info")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "environment" in data
        assert "platform" in data
        assert "uptime_seconds" in data
        assert "resources" in data
        assert "features" in data
        
        # Verify features
        features = data["features"]
        assert features["health_checks"] is True
        assert features["evaluation_api"] is True

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, client):
        """Test concurrent health check requests."""
        import asyncio
        import httpx
        
        # Make multiple concurrent requests
        async with httpx.AsyncClient(app=client.app, base_url="http://test") as async_client:
            tasks = [
                async_client.get("/health/"),
                async_client.get("/health/detailed"),
                async_client.get("/health/ready"),
                async_client.get("/health/live"),
                async_client.get("/health/metrics")
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests succeeded or failed gracefully
            for response in responses:
                if isinstance(response, Exception):
                    pytest.fail(f"Health check request failed: {response}")
                assert response.status_code in [200, 503]

    def test_health_check_performance(self, client):
        """Test health check response times."""
        import time
        
        # Basic health check should be fast
        start_time = time.time()
        response = client.get("/health/")
        basic_time = (time.time() - start_time) * 1000
        
        assert basic_time < 1000, f"Basic health check too slow: {basic_time}ms"
        
        # Detailed health check may be slower but should be reasonable
        start_time = time.time()
        response = client.get("/health/detailed")
        detailed_time = (time.time() - start_time) * 1000
        
        assert detailed_time < 5000, f"Detailed health check too slow: {detailed_time}ms"
        
        # Probes should be very fast
        start_time = time.time()
        response = client.get("/health/live")
        probe_time = (time.time() - start_time) * 1000
        
        assert probe_time < 500, f"Liveness probe too slow: {probe_time}ms"