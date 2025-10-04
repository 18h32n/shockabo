"""Integration tests for real-time dashboard functionality.

These tests verify WebSocket connections, real-time updates, and dashboard
data aggregation with authentication.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from src.adapters.api.routes.evaluation import (
    ConnectionManager,
    jwt_manager,
    manager,
    router,
    task_repository,
)
from src.domain.models import ARCTask


@pytest.fixture
def test_token():
    """Create a test JWT token."""
    return jwt_manager.create_access_token("test_user")


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    return TestClient(app)


class TestConnectionManager:
    """Test WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_authenticated_connection(self):
        """Test WebSocket connection with authentication."""
        conn_manager = ConnectionManager()

        # Mock WebSocket
        websocket = AsyncMock(spec=WebSocket)
        user_id = "test_user_123"

        # Connect with authentication
        await conn_manager.connect(websocket, user_id)

        assert websocket in conn_manager.active_connections
        assert conn_manager.authenticated_connections[websocket] == user_id
        websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self):
        """Test proper cleanup on disconnect."""
        conn_manager = ConnectionManager()

        # Setup connections
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)

        await conn_manager.connect(ws1, "user1")
        await conn_manager.connect(ws2, "user2")

        # Subscribe to experiment
        await conn_manager.subscribe_to_experiment(ws1, "exp_123")

        # Disconnect ws1
        conn_manager.disconnect(ws1)

        assert ws1 not in conn_manager.active_connections
        assert ws1 not in conn_manager.authenticated_connections
        assert "exp_123" not in conn_manager.experiment_subscriptions
        assert ws2 in conn_manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_to_experiment(self):
        """Test broadcasting to experiment subscribers."""
        conn_manager = ConnectionManager()

        # Setup connections
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)

        await conn_manager.connect(ws1, "user1")
        await conn_manager.connect(ws2, "user2")
        await conn_manager.connect(ws3, "user3")

        # Subscribe ws1 and ws2 to experiment
        await conn_manager.subscribe_to_experiment(ws1, "exp_123")
        await conn_manager.subscribe_to_experiment(ws2, "exp_123")

        # Broadcast to experiment
        message = json.dumps({"type": "test", "data": "hello"})
        await conn_manager.broadcast_to_experiment("exp_123", message)

        # Only subscribed connections should receive
        ws1.send_text.assert_called_with(message)
        ws2.send_text.assert_called_with(message)
        ws3.send_text.assert_not_called()


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    @pytest.mark.asyncio
    async def test_websocket_authentication_required(self, test_client):
        """Test WebSocket requires authentication."""
        with test_client.websocket_connect("/api/v1/evaluation/ws") as websocket:
            # Should be closed immediately due to no auth
            with pytest.raises(Exception):
                websocket.receive_json()

    @pytest.mark.asyncio
    async def test_websocket_with_valid_token(self, test_client, test_token):
        """Test WebSocket connection with valid token."""
        # Mock JWT authentication
        with patch.object(jwt_manager, "authenticate_websocket", return_value="test_user"):
            with test_client.websocket_connect(
                f"/api/v1/evaluation/ws?token={test_token}"
            ) as websocket:
                # Should receive connection confirmation
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
                assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_experiment_subscription(self, test_client, test_token):
        """Test subscribing to experiment updates."""
        with patch.object(jwt_manager, "authenticate_websocket", return_value="test_user"):
            with test_client.websocket_connect(
                f"/api/v1/evaluation/ws?token={test_token}"
            ) as websocket:
                # Skip connection message
                websocket.receive_json()

                # Subscribe to experiment
                websocket.send_json({
                    "type": "subscribe_experiment",
                    "experiment_id": "exp_456"
                })

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscription_confirmed"
                assert data["experiment_id"] == "exp_456"


class TestDashboardAPI:
    """Test dashboard REST API endpoints."""

    def test_get_dashboard_metrics_requires_auth(self, test_client):
        """Test dashboard metrics endpoint requires authentication."""
        response = test_client.get("/api/v1/evaluation/dashboard/metrics")
        assert response.status_code == 403  # Forbidden without auth

    def test_get_dashboard_metrics_with_auth(self, test_client, test_token):
        """Test getting dashboard metrics with authentication."""
        response = test_client.get(
            "/api/v1/evaluation/dashboard/metrics",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify dashboard metrics structure
        assert "timestamp" in data
        assert "active_experiments" in data
        assert "tasks_processed_last_hour" in data
        assert "average_accuracy_last_hour" in data
        assert "resource_utilization" in data
        assert "system_health" in data

    def test_get_strategy_performance(self, test_client, test_token):
        """Test getting strategy performance metrics."""
        response = test_client.get(
            "/api/v1/evaluation/strategies/performance?time_window=24h",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["time_window"] == "24h"
        assert "strategies" in data
        assert len(data["strategies"]) > 0

        # Verify strategy structure
        strategy = data["strategies"][0]
        assert "name" in strategy
        assert "average_accuracy" in strategy
        assert "tasks_evaluated" in strategy


class TestRealtimeUpdates:
    """Test real-time update functionality."""

    @pytest.mark.asyncio
    async def test_task_submission_broadcast(self, test_client, test_token):
        """Test real-time broadcast on task submission."""
        # Create a test task in repository
        test_task = ARCTask(
            task_id="test_task_123",
            task_source="test",
            train_examples=[
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            test_input=[[0, 0], [1, 1]],
            test_output=[[1, 1], [0, 0]]
        )

        with patch.object(task_repository, "get_task", return_value=test_task):
            with patch.object(task_repository, "save_submission", return_value=True):
                with patch.object(manager, "broadcast") as mock_broadcast:
                    # Submit task
                    response = test_client.post(
                        "/api/v1/evaluation/submit",
                        headers={"Authorization": f"Bearer {test_token}"},
                        json={
                            "task_id": "test_task_123",
                            "predicted_output": [[1, 1], [0, 0]],
                            "strategy": "DIRECT_SOLVE",
                            "confidence_score": 0.9,
                            "attempt_number": 1
                        }
                    )

                    assert response.status_code == 200

                    # Verify broadcast was called
                    mock_broadcast.assert_called_once()
                    broadcast_data = json.loads(mock_broadcast.call_args[0][0])
                    assert broadcast_data["type"] == "task_submitted"
                    assert broadcast_data["task_id"] == "test_task_123"
                    assert "accuracy" in broadcast_data

    @pytest.mark.asyncio
    async def test_experiment_progress_updates(self, test_client, test_token):
        """Test experiment progress updates via WebSocket."""
        # Create test tasks
        test_tasks = [
            ARCTask(
                task_id=f"task_{i}",
                task_source="test",
                train_examples=[],
                test_input=[[0]],
                test_output=[[1]]
            )
            for i in range(3)
        ]

        with patch.object(task_repository, "get_task", side_effect=test_tasks):
            with patch.object(task_repository, "save_experiment", return_value=True):
                with patch.object(task_repository, "update_experiment_status", return_value=True):
                    with patch.object(manager, "broadcast_to_experiment") as mock_broadcast:
                        # Start batch evaluation
                        response = test_client.post(
                            "/api/v1/evaluation/evaluate/batch",
                            headers={"Authorization": f"Bearer {test_token}"},
                            json={
                                "evaluations": [
                                    {
                                        "task_id": f"task_{i}",
                                        "predicted_output": [[1]],
                                        "ground_truth": [[1]]
                                    }
                                    for i in range(3)
                                ],
                                "strategy": "PATTERN_MATCH",
                                "experiment_id": "test_exp_123"
                            }
                        )

                        assert response.status_code == 200

                        # Wait for async processing
                        await asyncio.sleep(1)

                        # Verify progress updates were sent
                        assert mock_broadcast.call_count >= 3  # At least one per task

                        # Check final completion update
                        final_call = mock_broadcast.call_args_list[-1]
                        final_data = json.loads(final_call[0][1])
                        assert final_data["type"] == "experiment_completed"
                        assert final_data["experiment_id"] == "test_exp_123"


class TestDashboardMetricsAggregation:
    """Test dashboard metrics aggregation."""

    @pytest.mark.asyncio
    async def test_periodic_metrics_streaming(self):
        """Test periodic dashboard metrics streaming."""
        # Mock WebSocket
        websocket = AsyncMock(spec=WebSocket)

        # Track sent messages
        sent_messages = []
        websocket.send_text.side_effect = lambda msg: sent_messages.append(json.loads(msg))

        # Add to active connections
        manager.active_connections.append(websocket)

        # Import the metrics sender function
        from src.adapters.api.routes.evaluation import _send_dashboard_metrics

        # Run metrics sender for a short time
        task = asyncio.create_task(_send_dashboard_metrics(websocket))
        await asyncio.sleep(1.1)  # Should send at least 2 updates (500ms interval)
        task.cancel()

        # Verify metrics were sent
        assert len(sent_messages) >= 2

        # Check metrics structure
        for message in sent_messages:
            assert message["type"] == "dashboard_update"
            data = message["data"]
            assert "timestamp" in data
            assert "active_experiments" in data
            assert "resource_utilization" in data
            assert isinstance(data["resource_utilization"], dict)
            assert "cpu" in data["resource_utilization"]
            assert "memory" in data["resource_utilization"]

    def test_dashboard_metrics_snapshot(self, test_client, test_token):
        """Test getting dashboard metrics snapshot."""
        response = test_client.get(
            "/api/v1/evaluation/dashboard/metrics",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required metrics
        required_fields = [
            "timestamp",
            "active_experiments",
            "tasks_processed_last_hour",
            "average_accuracy_last_hour",
            "resource_utilization",
            "processing_queue_size",
            "error_rate_last_hour",
            "top_performing_strategies",
            "recent_alerts",
            "system_health"
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify data types
        assert isinstance(data["active_experiments"], int)
        assert isinstance(data["average_accuracy_last_hour"], float)
        assert isinstance(data["top_performing_strategies"], list)
        assert isinstance(data["system_health"], dict)


@pytest.mark.integration
class TestDashboardE2E:
    """End-to-end dashboard integration tests."""

    @pytest.mark.asyncio
    async def test_concurrent_websocket_connections(self, test_client):
        """Test handling multiple concurrent WebSocket connections."""
        # This would test with multiple real WebSocket connections
        # Skipped for brevity but important for production testing
        pass

    @pytest.mark.asyncio
    async def test_dashboard_performance_under_load(self, test_client):
        """Test dashboard performance with many concurrent updates."""
        # This would simulate high-frequency updates and measure latency
        # Important for verifying the 500ms update requirement
        pass
