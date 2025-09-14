"""Performance tests for real-time dashboard updates.

These tests verify that the dashboard meets the 500ms update latency requirement
and can handle concurrent connections efficiently.
"""

import asyncio
import gc
import json
import time
from datetime import datetime
from statistics import mean, stdev
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket

from src.adapters.api.routes.evaluation import ConnectionManager, _send_dashboard_metrics
from src.domain.evaluation_models import DashboardMetrics


class TestDashboardUpdateLatency:
    """Test dashboard update latency requirements."""

    @pytest.mark.asyncio
    async def test_single_client_update_latency(self):
        """Test update latency for a single client."""
        # Mock WebSocket with latency tracking
        websocket = AsyncMock(spec=WebSocket)
        send_times = []

        async def track_send_time(msg):
            send_times.append(time.time())

        websocket.send_text.side_effect = track_send_time

        # Run dashboard updates for 5 seconds
        task = asyncio.create_task(_send_dashboard_metrics(websocket))
        await asyncio.sleep(5)
        task.cancel()

        # Calculate intervals between updates
        intervals = []
        for i in range(1, len(send_times)):
            interval_ms = (send_times[i] - send_times[i-1]) * 1000
            intervals.append(interval_ms)

        # Verify 500ms update requirement
        avg_interval = mean(intervals)
        assert 450 <= avg_interval <= 550, f"Average interval {avg_interval}ms not within 500ms ± 10%"

        # Check consistency
        interval_stdev = stdev(intervals) if len(intervals) > 1 else 0
        assert interval_stdev < 50, f"Update interval variance too high: {interval_stdev}ms"

    @pytest.mark.asyncio
    async def test_concurrent_clients_update_latency(self):
        """Test update latency with multiple concurrent clients."""
        num_clients = 10
        manager = ConnectionManager()

        # Create mock websockets
        websockets = []
        client_latencies = {i: [] for i in range(num_clients)}

        for i in range(num_clients):
            ws = AsyncMock(spec=WebSocket)
            last_send_time = {"time": None}

            async def track_latency(msg, client_id=i, lst=last_send_time):
                current_time = time.time()
                if lst["time"] is not None:
                    latency = (current_time - lst["time"]) * 1000
                    client_latencies[client_id].append(latency)
                lst["time"] = current_time

            ws.send_text.side_effect = track_latency
            websockets.append(ws)
            await manager.connect(ws, f"user_{i}")

        # Start dashboard updates for all clients
        tasks = []
        for ws in websockets:
            task = asyncio.create_task(_send_dashboard_metrics(ws))
            tasks.append(task)

        # Run for 5 seconds
        await asyncio.sleep(5)

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Analyze latencies
        all_latencies = []
        for client_id, latencies in client_latencies.items():
            if latencies:
                all_latencies.extend(latencies)
                client_avg = mean(latencies)
                # Each client should maintain ~500ms updates
                assert 450 <= client_avg <= 600, f"Client {client_id} avg latency {client_avg}ms out of range"

        # Overall system should maintain performance
        overall_avg = mean(all_latencies)
        assert 450 <= overall_avg <= 550, f"Overall average latency {overall_avg}ms exceeds requirement"

    @pytest.mark.asyncio
    async def test_broadcast_performance(self):
        """Test broadcast performance to many clients."""
        manager = ConnectionManager()
        num_clients = 100

        # Create mock websockets
        for i in range(num_clients):
            ws = AsyncMock(spec=WebSocket)
            ws.send_text = AsyncMock()
            await manager.connect(ws, f"user_{i}")

        # Measure broadcast time
        message = json.dumps({
            "type": "test_broadcast",
            "timestamp": datetime.now().isoformat(),
            "data": {"value": 123}
        })

        start_time = time.time()
        await manager.broadcast(message)
        broadcast_time = (time.time() - start_time) * 1000

        # Broadcast to 100 clients should be fast
        assert broadcast_time < 100, f"Broadcast took {broadcast_time}ms, exceeding 100ms limit"

        # Verify all clients received the message
        for ws in manager.active_connections:
            ws.send_text.assert_called_with(message)


class TestDashboardScalability:
    """Test dashboard scalability under load."""

    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self):
        """Test connection pooling efficiency."""
        manager = ConnectionManager()

        # Simulate rapid connect/disconnect cycles
        connect_times = []
        disconnect_times = []

        for cycle in range(10):
            # Connect 50 clients
            cycle_websockets = []
            start_connect = time.time()

            for i in range(50):
                ws = AsyncMock(spec=WebSocket)
                await manager.connect(ws, f"cycle_{cycle}_user_{i}")
                cycle_websockets.append(ws)

            connect_time = time.time() - start_connect
            connect_times.append(connect_time)

            # Disconnect all
            start_disconnect = time.time()
            for ws in cycle_websockets:
                manager.disconnect(ws)

            disconnect_time = time.time() - start_disconnect
            disconnect_times.append(disconnect_time)

        # Connection/disconnection should be efficient
        avg_connect = mean(connect_times) * 1000
        avg_disconnect = mean(disconnect_times) * 1000

        assert avg_connect < 50, f"Average connection time {avg_connect}ms too high"
        assert avg_disconnect < 20, f"Average disconnection time {avg_disconnect}ms too high"

    @pytest.mark.asyncio
    async def test_memory_usage_with_connections(self):
        """Test memory usage doesn't grow unbounded with connections."""
        manager = ConnectionManager()

        # Track initial state
        import sys
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0

        # Create and destroy many connections
        for iteration in range(100):
            websockets = []

            # Create 10 connections
            for i in range(10):
                ws = AsyncMock(spec=WebSocket)
                await manager.connect(ws, f"iter_{iteration}_user_{i}")
                websockets.append(ws)

                # Subscribe to experiments
                await manager.subscribe_to_experiment(ws, f"exp_{iteration}_{i % 3}")

            # Send some messages
            for _ in range(5):
                await manager.broadcast(json.dumps({"iter": iteration}))

            # Disconnect all
            for ws in websockets:
                manager.disconnect(ws)

        # Verify cleanup
        assert len(manager.active_connections) == 0
        assert len(manager.authenticated_connections) == 0
        assert len(manager.experiment_subscriptions) == 0

        # Check object count didn't grow significantly
        if 'gc' in sys.modules:
            import gc
            gc.collect()
            final_objects = len(gc.get_objects())
            object_growth = final_objects - initial_objects
            # Allow some growth but not unbounded
            assert object_growth < 1000, f"Object count grew by {object_growth}"


class TestDashboardDataProcessing:
    """Test dashboard data processing performance."""

    @pytest.mark.asyncio
    async def test_metrics_aggregation_performance(self):
        """Test performance of metrics aggregation."""
        from src.domain.services.dashboard_aggregator import DashboardAggregator

        aggregator = DashboardAggregator()

        # Simulate high-frequency metric updates
        update_times = []

        for i in range(1000):
            start_time = time.time()

            # Add various metrics
            aggregator.add_task_result({
                "task_id": f"task_{i}",
                "accuracy": 0.8 + (i % 20) / 100,
                "processing_time_ms": 100 + i % 50,
                "strategy": ["direct", "pattern", "ensemble"][i % 3]
            })

            if i % 10 == 0:
                aggregator.add_resource_usage({
                    "cpu": 40 + i % 20,
                    "memory": 60 + i % 15,
                    "gpu": i % 2 * 30
                })

            update_time = (time.time() - start_time) * 1000
            update_times.append(update_time)

        # Get aggregated metrics
        start_time = time.time()
        metrics = aggregator.get_dashboard_metrics()
        aggregation_time = (time.time() - start_time) * 1000

        # Performance requirements
        avg_update_time = mean(update_times)
        assert avg_update_time < 1, f"Average update time {avg_update_time}ms too high"
        assert aggregation_time < 10, f"Aggregation time {aggregation_time}ms too high"

        # Verify metrics accuracy
        assert metrics.tasks_processed_last_hour > 0
        assert 0 <= metrics.average_accuracy_last_hour <= 1
        assert len(metrics.top_performing_strategies) > 0

    @pytest.mark.asyncio
    async def test_websocket_message_serialization_performance(self):
        """Test WebSocket message serialization performance."""
        # Create a complex dashboard metrics object
        metrics = DashboardMetrics(
            timestamp=datetime.now(),
            active_experiments=25,
            tasks_processed_last_hour=1500,
            average_accuracy_last_hour=0.827,
            resource_utilization={
                "cpu": 67.3,
                "memory": 82.1,
                "gpu": 45.6,
                "disk_io": 23.4
            },
            processing_queue_size=47,
            error_rate_last_hour=0.023,
            top_performing_strategies=[
                ("ensemble_v2", 0.912),
                ("pattern_match_advanced", 0.887),
                ("direct_solve_optimized", 0.834),
                ("hybrid_approach", 0.821),
                ("baseline", 0.756)
            ],
            recent_alerts=[
                {
                    "type": "performance",
                    "message": "High memory usage detected",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            system_health={
                "evaluation_service": "healthy",
                "database": "healthy",
                "cache": "degraded",
                "ml_models": "healthy"
            }
        )

        # Measure serialization time
        serialization_times = []

        for _ in range(1000):
            start_time = time.time()
            message = json.dumps(metrics.to_websocket_message())
            serialization_time = (time.time() - start_time) * 1000
            serialization_times.append(serialization_time)

        # Serialization should be fast
        avg_serialization = mean(serialization_times)
        assert avg_serialization < 0.5, f"Average serialization time {avg_serialization}ms too high"

        # Message size should be reasonable
        message_size = len(message.encode('utf-8'))
        assert message_size < 10000, f"Message size {message_size} bytes too large"


@pytest.mark.performance
class TestDashboardStressTest:
    """Stress tests for dashboard under extreme load."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_thousand_concurrent_connections(self):
        """Test system with 1000 concurrent connections."""
        # This test would create 1000 real connections
        # and verify system stability
        pass

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_high_frequency_updates(self):
        """Test sustained high-frequency updates for extended period."""
        # This test would run for several minutes with
        # continuous updates to verify no degradation
        pass


# Helper function to measure actual WebSocket latency
async def measure_websocket_latency(url: str, token: str, duration_seconds: int = 10) -> list[float]:
    """Measure actual WebSocket message latency.

    Args:
        url: WebSocket URL
        token: Authentication token
        duration_seconds: How long to measure

    Returns:
        List of latencies in milliseconds
    """
    import websockets

    latencies = []
    last_message_time = None

    async with websockets.connect(f"{url}?token={token}") as websocket:
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            await websocket.recv()
            current_time = time.time()

            if last_message_time is not None:
                latency = (current_time - last_message_time) * 1000
                latencies.append(latency)

            last_message_time = current_time

    return latencies
