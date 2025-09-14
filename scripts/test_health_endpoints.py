#!/usr/bin/env python3
"""Test script to demonstrate health check endpoints functionality.

This script tests all health check endpoints and displays their responses,
useful for verifying the health check system is working correctly.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Add source root to path
_src_root = Path(__file__).parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))


async def test_health_endpoints():
    """Test all health check endpoints."""
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("ARC Prize 2025 - Health Check Endpoints Test")
    print("=" * 60)

    endpoints_to_test = [
        ("Root Health", "/"),
        ("System Info", "/info"),
        ("Basic Health", "/health/"),
        ("Detailed Health", "/health/detailed"),
        ("Health Status", "/health/status"),
        ("Performance Metrics", "/health/metrics"),
        ("Health Config", "/health/config"),
        ("Readiness Probe", "/health/ready"),
        ("Liveness Probe", "/health/live"),
        ("Startup Probe", "/health/startup"),
        ("Cache Component", "/health/component/cache"),
        ("Memory Component", "/health/component/memory"),
        ("Storage Component", "/health/component/storage"),
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        print(f"\nTesting endpoints on {base_url}")
        print("-" * 60)

        for endpoint_name, path in endpoints_to_test:
            try:
                start_time = time.time()
                response = await client.get(f"{base_url}{path}")
                response_time = (time.time() - start_time) * 1000

                print(f"\n{endpoint_name}:")
                print(f"  URL: {path}")
                print(f"  Status: {response.status_code}")
                print(f"  Response Time: {response_time:.1f}ms")
                print(f"  Content-Type: {response.headers.get('content-type', 'N/A')}")

                # Show response content based on type
                if response.headers.get('content-type', '').startswith('application/json'):
                    try:
                        data = response.json()
                        if path == "/" or path == "/info":
                            # Show full response for simple endpoints
                            print(f"  Response: {json.dumps(data, indent=2)}")
                        else:
                            # Show key fields for complex responses
                            key_fields = []
                            if 'status' in data:
                                key_fields.append(f"status={data['status']}")
                            if 'overall_status' in data:
                                key_fields.append(f"overall_status={data['overall_status']}")
                            if 'uptime_seconds' in data:
                                key_fields.append(f"uptime={data['uptime_seconds']:.1f}s")
                            if 'components' in data:
                                if isinstance(data['components'], dict):
                                    component_count = len(data['components'])
                                    key_fields.append(f"components={component_count}")
                                elif isinstance(data['components'], list):
                                    component_count = len(data['components'])
                                    key_fields.append(f"components={component_count}")

                            print(f"  Key Fields: {', '.join(key_fields) if key_fields else 'N/A'}")

                            # Show component statuses for detailed health
                            if path == "/health/detailed" and 'components' in data:
                                print("  Component Status:")
                                for component in data['components']:
                                    comp_name = component.get('component_name', 'unknown')
                                    comp_status = component.get('status', 'unknown')
                                    comp_time = component.get('response_time_ms', 0)
                                    print(f"    - {comp_name}: {comp_status} ({comp_time:.1f}ms)")

                    except json.JSONDecodeError:
                        print(f"  Response: {response.text[:200]}{'...' if len(response.text) > 200 else ''}")
                else:
                    print(f"  Response: {response.text}")

                # Color code status
                status_color = "✅" if response.status_code == 200 else "⚠️" if response.status_code == 503 else "❌"
                print(f"  Result: {status_color}")

            except httpx.ConnectError:
                print(f"\n{endpoint_name}:")
                print(f"  URL: {path}")
                print("  Status: Connection Failed")
                print("  Result: ❌ (Server not running?)")

            except Exception as e:
                print(f"\n{endpoint_name}:")
                print(f"  URL: {path}")
                print(f"  Status: Error - {str(e)}")
                print("  Result: ❌")


async def test_concurrent_requests():
    """Test concurrent health check requests."""
    print("\n" + "=" * 60)
    print("Testing Concurrent Requests")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # Test concurrent basic health checks
    concurrent_count = 5

    async with httpx.AsyncClient(timeout=10.0) as client:
        print(f"Making {concurrent_count} concurrent requests to /health/...")

        start_time = time.time()

        tasks = [
            client.get(f"{base_url}/health/")
            for _ in range(concurrent_count)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = (time.time() - start_time) * 1000

        print(f"Total time: {total_time:.1f}ms")
        print(f"Average time per request: {total_time / concurrent_count:.1f}ms")

        success_count = 0
        status_codes = {}

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Request {i+1}: Exception - {response}")
            else:
                success_count += 1
                status_codes[response.status_code] = status_codes.get(response.status_code, 0) + 1

        print(f"Successful requests: {success_count}/{concurrent_count}")
        print(f"Status code distribution: {status_codes}")
        print(f"Result: {'✅' if success_count == concurrent_count else '⚠️'}")


async def performance_benchmark():
    """Benchmark health check endpoint performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    base_url = "http://localhost:8000"
    endpoints = [
        ("/health/live", "Liveness Probe"),
        ("/health/ready", "Readiness Probe"),
        ("/health/", "Basic Health"),
        ("/health/detailed", "Detailed Health")
    ]

    requests_per_endpoint = 10

    async with httpx.AsyncClient(timeout=30.0) as client:
        for path, name in endpoints:
            print(f"\nTesting {name} ({path}):")

            times = []
            statuses = []

            for i in range(requests_per_endpoint):
                start_time = time.time()
                try:
                    response = await client.get(f"{base_url}{path}")
                    response_time = (time.time() - start_time) * 1000
                    times.append(response_time)
                    statuses.append(response.status_code)
                except Exception as e:
                    print(f"  Request {i+1} failed: {e}")

            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                print(f"  Requests: {len(times)}")
                print(f"  Average: {avg_time:.1f}ms")
                print(f"  Min: {min_time:.1f}ms")
                print(f"  Max: {max_time:.1f}ms")

                success_rate = (statuses.count(200) / len(statuses)) * 100
                print(f"  Success Rate: {success_rate:.1f}%")

                # Performance assessment
                if avg_time < 100:
                    performance = "Excellent ✅"
                elif avg_time < 500:
                    performance = "Good ✅"
                elif avg_time < 1000:
                    performance = "Fair ⚠️"
                else:
                    performance = "Slow ❌"

                print(f"  Performance: {performance}")


def print_usage_info():
    """Print usage information for the health check system."""
    print("\n" + "=" * 60)
    print("Health Check System Usage")
    print("=" * 60)

    usage_info = """
Health Check Endpoints:

Basic Endpoints:
  GET /                      - Root health check (for load balancers)
  GET /info                  - System information
  GET /health/               - Basic health check (critical components)
  GET /health/status         - Simple status summary

Detailed Monitoring:
  GET /health/detailed       - Comprehensive health check (all components)
  GET /health/metrics        - Performance metrics
  GET /health/config         - Health check configuration

Kubernetes Probes:
  GET /health/live          - Liveness probe (basic responsiveness)
  GET /health/ready         - Readiness probe (ready to serve traffic)
  GET /health/startup       - Startup probe (finished initialization)

Component-Specific:
  GET /health/component/{name} - Individual component health
  
Available components: database, cache, wandb, storage, memory, cpu

HTTP Status Codes:
  200 - Healthy/Operational
  503 - Unhealthy/Not Ready
  404 - Component not found

Configuration:
  Health checks are configurable via configs/health_check.yaml
  Component thresholds, timeouts, and monitoring intervals can be adjusted
  Individual components can be enabled/disabled

Integration:
  - Use /health/live and /health/ready for Kubernetes probes
  - Use /health/ for load balancer health checks
  - Use /health/detailed for comprehensive monitoring dashboards
  - Use /health/metrics for performance monitoring systems
"""

    print(usage_info)


async def main():
    """Main test function."""
    print("Starting ARC Prize 2025 Health Check Tests...")
    print("Make sure the server is running on http://localhost:8000")

    try:
        # Test all endpoints
        await test_health_endpoints()

        # Test concurrent requests
        await test_concurrent_requests()

        # Performance benchmark
        await performance_benchmark()

        # Usage information
        print_usage_info()

        print("\n" + "=" * 60)
        print("Health Check Testing Complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if server might be running
    try:
        import requests
        response = requests.get("http://localhost:8000", timeout=2)
        print("✅ Server appears to be running")
    except:
        print("⚠️  Warning: Server may not be running on http://localhost:8000")
        print("   Start the server with: python -m src.main")
        print()

    asyncio.run(main())
