#!/usr/bin/env python3
"""Start the ARC system server and demonstrate health check functionality.

This script starts the FastAPI server and then runs health check demonstrations
to show the functionality of the health monitoring system.
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Add source root to path
_src_root = Path(__file__).parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))


class HealthCheckDemo:
    """Demonstrate health check functionality."""
    
    def __init__(self, base_url="http://localhost:8000"):
        """Initialize demo with server URL."""
        self.base_url = base_url
        self.server_process = None
    
    def start_server(self):
        """Start the FastAPI server."""
        print("🚀 Starting ARC Prize 2025 System Server...")
        print("-" * 50)
        
        # Change to project root directory
        os.chdir(_src_root)
        
        # Start the server process
        cmd = [sys.executable, "-m", "src.main"]
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"Server starting with command: {' '.join(cmd)}")
            print("Waiting for server to be ready...")
            
            # Wait for server to start
            max_wait = 30  # 30 seconds
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    response = httpx.get(f"{self.base_url}/", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Server ready after {wait_time} seconds!")
                        print(f"📍 Server running at: {self.base_url}")
                        print(f"📖 API Documentation: {self.base_url}/docs")
                        print()
                        return True
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                
                time.sleep(1)
                wait_time += 1
                
                # Show some server output
                if self.server_process.poll() is None:
                    print(f"⏳ Still waiting... ({wait_time}/{max_wait})")
                else:
                    print("❌ Server process exited unexpectedly")
                    if self.server_process.stdout:
                        output = self.server_process.stdout.read()
                        print(f"Server output: {output}")
                    return False
            
            print("❌ Server failed to start within timeout period")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            print("\n🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("✅ Server stopped")
    
    async def run_health_demo(self):
        """Run health check demonstrations."""
        print("🏥 Health Check System Demonstration")
        print("=" * 50)
        
        demonstrations = [
            ("Basic System Info", self.demo_system_info),
            ("Basic Health Check", self.demo_basic_health),
            ("Kubernetes Probes", self.demo_k8s_probes),
            ("Component Health", self.demo_component_health),
            ("Performance Metrics", self.demo_performance_metrics),
            ("Health Configuration", self.demo_health_config),
        ]
        
        for demo_name, demo_func in demonstrations:
            print(f"\n📋 {demo_name}")
            print("-" * 30)
            
            try:
                await demo_func()
                print("✅ Demo completed successfully")
            except Exception as e:
                print(f"❌ Demo failed: {e}")
            
            # Small pause between demonstrations
            await asyncio.sleep(1)
        
        print(f"\n🎉 All health check demonstrations completed!")
        print(f"🌐 Explore more at: {self.base_url}/docs")
    
    async def demo_system_info(self):
        """Demonstrate system info endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/info")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Service: {data.get('service', 'N/A')}")
                print(f"Version: {data.get('version', 'N/A')}")
                print(f"Environment: {data.get('environment', 'N/A')}")
                print(f"Platform: {data.get('platform', 'N/A')}")
                print(f"Uptime: {data.get('uptime_seconds', 0):.1f} seconds")
                
                features = data.get('features', {})
                enabled_features = [name for name, enabled in features.items() if enabled]
                print(f"Features: {', '.join(enabled_features)}")
    
    async def demo_basic_health(self):
        """Demonstrate basic health check."""
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.get(f"{self.base_url}/health/")
            response_time = (time.time() - start_time) * 1000
            
            print(f"Response Time: {response_time:.1f}ms")
            print(f"HTTP Status: {response.status_code}")
            
            if response.status_code in [200, 503]:
                data = response.json()
                print(f"Overall Status: {data.get('status', 'N/A')}")
                print(f"Version: {data.get('version', 'N/A')}")
                print(f"Uptime: {data.get('uptime_seconds', 0):.1f} seconds")
                
                components = data.get('components', {})
                print(f"Components Checked: {len(components)}")
                for name, comp_data in components.items():
                    status = comp_data.get('status', 'unknown')
                    resp_time = comp_data.get('response_time_ms', 0)
                    status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                    print(f"  {status_emoji} {name}: {status} ({resp_time:.1f}ms)")
    
    async def demo_k8s_probes(self):
        """Demonstrate Kubernetes probe endpoints."""
        probes = [
            ("Liveness", "/health/live"),
            ("Readiness", "/health/ready"), 
            ("Startup", "/health/startup")
        ]
        
        async with httpx.AsyncClient() as client:
            for probe_name, endpoint in probes:
                start_time = time.time()
                response = await client.get(f"{self.base_url}{endpoint}")
                response_time = (time.time() - start_time) * 1000
                
                status_emoji = "✅" if response.status_code == 200 else "❌"
                print(f"{status_emoji} {probe_name} Probe: {response.status_code} - {response.text} ({response_time:.1f}ms)")
    
    async def demo_component_health(self):
        """Demonstrate component-specific health checks."""
        components = ["cache", "memory", "storage", "cpu"]
        
        async with httpx.AsyncClient() as client:
            for component in components:
                try:
                    response = await client.get(f"{self.base_url}/health/component/{component}")
                    
                    if response.status_code in [200, 503]:
                        data = response.json()
                        status = data.get('status', 'unknown')
                        response_time = data.get('response_time_ms', 0)
                        
                        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                        print(f"{status_emoji} {component}: {status} ({response_time:.1f}ms)")
                        
                        # Show some key details if available
                        details = data.get('details', {})
                        if details:
                            key_details = []
                            if 'usage_percent' in details:
                                key_details.append(f"usage={details['usage_percent']:.1f}%")
                            if 'hit_rate' in details:
                                key_details.append(f"hit_rate={details['hit_rate']:.1%}")
                            if 'available_gb' in details:
                                key_details.append(f"available={details['available_gb']:.1f}GB")
                            
                            if key_details:
                                print(f"    📊 {', '.join(key_details)}")
                    else:
                        print(f"❌ {component}: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ {component}: Error - {str(e)}")
    
    async def demo_performance_metrics(self):
        """Demonstrate performance metrics endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health/metrics")
            
            if response.status_code in [200, 503]:
                data = response.json()
                perf_metrics = data.get('performance_metrics', {})
                
                print(f"📊 System Performance:")
                print(f"  Memory: {perf_metrics.get('memory_usage_percent', 0):.1f}% used")
                print(f"  Disk: {perf_metrics.get('disk_usage_percent', 0):.1f}% used")
                print(f"  Uptime: {perf_metrics.get('uptime_seconds', 0):.1f} seconds")
                
                if 'cpu_usage_percent' in perf_metrics:
                    print(f"  CPU: {perf_metrics['cpu_usage_percent']:.1f}% used")
    
    async def demo_health_config(self):
        """Demonstrate health configuration endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health/config")
            
            if response.status_code == 200:
                data = response.json()
                config = data.get('health_check_config', {})
                registered = data.get('registered_checks', [])
                
                print(f"⚙️  Configuration:")
                print(f"  Enabled: {config.get('enabled', False)}")
                print(f"  Check Interval: {config.get('check_interval_seconds', 0)}s")
                print(f"  Timeout: {config.get('timeout_seconds', 0)}s")
                print(f"  Degraded Threshold: {config.get('degraded_threshold_ms', 0)}ms")
                print(f"  Unhealthy Threshold: {config.get('unhealthy_threshold_ms', 0)}ms")
                print(f"  Registered Components: {', '.join(registered)}")


async def main():
    """Main demo function."""
    demo = HealthCheckDemo()
    
    try:
        # Start the server
        if not demo.start_server():
            print("Failed to start server. Exiting.")
            return 1
        
        # Wait a moment for full startup
        await asyncio.sleep(2)
        
        # Run health check demonstrations
        await demo.run_health_demo()
        
        # Show final instructions
        print("\n" + "=" * 60)
        print("🎯 Next Steps:")
        print(f"• API Documentation: {demo.base_url}/docs")
        print(f"• Interactive API: {demo.base_url}/redoc") 
        print("• Run tests: python -m pytest tests/integration/test_health_checks.py")
        print("• Manual testing: python scripts/test_health_endpoints.py")
        print("• Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Keep server running
        try:
            print("\n⏳ Server running... Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
    
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    
    finally:
        demo.stop_server()
    
    return 0


if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        print("\n👋 Received interrupt signal. Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
        sys.exit(0)