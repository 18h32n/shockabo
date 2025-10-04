"""Setup and integration guide for the centralized error handling system.

This module provides utilities and examples for integrating the error handling
system into the ARC Prize evaluation framework.
"""

import asyncio
from typing import Any

import structlog
from fastapi import FastAPI

from .error_handling import get_error_logger
from .error_recovery import (
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_health_monitor,
)
from .middleware import setup_middleware


class ErrorHandlingDemo:
    """Demo class for error handling system."""

    async def run_all_demos(self):
        """Run all demo scenarios."""
        print("Running error handling demos...")
        # Demo implementation would go here

    def generate_demo_report(self):
        """Generate demo report."""
        return "Error handling demo completed successfully"


def setup_error_handling_system(
    app: FastAPI,
    enable_detailed_logging: bool = False,
    enable_health_monitoring: bool = True,
    enable_circuit_breakers: bool = True,
) -> dict[str, Any]:
    """Complete setup of the error handling system for a FastAPI application.

    Args:
        app: FastAPI application instance
        enable_detailed_logging: Whether to enable detailed request/response logging
        enable_health_monitoring: Whether to start health monitoring
        enable_circuit_breakers: Whether to initialize circuit breakers

    Returns:
        Dictionary with setup results and component references
    """
    setup_results = {
        "middleware_configured": False,
        "health_monitoring_started": False,
        "circuit_breakers_initialized": [],
        "error_logger_configured": False,
    }

    logger = structlog.get_logger(__name__)
    logger.info("initializing_error_handling_system")

    try:
        # 1. Configure middleware
        setup_middleware(app, enable_detailed_logging)
        setup_results["middleware_configured"] = True
        logger.info("middleware_configured_successfully")

        # 2. Initialize error logger
        get_error_logger()
        setup_results["error_logger_configured"] = True
        logger.info("error_logger_initialized")

        # 3. Set up health monitoring if enabled
        if enable_health_monitoring:
            health_monitor = get_health_monitor("arc_system", check_interval=30)

            # Add basic health checks
            def check_app_health():
                return True  # Basic check - app is running

            def check_memory_health():
                import psutil

                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage

            health_monitor.add_health_check(
                "application", check_app_health, critical=True
            )
            health_monitor.add_health_check(
                "memory", check_memory_health, critical=False
            )

            # Start monitoring in background
            asyncio.create_task(health_monitor.start_monitoring())
            setup_results["health_monitoring_started"] = True
            logger.info("health_monitoring_started")

        # 4. Initialize circuit breakers if enabled
        if enable_circuit_breakers:
            # Database circuit breaker
            db_config = CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=60, success_threshold=3
            )
            get_circuit_breaker("database", db_config)
            setup_results["circuit_breakers_initialized"].append("database")

            # External API circuit breaker
            api_config = CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=30, success_threshold=2
            )
            get_circuit_breaker("external_api", api_config)
            setup_results["circuit_breakers_initialized"].append("external_api")

            logger.info(
                "circuit_breakers_initialized",
                count=len(setup_results["circuit_breakers_initialized"]),
            )

        logger.info("error_handling_system_setup_complete", results=setup_results)
        return setup_results

    except Exception as e:
        logger.error("error_handling_setup_failed", error=str(e), exc_info=True)
        raise


def create_fastapi_app_with_error_handling() -> FastAPI:
    """Create a FastAPI application with complete error handling setup.

    Returns:
        Configured FastAPI application with error handling
    """
    app = FastAPI(
        title="ARC Prize Evaluation API",
        description="ARC Prize 2025 evaluation framework with comprehensive error handling",
        version="1.0.0",
    )

    # Setup error handling
    setup_results = setup_error_handling_system(
        app,
        enable_detailed_logging=True,
        enable_health_monitoring=True,
        enable_circuit_breakers=True,
    )

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint with detailed system status."""
        from .error_recovery import (
            get_all_circuit_breaker_stats,
            get_all_health_statuses,
        )

        health_statuses = get_all_health_statuses()
        circuit_breaker_stats = get_all_circuit_breaker_stats()

        overall_healthy = all(
            status.get("overall_healthy", True) for status in health_statuses.values()
        )

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": "2025-01-16T00:00:00Z",  # Would use datetime.now() in real code
            "components": {
                "health_monitors": health_statuses,
                "circuit_breakers": circuit_breaker_stats,
                "error_handling": {
                    "configured": True,
                    "middleware_active": setup_results["middleware_configured"],
                    "monitoring_active": setup_results["health_monitoring_started"],
                },
            },
        }

    return app


def example_usage():
    """Example of how to use the error handling system in your code."""
    from .error_handling import (
        ARCBaseException,
        ErrorCode,
        ErrorContext,
        TaskNotFoundException,
    )
    from .error_recovery import FallbackStrategy, RetryStrategy, get_circuit_breaker

    # Example 1: Raising custom exceptions
    def load_task_example(task_id: str):
        if not task_id:
            raise ARCBaseException(
                message="Task ID cannot be empty",
                error_code=ErrorCode.VALIDATION_ERROR,
                suggestions=["Provide a valid task ID"],
            )

        # Simulate task not found
        if task_id == "nonexistent":
            context = ErrorContext(
                task_id=task_id,
                additional_data={"attempted_source": "database"},
            )
            raise TaskNotFoundException(task_id, context=context)

        return {"id": task_id, "data": "..."}

    # Example 2: Using circuit breaker
    async def call_external_service_example():
        circuit_breaker = get_circuit_breaker("example_service")

        async def external_api_call():
            # Your external service call here
            return "api_response"

        try:
            result = await circuit_breaker.call(external_api_call)
            return result
        except ARCBaseException as e:
            # Handle circuit breaker errors
            if "circuit breaker" in str(e):
                # Service is down, use fallback or return error
                return None
            raise

    # Example 3: Using retry strategy
    async def retry_example():
        retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )

        async def unreliable_operation():
            # Operation that might fail
            import random

            if random.random() < 0.5:
                raise ConnectionError("Network issue")
            return "success"

        try:
            result = await retry_strategy.execute(unreliable_operation)
            return result
        except ARCBaseException:
            # All retries failed
            return None

    # Example 4: Using fallback strategy
    async def fallback_example():
        fallback_strategy = FallbackStrategy("data_access")

        # Add fallback functions
        fallback_strategy.add_fallback(lambda: "cached_data", priority=2)
        fallback_strategy.add_fallback(lambda: "default_data", priority=1)

        def primary_data_source():
            raise Exception("Primary source failed")

        try:
            result = await fallback_strategy.execute(primary_data_source)
            return result
        except ARCBaseException:
            return "all_sources_failed"

    return {
        "load_task": load_task_example,
        "circuit_breaker": call_external_service_example,
        "retry": retry_example,
        "fallback": fallback_example,
    }


def integration_checklist() -> dict[str, str]:
    """Checklist for integrating error handling into existing modules."""
    return {
        "1_import_exceptions": "from src.utils.error_handling import ARCBaseException, TaskNotFoundException, etc.",
        "2_replace_generic_exceptions": "Replace ValueError, Exception with specific ARC exceptions",
        "3_add_error_context": "Include ErrorContext with user_id, task_id, and relevant metadata",
        "4_use_structured_logging": "Use structlog for consistent error logging",
        "5_implement_circuit_breakers": "Add circuit breakers for external service calls",
        "6_add_retry_logic": "Use RetryStrategy for transient failure recovery",
        "7_setup_health_checks": "Add health checks for critical components",
        "8_update_api_responses": "Let middleware handle error response formatting",
        "9_test_error_scenarios": "Test various error conditions and recovery paths",
        "10_document_error_codes": "Document all custom error codes and their meanings",
    }


if __name__ == "__main__":
    # Print integration guidance
    print("ARC Prize Error Handling System Integration Guide")
    print("=" * 60)

    checklist = integration_checklist()
    for step, description in checklist.items():
        print(f"{step}: {description}")

    print("\n" + "=" * 60)
    print("Use setup_error_handling_system(app) to configure your FastAPI app")
    print("Run error_handling_demo.py to see the system in action")
    print("Check the health endpoint at /health for system status")

    # Run the demo
    async def main():
        demo = ErrorHandlingDemo()
        await demo.run_all_demos()
        report = demo.generate_demo_report()
        print(report)

    asyncio.run(main())
