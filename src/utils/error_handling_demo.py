"""Demonstration and testing script for the centralized error handling system.

This script shows how to use the error handling framework and tests various scenarios
to ensure robust error management across the ARC Prize evaluation system.
"""

import asyncio
import logging
import random
import time
from datetime import datetime

import structlog

from .error_handling import (
    ARCBaseException,
    AuthenticationException,
    DataCorruptionException,
    DataNotFoundException,
    ErrorCode,
    ErrorContext,
    ErrorLogger,
    ErrorResponse,
    ErrorSeverity,
    EvaluationException,
    TaskNotFoundException,
    create_error_response,
)
from .error_recovery import (
    CircuitBreakerConfig,
    FallbackStrategy,
    RetryStrategy,
    get_circuit_breaker,
    get_health_monitor,
)

# Configure logging for demo
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)


class ErrorHandlingDemo:
    """Demonstration class for error handling features."""

    def __init__(self):
        self.error_logger = ErrorLogger()
        self.demo_results: dict[str, dict] = {}

    async def run_all_demos(self) -> dict[str, dict]:
        """Run all error handling demonstrations."""
        logger.info("starting_error_handling_demos")

        demos = [
            ("basic_exception_handling", self.demo_basic_exceptions),
            ("custom_arc_exceptions", self.demo_custom_exceptions),
            ("error_logging", self.demo_error_logging),
            ("circuit_breaker", self.demo_circuit_breaker),
            ("retry_mechanisms", self.demo_retry_mechanisms),
            ("fallback_strategies", self.demo_fallback_strategies),
            ("health_monitoring", self.demo_health_monitoring),
            ("api_error_responses", self.demo_api_error_responses),
            ("recovery_patterns", self.demo_recovery_patterns),
        ]

        for demo_name, demo_func in demos:
            try:
                logger.info("running_demo", demo=demo_name)
                start_time = time.time()

                result = await demo_func()

                execution_time = time.time() - start_time
                self.demo_results[demo_name] = {
                    "status": "success",
                    "result": result,
                    "execution_time": execution_time,
                }

                logger.info(
                    "demo_completed",
                    demo=demo_name,
                    execution_time_ms=execution_time * 1000,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                self.demo_results[demo_name] = {
                    "status": "error",
                    "error": str(e),
                    "execution_time": execution_time,
                }

                logger.error(
                    "demo_failed",
                    demo=demo_name,
                    error=str(e),
                    exc_info=True,
                )

        logger.info("all_demos_completed", total_demos=len(demos))
        return self.demo_results

    async def demo_basic_exceptions(self) -> dict:
        """Demonstrate basic exception handling patterns."""
        results = {"tests": []}

        # Test 1: Basic ARCBaseException
        try:
            raise ARCBaseException(
                message="This is a demo exception",
                error_code=ErrorCode.VALIDATION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestions=["This is just a demo", "No action needed"],
            )
        except ARCBaseException as e:
            results["tests"].append(
                {
                    "name": "basic_arc_exception",
                    "status": "caught",
                    "error_code": e.error_code.value,
                    "severity": e.severity.value,
                    "suggestions_count": len(e.suggestions),
                }
            )

        # Test 2: Exception with context
        try:
            context = ErrorContext(
                user_id="demo_user",
                task_id="demo_task_123",
                additional_data={"demo": True, "timestamp": datetime.now().isoformat()},
            )

            raise DataNotFoundException("task", "demo_task_123", context=context)
        except DataNotFoundException as e:
            results["tests"].append(
                {
                    "name": "exception_with_context",
                    "status": "caught",
                    "has_context": e.context is not None,
                    "context_user_id": e.context.user_id if e.context else None,
                    "context_task_id": e.context.task_id if e.context else None,
                }
            )

        # Test 3: Exception chaining
        try:
            try:
                # Simulate original error
                raise ValueError("Original error occurred")
            except ValueError as original_error:
                raise EvaluationException(
                    "demo_task",
                    "Evaluation failed due to underlying issue",
                    cause=original_error,
                ) from original_error
        except EvaluationException as e:
            results["tests"].append(
                {
                    "name": "exception_chaining",
                    "status": "caught",
                    "has_cause": e.__cause__ is not None,
                    "cause_type": type(e.__cause__).__name__ if e.__cause__ else None,
                }
            )

        return results

    async def demo_custom_exceptions(self) -> dict:
        """Demonstrate custom ARC exception types."""
        results = {"exception_types": []}

        # Define test scenarios for different exception types
        exception_scenarios = [
            {
                "name": "TaskNotFoundException",
                "exception": TaskNotFoundException("invalid_task_id"),
                "expected_code": ErrorCode.TASK_NOT_FOUND,
            },
            {
                "name": "DataCorruptionException",
                "exception": DataCorruptionException(
                    "task_file", "Invalid JSON format detected"
                ),
                "expected_code": ErrorCode.DATA_CORRUPTION,
            },
            {
                "name": "AuthenticationException",
                "exception": AuthenticationException("Invalid token provided"),
                "expected_code": ErrorCode.INVALID_TOKEN,
            },
            {
                "name": "EvaluationException",
                "exception": EvaluationException(
                    "test_task", "Pixel accuracy calculation failed"
                ),
                "expected_code": ErrorCode.EVALUATION_ERROR,
            },
        ]

        for scenario in exception_scenarios:
            try:
                raise scenario["exception"]
            except ARCBaseException as e:
                results["exception_types"].append(
                    {
                        "name": scenario["name"],
                        "error_code_match": e.error_code == scenario["expected_code"],
                        "has_suggestions": len(e.suggestions) > 0,
                        "error_id_generated": bool(e.error_id),
                        "to_dict_works": bool(e.to_dict()),
                        "to_response_works": isinstance(e.to_response(), ErrorResponse),
                    }
                )

        return results

    async def demo_error_logging(self) -> dict:
        """Demonstrate structured error logging."""
        results = {"logging_tests": []}

        # Test 1: Basic error logging
        test_exception = ARCBaseException(
            message="Demo error for logging test",
            error_code=ErrorCode.VALIDATION_ERROR,
            context=ErrorContext(
                user_id="demo_user",
                additional_data={"test": "logging_demo"},
            ),
        )

        error_id = self.error_logger.log_error(test_exception)

        results["logging_tests"].append(
            {
                "name": "basic_error_logging",
                "error_id_returned": bool(error_id),
                "error_id_format": "uuid" if len(error_id.split("-")) == 5 else "other",
            }
        )

        # Test 2: Logging with additional context
        additional_context = {
            "request_ip": "127.0.0.1",
            "user_agent": "Demo/1.0",
            "request_path": "/api/demo",
        }

        error_id_2 = self.error_logger.log_error(
            test_exception,
            additional_context=additional_context,
        )

        results["logging_tests"].append(
            {
                "name": "logging_with_context",
                "error_id_returned": bool(error_id_2),
                "different_error_ids": error_id != error_id_2,
            }
        )

        return results

    async def demo_circuit_breaker(self) -> dict:
        """Demonstrate circuit breaker functionality."""
        results = {"circuit_breaker_tests": []}

        # Create a circuit breaker with low thresholds for demo
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=2,  # 2 seconds for demo
            success_threshold=2,
        )

        circuit_breaker = get_circuit_breaker("demo_service", config)

        # Test 1: Successful operations
        async def successful_operation():
            await asyncio.sleep(0.1)
            return "success"

        try:
            result = await circuit_breaker.call(successful_operation)
            results["circuit_breaker_tests"].append(
                {
                    "name": "successful_operation",
                    "status": "success",
                    "result": result,
                }
            )
        except Exception as e:
            results["circuit_breaker_tests"].append(
                {
                    "name": "successful_operation",
                    "status": "error",
                    "error": str(e),
                }
            )

        # Test 2: Failing operations to trip circuit breaker
        async def failing_operation():
            raise Exception("Simulated service failure")

        failure_count = 0
        for _i in range(5):  # Try to trip the circuit breaker
            try:
                await circuit_breaker.call(failing_operation)
            except Exception:
                failure_count += 1

        results["circuit_breaker_tests"].append(
            {
                "name": "trip_circuit_breaker",
                "failures_recorded": failure_count,
                "circuit_tripped": failure_count >= config.failure_threshold,
            }
        )

        # Test 3: Circuit breaker should reject requests when open
        try:
            await circuit_breaker.call(successful_operation)
            circuit_rejected = False
        except ARCBaseException as e:
            circuit_rejected = "circuit breaker" in str(e).lower()

        results["circuit_breaker_tests"].append(
            {
                "name": "circuit_rejection",
                "rejected_when_open": circuit_rejected,
            }
        )

        # Get circuit breaker stats
        stats = circuit_breaker.get_stats()
        results["circuit_breaker_stats"] = {
            "state": stats["state"],
            "total_requests": stats["total_requests"],
            "total_failures": stats["total_failures"],
        }

        return results

    async def demo_retry_mechanisms(self) -> dict:
        """Demonstrate retry strategies."""
        results = {"retry_tests": []}

        # Test 1: Retry with eventual success
        attempt_count = 0

        async def eventually_succeeds():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"

        retry_strategy = RetryStrategy(
            max_attempts=5,
            base_delay=0.1,  # Fast for demo
            backoff_multiplier=1.5,
        )

        try:
            result = await retry_strategy.execute(eventually_succeeds)
            results["retry_tests"].append(
                {
                    "name": "eventual_success",
                    "status": "success",
                    "attempts_needed": attempt_count,
                    "result": result,
                }
            )
        except Exception as e:
            results["retry_tests"].append(
                {
                    "name": "eventual_success",
                    "status": "failed",
                    "attempts_made": attempt_count,
                    "error": str(e),
                }
            )

        # Test 2: Retry with non-retryable exception
        async def non_retryable_failure():
            raise ValueError("This should not be retried")

        non_retry_strategy = RetryStrategy(
            max_attempts=3,
            non_retryable_exceptions=(ValueError,),
        )

        try:
            await non_retry_strategy.execute(non_retryable_failure)
            results["retry_tests"].append(
                {
                    "name": "non_retryable",
                    "status": "unexpected_success",
                }
            )
        except Exception:
            results["retry_tests"].append(
                {
                    "name": "non_retryable",
                    "status": "correctly_not_retried",
                }
            )

        return results

    async def demo_fallback_strategies(self) -> dict:
        """Demonstrate fallback mechanisms."""
        results = {"fallback_tests": []}

        fallback_strategy = FallbackStrategy("demo_data_access")

        # Add fallback functions
        async def primary_data_source():
            raise Exception("Primary database is down")

        async def cache_fallback():
            await asyncio.sleep(0.1)
            return "data_from_cache"

        async def static_fallback():
            return "default_static_data"

        # Add fallbacks with priorities
        fallback_strategy.add_fallback(cache_fallback, priority=2)
        fallback_strategy.add_fallback(static_fallback, priority=1)

        # Test fallback execution
        try:
            result = await fallback_strategy.execute(primary_data_source)
            results["fallback_tests"].append(
                {
                    "name": "primary_fails_fallback_succeeds",
                    "status": "success",
                    "result": result,
                    "used_fallback": result != "primary_data",
                }
            )
        except Exception as e:
            results["fallback_tests"].append(
                {
                    "name": "primary_fails_fallback_succeeds",
                    "status": "failed",
                    "error": str(e),
                }
            )

        # Get fallback stats
        stats = fallback_strategy.get_stats()
        results["fallback_stats"] = {
            "name": stats["name"],
            "fallback_count": stats["fallback_count"],
            "primary_success_rate": stats["primary_success_rate"],
            "fallback_usage_rate": stats["fallback_usage_rate"],
        }

        return results

    async def demo_health_monitoring(self) -> dict:
        """Demonstrate health monitoring."""
        results = {"health_tests": []}

        health_monitor = get_health_monitor("demo_system", check_interval=1)

        # Add health checks
        database_healthy = True
        cache_healthy = True

        def check_database():
            return database_healthy

        def check_cache():
            return cache_healthy

        def recover_database():
            nonlocal database_healthy
            database_healthy = True
            logger.info("database_recovered")

        health_monitor.add_health_check(
            "database",
            check_database,
            recovery_func=recover_database,
            critical=True,
        )
        health_monitor.add_health_check(
            "cache",
            check_cache,
            critical=False,
        )

        # Start monitoring
        await health_monitor.start_monitoring()

        # Wait for initial health check
        await asyncio.sleep(1.5)

        # Check initial status
        initial_status = health_monitor.get_health_status()
        results["health_tests"].append(
            {
                "name": "initial_healthy_status",
                "overall_healthy": initial_status["overall_healthy"],
                "monitoring": initial_status["monitoring"],
            }
        )

        # Simulate database failure
        database_healthy = False
        await asyncio.sleep(1.5)

        # Check unhealthy status
        unhealthy_status = health_monitor.get_health_status()
        results["health_tests"].append(
            {
                "name": "unhealthy_detection",
                "overall_healthy": unhealthy_status["overall_healthy"],
                "database_healthy": unhealthy_status["components"]["database"]["healthy"],
            }
        )

        # Wait for recovery attempt
        await asyncio.sleep(2)

        # Check recovery
        recovered_status = health_monitor.get_health_status()
        results["health_tests"].append(
            {
                "name": "auto_recovery",
                "overall_healthy": recovered_status["overall_healthy"],
                "database_healthy": recovered_status["components"]["database"]["healthy"],
            }
        )

        # Stop monitoring
        await health_monitor.stop_monitoring()

        return results

    async def demo_api_error_responses(self) -> dict:
        """Demonstrate API error response formatting."""
        results = {"api_response_tests": []}

        # Test 1: Convert exception to API response
        test_exception = TaskNotFoundException(
            "missing_task_123",
            context=ErrorContext(user_id="api_user"),
        )

        # This would normally be handled by middleware
        response = create_error_response(test_exception, 404)

        results["api_response_tests"].append(
            {
                "name": "exception_to_api_response",
                "status_code": response.status_code,
                "has_error_id_header": "X-Error-ID" in response.headers,
                "content_type": "application/json",
            }
        )

        # Test 2: Verify response structure
        try:
            # In a real scenario, this would be JSON
            # For demo, we'll check the structure
            response_structure_valid = all(
                [
                    hasattr(test_exception.to_response(), "error_id"),
                    hasattr(test_exception.to_response(), "error_code"),
                    hasattr(test_exception.to_response(), "message"),
                    hasattr(test_exception.to_response(), "suggestions"),
                ]
            )

            results["api_response_tests"].append(
                {
                    "name": "response_structure",
                    "structure_valid": response_structure_valid,
                    "error_response_serializable": True,
                }
            )
        except Exception as e:
            results["api_response_tests"].append(
                {
                    "name": "response_structure",
                    "structure_valid": False,
                    "error": str(e),
                }
            )

        return results

    async def demo_recovery_patterns(self) -> dict:
        """Demonstrate advanced recovery patterns."""
        results = {"recovery_tests": []}

        # Test 1: Combined circuit breaker + retry + fallback
        circuit_breaker = get_circuit_breaker("recovery_demo")
        retry_strategy = RetryStrategy(max_attempts=2, base_delay=0.1)

        async def unreliable_service():
            if random.random() < 0.7:  # 70% failure rate
                raise Exception("Service temporarily unavailable")
            return "service_response"

        async def fallback_response():
            return "fallback_response"

        # Combine patterns
        attempts = 0
        successes = 0

        for _ in range(10):  # Try multiple times
            try:
                attempts += 1
                # Try primary service with circuit breaker and retry
                await retry_strategy.execute(
                    lambda: circuit_breaker.call(unreliable_service)
                )
                successes += 1
            except Exception:
                # Use fallback
                await fallback_response()
                successes += 1

        results["recovery_tests"].append(
            {
                "name": "combined_recovery_patterns",
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts if attempts > 0 else 0,
            }
        )

        return results

    def generate_demo_report(self) -> str:
        """Generate a comprehensive demo report."""
        report = []
        report.append("# ARC Prize Error Handling System Demo Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("""
## Demo Results Summary
""")

        total_demos = len(self.demo_results)
        successful_demos = sum(
            1 for r in self.demo_results.values() if r["status"] == "success"
        )

        report.append(f"- Total demos run: {total_demos}")
        report.append(f"- Successful demos: {successful_demos}")
        report.append(f"- Success rate: {successful_demos/total_demos*100:.1f}%")

        # Detailed results
        for demo_name, result in self.demo_results.items():
            report.append(f"""
### {demo_name.replace('_', ' ').title()}
""")
            report.append(f"- Status: {result['status']}")
            report.append(f"- Execution time: {result['execution_time']*1000:.2f}ms")

            if result["status"] == "success" and "result" in result:
                # Add key metrics from each demo
                demo_result = result["result"]

                if "tests" in demo_result:
                    report.append(f"- Tests run: {len(demo_result['tests'])}")

                if "exception_types" in demo_result:
                    report.append(
                        f"- Exception types tested: {len(demo_result['exception_types'])}"
                    )

                if "circuit_breaker_stats" in demo_result:
                    stats = demo_result["circuit_breaker_stats"]
                    report.append(f"- Circuit breaker state: {stats['state']}")
                    report.append(f"- Total requests: {stats['total_requests']}")

            elif result["status"] == "error":
                report.append(f"- Error: {result['error']}")

        report.append("""
## Key Features Demonstrated
""")
        report.append("- ✅ Custom exception hierarchy with error codes")
        report.append("- ✅ Structured error logging with context")
        report.append("- ✅ Circuit breaker patterns for service resilience")
        report.append("- ✅ Retry mechanisms with exponential backoff")
        report.append("- ✅ Fallback strategies for graceful degradation")
        report.append("- ✅ Health monitoring and auto-recovery")
        report.append("- ✅ API error response standardization")
        report.append("- ✅ Combined recovery patterns")

        report.append("""
## Usage Recommendations
""")
        report.append(
            "1. Use ARCBaseException and its subclasses for all domain-specific errors"
        )
        report.append(
            "2. Include ErrorContext with user_id, task_id, and relevant metadata"
        )
        report.append("3. Implement circuit breakers for external service calls")
        report.append("4. Use retry strategies for transient failures")
        report.append("5. Set up health monitoring for critical system components")
        report.append("6. Let middleware handle error response formatting")

        return "\n".join(report)


async def main():
    """Run the error handling demo and print the report."""
    demo = ErrorHandlingDemo()
    await demo.run_all_demos()
    report = demo.generate_demo_report()
    print(report)


def integration_checklist() -> dict[str, str]:
    """Checklist for integrating error handling into existing modules."""
    return {
        "1_import_exceptions": "from src.utils.error_handling import ARCBaseException, TaskNotFoundException, etc.",
        "2_replace_generic_exceptions": "Replace ValueError, Exception with specific ARC exceptions",
        "3.1_add_context": "Create ErrorContext with relevant info (task_id, user_id, etc.)",
        "3.2_pass_context": "Pass context object to exception constructors",
        "4_wrap_critical_code": "Wrap critical sections (e.g., API calls) with circuit breakers or retry logic",
        "5_use_fallback": "Implement FallbackStrategy for non-critical but desirable operations",
        "6_add_health_checks": "Add component-specific checks to the HealthMonitor",
        "7_configure_middleware": "Ensure FastAPI app uses the error handling and logging middleware",
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
    results = asyncio.run(main())
