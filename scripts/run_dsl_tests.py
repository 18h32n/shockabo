#!/usr/bin/env python3
"""
DSL Test Runner Script

This script runs all DSL tests, generates coverage reports, runs performance benchmarks,
and outputs a comprehensive summary report.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], capture_output: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command with error handling and timeout."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}")
        raise
    except Exception as e:
        print(f"Command failed: {' '.join(cmd)} - {str(e)}")
        raise


class DSLTestRunner:
    """Comprehensive DSL test runner with reporting."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "coverage": {},
            "errors": []
        }

        # Ensure we're in the right directory
        os.chdir(self.project_root)

        # Check if required directories exist
        self.test_dirs = {
            "unit": self.project_root / "tests" / "unit" / "domain" / "dsl",
            "integration": self.project_root / "tests" / "integration",
            "performance": self.project_root / "tests" / "performance"
        }

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("Checking prerequisites...")

        # Check if pytest is available
        result = run_command([sys.executable, "-m", "pytest", "--version"])
        if result.returncode != 0:
            self.results["errors"].append("pytest not available")
            return False

        # Check if coverage is available
        result = run_command([sys.executable, "-m", "coverage", "--version"])
        if result.returncode != 0:
            print("Warning: coverage not available, skipping coverage reports")

        # Check if test directories exist
        missing_dirs = []
        for name, path in self.test_dirs.items():
            if not path.exists():
                missing_dirs.append(f"{name}: {path}")

        if missing_dirs:
            self.results["errors"].extend([f"Missing test directory: {d}" for d in missing_dirs])
            print("Warning: Some test directories missing:", missing_dirs)

        # Check if DSL modules are importable
        try:
            result = run_command([
                sys.executable, "-c",
                "import src.domain.dsl; print('DSL modules available')"
            ])
            if result.returncode != 0:
                self.results["errors"].append("DSL modules not importable")
                return False
        except Exception as e:
            self.results["errors"].append(f"Failed to check DSL imports: {str(e)}")
            return False

        print("Prerequisites check completed.")
        return True

    def run_unit_tests(self) -> dict[str, Any]:
        """Run unit tests for DSL operations."""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)

        unit_results = {
            "status": "not_run",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0,
            "test_files": {}
        }

        if not self.test_dirs["unit"].exists():
            unit_results["status"] = "directory_missing"
            self.results["errors"].append("Unit test directory missing")
            return unit_results

        start_time = time.time()

        # Run unit tests with verbose output
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dirs["unit"]),
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-x"  # Stop on first failure for faster feedback
        ]

        try:
            result = run_command(cmd, timeout=120)

            unit_results["duration"] = time.time() - start_time
            unit_results["return_code"] = result.returncode
            unit_results["stdout"] = result.stdout
            unit_results["stderr"] = result.stderr

            # Parse pytest output
            if result.returncode == 0:
                unit_results["status"] = "passed"
            else:
                unit_results["status"] = "failed"
                self.results["errors"].append("Unit tests failed")

            # Extract test counts from pytest output
            self._parse_pytest_output(result.stdout, unit_results)

        except subprocess.TimeoutExpired:
            unit_results["status"] = "timeout"
            unit_results["duration"] = 120
            self.results["errors"].append("Unit tests timed out")
        except Exception as e:
            unit_results["status"] = "error"
            unit_results["error"] = str(e)
            self.results["errors"].append(f"Unit test error: {str(e)}")

        self.results["unit_tests"] = unit_results
        return unit_results

    def run_integration_tests(self) -> dict[str, Any]:
        """Run integration tests for DSL chaining."""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)

        integration_results = {
            "status": "not_run",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0
        }

        # Look for DSL chaining tests specifically
        test_files = list(self.test_dirs["integration"].glob("*dsl*"))
        if not test_files:
            integration_results["status"] = "no_tests_found"
            print("No DSL integration tests found")
            return integration_results

        start_time = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
        ] + [str(f) for f in test_files] + [
            "-v",
            "--tb=short",
            "--disable-warnings"
        ]

        try:
            result = run_command(cmd, timeout=180)

            integration_results["duration"] = time.time() - start_time
            integration_results["return_code"] = result.returncode
            integration_results["stdout"] = result.stdout
            integration_results["stderr"] = result.stderr

            if result.returncode == 0:
                integration_results["status"] = "passed"
            else:
                integration_results["status"] = "failed"
                self.results["errors"].append("Integration tests failed")

            self._parse_pytest_output(result.stdout, integration_results)

        except subprocess.TimeoutExpired:
            integration_results["status"] = "timeout"
            integration_results["duration"] = 180
            self.results["errors"].append("Integration tests timed out")
        except Exception as e:
            integration_results["status"] = "error"
            integration_results["error"] = str(e)
            self.results["errors"].append(f"Integration test error: {str(e)}")

        self.results["integration_tests"] = integration_results
        return integration_results

    def run_performance_tests(self) -> dict[str, Any]:
        """Run performance tests and benchmarks."""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)

        performance_results = {
            "status": "not_run",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0,
            "benchmarks": {}
        }

        # Look for DSL performance tests
        test_files = list(self.test_dirs["performance"].glob("*dsl*"))
        if not test_files:
            performance_results["status"] = "no_tests_found"
            print("No DSL performance tests found")
            return performance_results

        start_time = time.time()

        # Run with longer timeout for performance tests
        cmd = [
            sys.executable, "-m", "pytest",
        ] + [str(f) for f in test_files] + [
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-s",  # Don't capture output for performance tests
            "--durations=10"  # Show slowest tests
        ]

        try:
            result = run_command(cmd, timeout=300)

            performance_results["duration"] = time.time() - start_time
            performance_results["return_code"] = result.returncode
            performance_results["stdout"] = result.stdout
            performance_results["stderr"] = result.stderr

            if result.returncode == 0:
                performance_results["status"] = "passed"
            else:
                performance_results["status"] = "failed"
                self.results["errors"].append("Performance tests failed")

            self._parse_pytest_output(result.stdout, performance_results)

        except subprocess.TimeoutExpired:
            performance_results["status"] = "timeout"
            performance_results["duration"] = 300
            self.results["errors"].append("Performance tests timed out")
        except Exception as e:
            performance_results["status"] = "error"
            performance_results["error"] = str(e)
            self.results["errors"].append(f"Performance test error: {str(e)}")

        self.results["performance_tests"] = performance_results
        return performance_results

    def generate_coverage_report(self) -> dict[str, Any]:
        """Generate code coverage report for DSL modules."""
        print("\n" + "="*60)
        print("GENERATING COVERAGE REPORT")
        print("="*60)

        coverage_results = {
            "status": "not_run",
            "total_coverage": 0,
            "module_coverage": {},
            "missing_lines": 0
        }

        try:
            # Run tests with coverage
            cmd = [
                sys.executable, "-m", "coverage", "run",
                "--source=src/domain/dsl",
                "-m", "pytest",
                str(self.test_dirs["unit"]),
                "--tb=no",
                "-q"
            ]

            result = run_command(cmd, timeout=120)

            if result.returncode != 0:
                coverage_results["status"] = "test_failed"
                self.results["errors"].append("Coverage test run failed")
                return coverage_results

            # Generate coverage report
            cmd = [sys.executable, "-m", "coverage", "report", "--format=json"]
            result = run_command(cmd)

            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                coverage_results["status"] = "completed"
                coverage_results["total_coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0)
                coverage_results["files"] = coverage_data.get("files", {})
                coverage_results["totals"] = coverage_data.get("totals", {})

                # Generate HTML report
                html_cmd = [sys.executable, "-m", "coverage", "html", "-d", "htmlcov"]
                html_result = run_command(html_cmd)
                if html_result.returncode == 0:
                    coverage_results["html_report"] = "htmlcov/index.html"

            else:
                coverage_results["status"] = "report_failed"
                coverage_results["error"] = result.stderr
                self.results["errors"].append("Coverage report generation failed")

        except Exception as e:
            coverage_results["status"] = "error"
            coverage_results["error"] = str(e)
            self.results["errors"].append(f"Coverage error: {str(e)}")

        self.results["coverage"] = coverage_results
        return coverage_results

    def run_property_tests(self) -> dict[str, Any]:
        """Run property-based tests with hypothesis."""
        print("\n" + "="*60)
        print("RUNNING PROPERTY-BASED TESTS")
        print("="*60)

        property_results = {
            "status": "not_run",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "duration": 0
        }

        # Look for property test file
        property_test_file = self.test_dirs["unit"] / "test_properties.py"
        if not property_test_file.exists():
            property_results["status"] = "no_tests_found"
            print("Property-based tests not found")
            return property_results

        start_time = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
            str(property_test_file),
            "-v",
            "--tb=short",
            "--disable-warnings",
            "--hypothesis-show-statistics"
        ]

        try:
            result = run_command(cmd, timeout=180)

            property_results["duration"] = time.time() - start_time
            property_results["return_code"] = result.returncode
            property_results["stdout"] = result.stdout
            property_results["stderr"] = result.stderr

            if result.returncode == 0:
                property_results["status"] = "passed"
            else:
                property_results["status"] = "failed"
                self.results["errors"].append("Property-based tests failed")

            self._parse_pytest_output(result.stdout, property_results)

        except subprocess.TimeoutExpired:
            property_results["status"] = "timeout"
            property_results["duration"] = 180
            self.results["errors"].append("Property-based tests timed out")
        except Exception as e:
            property_results["status"] = "error"
            property_results["error"] = str(e)
            self.results["errors"].append(f"Property test error: {str(e)}")

        return property_results

    def _parse_pytest_output(self, output: str, results: dict[str, Any]) -> None:
        """Parse pytest output to extract test statistics."""
        lines = output.split('\n')

        for line in lines:
            # Look for test summary line
            if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                # Parse format like "5 failed, 10 passed, 2 skipped in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        results["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        results["failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        results["skipped"] = int(parts[i-1])

                results["total_tests"] = results["passed"] + results["failed"] + results["skipped"]
                break
            elif "passed in" in line:
                # Simple case: "10 passed in 1.23s"
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "passed":
                    results["passed"] = int(parts[0])
                    results["total_tests"] = results["passed"]

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        summary = []
        summary.append("DSL TEST EXECUTION SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Timestamp: {self.results['timestamp']}")
        summary.append("")

        # Overall status
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = len(self.results["errors"])

        # Unit tests summary
        if self.results["unit_tests"]:
            unit = self.results["unit_tests"]
            summary.append(f"UNIT TESTS: {unit['status'].upper()}")
            summary.append(f"  Tests: {unit.get('total_tests', 0)} total, {unit.get('passed', 0)} passed, {unit.get('failed', 0)} failed")
            summary.append(f"  Duration: {unit.get('duration', 0):.2f}s")
            total_tests += unit.get('total_tests', 0)
            total_passed += unit.get('passed', 0)
            total_failed += unit.get('failed', 0)
            summary.append("")

        # Integration tests summary
        if self.results["integration_tests"]:
            integration = self.results["integration_tests"]
            summary.append(f"INTEGRATION TESTS: {integration['status'].upper()}")
            summary.append(f"  Tests: {integration.get('total_tests', 0)} total, {integration.get('passed', 0)} passed, {integration.get('failed', 0)} failed")
            summary.append(f"  Duration: {integration.get('duration', 0):.2f}s")
            total_tests += integration.get('total_tests', 0)
            total_passed += integration.get('passed', 0)
            total_failed += integration.get('failed', 0)
            summary.append("")

        # Performance tests summary
        if self.results["performance_tests"]:
            performance = self.results["performance_tests"]
            summary.append(f"PERFORMANCE TESTS: {performance['status'].upper()}")
            summary.append(f"  Tests: {performance.get('total_tests', 0)} total, {performance.get('passed', 0)} passed, {performance.get('failed', 0)} failed")
            summary.append(f"  Duration: {performance.get('duration', 0):.2f}s")
            total_tests += performance.get('total_tests', 0)
            total_passed += performance.get('passed', 0)
            total_failed += performance.get('failed', 0)
            summary.append("")

        # Coverage summary
        if self.results["coverage"]:
            coverage = self.results["coverage"]
            summary.append(f"COVERAGE: {coverage['status'].upper()}")
            if coverage.get('total_coverage'):
                summary.append(f"  Total Coverage: {coverage['total_coverage']:.1f}%")
                if coverage['total_coverage'] >= 100:
                    summary.append("  ✓ 100% coverage requirement met")
                else:
                    summary.append("  ✗ Coverage below 100% requirement")
            summary.append("")

        # Overall summary
        summary.append("OVERALL RESULTS")
        summary.append(f"  Total Tests: {total_tests}")
        summary.append(f"  Passed: {total_passed}")
        summary.append(f"  Failed: {total_failed}")
        summary.append(f"  Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
        summary.append(f"  Errors: {total_errors}")

        if total_failed == 0 and total_errors == 0 and total_tests > 0:
            summary.append("  🎉 ALL TESTS PASSED!")
        else:
            summary.append("  ❌ SOME TESTS FAILED")

        # Errors
        if self.results["errors"]:
            summary.append("")
            summary.append("ERRORS:")
            for error in self.results["errors"]:
                summary.append(f"  - {error}")

        summary.append("")
        summary.append("=" * 60)

        return "\n".join(summary)

    def save_results(self, output_file: str | None = None) -> None:
        """Save detailed results to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"dsl_test_results_{timestamp}.json"

        output_path = self.project_root / output_file

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Detailed results saved to: {output_path}")

    def run_all_tests(self) -> bool:
        """Run all DSL tests and generate reports."""
        print("DSL Test Suite Runner")
        print(f"Project root: {self.project_root}")
        print("")

        if not self.check_prerequisites():
            print("Prerequisites check failed. Aborting.")
            return False

        success = True

        # Run all test suites
        unit_results = self.run_unit_tests()
        integration_results = self.run_integration_tests()
        performance_results = self.run_performance_tests()
        property_results = self.run_property_tests()
        coverage_results = self.generate_coverage_report()

        # Check if any tests failed
        if unit_results.get("status") == "failed":
            success = False
        if integration_results.get("status") == "failed":
            success = False
        if performance_results.get("status") == "failed":
            success = False
        if property_results.get("status") == "failed":
            success = False

        # Generate and display summary
        summary = self.generate_summary_report()
        print("\n" + summary)

        # Save detailed results
        self.save_results()

        return success


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DSL test suite with comprehensive reporting")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage report")
    parser.add_argument("--output", type=str, help="Output file for detailed results")

    args = parser.parse_args()

    runner = DSLTestRunner()

    if not runner.check_prerequisites():
        sys.exit(1)

    success = True

    if args.unit_only:
        runner.run_unit_tests()
    elif args.integration_only:
        runner.run_integration_tests()
    elif args.performance_only:
        runner.run_performance_tests()
    else:
        success = runner.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
