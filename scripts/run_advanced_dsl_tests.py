#!/usr/bin/env python3
"""
Test runner for advanced DSL operations integration tests.

This script runs the comprehensive integration tests for advanced DSL operations
including color manipulations, pattern operations, geometric transforms,
connectivity analysis, edge detection, and symmetry operations.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list[str], cwd: str = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,  # 5 minute timeout
            cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 5 minutes"
    except Exception as e:
        return -1, "", f"Command failed: {e}"


def run_pytest_with_coverage(test_file: str, markers: str = None) -> dict[str, Any]:
    """Run pytest on a specific test file with coverage reporting."""
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--durations=10",
        "--strict-markers"
    ]

    if markers:
        cmd.extend(["-m", markers])

    # Add coverage if available
    try:
        import coverage
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/advanced_dsl"
        ])
    except ImportError:
        print("Coverage not available, running without coverage reporting")

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()

    exit_code, stdout, stderr = run_command(cmd, cwd=str(project_root))

    duration = time.time() - start_time

    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration": duration,
        "test_file": test_file
    }


def parse_pytest_results(stdout: str) -> dict[str, Any]:
    """Parse pytest output to extract test results."""
    lines = stdout.split('\n')

    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "warnings": 0,
        "total_duration": 0.0,
        "slowest_tests": []
    }

    for line in lines:
        if "passed" in line and "failed" in line:
            # Parse summary line like "5 passed, 2 failed in 1.23s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed,":
                    try:
                        results["passed"] = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                elif part == "failed,":
                    try:
                        results["failed"] = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                elif part == "skipped,":
                    try:
                        results["skipped"] = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass

        # Parse duration info
        if "slowest durations" in line.lower():
            # Next lines contain slowest tests
            continue

        if line.strip().endswith("s") and "::" in line:
            # Parse individual test duration
            parts = line.split()
            if len(parts) >= 2:
                try:
                    duration = float(parts[0][:-1])  # Remove 's' suffix
                    test_name = parts[1]
                    results["slowest_tests"].append((test_name, duration))
                except ValueError:
                    pass

    return results


def print_summary(results: list[dict[str, Any]]):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("ADVANCED DSL OPERATIONS TEST SUMMARY")
    print("="*80)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_duration = 0.0
    failed_files = []

    for result in results:
        print(f"\nFile: {result['test_file']}")
        print(f"Exit Code: {result['exit_code']}")
        print(f"Duration: {result['duration']:.2f}s")

        if result['exit_code'] == 0:
            print("Status: PASSED")
        else:
            print("Status: FAILED")
            failed_files.append(result['test_file'])

        # Parse detailed results if available
        parsed = parse_pytest_results(result['stdout'])
        if parsed['passed'] > 0 or parsed['failed'] > 0:
            print(f"Tests: {parsed['passed']} passed, {parsed['failed']} failed, {parsed['skipped']} skipped")
            total_passed += parsed['passed']
            total_failed += parsed['failed']
            total_skipped += parsed['skipped']

        total_duration += result['duration']

    print("\nOVERALL SUMMARY:")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    print(f"Total Duration: {total_duration:.2f}s")

    if failed_files:
        print("\nFAILED FILES:")
        for file in failed_files:
            print(f"  - {file}")

    print("="*80)


def main():
    """Main test runner function."""
    print("Advanced DSL Operations Integration Test Runner")
    print("=" * 50)

    # Check if we're in the right directory
    if not (project_root / "src" / "domain" / "dsl").exists():
        print("Error: Not in project root directory or DSL module not found")
        return 1

    # Define test files to run
    test_files = [
        "tests/integration/test_advanced_dsl_operations.py",
        "tests/integration/test_dsl_to_python_integration.py"  # Run existing tests too for comparison
    ]

    # Check that test files exist
    missing_files = []
    for test_file in test_files:
        if not (project_root / test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print("Error: Missing test files:")
        for file in missing_files:
            print(f"  - {file}")
        return 1

    # Run tests
    results = []

    print(f"\nRunning {len(test_files)} test files...")

    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print('='*60)

        result = run_pytest_with_coverage(test_file)
        results.append(result)

        # Print immediate feedback
        if result['exit_code'] == 0:
            print(f"✓ {test_file} PASSED ({result['duration']:.2f}s)")
        else:
            print(f"✗ {test_file} FAILED ({result['duration']:.2f}s)")
            if result['stderr']:
                print(f"Error output:\n{result['stderr']}")

    # Print comprehensive summary
    print_summary(results)

    # Return overall exit code
    overall_success = all(r['exit_code'] == 0 for r in results)
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
