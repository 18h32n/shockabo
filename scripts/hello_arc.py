#!/usr/bin/env python3
"""
Hello ARC - Cross-platform validation script for ARC Prize 2025.
Tests that the development environment is properly configured across all platforms.
"""

import json
import logging
import platform as py_platform
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from infrastructure.config import Platform, PlatformDetector, get_config
except ImportError as e:
    print(f"ERROR: Cannot import configuration modules: {e}")
    print("Make sure the src directory is properly set up and PYTHONPATH is correct")
    sys.exit(1)


def setup_logging() -> logging.Logger:
    """Set up logging for Hello ARC test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def print_banner():
    """Print the Hello ARC banner."""
    banner = """
    ================================================================
    
                        HELLO ARC!
    
                ARC Prize 2025 Environment Test
              Cross-Platform Configuration Validator
    
    ================================================================
    """
    print(banner)


def test_basic_python_environment(logger: logging.Logger) -> dict[str, Any]:
    """Test basic Python environment."""
    logger.info("Testing basic Python environment...")

    results = {
        'test_name': 'Basic Python Environment',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    try:
        # Test Python version
        python_version = sys.version_info
        results['details']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"

        if python_version.major != 3 or python_version.minor < 10:
            results['errors'].append(f"Python version {results['details']['python_version']} may not be fully supported")

        # Test platform info
        results['details']['system_platform'] = py_platform.system()
        results['details']['machine'] = py_platform.machine()
        results['details']['python_implementation'] = py_platform.python_implementation()

        # Test basic imports
        results['details']['basic_imports'] = 'OK'

        logger.info("[PASS] Basic Python environment: PASSED")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"Basic Python test failed: {str(e)}")
        logger.error(f"[FAIL] Basic Python environment: FAILED - {e}")

    return results


def test_platform_detection(logger: logging.Logger) -> dict[str, Any]:
    """Test platform detection functionality."""
    logger.info("Testing platform detection...")

    results = {
        'test_name': 'Platform Detection',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    try:
        # Test platform detection
        detected_platform = PlatformDetector.detect_platform()
        results['details']['detected_platform'] = detected_platform.value

        # Get platform info
        platform_info = PlatformDetector.get_platform_info()
        results['details']['gpu_hours_limit'] = platform_info.gpu_hours_limit
        results['details']['max_memory_gb'] = platform_info.max_memory_gb
        results['details']['has_persistent_storage'] = platform_info.has_persistent_storage

        # Test resource limits
        resource_limits = PlatformDetector.get_resource_limits()
        results['details']['resource_limits'] = resource_limits

        # Test GPU detection
        gpu_available = PlatformDetector.is_gpu_available()
        results['details']['gpu_available'] = gpu_available

        logger.info(f"[PASS] Platform detection: PASSED (Platform: {detected_platform.value})")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"Platform detection failed: {str(e)}")
        logger.error(f"[FAIL] Platform detection: FAILED - {e}")

    return results


def test_configuration_loading(logger: logging.Logger) -> dict[str, Any]:
    """Test configuration loading and management."""
    logger.info("Testing configuration loading...")

    results = {
        'test_name': 'Configuration Loading',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    try:
        # Test configuration loading
        config = get_config()

        # Test basic configuration access
        platform_name = config.get('platform.name', 'unknown')
        results['details']['config_platform'] = platform_name

        # Test directory configuration
        data_dir = config.get_data_dir()
        output_dir = config.get_output_dir()
        cache_dir = config.get_cache_dir()

        results['details']['data_dir'] = str(data_dir)
        results['details']['output_dir'] = str(output_dir)
        results['details']['cache_dir'] = str(cache_dir)

        # Test platform info from config
        platform_info = config.get_platform_info()
        results['details']['platform_from_config'] = platform_info.platform.value

        # Test development mode detection
        is_dev = config.is_development()
        results['details']['development_mode'] = is_dev

        logger.info("[PASS] Configuration loading: PASSED")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"Configuration loading failed: {str(e)}")
        logger.error(f"[FAIL] Configuration loading: FAILED - {e}")

    return results


def test_directory_access(logger: logging.Logger) -> dict[str, Any]:
    """Test directory access and permissions."""
    logger.info("Testing directory access...")

    results = {
        'test_name': 'Directory Access',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    try:
        config = get_config()

        # Test directory creation and access
        directories_to_test = [
            ('data', config.get_data_dir()),
            ('output', config.get_output_dir()),
            ('cache', config.get_cache_dir())
        ]

        for dir_name, dir_path in directories_to_test:
            try:
                # Create directory if it doesn't exist
                dir_path.mkdir(parents=True, exist_ok=True)

                # Test write access
                test_file = dir_path / f'test_{dir_name}.txt'
                test_file.write_text(f'Hello ARC test file for {dir_name}')

                # Test read access
                content = test_file.read_text()
                if content.strip() != f'Hello ARC test file for {dir_name}':
                    raise ValueError(f"Content mismatch in {dir_name} directory")

                # Clean up test file
                test_file.unlink()

                results['details'][f'{dir_name}_access'] = 'OK'
                logger.info(f"[PASS] {dir_name} directory access: OK")

            except Exception as e:
                results['errors'].append(f"{dir_name} directory access failed: {str(e)}")
                results['details'][f'{dir_name}_access'] = 'FAIL'
                logger.error(f"[FAIL] {dir_name} directory access: FAILED - {e}")

        if results['errors']:
            results['status'] = 'FAIL'
        else:
            logger.info("[PASS] Directory access: PASSED")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"Directory access test failed: {str(e)}")
        logger.error(f"[FAIL] Directory access: FAILED - {e}")

    return results


def test_dependencies(logger: logging.Logger) -> dict[str, Any]:
    """Test that required dependencies are available."""
    logger.info("Testing dependencies...")

    results = {
        'test_name': 'Dependencies',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    # List of critical dependencies to test
    dependencies_to_test = [
        ('numpy', 'numpy'),
        ('psutil', 'psutil'),
        ('yaml', 'pyyaml'),
        ('dotenv', 'python-dotenv'),
        ('dataclasses', None),  # Built-in Python 3.7+
        ('pathlib', None),      # Built-in Python 3.4+
        ('typing', None),       # Built-in Python 3.5+
        ('json', None),         # Built-in
        ('os', None),           # Built-in
        ('sys', None),          # Built-in
    ]

    try:
        for module_name, package_name in dependencies_to_test:
            try:
                __import__(module_name)
                results['details'][f'{module_name}_import'] = 'OK'
                logger.info(f"[PASS] {module_name}: OK")
            except ImportError as e:
                results['errors'].append(f"Cannot import {module_name}: {str(e)}")
                results['details'][f'{module_name}_import'] = 'FAIL'
                logger.error(f"[FAIL] {module_name}: FAILED - {e}")

        if results['errors']:
            results['status'] = 'FAIL'
        else:
            logger.info("[PASS] Dependencies: PASSED")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"Dependencies test failed: {str(e)}")
        logger.error(f"[FAIL] Dependencies: FAILED - {e}")

    return results


def test_arc_sample_computation(logger: logging.Logger) -> dict[str, Any]:
    """Test a simple ARC-like computation to verify everything works."""
    logger.info("Testing ARC sample computation...")

    results = {
        'test_name': 'ARC Sample Computation',
        'status': 'PASS',
        'details': {},
        'errors': []
    }

    try:
        import numpy as np

        # Create a simple ARC-like grid transformation
        # Input: 3x3 grid with some pattern
        input_grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        # Simple transformation: invert the pattern
        expected_output = 1 - input_grid

        # Perform the transformation
        actual_output = 1 - input_grid

        # Verify the transformation
        if np.array_equal(expected_output, actual_output):
            results['details']['grid_transformation'] = 'OK'
            results['details']['input_shape'] = input_grid.shape
            results['details']['output_shape'] = actual_output.shape
            logger.info("[PASS] Grid transformation: OK")
        else:
            results['errors'].append("Grid transformation produced incorrect result")
            results['status'] = 'FAIL'

        # Test basic numpy operations
        sum_result = np.sum(input_grid)
        results['details']['grid_sum'] = int(sum_result)

        # Test grid shape manipulation
        flattened = input_grid.flatten()
        reshaped = flattened.reshape(input_grid.shape)

        if np.array_equal(input_grid, reshaped):
            results['details']['shape_manipulation'] = 'OK'
            logger.info("[PASS] Shape manipulation: OK")
        else:
            results['errors'].append("Shape manipulation failed")
            results['status'] = 'FAIL'

        if results['status'] == 'PASS':
            logger.info("[PASS] ARC sample computation: PASSED")

    except Exception as e:
        results['status'] = 'FAIL'
        results['errors'].append(f"ARC sample computation failed: {str(e)}")
        logger.error(f"[FAIL] ARC sample computation: FAILED - {e}")

    return results


def generate_report(test_results: list[dict[str, Any]], logger: logging.Logger) -> dict[str, Any]:
    """Generate a comprehensive test report."""
    logger.info("Generating test report...")

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['status'] == 'PASS')
    failed_tests = total_tests - passed_tests

    # Platform information
    try:
        platform = PlatformDetector.detect_platform()
        platform_info = PlatformDetector.get_platform_info()
        resource_limits = PlatformDetector.get_resource_limits()
    except Exception as e:
        platform = "Unknown"
        platform_info = None
        resource_limits = {}
        logger.error(f"Could not get platform info for report: {e}")

    report = {
        'test_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        },
        'platform_info': {
            'detected_platform': platform.value if hasattr(platform, 'value') else str(platform),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'system_platform': py_platform.system(),
            'machine': py_platform.machine(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        },
        'resource_info': resource_limits,
        'test_results': test_results
    }

    if platform_info:
        report['platform_info']['gpu_hours_limit'] = platform_info.gpu_hours_limit
        report['platform_info']['max_memory_gb'] = platform_info.max_memory_gb
        report['platform_info']['has_persistent_storage'] = platform_info.has_persistent_storage

    return report


def save_report(report: dict[str, Any], logger: logging.Logger) -> str | None:
    """Save the test report to file."""
    try:
        config = get_config()
        output_dir = config.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and platform
        platform_name = report['platform_info']['detected_platform']
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        filename = f'hello_arc_report_{platform_name}_{timestamp}.json'

        report_path = output_dir / filename

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"[PASS] Report saved: {report_path}")
        return str(report_path)

    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return None


def print_summary(report: dict[str, Any]):
    """Print a summary of test results."""
    summary = report['test_summary']
    platform_info = report['platform_info']

    print("\n" + "="*60)
    print("HELLO ARC - TEST SUMMARY")
    print("="*60)

    print(f"Platform: {platform_info['detected_platform']}")
    print(f"Python Version: {platform_info['python_version']}")
    print(f"System: {platform_info['system_platform']} ({platform_info['machine']})")
    print(f"Timestamp: {platform_info['timestamp']}")

    print("\nTest Results:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']} [PASS]")
    print(f"  Failed: {summary['failed_tests']} [FAIL]")
    print(f"  Success Rate: {summary['success_rate']}")

    print(f"\nOverall Status: {summary['overall_status']}")

    if summary['failed_tests'] > 0:
        print("\nFailed Tests:")
        for result in report['test_results']:
            if result['status'] == 'FAIL':
                print(f"  [FAIL] {result['test_name']}")
                for error in result['errors']:
                    print(f"    - {error}")

    print("="*60)

    if summary['overall_status'] == 'PASS':
        print("*** Congratulations! Your ARC Prize 2025 environment is ready!")
        print("You can now start developing ARC solutions.")
    else:
        print("*** Some tests failed. Please check the errors above.")
        print("You may need to run the platform setup script for your platform:")
        print("  Kaggle: python scripts/platform_deploy/kaggle_setup.py")
        print("  Colab: python scripts/platform_deploy/colab_setup.py")
        print("  Paperspace: python scripts/platform_deploy/paperspace_setup.py")

    print("="*60)


def main():
    """Main function for Hello ARC test."""
    print_banner()

    logger = setup_logging()
    logger.info("Starting Hello ARC validation tests...")

    # Run all tests
    test_functions = [
        test_basic_python_environment,
        test_platform_detection,
        test_configuration_loading,
        test_directory_access,
        test_dependencies,
        test_arc_sample_computation
    ]

    test_results = []

    for test_func in test_functions:
        try:
            result = test_func(logger)
            test_results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            test_results.append({
                'test_name': test_func.__name__.replace('test_', '').replace('_', ' ').title(),
                'status': 'FAIL',
                'details': {},
                'errors': [f"Test crashed: {str(e)}"]
            })

    # Generate and save report
    report = generate_report(test_results, logger)
    report_path = save_report(report, logger)

    # Print summary
    print_summary(report)

    if report_path:
        print(f"\nDetailed report saved to: {report_path}")

    # Exit with appropriate code
    overall_status = report['test_summary']['overall_status']
    sys.exit(0 if overall_status == 'PASS' else 1)


if __name__ == "__main__":
    main()
