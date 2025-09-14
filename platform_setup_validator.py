#!/usr/bin/env python3
"""
Platform Setup Validator for TTT Implementation
==============================================

This script validates platform setup and provides recommendations
for optimal TTT implementation deployment.

Run with: python platform_setup_validator.py
"""

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


@dataclass
class ValidationResult:
    """Platform validation result."""
    check_name: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    recommendation: str | None = None
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PlatformSetupValidator:
    """Validate platform setup for TTT implementation."""

    def __init__(self):
        """Initialize platform setup validator."""
        self.results: list[ValidationResult] = []
        self.platform_info = self._detect_platform()
        print(f"Detected platform: {self.platform_info['platform']}")

    def _detect_platform(self) -> dict[str, Any]:
        """Detect current platform and gather system information."""
        # Platform detection logic
        if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            platform_name = 'kaggle'
        elif 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
            platform_name = 'colab'
        elif 'PS_API_KEY' in os.environ or os.path.exists('/storage'):
            platform_name = 'paperspace'
        else:
            platform_name = 'local'

        # System information
        memory_info = psutil.virtual_memory()

        info = {
            'platform': platform_name,
            'os': platform.system(),
            'os_version': platform.release(),
            'python_version': platform.python_version(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': memory_info.total / (1024**3),
            'available_memory_gb': memory_info.available / (1024**3),
            'disk_usage': {}
        }

        # Disk usage information
        try:
            disk_usage = psutil.disk_usage('/')
            info['disk_usage'] = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3)
            }
        except:
            info['disk_usage'] = {'error': 'Could not determine disk usage'}

        return info

    def _add_result(self, check_name: str, status: str, message: str,
                   recommendation: str = None, details: dict[str, Any] = None):
        """Add validation result."""
        result = ValidationResult(
            check_name=check_name,
            status=status,
            message=message,
            recommendation=recommendation,
            details=details or {}
        )
        self.results.append(result)

        # Print immediate feedback
        status_icon = {'pass': '[PASS]', 'warning': '[WARN]', 'fail': '[FAIL]'}.get(status, '[INFO]')
        print(f"{status_icon} {check_name}: {message}")
        if recommendation:
            print(f"  RECOMMENDATION: {recommendation}")

    def validate_python_environment(self):
        """Validate Python environment and dependencies."""
        print("\nValidating Python Environment...")

        # Python version check
        python_version = tuple(map(int, platform.python_version().split('.')))
        if python_version >= (3, 8):
            self._add_result(
                "Python Version",
                "pass",
                f"Python {platform.python_version()} (supported)"
            )
        else:
            self._add_result(
                "Python Version",
                "fail",
                f"Python {platform.python_version()} (requires 3.8+)",
                "Upgrade to Python 3.8 or higher"
            )

        # Check critical dependencies
        critical_deps = {
            'torch': 'PyTorch for model training',
            'yaml': 'YAML configuration parsing',
            'psutil': 'System resource monitoring',
            'transformers': 'Hugging Face transformers',
            'datasets': 'Hugging Face datasets',
            'peft': 'Parameter-efficient fine-tuning'
        }

        missing_deps = []
        available_deps = []

        for dep_name, description in critical_deps.items():
            try:
                __import__(dep_name)
                available_deps.append(dep_name)
            except ImportError:
                missing_deps.append((dep_name, description))

        if not missing_deps:
            self._add_result(
                "Dependencies",
                "pass",
                f"All critical dependencies available ({len(available_deps)}/{'len(critical_deps)'})",
                details={'available': available_deps}
            )
        else:
            self._add_result(
                "Dependencies",
                "warning" if len(missing_deps) < len(critical_deps) // 2 else "fail",
                f"Missing {len(missing_deps)} dependencies",
                f"Install missing packages: {', '.join(dep[0] for dep in missing_deps)}",
                details={'missing': missing_deps, 'available': available_deps}
            )

    def validate_gpu_setup(self):
        """Validate GPU setup and CUDA availability."""
        print("\nValidating GPU Setup...")

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_info = []

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}"
                    })

                self._add_result(
                    "CUDA Availability",
                    "pass",
                    f"CUDA available with {gpu_count} GPU(s)",
                    details={'gpus': gpu_info}
                )

                # Memory check
                total_gpu_memory = sum(gpu['memory_gb'] for gpu in gpu_info)
                if total_gpu_memory >= 8:
                    self._add_result(
                        "GPU Memory",
                        "pass",
                        f"Sufficient GPU memory: {total_gpu_memory:.1f}GB"
                    )
                elif total_gpu_memory >= 4:
                    self._add_result(
                        "GPU Memory",
                        "warning",
                        f"Limited GPU memory: {total_gpu_memory:.1f}GB",
                        "Use quantization and small batch sizes"
                    )
                else:
                    self._add_result(
                        "GPU Memory",
                        "warning",
                        f"Very limited GPU memory: {total_gpu_memory:.1f}GB",
                        "Consider CPU-only mode or cloud GPU"
                    )
            else:
                self._add_result(
                    "CUDA Availability",
                    "warning",
                    "CUDA not available - CPU mode only",
                    "TTT training will be significantly slower on CPU"
                )

        except ImportError:
            self._add_result(
                "PyTorch",
                "fail",
                "PyTorch not available",
                "Install PyTorch with CUDA support for GPU acceleration"
            )

    def validate_memory_requirements(self):
        """Validate memory requirements for TTT implementation."""
        print("\nValidating Memory Requirements...")

        memory_info = psutil.virtual_memory()
        total_gb = memory_info.total / (1024**3)
        available_gb = memory_info.available / (1024**3)

        # Platform-specific memory requirements
        platform_requirements = {
            'kaggle': {'min': 16, 'recommended': 28},
            'colab': {'min': 8, 'recommended': 12},
            'paperspace': {'min': 8, 'recommended': 16},
            'local': {'min': 8, 'recommended': 16}
        }

        requirements = platform_requirements.get(
            self.platform_info['platform'],
            {'min': 8, 'recommended': 16}
        )

        if total_gb >= requirements['recommended']:
            self._add_result(
                "System Memory",
                "pass",
                f"Excellent memory: {total_gb:.1f}GB (recommended: {requirements['recommended']}GB)"
            )
        elif total_gb >= requirements['min']:
            self._add_result(
                "System Memory",
                "warning",
                f"Adequate memory: {total_gb:.1f}GB (minimum: {requirements['min']}GB)",
                "Consider enabling memory optimizations"
            )
        else:
            self._add_result(
                "System Memory",
                "fail",
                f"Insufficient memory: {total_gb:.1f}GB (minimum: {requirements['min']}GB)",
                "Upgrade system memory or use smaller models"
            )

        # Available memory check
        if available_gb < requirements['min'] * 0.5:
            self._add_result(
                "Available Memory",
                "warning",
                f"Limited available memory: {available_gb:.1f}GB",
                "Close other applications or restart system"
            )

    def validate_storage_requirements(self):
        """Validate storage requirements."""
        print("\nValidating Storage Requirements...")

        disk_info = self.platform_info['disk_usage']
        if 'error' in disk_info:
            self._add_result(
                "Disk Space",
                "warning",
                "Could not determine disk usage",
                "Manually verify sufficient disk space (>10GB recommended)"
            )
            return

        free_gb = disk_info['free_gb']

        # Storage requirements
        requirements = {
            'minimum': 5,    # 5GB minimum
            'recommended': 20,  # 20GB recommended for models + data + checkpoints
            'optimal': 50    # 50GB for full development
        }

        if free_gb >= requirements['optimal']:
            self._add_result(
                "Disk Space",
                "pass",
                f"Excellent storage: {free_gb:.1f}GB free"
            )
        elif free_gb >= requirements['recommended']:
            self._add_result(
                "Disk Space",
                "pass",
                f"Good storage: {free_gb:.1f}GB free"
            )
        elif free_gb >= requirements['minimum']:
            self._add_result(
                "Disk Space",
                "warning",
                f"Limited storage: {free_gb:.1f}GB free",
                "Monitor disk usage during training"
            )
        else:
            self._add_result(
                "Disk Space",
                "fail",
                f"Insufficient storage: {free_gb:.1f}GB free",
                "Free up disk space or use external storage"
            )

    def validate_network_connectivity(self):
        """Validate network connectivity for model downloads."""
        print("\nValidating Network Connectivity...")

        # Test Hugging Face Hub connectivity
        test_urls = [
            ('huggingface.co', 'Hugging Face Hub'),
            ('pypi.org', 'PyPI Package Index'),
            ('github.com', 'GitHub')
        ]

        connectivity_results = []

        for url, description in test_urls:
            try:
                import socket
                socket.create_connection((url, 443), timeout=5)
                connectivity_results.append((url, True))
            except (TimeoutError, OSError):
                connectivity_results.append((url, False))

        successful_connections = sum(1 for _, success in connectivity_results if success)

        if successful_connections == len(test_urls):
            self._add_result(
                "Network Connectivity",
                "pass",
                "All required services accessible"
            )
        elif successful_connections > 0:
            failed_services = [url for url, success in connectivity_results if not success]
            self._add_result(
                "Network Connectivity",
                "warning",
                f"Some services inaccessible: {', '.join(failed_services)}",
                "Check network connection and firewall settings"
            )
        else:
            self._add_result(
                "Network Connectivity",
                "fail",
                "No network connectivity detected",
                "Ensure internet connection for model downloads"
            )

    def validate_platform_specific_requirements(self):
        """Validate platform-specific requirements."""
        print(f"\nValidating {self.platform_info['platform'].title()} Platform Requirements...")

        platform = self.platform_info['platform']

        if platform == 'kaggle':
            # Kaggle-specific validations
            if os.path.exists('/kaggle/input'):
                self._add_result(
                    "Kaggle Input Mount",
                    "pass",
                    "Kaggle input directory accessible"
                )
            else:
                self._add_result(
                    "Kaggle Input Mount",
                    "warning",
                    "Kaggle input directory not found",
                    "Ensure datasets are properly mounted"
                )

            if os.path.exists('/kaggle/working') and os.access('/kaggle/working', os.W_OK):
                self._add_result(
                    "Kaggle Working Directory",
                    "pass",
                    "Kaggle working directory writable"
                )
            else:
                self._add_result(
                    "Kaggle Working Directory",
                    "fail",
                    "Kaggle working directory not writable",
                    "Check kernel permissions"
                )

        elif platform == 'colab':
            # Colab-specific validations
            try:
                import google.colab
                self._add_result(
                    "Colab Environment",
                    "pass",
                    "Google Colab environment detected"
                )
            except ImportError:
                self._add_result(
                    "Colab Environment",
                    "warning",
                    "Colab environment not properly detected"
                )

            # Check for GPU allocation
            if 'COLAB_GPU' in os.environ:
                self._add_result(
                    "Colab GPU",
                    "pass",
                    "GPU runtime allocated"
                )
            else:
                self._add_result(
                    "Colab GPU",
                    "warning",
                    "No GPU runtime detected",
                    "Enable GPU runtime for better performance"
                )

        elif platform == 'paperspace':
            # Paperspace-specific validations
            if os.path.exists('/storage'):
                self._add_result(
                    "Paperspace Storage",
                    "pass",
                    "Paperspace persistent storage available"
                )
            else:
                self._add_result(
                    "Paperspace Storage",
                    "warning",
                    "Paperspace storage not found"
                )

        else:  # local
            # Local environment validations
            self._add_result(
                "Local Environment",
                "pass",
                "Local development environment"
            )

    def validate_configuration_files(self):
        """Validate configuration files exist and are valid."""
        print("\nValidating Configuration Files...")

        config_dir = Path('configs')
        required_configs = [
            f'{self.platform_info["platform"]}.yaml',
            'strategies/ttt.yaml'
        ]

        if not config_dir.exists():
            self._add_result(
                "Configuration Directory",
                "fail",
                "Configuration directory not found",
                "Ensure configs/ directory exists with platform configurations"
            )
            return

        for config_file in required_configs:
            config_path = config_dir / config_file
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, encoding='utf-8') as f:
                        yaml.safe_load(f)
                    self._add_result(
                        f"Config: {config_file}",
                        "pass",
                        "Configuration file valid"
                    )
                except Exception as e:
                    self._add_result(
                        f"Config: {config_file}",
                        "fail",
                        f"Configuration file invalid: {str(e)}",
                        "Fix YAML syntax errors"
                    )
            else:
                self._add_result(
                    f"Config: {config_file}",
                    "warning",
                    "Configuration file not found",
                    f"Create {config_file} for platform-specific settings"
                )

    def run_all_validations(self) -> dict[str, Any]:
        """Run all platform validation checks."""
        print("Starting Platform Setup Validation...")
        print(f"Platform: {self.platform_info['platform']}")
        print(f"System: {self.platform_info['os']} {self.platform_info['os_version']}")
        print(f"Memory: {self.platform_info['total_memory_gb']:.1f}GB total")

        validation_methods = [
            self.validate_python_environment,
            self.validate_gpu_setup,
            self.validate_memory_requirements,
            self.validate_storage_requirements,
            self.validate_network_connectivity,
            self.validate_platform_specific_requirements,
            self.validate_configuration_files
        ]

        for validation_method in validation_methods:
            validation_method()

        # Generate summary
        summary = self._generate_summary()
        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate validation summary."""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == 'pass'])
        warnings = len([r for r in self.results if r.status == 'warning'])
        failed = len([r for r in self.results if r.status == 'fail'])

        overall_status = 'ready'
        if failed > 0:
            overall_status = 'needs_attention'
        elif warnings > 0:
            overall_status = 'ready_with_warnings'

        summary = {
            'platform': self.platform_info['platform'],
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_checks': total_checks,
                'passed': passed,
                'warnings': warnings,
                'failed': failed,
                'success_rate': passed / total_checks if total_checks > 0 else 0
            },
            'platform_info': self.platform_info,
            'validation_results': [asdict(result) for result in self.results],
            'recommendations': self._generate_recommendations()
        }

        return summary

    def _generate_recommendations(self) -> list[str]:
        """Generate setup recommendations."""
        recommendations = []

        # Critical failures
        failed_checks = [r for r in self.results if r.status == 'fail']
        if failed_checks:
            recommendations.append("CRITICAL: Address failed checks before proceeding:")
            for check in failed_checks:
                if check.recommendation:
                    recommendations.append(f"  - {check.check_name}: {check.recommendation}")

        # Warnings
        warning_checks = [r for r in self.results if r.status == 'warning']
        if warning_checks:
            recommendations.append("WARNINGS: Consider addressing these issues:")
            for check in warning_checks:
                if check.recommendation:
                    recommendations.append(f"  - {check.check_name}: {check.recommendation}")

        # Platform-specific recommendations
        platform = self.platform_info['platform']
        if platform == 'kaggle':
            recommendations.extend([
                "KAGGLE TIPS:",
                "  - Monitor GPU hours (30-hour weekly limit)",
                "  - Use /kaggle/working for outputs",
                "  - Enable internet for model downloads"
            ])
        elif platform == 'colab':
            recommendations.extend([
                "COLAB TIPS:",
                "  - Use Pro/Pro+ for better resources",
                "  - Mount Google Drive for persistence",
                "  - Enable GPU runtime for training"
            ])
        elif platform == 'local':
            recommendations.extend([
                "LOCAL TIPS:",
                "  - Use virtual environment",
                "  - Monitor system resources",
                "  - Enable GPU if available"
            ])

        return recommendations

    def save_report(self, output_file: Path = None) -> Path:
        """Save validation report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"platform_validation_report_{self.platform_info['platform']}_{timestamp}.json")

        summary = self._generate_summary()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        return output_file


def main():
    """Main function for platform setup validation."""
    try:
        validator = PlatformSetupValidator()
        summary = validator.run_all_validations()
        report_file = validator.save_report()

        # Print final summary
        print(f"\n{'='*60}")
        print("PLATFORM SETUP VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Platform: {summary['platform']}")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Checks: {summary['summary']['passed']}/{summary['summary']['total_checks']} passed")
        print(f"Warnings: {summary['summary']['warnings']}")
        print(f"Failures: {summary['summary']['failed']}")
        print(f"Success Rate: {summary['summary']['success_rate']:.1%}")

        if summary['recommendations']:
            print("\nRECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(rec)

        print(f"\nFull report saved to: {report_file}")

        # Exit code based on validation results
        if summary['overall_status'] == 'ready':
            print("\n[SUCCESS] Platform is ready for TTT implementation!")
            return 0
        elif summary['overall_status'] == 'ready_with_warnings':
            print("\n[WARNING] Platform is ready but has warnings - review recommendations")
            return 0
        else:
            print("\n[ERROR] Platform needs attention before TTT implementation")
            return 1

    except Exception as e:
        print(f"[ERROR] Error during platform validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
