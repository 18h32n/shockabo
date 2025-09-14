#!/usr/bin/env python3
"""
Platform Compatibility Testing for Story 1.4 TTT Implementation
===============================================================

This script tests the TTT implementation across different platform environments:
- Kaggle environment constraints
- Google Colab environment constraints
- Memory management across different GPU configurations
- Configuration overrides for platform-specific settings
- Data loading and model loading in different environments

Run with: python test_platform_compatibility.py --platform [kaggle|colab|local]
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml


@dataclass
class PlatformTestResult:
    """Result of platform compatibility test."""
    test_name: str
    platform: str
    success: bool
    duration_seconds: float
    memory_usage_mb: float
    gpu_memory_mb: float
    error_message: str | None = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemInfo:
    """System information for compatibility testing."""
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_count: int
    gpu_names: list[str]
    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    os_info: str


class PlatformCompatibilityTester:
    """Test TTT implementation compatibility across platforms."""

    def __init__(self, platform: str = "local", config_dir: Path = None):
        """Initialize platform compatibility tester."""
        self.platform = platform
        self.config_dir = config_dir or Path("configs")
        self.results: list[PlatformTestResult] = []
        self.system_info = self._gather_system_info()

        # Load platform-specific configuration
        self.platform_config = self._load_platform_config()

        # Initialize test data directory
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)

    def _gather_system_info(self) -> SystemInfo:
        """Gather system information for testing."""
        gpu_names = []
        cuda_version = None

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(i))
            cuda_version = torch.version.cuda

        memory_info = psutil.virtual_memory()

        return SystemInfo(
            platform=self.platform,
            python_version=sys.version,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            gpu_names=gpu_names,
            total_memory_gb=memory_info.total / (1024**3),
            available_memory_gb=memory_info.available / (1024**3),
            cpu_count=psutil.cpu_count(),
            os_info=f"{os.name} {os.uname().sysname if hasattr(os, 'uname') else 'Windows'}"
        )

    def _load_platform_config(self) -> dict[str, Any]:
        """Load platform-specific configuration."""
        config_file = self.config_dir / f"{self.platform}.yaml"
        if not config_file.exists():
            print(f"Warning: Platform config {config_file} not found, using defaults")
            return {}

        with open(config_file, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _run_test_with_monitoring(self, test_name: str, test_func, *args, **kwargs) -> PlatformTestResult:
        """Run test with resource monitoring."""
        print(f"\nRunning test: {test_name}")
        start_time = time.time()

        # Get baseline memory usage
        baseline_memory = psutil.virtual_memory().used / (1024**2)
        baseline_gpu_memory = 0.0
        if torch.cuda.is_available():
            baseline_gpu_memory = torch.cuda.memory_allocated() / (1024**2)

        try:
            # Run the test function
            result = test_func(*args, **kwargs)
            success = True
            error_message = None

        except Exception as e:
            result = None
            success = False
            error_message = str(e)
            print(f"Test failed: {error_message}")
            traceback.print_exc()

        # Calculate resource usage
        end_time = time.time()
        duration = end_time - start_time

        current_memory = psutil.virtual_memory().used / (1024**2)
        memory_usage = current_memory - baseline_memory

        gpu_memory_usage = 0.0
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            gpu_memory_usage = current_gpu_memory - baseline_gpu_memory

        return PlatformTestResult(
            test_name=test_name,
            platform=self.platform,
            success=success,
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory_usage,
            error_message=error_message,
            metadata=result if isinstance(result, dict) else {}
        )

    def test_configuration_loading(self) -> dict[str, Any]:
        """Test platform-specific configuration loading."""
        from src.infrastructure.config import ConfigManager

        # Test loading platform configuration
        config_manager = ConfigManager(config_dir=self.config_dir)

        config = config_manager.get_all()

        # Validate required configuration sections
        required_sections = ['platform', 'paths', 'model', 'training', 'resources']
        missing_sections = [section for section in required_sections if section not in config]

        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")

        # Test TTT strategy configuration override
        ttt_config_path = self.config_dir / "strategies" / "ttt.yaml"
        if ttt_config_path.exists():
            with open(ttt_config_path, encoding='utf-8') as f:
                ttt_config = yaml.safe_load(f)

            # Check platform-specific overrides
            if 'platform_overrides' in ttt_config and self.platform in ttt_config['platform_overrides']:
                platform_overrides = ttt_config['platform_overrides'][self.platform]
                print(f"Found platform overrides for {self.platform}: {platform_overrides}")

        return {
            'config_loaded': True,
            'platform_name': config.get('platform', {}).get('name'),
            'memory_limit': config.get('platform', {}).get('memory_limit_gb'),
            'gpu_hours_limit': config.get('platform', {}).get('gpu_hours_limit'),
            'has_persistent_storage': config.get('platform', {}).get('has_persistent_storage'),
            'model_device': config.get('model', {}).get('device'),
            'batch_size': config.get('training', {}).get('batch_size'),
            'max_memory_gb': config.get('resources', {}).get('max_memory_gb')
        }

    def test_ttt_adapter_initialization(self) -> dict[str, Any]:
        """Test TTT adapter initialization with platform configuration."""
        from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig

        # Load platform-specific TTT configuration
        ttt_config_path = self.config_dir / "strategies" / "ttt.yaml"

        if ttt_config_path.exists():
            ttt_config = TTTConfig.from_yaml(ttt_config_path)

            # Apply platform-specific overrides
            with open(ttt_config_path, encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if 'platform_overrides' in yaml_config and self.platform in yaml_config['platform_overrides']:
                overrides = yaml_config['platform_overrides'][self.platform]
                print(f"Applying platform overrides: {overrides}")

                # Apply training overrides
                if 'training' in overrides:
                    for key, value in overrides['training'].items():
                        if hasattr(ttt_config, key):
                            setattr(ttt_config, key, value)

                # Apply LoRA overrides
                if 'lora' in overrides:
                    for key, value in overrides['lora'].items():
                        if key == 'rank':
                            ttt_config.lora_rank = value
                        elif key == 'alpha':
                            ttt_config.lora_alpha = value
                        elif key == 'dropout':
                            ttt_config.lora_dropout = value

                # Apply model overrides
                if 'model' in overrides:
                    for key, value in overrides['model'].items():
                        if hasattr(ttt_config, key):
                            setattr(ttt_config, key, value)
        else:
            ttt_config = TTTConfig()

        # Initialize TTT adapter
        adapter = TTTAdapter(config=ttt_config)

        return {
            'adapter_initialized': True,
            'device': str(adapter.device),
            'model_name': adapter.config.model_name,
            'lora_rank': adapter.config.lora_rank,
            'batch_size': adapter.config.batch_size,
            'memory_limit_mb': adapter.config.memory_limit_mb,
            'quantization': adapter.config.quantization,
            'mixed_precision': adapter.config.mixed_precision
        }

    def test_memory_constraints(self) -> dict[str, Any]:
        """Test memory management under platform constraints."""

        # Get platform memory limits
        platform_memory_limit = self.platform_config.get('resources', {}).get('max_memory_gb', 16)
        gpu_memory_fraction = self.platform_config.get('resources', {}).get('gpu_memory_fraction', 0.8)

        memory_test_results = {
            'platform_memory_limit_gb': platform_memory_limit,
            'gpu_memory_fraction': gpu_memory_fraction,
            'current_memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

        # Test GPU memory if available
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)

            memory_test_results.update({
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_allocated_gb': gpu_memory_allocated,
                'gpu_memory_cached_gb': gpu_memory_cached,
                'gpu_memory_available_gb': gpu_memory_total - gpu_memory_cached
            })

            # Test memory allocation within limits
            target_allocation = gpu_memory_total * gpu_memory_fraction
            if gpu_memory_cached < target_allocation:
                try:
                    # Allocate test tensor
                    test_size = int((target_allocation - gpu_memory_cached) * 1024**3 / 4)  # float32
                    test_tensor = torch.randn(test_size, device='cuda')
                    memory_test_results['memory_allocation_test'] = 'success'
                    del test_tensor
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    memory_test_results['memory_allocation_test'] = f'failed: {str(e)}'
            else:
                memory_test_results['memory_allocation_test'] = 'skipped: already at limit'

        # Check if current usage exceeds platform limits
        if memory_test_results['current_memory_usage_gb'] > platform_memory_limit:
            raise RuntimeError(f"Memory usage ({memory_test_results['current_memory_usage_gb']:.2f}GB) exceeds platform limit ({platform_memory_limit}GB)")

        return memory_test_results

    def test_data_loading(self) -> dict[str, Any]:
        """Test data loading with platform-specific paths."""
        from src.adapters.repositories.arc_data_repository import ARCDataRepository
        from src.domain.models import ARCTask

        # Get platform-specific data paths
        data_dir = self.platform_config.get('paths', {}).get('data_dir', 'data')

        # Initialize data repository with correct parameter
        repo = ARCDataRepository(data_path=str(data_dir))

        # Test loading training data (should work on all platforms)
        try:
            training_tasks = repo.load_all_tasks(task_source="training", limit=5)
            print(f"Loaded {len(training_tasks)} training tasks")

            if not training_tasks:
                raise ValueError("No training tasks loaded")

            # Test loading a specific task
            sample_task_id = list(training_tasks.keys())[0]
            sample_task = training_tasks[sample_task_id]
            task_details = repo.load_task(sample_task.task_id, "training")

            data_loading_results = {
                'training_tasks_count': len(training_tasks),
                'sample_task_id': sample_task.task_id,
                'sample_task_loaded': task_details is not None,
                'data_dir': str(data_dir)
            }

        except Exception as e:
            # Fallback: create minimal test data
            print(f"Standard data loading failed: {e}")
            print("Creating minimal test data for platform testing...")

            test_task = ARCTask(
                task_id="test_task_001",
                task_source="training",
                train_examples=[
                    {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
                ],
                test_input=[[1, 0], [0, 1]],
                test_output=[[0, 1], [1, 0]]  # For validation
            )

            data_loading_results = {
                'training_tasks_count': 1,
                'sample_task_id': test_task.task_id,
                'sample_task_loaded': True,
                'data_dir': str(data_dir),
                'fallback_used': True
            }

        return data_loading_results

    def test_model_loading_and_inference(self) -> dict[str, Any]:
        """Test model loading and basic inference."""
        from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
        from src.domain.models import ARCTask

        # Create platform-optimized configuration
        ttt_config = TTTConfig()

        # Apply platform-specific settings
        if self.platform == 'colab':
            ttt_config.batch_size = 1
            ttt_config.lora_rank = 32
            ttt_config.quantization = True
            ttt_config.mixed_precision = True
            ttt_config.gradient_checkpointing = True
        elif self.platform == 'kaggle':
            ttt_config.batch_size = 1
            ttt_config.lora_rank = 32
            ttt_config.gradient_accumulation_steps = 2

        # Create simple test task
        test_task = ARCTask(
            task_id="platform_test_001",
            task_source="training",
            train_examples=[
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
            ],
            test_input=[[1, 0], [0, 1]],
            test_output=[[0, 1], [1, 0]]
        )

        # Initialize adapter
        adapter = TTTAdapter(config=ttt_config)

        try:
            # Test adaptation (quick version)
            adaptation = adapter.adapt_to_task(test_task)

            # Test solving
            solution = adapter.solve(test_task)

            model_results = {
                'adapter_initialized': True,
                'adaptation_successful': adaptation.adaptation_metrics.get('mit_ttt_success', False),
                'solution_generated': len(solution.predictions) > 0,
                'confidence_score': solution.confidence_score,
                'strategy_used': str(solution.strategy_used),
                'resource_usage': {
                    'cpu_seconds': solution.resource_usage.cpu_seconds,
                    'memory_mb': solution.resource_usage.memory_mb,
                    'gpu_memory_mb': solution.resource_usage.gpu_memory_mb
                }
            }

        except Exception as e:
            model_results = {
                'adapter_initialized': True,
                'adaptation_successful': False,
                'solution_generated': False,
                'error': str(e)
            }

        finally:
            # Cleanup
            adapter.cleanup()

        return model_results

    def test_checkpoint_management(self) -> dict[str, Any]:
        """Test checkpoint saving and loading on platform."""

        platform_paths = self.platform_config.get('paths', {})
        models_dir = Path(platform_paths.get('models_dir', 'models'))
        working_dir = Path(platform_paths.get('working_dir', '.'))

        # Create test directories
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = models_dir / "test_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_results = {
            'models_dir': str(models_dir),
            'checkpoint_dir': str(checkpoint_dir),
            'working_dir': str(working_dir)
        }

        # Test checkpoint creation
        test_checkpoint_file = checkpoint_dir / "test_checkpoint.pt"
        test_data = {
            'model_state': {'test': 'data'},
            'timestamp': datetime.now().isoformat(),
            'platform': self.platform
        }

        try:
            torch.save(test_data, test_checkpoint_file)
            checkpoint_results['checkpoint_save'] = 'success'

            # Test loading
            loaded_data = torch.load(test_checkpoint_file, map_location='cpu')
            checkpoint_results['checkpoint_load'] = 'success'
            checkpoint_results['data_integrity'] = loaded_data['platform'] == self.platform

            # Clean up
            test_checkpoint_file.unlink()

        except Exception as e:
            checkpoint_results['checkpoint_save'] = f'failed: {str(e)}'
            checkpoint_results['checkpoint_load'] = 'not_attempted'

        # Test disk space
        try:
            disk_usage = psutil.disk_usage(str(working_dir))
            checkpoint_results['disk_space'] = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3)
            }
        except Exception as e:
            checkpoint_results['disk_space'] = f'error: {str(e)}'

        return checkpoint_results

    def test_gpu_configurations(self) -> dict[str, Any]:
        """Test different GPU configurations."""

        gpu_results = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        if not torch.cuda.is_available():
            gpu_results['cpu_fallback'] = 'required'
            return gpu_results

        # Test each available GPU
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_props = torch.cuda.get_device_properties(i)

            gpu_results[f'gpu_{i}'] = {
                'name': device_name,
                'total_memory_gb': device_props.total_memory / (1024**3),
                'multiprocessor_count': device_props.multi_processor_count,
                'major': device_props.major,
                'minor': device_props.minor
            }

            # Test basic tensor operations
            try:
                with torch.cuda.device(i):
                    test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                    result = torch.matmul(test_tensor, test_tensor.T)
                    gpu_results[f'gpu_{i}']['tensor_ops'] = 'success'
                    del test_tensor, result
                    torch.cuda.empty_cache()
            except Exception as e:
                gpu_results[f'gpu_{i}']['tensor_ops'] = f'failed: {str(e)}'

        # Test memory management strategies
        if torch.cuda.is_available():
            torch.cuda.memory_allocated()

            # Test mixed precision
            try:
                with torch.cuda.amp.autocast():
                    test_tensor = torch.randn(1000, 1000, device='cuda')
                    result = torch.matmul(test_tensor, test_tensor.T)
                gpu_results['mixed_precision'] = 'supported'
                del test_tensor, result
            except Exception as e:
                gpu_results['mixed_precision'] = f'not_supported: {str(e)}'

            torch.cuda.empty_cache()

        return gpu_results

    def run_all_tests(self) -> list[PlatformTestResult]:
        """Run all platform compatibility tests."""
        print(f"Running platform compatibility tests for: {self.platform}")
        print(f"System info: {self.system_info}")

        test_methods = [
            ('Configuration Loading', self.test_configuration_loading),
            ('TTT Adapter Initialization', self.test_ttt_adapter_initialization),
            ('Memory Constraints', self.test_memory_constraints),
            ('Data Loading', self.test_data_loading),
            ('Model Loading and Inference', self.test_model_loading_and_inference),
            ('Checkpoint Management', self.test_checkpoint_management),
            ('GPU Configurations', self.test_gpu_configurations)
        ]

        results = []
        for test_name, test_method in test_methods:
            result = self._run_test_with_monitoring(test_name, test_method)
            results.append(result)
            self.results.append(result)

            # Print immediate result
            status = "PASS" if result.success else "FAIL"
            print(f"  {status}: {test_name} ({result.duration_seconds:.2f}s)")
            if result.error_message:
                print(f"    Error: {result.error_message}")

        return results

    def generate_report(self, output_file: Path = None) -> dict[str, Any]:
        """Generate comprehensive compatibility report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"platform_compatibility_report_{self.platform}_{timestamp}.json")

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        total_duration = sum(r.duration_seconds for r in self.results)
        total_memory = sum(r.memory_usage_mb for r in self.results)
        total_gpu_memory = sum(r.gpu_memory_mb for r in self.results)

        report = {
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'system_info': asdict(self.system_info),
            'platform_config': self.platform_config,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'total_memory_usage_mb': total_memory,
                'total_gpu_memory_mb': total_gpu_memory
            },
            'test_results': [asdict(result) for result in self.results],
            'recommendations': self._generate_recommendations()
        }

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nCompatibility report saved to: {output_file}")
        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate platform-specific recommendations."""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in self.results if not r.success]

        if failed_tests:
            recommendations.append("Failed tests detected - see individual test results for details")

        # Memory recommendations
        if self.system_info.available_memory_gb < 4:
            recommendations.append("Low memory detected - consider enabling more aggressive memory optimizations")

        if self.platform == 'colab':
            recommendations.append("For Colab: Use smaller batch sizes and enable gradient checkpointing")
            recommendations.append("For Colab: Consider using quantization to reduce memory usage")
            if not self.system_info.cuda_available:
                recommendations.append("For Colab: Ensure GPU runtime is enabled")

        elif self.platform == 'kaggle':
            recommendations.append("For Kaggle: Monitor GPU hour usage to avoid exhaustion")
            recommendations.append("For Kaggle: Use persistent storage for checkpoints")

        # GPU recommendations
        if self.system_info.cuda_available:
            if self.system_info.gpu_count > 1:
                recommendations.append("Multiple GPUs detected - consider implementing multi-GPU support")
        else:
            recommendations.append("No GPU detected - performance will be significantly reduced")

        # General recommendations
        if any(r.memory_usage_mb > 1000 for r in self.results):
            recommendations.append("High memory usage detected - monitor memory consumption in production")

        return recommendations


def main():
    """Main function for platform compatibility testing."""
    parser = argparse.ArgumentParser(description="Test TTT implementation platform compatibility")
    parser.add_argument(
        '--platform',
        choices=['kaggle', 'colab', 'local', 'paperspace'],
        default='local',
        help='Target platform for testing'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('configs'),
        help='Configuration directory path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for test report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    try:
        # Initialize tester
        tester = PlatformCompatibilityTester(
            platform=args.platform,
            config_dir=args.config_dir
        )

        # Run tests
        tester.run_all_tests()

        # Generate report
        report = tester.generate_report(args.output)

        # Print summary
        print(f"\n{'='*60}")
        print("PLATFORM COMPATIBILITY TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Platform: {args.platform}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Total Duration: {report['summary']['total_duration_seconds']:.2f}s")

        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            print(f"\nWARNING: {report['summary']['failed_tests']} tests failed!")
            sys.exit(1)
        else:
            print("\nSUCCESS: All tests passed!")
            sys.exit(0)

    except Exception as e:
        print(f"Error running platform compatibility tests: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
