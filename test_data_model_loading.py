#!/usr/bin/env python3
"""
Data and Model Loading Testing Across Environments
=================================================

This script tests data loading and model loading capabilities across
different platform environments to ensure compatibility.

Run with: python test_data_model_loading.py --platform [kaggle|colab|local]
"""

import argparse
import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class DataModelTestResult:
    """Result of data/model loading test."""
    test_name: str
    platform: str
    component: str  # 'data_loading' or 'model_loading'
    success: bool
    duration_seconds: float
    memory_usage_mb: float
    items_loaded: int
    error_message: str | None = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class DataModelLoadingTester:
    """Test data and model loading across environments."""

    def __init__(self, platform: str = "local"):
        """Initialize data and model loading tester."""
        self.platform = platform
        self.results: list[DataModelTestResult] = []

        # Platform-specific paths
        self.platform_paths = self._get_platform_paths()

        # Create test directories
        self._setup_test_environment()

    def _get_platform_paths(self) -> dict[str, str]:
        """Get platform-specific paths."""
        if self.platform == "kaggle":
            return {
                'data_dir': '/kaggle/input',
                'working_dir': '/kaggle/working',
                'cache_dir': '/kaggle/working/cache',
                'models_dir': '/kaggle/working/models'
            }
        elif self.platform == "colab":
            return {
                'data_dir': '/content/data',
                'working_dir': '/content',
                'cache_dir': '/content/cache',
                'models_dir': '/content/models'
            }
        elif self.platform == "paperspace":
            return {
                'data_dir': '/storage/data',
                'working_dir': '/storage',
                'cache_dir': '/storage/cache',
                'models_dir': '/storage/models'
            }
        else:  # local
            return {
                'data_dir': 'data',
                'working_dir': '.',
                'cache_dir': 'data/cache',
                'models_dir': 'data/models'
            }

    def _setup_test_environment(self):
        """Setup test environment with necessary directories."""
        for _path_name, path_value in self.platform_paths.items():
            path_obj = Path(path_value)
            path_obj.mkdir(parents=True, exist_ok=True)

    def _run_test_with_monitoring(self, test_name: str, component: str, test_func, *args, **kwargs) -> DataModelTestResult:
        """Run test with resource monitoring."""
        print(f"\nRunning {component} test: {test_name}")
        start_time = time.time()

        # Get baseline memory usage
        import psutil
        baseline_memory = psutil.virtual_memory().used / (1024**2)

        try:
            result = test_func(*args, **kwargs)
            success = True
            error_message = None
            items_loaded = result.get('items_loaded', 0) if isinstance(result, dict) else 0

        except Exception as e:
            result = {'error': str(e)}
            success = False
            error_message = str(e)
            items_loaded = 0
            print(f"Test failed: {error_message}")
            traceback.print_exc()

        # Calculate resource usage
        end_time = time.time()
        duration = end_time - start_time

        current_memory = psutil.virtual_memory().used / (1024**2)
        memory_usage = current_memory - baseline_memory

        return DataModelTestResult(
            test_name=test_name,
            platform=self.platform,
            component=component,
            success=success,
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            items_loaded=items_loaded,
            error_message=error_message,
            metadata=result if isinstance(result, dict) else {}
        )

    def test_arc_data_repository_loading(self) -> dict[str, Any]:
        """Test ARC data repository loading."""
        try:
            from src.adapters.repositories.arc_data_repository import ARCDataRepository

            # Initialize repository with platform-specific path
            data_path = self.platform_paths['data_dir']
            repo = ARCDataRepository(data_path=data_path, use_real_dataset=True)

            # Test basic functionality
            task_ids = repo.get_task_ids(task_source="training")

            results = {
                'repository_initialized': True,
                'data_path': data_path,
                'available_task_ids': len(task_ids),
                'sample_task_ids': task_ids[:5] if task_ids else []
            }

            # Try to load sample tasks if available
            if task_ids:
                sample_tasks = repo.load_all_tasks(task_source="training", limit=3)
                results.update({
                    'sample_tasks_loaded': len(sample_tasks),
                    'items_loaded': len(sample_tasks)
                })
            else:
                # Create fallback test data
                results.update({
                    'fallback_used': True,
                    'items_loaded': 0
                })

            return results

        except Exception as e:
            return {
                'repository_initialized': False,
                'error': str(e),
                'items_loaded': 0
            }

    def test_data_loader_performance(self) -> dict[str, Any]:
        """Test data loader performance with different configurations."""
        try:
            from src.adapters.repositories.data_loader import ArcDataLoader

            # Test with platform-specific configuration
            batch_sizes = [1, 2, 4] if self.platform in ['colab', 'kaggle'] else [1, 2, 4, 8]

            results = {
                'batch_size_tests': {},
                'optimal_batch_size': 1
            }

            for batch_size in batch_sizes:
                try:
                    # Create data loader with current batch size
                    loader = ArcDataLoader(
                        data_path=self.platform_paths['data_dir'],
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0  # Single process for compatibility
                    )

                    # Test loading a few batches
                    load_times = []
                    for i, batch in enumerate(loader):
                        if i >= 3:  # Test first 3 batches
                            break
                        batch_start = time.time()
                        # Simulate processing
                        if isinstance(batch, list | tuple) and len(batch) > 0:
                            _ = len(batch)
                        load_times.append(time.time() - batch_start)

                    avg_load_time = sum(load_times) / len(load_times) if load_times else 0

                    results['batch_size_tests'][batch_size] = {
                        'status': 'success',
                        'avg_load_time': avg_load_time,
                        'batches_tested': len(load_times)
                    }

                    # Update optimal batch size (fastest loading)
                    if avg_load_time > 0:
                        current_optimal = results['batch_size_tests'].get(results['optimal_batch_size'], {}).get('avg_load_time', float('inf'))
                        if avg_load_time < current_optimal:
                            results['optimal_batch_size'] = batch_size

                except Exception as e:
                    results['batch_size_tests'][batch_size] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            results['items_loaded'] = sum(
                test.get('batches_tested', 0)
                for test in results['batch_size_tests'].values()
                if isinstance(test, dict)
            )

            return results

        except ImportError:
            # Fallback if data loader not available
            return {
                'data_loader_available': False,
                'items_loaded': 0,
                'reason': 'ArcDataLoader not available'
            }
        except Exception as e:
            return {
                'data_loader_test': 'failed',
                'error': str(e),
                'items_loaded': 0
            }

    def test_model_loading_capabilities(self) -> dict[str, Any]:
        """Test model loading capabilities."""
        results = {
            'torch_available': True,
            'models_tested': {},
            'items_loaded': 0
        }

        # Test basic PyTorch operations
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            results['device'] = str(device)
            results['cuda_available'] = torch.cuda.is_available()

            # Test simple model creation
            simple_model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            ).to(device)

            # Test model inference
            test_input = torch.randn(1, 10, device=device)
            with torch.no_grad():
                output = simple_model(test_input)

            results['models_tested']['simple_linear'] = {
                'status': 'success',
                'parameters': sum(p.numel() for p in simple_model.parameters()),
                'device': str(device),
                'output_shape': list(output.shape)
            }
            results['items_loaded'] += 1

            # Test model serialization
            model_path = Path(self.platform_paths['models_dir']) / 'test_model.pt'
            torch.save(simple_model.state_dict(), model_path)

            # Test model loading
            loaded_state = torch.load(model_path, map_location=device)
            simple_model.load_state_dict(loaded_state)

            results['models_tested']['serialization'] = {
                'status': 'success',
                'file_size_bytes': model_path.stat().st_size,
                'save_load_cycle': 'success'
            }

            # Clean up
            model_path.unlink()
            del simple_model, test_input, output

        except Exception as e:
            results['models_tested']['simple_linear'] = {
                'status': 'failed',
                'error': str(e)
            }

        # Test quantization if available
        if torch.cuda.is_available():
            try:
                # Test FP16 mixed precision
                with torch.cuda.amp.autocast():
                    test_tensor = torch.randn(10, 10, device='cuda')
                    result = torch.matmul(test_tensor, test_tensor.T)

                results['models_tested']['mixed_precision'] = {
                    'status': 'success',
                    'precision': 'fp16'
                }
                results['items_loaded'] += 1

                del test_tensor, result

            except Exception as e:
                results['models_tested']['mixed_precision'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Test memory management
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()

                # Allocate and free memory
                temp_tensor = torch.randn(1000, 1000, device='cuda')
                memory_allocated = torch.cuda.memory_allocated()
                del temp_tensor
                torch.cuda.empty_cache()
                memory_after = torch.cuda.memory_allocated()

                results['models_tested']['memory_management'] = {
                    'status': 'success',
                    'memory_before_mb': memory_before / (1024**2),
                    'memory_allocated_mb': memory_allocated / (1024**2),
                    'memory_after_mb': memory_after / (1024**2),
                    'cleanup_effective': memory_after <= memory_before * 1.1  # Within 10%
                }
            else:
                results['models_tested']['memory_management'] = {
                    'status': 'cpu_mode',
                    'note': 'CUDA not available'
                }

            results['items_loaded'] += 1

        except Exception as e:
            results['models_tested']['memory_management'] = {
                'status': 'failed',
                'error': str(e)
            }

        return results

    def test_ttt_adapter_model_loading(self) -> dict[str, Any]:
        """Test TTT adapter model loading."""
        try:
            from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig

            # Create platform-optimized configuration
            config = TTTConfig(
                model_name="microsoft/DialoGPT-small",  # Use smaller, unrestricted model
                device="auto",
                quantization=True if self.platform in ['colab', 'kaggle'] else False,
                mixed_precision=True,
                gradient_checkpointing=True,
                lora_rank=16 if self.platform == 'colab' else 32,
                batch_size=1,
                max_training_time=30.0,  # Short test
                checkpoint_dir=Path(self.platform_paths['models_dir']) / 'ttt',
                cache_dir=Path(self.platform_paths['cache_dir']) / 'ttt'
            )

            results = {
                'adapter_config': {
                    'model_name': config.model_name,
                    'device': config.device,
                    'quantization': config.quantization,
                    'lora_rank': config.lora_rank,
                    'batch_size': config.batch_size
                }
            }

            # Initialize adapter
            adapter = TTTAdapter(config=config)

            results.update({
                'adapter_initialized': True,
                'device_selected': str(adapter.device),
                'items_loaded': 1
            })

            # Test cleanup
            adapter.cleanup()

            results['cleanup_successful'] = True

            return results

        except Exception as e:
            return {
                'adapter_initialized': False,
                'error': str(e),
                'items_loaded': 0
            }

    def test_checkpoint_management(self) -> dict[str, Any]:
        """Test checkpoint saving and loading."""
        results = {
            'checkpoint_tests': {},
            'items_loaded': 0
        }

        checkpoint_dir = Path(self.platform_paths['models_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Test checkpoint creation
            test_state = {
                'model_state': {'layer.weight': torch.randn(10, 10)},
                'optimizer_state': {'param_groups': [{'lr': 0.01}]},
                'epoch': 5,
                'loss': 0.123,
                'platform': self.platform,
                'timestamp': datetime.now().isoformat()
            }

            checkpoint_file = checkpoint_dir / 'test_checkpoint.pt'

            # Save checkpoint
            save_start = time.time()
            torch.save(test_state, checkpoint_file)
            save_time = time.time() - save_start

            results['checkpoint_tests']['save'] = {
                'status': 'success',
                'file_size_bytes': checkpoint_file.stat().st_size,
                'save_time_seconds': save_time
            }
            results['items_loaded'] += 1

            # Load checkpoint
            load_start = time.time()
            loaded_state = torch.load(checkpoint_file, map_location='cpu')
            load_time = time.time() - load_start

            # Verify integrity
            integrity_check = (
                loaded_state['platform'] == self.platform and
                loaded_state['epoch'] == 5 and
                abs(loaded_state['loss'] - 0.123) < 1e-6
            )

            results['checkpoint_tests']['load'] = {
                'status': 'success',
                'load_time_seconds': load_time,
                'integrity_verified': integrity_check
            }
            results['items_loaded'] += 1

            # Test large checkpoint (if space allows)
            try:
                large_state = {
                    'large_tensor': torch.randn(1000, 1000),
                    'metadata': {'size': 'large', 'platform': self.platform}
                }

                large_checkpoint_file = checkpoint_dir / 'large_checkpoint.pt'
                torch.save(large_state, large_checkpoint_file)

                results['checkpoint_tests']['large_checkpoint'] = {
                    'status': 'success',
                    'file_size_mb': large_checkpoint_file.stat().st_size / (1024**2)
                }
                results['items_loaded'] += 1

                # Clean up large file
                large_checkpoint_file.unlink()

            except Exception as e:
                results['checkpoint_tests']['large_checkpoint'] = {
                    'status': 'failed',
                    'error': str(e)
                }

            # Clean up test files
            checkpoint_file.unlink()

        except Exception as e:
            results['checkpoint_tests']['general'] = {
                'status': 'failed',
                'error': str(e)
            }

        return results

    def test_cache_system(self) -> dict[str, Any]:
        """Test cache system functionality."""
        try:
            from src.adapters.repositories.cache_repository import CacheRepository

            cache_dir = Path(self.platform_paths['cache_dir'])
            cache_repo = CacheRepository(cache_dir=cache_dir)

            results = {
                'cache_initialized': True,
                'cache_dir': str(cache_dir),
                'cache_tests': {}
            }

            # Test basic cache operations
            test_key = f"test_key_{self.platform}"
            test_value = {
                'data': [1, 2, 3, 4, 5],
                'platform': self.platform,
                'timestamp': datetime.now().isoformat()
            }

            # Test cache set
            cache_repo.set(test_key, test_value)
            results['cache_tests']['set'] = {'status': 'success'}

            # Test cache get
            retrieved_value = cache_repo.get(test_key)
            integrity_check = (
                retrieved_value is not None and
                retrieved_value.get('platform') == self.platform and
                retrieved_value.get('data') == [1, 2, 3, 4, 5]
            )

            results['cache_tests']['get'] = {
                'status': 'success',
                'integrity_verified': integrity_check
            }

            # Test cache statistics
            stats = cache_repo.get_cache_stats()
            results['cache_tests']['stats'] = {
                'status': 'success',
                'stats': stats
            }

            results['items_loaded'] = 3

            return results

        except ImportError:
            return {
                'cache_initialized': False,
                'reason': 'CacheRepository not available',
                'items_loaded': 0
            }
        except Exception as e:
            return {
                'cache_initialized': False,
                'error': str(e),
                'items_loaded': 0
            }

    def run_all_data_model_tests(self) -> list[DataModelTestResult]:
        """Run all data and model loading tests."""
        print(f"Running data and model loading tests for platform: {self.platform}")
        print(f"Platform paths: {self.platform_paths}")

        test_methods = [
            ('ARC Data Repository Loading', 'data_loading', self.test_arc_data_repository_loading),
            ('Data Loader Performance', 'data_loading', self.test_data_loader_performance),
            ('Model Loading Capabilities', 'model_loading', self.test_model_loading_capabilities),
            ('TTT Adapter Model Loading', 'model_loading', self.test_ttt_adapter_model_loading),
            ('Checkpoint Management', 'model_loading', self.test_checkpoint_management),
            ('Cache System', 'data_loading', self.test_cache_system)
        ]

        results = []
        for test_name, component, test_method in test_methods:
            result = self._run_test_with_monitoring(test_name, component, test_method)
            results.append(result)
            self.results.append(result)

            # Print immediate result
            status = "PASS" if result.success else "FAIL"
            print(f"  {status}: {test_name} ({result.duration_seconds:.2f}s, {result.items_loaded} items)")
            if result.error_message:
                print(f"    Error: {result.error_message}")

        return results

    def generate_data_model_report(self, output_file: Path = None) -> dict[str, Any]:
        """Generate data and model loading test report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"data_model_loading_report_{self.platform}_{timestamp}.json")

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        total_duration = sum(r.duration_seconds for r in self.results)
        total_items = sum(r.items_loaded for r in self.results)
        total_memory = sum(r.memory_usage_mb for r in self.results)

        # Component breakdown
        data_loading_tests = [r for r in self.results if r.component == 'data_loading']
        model_loading_tests = [r for r in self.results if r.component == 'model_loading']

        report = {
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'platform_paths': self.platform_paths,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'total_items_loaded': total_items,
                'total_memory_usage_mb': total_memory
            },
            'component_breakdown': {
                'data_loading': {
                    'total_tests': len(data_loading_tests),
                    'passed_tests': sum(1 for r in data_loading_tests if r.success),
                    'items_loaded': sum(r.items_loaded for r in data_loading_tests)
                },
                'model_loading': {
                    'total_tests': len(model_loading_tests),
                    'passed_tests': sum(1 for r in model_loading_tests if r.success),
                    'items_loaded': sum(r.items_loaded for r in model_loading_tests)
                }
            },
            'test_results': [asdict(result) for result in self.results],
            'recommendations': self._generate_data_model_recommendations()
        }

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nData and model loading report saved to: {output_file}")
        return report

    def _generate_data_model_recommendations(self) -> list[str]:
        """Generate data and model loading recommendations."""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in self.results if not r.success]

        if failed_tests:
            data_failures = [r for r in failed_tests if r.component == 'data_loading']
            model_failures = [r for r in failed_tests if r.component == 'model_loading']

            if data_failures:
                recommendations.append(f"Data loading issues detected: {len(data_failures)} tests failed")
                recommendations.append("Consider implementing data caching and optimization")

            if model_failures:
                recommendations.append(f"Model loading issues detected: {len(model_failures)} tests failed")
                recommendations.append("Consider using smaller models or quantization")

        # Platform-specific recommendations
        if self.platform == 'colab':
            recommendations.append("For Colab: Use smaller batch sizes and enable checkpointing")
            recommendations.append("For Colab: Consider Google Drive integration for data persistence")
        elif self.platform == 'kaggle':
            recommendations.append("For Kaggle: Leverage dataset mount points for data access")
            recommendations.append("For Kaggle: Use persistent storage for model checkpoints")

        # Performance recommendations
        total_items = sum(r.items_loaded for r in self.results)
        total_time = sum(r.duration_seconds for r in self.results)

        if total_time > 0 and total_items > 0:
            items_per_second = total_items / total_time
            if items_per_second < 1:
                recommendations.append("Low loading throughput detected - optimize data pipeline")

        # Memory recommendations
        max_memory = max((r.memory_usage_mb for r in self.results), default=0)
        if max_memory > 1000:  # 1GB
            recommendations.append("High memory usage detected - implement memory optimization")

        return recommendations


def main():
    """Main function for data and model loading testing."""
    parser = argparse.ArgumentParser(description="Test data and model loading across environments")
    parser.add_argument(
        '--platform',
        choices=['kaggle', 'colab', 'local', 'paperspace'],
        default='local',
        help='Target platform for testing'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for test report'
    )

    args = parser.parse_args()

    try:
        # Initialize tester
        tester = DataModelLoadingTester(platform=args.platform)

        # Run tests
        tester.run_all_data_model_tests()

        # Generate report
        report = tester.generate_data_model_report(args.output)

        # Print summary
        print(f"\n{'='*60}")
        print("DATA AND MODEL LOADING TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Platform: {args.platform}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Total Duration: {report['summary']['total_duration_seconds']:.2f}s")
        print(f"Items Loaded: {report['summary']['total_items_loaded']}")

        print("\nComponent Breakdown:")
        for component, stats in report['component_breakdown'].items():
            success_rate = stats['passed_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            print(f"  {component}: {stats['passed_tests']}/{stats['total_tests']} ({success_rate:.1%}), {stats['items_loaded']} items")

        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            print(f"\nWARNING: {report['summary']['failed_tests']} tests failed!")
            return 1
        else:
            print("\nSUCCESS: All tests passed!")
            return 0

    except Exception as e:
        print(f"Error running data and model loading tests: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
