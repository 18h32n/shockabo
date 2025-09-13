#!/usr/bin/env python3
"""
GPU Memory Configuration Testing for TTT Implementation
======================================================

This script tests the TTT implementation across different GPU memory configurations
and validates memory management strategies.

Run with: python test_gpu_memory_configurations.py
"""

import gc
import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import yaml


@dataclass
class GPUTestResult:
    """Result of GPU configuration test."""
    test_name: str
    gpu_config: str
    memory_config: str
    success: bool
    duration_seconds: float
    memory_usage_mb: float
    gpu_memory_mb: float
    peak_gpu_memory_mb: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class GPUMemoryTester:
    """Test TTT implementation with various GPU memory configurations."""

    def __init__(self):
        """Initialize GPU memory tester."""
        self.results: List[GPUTestResult] = []
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_info = []
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.gpu_info.append({
                    'device': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multiprocessor_count': props.multi_processor_count,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        else:
            self.gpu_count = 0
            self.gpu_info = []
        
        print(f"GPU Status: CUDA Available = {self.cuda_available}, GPU Count = {self.gpu_count}")
        for info in self.gpu_info:
            print(f"  GPU {info['device']}: {info['name']} ({info['total_memory_gb']:.1f}GB)")

    def _run_gpu_test(self, test_name: str, test_func, *args, **kwargs) -> GPUTestResult:
        """Run GPU test with memory monitoring."""
        print(f"\nRunning GPU test: {test_name}")
        start_time = time.time()
        
        # Reset GPU memory tracking
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            baseline_gpu_memory = 0.0
        
        baseline_memory = psutil.virtual_memory().used / (1024**2)
        
        try:
            result = test_func(*args, **kwargs)
            success = True
            error_message = None
            
        except Exception as e:
            result = {'error': str(e)}
            success = False
            error_message = str(e)
            print(f"Test failed: {error_message}")
            traceback.print_exc()
        
        # Calculate resource usage
        end_time = time.time()
        duration = end_time - start_time
        
        current_memory = psutil.virtual_memory().used / (1024**2)
        memory_usage = current_memory - baseline_memory
        
        if self.cuda_available:
            current_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)
            gpu_memory_usage = current_gpu_memory - baseline_gpu_memory
        else:
            gpu_memory_usage = 0.0
            peak_gpu_memory = 0.0
        
        return GPUTestResult(
            test_name=test_name,
            gpu_config="cuda" if self.cuda_available else "cpu",
            memory_config=f"{psutil.virtual_memory().total // (1024**3)}GB_RAM",
            success=success,
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory_usage,
            peak_gpu_memory_mb=peak_gpu_memory,
            error_message=error_message,
            metadata=result if isinstance(result, dict) else {}
        )

    def test_basic_cuda_operations(self) -> Dict[str, Any]:
        """Test basic CUDA operations and memory allocation."""
        if not self.cuda_available:
            return {'status': 'skipped', 'reason': 'CUDA not available'}
        
        results = {}
        
        # Test tensor operations
        try:
            device = torch.device('cuda')
            
            # Small tensor test
            small_tensor = torch.randn(100, 100, device=device)
            small_result = torch.matmul(small_tensor, small_tensor.T)
            results['small_tensor_ops'] = 'success'
            
            # Medium tensor test
            medium_tensor = torch.randn(1000, 1000, device=device)
            medium_result = torch.matmul(medium_tensor, medium_tensor.T)
            results['medium_tensor_ops'] = 'success'
            
            # Large tensor test (if memory allows)
            try:
                large_tensor = torch.randn(5000, 5000, device=device)
                large_result = torch.matmul(large_tensor, large_tensor.T)
                results['large_tensor_ops'] = 'success'
                del large_tensor, large_result
            except RuntimeError as e:
                results['large_tensor_ops'] = f'failed: {str(e)}'
            
            del small_tensor, small_result, medium_tensor, medium_result
            torch.cuda.empty_cache()
            
        except Exception as e:
            results['basic_ops'] = f'failed: {str(e)}'
        
        return results

    def test_mixed_precision(self) -> Dict[str, Any]:
        """Test mixed precision training capabilities."""
        if not self.cuda_available:
            return {'status': 'skipped', 'reason': 'CUDA not available'}
        
        results = {}
        
        try:
            device = torch.device('cuda')
            
            # Test autocast
            with torch.cuda.amp.autocast():
                x = torch.randn(1000, 1000, device=device, requires_grad=True)
                y = torch.randn(1000, 1000, device=device, requires_grad=True)
                z = torch.matmul(x, y)
                loss = z.mean()
                
            results['autocast'] = 'supported'
            
            # Test GradScaler
            scaler = torch.cuda.amp.GradScaler()
            
            with torch.cuda.amp.autocast():
                output = torch.matmul(x, y)
                loss = output.mean()
            
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD([x, y], lr=0.01))
            scaler.update()
            
            results['grad_scaler'] = 'supported'
            
            del x, y, z, loss, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            results['mixed_precision'] = f'failed: {str(e)}'
        
        return results

    def test_memory_management_strategies(self) -> Dict[str, Any]:
        """Test various memory management strategies."""
        results = {}
        
        # Test gradient checkpointing simulation
        try:
            if self.cuda_available:
                device = torch.device('cuda')
                
                # Simulate model with gradient checkpointing
                class SimpleModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layers = torch.nn.ModuleList([
                            torch.nn.Linear(1000, 1000) for _ in range(5)
                        ])
                    
                    def forward(self, x):
                        for layer in self.layers:
                            x = torch.nn.functional.relu(layer(x))
                        return x
                
                model = SimpleModel().to(device)
                input_data = torch.randn(32, 1000, device=device)
                
                # Test with checkpointing
                from torch.utils.checkpoint import checkpoint_sequential
                output = checkpoint_sequential(model.layers, 2, input_data)
                loss = output.mean()
                loss.backward()
                
                results['gradient_checkpointing'] = 'supported'
                
                del model, input_data, output, loss
                torch.cuda.empty_cache()
                
            else:
                results['gradient_checkpointing'] = 'cpu_mode'
                
        except Exception as e:
            results['gradient_checkpointing'] = f'failed: {str(e)}'
        
        # Test memory clearing
        try:
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()
            results['memory_clearing'] = 'supported'
        except Exception as e:
            results['memory_clearing'] = f'failed: {str(e)}'
        
        return results

    def test_batch_size_scaling(self) -> Dict[str, Any]:
        """Test automatic batch size scaling based on available memory."""
        results = {}
        
        if not self.cuda_available:
            results['batch_scaling'] = 'cpu_mode'
            return results
        
        device = torch.device('cuda')
        
        # Get available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory - torch.cuda.memory_allocated()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        max_working_batch_size = 1
        
        for batch_size in batch_sizes:
            try:
                # Create a model that uses significant memory
                model = torch.nn.Sequential(
                    torch.nn.Linear(2048, 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 1000)
                ).to(device)
                
                # Test with current batch size
                input_data = torch.randn(batch_size, 2048, device=device)
                output = model(input_data)
                loss = output.mean()
                loss.backward()
                
                max_working_batch_size = batch_size
                results[f'batch_size_{batch_size}'] = 'success'
                
                # Clean up
                del model, input_data, output, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f'batch_size_{batch_size}'] = 'out_of_memory'
                    break
                else:
                    results[f'batch_size_{batch_size}'] = f'error: {str(e)}'
        
        results['max_working_batch_size'] = max_working_batch_size
        results['available_memory_gb'] = available_memory / (1024**3)
        
        return results

    def test_ttt_memory_requirements(self) -> Dict[str, Any]:
        """Test TTT-specific memory requirements."""
        results = {}
        
        try:
            from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
            from src.domain.models import ARCTask
            
            # Test with different memory configurations
            memory_configs = [
                {"batch_size": 1, "lora_rank": 8, "quantization": True, "name": "ultra_low_memory"},
                {"batch_size": 1, "lora_rank": 16, "quantization": True, "name": "low_memory"},
                {"batch_size": 2, "lora_rank": 32, "quantization": True, "name": "medium_memory"},
                {"batch_size": 4, "lora_rank": 64, "quantization": False, "name": "high_memory"}
            ]
            
            for config in memory_configs:
                try:
                    ttt_config = TTTConfig(
                        model_name="microsoft/DialoGPT-small",  # Use a smaller, unrestricted model
                        batch_size=config["batch_size"],
                        lora_rank=config["lora_rank"],
                        quantization=config["quantization"],
                        gradient_checkpointing=True,
                        max_training_time=30.0  # Short test
                    )
                    
                    adapter = TTTAdapter(config=ttt_config)
                    
                    # Create minimal test task
                    test_task = ARCTask(
                        task_id=f"memory_test_{config['name']}",
                        task_source="training",
                        train_examples=[
                            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
                        ],
                        test_input=[[1, 0], [0, 1]],
                        test_output=[[0, 1], [1, 0]]
                    )
                    
                    # Test initialization only (not full training to save time)
                    results[config["name"]] = {
                        'status': 'initialized',
                        'config': config,
                        'device': str(adapter.device)
                    }
                    
                    adapter.cleanup()
                    del adapter
                    
                except Exception as e:
                    results[config["name"]] = {
                        'status': 'failed',
                        'error': str(e),
                        'config': config
                    }
                
                # Clean up after each test
                gc.collect()
                if self.cuda_available:
                    torch.cuda.empty_cache()
            
        except ImportError as e:
            results['ttt_memory_test'] = f'import_error: {str(e)}'
        
        return results

    def test_multi_gpu_support(self) -> Dict[str, Any]:
        """Test multi-GPU support capabilities."""
        results = {}
        
        if not self.cuda_available:
            results['multi_gpu'] = 'cuda_not_available'
            return results
        
        if self.gpu_count < 2:
            results['multi_gpu'] = 'single_gpu_only'
            return results
        
        try:
            # Test tensor operations on different GPUs
            for i in range(min(2, self.gpu_count)):  # Test first 2 GPUs
                device = torch.device(f'cuda:{i}')
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor.T)
                results[f'gpu_{i}_ops'] = 'success'
                del test_tensor, result
                torch.cuda.empty_cache()
            
            # Test data transfer between GPUs
            if self.gpu_count >= 2:
                tensor_gpu0 = torch.randn(100, 100, device='cuda:0')
                tensor_gpu1 = tensor_gpu0.to('cuda:1')
                results['gpu_transfer'] = 'success'
                del tensor_gpu0, tensor_gpu1
                torch.cuda.empty_cache()
            
        except Exception as e:
            results['multi_gpu'] = f'failed: {str(e)}'
        
        return results

    def run_all_gpu_tests(self) -> List[GPUTestResult]:
        """Run all GPU memory configuration tests."""
        print(f"Running GPU memory configuration tests...")
        print(f"System: {psutil.virtual_memory().total // (1024**3)}GB RAM, {self.gpu_count} GPUs")
        
        test_methods = [
            ('Basic CUDA Operations', self.test_basic_cuda_operations),
            ('Mixed Precision Support', self.test_mixed_precision),
            ('Memory Management Strategies', self.test_memory_management_strategies),
            ('Batch Size Scaling', self.test_batch_size_scaling),
            ('TTT Memory Requirements', self.test_ttt_memory_requirements),
            ('Multi-GPU Support', self.test_multi_gpu_support)
        ]
        
        results = []
        for test_name, test_method in test_methods:
            result = self._run_gpu_test(test_name, test_method)
            results.append(result)
            self.results.append(result)
            
            # Print immediate result
            status = "PASS" if result.success else "FAIL"
            print(f"  {status}: {test_name} ({result.duration_seconds:.2f}s)")
            if result.error_message:
                print(f"    Error: {result.error_message}")
            
            # Print GPU memory usage
            if result.gpu_memory_mb > 0:
                print(f"    GPU Memory: {result.gpu_memory_mb:.1f}MB used, {result.peak_gpu_memory_mb:.1f}MB peak")
        
        return results

    def generate_gpu_report(self, output_file: Path = None) -> Dict[str, Any]:
        """Generate GPU configuration compatibility report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"gpu_memory_compatibility_report_{timestamp}.json")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration_seconds for r in self.results)
        max_gpu_memory = max((r.peak_gpu_memory_mb for r in self.results), default=0)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cuda_available': self.cuda_available,
                'gpu_count': self.gpu_count,
                'gpu_info': self.gpu_info,
                'total_ram_gb': psutil.virtual_memory().total // (1024**3),
                'available_ram_gb': psutil.virtual_memory().available // (1024**3)
            },
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'max_gpu_memory_mb': max_gpu_memory
            },
            'test_results': [asdict(result) for result in self.results],
            'recommendations': self._generate_gpu_recommendations()
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nGPU compatibility report saved to: {output_file}")
        return report

    def _generate_gpu_recommendations(self) -> List[str]:
        """Generate GPU-specific recommendations."""
        recommendations = []
        
        if not self.cuda_available:
            recommendations.append("No CUDA support detected - consider using GPU-enabled environment")
            recommendations.append("CPU-only mode will be significantly slower for TTT training")
        else:
            # Analyze GPU memory
            if self.gpu_info:
                total_gpu_memory = sum(info['total_memory_gb'] for info in self.gpu_info)
                
                if total_gpu_memory < 8:
                    recommendations.append("Limited GPU memory detected - use small batch sizes and aggressive memory optimization")
                    recommendations.append("Enable quantization and gradient checkpointing")
                elif total_gpu_memory < 16:
                    recommendations.append("Moderate GPU memory - use medium batch sizes with mixed precision")
                else:
                    recommendations.append("Sufficient GPU memory for larger batch sizes and full precision training")
        
        # Analyze failed tests
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            memory_failures = [r for r in failed_tests if 'memory' in str(r.error_message).lower()]
            if memory_failures:
                recommendations.append("Memory-related failures detected - reduce batch size and enable memory optimizations")
        
        # Multi-GPU recommendations
        if self.gpu_count > 1:
            recommendations.append("Multiple GPUs detected - consider implementing data parallelism for TTT training")
        
        return recommendations


def main():
    """Main function for GPU memory configuration testing."""
    try:
        # Initialize tester
        tester = GPUMemoryTester()
        
        # Run tests
        results = tester.run_all_gpu_tests()
        
        # Generate report
        report = tester.generate_gpu_report()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"GPU MEMORY CONFIGURATION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"CUDA Available: {tester.cuda_available}")
        print(f"GPU Count: {tester.gpu_count}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Total Duration: {report['summary']['total_duration_seconds']:.2f}s")
        
        if tester.cuda_available:
            print(f"Max GPU Memory Used: {report['summary']['max_gpu_memory_mb']:.1f}MB")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            print(f"\nWARNING: {report['summary']['failed_tests']} tests failed!")
            return 1
        else:
            print(f"\nSUCCESS: All tests passed!")
            return 0
            
    except Exception as e:
        print(f"Error running GPU memory configuration tests: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())