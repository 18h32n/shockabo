"""
Pipeline Test Utilities for Task 6

Supporting utilities for the 100-task pipeline test including:
- Test configuration validation
- Result analysis and reporting
- Performance benchmarking
- Error pattern analysis
"""

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestValidationResult:
    """Result of test configuration validation."""
    is_valid: bool
    warnings: list[str]
    recommendations: list[str]
    estimated_duration_minutes: float
    estimated_memory_gb: float


class PipelineTestValidator:
    """Validates test configurations and system readiness."""

    def __init__(self):
        """Initialize test validator."""
        self.system_specs = self._detect_system_specs()

    def _detect_system_specs(self) -> dict[str, Any]:
        """Detect system specifications."""
        import psutil
        import torch

        specs = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_gb": 0,
            "gpu_count": 0
        }

        if torch.cuda.is_available():
            specs["gpu_count"] = torch.cuda.device_count()
            specs["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return specs

    def validate_test_config(self, config: Any) -> TestValidationResult:
        """Validate test configuration against system capabilities."""
        warnings = []
        recommendations = []

        # Memory validation
        estimated_memory = self._estimate_memory_usage(config)
        available_memory = self.system_specs["memory_gb"]

        if estimated_memory > available_memory * 0.8:
            warnings.append(f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds 80% of available memory")
            recommendations.append("Consider reducing batch size or using quantization")

        # GPU memory validation
        if config.use_qlora and self.system_specs["cuda_available"]:
            gpu_memory_required = self._estimate_gpu_memory(config)
            available_gpu_memory = self.system_specs["gpu_memory_gb"]

            if gpu_memory_required > available_gpu_memory * 0.9:
                warnings.append(f"GPU memory requirement ({gpu_memory_required:.1f}GB) exceeds available GPU memory")
                recommendations.append("Enable gradient checkpointing or reduce model size")

        # Time estimation
        estimated_duration = self._estimate_test_duration(config)

        if estimated_duration > 480:  # 8 hours
            warnings.append(f"Estimated test duration ({estimated_duration/60:.1f}h) is very long")
            recommendations.append("Consider reducing number of tasks or increasing parallelism")

        # Concurrency validation
        if config.max_concurrent_tasks > 1 and config.use_qlora:
            warnings.append("Concurrent execution with 8B model may cause memory issues")
            recommendations.append("Use sequential execution (max_concurrent_tasks=1) for 8B model")

        # Model availability check
        if "llama" in config.model_name.lower() and not self.system_specs["cuda_available"]:
            warnings.append("Llama models require GPU - falling back to CPU-compatible model")
            recommendations.append("Use smaller model like GPT-2 for CPU testing")

        is_valid = len([w for w in warnings if "exceeds" in w or "require" in w]) == 0

        return TestValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            recommendations=recommendations,
            estimated_duration_minutes=estimated_duration,
            estimated_memory_gb=estimated_memory
        )

    def _estimate_memory_usage(self, config: Any) -> float:
        """Estimate system memory usage."""
        base_memory = 2.0  # Base Python + libraries

        if "8B" in config.model_name or "8b" in config.model_name.lower():
            model_memory = 16.0 if config.use_qlora else 32.0
        elif "1B" in config.model_name or "1b" in config.model_name.lower():
            model_memory = 4.0 if config.use_qlora else 8.0
        else:
            model_memory = 2.0  # Small model

        data_memory = config.num_tasks * 0.01  # ~10MB per task
        buffer_memory = 4.0  # Buffer for operations

        return base_memory + model_memory + data_memory + buffer_memory

    def _estimate_gpu_memory(self, config: Any) -> float:
        """Estimate GPU memory usage."""
        if not config.use_qlora:
            return 0.0

        if "8B" in config.model_name or "8b" in config.model_name.lower():
            base_memory = 8.0  # 4-bit quantized 8B model
        elif "1B" in config.model_name or "1b" in config.model_name.lower():
            base_memory = 2.0  # 4-bit quantized 1B model
        else:
            base_memory = 1.0  # Small model

        # Add memory for gradients, optimizer states, etc.
        training_overhead = base_memory * 0.5
        buffer_memory = 2.0

        return base_memory + training_overhead + buffer_memory

    def _estimate_test_duration(self, config: Any) -> float:
        """Estimate test duration in minutes."""
        if "8B" in config.model_name or "8b" in config.model_name.lower():
            time_per_task = 12.0  # minutes
        elif "1B" in config.model_name or "1b" in config.model_name.lower():
            time_per_task = 5.0  # minutes
        else:
            time_per_task = 2.0  # minutes for small models

        # Account for parallelism
        effective_time = (config.num_tasks * time_per_task) / config.max_concurrent_tasks

        # Add overhead
        setup_time = 5.0
        overhead = effective_time * 0.1  # 10% overhead

        return setup_time + effective_time + overhead


class TestResultAnalyzer:
    """Analyzes test results and generates insights."""

    def __init__(self, results_dir: Path):
        """Initialize result analyzer."""
        self.results_dir = Path(results_dir)
        self.task_results = []
        self.summary_stats = {}
        self._load_results()

    def _load_results(self):
        """Load test results from files."""
        try:
            # Load task results
            task_results_file = self.results_dir / "task_results.json"
            if task_results_file.exists():
                with open(task_results_file) as f:
                    self.task_results = json.load(f)

            # Load summary stats
            summary_file = self.results_dir / "pipeline_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    self.summary_stats = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load results: {e}")

    def analyze_performance_patterns(self) -> dict[str, Any]:
        """Analyze performance patterns across tasks."""
        if not self.task_results:
            return {"error": "No task results available"}

        successful_tasks = [r for r in self.task_results if r["status"] == "success"]

        if not successful_tasks:
            return {"error": "No successful tasks to analyze"}

        # Execution time analysis
        execution_times = [r["execution_time"] for r in successful_tasks]
        time_stats = {
            "mean": statistics.mean(execution_times),
            "median": statistics.median(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "percentile_90": np.percentile(execution_times, 90),
            "percentile_95": np.percentile(execution_times, 95)
        }

        # Memory usage analysis
        memory_peaks = [r["memory_peak_mb"] for r in successful_tasks if r["memory_peak_mb"] > 0]
        memory_stats = {
            "mean": statistics.mean(memory_peaks) if memory_peaks else 0,
            "max": max(memory_peaks) if memory_peaks else 0,
            "std_dev": statistics.stdev(memory_peaks) if len(memory_peaks) > 1 else 0
        }

        # Accuracy analysis
        accuracies = [r["accuracy"] for r in successful_tasks if r["accuracy"] is not None]
        accuracy_stats = {
            "mean": statistics.mean(accuracies) if accuracies else 0,
            "median": statistics.median(accuracies) if accuracies else 0,
            "perfect_predictions": len([a for a in accuracies if a == 1.0]),
            "above_threshold": len([a for a in accuracies if a >= 0.5])
        }

        # Performance correlation analysis
        correlations = self._analyze_correlations(successful_tasks)

        return {
            "execution_time": time_stats,
            "memory_usage": memory_stats,
            "accuracy": accuracy_stats,
            "correlations": correlations,
            "task_count": len(successful_tasks)
        }

    def _analyze_correlations(self, tasks: list[dict]) -> dict[str, float]:
        """Analyze correlations between different metrics."""
        try:
            # Extract metrics for correlation analysis
            execution_times = [r["execution_time"] for r in tasks]
            memory_peaks = [r["memory_peak_mb"] for r in tasks if r["memory_peak_mb"] > 0]
            accuracies = [r["accuracy"] for r in tasks if r["accuracy"] is not None]

            correlations = {}

            # Only calculate if we have enough data points
            if len(execution_times) > 5 and len(memory_peaks) > 5:
                # Align arrays (use minimum length)
                min_len = min(len(execution_times), len(memory_peaks))
                time_memory_corr = np.corrcoef(
                    execution_times[:min_len],
                    memory_peaks[:min_len]
                )[0, 1]
                correlations["time_memory"] = float(time_memory_corr) if not np.isnan(time_memory_corr) else 0.0

            if len(execution_times) > 5 and len(accuracies) > 5:
                min_len = min(len(execution_times), len(accuracies))
                time_accuracy_corr = np.corrcoef(
                    execution_times[:min_len],
                    accuracies[:min_len]
                )[0, 1]
                correlations["time_accuracy"] = float(time_accuracy_corr) if not np.isnan(time_accuracy_corr) else 0.0

            return correlations

        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {}

    def identify_failure_patterns(self) -> dict[str, Any]:
        """Identify patterns in task failures."""
        failed_tasks = [r for r in self.task_results if r["status"] != "success"]

        if not failed_tasks:
            return {"message": "No task failures to analyze"}

        # Error category distribution
        error_categories = {}
        for task in failed_tasks:
            category = task.get("error_category", "unknown")
            error_categories[category] = error_categories.get(category, 0) + 1

        # Recovery attempt analysis
        recovery_stats = {
            "tasks_with_recovery": len([r for r in failed_tasks if r.get("recovery_attempts", 0) > 0]),
            "total_recovery_attempts": sum(r.get("recovery_attempts", 0) for r in failed_tasks),
            "max_recovery_attempts": max([r.get("recovery_attempts", 0) for r in failed_tasks], default=0)
        }

        # Time-based failure analysis
        timeout_failures = len([r for r in failed_tasks if r["status"] == "timeout"])

        return {
            "total_failures": len(failed_tasks),
            "error_categories": error_categories,
            "recovery_analysis": recovery_stats,
            "timeout_failures": timeout_failures,
            "failure_rate": len(failed_tasks) / len(self.task_results) if self.task_results else 0
        }

    def generate_performance_visualizations(self, output_dir: Path):
        """Generate performance visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        successful_tasks = [r for r in self.task_results if r["status"] == "success"]

        if not successful_tasks:
            logger.warning("No successful tasks for visualization")
            return

        # Execution time distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        execution_times = [r["execution_time"] for r in successful_tasks]
        plt.hist(execution_times, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Task Execution Time Distribution')

        # Memory usage distribution
        plt.subplot(2, 2, 2)
        memory_peaks = [r["memory_peak_mb"] for r in successful_tasks if r["memory_peak_mb"] > 0]
        if memory_peaks:
            plt.hist(memory_peaks, bins=20, edgecolor='black', alpha=0.7, color='orange')
            plt.xlabel('Peak Memory Usage (MB)')
            plt.ylabel('Frequency')
            plt.title('Memory Usage Distribution')

        # Accuracy distribution
        plt.subplot(2, 2, 3)
        accuracies = [r["accuracy"] for r in successful_tasks if r["accuracy"] is not None]
        if accuracies:
            plt.hist(accuracies, bins=20, edgecolor='black', alpha=0.7, color='green')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
            plt.title('Accuracy Distribution')

        # Success rate over time
        plt.subplot(2, 2, 4)
        task_indices = list(range(len(self.task_results)))
        success_indicators = [1 if r["status"] == "success" else 0 for r in self.task_results]

        # Calculate rolling success rate
        window_size = max(10, len(self.task_results) // 10)
        rolling_success = []
        for i in range(len(success_indicators)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_success = sum(success_indicators[start_idx:end_idx]) / len(success_indicators[start_idx:end_idx])
            rolling_success.append(window_success * 100)

        plt.plot(task_indices, rolling_success, linewidth=2, color='blue')
        plt.xlabel('Task Index')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Rolling Success Rate (Window: {window_size})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance visualizations saved to: {output_dir}")

    def generate_detailed_report(self, output_file: Path):
        """Generate detailed analysis report."""
        performance_patterns = self.analyze_performance_patterns()
        failure_patterns = self.identify_failure_patterns()

        report_lines = []

        # Header
        report_lines.extend([
            "="*80,
            "DETAILED PIPELINE TEST ANALYSIS REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Results Directory: {self.results_dir}",
            f"Total Tasks Analyzed: {len(self.task_results)}",
            ""
        ])

        # Performance Analysis
        if "error" not in performance_patterns:
            report_lines.extend([
                "PERFORMANCE ANALYSIS",
                "-"*30,
                "Execution Time (seconds):",
                f"  Mean: {performance_patterns['execution_time']['mean']:.2f}",
                f"  Median: {performance_patterns['execution_time']['median']:.2f}",
                f"  90th Percentile: {performance_patterns['execution_time']['percentile_90']:.2f}",
                f"  95th Percentile: {performance_patterns['execution_time']['percentile_95']:.2f}",
                "",
                "Memory Usage (MB):",
                f"  Mean Peak: {performance_patterns['memory_usage']['mean']:.2f}",
                f"  Maximum Peak: {performance_patterns['memory_usage']['max']:.2f}",
                "",
                "Accuracy Analysis:",
                f"  Mean Accuracy: {performance_patterns['accuracy']['mean']:.3f}",
                f"  Perfect Predictions: {performance_patterns['accuracy']['perfect_predictions']}",
                f"  Above 50% Accuracy: {performance_patterns['accuracy']['above_threshold']}",
                ""
            ])

        # Failure Analysis
        if "message" not in failure_patterns:
            report_lines.extend([
                "FAILURE ANALYSIS",
                "-"*20,
                f"Total Failures: {failure_patterns['total_failures']}",
                f"Failure Rate: {failure_patterns['failure_rate']:.1%}",
                f"Timeout Failures: {failure_patterns['timeout_failures']}",
                "",
                "Error Categories:",
            ])

            for category, count in failure_patterns['error_categories'].items():
                report_lines.append(f"  {category.replace('_', ' ').title()}: {count}")

            report_lines.extend([
                "",
                "Recovery Analysis:",
                f"  Tasks with Recovery: {failure_patterns['recovery_analysis']['tasks_with_recovery']}",
                f"  Total Recovery Attempts: {failure_patterns['recovery_analysis']['total_recovery_attempts']}",
                f"  Max Recovery Attempts: {failure_patterns['recovery_analysis']['max_recovery_attempts']}",
                ""
            ])

        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-"*20,
        ])

        if "error" not in performance_patterns:
            if performance_patterns['execution_time']['percentile_95'] > 600:  # 10 minutes
                report_lines.append("• Consider optimizing slow tasks (95th percentile > 10min)")

            if performance_patterns['memory_usage']['max'] > 20000:  # 20GB
                report_lines.append("• High memory usage detected - investigate memory optimization")

            if performance_patterns['accuracy']['mean'] < 0.5:
                report_lines.append("• Low average accuracy - review training configuration")

        if "message" not in failure_patterns:
            if failure_patterns['failure_rate'] > 0.3:
                report_lines.append("• High failure rate - investigate error patterns")

            if failure_patterns['timeout_failures'] > failure_patterns['total_failures'] * 0.5:
                report_lines.append("• Many timeout failures - consider increasing task timeout")

        if len(report_lines) == len(report_lines) - 1:  # No recommendations added
            report_lines.append("• Performance metrics within acceptable ranges")

        report_lines.extend([
            "",
            "="*80,
            "End of Analysis Report",
            "="*80
        ])

        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Detailed analysis report saved to: {output_file}")


def validate_pipeline_readiness() -> tuple[bool, list[str]]:
    """Validate that the system is ready for pipeline testing."""
    warnings = []

    # Check Python packages
    required_packages = [
        "torch", "transformers", "datasets", "peft",
        "bitsandbytes", "psutil", "matplotlib", "numpy"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        warnings.append(f"Missing packages: {', '.join(missing_packages)}")

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - testing will use CPU fallback")
    except ImportError:
        warnings.append("PyTorch not available")

    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)

        if free_gb < 50:
            warnings.append(f"Low disk space: {free_gb:.1f}GB available (recommend 50GB+)")
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")

    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)

        if memory_gb < 16:
            warnings.append(f"Low system memory: {memory_gb:.1f}GB (recommend 32GB+)")
    except Exception:
        warnings.append("Could not check system memory")

    is_ready = len([w for w in warnings if "Missing" in w or "not available" in w]) == 0

    return is_ready, warnings


if __name__ == "__main__":
    # Test the utilities
    logging.basicConfig(level=logging.INFO)

    # Validate system readiness
    is_ready, warnings = validate_pipeline_readiness()

    print("PIPELINE READINESS CHECK")
    print("="*40)
    print(f"System Ready: {'✅ YES' if is_ready else '❌ NO'}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
    else:
        print("✅ No issues detected")

    print("="*40)
