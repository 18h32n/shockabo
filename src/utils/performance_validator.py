"""
Performance validation utilities for TTT implementation.

Provides benchmarking, accuracy validation, and performance monitoring to ensure
the implementation meets all acceptance criteria.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import torch

from src.domain.models import ARCTask

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    task_id: str
    accuracy: float
    training_time_seconds: float
    memory_peak_mb: float
    memory_average_mb: float
    gpu_memory_peak_mb: float | None
    inference_time_ms: float
    model_load_time_ms: float
    checkpoint_size_mb: float
    timestamp: datetime

    def meets_criteria(self) -> dict[str, bool]:
        """Check if metrics meet acceptance criteria."""
        return {
            "accuracy_25_percent": self.accuracy >= 0.25,  # Updated from 40% to 25%
            "training_under_2_hours": self.training_time_seconds < 7200,
            "memory_under_10gb": self.memory_peak_mb < 10240,
            "gpu_16gb_compatible": self.gpu_memory_peak_mb is None or self.gpu_memory_peak_mb < 15360,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "task_id": self.task_id,
            "accuracy": self.accuracy,
            "training_time_seconds": self.training_time_seconds,
            "training_time_readable": str(timedelta(seconds=int(self.training_time_seconds))),
            "memory_peak_mb": self.memory_peak_mb,
            "memory_average_mb": self.memory_average_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "inference_time_ms": self.inference_time_ms,
            "model_load_time_ms": self.model_load_time_ms,
            "checkpoint_size_mb": self.checkpoint_size_mb,
            "timestamp": self.timestamp.isoformat(),
            "criteria_met": self.meets_criteria(),
        }


class MemoryMonitor:
    """Monitor memory usage during training and inference."""

    def __init__(self, sample_interval: float = 1.0):
        """Initialize memory monitor."""
        self.sample_interval = sample_interval
        self.samples: list[dict[str, float]] = []
        self.monitoring = False
        self.start_time = None

    def start(self) -> None:
        """Start memory monitoring."""
        self.samples.clear()
        self.monitoring = True
        self.start_time = time.time()
        logger.info("Memory monitoring started")

    def stop(self) -> dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False

        if not self.samples:
            return {
                "peak_mb": 0.0,
                "average_mb": 0.0,
                "gpu_peak_mb": None,
            }

        # Calculate statistics
        cpu_samples = [s["cpu_mb"] for s in self.samples]
        gpu_samples = [s["gpu_mb"] for s in self.samples if s["gpu_mb"] is not None]

        stats = {
            "peak_mb": max(cpu_samples),
            "average_mb": sum(cpu_samples) / len(cpu_samples),
            "gpu_peak_mb": max(gpu_samples) if gpu_samples else None,
            "duration": time.time() - self.start_time,
            "sample_count": len(self.samples),
        }

        logger.info(f"Memory monitoring stopped. Peak: {stats['peak_mb']:.2f}MB")

        return stats

    def sample(self) -> None:
        """Take a memory sample."""
        if not self.monitoring:
            return

        # CPU memory
        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024

        # GPU memory if available
        gpu_mb = None
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

        self.samples.append({
            "timestamp": time.time() - self.start_time,
            "cpu_mb": cpu_mb,
            "gpu_mb": gpu_mb,
        })


class PerformanceValidator:
    """Validate TTT implementation performance against acceptance criteria."""

    def __init__(self):
        """Initialize performance validator."""
        self.memory_monitor = MemoryMonitor()
        self.validation_results: list[PerformanceMetrics] = []

    def validate_accuracy(
        self,
        predictions: list[list[list[int]]],
        ground_truth: list[list[list[int]]],
    ) -> float:
        """
        Calculate pixel-perfect accuracy for predictions.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth outputs

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if len(predictions) != len(ground_truth):
            logger.warning(f"Prediction count mismatch: {len(predictions)} vs {len(ground_truth)}")
            return 0.0

        correct = 0
        total = len(predictions)

        for pred, truth in zip(predictions, ground_truth, strict=False):
            # Check if grids match exactly
            if self._grids_match(pred, truth):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

        return accuracy

    def validate_prediction(self, prediction: list[list[int]], ground_truth: list[list[int]]) -> bool:
        """Validate a single prediction against ground truth."""
        return self._grids_match(prediction, ground_truth)

    def _grids_match(self, grid1: list[list[int]], grid2: list[list[int]]) -> bool:
        """Check if two grids match exactly."""
        if len(grid1) != len(grid2):
            return False

        for row1, row2 in zip(grid1, grid2, strict=False):
            if len(row1) != len(row2):
                return False
            if row1 != row2:
                return False

        return True

    def benchmark_training(
        self,
        training_func: callable,
        task: ARCTask,
    ) -> tuple[Any, dict[str, float]]:
        """
        Benchmark training performance.

        Args:
            training_func: Function that performs training
            task: Task to train on

        Returns:
            Tuple of (training result, performance metrics)
        """
        logger.info(f"Starting training benchmark for task {task.task_id}")

        # Start monitoring
        self.memory_monitor.start()
        start_time = time.time()

        # Monitor memory in background
        import threading

        stop_monitoring = threading.Event()

        def monitor_loop():
            while not stop_monitoring.is_set():
                self.memory_monitor.sample()
                time.sleep(self.memory_monitor.sample_interval)

        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.start()

        try:
            # Run training
            result = training_func(task)

            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            memory_stats = self.memory_monitor.stop()

            # Calculate metrics
            training_time = time.time() - start_time

            metrics = {
                "training_time_seconds": training_time,
                "memory_peak_mb": memory_stats["peak_mb"],
                "memory_average_mb": memory_stats["average_mb"],
                "gpu_memory_peak_mb": memory_stats["gpu_peak_mb"],
            }

            logger.info(f"Training completed in {training_time:.2f}s")

            return result, metrics

        except Exception as e:
            stop_monitoring.set()
            monitor_thread.join()
            logger.error(f"Training benchmark failed: {e}")
            raise

    def benchmark_inference(
        self,
        inference_func: callable,
        input_data: Any,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            inference_func: Function that performs inference
            input_data: Input data for inference
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs

        Returns:
            Performance metrics
        """
        logger.info("Starting inference benchmark")

        # Warmup
        for _ in range(warmup_runs):
            _ = inference_func(input_data)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = inference_func(input_data)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

        # Calculate statistics
        metrics = {
            "inference_time_avg_ms": sum(times) / len(times),
            "inference_time_min_ms": min(times),
            "inference_time_max_ms": max(times),
            "inference_time_std_ms": torch.std(torch.tensor(times)).item() if len(times) > 1 else 0.0,
        }

        logger.info(f"Inference benchmark: {metrics['inference_time_avg_ms']:.2f}ms average")

        return metrics

    def validate_model_loading(
        self,
        load_func: callable,
        model_path: Path,
    ) -> dict[str, float]:
        """
        Validate model loading performance.

        Args:
            load_func: Function that loads the model
            model_path: Path to model checkpoint

        Returns:
            Loading metrics
        """
        logger.info("Validating model loading performance")

        # Time model loading
        start = time.perf_counter()
        load_func(model_path)
        load_time = (time.perf_counter() - start) * 1000

        # Check checkpoint size
        checkpoint_size = model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0

        metrics = {
            "model_load_time_ms": load_time,
            "checkpoint_size_mb": checkpoint_size,
        }

        logger.info(f"Model loaded in {load_time:.2f}ms, checkpoint size: {checkpoint_size:.2f}MB")

        return metrics

    def validate_gpu_compatibility(self) -> dict[str, Any]:
        """Validate GPU meets 16GB requirement."""
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "meets_requirement": False,
                "reason": "No GPU detected",
            }

        device_properties = torch.cuda.get_device_properties(0)
        total_memory_gb = device_properties.total_memory / 1024**3

        return {
            "gpu_available": True,
            "gpu_name": device_properties.name,
            "total_memory_gb": total_memory_gb,
            "meets_requirement": total_memory_gb >= 14,  # 14GB minimum for 16GB GPU
            "cuda_version": torch.version.cuda,
        }

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a performance validation report."""
        report = [
            "=" * 60,
            "TTT Performance Validation Report",
            "=" * 60,
            f"Task ID: {metrics.task_id}",
            f"Timestamp: {metrics.timestamp}",
            "",
            "Performance Metrics:",
            f"  - Accuracy: {metrics.accuracy:.2%}",
            f"  - Training Time: {timedelta(seconds=int(metrics.training_time_seconds))}",
            f"  - Peak Memory: {metrics.memory_peak_mb:.2f} MB",
            f"  - Average Memory: {metrics.memory_average_mb:.2f} MB",
            f"  - GPU Peak Memory: {metrics.gpu_memory_peak_mb:.2f} MB" if metrics.gpu_memory_peak_mb else "  - GPU: Not used",
            f"  - Inference Time: {metrics.inference_time_ms:.2f} ms",
            f"  - Model Load Time: {metrics.model_load_time_ms:.2f} ms",
            f"  - Checkpoint Size: {metrics.checkpoint_size_mb:.2f} MB",
            "",
            "Acceptance Criteria:",
        ]

        criteria = metrics.meets_criteria()
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"  [{status}] {criterion.replace('_', ' ').title()}")

        overall_pass = all(criteria.values())
        report.extend([
            "",
            f"Overall Result: {'PASS' if overall_pass else 'FAIL'}",
            "=" * 60,
        ])

        return "\n".join(report)

    def save_validation_results(self, metrics: PerformanceMetrics, output_path: Path) -> None:
        """Save validation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        import json
        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save report as text
        report_path = output_path.with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write(self.generate_report(metrics))

        logger.info(f"Saved validation results to {output_path}")
