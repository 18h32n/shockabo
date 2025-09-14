"""
Advanced training monitoring and execution utilities for 8B model performance.

This module provides comprehensive monitoring, logging, and execution management
for the 8B model training process to achieve 53%+ validation accuracy.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import psutil
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingSnapshot:
    """Snapshot of training state at a specific point in time."""
    timestamp: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    accuracy: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_usage_percent: float
    training_time_elapsed: float
    task_id: str
    

@dataclass
class PerformanceMetrics:
    """Performance metrics for training monitoring."""
    average_loss: float
    loss_trend: str  # "improving", "stable", "degrading"
    accuracy_trend: str
    memory_efficiency: float
    time_per_epoch: float
    convergence_rate: float
    stability_score: float


class TrainingMonitor:
    """
    Comprehensive training monitor for 8B model performance tracking.
    
    Provides real-time monitoring, alerting, and performance analysis
    for the training process targeting 53%+ validation accuracy.
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        checkpoint_interval: int = 100,
        memory_threshold_mb: float = 20480,  # 20GB warning threshold
        save_dir: Optional[Path] = None
    ):
        """
        Initialize the training monitor.
        
        Args:
            log_interval: Steps between logging updates
            checkpoint_interval: Steps between checkpoints
            memory_threshold_mb: Memory usage warning threshold
            save_dir: Directory to save monitoring data
        """
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.save_dir = Path(save_dir) if save_dir else Path("logs/training_monitor")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state tracking
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.current_task_id = None
        self.training_start_time = None
        self.last_checkpoint_time = None
        self.performance_alerts: List[str] = []
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance analysis
        self.loss_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        
        logger.info(f"Training monitor initialized - Save dir: {self.save_dir}")
    
    def start_task_monitoring(self, task_id: str) -> None:
        """Start monitoring a new training task."""
        self.current_task_id = task_id
        self.training_start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.performance_alerts.clear()
        
        # Clear histories for new task
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.memory_history.clear()
        
        # Start monitoring thread
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._continuous_monitoring,
                daemon=True
            )
            self._monitor_thread.start()
        
        logger.info(f"Started monitoring for task: {task_id}")
    
    def stop_task_monitoring(self) -> None:
        """Stop monitoring current task."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        if self.current_task_id:
            # Save final monitoring data
            self._save_task_summary()
            logger.info(f"Stopped monitoring for task: {self.current_task_id}")
            self.current_task_id = None
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        accuracy: float = 0.0
    ) -> None:
        """Log a training step with performance metrics."""
        timestamp = datetime.now().isoformat()
        
        # Get system metrics
        memory_usage_mb = self._get_memory_usage()
        gpu_memory_mb = self._get_gpu_memory_usage()
        cpu_usage = psutil.cpu_percent()
        
        # Calculate elapsed time
        training_time_elapsed = (
            time.time() - self.training_start_time 
            if self.training_start_time else 0
        )
        
        # Create snapshot
        snapshot = TrainingSnapshot(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            accuracy=accuracy,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_usage_percent=cpu_usage,
            training_time_elapsed=training_time_elapsed,
            task_id=self.current_task_id or "unknown"
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        # Update histories
        self.loss_history.append(loss)
        if accuracy > 0:
            self.accuracy_history.append(accuracy)
        self.memory_history.append(memory_usage_mb)
        
        # Log at intervals
        if step % self.log_interval == 0:
            self._log_training_progress(snapshot)
        
        # Check for performance issues
        self._check_performance_alerts(snapshot)
        
        # Save checkpoint data at intervals
        if step % self.checkpoint_interval == 0:
            self._save_checkpoint_data(snapshot)
    
    def _log_training_progress(self, snapshot: TrainingSnapshot) -> None:
        """Log training progress with performance analysis."""
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        logger.info(
            f"Task {snapshot.task_id} | "
            f"Epoch {snapshot.epoch} Step {snapshot.step} | "
            f"Loss: {snapshot.loss:.4f} ({metrics.loss_trend}) | "
            f"Acc: {snapshot.accuracy:.2%} ({metrics.accuracy_trend}) | "
            f"LR: {snapshot.learning_rate:.2e} | "
            f"Mem: {snapshot.memory_usage_mb:.0f}MB | "
            f"GPU: {snapshot.gpu_memory_mb:.0f}MB | "
            f"Time: {snapshot.training_time_elapsed:.1f}s"
        )
        
        # Log performance analysis
        if len(self.snapshots) > 10:  # Only after some training
            logger.info(
                f"Performance | "
                f"Stability: {metrics.stability_score:.2f} | "
                f"Convergence: {metrics.convergence_rate:.4f} | "
                f"Efficiency: {metrics.memory_efficiency:.2%} | "
                f"ETA: {self._estimate_completion_time():.1f}min"
            )
    
    def _check_performance_alerts(self, snapshot: TrainingSnapshot) -> None:
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Memory usage alerts
        if snapshot.memory_usage_mb > self.memory_threshold_mb:
            alert = f"High memory usage: {snapshot.memory_usage_mb:.0f}MB > {self.memory_threshold_mb:.0f}MB"
            alerts.append(alert)
        
        # GPU memory alerts
        if snapshot.gpu_memory_mb > 20480:  # 20GB GPU memory warning
            alert = f"High GPU memory: {snapshot.gpu_memory_mb:.0f}MB"
            alerts.append(alert)
        
        # Loss instability alerts
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std / loss_mean > 0.5:  # High volatility
                alert = f"Loss instability detected: std/mean = {loss_std/loss_mean:.2f}"
                alerts.append(alert)
        
        # Training stagnation alerts
        if len(self.accuracy_history) >= 20:
            recent_accuracies = list(self.accuracy_history)[-20:]
            if np.std(recent_accuracies) < 0.001:  # Very low variance
                alert = "Training stagnation: accuracy not improving"
                alerts.append(alert)
        
        # CPU usage alerts
        if snapshot.cpu_usage_percent > 90:
            alert = f"High CPU usage: {snapshot.cpu_usage_percent:.1f}%"
            alerts.append(alert)
        
        # Log new alerts
        for alert in alerts:
            if alert not in self.performance_alerts:
                logger.warning(f"PERFORMANCE ALERT: {alert}")
                self.performance_alerts.append(alert)
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if len(self.snapshots) < 2:
            return PerformanceMetrics(
                average_loss=0.0,
                loss_trend="unknown",
                accuracy_trend="unknown",
                memory_efficiency=0.0,
                time_per_epoch=0.0,
                convergence_rate=0.0,
                stability_score=0.0
            )
        
        recent_snapshots = list(self.snapshots)[-20:]  # Last 20 snapshots
        
        # Loss analysis
        losses = [s.loss for s in recent_snapshots]
        average_loss = np.mean(losses)
        
        if len(losses) >= 5:
            early_losses = np.mean(losses[:len(losses)//2])
            late_losses = np.mean(losses[len(losses)//2:])
            
            if late_losses < early_losses * 0.95:
                loss_trend = "improving"
            elif late_losses > early_losses * 1.05:
                loss_trend = "degrading"
            else:
                loss_trend = "stable"
        else:
            loss_trend = "unknown"
        
        # Accuracy analysis
        accuracies = [s.accuracy for s in recent_snapshots if s.accuracy > 0]
        if len(accuracies) >= 5:
            early_acc = np.mean(accuracies[:len(accuracies)//2])
            late_acc = np.mean(accuracies[len(accuracies)//2:])
            
            if late_acc > early_acc + 0.01:
                accuracy_trend = "improving"
            elif late_acc < early_acc - 0.01:
                accuracy_trend = "degrading"
            else:
                accuracy_trend = "stable"
        else:
            accuracy_trend = "unknown"
        
        # Memory efficiency (lower memory usage is better)
        memory_usages = [s.memory_usage_mb for s in recent_snapshots]
        max_memory = max(memory_usages) if memory_usages else 1
        memory_efficiency = max(0.0, 1.0 - (max_memory / 24576))  # Normalized to 24GB
        
        # Time per epoch estimation
        if len(recent_snapshots) >= 2:
            time_per_step = (recent_snapshots[-1].training_time_elapsed - 
                           recent_snapshots[0].training_time_elapsed) / len(recent_snapshots)
            # Estimate based on typical epoch size
            time_per_epoch = time_per_step * 100  # Assume 100 steps per epoch
        else:
            time_per_epoch = 0.0
        
        # Convergence rate (rate of loss improvement)
        convergence_rate = 0.0
        if len(losses) >= 10:
            # Linear regression on recent losses
            x = np.arange(len(losses))
            slope = np.polyfit(x, losses, 1)[0]
            convergence_rate = -slope  # Negative slope means improving
        
        # Stability score (inverse of coefficient of variation)
        stability_score = 0.0
        if len(losses) >= 5:
            cv = np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 1
            stability_score = max(0.0, 1.0 - cv)
        
        return PerformanceMetrics(
            average_loss=average_loss,
            loss_trend=loss_trend,
            accuracy_trend=accuracy_trend,
            memory_efficiency=memory_efficiency,
            time_per_epoch=time_per_epoch,
            convergence_rate=convergence_rate,
            stability_score=stability_score
        )
    
    def _estimate_completion_time(self) -> float:
        """Estimate time to completion in minutes."""
        if len(self.snapshots) < 5 or not self.training_start_time:
            return 0.0
        
        recent_snapshots = list(self.snapshots)[-10:]
        
        # Calculate training velocity (steps per second)
        if len(recent_snapshots) >= 2:
            time_diff = recent_snapshots[-1].training_time_elapsed - recent_snapshots[0].training_time_elapsed
            steps_diff = recent_snapshots[-1].step - recent_snapshots[0].step
            
            if time_diff > 0 and steps_diff > 0:
                steps_per_second = steps_diff / time_diff
                
                # Estimate remaining steps (assume target of 1000 steps total)
                current_step = recent_snapshots[-1].step
                remaining_steps = max(0, 1000 - current_step)
                
                remaining_seconds = remaining_steps / steps_per_second
                return remaining_seconds / 60  # Convert to minutes
        
        return 0.0
    
    def _continuous_monitoring(self) -> None:
        """Background monitoring thread."""
        while self._monitoring:
            try:
                # Monitor system resources
                if len(self.snapshots) > 0:
                    latest = self.snapshots[-1]
                    
                    # Check for resource exhaustion
                    if latest.memory_usage_mb > self.memory_threshold_mb * 0.9:
                        logger.warning(f"Approaching memory limit: {latest.memory_usage_mb:.0f}MB")
                    
                    # Check for training stagnation
                    if len(self.loss_history) >= 50:
                        recent_losses = list(self.loss_history)[-50:]
                        if np.std(recent_losses) < 0.001:
                            logger.warning("Potential training stagnation detected")
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def _save_checkpoint_data(self, snapshot: TrainingSnapshot) -> None:
        """Save checkpoint monitoring data."""
        checkpoint_file = self.save_dir / f"checkpoint_{snapshot.task_id}_{snapshot.step}.json"
        
        # Prepare checkpoint data
        data = {
            "snapshot": asdict(snapshot),
            "performance_metrics": asdict(self._calculate_performance_metrics()),
            "recent_history": {
                "losses": list(self.loss_history)[-10:],
                "accuracies": list(self.accuracy_history)[-10:],
                "memory_usage": list(self.memory_history)[-10:]
            },
            "alerts": self.performance_alerts[-5:]  # Last 5 alerts
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint data: {e}")
    
    def _save_task_summary(self) -> None:
        """Save comprehensive task training summary."""
        if not self.current_task_id or not self.snapshots:
            return
        
        summary_file = self.save_dir / f"task_summary_{self.current_task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate final metrics
        final_metrics = self._calculate_performance_metrics()
        
        # Prepare summary data
        summary = {
            "task_id": self.current_task_id,
            "training_duration": time.time() - self.training_start_time if self.training_start_time else 0,
            "total_steps": len(self.snapshots),
            "final_metrics": asdict(final_metrics),
            "performance_alerts": self.performance_alerts,
            "resource_usage": {
                "peak_memory_mb": max([s.memory_usage_mb for s in self.snapshots]) if self.snapshots else 0,
                "peak_gpu_memory_mb": max([s.gpu_memory_mb for s in self.snapshots]) if self.snapshots else 0,
                "average_cpu_percent": np.mean([s.cpu_usage_percent for s in self.snapshots]) if self.snapshots else 0
            },
            "training_progress": {
                "initial_loss": self.snapshots[0].loss if self.snapshots else 0,
                "final_loss": self.snapshots[-1].loss if self.snapshots else 0,
                "best_accuracy": max([s.accuracy for s in self.snapshots]) if self.snapshots else 0,
                "final_accuracy": self.snapshots[-1].accuracy if self.snapshots else 0
            }
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            logger.info(f"Task summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save task summary: {e}")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current training performance metrics."""
        if not self.snapshots:
            return {"status": "no_data"}
        
        latest = self.snapshots[-1]
        metrics = self._calculate_performance_metrics()
        
        return {
            "status": "training",
            "task_id": self.current_task_id,
            "current_step": latest.step,
            "current_epoch": latest.epoch,
            "current_loss": latest.loss,
            "current_accuracy": latest.accuracy,
            "loss_trend": metrics.loss_trend,
            "accuracy_trend": metrics.accuracy_trend,
            "memory_usage_mb": latest.memory_usage_mb,
            "gpu_memory_mb": latest.gpu_memory_mb,
            "training_time_elapsed": latest.training_time_elapsed,
            "stability_score": metrics.stability_score,
            "estimated_completion_min": self._estimate_completion_time(),
            "recent_alerts": self.performance_alerts[-3:]
        }
    
    def export_training_logs(self, output_path: Optional[Path] = None) -> Path:
        """Export all training logs to a comprehensive file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.save_dir / f"training_export_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_snapshots": len(self.snapshots),
                "current_task": self.current_task_id
            },
            "training_snapshots": [asdict(s) for s in self.snapshots],
            "performance_metrics": asdict(self._calculate_performance_metrics()) if self.snapshots else {},
            "alerts_history": self.performance_alerts,
            "system_info": {
                "platform": "8B_Model_QLoRA_Training",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "total_gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Training logs exported to: {output_path}")
        return output_path