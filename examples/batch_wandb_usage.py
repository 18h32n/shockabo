"""Example usage of the BatchedWandBClient for high-performance logging.

This example demonstrates how to use the async batch processing capabilities
of the W&B client to achieve better performance when logging many metrics.
"""

import asyncio
from datetime import datetime
from typing import List

from src.adapters.external.wandb_client import BatchedWandBClient, get_batched_wandb_client
from src.domain.evaluation_models import ExperimentRun, ResourceUsage, StrategyType, TaskStatus
from src.domain.services.evaluation_service import (
    AttemptNumber,
    EvaluationResult, 
    PixelAccuracy, 
    TaskMetrics
)


async def example_batch_evaluation_logging():
    """Example of batched evaluation result logging."""
    print("Example: Batched Evaluation Result Logging")
    
    # Get a batched client with custom settings
    client = get_batched_wandb_client(
        batch_size=25,           # Process 25 items per batch
        flush_interval_seconds=3.0,  # Flush every 3 seconds
        max_retry_attempts=2,    # Retry failed operations twice
        enable_batching=True
    )
    
    # Use as async context manager for automatic lifecycle management
    async with client:
        # Initialize W&B
        if not client.initialize():
            print("Failed to initialize W&B client")
            return
            
        # Start an experiment
        experiment = ExperimentRun(
            run_id="batch_example_001",
            experiment_name="Batch Processing Demo",
            task_ids=[f"task_{i}" for i in range(100)],
            strategy_config={"strategy": "batch_demo"},
            metrics={},
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now()
        )
        
        run_id = client.start_experiment(experiment, {"demo": True})
        if not run_id:
            print("Failed to start experiment")
            return
            
        print(f"Started experiment with run ID: {run_id}")
        
        # Log many evaluation results - these will be batched automatically
        for i in range(50):
            result = EvaluationResult(
                task_id=f"task_{i}",
                strategy_used="demo_strategy",
                attempts=[
                    TaskMetrics(
                        task_id=f"task_{i}",
                        attempt_number=AttemptNumber.FIRST,
                        pixel_accuracy=PixelAccuracy(
                            accuracy=0.7 + (i * 0.005),  # Gradually improving accuracy
                            total_pixels=100,
                            correct_pixels=70 + i // 2,
                            perfect_match=(i % 20) == 0
                        ),
                        confidence_score=0.8 + (i * 0.002),
                        processing_time_ms=100 + (i * 2),
                        error_category=None,
                        error_details={}
                    )
                ],
                total_processing_time_ms=100 + (i * 2)
            )
            
            # This queues the result for batch processing
            success = await client.log_evaluation_result_async(result)
            if not success:
                print(f"Failed to queue result for task_{i}")
            
            # Add some delay to simulate processing time
            await asyncio.sleep(0.01)
            
        # Log resource usage metrics
        for i in range(20):
            usage = ResourceUsage(
                task_id=f"resource_task_{i}",
                strategy_type=StrategyType.DIRECT_SOLVE,
                cpu_seconds=5.0 + (i * 0.5),
                memory_mb=128.0 + (i * 8),
                gpu_memory_mb=512.0,
                api_calls={"openai": i + 1, "anthropic": i // 2},
                total_tokens=500 + (i * 50),
                estimated_cost=0.005 + (i * 0.001),
                timestamp=datetime.now()
            )
            
            success = await client.log_resource_usage_async(usage)
            if not success:
                print(f"Failed to queue resource usage for resource_task_{i}")
                
        # Log custom metrics
        for i in range(30):
            custom_metrics = {
                "iteration": i,
                "learning_rate": 0.001 * (0.9 ** (i // 10)),
                "loss": 2.5 * (0.95 ** i),
                "validation_accuracy": 0.6 + (i * 0.01),
                "batch_processed": True
            }
            
            success = await client.log_custom_metrics_async(custom_metrics)
            if not success:
                print(f"Failed to queue custom metrics for iteration {i}")
                
        # Wait a moment for final batches to process
        await asyncio.sleep(2.0)
        
        # Get and display batch processing metrics
        batch_metrics = client.get_batch_metrics()
        queue_sizes = client.get_queue_sizes()
        
        print("\nBatch Processing Performance:")
        print(f"  Total operations: {batch_metrics['total_operations']}")
        print(f"  Successful operations: {batch_metrics['successful_operations']}")
        print(f"  Success rate: {batch_metrics['success_rate']:.2%}")
        print(f"  Total batches: {batch_metrics['total_batches']}")
        print(f"  Average batch size: {batch_metrics['average_batch_size']:.1f}")
        print(f"  Average flush time: {batch_metrics['average_flush_time_ms']:.1f}ms")
        print(f"  Partial failures: {batch_metrics['partial_failures']}")
        print(f"  Retry operations: {batch_metrics['retry_operations']}")
        
        print(f"\nQueue Status:")
        print(f"  Pending operations: {queue_sizes['batch_queue']}")
        print(f"  Retry operations: {queue_sizes['retry_queue']}")
        
        # End the experiment
        client.end_experiment()
        print("\nExperiment completed!")


async def example_manual_batch_control():
    """Example of manual batch control with explicit flushing."""
    print("\nExample: Manual Batch Control")
    
    client = BatchedWandBClient(
        batch_size=100,  # Large batch size
        flush_interval_seconds=60.0,  # Long interval
        enable_batching=True
    )
    
    # Manual lifecycle management
    await client.start_batch_processing()
    
    try:
        if client.initialize():
            # Log some metrics
            for i in range(15):
                await client.log_custom_metrics_async({
                    "manual_metric": i,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Manually force a flush before the batch is full
            print("Forcing batch flush...")
            await client._flush_batches(force=True)
            
            # Check metrics after manual flush
            metrics = client.get_batch_metrics()
            print(f"Operations after manual flush: {metrics['total_operations']}")
            
    finally:
        await client.stop_batch_processing()


def example_sync_compatibility():
    """Example showing backwards compatibility with sync methods."""
    print("\nExample: Sync Method Compatibility")
    
    # Regular client works exactly as before
    from src.adapters.external.wandb_client import get_wandb_client
    
    sync_client = get_wandb_client()
    
    if sync_client.initialize():
        print("Sync client initialized successfully")
        
        # All existing sync methods work unchanged
        result = EvaluationResult(
            task_id="sync_task",
            strategy_used="sync_strategy", 
            attempts=[
                TaskMetrics(
                    task_id="sync_task",
                    attempt_number=AttemptNumber.FIRST,
                    pixel_accuracy=PixelAccuracy(
                        accuracy=0.95,
                        total_pixels=100,
                        correct_pixels=95,
                        perfect_match=True
                    ),
                    confidence_score=0.98,
                    processing_time_ms=75,
                    error_category=None,
                    error_details={}
                )
            ],
            total_processing_time_ms=75
        )
        
        # This works exactly as before (no batching)
        success = sync_client.log_evaluation_result(result)
        print(f"Sync logging successful: {success}")


async def main():
    """Main example runner."""
    print("BatchedWandBClient Examples")
    print("=" * 50)
    
    # Note: These examples require a valid W&B setup
    # In a real environment, ensure WANDB_API_KEY is set
    
    try:
        # Run async examples
        await example_batch_evaluation_logging()
        await example_manual_batch_control()
        
        # Run sync example
        example_sync_compatibility()
        
    except Exception as e:
        print(f"Example failed (this is expected without W&B setup): {e}")


if __name__ == "__main__":
    asyncio.run(main())