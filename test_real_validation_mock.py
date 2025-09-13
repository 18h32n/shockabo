#!/usr/bin/env python3
"""
Quick test script to validate data pipeline with real ARC dataset using mock adapter.
Tests data loading and processing without requiring model authentication.
"""
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from adapters.repositories.arc_data_repository import ARCDataRepository
from domain.models import ARCTask, ARCTaskSolution, ResourceUsage, StrategyType
from utils.performance_validator import PerformanceValidator

class MockTTTAdapter:
    """Mock TTT adapter for testing data pipeline without model requirements."""
    
    def __init__(self):
        self.solved_count = 0
    
    def solve(self, task: ARCTask) -> ARCTaskSolution:
        """Mock solve method that returns a simple prediction."""
        start_time = time.time()
        
        # Create a mock prediction (just return test input as prediction)
        if task.test_input:
            prediction = task.test_input
        else:
            prediction = [[0]]  # Fallback prediction
        
        # Simulate some processing time
        time.sleep(0.1)
        
        solve_time = time.time() - start_time
        self.solved_count += 1
        
        # Create resource usage
        resource_usage = ResourceUsage(
            task_id=task.task_id,
            strategy_type=StrategyType.TEST_TIME_TRAINING,
            cpu_seconds=solve_time,
            memory_mb=100.0,  # Mock memory usage
            gpu_memory_mb=0.0,  # No GPU usage
            api_calls={},
            total_tokens=0,
            estimated_cost=0.0,
            timestamp=datetime.now()
        )
        
        # Create solution
        solution = ARCTaskSolution(
            task_id=task.task_id,
            predictions=[prediction],
            confidence_score=0.5,  # Mock confidence
            strategy_used=StrategyType.TEST_TIME_TRAINING,
            resource_usage=resource_usage,
            metadata={
                "adaptation_metrics": {"mock_score": 0.75},
                "processing_time": solve_time
            }
        )
        
        return solution
    
    def cleanup(self):
        """Mock cleanup method."""
        logger.info(f"Mock adapter cleaned up. Solved {self.solved_count} tasks.")

def test_real_dataset_validation_pipeline():
    """Test complete validation pipeline with real dataset and mock adapter."""
    logger.info("Testing complete validation pipeline with real ARC dataset...")
    
    try:
        # Initialize repository with real dataset
        repo = ARCDataRepository(use_real_dataset=True)
        validator = PerformanceValidator()
        
        # Load a small set of tasks for testing
        test_tasks = repo.load_all_tasks("training", limit=10)
        logger.info(f"Loaded {len(test_tasks)} tasks for validation testing")
        
        # Initialize mock adapter
        adapter = MockTTTAdapter()
        
        # Process tasks
        results = []
        correct_count = 0
        total_time = 0
        
        for task_id, task in test_tasks.items():
            logger.info(f"Processing task {task_id}...")
            
            # Solve task
            solution = adapter.solve(task)
            
            # Validate prediction if we have expected output
            is_correct = False
            if task.test_output is not None:
                is_correct = validator.validate_prediction(
                    solution.predictions[0],
                    task.test_output
                )
                if is_correct:
                    correct_count += 1
            
            # Record result
            result = {
                "task_id": task_id,
                "correct": is_correct,
                "has_expected_output": task.test_output is not None,
                "processing_time": solution.metadata.get("processing_time", 0),
                "memory_usage": solution.resource_usage.memory_mb
            }
            results.append(result)
            total_time += result["processing_time"]
        
        # Calculate metrics
        accuracy = correct_count / len([r for r in results if r["has_expected_output"]]) if any(r["has_expected_output"] for r in results) else 0
        avg_time = total_time / len(results) if results else 0
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("VALIDATION PIPELINE TEST RESULTS")
        logger.info("="*50)
        logger.info(f"Tasks processed: {len(results)}")
        logger.info(f"Tasks with expected output: {sum(1 for r in results if r['has_expected_output'])}")
        logger.info(f"Correct predictions: {correct_count}")
        logger.info(f"Mock accuracy: {accuracy:.2%}")
        logger.info(f"Average processing time: {avg_time:.3f}s")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info("="*50)
        
        # Show sample results
        logger.info("\nSample task results:")
        for result in results[:5]:
            logger.info(f"  {result['task_id']}: {'✓' if result['correct'] else '✗'} "
                       f"({result['processing_time']:.3f}s, {result['memory_usage']:.1f}MB)")
        
        # Cleanup
        adapter.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Validation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_sources():
    """Test all available dataset sources."""
    logger.info("Testing available dataset sources...")
    
    try:
        repo = ARCDataRepository(use_real_dataset=True)
        
        sources = ["training", "evaluation", "test"]
        for source in sources:
            task_ids = repo.get_task_ids(source)
            logger.info(f"{source} dataset: {len(task_ids)} tasks")
            
            if len(task_ids) > 0:
                # Test loading a few tasks from each source
                sample_tasks = repo.load_all_tasks(source, limit=3)
                logger.info(f"  Successfully loaded {len(sample_tasks)} sample tasks")
                
                # Check if tasks have solutions
                has_solutions = 0
                for task in sample_tasks.values():
                    if task.test_output is not None:
                        has_solutions += 1
                
                logger.info(f"  Tasks with solutions: {has_solutions}/{len(sample_tasks)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset sources test failed: {e}")
        return False

def main():
    """Run comprehensive validation tests."""
    logger.info("=== Starting Real ARC Dataset Pipeline Validation ===")
    
    # Test 1: Dataset sources
    success = test_dataset_sources()
    if not success:
        logger.error("Dataset sources test failed - aborting")
        return False
    
    # Test 2: Complete validation pipeline
    success = test_real_dataset_validation_pipeline()
    if not success:
        logger.error("Validation pipeline test failed")
        return False
    
    logger.info("=== All validation pipeline tests passed! ===")
    logger.info("Real ARC dataset integration is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)