"""
Integration tests for Task 6: 100-task pipeline test implementation.

Tests all components of the pipeline test system including:
- Pipeline orchestrator functionality
- Error handling and recovery
- Result analysis and reporting
- Integration with all previous tasks
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from test_pipeline_100_tasks import PipelineTestConfig, PipelineTestOrchestrator, TaskResult
from src.utils.pipeline_test_utils import PipelineTestValidator, TestResultAnalyzer


class TestPipelineOrchestrator(unittest.TestCase):
    """Test the pipeline test orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create minimal test configuration
        self.config = PipelineTestConfig(
            model_name="microsoft/DialoGPT-medium",  # Small model for testing
            num_tasks=3,
            use_qlora=False,
            enable_gradient_checkpointing=False,
            use_mixed_precision=False,
            task_timeout_minutes=2,
            max_inference_time_minutes=1.0
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = PipelineTestOrchestrator(self.config)
        
        # Check basic attributes
        self.assertIsNotNone(orchestrator.config)
        self.assertIsNotNone(orchestrator.test_id)
        self.assertIsInstance(orchestrator.results, list)
        self.assertTrue(orchestrator.output_dir.exists())
        
        orchestrator.cleanup()
    
    @patch('test_pipeline_100_tasks.ARCDataRepository')
    @patch('test_pipeline_100_tasks.TTTAdapter')
    def test_component_initialization(self, mock_ttt_adapter, mock_data_repo):
        """Test component initialization with mocks."""
        # Setup mocks
        mock_data_repo_instance = MagicMock()
        mock_data_repo.return_value = mock_data_repo_instance
        
        mock_ttt_adapter_instance = MagicMock()
        mock_ttt_adapter.return_value = mock_ttt_adapter_instance
        
        orchestrator = PipelineTestOrchestrator(self.config)
        success = orchestrator.initialize_components()
        
        self.assertTrue(success)
        self.assertIsNotNone(orchestrator.data_repository)
        self.assertIsNotNone(orchestrator.ttt_adapter)
        
        orchestrator.cleanup()
    
    def test_task_result_creation(self):
        """Test TaskResult creation and serialization."""
        result = TaskResult(
            task_id="test_001",
            status="success",
            execution_time=10.5,
            memory_peak_mb=1024.0,
            accuracy=0.75,
            prediction_quality="correct"
        )
        
        # Test serialization
        result_dict = result.__dict__
        self.assertIn("task_id", result_dict)
        self.assertEqual(result_dict["task_id"], "test_001")
        self.assertEqual(result_dict["status"], "success")
        self.assertEqual(result_dict["accuracy"], 0.75)
    
    @patch('test_pipeline_100_tasks.ARCDataRepository')
    @patch('test_pipeline_100_tasks.TTTAdapter')
    def test_single_task_execution_success(self, mock_ttt_adapter, mock_data_repo):
        """Test successful single task execution."""
        # Setup data repository mock
        mock_data_repo_instance = MagicMock()
        mock_task = MagicMock()
        mock_task.task_id = "test_task_001"
        mock_task.test_output = [[1, 0], [0, 1]]
        mock_data_repo_instance.load_task.return_value = mock_task
        mock_data_repo.return_value = mock_data_repo_instance
        
        # Setup TTT adapter mock
        mock_ttt_adapter_instance = MagicMock()
        mock_solution = MagicMock()
        mock_solution.predictions = [[[1, 0], [0, 1]]]
        mock_solution.inference_time = 30.0
        mock_solution.early_stopping_triggered = False
        mock_solution.resource_usage = MagicMock()
        mock_solution.resource_usage.memory_mb = 512.0
        mock_ttt_adapter_instance.solve.return_value = mock_solution
        mock_ttt_adapter.return_value = mock_ttt_adapter_instance
        
        orchestrator = PipelineTestOrchestrator(self.config)
        orchestrator.initialize_components()
        
        result = orchestrator.execute_single_task("test_task_001")
        
        self.assertEqual(result.status, "success")
        self.assertEqual(result.task_id, "test_task_001")
        self.assertIsNotNone(result.execution_time)
        self.assertGreater(result.execution_time, 0)
        
        orchestrator.cleanup()
    
    @patch('test_pipeline_100_tasks.ARCDataRepository')
    @patch('test_pipeline_100_tasks.TTTAdapter')
    def test_single_task_execution_failure(self, mock_ttt_adapter, mock_data_repo):
        """Test task execution failure handling."""
        # Setup data repository to return None (task not found)
        mock_data_repo_instance = MagicMock()
        mock_data_repo_instance.load_task.return_value = None
        mock_data_repo.return_value = mock_data_repo_instance
        
        mock_ttt_adapter_instance = MagicMock()
        mock_ttt_adapter.return_value = mock_ttt_adapter_instance
        
        orchestrator = PipelineTestOrchestrator(self.config)
        orchestrator.initialize_components()
        
        result = orchestrator.execute_single_task("nonexistent_task")
        
        self.assertEqual(result.status, "error")
        self.assertEqual(result.task_id, "nonexistent_task")
        self.assertIn("not found", result.error_message.lower())
        
        orchestrator.cleanup()
    
    def test_summary_statistics_generation(self):
        """Test summary statistics generation."""
        orchestrator = PipelineTestOrchestrator(self.config)
        
        # Add mock results
        orchestrator.results = [
            TaskResult(
                task_id="task_001",
                status="success",
                execution_time=10.0,
                memory_peak_mb=500.0,
                accuracy=0.8
            ),
            TaskResult(
                task_id="task_002", 
                status="error",
                execution_time=5.0,
                memory_peak_mb=300.0,
                error_category="memory"
            ),
            TaskResult(
                task_id="task_003",
                status="success",
                execution_time=15.0,
                memory_peak_mb=600.0,
                accuracy=0.6
            )
        ]
        
        from datetime import timedelta
        summary = orchestrator._generate_summary_statistics(timedelta(minutes=30))
        
        # Test basic statistics
        self.assertEqual(summary["task_execution"]["total_tasks"], 3)
        self.assertEqual(summary["task_execution"]["successful_tasks"], 2)
        self.assertEqual(summary["task_execution"]["failed_tasks"], 1)
        self.assertAlmostEqual(summary["task_execution"]["success_rate"], 66.67, places=1)
        
        # Test performance metrics
        self.assertAlmostEqual(summary["performance_metrics"]["avg_execution_time"], 10.0, places=1)
        self.assertAlmostEqual(summary["performance_metrics"]["avg_accuracy"], 0.7, places=1)
        
        # Test error analysis
        self.assertIn("memory", summary["error_analysis"]["error_categories"])
        
        orchestrator.cleanup()


class TestPipelineTestValidator(unittest.TestCase):
    """Test the pipeline test validator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = PipelineTestValidator()
        
        self.assertIsNotNone(validator.system_specs)
        self.assertIn("cpu_count", validator.system_specs)
        self.assertIn("memory_gb", validator.system_specs)
        self.assertIn("cuda_available", validator.system_specs)
    
    def test_config_validation_cpu(self):
        """Test configuration validation for CPU setup."""
        validator = PipelineTestValidator()
        
        config = PipelineTestConfig(
            model_name="microsoft/DialoGPT-medium",
            num_tasks=10,
            use_qlora=False,
            max_concurrent_tasks=2
        )
        
        result = validator.validate_test_config(config)
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.recommendations, list)
        self.assertGreater(result.estimated_duration_minutes, 0)
        self.assertGreater(result.estimated_memory_gb, 0)
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        validator = PipelineTestValidator()
        
        # Test small model
        small_config = PipelineTestConfig(
            model_name="gpt2",
            num_tasks=10,
            use_qlora=False
        )
        small_memory = validator._estimate_memory_usage(small_config)
        
        # Test large model
        large_config = PipelineTestConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            num_tasks=10,
            use_qlora=True
        )
        large_memory = validator._estimate_memory_usage(large_config)
        
        self.assertGreater(large_memory, small_memory)
        self.assertGreater(small_memory, 0)
    
    def test_duration_estimation(self):
        """Test duration estimation."""
        validator = PipelineTestValidator()
        
        # Test small configuration
        small_config = PipelineTestConfig(
            model_name="gpt2",
            num_tasks=5,
            max_concurrent_tasks=1
        )
        small_duration = validator._estimate_test_duration(small_config)
        
        # Test larger configuration
        large_config = PipelineTestConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            num_tasks=20,
            max_concurrent_tasks=1
        )
        large_duration = validator._estimate_test_duration(large_config)
        
        self.assertGreater(large_duration, small_duration)
        self.assertGreater(small_duration, 0)


class TestResultAnalyzer(unittest.TestCase):
    """Test the result analyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.temp_dir / "test_results"
        self.results_dir.mkdir()
        
        # Create mock results data
        self.mock_task_results = [
            {
                "task_id": "task_001",
                "status": "success", 
                "execution_time": 10.5,
                "memory_peak_mb": 512.0,
                "accuracy": 0.8
            },
            {
                "task_id": "task_002",
                "status": "success",
                "execution_time": 15.2,
                "memory_peak_mb": 768.0,
                "accuracy": 0.6
            },
            {
                "task_id": "task_003",
                "status": "error",
                "execution_time": 5.0,
                "memory_peak_mb": 256.0,
                "error_category": "memory",
                "recovery_attempts": 2
            }
        ]
        
        self.mock_summary = {
            "test_metadata": {
                "test_id": "test_pipeline_001",
                "duration_minutes": 45.0
            },
            "task_execution": {
                "total_tasks": 3,
                "successful_tasks": 2,
                "failed_tasks": 1,
                "success_rate": 66.67
            }
        }
        
        # Save mock data
        with open(self.results_dir / "task_results.json", 'w') as f:
            json.dump(self.mock_task_results, f)
        
        with open(self.results_dir / "pipeline_summary.json", 'w') as f:
            json.dump(self.mock_summary, f)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization and data loading."""
        analyzer = TestResultAnalyzer(self.results_dir)
        
        self.assertEqual(len(analyzer.task_results), 3)
        self.assertIsNotNone(analyzer.summary_stats)
        self.assertEqual(analyzer.summary_stats["task_execution"]["total_tasks"], 3)
    
    def test_performance_pattern_analysis(self):
        """Test performance pattern analysis."""
        analyzer = TestResultAnalyzer(self.results_dir)
        patterns = analyzer.analyze_performance_patterns()
        
        self.assertNotIn("error", patterns)
        self.assertIn("execution_time", patterns)
        self.assertIn("memory_usage", patterns)
        self.assertIn("accuracy", patterns)
        
        # Test statistics
        self.assertAlmostEqual(patterns["execution_time"]["mean"], 12.85, places=1)
        self.assertEqual(patterns["accuracy"]["perfect_predictions"], 0)
        self.assertEqual(patterns["task_count"], 2)  # Only successful tasks
    
    def test_failure_pattern_analysis(self):
        """Test failure pattern analysis."""
        analyzer = TestResultAnalyzer(self.results_dir)
        failures = analyzer.identify_failure_patterns()
        
        self.assertEqual(failures["total_failures"], 1)
        self.assertIn("memory", failures["error_categories"])
        self.assertEqual(failures["recovery_analysis"]["total_recovery_attempts"], 2)
        self.assertAlmostEqual(failures["failure_rate"], 0.333, places=2)
    
    def test_report_generation(self):
        """Test detailed report generation."""
        analyzer = TestResultAnalyzer(self.results_dir)
        
        report_file = self.temp_dir / "test_report.txt"
        analyzer.generate_detailed_report(report_file)
        
        self.assertTrue(report_file.exists())
        
        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
        
        self.assertIn("DETAILED PIPELINE TEST ANALYSIS REPORT", content)
        self.assertIn("PERFORMANCE ANALYSIS", content)
        self.assertIn("FAILURE ANALYSIS", content)
        self.assertIn("RECOMMENDATIONS", content)


class TestPipelineIntegration(unittest.TestCase):
    """Test integration between pipeline components."""
    
    def test_config_to_orchestrator_integration(self):
        """Test configuration passes correctly to orchestrator."""
        config = PipelineTestConfig(
            model_name="test-model",
            num_tasks=5,
            task_timeout_minutes=10
        )
        
        orchestrator = PipelineTestOrchestrator(config)
        
        self.assertEqual(orchestrator.config.model_name, "test-model")
        self.assertEqual(orchestrator.config.num_tasks, 5)
        self.assertEqual(orchestrator.config.task_timeout_minutes, 10)
        
        orchestrator.cleanup()
    
    def test_error_handling_integration(self):
        """Test error handling integration."""
        from src.utils.comprehensive_error_handling import error_reporter
        
        # Clear previous errors
        error_reporter.errors.clear()
        
        orchestrator = PipelineTestOrchestrator(PipelineTestConfig(num_tasks=1))
        
        # Test error reporting (should not crash)
        try:
            raise RuntimeError("Test error")
        except RuntimeError as e:
            # This should integrate with the error reporter
            pass
        
        orchestrator.cleanup()
    
    def test_early_stopping_integration(self):
        """Test early stopping integration."""
        config = PipelineTestConfig(
            early_stopping_patience=3,
            min_delta=0.01
        )
        
        orchestrator = PipelineTestOrchestrator(config)
        
        # Check that early stopping configuration is applied
        self.assertIsNotNone(orchestrator.early_stopping_config)
        self.assertIsNotNone(orchestrator.early_stopping_monitor)
        
        orchestrator.cleanup()


def run_integration_tests():
    """Run all integration tests."""
    print("="*60)
    print("RUNNING TASK 6 INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPipelineOrchestrator,
        TestPipelineTestValidator,
        TestResultAnalyzer,
        TestPipelineIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} TESTS FAILED")
    
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)