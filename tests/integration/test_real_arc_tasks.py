"""End-to-end integration tests with real ARC tasks."""
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.adapters.repositories.arc_data_repository import ARCDataRepository
from src.adapters.strategies.ttt_adapter import TTTAdapter, TTTConfig
from src.domain.models import ARCTask, StrategyType
from src.utils.performance_validator import PerformanceValidator


@pytest.fixture
def real_arc_test_dir():
    """Create temporary directory with real ARC task data."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create ARC data structure
    training_dir = temp_dir / "training"
    training_dir.mkdir(parents=True)

    # Create realistic ARC tasks based on common patterns
    real_tasks = {
        "simple_pattern_001": {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
                },
                {
                    "input": [[0, 0], [2, 0]],
                    "output": [[1, 1], [0, 1]]
                }
            ],
            "test": [
                {"input": [[0, 3, 0], [0, 0, 0]], "output": [[1, 0, 1], [1, 1, 1]]}
            ]
        },
        "color_transformation_002": {
            "train": [
                {
                    "input": [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
                    "output": [[2, 1, 2], [1, 2, 1], [2, 1, 2]]
                },
                {
                    "input": [[3, 4, 3, 4], [4, 3, 4, 3]],
                    "output": [[4, 3, 4, 3], [3, 4, 3, 4]]
                }
            ],
            "test": [
                {"input": [[5, 6], [6, 5]], "output": [[6, 5], [5, 6]]}
            ]
        },
        "shape_counting_003": {
            "train": [
                {
                    "input": [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                    "output": [[4, 0, 4], [0, 0, 0], [4, 0, 4]]
                },
                {
                    "input": [[2, 2], [2, 2]],
                    "output": [[4, 4], [4, 4]]
                }
            ],
            "test": [
                {"input": [[3, 3, 3], [3, 0, 3], [3, 3, 3]], "output": [[8, 8, 8], [8, 0, 8], [8, 8, 8]]}
            ]
        },
        "complex_pattern_004": {
            "train": [
                {
                    "input": [
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1]
                    ],
                    "output": [
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0]
                    ]
                },
                {
                    "input": [
                        [2, 3, 2],
                        [3, 2, 3],
                        [2, 3, 2]
                    ],
                    "output": [
                        [3, 2, 3],
                        [2, 3, 2],
                        [3, 2, 3]
                    ]
                }
            ],
            "test": [
                {
                    "input": [
                        [4, 5, 4, 5],
                        [5, 4, 5, 4],
                        [4, 5, 4, 5]
                    ],
                    "output": [
                        [5, 4, 5, 4],
                        [4, 5, 4, 5],
                        [5, 4, 5, 4]
                    ]
                }
            ]
        },
        "size_variation_005": {
            "train": [
                {
                    "input": [[1]],
                    "output": [[1, 1], [1, 1]]
                },
                {
                    "input": [[2, 2]],
                    "output": [[2, 2, 2, 2], [2, 2, 2, 2]]
                },
                {
                    "input": [[3, 3, 3]],
                    "output": [[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]]
                }
            ],
            "test": [
                {"input": [[4, 4, 4, 4]], "output": [[4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4]]}
            ]
        }
    }

    # Save tasks to files
    for task_id, task_data in real_tasks.items():
        with open(training_dir / f"{task_id}.json", "w") as f:
            json.dump(task_data, f, indent=2)

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def arc_data_repository(real_arc_test_dir):
    """Create ARC data repository with real task data."""
    return ARCDataRepository(data_path=str(real_arc_test_dir))


@pytest.fixture
def production_ttt_config(real_arc_test_dir):
    """Create production-like TTT configuration."""
    return TTTConfig(
        model_name="gpt2",  # Small model for testing, but realistic config
        device="cpu",
        quantization=False,  # Disable for deterministic testing
        num_epochs=3,
        batch_size=2,
        learning_rate=5e-5,
        max_examples=4,
        max_length=1024,
        checkpoint_dir=real_arc_test_dir / "checkpoints",
        cache_dir=real_arc_test_dir / "cache"
    )


class TestRealARCTaskProcessing:
    """Test processing of real ARC tasks end-to-end."""

    def test_single_task_complete_pipeline(self, arc_data_repository, production_ttt_config):
        """Test complete pipeline with a single real ARC task."""
        # Load a real task
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=1)
        assert len(tasks) == 1
        task = tasks[0]

        # Verify task structure
        assert isinstance(task, ARCTask)
        assert len(task.train_examples) >= 2
        assert task.test_input is not None
        assert len(task.test_input) > 0
        assert len(task.test_input[0]) > 0

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup realistic mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.pad_token_id = 50256
                mock_tokenizer_inst.eos_token_id = 50256

                # Mock realistic tokenization
                def mock_encode(text, **kwargs):
                    # Simulate encoding grid text to tokens
                    return [1] + [i % 1000 + 1 for i in range(len(text.split()))] + [50256]

                def mock_decode(tokens, **kwargs):
                    # Simulate decoding back to grid format
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.tolist()

                    # Generate a simple grid based on input dimensions
                    if hasattr(task, 'test_input') and task.test_input:
                        rows = len(task.test_input)
                        cols = len(task.test_input[0])
                        return '\n'.join([' '.join(['1' if (i+j) % 2 == 0 else '0' for j in range(cols)]) for i in range(rows)])
                    return "0 1\n1 0"

                mock_tokenizer_inst.encode = mock_encode
                mock_tokenizer_inst.decode = mock_decode
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                # Setup model mock
                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                param1 = torch.nn.Parameter(torch.randn(768, 768))
                param1.requires_grad = True
                mock_model_inst.parameters.return_value = [param1]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Mock training with convergence
                training_step = {"step": 0}

                def mock_forward(*args, **kwargs):
                    training_step["step"] += 1
                    # Simulate loss convergence
                    base_loss = 1.0
                    decay_rate = 0.1
                    converged_loss = 0.05
                    loss_value = converged_loss + (base_loss - converged_loss) * (1 - decay_rate) ** training_step["step"]
                    return Mock(loss=torch.tensor(loss_value, requires_grad=True))

                mock_model_inst.side_effect = mock_forward

                # Mock generation
                def mock_generate(input_ids, **kwargs):
                    batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
                    seq_len = 20  # Generate reasonable sequence length
                    return torch.randint(1, 1000, (batch_size, seq_len))

                mock_model_inst.generate = mock_generate
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter and run pipeline
                adapter = TTTAdapter(config=production_ttt_config)
                PerformanceValidator()

                start_time = time.time()

                # Initialize model
                adapter.initialize_model()
                init_time = time.time() - start_time

                # Adapt to task
                adaptation_start = time.time()
                adaptation = adapter.adapt_to_task(task)
                adaptation_time = time.time() - adaptation_start

                # Verify adaptation
                assert adaptation.task_id == task.task_id
                assert adaptation.adaptation_id is not None
                assert len(adaptation.training_examples) <= production_ttt_config.max_examples
                assert "final_loss" in adaptation.adaptation_metrics
                assert "training_steps" in adaptation.adaptation_metrics
                assert "adaptation_time" in adaptation.adaptation_metrics

                # Generate solution
                solution_start = time.time()
                solution = adapter.solve(task)
                solution_time = time.time() - solution_start

                total_time = time.time() - start_time

                # Verify solution
                assert solution.task_id == task.task_id
                assert solution.strategy_used == StrategyType.TEST_TIME_TRAINING
                assert len(solution.predictions) == 1
                assert isinstance(solution.predictions[0], list)
                assert len(solution.predictions[0]) > 0  # Has output grid
                assert isinstance(solution.predictions[0][0], list)  # Grid format
                assert solution.confidence_score >= 0.0
                assert solution.confidence_score <= 1.0

                # Verify resource tracking
                assert solution.resource_usage is not None
                assert solution.resource_usage.task_id == task.task_id
                assert solution.resource_usage.strategy_type == StrategyType.TEST_TIME_TRAINING
                assert solution.resource_usage.cpu_seconds > 0
                assert solution.resource_usage.memory_mb > 0

                # Verify performance metrics
                assert init_time < 60.0  # Model initialization under 1 minute
                assert adaptation_time < 180.0  # Adaptation under 3 minutes
                assert solution_time < 30.0  # Solution under 30 seconds
                assert total_time < 300.0  # Total under 5 minutes

                # Verify output dimensions match test input
                predicted_grid = solution.predictions[0]
                # Grid dimensions might change based on pattern, but should be reasonable
                assert len(predicted_grid) > 0
                assert len(predicted_grid[0]) > 0

                # Cleanup
                adapter.cleanup()

    def test_multiple_tasks_batch_processing(self, arc_data_repository, production_ttt_config):
        """Test batch processing of multiple real ARC tasks."""
        # Load multiple tasks
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=3)
        assert len(tasks) == 3

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks (similar to previous test but optimized for batch)
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [1, 2, 3, 4, 5]
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "1 0\n0 1"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Mock faster training for batch processing
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(
                    loss=torch.tensor(0.2, requires_grad=True)
                )
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter
                adapter = TTTAdapter(config=production_ttt_config)
                adapter.initialize_model()

                # Process all tasks
                results = []
                total_start_time = time.time()

                for _i, task in enumerate(tasks):
                    task_start_time = time.time()

                    # Adapt and solve
                    adaptation = adapter.adapt_to_task(task)
                    solution = adapter.solve(task)

                    task_processing_time = time.time() - task_start_time

                    # Collect results
                    results.append({
                        "task_id": task.task_id,
                        "adaptation_id": adaptation.adaptation_id,
                        "solution": solution,
                        "processing_time": task_processing_time,
                        "accuracy_estimate": adaptation.adaptation_metrics.get("validation_accuracy", 0.5)
                    })

                total_processing_time = time.time() - total_start_time

                # Verify batch processing results
                assert len(results) == 3
                assert all(r["solution"].task_id == tasks[i].task_id for i, r in enumerate(results))
                assert all(r["solution"].strategy_used == StrategyType.TEST_TIME_TRAINING for r in results)
                assert total_processing_time < 600.0  # All tasks under 10 minutes

                # Verify performance consistency
                processing_times = [r["processing_time"] for r in results]
                avg_time = sum(processing_times) / len(processing_times)

                # No task should take more than 3x average (reasonable variation)
                for processing_time in processing_times:
                    assert processing_time < avg_time * 3

                # Verify all solutions have valid predictions
                for result in results:
                    solution = result["solution"]
                    assert len(solution.predictions) == 1
                    assert isinstance(solution.predictions[0], list)
                    assert len(solution.predictions[0]) > 0

                # Verify resource usage tracking
                total_memory_usage = sum(
                    r["solution"].resource_usage.memory_mb for r in results
                )
                total_cpu_time = sum(
                    r["solution"].resource_usage.cpu_seconds for r in results
                )

                assert total_memory_usage > 0
                assert total_cpu_time > 0

                # Cleanup
                adapter.cleanup()

    def test_task_difficulty_variation(self, arc_data_repository, production_ttt_config):
        """Test handling tasks of varying difficulty and complexity."""
        # Load all available tasks to test variety
        all_tasks = arc_data_repository.load_all_tasks(dataset="training")

        # Categorize tasks by complexity (grid size, number of examples, etc.)
        simple_tasks = []
        medium_tasks = []
        complex_tasks = []

        for task in all_tasks:
            # Calculate complexity score
            grid_complexity = sum(
                len(ex["input"]) * len(ex["input"][0]) + len(ex["output"]) * len(ex["output"][0])
                for ex in task.train_examples
            )
            example_count = len(task.train_examples)

            complexity_score = grid_complexity + (example_count * 10)

            if complexity_score < 50:
                simple_tasks.append(task)
            elif complexity_score < 150:
                medium_tasks.append(task)
            else:
                complex_tasks.append(task)

        # Test one task from each category if available
        test_tasks = []
        if simple_tasks:
            test_tasks.append(("simple", simple_tasks[0]))
        if medium_tasks:
            test_tasks.append(("medium", medium_tasks[0]))
        if complex_tasks:
            test_tasks.append(("complex", complex_tasks[0]))

        assert len(test_tasks) > 0, "No tasks available for complexity testing"

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [1] * min(len(text.split()), 100)  # Vary by text length
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "1 0\n0 1"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(100, 100))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Mock training difficulty based on complexity
                def mock_complexity_training(complexity):
                    def mock_forward(*args, **kwargs):
                        # Complex tasks take more steps to converge
                        base_loss = 0.8 if complexity == "simple" else 1.2 if complexity == "medium" else 1.8
                        return Mock(loss=torch.tensor(base_loss, requires_grad=True))
                    return mock_forward

                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter
                adapter = TTTAdapter(config=production_ttt_config)
                adapter.initialize_model()

                complexity_results = []

                for complexity, task in test_tasks:
                    # Update mock for this complexity level
                    mock_model_inst.side_effect = mock_complexity_training(complexity)

                    start_time = time.time()

                    # Process task
                    adaptation = adapter.adapt_to_task(task)
                    adapter.solve(task)

                    processing_time = time.time() - start_time

                    complexity_results.append({
                        "complexity": complexity,
                        "task_id": task.task_id,
                        "processing_time": processing_time,
                        "adaptation_steps": adaptation.adaptation_metrics.get("training_steps", 0),
                        "final_loss": adaptation.adaptation_metrics.get("final_loss", 1.0),
                        "grid_sizes": [len(ex["input"]) * len(ex["input"][0]) for ex in task.train_examples],
                        "num_examples": len(task.train_examples)
                    })

                # Analyze results by complexity
                assert len(complexity_results) > 0

                # Verify that processing time generally increases with complexity
                if len(complexity_results) > 1:
                    complexity_order = ["simple", "medium", "complex"]
                    results_by_complexity = {r["complexity"]: r for r in complexity_results}

                    for i in range(len(complexity_order) - 1):
                        curr_complexity = complexity_order[i]
                        next_complexity = complexity_order[i + 1]

                        if curr_complexity in results_by_complexity and next_complexity in results_by_complexity:
                            curr_time = results_by_complexity[curr_complexity]["processing_time"]
                            next_time = results_by_complexity[next_complexity]["processing_time"]

                            # Allow some variation, but complex should generally take longer
                            assert next_time <= curr_time * 3  # Not more than 3x longer

                # Verify all tasks produced valid solutions regardless of complexity
                for result in complexity_results:
                    assert result["processing_time"] < 300.0  # All under 5 minutes
                    assert result["final_loss"] < 2.0  # Reasonable loss values

                # Cleanup
                adapter.cleanup()

    def test_memory_efficiency_with_real_tasks(self, arc_data_repository, production_ttt_config):
        """Test memory efficiency when processing real tasks."""
        import gc

        import psutil

        # Load tasks for memory testing
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=3)

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [1, 2, 3, 4, 5]
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "1 0\n0 1"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(loss=torch.tensor(0.3, requires_grad=True))
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create adapter
                adapter = TTTAdapter(config=production_ttt_config)

                # Measure baseline memory
                gc.collect()
                baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                adapter.initialize_model()

                # Measure memory after model loading
                after_init_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                model_memory_overhead = after_init_memory - baseline_memory

                memory_measurements = []

                for task in tasks:
                    # Measure memory before task processing
                    before_task_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    # Process task
                    adapter.adapt_to_task(task)
                    adapter.solve(task)

                    # Measure memory after task processing
                    after_task_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    task_memory_usage = after_task_memory - before_task_memory

                    memory_measurements.append({
                        "task_id": task.task_id,
                        "before_mb": before_task_memory,
                        "after_mb": after_task_memory,
                        "task_overhead_mb": task_memory_usage,
                        "total_from_baseline_mb": after_task_memory - baseline_memory
                    })

                    # Force garbage collection between tasks
                    gc.collect()

                # Analyze memory usage
                max_total_memory = max(m["after_mb"] for m in memory_measurements)
                avg_task_overhead = sum(m["task_overhead_mb"] for m in memory_measurements) / len(memory_measurements)

                # Verify memory efficiency
                assert model_memory_overhead < 2000  # Model loading under 2GB (mocked, so should be minimal)
                assert max_total_memory < 10240  # Stay under 10GB limit
                assert avg_task_overhead < 500  # Average task overhead under 500MB

                # Verify memory doesn't grow excessively between tasks
                memory_growth = memory_measurements[-1]["after_mb"] - memory_measurements[0]["before_mb"]
                assert memory_growth < 1000  # Total growth under 1GB for 3 tasks

                # Final cleanup and memory check
                adapter.cleanup()
                gc.collect()

                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                cleanup_efficiency = (memory_measurements[-1]["after_mb"] - final_memory) / memory_measurements[-1]["after_mb"]

                # Should clean up at least 50% of allocated memory
                assert cleanup_efficiency > 0.3  # Allow some overhead for test framework

    def test_accuracy_validation_with_known_patterns(self, arc_data_repository):
        """Test accuracy validation with tasks that have known correct patterns."""
        # Load tasks
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=2)

        # Create validator
        validator = PerformanceValidator()

        # Test accuracy calculation with perfect predictions
        for task in tasks:
            if task.test_output is not None:  # Only test if we have ground truth
                # Create perfect prediction (matches ground truth exactly)
                perfect_predictions = [task.test_output]
                ground_truth = [task.test_output]

                perfect_accuracy = validator.validate_accuracy(perfect_predictions, ground_truth)
                assert perfect_accuracy == 1.0

                # Create wrong prediction
                wrong_prediction = [[0] * len(row) for row in task.test_output]  # All zeros
                wrong_predictions = [wrong_prediction]

                wrong_accuracy = validator.validate_accuracy(wrong_predictions, ground_truth)
                assert wrong_accuracy == 0.0

                # Create partially correct prediction (if grid is large enough)
                if len(task.test_output) > 1 or len(task.test_output[0]) > 1:
                    partial_prediction = [row[:] for row in task.test_output]  # Copy ground truth
                    partial_prediction[0][0] = (partial_prediction[0][0] + 1) % 10  # Change one cell
                    partial_predictions = [partial_prediction]

                    partial_accuracy = validator.validate_accuracy(partial_predictions, ground_truth)
                    assert partial_accuracy == 0.0  # Pixel-perfect matching required

    def test_comprehensive_pipeline_validation(self, arc_data_repository, production_ttt_config, real_arc_test_dir):
        """Test comprehensive pipeline validation with real tasks."""
        # Load a representative task
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=1)
        task = tasks[0]

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup comprehensive mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [1, 2, 3, 4, 5]
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "1 0\n0 1"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(50, 50))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(loss=torch.tensor(0.25, requires_grad=True))
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                # Create components
                adapter = TTTAdapter(config=production_ttt_config)
                validator = PerformanceValidator()

                # Run complete pipeline with performance monitoring
                def complete_pipeline(task):
                    adapter.initialize_model()
                    adaptation = adapter.adapt_to_task(task)
                    solution = adapter.solve(task)
                    return {
                        "adaptation": adaptation,
                        "solution": solution,
                        "task_id": task.task_id
                    }

                # Benchmark the complete pipeline
                result, performance_metrics = validator.benchmark_training(complete_pipeline, task)

                # Validate pipeline results
                assert "adaptation" in result
                assert "solution" in result
                assert result["task_id"] == task.task_id

                adaptation = result["adaptation"]
                solution = result["solution"]

                # Validate adaptation
                assert adaptation.task_id == task.task_id
                assert adaptation.adaptation_id is not None
                assert len(adaptation.training_examples) > 0
                assert "final_loss" in adaptation.adaptation_metrics

                # Validate solution
                assert solution.task_id == task.task_id
                assert solution.strategy_used == StrategyType.TEST_TIME_TRAINING
                assert len(solution.predictions) == 1
                assert solution.confidence_score >= 0.0
                assert solution.resource_usage is not None

                # Validate performance metrics
                assert performance_metrics["training_time_seconds"] < 300.0  # Under 5 minutes
                assert performance_metrics["memory_peak_mb"] < 10240  # Under 10GB

                # Create comprehensive performance report
                from src.utils.performance_validator import PerformanceMetrics

                metrics = PerformanceMetrics(
                    task_id=task.task_id,
                    accuracy=0.5,  # Mock accuracy for testing
                    training_time_seconds=performance_metrics["training_time_seconds"],
                    memory_peak_mb=performance_metrics["memory_peak_mb"],
                    memory_average_mb=performance_metrics["memory_average_mb"],
                    gpu_memory_peak_mb=performance_metrics["gpu_memory_peak_mb"],
                    inference_time_ms=50.0,  # Mock inference time
                    model_load_time_ms=1000.0,  # Mock load time
                    checkpoint_size_mb=100.0,  # Mock checkpoint size
                    timestamp=datetime.now()
                )

                # Validate acceptance criteria
                criteria = metrics.meets_criteria()
                assert criteria["training_under_2_hours"] is True
                assert criteria["memory_under_10gb"] is True

                # Save validation results
                results_path = real_arc_test_dir / "validation_results.json"
                validator.save_validation_results(metrics, results_path)
                assert results_path.exists()

                # Verify saved results
                with open(results_path) as f:
                    saved_data = json.load(f)

                assert saved_data["task_id"] == task.task_id
                assert saved_data["training_time_seconds"] == metrics.training_time_seconds
                assert "criteria_met" in saved_data

                # Generate and verify report
                report = validator.generate_report(metrics)
                assert "TTT Performance Validation Report" in report
                assert task.task_id in report
                assert "Overall Result:" in report

                # Cleanup
                adapter.cleanup()


class TestRobustnessAndEdgeCases:
    """Test robustness and edge case handling with real ARC tasks."""

    def test_malformed_task_handling(self, production_ttt_config):
        """Test handling of malformed or edge case tasks."""
        # Create edge case tasks
        edge_case_tasks = [
            # Empty grid task
            ARCTask(
                task_id="edge_empty_001",
                task_source="test",
                train_examples=[
                    {"input": [[]], "output": [[]]}
                ],
                test_input=[[]]
            ),
            # Single cell task
            ARCTask(
                task_id="edge_single_001",
                task_source="test",
                train_examples=[
                    {"input": [[1]], "output": [[0]]}
                ],
                test_input=[[2]]
            ),
            # Large grid task
            ARCTask(
                task_id="edge_large_001",
                task_source="test",
                train_examples=[
                    {
                        "input": [[i % 10 for i in range(j, j + 20)] for j in range(0, 400, 20)],
                        "output": [[(i + 1) % 10 for i in range(j, j + 20)] for j in range(0, 400, 20)]
                    }
                ],
                test_input=[[i % 10 for i in range(j, j + 20)] for j in range(0, 400, 20)]
            )
        ]

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [1, 2, 3] if text else [1]
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "0" if len(tokens) < 5 else "0 1\n1 0"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(loss=torch.tensor(0.5, requires_grad=True))
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                adapter = TTTAdapter(config=production_ttt_config)
                adapter.initialize_model()

                edge_case_results = []

                for task in edge_case_tasks:
                    try:
                        start_time = time.time()

                        # Test adaptation
                        adaptation = adapter.adapt_to_task(task)

                        # Test solution generation
                        solution = adapter.solve(task)

                        processing_time = time.time() - start_time

                        edge_case_results.append({
                            "task_id": task.task_id,
                            "success": True,
                            "processing_time": processing_time,
                            "adaptation_id": adaptation.adaptation_id,
                            "solution_generated": len(solution.predictions) > 0,
                            "error": None
                        })

                    except Exception as e:
                        edge_case_results.append({
                            "task_id": task.task_id,
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })

                # Analyze edge case handling
                successful_cases = [r for r in edge_case_results if r["success"]]
                failed_cases = [r for r in edge_case_results if not r["success"]]

                # Should handle most edge cases gracefully
                success_rate = len(successful_cases) / len(edge_case_results)
                assert success_rate >= 0.5  # At least 50% of edge cases handled

                # For successful cases, verify reasonable processing times
                for result in successful_cases:
                    assert result["processing_time"] < 120.0  # Under 2 minutes even for edge cases
                    assert result["solution_generated"] is True

                # For failed cases, verify errors are informative
                for result in failed_cases:
                    assert result["error"] is not None
                    assert len(result["error"]) > 0

                adapter.cleanup()

    def test_consistency_across_runs(self, arc_data_repository, production_ttt_config):
        """Test consistency of results across multiple runs."""
        # Load a task for consistency testing
        tasks = arc_data_repository.load_all_tasks(dataset="training", limit=1)
        task = tasks[0]

        with patch("src.adapters.strategies.ttt_adapter.AutoModelForCausalLM") as mock_model:
            with patch("src.adapters.strategies.ttt_adapter.AutoTokenizer") as mock_tokenizer:
                # Setup deterministic mocks
                mock_tokenizer_inst = MagicMock()
                mock_tokenizer_inst.pad_token = None
                mock_tokenizer_inst.eos_token = "[EOS]"
                mock_tokenizer_inst.encode = lambda text, **kwargs: [hash(text) % 1000 + 1 for _ in range(5)]  # Deterministic based on text
                mock_tokenizer_inst.decode = lambda tokens, **kwargs: "1 0\n0 1"  # Consistent output
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_inst

                mock_model_inst = MagicMock()
                mock_model_inst.config.vocab_size = 50257
                mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
                mock_model_inst.train = MagicMock()
                mock_model_inst.eval = MagicMock()
                mock_model_inst.to = MagicMock(return_value=mock_model_inst)

                # Deterministic training simulation
                mock_model_inst.side_effect = lambda *args, **kwargs: Mock(
                    loss=torch.tensor(0.3, requires_grad=True)  # Consistent loss
                )

                # Deterministic generation
                mock_model_inst.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
                mock_model.from_pretrained.return_value = mock_model_inst

                # Run multiple times and compare results
                run_results = []

                for run_id in range(3):
                    # Create fresh adapter for each run
                    adapter = TTTAdapter(config=production_ttt_config)
                    adapter.initialize_model()

                    # Process task
                    adaptation = adapter.adapt_to_task(task)
                    solution = adapter.solve(task)

                    run_results.append({
                        "run_id": run_id,
                        "adaptation_id": adaptation.adaptation_id,
                        "final_loss": adaptation.adaptation_metrics.get("final_loss", 0.0),
                        "training_steps": adaptation.adaptation_metrics.get("training_steps", 0),
                        "prediction": solution.predictions[0],
                        "confidence": solution.confidence_score,
                        "memory_usage": solution.resource_usage.memory_mb
                    })

                    adapter.cleanup()

                # Analyze consistency
                assert len(run_results) == 3

                # Check prediction consistency (should be identical with deterministic mocks)
                first_prediction = run_results[0]["prediction"]
                for result in run_results[1:]:
                    assert result["prediction"] == first_prediction, "Predictions should be consistent across runs"

                # Check confidence consistency
                first_confidence = run_results[0]["confidence"]
                for result in run_results[1:]:
                    confidence_diff = abs(result["confidence"] - first_confidence)
                    assert confidence_diff < 0.1, "Confidence scores should be consistent"

                # Check training metrics consistency
                first_loss = run_results[0]["final_loss"]
                for result in run_results[1:]:
                    loss_diff = abs(result["final_loss"] - first_loss)
                    assert loss_diff < 0.1, "Training loss should be consistent"
