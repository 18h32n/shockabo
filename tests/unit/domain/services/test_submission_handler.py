"""Unit tests for SubmissionHandler."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.domain.evaluation_models import EvaluationResult, PerformanceMetrics
from src.domain.services.submission_handler import SubmissionHandler


class TestSubmissionHandler:
    """Test suite for SubmissionHandler."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.submission = Mock()
        config.submission.output_path = "submissions"
        config.submission.format = "json"
        config.submission.include_metadata = True
        config.submission.include_programs = True
        config.submission.max_programs_per_task = 10
        config.submission.min_confidence_threshold = 0.5
        config.submission.compression_enabled = True
        return config

    @pytest.fixture
    def handler(self, mock_config):
        """Create submission handler instance."""
        return SubmissionHandler(mock_config)

    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample evaluation results."""
        results = []

        # Successful result
        result1 = EvaluationResult(
            task_id="task_001",
            strategy="evolution",
            success=True,
            solution=[[1, 2], [3, 4]],
            confidence=0.95,
            programs=[
                {"operations": [{"op": "flip", "axis": "horizontal"}], "fitness": 0.95},
                {"operations": [{"op": "rotate", "k": 90}], "fitness": 0.90}
            ],
            metadata={
                "generations": 50,
                "best_fitness": 0.95,
                "mutation_success_rate": 0.3
            },
            performance=PerformanceMetrics(
                total_time=120.5,
                inference_time=0.5,
                programs_evaluated=5000,
                memory_peak_mb=512.0,
                evolution_generations=50,
                strategy="evolution"
            )
        )

        # Failed result
        result2 = EvaluationResult(
            task_id="task_002",
            strategy="ttt",
            success=False,
            confidence=0.3,
            error="No solution found",
            metadata={"attempts": 100}
        )

        # Another successful result
        result3 = EvaluationResult(
            task_id="task_003",
            strategy="synthesis",
            success=True,
            solution=[[5, 6], [7, 8]],
            confidence=0.85,
            programs=[{"operations": [{"op": "identity"}], "score": 1.0}],
            metadata={"llm_model": "gpt-4"}
        )

        results.extend([result1, result2, result3])
        return results

    def test_initialization(self, mock_config):
        """Test handler initialization."""
        handler = SubmissionHandler(mock_config)
        assert handler.config == mock_config
        assert handler.submission_data == {}

    def test_add_result(self, handler, sample_evaluation_results):
        """Test adding evaluation results."""
        result = sample_evaluation_results[0]

        handler.add_result(result)

        assert "task_001" in handler.submission_data
        submission = handler.submission_data["task_001"]
        assert submission["solution"] == [[1, 2], [3, 4]]
        assert submission["confidence"] == 0.95
        assert submission["strategy"] == "evolution"
        assert len(submission["programs"]) == 2

    def test_add_result_below_threshold(self, handler, sample_evaluation_results):
        """Test adding result below confidence threshold."""
        result = sample_evaluation_results[1]  # Failed result with low confidence

        handler.add_result(result)

        # Should not be added due to low confidence
        assert "task_002" not in handler.submission_data

    def test_add_result_override_lower_confidence(self, handler, sample_evaluation_results):
        """Test that higher confidence results override lower ones."""
        # Add result with lower confidence
        low_conf_result = EvaluationResult(
            task_id="task_001",
            strategy="ttt",
            success=True,
            solution=[[0, 0], [0, 0]],
            confidence=0.7
        )
        handler.add_result(low_conf_result)

        # Add result with higher confidence
        high_conf_result = sample_evaluation_results[0]  # confidence 0.95
        handler.add_result(high_conf_result)

        # Higher confidence should override
        assert handler.submission_data["task_001"]["confidence"] == 0.95
        assert handler.submission_data["task_001"]["strategy"] == "evolution"

    def test_create_submission_json(self, handler, sample_evaluation_results):
        """Test creating JSON submission."""
        # Add results
        for result in sample_evaluation_results:
            handler.add_result(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.config.submission.output_path = temp_dir

            # Create submission
            filepath = handler.create_submission("test_submission")

            assert filepath.exists()
            assert filepath.suffix == ".json"

            # Load and verify content
            with open(filepath) as f:
                submission = json.load(f)

            assert len(submission) == 2  # Only successful results above threshold
            assert "task_001" in submission
            assert "task_003" in submission
            assert submission["task_001"]["confidence"] == 0.95
            assert submission["task_003"]["confidence"] == 0.85

    def test_create_submission_without_metadata(self, handler, sample_evaluation_results):
        """Test creating submission without metadata."""
        handler.config.submission.include_metadata = False

        # Add result
        handler.add_result(sample_evaluation_results[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.config.submission.output_path = temp_dir

            filepath = handler.create_submission("test")

            with open(filepath) as f:
                submission = json.load(f)

            # Should not have metadata
            assert "metadata" not in submission["task_001"]
            assert "performance" not in submission["task_001"]

    def test_create_submission_without_programs(self, handler, sample_evaluation_results):
        """Test creating submission without programs."""
        handler.config.submission.include_programs = False

        # Add result
        handler.add_result(sample_evaluation_results[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.config.submission.output_path = temp_dir

            filepath = handler.create_submission("test")

            with open(filepath) as f:
                submission = json.load(f)

            # Should not have programs
            assert "programs" not in submission["task_001"]

    def test_create_submission_with_compression(self, handler, sample_evaluation_results):
        """Test creating compressed submission."""
        handler.config.submission.compression_enabled = True

        # Add all results
        for result in sample_evaluation_results:
            handler.add_result(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.config.submission.output_path = temp_dir

            filepath = handler.create_submission("compressed_test")

            assert filepath.exists()
            assert filepath.suffix == ".gz"

            # Verify compressed file can be read
            import gzip
            with gzip.open(filepath, 'rt') as f:
                submission = json.load(f)

            assert len(submission) == 2

    def test_max_programs_per_task(self, handler):
        """Test limiting programs per task."""
        handler.config.submission.max_programs_per_task = 3

        # Create result with many programs
        programs = [{"operations": [{"op": f"op_{i}"}], "fitness": 1.0 - i*0.1}
                   for i in range(10)]

        result = EvaluationResult(
            task_id="task_many_programs",
            strategy="evolution",
            success=True,
            solution=[[1, 2]],
            confidence=0.9,
            programs=programs
        )

        handler.add_result(result)

        # Should only keep top 3 programs
        assert len(handler.submission_data["task_many_programs"]["programs"]) == 3
        # Should be sorted by fitness (descending)
        assert all(handler.submission_data["task_many_programs"]["programs"][i]["fitness"] >=
                  handler.submission_data["task_many_programs"]["programs"][i+1]["fitness"]
                  for i in range(2))

    def test_get_submission_summary(self, handler, sample_evaluation_results):
        """Test getting submission summary."""
        # Add results
        for result in sample_evaluation_results:
            handler.add_result(result)

        summary = handler.get_submission_summary()

        assert summary["total_tasks"] == 2
        assert summary["average_confidence"] == 0.9  # (0.95 + 0.85) / 2
        assert summary["strategies_used"] == {"evolution", "synthesis"}
        assert summary["tasks_by_strategy"]["evolution"] == 1
        assert summary["tasks_by_strategy"]["synthesis"] == 1

    def test_clear_submission_data(self, handler, sample_evaluation_results):
        """Test clearing submission data."""
        # Add results
        handler.add_result(sample_evaluation_results[0])
        assert len(handler.submission_data) == 1

        # Clear
        handler.clear()
        assert len(handler.submission_data) == 0

    def test_validate_submission(self, handler):
        """Test submission validation."""
        # Empty submission should fail
        assert handler.validate_submission() is False

        # Add valid result
        result = EvaluationResult(
            task_id="task_001",
            strategy="evolution",
            success=True,
            solution=[[1, 2], [3, 4]],
            confidence=0.9
        )
        handler.add_result(result)

        # Should now be valid
        assert handler.validate_submission() is True

    def test_merge_results(self, handler):
        """Test merging results from multiple sources."""
        # Add initial result
        result1 = EvaluationResult(
            task_id="task_001",
            strategy="evolution",
            success=True,
            solution=[[1, 2]],
            confidence=0.8,
            metadata={"source": "evolution"}
        )
        handler.add_result(result1)

        # Create another handler with different results
        handler2 = SubmissionHandler(handler.config)
        result2 = EvaluationResult(
            task_id="task_002",
            strategy="synthesis",
            success=True,
            solution=[[3, 4]],
            confidence=0.85
        )
        handler2.add_result(result2)

        # Also add a higher confidence result for task_001
        result3 = EvaluationResult(
            task_id="task_001",
            strategy="ttt",
            success=True,
            solution=[[5, 6]],
            confidence=0.9
        )
        handler2.add_result(result3)

        # Merge
        handler.merge_from(handler2)

        # Should have both tasks, with higher confidence for task_001
        assert len(handler.submission_data) == 2
        assert handler.submission_data["task_001"]["confidence"] == 0.9
        assert handler.submission_data["task_001"]["strategy"] == "ttt"
        assert handler.submission_data["task_002"]["confidence"] == 0.85

    def test_export_analysis_report(self, handler, sample_evaluation_results):
        """Test exporting analysis report."""
        # Add results
        for result in sample_evaluation_results:
            handler.add_result(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "analysis_report.json"

            handler.export_analysis_report(report_path)

            assert report_path.exists()

            with open(report_path) as f:
                report = json.load(f)

            assert "summary" in report
            assert "tasks" in report
            assert "performance_metrics" in report
            assert report["summary"]["total_tasks"] == 2
            assert "task_001" in report["tasks"]
            assert report["tasks"]["task_001"]["strategy"] == "evolution"

    def test_create_submission_with_custom_format(self, handler, sample_evaluation_results):
        """Test creating submission with custom formatting function."""
        # Add result
        handler.add_result(sample_evaluation_results[0])

        # Define custom formatter
        def custom_formatter(task_id, data):
            return {
                "id": task_id,
                "answer": data["solution"],
                "score": data["confidence"]
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.config.submission.output_path = temp_dir

            filepath = handler.create_submission("custom", formatter=custom_formatter)

            with open(filepath) as f:
                submission = json.load(f)

            # Check custom format
            assert "task_001" in submission
            assert "id" in submission["task_001"]
            assert "answer" in submission["task_001"]
            assert "score" in submission["task_001"]
            assert submission["task_001"]["score"] == 0.95
