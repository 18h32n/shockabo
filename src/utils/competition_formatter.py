"""Competition format export utilities for ARC Prize 2025 submissions.

This module provides functionality to format evaluation results into the exact
submission format required by the ARC Prize 2025 competition, including validation
and submission preparation utilities.
"""

import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import structlog

from src.domain.evaluation_models import ExperimentRun, TaskSubmission
from src.domain.services.evaluation_service import EvaluationResult

logger = structlog.get_logger(__name__)


class SubmissionValidator:
    """Validates submissions against competition requirements."""

    # Competition constants
    MAX_GRID_SIZE = 30
    VALID_COLORS = list(range(10))  # 0-9 are valid colors
    MAX_ATTEMPTS = 2

    @classmethod
    def validate_grid(cls, grid: list[list[int]], task_id: str) -> tuple[bool, str | None]:
        """Validate a single grid output.

        Args:
            grid: Output grid to validate
            task_id: Task ID for error reporting

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if grid is non-empty
        if not grid or not all(isinstance(row, list) for row in grid):
            return False, f"Task {task_id}: Invalid grid structure"

        # Check grid dimensions
        height = len(grid)
        if height > cls.MAX_GRID_SIZE:
            return False, f"Task {task_id}: Grid height {height} exceeds maximum {cls.MAX_GRID_SIZE}"

        # Check each row
        for i, row in enumerate(grid):
            if not row:
                return False, f"Task {task_id}: Row {i} is empty"

            width = len(row)
            if width > cls.MAX_GRID_SIZE:
                return False, f"Task {task_id}: Row {i} width {width} exceeds maximum {cls.MAX_GRID_SIZE}"

            # Check all values are valid colors
            for j, value in enumerate(row):
                if not isinstance(value, int) or value not in cls.VALID_COLORS:
                    return False, f"Task {task_id}: Invalid color {value} at position ({i},{j})"

        # Check all rows have same width
        widths = [len(row) for row in grid]
        if len(set(widths)) > 1:
            return False, f"Task {task_id}: Inconsistent row widths: {widths}"

        return True, None

    @classmethod
    def validate_submission(
        cls, task_id: str, attempts: list[list[list[int]]]
    ) -> tuple[bool, str | None]:
        """Validate a complete task submission.

        Args:
            task_id: Task ID
            attempts: List of attempt grids (1 or 2)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check number of attempts
        if not attempts:
            return False, f"Task {task_id}: No attempts provided"

        if len(attempts) > cls.MAX_ATTEMPTS:
            return False, f"Task {task_id}: Too many attempts ({len(attempts)}), maximum is {cls.MAX_ATTEMPTS}"

        # Validate each attempt
        for i, grid in enumerate(attempts):
            is_valid, error = cls.validate_grid(grid, task_id)
            if not is_valid:
                return False, f"Attempt {i+1}: {error}"

        return True, None


class CompetitionFormatter:
    """Formats evaluation results for competition submission."""

    def __init__(self):
        """Initialize the competition formatter."""
        self.logger = structlog.get_logger(__name__).bind(service="competition_formatter")
        self.validator = SubmissionValidator()

    def format_task_result(self, result: EvaluationResult) -> dict[str, Any]:
        """Format a single task evaluation result.

        Args:
            result: Evaluation result for a task

        Returns:
            Dictionary with task_id and attempts
        """
        attempts = []

        # Sort attempts by attempt number
        sorted_attempts = sorted(result.attempts, key=lambda x: x.attempt_number.value)

        # Extract predicted outputs
        for _attempt in sorted_attempts[:2]:  # Maximum 2 attempts
            # In production, we'd need to store the actual predictions
            # For now, we'll use a placeholder
            predicted_output = [[0]]  # This should be the actual prediction
            attempts.append(predicted_output)

        return {
            "task_id": result.task_id,
            "attempts": attempts,
            "metadata": {
                "strategy": result.strategy_used,
                "accuracy": result.final_accuracy,
                "processing_time_ms": result.total_processing_time_ms,
            },
        }

    def format_experiment_results(
        self, experiment: ExperimentRun, results: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Format experiment results for submission.

        Args:
            experiment: Experiment run details
            results: List of evaluation results

        Returns:
            Formatted submission dictionary
        """
        submission = {
            "experiment_id": experiment.run_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "experiment_name": experiment.experiment_name,
                "strategy": experiment.strategy_config.get("strategy", "unknown"),
                "duration_seconds": experiment.duration_seconds,
            },
            "predictions": [],
        }

        # Format each task result
        for result in results:
            task_data = self.format_task_result(result)
            submission["predictions"].append(task_data)

        return submission

    def export_to_csv(
        self,
        results: list[EvaluationResult],
        output_path: str | None = None,
        include_metadata: bool = False,
    ) -> str:
        """Export results to CSV format matching competition requirements.

        The CSV format is:
        - task_id: Task identifier
        - attempt_1: JSON-encoded first attempt grid
        - attempt_2: JSON-encoded second attempt grid (optional)

        Args:
            results: List of evaluation results
            output_path: Optional path to save CSV file
            include_metadata: Whether to include additional metadata columns

        Returns:
            CSV content as string
        """
        # Create CSV in memory
        output = StringIO()

        # Define field names
        fieldnames = ["task_id", "attempt_1", "attempt_2"]
        if include_metadata:
            fieldnames.extend(["strategy", "accuracy", "processing_time_ms"])

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Write each task result
        for result in results:
            row = {"task_id": result.task_id}

            # Get attempts (in production, we'd need the actual predictions)
            # For now, using placeholder data
            sorted_attempts = sorted(result.attempts, key=lambda x: x.attempt_number.value)

            if len(sorted_attempts) >= 1:
                row["attempt_1"] = json.dumps([[0]])  # Placeholder - should be actual prediction
            else:
                row["attempt_1"] = json.dumps([[0]])  # Default if no attempts

            if len(sorted_attempts) >= 2:
                row["attempt_2"] = json.dumps([[0]])  # Placeholder - should be actual prediction
            else:
                row["attempt_2"] = ""  # Empty for single attempt

            # Add metadata if requested
            if include_metadata:
                row["strategy"] = result.strategy_used
                row["accuracy"] = f"{result.final_accuracy:.4f}"
                row["processing_time_ms"] = f"{result.total_processing_time_ms:.1f}"

            writer.writerow(row)

        # Get CSV content
        csv_content = output.getvalue()
        output.close()

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="") as f:
                f.write(csv_content)
            self.logger.info("csv_exported", path=output_path, num_tasks=len(results))

        return csv_content

    def export_to_json(
        self,
        experiment: ExperimentRun,
        results: list[EvaluationResult],
        submissions: list[TaskSubmission],
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Export results to JSON format with full predictions.

        Args:
            experiment: Experiment run details
            results: List of evaluation results
            submissions: List of task submissions with predictions
            output_path: Optional path to save JSON file

        Returns:
            JSON-serializable dictionary
        """
        # Create submission dictionary
        submission_data = {
            "experiment_id": experiment.run_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "experiment_name": experiment.experiment_name,
                "strategy": experiment.strategy_config.get("strategy", "unknown"),
                "duration_seconds": experiment.duration_seconds,
                "total_tasks": len(results),
                "average_accuracy": sum(r.final_accuracy for r in results) / len(results) if results else 0.0,
            },
            "predictions": {},
        }

        # Group submissions by task ID
        task_submissions = {}
        for sub in submissions:
            if sub.task_id not in task_submissions:
                task_submissions[sub.task_id] = []
            task_submissions[sub.task_id].append(sub)

        # Format each task's predictions
        for task_id, subs in task_submissions.items():
            # Sort by submission time to get attempt order
            sorted_subs = sorted(subs, key=lambda x: x.submitted_at)

            attempts = []
            for i, sub in enumerate(sorted_subs[:2]):  # Max 2 attempts
                # Validate the prediction
                is_valid, error = self.validator.validate_grid(sub.predicted_output, task_id)
                if not is_valid:
                    self.logger.warning(
                        "invalid_prediction",
                        task_id=task_id,
                        attempt=i+1,
                        error=error,
                    )
                    continue

                attempts.append(sub.predicted_output)

            if attempts:
                submission_data["predictions"][task_id] = {
                    "attempts": attempts,
                    "strategy": sorted_subs[0].strategy_used.value,
                    "confidence_scores": [sub.confidence_score for sub in sorted_subs[:len(attempts)]],
                }

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(submission_data, f, indent=2)
            self.logger.info(
                "json_exported",
                path=output_path,
                num_tasks=len(submission_data["predictions"]),
            )

        return submission_data

    def validate_submission_file(self, file_path: str) -> tuple[bool, list[str]]:
        """Validate a submission file against competition requirements.

        Args:
            file_path: Path to submission file (CSV or JSON)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        file_path = Path(file_path)

        if not file_path.exists():
            return False, ["File does not exist"]

        try:
            if file_path.suffix.lower() == ".csv":
                errors.extend(self._validate_csv_file(file_path))
            elif file_path.suffix.lower() == ".json":
                errors.extend(self._validate_json_file(file_path))
            else:
                errors.append(f"Unsupported file format: {file_path.suffix}")

        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")

        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info("submission_validated", file_path=str(file_path))
        else:
            self.logger.error(
                "submission_validation_failed",
                file_path=str(file_path),
                num_errors=len(errors),
            )

        return is_valid, errors

    def _validate_csv_file(self, file_path: Path) -> list[str]:
        """Validate CSV submission file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of validation errors
        """
        errors = []

        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)

            # Check required columns
            if not reader.fieldnames:
                errors.append("CSV file has no headers")
                return errors

            required_fields = {"task_id", "attempt_1"}
            missing_fields = required_fields - set(reader.fieldnames)
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
                return errors

            # Validate each row
            for i, row in enumerate(reader):
                row_num = i + 2  # Account for header row

                # Check task_id
                if not row.get("task_id"):
                    errors.append(f"Row {row_num}: Missing task_id")
                    continue

                # Validate attempt_1
                try:
                    attempt_1 = json.loads(row["attempt_1"])
                    is_valid, error = self.validator.validate_grid(
                        attempt_1, f"{row['task_id']} (row {row_num})"
                    )
                    if not is_valid:
                        errors.append(f"Row {row_num}: {error}")
                except json.JSONDecodeError:
                    errors.append(f"Row {row_num}: Invalid JSON in attempt_1")
                except Exception as e:
                    errors.append(f"Row {row_num}: Error parsing attempt_1: {str(e)}")

                # Validate attempt_2 if present
                if row.get("attempt_2"):
                    try:
                        attempt_2 = json.loads(row["attempt_2"])
                        is_valid, error = self.validator.validate_grid(
                            attempt_2, f"{row['task_id']} (row {row_num})"
                        )
                        if not is_valid:
                            errors.append(f"Row {row_num}: {error}")
                    except json.JSONDecodeError:
                        errors.append(f"Row {row_num}: Invalid JSON in attempt_2")
                    except Exception as e:
                        errors.append(f"Row {row_num}: Error parsing attempt_2: {str(e)}")

        return errors

    def _validate_json_file(self, file_path: Path) -> list[str]:
        """Validate JSON submission file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of validation errors
        """
        errors = []

        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return errors

        # Check required fields
        if "predictions" not in data:
            errors.append("Missing 'predictions' field")
            return errors

        predictions = data["predictions"]
        if not isinstance(predictions, dict):
            errors.append("'predictions' must be a dictionary")
            return errors

        # Validate each task prediction
        for task_id, task_data in predictions.items():
            if not isinstance(task_data, dict):
                errors.append(f"Task {task_id}: Invalid data structure")
                continue

            if "attempts" not in task_data:
                errors.append(f"Task {task_id}: Missing 'attempts' field")
                continue

            attempts = task_data["attempts"]
            if not isinstance(attempts, list):
                errors.append(f"Task {task_id}: 'attempts' must be a list")
                continue

            # Validate submission
            is_valid, error = self.validator.validate_submission(task_id, attempts)
            if not is_valid:
                errors.append(error)

        return errors

    def prepare_submission(
        self,
        experiment_id: str,
        output_dir: str = "./submissions",
        format_type: str = "csv",
    ) -> tuple[str, dict[str, Any]]:
        """Prepare a complete submission package for the competition.

        Args:
            experiment_id: ID of the experiment to prepare
            output_dir: Directory to save submission files
            format_type: Format type ('csv' or 'json')

        Returns:
            Tuple of (submission_path, metadata)
        """
        # TODO: Load experiment and results from storage
        # For now, return placeholder

        output_path = Path(output_dir) / f"{experiment_id}_submission.{format_type}"
        metadata = {
            "experiment_id": experiment_id,
            "format": format_type,
            "created_at": datetime.now().isoformat(),
            "status": "prepared",
        }

        self.logger.info(
            "submission_prepared",
            experiment_id=experiment_id,
            output_path=str(output_path),
        )

        return str(output_path), metadata


# Utility functions for quick access
def export_results_to_csv(
    results: list[EvaluationResult],
    output_path: str,
    include_metadata: bool = False,
) -> None:
    """Export evaluation results to competition CSV format.

    Args:
        results: List of evaluation results
        output_path: Path to save CSV file
        include_metadata: Whether to include additional metadata columns
    """
    formatter = CompetitionFormatter()
    formatter.export_to_csv(results, output_path, include_metadata)


def validate_submission(file_path: str) -> tuple[bool, list[str]]:
    """Validate a submission file.

    Args:
        file_path: Path to submission file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    formatter = CompetitionFormatter()
    return formatter.validate_submission_file(file_path)
