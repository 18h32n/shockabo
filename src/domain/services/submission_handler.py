"""
Unified submission handler for ARC task solutions.

This module provides a centralized system for handling submissions from
different strategies, ensuring consistent format and tracking.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.domain.evaluation_models import StrategyType
from src.domain.services.evaluation_service import EvaluationResult

logger = structlog.get_logger(__name__)


@dataclass
class UnifiedSubmission:
    """
    Unified submission format for ARC solutions.

    This format is compatible with competition requirements and
    supports multiple attempts per task.
    """

    task_id: str
    predictions: list[list[list[int]]]  # Up to 2 predictions
    confidence_scores: list[float]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate submission format."""
        if len(self.predictions) != len(self.confidence_scores):
            raise ValueError("Number of predictions must match confidence scores")

        if len(self.predictions) > 2:
            raise ValueError("Maximum 2 predictions allowed per task")

        if not all(0.0 <= score <= 1.0 for score in self.confidence_scores):
            raise ValueError("Confidence scores must be between 0 and 1")

    def to_competition_format(self) -> dict[str, Any]:
        """
        Convert to ARC competition submission format.

        Returns:
            Dictionary in competition-required format
        """
        # Competition format expects: {task_id: [prediction1, prediction2]}
        return {
            self.task_id: self.predictions
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Complete submission data as dictionary
        """
        return {
            "task_id": self.task_id,
            "predictions": self.predictions,
            "confidence_scores": self.confidence_scores,
            "strategy": self.strategy,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class SubmissionHandler:
    """
    Handles unified submissions from all strategies.

    Provides functionality for creating, validating, storing, and
    exporting submissions in various formats.
    """

    def __init__(self, output_dir: str | Path = "submissions"):
        """
        Initialize submission handler.

        Args:
            output_dir: Directory for storing submissions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = structlog.get_logger(__name__).bind(component="submission_handler")

        # Track submissions
        self.submissions: dict[str, UnifiedSubmission] = {}
        self.submission_history: list[UnifiedSubmission] = []

    def create_submission_from_evaluation(
        self,
        evaluation_result: EvaluationResult
    ) -> UnifiedSubmission:
        """
        Create unified submission from evaluation result.

        Args:
            evaluation_result: Result from evaluation service

        Returns:
            UnifiedSubmission object
        """
        # Extract predictions and confidence scores from attempts
        predictions = []
        confidence_scores = []

        for attempt in evaluation_result.attempts[:2]:  # Max 2 attempts
            # Get predicted output from attempt
            # Note: This assumes the predicted output is stored somewhere in the attempt
            # In practice, this would need to be passed along with the evaluation
            predicted_output = attempt.error_details.get('predicted_output')
            if predicted_output is None:
                # Skip attempts without predictions
                continue

            predictions.append(predicted_output)
            confidence_scores.append(attempt.confidence_score)

        # If no valid predictions, create empty submission
        if not predictions:
            self.logger.warning(
                "no_valid_predictions",
                task_id=evaluation_result.task_id,
                strategy=evaluation_result.strategy_used
            )
            predictions = [[[]]]  # Empty grid
            confidence_scores = [0.0]

        # Create unified submission
        submission = UnifiedSubmission(
            task_id=evaluation_result.task_id,
            predictions=predictions,
            confidence_scores=confidence_scores,
            strategy=evaluation_result.strategy_used,
            metadata={
                "evaluation_metadata": evaluation_result.metadata,
                "final_accuracy": evaluation_result.final_accuracy,
                "total_processing_time_ms": evaluation_result.total_processing_time_ms,
                "attempts_count": len(evaluation_result.attempts)
            }
        )

        # Store submission
        self.submissions[evaluation_result.task_id] = submission
        self.submission_history.append(submission)

        self.logger.info(
            "submission_created",
            task_id=submission.task_id,
            strategy=submission.strategy,
            predictions_count=len(predictions),
            confidence_scores=confidence_scores
        )

        return submission

    def create_submission(
        self,
        task_id: str,
        predictions: list[list[list[int]]],
        confidence_scores: list[float],
        strategy: str | StrategyType,
        metadata: dict[str, Any] | None = None
    ) -> UnifiedSubmission:
        """
        Create a unified submission directly.

        Args:
            task_id: Task identifier
            predictions: List of predicted outputs (up to 2)
            confidence_scores: Confidence for each prediction
            strategy: Strategy used (name or enum)
            metadata: Optional metadata

        Returns:
            UnifiedSubmission object
        """
        # Convert strategy enum to string if needed
        if isinstance(strategy, StrategyType):
            strategy = strategy.value

        submission = UnifiedSubmission(
            task_id=task_id,
            predictions=predictions,
            confidence_scores=confidence_scores,
            strategy=strategy,
            metadata=metadata or {}
        )

        # Store submission
        self.submissions[task_id] = submission
        self.submission_history.append(submission)

        return submission

    def get_submission(self, task_id: str) -> UnifiedSubmission | None:
        """
        Get submission for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            UnifiedSubmission or None if not found
        """
        return self.submissions.get(task_id)

    def export_competition_format(
        self,
        filename: str = "submission.json",
        task_ids: list[str] | None = None
    ) -> Path:
        """
        Export submissions in ARC competition format.

        Args:
            filename: Output filename
            task_ids: Specific task IDs to export (None for all)

        Returns:
            Path to exported file
        """
        # Collect submissions to export
        if task_ids:
            submissions_to_export = {
                tid: self.submissions[tid]
                for tid in task_ids
                if tid in self.submissions
            }
        else:
            submissions_to_export = self.submissions

        # Convert to competition format
        competition_data = {}
        for _task_id, submission in submissions_to_export.items():
            competition_data.update(submission.to_competition_format())

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(competition_data, f, indent=2)

        self.logger.info(
            "exported_competition_format",
            filename=filename,
            tasks_count=len(competition_data),
            output_path=str(output_path)
        )

        return output_path

    def export_detailed_format(
        self,
        filename: str = "submission_detailed.json",
        task_ids: list[str] | None = None
    ) -> Path:
        """
        Export submissions with full details and metadata.

        Args:
            filename: Output filename
            task_ids: Specific task IDs to export (None for all)

        Returns:
            Path to exported file
        """
        # Collect submissions to export
        if task_ids:
            submissions_to_export = [
                self.submissions[tid].to_dict()
                for tid in task_ids
                if tid in self.submissions
            ]
        else:
            submissions_to_export = [
                sub.to_dict() for sub in self.submissions.values()
            ]

        # Add summary statistics
        export_data = {
            "submissions": submissions_to_export,
            "summary": self.get_submission_summary(),
            "export_timestamp": datetime.now().isoformat()
        }

        # Write to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(
            "exported_detailed_format",
            filename=filename,
            tasks_count=len(submissions_to_export),
            output_path=str(output_path)
        )

        return output_path

    def get_submission_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for all submissions.

        Returns:
            Dictionary with summary statistics
        """
        if not self.submissions:
            return {
                "total_tasks": 0,
                "strategies_used": [],
                "average_confidence": 0.0,
                "predictions_per_task": {}
            }

        # Calculate statistics
        all_confidences = []
        strategies = set()
        predictions_count = {}

        for submission in self.submissions.values():
            all_confidences.extend(submission.confidence_scores)
            strategies.add(submission.strategy)
            predictions_count[len(submission.predictions)] = (
                predictions_count.get(len(submission.predictions), 0) + 1
            )

        return {
            "total_tasks": len(self.submissions),
            "strategies_used": sorted(strategies),
            "average_confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
            "predictions_per_task": predictions_count,
            "submission_history_length": len(self.submission_history)
        }

    def clear_submissions(self) -> None:
        """Clear all stored submissions."""
        self.submissions.clear()
        self.submission_history.clear()
        self.logger.info("submissions_cleared")
