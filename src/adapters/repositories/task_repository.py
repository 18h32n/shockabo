"""Repository for task storage and retrieval using the configured database.

This module provides database operations for ARC tasks, submissions,
and experiment data with proper abstraction and error handling.
"""

from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from src.domain.evaluation_models import ExperimentRun, StrategyType, TaskStatus, TaskSubmission
from src.domain.models import ARCTask

logger = structlog.get_logger(__name__)

Base = declarative_base()


class TaskRecord(Base):
    """Database model for ARC tasks."""

    __tablename__ = "arc_tasks"

    task_id = Column(String(100), primary_key=True)
    task_source = Column(String(50), nullable=False)
    train_examples = Column(JSON, nullable=False)
    test_input = Column(JSON, nullable=False)
    test_output = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SubmissionRecord(Base):
    """Database model for task submissions."""

    __tablename__ = "task_submissions"

    submission_id = Column(String(100), primary_key=True)
    task_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    predicted_output = Column(JSON, nullable=False)
    strategy_used = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    resource_usage = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=True)
    submitted_at = Column(DateTime, nullable=False)
    accuracy = Column(Float, nullable=True)
    perfect_match = Column(Integer, nullable=True)  # 0 or 1
    error_category = Column(String(50), nullable=True)


class ExperimentRecord(Base):
    """Database model for experiment runs."""

    __tablename__ = "experiment_runs"

    run_id = Column(String(100), primary_key=True)
    experiment_name = Column(String(200), nullable=False)
    task_ids = Column(JSON, nullable=False)  # List of task IDs
    strategy_config = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    error_log = Column(Text, nullable=True)
    user_id = Column(String(100), nullable=False)


class TaskRepository:
    """Repository for task-related database operations."""

    def __init__(self, database_url: str | None = None):
        """Initialize the repository.

        Args:
            database_url: Database connection URL (defaults to SQLite)
        """
        if not database_url:
            database_url = "sqlite:///./evaluation.db"

        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)

        logger.info("task_repository_initialized", database_url=database_url)

    def get_session(self) -> Session:
        """Get a database session.

        Returns:
            Database session
        """
        return self.SessionLocal()

    def save_task(self, task: ARCTask) -> bool:
        """Save an ARC task to the database.

        Args:
            task: ARC task to save

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            # Convert task to database record
            record = TaskRecord(
                task_id=task.task_id,
                task_source=task.task_source,
                train_examples=[
                    {"input": ex.input_grid, "output": ex.output_grid}
                    for ex in task.train_examples
                ],
                test_input=task.test_input,
                test_output=task.test_output,
                metadata=task.metadata or {}
            )

            # Save to database
            session.merge(record)  # Use merge to handle updates
            session.commit()

            logger.info("task_saved", task_id=task.task_id)
            return True

        except Exception as e:
            session.rollback()
            logger.error("task_save_failed", task_id=task.task_id, error=str(e))
            return False
        finally:
            session.close()

    def get_task(self, task_id: str) -> ARCTask | None:
        """Retrieve an ARC task from the database.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            ARC task if found, None otherwise
        """
        session = self.get_session()
        try:
            record = session.query(TaskRecord).filter_by(task_id=task_id).first()

            if not record:
                logger.warning("task_not_found", task_id=task_id)
                return None

            # Convert database record to domain model
            from src.domain.models import Example

            train_examples = [
                Example(
                    input_grid=ex["input"],
                    output_grid=ex["output"]
                )
                for ex in record.train_examples
            ]

            task = ARCTask(
                task_id=record.task_id,
                task_source=record.task_source,
                train_examples=train_examples,
                test_input=record.test_input,
                test_output=record.test_output,
                metadata=record.metadata
            )

            logger.info("task_retrieved", task_id=task_id)
            return task

        except Exception as e:
            logger.error("task_retrieval_failed", task_id=task_id, error=str(e))
            return None
        finally:
            session.close()

    def save_submission(self, submission: TaskSubmission, evaluation_result: Any) -> bool:
        """Save a task submission to the database.

        Args:
            submission: Task submission to save
            evaluation_result: Evaluation results

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            # Create submission record with evaluation results
            record = SubmissionRecord(
                submission_id=submission.submission_id,
                task_id=submission.task_id,
                user_id=submission.user_id,
                predicted_output=submission.predicted_output,
                strategy_used=submission.strategy_used.value,
                confidence_score=submission.confidence_score,
                processing_time_ms=submission.processing_time_ms,
                resource_usage=submission.resource_usage,
                metadata=submission.metadata,
                submitted_at=submission.submitted_at,
                accuracy=evaluation_result.pixel_accuracy.accuracy if evaluation_result else None,
                perfect_match=1 if evaluation_result and evaluation_result.pixel_accuracy.perfect_match else 0,
                error_category=evaluation_result.error_category.value if evaluation_result and evaluation_result.error_category else None
            )

            session.add(record)
            session.commit()

            logger.info("submission_saved", submission_id=submission.submission_id)
            return True

        except Exception as e:
            session.rollback()
            logger.error("submission_save_failed", submission_id=submission.submission_id, error=str(e))
            return False
        finally:
            session.close()

    def save_experiment(self, experiment: ExperimentRun) -> bool:
        """Save an experiment run to the database.

        Args:
            experiment: Experiment run to save

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            record = ExperimentRecord(
                run_id=experiment.run_id,
                experiment_name=experiment.experiment_name,
                task_ids=experiment.task_ids,
                strategy_config=experiment.strategy_config,
                metrics=experiment.metrics,
                status=experiment.status.value,
                started_at=experiment.started_at,
                completed_at=experiment.completed_at,
                error_log=experiment.error_log,
                user_id="system"  # TODO: Get from context
            )

            session.merge(record)
            session.commit()

            logger.info("experiment_saved", run_id=experiment.run_id)
            return True

        except Exception as e:
            session.rollback()
            logger.error("experiment_save_failed", run_id=experiment.run_id, error=str(e))
            return False
        finally:
            session.close()

    def get_experiment(self, run_id: str) -> ExperimentRun | None:
        """Retrieve an experiment run from the database.

        Args:
            run_id: ID of the experiment to retrieve

        Returns:
            Experiment run if found, None otherwise
        """
        session = self.get_session()
        try:
            record = session.query(ExperimentRecord).filter_by(run_id=run_id).first()

            if not record:
                logger.warning("experiment_not_found", run_id=run_id)
                return None

            experiment = ExperimentRun(
                run_id=record.run_id,
                experiment_name=record.experiment_name,
                task_ids=record.task_ids,
                strategy_config=record.strategy_config,
                metrics=record.metrics,
                status=TaskStatus(record.status),
                started_at=record.started_at,
                completed_at=record.completed_at,
                error_log=record.error_log
            )

            logger.info("experiment_retrieved", run_id=run_id)
            return experiment

        except Exception as e:
            logger.error("experiment_retrieval_failed", run_id=run_id, error=str(e))
            return None
        finally:
            session.close()

    def get_experiment_submissions(self, experiment_id: str) -> list[TaskSubmission]:
        """Get all submissions for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of submissions
        """
        session = self.get_session()
        try:
            # Get experiment to find task IDs
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                return []

            # Query submissions for those tasks
            records = session.query(SubmissionRecord).filter(
                SubmissionRecord.task_id.in_(experiment.task_ids)
            ).all()

            submissions = []
            for record in records:
                submission = TaskSubmission(
                    submission_id=record.submission_id,
                    task_id=record.task_id,
                    user_id=record.user_id,
                    predicted_output=record.predicted_output,
                    strategy_used=StrategyType(record.strategy_used),
                    confidence_score=record.confidence_score,
                    processing_time_ms=record.processing_time_ms,
                    resource_usage=record.resource_usage,
                    metadata=record.metadata,
                    submitted_at=record.submitted_at
                )
                submissions.append(submission)

            logger.info("experiment_submissions_retrieved",
                       experiment_id=experiment_id,
                       count=len(submissions))
            return submissions

        except Exception as e:
            logger.error("experiment_submissions_failed",
                        experiment_id=experiment_id,
                        error=str(e))
            return []
        finally:
            session.close()

    def update_experiment_status(
        self,
        run_id: str,
        status: TaskStatus,
        metrics: dict[str, Any] | None = None,
        error_log: str | None = None
    ) -> bool:
        """Update experiment status and metrics.

        Args:
            run_id: Experiment run ID
            status: New status
            metrics: Updated metrics
            error_log: Error log if failed

        Returns:
            True if updated successfully
        """
        session = self.get_session()
        try:
            record = session.query(ExperimentRecord).filter_by(run_id=run_id).first()

            if not record:
                logger.warning("experiment_not_found_for_update", run_id=run_id)
                return False

            record.status = status.value

            if metrics:
                record.metrics = metrics

            if error_log:
                record.error_log = error_log

            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                record.completed_at = datetime.utcnow()

            session.commit()

            logger.info("experiment_status_updated", run_id=run_id, status=status.value)
            return True

        except Exception as e:
            session.rollback()
            logger.error("experiment_update_failed", run_id=run_id, error=str(e))
            return False
        finally:
            session.close()


# Global repository instance
_task_repository: TaskRepository | None = None


def get_task_repository() -> TaskRepository:
    """Get the global task repository instance.

    Returns:
        TaskRepository instance
    """
    global _task_repository
    if _task_repository is None:
        import os
        database_url = os.environ.get("DATABASE_URL", "sqlite:///./evaluation.db")
        _task_repository = TaskRepository(database_url)
    return _task_repository
