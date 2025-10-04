"""Unified structured logging for cross-strategy integration.

This module configures structlog for JSON structured logging with correlation
IDs, strategy context, and unified error formatting across all strategies.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any

import structlog

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
strategy_context_var: ContextVar[str | None] = ContextVar("strategy_context", default=None)
task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)


def add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add correlation ID to log entries.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Updated event dict with correlation_id
    """
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_strategy_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add strategy context to log entries.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Updated event dict with strategy
    """
    strategy = strategy_context_var.get()
    if strategy:
        event_dict["strategy"] = strategy
    return event_dict


def add_task_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add task ID to log entries.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Updated event dict with task_id
    """
    task_id = task_id_var.get()
    if task_id:
        event_dict["task_id"] = task_id
    return event_dict


def add_timestamp(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add ISO 8601 timestamp to log entries.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Updated event dict with timestamp
    """
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def sanitize_reasoning_trace(trace: list[str], max_entries: int = 100) -> list[str]:
    """Sanitize reasoning trace to prevent log injection and memory exhaustion.

    Args:
        trace: Reasoning trace entries
        max_entries: Maximum number of entries to keep

    Returns:
        Sanitized trace limited to max_entries

    Security:
        - Limits trace size to prevent memory exhaustion
        - Removes newlines and control characters to prevent log injection
    """
    if not trace:
        return []

    sanitized = []
    for entry in trace[:max_entries]:
        sanitized_entry = "".join(
            c if c.isprintable() and c not in "\n\r\t" else " " for c in str(entry)
        )
        sanitized.append(sanitized_entry[:1000])

    return sanitized


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    include_traceback: bool = True,
) -> None:
    """Configure structlog for unified cross-strategy logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format (True) or console format (False)
        include_traceback: Include exception tracebacks in error logs

    Example:
        configure_logging(log_level="DEBUG", json_output=True)

        log = get_logger()
        with log_context(correlation_id="req_123", strategy="program_synthesis"):
            log.info(
                "strategy_execution_start",
                task_id="task_001",
                timeout_ms=300000
            )
    """
    processors: list[object] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_timestamp,
        add_correlation_id,
        add_strategy_context,
        add_task_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    if include_traceback:
        processors.append(structlog.processors.format_exc_info)

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured structlog logger.

    Args:
        name: Optional logger name (defaults to calling module)

    Returns:
        Configured BoundLogger instance

    Example:
        log = get_logger(__name__)
        log.info("task_processing", task_id="001", status="started")
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


class LogContext:
    """Context manager for adding logging context.

    Example:
        with LogContext(correlation_id="req_123", strategy="program_synthesis"):
            log.info("processing_task")
            # correlation_id and strategy automatically added to all logs
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        strategy: str | None = None,
        task_id: str | None = None,
        auto_generate_correlation_id: bool = True,
    ):
        """Initialize log context.

        Args:
            correlation_id: Optional correlation ID (generated if not provided)
            strategy: Strategy name
            task_id: Task identifier
            auto_generate_correlation_id: Generate correlation ID if not provided
        """
        self.correlation_id = correlation_id or (
            str(uuid.uuid4()) if auto_generate_correlation_id else None
        )
        self.strategy = strategy
        self.task_id = task_id

        self._correlation_token: object | None = None
        self._strategy_token: object | None = None
        self._task_token: object | None = None

    def __enter__(self) -> "LogContext":
        """Enter context and set context vars."""
        if self.correlation_id:
            self._correlation_token = correlation_id_var.set(self.correlation_id)
        if self.strategy:
            self._strategy_token = strategy_context_var.set(self.strategy)
        if self.task_id:
            self._task_token = task_id_var.set(self.task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and reset context vars."""
        if self._correlation_token:
            correlation_id_var.reset(self._correlation_token)
        if self._strategy_token:
            strategy_context_var.reset(self._strategy_token)
        if self._task_token:
            task_id_var.reset(self._task_token)


def log_strategy_execution_start(
    log: structlog.BoundLogger,
    task_id: str,
    timeout_ms: int,
    resource_budget: dict[str, Any] | None = None,
) -> None:
    """Log strategy execution start with standard format.

    Args:
        log: Configured logger
        task_id: Task identifier
        timeout_ms: Execution timeout in milliseconds
        resource_budget: Optional resource budget details
    """
    log.info(
        "strategy_execution_start",
        task_id=task_id,
        timeout_ms=timeout_ms,
        resource_budget=resource_budget or {},
    )


def log_strategy_execution_complete(
    log: structlog.BoundLogger,
    task_id: str,
    confidence: float,
    processing_time_ms: int,
    programs_evaluated: int | None = None,
) -> None:
    """Log strategy execution completion with standard format.

    Args:
        log: Configured logger
        task_id: Task identifier
        confidence: Output confidence score
        processing_time_ms: Total processing time
        programs_evaluated: Optional number of programs evaluated
    """
    log.info(
        "strategy_execution_complete",
        task_id=task_id,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        programs_evaluated=programs_evaluated,
    )


def log_strategy_error(
    log: structlog.BoundLogger,
    task_id: str,
    error_type: str,
    error_message: str,
    traceback: str | None = None,
) -> None:
    """Log strategy error with unified format.

    Args:
        log: Configured logger
        task_id: Task identifier
        error_type: Error classification
        error_message: Error description
        traceback: Optional exception traceback

    Security:
        Sanitizes error messages to prevent log injection
    """
    sanitized_message = "".join(
        c if c.isprintable() and c not in "\n\r" else " " for c in error_message
    )

    log.error(
        "strategy_execution_error",
        task_id=task_id,
        error_type=error_type,
        error_message=sanitized_message[:2000],
        traceback=traceback[:5000] if traceback else None,
    )
