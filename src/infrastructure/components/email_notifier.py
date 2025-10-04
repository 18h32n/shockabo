"""Secure email notification system for experiment status updates.

This module provides a secure email notification system for the ARC Prize 2025 platform
rotation system. It supports multiple notification types, rate limiting, template-based
formatting, and integrates with the existing secure credential storage system.

Features:
- TLS-encrypted SMTP connections
- Multiple notification types (completion, failures, quota warnings, queue status)
- Rate limiting (configurable, default 10 emails per hour)
- Queue-based email sending with retry logic
- Template-based HTML and plain text email formatting
- Asynchronous operation to avoid blocking
- Integration with secure credential storage
"""

import asyncio
import smtplib
import ssl
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from ...utils.secure_credentials import get_platform_credential_manager

logger = structlog.get_logger(__name__)


class NotificationType(Enum):
    """Email notification types."""
    EXPERIMENT_COMPLETION = "experiment_completion"
    EXPERIMENT_FAILURE = "experiment_failure"
    PLATFORM_FAILURE = "platform_failure"
    QUOTA_WARNING = "quota_warning"
    QUEUE_STATUS = "queue_status"
    SYSTEM_ALERT = "system_alert"
    DAILY_SUMMARY = "daily_summary"


class Priority(Enum):
    """Email notification priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EmailTemplate:
    """Email template configuration."""
    subject_template: str
    body_template: str
    html_template: str | None = None

    def render_subject(self, **context) -> str:
        """Render subject template with context variables."""
        try:
            return self.subject_template.format(**context)
        except KeyError as e:
            logger.warning("missing_subject_template_variable", variable=str(e))
            return self.subject_template

    def render_body(self, **context) -> str:
        """Render body template with context variables."""
        try:
            return self.body_template.format(**context)
        except KeyError as e:
            logger.warning("missing_body_template_variable", variable=str(e))
            return self.body_template

    def render_html(self, **context) -> str | None:
        """Render HTML template with context variables."""
        if not self.html_template:
            return None
        try:
            return self.html_template.format(**context)
        except KeyError as e:
            logger.warning("missing_html_template_variable", variable=str(e))
            return self.html_template


@dataclass
class EmailNotification:
    """Email notification message."""
    notification_type: NotificationType
    priority: Priority
    subject: str
    body: str
    html_body: str | None = None
    recipients: list[str] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    attachments: list[Path] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_emails_per_hour: int = 10
    burst_limit: int = 3
    burst_window_minutes: int = 5
    cooldown_minutes: int = 15


@dataclass
class EmailConfig:
    """Email system configuration."""
    smtp_server: str | None = None
    smtp_port: int = 587
    use_tls: bool = True
    timeout_seconds: int = 30
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    default_recipients: list[str] = field(default_factory=list)
    retry_delays: list[int] = field(default_factory=lambda: [60, 300, 900])  # 1min, 5min, 15min
    queue_max_size: int = 1000


class EmailNotifier:
    """Secure email notification system with rate limiting and queue management."""

    def __init__(self, config: EmailConfig | None = None):
        """Initialize email notification system.

        Args:
            config: Email system configuration
        """
        self.config = config or EmailConfig()
        self.credential_manager = get_platform_credential_manager()

        # Email queue and processing
        self.email_queue: deque[EmailNotification] = deque(maxlen=self.config.queue_max_size)
        self.failed_queue: deque[EmailNotification] = deque()
        self.processing_task: asyncio.Task | None = None
        self.running = False

        # Rate limiting tracking
        self.rate_limit_tracker: dict[str, list[datetime]] = defaultdict(list)
        self.burst_tracker: dict[str, list[datetime]] = defaultdict(list)
        self.last_cooldown: dict[str, datetime] = {}

        # Statistics
        self.stats = {
            'emails_sent': 0,
            'emails_failed': 0,
            'emails_rate_limited': 0,
            'queue_overflows': 0,
            'start_time': datetime.now()
        }

        # Email templates
        self.templates = self._initialize_templates()

        # Validate configuration and credentials
        self._validate_setup()

    def _validate_setup(self) -> bool:
        """Validate email configuration and credentials.

        Returns:
            True if setup is valid
        """
        try:
            # Get email credentials
            smtp_server = self._get_smtp_server()
            username = self._get_username()
            password = self._get_password()
            from_address = self._get_from_address()

            if not all([smtp_server, username, password, from_address]):
                missing = []
                if not smtp_server:
                    missing.append('smtp_server')
                if not username:
                    missing.append('username')
                if not password:
                    missing.append('password')
                if not from_address:
                    missing.append('from_address')

                logger.warning("email_credentials_incomplete", missing=missing)
                return False

            # Validate SMTP server format
            if smtp_server and ':' not in smtp_server:
                logger.warning("smtp_server_missing_port", server=smtp_server)
                return False

            # Set default recipients if none configured
            if not self.config.default_recipients and from_address:
                self.config.default_recipients = [from_address]
                logger.info("using_from_address_as_default_recipient", address=from_address)

            logger.info("email_notifier_configuration_valid",
                       server=smtp_server.split(':')[0],
                       recipients=len(self.config.default_recipients))
            return True

        except Exception as e:
            logger.error("email_configuration_validation_failed", error=str(e))
            return False

    def _get_smtp_server(self) -> str | None:
        """Get SMTP server configuration."""
        return (self.credential_manager.get_platform_credential('email', 'smtp_server') or
                self.config.smtp_server)

    def _get_username(self) -> str | None:
        """Get email username."""
        return self.credential_manager.get_platform_credential('email', 'username')

    def _get_password(self) -> str | None:
        """Get email password."""
        return self.credential_manager.get_platform_credential('email', 'password')

    def _get_from_address(self) -> str | None:
        """Get from email address."""
        return (self.credential_manager.get_platform_credential('email', 'from_address') or
                self._get_username())

    def _initialize_templates(self) -> dict[NotificationType, EmailTemplate]:
        """Initialize default email templates."""
        return {
            NotificationType.EXPERIMENT_COMPLETION: EmailTemplate(
                subject_template="[ARC Prize] Experiment Completed: {experiment_name}",
                body_template="""Experiment Completion Notification

Experiment: {experiment_name}
Job ID: {job_id}
Platform: {platform}
Status: COMPLETED
Duration: {duration_minutes:.1f} minutes
Completed at: {completion_time}

Results Summary:
{results_summary}

Platform Stats:
- GPU Utilization: {gpu_utilization:.1f}%
- Tasks Processed: {tasks_completed}

Next experiments in queue: {queue_size}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #28a745;">Experiment Completed Successfully</h2>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
    <h3>{experiment_name}</h3>
    <p><strong>Job ID:</strong> {job_id}</p>
    <p><strong>Platform:</strong> {platform}</p>
    <p><strong>Duration:</strong> {duration_minutes:.1f} minutes</p>
    <p><strong>Completed:</strong> {completion_time}</p>
</div>
<div style="background-color: #e9ecef; padding: 10px; border-radius: 5px;">
    <h4>Results Summary</h4>
    <pre>{results_summary}</pre>
</div>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.EXPERIMENT_FAILURE: EmailTemplate(
                subject_template="[ARC Prize] Experiment Failed: {experiment_name}",
                body_template="""Experiment Failure Notification

Experiment: {experiment_name}
Job ID: {job_id}
Platform: {platform}
Status: FAILED
Duration: {duration_minutes:.1f} minutes
Failed at: {failure_time}

Error Details:
{error_message}

Stack Trace:
{stack_trace}

Retry Info:
- Attempt: {retry_count}/{max_retries}
- Next retry: {next_retry_time}

Action Required: {action_required}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #dc3545;">Experiment Failed</h2>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
    <h3>{experiment_name}</h3>
    <p><strong>Job ID:</strong> {job_id}</p>
    <p><strong>Platform:</strong> {platform}</p>
    <p><strong>Duration:</strong> {duration_minutes:.1f} minutes</p>
    <p><strong>Failed:</strong> {failure_time}</p>
</div>
<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
    <h4>Error Details</h4>
    <p>{error_message}</p>
    <details>
        <summary>Stack Trace</summary>
        <pre style="font-size: 12px;">{stack_trace}</pre>
    </details>
</div>
<p><strong>Retry:</strong> Attempt {retry_count}/{max_retries}</p>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.PLATFORM_FAILURE: EmailTemplate(
                subject_template="[ARC Prize] Platform Failure: {platform}",
                body_template="""Platform Failure Alert

Platform: {platform}
Failure Type: {failure_type}
Detected at: {detection_time}

Error Details:
{error_details}

Impact:
- Suspended Jobs: {suspended_jobs}
- Queue Status: {queue_status}
- Next Platform: {next_platform}

Automatic Actions Taken:
{actions_taken}

Estimated Recovery Time: {recovery_estimate}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #fd7e14;">Platform Failure Alert</h2>
<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7;">
    <h3>Platform: {platform}</h3>
    <p><strong>Failure Type:</strong> {failure_type}</p>
    <p><strong>Detected:</strong> {detection_time}</p>
</div>
<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
    <h4>Impact Assessment</h4>
    <ul>
        <li>Suspended Jobs: {suspended_jobs}</li>
        <li>Queue Status: {queue_status}</li>
        <li>Next Platform: {next_platform}</li>
    </ul>
</div>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.QUOTA_WARNING: EmailTemplate(
                subject_template="[ARC Prize] Quota Warning: {platform} - {usage_percentage:.0f}%",
                body_template="""Resource Quota Warning

Platform: {platform}
Resource Type: {resource_type}
Current Usage: {current_usage} / {quota_limit} ({usage_percentage:.1f}%)
Warning Threshold: {warning_threshold:.1f}%

Estimated Time to Limit: {time_to_limit}
Recommended Actions: {recommended_actions}

Current Queue Status:
- Pending Jobs: {pending_jobs}
- Running Jobs: {running_jobs}
- Estimated Completion: {estimated_completion}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #ffc107;">Resource Quota Warning</h2>
<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;">
    <h3>{platform} - {resource_type}</h3>
    <div style="background-color: #495057; color: white; padding: 5px; border-radius: 3px;">
        <div style="background-color: #ffc107; width: {usage_percentage}%; height: 20px; border-radius: 3px;"></div>
    </div>
    <p>{current_usage} / {quota_limit} ({usage_percentage:.1f}%)</p>
</div>
<p><strong>Time to Limit:</strong> {time_to_limit}</p>
<p><strong>Recommended Actions:</strong> {recommended_actions}</p>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.QUEUE_STATUS: EmailTemplate(
                subject_template="[ARC Prize] Queue Status Update - {queue_size} jobs",
                body_template="""Experiment Queue Status

Queue Statistics:
- Total Jobs: {total_jobs}
- Pending: {pending_jobs}
- Running: {running_jobs}
- Completed: {completed_jobs}
- Failed: {failed_jobs}

Platform Status:
{platform_status}

Performance Metrics:
- Jobs/Hour: {jobs_per_hour:.1f}
- Average Runtime: {avg_runtime_minutes:.1f} minutes
- Success Rate: {success_rate:.1f}%

Estimated Queue Completion: {queue_completion_estimate}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #17a2b8;">Experiment Queue Status</h2>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 20px 0;">
    <div style="background-color: #e9ecef; padding: 10px; text-align: center; border-radius: 5px;">
        <div style="font-size: 24px; font-weight: bold;">{total_jobs}</div>
        <div>Total</div>
    </div>
    <div style="background-color: #cfe2ff; padding: 10px; text-align: center; border-radius: 5px;">
        <div style="font-size: 24px; font-weight: bold;">{pending_jobs}</div>
        <div>Pending</div>
    </div>
    <div style="background-color: #d1ecf1; padding: 10px; text-align: center; border-radius: 5px;">
        <div style="font-size: 24px; font-weight: bold;">{running_jobs}</div>
        <div>Running</div>
    </div>
    <div style="background-color: #d4edda; padding: 10px; text-align: center; border-radius: 5px;">
        <div style="font-size: 24px; font-weight: bold;">{completed_jobs}</div>
        <div>Completed</div>
    </div>
    <div style="background-color: #f8d7da; padding: 10px; text-align: center; border-radius: 5px;">
        <div style="font-size: 24px; font-weight: bold;">{failed_jobs}</div>
        <div>Failed</div>
    </div>
</div>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
    <h4>Platform Status</h4>
    <pre>{platform_status}</pre>
</div>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.SYSTEM_ALERT: EmailTemplate(
                subject_template="[ARC Prize] System Alert: {alert_type}",
                body_template="""System Alert Notification

Alert Type: {alert_type}
Severity: {severity}
Component: {component}
Detected at: {detection_time}

Description:
{alert_description}

System Impact:
{system_impact}

Automated Response:
{automated_response}

Manual Action Required: {manual_action_required}
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #dc3545;">System Alert</h2>
<div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;">
    <h3>{alert_type}</h3>
    <p><strong>Severity:</strong> <span style="color: #dc3545;">{severity}</span></p>
    <p><strong>Component:</strong> {component}</p>
    <p><strong>Detected:</strong> {detection_time}</p>
</div>
<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
    <h4>Description</h4>
    <p>{alert_description}</p>
</div>
<p><strong>Manual Action Required:</strong> {manual_action_required}</p>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            ),

            NotificationType.DAILY_SUMMARY: EmailTemplate(
                subject_template="[ARC Prize] Daily Summary - {date}",
                body_template="""Daily System Summary - {date}

Experiment Statistics:
- Jobs Completed: {jobs_completed}
- Jobs Failed: {jobs_failed}
- Success Rate: {success_rate:.1f}%
- Total Runtime: {total_runtime_hours:.1f} hours

Platform Distribution:
{platform_distribution}

Resource Usage:
- GPU Hours Used: {gpu_hours_used:.1f}
- Cost Estimate: ${cost_estimate:.2f}
- Efficiency Rating: {efficiency_rating}/10

Notable Events:
{notable_events}

Tomorrow's Schedule:
- Queued Jobs: {queued_jobs}
- Estimated Runtime: {estimated_runtime_hours:.1f} hours
""",
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #6f42c1;">Daily Summary - {date}</h2>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0;">
    <div style="background-color: #d4edda; padding: 15px; text-align: center; border-radius: 5px;">
        <div style="font-size: 28px; font-weight: bold; color: #155724;">{jobs_completed}</div>
        <div>Completed</div>
    </div>
    <div style="background-color: #f8d7da; padding: 15px; text-align: center; border-radius: 5px;">
        <div style="font-size: 28px; font-weight: bold; color: #721c24;">{jobs_failed}</div>
        <div>Failed</div>
    </div>
    <div style="background-color: #e2e3e5; padding: 15px; text-align: center; border-radius: 5px;">
        <div style="font-size: 28px; font-weight: bold; color: #383d41;">{success_rate:.0f}%</div>
        <div>Success Rate</div>
    </div>
    <div style="background-color: #cfe2ff; padding: 15px; text-align: center; border-radius: 5px;">
        <div style="font-size: 28px; font-weight: bold; color: #084298;">{total_runtime_hours:.1f}</div>
        <div>Runtime Hours</div>
    </div>
</div>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
    <h4>Platform Distribution</h4>
    <pre>{platform_distribution}</pre>
</div>
<p><small>Generated by ARC Prize 2025 Email Notifier at {timestamp}</small></p>
</body>
</html>
"""
            )
        }

    async def start(self) -> bool:
        """Start the email notification system.

        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("email_notifier_already_running")
            return False

        if not self._validate_setup():
            logger.error("email_notifier_setup_invalid")
            return False

        try:
            self.running = True
            self.processing_task = asyncio.create_task(self._process_email_queue())
            logger.info("email_notifier_started",
                       queue_max_size=self.config.queue_max_size,
                       rate_limit=self.config.rate_limit.max_emails_per_hour)
            return True

        except Exception as e:
            logger.error("email_notifier_start_failed", error=str(e))
            self.running = False
            return False

    async def stop(self) -> bool:
        """Stop the email notification system.

        Returns:
            True if stopped successfully
        """
        if not self.running:
            return True

        try:
            self.running = False

            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            # Process remaining emails with shorter timeout
            await self._flush_remaining_emails()

            logger.info("email_notifier_stopped",
                       emails_sent=self.stats['emails_sent'],
                       emails_failed=self.stats['emails_failed'])
            return True

        except Exception as e:
            logger.error("email_notifier_stop_failed", error=str(e))
            return False

    def send_notification(self,
                         notification_type: NotificationType,
                         priority: Priority = Priority.NORMAL,
                         recipients: list[str] | None = None,
                         **context) -> bool:
        """Send email notification.

        Args:
            notification_type: Type of notification to send
            priority: Notification priority
            recipients: Email recipients (uses default if None)
            **context: Template context variables

        Returns:
            True if notification was queued successfully
        """
        try:
            # Use default recipients if none provided
            if recipients is None:
                recipients = self.config.default_recipients

            if not recipients:
                logger.error("no_email_recipients_available")
                return False

            # Get template for notification type
            template = self.templates.get(notification_type)
            if not template:
                logger.error("no_template_for_notification_type", type=notification_type)
                return False

            # Add timestamp to context
            context['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

            # Render email content
            subject = template.render_subject(**context)
            body = template.render_body(**context)
            html_body = template.render_html(**context)

            # Create notification
            notification = EmailNotification(
                notification_type=notification_type,
                priority=priority,
                subject=subject,
                body=body,
                html_body=html_body,
                recipients=recipients,
                context=context
            )

            # Check rate limits
            if self._is_rate_limited(recipients[0] if recipients else 'default'):
                logger.warning("notification_rate_limited",
                              type=notification_type.value,
                              priority=priority.value)
                self.stats['emails_rate_limited'] += 1
                return False

            # Add to queue
            if len(self.email_queue) >= self.config.queue_max_size:
                # Remove lowest priority items to make room
                self._make_queue_space()
                self.stats['queue_overflows'] += 1

            self.email_queue.append(notification)

            logger.info("notification_queued",
                       type=notification_type.value,
                       priority=priority.value,
                       queue_size=len(self.email_queue))
            return True

        except Exception as e:
            logger.error("notification_queue_failed",
                        type=notification_type.value,
                        error=str(e))
            return False

    def send_experiment_completion(self, experiment_name: str, job_id: str,
                                 platform: str, duration_minutes: float,
                                 results_summary: str, **kwargs) -> bool:
        """Send experiment completion notification."""
        return self.send_notification(
            NotificationType.EXPERIMENT_COMPLETION,
            Priority.NORMAL,
            experiment_name=experiment_name,
            job_id=job_id,
            platform=platform,
            duration_minutes=duration_minutes,
            completion_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            results_summary=results_summary,
            gpu_utilization=kwargs.get('gpu_utilization', 0.0),
            tasks_completed=kwargs.get('tasks_completed', 0),
            queue_size=kwargs.get('queue_size', len(self.email_queue))
        )

    def send_experiment_failure(self, experiment_name: str, job_id: str,
                              platform: str, duration_minutes: float,
                              error_message: str, retry_count: int = 0,
                              max_retries: int = 3, **kwargs) -> bool:
        """Send experiment failure notification."""
        next_retry = "Not scheduled" if retry_count >= max_retries else f"In {self.config.retry_delays[min(retry_count, len(self.config.retry_delays)-1)]} seconds"
        action_required = "Manual intervention required" if retry_count >= max_retries else "Automatic retry scheduled"

        return self.send_notification(
            NotificationType.EXPERIMENT_FAILURE,
            Priority.HIGH,
            experiment_name=experiment_name,
            job_id=job_id,
            platform=platform,
            duration_minutes=duration_minutes,
            failure_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            error_message=error_message,
            stack_trace=kwargs.get('stack_trace', 'Not available'),
            retry_count=retry_count,
            max_retries=max_retries,
            next_retry_time=next_retry,
            action_required=action_required
        )

    def send_platform_failure(self, platform: str, failure_type: str,
                            error_details: str, **kwargs) -> bool:
        """Send platform failure notification."""
        return self.send_notification(
            NotificationType.PLATFORM_FAILURE,
            Priority.CRITICAL,
            platform=platform,
            failure_type=failure_type,
            detection_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            error_details=error_details,
            suspended_jobs=kwargs.get('suspended_jobs', 0),
            queue_status=kwargs.get('queue_status', 'Unknown'),
            next_platform=kwargs.get('next_platform', 'Not determined'),
            actions_taken=kwargs.get('actions_taken', 'Platform rotation initiated'),
            recovery_estimate=kwargs.get('recovery_estimate', 'Unknown')
        )

    def send_quota_warning(self, platform: str, resource_type: str,
                         current_usage: int | float, quota_limit: int | float,
                         warning_threshold: float = 80.0, **kwargs) -> bool:
        """Send resource quota warning notification."""
        usage_percentage = (current_usage / quota_limit) * 100

        return self.send_notification(
            NotificationType.QUOTA_WARNING,
            Priority.HIGH if usage_percentage >= 90 else Priority.NORMAL,
            platform=platform,
            resource_type=resource_type,
            current_usage=current_usage,
            quota_limit=quota_limit,
            usage_percentage=usage_percentage,
            warning_threshold=warning_threshold,
            time_to_limit=kwargs.get('time_to_limit', 'Unknown'),
            recommended_actions=kwargs.get('recommended_actions', 'Monitor usage and consider platform rotation'),
            pending_jobs=kwargs.get('pending_jobs', 0),
            running_jobs=kwargs.get('running_jobs', 0),
            estimated_completion=kwargs.get('estimated_completion', 'Unknown')
        )

    def send_queue_status(self, total_jobs: int, pending_jobs: int, running_jobs: int,
                        completed_jobs: int, failed_jobs: int, **kwargs) -> bool:
        """Send queue status notification."""
        success_rate = (completed_jobs / max(completed_jobs + failed_jobs, 1)) * 100

        return self.send_notification(
            NotificationType.QUEUE_STATUS,
            Priority.LOW,
            queue_size=total_jobs,
            total_jobs=total_jobs,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            success_rate=success_rate,
            platform_status=kwargs.get('platform_status', 'All platforms operational'),
            jobs_per_hour=kwargs.get('jobs_per_hour', 0.0),
            avg_runtime_minutes=kwargs.get('avg_runtime_minutes', 0.0),
            queue_completion_estimate=kwargs.get('queue_completion_estimate', 'Unknown')
        )

    def send_system_alert(self, alert_type: str, severity: str, component: str,
                        alert_description: str, **kwargs) -> bool:
        """Send system alert notification."""
        priority_map = {
            'low': Priority.LOW,
            'normal': Priority.NORMAL,
            'high': Priority.HIGH,
            'critical': Priority.CRITICAL
        }

        return self.send_notification(
            NotificationType.SYSTEM_ALERT,
            priority_map.get(severity.lower(), Priority.NORMAL),
            alert_type=alert_type,
            severity=severity.upper(),
            component=component,
            detection_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            alert_description=alert_description,
            system_impact=kwargs.get('system_impact', 'Unknown'),
            automated_response=kwargs.get('automated_response', 'None'),
            manual_action_required=kwargs.get('manual_action_required', 'Review alert and take appropriate action')
        )

    def send_daily_summary(self, date: str, jobs_completed: int, jobs_failed: int,
                         total_runtime_hours: float, **kwargs) -> bool:
        """Send daily summary notification."""
        success_rate = (jobs_completed / max(jobs_completed + jobs_failed, 1)) * 100

        return self.send_notification(
            NotificationType.DAILY_SUMMARY,
            Priority.LOW,
            date=date,
            jobs_completed=jobs_completed,
            jobs_failed=jobs_failed,
            success_rate=success_rate,
            total_runtime_hours=total_runtime_hours,
            platform_distribution=kwargs.get('platform_distribution', 'No data available'),
            gpu_hours_used=kwargs.get('gpu_hours_used', 0.0),
            cost_estimate=kwargs.get('cost_estimate', 0.0),
            efficiency_rating=kwargs.get('efficiency_rating', 0),
            notable_events=kwargs.get('notable_events', 'None'),
            queued_jobs=kwargs.get('queued_jobs', 0),
            estimated_runtime_hours=kwargs.get('estimated_runtime_hours', 0.0)
        )

    def _is_rate_limited(self, key: str) -> bool:
        """Check if key is rate limited.

        Args:
            key: Rate limiting key (usually email address or 'default')

        Returns:
            True if rate limited
        """
        now = datetime.now()

        # Check if in cooldown period
        if key in self.last_cooldown:
            cooldown_end = self.last_cooldown[key] + timedelta(minutes=self.config.rate_limit.cooldown_minutes)
            if now < cooldown_end:
                return True

        # Clean old entries
        hour_ago = now - timedelta(hours=1)
        burst_window_ago = now - timedelta(minutes=self.config.rate_limit.burst_window_minutes)

        self.rate_limit_tracker[key] = [ts for ts in self.rate_limit_tracker[key] if ts > hour_ago]
        self.burst_tracker[key] = [ts for ts in self.burst_tracker[key] if ts > burst_window_ago]

        # Check burst limit
        if len(self.burst_tracker[key]) >= self.config.rate_limit.burst_limit:
            self.last_cooldown[key] = now
            logger.warning("rate_limit_burst_exceeded", key=key,
                          burst_count=len(self.burst_tracker[key]))
            return True

        # Check hourly limit
        if len(self.rate_limit_tracker[key]) >= self.config.rate_limit.max_emails_per_hour:
            logger.warning("rate_limit_hourly_exceeded", key=key,
                          hourly_count=len(self.rate_limit_tracker[key]))
            return True

        # Update trackers
        self.rate_limit_tracker[key].append(now)
        self.burst_tracker[key].append(now)

        return False

    def _make_queue_space(self) -> None:
        """Remove lowest priority items from queue to make space."""
        if not self.email_queue:
            return

        # Sort by priority (lowest first) and remove items
        queue_list = list(self.email_queue)
        queue_list.sort(key=lambda x: x.priority.value)

        # Remove lowest priority items (keep at least half the queue)
        items_to_remove = min(len(queue_list) // 4, 10)
        for _ in range(items_to_remove):
            if self.email_queue:
                removed = self.email_queue.popleft()
                logger.warning("email_dropped_for_queue_space",
                              type=removed.notification_type.value,
                              priority=removed.priority.value)

    async def _process_email_queue(self) -> None:
        """Process email queue continuously."""
        while self.running:
            try:
                if not self.email_queue:
                    await asyncio.sleep(5)  # Check every 5 seconds when queue is empty
                    continue

                # Get next email (priority queue)
                notification = self._get_next_notification()
                if notification:
                    success = await self._send_email_with_retry(notification)
                    if success:
                        self.stats['emails_sent'] += 1
                    else:
                        self.stats['emails_failed'] += 1
                        # Add to failed queue for later retry if not exceeded max retries
                        if notification.retry_count < notification.max_retries:
                            self.failed_queue.append(notification)

                # Process failed emails periodically
                if len(self.failed_queue) > 0 and time.time() % 300 < 5:  # Every 5 minutes
                    await self._retry_failed_emails()

                await asyncio.sleep(1)  # Brief pause between emails

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("email_processing_error", error=str(e), traceback=traceback.format_exc())
                await asyncio.sleep(30)  # Wait longer on error

    def _get_next_notification(self) -> EmailNotification | None:
        """Get next notification from queue based on priority."""
        if not self.email_queue:
            return None

        # Find highest priority notification
        max_priority = max(notification.priority.value for notification in self.email_queue)

        # Get first notification with max priority
        for i, notification in enumerate(self.email_queue):
            if notification.priority.value == max_priority:
                # Remove from queue and return
                del self.email_queue[i]
                return notification

        return None

    async def _send_email_with_retry(self, notification: EmailNotification) -> bool:
        """Send email with retry logic.

        Args:
            notification: Email notification to send

        Returns:
            True if sent successfully
        """
        for attempt in range(notification.max_retries + 1):
            try:
                success = await self._send_single_email(notification)
                if success:
                    return True

                # Wait before retry
                if attempt < notification.max_retries:
                    delay = self.config.retry_delays[min(attempt, len(self.config.retry_delays) - 1)]
                    logger.info("email_retry_scheduled",
                              attempt=attempt + 1,
                              max_retries=notification.max_retries,
                              delay_seconds=delay)
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.error("email_send_attempt_failed",
                           attempt=attempt + 1,
                           error=str(e))
                if attempt == notification.max_retries:
                    return False

                # Wait before retry
                delay = self.config.retry_delays[min(attempt, len(self.config.retry_delays) - 1)]
                await asyncio.sleep(delay)

        return False

    async def _send_single_email(self, notification: EmailNotification) -> bool:
        """Send a single email notification.

        Args:
            notification: Email notification to send

        Returns:
            True if sent successfully
        """
        try:
            # Get email configuration
            smtp_server_config = self._get_smtp_server()
            if not smtp_server_config or ':' not in smtp_server_config:
                logger.error("invalid_smtp_server_configuration", server=smtp_server_config)
                return False

            smtp_server, smtp_port_str = smtp_server_config.split(':', 1)
            smtp_port = int(smtp_port_str)
            username = self._get_username()
            password = self._get_password()
            from_address = self._get_from_address()

            if not all([smtp_server, username, password, from_address]):
                logger.error("email_credentials_missing_for_send")
                return False

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = from_address
            msg['To'] = ', '.join(notification.recipients)

            # Add text body
            msg.attach(MIMEText(notification.body, 'plain', 'utf-8'))

            # Add HTML body if available
            if notification.html_body:
                msg.attach(MIMEText(notification.html_body, 'html', 'utf-8'))

            # Add attachments
            for attachment_path in notification.attachments:
                if attachment_path.exists():
                    try:
                        with open(attachment_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())

                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {attachment_path.name}'
                        )
                        msg.attach(part)
                    except Exception as e:
                        logger.warning("attachment_failed", path=str(attachment_path), error=str(e))

            # Send email with timeout
            await asyncio.wait_for(
                self._smtp_send(msg, smtp_server, smtp_port, username, password, from_address, notification.recipients),
                timeout=self.config.timeout_seconds
            )

            logger.info("email_sent_successfully",
                       type=notification.notification_type.value,
                       priority=notification.priority.value,
                       recipients=len(notification.recipients))

            return True

        except TimeoutError:
            logger.error("email_send_timeout", timeout=self.config.timeout_seconds)
            return False
        except Exception as e:
            logger.error("email_send_failed",
                        type=notification.notification_type.value,
                        error=str(e))
            return False

    async def _smtp_send(self, msg: MIMEMultipart, smtp_server: str, smtp_port: int,
                        username: str, password: str, from_address: str, recipients: list[str]) -> None:
        """Send email via SMTP (async wrapper for synchronous operation)."""
        def _send():
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port, timeout=self.config.timeout_seconds) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(username, password)
                server.send_message(msg, from_address, recipients)

        # Run synchronous SMTP in thread pool
        await asyncio.get_event_loop().run_in_executor(None, _send)

    async def _retry_failed_emails(self) -> None:
        """Retry failed emails."""
        retry_queue = []

        while self.failed_queue:
            notification = self.failed_queue.popleft()
            notification.retry_count += 1

            if notification.retry_count <= notification.max_retries:
                success = await self._send_email_with_retry(notification)
                if success:
                    self.stats['emails_sent'] += 1
                    logger.info("failed_email_retry_success",
                              type=notification.notification_type.value,
                              retry_count=notification.retry_count)
                else:
                    retry_queue.append(notification)
            else:
                logger.error("failed_email_max_retries_exceeded",
                           type=notification.notification_type.value,
                           max_retries=notification.max_retries)
                self.stats['emails_failed'] += 1

        # Re-queue items that still need retries
        for notification in retry_queue:
            self.failed_queue.append(notification)

    async def _flush_remaining_emails(self) -> None:
        """Send remaining emails in queue with reduced timeout."""
        original_timeout = self.config.timeout_seconds
        self.config.timeout_seconds = 10  # Reduced timeout for shutdown

        try:
            # Process remaining emails with shorter timeout
            processed = 0
            max_process = 20  # Limit during shutdown

            while self.email_queue and processed < max_process:
                notification = self._get_next_notification()
                if notification:
                    await self._send_single_email(notification)
                    processed += 1

            if self.email_queue:
                logger.warning("emails_not_sent_during_shutdown", remaining=len(self.email_queue))

        finally:
            self.config.timeout_seconds = original_timeout

    def get_statistics(self) -> dict[str, Any]:
        """Get email notifier statistics.

        Returns:
            Dictionary of statistics
        """
        uptime = datetime.now() - self.stats['start_time']

        return {
            'running': self.running,
            'uptime_seconds': uptime.total_seconds(),
            'emails_sent': self.stats['emails_sent'],
            'emails_failed': self.stats['emails_failed'],
            'emails_rate_limited': self.stats['emails_rate_limited'],
            'queue_overflows': self.stats['queue_overflows'],
            'queue_size': len(self.email_queue),
            'failed_queue_size': len(self.failed_queue),
            'rate_limit_per_hour': self.config.rate_limit.max_emails_per_hour,
            'success_rate': (self.stats['emails_sent'] / max(self.stats['emails_sent'] + self.stats['emails_failed'], 1)) * 100,
            'configuration': {
                'smtp_server': self._get_smtp_server(),
                'use_tls': self.config.use_tls,
                'timeout_seconds': self.config.timeout_seconds,
                'queue_max_size': self.config.queue_max_size,
                'default_recipients': len(self.config.default_recipients)
            }
        }

    def update_template(self, notification_type: NotificationType,
                       subject_template: str | None = None,
                       body_template: str | None = None,
                       html_template: str | None = None) -> bool:
        """Update email template for notification type.

        Args:
            notification_type: Type of notification
            subject_template: New subject template
            body_template: New body template
            html_template: New HTML template

        Returns:
            True if updated successfully
        """
        try:
            if notification_type not in self.templates:
                logger.error("template_not_found", type=notification_type.value)
                return False

            current_template = self.templates[notification_type]

            if subject_template:
                current_template.subject_template = subject_template
            if body_template:
                current_template.body_template = body_template
            if html_template:
                current_template.html_template = html_template

            logger.info("template_updated", type=notification_type.value)
            return True

        except Exception as e:
            logger.error("template_update_failed", type=notification_type.value, error=str(e))
            return False

    def test_connection(self) -> bool:
        """Test email connection and credentials.

        Returns:
            True if connection test successful
        """
        try:
            smtp_server_config = self._get_smtp_server()
            if not smtp_server_config or ':' not in smtp_server_config:
                logger.error("invalid_smtp_server_for_test")
                return False

            smtp_server, smtp_port_str = smtp_server_config.split(':', 1)
            smtp_port = int(smtp_port_str)
            username = self._get_username()
            password = self._get_password()

            if not all([smtp_server, username, password]):
                logger.error("email_credentials_missing_for_test")
                return False

            # Test connection
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(username, password)

            logger.info("email_connection_test_successful", server=smtp_server)
            return True

        except Exception as e:
            logger.error("email_connection_test_failed", error=str(e))
            return False


# Singleton instance
_email_notifier: EmailNotifier | None = None


def get_email_notifier(config: EmailConfig | None = None) -> EmailNotifier:
    """Get singleton email notifier instance.

    Args:
        config: Email configuration (only used for first initialization)

    Returns:
        EmailNotifier instance
    """
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier(config)
    return _email_notifier


# Convenience functions for common notifications
async def send_experiment_completion_notification(experiment_name: str, job_id: str,
                                                platform: str, duration_minutes: float,
                                                results_summary: str, **kwargs) -> bool:
    """Send experiment completion notification."""
    notifier = get_email_notifier()
    return notifier.send_experiment_completion(
        experiment_name, job_id, platform, duration_minutes, results_summary, **kwargs
    )


async def send_experiment_failure_notification(experiment_name: str, job_id: str,
                                             platform: str, duration_minutes: float,
                                             error_message: str, retry_count: int = 0,
                                             max_retries: int = 3, **kwargs) -> bool:
    """Send experiment failure notification."""
    notifier = get_email_notifier()
    return notifier.send_experiment_failure(
        experiment_name, job_id, platform, duration_minutes,
        error_message, retry_count, max_retries, **kwargs
    )


async def send_platform_failure_notification(platform: str, failure_type: str,
                                           error_details: str, **kwargs) -> bool:
    """Send platform failure notification."""
    notifier = get_email_notifier()
    return notifier.send_platform_failure(platform, failure_type, error_details, **kwargs)


async def send_quota_warning_notification(platform: str, resource_type: str,
                                        current_usage: int | float,
                                        quota_limit: int | float,
                                        warning_threshold: float = 80.0, **kwargs) -> bool:
    """Send quota warning notification."""
    notifier = get_email_notifier()
    return notifier.send_quota_warning(
        platform, resource_type, current_usage, quota_limit, warning_threshold, **kwargs
    )
