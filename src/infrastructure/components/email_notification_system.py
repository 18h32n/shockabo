"""Email notification system for experiment status updates.

This module provides comprehensive email notifications for experiment status,
platform rotation events, and system alerts.
"""

import asyncio
import json
import smtplib
import ssl
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import jinja2
import structlog

from ..utils.secure_credentials import get_platform_credential_manager
from .experiment_queue import ExperimentJob
from .queue_monitor import AlertSeverity, QueueAlert

logger = structlog.get_logger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of notifications."""
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    EXPERIMENT_SUSPENDED = "experiment_suspended"
    EXPERIMENT_RESUMED = "experiment_resumed"
    PLATFORM_ROTATED = "platform_rotated"
    SESSION_WARNING = "session_warning"
    QUEUE_ALERT = "queue_alert"
    SYSTEM_STATUS = "system_status"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class NotificationConfig:
    """Email notification configuration."""
    enabled: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    rate_limit_per_hour: int = 10
    batch_notifications: bool = True
    batch_interval_minutes: int = 15

    # Notification levels
    min_level: NotificationLevel = NotificationLevel.INFO
    notification_types: list[NotificationType] = None

    # Recipients
    recipients: list[str] = None
    cc_recipients: list[str] = None

    def __post_init__(self):
        if self.notification_types is None:
            self.notification_types = list(NotificationType)
        if self.recipients is None:
            self.recipients = []
        if self.cc_recipients is None:
            self.cc_recipients = []


@dataclass
class EmailNotification:
    """Email notification message."""
    notification_type: NotificationType
    level: NotificationLevel
    subject: str
    body: str
    html_body: str | None = None
    attachments: list[Path] = None
    timestamp: datetime = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}


class EmailTemplateManager:
    """Manages email templates for different notification types."""

    def __init__(self, template_dir: Path | None = None):
        self.template_dir = template_dir or Path(__file__).parent / "email_templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # Create default templates
        self._create_default_templates()

    def _create_default_templates(self):
        """Create default email templates."""
        templates = {
            'experiment_completed.html': '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .header { background-color: #28a745; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; font-family: Arial, sans-serif; }
        .details { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .footer { background-color: #6c757d; color: white; padding: 10px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>🎉 Experiment Completed Successfully</h2>
    </div>
    <div class="content">
        <h3>{{ job.config.name }}</h3>
        <div class="details">
            <p><strong>Job ID:</strong> {{ job.id }}</p>
            <p><strong>Platform:</strong> {{ job.platform }}</p>
            <p><strong>Model Size:</strong> {{ job.config.model_size }}</p>
            <p><strong>Runtime:</strong> {{ runtime_minutes|round(1) }} minutes</p>
            <p><strong>Tasks Completed:</strong> {{ job.config.dataset_tasks|length }}</p>
            {% if job.results %}
            <p><strong>Results:</strong></p>
            <ul>
            {% for key, value in job.results.items() %}
                <li>{{ key }}: {{ value }}</li>
            {% endfor %}
            </ul>
            {% endif %}
        </div>
        <p>The experiment has completed successfully and results are available.</p>
    </div>
    <div class="footer">
        <p>ARC Prize 2025 Platform Rotation System</p>
        <p>Generated at {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
    </div>
</body>
</html>
            ''',

            'experiment_failed.html': '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .header { background-color: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; font-family: Arial, sans-serif; }
        .details { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; }
        .footer { background-color: #6c757d; color: white; padding: 10px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>❌ Experiment Failed</h2>
    </div>
    <div class="content">
        <h3>{{ job.config.name }}</h3>
        <div class="details">
            <p><strong>Job ID:</strong> {{ job.id }}</p>
            <p><strong>Platform:</strong> {{ job.platform }}</p>
            <p><strong>Model Size:</strong> {{ job.config.model_size }}</p>
            <p><strong>Runtime:</strong> {{ runtime_minutes|round(1) }} minutes</p>
            <p><strong>Retry Count:</strong> {{ job.retry_count }}/{{ job.max_retries }}</p>
        </div>
        {% if job.last_error %}
        <div class="error">
            <h4>Error Details:</h4>
            <p>{{ job.last_error }}</p>
        </div>
        {% endif %}
        <p>The experiment has failed. {% if job.retry_count < job.max_retries %}A retry will be attempted automatically.{% else %}No more retries available.{% endif %}</p>
    </div>
    <div class="footer">
        <p>ARC Prize 2025 Platform Rotation System</p>
        <p>Generated at {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
    </div>
</body>
</html>
            ''',

            'platform_rotated.html': '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .header { background-color: #17a2b8; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; font-family: Arial, sans-serif; }
        .details { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .footer { background-color: #6c757d; color: white; padding: 10px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>🔄 Platform Rotation</h2>
    </div>
    <div class="content">
        <h3>Platform Switch Completed</h3>
        <div class="details">
            <p><strong>From:</strong> {{ from_platform }}</p>
            <p><strong>To:</strong> {{ to_platform }}</p>
            <p><strong>Reason:</strong> {{ reason }}</p>
            <p><strong>Jobs Suspended:</strong> {{ suspended_jobs|length }}</p>
            {% if session_stats %}
            <p><strong>Previous Session Stats:</strong></p>
            <ul>
                <li>Jobs Completed: {{ session_stats.jobs_completed }}</li>
                <li>Runtime: {{ session_stats.total_runtime_minutes|round(1) }} minutes</li>
                <li>GPU Utilization: {{ (session_stats.gpu_utilization * 100)|round(1) }}%</li>
            </ul>
            {% endif %}
        </div>
        <p>The platform rotation has completed successfully. Suspended experiments will resume on the new platform.</p>
    </div>
    <div class="footer">
        <p>ARC Prize 2025 Platform Rotation System</p>
        <p>Generated at {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
    </div>
</body>
</html>
            ''',

            'daily_summary.html': '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .header { background-color: #6f42c1; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; font-family: Arial, sans-serif; }
        .section { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric { display: inline-block; background-color: #e9ecef; padding: 10px; margin: 5px; border-radius: 3px; }
        .footer { background-color: #6c757d; color: white; padding: 10px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h2>📊 Daily System Summary</h2>
        <p>{{ date.strftime('%Y-%m-%d') }}</p>
    </div>
    <div class="content">
        <div class="section">
            <h3>Experiment Statistics</h3>
            <div class="metric">
                <strong>{{ stats.completed_jobs }}</strong><br>
                Completed
            </div>
            <div class="metric">
                <strong>{{ stats.failed_jobs }}</strong><br>
                Failed
            </div>
            <div class="metric">
                <strong>{{ stats.queued_jobs }}</strong><br>
                Queued
            </div>
            <div class="metric">
                <strong>{{ (stats.success_rate * 100)|round(1) }}%</strong><br>
                Success Rate
            </div>
        </div>

        <div class="section">
            <h3>Platform Usage</h3>
            {% for platform, count in stats.platform_distribution.items() %}
            <div class="metric">
                <strong>{{ count }}</strong><br>
                {{ platform.title() }}
            </div>
            {% endfor %}
        </div>

        <div class="section">
            <h3>Performance Metrics</h3>
            <div class="metric">
                <strong>{{ stats.throughput_jobs_per_hour|round(2) }}</strong><br>
                Jobs/Hour
            </div>
            <div class="metric">
                <strong>{{ (stats.estimated_gpu_utilization * 100)|round(1) }}%</strong><br>
                GPU Utilization
            </div>
            <div class="metric">
                <strong>${{ stats.estimated_cost_savings|round(2) }}</strong><br>
                Cost Savings
            </div>
        </div>

        {% if alerts %}
        <div class="section">
            <h3>Recent Alerts ({{ alerts|length }})</h3>
            {% for alert in alerts %}
            <div style="background-color: {% if alert.severity == 'critical' %}#f8d7da{% elif alert.severity == 'error' %}#fff3cd{% else %}#d4edda{% endif %}; padding: 5px; margin: 5px 0; border-radius: 3px;">
                <strong>{{ alert.severity.title() }}:</strong> {{ alert.message }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    <div class="footer">
        <p>ARC Prize 2025 Platform Rotation System</p>
        <p>Generated at {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
    </div>
</body>
</html>
            '''
        }

        for template_name, template_content in templates.items():
            template_file = self.template_dir / template_name
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(template_content)

    def render_template(self, template_name: str, **context) -> str:
        """Render email template with context."""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error("template_render_failed", template=template_name, error=str(e))
            return f"Template rendering failed: {str(e)}"

    def get_template_for_notification(self, notification_type: NotificationType) -> str:
        """Get template filename for notification type."""
        template_map = {
            NotificationType.EXPERIMENT_COMPLETED: "experiment_completed.html",
            NotificationType.EXPERIMENT_FAILED: "experiment_failed.html",
            NotificationType.EXPERIMENT_SUSPENDED: "experiment_suspended.html",
            NotificationType.EXPERIMENT_RESUMED: "experiment_resumed.html",
            NotificationType.PLATFORM_ROTATED: "platform_rotated.html",
            NotificationType.SESSION_WARNING: "session_warning.html",
            NotificationType.QUEUE_ALERT: "queue_alert.html",
            NotificationType.SYSTEM_STATUS: "system_status.html",
            NotificationType.DAILY_SUMMARY: "daily_summary.html"
        }

        return template_map.get(notification_type, "default.html")


class EmailNotificationSystem:
    """Comprehensive email notification system."""

    def __init__(self, config: NotificationConfig | None = None):
        """Initialize email notification system.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self.credential_manager = get_platform_credential_manager()
        self.template_manager = EmailTemplateManager()

        # State
        self.notification_queue: list[EmailNotification] = []
        self.sent_notifications: list[EmailNotification] = []
        self.rate_limit_tracker: dict[str, list[datetime]] = defaultdict(list)
        self.batch_notifications: list[EmailNotification] = []

        # Background tasks
        self.notification_task: asyncio.Task | None = None
        self.batch_task: asyncio.Task | None = None
        self.running = False

        self.logger = structlog.get_logger('email_notifications')

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate email configuration."""
        try:
            # Check for email credentials
            smtp_server = self.credential_manager.get_platform_credential('email', 'smtp_server')
            username = self.credential_manager.get_platform_credential('email', 'username')
            password = self.credential_manager.get_platform_credential('email', 'password')

            if not all([smtp_server, username, password]):
                self.logger.warning("email_credentials_incomplete",
                                  has_server=bool(smtp_server),
                                  has_username=bool(username),
                                  has_password=bool(password))
                self.config.enabled = False

            # Check recipients
            if not self.config.recipients:
                # Try to get from credentials
                from_address = self.credential_manager.get_platform_credential('email', 'from_address')
                if from_address:
                    self.config.recipients = [from_address]
                else:
                    self.logger.warning("no_email_recipients_configured")
                    self.config.enabled = False

            self.logger.info("email_configuration_validated",
                           enabled=self.config.enabled,
                           recipients=len(self.config.recipients))

        except Exception as e:
            self.logger.error("email_configuration_validation_failed", error=str(e))
            self.config.enabled = False

    async def start_notification_system(self) -> bool:
        """Start the email notification system.

        Returns:
            True if started successfully
        """
        if self.running:
            self.logger.warning("notification_system_already_running")
            return False

        if not self.config.enabled:
            self.logger.warning("notification_system_disabled")
            return False

        try:
            self.running = True

            # Start notification processing task
            self.notification_task = asyncio.create_task(self._notification_processing_loop())

            # Start batch processing task if enabled
            if self.config.batch_notifications:
                self.batch_task = asyncio.create_task(self._batch_processing_loop())

            self.logger.info("email_notification_system_started")
            return True

        except Exception as e:
            self.logger.error("notification_system_start_failed", error=str(e))
            self.running = False
            return False

    async def stop_notification_system(self) -> bool:
        """Stop the email notification system.

        Returns:
            True if stopped successfully
        """
        if not self.running:
            return True

        try:
            self.running = False

            # Cancel tasks
            if self.notification_task:
                self.notification_task.cancel()
                try:
                    await self.notification_task
                except asyncio.CancelledError:
                    pass

            if self.batch_task:
                self.batch_task.cancel()
                try:
                    await self.batch_task
                except asyncio.CancelledError:
                    pass

            # Send any remaining notifications
            await self._flush_notifications()

            self.logger.info("email_notification_system_stopped")
            return True

        except Exception as e:
            self.logger.error("notification_system_stop_failed", error=str(e))
            return False

    async def send_notification(self, notification: EmailNotification) -> bool:
        """Send email notification.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            return False

        # Check notification level
        if notification.level.value < self.config.min_level.value:
            self.logger.debug("notification_below_min_level",
                            level=notification.level.value,
                            min_level=self.config.min_level.value)
            return False

        # Check notification type filter
        if notification.notification_type not in self.config.notification_types:
            self.logger.debug("notification_type_filtered",
                            type=notification.notification_type.value)
            return False

        # Check rate limit
        if self._is_rate_limited(notification):
            self.logger.warning("notification_rate_limited",
                              type=notification.notification_type.value)
            return False

        try:
            if self.config.batch_notifications:
                # Add to batch queue
                self.batch_notifications.append(notification)
                self.logger.debug("notification_added_to_batch",
                                type=notification.notification_type.value)
                return True
            else:
                # Send immediately
                return await self._send_email(notification)

        except Exception as e:
            self.logger.error("notification_send_failed",
                            type=notification.notification_type.value,
                            error=str(e))
            return False

    async def send_experiment_completed(self, job: ExperimentJob) -> bool:
        """Send experiment completed notification."""
        runtime_minutes = 0
        if job.started_at and job.completed_at:
            runtime_minutes = (job.completed_at - job.started_at).total_seconds() / 60

        # Generate HTML body
        html_body = self.template_manager.render_template(
            "experiment_completed.html",
            job=job,
            runtime_minutes=runtime_minutes,
            timestamp=datetime.now()
        )

        notification = EmailNotification(
            notification_type=NotificationType.EXPERIMENT_COMPLETED,
            level=NotificationLevel.INFO,
            subject=f"✅ Experiment Completed: {job.config.name}",
            body=f"Experiment '{job.config.name}' completed successfully in {runtime_minutes:.1f} minutes on {job.platform}.",
            html_body=html_body,
            metadata={'job_id': job.id, 'platform': job.platform}
        )

        return await self.send_notification(notification)

    async def send_experiment_failed(self, job: ExperimentJob) -> bool:
        """Send experiment failed notification."""
        runtime_minutes = 0
        if job.started_at and job.completed_at:
            runtime_minutes = (job.completed_at - job.started_at).total_seconds() / 60

        # Generate HTML body
        html_body = self.template_manager.render_template(
            "experiment_failed.html",
            job=job,
            runtime_minutes=runtime_minutes,
            timestamp=datetime.now()
        )

        notification = EmailNotification(
            notification_type=NotificationType.EXPERIMENT_FAILED,
            level=NotificationLevel.ERROR,
            subject=f"❌ Experiment Failed: {job.config.name}",
            body=f"Experiment '{job.config.name}' failed after {runtime_minutes:.1f} minutes. Error: {job.last_error or 'Unknown error'}",
            html_body=html_body,
            metadata={'job_id': job.id, 'platform': job.platform, 'error': job.last_error}
        )

        return await self.send_notification(notification)

    async def send_platform_rotation(self, from_platform: str, to_platform: str,
                                   reason: str, metadata: dict[str, Any] | None = None) -> bool:
        """Send platform rotation notification."""
        # Generate HTML body
        html_body = self.template_manager.render_template(
            "platform_rotated.html",
            from_platform=from_platform,
            to_platform=to_platform,
            reason=reason,
            suspended_jobs=metadata.get('suspended_jobs', []) if metadata else [],
            session_stats=metadata.get('session_stats') if metadata else None,
            timestamp=datetime.now()
        )

        notification = EmailNotification(
            notification_type=NotificationType.PLATFORM_ROTATED,
            level=NotificationLevel.INFO,
            subject=f"🔄 Platform Rotation: {from_platform} → {to_platform}",
            body=f"Platform rotated from {from_platform} to {to_platform}. Reason: {reason}",
            html_body=html_body,
            metadata=metadata or {}
        )

        return await self.send_notification(notification)

    async def send_session_warning(self, platform: str, remaining_minutes: float) -> bool:
        """Send session timeout warning notification."""
        notification = EmailNotification(
            notification_type=NotificationType.SESSION_WARNING,
            level=NotificationLevel.WARNING,
            subject=f"⚠️ Session Timeout Warning: {platform}",
            body=f"Session on {platform} will timeout in {remaining_minutes:.1f} minutes. Experiments will be suspended.",
            metadata={'platform': platform, 'remaining_minutes': remaining_minutes}
        )

        return await self.send_notification(notification)

    async def send_queue_alert(self, alert: QueueAlert) -> bool:
        """Send queue monitoring alert notification."""
        level_map = {
            AlertSeverity.INFO: NotificationLevel.INFO,
            AlertSeverity.WARNING: NotificationLevel.WARNING,
            AlertSeverity.ERROR: NotificationLevel.ERROR,
            AlertSeverity.CRITICAL: NotificationLevel.CRITICAL
        }

        notification = EmailNotification(
            notification_type=NotificationType.QUEUE_ALERT,
            level=level_map.get(alert.severity, NotificationLevel.WARNING),
            subject=f"🚨 Queue Alert: {alert.message}",
            body=f"Queue alert: {alert.message}\\nDetails: {json.dumps(alert.details, indent=2)}",
            metadata={'alert_id': alert.id, 'severity': alert.severity.value}
        )

        return await self.send_notification(notification)

    async def send_daily_summary(self, stats: dict[str, Any], alerts: list[QueueAlert]) -> bool:
        """Send daily system summary notification."""
        # Generate HTML body
        html_body = self.template_manager.render_template(
            "daily_summary.html",
            date=datetime.now(),
            stats=stats,
            alerts=alerts,
            timestamp=datetime.now()
        )

        notification = EmailNotification(
            notification_type=NotificationType.DAILY_SUMMARY,
            level=NotificationLevel.INFO,
            subject=f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
            body=f"Daily system summary: {stats.get('completed_jobs', 0)} jobs completed, {stats.get('failed_jobs', 0)} failed.",
            html_body=html_body,
            metadata={'stats': stats, 'alert_count': len(alerts)}
        )

        return await self.send_notification(notification)

    def _is_rate_limited(self, notification: EmailNotification) -> bool:
        """Check if notification is rate limited."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Clean old entries
        notification_key = f"{notification.notification_type.value}_{notification.level.value}"
        self.rate_limit_tracker[notification_key] = [
            timestamp for timestamp in self.rate_limit_tracker[notification_key]
            if timestamp > hour_ago
        ]

        # Check limit
        if len(self.rate_limit_tracker[notification_key]) >= self.config.rate_limit_per_hour:
            return True

        # Add current notification
        self.rate_limit_tracker[notification_key].append(now)
        return False

    async def _notification_processing_loop(self):
        """Process notifications from queue."""
        while self.running:
            try:
                if self.notification_queue:
                    notification = self.notification_queue.pop(0)
                    await self._send_email(notification)
                else:
                    await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("notification_processing_error", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error

    async def _batch_processing_loop(self):
        """Process batched notifications."""
        while self.running:
            try:
                await asyncio.sleep(self.config.batch_interval_minutes * 60)

                if self.batch_notifications:
                    await self._send_batch_notifications()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("batch_processing_error", error=str(e))
                await asyncio.sleep(60)

    async def _send_batch_notifications(self):
        """Send batched notifications."""
        if not self.batch_notifications:
            return

        try:
            # Group notifications by type
            grouped_notifications = defaultdict(list)
            for notification in self.batch_notifications:
                grouped_notifications[notification.notification_type].append(notification)

            # Send batch for each type
            for notification_type, notifications in grouped_notifications.items():
                if len(notifications) == 1:
                    # Send single notification
                    await self._send_email(notifications[0])
                else:
                    # Create summary notification
                    await self._send_batch_summary(notification_type, notifications)

            # Clear batch queue
            self.batch_notifications.clear()

            self.logger.info("batch_notifications_sent",
                           total=sum(len(notifs) for notifs in grouped_notifications.values()))

        except Exception as e:
            self.logger.error("batch_send_failed", error=str(e))

    async def _send_batch_summary(self, notification_type: NotificationType,
                                notifications: list[EmailNotification]):
        """Send summary of batched notifications."""
        count = len(notifications)

        # Create summary subject and body
        subject = f"📦 Batch Update: {count} {notification_type.value.replace('_', ' ').title()} Notifications"

        body_parts = [f"Summary of {count} {notification_type.value.replace('_', ' ')} notifications:"]
        for i, notification in enumerate(notifications[:10]):  # Limit to first 10
            body_parts.append(f"{i+1}. {notification.subject}")

        if count > 10:
            body_parts.append(f"... and {count - 10} more")

        batch_notification = EmailNotification(
            notification_type=notification_type,
            level=NotificationLevel.INFO,
            subject=subject,
            body="\\n".join(body_parts),
            metadata={'batch_size': count, 'notification_type': notification_type.value}
        )

        await self._send_email(batch_notification)

    async def _send_email(self, notification: EmailNotification) -> bool:
        """Send individual email notification."""
        try:
            # Get credentials
            smtp_server = self.credential_manager.get_platform_credential('email', 'smtp_server') or self.config.smtp_server
            smtp_port = int(self.credential_manager.get_platform_credential('email', 'smtp_port') or self.config.smtp_port)
            username = self.credential_manager.get_platform_credential('email', 'username')
            password = self.credential_manager.get_platform_credential('email', 'password')
            from_address = self.credential_manager.get_platform_credential('email', 'from_address') or username

            if not all([smtp_server, username, password]):
                self.logger.error("email_credentials_missing")
                return False

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = from_address
            msg['To'] = ', '.join(self.config.recipients)

            if self.config.cc_recipients:
                msg['Cc'] = ', '.join(self.config.cc_recipients)

            # Add text body
            msg.attach(MIMEText(notification.body, 'plain'))

            # Add HTML body if available
            if notification.html_body:
                msg.attach(MIMEText(notification.html_body, 'html'))

            # Add attachments
            for attachment_path in notification.attachments:
                if attachment_path.exists():
                    with open(attachment_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment_path.name}'
                    )
                    msg.attach(part)

            # Send email
            context = ssl.create_default_context()

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if self.config.use_tls:
                    server.starttls(context=context)

                server.login(username, password)

                recipients = self.config.recipients + self.config.cc_recipients
                server.send_message(msg, from_address, recipients)

            # Track sent notification
            self.sent_notifications.append(notification)

            self.logger.info("email_sent_successfully",
                           type=notification.notification_type.value,
                           recipients=len(recipients))

            return True

        except Exception as e:
            self.logger.error("email_send_failed",
                            type=notification.notification_type.value,
                            error=str(e))
            return False

    async def _flush_notifications(self):
        """Send any remaining notifications."""
        try:
            # Send remaining batch notifications
            if self.batch_notifications:
                await self._send_batch_notifications()

            # Send remaining queued notifications
            while self.notification_queue:
                notification = self.notification_queue.pop(0)
                await self._send_email(notification)

        except Exception as e:
            self.logger.error("notification_flush_failed", error=str(e))

    def get_notification_statistics(self) -> dict[str, Any]:
        """Get notification system statistics."""
        return {
            'enabled': self.config.enabled,
            'running': self.running,
            'sent_count': len(self.sent_notifications),
            'queued_count': len(self.notification_queue),
            'batched_count': len(self.batch_notifications),
            'recipients': len(self.config.recipients),
            'rate_limit_per_hour': self.config.rate_limit_per_hour,
            'batch_enabled': self.config.batch_notifications,
            'notification_types': [t.value for t in self.config.notification_types]
        }


# Singleton instance
_email_notification_system = None


def get_email_notification_system(config: NotificationConfig | None = None) -> EmailNotificationSystem:
    """Get singleton email notification system instance."""
    global _email_notification_system
    if _email_notification_system is None:
        _email_notification_system = EmailNotificationSystem(config)
    return _email_notification_system
